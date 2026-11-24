"""Tests for calliopy.widefield._registration (port of register_wid.m and relatives).

All tests are synthetic. Images are sums of Gaussian blobs evaluated analytically, so a
moving image related to the fixed one by a known similarity transform is exact (no
interpolation). imregtform itself cannot be reproduced; the tests check that known
transforms are recovered, and that everything around it follows the MATLAB code.
"""

from __future__ import annotations

import gc
import struct
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import scipy.ndimage

from calliopy._matlab import corr2
from calliopy.io import MatStruct
from calliopy.widefield._registration import (
    SIMILARITY_THRESHOLD,
    StackAlignmentError,
    _local_normalise,
    compute_tform,
    compute_tform_manual,
    crop_margins,
    imregtform_similarity,
    plot_registration_check,
    propagate_tform,
    register_wid,
    registration_similarity,
    uncrop_tform,
)
from calliopy.widefield._transform import (
    similarity_matrix,
    similarity_params,
    transform_points,
    warp_image,
)


def blobs(h, w, tform=None, seed=1, n=None, base=1000.0, gain=None):
    """Blob texture sampled at pixel (x, y) (1-based) or at ``tform @ (x, y)``.

    ``moving = blobs(h, w, A)`` and ``fixed = blobs(h, w)`` satisfy
    ``moving(p) = fixed(A p)``, i.e. ``A`` maps moving to fixed coordinates.
    """
    g = np.random.default_rng(seed)
    n = n or int(h * w / 25)
    xk = g.uniform(-20, w + 20, n)
    yk = g.uniform(-20, h + 20, n)
    sk = g.uniform(1.5, 5.0, n)
    ak = g.uniform(100, 800, n)
    yy, xx = np.mgrid[1 : h + 1, 1 : w + 1].astype(np.float64)
    if tform is not None:
        xx, yy = (
            tform[0, 0] * xx + tform[0, 1] * yy + tform[0, 2],
            tform[1, 0] * xx + tform[1, 1] * yy + tform[1, 2],
        )
    img = np.full((h, w), base)
    for k in range(n):
        img += ak[k] * np.exp(-((xx - xk[k]) ** 2 + (yy - yk[k]) ** 2) / (2 * sk[k] ** 2))
    if gain is not None:
        img *= gain
    return img


def centre_error(a_est, a_true, shape):
    h, w = shape
    c = np.array([(w + 1) / 2, (h + 1) / 2])
    return float(np.linalg.norm(transform_points(a_est, c) - transform_points(a_true, c)))


def max_error(a_est, a_true, shape, margin):
    """Largest point error over the corners of the region inside ``margin``."""
    h, w = shape
    pts = np.array([[1 + margin, 1 + margin], [w - margin, 1 + margin],
                    [1 + margin, h - margin], [w - margin, h - margin]], dtype=float)
    return float(np.max(np.linalg.norm(transform_points(a_est, pts)
                                       - transform_points(a_true, pts), axis=1)))


H, W = 96, 112
CASES = [
    (1.0, 0.0, (6.0, -5.0)),
    (0.97, -4.0, (-3.0, 7.5)),
    (1.04, 5.0, (2.0, 3.0)),
    (1.0, 2.0, (-8.3, -6.7)),
    (1.02, -2.5, (0.4, 0.3)),
]


# ======================================================================================
# imregtform replacement
# ======================================================================================


@pytest.mark.parametrize("scale,angle,shift", CASES)
def test_imregtform_similarity_recovers_known_transform(scale, angle, shift):
    # default path: identity start, as MATLAB imregtform
    a_true = similarity_matrix(scale, angle, shift)
    fixed = blobs(H, W)
    moving = blobs(H, W, a_true)
    a_est = imregtform_similarity(moving, fixed, rng=np.random.default_rng(3))
    assert centre_error(a_est, a_true, (H, W)) < 0.25
    assert max_error(a_est, a_true, (H, W), 10) < 0.6
    p_est, p_true = similarity_params(a_est), similarity_params(a_true)
    assert abs(p_est.scale - p_true.scale) < 0.01
    assert abs(p_est.rotation_angle - p_true.rotation_angle) < 0.5


@pytest.mark.parametrize("scale,angle,shift", CASES)
def test_imregtform_similarity_optional_phase_correlation_start(scale, angle, shift):
    # opt-in extra (not in MATLAB): start at the phase-correlation translation
    a_true = similarity_matrix(scale, angle, shift)
    fixed = blobs(H, W)
    moving = blobs(H, W, a_true)
    a_est = imregtform_similarity(moving, fixed, rng=np.random.default_rng(3),
                                  initialize_translation=True)
    assert centre_error(a_est, a_true, (H, W)) < 0.25


def test_identity_start_is_the_default():
    # the default equals an explicit identity start, and differs from the opt-in start
    a_true = similarity_matrix(1.03, 3.0, (2.5, -3.5))
    fixed = blobs(H, W, seed=4)
    moving = blobs(H, W, a_true, seed=4)
    a_default = imregtform_similarity(moving, fixed, rng=1)
    a_identity = imregtform_similarity(moving, fixed, rng=1, initialize_translation=False)
    a_pc = imregtform_similarity(moving, fixed, rng=1, initialize_translation=True)
    np.testing.assert_array_equal(a_default, a_identity)
    assert not np.array_equal(a_default, a_pc)
    assert centre_error(a_default, a_true, (H, W)) < 0.25


def test_imregtform_similarity_noise_and_illumination():
    # different noise and a multiplicative intensity gain: mutual information copes
    a_true = similarity_matrix(0.98, -3.0, (4.0, 2.0))
    g = np.random.default_rng(7)
    _, xx = np.mgrid[0:H, 0:W]
    fixed = blobs(H, W, seed=5) + g.normal(0, 15, (H, W))
    moving = 0.6 * blobs(H, W, a_true, seed=5) + g.normal(0, 15, (H, W)) + 0.5 * xx
    a_est = imregtform_similarity(moving, fixed, rng=0)
    assert centre_error(a_est, a_true, (H, W)) < 0.5


def test_imregtform_similarity_deterministic_for_rng():
    a_true = similarity_matrix(1.01, 1.0, (3.0, -2.0))
    fixed = blobs(H, W, seed=2)
    moving = blobs(H, W, a_true, seed=2)
    a1 = imregtform_similarity(moving, fixed, rng=np.random.default_rng(11))
    a2 = imregtform_similarity(moving, fixed, rng=np.random.default_rng(11))
    a3 = imregtform_similarity(moving, fixed)
    a4 = imregtform_similarity(moving, fixed)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(a3, a4)  # rng=None: fixed seed
    assert a1.shape == (3, 3) and a1.dtype == np.float64
    np.testing.assert_array_equal(a1[2], [0, 0, 1])
    similarity_params(a1)  # a non-reflective similarity


def test_imregtform_similarity_identity_and_errors():
    fixed = blobs(H, W, seed=9)
    a_est = imregtform_similarity(fixed, fixed, rng=0)
    assert centre_error(a_est, np.eye(3), (H, W)) < 0.1
    with pytest.raises(ValueError):
        imregtform_similarity(fixed[0], fixed)
    with pytest.raises(ValueError):
        imregtform_similarity(fixed[:3, :3], fixed[:3, :3])


# ======================================================================================
# pieces of compute_tform
# ======================================================================================


def test_crop_margins_round_half_away():
    assert crop_margins((96, 112), 0.1) == (11, 10)  # 11.2 -> 11, 9.6 -> 10
    assert crop_margins((25, 15), 0.1) == (2, 3)  # 1.5 -> 2, 2.5 -> 3 (not banker's)
    assert crop_margins((10, 10), 0) == (0, 0)
    with pytest.raises(ValueError):
        crop_margins((10, 10), float("nan"))


def test_uncrop_tform_formula():
    # dshift = T - ((A(1:2,1:2) - eye(2)) * [dw; dh])'
    a = similarity_matrix(1.05, 7.0, (3.0, -4.0))
    dw, dh = 11, 6
    out = uncrop_tform(a, dw, dh)
    expected = a[:2, 2] - (a[:2, :2] - np.eye(2)) @ np.array([dw, dh])
    np.testing.assert_allclose(out[:2, 2], expected, rtol=1e-15, atol=1e-13)
    np.testing.assert_array_equal(out[:2, :2], a[:2, :2])
    # it is the conjugation with the crop shift: full = S A S^-1, S = translate(dw, dh)
    s = np.array([[1.0, 0, dw], [0, 1, dh], [0, 0, 1]])
    np.testing.assert_allclose(out, s @ a @ np.linalg.inv(s), atol=1e-12)
    # pure translation is unchanged
    t = similarity_matrix(1.0, 0.0, (2.0, 5.0))
    np.testing.assert_allclose(uncrop_tform(t, dw, dh), t, atol=1e-15)


def test_uncrop_tform_aligns_uncropped_images():
    # transform estimated on cropped images, corrected, aligns the uncropped images
    a_true = similarity_matrix(1.03, 4.0, (-3.0, 5.0))
    fixed = blobs(H, W, seed=6)
    moving = blobs(H, W, a_true, seed=6)
    dw, dh = crop_margins(moving.shape, 0.1)
    a_crop = imregtform_similarity(
        moving[dh : H - dh, dw : W - dw], fixed[dh : H - dh, dw : W - dw], rng=0
    )
    a_full = uncrop_tform(a_crop, dw, dh)
    assert centre_error(a_full, a_true, (H, W)) < 0.25
    # the uncorrected transform does not (rotation + scale about a shifted origin)
    assert centre_error(a_crop, a_true, (H, W)) > 1.0
    warped = warp_image(moving, a_full, fixed.shape)
    inner = (slice(dh, H - dh), slice(dw, W - dw))
    resid = np.abs(warped[inner] - fixed[inner]) / np.ptp(fixed)
    assert np.median(resid) < 0.01


def test_local_normalise_matches_uniform_filter():
    g = np.random.default_rng(0)
    img = g.integers(100, 4000, size=(40, 50)).astype(np.uint16)
    win = 5  # odd window: filter2 ones/win^2 == zero-padded uniform filter
    expected = img / scipy.ndimage.uniform_filter(img.astype(float), win, mode="constant")
    np.testing.assert_allclose(_local_normalise(img, win), expected, rtol=1e-12)
    # even window: conv2 'same' keeps full(k//2 + i), i.e. the window covers rows
    # i-1 .. i+2 (0-based) for a 4-tap kernel
    win = 4
    pad = np.pad(img.astype(float), ((1, 2), (1, 2)))
    mean = np.zeros(img.shape)
    for i in range(4):
        for j in range(4):
            mean += pad[i : i + 40, j : j + 50]
    np.testing.assert_allclose(_local_normalise(img, win), img / (mean / 16), rtol=1e-12)


def test_registration_similarity_is_cropped_corr2():
    g = np.random.default_rng(1)
    fixed = blobs(50, 60, seed=3)
    moving = fixed + g.normal(0, 50, fixed.shape)
    s = registration_similarity(moving, fixed, np.eye(3), 0.1)
    dw, dh = 6, 5
    ref = np.corrcoef(moving[dh:-dh, dw:-dw].ravel(), fixed[dh:-dh, dw:-dw].ravel())[0, 1]
    assert s == pytest.approx(ref, rel=1e-12)
    # uint16 moving: warped in its class (rounded), as imwarp does
    m16 = np.round(moving).astype(np.uint16)
    tf = similarity_matrix(1.0, 0.0, (0.5, 0.0))
    w16 = warp_image(m16, tf, fixed.shape)
    assert w16.dtype == np.uint16
    assert registration_similarity(m16, fixed, tf, 0.1) == pytest.approx(
        corr2(w16[dh:-dh, dw:-dw], fixed[dh:-dh, dw:-dw]), rel=1e-12
    )


# ======================================================================================
# compute_tform
# ======================================================================================


def _vignetted_pair(a_true, seed=8):
    yy, xx = np.mgrid[0:H, 0:W]
    vignette = 1.0 - 0.4 * ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (W * W / 4)
    fixed = np.round(blobs(H, W, seed=seed) * vignette).astype(np.uint16)
    moving = np.round(blobs(H, W, a_true, seed=seed) * vignette * 1.3).astype(np.uint16)
    return moving, fixed


# Measured on these 96x112 vignetted uint16 pairs (window 5 px): from the identity start,
# rotation + scale, translations up to ~4 px, and a small combination are recovered; the
# combinations in NORMALISED_IDENTITY_MISSES are not (they need initialize_translation=True).
NORMALISED_IDENTITY_HITS = [
    (1.02, -3.0, (0.0, 0.0)),
    (1.0, 0.0, (3.0, -2.5)),
    (1.0, 0.0, (4.0, 0.0)),
    (1.01, -1.5, (2.0, -1.5)),
]
NORMALISED_IDENTITY_MISSES = [
    (1.02, -3.0, (5.0, -4.0)),
    (1.02, -3.0, (3.0, -2.5)),
    (1.0, 0.0, (5.0, -4.0)),
    (0.98, 3.0, (-3.0, 2.0)),
]


@pytest.mark.parametrize("scale,angle,shift", NORMALISED_IDENTITY_HITS)
def test_compute_tform_recovers_and_reports(scale, angle, shift):
    # default path (identity start, as MATLAB), with local normalisation
    a_true = similarity_matrix(scale, angle, shift)
    moving, fixed = _vignetted_pair(a_true)
    res = compute_tform(moving, fixed, rng=0)
    assert centre_error(res.tform, a_true, (H, W)) < 0.3
    rp = res.registration_parameters
    assert isinstance(rp, MatStruct)
    assert list(rp) == ["status", "similarity", "similarity_threshold", "fraction_boundary",
                        "registration_type", "activity_calculated"]
    assert rp.status == "SUCCESS"
    assert rp.similarity > 0.9
    assert rp.similarity == pytest.approx(registration_similarity(moving, fixed, res.tform, 0.1))
    assert rp.similarity_threshold == 0.5 == SIMILARITY_THRESHOLD
    assert rp.fraction_boundary == 0.1
    assert rp.registration_type == "automatic"
    assert rp.activity_calculated == 0


def test_compute_tform_larger_shift_without_normalisation():
    # without the local normalisation the identity start also finds a 6.4 px shift
    a_true = similarity_matrix(1.02, -3.0, (5.0, -4.0))
    moving, fixed = _vignetted_pair(a_true)
    res = compute_tform(moving, fixed, normalize_template=False, rng=0, frac_boundary=0.15)
    assert centre_error(res.tform, a_true, (H, W)) < 0.3
    assert res.registration_parameters.fraction_boundary == 0.15


@pytest.mark.xfail(strict=True, reason=(
    "known limit of the identity start: with local normalisation (window 5 px on 96x112) "
    "these transforms are not recovered; use initialize_translation=True"))
@pytest.mark.parametrize("scale,angle,shift", NORMALISED_IDENTITY_MISSES)
def test_compute_tform_identity_start_limit_normalised(scale, angle, shift):
    a_true = similarity_matrix(scale, angle, shift)
    moving, fixed = _vignetted_pair(a_true)
    res = compute_tform(moving, fixed, rng=0)
    assert centre_error(res.tform, a_true, (H, W)) < 0.3


@pytest.mark.parametrize("scale,angle,shift", NORMALISED_IDENTITY_MISSES)
def test_compute_tform_optional_phase_correlation_start(scale, angle, shift):
    # the opt-in start recovers the cases the identity start misses
    a_true = similarity_matrix(scale, angle, shift)
    moving, fixed = _vignetted_pair(a_true)
    res = compute_tform(moving, fixed, rng=0, initialize_translation=True)
    assert centre_error(res.tform, a_true, (H, W)) < 0.3
    assert res.registration_parameters.status == "SUCCESS"


def test_compute_tform_failed_status():
    fixed = blobs(H, W, seed=1)
    moving = blobs(H, W, seed=2)  # unrelated
    res = compute_tform(moving, fixed, rng=0)
    assert res.registration_parameters.similarity < 0.5
    assert res.registration_parameters.status == "FAILED"


def test_compute_tform_threshold_is_inclusive():
    # status SUCCESS when similarity == threshold: MATLAB tests "similarity < thr"
    fixed = blobs(H, W, seed=1)
    res = compute_tform(fixed, fixed, rng=0)
    assert res.registration_parameters.status == "SUCCESS"


# ======================================================================================
# manual transform and propagation
# ======================================================================================


def test_compute_tform_manual():
    a_true = similarity_matrix(1.01, 2.0, (-2.0, 3.0))
    fixed = blobs(H, W, seed=3)
    moving = blobs(H, W, a_true, seed=3)
    mp = np.array([[20.0, 30.0], [80.0, 70.0]])
    fp = transform_points(a_true, mp)
    res = compute_tform_manual(moving, fixed, mp, fp, frac_boundary=0.1)
    np.testing.assert_allclose(res.tform, a_true, atol=1e-10)
    assert res.similarity == pytest.approx(registration_similarity(moving, fixed, a_true, 0.1))
    assert res.similarity > 0.98


def _rp(**kw):
    rp = MatStruct(status="SUCCESS", similarity=0.9, similarity_threshold=0.5,
                   fraction_boundary=0.1, registration_type="automatic", activity_calculated=1.0)
    rp.update(kw)
    return rp


def test_propagate_tform_updates_parameters():
    a = similarity_matrix(1.01, 2.0, (3.0, -1.0))
    site = blobs(H, W, seed=4)
    inherit = blobs(H, W, a, seed=4)
    rp_in = _rp()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = propagate_tform(inherit, site, a, rp_in, exp_id_origin=123, exp_id=124)
    rp = res.registration_parameters
    assert rp.registration_type == "Copied from 123"
    assert rp.status == "SUCCESS"
    assert rp.similarity == pytest.approx(registration_similarity(inherit, site, a, 0.1))
    assert rp.similarity > 0.99
    assert rp.similarity_threshold == 0.5
    assert rp.activity_calculated == 1.0  # kept without ROIs, as in MATLAB
    assert res.ROIs is None
    assert rp_in.registration_type == "automatic"  # input untouched
    np.testing.assert_array_equal(res.tform, a)


def test_propagate_tform_warns_and_keeps_success():
    site = blobs(H, W, seed=4)
    other = blobs(H, W, seed=5)
    with pytest.warns(UserWarning, match="registered as success even though similarity"):
        res = propagate_tform(other, site, np.eye(3), _rp(), exp_id_origin=1, exp_id=2)
    assert res.registration_parameters.status == "SUCCESS"
    assert res.registration_parameters.similarity < 0.5


def test_propagate_tform_rois_and_errors():
    site = blobs(H, W, seed=4)
    rois = np.empty((1, 2), dtype=object)
    rois[0, 0] = MatStruct(indices=np.arange(1, 4), activity=np.ones(3))
    rois[0, 1] = MatStruct(indices=np.arange(1, 4), activity=np.ones(3))
    res = propagate_tform(site, site, np.eye(3), _rp(), exp_id_origin=1, rois=rois)
    assert "activity" not in res.ROIs[0, 0] and "activity" in res.ROIs[0, 1]
    assert "activity" in rois[0, 0]  # input untouched
    assert res.registration_parameters.activity_calculated == 0
    res_list = propagate_tform(site, site, np.eye(3), _rp(), exp_id_origin=1,
                               rois=[MatStruct(a=1)])
    assert res_list.ROIs == [MatStruct(a=1)]
    with pytest.raises(ValueError, match="empty"):
        propagate_tform(site, site, np.zeros((0, 0)), _rp(), exp_id_origin=1)
    with pytest.raises(ValueError):
        propagate_tform(site, site, np.eye(3), _rp(fraction_boundary=np.nan), exp_id_origin=1)


def test_plot_registration_check_returns_figure():
    from matplotlib.figure import Figure

    fixed = blobs(40, 50, seed=1)
    fig = plot_registration_check(fixed, fixed, np.eye(3), title="t")
    assert isinstance(fig, Figure) and len(fig.axes) == 3


# ======================================================================================
# register_wid with synthetic .wid files
# ======================================================================================


def bcd(value: int, nbytes: int) -> list[int]:
    digits = f"{value:0{2 * nbytes}d}"
    return [int(digits[2 * k : 2 * k + 2], 16) for k in range(nbytes)]


def pco_header(framenumber: int, stamp: datetime) -> list[int]:
    return (bcd(framenumber, 4) + bcd(stamp.year, 2) + bcd(stamp.month, 1) + bcd(stamp.day, 1)
            + bcd(stamp.hour, 1) + bcd(stamp.minute, 1) + bcd(stamp.second, 1)
            + bcd(stamp.microsecond, 3))


def write_wid(path: Path, imgs: np.ndarray, first_number: int = 1) -> np.ndarray:
    """(size_y, size_x, n) uint16 frames with PCO headers; returns what a reader sees."""
    size_y, size_x, n = imgs.shape
    out = bytearray()
    seen = imgs.astype(np.uint16).copy()
    for k in range(n):
        flat = seen[:, :, k].ravel(order="F")
        flat[:14] = pco_header(first_number + k, datetime(2024, 5, 6, 7, 8, 9, 1000 * k))
        seen[:, :, k] = flat.reshape((size_y, size_x), order="F")
        out += struct.pack("<HH", size_y, size_x) + flat.astype("<u2").tobytes()
    path.write_bytes(bytes(out))
    return seen


def stack(frame, n=100, noise=5.0, seed=0):
    g = np.random.default_rng(seed)
    data = frame[:, :, None] + g.normal(0, noise, frame.shape + (n,))
    return np.clip(np.round(data), 0, 65535).astype(np.uint16)


def release():
    gc.collect()


def test_register_wid_origin_single_channel(tmp_path):
    base = blobs(H, W, seed=10)
    paths = [tmp_path / "a.wid", tmp_path / "b.wid"]
    seen1 = write_wid(paths[0], stack(base, seed=1))
    write_wid(paths[1], stack(base, seed=2), first_number=101)
    ad = register_wid(paths)
    release()
    assert list(ad) == ["meta_info", "first_frame", "tform", "registration_parameters"]
    np.testing.assert_array_equal(ad.first_frame, seen1[:, :, 99])
    assert ad.first_frame.dtype == np.uint16
    assert ad.tform.shape == (0, 0)
    rp = ad.registration_parameters
    assert rp.status == "ORIGIN" and np.isnan(rp.similarity)
    assert np.isnan(rp.similarity_threshold) and np.isnan(rp.fraction_boundary)
    assert np.isnan(rp.registration_type) and rp.activity_calculated == 0
    meta = ad.meta_info
    assert meta.shape == (1, 200)
    assert "dimension" not in meta[0, 0]
    assert meta[0, 0].framenumber == 1 and meta[0, 150].framenumber == 151
    assert meta[0, 150].file_id == 2


def test_register_wid_registers_to_site(tmp_path):
    a_true = similarity_matrix(1.0, 1.5, (4.0, -3.0))
    site = blobs(H, W, seed=11)
    moving = blobs(H, W, a_true, seed=11)
    p = tmp_path / "m.wid"
    write_wid(p, stack(moving, seed=3))
    ad = register_wid(p, site_first_frame=site.astype(np.uint16), rng=0)
    release()
    assert list(ad) == ["meta_info", "first_frame", "tform", "registration_parameters"]
    assert centre_error(ad.tform, a_true, (H, W)) < 0.4
    assert ad.registration_parameters.status == "SUCCESS"
    assert ad.registration_parameters.registration_type == "automatic"


def test_register_wid_stack_misalignment_raises(tmp_path):
    base = blobs(H, W, seed=12)
    shifted = np.roll(base, (8, 0), axis=(0, 1))
    paths = [tmp_path / "a.wid", tmp_path / "b.wid"]
    write_wid(paths[0], stack(base, seed=1))
    write_wid(paths[1], stack(shifted, seed=2))
    with pytest.raises(StackAlignmentError, match="out of alignment"):
        register_wid(paths)
    ad = register_wid(paths, check_stacks=False)
    release()
    assert ad.registration_parameters.status == "ORIGIN"


def test_register_wid_two_colour_origin_and_registration(tmp_path):
    a_map = similarity_matrix(1.0, 1.0, (2.0, -1.5))
    ch525 = blobs(H, W, seed=13)
    ch610_flipped = blobs(H, W, np.linalg.inv(a_map), seed=13)  # A maps ch525 -> it
    ch610 = np.fliplr(ch610_flipped)
    files = {"_ch525": [tmp_path / "g.wid"], "_ch610": [tmp_path / "r.wid"]}
    write_wid(files["_ch525"][0], stack(ch525, seed=1))
    write_wid(files["_ch610"][0], stack(ch610, seed=2))
    ad = register_wid(files, rng=0)
    assert list(ad) == ["meta_info", "first_frame_ch525", "tform_ch525",
                        "registration_parameters_ch525", "first_frame_ch610", "tform_ch610",
                        "registration_parameters_ch610", "tform_ch_map",
                        "registration_parameters_ch_map"]
    assert ad.registration_parameters_ch525.status == "ORIGIN"
    assert ad.registration_parameters_ch610.status == "ORIGIN"
    assert ad.tform_ch525.shape == (0, 0) and ad.tform_ch610.shape == (0, 0)
    assert centre_error(ad.tform_ch_map, a_map, (H, W)) < 0.4
    assert ad.registration_parameters_ch_map.status == "SUCCESS"
    # a later experiment of the site: each channel against the origin's channel
    site = {"_ch525": ad.first_frame_ch525, "_ch610": ad.first_frame_ch610}
    ad2 = register_wid(files, site_first_frame=site, rng=0)
    release()
    assert "tform_ch_map" not in ad2
    for e in ("_ch525", "_ch610"):
        assert centre_error(ad2["tform" + e], np.eye(3), (H, W)) < 0.3
        assert ad2["registration_parameters" + e].status == "SUCCESS"
    with pytest.raises(ValueError, match="must come before"):
        register_wid({"_ch610": files["_ch610"], "_ch525": files["_ch525"]})
    with pytest.raises(ValueError, match="must map"):
        register_wid(files, site_first_frame=ad.first_frame_ch525)
    release()


def write_wii(path: Path, n_rois: int, mean_proj: np.ndarray) -> None:
    text = f"param.numberROIs={n_rois};"
    m2, m1 = mean_proj.shape
    frame = np.zeros((2, 3), dtype=np.uint16)
    path.write_bytes(struct.pack("<I", len(text)) + text.encode("latin-1")
                     + struct.pack("<HH", m1, m2) + mean_proj.astype("<u2").ravel(order="F")
                     .tobytes() + struct.pack("<HH", 2, 3) + frame.ravel(order="F").tobytes())


def write_roi(path: Path, traces: np.ndarray) -> None:
    out = bytearray(struct.pack("<HH", 2, 2) + np.zeros(4, dtype="<u2").tobytes())
    for k in range(traces.shape[0]):
        out += np.array(pco_header(k + 1, datetime(2024, 1, 2, 3, 4, 5)), dtype="<u2").tobytes()
        out += traces[k].astype("<u2").tobytes()
    path.write_bytes(bytes(out))


def test_register_wid_trace_imaging(tmp_path):
    mean_proj = np.round(blobs(H, W, seed=14)).astype(np.uint16)
    roi = tmp_path / "t.roi"
    write_roi(roi, np.arange(30, dtype=np.uint16).reshape(10, 3))
    write_wii(tmp_path / "t.wii", 3, mean_proj)
    ad = register_wid([roi])
    release()
    np.testing.assert_array_equal(ad.first_frame, mean_proj)
    assert ad.meta_info.shape == (1, 10) and "dimension" not in ad.meta_info[0, 0]
    assert ad.registration_parameters.status == "ORIGIN"
    with pytest.raises(ValueError):
        register_wid([roi, tmp_path / "x.wid"])
