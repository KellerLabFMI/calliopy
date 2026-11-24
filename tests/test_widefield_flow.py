"""Tests for calliopy.widefield._flow (Velocity_fields, WF_grid2vid.m)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.ndimage
import scipy.signal

from calliopy._matlab import imgaussfilt
from calliopy.io import MatStruct
from calliopy.widefield._flow import (
    horn_schunck,
    match_rois_to_img_indices,
    quiver_video,
    uv_img_to_linear,
    uv_linear_to_img,
    velocity_field_horn_schunck,
    wf_grid2vid,
)

# --- Horn-Schunck ---------------------------------------------------------------------------


def _pattern(dx=0.0, dy=0.0, n=40):
    y, x = np.mgrid[0:n, 0:n].astype(float)
    x, y = x - dx, y - dy
    return (
        np.sin(2 * np.pi * x / 13)
        + np.cos(2 * np.pi * y / 11)
        + 0.5 * np.sin(2 * np.pi * (x + y) / 17)
    )


def _ref_hs(im1, im2, alpha, n_iter):
    """Independent route: explicit cube differences on zero-padded images (paper eq. 7-9)."""
    a = np.pad(im1, ((0, 1), (0, 1)))
    b = np.pad(im2, ((0, 1), (0, 1)))

    def at(e, di, dj):
        return e[di : di + im1.shape[0], dj : dj + im1.shape[1]]

    ex = 0.25 * sum(at(e, 0, 1) - at(e, 0, 0) + at(e, 1, 1) - at(e, 1, 0) for e in (a, b))
    ey = 0.25 * sum(at(e, 1, 0) - at(e, 0, 0) + at(e, 1, 1) - at(e, 0, 1) for e in (a, b))
    et = 0.25 * sum(at(b, i, j) - at(a, i, j) for i in (0, 1) for j in (0, 1))
    w = np.array([[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]])
    u = np.zeros_like(im1)
    v = np.zeros_like(im1)
    for _ in range(n_iter):
        ub = scipy.ndimage.correlate(u, w, mode="constant", cval=0.0)
        vb = scipy.ndimage.correlate(v, w, mode="constant", cval=0.0)
        r = (ex * ub + ey * vb + et) / (alpha**2 + ex**2 + ey**2)
        u, v = ub - ex * r, vb - ey * r
    return u, v


def _conv2_same(a, k):
    full = scipy.signal.convolve2d(a, k, mode="full")
    r0, c0 = k.shape[0] // 2, k.shape[1] // 2
    return full[r0 : r0 + a.shape[0], c0 : c0 + a.shape[1]]


def test_hs_matches_independent_route():
    rng = np.random.default_rng(0)
    im1 = rng.random((9, 12))
    im2 = rng.random((9, 12))
    u, v = horn_schunck(im1, im2, 0.7, 25)
    ur, vr = _ref_hs(im1, im2, 0.7, 25)
    np.testing.assert_allclose(u, ur, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(v, vr, rtol=1e-10, atol=1e-12)


def test_hs_invariant_to_negated_derivatives():
    # derivatives with all three signs reversed (another common convention) give the same flow
    rng = np.random.default_rng(1)
    im1 = rng.random((8, 10))
    im2 = rng.random((8, 10))
    kx = 0.25 * np.array([[-1.0, 1.0], [-1.0, 1.0]])
    ky = 0.25 * np.array([[-1.0, -1.0], [1.0, 1.0]])
    fx = _conv2_same(im1, kx) + _conv2_same(im2, kx)
    fy = _conv2_same(im1, ky) + _conv2_same(im2, ky)
    ft = _conv2_same(im1, 0.25 * np.ones((2, 2))) - _conv2_same(im2, 0.25 * np.ones((2, 2)))
    w = np.array([[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]])
    u = np.zeros_like(im1)
    v = np.zeros_like(im1)
    for _ in range(30):
        ub, vb = _conv2_same(u, w), _conv2_same(v, w)
        r = (fx * ub + fy * vb + ft) / (0.3**2 + fx**2 + fy**2)
        u, v = ub - fx * r, vb - fy * r
    got = horn_schunck(im1, im2, 0.3, 30)
    np.testing.assert_allclose(got.u, u, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(got.v, v, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize(("dx", "dy"), [(0.3, -0.2), (-0.25, 0.0), (0.0, 0.4)])
def test_hs_recovers_translation(dx, dy):
    u, v = horn_schunck(_pattern(), _pattern(dx, dy), 0.005, 100)
    c = slice(8, 32)  # away from the zero-padded border
    assert np.median(u[c, c]) == pytest.approx(dx, abs=0.03)
    assert np.median(v[c, c]) == pytest.approx(dy, abs=0.03)


def test_hs_identical_frames_zero_flow():
    im = _pattern()
    u, v = horn_schunck(im, im, 0.005, 50)
    assert np.all(u == 0) and np.all(v == 0)


def test_hs_alpha_zero_nan_to_zero_and_initial_values():
    im = np.zeros((5, 5))
    u, v = horn_schunck(im, im, 0.0, 3)  # 0/0 everywhere -> NaN -> 0
    assert np.all(u == 0) and np.all(v == 0)
    u0 = np.full((5, 5), 0.5)
    got = horn_schunck(im + 1, im + 1, 1.0, 0, u_initial=u0, v_initial=-u0)
    np.testing.assert_array_equal(got.u, u0)
    np.testing.assert_array_equal(got.v, -u0)
    with pytest.raises(ValueError):
        horn_schunck(np.zeros((3, 3)), np.zeros((3, 4)))


def test_velocity_field_stack():
    frames = [_pattern(0.2 * k, 0.1 * k, n=24) for k in range(4)]
    vid = np.stack(frames, axis=2)
    vid[0, 0, 1] = np.nan
    vid_in = vid.copy()
    U, V = velocity_field_horn_schunck(vid, alpha=0.01, max_iter=20, sigma=1)
    np.testing.assert_array_equal(vid, vid_in, strict=True)  # input untouched (NaN kept)
    assert U.shape == vid.shape and U.dtype == np.float64
    filled = np.nan_to_num(vid, nan=0.0)
    for i in range(3):
        a = imgaussfilt(filled[:, :, i], 1)
        b = imgaussfilt(filled[:, :, i + 1], 1)
        u, v = horn_schunck(a, b, 0.01, 20)
        np.testing.assert_array_equal(U[:, :, i], u)
        np.testing.assert_array_equal(V[:, :, i], v)
    np.testing.assert_array_equal(U[:, :, 3], U[:, :, 2])
    np.testing.assert_array_equal(V[:, :, 3], V[:, :, 2])


def test_velocity_field_needs_two_frames():
    with pytest.raises(ValueError):
        velocity_field_horn_schunck(np.zeros((4, 4, 1)))
    with pytest.raises(ValueError):
        velocity_field_horn_schunck(np.zeros((4, 4)))


# --- ROI grid --------------------------------------------------------------------------------

# (AP, ML) per ROI; the grid spans AP -1..2 (H = 4) and ML -2..1 (W = 4)
_POS = [(-1, -2), (0, -2), (2, -2), (1, 0), (2, 1), (0, 1), (-1, 0)]


def _rois(with_corner=False):
    rois = [MatStruct(AP_index=np.array([[ap]], float), ML_index=float(ml)) for ap, ml in _POS]
    if with_corner:  # a corner ROI: no AP_index, but its ML_index still counts for min/max
        rois.insert(2, MatStruct(AP_index=np.zeros((0, 0)), ML_index=np.array([[3.0]])))
    return rois


def test_match_rois_matlab_formula():
    rois = _rois()
    H, W, min_ap, min_ml = 4, 4, -1, -2
    got = match_rois_to_img_indices(rois)
    row1 = [H + min_ap - ap for ap, _ in _POS]  # MATLAB sub2ind arguments (1-based)
    col1 = [ml - min_ml + 1 for _, ml in _POS]
    expected = [r + (c - 1) * H for r, c in zip(row1, col1, strict=True)]
    assert got.inds_on_img.shape == (1, len(_POS))
    np.testing.assert_array_equal(got.inds_on_img[0], expected)
    r, c = np.unravel_index(got.inds_on_img[0] - 1, (H, W), order="F")
    np.testing.assert_array_equal(r, got.rows)
    np.testing.assert_array_equal(c, got.cols)
    sub = match_rois_to_img_indices(rois, [3, 0])
    np.testing.assert_array_equal(sub.inds_on_img[0], [expected[3], expected[0]])
    assert match_rois_to_img_indices(rois, []).inds_on_img.shape == (0, 0)
    # object array (1 x N, MATLAB struct array) is read column-major
    arr = np.empty((1, len(rois)), dtype=object)
    for i, s in enumerate(rois):
        arr[0, i] = s
    np.testing.assert_array_equal(match_rois_to_img_indices(arr).inds_on_img, got.inds_on_img)


def test_match_rois_corner_roi():
    rois = _rois(with_corner=True)
    with pytest.raises(ValueError, match="AP_index"):
        match_rois_to_img_indices(rois)
    got = match_rois_to_img_indices(rois, [0, 1, 3])
    # W now spans ML -2..3 = 6 columns because the corner's ML_index counts (as in MATLAB)
    a = wf_grid2vid(np.arange(1, 9, dtype=float), rois)
    assert a.shape == (4, 6, 1)
    np.testing.assert_array_equal(a[got.rows, got.cols, 0], [1, 2, 4])
    assert np.count_nonzero(a) == 7  # the corner ROI (value 3) is skipped


def test_match_rois_non_integer_index():
    rois = _rois()
    rois[0]["AP_index"] = 0.5
    with pytest.raises(ValueError):
        match_rois_to_img_indices(rois)


def test_uv_round_trip_and_grid2vid():
    rois = _rois()
    rng = np.random.default_rng(2)
    us = rng.standard_normal((len(rois), 3))
    vs = rng.standard_normal((len(rois), 3))
    U, V = uv_linear_to_img(us, vs, rois)
    assert U.shape == (4, 4, 3)
    assert np.isnan(U).sum() == (16 - len(rois)) * 3
    back = uv_img_to_linear(U, V, rois)
    np.testing.assert_array_equal(back.U_selected, us)
    np.testing.assert_array_equal(back.V_selected, vs)
    A = wf_grid2vid(us, rois)
    np.testing.assert_array_equal(A[~np.isnan(U)], U[~np.isnan(U)])
    assert np.all(A[np.isnan(U)] == 0)
    # the top row is the largest AP (anterior up), the first column the smallest ML
    assert A[0, 0, 0] == us[2, 0]  # AP 2, ML -2
    with pytest.raises(ValueError):
        uv_linear_to_img(us[:-1], vs[:-1], rois)
    with pytest.raises(ValueError):
        uv_img_to_linear(U[:3], V[:3], rois)


def test_grid2vid_resize():
    rois = _rois()
    act = np.arange(len(rois) * 2, dtype=float).reshape(len(rois), 2)
    A = wf_grid2vid(act, rois)
    A3 = wf_grid2vid(act, rois, 3)
    np.testing.assert_array_equal(A3, np.repeat(np.repeat(A, 3, axis=0), 3, axis=1))
    assert wf_grid2vid(act, rois, 1.5).shape == (6, 6, 2)
    with pytest.raises(ValueError):
        wf_grid2vid(act[:3], rois)


# --- quiver video ---------------------------------------------------------------------------


def _check_line(pixels, x1, y1, x2, y2):
    """Properties of a Bresenham line: endpoints, max(dx, dy)+1 pixels, deviation <= 0.5."""
    assert (y1, x1) in pixels and (y2, x2) in pixels
    n = max(abs(x2 - x1), abs(y2 - y1)) + 1
    line = [
        (y, x)
        for (y, x) in pixels
        if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)
    ]
    dist = [
        abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)) / max(abs(x2 - x1), abs(y2 - y1))
        for (y, x) in line
    ]
    assert len([d for d in dist if d <= 0.5 + 1e-12]) == n


def test_quiver_single_arrow():
    img = quiver_video(np.full((1, 1, 1), 0.1), np.zeros((1, 1, 1)))
    assert img.shape == (30, 30, 1) and img.dtype == np.uint8
    on = {(int(r) + 1, int(c) + 1) for r, c in zip(*np.nonzero(img[:, :, 0]), strict=True)}
    assert set(np.unique(img)) == {0, 255}
    # main line: start (x, y) = (15, 15), end (25, 15), 1-based
    assert {(15, x) for x in range(15, 26)} <= on
    # head: end - round(5 * (cos, sin)(+-30 deg)); 5 * sin(pi/6) is 2.4999999999999996 in
    # float64, so the offsets are (4, 2): heads end at (21, 13) and (21, 17)
    hx = [25 - int(np.floor(5 * math.cos(a) + 0.5)) for a in (math.pi / 6, -math.pi / 6)]
    hy = [15 - int(np.sign(s) * np.floor(abs(5 * s) + 0.5))
          for s in (math.sin(math.pi / 6), math.sin(-math.pi / 6))]
    assert (hx, hy) == ([21, 21], [13, 17])
    _check_line(on, 25, 15, hx[0], hy[0])
    _check_line(on, 25, 15, hx[1], hy[1])
    # each head has 5 pixels; its first two, (25, 15) and (24, 15), lie on the main line
    assert len(on) == 11 + 3 + 3


def test_quiver_direction_thickness_nan():
    down = quiver_video(np.zeros((1, 1, 1)), np.full((1, 1, 1), 0.1))
    assert down[24, 14, 0] == 255 and down[5, 14, 0] == 0  # V > 0 points down (rows)
    thick = quiver_video(np.full((1, 1, 1), 0.1), np.zeros((1, 1, 1)), thickness=3)
    assert np.all(thick[13:16, 13:26, 0] == 255)
    none = quiver_video(np.full((2, 2, 1), np.nan), np.zeros((2, 2, 1)))
    assert not none.any()
    odd = quiver_video(np.full((1, 1, 1), 0.1), np.zeros((1, 1, 1)), spacing=31)
    assert odd[15, 15, 0] == 255  # start rounded from 15.5 to 16 (1-based)
    clipped = quiver_video(np.full((1, 1, 1), 5.0), np.zeros((1, 1, 1)))
    assert clipped[14, 14:, 0].all()  # runs off the image without error
    with pytest.raises(ValueError):
        quiver_video(np.full((1, 1, 1), np.inf), np.zeros((1, 1, 1)))


def test_quiver_background():
    bg = np.array([[0.5, 2.0], [1.0, 0.0]])[:, :, None]
    U = np.full((2, 2, 2), np.nan)
    U[0, 0, 1] = 0.05
    V = np.zeros((2, 2, 2))
    img = quiver_video(U, V, spacing=10, vec_length=100, bg_mat=bg)
    rep = np.repeat(np.repeat(bg, 10, axis=0), 10, axis=1)[:, :, 0]
    np.testing.assert_array_equal(img[:, :, 0], rep)  # no arrows in frame 0
    drawn = img[:, :, 1] != rep
    assert drawn.any() and np.all(img[:, :, 1][drawn] == 4.0)  # 2 * max(bg)
    bg8 = (bg * 100).astype(np.uint8)
    img8 = quiver_video(U, V, spacing=10, bg_mat=bg8)
    assert img8.dtype == np.uint8 and img8.max() == 255  # 2 * 200 saturates
    with pytest.raises(ValueError):
        quiver_video(U, V, bg_mat=np.zeros((2, 2, 3)))


def test_quiver_roi_matrices():
    rois = _rois()
    rng = np.random.default_rng(5)
    us = rng.uniform(-0.1, 0.1, (len(rois), 2))
    vs = rng.uniform(-0.1, 0.1, (len(rois), 2))
    act = rng.random((len(rois), 2))
    got = quiver_video(us, vs, spacing=12, bg_mat=act, roi_info=rois)
    U, V = uv_linear_to_img(us, vs, rois)
    want = quiver_video(U, V, spacing=12, bg_mat=wf_grid2vid(act, rois))
    np.testing.assert_array_equal(got, want)
    with pytest.raises(ValueError):
        quiver_video(us, vs)
