"""Synthetic tests of calliopy.widefield._quality.

image_similarity_snippets (and its figure), image_similarity, wid_mean_roi_trace,
activity_similarity, quality_check and the wid_adata_check wrapper.
"""

from __future__ import annotations

import gc
import struct
from pathlib import Path

import numpy as np
import pytest
import scipy.ndimage
from matplotlib.figure import Figure

from calliopy.io import MatStruct
from calliopy.widefield._activity import calc_widefield_act
from calliopy.widefield._quality import (
    activity_similarity,
    image_similarity,
    image_similarity_snippets,
    plot_image_similarity_snippets,
    quality_check,
    wid_adata_check,
    wid_mean_roi_trace,
)
from calliopy.widefield._transform import similarity_matrix

RNG = np.random.default_rng(424242)


def _texture(shape=(520, 560), seed=1):
    rng = np.random.default_rng(seed)
    return scipy.ndimage.gaussian_filter(rng.normal(size=shape), 3.0) * 1000 + 5000


def write_wid(path: Path, imgs: np.ndarray) -> None:
    size_y, size_x, n = imgs.shape
    out = bytearray()
    for k in range(n):
        out += struct.pack("<HH", size_y, size_x)
        out += imgs[:, :, k].astype("<u2").ravel(order="F").tobytes()
    path.write_bytes(bytes(out))


def _cell(structs):
    arr = np.empty((1, len(structs)), dtype=object)
    for j, s in enumerate(structs):
        arr[0, j] = s
    cell = np.empty((1, 1), dtype=object)
    cell[0, 0] = arr
    return cell


# --- image_similarity_snippets --------------------------------------------------------------


def test_snippets_identical_images_give_zero_shift():
    img = _texture()
    res = image_similarity_snippets(img, img, offsets=np.zeros((3, 2), dtype=int))
    # matched centre = snippet centre (MATLAB would give sqrt(2), see the docstring)
    assert res.average_shift == 0.0
    np.testing.assert_array_equal(res.matched_locations, res.snippet_centers)
    rnd = image_similarity_snippets(img, img, rng=np.random.default_rng(7))
    assert rnd.average_shift == 0.0
    np.testing.assert_array_equal(rnd.matched_locations, rnd.snippet_centers)
    # centre: round(520/2)=260, round(560/2)=280; start = centre - 50; centre = start + 50
    np.testing.assert_array_equal(res.snippet_centers, [[260, 280]] * 3)


def test_snippets_shifted_image_random_offsets():
    img = _texture()
    shifted = np.roll(img, (5, -3), axis=(0, 1))  # content 5 rows down, 3 columns left
    res = image_similarity_snippets(img, shifted, rng=np.random.default_rng(3))
    assert res.snippet_centers.shape == (100, 2) and res.matched_locations.shape == (100, 2)
    np.testing.assert_array_equal(res.matched_locations - res.snippet_centers,
                                  np.tile([5, -3], (100, 1)))
    assert res.average_shift == pytest.approx(np.sqrt(34.0), abs=1e-12)
    # centres lie within +-200 of the image centre, as integers
    assert np.all(np.abs(res.snippet_centers - [260, 280]) <= 200)
    assert np.all(res.snippet_centers == np.round(res.snippet_centers))
    # the same generator state gives the same snippets
    again = image_similarity_snippets(img, shifted, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(again.snippet_centers, res.snippet_centers)
    other = image_similarity_snippets(img, shifted, rng=np.random.default_rng(4))
    assert not np.array_equal(other.snippet_centers, res.snippet_centers)


def test_snippets_clamped_to_image():
    img = _texture((300, 310))
    res = image_similarity_snippets(img, img, offsets=[[-200, -200], [200, 200], [-200, 200]])
    # r0 = 150 - 50 - 200 -> 1 (centre 51); r0 = 300 -> 300-100+1 = 201 (centre 251);
    # columns: centre 155: 155-50-200 -> 1 (51); 305 -> 211 (261)
    np.testing.assert_array_equal(res.snippet_centers, [[51, 51], [251, 261], [51, 261]])
    np.testing.assert_array_equal(res.matched_locations, res.snippet_centers)


def test_snippets_errors():
    img = _texture((150, 150))
    with pytest.raises(ValueError, match="2-D"):
        image_similarity_snippets(img[:, :, None], img)
    with pytest.raises(ValueError, match="at least"):
        image_similarity_snippets(img[:99], img)
    flat = np.ones((150, 150))
    with pytest.raises(ValueError):
        image_similarity_snippets(flat, img, offsets=[[0, 0]])


def test_plot_image_similarity_snippets():
    img = _texture((200, 220))
    res = image_similarity_snippets(img, img, offsets=[[0, 0], [10, -5]])
    fig = plot_image_similarity_snippets(img, img, res)
    assert isinstance(fig, Figure) and len(fig.axes) == 2
    x, y = fig.axes[0].lines[0].get_data()
    np.testing.assert_array_equal(x, res.snippet_centers[:, 1] - 1)
    np.testing.assert_array_equal(y, res.snippet_centers[:, 0] - 1)


# --- image_similarity -----------------------------------------------------------------------


def test_image_similarity_identity_and_translation():
    img = _texture()
    rng = np.random.default_rng(9)
    same = image_similarity(img, np.eye(3), img, rng=rng)
    assert same == 1.0
    # tform moves content 3 columns right (x) and 2 rows up (y); the site frame shows the
    # content there: the warped frame then matches the site frame
    tform = similarity_matrix(1.0, 0.0, [3.0, -2.0])
    site = np.roll(img, (-2, 3), axis=(0, 1))
    got = image_similarity(img, tform, site, rng=np.random.default_rng(9))
    assert got == 1.0
    # without the warp the frames are (-2, 3) px apart
    off = image_similarity(img, np.eye(3), site, rng=np.random.default_rng(9))
    assert off == pytest.approx(1 - np.hypot(2, 3) / 100, abs=1e-12)
    # integer frames as in adata (uint16) work
    got16 = image_similarity(img.astype(np.uint16), tform, site.astype(np.uint16),
                             rng=np.random.default_rng(9))
    assert got16 == 1.0


# --- activity similarity --------------------------------------------------------------------


def test_wid_mean_roi_trace():
    data = RNG.integers(0, 1000, size=(6, 7, 9), dtype=np.uint16)
    rois = [[1], [], [3, 4, 40], np.array([[10.0], [11.0]])]
    got = wid_mean_roi_trace(data, rois, chunk_frames=4)
    assert got.dtype == np.float32 and got.shape == (9,)
    flat = data.reshape(-1, 9, order="F").astype(np.float64)
    per = [flat[0], flat[[2, 3, 39]].mean(0), flat[[9, 10]].mean(0)]  # pixel-1 ROI counts here
    np.testing.assert_allclose(got, np.mean(per, axis=0), rtol=1e-6)
    assert np.isnan(wid_mean_roi_trace(data, [[]])).all()


def test_activity_similarity():
    t = RNG.normal(size=50)
    acts = [(2 * t + 1).reshape(-1, 1), np.empty((0, 0)), (4 * t - 3).reshape(-1, 1).astype(
        np.float32), np.full((50, 1), np.nan)]
    assert activity_similarity(acts, t) == pytest.approx(1.0, abs=1e-6)
    noisy = t + RNG.normal(scale=0.5, size=50)
    want = np.corrcoef(np.nanmean(np.stack([2 * t + 1, 4 * t - 3], 1), 1), noisy)[0, 1]
    assert activity_similarity(acts, noisy) == pytest.approx(want, rel=1e-6)
    with_nan = t.copy()
    with_nan[3] = np.nan
    assert np.isnan(activity_similarity(acts, with_nan))
    assert np.isnan(activity_similarity([np.full((50, 1), np.nan)], t))
    with pytest.raises(ValueError):
        activity_similarity(acts, t[:-1])
    with pytest.raises(ValueError):
        activity_similarity([t, t[:-1]], t)
    with pytest.raises(ValueError):
        activity_similarity([np.empty((0, 0))], t)


def test_quality_check_rule():
    ok = quality_check(0.99995, 0.9)
    assert list(ok.keys()) == ["passed", "img_similarity", "act_similarity"]
    assert ok.passed == 1.0 and isinstance(ok.passed, np.float64)
    assert quality_check(0.9999, 0.9).passed == 0.0      # strict >
    assert quality_check(0.99995, 0.85).passed == 0.0
    assert quality_check(float("nan"), 1.0).passed == 0.0
    assert quality_check(1.0, 0.86).passed == 1.0


# --- wid_adata_check ------------------------------------------------------------------------


def _experiment(tmp_path, n_rois=8, two=False):
    h, w = 16, 14
    rng = np.random.default_rng(11)
    idx = [(rng.choice(np.arange(2, h * w + 1), size=6, replace=False) * 1.0).reshape(-1, 1)
           for _ in range(n_rois)]
    idx[2] = np.empty((0, 0))
    fields = ("indices_ch525", "indices_ch610") if two else ("indices",)
    structs = [MatStruct({f: ix for f in fields}) for ix in idx]
    base = rng.integers(100, 4000, size=(h, w, 1))
    stacks = {}
    for ch in ("ch525", "ch610") if two else ("",):
        paths = []
        for k, n in enumerate((5, 4)):
            p = tmp_path / f"s{ch}{k}.wid"
            write_wid(p, (base + rng.integers(0, 3000, size=(h, w, n))).astype(np.uint16))
            paths.append(p)
        stacks[ch] = paths
    if two:
        res = calc_widefield_act(_cell(structs), (h, w), wid_fnames_ch525=stacks["ch525"],
                                 wid_fnames_ch610=stacks["ch610"])
        return res.ROIs, {"ch525": stacks["ch525"], "ch610": stacks["ch610"]}
    res = calc_widefield_act(_cell(structs), (h, w), wid_fnames=stacks[""])
    return res.ROIs, stacks[""]


def test_wid_adata_check_one_channel(tmp_path):
    rois, paths = _experiment(tmp_path)
    check = wid_adata_check(rois, paths, is_site=True, chunk_frames=2)
    assert check.passed == 1.0 and check.img_similarity == 1.0
    assert check.act_similarity == pytest.approx(1.0, abs=1e-6)
    # not the site: image similarity from the frames (identical frames -> exactly 1)
    ff = _texture((260, 240))
    check2 = wid_adata_check(rois, paths, first_frame=ff, tform=np.eye(3), site_first_frame=ff,
                             rng=np.random.default_rng(0))
    assert check2.img_similarity == 1.0
    assert check2.passed == 1.0
    # wrong stored activity (stacks in the wrong order) fails
    check3 = wid_adata_check(rois, paths[::-1], img_similarity=1.0)
    assert check3.passed == 0.0
    with pytest.raises(ValueError, match="required"):
        wid_adata_check(rois, paths)
    bare = _cell([MatStruct(indices=np.array([[2.0]]))] * 5)
    with pytest.raises(ValueError, match="not calculated"):
        wid_adata_check(bare, paths, is_site=True)
    gc.collect()


def test_wid_adata_check_two_channels(tmp_path):
    rois, paths = _experiment(tmp_path, n_rois=390, two=True)
    with pytest.raises(ValueError, match="two-channel site"):
        wid_adata_check(rois, paths, is_site=True)
    check = wid_adata_check(rois, paths, is_site=True, img_similarity=1.0)
    assert check.passed == 1.0 and check.act_similarity == pytest.approx(1.0, abs=1e-6)
    ff = _texture(seed=2)  # large enough that no snippet reaches the zero-filled border
    frames = {"ch525": ff, "ch610": ff}
    tf = {"ch525": np.eye(3), "ch610": similarity_matrix(1.0, 0.0, [4.0, 0.0])}
    check2 = wid_adata_check(rois, paths, first_frame=frames, tform=tf, site_first_frame=frames,
                             rng=np.random.default_rng(1))
    # minimum over channels: the warped ch610 frame is 4 columns right of the site frame, so
    # its snippets are found 4 columns left: shift (0, -4)
    assert check2.img_similarity == pytest.approx(1 - 4 / 100, abs=1e-12)
    gc.collect()
