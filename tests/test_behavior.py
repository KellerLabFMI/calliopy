"""Synthetic tests for calliopy.behavior (calliope 10_Behavior)."""

from __future__ import annotations

import numpy as np
import pytest
from calliopy.behavior import (
    find_saccades,
    leinweber_performance,
    map_eye_data_to_frame_times,
    preprocess_owl,
    rgb2gray_uint8,
)
from calliopy.behavior._owl import _largest_blob_centroid


def test_leinweber_performance():
    v = np.ones(10)
    assert leinweber_performance(v, 0 * v, np.full(10, 2.5)) == pytest.approx(1.0)
    assert leinweber_performance(v, 0 * v, np.zeros(10)) == pytest.approx(-1.0)
    vm = np.r_[np.ones(5), np.zeros(5)]
    assert leinweber_performance(vm, 0 * vm, np.full(10, 2.5)) == pytest.approx(0.5)
    # one extra angle sample: the first is dropped
    ang = np.r_[0.0, np.full(10, 2.5)]
    assert leinweber_performance(v, 0 * v, ang) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        leinweber_performance(v, v, np.ones(4))


def test_map_eye_data_to_frame_times():
    it = np.array([0.0, 1.0, 2.0, 3.0])
    ft = np.array([0.4, 0.5, 2.9, -5.0])
    pd = np.array([10.0, 11, 12, 13])
    pos = np.vstack([pd, pd + 100])
    r = map_eye_data_to_frame_times(pd, pos, pd, pd, pos, ft, it)
    np.testing.assert_array_equal(r.pupil_diam, [[10, 10, 13, 10]])  # tie 0.5 -> first
    assert r.pupil_pos.shape == (2, 4) and r.whisker_pos[1, 2] == 113
    it_nan = np.array([np.nan, 1.0, 2.0, 3.0])
    r2 = map_eye_data_to_frame_times(pd, pos, pd, pd, pos, ft, it_nan)
    np.testing.assert_array_equal(r2.blink, [[11, 11, 13, 11]])


def _eye_movie(n_frames=60, jump_at=30):
    h, w = 100, 120
    rr, cc = np.mgrid[1 : h + 1, 1 : w + 1]
    movie = np.full((h, w, n_frames), 50, dtype=np.uint8)
    for k in range(n_frames):
        cr = 50 if k + 1 < jump_at else 58
        disk = (rr - cr) ** 2 + (cc - 60) ** 2 <= 15**2
        movie[:, :, k][disk] = 200
    return movie


def test_find_saccades_tracks_bright_pupil():
    movie = _eye_movie()
    res = find_saccades(movie)
    assert res.pupil_pos.shape == (2, 120)  # length(idata) = longest dimension
    assert res.pupil_diam.shape == (120, 1) and res.facial_mov.shape == (60, 1)
    np.testing.assert_allclose(res.pupil_pos[:, 9], [50, 60], atol=0.5)
    np.testing.assert_allclose(res.pupil_pos[:, 45], [58, 60], atol=0.5)
    assert 15 <= res.pupil_diam[20, 0] <= 18
    assert np.any(np.abs(res.saccade_frames.ravel() - 30) <= 1)
    assert res.saccade_frames.shape[1] == 1
    np.testing.assert_array_equal(res.pupil_diam[:3, 0], 0)  # frames 1-3 untouched


def test_find_saccades_facial_movement_saturates_for_uint8():
    movie = _eye_movie()
    movie[:50, :50, 40:] += 10  # top-left corner brightens at frame 41, then stays
    movie[:50, :50, 50:] -= 10  # darkens again at frame 51: uint8 diff saturates to 0
    res = find_saccades(movie)
    fm = res.facial_mov.ravel()
    assert fm[39] == pytest.approx(10 * 2500 / 4)
    assert fm[49] == 0.0
    assert fm[-1] == 0.0
    res_f = find_saccades(movie.astype(np.float64))
    assert res_f.facial_mov.ravel()[49] == pytest.approx(-10 * 2500 / 4)
    with pytest.raises(ValueError):
        find_saccades(np.zeros((40, 40, 10)))


def test_rgb2gray_uint8():
    rng = np.random.default_rng(0)
    f = rng.integers(0, 256, size=(4, 5, 3), dtype=np.uint8)
    w = np.array([0.298936021293775, 0.587043074451121, 0.114020904255103])
    exp = np.floor(f.astype(float) @ w + 0.5).astype(np.uint8)
    np.testing.assert_array_equal(rgb2gray_uint8(f), exp)
    gray = np.full((2, 2, 3), 200, dtype=np.uint8)
    np.testing.assert_array_equal(rgb2gray_uint8(gray), 200)


def test_largest_blob_order_is_column_major():
    m = np.zeros((40, 40), dtype=bool)
    m[1:4, 30:33] = True  # top right: first in row-major order
    m[30:33, 1:4] = True  # bottom left: first in column-major order (MATLAB)
    x, y = _largest_blob_centroid(m, 1)
    assert (x, y) == (3.0, 32.0)
    m[5, 5] = True
    assert _largest_blob_centroid(m, 10) is None


def test_preprocess_owl_tracks_dark_mouse():
    h, w, t = 60, 80, 12
    frames = np.full((h, w, 3, t), 200, dtype=np.uint8)
    for k in range(t):
        r0, c0 = 10 + 2 * k, 20 + 3 * k
        frames[r0 : r0 + 10, c0 : c0 + 10, :, k] = 40
    frames[50:53, 5:8, :, 3] = 40  # 9-pixel speck, removed by bwareaopen
    out = preprocess_owl(frames, n_samples=50)
    owl = out.owl
    xs, ys = owl.KP.mouse_x.ravel(), owl.KP.mouse_y.ravel()
    np.testing.assert_allclose(xs, 20 + 3 * np.arange(t) + 5.5)  # 1-based centroid
    np.testing.assert_allclose(ys, 10 + 2 * np.arange(t) + 5.5)
    assert owl.median_frame.dtype == np.uint8 and owl.median_frame[0, 0] == 200
    assert owl.first_frame.shape == (h, w, 3)
    assert owl.last_frame.shape == (h, w)
    gray_out = preprocess_owl(frames[:, :, 0, :])
    np.testing.assert_array_equal(gray_out.owl.KP.mouse_x, owl.KP.mouse_x)
    empty = np.full((h, w, 5), 200, dtype=np.uint8)
    assert np.isnan(preprocess_owl(empty).owl.KP.mouse_x).all()
