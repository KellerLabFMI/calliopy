"""Synthetic tests of calliopy.oii (Fourier retinotopy, wave-stimulus maps, patch maps)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from matplotlib.figure import Figure
from scipy.signal import lfilter

from calliopy.oii import (
    analyze_oii_fourier,
    fourier_maps,
    oii_normalize,
    oii_patch_condition_images,
    oii_stim_bounds,
    oii_stim_conditions,
    oii_stim_epochs,
    plot_fourier_maps,
    plot_patch_montage,
    power_overlay,
    smooth2,
    wave_reference_image,
    wave_stimulus_maps,
)
from calliopy.oii._fourier import _smooth_rows
from calliopy.twophoton._builtins import smooth_moving


def wrap(p):
    return np.mod(p + np.pi, 2 * np.pi) - np.pi


def test_smooth2_matches_matlab_algorithm():
    rng = np.random.default_rng(0)
    v = rng.normal(size=50)
    w = 7
    pad = np.r_[np.full(4, v[0]), v, np.full(3, v[-1])]
    ref = lfilter(np.ones(w) / w, 1, pad)[w:]
    np.testing.assert_allclose(smooth2(v, w), ref)
    np.testing.assert_allclose(smooth2(v[:, None], w)[:, 0], ref)  # column: dim 1 in MATLAB
    A = rng.normal(size=(30, 4))
    out = smooth2(A, 5)
    np.testing.assert_allclose(out[:, 2], smooth2(A[:, 2], 5))


def test_smooth_rows_matches_1d():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(3, 41))
    for span in (5, 6, 80.7):
        np.testing.assert_allclose(_smooth_rows(y, span)[2], smooth_moving(y[2], span))


def _aux_and_frames(hfreq=0.125, vfreq=0.2):
    n = 86000
    aux = np.zeros((4, n + 499))
    aux[0, 499:] = 5  # shutter opens: crop from 1-based sample 500
    sid = np.zeros(n)
    pos = np.zeros(n)
    bounds = [(1000, 21000), (22000, 42000), (43000, 63000), (64000, 84000)]
    for k, (a, b) in enumerate(bounds):
        sid[a:b] = k + 1
        per = 1000 / (hfreq if k < 2 else vfreq)
        pos[a:b] = 4 * np.mod(np.arange(b - a), per) / per
    aux[2, 499:] = sid
    aux[1, 499:] = pos
    fr = np.zeros((4, 840))
    fr[0] = 50 + 100 * np.arange(840)
    return aux, fr


def test_stim_bounds_and_epochs():
    aux, fr = _aux_and_frames()
    cropped, sb = oii_stim_bounds(aux)
    # MATLAB crops from the last low sample (strfind(x, [0 1])), so one sample more
    assert cropped.shape[1] == aux.shape[1] - 498
    np.testing.assert_array_equal(sb, [1001, 21001, 22001, 42001, 43001, 63001, 64001, 84001, 86001])
    frm = fr[0].copy()
    frm[0] = 1
    ep = oii_stim_epochs(frm, cropped, sb)
    assert ep.start[0] == 10 and ep.stop[0] == 209
    np.testing.assert_allclose(ep.stimfreqs, [0.125, 0.125, 0.2, 0.2], rtol=1e-3)
    swapped, _ = oii_stim_bounds(aux, swap_stim_channels=True)
    np.testing.assert_array_equal(swapped[1], cropped[2])


def test_analyze_oii_fourier_end_to_end():
    aux, fr = _aux_and_frames()
    h, w = 16, 20
    t = fr[0] / 1000
    col = np.arange(w)[None, :, None]
    dat = 1000 + 100 * np.cos(2 * np.pi * 0.125 * t[None, None, :] + 0.2 * col)
    dat = np.broadcast_to(dat, (h, w, t.size)).round().astype(np.uint16)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = analyze_oii_fourier(dat, fr, aux, 0.125, 0.2)
    r = res.result
    assert r["Power"].shape == (8, 10) and r["Ibv"].shape == (8, 10)
    assert np.nanmin(r["Power"]) >= 0 and np.nanmax(r["Power"]) <= 1
    assert len(r["FourierSpecs"]) == 4 and r["FourierSpecs"][0].dtype == np.complex64
    assert res.dat.dtype == np.float32
    assert res.fr[0, 0] == 1
    assert res.sampling_freq == pytest.approx(1000 / ((fr[0, -1] - 1) / 839))
    sel = analyze_oii_fourier(dat, fr, aux, 0.125, 0.2, frequency_select_range=1, selected=1,
                              spatial_filter_std=1, normalize="Median")
    assert sel.result["FourierSpecs"][0].shape == (8, 10, 3)
    with pytest.warns(UserWarning):
        analyze_oii_fourier(dat, fr, aux, 0.2, 0.2)


def test_fourier_maps_analytic():
    rng = np.random.default_rng(2)
    a = rng.uniform(-3, 3, size=(6, 7))
    b = rng.uniform(-3, 3, size=(6, 7))
    mag = rng.uniform(1, 3, size=(4, 6, 7))
    F = [mag[0] * np.exp(1j * a), mag[1] * np.exp(1j * b), mag[2] * np.exp(1j * b), mag[3] * np.exp(1j * a)]
    mask = np.ones((6, 7), dtype=bool)
    mask[0, 0] = False
    m = fourier_maps(F, 0, mask)
    np.testing.assert_allclose(m.PhaseHor, wrap(b - a)[:, ::-1], atol=1e-12)
    np.testing.assert_allclose(m.PhaseVer, wrap(a - b)[:, ::-1], atol=1e-12)
    np.testing.assert_allclose(m.Phase[0], wrap(a)[:, ::-1], atol=1e-12)
    assert m.power_raw[0, -1] == 0  # masked pixel, flipped
    assert np.nanmin(m.Power) >= 0 and np.nanmax(m.Power) <= 1
    np.testing.assert_allclose(m.PhaseMap[0].max(), 1)
    ov = power_overlay(np.ones((6, 7)), m.Power, np.ones((6, 7)), min_area=5)
    assert ov.shape == (6, 7, 3)
    assert isinstance(plot_fourier_maps(m, np.ones((6, 7))), Figure)


def test_oii_normalize():
    rng = np.random.default_rng(3)
    d = rng.uniform(100, 200, size=(4, 5, 6)).astype(np.float32)
    np.testing.assert_allclose(oii_normalize(d, "Mean").mean(axis=(0, 1)), 1, rtol=1e-5)
    med = oii_normalize(d, "Median")
    np.testing.assert_allclose(med[:, :, 0], d[:, :, 0] / np.median(np.median(d[:, :, 0], axis=0)), rtol=1e-6)
    mm = oii_normalize(d, "minmax")
    x = d[:, :, 0].astype(float)
    p2 = np.percentile(x, 2, axis=0, method="hazen")
    p98 = np.percentile(x, 98, axis=0, method="hazen")
    np.testing.assert_allclose(mm[:, :, 0], np.clip(x - p2 / (p98 - p2), 0, 255), rtol=1e-5)
    assert oii_normalize(d, "GlobalSmooth").shape == d.shape
    with pytest.raises(ValueError):
        oii_normalize(d, "bogus")


def test_wave_stimulus_maps_phase():
    h, w, n, sf, f = 6, 8, 200, 10.0, 0.5
    t = np.arange(n) / sf
    col = np.arange(w)[None, :, None]
    d1 = np.broadcast_to(np.cos(2 * np.pi * f * t)[None, None, :] + 0 * col, (h, w, n)) + 5
    d3 = np.broadcast_to(np.cos(2 * np.pi * f * t[None, None, :] + 0.5 * col), (h, w, n)) + 5
    d2 = np.broadcast_to(np.cos(2 * np.pi * f * t[None, None, :] - 0.3 * col), (h, w, n)) + 5
    out = wave_stimulus_maps([d1, d2, d3, d1], [sf] * 4, [f] * 4, spatial_filter_std=0,
                             do_high_pass=False, normalize_per_frame=False)
    np.testing.assert_allclose(out["PhaseHor"], wrap(0.5 * np.arange(w))[None, :] * np.ones((h, 1)), atol=1e-6)
    for k in ("PhaseMapHor", "ScaledPhaseMapVerBlack", "HorVerRetinotopy", "HorRetinotopyBLoverlay"):
        assert out[k].shape == (h, w, 3)
    hp = wave_stimulus_maps([d1, d2, d3, d1], [sf] * 4, [f] * 4, ibv=np.ones((h, w)))
    assert hp["Power"].shape == (h, w)
    with pytest.raises(ValueError):
        wave_stimulus_maps([d1, d2[:2], d3, d1], [sf] * 4, [f] * 4)


def test_wave_reference_image():
    img = np.random.default_rng(4).integers(0, 4000, size=(10, 12)).astype(np.uint16)
    out = wave_reference_image(img)
    assert out.min() >= 0 and out.max() <= 1


def test_stim_conditions_merge_rare():
    s = np.r_[np.zeros(30), np.full(30, 0.1), np.full(5, 0.35), np.full(30, 0.2)]
    c = oii_stim_conditions(s)
    assert c[:30].tolist() == [0] * 30 and c[30] == 1 and c[60] == 0 and c[-1] == 2


def test_patch_condition_images():
    ids = ([0] * 5 + [1] * 5 + [0] * 5 + [2] * 5) * 2
    stim = np.repeat(np.asarray(ids) / 10, 10)
    frame_times = 10 * np.arange(len(ids)) + 5
    vd = np.repeat((10.0 * np.asarray(ids))[None, None, :], 2, axis=0).repeat(3, axis=1)
    res = oii_patch_condition_images(vd, frame_times, stim, patches=(1, 2))
    np.testing.assert_allclose(res.condimg[:, :, 0], 16)
    np.testing.assert_allclose(res.condimg[:, :, 1], 32)  # 20 - mean([0 0 0 0 20]), twice
    assert res.frame_id_reps.max() == 2
    assert isinstance(plot_patch_montage(res.condimg, (1, 2)), Figure)
