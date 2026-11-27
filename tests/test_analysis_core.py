"""Synthetic tests for the analysis-core ports: psmooth, smooth2, ftfil, nGSC, AvgPsdInRange,
extract_pulse_envelope, check_vis_speed (21_Analysis) and expmat, flattenCell, isnumber,
nan_sem (06_Library)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.signal
from calliopy.analysis import (
    avg_psd_in_range,
    check_vis_speed,
    expmat,
    extract_pulse_envelope,
    flatten_cell,
    ftfil,
    is_number,
    ngsc,
    psmooth,
    smooth2,
)
from calliopy.stats import nan_sem


# --------------------------------------------------------------------------- psmooth
def _psmooth_literal(trace, w):
    """Line-by-line transcription of psmooth.m (sorted list, MATLAB find semantics)."""
    x = [float(v) for v in trace]
    lt = len(x)
    med_lo = float(np.median(x[:w]))
    med_hi = float(np.median(x[lt - w :]))
    p = [med_lo] * (w // 2) + x + [med_hi] * (w // 2)
    srt = sorted(p[:w], key=lambda v: (math.isnan(v), v))
    k = math.floor(w * 8 / 100 + 0.5)
    base = []
    for i in range(lt):
        base.append(srt[k - 1])
        if i < lt - 1:
            leave = p[i]
            for j, v in enumerate(srt):
                if v == leave:
                    del srt[j]
                    break
            new = p[i + w]
            for j, v in enumerate(srt):
                if v > new:
                    srt.insert(j, new)
                    break
            else:
                srt.append(new)
    base = np.array(base)
    return np.array(x) - base + np.median(base)


def test_psmooth_matches_literal_transcription():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(size=400)) + 5 * np.sin(np.arange(400) / 15)
    x[50:60] = x[50]  # ties
    np.testing.assert_allclose(psmooth(x, 40), _psmooth_literal(x, 40), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(psmooth(x, 100), _psmooth_literal(x, 100), rtol=1e-12, atol=1e-12)


def test_psmooth_nan_path_matches_literal():
    rng = np.random.default_rng(2)
    x = rng.normal(size=300)
    x[[70, 71, 200]] = np.nan
    got = psmooth(x, 50)
    exp = _psmooth_literal(x, 50)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(exp))


def test_sorted_window_equals_partition_without_nan():
    from calliopy.analysis._signal import _sliding_low_order_matlab

    rng = np.random.default_rng(3)
    p = rng.normal(size=260)
    view = np.lib.stride_tricks.sliding_window_view(p, 60)[:200]
    exp = np.partition(view, 4, axis=1)[:, 4]
    np.testing.assert_array_equal(_sliding_low_order_matlab(p, 60, 4, 200), exp)


def test_psmooth_shapes_defaults_and_errors():
    x = np.linspace(0, 1, 50)
    assert psmooth(x).shape == (50,)  # default window 50 (= 2*floor(50/2))
    col = psmooth(x.reshape(1, -1), 20)
    assert col.shape == (50, 1)  # MATLAB returns a column
    short = psmooth(x[:10].reshape(1, -1), 20)
    assert short.shape == (10, 1)
    np.testing.assert_array_equal(short.ravel(), x[:10])
    np.testing.assert_allclose(psmooth(np.full(30, 4.0), 10), 4.0)
    with pytest.raises(ValueError):
        psmooth(x, 21)
    with pytest.raises(ValueError):
        psmooth(x, 6)  # round(0.48) == 0
    with pytest.raises(ValueError):
        psmooth(np.ones((3, 3)), 2)


def test_psmooth_keeps_float32_and_integer_class():
    x = (np.arange(100) % 7).astype(np.float32)
    assert psmooth(x, 20).dtype == np.float32
    xi = np.array([0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint16)
    out = psmooth(xi, 10)
    assert out.dtype == np.uint16 and out.min() >= 0


# --------------------------------------------------------------------------- smooth2
def test_smooth2_equals_window_mean():
    rng = np.random.default_rng(4)
    x = rng.normal(size=37)
    for w in (1, 4, 5):
        p = np.concatenate([np.full(math.ceil(w / 2), x[0]), x, np.full(w // 2, x[-1])])
        exp = np.array([p[i + 1 : i + w + 1].mean() for i in range(x.size)])
        np.testing.assert_allclose(smooth2(x, w), exp, rtol=1e-12, atol=1e-12)


def test_smooth2_default_axis_rule():
    rng = np.random.default_rng(5)
    tall = rng.normal(size=(20, 3))
    np.testing.assert_allclose(smooth2(tall, 3)[:, 1], smooth2(tall[:, 1], 3))
    wide = tall.T
    np.testing.assert_allclose(smooth2(wide, 3)[1], smooth2(wide[1], 3))
    sq = rng.normal(size=(4, 4))  # square: along rows, as MATLAB
    np.testing.assert_allclose(smooth2(sq, 2)[2], smooth2(sq[2], 2))
    np.testing.assert_allclose(smooth2(sq, 2, axis=0)[:, 2], smooth2(sq[:, 2], 2))


# --------------------------------------------------------------------------- ftfil
def test_ftfil_band_pass_keeps_in_band_sine():
    fs, n = 1000.0, 1000
    t = np.arange(n) / fs
    slow, fast = np.sin(2 * np.pi * 5 * t), np.sin(2 * np.pi * 120 * t)
    out = ftfil(slow + fast + 3.0, fs, 1, 50)
    np.testing.assert_allclose(out, slow, atol=1e-10)


def test_ftfil_transposes_tall_input():
    x = np.random.default_rng(6).normal(size=(64, 2))
    out = ftfil(x, 64.0, 0, 20)
    assert out.shape == (2, 64)  # MATLAB does not transpose back
    np.testing.assert_allclose(out[0], ftfil(x[:, 0], 64.0, 0, 20))
    with pytest.raises(ValueError):
        ftfil(x, 64.0, 0, 0)


# --------------------------------------------------------------------------- nGSC
def test_ngsc_equal_independent_components_is_one():
    t = np.arange(400)
    data = np.vstack([np.sin(2 * np.pi * 3 * t / 400), np.cos(2 * np.pi * 3 * t / 400)])
    assert ngsc(data) == pytest.approx(1.0, abs=1e-12)


def test_ngsc_matches_covariance_eigenvalues():
    rng = np.random.default_rng(7)
    data = rng.normal(size=(4, 200)) * np.array([[1], [2], [0.5], [3]])
    ev = np.linalg.eigvalsh(np.cov(data))
    ev = ev / ev.sum()
    exp = -np.sum(ev * np.log(ev)) / np.log(4)
    assert ngsc(data) == pytest.approx(exp, rel=1e-10)
    data_nan = data.copy()
    data_nan[1, 5] = np.nan  # pca drops that time point
    assert ngsc(data_nan) == pytest.approx(ngsc(np.delete(data, 5, axis=1)), rel=1e-12)


# --------------------------------------------------------------------------- AvgPsdInRange
@pytest.mark.parametrize("n", [256, 255])
def test_avg_psd_matches_scipy_periodogram(n):
    rng = np.random.default_rng(8)
    x = rng.normal(size=n)
    fs = 100.0
    f, pxx = scipy.signal.periodogram(x, fs, window="boxcar", detrend=False)
    sel = (f >= 5) & (f <= 20)
    assert avg_psd_in_range(x, fs, [5, 20]) == pytest.approx(pxx[sel].mean(), rel=1e-10)
    geo = 10 ** (np.mean(10 * np.log10(pxx[sel] + np.finfo(float).eps)) / 10)
    assert avg_psd_in_range(x, fs, [5, 20], True) == pytest.approx(geo, rel=1e-10)


def test_avg_psd_matrix_axis_and_edge_cases():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(3, 128))
    out = avg_psd_in_range(x, 50.0, [1, 10], axis=1)
    assert out.shape == (3, 1)
    assert out[1, 0] == pytest.approx(avg_psd_in_range(x[1], 50.0, [1, 10]))
    out0 = avg_psd_in_range(x.T, 50.0, [1, 10])  # default: first non-singleton axis (0)
    assert out0.shape == (1, 3)
    np.testing.assert_allclose(out0.ravel(), out.ravel())
    assert math.isnan(avg_psd_in_range([1.0], 10, [0, 5]))
    with pytest.warns(UserWarning):
        assert math.isnan(avg_psd_in_range(x[0], 50.0, [30, 40]))


# --------------------------------------------------------------------------- envelope
def _pulses(amps, period=20, width=6, lead=7):
    s = np.zeros(lead + period * len(amps) + 10)
    for k, a in enumerate(amps):
        s[lead + k * period : lead + k * period + width] = a
    return s


def test_envelope_constant_pulses():
    s = _pulses(np.ones(10))
    env = extract_pulse_envelope(s, dt=4)
    assert env.shape == (1, s.size)
    np.testing.assert_allclose(env, 1.0)


def test_envelope_values_at_onsets_and_gap_filling():
    amps = np.array([1.0, 2.0, 3.0, 0, 0, 0, 7.0, 8.0])  # three missing pulses
    s = _pulses(amps, period=20, width=6)
    env = extract_pulse_envelope(s, dt=4).ravel()
    onsets = 7 + 20 * np.arange(len(amps))
    # median over [onset, onset+4] of a 6-sample plateau is the amplitude
    for k in (0, 1, 2, 6, 7):
        assert env[onsets[k]] == pytest.approx(amps[k])
    # synthetic onsets inside the gap sample |signal| = 0
    assert env[onsets[4]] == pytest.approx(0.0)
    # a short spike (shorter than round(dt/2)) is not an onset
    s2 = s.copy()
    s2[100] = 5
    env2 = extract_pulse_envelope(s2, dt=4).ravel()
    assert env2[onsets[4]] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        extract_pulse_envelope(_pulses([1.0]), dt=4)


# --------------------------------------------------------------------------- vis speed
def test_check_vis_speed_square_wave():
    x = np.tile(np.r_[np.zeros(100), np.ones(100)], 10)
    v = check_vis_speed(x)
    assert v.shape == (1, x.size)
    # smoothing makes the supra-threshold half shorter: the interior alternates between two
    # speeds 500/a and 500/b with a + b = 200 samples (one period)
    vals = np.unique(np.round(v[0, 300:1700], 9))
    assert vals.size == 2
    assert 500 / vals[0] + 500 / vals[1] == pytest.approx(200)
    assert v.max() <= 10
    np.testing.assert_array_equal(check_vis_speed(x.reshape(-1, 1)), v)


# --------------------------------------------------------------------------- 06_Library
def test_expmat_matches_matlab_reshape():
    b = np.array([[1, 2, 3], [4, 5, 6]])
    m = 3
    lit = np.tile(b.T, (1, m)).T.reshape(2, m * 3, order="F")  # reshape(repmat(b',1,m)',...)
    np.testing.assert_array_equal(expmat(b, m), lit)
    np.testing.assert_array_equal(expmat([[1, 2], [3, 4]], 3), [[1, 1, 1, 2, 2, 2], [3, 3, 3, 4, 4, 4]])
    assert expmat(np.arange(1, 4)[:, None], 2).shape == (3, 2)
    assert expmat([1, 2], 2).shape == (1, 4)


def test_flatten_cell():
    assert flatten_cell(["a", ["b", ["c", "d"]], "e"]) == ["a", "b", "c", "d", "e"]
    obj = np.empty((2, 2), dtype=object)
    obj[0, 0], obj[1, 0], obj[0, 1], obj[1, 1] = "a", "b", ["c"], np.array(["x"], dtype=object)
    assert flatten_cell(obj) == ["a", "b", "c", "x"]  # column-major
    assert flatten_cell([]) == []


def test_is_number():
    assert is_number(3) and is_number(2.5) and is_number(np.float32(1)) and is_number(np.array([]))
    assert is_number("3") and is_number(" -1e3 ") and is_number("Inf") and is_number("1,000")
    assert is_number("1+2i")
    assert not is_number("NaN") and not is_number("abc") and not is_number(True)
    assert not is_number(None) and not is_number(["1"]) and not is_number("")


def test_nan_sem():
    x = np.array([[1.0, 2.0, np.nan, 4.0], [np.nan, np.nan, np.nan, np.nan], [5.0, np.nan, np.nan, np.nan]])
    out = nan_sem(x, axis=1)
    assert out.shape == (3, 1)
    v = np.array([1.0, 2.0, 4.0])
    assert out[0, 0] == pytest.approx(v.std(ddof=1) / np.sqrt(3), rel=1e-12)
    assert np.isnan(out[1, 0])
    assert out[2, 0] == 0.0  # one value: std 0 as MATLAB
    # default axis: first non-singleton
    assert nan_sem(x).shape == (1, 4)
    assert nan_sem(np.array([[1.0, 3.0]])).shape == (1, 1)
    assert nan_sem([1.0, 3.0]) == pytest.approx(1.0)
    assert np.isnan(nan_sem([]))
