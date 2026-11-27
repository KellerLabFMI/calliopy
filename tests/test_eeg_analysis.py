"""Synthetic tests of calliopy.eeg (get_eeg_times, ERPs, myResample, myWelch, EEG checks)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from matplotlib.figure import Figure

from calliopy.eeg import (
    TriggerAlignmentError,
    create_erp_auxaligned,
    eeg_average_spectrum,
    eeg_average_spectrum_files,
    eeg_grating_erps,
    get_eeg_times,
    my_resample,
    my_welch,
    plot_eeg_spectrum,
    plot_erp,
    read_eeg_dat,
    rising_edges,
)
from calliopy.io import EEG_CHANNEL_ORDER


def pulses(n, starts, width, amp=5.0):
    x = np.zeros(n)
    for s in starts:
        x[int(s) : int(s) + width] = amp
    return x


def test_rising_edges_matlab_values():
    x = np.array([0, 0, 2, 2, 0, 3])
    np.testing.assert_array_equal(rising_edges(x), [2, 5])


def test_get_eeg_times_linear_map():
    a_starts = 5000 + 10000 * np.arange(15)
    e_starts = 300 + 1000 * np.arange(15)
    aux = pulses(160000, a_starts, 100)
    eeg = pulses(15500, e_starts, 10)
    t = get_eeg_times(aux, eeg)
    assert t.shape == (1, 15500)
    i = np.arange(1, 15501)
    np.testing.assert_array_equal(t[0], np.floor(5000 + (i - 300) * 10 + 0.5))
    # one extra aux trigger at the start shifts everything: rejected
    aux_bad = pulses(160000, np.r_[2000, a_starts], 100)
    with pytest.raises(TriggerAlignmentError):
        get_eeg_times(aux_bad, eeg)


def test_create_erp_auxaligned():
    n = 5000
    d = np.zeros((2, n))
    d[0, 1000] = 1.0
    d[1, 1000:1010] = 2.0
    times = 10.0 * np.arange(1, n + 1)
    res = create_erp_auxaligned(d, times, [10.0 * 1001], 1000, 0.1, 0.2)
    assert res.time.shape == (1, 301)
    assert res.time[0, 0] == -0.1 and res.time[0, -1] == pytest.approx(0.2)
    k0 = int(np.flatnonzero(np.isclose(res.time[0], 0))[0])
    assert res.erp[0, k0] == 1.0 and res.erp[0].sum() == 1.0
    assert res.epochs_ids[0, k0] == 1000
    with pytest.raises(IndexError):
        create_erp_auxaligned(d, times, [10.0], 1000, 0.1, 0.2)
    empty = create_erp_auxaligned(d, times, [], 1000, 0.1, 0.2)
    assert np.all(np.isnan(empty.erp))


def test_my_resample():
    x = np.ones(400)
    y = my_resample(x, 1, 2)
    assert y.shape == (200,)
    np.testing.assert_allclose(y[20:-20], 1.0)
    rng = np.random.default_rng(0)
    m = rng.normal(size=(3, 300))
    y2 = my_resample(m, 1, 3, 1)
    assert y2.shape == (3, 100)
    N = 30
    n = np.arange(-N // 2, N // 2 + 1)
    h = np.sinc(n / 6) * 0.5 * (1 - np.cos(2 * np.pi * np.arange(N + 1) / N))
    h /= h.sum()
    ref = np.convolve(m[1], h)[N // 2 : N // 2 + 300][::3]
    np.testing.assert_allclose(y2[1], ref, atol=1e-12)
    y3 = my_resample(np.ones(50), 2, 1)
    # unit-sum filter: kept 1/p gain; the polyphase ripple of the windowed sinc is ~1e-4
    np.testing.assert_allclose(y3[30:70], 0.5, atol=1e-4)
    with pytest.raises(ValueError):
        my_resample(x, 1.5, 2)


def test_my_welch_sinusoid():
    fs, N = 1000.0, 200
    t = np.arange(2000) / fs
    x = 2.0 * np.cos(2 * np.pi * 50 * t)
    p, f = my_welch(x, np.ones(N), N / 2, N, fs)
    assert p.shape == (101, 1) and f[1, 0] == 5.0
    k = int(np.argmax(p))
    assert f[k, 0] == 50.0
    assert p[k, 0] == pytest.approx((2.0 * N / 2) ** 2 / (N * fs))
    p2, _ = my_welch(x, N, N / 2, N + 1, fs)
    assert p2.shape == (101, 1)


def test_plot_erp():
    fig = plot_erp(np.random.default_rng(1).normal(size=(4, 50)), np.linspace(-0.5, 1, 50), [1, 2, 3, 4])
    assert isinstance(fig, Figure)


def _write_dat(tmp_path, data_int16, bit_volts, fs, name="S1-T1_EEG"):
    nch = data_int16.shape[0]
    (tmp_path / f"{name}.dat").write_bytes(data_int16.T.astype("<i2").tobytes())
    hdr = {"continuous": [{"num_channels": nch, "sample_rate": fs,
                           "channels": [{"bit_volts": bit_volts} for _ in range(nch)]}]}
    (tmp_path / f"{name}.oebin").write_text(json.dumps(hdr))
    return tmp_path / f"{name}.dat"


def test_read_eeg_dat_and_spectrum(tmp_path):
    fs = 1000
    t = np.arange(20 * fs) / fs
    raw = np.zeros((33, t.size), dtype=np.int16)
    for c in range(33):
        raw[c] = np.round((c + 1) * 10 * np.sin(2 * np.pi * 10 * t))
    p = _write_dat(tmp_path, raw, 0.5, fs)
    d, rate = read_eeg_dat(p)
    assert rate == fs and d.dtype == np.float32
    np.testing.assert_allclose(d, raw * 0.5)
    spec = eeg_average_spectrum([d, d[:, :15000]], fs, common_avg=False, window_size_sec=1,
                                freq_lim=(0, 40))
    assert spec.freq_spec[-1, 0] == 40 and spec.rec_means.shape == (41, 2)
    assert spec.freq_spec[int(np.argmax(spec.rec_means[:, 0])), 0] == 10
    ca = eeg_average_spectrum([d], fs, window_size_sec=1, exclude_chan=[0, 1])
    assert ca.rec_means.shape[1] == 1
    files = eeg_average_spectrum_files([p], window_size_sec=1, common_avg=False)
    np.testing.assert_allclose(files.rec_means[:, 0], eeg_average_spectrum([d], fs, window_size_sec=1,
                                                                            common_avg=False).rec_means[:, 0])
    assert isinstance(plot_eeg_spectrum(spec, rng=np.random.default_rng(0), nbss=20), Figure)


def test_eeg_grating_erps_end_to_end():
    fs, f_aux, dur = 2000, 10000, 40
    n = fs * dur
    eeg = np.zeros((33, n), dtype=np.float32)
    sync_s = 1 + 2 * np.arange(19)
    eeg[32] = pulses(n, sync_s * fs, int(0.1 * fs))
    off = 50000

    def aux_of(t_s):
        return off + t_s * f_aux

    na = int(aux_of(dur + 2))
    aux_sync = pulses(na, aux_of(sync_s), int(0.1 * f_aux))
    on_s = 5.25 + 4 * np.arange(8)
    grating = pulses(na, aux_of(on_s), f_aux)
    probe0 = EEG_CHANNEL_ORDER[0]
    for s in on_s:
        eeg[probe0, int(s * fs) : int((s + 0.1) * fs)] = 50.0
    res = eeg_grating_erps([{"eeg": eeg, "fs": fs, "grating": grating, "aux_sync": aux_sync}],
                           common_avg=False)
    assert res.time_on.shape == (1, 1501)
    k0 = int(np.flatnonzero(np.isclose(res.time_on[0], 0.05))[0])
    assert res.erp_on[0, k0] > 30
    assert abs(res.erp_on[1, k0]) < 1
    assert res.erp_off.shape == (30, 1501)
