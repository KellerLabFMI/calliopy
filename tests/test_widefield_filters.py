"""Tests for calliopy.widefield._filters (zeroShiftBandPass.m, zeroShiftNotch.m)."""

from __future__ import annotations

import numpy as np
import pytest

from calliopy.widefield._filters import (
    frequency_keep_mask,
    zero_shift_bandpass,
    zero_shift_notch,
)

# --- independent reference: MATLAB statements on the full spectrum -------------------------


def _find1(values, pred):
    """MATLAB find(pred, 1) by an explicit loop (1-based, None when empty)."""
    for i, v in enumerate(values, start=1):
        if pred(v):
            return i
    return None


def _set_zero(y, a, b):
    """MATLAB y(a:b) = 0 with 1-based inclusive limits; None operand -> empty range.

    Indices beyond the end are ignored (where MATLAB would grow the array and fail).
    """
    if a is None or b is None:
        return
    for i in range(a, b + 1):
        if 1 <= i <= len(y):
            y[i - 1] = 0


def _ifft_symmetric(y):
    """MATLAB ifft(y, 'symmetric'): second half replaced by the conjugate of the first."""
    n = len(y)
    z = np.array(y, dtype=complex)
    for k in range(n // 2 + 1, n):
        z[k] = np.conj(z[n - k])
    z[0] = z[0].real
    if n % 2 == 0:
        z[n // 2] = z[n // 2].real
    return np.fft.ifft(z).real


def _ref_filter(x, fs, zero_fn):
    out = np.empty_like(x, dtype=float)
    for c in range(x.shape[0]):
        n = x.shape[1]
        f = [fs * k / n for k in range(n // 2 + 1)]
        y = np.fft.fft(x[c])
        zero_fn(y, f, n)
        s = _ifft_symmetric(y)[::-1]
        y2 = np.fft.fft(s)
        zero_fn(y2, f, n)
        out[c] = _ifft_symmetric(y2)[::-1]
    return out


def _ref_bandpass(x, fs, lb, ub):
    def zero(y, f, n):
        hi = _find1(f, lambda v: v > ub)
        lo = _find1(f, lambda v: v >= lb)
        _set_zero(y, hi, None if hi is None else n - hi + 2)
        _set_zero(y, 1, None if lo is None else lo - 1)
        _set_zero(y, None if lo is None else n - lo + 2, n)

    return _ref_filter(x, fs, zero)


def _ref_notch(x, fs, nf, bw):
    def zero(y, f, n):
        li = _find1(f, lambda v: v >= nf - bw / 2)
        ui = _find1(f, lambda v: v >= nf + bw / 2)
        _set_zero(y, li, ui)
        _set_zero(y, None if ui is None else n - ui + 2, None if li is None else n - li + 2)

    return _ref_filter(x, fs, zero)


def _signal(n_ch, n, seed=0):
    return np.random.default_rng(seed).standard_normal((n_ch, n))


# --- analytic cases -----------------------------------------------------------------------


@pytest.mark.parametrize("n", [200, 201])
def test_bandpass_keeps_only_band(n):
    fs = 100.0
    t = np.arange(n) / fs
    k_low, k_mid, k_high = 4, 20, 80  # bin numbers, frequencies k * fs / n
    sines = [np.sin(2 * np.pi * k * fs / n * t + 0.3 * k) for k in (k_low, k_mid, k_high)]
    x = np.vstack([sines[0] + sines[1] + sines[2], 2 * sines[1]])
    out = zero_shift_bandpass(x, fs, 5.0, 20.0)
    np.testing.assert_allclose(out[0], sines[1], atol=1e-12)
    np.testing.assert_allclose(out[1], 2 * sines[1], atol=1e-12)


def test_notch_removes_line_frequency():
    fs, n = 1000.0, 1000
    t = np.arange(n) / fs
    keep = np.sin(2 * np.pi * 10 * t)
    x = keep + 0.5 * np.sin(2 * np.pi * 50 * t) + 0.2 * np.cos(2 * np.pi * 51 * t)
    out = zero_shift_notch(x, fs, 50.0, 2.0)  # bins 49, 50, 51 Hz removed
    assert out.shape == x.shape
    np.testing.assert_allclose(out, keep, atol=1e-12)


def test_notch_upper_index_includes_next_bin():
    # band 49.5 .. 50.5 Hz on a 1 Hz grid: bins 50 and 51 (first f >= 50.5) are removed
    keep = frequency_keep_mask(1000, 1000.0, notch=(50.0, 1.0))
    removed = np.flatnonzero(~keep[:501])
    np.testing.assert_array_equal(removed, [50, 51])
    removed_neg = np.flatnonzero(~keep[501:]) + 501
    np.testing.assert_array_equal(removed_neg, [1000 - 51, 1000 - 50])


# --- agreement with the literal MATLAB index arithmetic -----------------------------------


BAND_CASES = [
    (200, 100.0, 5.0, 20.0),
    (201, 100.0, 5.0, 20.0),
    (64, 64.0, 0.0, 10.0),  # lb = 0: nothing removed at the low end
    (64, 64.0, -3.0, 10.0),
    (64, 64.0, 3.0, 32.0),  # ub = Nyquist: find(f > ub) empty -> no low-pass
    (65, 64.0, 3.0, 100.0),  # ub beyond the frequency vector
    (64, 64.0, 40.0, 50.0),  # lb beyond the frequency vector -> nothing removed at all
    (64, 64.0, 31.5, 50.0),  # even L, lb in (f[-2], f[-1]]: Nyquist removed too
    (65, 64.0, 31.0, 50.0),  # odd L, lb just below the last bin
    (64, 64.0, 12.0, 6.0),  # lb > ub
    (64, 64.0, 3.0, -1.0),  # ub < 0: MATLAB would fail; everything zeroed here
    (1, 10.0, 1.0, 4.0),
    (2, 10.0, 1.0, 4.0),
    (3, 10.0, 0.0, 4.0),
]


@pytest.mark.parametrize(("n", "fs", "lb", "ub"), BAND_CASES)
def test_bandpass_matches_literal_matlab(n, fs, lb, ub):
    x = _signal(3, n, seed=n)
    np.testing.assert_allclose(
        zero_shift_bandpass(x, fs, lb, ub), _ref_bandpass(x, fs, lb, ub), atol=1e-12
    )


NOTCH_CASES = [
    (1000, 1000.0, 50.0, 2.0),
    (999, 1000.0, 50.0, 2.0),
    (100, 100.0, 49.0, 4.0),  # upper edge beyond Nyquist: nothing removed
    (101, 100.0, 49.0, 4.0),
    (100, 100.0, 1.0, 4.0),  # lower edge <= 0: lower_idx = 1 (MATLAB index L+1)
    (100, 100.0, 50.0, 0.0),  # zero width on the Nyquist bin
    (100, 100.0, 10.3, 0.1),  # band between two bins: the next bin is removed
]


@pytest.mark.parametrize(("n", "fs", "nf", "bw"), NOTCH_CASES)
def test_notch_matches_literal_matlab(n, fs, nf, bw):
    x = _signal(2, n, seed=n + 1)
    np.testing.assert_allclose(
        zero_shift_notch(x, fs, nf, bw), _ref_notch(x, fs, nf, bw), atol=1e-12
    )


# --- MATLAB special cases, stated explicitly ----------------------------------------------


def test_bandpass_lb_beyond_nyquist_removes_nothing():
    x = _signal(2, 64)
    np.testing.assert_allclose(zero_shift_bandpass(x, 64.0, 40.0, 50.0), x, atol=1e-12)


def test_bandpass_even_length_nyquist_case_removes_everything():
    x = _signal(1, 64)
    np.testing.assert_allclose(zero_shift_bandpass(x, 64.0, 31.5, 50.0), 0.0, atol=1e-12)
    # the same lb on odd length keeps the last bin
    keep = frequency_keep_mask(65, 64.0, band=(31.0, 50.0))
    np.testing.assert_array_equal(np.flatnonzero(keep[:33]), [32])


def test_bandpass_ub_negative_zeroes_all():
    x = _signal(2, 30)
    np.testing.assert_allclose(zero_shift_bandpass(x, 10.0, 1.0, -1.0), 0.0, atol=1e-12)


def test_notch_upper_beyond_nyquist_removes_nothing():
    x = _signal(2, 100)
    np.testing.assert_allclose(zero_shift_notch(x, 100.0, 49.0, 4.0), x, atol=1e-12)


def test_bandpass_mask_matches_frequency_rule():
    # independent of the index arithmetic: for ordinary cutoffs the positive half keeps
    # exactly lb <= f <= ub
    for n in (100, 101):
        f = 250.0 * np.arange(n // 2 + 1) / n
        keep = frequency_keep_mask(n, 250.0, band=(7.0, 60.0))[: n // 2 + 1]
        np.testing.assert_array_equal(keep, (f >= 7.0) & (f <= 60.0))


# --- shapes and types ---------------------------------------------------------------------


def test_two_passes_equal_one_masked_pass():
    x = _signal(2, 128)
    keep = frequency_keep_mask(128, 128.0, band=(5.0, 30.0))[:65]
    once = np.fft.irfft(np.fft.rfft(x, axis=1) * keep, n=128, axis=1)
    np.testing.assert_allclose(zero_shift_bandpass(x, 128.0, 5.0, 30.0), once, atol=1e-12)


def test_rows_are_independent_and_1d_input():
    x = _signal(3, 90)
    full = zero_shift_bandpass(x, 30.0, 2.0, 8.0)
    for c in range(3):
        row = zero_shift_bandpass(x[c], 30.0, 2.0, 8.0)
        assert row.shape == (90,)
        np.testing.assert_allclose(row, full[c], atol=1e-12)


def test_dtypes():
    x = _signal(2, 50)
    assert zero_shift_notch(x.astype(np.float32), 50.0, 10.0, 2.0).dtype == np.float32
    np.testing.assert_allclose(
        zero_shift_notch(x.astype(np.float32), 50.0, 10.0, 2.0),
        zero_shift_notch(x.astype(np.float32).astype(float), 50.0, 10.0, 2.0),
        rtol=1e-5,
        atol=1e-6,
    )
    xi = (x * 100).astype(np.int16)
    assert zero_shift_bandpass(xi, 50.0, 1.0, 10.0).dtype == np.float64
    assert zero_shift_bandpass(np.zeros((2, 0)), 50.0, 1.0, 10.0).shape == (2, 0)


def test_invalid_input():
    with pytest.raises(ValueError):
        zero_shift_bandpass(np.zeros((2, 3, 4)), 10.0, 1.0, 2.0)
    with pytest.raises(ValueError):
        zero_shift_bandpass(np.zeros((2, 8), complex), 10.0, 1.0, 2.0)
    with pytest.raises(ValueError):
        frequency_keep_mask(8, 10.0)
