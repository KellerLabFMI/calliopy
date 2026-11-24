"""Zero-phase FFT band-pass and notch filters (port of calliope 12_Widefield).

Ports of ``zeroShiftBandPass.m`` and ``zeroShiftNotch.m``. Each row of the signal is
Fourier transformed, a set of frequency bins is set to zero, the result is transformed back
with MATLAB's ``ifft(..., 'symmetric')``, time-reversed, and the same bins are zeroed a second
time before the signal is reversed back.

The bins that are zeroed are taken from MATLAB's index arithmetic, reproduced exactly
(see :func:`zero_shift_bandpass`), including the cases in which MATLAB's ``find`` returns
an empty result. ``ifft(Y, 'symmetric')`` only uses the first half of ``Y`` (bins
``0 .. L//2``) and treats the rest as its complex conjugate; it is computed here with
:func:`numpy.fft.irfft`. Zeroing that MATLAB applies to the second half therefore has no
effect, except where it reaches the Nyquist bin.

None of these functions has been compared with MATLAB output.
"""

from __future__ import annotations

import numpy as np

__all__ = ["frequency_keep_mask", "zero_shift_bandpass", "zero_shift_notch"]


def _first(cond: np.ndarray) -> int | None:
    """MATLAB ``find(cond, 1)``: 1-based index of the first true element, None if empty."""
    idx = np.flatnonzero(cond)
    return int(idx[0]) + 1 if idx.size else None


def _zero_range(keep: np.ndarray, first: int | None, last: int | None) -> None:
    """MATLAB ``Y(first:last) = 0`` on a 1-based, inclusive range (in place on ``keep``).

    An empty operand (``find`` found nothing) gives an empty range, as ``[]:n`` does in
    MATLAB. Indices beyond ``len(keep)`` are dropped (MATLAB would grow the array and then
    fail; see the callers).
    """
    if first is None or last is None:
        return
    lo = max(first, 1)
    hi = min(last, keep.size)
    if lo <= hi:
        keep[lo - 1 : hi] = False


def frequency_keep_mask(n_samples: int, fs: float, *, band=None, notch=None) -> np.ndarray:
    """Bins of a length-``n_samples`` FFT that the MATLAB filters keep.

    Parameters
    ----------
    n_samples
        Signal length ``L``.
    fs
        Sampling rate in Hz.
    band
        ``(lb, ub)`` in Hz for :func:`zero_shift_bandpass`.
    notch
        ``(nf, bw)`` in Hz for :func:`zero_shift_notch`. Exactly one of ``band`` and
        ``notch`` is given.

    Returns
    -------
    numpy.ndarray
        Boolean array of length ``L`` over all FFT bins (0-based bin ``k`` is frequency
        ``k * fs / L``; bins above ``L // 2`` are the negative frequencies), ``True`` where
        MATLAB leaves the bin unchanged. Only the first ``L // 2 + 1`` entries affect the
        filtered signal (``ifft(..., 'symmetric')``).
    """
    if (band is None) == (notch is None):
        raise ValueError("give exactly one of band=(lb, ub) and notch=(nf, bw)")
    L = int(n_samples)
    # f = Fs * (0:(L/2)) / L: the colon operator stops at floor(L/2)
    f = fs * np.arange(L // 2 + 1, dtype=np.float64) / L
    keep = np.ones(L, dtype=bool)
    if band is not None:
        lb, ub = band
        hi = _first(f > ub)
        lo = _first(f >= lb)
        _zero_range(keep, hi, None if hi is None else L - hi + 2)
        _zero_range(keep, 1, None if lo is None else lo - 1)
        _zero_range(keep, None if lo is None else L - lo + 2, L)
    else:
        nf, bw = notch
        lower_idx = _first(f >= nf - bw / 2)
        upper_idx = _first(f >= nf + bw / 2)
        _zero_range(keep, lower_idx, upper_idx)
        if lower_idx is not None and upper_idx is not None:
            _zero_range(keep, L - upper_idx + 2, L - lower_idx + 2)
        # with an empty operand the MATLAB colon gives an empty range: nothing is zeroed
    return keep


def _as_signal(signal) -> tuple[np.ndarray, bool]:
    x = np.asarray(signal)
    if np.iscomplexobj(x):
        raise ValueError("signal must be real")
    if x.ndim == 1:
        return x[np.newaxis, :], True
    if x.ndim != 2:
        raise ValueError(f"signal must be [nChannels x nSamples], got shape {x.shape}")
    return x, False


def _filter_two_pass(signal, fs, **spec) -> np.ndarray:
    x, was_1d = _as_signal(signal)
    out_dtype = np.float32 if x.dtype == np.float32 else np.float64
    L = x.shape[1]
    if L == 0 or x.shape[0] == 0:
        out = x.astype(out_dtype)
    else:
        keep = frequency_keep_mask(L, fs, **spec)[: L // 2 + 1]
        work = x.astype(np.float64)
        once = np.fft.irfft(np.fft.rfft(work, axis=1) * keep, n=L, axis=1)
        mirrored = once[:, ::-1]
        twice = np.fft.irfft(np.fft.rfft(mirrored, axis=1) * keep, n=L, axis=1)
        out = np.ascontiguousarray(twice[:, ::-1]).astype(out_dtype, copy=False)
    return out[0] if was_1d else out


def zero_shift_bandpass(signal, fs: float, lb: float, ub: float) -> np.ndarray:
    """Zero-phase band-pass filter by zeroing FFT bins, forward and backward.

    Port of calliope ``12_Widefield/zeroShiftBandPass.m``.

    With ``L`` samples per channel and the frequency vector
    ``f = fs * (0 .. floor(L/2)) / L`` (Hz), MATLAB computes the 1-based indices
    ``high = find(f > ub, 1)`` and ``low = find(f >= lb, 1)`` and sets
    ``Y(high : L-high+2)``, ``Y(1 : low-1)`` and ``Y(L-low+2 : L)`` to zero. The same
    indices are used here (:func:`frequency_keep_mask`). In effect the bins with
    ``f < lb`` or ``f > ub`` are removed, with these MATLAB special cases:

    - ``ub`` at or above the largest frequency of ``f`` (``find`` empty): no low-pass.
    - ``lb`` above the largest frequency of ``f`` (``find`` empty): **no high-pass at all**
      (nothing below ``lb`` is removed).
    - Even ``L`` and ``lb`` in ``(f[-2], f[-1]]``: the Nyquist bin is removed together with
      everything else, although ``f[-1] >= lb`` (the negative-frequency range
      ``L-low+2 : L`` starts at the Nyquist bin).

    Parameters
    ----------
    signal
        Real array ``[nChannels x nSamples]``; a 1-D array is one channel (MATLAB row
        vector) and the result is 1-D as well.
    fs
        Sampling rate in Hz.
    lb, ub
        Lower and upper edge of the pass band in Hz (``lb`` kept: ``f >= lb``; ``ub`` kept:
        ``f <= ub``).

    Returns
    -------
    numpy.ndarray
        Filtered signal, same shape as ``signal``; float32 for float32 input, else float64.

    Notes
    -----
    The zeroing is applied twice (forward, then on the time-reversed signal), as in MATLAB.
    Because the bins are set to exactly zero, the result equals one masked pass up to
    floating-point rounding.

    Differences from MATLAB:

    - ``ub < 0`` makes MATLAB write beyond the end of the spectrum (``Y(1:L+1) = 0``) and
      then fail on the size mismatch; here the range is clipped, which zeroes every bin.
    - Integer input is filtered in float64 and returned as float64 (MATLAB ``fft`` requires
      floating-point input). float32 input is filtered in float64 and returned as float32
      (MATLAB computes in single precision). Complex input raises ``ValueError``.
    - Input with more than two dimensions raises ``ValueError``.
    """
    return _filter_two_pass(signal, fs, band=(lb, ub))


def zero_shift_notch(signal, fs: float, nf: float, bw: float) -> np.ndarray:
    """Zero-phase notch filter by zeroing FFT bins, forward and backward.

    Port of calliope ``12_Widefield/zeroShiftNotch.m``.

    With ``f = fs * (0 .. floor(L/2)) / L`` (Hz), MATLAB computes the 1-based indices
    ``lower_idx = find(f >= nf - bw/2, 1)`` and ``upper_idx = find(f >= nf + bw/2, 1)`` and
    zeroes ``Y(lower_idx : upper_idx)`` and ``Y(L-upper_idx+2 : L-lower_idx+2)``. The same
    indices are used here (:func:`frequency_keep_mask`). In effect the bins from the first
    frequency ``>= nf - bw/2`` up to and including the first frequency ``>= nf + bw/2`` are
    removed (so one bin above the band edge is removed when no bin lies exactly on it), with
    this MATLAB special case:

    - ``nf + bw/2`` above the largest frequency of ``f`` (``find`` empty): **nothing is
      removed**, even if part of the notch band lies below the Nyquist frequency.

    Parameters
    ----------
    signal
        Real array ``[nChannels x nSamples]``; a 1-D array is one channel and the result is
        1-D as well.
    fs
        Sampling rate in Hz.
    nf
        Notch frequency in Hz.
    bw
        Width of the notch band in Hz (``nf - bw/2`` to ``nf + bw/2``).

    Returns
    -------
    numpy.ndarray
        Filtered signal, same shape as ``signal``; float32 for float32 input, else float64.

    Notes
    -----
    Differences from MATLAB:

    - ``nf - bw/2 <= 0`` (``lower_idx = 1``) makes MATLAB write one element beyond the end
      of the spectrum (``L - lower_idx + 2 = L + 1``) and then fail on the size mismatch;
      here that index is dropped. It lies in the negative-frequency half, which
      ``ifft(..., 'symmetric')`` ignores, so the result is what MATLAB would have computed
      without the failure.
    - Integer, float32, complex and >2-D input: as in :func:`zero_shift_bandpass`.
    """
    return _filter_two_pass(signal, fs, notch=(nf, bw))
