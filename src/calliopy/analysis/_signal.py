"""Signal processing helpers of calliope ``21_Analysis``: baseline and running-mean smoothing,
Fourier filtering, band power, pulse envelopes, global spatial complexity and the
photodiode speed estimate.
"""

from __future__ import annotations

import bisect
import itertools
import logging
import math
import warnings

import numpy as np
import scipy.signal

from .._matlab import matlab_cast, mround, prctile

__all__ = [
    "avg_psd_in_range",
    "check_vis_speed",
    "extract_pulse_envelope",
    "ftfil",
    "ngsc",
    "psmooth",
    "smooth2",
]

log = logging.getLogger(__name__)

_EPS = float(np.finfo(np.float64).eps)


# --------------------------------------------------------------------------- helpers


def _median_like_matlab(x: np.ndarray):
    """MATLAB ``median`` of a vector, keeping MATLAB's output class.

    Integer input: the mean of the two middle values rounded half away from zero, in the
    input class. float32 stays float32. NaN anywhere gives NaN (MATLAB default).
    """
    if x.dtype.kind in "biu":
        return matlab_cast(np.median(x.astype(np.float64)), x.dtype)
    return np.median(x)


def _sliding_low_order_matlab(padded: np.ndarray, window: int, k: int, n_out: int):
    """Step-by-step replica of psmooth's sorted-window update (``sorted(window)[k]``).

    A sorted list is updated with binary search for every step, as MATLAB does with
    ``find``. With NaNs: MATLAB removes the leaving element with
    ``find(sorted_window == leaving, 1)``, which never matches a NaN: a NaN that enters the window is never removed, so the window grows
    by one element per NaN that leaves. New elements are inserted before the first larger
    value (NaN after everything). NaNs therefore always sit at the end of the window.
    """
    first = padded[:window]
    finite = sorted(float(v) for v in first if not math.isnan(v))
    n_nan = int(window - len(finite))
    out = np.empty(n_out, dtype=np.float64)
    for i in range(n_out):
        out[i] = finite[k] if k < len(finite) else math.nan
        if i < n_out - 1:
            leaving = float(padded[i])
            if not math.isnan(leaving):
                j = bisect.bisect_left(finite, leaving)
                if j < len(finite) and finite[j] == leaving:
                    del finite[j]
            new = float(padded[i + window])
            if math.isnan(new):
                n_nan += 1
            else:
                bisect.insort_right(finite, new)
    return out


def _as_signal_rows(data):
    """MATLAB 2-D view of ``data``: 1-D input becomes a row ``(1, N)``."""
    a = np.asarray(data)
    if a.ndim == 0:
        return a.reshape(1, 1), True
    if a.ndim == 1:
        return a[np.newaxis, :], True
    if a.ndim != 2:
        raise ValueError(f"data must be a vector or a 2-D matrix, got shape {a.shape}")
    return a, False


def _work_float(a: np.ndarray) -> np.ndarray:
    return a if a.dtype in (np.float32, np.float64) else a.astype(np.float64)


# --------------------------------------------------------------------------- psmooth


def psmooth(trace, win_size: int | None = None) -> np.ndarray:
    """Remove a slow baseline with a running 8th percentile (Dombeck et al. 2007).

    Port of calliope ``21_Analysis/psmooth.m``. The trace is padded at both ends with
    ``win_size // 2`` copies of the median of its first / last ``win_size`` samples; the
    8th percentile of every window of ``win_size`` padded samples (the sample at sorted
    position ``round(0.08 * win_size)``, 1-based, no interpolation) is subtracted from the
    trace and the median of that baseline is added back.

    Parameters
    ----------
    trace : array_like
        Fluorescence trace: a vector (1-D, ``(1, N)`` or ``(N, 1)``).
    win_size : int, optional
        Window length in samples; must be even (MATLAB errors on odd windows, see below).
        Default ``2 * floor(min(N, 1000) / 2)``, as MATLAB.

    Returns
    -------
    numpy.ndarray
        The corrected trace. 1-D input gives a 1-D result; a 2-D vector gives a column
        ``(N, 1)`` (MATLAB transposes rows to columns). float32 input stays float32, integer
        input keeps its class with MATLAB's rounding and saturation; other input is float64.
        When ``N < win_size`` the trace is returned unchanged (as a column for 2-D input),
        as MATLAB does, and a message is logged.

    Notes
    -----
    MATLAB's sorted-window update is reproduced step by step (values are selected, not
    computed, so the baseline is exact). With NaNs this includes MATLAB's behaviour that a
    NaN is never removed from the window (see :func:`_sliding_low_order_matlab`). About
    0.1 s for 60 000 samples and a 1000-sample window.

    Differences from MATLAB
    -----------------------
    - An odd ``win_size`` or one with ``round(0.08 * win_size) == 0`` raises ``ValueError``
      (MATLAB fails inside ``repmat`` / with an invalid index).
    - p-ECT refers to a ``psmooth_fast``, which does not exist at r2003; this port is the
      fast form of ``psmooth``.
    """
    a = np.asarray(trace)
    if a.ndim > 2 or (a.ndim == 2 and min(a.shape) > 1):
        raise ValueError(f"trace must be a vector, got shape {a.shape}")
    one_d = a.ndim <= 1
    x = a.ravel()
    lt = x.size
    if win_size is None:
        window = 2 * (min(lt, 1000) // 2)
    else:
        if float(win_size) != int(win_size):
            raise ValueError("win_size must be an integer")
        window = int(win_size)
    if window % 2:
        raise ValueError("win_size must be even (MATLAB fails in repmat for odd windows)")
    col = x if one_d else x.reshape(-1, 1)
    if lt < window:
        log.info("Stack shorter than window size, psmooth skipped.")
        return col.copy()
    k = int(mround(window * 8 / 100))
    if k < 1:
        raise ValueError("win_size too small: round(0.08 * win_size) must be at least 1")
    half = window // 2
    pad_lo = _median_like_matlab(x[:window])
    pad_hi = _median_like_matlab(x[lt - window :])
    padded = np.concatenate(
        [np.full(half, pad_lo, dtype=x.dtype), x, np.full(half, pad_hi, dtype=x.dtype)]
    ).astype(np.float64)
    base = _sliding_low_order_matlab(padded, window, k - 1, lt)
    med = np.median(base)
    if x.dtype.kind in "biu":
        out = matlab_cast(matlab_cast(x.astype(np.float64) - base, x.dtype) + med, x.dtype)
    elif x.dtype == np.float32:
        out = (x - base.astype(np.float32)) + np.float32(med)
    else:
        out = x.astype(np.float64) - base + med
    out = np.asarray(out)
    return out if one_d else out.reshape(-1, 1)


# --------------------------------------------------------------------------- smooth2


def smooth2(data, window_size: int, axis: int | None = None) -> np.ndarray:
    """Running mean over ``window_size`` samples with edge replication.

    Port of calliope ``21_Analysis/smooth2.m``. Every signal is padded with
    ``ceil(w/2)`` copies of its first and ``floor(w/2)`` copies of its last sample, filtered
    with the causal box filter ``filter(ones(1,w)/w, 1, ...)`` and the first ``w`` filtered
    samples are dropped, so the output has the input length. Sample ``i`` of the output is
    the mean of input samples ``i - ceil(w/2) + 1 ... i + floor(w/2)`` (0-based, replicated
    at the edges).

    Parameters
    ----------
    data : array_like
        A vector or a 2-D matrix.
    window_size : int
        Window length ``w`` in samples (positive).
    axis : int, optional
        **0-based** axis to smooth along (MATLAB ``dim - 1``). Default as MATLAB: along the
        longer dimension (along columns, axis 0, only when the matrix has fewer columns than
        rows; a square matrix is smoothed along rows). 1-D input is smoothed along itself.

    Returns
    -------
    numpy.ndarray
        Same shape as ``data``; float32 for float32 input, else float64.

    Differences from MATLAB
    -----------------------
    - Integer input is converted to float64 (MATLAB ``filter`` rejects integer classes).
    """
    a, was_1d = _as_signal_rows(data)
    a = _work_float(a)
    w = int(window_size)
    if w < 1:
        raise ValueError("window_size must be a positive integer")
    if was_1d:
        ax = 1
    elif axis is None:
        ax = 0 if a.shape[1] < a.shape[0] else 1
    else:
        ax = int(axis) % 2
    if ax == 0:
        a = a.T
    if a.shape[1] == 0:
        out = a.copy()
    else:
        left = np.repeat(a[:, :1], math.ceil(w / 2), axis=1)
        right = np.repeat(a[:, -1:], w // 2, axis=1)
        p = np.concatenate([left, a, right], axis=1)
        b = np.full(w, 1.0 / w, dtype=a.dtype)
        out = scipy.signal.lfilter(b, np.ones(1, dtype=a.dtype), p, axis=1)[:, w:]
        out = out.astype(a.dtype, copy=False)
    if ax == 0:
        out = out.T
    return out.ravel() if was_1d else np.ascontiguousarray(out)


# --------------------------------------------------------------------------- ftfil


def ftfil(data, frequency: float, hi_pass: float, lo_pass: float) -> np.ndarray:
    """Band-pass a signal by zeroing Fourier coefficients.

    Port of calliope ``21_Analysis/ftfil.m``. For every row: ``y = fft(row)``; with
    ``lim = round(hi_pass / frequency * N)`` the first ``lim`` coefficients and their mirror
    images are set to zero (high pass); with ``lim = round(lo_pass / frequency * N)`` the
    coefficients ``lim-1 ... N-lim`` (0-based) are set to zero (low pass); the result is
    ``real(ifft(y))``.

    Parameters
    ----------
    data : array_like
        A vector or a 2-D matrix. If it has more rows than columns it is transposed first,
        so the transform always runs along the longer dimension.
    frequency : float
        Sampling rate in Hz.
    hi_pass, lo_pass : float
        Cut-offs in Hz. ``lo_pass`` must give ``lim >= 1``.

    Returns
    -------
    numpy.ndarray
        Filtered data, ``(n_signals, N)``: **transposed** with respect to the input when the
        input had more rows than columns (MATLAB does not transpose back). 1-D input gives
        1-D output. float32 stays float32, everything else is float64.

    Differences from MATLAB
    -----------------------
    - ``lo_pass`` with ``lim == 0`` and ``hi_pass`` with ``lim > N`` raise ``ValueError``
      (MATLAB fails with an index error / a size mismatch).
    - Integer input is converted to float64 (MATLAB ``fft`` rejects integer classes).
    """
    a, was_1d = _as_signal_rows(data)
    if a.shape[0] > a.shape[1]:
        a = a.T
    a = _work_float(a)
    n = a.shape[1]
    lim_hi = int(mround(hi_pass / frequency * n))
    lim_lo = int(mround(lo_pass / frequency * n))
    if lim_hi > n:
        raise ValueError("hi_pass above the sampling frequency (MATLAB fails here)")
    if lim_lo < 1:
        raise ValueError("lo_pass too low: round(lo_pass/frequency*N) must be at least 1")
    y = np.fft.fft(a.astype(np.float64), axis=1)
    if lim_hi >= 1:
        y[:, :lim_hi] = 0
    if lim_hi >= 2:
        y[:, n - lim_hi + 1 :] = 0
    y[:, lim_lo - 1 : n - lim_lo + 1] = 0
    out = np.fft.ifft(y, axis=1).real.astype(a.dtype, copy=False)
    return out.ravel() if was_1d else out


# --------------------------------------------------------------------------- nGSC


def ngsc(data) -> float:
    """Normalised global spatial complexity of multichannel activity.

    Port of calliope ``21_Analysis/nGSC.m`` (Jia et al., Sci. Rep. 2018,
    doi:10.1038/s41598-018-26329-0). With ``lambda_i`` the PCA eigenvalues of the
    channels (observations are time points) normalised to sum 1,
    ``nGSC = -sum(lambda_i * log(lambda_i)) / log(n_channels)``.

    Parameters
    ----------
    data : array_like, shape (n_rois, n_timepoints)
        One channel per row.

    Returns
    -------
    float
        Between 0 (one component) and 1 (all components equal). NaN when an eigenvalue is
        exactly zero (``0 * log(0)``), as in MATLAB.

    Notes
    -----
    MATLAB ``pca`` is reproduced: time points with a NaN in any channel are dropped
    (``'Rows','complete'``), columns are centred, the eigenvalues are the squared singular
    values divided by ``n - 1``, and only the first ``min(n - 1, n_rois)`` are kept.
    """
    d = np.asarray(data, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError("data must be 2-D (n_rois x n_timepoints)")
    x = d.T
    x = x[~np.isnan(x).any(axis=1)]
    n, p = x.shape
    if n < 2:
        raise ValueError("need at least two time points without NaN")
    x = x - x.mean(axis=0)
    s = np.linalg.svd(x, compute_uv=False)
    latent = s**2 / (n - 1)
    latent = latent[: min(n - 1, p)]
    ev = latent / latent.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(-np.sum(np.log(ev) * ev) / math.log(d.shape[0]))


# --------------------------------------------------------------------------- AvgPsdInRange


def avg_psd_in_range(
    signal, fs: float, freq_range, use_db_averaging: bool = False, axis: int | None = None
):
    """Mean one-sided power spectral density of a signal within a frequency band.

    Port of calliope ``21_Analysis/AvgPsdInRange.m``. ``PSD = |fft(x)|**2 / (N * fs)``
    is cut to the one-sided spectrum (``N//2 + 1`` bins for even ``N``, ``(N+1)/2`` for
    odd ``N``), every bin except DC (and Nyquist for even ``N``) is doubled, and the bins
    with frequency ``k * fs / N`` inside ``[freq_range[0], freq_range[1]]`` (inclusive) are
    averaged, ignoring NaN.

    Parameters
    ----------
    signal : array_like
        Vector, matrix or N-D array.
    fs : float
        Sampling rate in Hz.
    freq_range : sequence of two floats
        ``[lower, upper]`` in Hz.
    use_db_averaging : bool, default False
        Average ``10*log10(PSD + eps)`` and convert back (geometric mean) instead of the
        linear mean.
    axis : int, optional
        **0-based** axis of the transform (MATLAB ``dim - 1``). Default: the first axis of
        length > 1 (axis 0 if none).

    Returns
    -------
    float or numpy.ndarray
        1-D input: a float. N-D input: an array with ``axis`` reduced to length 1, as MATLAB.
        ``nan`` (a float) when the signal is empty or shorter than 2 samples along ``axis``,
        or when no frequency bin lies in the range (with a ``UserWarning``, as MATLAB's
        ``warning``).

    Differences from MATLAB
    -----------------------
    - Integer input is converted to float64 (MATLAB ``fft`` rejects integer classes).
    - Frequencies are computed as ``k * (fs / N)``; MATLAB's colon operator
      ``0:fs/N:fs/2`` can differ in the last bit, which matters only for a band edge that
      falls exactly on a bin.
    """
    x = np.asarray(signal)
    if x.dtype.kind in "biu":
        x = x.astype(np.float64)
    if axis is None:
        nonsingle = [i for i, s in enumerate(x.shape) if s > 1]
        ax = nonsingle[0] if nonsingle else 0
    else:
        ax = int(axis)
    if x.size == 0 or x.ndim == 0 or x.shape[ax] < 2:
        return math.nan
    n = x.shape[ax]
    psd = np.abs(np.fft.fft(x, axis=ax)) ** 2 / (n * fs)
    cutoff = n // 2 + 1 if n % 2 == 0 else (n + 1) // 2
    psd = np.take(psd, np.arange(cutoff), axis=ax)
    stop = cutoff - 1 if n % 2 == 0 else cutoff
    sl = [slice(None)] * psd.ndim
    sl[ax] = slice(1, stop)
    psd[tuple(sl)] *= 2
    freqs = np.arange(cutoff) * (fs / n)
    sel = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if not sel.any():
        warnings.warn(
            f"No frequencies found within the specified range "
            f"[{freq_range[0]:.2f}, {freq_range[1]:.2f}] Hz",
            stacklevel=2,
        )
        return math.nan
    part = np.compress(sel, psd, axis=ax)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if use_db_averaging:
            mean_db = np.nanmean(10 * np.log10(part + _EPS), axis=ax, keepdims=True)
            out = 10 ** (mean_db / 10)
        else:
            out = np.nanmean(part, axis=ax, keepdims=True)
    if np.asarray(signal).ndim == 1:
        return float(out.ravel()[0])
    return out


# --------------------------------------------------------------------------- envelope


def _interp1_linear_extrap(xp: np.ndarray, fp: np.ndarray, x: np.ndarray) -> np.ndarray:
    """MATLAB ``interp1(xp, fp, x, 'linear', 'extrap')`` for increasing ``xp``."""
    out = np.interp(x, xp, fp)
    lo = x < xp[0]
    hi = x > xp[-1]
    if lo.any():
        s = (fp[1] - fp[0]) / (xp[1] - xp[0])
        out[lo] = fp[0] + (x[lo] - xp[0]) * s
    if hi.any():
        s = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        out[hi] = fp[-1] + (x[hi] - xp[-1]) * s
    return out


def extract_pulse_envelope(signal, *, threshold: float = 0.03, dt: int = 10) -> np.ndarray:
    """Piecewise-linear envelope of a pulse-modulated signal.

    Port of calliope ``21_Analysis/extract_pulse_envelope.m``. ``|signal| > threshold`` is
    binarised; supra-threshold segments shorter than ``round(dt/2)`` samples are removed;
    the rising edges of the cleaned mask are the pulse onsets. Gaps longer than 1.5 times
    the expected interval (rounded 25th percentile of the onset intervals) are filled with
    synthetic onsets every expected interval. At every onset the median of ``|signal|`` over
    ``dt + 1`` samples (fewer at the end) is taken, and these values are interpolated
    linearly (and extrapolated) to every sample.

    Parameters
    ----------
    signal : array_like
        The signal (any shape, flattened column-major as MATLAB ``signal(:)``).
    threshold : float, default 0.03
        Binarisation threshold (MATLAB ``'Threshold'``; the MATLAB help text mentions
        ``0.3*max(abs(signal))`` but the code uses 0.03).
    dt : int, default 10
        Averaging window in samples (MATLAB ``'dt'``).

    Returns
    -------
    numpy.ndarray
        float64 row ``(1, N)``, as MATLAB.

    Differences from MATLAB
    -----------------------
    - Fewer than two distinct onsets raise ``ValueError`` (MATLAB fails in ``interp1`` or
      when indexing an empty edge list).
    """
    s = np.asarray(signal, dtype=np.float64).ravel(order="F")
    n = s.size
    b = np.abs(s) > threshold
    edge = np.concatenate([[0], np.diff(b.astype(np.int8))])
    rising = np.flatnonzero(edge > 0)
    falling = np.flatnonzero(edge < 0)
    min_len = int(mround(dt / 2))
    cleaned = b.copy()
    for r in rising:
        nxt = falling[falling > r]
        if nxt.size and nxt[0] - r < min_len:
            cleaned[r : nxt[0]] = False
    edge_c = np.concatenate([[0], np.diff(cleaned.astype(np.int8))])
    edges = np.flatnonzero(edge_c > 0)  # 0-based onsets
    if edges.size == 0:
        raise ValueError("no pulse onsets found above the threshold")
    inter = np.diff(edges)
    expected = int(mround(prctile(inter, 25))) if inter.size else 0
    all_idx = []
    for i in range(edges.size - 1):
        cur, nxt = int(edges[i]), int(edges[i + 1])
        all_idx.append(cur)
        if nxt - cur > 1.5 * expected:
            all_idx.extend(range(cur + expected, nxt + 1, expected))
    all_idx.append(int(edges[-1]))
    all_idx = np.asarray(all_idx, dtype=np.int64)
    vals = np.array([np.median(np.abs(s[i : min(i + dt, n - 1) + 1])) for i in all_idx])
    uniq, first = np.unique(all_idx, return_index=True)
    if uniq.size < 2:
        raise ValueError("need at least two distinct pulse onsets to interpolate")
    env = _interp1_linear_extrap(
        (uniq + 1).astype(np.float64), vals[first], np.arange(1, n + 1, dtype=np.float64)
    )
    return env.reshape(1, -1)


# --------------------------------------------------------------------------- vis speed


def check_vis_speed(pd_data) -> np.ndarray:
    """Grating speed estimate from a photodiode trace.

    Port of calliope ``21_Analysis/check_vis_speed.m``. The trace is smoothed with
    :func:`smooth2` (50 samples), thresholded at
    ``p2 + 0.85 * (p98 - p2)`` (percentiles, :func:`calliopy._matlab.prctile`), and every
    interval between threshold crossings (plus the first and last sample) is assigned
    ``1000 * 0.5 / interval_length`` (cycles per second at 1 kHz sampling, half a cycle per
    interval); the boundary sample belongs to the later interval. Values above 10 are set
    to 10.

    Parameters
    ----------
    pd_data : array_like
        Photodiode vector (1 kHz sampling assumed, as in MATLAB).

    Returns
    -------
    numpy.ndarray
        float64 row ``(1, N)``.

    Differences from MATLAB
    -----------------------
    - A column vector is accepted (MATLAB fails when concatenating ``[1 find(...) N]``
      with a column of crossings).
    """
    x = np.asarray(pd_data, dtype=np.float64).ravel(order="F")
    x = smooth2(x, 50)
    thr = prctile(x, 2) + 0.85 * (prctile(x, 98) - prctile(x, 2))
    k = np.flatnonzero(np.diff((x > thr).astype(np.int8))) + 1  # 1-based, as MATLAB find
    crossings = np.concatenate([[1], k, [x.size]]).astype(np.int64)
    out = np.zeros(int(crossings.max()))
    with np.errstate(divide="ignore"):
        for a, b in itertools.pairwise(crossings):
            out[a - 1 : b] = 1000 * 0.5 / (b - a) if b != a else np.inf
    out[out > 10] = 10
    return out.reshape(1, -1)
