"""Private equivalents of MATLAB built-ins used by the 2P / EEG / OII ports.

These are candidates for :mod:`calliopy._matlab` (consolidation is left to the integration
step; this module is owned by the 2P/EEG/OII track). Each reproduces the documented MATLAB
behaviour with numpy/scipy; MathWorks code is not translated.

None of these functions has been compared with MATLAB output.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from .._matlab import filter2, matlab_cast

__all__ = [
    "fspecial_average",
    "fspecial_gaussian",
    "imfilter",
    "impyramid_reduce",
    "mat2gray",
    "matchpairs",
    "matlab_colon",
    "ntzo",
    "smooth_moving",
    "sub2ind_f",
]


def matlab_colon(a: float, step: float, b: float) -> np.ndarray:
    """MATLAB ``a:step:b`` for floating point operands.

    The number of elements is ``floor((b - a) / step)`` (with a relative tolerance of a few
    eps, as MATLAB) plus one. As MATLAB's colon, the first half of the values is computed
    from the start (``a + k*step``) and the second half from the end point
    (``last - (n-1-k)*step``), where ``last`` is ``b`` when it lies on the grid within that
    tolerance; this keeps ``b`` exact and the vector symmetric (unverified in the last bit
    for unusual step sizes).
    """
    if step == 0 or (b - a) / step < 0:
        return np.zeros(0)
    q = (b - a) / step
    n = int(np.floor(q + 4 * np.finfo(float).eps * max(1.0, abs(q)))) + 1
    k = np.arange(n, dtype=np.float64)
    last = a + (n - 1) * step
    if abs(last - b) <= 4 * np.finfo(float).eps * max(abs(a), abs(b), 1.0) * max(n, 1):
        last = b
    out = a + k * step
    hi = k > (n - 1) / 2
    out[hi] = last - (n - 1 - k[hi]) * step
    return out


def sub2ind_f(shape, *subs) -> np.ndarray:
    """1-based column-major linear index from 1-based subscripts (MATLAB ``sub2ind``)."""
    subs0 = [np.asarray(s, dtype=np.int64) - 1 for s in subs]
    return np.ravel_multi_index(subs0, shape, order="F") + 1


def ntzo(x) -> np.ndarray:
    """Normalise to the range 0..1 (calliope ``06_Library/ntzo.m``, one input).

    Integer input is converted to float32 (MATLAB ``single``); the minimum is subtracted and,
    when the result is not constant, it is divided by its maximum. NaNs are ignored by
    min/max, as MATLAB ``min``/``max`` do.
    """
    a = np.asarray(x)
    if a.dtype.kind in "iub":
        a = a.astype(np.float32)
    elif a.dtype.kind != "f":
        a = a.astype(np.float64)
    a = a - np.nanmin(a)
    mx, mn = np.nanmax(a), np.nanmin(a)
    if mx > mn:
        return a / mx
    return a


def mat2gray(a, limits=None) -> np.ndarray:
    """MATLAB ``mat2gray``: scale linearly so that ``limits`` map to 0 and 1, then clip.

    Default limits are the minimum and maximum of ``a`` (NaN ignored). NaNs stay NaN.
    For constant input (``limits[0] == limits[1]``) the input is returned as float64
    (unverified MATLAB edge case).
    """
    x = np.asarray(a, dtype=np.float64)
    if limits is None:
        if x.size == 0 or np.all(np.isnan(x)):
            return x.copy()
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    else:
        lo, hi = float(limits[0]), float(limits[1])
    if hi == lo:
        return x.copy()
    out = (x - lo) / (hi - lo)
    with np.errstate(invalid="ignore"):
        out = np.where(out < 0, 0.0, np.where(out > 1, 1.0, out))
    return out


def fspecial_gaussian(hsize=3, sigma=0.5) -> np.ndarray:
    """MATLAB ``fspecial('gaussian', hsize, sigma)``.

    ``exp(-(x^2 + y^2) / (2 sigma^2))`` on the grid ``-(hsize-1)/2 .. (hsize-1)/2``, values
    below ``eps * max`` set to 0, normalised to sum 1. ``hsize`` may be a scalar or
    ``(rows, cols)``.
    """
    hs = np.atleast_1d(np.asarray(hsize, dtype=np.float64))
    if hs.size == 1:
        hs = np.array([hs[0], hs[0]])
    sr, sc = (hs[0] - 1) / 2, (hs[1] - 1) / 2
    y, x = np.meshgrid(np.arange(-sr, sr + 1), np.arange(-sc, sc + 1), indexing="ij")
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(float).eps * h.max()] = 0
    s = h.sum()
    if s != 0:
        h = h / s
    return h


def fspecial_average(hsize=3) -> np.ndarray:
    """MATLAB ``fspecial('average', hsize)``: ``ones(hsize) / numel``."""
    hs = np.atleast_1d(np.asarray(hsize, dtype=np.int64))
    if hs.size == 1:
        hs = np.array([hs[0], hs[0]])
    return np.full((int(hs[0]), int(hs[1])), 1.0 / (hs[0] * hs[1]))


def imfilter(a, h) -> np.ndarray:
    """MATLAB ``imfilter(a, h)`` with its defaults: correlation, zero padding, ``'same'``.

    Every 2-D plane of ``a`` is filtered with the 2-D kernel ``h``
    (:func:`calliopy._matlab.filter2`). The result has the class of ``a``: integer input is
    rounded half away from zero and saturated (MATLAB), float32 stays float32.

    Differences from MATLAB: MATLAB may use separable or FFT-based filtering internally;
    float results can differ in the last bits.
    """
    arr = np.asarray(a)
    k = np.asarray(h, dtype=np.float64)
    if arr.ndim == 2:
        out = filter2(k, arr.astype(np.float64), "same")
    else:
        flat = arr.reshape(arr.shape[0], arr.shape[1], -1, order="F")
        out = np.empty(flat.shape, dtype=np.float64)
        for i in range(flat.shape[2]):
            out[:, :, i] = filter2(k, flat[:, :, i].astype(np.float64), "same")
        out = out.reshape(arr.shape, order="F")
    if arr.dtype.kind in "iu":
        return matlab_cast(out, arr.dtype)
    if arr.dtype == np.float32:
        return out.astype(np.float32)
    return out


_PYR_KERNEL = np.array([0.25 - 0.375 / 2, 0.25, 0.375, 0.25, 0.25 - 0.375 / 2])


def impyramid_reduce(a) -> np.ndarray:
    """One level of Gaussian pyramid reduction, as MATLAB ``impyramid(a, 'reduce')``.

    Implemented from Burt & Adelson (1983), the algorithm MATLAB documents: the 5-tap
    kernel ``[1/4 - a/2, 1/4, a, 1/4, 1/4 - a/2]`` with ``a = 0.375`` is applied along rows
    and columns with symmetric boundary extension, then every second row and column is kept
    starting with the first, giving ``ceil(M/2) x ceil(N/2)`` (further dimensions are kept).
    The result has the class of ``a`` (integers rounded and saturated).

    Notes
    -----
    MathWorks code is not translated. That MATLAB's implementation matches this at the image
    border and in the last bits is unverified.
    """
    arr = np.asarray(a)
    x = arr.astype(np.float64)
    x = ndimage.correlate1d(x, _PYR_KERNEL, axis=0, mode="reflect")
    x = ndimage.correlate1d(x, _PYR_KERNEL, axis=1, mode="reflect")
    x = x[::2, ::2]
    if arr.dtype.kind in "iu":
        return matlab_cast(x, arr.dtype)
    if arr.dtype == np.float32:
        return x.astype(np.float32)
    return x


def smooth_moving(y, span: float) -> np.ndarray:
    """Moving-average smoothing as Curve Fitting Toolbox ``smooth(y, span)`` (``'moving'``).

    The window width is made odd (``span - 1 + mod(span, 2)`` after flooring ``span`` and
    limiting it to the data length); interior points are the centred mean over the window,
    and the first and last ``(width-1)/2`` points use the largest symmetric window that fits
    (1, 3, 5, ... samples), as the toolbox documents.

    Notes
    -----
    Flooring a non-integer ``span`` and limiting it to the data length are assumptions
    (unverified against the toolbox).
    """
    v = np.asarray(y, dtype=np.float64).ravel(order="F")
    n = v.size
    if n == 0:
        return v.copy()
    span = int(np.floor(span))
    span = max(1, min(span, n))
    width = span - 1 + span % 2
    if width <= 1:
        return v.copy()
    half = (width - 1) // 2
    c = np.convolve(v, np.ones(width) / width, mode="valid")
    out = np.empty(n)
    out[half : n - half] = c
    cs = np.cumsum(v)
    for i in range(half):
        w = 2 * i + 1
        out[i] = cs[w - 1] / w
        out[n - 1 - i] = v[n - w :].sum() / w
    return out


def matchpairs(cost, cost_unmatched: float) -> np.ndarray:
    """Minimum-cost partial assignment, as MATLAB ``matchpairs(Cost, costUnmatched)``.

    Minimises the sum of the costs of the matched pairs plus ``cost_unmatched`` for every
    unmatched row and every unmatched column (so a pair is only matched when its cost is at
    most ``2 * cost_unmatched``), solved with
    :func:`scipy.optimize.linear_sum_assignment` on the standard augmented matrix.

    Returns
    -------
    numpy.ndarray
        ``(nPairs, 2)`` int64 array of **0-based** ``(row, column)`` pairs, sorted by row.

    Notes
    -----
    When several assignments have the same total cost, the one chosen can differ from
    MATLAB's (tie-breaking is implementation specific).
    """
    c = np.asarray(cost, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError("cost must be 2-D")
    n, m = c.shape
    if n == 0 or m == 0:
        return np.zeros((0, 2), dtype=np.int64)
    finite = np.isfinite(c)
    scale = float(np.abs(c[finite]).max()) if finite.any() else 0.0
    big = (2 * abs(cost_unmatched) + scale + 1) * (n + m + 1)
    aug = np.full((n + m, m + n), big)
    aug[:n, :m] = np.where(finite, c, big)
    aug[:n, m:] = np.where(np.eye(n, dtype=bool), cost_unmatched, big)
    aug[n:, :m] = np.where(np.eye(m, dtype=bool), cost_unmatched, big)
    aug[n:, m:] = 0
    r, k = linear_sum_assignment(aug)
    keep = (r < n) & (k < m)
    pairs = np.stack([r[keep], k[keep]], axis=1).astype(np.int64)
    if pairs.size:
        pairs = pairs[finite[pairs[:, 0], pairs[:, 1]]]
    if not pairs.size:
        return np.zeros((0, 2), dtype=np.int64)
    return pairs[np.argsort(pairs[:, 0], kind="stable")]
