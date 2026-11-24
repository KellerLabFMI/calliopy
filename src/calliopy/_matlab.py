"""Private equivalents of the MATLAB built-ins that the ported calliope code needs.

Each helper reproduces the documented behaviour of one MATLAB function (or of the MathWorks
copy that calliope ships in ``06_Library``) with numpy/scipy. They are private: ported
modules import them from :mod:`calliopy._matlab`; they are not part of the public API.

=====================  ==============================================================
Python                 MATLAB
=====================  ==============================================================
:func:`mround`         ``round`` (halves away from zero)
:func:`linspace`       ``linspace(d1, d2, n)`` (MATLAB's arithmetic)
:func:`matlab_cast`    ``cast(x, class)`` / ``uint16(x)``, ... (round and saturate)
:func:`prctile`        ``prctile`` (``06_Library/prctile.m``, a MathWorks copy)
:func:`corr2`          ``corr2``
:func:`conv2`          ``conv2(a, k, shape)``
:func:`filter2`        ``filter2(k, a, shape)``
:func:`imgaussfilt`    ``imgaussfilt(a, sigma)``
:func:`normxcorr2`     ``normxcorr2(template, a)``
=====================  ==============================================================

None of these has been compared with MATLAB output; agreement rests on MATLAB's
documentation and on the tests in ``tests/test_matlab_helpers.py``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import scipy.ndimage
import scipy.signal

__all__ = [
    "conv2",
    "corr2",
    "filter2",
    "imgaussfilt",
    "linspace",
    "matlab_cast",
    "mround",
    "normxcorr2",
    "prctile",
]


def mround(x):
    """Round half away from zero, like MATLAB ``round``.

    Parameters
    ----------
    x
        Scalar or array. Integer and boolean input is returned unchanged.

    Returns
    -------
    numpy.ndarray or numpy scalar
        Same shape and (floating) dtype as ``x``; a numpy scalar for scalar input.

    Notes
    -----
    ``np.round`` rounds halves to even (``np.round(2.5) == 2``); MATLAB gives 3.
    The fractional part is taken as ``x - trunc(x)``, which is exact in floating point, so
    values just below a half (``0.49999999999999994``) are not rounded up. NaN and Inf
    pass through.
    """
    a = np.asarray(x)
    if a.dtype.kind in "biu":
        out = a.copy()
    else:
        t = np.trunc(a)
        with np.errstate(invalid="ignore"):
            frac = a - t
            out = t + np.where(np.abs(frac) >= 0.5, np.sign(a), 0).astype(a.dtype)
    return out[()] if out.ndim == 0 else out


def linspace(d1, d2, n=100) -> np.ndarray:
    """Evenly spaced values with MATLAB's arithmetic, MATLAB ``linspace(d1, d2, n)``.

    MATLAB computes ``d1 + (0:n-1).*(d2-d1)./(n-1)`` (product first, then division) and sets
    the end points to ``d1`` and ``d2`` exactly (all values to ``d1`` when ``d1 == d2``).
    ``numpy.linspace`` uses ``d1 + k*step``, which can differ in the last bit and so change a
    following :func:`mround` at a half.

    Parameters
    ----------
    d1, d2
        Real end points.
    n
        Number of points (default 100, as MATLAB); floored. ``n < 1`` gives an empty
        array, ``n == 1`` gives ``[d2]`` (as MATLAB).

    Returns
    -------
    numpy.ndarray
        float64 1-D array of length ``max(floor(n), 0)`` (MATLAB returns a 1 x n row vector:
        use ``.reshape(1, -1)`` where the shape matters).

    Notes
    -----
    When ``(d2 - d1)*(n - 2)`` overflows, MATLAB divides before multiplying
    (``d1 + (0:n-1).*((d2-d1)./(n-1))``, or, if ``d2 - d1`` itself overflows,
    ``d1 + (d2./(n-1)).*(0:n-1) - (d1./(n-1)).*(0:n-1)``); the same branches are used here.
    Not compared with MATLAB output: agreement in the last bit is unverified.
    """
    n = math.floor(n)
    if n < 1:
        return np.empty(0)
    d1 = float(d1)
    d2 = float(d2)
    if n == 1:
        return np.array([d2])
    n1 = n - 1
    k = np.arange(n, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        if math.isinf((d2 - d1) * (n1 - 1)):
            if math.isinf(d2 - d1):
                y = d1 + (d2 / n1) * k - (d1 / n1) * k
            else:
                y = d1 + k * ((d2 - d1) / n1)
        else:
            y = d1 + k * (d2 - d1) / n1
    if d1 == d2:
        y[:] = d1
    else:
        y[0] = d1
        y[-1] = d2
    return y


def matlab_cast(x, dtype):
    """Convert to a numeric class the way MATLAB does (``cast(x, class)``, ``uint16(x)``).

    Parameters
    ----------
    x
        Scalar or array.
    dtype
        Target numpy dtype. Integer targets: round half away from zero, saturate at the
        type's limits, NaN becomes 0. Floating targets: plain ``astype``. Boolean target:
        nonzero is ``True`` (MATLAB ``logical``); NaN raises.

    Returns
    -------
    numpy.ndarray or numpy scalar
        Array of ``dtype`` with the shape of ``x``; a numpy scalar for scalar input.

    Notes
    -----
    Use it wherever MATLAB arithmetic mixes a ``double`` with an integer or ``single``
    value: MATLAB computes the result and converts it to the integer/``single`` class
    (``uint16(3) - 5.4`` is ``0``, ``uint16(3)/2`` is ``2``); numpy would wrap and truncate.

    Differences from MATLAB: int64/uint64 values beyond 2**53 lose precision because the
    conversion goes through float64.
    """
    dt = np.dtype(dtype)
    a = np.asarray(x)
    if dt.kind in "iu":
        if a.dtype.kind in "biu":
            info = np.iinfo(dt)
            if a.dtype.kind == "u":
                top = min(info.max, np.iinfo(a.dtype).max)
                out = np.minimum(a, np.asarray(top, dtype=a.dtype)).astype(dt)
            else:
                out = np.clip(a.astype(np.int64), max(info.min, -(2**63)), min(info.max, 2**63 - 1))
                out = out.astype(dt)
        else:
            info = np.iinfo(dt)
            r = mround(a.astype(np.float64))
            r = np.nan_to_num(np.asarray(r), nan=0.0, posinf=info.max, neginf=info.min)
            out = np.clip(r, info.min, info.max).astype(dt)
    elif dt.kind == "b":
        if a.dtype.kind == "f" and np.isnan(a).any():
            raise ValueError("NaN cannot be converted to logical")
        out = a != 0
    else:
        out = a.astype(dt)
    out = np.asarray(out)
    return out[()] if out.ndim == 0 else out


def prctile(x, p, axis=None):
    """Percentiles as MATLAB ``prctile`` computes them (NaNs ignored).

    Replaces ``prctile`` (calliope ships the MathWorks copy ``06_Library/prctile.m``, which
    shadows the toolbox function). For ``n`` values the sorted data are the
    ``100*(0.5:n-0.5)/n`` percentiles; values in between are interpolated linearly and
    percentiles outside that range take the minimum or maximum. This is numpy's
    ``method="hazen"``.

    Parameters
    ----------
    x
        Data, any shape.
    p
        Percent value(s) in ``[0, 100]``: a scalar or a non-empty 1-D sequence.
    axis
        ``None`` (default): over all elements, as MATLAB ``prctile(x(:), p)``. Otherwise
        the 0-based numpy axis (MATLAB ``dim - 1``).

    Returns
    -------
    numpy.ndarray or numpy scalar
        ``axis=None``: a scalar for scalar ``p``, else an array of shape ``(len(p),)``.
        With ``axis``: the shape of ``x`` with ``axis`` replaced by ``len(p)`` (1 for scalar
        ``p``), as MATLAB ``prctile(x, p, dim)``. float32 for float32 input, else float64.
        An all-NaN or empty slice gives NaN.

    Notes
    -----
    Differences from MATLAB: MATLAB's default dimension (first non-singleton) is not used;
    pass ``axis`` explicitly. For integer input MATLAB returns the integer class (the
    interpolated value rounded); here the result is float64: apply :func:`mround` (or
    :func:`matlab_cast`) where the MATLAB code relied on that.
    """
    a = np.asarray(x)
    out_dtype = np.float32 if a.dtype == np.float32 else np.float64
    pa = np.asarray(p, dtype=np.float64)
    if pa.ndim > 1 or pa.size == 0:
        raise ValueError("p must be a scalar or a non-empty vector")
    if np.any((pa < 0) | (pa > 100)) or np.any(np.isnan(pa)):
        raise ValueError("p must take values between 0 and 100")
    af = a.astype(np.float64)
    if axis is None:
        af = af.ravel(order="F")
        ax = 0
    else:
        ax = axis % max(a.ndim, 1)
    if af.shape[ax] == 0:
        shape = list(af.shape)
        del shape[ax]
        y = np.full((pa.size, *shape), np.nan)
        if pa.ndim == 0:
            y = y[0]
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            y = np.nanpercentile(af, pa, axis=ax, method="hazen")
    if axis is None:
        y = np.asarray(y, dtype=out_dtype)
        return y[()] if y.ndim == 0 else y
    y = np.asarray(y)
    y = np.expand_dims(y, ax) if pa.ndim == 0 else np.moveaxis(y, 0, ax)
    return y.astype(out_dtype)


def corr2(a, b) -> float:
    """2-D correlation coefficient, MATLAB ``corr2(a, b)``.

    Parameters
    ----------
    a, b
        Arrays of the same shape (MATLAB: 2-D matrices); converted to float64.

    Returns
    -------
    float
        ``sum((a-mean(a)).*(b-mean(b))) / sqrt(sum((a-mean(a)).^2) * sum((b-mean(b)).^2))``.
        NaN when either input is constant or contains NaN, as in MATLAB.

    Notes
    -----
    Differences from MATLAB: arrays of any number of dimensions are accepted (MATLAB
    requires 2-D); they are compared element by element.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {a.shape} and {b.shape}")
    am = a - a.mean()
    bm = b - b.mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.sum(am * bm) / math.sqrt(np.sum(am * am) * np.sum(bm * bm)))


def _as_2d(x, name):
    a = np.asarray(x)
    if a.ndim == 1:
        a = a[np.newaxis, :]
    elif a.ndim == 0:
        a = a.reshape(1, 1)
    elif a.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {a.shape}")
    if a.dtype.kind in "biu":
        a = a.astype(np.float64)
    return a


def conv2(a, k, shape: str = "full") -> np.ndarray:
    """2-D convolution with MATLAB's output sizes, MATLAB ``conv2(a, k, shape)``.

    Parameters
    ----------
    a
        Image, ``(ma, na)``. A 1-D array is treated as a row vector ``(1, n)``.
    k
        Kernel, ``(mk, nk)``; 1-D as for ``a``.
    shape
        ``'full'``: ``(ma+mk-1, na+nk-1)``. ``'same'``: the central part of the full result
        with the size of ``a``, starting at 0-based row ``mk // 2`` and column ``nk // 2``
        of the full result (for even kernel sizes this is one sample further than
        ``scipy.signal.convolve2d(..., mode="same")``). ``'valid'``: only the samples
        computed without zero padding, ``(max(ma-mk+1, 0), max(na-nk+1, 0))``.

    Returns
    -------
    numpy.ndarray
        float32 if either input is float32 (MATLAB ``single``), else float64 (complex if an
        input is complex). Integer input is converted to float64 (MATLAB requires
        floating-point input).

    Notes
    -----
    Direct (not FFT) convolution, like MATLAB. Differences from MATLAB: the separable
    form ``conv2(u, v, A)`` is not provided. The output of empty inputs (zeros of the
    sizes above) is unverified.
    """
    if shape not in ("full", "same", "valid"):
        raise ValueError(f"shape must be 'full', 'same' or 'valid', got {shape!r}")
    a2 = _as_2d(a, "a")
    k2 = _as_2d(k, "k")
    single = np.float32 in (a2.dtype.type, k2.dtype.type)
    ma, na = a2.shape
    mk, nk = k2.shape
    rtype = np.result_type(a2.dtype, k2.dtype, np.float64)
    if a2.size == 0 or k2.size == 0:
        if shape == "full":
            sz = (max(ma + mk - 1, 0), max(na + nk - 1, 0))
        elif shape == "same":
            sz = (ma, na)
        else:
            sz = (max(ma - max(0, mk - 1), 0), max(na - max(0, nk - 1), 0))
        out = np.zeros(sz, dtype=rtype)
    else:
        full = scipy.signal.convolve2d(a2.astype(rtype), k2.astype(rtype), mode="full")
        if shape == "full":
            out = full
        elif shape == "same":
            r0, c0 = mk // 2, nk // 2
            out = full[r0 : r0 + ma, c0 : c0 + na]
        else:
            out = full[mk - 1 : ma, nk - 1 : na]
            if out.size == 0:
                out = np.zeros((max(ma - mk + 1, 0), max(na - nk + 1, 0)), dtype=rtype)
    if single:
        out = out.astype(np.complex64 if np.iscomplexobj(out) else np.float32)
    return np.ascontiguousarray(out)


def filter2(k, a, shape: str = "same") -> np.ndarray:
    """2-D correlation with zero padding, MATLAB ``filter2(k, a, shape)``.

    MATLAB defines ``filter2(k, a, shape)`` as ``conv2(a, rot90(k, 2), shape)``; this is
    exactly how it is computed here (see :func:`conv2` for the output sizes and for the
    position of the ``'same'`` window with even-sized kernels).

    Parameters
    ----------
    k
        Kernel, ``(mk, nk)``.
    a
        Image, ``(ma, na)``; integer images are converted to float64, as in MATLAB.
    shape
        ``'same'`` (default), ``'full'`` or ``'valid'``.

    Returns
    -------
    numpy.ndarray
        As :func:`conv2`.

    Notes
    -----
    Differences from MATLAB: MATLAB evaluates separable kernels as two 1-D
    convolutions, which can differ from the 2-D sum in the last bits.
    """
    k2 = _as_2d(k, "k")
    return conv2(a, k2[::-1, ::-1], shape)


def imgaussfilt(a, sigma=0.5) -> np.ndarray:
    """2-D Gaussian smoothing, MATLAB ``imgaussfilt(a, sigma)`` with default options.

    The kernel has ``2*ceil(2*sigma)+1`` taps per dimension, is normalised to sum 1 over
    those taps, and the image is extended by replicating its border ('replicate').

    Parameters
    ----------
    a
        Image ``(height, width)`` or a stack ``(height, width, ...)``; every 2-D plane is
        filtered separately, as in MATLAB.
    sigma
        Positive standard deviation in pixels: a scalar, or ``(sigma_row, sigma_col)``.

    Returns
    -------
    numpy.ndarray
        Same shape and dtype as ``a``. Integer images are filtered in float64 and converted
        back with :func:`matlab_cast` (rounded, saturated); float32 stays float32.

    Notes
    -----
    Unverified: the order ``(sigma_row, sigma_col)`` of a 2-element ``sigma``; the
    rounding for integer images; MATLAB's frequency-domain path for large kernels
    ('FilterDomain' 'auto') is assumed to agree up to floating-point rounding.
    Differences from MATLAB: the name-value options ('FilterSize', 'Padding',
    'FilterDomain') are not provided.
    """
    arr = np.asarray(a)
    if arr.ndim < 2:
        raise ValueError(f"a must be at least 2-D, got shape {arr.shape}")
    s = np.broadcast_to(np.asarray(sigma, dtype=np.float64), (2,))
    if np.any(~(s > 0)):
        raise ValueError(f"sigma must be positive, got {sigma!r}")
    work = np.float32 if arr.dtype == np.float32 else np.float64
    sig = [float(s[0]), float(s[1])] + [0.0] * (arr.ndim - 2)
    rad = [math.ceil(2 * s[0]), math.ceil(2 * s[1])] + [0] * (arr.ndim - 2)
    out = scipy.ndimage.gaussian_filter(
        arr.astype(work), sigma=sig, mode="nearest", radius=rad
    )
    if arr.dtype.kind in "biu":
        return matlab_cast(out, arr.dtype)
    return out


def _local_sum(x, m, n):
    """Sum over every m x n window of ``x`` zero-padded to the 'full' correlation size."""
    p = np.pad(x, ((m, m - 1), (n, n - 1)))
    c = np.cumsum(p, axis=0)
    c = c[m:] - c[:-m]
    c = np.cumsum(c, axis=1)
    return c[:, n:] - c[:, :-n]


def normxcorr2(template, a) -> np.ndarray:
    """Normalised 2-D cross-correlation, MATLAB ``normxcorr2(template, a)``.

    Implements the published definition (J. P. Lewis, "Fast normalized
    cross-correlation", 1995): for every placement of ``template`` over ``a`` (zero-padded
    outside ``a``)::

        C = sum((f - mean_window(f)) * (t - mean(t)))
            / sqrt(sum((f - mean_window(f))**2) * sum((t - mean(t))**2))

    The numerator is computed with :func:`scipy.signal.fftconvolve`, the window sums of
    ``f`` and ``f**2`` with running (cumulative) sums.

    Parameters
    ----------
    template
        ``(m, n)`` array, not larger than ``a`` in either dimension, at least 2 elements and
        not constant.
    a
        ``(M, N)`` image.

    Returns
    -------
    numpy.ndarray
        float64 ``(M+m-1, N+n-1)`` ('full' size, as MATLAB). Element ``[i, j]`` (0-based)
        places the template's last row/column on ``a[i, j]``; a peak at ``[i, j]`` means the
        template's top-left corner sits at 0-based ``(i - m + 1, j - n + 1)`` in ``a``.
        Windows whose variance is numerically zero give 0.

    Notes
    -----
    The zero threshold is ``sqrt(eps(max(denominator)))`` on the denominator; this and
    whether MATLAB clips values that exceed 1 by rounding are unverified. Differences
    from MATLAB: errors are :class:`ValueError`.
    """
    t = np.asarray(template, dtype=np.float64)
    f = np.asarray(a, dtype=np.float64)
    if t.ndim != 2 or f.ndim != 2:
        raise ValueError("template and a must be 2-D")
    m, n = t.shape
    if t.size < 2:
        raise ValueError("template must contain at least 2 elements")
    if m > f.shape[0] or n > f.shape[1]:
        raise ValueError("template must not be larger than a in either dimension")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(f))):
        raise ValueError("template and a must be finite")
    t0 = t - t.mean()
    sum_t2 = float(np.sum(t0 * t0))
    if sum_t2 == 0 or np.all(t == t.flat[0]):
        raise ValueError("the values of template cannot all be the same")
    num = scipy.signal.fftconvolve(f, t0[::-1, ::-1], mode="full")
    ls = _local_sum(f, m, n)
    ls2 = _local_sum(f * f, m, n)
    var_f = np.maximum(ls2 - ls * ls / (m * n), 0.0)
    denom = np.sqrt(var_f * sum_t2)
    tol = math.sqrt(np.spacing(np.max(np.abs(denom)))) if denom.size else 0.0
    out = np.zeros_like(num)
    ok = denom > tol
    out[ok] = num[ok] / denom[ok]
    return out
