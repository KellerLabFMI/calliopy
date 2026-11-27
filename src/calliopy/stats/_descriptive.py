"""Descriptive statistics: standard error of the mean ignoring NaN (calliope ``06_Library``)."""

from __future__ import annotations

import math

import numpy as np

__all__ = ["nan_sem"]


def nan_sem(x, axis: int | None = None):
    """Standard error of the mean, ignoring NaN.

    Replaces calliope ``06_Library/nan_sem.m`` (from the third-party "NaN suite" by
    J. Glaescher, not translated): ``std(x, [], dim, 'omitnan') ./ sqrt(count)`` where
    ``count`` is the number of non-NaN values; slices without any value give NaN.
    Implemented here from that definition with numpy.

    Parameters
    ----------
    x : array_like
        Data, any shape.
    axis : int, optional
        **0-based** axis (MATLAB ``dim - 1``). Default: the first axis whose length is not
        1 (axis 0 if all are 1), as MATLAB.

    Returns
    -------
    float or numpy.ndarray
        1-D input: a float. Otherwise float64 with ``axis`` kept as length 1 (MATLAB keeps
        the reduced dimension). Empty input gives ``nan``.

    Notes
    -----
    The standard deviation normalises by ``count - 1``; a slice with exactly one value has
    standard deviation 0 (MATLAB ``std`` of a scalar), so its SEM is 0, unlike
    ``scipy.stats.sem`` (NaN).
    """
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return math.nan
    if a.ndim == 0:
        a = a.reshape(1)
    if axis is None:
        nonsingle = [i for i, s in enumerate(a.shape) if s != 1]
        ax = nonsingle[0] if nonsingle else 0
    else:
        ax = int(axis)
    valid = ~np.isnan(a)
    count = valid.sum(axis=ax, keepdims=True)
    filled = np.where(valid, a, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = filled.sum(axis=ax, keepdims=True) / count
        dev = np.where(valid, a - mean, 0.0)
        ss = (dev * dev).sum(axis=ax, keepdims=True)
        var = np.where(count > 1, ss / np.maximum(count - 1, 1), 0.0)
        y = np.sqrt(var) / np.sqrt(np.maximum(count, 1))
    y = np.where(count == 0, np.nan, y)
    if np.asarray(x).ndim <= 1:
        return float(y.ravel()[0])
    return y
