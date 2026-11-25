"""Automatic multiscale peak detection: calliope ``21_Analysis/ampd.m``.

AMPD is the algorithm of Scholkmann, Boss & Wolf (2012), Algorithms 5(4):588-603; the
'extended' and 'adaptive' (ASS-AMPD) modes follow the variants of ampdLib (L. Cerina) that
calliope's ``ampd.m`` was translated from. This module implements the algorithm as
``ampd.m`` defines it.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.signal

__all__ = ["AmpdAdaptiveDebug", "AmpdExtendedDebug", "AmpdOriginalDebug", "ampd"]


class AmpdOriginalDebug(NamedTuple):
    """``[pks, LSM, l_scale]`` of mode 'original' (``l_scale`` 1-based row number as MATLAB)."""

    pks: np.ndarray
    lsm: np.ndarray
    l_scale: int


class AmpdExtendedDebug(NamedTuple):
    """``[pks, LSM, G, l_scale]`` of mode 'extended' (``l_scale`` 1-based row number)."""

    pks: np.ndarray
    lsm: np.ndarray
    g: np.ndarray
    l_scale: int


class AmpdAdaptiveDebug(NamedTuple):
    """``[pks, ass_LSM, adaptive_scale]`` of mode 'adaptive' (scales 0-based, as MATLAB)."""

    pks: np.ndarray
    ass_lsm: np.ndarray
    adaptive_scale: np.ndarray


def _lsm_both_sides(x: np.ndarray, n_rows: int) -> np.ndarray:
    """Local scalogram: row k-1 true where x is larger than both neighbours at distance k
    (a missing neighbour at the edge counts as smaller)."""
    n = x.size
    lsm = np.ones((n_rows, n), dtype=bool)
    for k in range(1, n_rows + 1):
        if k >= n:
            lsm[k - 1, :] = True  # no neighbour on either side: MATLAB leaves the row true
            continue
        lsm[k - 1, : n - k] &= x[: n - k] > x[k:]
        lsm[k - 1, k:] &= x[k:] > x[: n - k]
    return lsm


def _movmean_shrink(a: np.ndarray, window: int) -> np.ndarray:
    """MATLAB ``movmean(a, window, 2, 'Endpoints', 'shrink')`` along the columns."""
    n = a.shape[1]
    before = window // 2 if window % 2 == 0 else (window - 1) // 2
    after = window // 2 - 1 if window % 2 == 0 else (window - 1) // 2
    c = np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(a, axis=1)], axis=1)
    j = np.arange(n)
    lo = np.maximum(j - before, 0)
    hi = np.minimum(j + after, n - 1) + 1
    return (c[:, hi] - c[:, lo]) / (hi - lo)


def ampd(x, *, mode: str = "extended", scale: int | None = None, window: int | None = None,
         debug: bool = False):
    """Find peaks of a noisy, periodic or quasi-periodic signal with AMPD.

    Port of calliope ``21_Analysis/ampd.m``. The signal is linearly detrended first
    (MATLAB ``detrend``).

    Parameters
    ----------
    x : array_like
        Signal (flattened).
    mode : {'extended', 'original', 'adaptive'}, default 'extended'
        'original': AMPD of Scholkmann et al. (2012). 'extended': edge-robust local
        scalogram with scale weights. 'adaptive': ASS-AMPD with a per-sample scale.
    scale : int, optional
        Largest scale for 'original' and 'extended' (default ``len(x) // 2``).
    window : int, optional
        Window of 'adaptive' (default ``len(x)``).
    debug : bool, default False
        Also return the intermediate arrays (see the ``Ampd*Debug`` tuples).

    Returns
    -------
    numpy.ndarray or NamedTuple
        **0-based** sample positions of the peaks (int64), or a ``Ampd*Debug`` tuple whose
        ``pks`` field holds them.

    Differences from MATLAB
    -----------------------
    - Peak positions are 0-based int64 (MATLAB: 1-based int32 column).
    - The local scalogram is returned as a boolean array (rows = scales).
    """
    m = str(mode).lower()
    if m not in ("original", "extended", "adaptive"):
        raise ValueError('Unknown mode. Use "original", "extended", or "adaptive".')
    xs = np.asarray(x, dtype=np.float64).ravel(order="F")
    n = xs.size
    empty = np.empty(0, dtype=np.int64)
    if n:
        xs = scipy.signal.detrend(xs, type="linear") if n > 1 else xs - xs.mean()

    if m == "original":
        L = n // 2
        if scale is not None:
            L = min(int(scale), L)
        if L < 2:
            return empty
        lsm = np.zeros((L, n), dtype=bool)
        for k in range(1, L):
            i = np.arange(k, n - k)
            lsm[k - 1, i] = (xs[i - k] < xs[i]) & (xs[i] > xs[i + k])
        g = lsm.sum(axis=1)
        l_scale = int(np.argmax(g)) + 1
        pk = lsm[0] if l_scale <= 1 else lsm[: l_scale - 1].all(axis=0)
        pks = np.flatnonzero(pk).astype(np.int64)
        return AmpdOriginalDebug(pks, lsm, l_scale) if debug else pks

    if m == "extended":
        L = n // 2
        if scale is not None:
            L = min(int(scale), L)
        if L < 1:
            return empty
        lsm = _lsm_both_sides(xs, L)
        w = np.arange(n // 2, n // 2 - L, -1, dtype=np.float64)
        g = lsm.sum(axis=1) * w
        l_scale = int(np.argmax(g)) + 1
        pk = lsm[0] if l_scale <= 1 else lsm[: l_scale - 1].all(axis=0)
        pks = np.flatnonzero(pk).astype(np.int64)
        return AmpdExtendedDebug(pks, lsm, g, l_scale) if debug else pks

    # adaptive
    win = n if (window is None or window > n) else int(window)
    L = win // 2
    if L < 1:
        return empty
    lsm = _lsm_both_sides(xs, L)
    ass = _movmean_shrink(lsm.astype(np.float64) * win, win) * np.arange(L, 0, -1, dtype=np.float64)[:, None]
    scale0 = np.argmax(ass, axis=0).astype(np.int64)
    r = int(scale0.max())
    if r == 0:
        pk = lsm[0]
    else:
        red = lsm[:r].copy()
        red[np.arange(r)[:, None] > scale0[None, :]] = True
        pk = red.all(axis=0)
    pks = np.flatnonzero(pk).astype(np.int64)
    return AmpdAdaptiveDebug(pks, ass, scale0) if debug else pks
