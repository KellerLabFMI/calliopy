"""Several bootstrapped traces with per-time-point comparison strips: calliope
``22_Plotting/plotBSTRP_multi.m``."""

from __future__ import annotations

from itertools import combinations
from typing import Any, NamedTuple

import numpy as np

from ._bstrp import plot_bstrp

__all__ = ["PlotBstrpMultiResult", "bootstrap_overlap_pvalues", "plot_bstrp_multi"]

# MATLAB's default axes colour order, as returned by lines(7)
_LINES = np.array(
    [
        [0.0, 0.4470, 0.7410],
        [0.8500, 0.3250, 0.0980],
        [0.9290, 0.6940, 0.1250],
        [0.4940, 0.1840, 0.5560],
        [0.4660, 0.6740, 0.1880],
        [0.3010, 0.7450, 0.9330],
        [0.6350, 0.0780, 0.1840],
    ]
)


def _lines(n: int) -> np.ndarray:
    """MATLAB ``lines(n)`` with the default colour order (repeated cyclically)."""
    return _LINES[np.arange(n) % len(_LINES)]


class PlotBstrpMultiResult(NamedTuple):
    """Outputs of :func:`plot_bstrp_multi` (MATLAB ``[hLines, results]``)."""

    h_lines: list
    results: Any


def bootstrap_overlap_pvalues(b1, b2, tail: str = "both") -> np.ndarray:
    """Per-row bootstrap overlap test between two bootstrap matrices.

    The test of calliope ``22_Plotting/plotBSTRP_multi.m``: with ``less = mean(b1 < b2)``
    and ``greater = mean(b1 > b2)`` over the bootstrap iterations of a row (ties count for
    neither), ``p = min(2 * min(less, greater), 1)`` for ``'both'``, ``less`` for
    ``'right'`` (H1: b1 > b2) and ``greater`` for ``'left'`` (H1: b1 < b2).

    Parameters
    ----------
    b1, b2 : array_like, shape (n_time, nbss)
        Bootstrapped means. If the numbers of columns differ, both are cut to the smaller
        one, as MATLAB.
    tail : {'both', 'right', 'left'}

    Returns
    -------
    numpy.ndarray
        float64 ``(n_time,)`` p-values (no multiple-comparison correction).
    """
    a = np.asarray(b1, dtype=np.float64)
    b = np.asarray(b2, dtype=np.float64)
    c = min(a.shape[1], b.shape[1])
    a, b = a[:, :c], b[:, :c]
    less = np.sum(a < b, axis=1) / c
    greater = np.sum(a > b, axis=1) / c
    t = str(tail).lower()
    if t == "both":
        return np.minimum(2 * np.minimum(less, greater), 1.0)
    if t == "right":
        return less
    if t == "left":
        return greater
    raise ValueError("tail must be 'both', 'right' or 'left'")


def _bin_edges(x: np.ndarray) -> np.ndarray:
    if x.size == 1:
        return np.array([x[0], x[0]])
    return np.concatenate([[x[0]], x[:-1] + np.diff(x) / 2, [x[-1]]])


def plot_bstrp_multi(
    data_cell,
    group_cell,
    nbss: int,
    *,
    colors=None,
    pairs="all",
    tail="both",
    alpha: float = 0.05,
    sig_color=(0.0, 0.0, 0.0),
    ns_color=(0.85, 0.85, 0.85),
    show_pair_key: bool = True,
    band_frac: float = 0.02,
    gap_frac: float = 0.01,
    top_pad: float = 0.02,
    ylim=None,
    x_data=None,
    rng: np.random.Generator,
    ax=None,
    **plot_bstrp_kwargs,
) -> PlotBstrpMultiResult:
    """Plot N bootstrapped curves and per-time-point significance strips between pairs.

    Port of calliope ``22_Plotting/plotBSTRP_multi.m``. Every curve is drawn with
    :func:`calliopy.plotting.plot_bstrp` (band in the curve colour lightened 65 % towards
    white). For every pair, :func:`bootstrap_overlap_pvalues` compares the bootstrap
    matrices per time point; above the data one strip per pair shows significant bins in
    ``sig_color`` and the others in ``ns_color`` (pair 0 on top), with two squares in the
    left margin in the colours of the compared curves.

    Parameters
    ----------
    data_cell : sequence of array_like
        N matrices ``(n_time, n_samples_i)`` with the same ``n_time``.
    group_cell : sequence of array_like
        N group-label vectors (one per matrix).
    nbss : int
        Bootstrap iterations, passed to :func:`plot_bstrp`.
    colors : sequence of N colours, optional
        Default MATLAB ``lines(N)``.
    pairs : 'all', None or array_like (M, 2)
        **0-based** positions in ``data_cell`` of the curves to compare. ``'all'`` (default):
        every pair in ``itertools.combinations`` order (MATLAB ``nchoosek``); ``None`` or
        empty: no tests and no strips.
    tail : str or sequence of M str, default 'both'
        ``'both'``, ``'right'`` (H1: first > second) or ``'left'``, for all pairs or per
        pair.
    alpha : float, default 0.05
        Significance threshold.
    sig_color, ns_color : colours
        Strip colours of significant / non-significant bins.
    show_pair_key : bool, default True
        Draw the two colour squares left of each strip.
    band_frac, gap_frac, top_pad : float
        Strip height, gap between strips and extra space on top, as fractions of the
        y range of the data.
    ylim : (ymin, ymax), optional
        Fix the data y range before the strips are added.
    x_data : array_like, optional
        x values (default ``1..n_time``), passed to :func:`plot_bstrp`.
    rng : numpy.random.Generator
        Source of the random numbers (shared by all curves).
    ax : matplotlib.axes.Axes, optional
        Axes to draw into (default: current axes).
    **plot_bstrp_kwargs
        Passed on to :func:`plot_bstrp` (``ptil_range``, ``trns``, ``ms``, ``baseline``,
        ``bstrp_omit_nan``, ``verbose``, ``linewidth`` ...).

    Returns
    -------
    PlotBstrpMultiResult
        ``h_lines`` the N mean lines; ``results`` a :class:`calliopy.io.MatStruct` with
        ``bss_all`` (list of N ``(n_time, nbss)`` arrays), ``pairs`` (``(M, 2)`` int64,
        **0-based**), ``pvals`` and ``sig_mask`` (``(n_time, M)``).

    Differences from MATLAB
    -----------------------
    - ``pairs`` (argument and result) are 0-based.
    - Random draws differ from MATLAB's.
    - The y range on which the strips are stacked is matplotlib's automatic range (MATLAB's
      automatic limits are rounded to "nice" values), so strip positions differ unless
      ``ylim`` is given.
    - Strips are drawn as a ``PolyCollection``; the pair-key squares are unclipped patches.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Rectangle

    from ..io import MatStruct

    data_cell = list(data_cell)
    group_cell = list(group_cell)
    if len(data_cell) != len(group_cell):
        raise ValueError("data_cell and group_cell must have the same number of entries")
    n = len(data_cell)
    if n < 1:
        raise ValueError("Need at least one group")
    n_time = np.shape(data_cell[0])[0]
    if any(np.shape(d)[0] != n_time for d in data_cell):
        raise ValueError("All data matrices must share the same number of timepoints (rows)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    if colors is None:
        cols = [tuple(c) for c in _lines(n)]
    else:
        cols = list(colors)
        if len(cols) != n:
            raise ValueError("colors must have one entry per group")

    if isinstance(pairs, str):
        if pairs.lower() != "all":
            raise ValueError("pairs must be 'all', None or an (M, 2) array")
        pr = np.array(list(combinations(range(n), 2)), dtype=np.int64).reshape(-1, 2)
    elif pairs is None or np.size(pairs) == 0:
        pr = np.zeros((0, 2), dtype=np.int64)
    else:
        pr = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if pr.min() < 0 or pr.max() >= n:
            raise ValueError("pairs must hold 0-based indices into data_cell")
    m = pr.shape[0]

    if isinstance(tail, str):
        tails = [tail.lower()] * m
    else:
        tails = [str(t).lower() for t in tail]
        if len(tails) != m:
            raise ValueError(f"tail (sequence) must have one entry per pair (length {m})")
    for t in tails:
        if t not in ("both", "right", "left"):
            raise ValueError("tail must be 'both', 'right', or 'left'")

    if ax is None:
        ax = plt.gca()
    h_lines, bss_all = [], []
    for i in range(n):
        c = np.asarray(_to_rgb(cols[i]))
        sh = 0.35 * c + 0.65 * np.ones(3)
        res = plot_bstrp(
            data_cell[i],
            group_cell[i],
            nbss,
            x_data=x_data,
            sh_color=tuple(sh),
            rng=rng,
            ax=ax,
            color=cols[i],
            **plot_bstrp_kwargs,
        )
        h_lines.append(res.h)
        bss_all.append(res.bss_all)

    x = (
        np.arange(1, n_time + 1, dtype=np.float64)
        if x_data is None
        else np.asarray(x_data, dtype=np.float64).ravel()
    )
    ax.set_xlim(x[0], x[-1])
    if ylim is not None:
        ax.set_ylim(*ylim)

    pvals = np.full((n_time, m), np.nan)
    sig = np.zeros((n_time, m), dtype=bool)
    for k in range(m):
        p = bootstrap_overlap_pvalues(bss_all[pr[k, 0]], bss_all[pr[k, 1]], tails[k])
        pvals[:, k] = p
        sig[:, k] = p < alpha

    if m > 0:
        y0l, y1l = ax.get_ylim()
        rng_y = y1l - y0l
        if rng_y <= 0:
            rng_y = 1.0
        row_h, gap_h, pad_h = band_frac * rng_y, gap_frac * rng_y, top_pad * rng_y
        total = m * row_h + max(0, m - 1) * gap_h + pad_h
        ax.set_ylim(y0l, y1l + total)
        edges = _bin_edges(x)
        xl = ax.get_xlim()
        span = xl[1] - xl[0]
        for k in range(m):
            from_bottom = m - k
            y0 = y1l + (from_bottom - 1) * (row_h + gap_h) + 0.25 * pad_h
            y1 = y0 + row_h
            verts = [
                [(edges[i], y0), (edges[i + 1], y0), (edges[i + 1], y1), (edges[i], y1)]
                for i in range(n_time)
            ]
            fc = [sig_color if s else ns_color for s in sig[:, k]]
            ax.add_collection(
                PolyCollection(verts, facecolors=fc, edgecolors="none", label="_nolegend_")
            )
            if show_pair_key:
                sq_w, gap, margin = 0.015 * span, 0.004 * span, 0.006 * span
                xb1 = xl[0] - margin
                xb0 = xb1 - sq_w
                xa1 = xb0 - gap
                xa0 = xa1 - sq_w
                for x0, cc in ((xa0, cols[pr[k, 0]]), (xb0, cols[pr[k, 1]])):
                    ax.add_patch(
                        Rectangle(
                            (x0, y0), sq_w, row_h, facecolor=cc, edgecolor="none",
                            clip_on=False, label="_nolegend_",
                        )
                    )
        ax.set_xlim(*xl)

    results = MatStruct(bss_all=bss_all, pairs=pr, pvals=pvals, sig_mask=sig)
    return PlotBstrpMultiResult(h_lines, results)


def _to_rgb(c):
    from matplotlib.colors import to_rgb

    return to_rgb(c)
