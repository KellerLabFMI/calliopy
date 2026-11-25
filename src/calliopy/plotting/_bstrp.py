"""Bootstrapped mean trace with a shaded band: calliope ``22_Plotting/plotBSTRP.m``."""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

import numpy as np

from .._matlab import prctile
from ..stats import find_groups, nbstrp_get_indices
from ._linespec import linespec_kwargs

__all__ = ["PlotBstrpResult", "plot_bstrp"]

log = logging.getLogger(__name__)


class PlotBstrpResult(NamedTuple):
    """Outputs of :func:`plot_bstrp` (MATLAB ``[h, bss_all]``)."""

    h: Any
    bss_all: np.ndarray


def plot_bstrp(
    data,
    group_index,
    nbss: int | None = 100,
    line_spec: str | None = "-",
    *,
    x_data=None,
    sh_color=(0.9, 0.9, 0.9),
    trns: float = 1,
    ptil_range="sd",
    baseline=None,
    ms: bool = False,
    bstrp_omit_nan: bool = False,
    verbose: bool = False,
    rng: np.random.Generator,
    ax=None,
    **plot_kwargs,
) -> PlotBstrpResult:
    """Plot a hierarchically bootstrapped mean with an SD or percentile band.

    Port of calliope ``22_Plotting/plotBSTRP.m``. Whole traces (columns of ``data``) are
    resampled with :func:`calliopy.stats.nbstrp_get_indices` (groups, then traces within
    the group); the line is the mean of the ``nbss`` bootstrapped mean traces and the band is
    their mean +/- 1 SD (``ptil_range='sd'``) or the given percentiles.

    Parameters
    ----------
    data : array_like, shape (n_time, n_samples)
        One trace per column.
    group_index : array_like, length n_samples
        Group label of every trace (any labels; normalised as MATLAB ``findgroups``).
    nbss : int, default 100
        Number of bootstrap iterations (``None`` gives 100, as MATLAB's empty).
    line_spec : str, default '-'
        MATLAB LineSpec of the mean line (``'-k'``, ``'--r'``...). A ``color`` keyword
        overrides the LineSpec colour, as a ``'Color'`` name-value pair does in MATLAB.
    x_data : array_like, optional
        x values, length n_time (default ``1..n_time`` as MATLAB).
    sh_color : colour, default (0.9, 0.9, 0.9)
        Band colour.
    trns : float, default 1
        Band alpha.
    ptil_range : 'sd' or sequence of two percentiles
        Band definition (``[2.5, 97.5]`` for a 95 % interval).
    baseline : array_like of int, optional
        **0-based** row indices of the baseline period used when ``ms`` is true
        (default ``range(92, 100)``, MATLAB's ``93:100``).
    ms : bool, default False
        Subtract the mean of ``bss_all[baseline, :]`` (one number over all iterations).
    bstrp_omit_nan : bool, default False
        Drop every trace that has a NaN before resampling.
    verbose : bool, default False
        Log progress every 10 % (MATLAB prints it).
    rng : numpy.random.Generator
        Source of the random numbers.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into (default: current axes).
    **plot_kwargs
        Passed to ``ax.plot`` for the mean line (MATLAB's unmatched name-value pairs,
        ``Color`` -> ``color``, ``LineWidth`` -> ``linewidth``).

    Returns
    -------
    PlotBstrpResult
        ``h`` the mean ``Line2D``; ``bss_all`` float64 array (n_time, nbss) of the
        bootstrapped mean traces (baseline-subtracted when ``ms``).

    Differences from MATLAB
    -----------------------
    - ``baseline`` is 0-based.
    - Random draws differ from MATLAB's.
    - The band is drawn with ``fill`` below the line (``zorder``) and labelled
      ``'_nolegend_'``; MATLAB moves the patch into an hggroup hidden from the legend and
      flips the order of all axes children.
    - ``shg`` (bring the figure forward) is not done.
    """
    import matplotlib.pyplot as plt

    d = np.asarray(data, dtype=np.float64)
    if d.ndim == 1:
        d = d[:, None]
    if d.ndim != 2:
        raise ValueError("data must be a 2D matrix")
    g = np.asarray(group_index).ravel()
    if d.shape[1] != g.size:
        raise ValueError("group_index must match number of samples")
    if nbss is None:
        nbss = 100
    nbss = int(nbss)
    n_time = d.shape[0]

    if bstrp_omit_nan:
        valid = ~np.isnan(d).any(axis=0)
        d, g = d[:, valid], g[valid]

    idx = nbstrp_get_indices(find_groups(g), nbss, rng=rng)
    bss_all = np.empty((n_time, nbss), dtype=np.float64)
    step = max(1, nbss // 10)
    for ib, ii in enumerate(idx):
        bss_all[:, ib] = d[:, ii].mean(axis=1)
        if verbose and ((ib + 1) % step == 0 or ib + 1 == nbss):
            log.info("plot_bstrp is %d percent done", int(np.floor((ib + 1) / nbss * 100 + 0.5)))

    if ms:
        bl = np.arange(92, 100) if baseline is None else np.asarray(baseline, dtype=np.int64)
        bss_all = bss_all - np.nanmean(bss_all[bl, :])

    mean_data = bss_all.mean(axis=1)
    if isinstance(ptil_range, str):
        if ptil_range.lower() != "sd":
            raise ValueError("ptil_range must be 'sd' or two percentiles")
        sd = bss_all.std(axis=1, ddof=1) if nbss > 1 else np.zeros(n_time)
        lo, hi = mean_data - sd, mean_data + sd
    else:
        p = np.asarray(ptil_range, dtype=np.float64).ravel()
        if p.size < 2:
            raise ValueError("ptil_range needs two percentiles")
        ci = np.asarray(prctile(bss_all, p, axis=1)).reshape(n_time, p.size)
        lo, hi = ci[:, 0], ci[:, 1]  # MATLAB uses the first two percentiles

    if x_data is None:
        x = np.arange(1, n_time + 1, dtype=np.float64)
    else:
        x = np.asarray(x_data, dtype=np.float64).ravel()
        if x.size != n_time:
            raise ValueError("x_data must have same length as data time dimension")

    if ax is None:
        ax = plt.gca()
    ax.fill(
        np.concatenate([x, x[::-1]]),
        np.concatenate([hi, lo[::-1]]),
        color=sh_color,
        alpha=trns,
        linewidth=0,
        label="_nolegend_",
        zorder=1,
    )
    kw = linespec_kwargs(line_spec)
    kw.update({k.lower(): v for k, v in plot_kwargs.items()})
    kw.setdefault("zorder", 2)
    (h,) = ax.plot(x, mean_data, **kw)
    return PlotBstrpResult(h, bss_all)
