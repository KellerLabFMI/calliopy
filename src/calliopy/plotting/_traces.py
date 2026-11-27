"""Mean +/- SEM traces and stacked normalised traces: calliope ``22_Plotting/plotSEM.m`` and
``22_Plotting/specTraces.m``."""

from __future__ import annotations

import warnings
from typing import Any, NamedTuple

import numpy as np

from ..stats import nan_sem
from ._linespec import linespec_kwargs

__all__ = ["SpecTracesResult", "plot_sem", "spec_traces", "spec_traces_data"]


def plot_sem(
    data,
    line_spec: str | None = "-",
    *,
    sh_color=(0.9, 0.9, 0.9),
    trns: float = 1,
    ax=None,
    **plot_kwargs,
):
    """Plot the mean of trials with a shaded +/- SEM band (NaN ignored).

    Port of calliope ``22_Plotting/plotSEM.m``. The line is ``nanmean(data, axis=1)``, the
    band ``mean +/- nan_sem(data, axis=1)`` (:func:`calliopy.stats.nan_sem`), at
    x = ``1..n_time`` as in MATLAB.

    Parameters
    ----------
    data : array_like, shape (n_time, n_trials)
        Rows are measurements (time points), columns trials.
    line_spec : str, default '-'
        MATLAB LineSpec of the mean line.
    sh_color : colour, default (0.9, 0.9, 0.9)
        Band colour.
    trns : float, default 1
        Band alpha.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into (default: current axes).
    **plot_kwargs
        Passed to ``ax.plot`` (MATLAB's other name-value pairs, lower-cased).

    Returns
    -------
    matplotlib.lines.Line2D
        The mean line (MATLAB ``h``).

    Differences from MATLAB
    -----------------------
    - ``alpha(trns)`` in MATLAB sets the transparency of every patch in the axes; here only
      the new band gets ``trns``.
    - The band is kept behind the line with ``zorder`` and hidden from the legend; MATLAB
      regroups all patches of the axes in an hggroup and reverses the order of all
      children. ``shg`` is not done.
    """
    import matplotlib.pyplot as plt

    d = np.asarray(data, dtype=np.float64)
    if d.ndim == 1:
        d = d[:, None]
    if d.ndim != 2:
        raise ValueError("data must be 2-D (n_time x n_trials)")
    n = d.shape[0]
    sem = np.asarray(nan_sem(d, axis=1)).ravel() if d.shape[1] else np.full(n, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(d, axis=1)
    x = np.arange(1, n + 1, dtype=np.float64)
    if ax is None:
        ax = plt.gca()
    ax.fill(
        np.concatenate([x, x[::-1]]),
        np.concatenate([mean + sem, (mean - sem)[::-1]]),
        color=sh_color,
        alpha=trns,
        linewidth=0,
        label="_nolegend_",
        zorder=1,
    )
    kw = linespec_kwargs(line_spec)
    kw.update({k.lower(): v for k, v in plot_kwargs.items()})
    kw.setdefault("zorder", 2)
    (h,) = ax.plot(x, mean, **kw)
    return h


class SpecTracesResult(NamedTuple):
    """Outputs of :func:`spec_traces` (MATLAB ``[h, data]``)."""

    h: Any
    data: np.ndarray


def spec_traces_data(data, *, autotranspose: bool = True) -> np.ndarray:
    """Normalise traces to 0..1 and stack them with a spacing of 1.1.

    The computation of :func:`spec_traces` without the figure. Column ``j`` (0-based) of
    the result is ``(x_j - min(x_j)) / (max(x_j) - min(x_j)) + 1.1 * (n_traces - 1 - j)``
    (min/max ignore NaN, as MATLAB): the first trace is drawn on top.

    Parameters
    ----------
    data : array_like
        2-D array, one trace per column (or per row, see ``autotranspose``).
    autotranspose : bool, default True
        Transpose when there are fewer rows than columns, so traces are along the longer
        dimension.

    Returns
    -------
    numpy.ndarray
        float64 ``(n_samples, n_traces)``. A constant trace gives NaN (0/0), as MATLAB.
    """
    d = np.asarray(data, dtype=np.float64)
    if d.ndim == 1:
        d = d[None, :]
    if d.ndim != 2:
        raise ValueError("data must be 2-D")
    if autotranspose and d.shape[0] < d.shape[1]:
        d = d.T
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        mn = np.nanmin(d, axis=0)
        mx = np.nanmax(d, axis=0)
        out = (d - mn) / (mx - mn)
    n = d.shape[1]
    return out + np.arange(n - 1, -1, -1, dtype=np.float64) * 1.1


def spec_traces(
    data,
    *,
    plot: bool = True,
    autotranspose: bool = True,
    downsample: bool = False,
    legend_names=None,
    title: str | None = None,
) -> SpecTracesResult:
    """Plot normalised traces stacked on top of each other.

    Port of calliope ``22_Plotting/specTraces.m`` (ML 2013). The data are normalised and
    stacked with :func:`spec_traces_data` and drawn in a new figure, sample number on the x
    axis (``1..n``), trace numbers ``n..1`` from bottom to top on the y axis.

    Parameters
    ----------
    data : array_like
        One trace per column (see ``autotranspose``).
    plot : bool, default True
        Draw the figure.
    autotranspose : bool, default True
        As in :func:`spec_traces_data`.
    downsample : bool, default False
        Draw every 10th sample for 50 000 < n <= 500 000 samples and every 100th above;
        tick labels are then scaled by 0.01 / 0.1 (seconds at 1 kHz), as MATLAB.
    legend_names : sequence of str, optional
        Legend entries (placed outside to the upper right).
    title : str, optional
        Figure title.

    Returns
    -------
    SpecTracesResult
        ``h`` the new ``matplotlib.figure.Figure`` (``None`` when ``plot`` is false) and
        ``data`` the normalised, stacked traces.

    Notes
    -----
    The x label reads "Time [s]" also when the axis shows samples (MATLAB does the same).
    """
    out = spec_traces_data(data, autotranspose=autotranspose)
    if not plot:
        return SpecTracesResult(None, out)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    n_samp, n_tr = out.shape
    fig, ax = plt.subplots()
    step, scale = 1, None
    if downsample and 50000 < n_samp <= 500000:
        step, scale = 10, 0.01
    elif downsample and n_samp > 500000:
        step, scale = 100, 0.1
    shown = out[::step]
    ax.plot(np.arange(1, shown.shape[0] + 1), shown)
    ax.set_xlim(1, n_samp / step)
    if scale is not None:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p, s=scale: f"{v * s:g}"))
    if legend_names:
        ax.legend(list(legend_names), loc="upper left", bbox_to_anchor=(1.02, 1))
    if title:
        ax.set_title(title)
    ax.set_ylim(0, n_tr * 1.1)
    ax.set_yticks(0.5 + 1.1 * np.arange(n_tr))
    ax.set_yticklabels([str(k) for k in range(n_tr, 0, -1)])
    ax.set_xlabel("Time [s]")
    return SpecTracesResult(fig, out)
