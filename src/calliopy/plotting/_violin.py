"""Violin plots with matplotlib: replaces calliope ``22_Plotting/Violin.m`` and ``violinplot.m``.

The MATLAB files are a bundled copy of Bastian Bechtold's violinplot (BSD 3-clause). This
module is an independent matplotlib re-implementation of the violin plot of Hintze & Nelson
(1998, The American Statistician 52(2):181-184) with the same options and children, not a
translation of that code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["Violin", "violin", "violinplot"]

_HALF = ("full", "left", "right")
_QUART = ("boxplot", "shadow", "none")
_DSTYLE = ("scatter", "histogram", "none")
_ORIENT = ("vertical", "horizontal")


def _bandwidth(x: np.ndarray) -> float:
    """Normal-reference bandwidth with a robust spread, as MATLAB ``ksdensity`` uses."""
    n = x.size
    sig = np.median(np.abs(x - np.median(x))) / 0.6745
    if sig <= 0:
        sig = np.std(x, ddof=1) if n > 1 else 0.0
    bw = sig * (4.0 / (3.0 * n)) ** 0.2
    if not np.isfinite(bw) or bw <= 0:
        bw = max(abs(float(x[0])) * 0.01, 1e-3) if n else 1.0
    return float(bw)


def _kde(x: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    """Gaussian kernel density of ``x`` on ``grid``."""
    z = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * z * z).sum(axis=1) / (x.size * bw * np.sqrt(2 * np.pi))


def _is_set_list(data) -> bool:
    """True for a list/tuple of one or two data sets (MATLAB cell), each an array or list."""
    return (
        isinstance(data, (list, tuple))
        and len(data) in (1, 2)
        and all(isinstance(d, (list, tuple, np.ndarray)) for d in data)
    )


def _per_set(v, n: int, kind: str) -> list:
    """Spread a colour/alpha option over ``n`` data sets (a list gives one value per set)."""
    if v is None:
        return [None] * n
    single = np.isscalar(v) if kind == "alpha" else (
        isinstance(v, str) or (isinstance(v, (list, tuple, np.ndarray)) and all(np.isscalar(e) for e in v))
    )
    vals = [v] if single else list(v)
    if not 1 <= len(vals) <= 2:
        raise ValueError(f"violin_{kind} takes at most two values (one per data set)")
    return [vals[min(i, len(vals) - 1)] for i in range(n)]


class Violin:
    """One violin at position ``pos`` (MATLAB ``Violin(data, pos, ...)``).

    Parameters
    ----------
    data : array_like or sequence of two array_like
        Data points (NaNs are dropped). Two data sets give a split violin: the first on the
        left half, the second on the right half.
    pos : float
        Position on the category axis (MATLAB numbering, first violin at 1).
    ax : matplotlib.axes.Axes, optional
        Axes to draw into (MATLAB ``Parent``; ``parent`` is accepted too). Default: current axes.
    rng : numpy.random.Generator, optional
        Source of the jitter of the data points (default: a fresh unseeded generator).
    width : float, default 0.3
        Half width of the widest part of the violin.
    bandwidth : float, optional
        Kernel bandwidth (default: normal reference rule with a robust spread).
    violin_color : colour or sequence of up to two colours, optional
        Default: the next colour of the axes' colour cycle.
    violin_alpha : float or sequence of up to two floats, default (0.3, 0.15)
        Alpha of the violin area and data points of data set 1 and 2.
    marker_size, median_marker_size : float, default 24, 36
        Marker areas (points^2) of the data points and the median.
    line_width : float, default 0.75
    box_color, edge_color : colour, default (0.5, 0.5, 0.5)
    box_width : float, default 0.01
        Half width of the quartile box in data units.
    median_color : colour, default (1, 1, 1)
    show_data, show_box, show_median, show_whiskers : bool, default True
    show_notches, show_mean : bool, default False
    half_violin : {'full', 'left', 'right'}
    quartile_style : {'boxplot', 'shadow', 'none'}
    data_style : {'scatter', 'histogram', 'none'}
    orientation : {'vertical', 'horizontal'}

    Attributes
    ----------
    scatter_plot, scatter_plot2, violin_plot, violin_plot2, box_plot, whisker_plot,
    median_plot, notch_plots, mean_plot, histogram_plot, violin_plot_q, parent, orientation
        The matplotlib artists (``None`` when not drawn), named after the MATLAB children.

    Differences from MATLAB
    -----------------------
    Independent implementation: the bandwidth rule, the density grid, the jitter of the data
    points and the histogram style follow the published method, not the MATLAB code, so the
    shapes are close but not identical. Colour/visibility properties are attributes set at
    construction; changing them afterwards does not redraw (MATLAB's dependent properties do).
    """

    def __init__(
        self,
        data,
        pos: float,
        *,
        ax=None,
        parent=None,
        rng: np.random.Generator | None = None,
        width: float = 0.3,
        bandwidth: float | None = None,
        violin_color: Any = None,
        violin_alpha: Any = (0.3, 0.15),
        marker_size: float = 24,
        median_marker_size: float = 36,
        line_width: float = 0.75,
        box_color: Any = (0.5, 0.5, 0.5),
        box_width: float = 0.01,
        edge_color: Any = (0.5, 0.5, 0.5),
        median_color: Any = (1, 1, 1),
        show_data: bool = True,
        show_notches: bool = False,
        show_mean: bool = False,
        show_box: bool = True,
        show_median: bool = True,
        show_whiskers: bool = True,
        half_violin: str = "full",
        quartile_style: str = "boxplot",
        data_style: str = "scatter",
        orientation: str = "vertical",
    ) -> None:
        import matplotlib.pyplot as plt

        for name, val, ok in (("half_violin", half_violin, _HALF), ("quartile_style", quartile_style, _QUART),
                              ("data_style", data_style, _DSTYLE), ("orientation", orientation, _ORIENT)):
            if val not in ok:
                raise ValueError(f"{name} must be one of {ok}")
        if ax is None:
            ax = parent if parent is not None else plt.gca()
        rng = np.random.default_rng() if rng is None else rng
        sets = self._split_data(data)
        nset = len(sets)
        if nset == 2 and half_violin != "full":
            raise ValueError("half_violin applies to a single data set")
        colors = _per_set(violin_color, nset, "color")
        colors = [c if c is not None else ax._get_lines.get_next_color() for c in colors]
        alphas = [float(a) for a in _per_set(violin_alpha, nset, "alpha")]

        self.parent = ax
        self.orientation = orientation
        self.pos = float(pos)
        self.width = float(width)
        self.violin_color = colors
        self.violin_alpha = alphas
        self.edge_color, self.box_color, self.median_color = edge_color, box_color, median_color
        self.box_width = float(box_width)
        self.show_data, self.show_notches, self.show_mean = show_data, show_notches, show_mean
        self.show_box, self.show_median, self.show_whiskers = show_box, show_median, show_whiskers
        self.half_violin, self.quartile_style, self.data_style = half_violin, quartile_style, data_style
        self.scatter_plot = self.scatter_plot2 = None
        self.violin_plot = self.violin_plot2 = None
        self.box_plot = self.whisker_plot = self.median_plot = self.mean_plot = None
        self.histogram_plot = self.violin_plot_q = None
        self.notch_plots: list = []

        horiz = orientation == "horizontal"

        def xy(c, v):  # category-axis coordinate c, value v -> plot coordinates
            return (v, c) if horiz else (c, v)

        # side of each data set: -1 left only, +1 right only, 0 both
        sides = [-1, 1] if nset == 2 else [{"full": 0, "left": -1, "right": 1}[half_violin]]
        all_vals = np.concatenate(sets) if any(s.size for s in sets) else np.empty(0)
        for k, (x, side) in enumerate(zip(sets, sides)):
            if x.size == 0:
                continue
            bw = _bandwidth(x) if bandwidth is None else float(bandwidth)
            grid = np.linspace(x.min(), x.max(), 100) if x.max() > x.min() else np.array([x.min()])
            dens = _kde(x, grid, bw)
            scale = self.width / dens.max() if dens.max() > 0 else 0.0
            half = dens * scale
            left = self.pos - half * (side <= 0)
            right = self.pos + half * (side >= 0)
            c_outline = np.concatenate([right, left[::-1]])
            v_outline = np.concatenate([grid, grid[::-1]])
            px, py = xy(c_outline, v_outline)
            (patch,) = ax.fill(px, py, facecolor=colors[k], alpha=alphas[k], edgecolor=edge_color,
                               linewidth=line_width, label="_nolegend_")
            if k == 0:
                self.violin_plot = patch
            else:
                self.violin_plot2 = patch

            if show_data and data_style == "scatter":
                d_at = np.interp(x, grid, half) if grid.size > 1 else np.full(x.shape, half[0])
                u = rng.random(x.size)
                if side == 0:
                    jit = (2 * u - 1) * d_at
                else:
                    jit = side * u * d_at
                sx, sy = xy(self.pos + jit, x)
                sc = ax.scatter(sx, sy, s=marker_size, color=colors[k], alpha=min(1.0, alphas[k] + 0.4),
                                linewidths=0, label="_nolegend_")
                if k == 0:
                    self.scatter_plot = sc
                else:
                    self.scatter_plot2 = sc
            elif show_data and data_style == "histogram":
                counts, edges = np.histogram(x, bins="auto")
                if counts.max() > 0:
                    hw = counts / counts.max() * self.width
                    for c, lo, hi in zip(hw, edges[:-1], edges[1:]):
                        a = self.pos - c * (side <= 0)
                        b = self.pos + c * (side >= 0)
                        hx, hy = xy(np.array([a, b, b, a]), np.array([lo, lo, hi, hi]))
                        (bar,) = ax.fill(hx, hy, facecolor=colors[k], alpha=alphas[k], edgecolor="none",
                                         label="_nolegend_")
                        self.histogram_plot = bar

        if all_vals.size == 0:
            return
        q1, med, q3 = np.percentile(all_vals, [25, 50, 75])
        iqr = q3 - q1
        lo_w = all_vals[all_vals >= q1 - 1.5 * iqr].min()
        hi_w = all_vals[all_vals <= q3 + 1.5 * iqr].max()
        bw_box = self.box_width
        if quartile_style == "boxplot" and show_box:
            bx, by = xy(self.pos + np.array([-bw_box, bw_box, bw_box, -bw_box]), np.array([q1, q1, q3, q3]))
            (self.box_plot,) = ax.fill(bx, by, facecolor=box_color, edgecolor=box_color,
                                       linewidth=line_width, label="_nolegend_")
        elif quartile_style == "shadow" and self.violin_plot is not None:
            x0 = sets[0]
            bw = _bandwidth(x0) if bandwidth is None else float(bandwidth)
            gq = np.linspace(q1, q3, 50)
            dg = np.linspace(x0.min(), x0.max(), 100) if x0.max() > x0.min() else np.array([x0.min()])
            dmax = _kde(x0, dg, bw).max()
            hq = _kde(x0, gq, bw) * (self.width / dmax if dmax > 0 else 0.0)
            px, py = xy(np.concatenate([self.pos + hq, (self.pos - hq)[::-1]]), np.concatenate([gq, gq[::-1]]))
            (self.violin_plot_q,) = ax.fill(px, py, facecolor=colors[0], alpha=min(1.0, alphas[0] + 0.3),
                                            edgecolor="none", label="_nolegend_")
        if show_whiskers and quartile_style == "boxplot":
            wx, wy = xy(np.array([self.pos, self.pos]), np.array([lo_w, hi_w]))
            (self.whisker_plot,) = ax.plot(wx, wy, color=box_color, linewidth=line_width, label="_nolegend_")
        if show_median:
            mx, my = xy(self.pos, med)
            self.median_plot = ax.scatter([mx], [my], s=median_marker_size, facecolor=median_color,
                                          edgecolor=box_color, linewidths=line_width, zorder=3,
                                          label="_nolegend_")
        if show_notches:
            nd = 1.58 * iqr / np.sqrt(all_vals.size)
            for v in (med - nd, med + nd):
                nx, ny = xy(self.pos, v)
                self.notch_plots.append(ax.scatter([nx], [ny], s=median_marker_size / 2,
                                                   marker="_" if not horiz else "|", color=box_color,
                                                   zorder=3, label="_nolegend_"))
        if show_mean:
            mean = float(np.mean(all_vals))
            mxs, mys = xy(self.pos + np.array([-self.width, self.width]) / 2, np.array([mean, mean]))
            (self.mean_plot,) = ax.plot(mxs, mys, color="k", linewidth=line_width, label="_nolegend_")

    @staticmethod
    def _split_data(data) -> list[np.ndarray]:
        if _is_set_list(data):
            sets = [np.asarray(d, dtype=np.float64).ravel() for d in data]
        else:
            sets = [np.asarray(data, dtype=np.float64).ravel()]
        return [s[~np.isnan(s)] for s in sets]


def violin(data, pos: float, *, ax=None, rng: np.random.Generator | None = None, **options) -> Violin:
    """Draw one violin (MATLAB ``Violin(data, pos, Name, Value)``); see :class:`Violin`."""
    return Violin(data, pos, ax=ax, rng=rng, **options)


def violinplot(data, cats=None, *, ax=None, rng: np.random.Generator | None = None,
               group_order: Sequence[str] | None = None, **options) -> list[Violin]:
    """Violin plots of a vector, the columns of a matrix, categories or the fields of a mapping.

    Replaces calliope ``22_Plotting/violinplot.m`` (third-party, re-implemented).

    Parameters
    ----------
    data : array_like, mapping, or sequence of two arrays
        - vector: one violin at 1;
        - matrix (n, k): one violin per column at 1..k;
        - vector with ``cats`` of the same length: one violin per category;
        - mapping (dict, ``MatStruct``): one violin per numeric field, labelled by its name;
        - list or tuple of two vectors/matrices (MATLAB cell) with the same number of
          columns: split violins (first data set left, second right). A matrix must be a
          numpy array, since a list of two lists is read as two data sets.
    cats : sequence, optional
        Category of every point (vector data) or category names of the columns (matrix data).
    group_order : sequence of str, optional
        Order of the categories (MATLAB ``'GroupOrder'``); default sorted.
    ax, rng, **options
        As :class:`Violin`.

    Returns
    -------
    list of Violin

    Differences from MATLAB
    -----------------------
    MATLAB tables and datasets are taken as mappings. Category labels are set as tick labels
    of the category axis; MATLAB also switches the box on, which is matplotlib's default.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = options.pop("parent", None) or plt.gca()
    horiz = options.get("orientation", "vertical") == "horizontal"
    vio: list[Violin] = []
    labels: list[str] | None = None

    if isinstance(data, Mapping) or hasattr(data, "_fieldnames"):
        keys = list(data.keys()) if isinstance(data, Mapping) else list(data._fieldnames)
        getter = (lambda k: data[k]) if isinstance(data, Mapping) else (lambda k: getattr(data, k))
        names = [k for k in (group_order or sorted(keys)) if np.asarray(getter(k)).dtype.kind in "biuf"]
        for n, k in enumerate(names, start=1):
            vio.append(Violin(np.asarray(getter(k), dtype=np.float64), n, ax=ax, rng=rng, **options))
        labels = [str(k) for k in names]
    else:
        two = _is_set_list(data) and len(data) == 2
        if _is_set_list(data):
            sets = [np.asarray(d, dtype=np.float64) for d in data]
        else:
            sets = [np.asarray(data, dtype=np.float64)]
        # a vector (1-D, 1 x n or n x 1) is one column, as MATLAB treats row and column vectors
        sets = [s.reshape(-1, 1) if s.ndim <= 1 or (s.ndim == 2 and 1 in s.shape) else s for s in sets]
        if two and sets[0].shape[1] != sets[1].shape[1]:
            raise ValueError("the two data sets must have the same number of columns")
        if cats is not None and sets[0].shape[1] == 1 and len(cats) == sets[0].shape[0]:
            if two:
                raise ValueError("categories with two data sets are not supported")
            cat = np.asarray(cats).ravel()
            names = list(group_order) if group_order else sorted({str(c) for c in cat})
            cs = np.array([str(c) for c in cat])
            for n, c in enumerate(names, start=1):
                vio.append(Violin(sets[0][cs == c, 0], n, ax=ax, rng=rng, **options))
            labels = names
        else:
            for n in range(sets[0].shape[1]):
                this = [s[:, n] for s in sets] if two else sets[0][:, n]
                vio.append(Violin(this, n + 1, ax=ax, rng=rng, **options))
            if cats is not None:
                labels = [str(c) for c in cats]
    ticks = list(range(1, len(vio) + 1))
    if horiz:
        ax.set_yticks(ticks)
        if labels:
            ax.set_yticklabels(labels)
    else:
        ax.set_xticks(ticks)
        if labels:
            ax.set_xticklabels(labels)
    return vio
