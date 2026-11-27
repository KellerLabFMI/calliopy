"""Figures of the cell browser plugins: calliope ``20_MetaData/cb_plugin_ACX.m``,
``cb_plugin_M1.m``, ``cb_plugin_VML.m``, ``cb_plugin_trav.m`` and the stimulus snippets of
the ``s`` key of ``cell_browser.m``.

Every function takes one ``rd`` entry of a proj_meta (``proj_meta(site).rd(layer, tp)``,
a mapping), the **0-based** cell (row of ``act``) and, for M1 and VML, the field names of
the five channel rows of the browser (row 0 is MATLAB ``chansToPlot{1}``, the bottom row).
Samples are plotted at MATLAB's x positions (1, 2, ..., as ``plot(y)``), so the guide lines
of the M1 figures sit where MATLAB draws them.

The onset detection is :func:`calliopy.projmeta.activity_onsets` (ported earlier); the
position/speed binning of the trav plugin is :func:`calliopy.projmeta.traversal_activity`.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import scipy.signal
from matplotlib.figure import Figure

from calliopy._matlab import mround, prctile
from calliopy.projmeta import activity_onsets, traversal_activity

__all__ = [
    "plot_cb_plugin_acx",
    "plot_cb_plugin_m1",
    "plot_cb_plugin_trav",
    "plot_cb_plugin_vml",
    "plot_cb_stim_snippets",
]

log = logging.getLogger("calliopy.viewers")

_GREY = (0.4, 0.4, 0.4)


def _ntzo(x) -> np.ndarray:
    from calliopy.projmeta._struct import ntzo

    return ntzo(x)


def _act(rd: Mapping, cell: int) -> np.ndarray:
    act = np.asarray(rd["act"], dtype=np.float64)
    if act.ndim != 2 or not 0 <= int(cell) < act.shape[0]:
        raise IndexError(f"cell {cell} is not a row of act (shape {act.shape})")
    return act[int(cell)]


def _row(rd: Mapping, name: str, row: int = 0) -> np.ndarray | None:
    """``rd.(name)(row+1, :)`` as float64, or ``None`` when the field is missing or empty."""
    if name not in rd:
        return None
    a = np.asarray(rd[name], dtype=np.float64)
    if a.size == 0:
        return None
    a = np.atleast_2d(a)
    return a[row] if row < a.shape[0] else None


def _linear(rd: Mapping, name: str) -> np.ndarray | None:
    """``rd.(name)`` read with MATLAB linear indexing (column-major), or ``None``."""
    if name not in rd:
        return None
    a = np.asarray(rd[name])
    if a.size == 0 or a.dtype == object:
        return None
    return np.ravel(a, order="F").astype(np.float64)


def _channel_trace(rd: Mapping, field: str, cell: int) -> np.ndarray | None:
    """The trace a plugin cuts snippets from: ``ntzo(act(cell,:))`` or the raw field."""
    if field == "act":
        return _ntzo(_act(rd, cell))
    return _linear(rd, field)


def _snips(trace: np.ndarray, onsets: np.ndarray, before: int, after: int) -> np.ndarray:
    """``[before+after+1 x nOnsets]`` windows ``trace(o-before : o+after)`` (0-based ``o``)."""
    cols = [trace[o - before: o + after + 1] for o in onsets]
    n = before + after + 1
    return np.stack(cols, axis=1) if cols else np.zeros((n, 0))


def _x(n: int) -> np.ndarray:
    return np.arange(1, n + 1)


def _tight(ax) -> None:
    """MATLAB ``axis tight`` on the data drawn so far."""
    ax.relim()
    ax.autoscale(enable=True, tight=True)
    ax.autoscale_view(tight=True)
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())


def _guides(ax, win: int, lw_centre: float = 1.0, zero_line: bool = False) -> None:
    """The dash-dot lines of cb_plugin_M1 at ``win`` (black) and ``win +- 50`` (green)."""
    lo, hi = ax.get_ylim()
    ax.plot([win, win], [hi, lo], "-.k", linewidth=lw_centre)
    ax.plot([win + 50, win + 50], [hi, lo], "-.g", linewidth=1)
    ax.plot([win - 50, win - 50], [hi, lo], "-.g", linewidth=1)
    if zero_line:
        ax.plot([0, 2 * win], [0, 0], "-.k")


def _figure(label: str, size=(6.4, 4.8)) -> Figure:
    fig = Figure(figsize=size)
    fig.set_label(label)
    return fig


# ---------------------------------------------------------------------------------------
# 's' key of cell_browser.m
# ---------------------------------------------------------------------------------------
def plot_cb_stim_snippets(rd: Mapping, cell: int) -> Figure:
    """Activity around stimulus onsets, one panel per stimulus (``s`` key of ``cell_browser.m``).

    ``stim_id`` is rounded to one decimal (MATLAB ``round(10*sid)/10``); every value other
    than 0 is a condition, and an onset is the last sample before ``stim_id`` becomes that
    value. Each panel overlays ``act`` from 20 samples before to 30 samples after the
    onsets of one condition (``axis tight``, ``ylim([0.9 max(act)])``).

    Parameters
    ----------
    rd : mapping
        One ``rd`` entry with ``act`` and ``stim_id``.
    cell : int
        **0-based** row of ``act``.

    Returns
    -------
    matplotlib.figure.Figure
        MATLAB figure 10.

    Differences from MATLAB
    -----------------------
    Onsets whose window would start before the first or end after the last sample are
    skipped (MATLAB fails on the index).
    """
    act = _act(rd, cell)
    sid = _linear(rd, "stim_id")
    if sid is None:
        raise KeyError("rd has no stim_id")
    sid = mround(10 * sid) / 10
    conds = np.setdiff1d(np.unique(sid), [0.0])
    fig = _figure("cell_browser stimulus snippets (10)", (3 * max(len(conds), 1), 3))
    win_l, win_r = 20, 30
    for j, c in enumerate(conds):
        ax = fig.add_subplot(1, len(conds), j + 1)
        ons = np.flatnonzero(np.diff((sid == c).astype(np.int8)) == 1)
        for o in ons:
            if o - win_l < 0 or o + win_r >= act.size:
                continue
            ax.plot(_x(win_l + win_r + 1), act[o - win_l: o + win_r + 1])
        _tight(ax)
        ax.set_ylim(0.9, float(np.max(act)))
    return fig


# ---------------------------------------------------------------------------------------
# cb_plugin_ACX.m
# ---------------------------------------------------------------------------------------
_ACX_PANELS = (("act", None), ("velM_smoothed", "r"), ("airPuff", "k"), ("el_stim", "y"),
               ("aud_trig", "g"))


def plot_cb_plugin_acx(rd: Mapping, cell: int) -> Figure:
    """Activity onsets of one cell with the ACX behaviour channels (``cb_plugin_ACX.m``).

    Onsets: ``act > 1 + 2.5*std(act)`` (``ddof=1``), window 100 samples
    (:func:`calliopy.projmeta.activity_onsets`). Five stacked panels overlay, for every
    onset, ``act``, ``velM_smoothed`` (red), ``airPuff`` (black), ``el_stim`` (yellow) and
    ``aud_trig`` (green) from 100 samples before to 100 after the onset.

    Parameters
    ----------
    rd : mapping
        One ``rd`` entry.
    cell : int
        **0-based** row of ``act``.

    Returns
    -------
    matplotlib.figure.Figure
        MATLAB figure 90.

    Differences from MATLAB
    -----------------------
    A missing channel leaves its panel empty with a title saying so (MATLAB fails). The
    screen position of MATLAB's figure is not reproduced.
    """
    act = _act(rd, cell)
    win = 100
    onsets = activity_onsets(act, 1 + 2.5 * float(np.std(act, ddof=1)), win)
    fig = _figure("cb_plugin_ACX (90)", (6.66, 9.76))
    for k, (name, color) in enumerate(_ACX_PANELS):
        ax = fig.add_subplot(len(_ACX_PANELS), 1, k + 1)
        tr = act if name == "act" else _linear(rd, name)
        if tr is None:
            ax.set_title(f"{name} (missing)")
            continue
        style = {"color": color} if color else {}
        for o in onsets:
            seg = tr[o - win: o + win + 1]
            ax.plot(_x(seg.size), seg, **style)
        if onsets.size:
            _tight(ax)
    return fig


# ---------------------------------------------------------------------------------------
# cb_plugin_M1.m
# ---------------------------------------------------------------------------------------
def plot_cb_plugin_m1(rd: Mapping, cell: int, channel_fields: Sequence[Sequence[str]]) -> list[Figure]:
    """Movement parameters around activity onsets (``cb_plugin_M1.m``, function ``cb_plugin_turning``).

    Threshold ``1 + 2.5*std(act)`` (``ddof=1``) on the 20-sample running mean
    (:func:`calliopy.analysis.smooth2`) of the cell's ``act``; window 150 samples.

    - figure 12: ``velM_smoothed`` against ``velR_smoothed`` (limits +-0.1) with the 20
      samples before (red) and after (green) each onset;
    - figure 13: mean activity of all cells around the onsets, rows sorted by their maximum;
    - figure 11: one panel per channel row (row 0 at the bottom): the snippets of every
      selected field (grey), their mean (red) and dash-dot lines at the onset and +-50;
      for ``pupil_pos`` (except in row 3) the mean of its first row minus 36;
    - figure 14 (only when row 4 has a field): mean ``diff`` and mean of ``velM`` (sign
      as stored) and ``velR``, population mean, cell mean with the number of onsets;
    - figure 15: ``act`` against ``VRx`` for the samples above threshold, onsets in red.

    Parameters
    ----------
    rd : mapping
        One ``rd`` entry (``act``, ``velM_smoothed``, ``velR_smoothed``; ``pupil_pos``,
        ``VRx``, ``VRrew`` when present).
    cell : int
        **0-based** row of ``act``.
    channel_fields : sequence of 5 sequences of str
        Field names selected in the browser's channel rows (row 0 = MATLAB
        ``chansToPlot{1}``).

    Returns
    -------
    list of matplotlib.figure.Figure
        Figures 11, 12, 13, (14), 15 (labels name the MATLAB figure numbers). When the
        smoothed trace never crosses the threshold, figures 11, 12, 13 and 15 are returned
        with only the velocity plot drawn and the MATLAB message is logged.

    Notes
    -----
    Kept from MATLAB: a field with more than one row is cut with linear (column-major)
    indexing, as MATLAB's ``tmp_trace(a:b)`` does, so its snippets interleave the rows.

    Differences from MATLAB
    -----------------------
    - The ``m`` key of ``cell_browser.m`` calls ``cb_plugin_turning``, the name of the
      function inside ``cb_plugin_M1.m``; MATLAB resolves functions by file name, so the key
      fails there. Here the key runs this function.
    - The onset matrices are computed even when channel row 0 is empty (MATLAB then fails
      at figure 13); missing optional fields (``pupil_pos``, ``VRx``, ``VRrew``) skip only
      what needs them; no onset left after the window checks gives the figures without
      snippets (MATLAB fails).
    - Figure positions on the screen are not reproduced.
    """
    act_all = np.asarray(rd["act"], dtype=np.float64)
    a = _act(rd, cell)
    vel_m = _row(rd, "velM_smoothed")
    vel_r = _row(rd, "velR_smoothed")
    f11 = _figure("cb_plugin_M1 (11)", (5.85, 7.21))
    f12 = _figure("cb_plugin_M1 (12)", (4.01, 2.58))
    f13 = _figure("cb_plugin_M1 (13)", (4.01, 2.6))
    f15 = _figure("cb_plugin_M1 (15)")
    ax12 = f12.add_subplot(1, 1, 1)
    if vel_m is not None and vel_r is not None:
        ax12.plot(vel_m, vel_r)
    ax12.set_xlim(-0.1, 0.1)
    ax12.set_ylim(-0.1, 0.1)

    from calliopy.analysis import smooth2

    thresh = 1 + float(np.std(a, ddof=1)) * 2.5
    smoothed = smooth2(a, 20)
    tact = smoothed > thresh
    win = 150
    if not np.any(np.diff(tact.astype(np.int8)) == 1):
        log.info("No act onsets with threshold criteria detected. Aborting ...")
        return [f11, f12, f13, f15]
    onsets = activity_onsets(smoothed, thresh, win)
    n_on = onsets.size

    act_mat = (np.stack([act_all[:, o - win: o + win + 1] for o in onsets], axis=2)
               if n_on else np.zeros((act_all.shape[0], 2 * win + 1, 0)))
    vel_m_mat = -_snips(vel_m, onsets, win, win) if vel_m is not None else None
    vel_r_mat = _snips(vel_r, onsets, win, win) if vel_r is not None else None
    pupil = [_row(rd, "pupil_pos", 0), _row(rd, "pupil_pos", 1)]
    ipos1 = _snips(pupil[0], onsets, win, win) if pupil[0] is not None else None

    any_channel = any(len(r) for r in channel_fields)
    if channel_fields and len(channel_fields[0]) and vel_m is not None and vel_r is not None:
        for o in onsets:
            ax12.plot(vel_m[o - 20: o + 1], vel_r[o - 20: o + 1], "r", linewidth=2)
            ax12.plot(vel_m[o: o + 21], vel_r[o: o + 21], "g", linewidth=2)
    if any_channel:
        ax12.set_title("velM vs velR, 20 before onset red, 20 after onset green")
        ax13 = f13.add_subplot(1, 1, 1)
        if n_on:
            tmp = act_mat.mean(axis=2)
            order = np.argsort(tmp.max(axis=1), kind="stable")
            ax13.imshow(tmp[order], aspect="auto", interpolation="nearest")
        ax13.set_title("raster aligned to max any cell act")

    figs = [f11, f12, f13]
    f14 = None
    for ind, fields in enumerate(channel_fields[:5]):
        if not len(fields):
            continue
        ax = f11.add_subplot(5, 1, 5 - ind)
        for field in fields:
            tr = _channel_trace(rd, field, cell)
            snp = _snips(tr, onsets, win, win) if tr is not None else np.zeros((2 * win + 1, 0))
            if ind == 3 or field != "pupil_pos":
                if snp.shape[1]:
                    ax.plot(_x(snp.shape[0]), snp, color=_GREY)
                    _tight(ax)
                    ax.plot(_x(snp.shape[0]), snp.mean(axis=1), "r")
                    _guides(ax, win)
                ax.set_title(field)
            elif ipos1 is not None and ipos1.shape[1]:
                ax.plot(_x(ipos1.shape[0]), ipos1.mean(axis=1) - 36)
            if ind == 4:
                f14 = _figure("cb_plugin_M1 (14)", (9.87, 4.65))
                _m1_figure14(f14, vel_m_mat, vel_r_mat, act_mat, snp, win, n_on)
    if f14 is not None:
        figs.append(f14)

    ax15 = f15.add_subplot(1, 1, 1)
    vrx = _linear(rd, "VRx")
    if vrx is not None:
        sel = np.flatnonzero(tact)
        sel = sel[sel < vrx.size]
        ax15.plot(vrx[sel], a[sel], ".")
        ons15 = onsets[onsets < vrx.size]
        ax15.plot(vrx[ons15], a[ons15], "ro")
    ax15.set_xlabel("VRx")
    ax15.set_ylabel(r"$\Delta$F/F")
    ax15.set_xlim(0, 5)
    figs.append(f15)
    return figs


def _m1_figure14(fig, vel_m_mat, vel_r_mat, act_mat, snp, win, n_on) -> None:
    fig.clear()
    axs = [fig.add_subplot(4, 1, k + 1) for k in range(4)]
    if n_on and vel_m_mat is not None and vel_r_mat is not None:
        m_mean = (-vel_m_mat).mean(axis=1)
        r_mean = vel_r_mat.mean(axis=1)
        axs[0].plot(_x(m_mean.size - 1), np.diff(m_mean))
        axs[0].plot(_x(r_mean.size - 1), np.diff(r_mean), "r")
        _tight(axs[0])
        _guides(axs[0], win, 2, zero_line=True)
        axs[1].plot(_x(m_mean.size), m_mean)
        axs[1].plot(_x(r_mean.size), r_mean, "r")
        _tight(axs[1])
        _guides(axs[1], win, 2, zero_line=True)
    axs[0].set_title("diff(velM): blue   diff(velR):red")
    axs[1].set_title("velM: blue   velR: red")
    if n_on:
        pop = act_mat.mean(axis=0).mean(axis=1)
        axs[2].plot(_x(pop.size), pop)
        _tight(axs[2])
        _guides(axs[2], win, 2)
    axs[2].set_title("population mean")
    if snp.shape[1]:
        axs[3].plot(_x(snp.shape[0]), snp.mean(axis=1))
        _tight(axs[3])
        _guides(axs[3], win, 2)
    axs[3].set_title(f"cell mean, number of onsets: {n_on}")


# ---------------------------------------------------------------------------------------
# cb_plugin_VML.m
# ---------------------------------------------------------------------------------------
def _vml_pages(m_snips: list, p_snips: list, n: int) -> np.ndarray:
    """``snp_mat`` ``[n x max(nM, nP, 1) x 2]``: the snippets of both parts, NaN-padded.

    MATLAB grew ``NaN(n,1,2)`` by assignment, which pads with zeros; here unused columns
    stay NaN, so :func:`_vml_means` averages only real snippets.
    """
    cols = max(len(m_snips), len(p_snips), 1)
    out = np.full((n, cols, 2), np.nan)
    for j, s in enumerate(m_snips):
        out[:, j, 0] = s
    for j, s in enumerate(p_snips):
        out[:, j, 1] = s
    return out


def _vml_means(pages: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``nanmean`` over the snippets of each part (NaN for a part without onsets)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN part: NaN, as MATLAB nanmean
        return np.nanmean(pages[:, :, 0], axis=1), np.nanmean(pages[:, :, 1], axis=1)


def plot_cb_plugin_vml(rd: Mapping, cell: int, channel_fields: Sequence[Sequence[str]]) -> list[Figure]:
    """Visual-flow / running coupling around activity onsets (``cb_plugin_VML.m``).

    Onsets: ``act > 1.5``, window 200 samples. Onsets before ``nbr_frames(1)`` (MATLAB's
    1-based onset compared with the length of the first recording) belong to the
    closed-loop part (red), the others to playback (green).

    - figure 12: ``velM_smoothed`` against ``velP_smoothed`` (limits 0.001 to 0.025) with
      the 20 samples before (red) and after (green) each onset (when channel row 0 has a
      field, as in MATLAB);
    - figure 11: one panel per channel row (row 0 at the bottom): mean snippet of every
      selected field for the two parts, title with the number of onsets on the last panel.

    Parameters
    ----------
    rd : mapping
        One ``rd`` entry (``act``, ``nbr_frames``, ``velM_smoothed``, ``velP_smoothed``).
    cell : int
        **0-based** row of ``act``.
    channel_fields : sequence of 5 sequences of str
        Field names of the browser's channel rows.

    Returns
    -------
    list of matplotlib.figure.Figure
        Figures 11 and 12.

    Differences from MATLAB
    -----------------------
    - The mean of each part is the ``nanmean`` over that part's snippets only. MATLAB's
      ``snp_mat`` starts as ``NaN(2*win+1, 1, 2)`` and grows by assignment, which pads
      with zeros: when one part had fewer onsets than the other, its mean included zero
      columns and was pulled towards 0. With equal onset counts both give the same means;
      a part without onsets is NaN in both.
    - The title reads "Number of onsets" (MATLAB: "Nuber"). No onset left after the window
      checks gives figures without means (MATLAB fails).
    """
    a = _act(rd, cell)
    vel_m = _row(rd, "velM_smoothed")
    vel_p = _row(rd, "velP_smoothed")
    nbr = np.ravel(np.asarray(rd.get("nbr_frames", np.zeros(0)), dtype=np.float64))
    first_len = float(nbr[0]) if nbr.size else np.inf
    f12 = _figure("cb_plugin_VML (12)")
    ax12 = f12.add_subplot(1, 1, 1)
    if vel_m is not None and vel_p is not None:
        ax12.plot(vel_m, vel_p)
    ax12.set_xlim(0.001, 0.025)
    ax12.set_ylim(0.001, 0.025)
    f11 = _figure("cb_plugin_VML (11)")
    win = 200
    onsets = activity_onsets(a, 1.5, win)
    n = 2 * win + 1
    last_ax = None
    for ind, fields in enumerate(channel_fields[:5]):
        for knd, field in enumerate(fields):
            tr = _channel_trace(rd, field, cell)
            m_snips, p_snips = [], []
            for o in onsets:
                seg = tr[o - win: o + win + 1] if tr is not None else np.full(n, np.nan)
                (m_snips if o + 1 < first_len else p_snips).append(seg)
                if ind == 0 and knd == 0 and vel_m is not None and vel_p is not None:
                    ax12.plot(vel_m[o - 20: o + 1], vel_p[o - 20: o + 1], "r", linewidth=2)
                    ax12.plot(vel_m[o: o + 21], vel_p[o: o + 21], "g", linewidth=2)
            mean_m, mean_p = _vml_means(_vml_pages(m_snips, p_snips, n))
            ax = f11.add_subplot(5, 1, 5 - ind) if knd == 0 else last_ax
            ax.plot(_x(n), mean_m, "r")
            ax.plot(_x(n), mean_p, "g")
            _tight(ax)
            last_ax = ax
    if last_ax is not None:
        last_ax.set_title(f"Number of onsets: {onsets.size}")
    return [f11, f12]


# ---------------------------------------------------------------------------------------
# cb_plugin_trav.m
# ---------------------------------------------------------------------------------------
def _xcorr_unbiased(x: np.ndarray) -> np.ndarray:
    """MATLAB ``xcorr(x, 'unbiased')``: lags ``-(N-1) .. N-1`` divided by ``N - |lag|``."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    c = scipy.signal.correlate(x, x, mode="full")
    lags = np.arange(-(n - 1), n)
    return c / (n - np.abs(lags))


def plot_cb_plugin_trav(rd: Mapping, cell: int) -> list[Figure]:
    """Activity by corridor position per traversal and by running speed (``cb_plugin_trav.m``).

    Computation: :func:`calliopy.projmeta.traversal_activity` with ``VRx(1,:)``, the
    cell's ``act``, ``GratFlash`` and ``-velM_smoothed(1,:)``.

    - figure 169: traversals x 100 position bins (colour limits: 5th and 99th percentile of
      all bins), the grating signal per bin below, the mean over traversals above;
    - figure 170: unbiased autocorrelation of the bins (NaN set to 0, lags +-200 bins) and
      of ``act`` (lags +-2000 samples), plotted at MATLAB's x positions (lag 0 at ``N``);
    - figure 171: mean activity +- standard error in 14 speed bins (``ylim([1 1.2])``).

    Parameters
    ----------
    rd : mapping
        One ``rd`` entry (``VRx``, ``act``, ``GratFlash``, ``velM_smoothed``).
    cell : int
        **0-based** row of ``act``.

    Returns
    -------
    list of matplotlib.figure.Figure
        Figures 169, 170, 171.

    Differences from MATLAB
    -----------------------
    ``imagesc`` uses matplotlib's ``viridis`` (MATLAB's default ``parula`` is not in
    matplotlib); the menu bar of figure 169 is not removed.
    """
    for name in ("VRx", "GratFlash", "velM_smoothed"):
        if _linear(rd, name) is None:
            raise KeyError(f"rd has no {name}")
    act = _act(rd, cell)
    res = traversal_activity(_row(rd, "VRx"), act, _linear(rd, "GratFlash"), -_row(rd, "velM_smoothed"))
    bins = res.bins
    aaa = bins.ravel(order="C")  # MATLAB reshape(bins', 1, [])
    f169 = _figure("cb_plugin_trav (169)")
    ax = f169.add_axes((0, 0.2, 1, 0.7))
    lo = float(prctile(aaa, 5)) if aaa.size else 0.0
    hi = float(prctile(aaa, 99)) if aaa.size else 1.0
    ax.imshow(bins, aspect="auto", interpolation="nearest", vmin=lo, vmax=hi)
    ax.set_axis_off()
    ax = f169.add_axes((0, 0, 1, 0.2))
    ax.imshow(res.grats, aspect="auto", interpolation="nearest")
    ax.set_axis_off()
    ax = f169.add_axes((0, 0.9, 1, 0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # MATLAB nanmean of an all-NaN bin: NaN
        ax.imshow(np.nanmean(bins, axis=0)[None, :] if bins.size else np.zeros((1, 100)),
                  aspect="auto", interpolation="nearest")
    ax.set_axis_off()

    f170 = _figure("cb_plugin_trav (170)")
    ax1 = f170.add_subplot(1, 2, 1)
    a0 = np.where(np.isnan(aaa), 0.0, aaa)
    if a0.size:
        ax1.plot(_x(2 * a0.size - 1), _xcorr_unbiased(a0))
    ax1.set_xlim(a0.size - 200, a0.size + 200)
    ax1.set_xlabel("distance")
    ax1.set_ylabel("correlation")
    ax2 = f170.add_subplot(1, 2, 2)
    ax2.plot(_x(2 * act.size - 1), _xcorr_unbiased(act))
    ax2.set_xlim(act.size - 2000, act.size + 2000)
    ax2.set_xlabel("time")

    f171 = _figure("cb_plugin_trav (171)")
    ax = f171.add_subplot(1, 1, 1)
    ax.errorbar(_x(res.bin_act_speed.size), res.bin_act_speed, res.bin_std_speed)
    ax.set_ylim(1, 1.2)
    ax.set_xlabel("Speed")
    ax.set_ylabel("Activity")
    return [f169, f170, f171]


PLUGINS: dict[str, Any] = {
    "stim": plot_cb_stim_snippets,
    "acx": plot_cb_plugin_acx,
    "m1": plot_cb_plugin_m1,
    "vml": plot_cb_plugin_vml,
    "trav": plot_cb_plugin_trav,
}
