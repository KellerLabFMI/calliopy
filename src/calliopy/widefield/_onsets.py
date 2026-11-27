"""Event-triggered widefield responses across a proj_meta (calliope 21_Analysis/WF_onsets).

:func:`wf_onsets` extracts baseline-subtracted activity snippets of the ROIs of one Allen
area around grating, running, visual-flow or mismatch onsets, averages them per site and
timepoint group, and :func:`plot_wf_onsets` draws the mean +/- SEM (or a hierarchical
bootstrap) per group.

Frame numbers inside this module follow MATLAB (1-based) where they are compared with
``nbr_frames`` boundaries; arguments that select sessions (``tp_list``) are **0-based**.
proj_meta conventions: see :mod:`calliopy.widefield._proj_meta_tools`. Nothing here has
been compared with MATLAB output.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any

import numpy as np

from ._proj_meta_tools import _contains, _field, _isempty, _rd_grid, _sites, _str_list, _vec

__all__ = ["ONSET_TYPES", "onset_triggers", "plot_wf_onsets", "wf_onsets"]

logger = logging.getLogger(__name__)

ONSET_TYPES = ("gratings", "running", "visualflow", "mismatch")
_TIME_SCALE = 100.0  # MATLAB plots window_frames / 100 as seconds


def _entry(site, idx: int):
    """rd entry ``idx`` (0-based) as WF_onsets reads it: ``rd(1, idx)`` for a single row,
    else linear ``rd(idx)``; ``None`` when out of range."""
    g = _rd_grid(site)
    num = g.size
    if num == 0 or not 0 <= idx < num:
        return None
    if g.shape[1] >= idx + 1 and num == g.shape[1]:
        return g[0, idx]
    return g.ravel(order="F")[idx]


def _eligible(rd, onset_type: str) -> bool:
    if not isinstance(rd, dict) or not rd:
        return False
    if not all(f in rd for f in ("act", "ROIinfo", "nbr_frames", "comments")):
        return False
    if onset_type == "gratings":
        return "StimulusID" in rd or "GratingID" in rd
    if onset_type == "running":
        return "Running_smoothed" in rd
    if onset_type == "visualflow":
        return "VisualFlow_smoothed" in rd
    return ("PerturbationLFM" in rd or "Perturbation" in rd) and "Running_smoothed" in rd


def _rising(trace: np.ndarray, thresh: float) -> np.ndarray:
    """1-based frames where ``trace > thresh`` starts (MATLAB ``find(diff([0, x>t]) == 1)``)."""
    above = np.concatenate([[False], trace > thresh]).astype(np.int8)
    return np.flatnonzero(np.diff(above) == 1) + 1


def _after_stillness(onsets, trace, thresh, min_still) -> np.ndarray:
    keep = [o for o in onsets
            if o > min_still and np.all(trace[o - min_still - 1 : o - 1] <= thresh)]
    return np.array(keep, dtype=np.int64)


def onset_triggers(
    rd,
    onset_type: str,
    *,
    thresh_stim: float = 0.5,
    thresh_run: float = 0.001,
    thresh_visual_flow: float = 0.001,
    min_still_duration: int = 100,
) -> np.ndarray:
    """Onset frames of one rd entry (WF_onsets local function ``get_triggers_by_type``).

    - ``'gratings'``: rising crossings of ``StimulusID`` (else ``GratingID``) above
      ``thresh_stim``;
    - ``'running'``: rising crossings of ``Running_smoothed`` above ``thresh_run``
      preceded by ``min_still_duration`` frames ``<= thresh_run``;
    - ``'visualflow'``: the same with ``VisualFlow_smoothed`` and ``thresh_visual_flow``;
    - ``'mismatch'``: rising crossings of ``PerturbationLFM`` (else ``Perturbation``) above
      0.5 at which ``Running_smoothed >= thresh_run``.

    Returns
    -------
    numpy.ndarray
        int64 **1-based frame numbers** (MATLAB values; subtract 1 to index ``act``).
    """
    t = onset_type.lower()

    def trace(name):
        v = _field(rd, name)
        return None if _isempty(v) else _vec(v).astype(np.float64)

    if t == "gratings":
        tr = trace("StimulusID")
        tr = tr if tr is not None else trace("GratingID")
        return _rising(tr, thresh_stim) if tr is not None else np.zeros(0, np.int64)
    if t in ("running", "visualflow"):
        name, th = (("Running_smoothed", thresh_run) if t == "running"
                    else ("VisualFlow_smoothed", thresh_visual_flow))
        tr = trace(name)
        if tr is None:
            return np.zeros(0, np.int64)
        return _after_stillness(_rising(tr, th), tr, th, int(min_still_duration))
    if t == "mismatch":
        run = trace("Running_smoothed")
        pert = trace("PerturbationLFM")
        pert = pert if pert is not None else trace("Perturbation")
        if pert is None or run is None:
            return np.zeros(0, np.int64)
        trig = _rising(pert, 0.5)
        trig = trig[trig <= run.size]
        return trig[run[trig - 1] >= thresh_run]
    raise ValueError(f"Unknown onset type: {onset_type}")


def _filter_running_state(rd, triggers, state, run_thresh, min_duration, min_prob):
    run = _field(rd, "Running_smoothed")
    if _isempty(run) or triggers.size == 0:
        if _isempty(run):
            logger.warning("Cannot filter by running state: missing Running_smoothed")
        return np.zeros(0, np.int64)
    is_run = _vec(run).astype(np.float64) > run_thresh
    n = is_run.size
    keep = np.zeros(triggers.size, bool)
    for k, trig in enumerate(triggers):
        if trig > n:
            continue
        end = min(trig + int(min_duration) - 1, n)
        if end < trig:
            continue
        prob = is_run[trig - 1 : end].mean()
        if (state == "running" and prob >= min_prob) or (
                state == "stationary" and prob < 1 - min_prob):
            keep[k] = True
    return triggers[keep]


def _normalize(act: np.ndarray, method: str) -> np.ndarray:
    if act.size == 0:
        return act
    if method in ("median", "mean"):
        f = np.median(act, axis=1, keepdims=True) if method == "median" else act.mean(
            axis=1, keepdims=True)
        f[f == 0] = 1
        with np.errstate(divide="ignore", invalid="ignore"):
            out = act / f
    else:
        out = act.copy()
    out[~np.isfinite(out)] = 0
    return out


def _groups(tp_list, proj_sites) -> list[list[int]]:
    if tp_list is None:
        g = _rd_grid(proj_sites[0]) if proj_sites else np.empty((0, 0))
        n = g.shape[1] if g.size else 0
        if n == 0:
            raise ValueError("Cannot determine default tp_list: proj_meta(1).rd is empty; "
                             "give tp_list")
        return [[t] for t in range(n)]
    if isinstance(tp_list, (list, tuple)) and any(
            isinstance(t, (list, tuple, np.ndarray)) for t in tp_list):
        return [[int(x) for x in np.ravel(t)] for t in tp_list]
    return [[int(t)] for t in np.ravel(tp_list)]


def wf_onsets(
    proj_meta,
    onset_type: str,
    *,
    comment=(),
    paradigm=(),
    session=(),
    session_type: str = "all",
    roi: str = "VISp",
    tp_list=None,
    running_state: str = "all",
    thresh_stim: float = 0.5,
    thresh_run: float = 0.001,
    thresh_visual_flow: float = 0.001,
    min_still_duration: int = 100,
    window=(-100, 300),
    baseline_window=(-250, -200),
    normalization_method: str = "median",
    min_duration: int = 300,
    min_state_prob: float = 0.5,
) -> np.ndarray:
    """Average baseline-subtracted onset responses per ROI, site and timepoint group.

    Computation of calliope ``21_Analysis/WF_onsets.m`` (plot: :func:`plot_wf_onsets`).
    For every site, group of ``tp_list`` and rd entry of the group:

    1. the entry needs ``act``, ``ROIinfo``, ``nbr_frames``, ``comments`` and the onset
       signal; it must match ``comment``/``paradigm``/``session`` (substrings of
       ``comments``/``paradigm``/``session``) and ``session_type`` ('PB'/'FB' in comments);
    2. ROIs whose ``Allen_acronym`` equals ``roi`` are taken and normalised per ROI
       (``'median'``/``'mean'``: divided by it, 0 replaced by 1; NaN/Inf become 0);
    3. onsets from :func:`onset_triggers`; with ``running_state`` 'running'/'stationary'
       only onsets whose next ``min_duration`` frames are running with probability
       ``>= min_state_prob`` / ``< 1 - min_state_prob``;
    4. onsets whose window or baseline leaves the recording, or that lie closer than
       ``max(|window|, |baseline_window|) + 1`` frames to a stack boundary
       (``[0, cumsum(nbr_frames)]``), are dropped;
    5. snippet ``act[:, trig + window]`` minus the mean over ``trig + baseline_window``.

    Snippets of a site and group are averaged over all onsets of all its entries.

    Parameters
    ----------
    proj_meta
        proj_meta struct array (see :mod:`calliopy.widefield._proj_meta_tools`).
    onset_type
        ``'gratings'``, ``'running'``, ``'visualflow'`` or ``'mismatch'``.
    tp_list
        **0-based** rd entries: ints (one group each) or a list of sequences (groups).
        Default: one group per column of the first site's ``rd``.
    window, baseline_window
        ``[left, right]`` frames relative to the onset (both ends included); the baseline
        must be ``<= 0``.
    comment, paradigm, session, session_type, roi, running_state, thresh_stim, thresh_run,
    thresh_visual_flow, min_still_duration, normalization_method, min_duration,
    min_state_prob
        MATLAB parameters of the same names (CamelCase there).

    Returns
    -------
    numpy.ndarray
        float64 ``[nROI x nWindowFrames x nSites x nGroups]``, NaN where a site/group has no
        data; ``(0, 0)`` when no data at all. ``nROI`` is the ROI count of the first
        site/group with data; a site with a different count stays NaN (with a warning).

    Notes
    -----
    Differences from MATLAB: ``session_type`` 'PB'/'FB' with several comments (a cell)
    accepts the entry when any comment contains the tag (MATLAB failed: non-scalar
    ``&&``). Nothing is printed.
    """
    t = str(onset_type).lower()
    if t not in ONSET_TYPES:
        raise ValueError(f"onset_type must be one of {ONSET_TYPES}, got {onset_type!r}")
    st = session_type.lower()
    if st not in ("all", "pb", "fb"):
        raise ValueError("session_type must be 'all', 'PB' or 'FB'")
    rs = running_state.lower()
    if rs not in ("all", "running", "stationary"):
        raise ValueError("running_state must be 'all', 'running' or 'stationary'")
    norm = normalization_method.lower()
    if norm not in ("median", "mean", "none"):
        raise ValueError("normalization_method must be 'median', 'mean' or 'none'")
    window = [int(w) for w in window]
    baseline_window = [int(w) for w in baseline_window]
    if any(b > 0 for b in baseline_window):
        raise ValueError("baseline_window must be <= 0")
    comment, paradigm, session = _str_list(comment), _str_list(paradigm), _str_list(session)
    sites = _sites(proj_meta)
    groups = _groups(tp_list, sites)
    win = np.arange(window[0], window[1] + 1)
    bl = np.arange(baseline_window[0], baseline_window[1] + 1)
    edge = max(max(abs(w) for w in window), max(abs(b) for b in baseline_window)) + 1
    snps_all = None
    for s, site in enumerate(sites):
        if _rd_grid(site).size == 0:
            continue
        for k, grp in enumerate(groups):
            acc = []
            for idx in grp:
                rd = _entry(site, idx)
                if rd is None or not isinstance(rd, dict) or "comments" not in rd:
                    continue
                if not _eligible(rd, t):
                    continue
                if comment and not _contains(rd["comments"], comment).any():
                    continue
                if paradigm and (_isempty(_field(rd, "paradigm")) or
                                 not _contains(rd["paradigm"], paradigm).any()):
                    continue
                if session and (_isempty(_field(rd, "session")) or
                                not _contains(rd["session"], session).any()):
                    continue
                if st != "all" and not _contains(rd["comments"], st.upper()).any():
                    continue
                infos = _sites(rd["ROIinfo"]) if not _isempty(rd["ROIinfo"]) else []
                if not infos or "Allen_acronym" not in infos[0]:
                    continue
                rois = [i for i, r in enumerate(infos) if str(_field(r, "Allen_acronym", ""))
                        == roi and not _isempty(_field(r, "Allen_acronym"))]
                if not rois:
                    continue
                if _isempty(_field(rd, "nbr_frames")):
                    continue
                act = _normalize(np.asarray(rd["act"], dtype=np.float64)[rois, :], norm)
                trig = onset_triggers(rd, t, thresh_stim=thresh_stim, thresh_run=thresh_run,
                                      thresh_visual_flow=thresh_visual_flow,
                                      min_still_duration=min_still_duration)
                if trig.size == 0:
                    continue
                if rs != "all":
                    trig = _filter_running_state(rd, trig, rs, thresh_run, min_duration,
                                                 min_state_prob)
                if trig.size == 0:
                    continue
                bounds = np.concatenate([[0], np.cumsum(_vec(rd["nbr_frames"]))])
                ok = ((trig + window[0] >= 1) & (trig + window[1] <= act.shape[1])
                      & (trig + baseline_window[0] >= 1))
                near = np.array([np.any(np.abs(x - bounds) < edge) for x in trig], dtype=bool)
                trig = trig[ok & ~near]
                if trig.size == 0:
                    continue
                for x in trig:
                    snip = act[:, x + win - 1]
                    acc.append(snip - act[:, x + bl - 1].mean(axis=1, keepdims=True))
            if not acc:
                continue
            mean = np.mean(np.stack(acc, axis=2), axis=2)
            if snps_all is None:
                snps_all = np.full((mean.shape[0], mean.shape[1], len(sites), len(groups)),
                                   np.nan)
            if mean.shape == snps_all.shape[:2]:
                snps_all[:, :, s, k] = mean
            else:
                warnings.warn(f"Dimension mismatch for site {s}, group {k}: expected "
                              f"{snps_all.shape[:2]}, got {mean.shape}; left NaN",
                              stacklevel=2)
    return np.zeros((0, 0)) if snps_all is None else snps_all


def _format_time_axis(ax, t: np.ndarray, tick_interval: float = 1.0) -> None:
    """WF_onsets local ``format_time_axis``: data plotted against 1..N, ticks in seconds."""
    n = t.size
    if n == 0:
        return
    ax.set_xlim(0.5, n + 0.5)
    lo, hi = float(t.min()), float(t.max())
    first = math.ceil(lo / tick_interval) * tick_interval
    last = math.floor(hi / tick_interval) * tick_interval
    ticks = list(np.arange(first, last + tick_interval / 2, tick_interval))
    if lo <= 0 <= hi and 0 not in ticks:
        ticks = sorted([*ticks, 0.0])
    ticks = [x for x in ticks if lo <= x <= hi]
    if not ticks or min(abs(x - lo) for x in ticks) > tick_interval / 4:
        ticks = [lo, *ticks]
    if min(abs(x - hi) for x in ticks) > tick_interval / 4:
        ticks = [*ticks, hi]
    ticks = sorted(set(ticks))
    idx = [int(np.argmin(np.abs(t - x))) + 1 for x in ticks]
    uniq, first_pos = np.unique(idx, return_index=True)
    ticks = [ticks[i] for i in first_pos]
    labels = []
    for x in ticks:
        if abs(x) < 1e-9:
            labels.append("0")
        elif abs(x - round(x)) < 1e-9:
            labels.append(f"{round(x):d}")
        elif tick_interval >= 0.5 or abs(x) >= 5:
            labels.append(f"{x:.1f}")
        else:
            labels.append(f"{x:.2f}")
    ax.set_xticks(uniq, labels)


def _plot_sem(ax, data: np.ndarray, color=None, sh_color=(0.9, 0.9, 0.9), trns: float = 1.0):
    """Port of calliope ``22_Plotting/plotSEM.m`` (mean +/- SEM over columns, x = 1..N)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(data, axis=1)
        cnt = np.sum(~np.isnan(data), axis=1)
        sem = np.nanstd(data, axis=1, ddof=1) / np.sqrt(np.maximum(cnt, 1))
        sem[cnt == 0] = np.nan
    x = np.arange(1, data.shape[0] + 1)
    ax.fill_between(x, m - sem, m + sem, color=sh_color, alpha=trns, linewidth=0)
    (h,) = ax.plot(x, m, "-", color=color)
    return h


def plot_wf_onsets(
    snps_all,
    tp_list=None,
    *,
    roi: str = "VISp",
    onset_type: str = "",
    running_state: str = "all",
    normalization_method: str = "median",
    window=(-100, 300),
    overlay_plots: bool = False,
    bootstrap_plot: bool = False,
    shade_trans: float = 1.0,
    rng: np.random.Generator | None = None,
):
    """Mean onset responses of :func:`wf_onsets` (plot part of WF_onsets.m).

    For each group, every ROI of every site with data is one trace; the traces are drawn as
    mean +/- SEM (``plotSEM``) or, with ``bootstrap_plot``, with
    :func:`calliopy.plotting.plot_bstrp` (1000 resamples, sites as groups). Subplots in a
    near-square grid, or all groups overlaid with a legend (``overlay_plots``). The x axis
    shows ``window_frames / 100`` seconds; shared y limits are the data range +/- 10 %.

    Parameters
    ----------
    snps_all
        From :func:`wf_onsets`.
    tp_list
        The groups used (0-based; shown 1-based in the titles).
    rng
        Required with ``bootstrap_plot``.

    Returns
    -------
    matplotlib.figure.Figure or None
        ``None`` when there is nothing to plot (MATLAB only printed a message).
    """
    import matplotlib.pyplot as plt

    s = np.asarray(snps_all, dtype=np.float64)
    if s.size == 0 or np.all(np.isnan(s)):
        return None
    n_g = s.shape[3]
    win = np.arange(int(window[0]), int(window[1]) + 1)
    t_axis = win / _TIME_SCALE
    groups = [[g] for g in range(n_g)] if tp_list is None else _groups(tp_list, [])
    valid = s[~np.isnan(s)]
    y_min, y_max = float(valid.min()), float(valid.max())
    rng_y = y_max - y_min or max(abs(y_min), 0.1)
    y_lims = (y_min - 0.1 * rng_y, y_max + 0.1 * rng_y)
    if overlay_plots:
        fig, ax0 = plt.subplots()
        axes = [ax0] * n_g
        colors = plt.get_cmap("tab10")
    else:
        cols = math.ceil(math.sqrt(n_g))
        rows = math.ceil(n_g / cols)
        fig, ax_grid = plt.subplots(rows, cols, squeeze=False)
        axes = list(ax_grid.ravel())
        for ax in axes[n_g:]:
            ax.set_axis_off()
    fig.suptitle(f"ROI: {roi}, Onset: {onset_type}, State: {running_state}, "
                 f"Norm: {normalization_method}")
    zero_idx = int(np.argmin(np.abs(t_axis))) + 1
    handles, labels = [], []
    for g in range(n_g):
        ax = axes[g]
        d = s[:, :, :, g]
        has = ~np.all(np.isnan(d), axis=(0, 1))
        d = d[:, :, has]
        idx = groups[g] if g < len(groups) else [g]
        idx_txt = (f"{min(idx) + 1}..{max(idx) + 1} ({len(idx)} total)" if len(idx) > 10
                   else "[" + " ".join(str(i + 1) for i in idx) + "]")
        if d.size == 0:
            if not overlay_plots:
                ax.set_title(f"Group {g + 1} (Idx: {idx_txt})\n(No Data)")
                ax.set_facecolor((0.95, 0.95, 0.95))
                ax.set_ylim(*y_lims)
            continue
        traces = np.transpose(d, (1, 0, 2)).reshape(d.shape[1], -1, order="F")
        color: Any = colors(g % 10) if overlay_plots else None
        if bootstrap_plot:
            if rng is None:
                raise ValueError("bootstrap_plot needs rng")
            from calliopy.plotting import plot_bstrp

            grp = np.repeat(np.arange(d.shape[2]), d.shape[0])
            kw = {"color": color} if color is not None else {}
            h = plot_bstrp(traces, grp, 1000, sh_color=color or (0.9, 0.9, 0.9),
                           trns=shade_trans, rng=rng, ax=ax, **kw).h
        else:
            h = _plot_sem(ax, traces, color=color,
                          sh_color=color if color is not None else (0.9, 0.9, 0.9),
                          trns=shade_trans if overlay_plots else 1.0)
        handles.append(h)
        labels.append(f"Group {g + 1}: {idx_txt}")
        if not overlay_plots:
            _format_time_axis(ax, t_axis)
            ax.axvline(zero_idx, color="r", linestyle="--")
            ax.set_title(f"Group {g + 1} (Idx: {idx_txt})")
            ax.set_xlabel("Time from onset (s)")
            ax.set_ylabel("Activity (norm., baseline sub.)")
            ax.grid(True)
            ax.set_ylim(*y_lims)
    if overlay_plots:
        ax = axes[0]
        _format_time_axis(ax, t_axis)
        ax.set_xlabel("Time from onset (s)")
        ax.set_ylabel("Activity (norm., baseline sub.)")
        ax.grid(True)
        ax.set_ylim(*y_lims)
        ax.axvline(zero_idx, color="r", linestyle="--")
        ax.legend(handles, labels, loc="best")
    return fig
