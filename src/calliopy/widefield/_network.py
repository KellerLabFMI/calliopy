"""Correlation analyses of widefield ROI activity across a proj_meta (calliope 21_Analysis).

Ports of ``WF_corrmat``, ``WF_corrmat_plot``, ``WF_corrmat_ROI_filter``,
``WF_corrmat_matrix``, ``WF_corrmatBoxplot``, ``WF_findLowCorrcoef``,
``WF_plotROICountsByLocation``, ``WF_LesioningROIs``, ``WF_seedcorr``,
``WF_plotCorrChangeInteractive`` (static part) and ``WF_maskROIs`` (computational core).

Computation and plotting are separate: every ``plot_*`` function takes computed results and
returns matplotlib figure(s). proj_meta conventions (accepted forms, 0-based timepoint and
ROI arguments, copies instead of in-place changes) are those of
:mod:`calliopy.widefield._proj_meta_tools`. Correlation matrices keep MATLAB's layout
``[ROI x ROI x site x timepoint]``.

Nothing here has been compared with MATLAB output.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any, NamedTuple

import numpy as np
import scipy.stats

from calliopy._matlab import imgaussfilt, linspace, mround, prctile
from calliopy.io import MatStruct

from ._grid import plot_roi_values
from ._proj_meta_tools import (
    N_GRID_ROIS,
    _as_struct_array,
    _cell_field,
    _concat_field,
    _contains,
    _draw_roi_map,
    _field,
    _filter_sites,
    _isempty,
    _matlab_logical,
    _n_tp,
    _num2str,
    _rd_grid,
    _rd_linear,
    _rd_row,
    _roi_range,
    _sites,
    _str_list,
    _vec,
    wf_bandpass,
)

__all__ = [
    "CorrChangePairs",
    "CorrContour",
    "CorrDistanceHistogram",
    "CorrmatMatrixResult",
    "CorrmatResult",
    "CorrmatRoiFilterResult",
    "FindLowCorrcoefResult",
    "LesioningResult",
    "RoiDensity",
    "SeedCorrResult",
    "corr_change_pairs",
    "corr_change_roi_density",
    "corr_distance_histogram",
    "plot_corr_change",
    "plot_mask_rois",
    "plot_roi_counts_by_location",
    "plot_wf_corrmat",
    "plot_wf_corrmat_boxplot",
    "plot_wf_corrmat_matrix",
    "plot_wf_corrmat_roi_filter",
    "plot_wf_lesioning_rois",
    "plot_wf_seedcorr",
    "roi_border_indices",
    "roi_mean_values",
    "roi_threshold_mask",
    "wf_corrmat",
    "wf_corrmat_boxplot",
    "wf_corrmat_matrix",
    "wf_corrmat_plot",
    "wf_corrmat_roi_filter",
    "wf_find_low_corrcoef",
    "wf_lesioning_rois",
    "wf_seedcorr",
]

logger = logging.getLogger(__name__)

_RUNNING_STATES = ("all", "running", "stationary")
_HIST_RES = 40  # WF_corrmat_plot histogram bins per axis
_HIST_SIGMA = 2.5  # WF_corrmat_plot Gaussian smoothing (bins)
_DIST_SCALE = 16  # WF_corrmat_plot: distances divided by 16 (bregma-lambda distance)


# --- shared pieces ------------------------------------------------------------------------


class CorrmatResult(NamedTuple):
    """Result of :func:`wf_corrmat` (MATLAB ``[corrmat_all, proj_meta]``)."""

    corrmat_all: np.ndarray  #: [ROI x ROI x site x timepoint]
    proj_meta: np.ndarray  #: sites used (after the siteID filter and band-pass)


def _check_logic(logic: str) -> None:
    if logic not in ("and", "or"):
        raise ValueError(f"logic must be 'and' or 'or', got {logic!r}")


def _check_running_state(state: str) -> None:
    if str(state).lower() not in _RUNNING_STATES:
        raise ValueError(f"running_state must be one of {_RUNNING_STATES}, got {state!r}")


def _sessions_to_use(rd, comment, paradigm, logic) -> np.ndarray:
    """Per-stack selection from 'comment'/'paradigm' (MATLAB ``contains`` on rd fields)."""
    sel = np.zeros(0, dtype=bool)
    if comment or paradigm:
        if comment and not paradigm:
            sel = _contains(rd["comments"], comment)
        elif paradigm and not comment:
            sel = _contains(rd["session"], paradigm)
        elif logic == "and":
            sel = _contains(rd["comments"], comment) & _contains(rd["session"], paradigm)
        else:
            sel = _contains(rd["comments"], comment) | _contains(rd["session"], paradigm)
    if sel.size == 0:
        sel = np.ones(_vec(rd["nbr_frames"]).size, dtype=bool)
    return sel


def _frame_selection(site, tp, sessions, running_state, run_thresh) -> np.ndarray:
    """``fr_to_use`` of WF_corrmat / WF_LesioningROIs (sessions x running state)."""
    rd = _rd_row(site, tp)
    nbr = _vec(rd["nbr_frames"]).astype(np.int64)
    cum = np.concatenate([[0], np.cumsum(nbr)])
    parts = []
    for j in range(sessions.size):
        n_j = int(nbr[j])
        sess = np.repeat(bool(sessions[j]), n_j)
        if running_state in ("running", "stationary"):
            run = _vec(_rd_linear(site, tp)["Running"]).astype(np.float64)
            seg = run[cum[j] : cum[j + 1]]
            if seg.size != n_j:
                raise IndexError("Running is shorter than the frames in nbr_frames")
            sess = sess & (seg > run_thresh if running_state == "running" else seg <= run_thresh)
        parts.append(sess)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=bool)


def _median_divide_stacks(act: np.ndarray, rd) -> np.ndarray:
    """The 'medianDiv_on' block shared by WF_corrmat, WF_corrmat_matrix, WF_LesioningROIs
    and WF_seedcorr, with MATLAB's stack boundaries (see :func:`wf_corrmat`)."""
    med_act = _field(rd, "median_act")
    if not _isempty(med_act):
        act = act * np.asarray(med_act, dtype=np.float64).reshape(-1, 1)
    frms = np.concatenate([[1], np.cumsum(_vec(rd["nbr_frames"]))]).astype(np.int64)
    n = act.shape[1]
    pieces = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in range(frms.size - 1):
            s, e = int(frms[k]), int(frms[k + 1])
            if s < n and e <= n:
                st = act[:, s - 1 : e - 1]
                med = np.median(st, axis=1, keepdims=True)
                med[med == 0] = np.finfo(float).eps
                pieces.append(st / med)
    return np.concatenate(pieces, axis=1) if pieces else act


def _activity(site, tp, *, comment, paradigm, logic, running_state, run_thresh,
              median_div_on) -> np.ndarray:
    """Activity of one site and timepoint after session, running and median selection."""
    rd = _rd_row(site, tp)
    sessions = _sessions_to_use(rd, comment, paradigm, logic)
    state = running_state
    if state not in _RUNNING_STATES:  # MATLAB compared case-sensitively and fell back to 'all'
        warnings.warn(f"runningState unclear. {state} should be either 'all', 'running', "
                      "'stationary'. Using 'all'.", stacklevel=3)
        state = "all"
    fr = _frame_selection(site, tp, sessions, state, run_thresh)
    act = np.asarray(rd["act"], dtype=np.float64)
    act = act[:, _matlab_logical(fr, act.shape[1])]
    if median_div_on:
        act = _median_divide_stacks(act, rd)
    return act


def _corrcoef_rows(x: np.ndarray) -> np.ndarray:
    """MATLAB ``corrcoef(x')``: correlation between rows (variables), NaN propagates."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] == 0:
        return np.zeros((0, 0))
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        c = np.corrcoef(x) if x.shape[1] > 1 else np.full((x.shape[0], x.shape[0]), np.nan)
    return np.atleast_2d(c)


def _assemble(entries: dict, n_sites: int, n_tp: int, fill: float = 0.0) -> np.ndarray:
    """MATLAB ``A(:,:,site,tp) = M`` growth: unassigned slices are zero (or ``fill``)."""
    if not entries:
        return np.zeros((0, 0))
    shapes = {v.shape for v in entries.values()}
    if len(shapes) != 1:
        raise ValueError(f"correlation matrices of different sizes cannot be stacked: {shapes}")
    shape = shapes.pop()
    out = np.full((*shape, n_sites, n_tp), fill)
    for (s, t), v in entries.items():
        out[:, :, s, t] = v
    return out


def _prepare_sites(proj_meta, site_id, bandpass_on, bandpass_range, bandpass_filter_fn):
    sites = _sites(proj_meta)
    if site_id is not None and np.size(site_id):
        sites = _filter_sites(sites, site_id)
    if sites and bandpass_on:
        tag = f"WF_bandpass_{_num2str(bandpass_range[0])}-{_num2str(bandpass_range[1])}"
        uuid = _field(sites[0], "uuid")
        if uuid is None:
            raise KeyError("bandpass_on needs a uuid field in proj_meta (MATLAB checks it)")
        if tag in str(uuid):
            logger.info("data already bandpassed in the requested range - not recomputed")
        else:
            sites = _sites(wf_bandpass(sites, bandpass_range, filter_fn=bandpass_filter_fn))
    return sites


def _roiinfo(sites, tp: int = 0):
    return _rd_row(sites[0], tp)["ROIinfo"]


def _n_structs(s) -> int:
    return len(_sites(s)) if not _isempty(s) else 0


# --- WF_corrmat ---------------------------------------------------------------------------


def wf_corrmat(
    proj_meta,
    *,
    comment=(),
    paradigm=(),
    logic: str = "and",
    site_id=None,
    tp_list=None,
    roi_mask=None,
    bandpass_on: bool = False,
    bandpass_range=(0.2, 8.0),
    median_div_on: bool = False,
    running_state: str = "all",
    run_thresh: float = 0.001,
    bandpass_filter_fn=None,
) -> CorrmatResult:
    """Correlation matrices of ROI activity per site and timepoint.

    Computation of calliope ``21_Analysis/WF_corrmat.m`` (plot: :func:`plot_wf_corrmat`).
    For every site and timepoint ``rd[0, tp]``:

    1. stacks (sub-sessions) are selected with ``comment``/``paradigm``: substrings of the
       ``comments``/``session`` fields, combined with ``logic`` when both are given; all
       stacks when neither is given;
    2. frames are selected by running state (``Running > run_thresh`` is running);
    3. optionally every stack is divided by its median (``median_div_on``, after
       multiplying by ``median_act`` when that field exists);
    4. ``corrcoef`` of the rows selected by ``roi_mask``.

    Parameters
    ----------
    proj_meta
        proj_meta struct array (see :mod:`calliopy.widefield._proj_meta_tools`).
    comment, paradigm
        String or list of strings (MATLAB ``'comment'``, ``'paradigm'``).
    logic
        ``'and'`` or ``'or'`` when both filters are given.
    site_id
        Keep sites whose ``ExpGroup(1)`` is in this list.
    tp_list
        **0-based** timepoints: a sequence of ints used for every site, or a list with one
        sequence of ints per site (MATLAB cell array). Default: all of the first site.
    roi_mask
        Boolean mask over the rows of ``act`` (default all; a shorter mask is padded with
        ``False``, as MATLAB logical indexing does).
    bandpass_on, bandpass_range, bandpass_filter_fn
        Band-pass with :func:`wf_bandpass` first, unless the first site's ``uuid`` already
        holds ``'WF_bandpass_<low>-<high>'``; ``bandpass_filter_fn`` is its ``filter_fn``.
    median_div_on, running_state, run_thresh
        MATLAB ``'medianDiv_on'``, ``'runningState'``, ``'runThresh'``.

    Returns
    -------
    CorrmatResult
        ``corrmat_all`` float64 ``[nROI x nROI x nSites x nTimepoints]`` (slices that were
        never computed, e.g. a site with fewer timepoints, are 0 as in MATLAB);
        ``proj_meta`` the sites used, object array ``(1, nSites)``. When ``site_id`` matches
        no site: a warning, ``corrmat_all`` of shape ``(0, 0)`` and an empty proj_meta.

    Notes
    -----
    Suspected MATLAB bugs kept (they change numbers):

    - Median division uses ``frms = [1, cumsum(nbr_frames)]`` and the columns
      ``frms(k) : frms(k+1)-1``, so the stacks are shifted by one frame after the first,
      the last frame of the recording is dropped, and it is applied *after* the frame
      selection, so the stack boundaries no longer match when stacks or frames were
      excluded.
    - With a char (not cell) ``comments``, ``contains`` returns one value and only the first
      stack's frames are selected.

    Differences from MATLAB: timepoints are 0-based; nothing is printed.
    """
    comment, paradigm = _str_list(comment), _str_list(paradigm)
    _check_logic(logic)
    _check_running_state(running_state)
    sites = _prepare_sites(proj_meta, site_id, bandpass_on, bandpass_range, bandpass_filter_fn)
    if not sites:
        return CorrmatResult(np.zeros((0, 0)), _as_struct_array([]))
    if tp_list is None:
        tp_list = list(range(_n_tp(sites[0])))
    per_site = isinstance(tp_list, (list, tuple)) and any(
        isinstance(t, (list, tuple, np.ndarray)) for t in tp_list)
    entries = {}
    n_tp = 0
    for s, site in enumerate(sites):
        tps = [int(t) for t in np.ravel(tp_list[s] if per_site else tp_list)]
        n_tp = max(n_tp, len(tps))
        for k, tp in enumerate(tps):
            act = _activity(site, tp, comment=comment, paradigm=paradigm, logic=logic,
                            running_state=running_state, run_thresh=run_thresh,
                            median_div_on=median_div_on)
            mask = np.ones(act.shape[0], bool) if roi_mask is None else _matlab_logical(
                roi_mask, act.shape[0])
            entries[(s, k)] = _corrcoef_rows(act[mask, :])
    return CorrmatResult(_assemble(entries, len(sites), n_tp), _as_struct_array(sites))


# --- WF_corrmat_plot ----------------------------------------------------------------------


class CorrDistanceHistogram(NamedTuple):
    """Result of :func:`corr_distance_histogram`."""

    normalized_data: np.ndarray  #: [40 x 40] counts / total; row = correlation bin (low first)
    smoothed_data: np.ndarray  #: imgaussfilt(normalized_data, 2.5)
    x_edges: np.ndarray  #: distance / 16 bin edges
    y_edges: np.ndarray  #: correlation bin edges


class CorrContour(NamedTuple):
    """A contour to draw on correlation-distance histograms (MATLAB ``contour_obj``)."""

    smoothed_data: np.ndarray  #: data the contour is drawn on (not flipped)
    level: float  #: contour level (0.5 * max of the data)


def _tril_vector(m: np.ndarray, drop_zeros: bool = True) -> np.ndarray:
    """MATLAB ``t = tril(m, -1); t(t ~= 0)``: strictly lower triangle in column-major order;
    zeros are dropped (NaN is kept)."""
    t = np.tril(m, -1)
    v = t.ravel(order="F")
    return v[v != 0] if drop_zeros else v


def _tril_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    """0-based (row, col) of the strict lower triangle of an n x n matrix, column-major."""
    cols, rows = np.nonzero(np.tril(np.ones((n, n), bool), -1).T)
    return rows, cols


def _pdist2(ml: np.ndarray, ap: np.ndarray) -> np.ndarray:
    p = np.column_stack([ml, ap])
    return np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(axis=2))


def _histogram2(x, y, x_edges, y_edges) -> np.ndarray:
    """MATLAB histogram2/histcounts2 counts ``N[xbin, ybin]`` (last bins right-closed;
    values outside the limits and NaN are not counted)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    h, _, _ = np.histogram2d(x[ok], y[ok], bins=[x_edges, y_edges])
    return h


def corr_distance_histogram(corrmat, proj_meta, y_bin_limits=(0.0, 1.0)) -> CorrDistanceHistogram:
    """2-D histogram of ROI-pair correlation against ROI distance (WF_corrmat_plot).

    Computation of calliope ``21_Analysis/WF_corrmat_plot.m``. ROI distances are the
    Euclidean distances of ``(ML_index, AP_index)`` from ``proj_meta(1).rd(1,1).ROIinfo``,
    divided by 16; with 824 ROIs the first two (bregma, lambda) are excluded. The strictly
    lower triangles of the distance matrix and of every correlation matrix
    (``corrmat[:, :, k]``) are taken without their zeros (MATLAB ``tril(..., -1) ~= 0``).
    40 x 40 bins, x limits ``[-0.2, 2]``, y limits ``y_bin_limits``; the counts are
    divided by their total and smoothed with ``imgaussfilt(..., 2.5)``.

    Parameters
    ----------
    corrmat
        ``[nROI x nROI]`` or ``[nROI x nROI x n]`` (e.g. one timepoint of ``corrmat_all``,
        all sites).
    proj_meta
        proj_meta struct array (only the first site's first ``ROIinfo`` is used).
    y_bin_limits
        Correlation limits (MATLAB ``'YBinLimits'``).

    Returns
    -------
    CorrDistanceHistogram
        ``normalized_data`` row ``i`` is the ``i``-th correlation bin from the bottom (MATLAB
        ``h.Values'``; the plot shows it flipped up-down).

    Raises
    ------
    ValueError
        The distance and correlation vectors differ in length (a zero correlation or two
        ROIs at the same position; MATLAB's ``histogram2`` failed then).
    """
    cm = np.asarray(corrmat, dtype=np.float64)
    cm3 = cm.reshape(cm.shape[0], cm.shape[1], -1)
    n = cm3.shape[0]
    info = _roiinfo(_sites(proj_meta))
    ml = _concat_field(info, "ML_index")
    ap = _concat_field(info, "AP_index")
    valid = np.r_[False, False, np.ones(n - 2, bool)] if n == 824 else np.ones(n, bool)
    dist = _pdist2(ml, ap)
    vm = _matlab_logical(valid, dist.shape[0])
    dvec = _tril_vector(dist[np.ix_(vm, vm)])
    cvals, dvals = [], []
    vc = _matlab_logical(valid, n)
    for k in range(cm3.shape[2]):
        cv = _tril_vector(cm3[:, :, k][np.ix_(vc, vc)])
        if cv.size != dvec.size:
            raise ValueError(f"{cv.size} correlations but {dvec.size} distances (zeros dropped)")
        cvals.append(cv)
        dvals.append(dvec)
    x_edges = linspace(-0.2, 2.0, _HIST_RES + 1)
    y_edges = linspace(float(y_bin_limits[0]), float(y_bin_limits[1]), _HIST_RES + 1)
    counts = _histogram2(np.concatenate(dvals) / _DIST_SCALE, np.concatenate(cvals),
                         x_edges, y_edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = counts.T / counts.sum()
    return CorrDistanceHistogram(norm, imgaussfilt(norm, _HIST_SIGMA), x_edges, y_edges)


def _style_hist_axes(ax, data):
    ax.set_xticks([0, mround(data.shape[1] / 2) - 1, data.shape[1] - 1])
    ax.set_yticks([0, mround(data.shape[0] / 2) - 1, data.shape[0] - 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(direction="out", width=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(2)
    ax.set_aspect("equal")


def _draw_contour(ax, contour: CorrContour):
    return ax.contour(np.flipud(contour.smoothed_data), levels=[contour.level], colors="w",
                      linewidths=2)


def wf_corrmat_plot(
    corrmat,
    proj_meta,
    *,
    contour: CorrContour | None = None,
    ax=None,
    smoothed_img: bool = False,
    y_bin_limits=(0.0, 1.0),
    scale: float = 1.0,
):
    """Correlation-against-distance heat map with a half-maximum contour (WF_corrmat_plot).

    Port of calliope ``21_Analysis/WF_corrmat_plot.m``. Shows the normalised histogram of
    :func:`corr_distance_histogram` (or its smoothed version), flipped so that high
    correlations are at the top, colormap magma, colour limits ``[0, 3e-3]``, and a white
    contour at half the maximum of the smoothed histogram. When ``contour`` is given (from
    an earlier call), that contour is drawn instead of a new one (MATLAB ``'contour_obj'``).

    Parameters
    ----------
    corrmat, proj_meta, y_bin_limits
        As :func:`corr_distance_histogram`.
    contour
        Contour to reuse; default: computed from this histogram.
    ax
        Axes to draw into (default: a new figure).
    smoothed_img
        Show the smoothed histogram instead of the raw one.
    scale
        Factor applied to the shown histogram (used by :func:`plot_wf_corrmat_roi_filter`
        for its fixed normalisation; 1 in MATLAB's WF_corrmat_plot).

    Returns
    -------
    tuple
        ``(contour, normalized_data, fig)``: the contour drawn (MATLAB ``contour_obj``), the
        normalised histogram (MATLAB ``normalized_data``, not flipped), the figure.
    """
    import matplotlib.pyplot as plt

    hist = corr_distance_histogram(corrmat, proj_meta, y_bin_limits)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.7, 4.7))
    else:
        fig = ax.figure
    shown = hist.smoothed_data if smoothed_img else hist.normalized_data
    ax.imshow(np.flipud(shown) * scale, cmap="magma", vmin=0, vmax=3e-3,
              interpolation="nearest")
    if contour is None:
        contour = CorrContour(hist.smoothed_data, 0.5 * float(np.nanmax(hist.smoothed_data)))
    _draw_contour(ax, contour)
    _style_hist_axes(ax, hist.normalized_data)
    return contour, hist.normalized_data, fig


def _title_from(sites, comment, paradigm, site_index=-1) -> str:
    s = sites[site_index]
    return " ".join([str(_field(s, "projID", "")), str(_field(s, "Acode", "")),
                     " ".join([*comment, *paradigm])])


def plot_wf_corrmat(corrmat_all, proj_meta, *, comment=(), paradigm=()):
    """Figure of :func:`wf_corrmat`: one :func:`wf_corrmat_plot` panel per timepoint.

    The contour of the first timepoint is drawn on every panel; the title is
    ``projID Acode comments paradigms`` of the last site (as MATLAB, whose loop variable
    was left at the last site).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    cm = np.asarray(corrmat_all)
    n_tp = cm.shape[3] if cm.ndim == 4 else 1
    fig, axes = plt.subplots(1, n_tp, squeeze=False, figsize=(3.5 * n_tp, 4))
    contour = None
    for t in range(n_tp):
        sl = cm[:, :, :, t] if cm.ndim == 4 else cm
        contour, _, _ = wf_corrmat_plot(sl, proj_meta, contour=contour, ax=axes[0, t])
        if t == 0:
            axes[0, t].set_title(_title_from(_sites(proj_meta), _str_list(comment),
                                             _str_list(paradigm)))
    fig.colorbar(axes[0, -1].images[0], ax=axes.ravel().tolist())
    return fig


# --- WF_corrmat_ROI_filter ----------------------------------------------------------------


class CorrmatRoiFilterResult(NamedTuple):
    """Result of :func:`wf_corrmat_roi_filter`."""

    corrmat_all_filtered: np.ndarray
    proj_meta: np.ndarray
    keep_mask: np.ndarray  #: over the brain ROIs (818 for 818/824-ROI data)
    corrmat_all: np.ndarray  #: unfiltered matrices from :func:`wf_corrmat`
    total_rois: int  #: number of ROIs in ROIinfo


def wf_corrmat_roi_filter(proj_meta, *, exclude_rois=(), **corrmat_kwargs) -> CorrmatRoiFilterResult:
    """Correlation matrices without the ROIs of some Allen areas.

    Computation of calliope ``21_Analysis/WF_corrmat_ROI_filter.m`` (plot:
    :func:`plot_wf_corrmat_roi_filter`): :func:`wf_corrmat` with ``corrmat_kwargs``, then
    rows and columns whose ``Allen_acronym`` equals one of ``exclude_rois`` are removed.

    Parameters
    ----------
    proj_meta
        proj_meta struct array.
    exclude_rois
        Allen acronyms to exclude (exact match), string or list.
    **corrmat_kwargs
        Passed to :func:`wf_corrmat`.

    Returns
    -------
    CorrmatRoiFilterResult

    Notes
    -----
    Suspected MATLAB bug kept: the keep mask has one entry per brain ROI (ROIs 3:820 of an
    824-ROI ROIinfo, 818 entries) but is applied to the full correlation matrix (824 rows
    when ``roi_mask`` is not given), so with 824 ROIs the mask is shifted by two ROIs and
    the last six rows are always dropped.

    Suspected MATLAB bug kept in the plot (:func:`plot_wf_corrmat_roi_filter`): the
    distances of the filtered matrices are taken from the *first* ``n`` ROIs of the full
    ROIinfo (MATLAB's logical index shorter than the distance matrix), not from the ROIs
    that were kept, so the distance axis of the filtered histograms is misaligned.
    """
    excl = _str_list(exclude_rois)
    corrmat_all, pm = wf_corrmat(proj_meta, **corrmat_kwargs)
    sites = _sites(pm)
    info = _roiinfo(sites)
    total = _n_structs(info)
    rr = np.arange(2, 820) if total == 824 else np.arange(total)
    acr = _cell_field(info, "Allen_acronym")
    keep = np.array([acr[i] not in excl for i in rr], dtype=bool)
    logger.info("Keeping %d out of %d ROIs", int(keep.sum()), keep.size)
    if corrmat_all.size == 0:
        return CorrmatRoiFilterResult(corrmat_all, pm, keep, corrmat_all, total)
    m = _matlab_logical(keep, corrmat_all.shape[0])
    filt = corrmat_all[np.ix_(m, m, np.arange(corrmat_all.shape[2]),
                              np.arange(corrmat_all.shape[3]))]
    return CorrmatRoiFilterResult(filt, pm, keep, corrmat_all, total)


def plot_wf_corrmat_roi_filter(
    result: CorrmatRoiFilterResult,
    *,
    exclude_rois=(),
    contour_all_rois: bool = True,
    contour_filtered_rois: bool = False,
    comment=(),
    paradigm=(),
):
    """Figure of :func:`wf_corrmat_roi_filter` with a fixed normalisation.

    Each timepoint is drawn with :func:`wf_corrmat_plot` on the filtered matrices; the image
    is multiplied by ``n_pairs_filtered / n_pairs_reference`` (pairs ``n(n-1)/2``; the
    reference is 818 ROIs for 824-ROI data, else the ROIinfo length) so that filtered and
    unfiltered plots share one scale. Contour, as in MATLAB:

    - ``contour_all_rois`` and ROIs excluded: the contour of the unfiltered first timepoint
      on every panel;
    - else ``contour_filtered_rois``: the contour of the filtered first timepoint, reused;
    - else every panel draws the contour of its own histogram.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    excl = _str_list(exclude_rois)
    cmf = result.corrmat_all_filtered
    n_tp = cmf.shape[3]
    n_ref = 818 if result.total_rois == 824 else result.total_rois
    n_act = cmf.shape[0]
    factor = (n_act * (n_act - 1) / 2) / (n_ref * (n_ref - 1) / 2)
    baseline = None
    if contour_all_rois and excl:
        h = corr_distance_histogram(result.corrmat_all[:, :, :, 0], result.proj_meta)
        sm = imgaussfilt(h.normalized_data, _HIST_SIGMA)
        baseline = CorrContour(sm, 0.5 * float(np.nanmax(sm)))
    fig, axes = plt.subplots(1, n_tp, squeeze=False, figsize=(3.5 * n_tp, 4))
    reuse = None
    for t in range(n_tp):
        ax = axes[0, t]
        if baseline is not None:
            c = baseline
        elif contour_filtered_rois:
            c = reuse
        else:
            c = None
        drawn, _, _ = wf_corrmat_plot(cmf[:, :, :, t], result.proj_meta, contour=c, ax=ax,
                                      scale=factor)
        if t == 0:
            reuse = drawn
            sites = _sites(result.proj_meta)
            title = _title_from(sites, _str_list(comment), _str_list(paradigm))
            if excl:
                title += " - Exclude: " + ",".join(excl)
            ax.set_title(title)
    fig.colorbar(axes[0, -1].images[0], ax=axes.ravel().tolist())
    return fig


# --- WF_corrmat_matrix --------------------------------------------------------------------


class CorrmatMatrixResult(NamedTuple):
    """Result of :func:`wf_corrmat_matrix` (MATLAB ``[corrmat_all, proj_meta]`` + labels)."""

    corrmat_all: np.ndarray  #: [nRegions x nRegions x nTimepoints]
    region_labels: list
    proj_meta: np.ndarray


def _region_labels(regions, separate, organization) -> list[str]:
    if not separate:
        return list(regions)
    if organization == 2:
        return [f"{r}-L" for r in regions] + [f"{r}-R" for r in regions]
    return [f"{r}-{h}" for r in regions for h in ("L", "R")]


def _region_trace(all_act, rr, acr, names, region, hemi, tp) -> np.ndarray:
    """Mean activity of the ROIs of one region (and hemisphere) for WF_corrmat_matrix."""
    hits = [i for i in rr if region in acr[i] and (hemi is None or hemi in names[i])]
    if not hits:
        side = "" if hemi is None else hemi.split()[0] + " hemisphere "
        warnings.warn(f"No {side}ROIs found for region {region} at timepoint {tp}",
                      stacklevel=3)
        return np.full(all_act.shape[1], np.nan)
    return all_act[hits, :].mean(axis=0)


def wf_corrmat_matrix(
    proj_meta,
    *,
    comment=(),
    paradigm=(),
    logic: str = "and",
    site_id=None,
    tp_list=None,
    bandpass_on: bool = False,
    bandpass_range=(0.2, 8.0),
    median_div_on: bool = False,
    allen_regions=("VISp", "RSP", "MOp", "SSp"),
    separate_hemispheres: bool = True,
    hemisphere_organization: int = 1,
    bandpass_filter_fn=None,
) -> CorrmatMatrixResult:
    """Correlation between Allen regions (ROIs averaged per region and hemisphere).

    Computation of calliope ``21_Analysis/WF_corrmat_matrix.m`` (plot:
    :func:`plot_wf_corrmat_matrix`). Per timepoint the selected activity of **all sites** is
    concatenated in time; ROIs whose ``Allen_acronym`` contains a region name (substring,
    MATLAB ``contains``) and whose ``name`` contains ``'left hemisphere'`` /
    ``'right hemisphere'`` are averaged; the region traces are correlated (``corrcoef``;
    with NaN, only time points without NaN in any region, MATLAB ``'rows','complete'``).

    Parameters
    ----------
    proj_meta, comment, paradigm, logic, site_id, bandpass_on, bandpass_range,
    median_div_on, bandpass_filter_fn
        As :func:`wf_corrmat` (no running-state filter, as in MATLAB).
    tp_list
        **0-based** timepoints (default all of the first site).
    allen_regions
        Region names (MATLAB default order ``{'VISp','RSP','MOp','SSp'}``).
    separate_hemispheres
        One region per hemisphere (``'<region>-L'``, ``'<region>-R'``).
    hemisphere_organization
        1: L/R alternating; 2: all left, then all right; 3: alternating, then reordered by
        average-linkage hierarchical clustering of ``1 - mean correlation`` with optimal
        leaf order.

    Returns
    -------
    CorrmatMatrixResult
        ``corrmat_all`` float64 ``[nRegions x nRegions x nTimepoints]``, ``region_labels``
        in matrix order, ``proj_meta`` used.

    Notes
    -----
    Suspected MATLAB behaviour kept: empty ``Allen_acronym``/``name`` entries are removed
    before ROI positions are taken, so with empty entries the positions shift; region
    names match as substrings (``'SSp'`` also matches ``'SSp-bfd'``, ...). Median-division
    stack boundaries as :func:`wf_corrmat`.

    Differences from MATLAB: ordering 3 uses :func:`scipy.cluster.hierarchy.linkage`
    (``'average'``) and :func:`scipy.cluster.hierarchy.optimal_leaf_ordering`; MATLAB's
    ``optimalleaforder`` minimises the same sum of adjacent-leaf distances, but ties may be
    broken differently. Missing regions give a warning and a NaN row, as in MATLAB.
    """
    comment, paradigm = _str_list(comment), _str_list(paradigm)
    _check_logic(logic)
    if hemisphere_organization not in (1, 2, 3):
        raise ValueError("hemisphere_organization must be 1, 2 or 3")
    regions = _str_list(allen_regions)
    sites = _prepare_sites(proj_meta, site_id, bandpass_on, bandpass_range, bandpass_filter_fn)
    if not sites:
        return CorrmatMatrixResult(np.zeros((0, 0)), [], _as_struct_array([]))
    tps = list(range(_n_tp(sites[0]))) if tp_list is None else [int(t) for t in np.ravel(tp_list)]
    labels = _region_labels(regions, separate_hemispheres, hemisphere_organization)
    out = np.zeros((len(labels), len(labels), len(tps)))
    for k, tp in enumerate(tps):
        acts = [_activity(s, tp, comment=comment, paradigm=paradigm, logic=logic,
                          running_state="all", run_thresh=0.0, median_div_on=median_div_on)
                for s in sites]
        all_act = np.concatenate(acts, axis=1)
        info = _roiinfo(sites, tp)
        acr = [a for a in _cell_field(info, "Allen_acronym") if a]
        names = [a for a in _cell_field(info, "name") if a]
        total = len(acr)
        rr = np.arange(2, 820) if total == 824 else np.arange(total)
        grouped = np.zeros((len(labels), all_act.shape[1]))

        if separate_hemispheres:
            for i, lab in enumerate(labels):
                region, side = lab[:-2], lab[-1]
                hemi = "left hemisphere" if side == "L" else "right hemisphere"
                grouped[i] = _region_trace(all_act, rr, acr, names, region, hemi, tp)
        else:
            for i, region in enumerate(regions):
                grouped[i] = _region_trace(all_act, rr, acr, names, region, None, tp)
        if np.isnan(grouped).any():
            warnings.warn(f"NaN values found in grouped activity for timepoint {tp}",
                          stacklevel=2)
            grouped = grouped[:, ~np.isnan(grouped).any(axis=0)]
        out[:, :, k] = _corrcoef_rows(grouped)
    if separate_hemispheres and hemisphere_organization == 3:
        import scipy.cluster.hierarchy as sch
        import scipy.spatial.distance as ssd

        d = 1.0 - out.mean(axis=2)
        np.fill_diagonal(d, 0.0)
        d = (d + d.T) / 2.0
        y = ssd.squareform(d, checks=False)
        z = sch.optimal_leaf_ordering(sch.linkage(y, "average"), y)
        order = sch.leaves_list(z)
        out = out[np.ix_(order, order, np.arange(out.shape[2]))]
        labels = [labels[i] for i in order]
    return CorrmatMatrixResult(out, labels, _as_struct_array(sites))


def plot_wf_corrmat_matrix(
    corrmat_all,
    region_labels,
    tp_list=None,
    *,
    colormap_name: str = "magma",
    color_range=(0.0, 1.0),
    figure_title: str = "",
    proj_meta=None,
    comment=(),
    paradigm=(),
):
    """Heat maps of :func:`wf_corrmat_matrix`, one per timepoint (MATLAB local function
    ``plot_correlation_matrices_by_timepoint``).

    ``tp_list`` (0-based) labels the panels as MATLAB did (``Timepoint <1-based>``). The
    default title is ``projID Acode comments paradigms`` of the first site.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    cm = np.asarray(corrmat_all)
    n_tp = cm.shape[2]
    tps = list(range(n_tp)) if tp_list is None else [int(t) for t in np.ravel(tp_list)]
    if not figure_title:
        parts = []
        if proj_meta is not None and _sites(proj_meta):
            s0 = _sites(proj_meta)[0]
            parts += [str(s0[f]) for f in ("projID", "Acode") if f in s0]
        figure_title = " ".join(parts + _str_list(comment) + _str_list(paradigm))
    fig, axes = plt.subplots(1, n_tp, squeeze=False, figsize=(min(3 * n_tp + 1, 16), 4))
    n = cm.shape[0]
    im = None
    for t in range(n_tp):
        ax = axes[0, t]
        im = ax.imshow(cm[:, :, t], cmap=colormap_name, vmin=color_range[0],
                       vmax=color_range[1], interpolation="nearest")
        ax.set_xticks(range(n), region_labels, rotation=45, fontsize=8)
        ax.set_yticks(range(n), region_labels, fontsize=8)
        ax.set_title(f"Timepoint {tps[t] + 1}", fontsize=12)
    if im is not None:
        cb = fig.colorbar(im, ax=axes.ravel().tolist())
        cb.set_label("Correlation Coefficient", fontsize=11)
    fig.suptitle(figure_title, fontsize=14, fontweight="bold")
    return fig


# --- WF_corrmatBoxplot --------------------------------------------------------------------


def _significance_stars(p: float) -> str:
    """Local function ``get_significance_stars`` of WF_corrmatBoxplot.m."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _hierarchical_bootstrap(site1: list, site2: list | None, nbss: int, alpha: float,
                            kind: str, rng: np.random.Generator):
    """Local function ``calculate_hierarchical_bootstrap_stats`` of WF_corrmatBoxplot.m.

    Returns ``(ci, p_value, is_significant)``; sites are resampled with replacement and all
    their values pooled; ``p = 2 * min(mean(reps <= 0), mean(reps >= 0))``.
    """
    ci, p, sig = np.array([np.nan, np.nan]), math.nan, False
    n1 = len(site1)
    if kind == "pairwise_diff":
        n2 = len(site2 or [])
        if n1 == 0 or n2 == 0:
            return ci, p, sig
        if n1 != n2:
            raise ValueError("Number of sites in site_data1 and site_data2 must be the same "
                             "for pairwise_diff.")
        if n1 < 2:
            logger.warning("Need at least 2 sites for hierarchical bootstrap pairwise_diff.")
            return ci, p, sig
    elif kind == "vs_zero":
        if n1 == 0:
            return ci, p, sig
    else:
        raise ValueError("comparison_type must be 'vs_zero' or 'pairwise_diff'")
    reps = np.zeros(nbss)
    for i in range(nbss):
        idx = rng.integers(0, n1, size=n1)
        d1 = np.concatenate([site1[j] for j in idx])
        if kind == "pairwise_diff":
            d2 = np.concatenate([site2[j] for j in idx])
            reps[i] = d1.mean() - d2.mean() if d1.size and d2.size else np.nan
        else:
            reps[i] = d1.mean() if d1.size else np.nan
    reps = reps[~np.isnan(reps)]
    if reps.size == 0:
        return ci, p, sig
    ci = np.asarray(prctile(reps, [alpha / 2 * 100, (1 - alpha / 2) * 100]))
    sig = not (ci[0] <= 0 <= ci[1])
    p = 2 * min(np.mean(reps <= 0), np.mean(reps >= 0))
    return ci, float(max(0.0, min(1.0, p))), sig


def wf_corrmat_boxplot(
    corrmat_all,
    proj_meta,
    *,
    bregma_lambda_distance: float = 16,
    alpha_level: float = 0.05,
    epsilon: float = 1e-6,
    use_bootstrap: bool = False,
    num_bootstrap_samples: int = 5000,
    rng: np.random.Generator | None = None,
) -> list[MatStruct]:
    """Change of short- and long-range correlations from the first timepoint, with tests.

    Computation of calliope ``21_Analysis/WF_corrmatBoxplot.m`` (plot:
    :func:`plot_wf_corrmat_boxplot`). For every timepoint ``k >= 1`` and site: the strictly
    lower triangle of ``(C_k - C_0) / (C_0 + epsilon) * 100``, split at the distance
    ``0.9 * bregma_lambda_distance`` (``ML_index``/``AP_index`` units) into short range
    (``<=``) and long range. The first two ROIs are left out when ``ROIinfo`` has a field
    ``label`` and ROI 1 is ``'Bregma'`` or ROI 2 ``'Lambda'``.

    Tests per comparison: two-sample t-test short vs long (``ttest2``, equal variances) and
    one-sample t-tests against 0 (``ttest``), NaN omitted; or, with ``use_bootstrap``, the
    hierarchical bootstrap of the MATLAB file (sites resampled, values pooled; CI at
    ``alpha_level``; ``p = 2 min(P(rep <= 0), P(rep >= 0))``).

    Parameters
    ----------
    corrmat_all
        ``[ROI x ROI x site x timepoint]`` from :func:`wf_corrmat`.
    proj_meta
        The proj_meta used for ``corrmat_all`` (``ROIinfo`` of each site's first rd entry).
    bregma_lambda_distance, alpha_level, epsilon, use_bootstrap, num_bootstrap_samples
        MATLAB parameters of the same names.
    rng
        Random generator for the bootstrap (required when ``use_bootstrap``).

    Returns
    -------
    list of MatStruct
        One per comparison, fields ``baseline_tp``/``current_tp`` (0-based),
        ``short_range``/``long_range`` (pooled 1-D arrays), ``short_by_site``/
        ``long_by_site`` (lists of arrays of the sites that had ROIinfo),
        ``p_short_vs_long``, ``p_short_vs_no_change``, ``p_long_vs_no_change`` (NaN when a
        group is empty), ``ci_*`` for the bootstrap, ``method``. Empty list with fewer than
        two timepoints.

    Notes
    -----
    Differences from MATLAB: the t-test branch (the default) could not run in MATLAB: the
    star label of the short-range test used ``p_long_vs_no_change`` before it was defined
    and the long-range label an undefined variable. Here each label uses its own p-value.
    Random draws differ from MATLAB's.
    """
    cm = np.asarray(corrmat_all, dtype=np.float64)
    if cm.ndim < 4 or cm.shape[3] < 2:
        logger.info("Not enough timepoints in corrmat_all to perform comparisons (need 2).")
        return []
    if use_bootstrap and rng is None:
        raise ValueError("use_bootstrap needs rng (a numpy.random.Generator)")
    n_reg, n_sites, n_tp = cm.shape[0], cm.shape[2], cm.shape[3]
    sites = _sites(proj_meta)
    thr = 0.9 * bregma_lambda_distance
    out = []
    for cur in range(1, n_tp):
        shorts, longs = [], []
        for s in range(n_sites):
            g = _rd_grid(sites[s]) if s < len(sites) else np.empty((0, 0))
            info = _field(g.ravel(order="F")[0], "ROIinfo") if g.size else None
            if _isempty(info):
                logger.warning("ROIinfo not found for site %d; site skipped", s)
                continue
            ml, ap = _concat_field(info, "ML_index"), _concat_field(info, "AP_index")
            valid = np.ones(n_reg, bool)
            rois = _sites(info)
            if n_reg > 2 and "label" in rois[0] and (
                    str(rois[0]["label"]) == "Bregma" or str(rois[1]["label"]) == "Lambda"):
                valid[:2] = False
            vi = np.flatnonzero(valid)
            dist = _pdist2(ml[vi], ap[vi])
            r, c = _tril_pairs(vi.size)
            base = cm[:, :, s, 0][np.ix_(vi, vi)][r, c]
            now = cm[:, :, s, cur][np.ix_(vi, vi)][r, c]
            delta = (now - base) / (base + epsilon) * 100
            d = dist[r, c]
            shorts.append(delta[d <= thr])
            longs.append(delta[d > thr])
        short = np.concatenate(shorts) if shorts else np.zeros(0)
        long_ = np.concatenate(longs) if longs else np.zeros(0)
        res = MatStruct(baseline_tp=0, current_tp=cur, short_range=short, long_range=long_,
                        short_by_site=shorts, long_by_site=longs)
        sb = [x for x in shorts if x.size]
        lb = [x for x in longs if x.size]
        if use_bootstrap:
            res.method = "hierarchical bootstrap"
            nb = int(num_bootstrap_samples)
            ci, p, _ = (_hierarchical_bootstrap(sb, lb, nb, alpha_level, "pairwise_diff", rng)
                        if sb and lb else (np.full(2, np.nan), math.nan, False))
            res.p_short_vs_long, res.ci_short_vs_long = p, ci
            ci, p, _ = (_hierarchical_bootstrap(sb, None, nb, alpha_level, "vs_zero", rng)
                        if sb else (np.full(2, np.nan), math.nan, False))
            res.p_short_vs_no_change, res.ci_short_vs_no_change = p, ci
            ci, p, _ = (_hierarchical_bootstrap(lb, None, nb, alpha_level, "vs_zero", rng)
                        if lb else (np.full(2, np.nan), math.nan, False))
            res.p_long_vs_no_change, res.ci_long_vs_no_change = p, ci
        else:
            res.method = "t-test"
            res.p_short_vs_long = float(scipy.stats.ttest_ind(
                short, long_, nan_policy="omit").pvalue) if short.size and long_.size else math.nan
            res.p_short_vs_no_change = float(scipy.stats.ttest_1samp(
                short, 0.0, nan_policy="omit").pvalue) if short.size else math.nan
            res.p_long_vs_no_change = float(scipy.stats.ttest_1samp(
                long_, 0.0, nan_policy="omit").pvalue) if long_.size else math.nan
        out.append(res)
    return out


def plot_wf_corrmat_boxplot(results: list, *, ylim=(-80, 80)) -> list:
    """Box plots of :func:`wf_corrmat_boxplot`, one figure per comparison.

    Boxes without outliers at x = 1 (short range) and 2 (long range), the mean as an open
    circle, a dashed zero line, significance stars (``*`` p < 0.05, ``**`` < 0.01,
    ``***`` < 0.001, else ``n.s.``) above (short vs long, with a line) and below each box.

    Returns
    -------
    list of matplotlib.figure.Figure

    Notes
    -----
    Differences from MATLAB: matplotlib's box quartiles (linear interpolation) and
    whiskers (1.5 IQR) can differ slightly from MATLAB's ``boxplot``.
    """
    import matplotlib.pyplot as plt

    figs = []
    lo, hi = float(ylim[0]), float(ylim[1])
    for res in results:
        fig, ax = plt.subplots()
        for x, data in ((1, res.short_range), (2, res.long_range)):
            if data.size:
                ax.boxplot(data[~np.isnan(data)], positions=[x], widths=0.5, showfliers=False)
                ax.plot(x, np.mean(data), "ko", markerfacecolor="w", markersize=5)
        ax.set_ylabel("Change in correlation [%]")
        ax.set_title(f"Aggregated Correlation Change (TP {res.baseline_tp + 1} vs TP "
                     f"{res.current_tp + 1}) Across All Sites")
        ax.set_xticks([1, 2], ["Short-range", "Long-range"], fontsize=10)
        ax.tick_params(direction="out")
        ax.plot([0.5, 2.5], [0, 0], "k--")
        ax.set_ylim(lo, hi)
        ax.set_xlim(0.5, 2.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if not math.isnan(res.p_short_vs_long):
            ax.text(1.5, hi * 0.9, _significance_stars(res.p_short_vs_long), ha="center",
                    va="top", color="b", fontsize=10)
            ax.plot([1, 2], [hi * 0.88, hi * 0.88], color="b")
        for x, p in ((1, res.p_short_vs_no_change), (2, res.p_long_vs_no_change)):
            if not math.isnan(p):
                ax.text(x, lo + (hi - lo) * 0.05, _significance_stars(p), ha="center",
                        va="bottom", color="b", fontsize=10)
        figs.append(fig)
    return figs


# --- WF_findLowCorrcoef / WF_plotROICountsByLocation --------------------------------------


class FindLowCorrcoefResult(NamedTuple):
    """Result of :func:`wf_find_low_corrcoef` (MATLAB ``[idx_counts_sort_all, siteID]``)."""

    idx_counts_sort_all: list  #: per site: int64 [k x 2] (0-based ROI, count), or (0, 0)
    site_id: list  #: 0-based sites (third dimension of corrmat_all) analysed


def wf_find_low_corrcoef(
    corrmat_all,
    *,
    site_id=None,
    tp_list=None,
    cc_range=(0.0, 0.1),
    cc_th=None,
    lower_triangle_only: bool = False,
) -> FindLowCorrcoefResult:
    """Count how often each ROI takes part in pairs with a correlation in ``cc_range``.

    Computation of calliope ``21_Analysis/WF_findLowCorrcoef.m`` (plot:
    :func:`plot_roi_counts_by_location`). Per site and timepoint, entries with
    ``cc_range[0] <= C <= cc_range[1]`` are kept (the others set to NaN), the matrix is
    reduced with ``tril(C, -1)``, and every non-NaN position (row, column) counts once for
    its row ROI and once for its column ROI. Counts are summed over timepoints and sorted
    in descending order (ties in ascending ROI order).

    Parameters
    ----------
    corrmat_all
        ``[ROI x ROI x site x timepoint]``.
    site_id
        **0-based** sites (third dimension); default all.
    tp_list
        Timepoints; only its **length** matters (see Notes). Default: all.
    cc_range
        ``[min, max]`` inclusive. ``cc_th`` (deprecated in MATLAB) sets ``[0, cc_th]`` with a
        warning.
    lower_triangle_only
        ``False`` (default) reproduces MATLAB (see Notes). ``True`` counts only the strictly
        lower triangle, which is what the MATLAB comments describe.

    Returns
    -------
    FindLowCorrcoefResult
        ``idx_counts_sort_all[i]``: int64 ``[k x 2]``, column 0 the **0-based** ROI index,
        column 1 its count; ``(0, 0)`` when nothing is in range. ``site_id``: 0-based.

    Notes
    -----
    Suspected MATLAB bugs kept by default (they change numbers):

    - ``tril(C, -1)`` writes **zeros**, not NaN, into the diagonal and upper triangle, and
      ``find(~isnan(...))`` then counts all of those positions as well. Each ROI ``i``
      (0-based, ``n`` ROIs) therefore gets ``n + 1`` extra counts per timepoint.
    - The timepoint loop reads ``corrmat_all(:, :, site, tp_idx)`` with the loop counter, so
      the first ``len(tp_list)`` timepoints are used whatever ``tp_list`` contains.
    """
    cm = np.asarray(corrmat_all, dtype=np.float64)
    cm = cm.reshape(*cm.shape, *([1] * (4 - cm.ndim)))
    if cc_th is not None:
        warnings.warn("cc_th is deprecated. Use cc_range instead.", DeprecationWarning,
                      stacklevel=2)
        cc_range = (0.0, float(cc_th))
    lo, hi = float(cc_range[0]), float(cc_range[1])
    if lo > hi:
        raise ValueError("cc_range must be [min, max] with min <= max")
    sites = list(range(cm.shape[2])) if site_id is None else [int(s) for s in np.ravel(site_id)]
    n_tp = cm.shape[3] if tp_list is None else len(np.ravel(tp_list))
    out = []
    for s in sites:
        idx_all = []
        for t in range(n_tp):
            c = cm[:, :, s, t].copy()
            c[~((c >= lo) & (c <= hi))] = np.nan
            if lower_triangle_only:
                low = np.tril(np.ones(c.shape, bool), -1)
                c[~low] = np.nan
            else:
                c = np.tril(c, -1)
            cols, rows = np.nonzero(~np.isnan(c).T)  # MATLAB find: column-major order
            idx_all.append(np.column_stack([rows, cols]))
        pairs = np.concatenate(idx_all) if idx_all else np.zeros((0, 2), np.int64)
        if pairs.size:
            u, counts = np.unique(pairs.ravel(order="F"), return_counts=True)
            order = np.argsort(-counts, kind="stable")
            out.append(np.column_stack([u[order], counts[order]]).astype(np.int64))
        else:
            logger.info("No ROI pairs in range [%.3f, %.3f] for site %d", lo, hi, s)
            out.append(np.zeros((0, 0), dtype=np.int64))
    return FindLowCorrcoefResult(out, sites)


def plot_roi_counts_by_location(proj_meta, idx_counts_sort_all, site_id, plot_sz: float = 30000):
    """Scatter maps of ROI counts (WF_plotROICountsByLocation), one figure per site.

    Port of calliope ``21_Analysis/WF_plotROICountsByLocation.m``. All ROIs are drawn as
    black circles at ``(ML_index, AP_index)`` of ``proj_meta(1).rd(1).ROIinfo``; ROIs
    counted more than once are drawn filled, with area ``count / sum(counts) * plot_sz`` and
    the magma colour of their count (``magma(max_count)``).

    Parameters
    ----------
    proj_meta
        proj_meta struct array (``animal`` of each analysed site is the title).
    idx_counts_sort_all, site_id
        Output of :func:`wf_find_low_corrcoef` (0-based ROI indices and sites).
    plot_sz
        Size factor (MATLAB default 30000).

    Returns
    -------
    list of matplotlib.figure.Figure

    Notes
    -----
    Differences from MATLAB: a site with no counted ROI gets an empty map (MATLAB failed
    indexing the empty result). The ROI indices are 0-based. matplotlib's magma resampled to
    ``max_count`` colours may differ slightly from the lab's ``magma.m``.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    sites = _sites(proj_meta)
    info = _roiinfo(sites)
    ml, ap = _concat_field(info, "ML_index"), _concat_field(info, "AP_index")
    figs = []
    for k, ic in enumerate(idx_counts_sort_all):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(ml, ap, facecolors="none", edgecolors="k", label="All ROIs")
        ic = np.asarray(ic)
        if ic.size:
            idx, counts = ic[:, 0].astype(int), ic[:, 1].astype(int)
            total, max_count = counts.sum(), int(counts.max())
            cmap = matplotlib.colormaps["magma"].resampled(max_count)
            for i, c in zip(idx, counts, strict=True):
                if c == 1:
                    continue
                ax.scatter(ml[i], ap[i], s=c / total * plot_sz, color=cmap(c - 1),
                           label=f"ROI {i} (count: {c})")
            sm = matplotlib.cm.ScalarMappable(
                cmap=cmap, norm=matplotlib.colors.Normalize(1, max(max_count, 2)))
            fig.colorbar(sm, ax=ax).set_label("Occurrence Count")
            if idx.size <= 10:
                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        ax.set_xlabel("ML Index")
        ax.set_ylabel("AP Index")
        animal = _field(sites[site_id[k]], "animal", "") if site_id[k] < len(sites) else ""
        ax.set_title(f"ROI Locations with Occurrence Counts of {animal}", fontsize=15)
        ax.set_aspect("equal")
        figs.append(fig)
    return figs


# --- WF_LesioningROIs ---------------------------------------------------------------------


class LesioningResult(NamedTuple):
    """Result of :func:`wf_lesioning_rois`."""

    lesioning_scores: MatStruct  #: keys 'TP<n>' (n 1-based, as MATLAB's field names)
    corrmat_all: np.ndarray  #: [ROI x ROI x site x timepoint group]


def _tp_groups(tp_list, n_default: int) -> list[list[int]]:
    if tp_list is None:
        return [[t] for t in range(n_default)]
    groups = []
    for t in tp_list:
        groups.append([int(x) for x in np.ravel(t)])
    return groups


def _upper_mean_without(c: np.ndarray, remove: np.ndarray) -> float:
    keep = ~remove
    sub = c[np.ix_(keep, keep)]
    iu = np.triu_indices(sub.shape[0], 1)
    vals = sub[iu]
    return float(vals.mean()) if vals.size else math.nan


def wf_lesioning_rois(
    proj_meta,
    *,
    comment=(),
    paradigm=(),
    logic: str = "and",
    site_id=None,
    tp_list=None,
    roi_mask=None,
    bandpass_on: bool = False,
    bandpass_range=(0.2, 8.0),
    median_div_on: bool = False,
    running_state: str = "all",
    run_thresh: float = 0.001,
    bandpass_filter_fn=None,
) -> LesioningResult:
    """Average ROI correlation and the effect of removing ROIs or Allen regions.

    Computation of calliope ``21_Analysis/WF_LesioningROIs.m`` (plot:
    :func:`plot_wf_lesioning_rois`). Correlation matrices as :func:`wf_corrmat`, with each
    ``tp_list`` entry a group of timepoints whose activity is concatenated. Per group the
    matrix is averaged over sites, then:

    - ``baseline``: mean of all off-diagonal entries;
    - ``roi_avg_corr``: per ROI, the mean correlation with all other ROIs;
    - ``lesion_change``: per ROI, mean of the upper triangle without that ROI minus
      ``baseline``;
    - ``region_names``/``region_values``: the same with all ROIs of one Allen acronym
      removed (sorted unique acronyms of the brain ROIs, 3:820 of 824 or 1:818).

    Parameters
    ----------
    proj_meta, comment, paradigm, logic, site_id, bandpass_on, bandpass_range,
    median_div_on, running_state, run_thresh, bandpass_filter_fn
        As :func:`wf_corrmat`.
    tp_list
        **0-based** timepoints: ints (one group each) or a list of sequences (groups).
    roi_mask
        Rows of ``act`` used; default ROIs 3:820 (1-based) of 824, else all.

    Returns
    -------
    LesioningResult
        ``lesioning_scores['TP<n>']`` (``n`` = 1-based timepoint, MATLAB's field name; a
        group gives ``'TP[a b]'``) with fields ``roi_avg_corr``, ``lesion_change`` (float64
        ``(nROI,)``), ``region_names`` (list), ``region_values`` (float64), ``baseline``.

    Notes
    -----
    Differences from MATLAB: a group of several timepoints gives the key ``'TP[a b]'``,
    which MATLAB could not use as a field name (it failed). The region masks have one
    entry per brain ROI (818) and are applied with MATLAB's logical-index rule, so a
    ``roi_mask`` that does not select exactly those 818 ROIs misaligns regions, as in
    MATLAB.
    """
    comment, paradigm = _str_list(comment), _str_list(paradigm)
    _check_logic(logic)
    _check_running_state(running_state)
    sites = _sites(proj_meta)
    if roi_mask is None and sites:
        n0 = np.asarray(_rd_linear(sites[0], 0)["act"]).shape[0]
        roi_mask = np.zeros(824, bool) if n0 == 824 else np.ones(n0, bool)
        if n0 == 824:
            roi_mask[2:820] = True
    sites = _prepare_sites(sites, site_id, bandpass_on, bandpass_range, bandpass_filter_fn)
    if not sites:
        return LesioningResult(MatStruct(), np.zeros((0, 0)))
    groups = _tp_groups(tp_list, _n_tp(sites[0]))
    info = _roiinfo(sites)
    rr = _roi_range(_n_structs(info))
    acr_all = _cell_field(info, "Allen_acronym")
    acr = [acr_all[i] for i in rr]
    entries = {}
    for s, site in enumerate(sites):
        for k, grp in enumerate(groups):
            acts = [_activity(site, tp, comment=comment, paradigm=paradigm, logic=logic,
                              running_state=running_state.lower(), run_thresh=run_thresh,
                              median_div_on=median_div_on) for tp in grp]
            act = np.concatenate(acts, axis=1)
            m = _matlab_logical(roi_mask, act.shape[0])
            entries[(s, k)] = _corrcoef_rows(act[m, :])
    cm_all = _assemble(entries, len(sites), len(groups))
    scores = MatStruct()
    n = cm_all.shape[0]
    regions = sorted(set(acr))
    for k, grp in enumerate(groups):
        c = cm_all[:, :, :, k].mean(axis=2)
        off = ~np.eye(n, dtype=bool)
        base = float(c[off].mean())
        roi_avg = np.array([np.delete(c[i], i).mean() for i in range(n)])
        if np.isnan(c).any():
            lesion = np.array([_upper_mean_without(c, np.eye(1, n, i, dtype=bool)[0])
                               for i in range(n)]) - base
        else:
            iu = np.triu_indices(n, 1)
            total = c[iu].sum()
            row = c.sum(axis=1) - np.diag(c)
            lesion = (total - row) / ((n - 1) * (n - 2) / 2) - base
        reg_vals = np.array([
            _upper_mean_without(c, _matlab_logical(np.array([a == r for a in acr]), n))
            for r in regions]) - base
        name = f"TP{grp[0] + 1}" if len(grp) == 1 else "TP[" + " ".join(
            str(t + 1) for t in grp) + "]"
        scores[name] = MatStruct(roi_avg_corr=roi_avg, lesion_change=lesion,
                                 region_names=list(regions), region_values=reg_vals,
                                 baseline=base)
    return LesioningResult(scores, cm_all)


def plot_wf_lesioning_rois(
    lesioning_scores: MatStruct,
    proj_meta,
    *,
    template=None,
    show_atlas_borders: bool = True,
    border_color: Any = "w",
):
    """The three figures of WF_LesioningROIs (average correlation, ROI and region lesions).

    Parameters
    ----------
    lesioning_scores
        From :func:`wf_lesioning_rois`.
    proj_meta
        For the title (``projID``, ``Acode``) and, without ``template``, the ROI coordinates.
    template
        :class:`GridTemplate`: values painted into the atlas image (background 0, as
        MATLAB), Allen borders drawn. Without it the values are interpolated on the
        ML/AP grid (linear; MATLAB used ``'natural'``).

    Returns
    -------
    tuple of matplotlib.figure.Figure
        ``(fig1, fig2, fig3)``: magma with global min/max; diverging maps symmetric about 0.

    Notes
    -----
    Differences from MATLAB: the lab's ``redblue`` colormap is replaced by matplotlib's
    ``'bwr'``; the template is an argument.
    """
    import matplotlib.pyplot as plt

    sites = _sites(proj_meta)
    info = _roiinfo(sites)
    rr = _roi_range(_n_structs(info))
    acr_all = _cell_field(info, "Allen_acronym")
    acr = [acr_all[i] for i in rr]
    names = list(lesioning_scores.keys())
    v1 = [np.asarray(lesioning_scores[k]["roi_avg_corr"]) for k in names]
    v2 = [np.asarray(lesioning_scores[k]["lesion_change"]) for k in names]
    v3 = []
    for k in names:
        reg = dict(zip(lesioning_scores[k]["region_names"], lesioning_scores[k]["region_values"],
                       strict=True))
        v3.append(np.array([reg[acr[i]] for i in range(v2[-1].size)]))
    lim1 = (min(float(np.min(v)) for v in v1), max(float(np.max(v)) for v in v1))
    m2 = max([0.0] + [float(np.max(np.abs(v))) for v in v2])
    m3 = max([0.0] + [float(np.max(np.abs(lesioning_scores[k]["region_values"])))
                      for k in names])
    title_id = f"{_field(sites[0], 'projID', '')} {_field(sites[0], 'Acode', '')}"
    figs = []
    specs = (
        (v1, lim1, "magma", "Avg Correlation", f"Average ROI Correlations - {title_id}"),
        (v2, (-m2, m2), "bwr", "\u0394 Avg Correlation",
         (f"Individual ROI Lesioning Effects - {title_id}\n"
          "(Negative=ROI removal decreases correlation)")),
        (v3, (-m3, m3), "bwr", "\u0394 Avg Correlation",
         (f"Allen Region Lesioning Effects - {title_id}\n"
          "(Negative=Region removal decreases correlation)")),
    )
    for vals, lim, cmap, cb_label, title in specs:
        fig, axes = plt.subplots(1, len(names), squeeze=False, figsize=(4 * len(names), 4))
        im = None
        for t, key in enumerate(names):
            im = _draw_roi_map(axes[0, t], vals[t], template=template, roi_info=info,
                               clim=lim, cmap=cmap, border_color=border_color,
                               show_borders=show_atlas_borders, background=0.0, marker_size=1)
            axes[0, t].set_title(key.replace("TP", "TP ", 1), fontsize=12)
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist()).set_label(cb_label)
        fig.suptitle(title, fontsize=14, fontweight="bold")
        figs.append(fig)
    return tuple(figs)


# --- WF_seedcorr --------------------------------------------------------------------------


class SeedCorrResult(NamedTuple):
    """Result of :func:`wf_seedcorr`."""

    seed_corr: np.ndarray  #: [nSeeds x nROI x nSites x nTimepointGroups]
    inds_roi_seed: np.ndarray  #: 0-based seed ROIs (positions in the concatenated ROIinfo)


def _corr_columns(a: np.ndarray, b: np.ndarray, pairwise: bool) -> np.ndarray:
    """MATLAB ``corr(a', b')`` for row variables ``a`` [p x T], ``b`` [q x T] (Pearson);
    ``pairwise``: each pair uses the time points where both are not NaN."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not pairwise:
        ac = a - a.mean(axis=1, keepdims=True)
        bc = b - b.mean(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (ac @ bc.T) / np.sqrt(np.outer((ac * ac).sum(1), (bc * bc).sum(1)))
    out = np.full((a.shape[0], b.shape[0]), np.nan)
    bn = ~np.isnan(b)
    with np.errstate(invalid="ignore", divide="ignore"):
        for i in range(a.shape[0]):
            m = bn & ~np.isnan(a[i])[None, :]
            n = m.sum(axis=1)
            x = np.where(m, a[i][None, :], 0.0)
            y = np.where(m, b, 0.0)
            xm = np.where(m, x - (x.sum(1) / n)[:, None], 0.0)
            ym = np.where(m, y - (y.sum(1) / n)[:, None], 0.0)
            r = (xm * ym).sum(1) / np.sqrt((xm * xm).sum(1) * (ym * ym).sum(1))
            r[n < 2] = np.nan
            out[i] = r
    return out


def _seed_rois(info, pattern: str, seed_rois) -> np.ndarray:
    ml, ap = _concat_field(info, "ML_index"), _concat_field(info, "AP_index")
    if pattern == "standard":
        seeds = []
        if ml.size and ap.size:
            for i in np.arange(ml.min(), 0 + 1e-9, 4):
                for j in np.arange(ap.min(), ap.max() + 1e-9, 5):
                    seeds.extend(np.flatnonzero((ml == i) & (ap == j)).tolist())
        return np.array(seeds, dtype=np.int64)
    if pattern == "allen":
        acr = _cell_field(info, "Allen_acronym")
        names = _cell_field(info, "name")
        seeds = []
        for a in sorted(set(acr)):
            tmp = [i for i in range(len(acr)) if acr[i] == a and names[i] == "left hemisphere"]
            if not tmp or max(tmp) >= ml.size:
                continue
            cx, cy = mround(ml[tmp].mean()), mround(ap[tmp].mean())
            seeds.extend(np.flatnonzero((ml == cx) & (ap == cy)).tolist())
        return np.sort(np.array(seeds, dtype=np.int64))
    if pattern == "manual":
        if seed_rois is None:
            raise ValueError("seed_pattern 'manual' needs seed_rois (0-based ROI indices)")
        s = np.asarray(seed_rois, dtype=np.int64).ravel()
        if np.any(s < 0) or np.any(s > N_GRID_ROIS - 1):
            raise ValueError("Valid seed ROIs are between 0 and 817 (0-based)")
        return s
    raise ValueError(f"seed_pattern must be 'standard', 'allen' or 'manual', got {pattern!r}")


def wf_seedcorr(
    proj_meta,
    *,
    comment=(),
    paradigm=(),
    logic: str = "and",
    tp_list=None,
    seed_pattern: str = "standard",
    seed_rois=None,
    median_div_on: bool = False,
    running_state: str = "all",
    thresh_run: float = 0.001,
) -> SeedCorrResult:
    """Correlation of seed ROIs with every ROI, per site and timepoint group.

    Computation of calliope ``21_Analysis/WF_seedcorr.m`` (plot: :func:`plot_wf_seedcorr`).
    Seeds (positions in ``[ROIinfo.ML_index]`` of ``proj_meta(1).rd(1,1)``):

    - ``'standard'``: every ROI at ``ML = min(ML):4:0`` and ``AP = min(AP):5:max(AP)``;
    - ``'allen'``: per Allen acronym (sorted), the ROI at the rounded mean ML/AP of its
      left-hemisphere ROIs (none when no ROI sits exactly there);
    - ``'manual'``: ``seed_rois`` (0-based, 0..817).

    Activity: stacks selected by ``comment``/``paradigm``/``logic`` as :func:`wf_corrmat`;
    with ``running_state`` 'running'/'stationary' only frames with ``Running_smoothed``
    above / not above ``thresh_run`` (all frames, with a warning, when the field is
    missing; the timepoint is skipped when no frame matches); optional median division.
    Correlations are Pearson; when the activity has NaN, pairwise-complete.

    Parameters
    ----------
    tp_list
        **0-based** timepoints: ints (one group each) or a list of sequences (groups whose
        activity is concatenated). Default all of the first site.
    seed_pattern, seed_rois, median_div_on, running_state, thresh_run, comment, paradigm,
    logic
        See above (MATLAB ``'seed_pattern'``, ``'RunningState'``, ``'ThreshRun'``, ...).

    Returns
    -------
    SeedCorrResult
        ``seed_corr`` float64 ``[nSeeds x nROI x nSites x nGroups]`` (a group without data is
        NaN; slices never written are 0, as MATLAB's array growth) and the 0-based seeds.

    Notes
    -----
    Differences from MATLAB: ``'allen'`` failed in MATLAB (it used the undefined variables
    ``ROIinfo``, ``inds_ROI``, ``inds_ML``); here it does what the code intended. ``'manual'``
    takes ``seed_rois`` instead of asking with ``input``. The unused ``'areas'`` parameter
    is not provided.
    """
    comment, paradigm = _str_list(comment), _str_list(paradigm)
    _check_logic(logic)
    _check_running_state(running_state)
    state = running_state.lower()
    sites = _sites(proj_meta)
    groups = _tp_groups(tp_list, _n_tp(sites[0]))
    seeds = _seed_rois(_roiinfo(sites), seed_pattern, seed_rois)
    entries: dict = {}
    first_empty = None
    for s, site in enumerate(sites):
        for k, grp in enumerate(groups):
            pieces = []
            for tp in grp:
                rd = _rd_row(site, tp)
                sessions = _sessions_to_use(rd, comment, paradigm, logic)
                fr = _frame_selection(site, tp, sessions, "all", 0.0)
                act = np.asarray(rd["act"], dtype=np.float64)
                sel = _matlab_logical(fr, act.shape[1])
                act = act[:, sel]
                if state != "all":
                    run = _field(rd, "Running_smoothed")
                    if _isempty(run):
                        logger.warning("Cannot filter by running state for site %d, tp %d: "
                                       "missing Running_smoothed; using all frames", s, tp)
                    else:
                        r = _vec(run).astype(np.float64)[_matlab_logical(fr, _vec(run).size)]
                        keep = r > thresh_run if state == "running" else ~(r > thresh_run)
                        if not keep.any():
                            logger.warning("No frames match %s state for site %d, tp %d; "
                                           "skipped", state, s, tp)
                            continue
                        act = act[:, keep]
                if median_div_on:
                    act = _median_divide_stacks(act, rd)
                pieces.append(act)
            if not pieces or sum(p.shape[1] for p in pieces) == 0:
                n_roi = np.asarray(_rd_row(site, grp[0])["act"]).shape[0]
                entries[(s, k)] = np.full((seeds.size, n_roi), np.nan)
                if first_empty is None:  # MATLAB initialised seed_corr with NaN here
                    first_empty = True
                continue
            avg = np.concatenate(pieces, axis=1)
            entries[(s, k)] = _corr_columns(avg[seeds, :], avg, bool(np.isnan(avg).any()))
            if first_empty is None:
                first_empty = False
    seed_corr = _assemble(entries, len(sites), len(groups), fill=np.nan if first_empty else 0.0)
    return SeedCorrResult(seed_corr, seeds)


def plot_wf_seedcorr(
    seed_corr,
    inds_roi_seed,
    template,
    *,
    tp_list=None,
    paradigm=(),
    running_state: str = "all",
    cmap: Any = "BrBG_r",
) -> list:
    """Seed-correlation maps of :func:`wf_seedcorr`, one figure per timepoint group.

    Every seed gets a panel with the site-averaged correlation (NaN ignored; remaining NaN
    set to 0) drawn by :func:`plot_roi_values` with colour limits ``[-1, 1]`` and the seed
    marked.

    Parameters
    ----------
    seed_corr, inds_roi_seed
        From :func:`wf_seedcorr`. ``seed_corr`` must have one column per template ROI.
    template
        :class:`GridTemplate` (``818_ROIs_grid_template.mat``).
    tp_list, paradigm, running_state
        For the titles (timepoints 0-based; shown 1-based).
    cmap
        Colormap. Default ``'BrBG_r'`` replaces MATLAB's ``flipud(lbmap(100,'BrownBlue'))``
        (third-party colormap, not ported); colours differ.

    Returns
    -------
    list of matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    sc = np.asarray(seed_corr, dtype=np.float64)
    seeds = np.asarray(inds_roi_seed, dtype=np.int64).ravel()
    figs = []
    n_tp = sc.shape[3]
    groups = _tp_groups(tp_list, n_tp)
    for t in range(n_tp):
        n = seeds.size
        cols = max(1, math.ceil(math.sqrt(n)))
        rows = max(1, math.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, squeeze=False, figsize=(2.5 * cols, 2.5 * rows))
        im = None
        for i in range(rows * cols):
            ax = axes.ravel()[i]
            if i >= n:
                ax.set_axis_off()
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                corrs = np.nanmean(sc[i, :, :, t], axis=1)
            corrs[seeds[i]] = -np.inf
            corrs[np.isnan(corrs)] = 0.0
            im = plot_roi_values(corrs, (-1, 1), template=template, ax=ax, cmap=cmap,
                                 seed=int(seeds[i]))
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist()).set_label("Correlation coefficient")
        tp_txt = " ".join(str(x + 1) for x in groups[t]) if t < len(groups) else str(t + 1)
        fig.suptitle(f"tp: {tp_txt}, paradigm: {' '.join(_str_list(paradigm))}, "
                     f"running state: {running_state}")
        figs.append(fig)
    return figs


# --- WF_plotCorrChangeInteractive ---------------------------------------------------------


class CorrChangePairs(NamedTuple):
    """Result of :func:`corr_change_pairs` (data of the left panel)."""

    naive: np.ndarray  #: correlation per pair and sample
    treated: np.ndarray
    roi_pairs: np.ndarray  #: int64 [nPoints x 2], 0-based ROI (row, column) of each point
    density: np.ndarray  #: 50 x 50 histogram count of each point's bin (0 outside [0, 1])


class RoiDensity(NamedTuple):
    """Result of :func:`corr_change_roi_density` (right panel after a click)."""

    image: np.ndarray  #: [H x W] count per ROI grid pixel (anterior up)
    roi_counts: np.ndarray  #: count per ROI (0-based ROI = position in ROIinfo)
    n_pairs: int
    n_active_rois: int
    mean_naive: float
    mean_treated: float


def _discretize(x, edges) -> np.ndarray:
    """MATLAB ``discretize``: 0-based bin, -1 outside (last bin right-closed)."""
    x = np.asarray(x, dtype=np.float64)
    b = np.searchsorted(edges, x, side="right") - 1
    b[x == edges[-1]] = edges.size - 2
    b[(x < edges[0]) | (x > edges[-1]) | np.isnan(x)] = -1
    return b


def corr_change_pairs(corrmat_naive, corrmat_treated, proj_meta) -> CorrChangePairs:
    """Naive against treated correlation of every ROI pair (WF_plotCorrChangeInteractive).

    Computation of the left panel of calliope ``21_Analysis/WF_plotCorrChangeInteractive.m``.
    For every sample ``k`` of ``corrmat_naive`` (third dimension), the nonzero entries of
    the strictly lower triangle (column-major) are paired with those of
    ``corrmat_treated``; with 824 ROIs the first two (bregma, lambda) are left out. Each
    point's ``density`` is the count of its bin in a 50 x 50 histogram over ``[0, 1]^2``.

    Parameters
    ----------
    corrmat_naive
        ``[nROI x nROI]`` or ``[nROI x nROI x nSamples]``.
    corrmat_treated
        ``[nROI x nROI]``.
    proj_meta
        Not used for the pairs (kept for symmetry with :func:`corr_change_roi_density`).

    Returns
    -------
    CorrChangePairs

    Notes
    -----
    Differences from MATLAB: MATLAB took the treated values and the ROI pairs from the full
    matrix but the naive values without the first two ROIs; with 824 ROIs the lengths
    differed and ``scatter`` failed. Here the same ROIs are left out of both and the pairs
    are the original (0-based) ROI numbers. For other sizes the result is MATLAB's. A zero
    correlation (dropped, as MATLAB's ``~= 0``) in one matrix only raises ``ValueError``.
    """
    del proj_meta
    nv = np.asarray(corrmat_naive, dtype=np.float64)
    nv = nv.reshape(nv.shape[0], nv.shape[1], -1)
    tr = np.asarray(corrmat_treated, dtype=np.float64)
    if tr.ndim != 2:
        raise ValueError("corrmat_treated must be 2-D")
    n = nv.shape[0]
    valid = np.r_[False, False, np.ones(n - 2, bool)] if n == 824 else np.ones(n, bool)
    vi = np.flatnonzero(valid)
    tv = _tril_vector(tr[np.ix_(vi, vi)])
    r, c = _tril_pairs(vi.size)
    pairs = np.column_stack([vi[r], vi[c]])
    naive, treated, all_pairs = [], [], []
    for k in range(nv.shape[2]):
        cv = _tril_vector(nv[:, :, k][np.ix_(vi, vi)])
        if cv.size != tv.size or cv.size != pairs.shape[0]:
            raise ValueError("naive and treated pair counts differ (zero correlations dropped)")
        naive.append(cv)
        treated.append(tv)
        all_pairs.append(pairs)
    x, y = np.concatenate(naive), np.concatenate(treated)
    edges = linspace(0.0, 1.0, 51)
    h = _histogram2(x, y, edges, edges)
    bx, by = _discretize(x, edges), _discretize(y, edges)
    ok = (bx >= 0) & (by >= 0)
    dens = np.zeros(x.size)
    dens[ok] = h[bx[ok], by[ok]]
    return CorrChangePairs(x, y, np.concatenate(all_pairs).astype(np.int64), dens)


def corr_change_roi_density(pairs: CorrChangePairs, proj_meta, x_click: float, y_click: float,
                            radius: float = 0.02) -> RoiDensity | None:
    """ROIs of the pairs within ``radius`` of a point of the naive/treated plane.

    The click callback of ``WF_plotCorrChangeInteractive.m``, with the click as an argument.
    Every ROI of a selected pair counts once; counts are drawn at the ROI's grid position
    (row ``H + min(AP) - AP``, column ``ML - min(ML)``, 0-based) of
    ``proj_meta(1).rd(1).ROIinfo``.

    Returns
    -------
    RoiDensity or None
        ``None`` when the point is outside ``[0, 1]^2`` or no pair is within ``radius``
        (MATLAB ignored the click).
    """
    if not (0 <= x_click <= 1 and 0 <= y_click <= 1):
        return None
    sel = (pairs.naive - x_click) ** 2 + (pairs.treated - y_click) ** 2 <= radius ** 2
    if not sel.any():
        return None
    rois = _sites(_roiinfo(_sites(proj_meta)))
    ml_all = _concat_field(rois, "ML_index")
    ap_all = _concat_field(rois, "AP_index")
    min_ml, min_ap = ml_all.min(), ap_all.min()
    W, H = int(ml_all.max() - min_ml + 1), int(ap_all.max() - min_ap + 1)
    chosen = np.concatenate([pairs.roi_pairs[sel, 0], pairs.roi_pairs[sel, 1]])
    counts = np.bincount(chosen, minlength=len(rois))[: max(len(rois), chosen.max() + 1)]
    img = np.zeros((H, W))
    for i in np.flatnonzero(counts):
        if i < len(rois) and not _isempty(_field(rois[i], "AP_index")):
            row = int(H + min_ap - float(_vec(rois[i]["AP_index"])[0])) - 1
            col = int(float(_vec(rois[i]["ML_index"])[0]) - min_ml)
            img[row, col] = counts[i]
    return RoiDensity(img, counts, int(sel.sum()), int((counts > 0).sum()),
                      float(pairs.naive[sel].mean()), float(pairs.treated[sel].mean()))


def plot_corr_change(pairs: CorrChangePairs, proj_meta, click=None, *, radius: float = 0.02):
    """Two-panel figure of WF_plotCorrChangeInteractive without interaction.

    Left: naive against treated correlation, coloured by density (magma), diagonal
    reference. Right: the ROI density map for ``click = (x, y)`` (with its circle on the
    left panel), or an empty map when no click is given.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Differences from MATLAB: the interactive click is the ``click`` argument; a new click
    means a new call.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sc = ax1.scatter(pairs.naive, pairs.treated, s=10, c=pairs.density, cmap="magma",
                     alpha=0.6)
    fig.colorbar(sc, ax=ax1)
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect("equal")
    ax1.set_xlabel("Naive", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Treated", fontsize=14, fontweight="bold")
    ax1.set_title("Click on a point to show ROI density map")
    dens = corr_change_roi_density(pairs, proj_meta, *click, radius=radius) if click else None
    if dens is None:
        rois = _roiinfo(_sites(proj_meta))
        ml, ap = _concat_field(rois, "ML_index"), _concat_field(rois, "AP_index")
        img = np.zeros((int(ap.max() - ap.min() + 1), int(ml.max() - ml.min() + 1)))
        im = ax2.imshow(img, cmap="hot")
        ax2.set_title("ROI Density Map (click left plot)")
    else:
        th = np.linspace(0, 2 * np.pi, 50)
        ax1.plot(click[0] + radius * np.cos(th), click[1] + radius * np.sin(th), "r-",
                 linewidth=2)
        im = ax2.imshow(dens.image, cmap="hot", vmin=0, vmax=max(dens.roi_counts.max(), 1))
        ax2.set_title(f"ROI Density Map\n{dens.n_pairs} pairs | {dens.n_active_rois} ROIs "
                      f"active\nMean: Naive={dens.mean_naive:.3f}, "
                      f"Treated={dens.mean_treated:.3f}")
    fig.colorbar(im, ax=ax2).set_label("ROI Pair Count")
    ax2.set_xticks([])
    ax2.set_yticks([])
    return fig


# --- WF_maskROIs (computational core) -----------------------------------------------------


def _index_columns(roi_indices) -> list[np.ndarray]:
    """ROI pixel indices (cell array, list, or [nPix x nROI] array) as a list of vectors."""
    if isinstance(roi_indices, np.ndarray) and roi_indices.dtype != object:
        a = roi_indices.reshape(roi_indices.shape[0], -1)
        return [a[:, j].astype(np.float64) for j in range(a.shape[1])]
    seq = roi_indices.ravel(order="F") if isinstance(roi_indices, np.ndarray) else roi_indices
    return [np.zeros(0) if x is None else np.asarray(x, dtype=np.float64).ravel(order="F")
            for x in seq]


def roi_mean_values(frame, roi_indices) -> np.ndarray:
    """Mean pixel value of every ROI in a frame (WF_maskROIs ``roi_mean_values``).

    Parameters
    ----------
    frame
        2-D image (e.g. ``adata.first_frame``).
    roi_indices
        Per ROI the **1-based column-major linear** pixel indices as stored by MATLAB (a
        cell array / list of vectors, or ``[nPix x nROI]``). Non-positive and NaN entries
        are ignored (MATLAB ``valid_indices > 0``).

    Returns
    -------
    numpy.ndarray
        float64 ``(nROI,)``; ``inf`` for a ROI without pixels (as MATLAB).
    """
    f = np.asarray(frame, dtype=np.float64).ravel(order="F")
    out = []
    for idx in _index_columns(roi_indices):
        ok = idx[idx > 0]
        out.append(float(f[ok.astype(np.int64) - 1].mean()) if ok.size else math.inf)
    return np.array(out)


def roi_border_indices(roi_indices, shape) -> list[np.ndarray]:
    """Perimeter pixels of every ROI (WF_maskROIs local ``findBorderIdx``).

    The perimeter is ``bwperim`` with 4-connectivity: ROI pixels with a 4-neighbour outside
    the ROI; pixels on the image edge count as perimeter. Returns **1-based column-major**
    linear indices per ROI (empty for a ROI with NaN or no indices).

    Notes
    -----
    Unverified: whether MATLAB's ``bwperim`` treats the image edge as background.
    """
    import scipy.ndimage

    out = []
    cross = scipy.ndimage.generate_binary_structure(2, 1)
    for idx in _index_columns(roi_indices):
        if idx.size == 0 or np.isnan(idx).any():
            out.append(np.zeros(0, dtype=np.int64))
            continue
        m = np.zeros(int(np.prod(shape)), bool)
        m[idx.astype(np.int64) - 1] = True
        m = m.reshape(shape, order="F")
        per = m & ~scipy.ndimage.binary_erosion(m, cross, border_value=0)
        out.append(np.flatnonzero(per.ravel(order="F")) + 1)
    return out


def roi_threshold_mask(roi_mean, threshold: float, roi_mask=None, has_border=None) -> np.ndarray:
    """ROI mask after setting the WF_maskROIs threshold slider (no manual marks).

    Reproduces the slider callback of ``WF_maskROIs.m`` applied to the starting mask:
    ROIs whose mean value is below ``threshold`` are marked (mask ``False``); ROIs marked
    in ``roi_mask`` (which count as threshold-marked) are unmarked when they are not below
    the threshold. ROIs without a border (no pixels) never change.

    Parameters
    ----------
    roi_mean
        From :func:`roi_mean_values`.
    threshold
        Slider value.
    roi_mask
        Starting mask, ``True`` = keep (default all ``True``).
    has_border
        Per ROI, whether it has a border (default all ``True``).

    Returns
    -------
    numpy.ndarray
        Boolean ``(nROI,)``: ``True`` = keep (MATLAB ``roiMask`` exported with key 'm').
    """
    mean = np.asarray(roi_mean, dtype=np.float64).ravel()
    keep = np.ones(mean.size, bool) if roi_mask is None else np.asarray(roi_mask, bool).ravel()
    border = np.ones(mean.size, bool) if has_border is None else np.asarray(has_border, bool)
    below = mean < threshold
    out = keep.copy()
    out[border & ~below & ~keep] = True
    out[border & below] = False
    return out


def plot_mask_rois(frame, roi_indices, roi_mask, *, show_black_borders: bool = True,
                   cmap: Any = "gray"):
    """Static view of WF_maskROIs: the frame with ROI borders (marked white, others black).

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    The interactive parts of ``WF_maskROIs.m`` (dragging over ROIs, slider, keys,
    ``assignin``) are not ported; use :func:`roi_threshold_mask` and edit the mask array.
    The frame is an argument (MATLAB loaded ``first_frame`` of the adata file by ExpID).
    """
    import matplotlib.pyplot as plt

    f = np.asarray(frame, dtype=np.float64)
    disp = f.copy()
    flat = disp.reshape(-1, order="F")
    keep = np.asarray(roi_mask, bool).ravel()
    for i, b in enumerate(roi_border_indices(roi_indices, f.shape)):
        if b.size == 0:
            continue
        if keep[i] if i < keep.size else True:
            if show_black_borders:
                flat[b - 1] = 0
        else:
            flat[b - 1] = 65535
    disp = flat.reshape(f.shape, order="F")
    fig, ax = plt.subplots()
    ax.imshow(disp, cmap=cmap, vmin=f.min(), vmax=f.max(), interpolation="nearest")
    ax.set_axis_off()
    return fig

