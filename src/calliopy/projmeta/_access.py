"""Reading data out of a project meta file (``proj_meta``): calliope ``24_ProjMeta/pm_expids.m``,
``pm_info.m``, ``add_uuid_to_proj_meta.m``, ``20_MetaData/act2mat.m``, ``trig2snps.m``, and
the computational cores of the cell browser and its plugins (``cell_browser.m``,
``cb_plugin_ACX.m``, ``cb_plugin_M1.m``, ``cb_plugin_VML.m``, ``cb_plugin_trav.m``).

A ``proj_meta`` is the struct array MATLAB saves (``(1, nSites)`` object array of
:class:`calliopy.io.MatStruct` as :func:`calliopy.io.load_mat` returns it, or a list of
structs); ``rd`` of a site is the ``(nLayers, nTimepoints)`` struct array.
"""

from __future__ import annotations

import logging
import uuid as _uuid
from typing import NamedTuple

import numpy as np

from .._matlab import linspace, prctile
from ..io import MatStruct
from ._struct import elements, empty, struct_array

__all__ = [
    "Act2MatResult",
    "PmInfoResult",
    "TraversalActivityResult",
    "Trig2SnpsResult",
    "act2mat",
    "activity_onsets",
    "add_uuid_to_proj_meta",
    "cell_browser_good_cells",
    "pm_expids",
    "pm_info",
    "traversal_activity",
    "trig2snps",
]

log = logging.getLogger(__name__)


def _sites(proj_meta) -> list:
    return elements(proj_meta)


def _rd(site) -> np.ndarray:
    rd = site["rd"]
    if isinstance(rd, np.ndarray) and rd.dtype == object:
        return rd if rd.ndim == 2 else rd.reshape(1, -1)
    if isinstance(rd, dict):
        return struct_array([rd])
    return np.empty((0, 0), dtype=object)


def _get(s, name):
    v = s.get(name) if isinstance(s, dict) else None
    return empty() if v is None else v


def pm_expids(proj_meta, site_ids=None, tps=None, flat: bool = False) -> np.ndarray:
    """ExpIDs of every site and timepoint (calliope ``24_ProjMeta/pm_expids.m``).

    Parameters
    ----------
    proj_meta : struct array
    site_ids : ignored
        Kept from MATLAB, which accepts it but always returns every site.
    tps : sequence of int, optional
        **0-based** timepoints to return; default all timepoints of each site. A
        timepoint ``t`` goes to column ``t`` (earlier columns stay NaN).
    flat : bool, default False
        Return the non-NaN ExpIDs as a (1, N) array, column-major (site varies fastest).

    Returns
    -------
    numpy.ndarray
        [nSites x nTimepoints] float array of ExpIDs, NaN where a site has fewer
        timepoints.
    """
    sites = _sites(proj_meta)
    cells: dict = {}
    n_cols = 0
    for s, site in enumerate(sites):
        cols = range(_rd(site).shape[1]) if tps is None else [int(t) for t in np.ravel(tps)]
        group = np.ravel(np.asarray(site["ExpGroup"], dtype=np.float64))
        for t in cols:
            cells[(s, t)] = float(group[t])
            n_cols = max(n_cols, t + 1)
        if len(cols) == 0:
            n_cols = max(n_cols, 1)
    out = np.full((len(sites), n_cols), np.nan)
    for (s, t), v in cells.items():
        out[s, t] = v
    if flat:
        v = out.ravel(order="F")
        return v[~np.isnan(v)].reshape(1, -1)
    return out


class PmInfoResult(NamedTuple):
    """Summary of a proj_meta (:func:`pm_info`)."""

    text: str
    animals: list
    sites_per_animal: np.ndarray
    cells_per_animal: np.ndarray
    perc_run: np.ndarray
    avg_speed: np.ndarray


def _mean(v) -> float:
    v = np.asarray(v, dtype=np.float64)
    return float(np.mean(v)) if v.size else float("nan")


def pm_info(proj_meta, run_threshold: float | None = None) -> PmInfoResult:
    """Summary of a proj_meta: animals, sites, cells, running (calliope ``24_ProjMeta/pm_info.m``).

    Per site: number of cells (rows of ``act`` summed over the layers of the first
    timepoint), fraction of samples with ``velM_smoothed > run_threshold`` and mean
    ``velM_smoothed`` of ``rd[0, 0]`` (-Inf when missing). Per animal and in total, the
    text MATLAB prints.

    Parameters
    ----------
    proj_meta : struct array
    run_threshold : float, optional
        Default 0.0005.

    Returns
    -------
    PmInfoResult
        ``text`` (also logged at INFO level), the sorted animal IDs, and per animal the
        number of sites and cells, and per site the running fraction and speed.

    Differences from MATLAB
    -----------------------
    The text is returned and logged; writing it to ``_metaData/pm_info/<fname>.txt`` is
    left to the caller. Without an ``Acode`` field the first line is written once (MATLAB
    reuses the format for its surplus argument and prints a second, garbled line).
    """
    if run_threshold is None:
        run_threshold = 0.0005
    sites_l = _sites(proj_meta)
    animals_site, sites, cells, perc_run, avg_speed = [], [], [], [], []
    for site in sites_l:
        animals_site.append(str(site["animal"]))
        sites.append(int(np.ravel(site["ExpGroup"])[0]))
        tmp = 0
        rd = _rd(site)
        try:
            for j in range(rd.shape[0]):
                a = np.asarray(_get(rd[j, 0], "act"))
                tmp += a.shape[0] if a.ndim else 0
        except (IndexError, TypeError, KeyError):
            pass
        try:
            v = np.asarray(rd[0, 0]["velM_smoothed"], dtype=np.float64)
            if v.size == 0:
                raise ValueError
            perc_run.append(float(np.mean(v > run_threshold)))
            avg_speed.append(float(np.mean(v)))
        except (IndexError, KeyError, ValueError, TypeError):
            perc_run.append(-np.inf)
            avg_speed.append(-np.inf)
        cells.append(tmp)
    animals, ic = np.unique(np.asarray(animals_site, dtype=object).astype(str), return_inverse=True)
    animals = list(animals)
    sites_arr, cells_arr = np.asarray(sites), np.asarray(cells)
    pr, sp = np.asarray(perc_run), np.asarray(avg_speed)
    sites_per_animal = np.bincount(ic, minlength=len(animals)) if len(animals) else np.zeros(0, int)
    cells_per_animal = np.array([cells_arr[ic == i].sum() for i in range(len(animals))])
    sep = "-" * 101
    lines = [sep]
    has_acode = bool(sites_l) and all("Acode" in s for s in sites_l)
    if has_acode:
        acodes = sorted({str(s["Acode"]) for s in sites_l})
        lines.append(f"{len(animals)} animals in this proj_meta file (acodes: {', '.join(acodes)})")
    else:
        lines.append(f"{len(animals)} animals in this proj_meta file ")
    lines.append(sep)
    explength = (len(str(int(sites_arr.max()))) + 1) * int(sites_per_animal.max()) if sites_l else 0
    for i, a in enumerate(animals):
        site_str = "".join(f"{s} " for s in sites_arr[ic == i]).ljust(explength)
        lines.append(f"{sites_per_animal[i]} sites \t - {cells_per_animal[i]} cells \t - {a} \t sites: {site_str}"
                     f"\t%run: {_mean(pr[ic == i]) * 100:2.1f}\tspeed: {_mean(sp[ic == i]):0.3f} ")
    lines.append(sep)
    tps = _mean([_rd(s).shape[1] for s in sites_l])
    lines.append(f"{int(sites_per_animal.sum())} sites \t - {int(cells_arr.sum())} cells \t - {len(animals)} "
                 f"animals \t\t {len(sites_l)} sites, ~{tps:2.1f} tps {' ' * max(1, explength)} "
                 f"%run: {_mean(pr * 100):2.1f}\tspeed: {_mean(sp):0.3f} ")
    lines.append(sep)
    if sites_l and "uuid" in sites_l[0]:
        lines.append(str(sites_l[0]["uuid"]))
        lines.append(sep)
    text = "\n".join(lines) + "\n"
    log.info("\n%s", text)
    return PmInfoResult(text, animals, sites_per_animal, cells_per_animal, pr, sp)


def add_uuid_to_proj_meta(proj_meta, uuid: str | None = None) -> np.ndarray:
    """Give every site of a proj_meta the same UUID (``24_ProjMeta/add_uuid_to_proj_meta.m``).

    Parameters
    ----------
    proj_meta : struct array
    uuid : str, optional
        Default: a new random UUID (MATLAB ``java.util.UUID.randomUUID``).

    Returns
    -------
    numpy.ndarray
        The proj_meta struct array with the field ``uuid``; unchanged (and logged) when it
        already has one.

    Differences from MATLAB
    -----------------------
    Takes and returns the struct array; loading and saving the file is left to the caller.
    """
    sites = _sites(proj_meta)
    shape = proj_meta.shape if isinstance(proj_meta, np.ndarray) else (1, len(sites))
    if any("uuid" in s for s in sites):
        log.info("Meta file already has a uuid")
        return proj_meta if isinstance(proj_meta, np.ndarray) else struct_array(sites, shape)
    u = str(_uuid.uuid4()) if uuid is None else str(uuid)
    out = []
    for s in sites:
        t = MatStruct(s)
        t["uuid"] = u
        out.append(t)
    return struct_array(out, shape)


class Act2MatResult(NamedTuple):
    """Outputs of :func:`act2mat` (MATLAB ``[actmat, indROIinlayer, indlayer, spksmat, fissamat]``)."""

    actmat: np.ndarray
    ind_roi_in_layer: np.ndarray
    ind_layer: np.ndarray
    spksmat: np.ndarray
    fissamat: np.ndarray


def act2mat(proj_meta, site_id: int, tp: int) -> Act2MatResult:
    """Activity of all layers of one site and timepoint as one matrix (``20_MetaData/act2mat.m``).

    Parameters
    ----------
    proj_meta : struct array
    site_id, tp : int
        **0-based** site and timepoint.

    Returns
    -------
    Act2MatResult
        ``actmat`` [nCells x nFrames] (layers stacked in order); ``ind_roi_in_layer`` and
        ``ind_layer``: **0-based** ROI index within its layer and layer of every row;
        ``spksmat`` and ``fissamat``: zeros of the same shape (kept from MATLAB, which
        allocates them and never fills them). Empty (0, 0) arrays, with a logged error,
        when the site has no activity.

    Differences from MATLAB
    -----------------------
    The index outputs are 0-based (MATLAB 1-based). No activity gives empty outputs
    (MATLAB prints an error and leaves the outputs undefined).
    """
    site = _sites(proj_meta)[int(site_id)]
    rd = _rd(site)
    acts = [np.asarray(_get(rd[j, int(tp)], "act")) for j in range(rd.shape[0])]
    ncells = [a.shape[0] if a.ndim == 2 else 0 for a in acts]
    nframes = None
    for a, n in zip(acts, ncells, strict=True):
        if n > 0:
            nframes = a.shape[1]
    if nframes is None:
        log.error("ERROR - no activity found in site: %d tp: %d", site_id, tp)
        e = empty()
        return Act2MatResult(e, np.zeros(0, int), np.zeros(0, int), e, e)
    rows = [a for a, n in zip(acts, ncells, strict=True) if n > 0]
    actmat = np.vstack(rows).astype(np.float64)
    ind_roi = np.concatenate([np.arange(n) for n in ncells]).astype(np.int64)
    ind_layer = np.concatenate([np.full(n, j) for j, n in enumerate(ncells)]).astype(np.int64)
    return Act2MatResult(actmat, ind_roi, ind_layer, np.zeros(actmat.shape), np.zeros(actmat.shape))


class Trig2SnpsResult(NamedTuple):
    """Outputs of :func:`trig2snps` (MATLAB ``[snps, snps_array]``)."""

    snps: np.ndarray
    snps_array: list


def trig2snps(proj_meta, site_id: int, tp: int, trigs, win=(-100, 100), var_array=None) -> Trig2SnpsResult:
    """Activity snippets around triggers for every cell (``20_MetaData/trig2snps.m``).

    Parameters
    ----------
    proj_meta : struct array
    site_id, tp : int
        **0-based** site and timepoint.
    trigs : array of int
        **0-based** trigger frames. Triggers closer than ``max(abs(win)) + 1`` frames to
        the start, the end or a stack boundary (``cumsum(rd[0, tp].nbr_frames)``) are
        dropped.
    win : (int, int), default (-100, 100)
        Window in frames relative to the trigger (``win[0] <= 0 <= win[1]``).
    var_array : sequence of str, optional
        Other ``rd[0, tp]`` fields to cut the same way (a vector the length of
        ``frame_times``); a field that cannot be cut gives an empty array.

    Returns
    -------
    Trig2SnpsResult
        ``snps`` [nCells x nWindow x nTriggers]; ``snps_array``: per field in
        ``var_array`` an [nWindow x nTriggers] array (MATLAB linear indexing).
        ``snps`` is an empty (0, 0) array when the site has no activity.
    """
    act = act2mat(proj_meta, site_id, tp).actmat
    if act.size == 0:
        log.info("No actvity? Probably dead.")
        return Trig2SnpsResult(empty(), [])
    t1 = np.ravel(np.asarray(trigs, dtype=np.int64)) + 1  # MATLAB 1-based frames
    w0, w1 = int(win[0]), int(win[1])
    rd0 = _rd(_sites(proj_meta)[int(site_id)])[0, int(tp)]
    edges = np.concatenate(([0], np.cumsum(np.ravel(np.asarray(rd0["nbr_frames"], dtype=np.float64)))))
    lim = max(abs(w0), abs(w1)) + 1
    near = (np.abs(edges[:, None] - t1[None, :]) < lim).sum(axis=0) > 0
    t1 = t1[~near]
    length = abs(w0) + abs(w1) + 1
    offs = np.arange(w0, w1 + 1)
    idx = offs[:, None] + t1[None, :]  # 1-based, [nWindow x nTriggers]
    flat = idx.ravel(order="F") - 1
    snps = act[:, flat].reshape(act.shape[0], length, t1.size, order="F")
    out = []
    if var_array is not None:
        ft = np.asarray(_get(rd0, "frame_times"))
        n_ft = ft.shape[1] if ft.ndim == 2 else ft.size
        for name in var_array:
            cur = np.asarray(rd0[name])
            if cur.ndim == 2 and cur.shape[0] == n_ft:
                cur = cur.T
            try:
                v = cur.ravel(order="F")
                if flat.size and (flat.max() >= v.size or flat.min() < 0):
                    raise IndexError
                out.append(v[idx - 1])
            except IndexError:
                out.append(empty())
    return Trig2SnpsResult(snps, out)


def activity_onsets(act, threshold: float, win: int) -> np.ndarray:
    """Activity onsets as the cell browser plugins find them (``cb_plugin_ACX/M1/VML.m``).

    ``act > threshold`` is the active state. An onset is a sample that is not active and
    followed by an active one; onsets within ``win`` samples of either end, and onsets
    with any active sample in the ``win/2`` samples before them, are dropped.

    Parameters
    ----------
    act : array, shape (N,)
        Activity trace (the plugins use ``act`` with ``threshold = 1 + 2.5*std(act,
        ddof=1)`` (ACX), ``1.5`` (VML), or the 20-sample ``smooth2`` of ``act`` (M1)).
    threshold : float
    win : int
        Even window length in samples (ACX 100, M1 150, VML 200).

    Returns
    -------
    numpy.ndarray of int64
        **0-based** indices of the last inactive sample before each onset (MATLAB's
        1-based onset minus one).

    Differences from MATLAB
    -----------------------
    No onsets give an empty array (MATLAB fails on its one-element mask); an odd ``win``
    raises ``ValueError`` (MATLAB fails on the fractional range).
    """
    if int(win) % 2:
        raise ValueError("win must be even")
    win = int(win)
    tact = np.ravel(np.asarray(act) > threshold)
    ons = np.flatnonzero(np.diff(tact.astype(np.int8)) == 1) + 1  # 1-based
    ons = ons[ons >= win + 1]
    ons = ons[ons <= tact.size - win - 1]
    keep = [not tact[o - win // 2 - 1 : o].any() for o in ons]
    return (ons[np.asarray(keep, dtype=bool)] - 1).astype(np.int64) if ons.size else np.zeros(0, np.int64)


class TraversalActivityResult(NamedTuple):
    """Outputs of :func:`traversal_activity`."""

    bins: np.ndarray
    grats: np.ndarray
    binned_xpos: np.ndarray
    bin_act_speed: np.ndarray
    bin_std_speed: np.ndarray


def _nanmean_or_nan(v) -> float:
    v = np.asarray(v, dtype=np.float64)
    return float(np.mean(v)) if v.size else float("nan")


def traversal_activity(xpos, act, grat, velocity) -> TraversalActivityResult:
    """Activity by corridor position per traversal and by running speed (``cb_plugin_trav.m``).

    Parameters
    ----------
    xpos : array, shape (N,)
        Corridor position (``VRx``, 0 to 5.02); a drop by more than 0.2 starts a traversal.
    act : array, shape (N,)
        Activity of one cell.
    grat : array, shape (N,)
        Grating signal (``GratFlash``).
    velocity : array, shape (N,)
        Running speed (MATLAB uses ``-velM_smoothed``).

    Returns
    -------
    TraversalActivityResult
        ``bins`` [nTraversals x 100] mean activity per position bin (NaN when empty),
        ``grats`` the same for the grating signal, ``binned_xpos`` mean position per bin,
        ``bin_act_speed``/``bin_std_speed`` mean and standard error of the activity in
        14 speed bins between 0 and 0.014.
    """
    xpos, act, grat, velocity = (np.ravel(np.asarray(v, dtype=np.float64)) for v in (xpos, act, grat, velocity))
    edges = linspace(0, 5.02, 101)
    binned_xpos = np.array([_nanmean_or_nan(xpos[(xpos > edges[i]) & (xpos <= edges[i + 1])])
                            for i in range(100)])
    starts = np.concatenate((np.flatnonzero(np.diff(xpos) < -0.2) + 1, [xpos.size]))  # 1-based
    n_trav = starts.size - 1
    bins = np.full((n_trav, 100), np.nan)
    grats = np.full((n_trav, 100), np.nan)
    for n in range(n_trav):
        sl = slice(int(starts[n]), int(starts[n + 1]))
        pos, a, g = xpos[sl], act[sl], grat[sl]
        for j in range(100):
            sel = (pos > edges[j]) & (pos < edges[j + 1])
            bins[n, j] = _nanmean_or_nan(a[sel])
            grats[n, j] = _nanmean_or_nan(g[sel])
    sedges = linspace(0, 0.014, 15)
    m, s = [], []
    for j in range(14):
        sel = (velocity > sedges[j]) & (velocity <= sedges[j + 1])
        vals = act[sel]
        m.append(_nanmean_or_nan(vals))
        s.append(float(np.std(vals, ddof=1)) / np.sqrt(vals.size) if vals.size > 1 else
                 (0.0 if vals.size == 1 else float("nan")))
    return TraversalActivityResult(bins, grats, binned_xpos, np.asarray(m), np.asarray(s))


def cell_browser_good_cells(proj_meta) -> tuple[dict, float]:
    """Cells with the largest dynamic range, as ``cell_browser.m`` selects them.

    The range (max - min) of every cell's ``act`` in every site, timepoint and layer;
    cells above the 90th percentile (MATLAB ``prctile``) of all ranges are good.

    Returns
    -------
    (dict, float)
        ``{(site, tp, layer): bool array}`` with **0-based** keys, and the threshold.
    """
    ranges: dict = {}
    for s, site in enumerate(_sites(proj_meta)):
        rd = _rd(site)
        for t in range(rd.shape[1]):
            for z in range(rd.shape[0]):
                a = np.asarray(_get(rd[z, t], "act"), dtype=np.float64)
                if a.ndim == 2 and a.shape[0] and a.shape[1]:
                    ranges[(s, t, z)] = a.max(axis=1) - a.min(axis=1)
    allr = np.concatenate(list(ranges.values())) if ranges else np.zeros(0)
    if allr.size == 0:
        return {}, float("nan")
    p = float(prctile(allr, 90))
    return {k: v > p for k, v in ranges.items()}, p

