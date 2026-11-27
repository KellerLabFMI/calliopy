"""SpineQuest: spiny regions, DeepD3 hand-over and spine tracking across timepoints.

Computational cores of calliope ``04_Calliope/kalliope_callback_SpineQuest_select_regions.m``,
``kalliope_callback_SpineQuest_detect_spines.m`` and
``kalliope_callback_SpineQuest_track_spines.m``. The interactive parts (marking regions in
``view_stack``, the ``StackAligner`` window, the click-to-correct tracking window) become
arguments: marks, offsets and manual overrides. Running DeepD3 (an external Python
program) is the caller's.

Conventions (values as MATLAB stores them in adata):

- ``spiny_regions``: per region a ``3 x 3`` array of **1-based** ``(row, column, z)``
  subscripts into ``mean_data``: row 0 top-left corner (highest z layer), row 1 bottom-right
  corner (same layer), row 2 a point in the lowest layer.
- ``spines[region]``: list of structs whose ``indices`` are **1-based column-major** linear
  indices into that region's crop of ``mean_data``.
- Spine positions are ``(x, y, z) = (column, row, layer)`` centroids, **1-based**, in the
  region crop.

Arguments that select something (spine numbers, master rows, layers) are **0-based**.

None of these functions has been compared with MATLAB output.
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import tifffile

from ..io import MatStruct
from ._activity import _as_items
from ._builtins import matchpairs

__all__ = [
    "SPINE_SCALE",
    "MatchResult",
    "SpineTracking",
    "crop_spiny_region",
    "match_timepoint_to_master",
    "preceding_expid_in_site",
    "propagate_spiny_regions",
    "seed_label_map",
    "spine_positions",
    "spine_tracking_export",
    "spines_from_roi_map",
    "spiny_regions_from_marks",
    "track_spines",
    "write_label_tiff",
    "write_spine_region_tiff",
]

SPINE_SCALE = (0.27, 0.54, 2.0)
"""MATLAB default ``[x_size, y_size, z_size]`` in micrometres per pixel / step."""


def spiny_regions_from_marks(marks: Any, shape: Sequence[int]) -> list:
    """Spiny regions from the points marked in ``view_stack`` (``SpineQuest_select_regions``).

    Parameters
    ----------
    marks
        MATLAB linear indices into ``mean_data`` (**1-based**, column-major), three per
        region: top-left corner and bottom-right corner in the highest layer, then a point in
        the lowest layer.
    shape
        ``size(mean_data)``, ``(rows, cols, layers)``.

    Returns
    -------
    list of numpy.ndarray
        One float64 ``3 x 3`` array of 1-based ``(row, column, z)`` per region.

    Raises
    ------
    ValueError
        No marks, not a multiple of 3, or a region failing MATLAB's checks (corner 1 above
        and left of corner 2, both in the same layer, point 3 in a deeper layer).
    """
    m = np.asarray(marks, dtype=np.int64).ravel()
    msg = ("You must select markers in multiples of 3 - 1: top left corner of region in highest "
           "z-layer, 2: bottom right corner in highest z-layer, and 3: a point between the two in "
           "the lowest z-layer")
    if m.size == 0 or m.size % 3:
        raise ValueError(msg)
    sub = np.stack(np.unravel_index(m - 1, tuple(int(s) for s in shape[:3]), order="F"), axis=1) + 1
    regions = [sub[i : i + 3].astype(np.float64) for i in range(0, m.size, 3)]
    for r in regions:
        ok = r[0, 0] < r[1, 0] and r[0, 1] < r[1, 1] and r[0, 2] == r[1, 2] and r[2, 2] > r[0, 2]
        if not ok:
            raise ValueError(msg)
    return regions


def crop_spiny_region(mean_data: Any, region: Any) -> np.ndarray:
    """The sub-volume ``mean_data(r1:r2, c1:c2, z1:z3)`` of a spiny region (1-based, inclusive)."""
    r = np.asarray(region, dtype=np.int64)
    md = np.asarray(mean_data)
    return md[r[0, 0] - 1 : r[1, 0], r[0, 1] - 1 : r[1, 1], r[0, 2] - 1 : r[2, 2]]


def propagate_spiny_regions(orig_regions: Sequence[Any], final_offsets: Sequence[Any]) -> list:
    """Regions of a later timepoint from the site origin's regions and the aligned offsets.

    ``final_offsets`` are, per region, the ``[x, y, z]`` (1-based column, row, layer) top-left
    corner returned by MATLAB's interactive ``StackAligner``. As MATLAB: the new corner 1 is
    ``(y, x, z)``; corner 2 keeps the origin's region size in that layer; point 3 keeps the
    origin's row and column and the origin's depth below corner 2.
    """
    out = []
    for orig, off in zip(orig_regions, final_offsets, strict=True):
        o = np.asarray(orig, dtype=np.float64)
        x, y, z = (float(v) for v in np.asarray(off).ravel()[:3])
        r = np.zeros((3, 3))
        r[0] = [y, x, z]
        r[1] = [y + (o[1, 0] - o[0, 0]), x + (o[1, 1] - o[0, 1]), z]
        r[2] = [o[2, 0], o[2, 1], z + (o[2, 2] - o[1, 2])]
        out.append(r)
    return out


def write_spine_region_tiff(path: Any, region: Any) -> None:
    """Write a region crop as a multi-page float64 TIFF, one page per layer (DeepD3 input)."""
    vol = np.asarray(region, dtype=np.float64)
    tifffile.imwrite(path, np.moveaxis(vol, 2, 0), photometric="minisblack")


def seed_label_map(prev_spines: Any, shape: Sequence[int]) -> np.ndarray | None:
    """Label volume of the preceding timepoint's spines for one region (DeepD3 seed).

    Spine ``k`` (0-based) is drawn with label ``k + 1`` at its ``indices``. Returns None
    when an index lies outside ``shape`` (the crops differ) or no label is set, as MATLAB
    then writes no seed file.
    """
    n = int(np.prod([int(s) for s in shape]))
    lab = np.zeros(n)
    for k, s in enumerate(_as_items(prev_spines)):
        ix = np.asarray(s["indices"], dtype=np.int64).ravel()
        if ix.size and ix.max() > n:
            return None
        lab[ix - 1] = k + 1
    if not lab.any():
        return None
    return lab.reshape(tuple(int(s) for s in shape), order="F")


def write_label_tiff(path: Any, label_map: Any) -> None:
    """Write a label volume as a multi-page uint16 TIFF (``write_label_tiff``)."""
    from .._matlab import matlab_cast

    tifffile.imwrite(path, np.moveaxis(matlab_cast(np.asarray(label_map), np.uint16), 2, 0),
                     photometric="minisblack")


def spines_from_roi_map(combined: Any) -> tuple[list, np.ndarray]:
    """Spines and dendrite from DeepD3's combined ROI map (``SpineQuest_detect_spines``).

    Parameters
    ----------
    combined
        ``[rows x cols x 2P]`` volume: the spine label map (first ``P`` layers) followed by
        the dendrite mask (MATLAB ``tiffreadVolume`` order). A TIFF read with tifffile is
        ``(2P, rows, cols)``: pass ``np.moveaxis(vol, 0, 2)``.

    Returns
    -------
    (list of MatStruct, numpy.ndarray)
        One struct per non-zero label in ascending order, ``indices`` float64 column of
        1-based column-major indices into the ``[rows x cols x P]`` label map; and the
        dendrite indices (same convention).
    """
    vol = np.asarray(combined)
    p = vol.shape[2] // 2
    stack = vol[:, :, :p]
    dend = vol[:, :, p : 2 * p]
    flat = stack.ravel(order="F")
    spines = []
    for lab in np.unique(flat):
        if lab == 0:
            continue
        spines.append(MatStruct(indices=(np.flatnonzero(flat == lab) + 1).astype(np.float64)[:, None]))
    dendrite = (np.flatnonzero(dend.ravel(order="F") > 0) + 1).astype(np.float64)[:, None]
    return spines, dendrite


def preceding_expid_in_site(expid: int, expids: Sequence[int], siteids: Sequence[int],
                            stackdates: Sequence[Any]) -> int | None:
    """The ExpID just before ``expid`` in its site, by stack date (``get_preceding_expid_in_site``).

    ``expids``, ``siteids`` and ``stackdates`` are the ExpLog columns (one row per stack;
    dates as ``datetime`` or ISO strings). Returns None for the first or only timepoint.
    """
    e = np.asarray(expids)
    s = np.asarray(siteids)
    rows = np.flatnonzero(e == expid)
    if rows.size == 0:
        return None
    site = s[rows[0]]
    cur = np.unique(e[s == site])
    if cur.size < 2:
        return None
    dates = []
    for x in cur:
        d = stackdates[int(np.flatnonzero(e == x)[0])]
        dates.append(d if isinstance(d, datetime.datetime) else datetime.datetime.fromisoformat(str(d)))
    order = sorted(range(cur.size), key=lambda i: dates[i])
    srt = cur[order]
    pos = int(np.flatnonzero(srt == expid)[0])
    return None if pos == 0 else int(srt[pos - 1])


def spine_positions(spines: Any, region_size: Sequence[int]) -> np.ndarray:
    """``[nSpines x 3]`` centroids ``(x, y, z)`` = (column, row, layer), 1-based, in the crop."""
    items = _as_items(spines)
    out = np.full((len(items), 3), np.nan)
    shp = tuple(int(v) for v in region_size)
    for k, s in enumerate(items):
        ix = np.asarray(s["indices"], dtype=np.int64).ravel()
        if ix.size:
            r, c, z = np.unravel_index(ix - 1, shp, order="F")
            out[k] = [c.mean() + 1, r.mean() + 1, z.mean() + 1]
    return out


class MatchResult(NamedTuple):
    """Result of :func:`match_timepoint_to_master`."""

    reg_presence: np.ndarray
    reg_positions: np.ndarray
    active_rows: np.ndarray
    """0-based master rows eligible at the next timepoint."""
    reg_first_positions: np.ndarray
    assignment: np.ndarray
    """Per spine of this timepoint the 0-based master row it was assigned to."""


def match_timepoint_to_master(reg_presence: Any, reg_positions: Any, active_rows: Any,
                              cur_positions: Any, t: int, dist_threshold_um: float = 1,
                              scale: Sequence[float] = SPINE_SCALE,
                              overrides: Mapping[int, int | None] | None = None,
                              reg_first_positions: Any = None) -> MatchResult:
    """Match one timepoint's spines to the running master list (``matchTimepointToMaster``).

    Global minimum-cost assignment (:func:`~._builtins.matchpairs`) on the physical 3-D
    distance ``|(p_master - p_cur) * scale|`` with ``dist_threshold_um`` as cost of leaving a
    spine unmatched; then manual ``overrides`` (spine -> master row, ``None`` = new); when two
    spines claim one row, the later spine (higher number) keeps it and the earlier becomes
    new. Unmatched spines are appended as new master rows (present from column ``t`` on);
    only the rows matched or created here stay eligible (a lost spine never comes back).

    Parameters
    ----------
    reg_presence
        ``[nMaster x nTimepoints]`` presence (1/0).
    reg_positions
        ``[nMaster x 3]`` most recent positions.
    active_rows
        **0-based** master rows eligible for matching.
    cur_positions
        ``[nSpinesT x 3]`` positions at this timepoint.
    t
        **0-based** column of this timepoint.
    dist_threshold_um, scale
        As MATLAB (defaults 1 um and :data:`SPINE_SCALE`).
    overrides
        ``{spine (0-based): master row (0-based) or None}``.
    reg_first_positions
        ``[nMaster x 3]`` positions at creation.

    Notes
    -----
    Ties between equally good assignments may be broken differently from MATLAB's
    ``matchpairs``.
    """
    pres = np.array(reg_presence, dtype=np.float64, ndmin=2)
    pos = np.array(reg_positions, dtype=np.float64).reshape(-1, 3)
    first = (np.array(reg_first_positions, dtype=np.float64).reshape(-1, 3)
             if reg_first_positions is not None else np.zeros((0, 3)))
    act = np.asarray(active_rows, dtype=np.int64).ravel()
    cur = np.asarray(cur_positions, dtype=np.float64).reshape(-1, 3)
    sc = np.asarray(scale, dtype=np.float64)
    ns = cur.shape[0]
    assign = np.full(ns, -1, dtype=np.int64)
    if act.size and ns:
        d = (pos[act][:, None, :] - cur[None, :, :]) * sc
        cost = np.sqrt((d * d).sum(axis=2))
        for a, b in matchpairs(cost, dist_threshold_um):
            assign[b] = act[a]
    for k, v in (overrides or {}).items():
        assign[int(k)] = -1 if v is None else int(v)
    claimed: dict[int, int] = {}
    for i in range(ns):
        row = int(assign[i])
        if row < 0:
            continue
        if row in claimed:
            assign[claimed[row]] = -1
        claimed[row] = i
    ntp = max(pres.shape[1] if pres.size else 0, t + 1)
    if pres.shape[1] < ntp:
        pres = np.pad(pres.reshape(pres.shape[0], -1), ((0, 0), (0, ntp - pres.shape[1])))
    new_active = []
    for i in range(ns):
        row = int(assign[i])
        if row < 0:
            row = pos.shape[0]
            pos = np.vstack([pos, cur[i]])
            first = np.vstack([first, cur[i]])
            pres = np.vstack([pres, np.zeros((1, pres.shape[1]))])
            pres[row, t] = 1
            assign[i] = row
        else:
            pres[row, t] = 1
            pos[row] = cur[i]
        new_active.append(row)
    return MatchResult(pres, pos, np.asarray(new_active, dtype=np.int64), first, assign)


class SpineTracking(NamedTuple):
    """Result of :func:`track_spines` (MATLAB ``[presence, positions, regionIdx]``)."""

    presence: np.ndarray
    """``[nSpines x nTimepoints]`` 1/0."""
    positions: np.ndarray
    """``[nSpines x 3]`` most recent ``(x, y, z)`` positions (1-based, crop coordinates)."""
    region_idx: np.ndarray
    """``[nSpines x 1]`` **0-based** region of every row (MATLAB ``regionIdx`` is 1-based)."""
    first_positions: np.ndarray
    """``[nSpines x 3]`` position at the first timepoint each spine was present."""


def _region_size(region) -> tuple[int, int, int]:
    b = np.asarray(region, dtype=np.int64)
    return int(b[1, 0] - b[0, 0] + 1), int(b[1, 1] - b[0, 1] + 1), int(b[2, 2] - b[0, 2] + 1)


def track_spines(spines: Sequence[Sequence[Any]], spiny_regions: Sequence[Any], *,
                 dist_threshold_um: float = 1, scale: Sequence[float] = SPINE_SCALE,
                 overrides: Mapping[tuple[int, int], Mapping[int, int | None]] | None = None) -> SpineTracking:
    """Track spines across the timepoints of a site (``SpineQuest_track_spines``).

    Parameters
    ----------
    spines
        Per timepoint (chronological, only timepoints with detected spines), the ``spines``
        cell of its adata: per region a list of structs with ``indices``.
    spiny_regions
        The ``spiny_regions`` of the **first** timepoint; MATLAB uses its region sizes to
        convert the indices of every timepoint to positions.
    dist_threshold_um, scale
        Matching parameters (:func:`match_timepoint_to_master`).
    overrides
        Manual corrections made in MATLAB's window: ``{(region, t): {spine: row or None}}``
        with 0-based region, timepoint, spine and master row.

    Returns
    -------
    SpineTracking

    Notes
    -----
    Differences from MATLAB: the interactive review window is replaced by ``overrides``;
    ExpLog lookup, adata loading and :func:`spine_tracking_export`'s save are the caller's.
    """
    ntp = len(spines)
    if ntp < 2:
        raise ValueError(f"need at least 2 timepoints with spines, got {ntp}")
    pres_all, pos_all, reg_all, first_all = [], [], [], []
    for r in range(len(spines[0])):
        size = _region_size(spiny_regions[r])
        pos = spine_positions(spines[0][r], size)
        pres = np.zeros((pos.shape[0], ntp))
        pres[:, 0] = 1
        active = np.arange(pos.shape[0])
        first = pos.copy()
        for t in range(1, ntp):
            cur = spine_positions(spines[t][r], size)
            ov = (overrides or {}).get((r, t))
            m = match_timepoint_to_master(pres, pos, active, cur, t, dist_threshold_um, scale, ov, first)
            pres, pos, active, first = m.reg_presence, m.reg_positions, m.active_rows, m.reg_first_positions
        pres_all.append(pres)
        pos_all.append(pos)
        first_all.append(first)
        reg_all.append(np.full((pres.shape[0], 1), r, dtype=np.int64))
    return SpineTracking(np.vstack(pres_all), np.vstack(pos_all), np.vstack(reg_all),
                         np.vstack(first_all))


def spine_tracking_export(tracking: SpineTracking, spines: Sequence[Sequence[Any]],
                          spiny_regions: Sequence[Sequence[Any]]) -> list:
    """The ``spine_tracking`` variable MATLAB saves on the site origin's adata.

    Per region a list with one struct per tracked spine: ``indices`` of the spine at its
    first present timepoint (found by matching its first position exactly, as MATLAB;
    empty when none matches) and ``presence`` (``1 x nTimepoints``).

    Parameters
    ----------
    spiny_regions
        Per timepoint its ``spiny_regions`` (MATLAB uses the first present timepoint's
        region size here).
    """
    nreg = int(tracking.region_idx.max()) + 1 if tracking.region_idx.size else 0
    out = []
    for rr in range(nreg):
        rows = np.flatnonzero(tracking.region_idx[:, 0] == rr)
        lst = []
        for row in rows:
            pv = tracking.presence[row : row + 1, :]
            ft = int(np.flatnonzero(pv[0] == 1)[0])
            size = _region_size(spiny_regions[ft][rr])
            cand = spine_positions(spines[ft][rr], size)
            found = np.zeros((0, 1))
            for k, s in enumerate(_as_items(spines[ft][rr])):
                if np.array_equal(cand[k], tracking.first_positions[row]):
                    found = np.asarray(s["indices"])
                    break
            lst.append(MatStruct(indices=found, presence=pv.copy()))
        out.append(lst)
    return out
