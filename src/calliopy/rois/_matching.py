"""ROI matching across chronic recordings and assembly of what RoiQuest saves.

Port of calliope ``08_RoiQuest/+clbk/autoAdjustRois.m``, ``+clbk/autoRoiMatching.m``,
``+utils/checkRoiMatching.m``, ``+utils/roiMatchMessage.m``, ``+utils/saveRoisAcute.m``,
``+utils/saveRoisChronic.m`` and ``+utils/expid2expgroup.m`` (Leonardo Lupori, 2024).

Nothing here reads or writes adata files, looks up ExpIDs or asks the user: the caller
loads the adata variables (``calliopy.io.load_adata`` / ``load_mat``), passes them in, and
receives what MATLAB would have written as ``MatStruct`` values with MATLAB's variable
names. Writing them back is deferred (``calliopy`` does not write adata files yet).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np

from calliopy._matlab import mround
from calliopy.io import MatStruct
from calliopy.widefield._frames import register_frames

from ._images import adjust_img_contrast, filter_template
from ._roi import (
    _EMPTY,
    Roi,
    _cell_list,
    _cell_row,
    _get,
    _ind2sub,
    _is_empty,
    _lin,
    _struct_array,
    _struct_list,
    move_roi_structs,
    rois2struct,
)

__all__ = [
    "AutoAdjustResult",
    "AutoRoiMatchingResult",
    "ChronicRoiSave",
    "RoiMatchResult",
    "RoiSaveError",
    "auto_adjust_rois",
    "auto_roi_matching",
    "check_roi_matching",
    "exp_group",
    "roi_match_message",
    "save_rois_acute",
    "save_rois_chronic",
]

MINIMUM_OVERLAP = 0.5
_WIN_SIZE = 30


class RoiSaveError(RuntimeError):
    """Saving ROIs was refused (MATLAB logged a message and aborted the save)."""


# ---------------------------------------------------------------------------------------
# autoAdjustRois / autoRoiMatching
# ---------------------------------------------------------------------------------------
class AutoAdjustResult(NamedTuple):
    """Outputs of :func:`auto_adjust_rois` (MATLAB output names)."""

    dxdy: np.ndarray
    """``[nRois x 2]`` local displacement (rows, columns) of each ROI."""
    global_dxdy: np.ndarray
    """``(1, 2)`` global displacement ``[dxF, dyF]`` (``[0, 0]`` with ``ignore_global``)."""
    is_out_of_bounds: np.ndarray
    """``[nRois x 1]`` bool: the moved ROI would leave the image."""
    start_centroids: np.ndarray
    """``[nRois x 2]`` centroid (row, column) of each ROI, **1-based** (MATLAB subscripts)."""
    displacement: np.ndarray
    """``[nRois x 2]`` total displacement ``dxdy + [dxF, dyF]``."""


def _register(img: np.ndarray, ref: np.ndarray, perc: float) -> tuple[float, float]:
    res = register_frames(np.asarray(img, dtype=np.float64)[:, :, None],
                          np.asarray(ref, dtype=np.float64), perc)
    return float(np.ravel(res.shift_x)[0]), float(np.ravel(res.shift_y)[0])


def auto_adjust_rois(rois, img, ref_img, ignore_global: bool = False) -> AutoAdjustResult:
    """Displacement of each ROI between an image and a reference, MATLAB
    ``clbk.autoAdjustRois``.

    1. Global shift ``[dxF, dyF]`` of ``img`` relative to ``ref_img`` with
       :func:`calliopy.widefield.register_frames` (border 0.15); ``img`` is circularly shifted
       by ``-[dxF, dyF]`` (unless ``ignore_global``).
    2. For each ROI, a window of +-30 pixels around its rounded centroid is cut from the
       shifted image and from the reference (each minus its window mean, zero elsewhere) and
       registered with border 0, giving ``dxdy``.

    Parameters
    ----------
    rois
        Sequence of :class:`Roi` or ROI structs (``indices`` 1-based, as stored).
    img, ref_img
        2-D images of the same size (e.g. filtered templates of the current and the original
        recording).
    ignore_global
        Skip the global registration (``globalDxdy = [0, 0]``).

    Returns
    -------
    AutoAdjustResult

    Notes
    -----
    Kept as in MATLAB (suspected bug, reported): with ``ignore_global`` the global shift is
    still computed and still added to ``displacement`` and to the out-of-bounds test, although
    ``global_dxdy`` is returned as ``[0, 0]``. The out-of-bounds test uses ``< 0`` and
    ``> size`` (MATLAB), not ``< 1``.
    """
    im = np.asarray(img, dtype=np.float64)
    ref = np.asarray(ref_img, dtype=np.float64)
    if im.shape != ref.shape or im.ndim != 2:
        raise ValueError("img and ref_img must be 2-D images of the same size")
    h, w = im.shape
    dx_f, dy_f = _register(im, ref, 0.15)
    if ignore_global:
        shifted = im
        global_dxdy = np.zeros((1, 2))
    else:
        shifted = np.roll(im, (-int(dx_f), -int(dy_f)), axis=(0, 1))
        global_dxdy = np.array([[dx_f, dy_f]])
    seq = list(rois)
    n = len(seq)
    start = np.zeros((n, 2))
    disp = np.zeros((n, 2))
    oob = np.zeros((n, 1), dtype=bool)
    dxdy = np.zeros((n, 2))
    for i, roi in enumerate(seq):
        r_x, r_y = _ind2sub((h, w), _get(roi, "indices"))
        with np.errstate(invalid="ignore"):
            cx = r_x.mean() if r_x.size else np.nan
            cy = r_y.mean() if r_y.size else np.nan
        start[i] = (cx, cy)
        win = np.arange(-_WIN_SIZE, _WIN_SIZE + 1)
        if np.isnan(cx) or np.isnan(cy):
            txs = tys = np.array([], dtype=int)
        else:
            txs = win + int(mround(cx))
            tys = win + int(mround(cy))
            txs = txs[(txs > 0) & (txs <= h)]
            tys = tys[(tys > 0) & (tys <= w)]
        t_win = np.zeros_like(shifted)
        ot_win = np.zeros_like(ref)
        if txs.size and tys.size:
            ix = np.ix_(txs - 1, tys - 1)
            t_win[ix] = shifted[ix] - shifted[ix].mean()
            ot_win[ix] = ref[ix] - ref[ix].mean()
        dx, dy = _register(t_win, ot_win, 0)
        dxdy[i] = (dx, dy)
        disp[i] = (dx + dx_f, dy + dy_f)
        sx = r_x + dx + dx_f
        sy = r_y + dy + dy_f
        oob[i, 0] = bool(np.any(sx < 0) or np.any(sx > h) or np.any(sy < 0) or np.any(sy > w))
    return AutoAdjustResult(dxdy, global_dxdy, oob, start, disp)


class AutoRoiMatchingResult(NamedTuple):
    """Outputs of :func:`auto_roi_matching`."""

    roi_trans: np.ndarray
    """``(1, 2)`` global shift to store in ``ROItrans{z}(1:2)``."""
    start_centroids: np.ndarray
    """``[nSelected x 2]`` 1-based centroids (row, column) before moving (quiver origins)."""
    displacement: np.ndarray
    """``[nSelected x 2]`` applied displacement (``[0, 0]`` for ROIs not moved)."""
    is_out_of_bounds: np.ndarray
    """``[nSelected x 1]`` bool: ROI not moved because it would leave the image."""


def auto_roi_matching(rois: Sequence[Roi], current, orig, selected, *, local_only=False,
                      global_only=False) -> AutoRoiMatchingResult:
    """Move selected ROIs onto the current recording, MATLAB ``clbk.autoRoiMatching``.

    Runs :func:`auto_adjust_rois` on the selected ROIs (``current`` = image of this
    recording, ``orig`` = same image of the original recording), moves every ROI that stays
    inside the image with :meth:`Roi.move` (local ``dxdy`` plus the global shift), and
    deselects all ROIs. The ROIs are modified in place.

    Parameters
    ----------
    rois
        ROIs of the current Z layer (:class:`Roi` objects).
    current, orig
        2-D images (RoiQuest: the filtered templates, ``imgs{2, z}`` and ``origImgs{2, z}``).
    selected
        Which ROIs to match: **0-based** indices or a boolean mask over ``rois``.
    local_only
        MATLAB modifier ``'shift'``: ignore the global registration.
    global_only
        MATLAB modifier ``'control'``: apply only the global shift (``dxdy`` set to 0).

    Returns
    -------
    AutoRoiMatchingResult

    Raises
    ------
    ValueError
        No ROIs, or none selected (MATLAB logged a message and returned).

    Notes
    -----
    Not ported: the refusal for widefield data (the caller decides), the log messages and
    the quiver plot of the GUI.
    """
    seq = list(rois)
    if not seq:
        raise ValueError("No ROIs to correct.")
    sel = np.asarray(selected)
    if sel.dtype == bool:
        idx = np.flatnonzero(sel)
    else:
        idx = sel.astype(int).ravel()
    if idx.size == 0:
        raise ValueError("No ROIs selected for correction.")
    sub = [seq[i] for i in idx]
    res = auto_adjust_rois(sub, current, orig, ignore_global=local_only)
    dxdy = np.zeros_like(res.dxdy) if global_only else res.dxdy
    disp = res.displacement.copy()
    g = res.global_dxdy.ravel()
    for k, i in enumerate(idx):
        if res.is_out_of_bounds[k, 0]:
            disp[k] = (0.0, 0.0)
        else:
            seq[i].move(dxdy[k, 0], dxdy[k, 1], g)
            disp[k] = (dxdy[k, 0] + g[0], dxdy[k, 1] + g[1])
    for r in seq:
        r.selected = False
    return AutoRoiMatchingResult(res.global_dxdy.copy(), res.start_centroids, disp,
                                 res.is_out_of_bounds)


# ---------------------------------------------------------------------------------------
# checkRoiMatching / roiMatchMessage
# ---------------------------------------------------------------------------------------
class RoiMatchResult(NamedTuple):
    """Outputs of :func:`check_roi_matching` (MATLAB ``[bool, isOverlapping, isTypeMatched]``)."""

    matched: bool
    """MATLAB ``bool``: every ROI overlaps and has the same type."""
    is_overlapping: list
    """Per Z layer, bool ``(1, nRois)``: overlap >= 0.5 of the original ROI."""
    is_type_matched: list
    """Per Z layer, bool ``(1, nRois)``: same ``type``."""


def _layers(rois_per_z) -> list:
    if isinstance(rois_per_z, np.ndarray) and rois_per_z.dtype == object:
        return _cell_list(rois_per_z)
    return list(rois_per_z)


def _rois_in_layer(layer) -> list:
    if isinstance(layer, (list, tuple)):
        return list(layer)
    return _struct_list(layer)


def check_roi_matching(orig_rois, test_rois, roi_trans, img_hw) -> RoiMatchResult:
    """Check that the ROIs of a recording match the original ones, MATLAB
    ``utils.checkRoiMatching``.

    ROI ``i`` of each layer is translated back by ``ROItrans{z}(1:2) + shift`` and must cover
    at least half of the pixels of original ROI ``i`` (``MINIMUM_OVERLAP = 0.5``) and have the
    same ``type``.

    Parameters
    ----------
    orig_rois, test_rois
        Cell arrays (one entry per Z layer) of ROI struct arrays or :class:`Roi` lists, as in
        the adata ``ROIs`` variable.
    roi_trans
        Cell array of per-layer ``ROItrans`` vectors (first two values used).
    img_hw
        ``(height, width)``.

    Returns
    -------
    RoiMatchResult
        When the numbers of layers differ: ``matched = False`` and lists of ``None``.

    Notes
    -----
    Differences from MATLAB: pixels translated outside the image are not counted as common
    (MATLAB's ``sub2ind`` failed); a test layer with fewer ROIs than the original marks the
    missing ROIs as not overlapping and not type-matched (MATLAB failed on the index).
    """
    orig = _layers(orig_rois)
    test = _layers(test_rois)
    trans = _layers(roi_trans)
    if len(test) != len(orig):
        return RoiMatchResult(False, [None] * len(test), [None] * len(test))
    h, w = int(img_hw[0]), int(img_hw[1])
    ok = True
    ovl_all, typ_all = [], []
    for z in range(len(orig)):
        o = _rois_in_layer(orig[z])
        t = _rois_in_layer(test[z])
        tr = np.ravel(np.asarray(trans[z], dtype=np.float64))
        ovl = np.ones((1, len(o)), dtype=bool)
        typ = np.ones((1, len(o)), dtype=bool)
        for i, oroi in enumerate(o):
            if i >= len(t):
                ovl[0, i] = typ[0, i] = False
                ok = False
                continue
            troi = t[i]
            x, y = _ind2sub((h, w), _get(troi, "indices"))
            shift = np.ravel(np.asarray(_get(troi, "shift", [0, 0]), dtype=np.float64))
            if shift.size < 2:
                shift = np.zeros(2)
            x = x - (tr[0] + shift[0])
            y = y - (tr[1] + shift[1])
            inside = (x >= 1) & (x <= h) & (y >= 1) & (y <= w)
            test_idx = (x[inside] - 1) + (y[inside] - 1) * h + 1
            o_idx = _lin(_get(oroi, "indices"))
            common = np.intersect1d(o_idx, test_idx.astype(np.int64))
            with np.errstate(invalid="ignore", divide="ignore"):
                overlap = common.size / o_idx.size if o_idx.size else np.nan
            if overlap < MINIMUM_OVERLAP:
                ovl[0, i] = False
                ok = False
            ta, oa = _get(troi, "type"), _get(oroi, "type")
            if not (_is_empty(ta) or _is_empty(oa)) and np.ravel(ta)[0] != np.ravel(oa)[0]:
                typ[0, i] = False
                ok = False
        ovl_all.append(ovl)
        typ_all.append(typ)
    return RoiMatchResult(ok, ovl_all, typ_all)


def _num2str(values) -> str:
    """MATLAB ``num2str`` of a row of non-negative integers (``'1  2  10'``)."""
    v = [int(x) for x in np.ravel(values)]
    if not v:
        return ""
    width = max(len(str(x)) for x in v) + 2
    return "".join(f"{x:{width}d}" for x in v).lstrip()


def roi_match_message(is_overlapping, is_type_matched) -> str:
    """Error message for a failed :func:`check_roi_matching`, MATLAB ``utils.roiMatchMessage``.

    Z layers and ROIs are numbered from 1, as in the RoiQuest GUI.

    Differences from MATLAB (bugs fixed, they only affected this text or crashed): the type
    section printed the layer list with ``%u`` (character codes) and looped over the layers
    of the *overlap* section (failing when there was no overlap error); here it lists the
    type-mismatched layers and their ROIs.
    """
    ovl = [np.asarray(x, dtype=bool) for x in is_overlapping]
    typ = [np.asarray(x, dtype=bool) for x in is_type_matched]
    msg = "ERROR. ROI mismatch detected!\n"
    z_ovl = [i for i, x in enumerate(ovl) if not np.all(x)]
    if z_ovl:
        msg += f"\t- Z-Layer(s) [{_num2str(np.array(z_ovl) + 1)}] fail(s) ROI overlap criterion.\n"
        for z in z_ovl:
            msg += f"\t- ROIs(s) [{_num2str(np.flatnonzero(~ovl[z].ravel()) + 1)}].\n"
    z_typ = [i for i, x in enumerate(typ) if not np.all(x)]
    if z_typ:
        layers = _num2str(np.array(z_typ) + 1).replace("  ", ",")
        msg += f"\t- Z-Layer(s) {layers} fail(s) ROI type matching.\n"
        for z in z_typ:
            wrong = _num2str(np.flatnonzero(~typ[z].ravel()) + 1).replace("  ", ",")
            msg += f"\t\t- ROIs(s) [{wrong}].\n"
    return msg


# ---------------------------------------------------------------------------------------
# saving
# ---------------------------------------------------------------------------------------
def _rois_cell(rois_per_z) -> list:
    """Per-layer ROI lists -> list of struct arrays (``[]`` for empty layers)."""
    out = []
    for layer in _layers(rois_per_z):
        s = rois2struct(_rois_in_layer(layer) if not isinstance(layer, (list, tuple))
                        else layer)
        out.append(_EMPTY.copy() if isinstance(s, Mapping) and len(s) == 0 else s)
    return out


def _layer_empty(layer) -> bool:
    return len(_rois_in_layer(layer)) == 0


def _has_activity(old_rois) -> bool:
    layers = _layers(old_rois)
    if not layers:
        return False
    first = _rois_in_layer(layers[0])
    return bool(first) and "activity" in first[0]


def save_rois_acute(rois, roi_trans, *, old_rois=None, overwrite_activity: bool = False) -> MatStruct:
    """Variables RoiQuest writes for a single (not chronic) recording, MATLAB
    ``utils.saveRoisAcute``.

    Parameters
    ----------
    rois
        Per Z layer, the ROIs (:class:`Roi` lists or struct arrays): MATLAB ``app.ROIs``.
    roi_trans
        MATLAB ``app.ROItrans`` (cell of per-layer vectors), returned unchanged.
    old_rois
        The ``ROIs`` variable currently in the adata file, to check for activity traces.
    overwrite_activity
        The answer to MATLAB's dialog "Overwriting ROI activity traces?".

    Returns
    -------
    MatStruct
        ``ROIs`` (``(1, nZ)`` cell: struct array per layer, ``[]`` for empty layers) and
        ``ROItrans``. Writing them into the adata file is deferred.

    Raises
    ------
    RoiSaveError
        ``old_rois`` has activity traces and ``overwrite_activity`` is False.

    Notes
    -----
    MATLAB also marks the GUI's ROIs as not new after saving; set ``Roi.is_new`` yourself.
    """
    if old_rois is not None and _has_activity(old_rois) and not overwrite_activity:
        raise RoiSaveError("ROIs in the AData file have activity traces; ROI saving aborted.")
    return MatStruct(ROIs=_cell_row(_rois_cell(rois)), ROItrans=roi_trans)


class ChronicRoiSave(NamedTuple):
    """Result of :func:`save_rois_chronic`."""

    ROIs: np.ndarray
    """``(1, nZ)`` cell written first to the current adata (ROIs matching the original)."""
    ROItrans: object
    """``ROItrans`` written with it (the input, unchanged)."""
    rois_to_add: list
    """MATLAB ``r2aInfo``: one MatStruct per timepoint (``expID``, ``ROIs``, ``imgs``,
    ``titles``) with the new ROIs moved to that timepoint; empty without new ROIs."""
    updated_rois: dict
    """ExpID -> ``(1, nZ)`` cell of ROIs after appending the new ROIs: what MATLAB writes to
    each adata file when the user accepts the new ROIs."""


def exp_group(exp_id, explog_expid, explog_siteid) -> np.ndarray:
    """ExpIDs recorded at the same site, MATLAB ``utils.expid2expgroup`` (core).

    Parameters
    ----------
    exp_id
        The ExpID.
    explog_expid, explog_siteid
        ExpLog columns ``expid`` and ``siteid`` (same length).

    Returns
    -------
    numpy.ndarray
        Sorted unique ExpIDs whose ``siteid`` equals that of ``exp_id``.

    Raises
    ------
    ValueError
        ``exp_id`` has no or several site IDs (MATLAB failed or compared with a vector).
    """
    e = np.ravel(np.asarray(explog_expid, dtype=np.float64))
    s = np.ravel(np.asarray(explog_siteid, dtype=np.float64))
    site = np.unique(s[e == exp_id])
    if site.size != 1:
        raise ValueError(f"ExpID {exp_id} has {site.size} site IDs in the ExpLog")
    return np.unique(e[s == site[0]])


def _blank_like(s: Mapping) -> MatStruct:
    return MatStruct((k, _EMPTY.copy()) for k in s)


def _assign_gaps(items: dict) -> np.ndarray | None:
    """MATLAB ``A(j) = s`` growth: gaps filled with empty-field structs."""
    if not items:
        return None
    n = max(items) + 1
    tmpl = next(iter(items.values()))
    return _struct_array([items[j] if j in items else _blank_like(tmpl) for j in range(n)])


def _concat(old, new) -> object:
    a = _struct_list(old)
    b = _struct_list(new)
    if not b:
        return old if a else (_EMPTY.copy() if old is None else old)
    if a:
        if set(a[0].keys()) != set(b[0].keys()):
            raise RoiSaveError("Concatenation of ROI structs with different fields "
                               "(MATLAB error)")
        order = list(a[0].keys())
        b = [MatStruct((k, s[k]) for k in order) for s in b]
    return _struct_array(a + b)


def save_rois_chronic(rois, roi_trans, template, current_exp_id, exp_group_ids, timepoints, *,
                      filter_win_size, filter_type: str = "legacy") -> ChronicRoiSave:
    """Match and assemble ROIs of a chronic series, MATLAB ``utils.saveRoisChronic``.

    For a recording that is part of a chronic series and is not the original (first ExpID
    of the group):

    1. ROIs of every layer must be at least as many as in the original; extra ROIs ("new
       ROIs") are set aside.
    2. The remaining ROIs of the current recording, and those of every other analysed
       timepoint, must match the original (:func:`check_roi_matching`).
    3. ``ROIs``/``ROItrans`` of the current recording are what is saved first.
    4. New ROIs are moved back to the original's coordinates (negative
       :func:`auto_adjust_rois` displacement between the filtered templates, ``shift`` reset)
       and from there onto every analysed timepoint, and appended to each file's ROIs
       (``updated_rois``; MATLAB writes them after the user confirms in a check figure, see
       :func:`plot_rois_to_add`).

    Parameters
    ----------
    rois
        Per Z layer, ROIs of the current recording (``app.ROIs``).
    roi_trans
        ``app.ROItrans``.
    template
        Per Z layer, the template images of the current recording (``app.template``).
    current_exp_id
        ExpID of the current recording.
    exp_group_ids
        ExpIDs of the series (``expGroup``), the original first.
    timepoints
        Mapping ExpID -> adata variables of that recording (a mapping with ``ROIs``,
        ``ROItrans`` and ``template``, e.g. from :func:`calliopy.io.load_mat`), or ``None`` for
        an ExpID without adata file or that is a z-stack (MATLAB: adata name containing
        ``'mean_data'``). The original ExpID must be present; the current one is ignored (its
        data come from the other arguments).
    filter_win_size, filter_type
        Template filter (``app.filterWinSize``, ``app.filterType``), see
        :func:`filter_template`.

    Returns
    -------
    ChronicRoiSave

    Raises
    ------
    RoiSaveError
        Original without adata or ROIs, fewer ROIs than the original, or a ROI mismatch
        (message of :func:`roi_match_message`) -- MATLAB logged it and aborted.

    Notes
    -----
    Kept as in MATLAB (suspected bugs, reported):

    - a new ROI whose adjusted position is out of bounds is skipped by assigning only the
      other elements of the struct array, so a skipped ROI in the middle becomes a struct
      with all fields empty and a skipped last ROI disappears;
    - for the current recording, the appended new ROIs are the auto-adjusted copies coming
      back from the original's coordinates, not the ROIs as drawn;
    - the title of the original's panel uses the last ExpID of the matching loop.

    Differences from MATLAB: ``aDataFile`` is not part of ``rois_to_add`` (the caller knows
    the files); concatenating ROI structs with different fields raises ``RoiSaveError`` (a
    MATLAB error as well).
    """
    group = [int(v) for v in np.ravel(exp_group_ids)]
    orig_id = group[0]
    orig_tp = timepoints.get(orig_id)
    if orig_tp is None:
        raise RoiSaveError(f"Original ExpID ({orig_id}) has no AData files.")
    orig_rois = _layers(orig_tp["ROIs"])
    if not orig_rois or _layer_empty(orig_rois[0]):
        raise RoiSaveError(f"Original ExpID ({orig_id}) has no ROIs.")
    templ = _layers(template)
    nz = len(templ)
    img_hw = np.asarray(templ[0]).shape[:2]
    new_rois = _rois_cell(rois)

    diff = [len(_struct_list(new_rois[i])) - len(_rois_in_layer(orig_rois[i])) for i in range(nz)]
    rois_to_add_z: list = [None] * nz
    additional = False
    if any(d < 0 for d in diff):
        less = [i + 1 for i, d in enumerate(diff) if d < 0]
        raise RoiSaveError(
            f"The current ExpID ({current_exp_id}) has less ROIs than the original ({orig_id}) "
            f"in Z layer(s) [{_num2str(less)}].")
    if any(d > 0 for d in diff):
        additional = True
        for i in range(nz):
            items = _struct_list(new_rois[i])
            n_orig = len(_rois_in_layer(orig_rois[i]))
            rois_to_add_z[i] = items[n_orig:]
            kept = items[:n_orig]
            new_rois[i] = _struct_array(kept) if kept else _EMPTY.copy()

    last_exp = group[-1]
    for exp in group[1:]:
        if exp == current_exp_id:
            res = check_roi_matching(orig_rois, new_rois, roi_trans, img_hw)
        else:
            tp = timepoints.get(exp)
            if tp is None:
                continue
            test = _layers(tp["ROIs"])
            if not test or _layer_empty(test[0]):
                continue
            res = check_roi_matching(orig_rois, test, tp["ROItrans"],
                                     np.asarray(_layers(tp["template"])[0]).shape[:2])
        if not res.matched:
            raise RoiSaveError(roi_match_message(res.is_overlapping, res.is_type_matched))

    saved = _cell_row(new_rois)
    if not additional:
        return ChronicRoiSave(saved, roi_trans, [], {})

    orig_templ = _layers(orig_tp["template"])

    def filt(img):
        return filter_template(img, filter_win_size, filter_type)

    # original timepoint: back-adjust the new ROIs to the original coordinates
    rois_o, imgs_o, titles_o = [], [], []
    for i in range(nz):
        orig_f = filt(orig_templ[i])
        this_f = filt(templ[i])
        titles_o.append(f"ORIGINAL ExpID: {last_exp} - Z:{i + 1}")
        imgs_o.append(adjust_img_contrast(orig_f))
        add = rois_to_add_z[i] or []
        if not add:
            rois_o.append(_EMPTY.copy())
            continue
        res_a = auto_adjust_rois(add, this_f, orig_f)
        moved = {}
        for j, s in enumerate(add):
            if res_a.is_out_of_bounds[j, 0]:
                continue
            m = move_roi_structs([s], -res_a.dxdy[j, 0], -res_a.dxdy[j, 1],
                                 -res_a.global_dxdy.ravel(), img_hw)[0, 0]
            m["shift"] = np.zeros((1, 2))
            moved[j] = m
        arr = _assign_gaps(moved)
        rois_o.append(_EMPTY.copy() if arr is None else arr)
    r2a = [MatStruct(expID=orig_id, ROIs=_cell_row(rois_o), imgs=_cell_row(imgs_o),
                     titles=_cell_row(titles_o))]
    in_orig = rois_o

    for exp in group[1:]:
        if exp == current_exp_id:
            old = new_rois
            tmpl_check = templ
        else:
            tp = timepoints.get(exp)
            if tp is None:
                continue
            old = _layers(tp["ROIs"])
            tmpl_check = _layers(tp["template"])
        if all(_layer_empty(x) for x in old):
            continue
        rois_e, imgs_e, titles_e = [], [], []
        for j in range(nz):
            orig_f = filt(orig_templ[j])
            this_f = filt(tmpl_check[j])
            prefix = "CURRENT ExpID" if exp == current_exp_id else "ExpID"
            titles_e.append(f"{prefix}: {exp} - Z:{j + 1}")
            imgs_e.append(adjust_img_contrast(this_f))
            src = _struct_list(in_orig[j])
            if not src:
                rois_e.append(_EMPTY.copy())
                continue
            res_a = auto_adjust_rois(src, this_f, orig_f)
            moved = {}
            for k, s in enumerate(src):
                if res_a.is_out_of_bounds[k, 0]:
                    continue
                moved[k] = move_roi_structs([s], res_a.dxdy[k, 0], res_a.dxdy[k, 1],
                                            res_a.global_dxdy.ravel(), img_hw)[0, 0]
            arr = _assign_gaps(moved)
            rois_e.append(_EMPTY.copy() if arr is None else arr)
        r2a.append(MatStruct(expID=exp, ROIs=_cell_row(rois_e), imgs=_cell_row(imgs_e),
                             titles=_cell_row(titles_e)))

    updated = {}
    for entry in r2a:
        exp = int(entry["expID"])
        if exp == orig_id:
            base = list(orig_rois)
        elif exp == current_exp_id:
            base = list(new_rois)
        else:
            base = list(_layers(timepoints[exp]["ROIs"]))
        add_cells = _cell_list(entry["ROIs"])
        updated[exp] = _cell_row([_concat(base[z], add_cells[z]) for z in range(len(base))])
    return ChronicRoiSave(saved, roi_trans, r2a, updated)
