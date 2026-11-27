"""Writing ROIs back into adata files.

Port of the file-writing parts of calliope ``08_RoiQuest/+clbk/saveToAdata.m``,
``+utils/saveRoisAcute.m``, ``+utils/saveRoisChronic.m``, ``+utils/wfSaveRois.m`` and
``+clbk/deleteCallback.m`` (Leonardo Lupori, 2024). What is written is computed by the
:mod:`calliopy.rois` cores (:func:`~calliopy.rois.save_rois_acute`,
:func:`~calliopy.rois.save_rois_chronic`, :func:`~calliopy.rois.wf_save_rois`); this module
reads the other files of a series, makes the backup copies and writes with
:func:`calliopy.io.update_mat`.
"""

from __future__ import annotations

import datetime
import logging
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from calliopy.io import MatStruct, load_adata, load_mat
from calliopy.rois import (
    ChronicRoiSave,
    RoiSaveError,
    save_rois_acute,
    save_rois_chronic,
    wf_save_rois,
)

from ._load import _cells, _has_var, _struct_items, _var, adata_file_of, is_zstack_name, var_names
from ._load import tform_matrix as _tform_matrix
from ._types import RoiQuestRefused, SaveResult

if TYPE_CHECKING:  # pragma: no cover
    from ._session import RoiQuestSession

__all__: list[str] = []

log = logging.getLogger("calliopy.roiquest")

BACKUP_INFIX = ".roiquest_backup_"
"""Backup copies are named ``<adata file name>.roiquest_backup_<YYYYmmdd-HHMMSS>`` and sit
next to the adata file. The name does not end in ``.mat``, so lab code that lists adata files
does not pick them up; MATLAB reads them with ``load(file, '-mat')``."""


def _update_mat(path: Path, variables: Mapping[str, Any]) -> None:
    """Write variables into an existing v7.3 adata file (lazy import of
    :func:`calliopy.io.update_mat`)."""
    try:
        from calliopy.io import update_mat
    except ImportError as err:  # pragma: no cover - depends on the calliopy version
        raise ImportError("calliopy.io.update_mat is not available in this version of "
                          "calliopy; RoiQuest needs it to write ROIs into adata files") from err
    update_mat(path, variables)


def backup(s: RoiQuestSession, path: Path) -> Path | None:
    """Copy ``path`` next to itself once per session before it is first written.

    Returns
    -------
    pathlib.Path or None
        The new backup, or None when this session already made one for the file.
    """
    path = Path(path)
    key = path.resolve()
    if key in s.backups:
        return None
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = path.with_name(path.name + BACKUP_INFIX + stamp)
    n = 1
    while dest.exists():
        dest = path.with_name(f"{path.name}{BACKUP_INFIX}{stamp}-{n}")
        n += 1
    shutil.copy2(path, dest)
    s.backups[key] = dest
    log.info("Backup of %s written to %s", path.name, dest)
    return dest


def write(s: RoiQuestSession, path: Path, variables: Mapping[str, Any],
          written: dict | None = None) -> None:
    """Back up ``path`` (first time only) and write ``variables`` into it."""
    path = Path(path)
    backup(s, path)
    _update_mat(path, variables)
    if written is not None:
        written[path] = tuple(written.get(path, ())) + tuple(variables)


def _require_series(s: RoiQuestSession, what: str) -> None:
    if s.exp_log is None or s.adata_dir is None:
        raise RoiQuestRefused(f"{what} of a chronic series needs the ExpLog table and the adata "
                              "folder (pass exp_log and adata_dir to the session)")


def series_file(s: RoiQuestSession, exp_id: int) -> Path | None:
    """Adata file of an ExpID of the current series (the open file for the current ExpID)."""
    if s.curr_exp_id is not None and int(exp_id) == int(s.curr_exp_id):
        return s.adata_full_path
    return adata_file_of(exp_id, s.adata_dir, s.exp_log)


def _mark_saved(s: RoiQuestSession) -> None:
    for layer in s.rois:
        for r in layer:
            r.is_new = False


def _rois_variable(path: Path):
    names = var_names(path)
    if not _has_var(names, "ROIs"):
        return None
    return load_mat(path, [_var(names, "ROIs")])[_var(names, "ROIs")]


# ---------------------------------------------------------------------------------------
# saveToAdata
# ---------------------------------------------------------------------------------------
def save(s: RoiQuestSession, *, overwrite_activity: bool = False,
         add_new_rois: bool | Callable[[ChronicRoiSave], bool] = False,
         overwrite_policy: str | None = None, tforms: Mapping | None = None) -> SaveResult:
    """Implementation of :meth:`RoiQuestSession.save` (MATLAB ``clbk.saveToAdata``)."""
    written: dict = {}
    before = dict(s.backups)
    chronic = None
    if s.is_wf_eeg or s.is_widefield_data:
        _save_wf(s, written, overwrite_activity, overwrite_policy, tforms)
    elif s.orig_adata_full_path is None:
        old = _rois_variable(s.adata_full_path)
        variables = save_rois_acute(s.rois, s.roi_trans_cell(), old_rois=old,
                                    overwrite_activity=overwrite_activity)
        write(s, s.adata_full_path, variables, written)
        _mark_saved(s)
        log.info("New ROIs succesfully saved to:\n\t- %s", s.adata_full_path)
    else:
        chronic = _save_chronic(s, written, add_new_rois)
    backups = {p: b for p, b in s.backups.items() if p not in before}
    return SaveResult(written, backups, chronic)


def _save_chronic(s: RoiQuestSession, written: dict,
                  add_new_rois: bool | Callable[[ChronicRoiSave], bool]) -> ChronicRoiSave:
    """MATLAB ``utils.saveRoisChronic`` (with ``askConfirmation`` answered beforehand)."""
    _require_series(s, "Saving ROIs")
    group = [int(v) for v in s.exp_group]
    paths: dict[int, Path | None] = {}
    timepoints: dict[int, Any] = {}
    for exp in group:
        p = series_file(s, exp)
        paths[exp] = p
        if exp == s.curr_exp_id:
            continue
        if p is None or is_zstack_name(p):
            timepoints[exp] = None
            continue
        names = var_names(p)
        want = [_var(names, n) for n in ("ROIs", "ROItrans") if _has_var(names, n)]
        m = load_mat(p, ["template", *want])
        timepoints[exp] = MatStruct(
            template=m["template"],
            ROIs=m[_var(names, "ROIs")] if _has_var(names, "ROIs") else [],
            ROItrans=m[_var(names, "ROItrans")] if _has_var(names, "ROItrans") else [],
        )
    log.info("Matching ROIs identity to the existing ones in the original ExpID (%u)...",
             group[0])
    res = save_rois_chronic(s.rois, s.roi_trans_cell(), s.template, s.curr_exp_id, group,
                            timepoints, filter_win_size=s.filter_win_size,
                            filter_type=s.filter_type)
    write(s, s.adata_full_path, {"ROIs": res.ROIs, "ROItrans": res.ROItrans}, written)
    if not res.rois_to_add:
        log.info("Done saving ROIs.")
        _mark_saved(s)
        return res
    log.info("New ROIs defined in this expID.")
    accept = add_new_rois(res) if callable(add_new_rois) else bool(add_new_rois)
    if not accept:
        log.info("Done saving all existing ROIs. (skipped new ROIs)")
        _mark_saved(s)
        return res
    log.info("Saving newly defined ROIs in all AData files of the series.")
    for exp, cell in res.updated_rois.items():
        p = paths.get(exp)
        if p is None:
            continue
        write(s, p, {"ROIs": cell}, written)
        log.info("\t- Added ROIs in ExpID %u", exp)
    _mark_saved(s)
    log.info("Done saving ROIs.")
    return res


def _save_wf(s: RoiQuestSession, written: dict, overwrite_activity: bool,
             overwrite_policy: str | None, tforms: Mapping | None) -> None:
    """MATLAB ``utils.wfSaveRois``."""
    group = [int(v) for v in s.exp_group] or (
        [] if s.curr_exp_id is None else [int(s.curr_exp_id)])
    if s.curr_exp_id is None:
        raise RoiSaveError("The adata file name does not contain an ExpID (Adata-S1-T<ExpID>)")
    cur = int(s.curr_exp_id)
    rois = s.rois[0] if s.rois else []
    red = s.rois[1] if s.is_wf_two_channels and len(s.rois) > 1 else None
    hw = np.asarray(s.template[0]).shape[:2]
    timepoints: dict[int, Any] = {}
    paths: dict[int, Path | None] = {}
    if len(group) < 2:
        old = _rois_variable(s.adata_full_path)
        timepoints[cur] = None if old is None else {"ROIs": old}
        paths[cur] = s.adata_full_path
    else:
        _require_series(s, "Saving widefield ROIs")
        if overwrite_policy not in ("keep", "overwrite"):
            raise ValueError("overwrite_policy must be 'keep' or 'overwrite' for a series")
        for i, exp in enumerate(group):
            p = series_file(s, exp)
            paths[exp] = p
            if p is None:
                timepoints[exp] = None
                continue
            ad = load_adata(p)
            names = set(ad.variables)
            tp = MatStruct(type=ad.type)
            for n in ("registration_parameters", "registration_parameters_ch525",
                      "registration_parameters_ch610", "ROIs"):
                if n in names:
                    tp[n] = ad[n]
            needs_tform = i > 0 and (overwrite_policy == "overwrite" or "ROIs" not in names)
            given = None if tforms is None else tforms.get(exp)
            for n in ("tform", "tform_ch525", "tform_ch610"):
                mat = None
                if isinstance(given, Mapping):
                    mat = _tform_matrix(given.get(n))
                elif given is not None and n == "tform":
                    mat = _tform_matrix(given)
                if mat is None and n in names:
                    mat = _tform_matrix(ad[n])
                if mat is not None:
                    tp[n] = mat
            if needs_tform:
                need = ("tform_ch525", "tform_ch610") if ad.type == "2ch_wf" else ("tform",)
                missing = [n for n in need if n not in tp]
                if missing:
                    raise RoiSaveError(
                        f"ExpID {exp}: the transform(s) {missing} are MATLAB objects that "
                        "calliopy cannot read; pass them as 3x3 matrices in tforms={exp: ...}")
            timepoints[exp] = tp
    out = wf_save_rois(rois, exp_group_ids=group or [cur], current_exp_id=cur,
                       template_hw=hw, roi_template_parameters=s.roi_template_parameters,
                       timepoints=timepoints, overwrite_policy=overwrite_policy,
                       rois_ch610=red, overwrite_activity=overwrite_activity)
    for exp, variables in out.items():
        p = paths.get(exp)
        if p is None:
            continue
        write(s, p, variables, written)
        log.info("\t- ROIs saved to ExpID %u.", exp)


# ---------------------------------------------------------------------------------------
# deleteCallback, chronic part
# ---------------------------------------------------------------------------------------
def delete_tracked(s: RoiQuestSession, selected_idx: list[int], add_new_rois) -> dict:
    """Delete chronically tracked ROIs from every adata file of the series.

    MATLAB ``deleteCallback`` after both confirmations: autosave (``saveRoisChronic`` without
    confirmation), then for every ExpID of the series whose layer has as many ROIs as the
    current layer, remove the selected ROIs and write ``ROIs`` back; finally reload the ROIs
    of the current file (all layers empty when the file has no ROIs left).
    """
    _require_series(s, "Deleting tracked ROIs")
    z = s.current_z
    num_rois = len(s.rois[z])
    written: dict = {}
    _save_chronic(s, written, add_new_rois)
    sel = {int(i) for i in selected_idx}
    log.info("Deleting ROIs #[%s] from all timepoints.", " ".join(str(i + 1) for i in sorted(sel)))
    for exp in [int(v) for v in s.exp_group]:
        p = series_file(s, exp)
        if p is None:
            log.info("\t- ExpID %u does not have a Adata file - skipping", exp)
            continue
        if is_zstack_name(p):
            log.info("\t- ExpID %u is a z-stack - skipping", exp)
            continue
        var = _rois_variable(p)
        if var is None:
            log.info("\t- ExpID %u probably not analyzed yet.", exp)
            continue
        cells = _cells(var)
        items = _struct_items(cells[z]) if z < len(cells) else []
        if len(items) != num_rois:
            log.info("\t- ExpID %u probably not analyzed yet.", exp)
            continue
        keep = [it for k, it in enumerate(items) if k not in sel]
        if keep:
            arr = np.empty((1, len(keep)), dtype=object)
            for k, it in enumerate(keep):
                arr[0, k] = it
        else:
            arr = np.empty((0, 0))
        shape = var.shape if (isinstance(var, np.ndarray) and var.dtype == object) \
            else (1, len(cells))
        new = np.empty(shape, dtype=object)
        flat = list(cells)
        flat[z] = arr
        for k, c in enumerate(flat):
            new[np.unravel_index(k, shape, order="F")] = c
        write(s, p, {"ROIs": new}, written)
        log.info("\t- Updated ROIs in %s.", p)
    log.info("Done deleting ROIs! Now will reload ROIs from current AData file.")
    from ._load import load_rois

    if not load_rois(s, s.adata_full_path):
        # the file has no ROIs left: show and hold empty layers (MATLAB's loadRois aborted
        # and kept the deleted ROIs on screen, so a later save could write them back)
        s.rois = [[] for _ in s.rois]
        s.putative_rois = []
    return written
