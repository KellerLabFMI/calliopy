"""Saving two-photon results into adata files.

The adata-writing parts of calliope ``05_LoadData/register_bin.m``,
``11_TwoPhoton/calc_act.m``, ``11_TwoPhoton/tpi_adata_check.m``,
``11_TwoPhoton/tpi_alignment_visualize.m``, ``21_Analysis/register_exp_downsample.m``,
``21_Analysis/register_exp_avgplanes.m``, ``21_Analysis/register_exp_fixoutliers.m``,
``21_Analysis/correct_dxdy.m`` and the SpineQuest callbacks of ``04_Calliope``. Each
saver takes the adata path and the result of the matching computation in
:mod:`calliopy.twophoton`, and writes the variables MATLAB writes, with MATLAB's names,
classes and cell/struct layout:

- functions that use MATLAB's ``Adata`` class (``register_bin``, ``calc_act``,
  ``tpi_adata_check``, SpineQuest) go through :class:`calliopy.io.WritableAdata`, which
  also appends to its modification log;
- functions that load the whole file and ``save`` it again (the registration fixes) or
  use ``save(..., '-append')`` change only their variables with
  :func:`calliopy.io.update_mat`.

Finding the adata file of an ExpID, the ExpLog metadata and the database updates
(``update_raw_data_status``) are not ported: the path and the metadata are arguments.
"""

from __future__ import annotations

import datetime
import logging
import re
import warnings
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from ..io import MatStruct, WritableAdata, save_mat, update_mat
from ..io._adata_write import (
    _calliopy_version,
    _cell_row,
    _cells,
    _datestr,
    _empty_struct,
    _existing_v73,
    _float_row,
    _num2str_int,
    _read_variables,
    _set_check_raw_passed,
    _struct_array,
    _struct_items,
)

__all__ = [
    "save_calc_act_results",
    "save_check_raw_passed",
    "save_correct_dxdy_results",
    "save_detected_spines",
    "save_register_bin_results",
    "save_register_exp_avgplanes_results",
    "save_register_exp_downsample_results",
    "save_register_exp_fixoutliers_results",
    "save_spine_tracking",
    "save_spiny_regions",
    "save_tpi_adata_check_results",
]

logger = logging.getLogger(__name__)


def _layers_cell(values: Any) -> Any:
    """A per-layer list as MATLAB's ``1 x nLayers`` cell (``{}`` -> ``0 x 0`` cell);
    anything else (an array of a one-layer result) unchanged."""
    if isinstance(values, (list, tuple)):
        return _cell_row(values)
    return values


def _result_get(result: Any, name: str) -> Any:
    if isinstance(result, Mapping):
        return result[name]
    return getattr(result, name)


# =======================================================================================
# register_bin
# =======================================================================================


def save_register_bin_results(
    adata_path: str | PathLike,
    result: Mapping[str, Any],
    *,
    aux_files: Any = None,
    fnames: Any = None,
    mouse_id: str | None = None,
    user_id: str | None = None,
    create_dir: bool = True,
    adata_root: str | PathLike | None = None,
) -> Path:
    """Create the adata file of a registered two-photon experiment (``register_bin.m``).

    Writes a new file with :meth:`calliopy.io.WritableAdata.create` (``create_dir`` as in
    MATLAB) holding, for a time series, in MATLAB's order: ``ROIs``, ``bv``, ``np`` (one
    empty ``struct`` per layer), ``template``, ``dx``, ``dy``, ``aux_files``, ``fnames``,
    ``nbr_frames`` (double row), ``mouse_id``, ``userID``, ``act_map``, ``template_sec``,
    ``act_map_sec`` (per-layer ``1 x nLayers`` cells; ``{}`` (``0 x 0`` cell) without a
    second channel). For a z-stack only ``mean_data`` (and ``mean_data_sec``).

    Parameters
    ----------
    adata_path
        The new file, normally :func:`calliopy.io.adata_file_path` (``Adata-S1-T<ExpID>.mat``).
    result
        :func:`calliopy.twophoton.register_bin` (or :func:`register_bin_files`) output.
    aux_files, fnames, mouse_id, user_id
        ExpLog metadata (MATLAB ``ExpInfo.aux_files``, ``ExpInfo.fnames`` (``1 x nFiles``
        cells of paths), ``ExpInfo.mouse_id``, ``ExpInfo.userID``); required for a time
        series, unused for a z-stack.
    create_dir, adata_root
        Passed to :meth:`calliopy.io.WritableAdata.create`.

    Returns
    -------
    pathlib.Path

    Raises
    ------
    FileExistsError
        The file exists: MATLAB skipped the ExpID ("Adata file already exists") and
        ``Adata.create`` refuses to overwrite.

    Notes
    -----
    Not ported: finding whether an adata file exists for the ExpID under another name
    (``find_adata_file``) and saving the check figure
    (:func:`calliopy.twophoton.save_check_registration_figure` is separate).
    """
    ad = WritableAdata.create(adata_path, create_dir=create_dir, adata_root=adata_root)
    if "mean_data" in result:
        ad.add_variable(mean_data=result["mean_data"])
        if "mean_data_sec" in result:
            ad.add_variable(mean_data_sec=result["mean_data_sec"])
        ad.save_changes()
        return ad.path
    missing = [n for n, v in (("aux_files", aux_files), ("fnames", fnames),
                              ("mouse_id", mouse_id), ("user_id", user_id)) if v is None]
    if missing:
        raise ValueError(f"register_bin adata needs {', '.join(missing)}")
    ad.add_variable({
        "ROIs": _cell_row([_empty_struct() for _ in result["ROIs"]]),
        "bv": _cell_row([_empty_struct() for _ in result["bv"]]),
        "np": _cell_row([_empty_struct() for _ in result["np"]]),
        "template": _layers_cell(result["template"]),
        "dx": _layers_cell(result["dx"]),
        "dy": _layers_cell(result["dy"]),
        "aux_files": aux_files,
        "fnames": fnames,
        "nbr_frames": _float_row(result["nbr_frames"]),
        "mouse_id": str(mouse_id),
        "userID": str(user_id),
        "act_map": _layers_cell(result["act_map"]),
        "template_sec": _layers_cell(result["template_sec"]),
        "act_map_sec": _layers_cell(result["act_map_sec"]),
    })
    ad.save_changes()
    return ad.path


# =======================================================================================
# calc_act
# =======================================================================================


def save_calc_act_results(
    adata_path: str | PathLike,
    result: Any,
    *,
    rois_recalculated: bool = True,
    raw_data_status: Any = None,
) -> bool:
    """Save ROI activity, neuropil and projections into the adata file (``calc_act.m``).

    As MATLAB (``adata.ROIs{l} = ...``, ``adata.np{l} = ...``,
    ``adata.add_variable(projections)``, ``adata.save_changes``):

    - ``ROIs`` and ``np``: the layers that :func:`calliopy.twophoton.calc_act` computed
      (ROIs with ``indices``) are replaced, the other layers keep the file's value; one
      ROI is written as a struct, several as a ``1 x n`` struct array. With
      ``rois_recalculated=False`` (MATLAB ``skip_act_calc``) only ``np`` is replaced;
    - ``projections`` (``mean``, ``median``, ``max``, ``std``: ``1 x nLayers`` cells) is
      added only when the file has none: MATLAB's ``add_variable`` skips an existing
      variable with a warning, so recomputed projections are never saved.

    Parameters
    ----------
    adata_path
        Existing adata file of the experiment.
    result
        :class:`calliopy.twophoton.CalcActResult`.
    rois_recalculated
        False when ``calc_act`` ran with ``recalc_rois=False``.
    raw_data_status
        Passed to :meth:`calliopy.io.WritableAdata.save_changes` (raw-data lock).

    Returns
    -------
    bool
        Whether anything was written.

    Notes
    -----
    Not ported: the ExpLog lookup, the "already calculated" checks (the caller decides
    whether to run ``calc_act``) and ``override == 2`` (compute without saving: do not
    call this function).
    """
    ad = WritableAdata(_existing_v73(adata_path))
    if "ROIs" not in ad:
        raise KeyError(f"{ad.path.name} has no ROIs")
    file_rois = _cells(ad["ROIs"])
    file_np = _cells(ad["np"]) if "np" in ad else [_empty_struct() for _ in file_rois]
    res_rois = list(_result_get(result, "ROIs"))
    res_np = list(_result_get(result, "np"))
    computed = [bool(len(x)) for x in res_np]
    if len(computed) != len(file_rois):
        raise ValueError(f"result has {len(computed)} layers, {ad.path.name} has {len(file_rois)}")
    if any(computed):
        if rois_recalculated:
            ad["ROIs"] = _cell_row(
                [_struct_array(res_rois[q]) if computed[q] else file_rois[q]
                 for q in range(len(computed))]
            )
        new_np = _cell_row(
            [_struct_array(res_np[q]) if computed[q] else file_np[q] for q in range(len(computed))]
        )
        if "np" in ad:
            ad["np"] = new_np
        else:
            ad.add_variable(np=new_np)
    proj = _result_get(result, "projections")
    projections = MatStruct(
        (k, _layers_cell(list(proj[k]))) for k in ("mean", "median", "max", "std")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        ad.add_variable(projections=projections)
    if not ad.modified:
        warnings.warn("No variables have been modified. Nothing to save.", stacklevel=2)
        return False
    return ad.save_changes(raw_data_status=raw_data_status)


# =======================================================================================
# tpi_adata_check / tpi_alignment_visualize
# =======================================================================================


def save_tpi_adata_check_results(
    adata_path: str | PathLike, result: Mapping[str, Any], *, raw_data_status: Any = None
) -> bool:
    """Add ``check_raw`` to the adata file (save part of ``tpi_adata_check.m``).

    MATLAB adds ``check_raw`` (``passed``, ``roi_similarity``, ``act_similarity``) with
    ``add_variable`` and saves: when the file already has ``check_raw``, ``add_variable``
    skips it with a warning and nothing is written. Here the same.

    Parameters
    ----------
    adata_path
        Existing adata file.
    result
        :func:`calliopy.twophoton.tpi_adata_check` output (``MatStruct(check_raw=...)``).
    raw_data_status
        Passed to :meth:`calliopy.io.WritableAdata.save_changes`. MATLAB writes the new
        status to the ExpLog database (``update_raw_data_status``) just before saving;
        that update is not ported.

    Returns
    -------
    bool
        Whether ``check_raw`` was written.
    """
    ad = WritableAdata(_existing_v73(adata_path))
    if "check_raw" in ad:
        warnings.warn(
            'Variable "check_raw" already exists. Use assignment to modify.', stacklevel=2
        )
        warnings.warn("No variables have been modified. Nothing to save.", stacklevel=2)
        return False
    ad.add_variable(check_raw=MatStruct(result["check_raw"]))
    return ad.save_changes(raw_data_status=raw_data_status)


def save_check_raw_passed(adata_path: str | PathLike) -> bool:
    """Set ``check_raw.passed = 1`` in an adata file (the "y" answer of
    ``11_TwoPhoton/tpi_alignment_visualize.m`` and ``12_Widefield/wid_alignment_visualize.m``).

    ``check_raw`` is read, ``passed`` set to 1 (double) and the variable saved with
    ``-append`` semantics (:func:`calliopy.io.update_mat`); the other fields are kept.

    Returns
    -------
    bool
        False (with a warning) when the file has no ``check_raw`` ("not checked yet", as
        ``wid_alignment_visualize``; ``tpi_alignment_visualize`` failed instead).

    Notes
    -----
    The interactive question and the figures are the caller's
    (:func:`calliopy.twophoton.tpi_inspection_flags`,
    :func:`calliopy.twophoton.plot_tpi_activity_similarity`, widefield
    :func:`calliopy.widefield.compare_wid_frames`).
    """
    return _set_check_raw_passed(adata_path)


# =======================================================================================
# registration fixes (register_exp_downsample / avgplanes / fixoutliers, correct_dxdy)
# =======================================================================================


def _registration_log_entry(tail: str, now: datetime.datetime | None = None) -> str:
    """``[datestr(now) ' - SVN revision nbr: ' rev ' : ' tail]`` of the registration fixes.

    MATLAB writes the SVN revision of its own file; calliopy writes
    ``calliopy-<version>`` there.
    """
    return f"{_datestr(now)} - SVN revision nbr: calliopy-{_calliopy_version()} : {tail}"


def _append_log(existing: Any, entry: str) -> np.ndarray:
    """MATLAB ``registration_log{end+1} = entry`` (``registration_log{1}`` when absent)."""
    if existing is None:
        return _cell_row([entry])
    if isinstance(existing, np.ndarray) and existing.dtype == object and existing.ndim == 2:
        items = list(existing.ravel(order="F"))
        if existing.shape[1] == 1 and existing.shape[0] > 1:
            out = np.empty((len(items) + 1, 1), dtype=object)
            for i, v in enumerate([*items, entry]):
                out[i, 0] = v
            return out
        if existing.shape[0] <= 1:
            return _cell_row([*items, entry])
    if isinstance(existing, np.ndarray) and existing.size == 0:
        return _cell_row([entry])
    raise ValueError("registration_log in this file is not a cell vector; MATLAB fails as well")


def _save_registration_fix(
    adata_path: str | PathLike,
    variables: dict[str, Any],
    log_tail: str,
    *,
    new_file: Mapping[str, Any] | None = None,
) -> Path:
    """Write the variables of a registration fix into an existing or new adata file.

    MATLAB loads the whole file, changes some fields and saves the whole file again;
    replacing only those variables gives the same variables.
    """
    path = Path(adata_path)
    entry = _registration_log_entry(log_tail)
    if path.is_file():
        old = _read_variables(path, ["registration_log"])
        out = {"registration_log": _append_log(old.get("registration_log"), entry), **variables}
        update_mat(path, out)
        logger.info("----Done fixing your registration...----")
        return path
    if new_file is None:
        raise FileNotFoundError(f"adata file not found: {path}")
    out = {**new_file, "registration_log": _cell_row([entry])}
    n_layers = int(new_file.get("_n_layers", 1))
    out.pop("_n_layers", None)
    out["nbr_frames"] = variables.pop("nbr_frames")
    if n_layers < 2:
        out["ROIs"], out["bv"], out["np"] = _empty_struct(), _empty_struct(), _empty_struct()
    else:
        for name in ("ROIs", "bv", "np"):
            out[name] = _cell_row([_empty_struct() for _ in range(n_layers)])
    out.update(variables)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_mat(path, out)
    logger.info("---- NO AData file was found. Generated new one. ----")
    return path


def _new_file_vars(fnames, aux_files, mouse_id, user_id, n_layers) -> dict[str, Any] | None:
    given = [v is not None for v in (fnames, aux_files, mouse_id, user_id)]
    if not any(given):
        return None
    if not all(given):
        raise ValueError("a new adata file needs fnames, aux_files, mouse_id and user_id")
    return {"fnames": fnames, "aux_files": aux_files, "mouse_id": str(mouse_id),
            "userID": str(user_id), "_n_layers": n_layers}


def _template_range(template_frames: Sequence[int]) -> str:
    tf = [int(t) for t in template_frames]
    return f"{tf[0] + 1}:{tf[-1] + 1}"


def _registration_variables(result: Mapping[str, Any]) -> dict[str, Any]:
    if "nbr_frames" not in result:
        raise ValueError("result has no nbr_frames: pass nbr_frames to the registration "
                         "function (MATLAB always stores it)")
    return {
        "nbr_frames": _float_row(result["nbr_frames"]),
        "dx": _layers_cell(result["dx"]),
        "dy": _layers_cell(result["dy"]),
        "act_map": _layers_cell(result["act_map"]),
        "template": _layers_cell(result["template"]),
        "template_sec": _layers_cell(result["template_sec"]),
    }


def save_register_exp_downsample_results(
    adata_path: str | PathLike,
    result: Mapping[str, Any],
    *,
    fta: int = 3,
    template_frames: Sequence[int] = range(10, 30),
    fnames: Any = None,
    aux_files: Any = None,
    mouse_id: str | None = None,
    user_id: str | None = None,
) -> Path:
    """Save a downsampled re-registration (``21_Analysis/register_exp_downsample.m``).

    Existing file: ``registration_log`` gets the entry ``'<date> - SVN revision nbr:
    calliopy-<version> : register_exp_downsample - fta: <fta>, template_frames: <a>:<b>
    EOL'`` (frames 1-based, as MATLAB writes them) and ``nbr_frames``, ``dx``, ``dy``,
    ``act_map``, ``template``, ``template_sec`` are replaced; all other variables stay.

    No file at ``adata_path``: a new one is written (MATLAB names it with
    :func:`calliopy.io.adata_file_path_from_fnames`) with ``fnames``, ``aux_files``,
    ``mouse_id``, ``userID``, ``registration_log``, ``nbr_frames``, empty ``ROIs``,
    ``bv``, ``np`` (a ``struct`` for one layer, otherwise a cell of structs) and the
    registration variables.

    Parameters
    ----------
    adata_path
        The experiment's adata file.
    result
        :func:`calliopy.twophoton.register_exp_downsample` output, computed with
        ``nbr_frames``.
    fta, template_frames
        The arguments used (``template_frames`` **0-based**, as in calliopy), for the log.
    fnames, aux_files, mouse_id, user_id
        ExpLog metadata, needed only when the file does not exist.
    """
    variables = _registration_variables(result)
    tail = (f"register_exp_downsample - fta: {_num2str_int(fta)}, template_frames: "
            f"{_template_range(template_frames)} EOL")
    n_layers = len(result["dx"]) if isinstance(result["dx"], (list, tuple)) else 1
    return _save_registration_fix(
        adata_path, variables, tail,
        new_file=_new_file_vars(fnames, aux_files, mouse_id, user_id, n_layers),
    )


def save_register_exp_avgplanes_results(
    adata_path: str | PathLike,
    result: Mapping[str, Any],
    *,
    template_frames: Sequence[int] = range(10, 30),
    fta: int = 1,
    fnames: Any = None,
    aux_files: Any = None,
    mouse_id: str | None = None,
    user_id: str | None = None,
) -> Path:
    """Save a layer-average re-registration (``21_Analysis/register_exp_avgplanes.m``).

    As :func:`save_register_exp_downsample_results`; the log entry is
    ``'... : register_exp_avgplanes - template_frames: <a>:<b> fta: <fta> EOL'``. MATLAB's
    ``manual_check`` runs do not save: do not call this function for them.
    """
    variables = _registration_variables(result)
    tail = (f"register_exp_avgplanes - template_frames: {_template_range(template_frames)} "
            f"fta: {_num2str_int(fta)} EOL")
    n_layers = len(result["dx"]) if isinstance(result["dx"], (list, tuple)) else 1
    return _save_registration_fix(
        adata_path, variables, tail,
        new_file=_new_file_vars(fnames, aux_files, mouse_id, user_id, n_layers),
    )


def save_register_exp_fixoutliers_results(
    adata_path: str | PathLike, result: Mapping[str, Any]
) -> bool:
    """Save outlier-corrected shifts (``21_Analysis/register_exp_fixoutliers.m``).

    When :func:`calliopy.twophoton.register_exp_fixoutliers` corrected something, the
    fields it returns are replaced (one layer: arrays; several: ``1 x nLayers`` cells) and
    ``registration_log`` gets ``'... : register_exp_fixoutliers - EOL'``. An empty result
    (nothing corrected) writes nothing, as MATLAB.

    Returns
    -------
    bool
        Whether the file was changed.
    """
    if not result:
        logger.info("----Done - no fixing necessary...----")
        return False
    variables = {k: _layers_cell(v) for k, v in result.items()}
    _save_registration_fix(_existing_v73(adata_path), variables, "register_exp_fixoutliers - EOL")
    return True


def _merge_layers(file_value: Any, new: Any) -> Any:
    """Per-layer list with ``None`` for untouched layers -> the file's cell with the new
    layers put in (MATLAB ``curr_file_struct.x{rnd} = ...``)."""
    if not isinstance(new, (list, tuple)):
        return new
    new = list(new)
    if all(v is not None for v in new):
        return _cell_row(new)
    base = _cells(file_value) if file_value is not None else []
    out = base + [None] * max(0, len(new) - len(base))
    for i, v in enumerate(new):
        if v is not None:
            out[i] = v
    return _cell_row([np.zeros((0, 0)) if v is None else v for v in out])


def save_correct_dxdy_results(
    adata_path: str | PathLike,
    result: Mapping[str, Any],
    *,
    z_plane: int | None = None,
    z_to_correct: Sequence[int] = (),
) -> Path:
    """Save corrected registration shifts (``21_Analysis/correct_dxdy.m``).

    ``dx``, ``dy`` and, when recomputed, ``act_map``, ``template``, ``act_map_sec``,
    ``template_sec`` from :func:`calliopy.twophoton.correct_dxdy` replace the file's
    variables; layers the result leaves ``None`` (``z_plane`` mode) keep the file's value.
    ``registration_log`` gets the MATLAB entry:
    ``'... : correct_dxdy - z_plane, <z>, <layers>, : EOL'`` (1-based layer numbers, MATLAB
    ``num2str``) or ``'... : correct_dxdy - man_corr, : EOL'``.

    Parameters
    ----------
    adata_path
        Existing adata file.
    result
        :func:`calliopy.twophoton.correct_dxdy` output.
    z_plane, z_to_correct
        The **0-based** arguments given to ``correct_dxdy`` in ``z_plane`` mode; ``None``
        for ``man_corr`` mode.

    Notes
    -----
    Differences from MATLAB: in ``man_corr`` mode with numeric (one-layer) shifts MATLAB
    writes the shift values into the log through ``num2str`` of a column vector, which
    comes out as a column-major character dump; the log holds ``'man_corr, '`` only.
    """
    path = _existing_v73(adata_path)
    if z_plane is None:
        tail = "correct_dxdy - man_corr, : EOL"
    else:
        layers = [int(z) + 1 for z in z_to_correct]
        tail = f"correct_dxdy - z_plane, {int(z_plane) + 1}, {_num2str_int(layers)}, : EOL"
    names = [n for n in ("act_map", "template", "template_sec", "act_map_sec", "dx", "dy")
             if n in result]
    old = _read_variables(path, names)
    variables = {n: _merge_layers(old.get(n), result[n]) for n in names}
    return _save_registration_fix(path, variables, tail)


# =======================================================================================
# SpineQuest (04_Calliope/kalliope_callback_SpineQuest_*)
# =======================================================================================


def _set_or_add(ad: WritableAdata, name: str, value: Any) -> None:
    if name in ad:
        ad[name] = value
    else:
        ad.add_variable({name: value})


def save_detected_spines(
    adata_path: str | PathLike,
    spines: Sequence[Any],
    dendrite: Sequence[Any],
    *,
    raw_data_status: Any = None,
) -> bool:
    """Save DeepD3 spine detections (``kalliope_callback_SpineQuest_detect_spines.m``).

    ``spines``: ``1 x nRegions`` cell, each a struct array with ``indices`` (one element
    per spine, from :func:`calliopy.twophoton.spines_from_roi_map`); ``dendrite``:
    ``1 x nRegions`` cell of index columns. Both are set or added and saved with
    :meth:`calliopy.io.WritableAdata.save_changes`, replacing earlier values, as MATLAB.

    Parameters
    ----------
    adata_path
        The experiment's adata file.
    spines, dendrite
        Per spiny region: the spine structs and the dendrite indices (1-based, as stored).
    raw_data_status
        Passed to :meth:`~calliopy.io.WritableAdata.save_changes`.

    Notes
    -----
    A region without spines is written as ``[]``: MATLAB stores an empty struct array
    with the field ``indices``, which calliopy cannot write. Reading code that uses
    ``numel`` or loops over the elements behaves the same.
    """
    if len(spines) != len(dendrite):
        raise ValueError("spines and dendrite need one entry per region")
    ad = WritableAdata(_existing_v73(adata_path))
    _set_or_add(ad, "spines", _cell_row([_struct_array(s) for s in spines]))
    _set_or_add(ad, "dendrite", _cell_row(
        [np.asarray(d, dtype=np.float64).reshape(-1, 1) for d in dendrite]))
    return ad.save_changes(raw_data_status=raw_data_status)


_OLD_NAME = re.compile(r"mean_data-S1-T(\d+)\.mat")


def save_spiny_regions(
    adata_path: str | PathLike,
    spiny_regions: Sequence[Any],
    *,
    overwrite: bool = False,
    rename_old_name: bool = True,
    raw_data_status: Any = None,
) -> Path:
    """Save selected spiny regions (``kalliope_callback_SpineQuest_select_regions.m``).

    ``spiny_regions`` (``1 x nRegions`` cell of ``3 x 3`` double, 1-based ``(row, column,
    z)``) is set or added and saved. Afterwards, a file still named with the old
    convention ``mean_data-S1-T<ExpID>.mat`` is renamed to ``Adata-S1-T<ExpID>.mat``, as
    MATLAB.

    Parameters
    ----------
    adata_path
        The experiment's adata file.
    spiny_regions
        :func:`calliopy.twophoton.spiny_regions_from_marks` or
        :func:`calliopy.twophoton.propagate_spiny_regions` output.
    overwrite
        Replace existing ``spiny_regions`` (MATLAB asks "do you want to overwrite? 1 for
        yes"). When false and the file has them, nothing is written (warning).
    rename_old_name
        Rename an old-convention file after saving.
    raw_data_status
        Passed to :meth:`~calliopy.io.WritableAdata.save_changes`.

    Returns
    -------
    pathlib.Path
        The file's path after the rename.

    Notes
    -----
    Differences from MATLAB:

    - MATLAB's test of the answer, ``if ~num2str(answer)==1``, is false for every
      numeric answer, so existing regions were always overwritten; here ``overwrite``
      decides, as the question intends.
    - The rename refuses to replace an existing ``Adata-S1-T<ExpID>.mat`` (MATLAB's
      ``movefile`` replaced it) and logs an error, as MATLAB printed one when the rename
      failed.
    """
    ad = WritableAdata(_existing_v73(adata_path))
    if "spiny_regions" in ad and not overwrite:
        warnings.warn(f"{ad.path.name} already has spiny_regions; not overwritten", stacklevel=2)
        return ad.path
    value = _cell_row([np.asarray(r, dtype=np.float64) for r in spiny_regions])
    _set_or_add(ad, "spiny_regions", value)
    ad.save_changes(raw_data_status=raw_data_status)
    path = ad.path
    if rename_old_name and "mean_data" in path.name:
        m = _OLD_NAME.fullmatch(path.name)
        target = path.with_name(f"Adata-S1-T{m.group(1)}.mat") if m else None
        if target is None:
            logger.error("Error: cannot rename %s (not mean_data-S1-T<ExpID>.mat)", path.name)
        elif target.exists():
            logger.error("Error: cannot rename %s: %s exists", path.name, target.name)
        else:
            path.rename(target)
            logger.info("File renamed successfully!")
            path = target
    return path


def save_spine_tracking(
    adata_path: str | PathLike,
    spine_tracking: Sequence[Sequence[Any]],
    *,
    raw_data_status: Any = None,
) -> bool:
    """Save spine tracking on the site origin's adata (``..._SpineQuest_track_spines.m``).

    ``spine_tracking`` (from :func:`calliopy.twophoton.spine_tracking_export`): per region
    a struct array with ``indices`` (``[]`` when no spine matched, as MATLAB) and
    ``presence`` (``1 x nTimepoints``); written as a ``1 x nRegions`` cell, set or added,
    and saved.

    Parameters
    ----------
    adata_path
        Adata file of the site's origin ExpID (MATLAB saves nothing when the origin is
        not among the tracked timepoints: the caller's check).
    spine_tracking
        Per region the list of spine structs.
    raw_data_status
        Passed to :meth:`~calliopy.io.WritableAdata.save_changes`.

    Notes
    -----
    A region without tracked spines is written as ``[]`` (see :func:`save_detected_spines`).
    """
    regions = []
    for region in spine_tracking:
        items = []
        for s in _struct_items(region):
            st = MatStruct(s)
            idx = np.asarray(st.get("indices", np.zeros((0, 0))))
            st["indices"] = np.zeros((0, 0)) if idx.size == 0 else idx
            items.append(st)
        regions.append(_struct_array(items))
    ad = WritableAdata(_existing_v73(adata_path))
    _set_or_add(ad, "spine_tracking", _cell_row(regions))
    return ad.save_changes(raw_data_status=raw_data_status)
