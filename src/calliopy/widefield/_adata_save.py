"""Saving widefield results into adata files.

The adata-writing parts of calliope ``12_Widefield/register_wid.m``,
``propagate_tform_and_ROIs_from_expid.m``, ``calc_widefield_act.m``,
``wid_adata_check.m`` and ``wid_alignment_visualize.m``. Each saver takes the adata
path and the result of the matching function of :mod:`calliopy.widefield` and writes the
variables MATLAB writes, with MATLAB's names and layout.

Transforms
----------
MATLAB stores ``tform`` (and ``tform_ch525``, ``tform_ch610``, ``tform_ch_map``) as a
MATLAB geometric-transform object; calliopy works with its 3x3 matrix ``tform.A`` and
cannot write MATLAB objects. The savers write the matrix as a 3x3 double under the MATLAB
name (``[]`` for an origin experiment, as MATLAB). calliopy reads it back as it reads
other matrices; MATLAB code that passes ``adata.tform`` to ``imwarp`` needs
``affinetform2d(adata.tform)`` (or ``affine2d(adata.tform')`` before R2022b) for such
files.

The ExpLog lookups, the same-mouse/same-user checks, ``overwrite_flag`` decisions based on
the ExpLog and the database updates (``update_raw_data_status``) are not ported: the
adata path is an argument.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from ..io import MatStruct, WritableAdata, update_mat
from ..io._adata_write import _as_cell, _existing_v73, _set_check_raw_passed

__all__ = [
    "save_calc_widefield_act_results",
    "save_check_raw_passed",
    "save_propagate_tform_results",
    "save_register_wid_results",
    "save_wid_adata_check_results",
]

logger = logging.getLogger(__name__)

_CALC_ACT_VARIABLES = ("ROIs", "ROIs_wid", "meta_info", "registration_parameters")


def _tform_value(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 0))
    if arr.shape != (3, 3):
        raise ValueError(f"tform must be a 3x3 matrix (tform.A) or empty, got shape {arr.shape}")
    return arr


def save_register_wid_results(
    adata_path: str | PathLike,
    result: Mapping[str, Any],
    *,
    overwrite: bool = False,
    create_dir: bool = True,
    adata_root: str | PathLike | None = None,
) -> Path:
    """Create the adata file of a registered widefield experiment (``register_wid.m``).

    All variables of :func:`calliopy.widefield.register_wid` (``meta_info``,
    ``first_frame<ending>``, ``tform<ending>``, ``registration_parameters<ending>``, and
    ``tform_ch_map`` / ``registration_parameters_ch_map`` for a two-colour origin) are
    added in that order to a new file made with :meth:`calliopy.io.WritableAdata.create`
    and saved.

    Parameters
    ----------
    adata_path
        The new file (MATLAB: :func:`calliopy.io.adata_file_path`).
    result
        :func:`calliopy.widefield.register_wid` output.
    overwrite
        Replace an existing file (MATLAB ``overwrite_flag`` 1, or 2 after the caller found
        the stored status ``'FAILED'``). False raises ``FileExistsError``, MATLAB's
        behaviour for ``overwrite_flag = 0`` (skip).
    create_dir, adata_root
        Passed to :meth:`calliopy.io.WritableAdata.create`.

    Returns
    -------
    pathlib.Path

    Notes
    -----
    Transforms are written as 3x3 double matrices (see the module docstring).

    Differences from MATLAB: with ``overwrite_flag`` 1 or 2 and an existing file, MATLAB's
    ``Adata.create`` raises ``Adata:FileExists``, so MATLAB never replaced a file (the
    flags cannot have worked since ``register_wid`` moved to ``Adata.create``);
    ``overwrite=True`` does what the flags describe.
    """
    ad = WritableAdata.create(adata_path, create_dir=create_dir, adata_root=adata_root,
                              overwrite=overwrite)
    values = {}
    for name, value in result.items():
        values[name] = _tform_value(value) if name.startswith("tform") else value
    ad.add_variable(values)
    ad.save_changes()
    logger.info("Saved Adata (%s)", ad.path.name)
    return ad.path


def save_propagate_tform_results(adata_path: str | PathLike, result: Any) -> Path:
    """Save a copied transform (``propagate_tform_and_ROIs_from_expid.m``).

    ``tform``, ``registration_parameters`` and, when the origin had ROIs, ``ROIs`` (without
    ``activity``) from :func:`calliopy.widefield.propagate_tform` are written into the
    inheriting experiment's adata file with ``-append`` semantics
    (:func:`calliopy.io.update_mat`); other variables stay.

    Parameters
    ----------
    adata_path
        Existing adata file of the inheriting experiment.
    result
        :class:`calliopy.widefield.PropagateTformResult`.

    Notes
    -----
    ``tform`` is written as a 3x3 double (see the module docstring). Not ported: the
    lookups and the checks that origin and inheriting experiment belong to the same mouse
    and user (the caller's).
    """
    path = _existing_v73(adata_path)
    variables: dict[str, Any] = {
        "tform": _tform_value(result.tform),
        "registration_parameters": MatStruct(result.registration_parameters),
    }
    if result.ROIs is not None:
        variables["ROIs"] = _as_cell(result.ROIs)
    update_mat(path, variables)
    logger.info("Saved adata")
    return path


def save_calc_widefield_act_results(adata_path: str | PathLike, result: Mapping[str, Any]) -> Path:
    """Save widefield ROI activity (``calc_widefield_act.m``).

    The variables of :func:`calliopy.widefield.calc_widefield_act` are written with
    ``-append`` semantics, as MATLAB's three saving modes: ``ROIs`` and
    ``registration_parameters`` (mode 1), plus ``ROIs_wid`` and ``meta_info`` (mode 2) or
    ``meta_info`` (mode 3). Other variables stay.

    Raises
    ------
    ValueError
        ``result`` holds a variable MATLAB does not save here.

    Notes
    -----
    Kept from MATLAB (suspected bug): for two-channel data MATLAB saves a new
    ``registration_parameters`` that holds only ``activity_calculated = 1`` and leaves
    ``registration_parameters_ch525`` / ``_ch610`` unchanged; ``calc_widefield_act``
    reproduces that and this function saves what it returns.
    """
    path = _existing_v73(adata_path)
    extra = [k for k in result if k not in _CALC_ACT_VARIABLES]
    if extra:
        raise ValueError(f"calc_widefield_act does not save {extra}")
    variables = {k: _as_cell(result[k]) if k in ("ROIs", "ROIs_wid") else result[k]
                 for k in _CALC_ACT_VARIABLES if k in result}
    update_mat(path, variables)
    return path


def save_wid_adata_check_results(adata_path: str | PathLike, result: Mapping[str, Any]) -> Path:
    """Save ``check_raw`` (save part of ``wid_adata_check.m``).

    ``check_raw`` (``passed``, ``img_similarity``, ``act_similarity``) from
    :func:`calliopy.widefield.wid_adata_check` is written with ``-append`` semantics,
    replacing an earlier one, as MATLAB. The ExpLog update (``update_raw_data_status``)
    that MATLAB does first is not ported.
    """
    path = _existing_v73(adata_path)
    update_mat(path, {"check_raw": MatStruct(result["check_raw"])})
    return path


def save_check_raw_passed(adata_path: str | PathLike) -> bool:
    """Set ``check_raw.passed = 1`` (the "y" answer of ``wid_alignment_visualize.m``).

    The same function as :func:`calliopy.twophoton.save_check_raw_passed`.

    Returns
    -------
    bool
        False (with a warning) when the file has no ``check_raw``.
    """
    return _set_check_raw_passed(adata_path)
