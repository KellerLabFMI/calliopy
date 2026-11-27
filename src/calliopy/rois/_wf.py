"""Widefield ROIs of RoiQuest: propagation to other recordings and what is saved.

Port of calliope ``08_RoiQuest/+utils/wf_transformROis.m``, ``wf_transformROis_2ch.m`` and
``wfSaveRois.m`` (Leonardo Lupori / lab, 2024). Transforms are 3x3 matrices in the
convention of :mod:`calliopy.widefield` (MATLAB ``tform.A``, column vectors
``[x; y; 1]``, x = column, y = row, 1-based pixel centres); for a legacy ``affine2d`` pass
``T.T``. calliopy.io cannot decode MATLAB transform objects stored in adata files, so the
caller passes their matrices.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np

from calliopy._matlab import mround
from calliopy.io import MatStruct

from ._matching import RoiSaveError
from ._roi import (
    _EMPTY,
    _cell_row,
    _find,
    _ind2sub,
    _lin,
    _struct_array,
    _struct_list,
    _sub2ind,
    rois2struct,
)

__all__ = [
    "CORNER_ROI_NAMES_RQ",
    "wf_rois_struct",
    "wf_save_rois",
    "wf_transform_rois",
    "wf_transform_rois_2ch",
]

CORNER_ROI_NAMES_RQ = ("upper left corner", "upper right corner", "bottom left corner",
                       "bottom right corner")
"""Names of the corner ROIs that ``wfSaveRois`` removes (``calc_act`` creates them)."""


def _transform_field(rois: list, field: str, size_hw, tform) -> None:
    from calliopy.widefield import transform_points, warp_image

    a_inv = np.linalg.inv(np.asarray(tform, dtype=np.float64))
    hw = (int(size_hw[0]), int(size_hw[1]))
    for k in (0, 1):  # reference points: transform the point, not an image
        r, c = _ind2sub(hw, rois[k][field])
        if r.size != 1:
            raise ValueError("the first two ROIs must be single reference points")
        x, y = transform_points(a_inv, np.array([[c[0], r[0]]], dtype=np.float64))[0]
        try:
            rois[k][field] = np.asarray(_sub2ind(hw, mround(y), mround(x)),
                                        dtype=np.float64).reshape(1, 1)
        except ValueError:
            raise ValueError("One of your reference points moved out of the border of the "
                             "image - this data is too bad to be used") from None
    for k in range(2, len(rois)):
        img = np.zeros(hw)
        img.T.flat[_lin(rois[k][field]) - 1] = 1
        out = _find(warp_image(img, a_inv, hw, order=1) != 0)
        if out.size == 0:
            warnings.warn(f"ROI with ind {k + 1} is beyond template boundaries. Will assign "
                          "empty indices", stacklevel=3)
            out = _EMPTY.copy()
        rois[k][field] = out


def wf_transform_rois(rois_orig, size_template_orig, tform) -> np.ndarray:
    """Propagate widefield ROIs of the first recording to another one, MATLAB
    ``utils.wf_transformROis``.

    ``tform`` maps the other recording onto the first (as stored in its adata); the ROIs are
    mapped with ``invert(tform)``. ROIs 1 and 2 are the reference points (one pixel each):
    their pixel is transformed as a point and rounded. Every other ROI is drawn as a 0/1 image
    and warped (``imwarp`` with ``'SameAsInput'``, bilinear); every non-zero output pixel
    belongs to the new ROI.

    Parameters
    ----------
    rois_orig
        Widefield ROI struct array (fields as in the adata, ``indices`` 1-based).
    size_template_orig
        ``(height, width)`` of the template (the image size does not change within a site).
    tform
        3x3 matrix (see the module docstring).

    Returns
    -------
    numpy.ndarray
        ``(1, N)`` object array of copies of the ROI structs with new ``indices`` (``(N, 1)``
        columns, ``(1, 1)`` for the reference points, ``[]`` with a warning when a ROI leaves
        the image).

    Raises
    ------
    ValueError
        A reference point leaves the image (MATLAB ``error`` with the same message).

    Notes
    -----
    Unverified: :func:`calliopy.widefield.warp_image` documents that MATLAB's treatment of
    source points within half a pixel outside the image is not known exactly, so ROIs at
    the image border may differ by a pixel row/column.
    """
    rois = [MatStruct(s) for s in _struct_list(rois_orig)]
    _transform_field(rois, "indices", size_template_orig, tform)
    return _struct_array(rois)


def wf_transform_rois_2ch(rois_orig, size_template_orig, tform_525, tform_610) -> np.ndarray:
    """Two-channel version, MATLAB ``utils.wf_transformROis_2ch``: ``indices_ch525`` with
    ``tform_525`` and ``indices_ch610`` with ``tform_610``, each as
    :func:`wf_transform_rois`."""
    rois = [MatStruct(s) for s in _struct_list(rois_orig)]
    _transform_field(rois, "indices_ch525", size_template_orig, tform_525)
    _transform_field(rois, "indices_ch610", size_template_orig, tform_610)
    return _struct_array(rois)


def wf_rois_struct(rois, rois_ch610=None) -> np.ndarray:
    """Struct array of widefield ROIs as ``wfSaveRois`` saves it (``newROIs{1}``).

    Parameters
    ----------
    rois
        :class:`RoiWf` list (green channel for two-channel data), MATLAB ``app.ROIs{1}``.
    rois_ch610
        Red-channel :class:`RoiWf` list (``app.ROIs{2}``) for two-channel data, else ``None``.

    Returns
    -------
    numpy.ndarray
        ``(1, N)`` object array of MatStruct. Two channels: fields ``indices_ch525``,
        ``indices_ch610`` first, then the other ROI fields. Corner ROIs
        (:data:`CORNER_ROI_NAMES_RQ`) are removed.
    """
    s = _struct_list(rois2struct(list(rois)))
    if rois_ch610 is not None:
        red = _struct_list(rois2struct(list(rois_ch610)))
        if len(red) != len(s):
            raise ValueError("both channels must have the same number of ROIs")
        out = []
        for g, r in zip(s, red):
            t = MatStruct(indices_ch525=g["indices"], indices_ch610=r["indices"])
            t.update((k, v) for k, v in g.items() if k != "indices")
            out.append(t)
        s = out
    keep = [x for x in s if str(x.get("name", "")) not in CORNER_ROI_NAMES_RQ]
    return _struct_array(keep)


def _status_failed(rp) -> bool:
    return str(rp["status"]) == "FAILED"


def _without_corners(rois) -> list:
    return [x for x in _struct_list(rois) if str(x.get("name", "")) not in CORNER_ROI_NAMES_RQ]


def _first_cell(rois_var):
    if isinstance(rois_var, np.ndarray) and rois_var.dtype == object and rois_var.size and \
            not isinstance(rois_var.ravel(order="F")[0], Mapping):
        return rois_var.ravel(order="F")[0]
    if isinstance(rois_var, (list, tuple)):
        return rois_var[0]
    return rois_var


def wf_save_rois(rois, *, exp_group_ids, current_exp_id, template_hw,
                 roi_template_parameters, timepoints, overwrite_policy=None, rois_ch610=None,
                 overwrite_activity: bool = False) -> dict:
    """Variables ``wfSaveRois`` writes to each adata file of a widefield series.

    Parameters
    ----------
    rois
        :class:`RoiWf` list of the first recording (green channel for two channels).
    exp_group_ids
        ExpIDs of the series (``app.expGroup``), the first recording first.
    current_exp_id
        ExpID open in RoiQuest; must be the first of the group.
    template_hw
        ``(height, width)`` of the template.
    roi_template_parameters
        ``app.ROItemplate_parameters``, saved unchanged.
    timepoints
        Mapping ExpID -> adata variables (mapping, e.g. from :func:`calliopy.io.load_mat`),
        or ``None`` for an ExpID without adata file. Used: ``type`` (``'2ch_wf'`` for two
        channels), ``registration_parameters`` / ``registration_parameters_ch525`` /
        ``_ch610`` (``status``, ``activity_calculated``), ``ROIs`` (when present), ``tform`` /
        ``tform_ch525`` / ``tform_ch610`` as 3x3 matrices.
    overwrite_policy
        For a series: ``'keep'`` (MATLAB "Propagate without overwriting") or ``'overwrite'``
        ("Propagate and overwrite").
    rois_ch610
        Red-channel ROIs for two-channel data.
    overwrite_activity
        Single recording: the answer to "Overwriting ROI activity traces?".

    Returns
    -------
    dict
        ExpID -> MatStruct of the variables MATLAB writes into that adata file (``ROIs`` as a
        1x1 cell, ``ROItemplate_parameters`` and, when activity was calculated, the reset
        registration parameters). ExpIDs MATLAB skips are absent. Writing is deferred.

    Raises
    ------
    RoiSaveError
        Current ExpID is not the first of its group, no ROIs, existing activity without
        ``overwrite_activity``, or (``'keep'``) ROIs differing from those in the first file.
    ValueError
        Missing or unknown ``overwrite_policy`` for a series (MATLAB returned when the dialog
        was closed).

    Notes
    -----
    Kept as in MATLAB (suspected bug, reported): for two-channel data with activity already
    calculated, the reset registration parameters are written to new variables
    ``registration_parameters_525`` and ``registration_parameters_610`` instead of
    ``registration_parameters_ch525`` / ``_ch610``; they are returned under those names.
    """
    group = [int(v) for v in np.ravel(exp_group_ids)]
    if len(group) > 1 and current_exp_id != group[0]:
        raise RoiSaveError(f"The current ExpID ({current_exp_id}) is part of an experiment "
                           f"group but it's not the first ({group[0]}).")
    if rois is None or len(list(rois)) == 0:
        raise RoiSaveError("No ROIs to save.")
    two_ch = rois_ch610 is not None
    new = wf_rois_struct(rois, rois_ch610)
    if len(group) < 2:
        tp = timepoints.get(current_exp_id) if timepoints else None
        if tp is not None and "ROIs" in tp and not overwrite_activity:
            first = _struct_list(_first_cell(tp["ROIs"]))
            if first and "activity" in first[0]:
                raise RoiSaveError("ROIs in the AData file have activity traces; "
                                   "ROI saving aborted.")
        return {current_exp_id: MatStruct(ROIs=_cell_row([new]),
                                          ROItemplate_parameters=roi_template_parameters)}
    if overwrite_policy not in ("keep", "overwrite"):
        raise ValueError("overwrite_policy must be 'keep' or 'overwrite' for a series")
    out = {}
    fields = ("indices_ch525", "indices_ch610") if two_ch else ("indices",)
    for i, exp in enumerate(group):
        tp = timepoints.get(exp)
        if tp is None:
            continue
        is2 = str(tp.get("type", "")) == "2ch_wf"
        if is2:
            rp5 = MatStruct(tp["registration_parameters_ch525"])
            rp6 = MatStruct(tp["registration_parameters_ch610"])
            if _status_failed(rp5) or _status_failed(rp6):
                continue
        else:
            rp = MatStruct(tp["registration_parameters"])
            if _status_failed(rp):
                continue

        def transformed(exp=exp, tp=tp, is2=is2):
            if exp == group[0]:
                return new
            if is2:
                return wf_transform_rois_2ch(new, template_hw, tp["tform_ch525"],
                                             tp["tform_ch610"])
            return wf_transform_rois(new, template_hw, tp["tform"])

        if "ROIs" in tp:
            if i == 0 and overwrite_policy == "keep":
                old = _without_corners(_first_cell(tp["ROIs"]))
                cur = _struct_list(new)
                equal = len(old) == len(cur) and all(
                    np.array_equal(np.asarray(a[f]), np.asarray(b[f]))
                    for a, b in zip(cur, old) for f in fields)
                if not equal:
                    raise RoiSaveError("Current ROIs are different from ROIs in the first "
                                       "AData file. Propagating without overwriting is not "
                                       "allowed.")
            if overwrite_policy == "keep":
                continue
            v = MatStruct()
            if is2:
                if float(np.ravel(rp5["activity_calculated"])[0]):
                    rp5["activity_calculated"] = 0.0
                    rp6["activity_calculated"] = 0.0
                    v["registration_parameters_525"] = rp5
                    v["registration_parameters_610"] = rp6
            elif float(np.ravel(rp["activity_calculated"])[0]):
                rp["activity_calculated"] = 0.0
                v["registration_parameters"] = rp
            v["ROIs"] = _cell_row([transformed()])
            v["ROItemplate_parameters"] = roi_template_parameters
            out[exp] = v
        else:
            out[exp] = MatStruct(ROIs=_cell_row([transformed()]),
                                 ROItemplate_parameters=roi_template_parameters)
    return out
