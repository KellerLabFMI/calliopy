"""ROI templates: shared geometry, Allen-area assignment, transforms and plotting grids.

Port of the shared parts of calliope ``12_Widefield/ROItemplate_library/Build_code``
(``ROItemplate_generate_*.m``), ``12_Widefield/angles_from_top_left.m``,
``08_RoiQuest/+utils/wf_transform_ROIs_from_ROItemplate.m`` and
``12_Widefield/Plotting_templates/make_plotting_grid_ROIs_template.m``. The eight generators
themselves are thin functions in :mod:`calliopy.widefield._template_builders`.

Conventions
-----------
- A template is the ``struct_ROIs`` :class:`~calliopy.io.MatStruct` that MATLAB saves; its
  ``ROIs`` field is an object array of shape ``(1, nROIs)`` holding one ``MatStruct`` per ROI
  (a MATLAB 1xN struct array), with the MATLAB field names and order.
- ROI ``indices`` are MATLAB linear indices: **1-based, column-major** into an image of
  ``SIZE_TEMPLATE`` (rows x columns), float64 as MATLAB stores them, as a column ``(n, 1)``.
  A 1x1 value (the reference-point pseudo-ROIs) is a numpy scalar, ``[]`` is an empty
  ``(0, 0)`` array, as everywhere in calliopy.
- Points written ``VEC_*`` / ``ref_point*`` are MATLAB axes (``ginput``) coordinates
  ``(x, y) = (column, row)``, 1-based.
- The MAT-files of this folder are MATLAB v5/v7 files, which :func:`calliopy.io.load_mat`
  (v7.3/HDF5 only) does not read; the loaders here use :func:`scipy.io.loadmat`.

None of these functions has been run against MATLAB; see the docstrings for what is checked
against MATLAB-made template files and what is not.
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from collections.abc import Sequence
from os import PathLike
from typing import Any

import numpy as np
import scipy.io
import scipy.ndimage

from calliopy._matlab import imgaussfilt, linspace, mround
from calliopy.io import MatStruct
from calliopy.widefield._transform import fit_similarity, transform_points, warp_image

logger = logging.getLogger(__name__)

__all__ = [
    "ALLEN_REF_POINTS",
    "allen_area_borders",
    "allen_flat_map",
    "angles_from_top_left",
    "assign_allen_areas",
    "load_allen_flat_map",
    "load_allen_ontology",
    "load_plotting_grid_template",
    "load_roi_template",
    "make_plotting_grid_rois_template",
    "plot_roi_template",
    "roi_template_colour_image",
    "transform_rois_from_template",
]

_PathArg = str | PathLike

#: Reference points on the Allen flat map (MATLAB axes coordinates ``(x, y)``, 1-based) as
#: hard-coded in the MATLAB generators and in make_plotting_grid_ROIs_template.m.
ALLEN_REF_POINTS = {
    "lambda": (230.0, 372.0),
    "bregma": (229.0, 204.0),
    "icv": (229.0, 78.0),
    "cranial_window_posterior": (315.0, 390.0),
    "cranial_window_anterior": (315.0, 265.0),
}

# MATLAB fixes one pixel of the flat map (1-based (179, 259)) from MOp to MOs.
_FLATMAP_FIX = ((178, 258), 993.0)

# ======================================================================================
# Small helpers
# ======================================================================================


def _empty() -> np.ndarray:
    """MATLAB ``[]`` (a fresh ``(0, 0)`` float64 array)."""
    return np.empty((0, 0))


def _mat_num(value) -> Any:
    """A MATLAB double as calliopy returns it: 1x1 -> numpy scalar, else a 2-D array."""
    a = np.asarray(value, dtype=np.float64)
    if a.size == 1 and a.ndim <= 2:
        return np.float64(a.reshape(()))
    return a


def _mat_col(values) -> Any:
    """A MATLAB column vector of doubles: ``(n, 1)``; 1 element -> scalar."""
    a = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return np.float64(a[0, 0]) if a.size == 1 else a


def _row(values) -> np.ndarray:
    """A MATLAB 1xN double row ``(1, N)``."""
    return np.asarray(values, dtype=np.float64).reshape(1, -1)


def _sub2ind(shape, rows, cols) -> np.ndarray:
    """MATLAB ``sub2ind(shape, rows, cols)`` (1-based in, 1-based out), with range check."""
    r = np.asarray(rows, dtype=np.float64)
    c = np.asarray(cols, dtype=np.float64)
    if np.any(r < 1) or np.any(r > shape[0]) or np.any(c < 1) or np.any(c > shape[1]):
        raise ValueError("sub2ind: subscript out of range")
    if np.any(r != np.round(r)) or np.any(c != np.round(c)):
        raise ValueError("sub2ind: subscripts must be integers")
    return (c - 1) * shape[0] + r


def _ind2sub(shape, idx) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB ``[r, c] = ind2sub(shape, idx)`` for 2-D shapes, 1-based in and out."""
    i = np.asarray(idx, dtype=np.int64) - 1
    return i % int(shape[0]) + 1, i // int(shape[0]) + 1


def _indices_of(roi) -> np.ndarray:
    """1-based linear indices of one ROI as a flat int64 array."""
    v = np.asarray(roi["indices"], dtype=np.float64).ravel(order="F")
    return v.astype(np.int64)


def _roi_list(rois) -> list:
    """ROIs given as a (1, N) object array, a list or a single MatStruct -> list."""
    if isinstance(rois, dict):
        return [rois]
    if isinstance(rois, np.ndarray):
        return list(rois.ravel(order="F"))
    return list(rois)


def _struct_array(items: Sequence[dict]) -> np.ndarray:
    """List of MatStruct -> MATLAB 1xN struct array (object array of shape (1, N))."""
    out = np.empty((1, len(items)), dtype=object)
    for i, s in enumerate(items):
        out[0, i] = s
    return out


def new_roi(
    indices,
    *,
    type_=0.0,
    ml_index=None,
    ap_index=None,
    name: str = "",
    allen_area: str = "",
    allen_acronym: str = "",
    allen_index=None,
) -> MatStruct:
    """One ROI struct with the fields of the grid generators, in MATLAB's order.

    ``indices`` (1-based, column-major) is stored as a ``(n, 1)`` float64 column (scalar for
    one element); ``None`` for ``ML_index``/``AP_index``/``Allen_index`` means MATLAB ``[]``.
    """
    return MatStruct(
        indices=indices,
        type=_mat_num(type_) if type_ is not None else _empty(),
        shift=_row([0, 0]),
        ML_index=_empty() if ml_index is None else _mat_num(ml_index),
        AP_index=_empty() if ap_index is None else _mat_num(ap_index),
        name=name,
        Allen_area=allen_area,
        Allen_acronym=allen_acronym,
        Allen_index=_empty() if allen_index is None else _mat_num(allen_index),
    )


def reference_rois(size_template, points, names, *, fields: Sequence[str] | None = None):
    """The reference-point pseudo-ROIs that every generator puts first.

    Parameters
    ----------
    size_template
        ``(rows, cols)`` of the template.
    points
        Reference points in MATLAB axes coordinates ``(x, y)`` (1-based), e.g.
        ``[VEC_LAMBDA, VEC_BREGMA]``.
    names
        ``'Lambda'``, ``'Bregma'``, ... one per point.
    fields
        Field names of the ROI struct, in order (default: the grid generators' fields).
        Fields other than ``indices``, ``type``, ``shift`` and ``name`` are ``''`` for
        ``Allen_area``/``Allen_acronym`` and ``[]`` otherwise.

    Returns
    -------
    list of MatStruct
        ``indices`` = ``sub2ind(size_template, y, x)`` (1-based scalar), ``type`` 0,
        ``shift`` ``[0 0]``.
    """
    if fields is None:
        fields = list(new_roi(0.0).keys())
    out = []
    for (x, y), nm in zip(points, names):
        s = MatStruct()
        for f in fields:
            if f == "indices":
                s[f] = _mat_num(_sub2ind(size_template, y, x))
            elif f == "type":
                s[f] = np.float64(0.0)
            elif f == "shift":
                s[f] = _row([0, 0])
            elif f == "name":
                s[f] = nm
            elif f in ("Allen_area", "Allen_acronym"):
                s[f] = ""
            else:
                s[f] = _empty()
        out.append(s)
    return out


# ======================================================================================
# Shared geometry of the generators
# ======================================================================================


def _square_roi_indices(x: float, y: float, l_w: float, d_border: float, n_rows: int):
    """Linear indices of one grid square, as the MATLAB generators build them.

    MATLAB::

        for ind = 1:l_w-D_BORDER-1
            inds = cat(2, inds, (y-2+ind)*X_template + (x: x + l_w-D_BORDER-1));
        end

    i.e. rows ``x .. x+l_w-D_BORDER-1`` (``l_w-D_BORDER`` rows) and columns
    ``y .. y+l_w-D_BORDER-2`` (``l_w-D_BORDER-1`` columns), in column-major order.

    Parameters
    ----------
    x, y
        Top-left corner: ``x`` = row, ``y`` = column (1-based; MATLAB's "matrix
        coordinates" naming in the generators).
    l_w, d_border
        Grid pitch and border in pixels.
    n_rows
        ``X_template``, number of rows of the template.

    Returns
    -------
    numpy.ndarray
        float64 1-based linear indices, flat, in MATLAB's order. The square is
        ``(l_w-D_BORDER) x (l_w-D_BORDER-1)``: one column narrower than high, as in MATLAB.
    """
    n_side = int(l_w - d_border)
    rows = x + np.arange(n_side, dtype=np.float64)
    cols = y + np.arange(n_side - 1, dtype=np.float64)
    return (((cols - 1.0) * n_rows)[:, None] + rows[None, :]).ravel()


def cortex_grid(
    size_template,
    vec_lambda,
    vec_bregma,
    *,
    w_hem: float,
    l_back: float,
    l_front: float,
    n_w: int,
    tan_ant: float,
    tan_post: float,
    l_corner: float,
    d_border: float,
) -> list[MatStruct]:
    """Square grid ROIs over the dorsal cortex (lambda-bregma generators).

    Shared core of ``ROItemplate_generate_grid_whole_cortex_lambda_bregma``,
    ``..._lambda_ICV`` and ``..._grid_plus_eeg_circles_lambda_bregma``: grid construction,
    the three ROI-accepting conditions (frontal cone, internal triangle behind lambda, sharp
    corner around lambda) and the boundary check with a safety margin of ``l_w/2``.

    Parameters
    ----------
    size_template
        ``(rows, cols)`` = ``SIZE_TEMPLATE``.
    vec_lambda, vec_bregma
        ``VEC_LAMBDA``, ``VEC_BREGMA`` in MATLAB axes coordinates ``(x, y)``. Only
        ``vec_bregma[0]`` is used (``lbd = VEC_BREGMA(1) - VEC_LAMBDA(1)``).
    w_hem, l_back, l_front, n_w, tan_ant, tan_post, l_corner, d_border
        ``W_HEM``, ``L_BACK``, ``L_FRONT``, ``N_W``, ``TAN_ANT``, ``TAN_POST``,
        ``L_CORNER``, ``D_BORDER``.

    Returns
    -------
    list of MatStruct
        Accepted ROIs in MATLAB's order (grid rows outer, grid columns inner), fields of
        :func:`new_roi`: ``indices`` (1-based column-major, ``(n, 1)``; ``[]`` when the square
        lies within ``l_w/2`` of the template border), ``ML_index = ind_lambda_x - ind_x``,
        ``AP_index = ind_y - ind_lambda_y``, empty names and Allen fields.

    Notes
    -----
    Suspected MATLAB bug, kept: ``shift = min(abs(y_ticks - VEC_LAMBDA(1)))`` is subtracted
    from every tick, which moves the ticks *away* from lambda when the closest tick lies
    below lambda; no tick then equals lambda and ``AP_index`` becomes empty (MATLAB
    ``ind_y - []``). With the shipped parameters a tick falls on lambda (``shift = 0``).
    The same applies to ``x_ticks`` (``ind_lambda_x``), which is not shifted at all.
    """
    n_rows, n_cols = float(size_template[0]), float(size_template[1])
    lam1, lam2 = float(vec_lambda[0]), float(vec_lambda[1])
    lbd = float(vec_bregma[0]) - lam1
    l_w = float(mround(lbd * 2 * w_hem / n_w))
    n_l = float(mround(lbd * (1 + l_front + l_back) / l_w))
    x_ticks = mround(linspace(lam2 - l_w * n_w / 2, lam2 + l_w * (n_w / 2 - 1), n_w))
    y_ticks = mround(
        linspace(lam1 - lbd * l_back, lam1 - lbd * l_back + l_w * (n_l - 1), n_l)
    )
    shift = np.min(np.abs(y_ticks - lam1))
    y_ticks = y_ticks - shift
    ind_lambda_x = np.flatnonzero(x_ticks == lam2) + 1
    ind_lambda_y = np.flatnonzero(y_ticks == lam1) + 1
    front = math.ceil(lbd * (1 + tan_ant * w_hem))
    margin = l_w / 2
    rois = []
    for ix, x in enumerate(x_ticks, start=1):
        for iy, y in enumerate(y_ticks, start=1):
            a = abs(x - lam2 + l_w / 2)
            dy = y - lam1
            if not (
                dy <= -tan_ant * a + front
                and (y >= lam1 or dy >= -tan_post * a)
                and dy >= lbd * l_corner - tan_ant * a
            ):
                continue
            inds = _square_roi_indices(x, y, l_w, d_border, n_rows)
            r, c = _ind2sub((n_rows, n_cols), inds)
            if (
                np.any((r < 1 + margin) | (r > n_rows - margin))
                or np.any((c < 1 + margin) | (c > n_cols - margin))
            ):
                logger.info(
                    "ROI at x=%d, y=%d is beyond template boundaries; empty indices", x, y
                )
                indices = _empty()
            else:
                indices = _mat_col(inds)
            rois.append(
                new_roi(
                    indices,
                    ml_index=_index_diff(ind_lambda_x, ix, lambda_first=True),
                    ap_index=_index_diff(ind_lambda_y, iy, lambda_first=False),
                )
            )
    return rois


def _index_diff(ind_ref: np.ndarray, ind: int, *, lambda_first: bool):
    """``ind_ref - ind`` (or ``ind - ind_ref``) with MATLAB's empty propagation."""
    if ind_ref.size == 0:
        return np.empty((1, 0))
    v = ind_ref - ind if lambda_first else ind - ind_ref
    return v.astype(np.float64).reshape(1, -1)


def cranial_window_grid(
    size_template, vec_posterior, vec_anterior, *, n_w: int, d_border: float
) -> list[MatStruct]:
    """Square grid ROIs inside a circular cranial window (core of
    ``ROItemplate_generate_grid_cranial_window``).

    Parameters
    ----------
    size_template
        ``(rows, cols)``.
    vec_posterior, vec_anterior
        Reference points in MATLAB axes coordinates ``(x, y)``; the window diameter is their
        distance (rounded), its centre their rounded midpoint.
    n_w, d_border
        ``N_W`` squares along the diameter, ``D_BORDER`` pixels between squares.

    Returns
    -------
    list of MatStruct
        ROIs whose square centre lies within ``diam/2`` of the window centre, in MATLAB's
        order; ``ML_index = ind_post_x - ind_x``, ``AP_index = ind_y - ind_post_y``.
        There is no boundary check (as in MATLAB).
    """
    n_rows = float(size_template[0])
    vp = np.asarray(vec_posterior, dtype=np.float64)
    va = np.asarray(vec_anterior, dtype=np.float64)
    d = vp - va
    diam = float(mround(math.sqrt(d[0] ** 2 + d[1] ** 2)))
    centre = mround((vp + va) / 2)
    l_w = float(mround(diam / n_w))
    x_ticks = mround(linspace(vp[1] - l_w * n_w / 2, vp[1] + l_w * (n_w / 2 - 1), n_w))
    y_ticks = mround(linspace(vp[0], va[0], n_w))
    shift = np.min(np.abs(x_ticks - vp[1]))
    x_ticks = x_ticks - shift
    ind_post_x = np.flatnonzero(x_ticks == vp[1]) + 1
    ind_post_y = np.flatnonzero(y_ticks == vp[0]) + 1
    rois = []
    for ix, x in enumerate(x_ticks, start=1):
        for iy, y in enumerate(y_ticks, start=1):
            v0 = (y + l_w / 2) - centre[0]
            v1 = (x + l_w / 2) - centre[1]
            if math.sqrt(v0 * v0 + v1 * v1) <= diam / 2:
                inds = _square_roi_indices(x, y, l_w, d_border, n_rows)
                rois.append(
                    new_roi(
                        _mat_col(inds),
                        ml_index=_index_diff(ind_post_x, ix, lambda_first=True),
                        ap_index=_index_diff(ind_post_y, iy, lambda_first=False),
                    )
                )
    return rois


# Electrode positions of the NeuroNexus mouse EEG probe relative to bregma (mm), as typed in
# the MATLAB generators (multiplied by 100 there).
EEG_EL_X = (4.05, 2.24, 1.0, 4.13, 2.88, 1.13, 4.05, 2.88, 1.12, 3.5, 2.12, 1.93, 0.55, 1.5, 0.5,
            -0.5, -1.5, -0.55, -1.93, -2.12, -3.5, -1.12, -2.88, -4.05, -1.13, -2.88, -4.13,
            -1.0, -2.24, -4.05)  # fmt: skip
EEG_EL_Y = (-4.14, -4.14, -4.14, -3.04, -3.04, -3.04, -1.96, -1.96, -1.96, -0.48, -0.48, 1.04,
            1.04, 2.3, 2.3, 2.3, 2.3, 1.04, 1.04, -0.48, -0.48, -1.96, -1.96, -1.96, -3.04,
            -3.04, -3.04, -4.14, -4.14, -4.14)  # fmt: skip
EEG_ANGLES = ((0.1, 89.9), (90.1, 179.9), (-179.9, -90.1), (-89.9, -0.1))


def eeg_electrode_discs(
    size_template, vec_bregma, *, r_inner: float | None, r_outer: float,
    el_x=EEG_EL_X, el_y=EEG_EL_Y, scale: float = 100.0,
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """Pixels around each EEG electrode (shared part of the EEG generators).

    Parameters
    ----------
    size_template
        ``(rows, cols)``.
    vec_bregma
        ``VEC_BREGMA`` in MATLAB axes coordinates ``(x, y)``.
    r_inner, r_outer
        Ring: ``r_inner < dist < r_outer`` (``ElR``, ``ElRo``). Disc: ``r_inner=None`` and
        ``dist <= r_outer`` (``only_eeg_electrode_positions``, ``ElR``).
    el_x, el_y, scale
        Electrode positions (``ElX``, ``ElY`` before ``* 100``) and that factor.

    Returns
    -------
    (indices, centres)
        ``indices[k]``: float64 1-based column-major linear indices (MATLAB ``find``) of
        electrode ``k``; ``centres[k]``: ``MPOINT = [x_loc, y_loc]`` = (row, column), 1-based,
        with ``x_loc = round(-ElX + VEC_BREGMA(2))``, ``y_loc = round(ElY + VEC_BREGMA(1))``.
    """
    n_rows, n_cols = int(size_template[0]), int(size_template[1])
    ex = np.asarray(el_x, dtype=np.float64) * scale
    ey = np.asarray(el_y, dtype=np.float64) * scale
    rr = np.arange(1, n_rows + 1, dtype=np.float64)[:, None]
    cc = np.arange(1, n_cols + 1, dtype=np.float64)[None, :]
    out, centres = [], []
    for k in range(ex.size):
        x_loc = float(mround(-(ex[k]) + float(vec_bregma[1])))
        y_loc = float(mround(ey[k] + float(vec_bregma[0])))
        dist = np.sqrt((rr - x_loc) ** 2 + (cc - y_loc) ** 2)
        if r_inner is None:
            mask = dist <= r_outer
        else:
            mask = (r_inner < dist) & (dist < r_outer)
        out.append(np.flatnonzero(mask.ravel(order="F")).astype(np.float64) + 1)
        centres.append((x_loc, y_loc))
    return out, centres


def eeg_ring_rois(size_template, vec_bregma, *, r_inner=30.0, r_outer=40.0,
                  angles=EEG_ANGLES) -> list[MatStruct]:
    """Ring ROIs split into angular sectors around the 30 electrodes (the ring part of
    ``ROItemplate_generate_eeg_circles_lambda_bregma`` and ``..._grid_plus_eeg_...``).

    Returns
    -------
    list of MatStruct
        ``30 * len(angles)`` ROIs, electrode outer, sector inner; ``ML_index`` = channel
        (1..30), ``AP_index`` = sector (1..4), ``name`` ``'right hemisphere'`` for channels
        1..15, ``'left hemisphere'`` otherwise. Sector ``k`` keeps the ring pixels with
        ``lb <= angle <= ub`` (:func:`angles_from_top_left`, degrees).
    """
    discs, centres = eeg_electrode_discs(
        size_template, vec_bregma, r_inner=r_inner, r_outer=r_outer
    )
    rois = []
    for chan, (idx, mp) in enumerate(zip(discs, centres), start=1):
        hem = "right" if chan <= 15 else "left"
        theta = angles_from_top_left(idx, size_template, mp)
        for k, (lb, ub) in enumerate(angles, start=1):
            sel = idx[(theta >= lb) & (theta <= ub)]
            rois.append(
                new_roi(
                    sel.reshape(-1, 1) if sel.size != 1 else np.float64(sel[0]),
                    ml_index=chan,
                    ap_index=k,
                    name=f"{hem} hemisphere",
                )
            )
    return rois


def angles_from_top_left(idxs, matrix_size, mpoint) -> np.ndarray:
    """Angle of pixels around a centre pixel, port of ``12_Widefield/angles_from_top_left.m``.

    Pixels are turned into Cartesian coordinates with the origin at the top-left pixel and
    y pointing up (``x = col - 1``, ``y = rows - row``); the angle of each pixel seen from
    ``mpoint`` is ``atan2d(dy, dx)``.

    Parameters
    ----------
    idxs
        1-based linear indices (column-major) as a 1-D array or an ``(N, 1)`` column, or an
        ``(N, 2)`` array of 1-based ``[row, col]`` subscripts.
    matrix_size
        ``(rows, cols)`` of the matrix.
    mpoint
        Centre as 1-based ``(row, col)`` (not Cartesian).

    Returns
    -------
    numpy.ndarray
        float64 angles in degrees in ``[-180, 180]``, counter-clockwise from the +column
        direction (0 = right, 90 = up), one per pixel, shape ``(N,)``.

    Notes
    -----
    Differences from MATLAB: MATLAB treats a 1x2 *row* of linear indices as one
    ``[row, col]`` pair and returns a column; here a 1-D input is always linear indices and
    the result is 1-D.
    """
    a = np.asarray(idxs, dtype=np.float64)
    k = float(matrix_size[0])
    if a.ndim == 2 and a.shape[1] == 2:
        rows, cols = a[:, 0], a[:, 1]
    elif a.ndim <= 1 or (a.ndim == 2 and a.shape[1] == 1):
        rows, cols = (v.astype(np.float64) for v in _ind2sub(matrix_size, a.ravel()))
    else:
        raise ValueError(
            "idxs must be a column vector of linear indices or an Nx2 matrix of [row, col]"
        )
    dx = (cols - 1) - (float(mpoint[1]) - 1)
    dy = (k - rows) - (k - float(mpoint[0]))
    return np.degrees(np.arctan2(dy, dx))


# ======================================================================================
# Allen atlas
# ======================================================================================


def allen_flat_map(annotation_volume, *, fix_pixel: bool = True) -> np.ndarray:
    """Dorsal flat map of the Allen annotation: first nonzero label along the 2nd axis.

    Port of the block in the MATLAB generators (copied there from getFlatMap)::

        annotationVolume = permute(annotationVolume, [2 1 3]);
        flatMap(r,c) = first nonzero of annotationVolume(:,r,c)   (0 if none)
        flatMap(179, 259) = 993;

    Parameters
    ----------
    annotation_volume
        ``midAnnotation`` of ``annotation_25_mid_ctxOnly.mat``, ``(528, 320, 456)`` in
        MATLAB's axis order (any integer or float array of 3 dimensions).
    fix_pixel
        Apply MATLAB's fix of one MOp pixel inside MOs (1-based ``(179, 259)`` -> 993).

    Returns
    -------
    numpy.ndarray
        float64 ``(size(vol, 1), size(vol, 3))``.
    """
    vol = np.asarray(annotation_volume)
    if vol.ndim != 3:
        raise ValueError(f"annotation volume must be 3-D, got shape {vol.shape}")
    nz = vol != 0
    first = nz.argmax(axis=1)
    vals = np.take_along_axis(vol, first[:, None, :], axis=1)[:, 0, :]
    flat = np.where(nz.any(axis=1), vals, 0).astype(np.float64)
    if fix_pixel:
        (r, c), v = _FLATMAP_FIX
        if r < flat.shape[0] and c < flat.shape[1]:
            flat[r, c] = v
    return flat


def _as_flat_map(allen_map) -> np.ndarray:
    a = np.asarray(allen_map)
    if a.ndim == 3:
        return allen_flat_map(a)
    if a.ndim == 2:
        return a.astype(np.float64, copy=False)
    raise ValueError(f"expected an annotation volume (3-D) or a flat map (2-D), got {a.shape}")


def load_allen_flat_map(annotation_path: _PathArg, *, variable: str = "midAnnotation"):
    """Read ``annotation_25_mid_ctxOnly.mat`` (MATLAB v5/v7, read with
    :func:`scipy.io.loadmat`) and return :func:`allen_flat_map` of its volume.

    Parameters
    ----------
    annotation_path
        Path of the MAT-file.
    variable
        Name of the volume in the file (MATLAB ``m.midAnnotation``).

    Returns
    -------
    numpy.ndarray
        float64 flat map, ``(528, 456)`` for the calliope file.
    """
    data = scipy.io.loadmat(annotation_path, variable_names=[variable])
    if variable not in data:
        raise KeyError(f"{variable!r} not found in the annotation file")
    return allen_flat_map(data[variable])


def load_allen_ontology(structures_path: _PathArg) -> list[dict]:
    """Read the Allen ontology ``structures.json`` (MATLAB ``jsondecode(fileread(path))``).

    Returns
    -------
    list of dict
        One dict per structure with at least ``id``, ``name``, ``acronym``.
    """
    with open(structures_path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):  # tolerate {"msg": [...]} style exports
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    if not isinstance(data, list):
        raise TypeError("structures file does not hold a list of structures")
    return data


def _fit(moving_points, fixed_points) -> np.ndarray:
    return fit_similarity(
        np.asarray(moving_points, dtype=np.float64), np.asarray(fixed_points, dtype=np.float64)
    )


def _source_coords(tform: np.ndarray, out_shape, in_shape, rows=None, cols=None):
    """0-based source (row, col) of output pixels (a sub-grid ``rows`` x ``cols``, 1-based
    output coordinates, or the whole output) and the inside mask, with the same formula as
    :func:`calliopy.widefield._transform.warp_image`, so that :func:`_warp_mask_nonzero`
    can evaluate only an ROI's bounding box and still give ``warp_image(mask) != 0``."""
    h_in, w_in = int(in_shape[0]), int(in_shape[1])
    if rows is None:
        yy, xx = np.mgrid[1 : int(out_shape[0]) + 1, 1 : int(out_shape[1]) + 1].astype(
            np.float64
        )
    else:
        yy, xx = np.meshgrid(rows, cols, indexing="ij")
        yy, xx = yy.astype(np.float64), xx.astype(np.float64)
    ainv = np.linalg.inv(tform)
    src_c = ainv[0, 0] * xx + ainv[0, 1] * yy + ainv[0, 2] - 1.0
    src_r = ainv[1, 0] * xx + ainv[1, 1] * yy + ainv[1, 2] - 1.0
    tol = 1e-9
    inside = (src_r >= -tol) & (src_r <= h_in - 1 + tol)
    inside &= (src_c >= -tol) & (src_c <= w_in - 1 + tol)
    return np.clip(src_r, 0, h_in - 1), np.clip(src_c, 0, w_in - 1), inside, yy, xx


def warp_allen_map(flat_map, allen_points, target_points, size_target) -> np.ndarray:
    """Allen flat map in template coordinates.

    MATLAB::

        tform = fitgeotform2d([p1_allen; p2_allen], [p1_target; p2_target], "similarity");
        flatMap_transformed = imwarp(flatMap, tform, "OutputView", imref2d(size_target));

    Implemented with :func:`calliopy.widefield._transform.fit_similarity` and
    :func:`~calliopy.widefield._transform.warp_image` (bilinear). Pixels on area borders get
    non-integer (mixed) label values, as in MATLAB; inside a uniform area the label is
    returned exactly (``warp_image`` interpolates as ``a + t*(b - a)``), which the label
    count of :func:`assign_allen_areas` relies on.

    Parameters
    ----------
    flat_map
        2-D float label image (:func:`allen_flat_map`).
    allen_points, target_points
        Two ``(x, y)`` points each (MATLAB axes coordinates, 1-based).
    size_target
        ``(rows, cols)`` of the output.
    """
    tform = _fit(allen_points, target_points)
    return warp_image(np.asarray(flat_map, dtype=np.float64), tform, size_target, order=1)


def assign_allen_areas(
    rois,
    flat_map,
    ontology,
    allen_points,
    target_points,
    size_template,
    *,
    first: int = 2,
    name_hemisphere: bool = False,
) -> np.ndarray:
    """Give each ROI the Allen area that most of its pixels fall in.

    Port of the "Assigning ROIs to the Allen areas" block of the MATLAB generators: the
    flat map is warped into template coordinates (:func:`warp_allen_map`), and for each ROI
    (MATLAB ``for ind_ROI = 3:length(potROIs)``) the most frequent warped value
    (``groupcounts`` + ``max``: the smallest value among ties) becomes ``Allen_index``;
    ``Allen_area`` and ``Allen_acronym`` are the ``name`` and ``acronym`` of that id in the
    ontology.

    Parameters
    ----------
    rois
        ROI struct array ``(1, N)`` (or list); modified copies are returned.
    flat_map
        :func:`allen_flat_map` output, or the annotation volume (3-D) itself.
    ontology
        Parsed ``structures.json`` (:func:`load_allen_ontology`).
    allen_points, target_points
        Two reference points each, MATLAB axes ``(x, y)``: on the flat map
        (:data:`ALLEN_REF_POINTS`) and on the template (e.g. ``VEC_LAMBDA``, ``VEC_BREGMA``).
    size_template
        ``(rows, cols)`` of the template.
    first
        0-based position of the first ROI to assign (default 2: skip the two reference
        points; MATLAB ``3:length(potROIs)``).
    name_hemisphere
        Also set ``name`` to ``'right hemisphere'`` (``ML_index >= 1``) or
        ``'left hemisphere'`` (grid generators whole_cortex_lambda_bregma / _ICV).

    Returns
    -------
    numpy.ndarray
        New ``(1, N)`` struct array.

    Raises
    ------
    ValueError
        The most frequent value is 0 (MATLAB: "Most of your ROI area ... is outside of the
        Allen map"), or it is not an id of the ontology (MATLAB fails on the empty lookup).

    Notes
    -----
    Agreement with MATLAB rests on :func:`~calliopy.widefield._transform.warp_image`
    reproducing ``imwarp``'s bilinear interpolation; ROIs whose pixel majority is decided by
    interpolated border pixels may differ (see the comparison with the MATLAB-made template
    files in the tests).
    """
    fmt = warp_allen_map(_as_flat_map(flat_map), allen_points, target_points, size_template)
    flat_f = fmt.ravel(order="F")
    by_id = {}
    for s in ontology:
        by_id.setdefault(float(s["id"]), s)
    out = []
    for i, roi in enumerate(_roi_list(rois)):  # i: 0-based ROI position
        roi = MatStruct(roi)
        if i >= first:
            idx = _indices_of(roi)
            vals = flat_f[idx - 1]
            if vals.size == 0:
                raise ValueError(
                    f"ROI at position {i} (0-based) has no pixels; cannot assign an Allen area"
                )
            uniq, counts = np.unique(vals[~np.isnan(vals)], return_counts=True)
            most = float(uniq[np.argmax(counts)])
            if most == 0:
                raise ValueError(
                    f"Most of your ROI area at position {i} (0-based) is outside of the "
                    "Allen map"
                )
            if most not in by_id:
                raise ValueError(
                    f"ROI at position {i} (0-based): most frequent Allen value {most!r} is "
                    "not an ontology id"
                )
            roi["Allen_area"] = by_id[most]["name"]
            roi["Allen_acronym"] = by_id[most]["acronym"]
            roi["Allen_index"] = np.float64(most)
            if name_hemisphere:
                ml = np.asarray(roi["ML_index"], dtype=np.float64)
                right = ml.size > 0 and bool(np.all(ml >= 1))
                roi["name"] = "right hemisphere" if right else "left hemisphere"
        out.append(roi)
    return _struct_array(out)


def _hsv_colormap(n: int) -> np.ndarray:
    """MATLAB ``hsv(n)``: hue ``(0:n-1)/n``, full saturation and value; ``(n, 3)``."""
    from matplotlib.colors import hsv_to_rgb

    h = np.arange(n, dtype=np.float64) / max(n, 1)
    return hsv_to_rgb(np.column_stack([h, np.ones(n), np.ones(n)]))


def roi_template_colour_image(rois, flat_map, size_template, *, rng=None, first: int = 2):
    """Colour image of the ROIs, each coloured by its Allen area (``img_ROIs`` of the
    Allen-assigning generators).

    MATLAB::

        inds_allen = unique(flatMap); cmap = hsv(numel(inds_allen));
        rng(7); cmap = cmap(randperm(numel(inds_allen)), :);
        img_ROIs(x, y, :) = cmap(inds_allen == ROI.Allen_index, :)   (ROIs 3..end)

    Parameters
    ----------
    rois
        ROI struct array with ``Allen_index`` set (:func:`assign_allen_areas`).
    flat_map
        The (unwarped) flat map or annotation volume; its unique values index the colours.
    size_template
        ``(rows, cols)``.
    rng
        :class:`numpy.random.Generator` for the colour permutation (default: a new
        unseeded generator).
    first
        0-based position of the first ROI to draw (default 2: skip the reference points;
        MATLAB ``3:length(potROIs)``).

    Returns
    -------
    numpy.ndarray
        float64 ``(rows, cols, 3)`` RGB in [0, 1]; pixels outside ROIs are 0; a later ROI
        overwrites an earlier one where they overlap.

    Notes
    -----
    Differences from MATLAB: the colours are not MATLAB's. MATLAB permutes them with
    ``rng(7); randperm(...)``, which numpy cannot reproduce; the set of colours (``hsv(n)``)
    and which pixels are coloured with the same colour are the same.
    """
    rng = np.random.default_rng() if rng is None else rng
    uniq = np.unique(_as_flat_map(flat_map))
    cmap = _hsv_colormap(uniq.size)[rng.permutation(uniq.size)]
    n_rows, n_cols = int(size_template[0]), int(size_template[1])
    img = np.zeros((n_rows * n_cols, 3))
    for i, roi in enumerate(_roi_list(rois)):
        if i < first:
            continue
        aidx = np.asarray(roi["Allen_index"], dtype=np.float64).ravel()
        if aidx.size == 0:
            continue
        sel = np.flatnonzero(uniq == aidx[0])
        if sel.size == 0:
            continue
        img[_indices_of(roi) - 1] = cmap[sel[0]]
    return img.reshape(n_cols, n_rows, 3).transpose(1, 0, 2).copy()


def binary_roi_image(rois, size_template, *, value: float = 1.0) -> np.ndarray:
    """``img_ROIs = zeros(SIZE_TEMPLATE); img_ROIs(ROI.indices) = 1`` over all ROIs (including
    the reference points), times ``value`` (0.5 in the 14-ROI generators)."""
    n_rows, n_cols = int(size_template[0]), int(size_template[1])
    img = np.zeros(n_rows * n_cols)
    for roi in _roi_list(rois):
        img[_indices_of(roi) - 1] = 1.0
    return value * img.reshape(n_cols, n_rows).T.copy()


# ======================================================================================
# Transforming template ROIs to a target image
# ======================================================================================


def _warp_mask_nonzero(lin0: np.ndarray, in_shape, tform: np.ndarray, out_shape) -> np.ndarray:
    """0-based column-major indices of the nonzero pixels of
    ``warp_image(mask, tform, out_shape, order=1)`` for a 0/1 mask given by its 0-based
    column-major indices ``lin0`` in an image of ``in_shape``.

    Same source coordinates and inside test as :func:`warp_image`, evaluated only on the
    output bounding box of the ROI (a bilinear sample of a 0/1 mask is nonzero exactly
    when a neighbour with nonzero weight is set).
    """
    h_in, w_in = int(in_shape[0]), int(in_shape[1])
    h_out, w_out = int(out_shape[0]), int(out_shape[1])
    if lin0.size == 0:
        return np.empty(0, dtype=np.int64)
    r_in, c_in = lin0 % h_in, lin0 // h_in
    corners = np.array(
        [[c, r] for c in (c_in.min() - 1, c_in.max() + 3) for r in (r_in.min() - 1, r_in.max() + 3)],
        dtype=np.float64,
    )  # (x, y) 1-based, padded by 2 px
    pts = transform_points(tform, corners)
    x0 = max(math.floor(pts[:, 0].min()) - 1, 1)
    x1 = min(math.ceil(pts[:, 0].max()) + 1, w_out)
    y0 = max(math.floor(pts[:, 1].min()) - 1, 1)
    y1 = min(math.ceil(pts[:, 1].max()) + 1, h_out)
    if x0 > x1 or y0 > y1:
        return np.empty(0, dtype=np.int64)
    src_r, src_c, inside, yy, xx = _source_coords(
        tform, out_shape, in_shape, rows=np.arange(y0, y1 + 1), cols=np.arange(x0, x1 + 1)
    )
    mask = np.zeros((h_in + 1, w_in + 1), dtype=bool)
    mask[r_in, c_in] = True
    r0 = np.floor(src_r).astype(np.int64)
    c0 = np.floor(src_c).astype(np.int64)
    fr = src_r - r0
    fc = src_c - c0
    hit = mask[r0, c0].copy()
    hit |= (fr > 0) & mask[r0 + 1, c0]
    hit |= (fc > 0) & mask[r0, c0 + 1]
    hit |= (fr > 0) & (fc > 0) & mask[r0 + 1, c0 + 1]
    hit &= inside
    rr = (yy[hit] - 1).astype(np.int64)
    cc = (xx[hit] - 1).astype(np.int64)
    return np.sort(cc * h_out + rr)


def transform_rois_from_template(
    rois, size_orig, size_target, ref_point1_target, ref_point2_target
) -> np.ndarray:
    """Map template ROIs onto a target image by its two reference points.

    Port of ``08_RoiQuest/+utils/wf_transform_ROIs_from_ROItemplate.m``. The first two
    ROIs are the template's reference points; the similarity transform that maps them onto
    ``ref_point1_target`` / ``ref_point2_target`` (``fitgeotform2d(..., "similarity")``) is
    applied to a 0/1 image of each other ROI with bilinear interpolation (``imwarp``), and
    every nonzero output pixel belongs to the transformed ROI (``find``).

    Parameters
    ----------
    rois
        ROI struct array ``(1, N)`` (``struct_ROIs.ROIs``); ``indices`` 1-based column-major
        in an image of ``size_orig``.
    size_orig
        ``(rows, cols)`` of the template (``struct_ROIs.parameters.SIZE_TEMPLATE``).
    size_target
        ``(rows, cols)`` of the target image.
    ref_point1_target, ref_point2_target
        Reference points on the target, MATLAB axes coordinates ``(x, y)``, 1-based.

    Returns
    -------
    numpy.ndarray
        ``(1, N)`` struct array: a copy of ``rois`` whose ``indices`` are 1-based
        column-major linear indices into ``size_target`` (``(n, 1)`` float64, sorted); the
        reference ROIs get ``sub2ind(size_target, y, x)`` of the target points. An ROI that
        lands outside the target gets ``[]`` and a warning.

    Notes
    -----
    Every pixel with a nonzero bilinear weight on an ROI pixel is kept, so transformed ROIs
    grow by up to one pixel on each side (as in MATLAB). Pixels whose source point lies
    exactly on a pixel centre line (weight exactly 0 or 1e-16) decide that growth; there,
    agreement with MATLAB's floating point is not guaranteed.

    Differences from MATLAB: MATLAB pads the template image with columns only when
    ``size_target(1) > size_orig(1)`` (it tests the row count twice); the padding adds
    zeros only and does not move the image, so the result is the same; this port pads
    rows and columns independently. A template index outside ``size_orig`` raises
    (MATLAB: error on growing the matrix).
    """
    roi_l = _roi_list(rois)
    if len(roi_l) < 2:
        raise ValueError("the template needs the two reference-point ROIs first")
    so = (int(size_orig[0]), int(size_orig[1]))
    st = (int(size_target[0]), int(size_target[1]))
    pts = []
    for roi in roi_l[:2]:
        r, c = _ind2sub(so, _indices_of(roi)[:1])
        pts.append((float(c[0]), float(r[0])))
    tform = _fit(np.array(pts), np.array([ref_point1_target, ref_point2_target], float))
    in_shape = (max(so[0], st[0]), max(so[1], st[1]))
    out = []
    for i, roi in enumerate(roi_l):  # i: 0-based ROI position
        roi = MatStruct(roi)
        if i < 2:
            p = (ref_point1_target, ref_point2_target)[i]
            roi["indices"] = _mat_num(_sub2ind(st, p[1], p[0]))
        else:
            idx = _indices_of(roi)
            if np.any(idx < 1) or np.any(idx > so[0] * so[1]):
                raise ValueError(
                    f"ROI at position {i} (0-based): indices outside the template of size {so}"
                )
            r, c = _ind2sub(so, idx)
            lin0 = (c - 1) * in_shape[0] + (r - 1)
            hit = _warp_mask_nonzero(lin0, in_shape, tform, st)
            if hit.size == 0:
                warnings.warn(
                    f"ROI at position {i} (0-based) is beyond template boundaries. Will "
                    "assign empty indices",
                    stacklevel=2,
                )
                roi["indices"] = _empty()
            else:
                roi["indices"] = _mat_col(hit + 1)
        out.append(roi)
    return _struct_array(out)


def allen_area_borders(flat_map_transformed, *, line_width: float = 0.1, threshold: float = 0.1):
    """Area-border image of a warped Allen flat map (copied in MATLAB from getAreaBorders).

    MATLAB::

        gradients = imgradient(flatMap_transformed);     % Sobel magnitude
        filtered = imgaussfilt(gradients, opts.lineWidth);
        areaBorders = filtered > 0.1;

    ``imgradient`` is the magnitude ``hypot(Gx, Gy)`` of the Sobel responses with
    replicated borders (``imfilter(I, [1 2 1; 0 0 0; -1 -2 -1], 'replicate')`` and its
    transpose; the sign does not matter for the magnitude), computed with
    :func:`scipy.ndimage.correlate`.

    Returns
    -------
    numpy.ndarray
        bool, same shape as the input.
    """
    img = np.asarray(flat_map_transformed, dtype=np.float64)
    h = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    gy = scipy.ndimage.correlate(img, h, mode="nearest")
    gx = scipy.ndimage.correlate(img, h.T, mode="nearest")
    grad = np.hypot(gx, gy)
    return imgaussfilt(grad, line_width) > threshold


_PLOTTING_GRID = {
    "whole cortex": {
        "allen": (ALLEN_REF_POINTS["lambda"], ALLEN_REF_POINTS["icv"]),
        "d_border": 2,
    },
    "cranial window": {
        "allen": (
            ALLEN_REF_POINTS["cranial_window_posterior"],
            ALLEN_REF_POINTS["cranial_window_anterior"],
        ),
        "d_border": 5,
    },
}


def make_plotting_grid_rois_template(
    roi_template,
    allen_map,
    roi_type: str = "whole cortex",
    *,
    size_target=(1500, 1500),
    ref_point1_target=(750.0, 1200.0),
    ref_point2_target=(750.0, 200.0),
) -> MatStruct:
    """Grid ROIs and Allen area borders in a common frame, for plotting values per ROI.

    Port of ``12_Widefield/Plotting_templates/make_plotting_grid_ROIs_template.m``:

    1. The Allen flat map is warped into a 1500 x 1500 frame (the Allen reference points of
       ``roi_type`` go to ``ref_point1_target`` / ``ref_point2_target``) and its area borders
       are taken (:func:`allen_area_borders`).
    2. The template ROIs are mapped into the same frame
       (:func:`transform_rois_from_template`), the two reference ROIs are dropped, and each
       ROI is inflated by ``d_border`` pixels (2 for 'whole cortex', 5 for 'cranial
       window'): the union of the index shifts ``0, ±d, ±d*rows`` and the four diagonal
       ones, as linear indices (so a shift can wrap into the neighbouring column, as in
       MATLAB), sorted and unique.

    Parameters
    ----------
    roi_template
        ``struct_ROIs`` of the template: MATLAB used
        ``Grid_whole_cortex_based_on_lambda-inferior_cerebral_vein.mat`` for 'whole cortex'
        and ``Grid_for_circular_cranial_window.mat`` for 'cranial window' (load with
        :func:`load_roi_template` or build with the generators).
    allen_map
        Annotation volume or :func:`allen_flat_map` output.
    roi_type
        ``'whole cortex'`` (default, MATLAB's default for an empty argument) or
        ``'cranial window'``.
    size_target, ref_point1_target, ref_point2_target
        The common frame (MATLAB constants).

    Returns
    -------
    MatStruct
        ``areaBorders`` (bool ``size_target``) and ``ROIs_transformed`` (``(1, N-2)`` struct
        array, ``indices`` 1-based column-major into ``size_target``): the two variables MATLAB
        saves into ``<name>_grid_template.mat``. Nothing is written.

    Notes
    -----
    Differences from MATLAB: the template and the annotation are arguments instead of fixed
    paths, and nothing is saved.
    """
    if roi_type not in _PLOTTING_GRID:
        raise ValueError(f"roi_type must be 'whole cortex' or 'cranial window', got {roi_type!r}")
    cfg = _PLOTTING_GRID[roi_type]
    st = (int(size_target[0]), int(size_target[1]))
    fmt = warp_allen_map(
        _as_flat_map(allen_map), cfg["allen"], (ref_point1_target, ref_point2_target), st
    )
    borders = allen_area_borders(fmt)
    size_orig = np.asarray(roi_template["parameters"]["SIZE_TEMPLATE"]).ravel()
    trans = transform_rois_from_template(
        roi_template["ROIs"], size_orig, st, ref_point1_target, ref_point2_target
    )
    d = cfg["d_border"]
    n = st[0]
    shifts = np.array([0, d * n, d, d * n + d, -d * n, -d, -d * n - d, -d * n + d, d * n - d])
    out = []
    for roi in _roi_list(trans)[2:]:
        roi = MatStruct(roi)
        idx = np.asarray(roi["indices"], dtype=np.float64).ravel()
        if idx.size:
            roi["indices"] = _mat_col(np.unique((idx[None, :] + shifts[:, None]).ravel()))
        out.append(roi)
    return MatStruct(areaBorders=borders, ROIs_transformed=_struct_array(out))


# ======================================================================================
# Loading MATLAB-made template files (v5/v7)
# ======================================================================================


def _from_scipy(v) -> Any:
    """scipy.io.loadmat value (squeeze_me=False, struct_as_record=False) -> calliopy style."""
    if isinstance(v, scipy.io.matlab.mat_struct):
        return MatStruct((f, _from_scipy(getattr(v, f))) for f in v._fieldnames)
    if isinstance(v, np.ndarray):
        if v.dtype.kind == "U":
            return "".join(v.ravel().tolist()) if v.size else ""
        if v.dtype == object:
            if v.size and all(isinstance(e, scipy.io.matlab.mat_struct) for e in v.flat):
                if v.size == 1:  # 1x1 struct -> MatStruct, as calliopy.io.load_mat
                    return _from_scipy(v.flat[0])
                out = np.empty(v.shape, dtype=object)
                for i, e in np.ndenumerate(v):
                    out[i] = _from_scipy(e)
                return out
            out = np.empty(v.shape, dtype=object)
            for i, e in np.ndenumerate(v):
                out[i] = _from_scipy(e)
            return out
        if v.size == 1 and v.ndim == 2 and v.dtype.kind in "fiub":
            return v.reshape(())[()]
        return v
    return v


def _load_mat5(path: _PathArg) -> MatStruct:
    data = scipy.io.loadmat(
        path, squeeze_me=False, struct_as_record=False, chars_as_strings=True
    )
    return MatStruct((k, _from_scipy(v)) for k, v in data.items() if not k.startswith("__"))


def load_roi_template(path: _PathArg) -> MatStruct:
    """Read a ROI template file of ``ROItemplate_library`` (``struct_ROIs``).

    The files are MATLAB v5/v7 MAT-files, read with :func:`scipy.io.loadmat` (not
    :func:`calliopy.io.load_mat`, which reads v7.3 only).

    Returns
    -------
    MatStruct
        ``struct_ROIs`` with ``ROIs`` as a ``(1, N)`` object array of MatStruct; 1x1 numbers
        are numpy scalars, char arrays are ``str``; ``indices`` are as stored (1-based,
        column-major).
    """
    data = _load_mat5(path)
    if "struct_ROIs" not in data:
        raise KeyError("no variable 'struct_ROIs' in the file")
    return data["struct_ROIs"]


def load_plotting_grid_template(path: _PathArg) -> MatStruct:
    """Read a ``Plotting_templates/*_grid_template.mat`` file (MATLAB v5/v7, via
    :func:`scipy.io.loadmat`).

    Returns
    -------
    MatStruct
        ``areaBorders`` (bool) and ``ROIs_transformed`` (``(1, N)`` object array).
    """
    return _load_mat5(path)


# ======================================================================================
# Figures
# ======================================================================================


def plot_roi_template(struct_rois, *, ax=None):
    """Template image with the two reference points, as the MATLAB generators draw it.

    Parameters
    ----------
    struct_rois
        A template (generator output or :func:`load_roi_template`).
    ax
        Axes to draw into; default a new figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.figure import Figure

    if ax is None:
        fig = Figure(figsize=(5, 6))
        ax = fig.add_subplot()
    else:
        fig = ax.figure
    img = np.asarray(struct_rois["img_ROIs"], dtype=np.float64)
    if img.ndim == 2 and img.max(initial=0) <= 0.5:  # 14-ROI style: template + 0.5*ROIs
        img = img + 1.0
    ax.imshow(img, interpolation="nearest")
    size = img.shape[:2]
    rois = _roi_list(struct_rois["ROIs"])
    for roi, colour in zip(rois[:2], ("cyan", "red")):
        r, c = _ind2sub(size, _indices_of(roi)[:1])
        ax.scatter(c - 1, r - 1, c=colour, label=roi["name"])
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.05))
    return fig
