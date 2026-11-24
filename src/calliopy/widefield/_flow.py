"""Optical-flow velocity fields on widefield ROI grids (port of calliope 12_Widefield).

Contents:

- :func:`horn_schunck`: the Horn-Schunck optical-flow method, implemented from the paper
  (B. K. P. Horn and B. G. Schunck, "Determining optical flow", Artificial Intelligence 17
  (1981) 185-203). calliope calls a third-party implementation of this method that is under
  a non-commercial share-alike licence; that code is not translated here.
- :func:`velocity_field_horn_schunck`: flow between consecutive frames of a video
  (``Velocity_fields/calculate_velocity_field_Horn_Schunck.m``).
- :func:`match_rois_to_img_indices`, :func:`uv_img_to_linear`, :func:`uv_linear_to_img` and
  :func:`wf_grid2vid`: conversion between ROI-by-time matrices and images of the ROI grid
  (one ROI = one pixel). MATLAB loads the ROI grid template from a lab folder; here the ROI
  information is an argument.
- :func:`quiver_video`: renders a flow field as arrows into an image stack.

Flow convention: ``u`` is the velocity along columns (x, positive to the right) and ``v``
along rows (y, positive downwards), in pixels per frame.

ROI information (``roi_info``): a sequence of structs, e.g. the ``ROIs_transformed`` struct
array of a ROI grid template as a 1-D list or object array of
:class:`~calliopy.io.MatStruct` (a 2-D object array is read in MATLAB's column-major order).
Only the fields ``AP_index`` and ``ML_index`` (integer-valued numbers) are used; an empty
``AP_index`` (``None`` or an empty array) marks a ROI without a grid position. A ROI is
drawn at grid row ``max(AP_index) - AP_index`` and column ``ML_index - min(ML_index)``
(0-based), so anterior is up.

None of these functions has been compared with MATLAB output.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

from calliopy._matlab import conv2, imgaussfilt, matlab_cast, mround

__all__ = [
    "HornSchunckResult",
    "ImgIndices",
    "UVImage",
    "UVLinear",
    "VelocityField",
    "horn_schunck",
    "match_rois_to_img_indices",
    "quiver_video",
    "uv_img_to_linear",
    "uv_linear_to_img",
    "velocity_field_horn_schunck",
    "wf_grid2vid",
]

logger = logging.getLogger(__name__)

# Horn & Schunck (1981): the derivatives are averages of the four first differences on the
# 2 x 2 x 2 cube of samples (i..i+1, j..j+1, k..k+1); i = row, j = column, k = frame.
# Weights below are for correlation, w[a, b] multiplies E(i+a, j+b).
_WX = np.array([[-1.0, 1.0], [-1.0, 1.0]]) / 4.0
_WY = np.array([[-1.0, -1.0], [1.0, 1.0]]) / 4.0
_WT = np.ones((2, 2)) / 4.0
# Local average of the flow: 1/6 for the four nearest neighbours, 1/12 for the diagonals.
_AVG = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 2.0], [1.0, 2.0, 1.0]]) / 12.0
_ARROW_HEAD = 5  # arrow head length in pixels (MATLAB arrowHeadSize)


class HornSchunckResult(NamedTuple):
    """Result of :func:`horn_schunck` (MATLAB ``[u, v]``)."""

    u: np.ndarray
    v: np.ndarray


class VelocityField(NamedTuple):
    """Result of :func:`velocity_field_horn_schunck` (MATLAB ``[U, V]``)."""

    U: np.ndarray
    V: np.ndarray


class ImgIndices(NamedTuple):
    """Result of :func:`match_rois_to_img_indices`."""

    inds_on_img: np.ndarray  #: MATLAB output: 1-based column-major linear indices
    rows: np.ndarray  #: 0-based grid rows of the selected ROIs
    cols: np.ndarray  #: 0-based grid columns of the selected ROIs


class UVLinear(NamedTuple):
    """Result of :func:`uv_img_to_linear` (MATLAB ``[U_selected, V_selected]``)."""

    U_selected: np.ndarray
    V_selected: np.ndarray


class UVImage(NamedTuple):
    """Result of :func:`uv_linear_to_img` (MATLAB ``[U, V]``)."""

    U: np.ndarray
    V: np.ndarray


# --- Horn-Schunck ---------------------------------------------------------------------------


def _forward_stencil(img: np.ndarray, w: np.ndarray) -> np.ndarray:
    """sum_ab w[a, b] * img[i+a, j+b] with zeros outside the image (conv2 'same')."""
    return conv2(img, np.rot90(w, 2), "same")


def horn_schunck(
    im1, im2, alpha: float = 1.0, n_iter: int = 100, *, u_initial=None, v_initial=None
) -> HornSchunckResult:
    """Horn-Schunck optical flow between two images.

    Implements B. K. P. Horn and B. G. Schunck, "Determining optical flow", Artificial
    Intelligence 17 (1981) 185-203. Replaces the third-party ``HS.m`` (with
    ``computeDerivatives.m``) in calliope ``12_Widefield/Velocity_fields/
    Horn-Schunck_optical_flow``, as called by ``calculate_velocity_field_Horn_Schunck.m``:
    ``[u, v] = HS(im1, im2, alpha, ite, [], [], 0, [])``.

    Method (paper, sections 7-12):

    - ``Ex``, ``Ey``, ``Et`` at sample ``(i, j)`` are averages of the four first differences
      along columns, rows and time in the cube ``(i..i+1, j..j+1)`` of both images. Samples
      outside the image are zero (MATLAB ``conv2(..., 'same')`` zero padding), so the last
      row and column see a step to zero.
    - The local flow average ``ubar`` weights the four nearest neighbours by 1/6 and the
      four diagonal neighbours by 1/12, again with zeros outside the image.
    - ``n_iter`` Jacobi iterations of
      ``u = ubar - Ex (Ex ubar + Ey vbar + Et) / (alpha^2 + Ex^2 + Ey^2)``, and the same with
      ``Ey`` for ``v``, starting from ``u_initial``, ``v_initial`` (default zero).
    - NaN in the result (from ``0/0`` when ``alpha = 0``) is set to 0 at the end.

    Parameters
    ----------
    im1, im2
        Two consecutive frames, 2-D arrays of equal shape ``(height, width)``; converted to
        float64.
    alpha
        Smoothness weight (the paper's alpha, in image-intensity units; it enters squared).
    n_iter
        Number of iterations (MATLAB ``ite``).
    u_initial, v_initial
        Initial flow, ``(height, width)``; zeros when omitted.

    Returns
    -------
    HornSchunckResult
        ``u`` (along columns, positive to the right) and ``v`` (along rows, positive down),
        float64 ``(height, width)``, pixels per frame.

    Notes
    -----
    Differences from the replaced MATLAB function: no display options (MATLAB was called
    with display off); RGB images are not converted to gray (3-D input raises
    ``ValueError``); the defaults of the replaced function for omitted frames (loading
    sample images) are not provided. The derivative sign convention of the paper is used;
    the update is invariant to negating all three derivatives, so this does not change
    ``u`` and ``v`` beyond floating-point rounding.
    """
    a = np.asarray(im1, dtype=np.float64)
    b = np.asarray(im2, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError(f"im1 and im2 must be 2-D of equal shape, got {a.shape} and {b.shape}")
    n_iter = int(n_iter)
    ex = _forward_stencil(a, _WX) + _forward_stencil(b, _WX)
    ey = _forward_stencil(a, _WY) + _forward_stencil(b, _WY)
    et = _forward_stencil(b, _WT) - _forward_stencil(a, _WT)
    u = np.zeros_like(a) if u_initial is None else np.array(u_initial, dtype=np.float64)
    v = np.zeros_like(a) if v_initial is None else np.array(v_initial, dtype=np.float64)
    if u.shape != a.shape or v.shape != a.shape:
        raise ValueError("u_initial and v_initial must have the shape of the images")
    denom = alpha**2 + ex**2 + ey**2
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(n_iter):
            ubar = conv2(u, _AVG, "same")
            vbar = conv2(v, _AVG, "same")
            common = (ex * ubar + ey * vbar + et) / denom
            u = ubar - ex * common
            v = vbar - ey * common
    u[np.isnan(u)] = 0.0
    v[np.isnan(v)] = 0.0
    return HornSchunckResult(u, v)


def velocity_field_horn_schunck(
    mat_vid, *, alpha: float = 0.005, max_iter: int = 100, sigma: float = 1.0
) -> VelocityField:
    """Horn-Schunck flow between consecutive frames of a video.

    Port of calliope ``12_Widefield/Velocity_fields/calculate_velocity_field_Horn_Schunck.m``.
    NaN pixels are set to 0, every frame is smoothed with ``imgaussfilt(frame, sigma)``, and
    :func:`horn_schunck` (``alpha``, ``max_iter`` iterations) gives the flow from frame
    ``i`` to frame ``i+1``, stored at frame ``i``. The last frame repeats the flow of the
    one before.

    Parameters
    ----------
    mat_vid
        Video ``[height x width x nFrames]`` (for example the output of
        :func:`wf_grid2vid`), ``nFrames >= 2``.
    alpha, max_iter, sigma
        MATLAB ``'Alpha'``, ``'MaxIter'``, ``'Sigma'`` (Gaussian sigma in pixels).

    Returns
    -------
    VelocityField
        ``U`` (along columns) and ``V`` (along rows), float64 ``[height x width x nFrames]``,
        pixels per frame.

    Notes
    -----
    Differences from MATLAB: ``parfor`` is a plain loop; each frame is smoothed once (MATLAB
    smoothed every inner frame twice, with the same result). Fewer than two frames raise
    ``ValueError`` (MATLAB failed at ``U(:, :, end-1)``). Integer videos are smoothed in
    their class, as ``imgaussfilt`` does (see :func:`calliopy._matlab.imgaussfilt`).
    """
    vid = np.array(mat_vid)  # copy: NaN replacement must not touch the caller's data
    if vid.ndim != 3 or vid.shape[2] < 2:
        raise ValueError(f"mat_vid must be [height x width x nFrames >= 2], got {vid.shape}")
    if vid.dtype.kind == "f":
        vid[np.isnan(vid)] = 0
    smooth = imgaussfilt(vid, sigma)
    U = np.zeros(vid.shape, dtype=np.float64)
    V = np.zeros(vid.shape, dtype=np.float64)
    for i in range(vid.shape[2] - 1):
        U[:, :, i], V[:, :, i] = horn_schunck(
            smooth[:, :, i], smooth[:, :, i + 1], alpha, max_iter
        )
    U[:, :, -1] = U[:, :, -2]
    V[:, :, -1] = V[:, :, -2]
    return VelocityField(U, V)


# --- ROI grid <-> image -------------------------------------------------------------------


class _Grid(NamedTuple):
    ap: np.ndarray  # AP_index per ROI, NaN where empty
    ml: np.ndarray
    min_ap: int
    min_ml: int
    height: int
    width: int

    def rows(self) -> np.ndarray:  # 0-based, NaN where empty
        return self.height + self.min_ap - self.ap - 1

    def cols(self) -> np.ndarray:
        return self.ml - self.min_ml


def _field(s: Any, name: str):
    try:
        return s[name]
    except (KeyError, IndexError, TypeError, ValueError):
        return getattr(s, name)


def _scalar_or_nan(value, name: str, i: int) -> float:
    if value is None:
        return math.nan
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return math.nan
    if arr.size != 1:
        raise ValueError(f"ROI {i}: {name} must be a single number, got shape {arr.shape}")
    x = float(arr.reshape(-1)[0])
    if not (math.isfinite(x) and x == round(x)):
        raise ValueError(f"ROI {i}: {name} must be an integer value, got {x!r}")
    return x


def _roi_list(roi_info) -> list:
    if isinstance(roi_info, np.ndarray):
        return list(roi_info.ravel(order="F"))
    if isinstance(roi_info, Sequence):
        return list(roi_info)
    raise TypeError("roi_info must be a sequence or array of structs")


def _grid(roi_info) -> _Grid:
    rois = _roi_list(roi_info)
    ap = np.array(
        [_scalar_or_nan(_field(r, "AP_index"), "AP_index", i) for i, r in enumerate(rois)]
    )
    ml = np.array(
        [_scalar_or_nan(_field(r, "ML_index"), "ML_index", i) for i, r in enumerate(rois)]
    )
    if not np.any(np.isfinite(ap)) or not np.any(np.isfinite(ml)):
        raise ValueError("roi_info has no ROI with AP_index and ML_index")
    # MATLAB [ROIinfo.AP_index] skips empty fields; min/max over the non-empty ones
    min_ap, max_ap = int(np.nanmin(ap)), int(np.nanmax(ap))
    min_ml, max_ml = int(np.nanmin(ml)), int(np.nanmax(ml))
    return _Grid(ap, ml, min_ap, min_ml, max_ap - min_ap + 1, max_ml - min_ml + 1)


def match_rois_to_img_indices(roi_info, inds_roi_subselect=None) -> ImgIndices:
    """Grid-image positions of selected ROIs.

    Port of calliope ``12_Widefield/Velocity_fields/match_ROIs_to_img_indices.m``. The grid
    image has ``H = max(AP) - min(AP) + 1`` rows and ``W = max(ML) - min(ML) + 1`` columns,
    one pixel per ROI and no padding.

    Parameters
    ----------
    roi_info
        ROI information, see the module docstring (fields ``AP_index``, ``ML_index``).
    inds_roi_subselect
        **0-based** positions of the ROIs in ``roi_info`` (MATLAB: 1-based
        ``inds_ROI_subselect``; pass ``matlab_indices - 1``). Default: all ROIs.

    Returns
    -------
    ImgIndices
        ``inds_on_img``: what MATLAB returns, **1-based column-major** linear indices into the
        ``H x W`` image, shape ``(1, n)`` (MATLAB row vector; ``(0, 0)`` for an empty
        selection, MATLAB ``[]``), int64. ``rows``, ``cols``: the same positions as
        **0-based** row and column numbers, 1-D int64, so ``image[rows, cols]`` addresses
        them in numpy.

    Raises
    ------
    ValueError
        A selected ROI has an empty ``AP_index`` (MATLAB ``error``: reference points or
        corner ROIs must not be selected), or the ``AP_index``/``ML_index`` values are not
        integers.

    Notes
    -----
    Differences from MATLAB: the ROI selection is 0-based; ``rows``/``cols`` are new.
    """
    g = _grid(roi_info)
    n_roi = g.ap.size
    sel = np.arange(n_roi) if inds_roi_subselect is None else np.asarray(inds_roi_subselect)
    sel = sel.reshape(-1).astype(np.int64)
    if np.any((sel < 0) | (sel >= n_roi)):
        raise IndexError(f"inds_roi_subselect out of range for {n_roi} ROIs")
    empty = np.isnan(g.ap[sel])
    if np.any(empty):
        raise ValueError(
            f"ROI(s) {sel[empty].tolist()} (0-based) have no AP_index: reference points or "
            "corner ROIs cannot be placed on the image; remove them from inds_roi_subselect"
        )
    if np.any(np.isnan(g.ml[sel])):
        raise ValueError("a selected ROI has an AP_index but no ML_index")
    rows = g.rows()[sel].astype(np.int64)
    cols = g.cols()[sel].astype(np.int64)
    lin = rows + cols * g.height + 1
    inds = lin.reshape(1, -1) if lin.size else np.zeros((0, 0), dtype=np.int64)
    return ImgIndices(inds, rows, cols)


def _all_roi_positions(roi_info, n_rows: int, name: str) -> tuple[_Grid, ImgIndices]:
    g = _grid(roi_info)
    if g.ap.size != n_rows:
        raise ValueError(f"Sizes of ROIinfo ({g.ap.size}) and {name} ({n_rows}) don't match")
    return g, match_rois_to_img_indices(roi_info)


def uv_img_to_linear(U, V, roi_info) -> UVLinear:
    """Flow images of the ROI grid to ROI-by-time matrices.

    Port of calliope ``12_Widefield/Velocity_fields/UV_img_to_linear.m``, with the ROI
    template as an argument instead of ``818_ROIs_grid_template.mat`` from the lab folder.

    Parameters
    ----------
    U, V
        ``[H x W x nFrames]`` (or ``[H x W]`` for one frame) on the grid of ``roi_info``.
    roi_info
        ROI information (module docstring). Every ROI needs a grid position, as in MATLAB
        (which raises in :func:`match_rois_to_img_indices` otherwise).

    Returns
    -------
    UVLinear
        ``U_selected``, ``V_selected``: ``[nROIs x nFrames]``, row ``r`` is ROI ``r`` of
        ``roi_info``; same dtype as the input.

    Notes
    -----
    Differences from MATLAB: an image whose height and width differ from the grid of
    ``roi_info`` raises ``ValueError``; MATLAB reshaped it anyway and read the wrong pixels
    (or failed when an index exceeded the image).
    """
    u = np.asarray(U)
    v = np.asarray(V)
    g = _grid(roi_info)
    idx = match_rois_to_img_indices(roi_info)
    out = []
    for name, arr in (("U", u), ("V", v)):
        if arr.ndim not in (2, 3) or arr.shape[:2] != (g.height, g.width):
            raise ValueError(
                f"{name} must be [{g.height} x {g.width} x nFrames] for this roi_info, "
                f"got {arr.shape}"
            )
        a3 = arr.reshape(arr.shape[0], arr.shape[1], -1)
        out.append(a3[idx.rows, idx.cols, :])
    return UVLinear(out[0], out[1])


def uv_linear_to_img(U_selected, V_selected, roi_info) -> UVImage:
    """ROI-by-time flow matrices to images of the ROI grid, NaN where there is no ROI.

    Port of calliope ``12_Widefield/Velocity_fields/UV_linear_to_img.m``, with the ROI
    template as an argument.

    Parameters
    ----------
    U_selected, V_selected
        ``[nROIs x nFrames]``, one row per ROI of ``roi_info`` (a 1-D array is one frame).
    roi_info
        ROI information (module docstring); every ROI needs a grid position.

    Returns
    -------
    UVImage
        ``U``, ``V``: float64 ``[H x W x nFrames]``; pixels without a ROI are NaN.

    Raises
    ------
    ValueError
        The number of rows differs from the number of ROIs (MATLAB ``error``).
    """
    us = np.asarray(U_selected)
    vs = np.asarray(V_selected)
    us = us.reshape(-1, 1) if us.ndim == 1 else us
    vs = vs.reshape(-1, 1) if vs.ndim == 1 else vs
    g, idx = _all_roi_positions(roi_info, us.shape[0], "U")
    if vs.shape[0] != us.shape[0]:
        raise ValueError(f"Sizes of ROIinfo ({g.ap.size}) and V ({vs.shape[0]}) don't match")
    U = np.full((g.height, g.width, us.shape[1]), np.nan)
    V = np.full((g.height, g.width, vs.shape[1]), np.nan)
    for r in range(us.shape[0]):  # in ROI order: a later ROI on the same pixel wins
        U[idx.rows[r], idx.cols[r], :] = us[r, :]
        V[idx.rows[r], idx.cols[r], :] = vs[r, :]
    return UVImage(U, V)


def _nearest_index(n_in: int, n_out: int, scale: float) -> np.ndarray:
    """0-based source index of each output sample for imresize(..., 'nearest')."""
    x = np.arange(1, n_out + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1.0 - 1.0 / scale)
    j = np.floor(u + 0.5).astype(np.int64)
    return np.clip(j, 1, n_in) - 1


def _imresize_nearest(a: np.ndarray, out_rows: int, out_cols: int, scale=None) -> np.ndarray:
    """MATLAB imresize(a, ..., 'nearest') of every 2-D plane (no antialiasing).

    ``scale`` (rows, cols) defaults to the output/input size ratio. For an integer factor
    this is pixel repetition.
    """
    sr, sc = (out_rows / a.shape[0], out_cols / a.shape[1]) if scale is None else scale
    ri = _nearest_index(a.shape[0], out_rows, sr)
    ci = _nearest_index(a.shape[1], out_cols, sc)
    return a[ri][:, ci]


def wf_grid2vid(act, roi_info, resize_factor: float = 1) -> np.ndarray:
    """ROI activity matrix to a video of the ROI grid (zeros where there is no ROI).

    Port of calliope ``21_Analysis/WF_grid2vid.m``, with the ROI template (MATLAB:
    ``818_ROIs_grid_template.mat`` from the lab folder) as an argument.

    Parameters
    ----------
    act
        ``[nROIs x nFrames]`` activity; row ``r`` belongs to ROI ``r`` of ``roi_info``
        (extra rows are ignored, as in MATLAB). A 1-D array is one frame.
    roi_info
        ROI information (module docstring). ROIs without ``AP_index`` are skipped.
    resize_factor
        Spatial enlargement with MATLAB ``imresize(A, resize_factor, 'nearest')``: output
        ``ceil(resize_factor * H)`` x ``ceil(resize_factor * W)``; an integer factor repeats
        every pixel.

    Returns
    -------
    numpy.ndarray
        float64 ``[H x W x nFrames]`` (resized as requested).

    Notes
    -----
    Differences from MATLAB: none intended. Unverified: the nearest-neighbour index rule of
    ``imresize`` for non-integer factors (integer factors are pixel repetition).
    """
    a = np.asarray(act)
    a = a.reshape(-1, 1) if a.ndim == 1 else a
    g = _grid(roi_info)
    if a.shape[0] < g.ap.size:
        raise ValueError(f"act has {a.shape[0]} rows, roi_info has {g.ap.size} ROIs")
    rows, cols = g.rows(), g.cols()
    A = np.zeros((g.height, g.width, a.shape[1]), dtype=np.float64)
    for i in range(g.ap.size):
        if not np.isnan(rows[i]):
            A[int(rows[i]), int(cols[i]), :] = a[i, :]
    if resize_factor != 1:
        if not resize_factor > 0:
            raise ValueError(f"resize_factor must be positive, got {resize_factor!r}")
        A = _imresize_nearest(
            A,
            math.ceil(resize_factor * g.height),
            math.ceil(resize_factor * g.width),
            (resize_factor, resize_factor),
        )
    return A


# --- quiver video ---------------------------------------------------------------------------


def _draw_line(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color, thickness) -> None:
    """Bresenham line from (x1, y1) to (x2, y2), 0-based (x = column, y = row), in place.

    Every visited point sets the square of ``2*floor(thickness/2)+1`` pixels around it,
    clipped to the image. (MATLAB drawLine took 1-based coordinates; the visited points
    are the same after subtracting 1, since Bresenham's steps depend only on differences.)
    """
    h = math.floor(thickness / 2)
    n_rows, n_cols = img.shape
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    sx = (x2 > x1) - (x2 < x1)
    sy = (y2 > y1) - (y2 < y1)
    err = dx - dy
    while True:
        r0, r1 = max(y1 - h, 0), min(y1 + h, n_rows - 1)
        c0, c1 = max(x1 - h, 0), min(x1 + h, n_cols - 1)
        if r0 <= r1 and c0 <= c1:
            img[r0 : r1 + 1, c0 : c1 + 1] = color
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy


def quiver_video(
    U,
    V,
    *,
    spacing: int = 30,
    thickness: int = 1,
    vec_length: float = 100,
    bg_mat=None,
    roi_info=None,
) -> np.ndarray:
    """Render a flow field as arrows into an image stack.

    Port of calliope ``12_Widefield/Velocity_fields/quiver_video.m``. Every grid pixel
    ``(i, j)`` becomes a ``spacing x spacing`` block; an arrow starts at the block centre
    ``x = (j + 0.5) * spacing``, ``y = (i + 0.5) * spacing`` (0-based ``i``, ``j``; MATLAB
    ``(j - 0.5) * spacing`` with 1-based ``j``, in 1-based pixel coordinates) and ends at
    ``round(x + vec_length * U)``, ``round(y + vec_length * V)``. Lines are drawn with
    Bresenham's algorithm, each point as a square of ``2*floor(thickness/2)+1`` pixels;
    the head is two lines of length 5 pixels at +-30 degrees. Pixels where ``U`` or ``V``
    is NaN get no arrow.

    Parameters
    ----------
    U, V
        Flow ``[H x W x nFrames]``, or ``[nROIs x nFrames]`` matrices (2-D input), which are
        converted with :func:`uv_linear_to_img` and need ``roi_info``. A single frame must
        therefore be given as ``[H x W x 1]``.
    spacing
        Block size in output pixels per grid pixel.
    thickness
        Line thickness (see above; an even value gives ``thickness + 1`` pixels).
    vec_length
        Arrow length in output pixels per unit of flow.
    bg_mat
        Background: ``[H x W]``-grid video ``[H x W x 1]`` or ``[H x W x nFrames]``, or an
        activity matrix ``[nROIs x nFrames]`` (2-D input, converted with
        :func:`wf_grid2vid`, needs ``roi_info``). It is enlarged with
        ``imresize(bg_mat, [H*spacing, W*spacing], 'nearest')`` (pixel repetition when its
        grid is ``H x W``) and the arrows get the value ``2 * max(background)``.
        Without a background the output is uint8, black, with arrows of 255.
    roi_info
        ROI information (module docstring), needed only for 2-D ``U``/``V``/``bg_mat``.

    Returns
    -------
    numpy.ndarray
        ``[H*spacing x W*spacing x nFrames]``; uint8 without background, else the dtype of
        the (converted) background (``2 * max`` saturates for integer types, as in MATLAB).

    Notes
    -----
    Differences from MATLAB:

    - Odd ``spacing`` gives non-integer arrow start points, at which MATLAB failed (non-
      integer subscript); here they are rounded half away from zero.
    - Infinite ``U``/``V`` raise ``ValueError`` (MATLAB never left the line loop).
    - ``roi_info`` is an argument (MATLAB loaded the lab's 818-ROI template).
    - ``U`` and ``V`` of different dimensionality raise ``ValueError``.
    """
    u = np.asarray(U)
    v = np.asarray(V)
    if u.ndim != v.ndim:
        raise ValueError("U and V must both be images or both be ROI matrices")
    if u.ndim == 2:
        if roi_info is None:
            raise ValueError("2-D U/V are ROI matrices and need roi_info")
        u, v = uv_linear_to_img(u, v, roi_info)
    if u.ndim != 3 or u.shape != v.shape:
        raise ValueError(f"U and V must be [H x W x nFrames] of equal shape, got {u.shape}")
    finite_or_nan = np.isfinite(u) | np.isnan(u), np.isfinite(v) | np.isnan(v)
    if not (finite_or_nan[0].all() and finite_or_nan[1].all()):
        raise ValueError("U and V must not contain Inf")
    rows, cols, n_t = u.shape
    img_h, img_w = rows * spacing, cols * spacing
    if bg_mat is None:
        img = np.zeros((img_h, img_w, n_t), dtype=np.uint8)
        color: Any = np.uint8(255)
    else:
        bg = np.asarray(bg_mat)
        if bg.ndim == 2:
            if roi_info is None:
                raise ValueError("a 2-D bg_mat is an activity matrix and needs roi_info")
            bg = wf_grid2vid(bg, roi_info)
        bg_r = _imresize_nearest(bg, img_h, img_w)
        if bg.shape[2] == 1:
            bg_r = np.repeat(bg_r, n_t, axis=2)
        elif bg.shape[2] != n_t:
            raise ValueError(
                "bg_mat depth must match the number of timesteps in U and V or be a single slice."
            )
        img = np.array(bg_r)
        if img.dtype.kind in "iu":
            color = matlab_cast(2.0 * float(img.max()), img.dtype)
        elif img.dtype.kind == "b":
            color = True
        else:
            finite = img[~np.isnan(img)]
            color = 2 * finite.max() if finite.size else np.nan
    for t in range(n_t):
        plane = img[:, :, t]
        for i in range(rows):
            for j in range(cols):
                uu, vv = u[i, j, t], v[i, j, t]
                if np.isnan(uu) or np.isnan(vv):
                    continue
                # MATLAB's 1-based pixel coordinates are rounded first (rounding half away
                # from zero is not shift-invariant), then converted to 0-based
                sx = (j + 0.5) * spacing
                sy = (i + 0.5) * spacing
                ex = int(mround(sx + vec_length * uu))
                ey = int(mround(sy + vec_length * vv))
                x0, y0 = int(mround(sx)), int(mround(sy))
                _draw_line(plane, x0 - 1, y0 - 1, ex - 1, ey - 1, color, thickness)
                angle = math.atan2(vv, uu)
                for da in (math.pi / 6, -math.pi / 6):
                    hx = ex - int(mround(_ARROW_HEAD * math.cos(angle + da)))
                    hy = ey - int(mround(_ARROW_HEAD * math.sin(angle + da)))
                    _draw_line(plane, ex - 1, ey - 1, hx - 1, hy - 1, color, thickness)
    return img
