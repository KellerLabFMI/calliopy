"""Image computations of RoiQuest and ActiQuest.

Port of calliope ``08_RoiQuest/+gfx`` (``filterTemplate``, ``adjustImgContrast``,
``getLabelsColormap``, ``makeCross``, ``segmentNeuron``, ``uniformRgbImage``), of the image
parts of ``+clbk/getInfoFromAData.m`` (valid area, filter size) and ``+clbk/
autodetectCellpose.m`` (filtering of Cellpose labels), and of ``ActiQuest/
AQ_createVisualization.m`` (Leonardo Lupori, 2024). The interactive GUI is not ported.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import matplotlib.colors
import numpy as np
import scipy.ndimage
from matplotlib.figure import Figure

from calliopy._matlab import matlab_cast, mround

from ._builtins import (
    bwareaopen,
    bwmorph_spur,
    bwperim,
    fspecial_disk,
    imclose_binary,
    imtophat,
    mat2gray,
    quantile,
    strel_disk,
)
from ._roi import _cell_list, _find, _lin, _struct_list

__all__ = [
    "CellposeRois",
    "actiquest_activity",
    "actiquest_traces",
    "adjust_img_contrast",
    "default_filter_size",
    "filter_cellpose_labels",
    "filter_template",
    "get_labels_colormap",
    "make_cross",
    "plot_actiquest",
    "plot_rois_to_add",
    "segment_neuron",
    "uniform_rgb_image",
    "valid_area",
]


def _imfilter_replicate(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """MATLAB ``imfilter(img, h, 'replicate')`` for an odd-sized kernel, in float64."""
    return scipy.ndimage.correlate(img, kernel, mode="nearest")


def filter_template(template_image, filter_size, filter_type: str = "legacy") -> np.ndarray:
    """Normalise a template image by its local mean, MATLAB ``gfx.filterTemplate``.

    ``'legacy'`` (default): ``T ./ imfilter(T, fspecial('disk', filter_size), 'replicate')``.
    ``'updated'``: ``F = imtophat(T, strel('disk', filter_size))``, then
    ``F ./ imfilter(F, fspecial('disk', 3*filter_size), 'replicate')``.

    Parameters
    ----------
    template_image
        2-D image (template). float32 input gives float32 output (computed in float64);
        integer input is filtered and divided in its own class as MATLAB does (rounded and
        saturated).
    filter_size
        Disk radius in pixels (RoiQuest uses ``round(min(size)/20)``, see
        :func:`default_filter_size`).
    filter_type
        ``'legacy'`` or ``'updated'``.

    Returns
    -------
    numpy.ndarray
        Same shape as the input.

    Notes
    -----
    The disk kernel reproduces MATLAB's ``fspecial('disk')`` formula.

    Differences from MATLAB / unverified: for ``'updated'`` the top-hat uses the exact disk
    ``strel('disk', r, 0)``; MATLAB's ``strel('disk', r)`` approximates the disk with periodic
    lines (``N = 4``), so for radii above 2 the ``'updated'`` result differs from MATLAB.
    ``'legacy'`` (RoiQuest's default) has no such difference.
    """
    if filter_type not in ("legacy", "updated"):
        raise ValueError("filter_type must be 'legacy' or 'updated'")
    t = np.asarray(template_image)
    if t.ndim != 2:
        raise ValueError(f"template_image must be 2-D, got shape {t.shape}")
    dtype = t.dtype
    is_int = np.issubdtype(dtype, np.integer)
    tf = t.astype(np.float64)
    if filter_type == "legacy":
        num = tf
        den = _imfilter_replicate(tf, fspecial_disk(filter_size))
    else:
        num = imtophat(tf, strel_disk(filter_size))
        if is_int:
            num = matlab_cast(num, dtype).astype(np.float64)
        den = _imfilter_replicate(num, fspecial_disk(3 * filter_size))
    if is_int:
        den = matlab_cast(den, dtype).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return matlab_cast(num / den, dtype)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return out.astype(np.float32) if dtype == np.float32 else out


def default_filter_size(img_hw) -> int:
    """Template filter radius RoiQuest uses: ``round(min(imgSize) / 20)``
    (``getInfoFromAData.m``)."""
    return int(mround(min(int(img_hw[0]), int(img_hw[1])) / 20))


def adjust_img_contrast(double_img, contrast_mode: str = "bestFit", clims=(0.0, 1.0)) -> np.ndarray:
    """Contrast-stretch an image to uint8, MATLAB ``gfx.adjustImgContrast``.

    - ``'minMax'``: ``uint8(mat2gray(img) * 255)``;
    - ``'bestFit'`` (default): limits at the 0.5 % and 99.5 % quantiles (MATLAB
      ``quantile``, NaNs ignored);
    - ``'custom'``: with ``m``/``M`` the image minimum/maximum, limits
      ``[m + (M - m)*clims[0], M*clims[1]]``.

    Parameters
    ----------
    double_img
        2-D image (converted to float64, as MATLAB's ``double`` argument validation).
    contrast_mode
        ``'minMax'``, ``'bestFit'`` or ``'custom'``.
    clims
        Two values in ``[0, 1]`` for ``'custom'``; not ascending -> warning and ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        uint8 image (MATLAB rounding, NaN -> 0).

    Notes
    -----
    Kept as in MATLAB (suspected bug, reported): in ``'custom'`` mode the upper limit is
    ``M*clims[1]``, not ``m + (M - m)*clims[1]``.

    Differences from MATLAB: an unknown ``contrast_mode`` raises (MATLAB failed later with an
    undefined output).
    """
    img = np.asarray(double_img, dtype=np.float64)
    cl = np.asarray(clims, dtype=np.float64).ravel()
    if cl.size != 2 or np.any(cl < 0) or np.any(cl > 1):
        raise ValueError("clims must be two values in [0, 1]")
    if contrast_mode == "minMax":
        g = mat2gray(img)
    elif contrast_mode == "bestFit":
        lo = quantile(img, 0.005)
        hi = quantile(img, 0.995)
        g = mat2gray(img, [lo, hi])
    elif contrast_mode == "custom":
        if cl[1] <= cl[0]:
            warnings.warn("Input CLims are not ascending. Changed to default [0,1].",
                          stacklevel=2)
            cl = np.array([0.0, 1.0])
        m = np.nanmin(img)
        mx = np.nanmax(img)
        rng = mx - m
        g = mat2gray(img, [m + rng * cl[0], mx * cl[1]])
    else:
        raise ValueError("contrast_mode must be 'minMax', 'bestFit' or 'custom'")
    return matlab_cast(g * 255, np.uint8)


def get_labels_colormap() -> np.ndarray:
    """ROI label colormap, MATLAB ``gfx.getLabelsColormap``.

    Returns
    -------
    numpy.ndarray
        ``(12, 3)`` float64: black (background, label 0 -> row 0), ``hsv(10)`` (ROI types),
        fluorescent green ``[0.22, 0.9, 0.19]`` (selection). Row ``k`` is the colour of label
        value ``k`` (MATLAB row ``k + 1``).
    """
    h = np.arange(10) / 10.0
    hsv = matplotlib.colors.hsv_to_rgb(np.column_stack([h, np.ones(10), np.ones(10)]))
    return np.vstack([[0.0, 0.0, 0.0], hsv, [0.22, 0.9, 0.19]])


def uniform_rgb_image(image_size, color=(0.93, 0.13, 0.55)) -> np.ndarray:
    """Constant RGB image, MATLAB ``gfx.uniformRgbImage``: ``(H, W, 3)`` float64."""
    h, w = (int(v) for v in tuple(image_size)[:2])
    c = np.asarray(color, dtype=np.float64).ravel()
    if c.size != 3:
        raise ValueError("color must have 3 values")
    return np.broadcast_to(c, (h, w, 3)).copy()


def make_cross(center_rc, img_hw, cross_size=40) -> np.ndarray:
    """Pixels of a cross marker, MATLAB ``gfx.makeCross``.

    Two bars of width 5 and length ``2*round(cross_size/2) + 1`` centred on ``center_rc``,
    clipped to the image.

    Parameters
    ----------
    center_rc
        ``(row, col)`` of the centre, **0-based** integers.
    img_hw
        ``(height, width)``.
    cross_size
        MATLAB default 40.

    Returns
    -------
    numpy.ndarray
        float64 ``(N, 1)`` **1-based** column-major linear indices (as ``Roi.indices``),
        ascending.

    Differences from MATLAB: the centre is 0-based (MATLAB: 1-based subscripts).
    """
    r0, c0 = (float(v) for v in np.ravel(center_rc)[:2])
    if r0 != int(r0) or c0 != int(c0):
        raise ValueError("center_rc must be integers")
    h, w = int(img_hw[0]), int(img_hw[1])
    r, c = int(r0) + 1, int(c0) + 1
    add = int(mround(cross_size / 2))
    mask = np.zeros((h, w), dtype=bool)
    for xr, yr in (((r - add, r + add), (c - 2, c + 2)), ((r - 2, r + 2), (c - add, c + add))):
        x = np.arange(xr[0], xr[1] + 1)
        y = np.arange(yr[0], yr[1] + 1)
        x = x[(x >= 1) & (x <= h)]
        y = y[(y >= 1) & (y <= w)]
        mask[np.ix_(x - 1, y - 1)] = True
    return _find(mask)


def segment_neuron(img_data, thr, crop, *, min_roi_pix: int = 20, valid_area=None,
                   existing_mask=None, close_and_despur: bool = False) -> np.ndarray:
    """Threshold segmentation of a zoomed region, MATLAB ``gfx.segmentNeuron(app)``.

    Steps, as in MATLAB: pixels of the crop above ``thr``; 8-connected objects smaller than
    ``min_roi_pix`` removed (``bwareaopen``); put back into a full-size mask; restricted to
    ``valid_area``; pixels of existing ROIs removed; for the activity map
    (``close_and_despur``, MATLAB ``app.imgIdx == 3``) closing with ``strel('disk', 2)`` and
    one ``bwmorph 'spur'`` pass.

    Parameters
    ----------
    img_data
        2-D display image (RoiQuest passes the uint8 contrast-adjusted image).
    thr
        Threshold (MATLAB ``app.thr(app.imgIdx)``).
    crop
        ``(row_start, row_stop, col_start, col_stop)``, **0-based, stop exclusive** (the
        zoomed axes limits; MATLAB ``imgData(yl(1):yl(2), xl(1):xl(2))`` is
        ``(yl(1)-1, yl(2), xl(1)-1, xl(2))``).
    min_roi_pix
        MATLAB ``app.minRoiPix`` (default 20).
    valid_area
        Boolean ``(H, W)`` mask of valid pixels (``None``: all valid).
    existing_mask
        Boolean ``(H, W)`` union of the existing ROIs (``None``: no ROIs).
    close_and_despur
        Apply the closing and spur removal.

    Returns
    -------
    numpy.ndarray
        Boolean ``(H, W)`` mask; ``Roi(np.flatnonzero(mask.ravel(order='F')) + 1, mask.shape)``
        is the putative ROI RoiQuest creates.
    """
    img = np.asarray(img_data)
    r0, r1, c0, c1 = (int(v) for v in crop)
    mask_crop = img[r0:r1, c0:c1] > thr
    mask_crop = bwareaopen(mask_crop, min_roi_pix)
    mask = np.zeros(img.shape[:2], dtype=bool)
    mask[r0:r1, c0:c1] = mask_crop
    if valid_area is not None:
        mask &= np.asarray(valid_area, dtype=bool)
    if existing_mask is not None:
        mask &= ~np.asarray(existing_mask, dtype=bool)
    if close_and_despur:
        mask = imclose_binary(mask, strel_disk(2))
        mask = bwmorph_spur(mask)
    return mask


def valid_area(dx, dy, img_hw) -> np.ndarray:
    """Pixels not affected by the registration shifts, from ``getInfoFromAData.m``.

    For each Z layer the valid area is the image without a border of ``max|dx|`` rows and
    ``max|dy|`` columns on each side (NaNs ignored).

    Parameters
    ----------
    dx, dy
        Cell arrays (one entry per Z layer) of registration shifts, as stored in the adata
        (``dx`` = rows, ``dy`` = columns).
    img_hw
        ``(height, width)``.

    Returns
    -------
    numpy.ndarray
        Boolean ``(H, W, nZ)``.

    Notes
    -----
    Kept as in MATLAB (suspected bug, reported): MATLAB indexes with the vectors of all
    layers' maxima, so the colon operator uses the **first layer's** maxima for every layer.
    A layer with only NaN shifts gives an empty valid area (MATLAB: ``NaN:...`` is empty).
    """
    dxs = _cell_list(dx)
    dys = _cell_list(dy)

    def absmax(v):
        a = np.abs(np.asarray(v, dtype=np.float64))
        return np.nan if a.size == 0 or np.all(np.isnan(a)) else float(np.nanmax(a))

    max_dx = [absmax(v) for v in dxs]
    max_dy = [absmax(v) for v in dys]
    h, w = int(img_hw[0]), int(img_hw[1])
    nz = len(dxs)
    out = np.zeros((h, w, nz), dtype=bool)
    mdx, mdy = max_dx[0], max_dy[0]
    if np.isnan(mdx) or np.isnan(mdy):
        return out
    if mdx != int(mdx) or mdy != int(mdy):
        raise ValueError("registration shifts must be integers")
    x0, x1 = int(mdx) + 1, h - int(mdx)
    y0, y1 = int(mdy) + 1, w - int(mdy)
    if x1 >= x0 and y1 >= y0:
        out[x0 - 1 : x1, y0 - 1 : y1, :] = True
    return out


class CellposeRois(NamedTuple):
    """Result of :func:`filter_cellpose_labels`."""

    indices: list
    """1-based column-major linear indices (``(N, 1)`` float64) of each accepted ROI."""
    num_detected: int
    """``max(labels)``: number of labels Cellpose returned."""
    num_removed: int
    """Labels rejected (on the border, overlapping, too small)."""


def filter_cellpose_labels(labels, valid_area_z=None, existing_mask=None,
                           min_roi_pix: int = 20) -> CellposeRois:
    """Turn a Cellpose label image into putative ROIs, as ``clbk.autodetectCellpose``.

    For each label ``1..max(labels)``: rejected when any pixel lies outside
    ``valid_area_z``, when it overlaps ``existing_mask``, or when it has fewer than
    ``min_roi_pix`` pixels. Running Cellpose itself (``gfx.cellposeDetect``) is not part of
    calliopy.

    Parameters
    ----------
    labels
        Integer label image ``(H, W)`` (0 = background).
    valid_area_z
        Boolean ``(H, W)`` valid area of the layer (``None``: all valid).
    existing_mask
        Boolean ``(H, W)`` union of existing ROIs (``None``: none).
    min_roi_pix
        Minimum ROI size (MATLAB ``app.minRoiPix``).
    """
    lab = np.asarray(labels)
    nmax = int(lab.max()) if lab.size else 0
    out, removed = [], 0
    for i in range(1, nmax + 1):
        mask = lab == i
        if valid_area_z is not None and np.any(mask & ~np.asarray(valid_area_z, dtype=bool)):
            removed += 1
            continue
        if existing_mask is not None and np.any(mask & np.asarray(existing_mask, dtype=bool)):
            removed += 1
            continue
        if mask.sum() < min_roi_pix:
            removed += 1
            continue
        out.append(_find(mask))
    return CellposeRois(out, nmax, removed)


# ---------------------------------------------------------------------------------------
# ActiQuest
# ---------------------------------------------------------------------------------------
def actiquest_activity(activities) -> np.ndarray:
    """Median-divided activity matrix of ActiQuest (``AQ_createVisualization.m``).

    Parameters
    ----------
    activities
        Sequence of ROI activity traces, each ``[nFrames x 1]`` (the ``activity`` field of
        the ROIs), or ``[nFrames x nRois]`` array.

    Returns
    -------
    numpy.ndarray
        ``[nRois x nFrames]``: ``(act ./ median(act, 1))' - 1`` (NaN propagates through the
        median, as MATLAB's default).
    """
    if isinstance(activities, np.ndarray) and activities.dtype != object:
        act = np.asarray(activities, dtype=np.float64)
    else:
        act = np.column_stack([np.asarray(a, dtype=np.float64).ravel()
                               for a in (_cell_list(activities)
                                         if isinstance(activities, np.ndarray) else activities)])
    return (act / np.median(act, axis=0)).T - 1


def actiquest_traces(act_data, raster_hi: float = 2) -> np.ndarray:
    """Offset traces of ActiQuest: row ``i`` (0-based) is ``act_data[i] / (raster_hi + 0.5)
    + (i + 1)`` (``AQ_createVisualization.m``, ``plotData``)."""
    a = np.asarray(act_data, dtype=np.float64)
    return a / (raster_hi + 0.5) + np.arange(1, a.shape[0] + 1)[:, None]


def plot_actiquest(act_data, *, raster_lo: float = 0, raster_hi: float = 2,
                   roi_numbers=None, rand_color_idx=None) -> Figure:
    """ActiQuest figure: raster of ``act_data`` and offset traces (``AQ_createVisualization``).

    Parameters
    ----------
    act_data
        ``[nRois x nFrames]`` from :func:`actiquest_activity`.
    raster_lo, raster_hi
        Colour limits of the raster (ActiQuest defaults 0 and 2).
    roi_numbers
        Labels of the trace rows (default ``1..nRois``, the ROI numbers shown in RoiQuest).
    rand_color_idx
        ``Roi.rand_color_idx`` of each ROI (MATLAB row index into the labels colormap,
        1-based as stored by the Roi class); default: matplotlib's colour cycle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    a = np.asarray(act_data, dtype=np.float64)
    n = a.shape[0]
    fig = Figure(figsize=(10, 6))
    ax_r, ax_t = fig.subplots(2, 1, sharex=True)
    ax_r.imshow(a, cmap="gray_r", vmin=raster_lo, vmax=raster_hi, aspect="auto",
                origin="lower", interpolation="nearest",
                extent=(0.5, a.shape[1] + 0.5, 0.5, n + 0.5))
    ax_r.set_yticks([])
    traces = actiquest_traces(a, raster_hi)
    x = np.arange(1, a.shape[1] + 1)
    cmap = get_labels_colormap()
    for i in range(n):
        kw = {}
        if rand_color_idx is not None:
            kw["color"] = cmap[int(rand_color_idx[i]) - 1]
        ax_t.plot(x, traces[i], linewidth=0.8, **kw)
    ax_t.set_xlim(1, max(a.shape[1], 2))
    ax_t.set_ylim(0.5, n + 1)
    ax_t.set_yticks(np.arange(1, n + 1))
    labels = np.arange(1, n + 1) if roi_numbers is None else np.ravel(roi_numbers)
    ax_t.set_yticklabels([str(int(v)) for v in labels])
    ax_t.yaxis.grid(True)
    ax_t.set_xlabel("Time (frames)")
    ax_t.set_ylabel("Normalized Activity for each ROI number")
    return fig


def plot_rois_to_add(rois_to_add, img_hw) -> Figure:
    """Check figure of :func:`save_rois_chronic`: per timepoint the contrast-adjusted images
    of all Z layers stacked vertically, with the perimeters of the ROIs to add in cyan.

    Parameters
    ----------
    rois_to_add
        ``rois_to_add`` of :class:`ChronicRoiSave` (entries with ``ROIs``, ``imgs``,
        ``titles``).
    img_hw
        ``(height, width)``.
    """
    entries = list(rois_to_add)
    h, w = int(img_hw[0]), int(img_hw[1])
    fig = Figure(figsize=(2.8 * max(len(entries), 1), 8), facecolor=(0.15, 0.15, 0.15))
    axes = fig.subplots(1, max(len(entries), 1), squeeze=False)[0]
    for ax, e in zip(axes, entries):
        stacked = []
        for z, img in enumerate(_cell_list(e["imgs"])):
            rois_z = _cell_list(e["ROIs"])[z]
            bw = np.zeros((h, w), dtype=bool)
            for s in _struct_list(rois_z):
                if np.size(s["indices"]):
                    bw.T.flat[_lin(s["indices"]) - 1] = True
            bw = bwperim(bw)
            g = np.asarray(img, dtype=np.float64) / 255.0
            rgb = np.repeat(g[:, :, None], 3, axis=2)
            rgb[bw] = (0.0, 1.0, 1.0)
            stacked.append(rgb)
        if stacked:
            ax.imshow(np.concatenate(stacked, axis=0), interpolation="nearest")
        titles = _cell_list(e["titles"])
        if titles:
            ax.set_title(str(titles[0])[:-3] + "All Z Layers", color=(0.85, 0.85, 0.85),
                         fontsize=8)
        ax.set_axis_off()
    return fig
