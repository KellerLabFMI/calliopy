"""ROI grid template, ROI-to-grid images and ROI maps (WF_grid2vid, WF_imageROIvalues)."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np


class GridTemplate(NamedTuple):
    """The ROI grid template (``ROIs_transformed`` and ``areaBorders`` of the template file).

    ``ap_index``/``ml_index`` are the grid coordinates of each ROI (NaN when empty).
    ``indices`` holds each ROI's pixels in the atlas image as stored: 1-based linear
    indices in MATLAB (column-major) order. ``area_borders`` is the Allen border mask.
    """

    ap_index: np.ndarray
    ml_index: np.ndarray
    indices: list[np.ndarray]
    names: list[str]
    allen_acronym: list[str]
    area_borders: np.ndarray
    path: Path

    @property
    def n_roi(self) -> int:
        return len(self.indices)

    @property
    def image_shape(self) -> tuple[int, int]:
        return self.area_borders.shape


def _as_float(v: Any) -> float:
    return float(v) if np.size(v) == 1 else np.nan


def load_grid_template(path: str | PathLike) -> GridTemplate:
    """Read a ROI grid template file (e.g. ``818_ROIs_grid_template.mat``).

    Reads MATLAB v5 files with scipy and v7.3 files with :func:`calliopy.io.load_mat`.

    Differences from MATLAB
    -----------------------
    - WF_grid2vid/WF_imageROIvalues find the template through ``set_lab_paths``;
      here the file is always passed in.
    """
    path = Path(path)
    try:
        from scipy.io import loadmat

        raw = loadmat(path, squeeze_me=True, struct_as_record=False,
                      variable_names=["ROIs_transformed", "areaBorders"])
        rois = np.atleast_1d(raw["ROIs_transformed"])
        borders = np.asarray(raw["areaBorders"]).astype(bool)
    except NotImplementedError:  # v7.3 (HDF5)
        from calliopy.io import load_mat

        raw = load_mat(path, ["ROIs_transformed", "areaBorders"])
        rois = np.ravel(raw.ROIs_transformed)  # MatStruct: fields are also attributes
        borders = np.asarray(raw.areaBorders).astype(bool)
    ap = np.array([_as_float(r.AP_index) for r in rois])
    ml = np.array([_as_float(r.ML_index) for r in rois])
    idx = [np.atleast_1d(np.asarray(r.indices, dtype=np.int64)).ravel() for r in rois]
    names = [str(r.name) for r in rois]
    acr = [str(r.Allen_acronym) for r in rois]
    return GridTemplate(ap, ml, idx, names, acr, borders, path)


def grid2vid(act: np.ndarray, template: GridTemplate, resize_factor: int = 1) -> np.ndarray:
    """Place ROI values on the ROI grid, one pixel per ROI (WF_grid2vid).

    Parameters
    ----------
    act
        ``[nROI x T]`` (or ``[nROI]``) values, one row per ROI in template order.
    template
        From :func:`load_grid_template`.
    resize_factor
        Integer up-sampling of the grid (nearest neighbour), as ``imresize(..., 'nearest')``.

    Returns
    -------
    numpy.ndarray
        ``[H x W x T]``; anterior at the top (row 0), grid positions without an
        ROI are 0, as in MATLAB.
    """
    act = np.asarray(act, dtype=float)
    if act.ndim == 1:
        act = act[:, None]
    if act.shape[0] != template.n_roi:
        raise ValueError(f"act has {act.shape[0]} rows, the template {template.n_roi} ROIs")
    ok = ~(np.isnan(template.ap_index) | np.isnan(template.ml_index))  # corner ROIs without index
    ap = template.ap_index[ok].astype(int)
    ml = template.ml_index[ok].astype(int)
    h = ap.max() - ap.min() + 1
    w = ml.max() - ml.min() + 1
    out = np.zeros((h, w, act.shape[1]))
    out[ap.max() - ap, ml - ml.min(), :] = act[ok]
    if resize_factor != 1:
        out = np.repeat(np.repeat(out, resize_factor, axis=0), resize_factor, axis=1)
    return out


def roi_image(values: np.ndarray, template: GridTemplate, background: float = np.nan) -> np.ndarray:
    """ROI values painted into the atlas-space image, flipped left-right as WF_imageROIvalues shows it."""
    values = np.asarray(values, dtype=float).ravel()
    if values.size != template.n_roi:
        raise ValueError(f"{values.size} values for {template.n_roi} ROIs")
    img = np.full(template.image_shape, background, dtype=float)
    for v, idx in zip(values, template.indices):
        rows, cols = np.unravel_index(idx - 1, template.image_shape, order="F")  # MATLAB: column-major
        img[rows, cols] = v
    return img[:, ::-1]


def _disk(radius: int) -> np.ndarray:
    r = int(radius)
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    return x**2 + y**2 <= r**2 + r  # close to MATLAB's strel('disk', r) for small r


def plot_roi_values(
    values: np.ndarray,
    clim: tuple[float, float] | None = None,
    *,
    template: GridTemplate,
    ax: Any = None,
    cmap: Any = None,
    border_color: Any = "w",
    border_width: int = 2,
    seed: int | None = None,
    seed_color: Any = (1.0, 0.0, 1.0),
    seed_size: float = 15,
):
    """Draw one value per ROI with the Allen area borders on top (WF_imageROIvalues).

    Parameters
    ----------
    values
        One value per ROI (template order); NaN ROIs are transparent.
    clim
        Colour limits; default: 0.01th and 99th percentile of ``values``.
    template
        From :func:`load_grid_template`.
    ax, cmap
        Matplotlib axes (default: current) and colormap (name, Colormap, or ``[n x 3]`` RGB).
    border_color, border_width
        Colour and thickness (disk radius in pixels) of the area borders.
    seed
        0-based ROI index to mark with a dot, or ``None``.

    Returns
    -------
    matplotlib.image.AxesImage
        The ROI image (for a colorbar).

    Differences from MATLAB
    -----------------------
    - ``seed`` is 0-based and must be given explicitly (MATLAB's default 'auto'
      marked the maximum).
    - NaN ROIs and the background are always transparent (MATLAB: ``bg_value``).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, to_rgb
    from scipy.ndimage import binary_dilation

    values = np.asarray(values, dtype=float).ravel()
    if ax is None:
        ax = plt.gca()
    if cmap is not None and not isinstance(cmap, str) and np.ndim(cmap) == 2:
        cmap = ListedColormap(np.asarray(cmap, dtype=float))
    if clim is None:
        finite = values[np.isfinite(values)]
        clim = (np.percentile(finite, 0.01, method="hazen"), np.percentile(finite, 99, method="hazen"))
    img = roi_image(values, template)
    im = ax.imshow(img, cmap=cmap, vmin=clim[0], vmax=clim[1], interpolation="nearest")
    border = binary_dilation(template.area_borders[:, ::-1], structure=_disk(border_width))
    rgba = np.zeros((*border.shape, 4))
    rgba[..., :3] = to_rgb(border_color)
    rgba[..., 3] = border
    ax.imshow(rgba, interpolation="nearest")
    if seed is not None:
        rows, cols = np.unravel_index(template.indices[seed] - 1, template.image_shape, order="F")
        x = template.image_shape[1] - 1 - cols.mean()
        ax.scatter([x], [rows.mean()], s=seed_size, color=seed_color, zorder=3)
    ax.set_axis_off()
    ax.set_aspect("equal")
    return im
