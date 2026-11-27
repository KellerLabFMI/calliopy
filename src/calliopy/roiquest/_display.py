"""What RoiQuest shows, computed without any graphics toolkit.

Port of the computational parts of calliope ``08_RoiQuest/+gfx/updateOverlayImage.m``,
``+gfx/updateHistogram.m``, ``+clbk/createHistogram.m``, ``+clbk/setImageColormap.m`` and
``+utils/updateText.m`` (Leonardo Lupori, 2024). The matplotlib renderer
(:mod:`calliopy.roiquest._render`) draws these arrays and strings; a window toolkit only
forwards events.

Coordinates: 0-based pixel centres, ``x`` = column and ``y`` = row (matplotlib's ``imshow``
convention). Texts that RoiQuest shows to the user keep MATLAB's 1-based numbering of Z
layers and ROIs.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import matplotlib
import matplotlib.colors
import numpy as np

from calliopy.rois import (
    rois_binary_mask,
    rois_label_image,
    rois_label_type_image,
)

if TYPE_CHECKING:  # pragma: no cover
    from ._session import RoiQuestSession

__all__ = [
    "COLORMAP_NAMES",
    "OverlayImage",
    "colormap",
    "histogram_data",
    "info_text_anchor",
    "overlay_image",
    "overlay_text",
    "roi_number_labels",
]

COLORMAP_NAMES = ("gray", "copper", "red", "parula", "hot")
"""Colormaps of RoiQuest's View > Colormap menu."""

SELECTED_ALPHA = 0.9
"""Opacity of selected ROIs and selected reference points (``updateOverlayImage.m``)."""


def colormap(name: str, n: int = 256) -> np.ndarray:
    """``(n, 3)`` RGB colormap of the View > Colormap menu (``clbk.setImageColormap``).

    Parameters
    ----------
    name
        ``'gray'``, ``'copper'``, ``'red'`` (black to red, lab definition), ``'parula'`` or
        ``'hot'``; any other name gives ``'gray'``, as MATLAB's ``otherwise`` branch.
    n
        Number of colours (RoiQuest uses 256).

    Returns
    -------
    numpy.ndarray
        float64 ``(n, 3)`` in ``[0, 1]``; row ``k`` is the colour of uint8 value ``k``.

    Notes
    -----
    Differences from MATLAB (display only): ``'parula'`` is MATLAB's own colormap and is not
    available outside MATLAB; matplotlib's ``viridis`` (also perceptually uniform, dark blue
    to yellow) is used instead. ``'gray'``, ``'copper'`` and ``'hot'`` are matplotlib's
    colormaps of the same name; ``gray`` and ``copper`` follow the same formulas as MATLAB,
    ``hot`` differs slightly in where the red/yellow/white segments start.
    """
    x = np.linspace(0.0, 1.0, n)
    if name == "red":
        return np.column_stack([x, np.zeros(n), np.zeros(n)])
    mpl_name = {"copper": "copper", "parula": "viridis", "hot": "hot"}.get(name, "gray")
    return np.asarray(matplotlib.colormaps[mpl_name](x)[:, :3], dtype=np.float64)


class OverlayImage(NamedTuple):
    """Result of :func:`overlay_image`."""

    rgba: np.ndarray
    """``(H, W, 4)`` float64 RGBA in ``[0, 1]``: colours and per-pixel opacity."""
    overlay: np.ndarray
    """``(H, W)`` the overlay values MATLAB derives the opacity from (binary mask, summed
    label image, ...)."""
    n_pixels: float
    """``sum(overlay(:))``: the pixel count shown in the zoomed info text."""


def _hsv(n: int) -> np.ndarray:
    h = np.arange(n) / max(n, 1)
    return matplotlib.colors.hsv_to_rgb(np.column_stack([h, np.ones(n), np.ones(n)]))


def _ind2rgb(index: np.ndarray, cmap: np.ndarray) -> np.ndarray:
    """MATLAB ``ind2rgb`` of a double index image: 1-based, clamped to the colormap."""
    idx = np.clip(np.asarray(index, dtype=np.float64), 1, cmap.shape[0]).astype(np.int64)
    return cmap[idx - 1]


def overlay_image(s: RoiQuestSession) -> OverlayImage:
    """ROI overlay of the current Z layer, MATLAB ``gfx.updateOverlayImage`` (image part).

    Display modes, as in MATLAB:

    - not zoomed and no putative ROIs: the ROIs of the layer, coloured by
      ``s.roi_color_mode`` (``'flat'``: ``s.roi_bg_color``; ``'random'``: each ROI's
      ``rand_color_idx`` row of the labels colormap; ``'tags'``: ``type + 1`` row of the labels
      colormap for two-photon data, and for widefield data a ``hsv`` colour per distinct
      ``Allen_index`` rank);
    - zoomed without putative ROIs: nothing;
    - with putative ROIs (zoom selection or Cellpose): the putative ROIs in
      ``s.roi_sel_bg_color``.

    Widefield reference points are drawn on top. The opacity is ``s.ovrl_opacity`` on
    overlay pixels, 0.9 on selected ROIs (only when there are no putative ROIs) and on
    selected reference points, and 0 everywhere when ``s.roi_visible`` is False.

    Returns
    -------
    OverlayImage

    Notes
    -----
    Kept as in MATLAB: the reference-point colour is written as ``uint8(255*c)`` into the RGB
    image; when that image is double (all modes except "no ROI in the layer"), MATLAB's
    display clips it to ``[1, 1, 0]`` (yellow) and only in the "no ROI" mode it shows
    ``c = [0.8, 1, 0]``. Both colours are reproduced.
    """
    h, w = s.image_hw
    rgb = np.zeros((h, w, 3))
    overlay = np.zeros((h, w))
    if not s.roi_visible:
        return OverlayImage(np.zeros((h, w, 4)), overlay, 0.0)
    rois = s.rois[s.current_z] if s.rois else []
    uint8_rgb = False
    if not s.putative_rois and not s.is_zoomed_in:
        if not rois:
            uint8_rgb = True
        elif s.roi_color_mode == "flat":
            overlay = rois_binary_mask(rois).astype(np.float64)
            rgb = np.broadcast_to(np.asarray(s.roi_bg_color, dtype=np.float64), (h, w, 3)).copy()
        elif s.roi_color_mode == "random":
            lab = rois_label_image(rois).astype(np.float64)
            overlay = lab.sum(axis=2) if lab.ndim == 3 else lab
            rgb = _ind2rgb(overlay, s.lab_cmap)
        else:  # 'tags'
            lab = np.asarray(rois_label_type_image(rois), dtype=np.float64)
            overlay = lab.sum(axis=2) if lab.ndim == 3 else lab
            if s.is_widefield_data:
                k = np.unique(overlay).size
                rgb = _ind2rgb(overlay + 1, np.vstack([[0.0, 0.0, 0.0], _hsv(k)]))
            else:
                rgb = _ind2rgb(overlay + 1, s.lab_cmap)
    elif not s.putative_rois and s.is_zoomed_in:
        pass
    else:
        overlay = rois_binary_mask(s.putative_rois).astype(np.float64)
        rgb = np.broadcast_to(np.asarray(s.roi_sel_bg_color, dtype=np.float64),
                              (h, w, 3)).copy()

    if s.is_widefield_data and s.ref_rois:
        ref_mask = rois_binary_mask(s.ref_rois)
        overlay = ((overlay != 0) | ref_mask).astype(np.float64)
        col = np.floor(s.lab_cmap[3] * 255 + 0.5)  # uint8(labCmap(4,:)*255)
        rgb[ref_mask] = col / 255.0 if uint8_rgb else np.minimum(col, 1.0)

    alpha = (overlay > 0) * float(s.ovrl_opacity)
    if not s.putative_rois and rois:
        sel = [r for r in rois if r.selected]
        if sel:
            alpha[rois_binary_mask(sel)] = SELECTED_ALPHA
    if s.ref_rois:
        sel = [r for r in s.ref_rois if r.selected]
        if sel:
            alpha[rois_binary_mask(sel)] = SELECTED_ALPHA
    rgba = np.dstack([np.clip(rgb, 0.0, 1.0), alpha])
    return OverlayImage(rgba, overlay, float(overlay.sum()))


def _fmt_u(v) -> str:
    """MATLAB ``sprintf('%u', v)`` for the values RoiQuest prints (NaN -> ``'NaN'``)."""
    if v is None:
        return "NaN"
    f = float(v)
    if math.isnan(f):
        return "NaN"
    return str(int(f)) if f == int(f) else f"{f:e}"


def overlay_text(s: RoiQuestSession, n_pixels: float | None = None,
                 free_form_msg: str | None = None) -> str:
    """Info text drawn on the image, MATLAB ``utils.updateText`` as RoiQuest calls it.

    Parameters
    ----------
    s
        The session.
    n_pixels
        Pixel count of the overlay (:attr:`OverlayImage.n_pixels`), shown when zoomed in
        (two-photon data only).
    free_form_msg
        Widefield message shown instead of the channel (``wfAddRefPoints`` uses
        ``'Select reference point #1 (left-click)'``).

    Returns
    -------
    str
        ``'ExpID: 7 - Z:1 - 12 ROI(s)'`` (two-photon, not zoomed), ``'ExpID: 7 - Z:1 - 140
        pixels'`` (zoomed), ``'ExpID: 7 - Channel: 1 - 40 ROI(s)'`` (widefield);
        ``'ORIGINAL - '`` is prepended while the original recording is shown (not zoomed);
        ``''`` when the ROIs are hidden. Z layers are numbered from 1 as in MATLAB.
    """
    if not s.roi_visible and free_form_msg is None:
        return ""
    exp = _fmt_u(s.curr_exp_id)
    z = s.current_z + 1
    if s.is_widefield_data:
        if free_form_msg:
            return f"ExpID: {exp} - {free_form_msg}"
        msg = f"ExpID: {exp} - Channel: {z}"
    else:
        msg = f"ExpID: {exp} - Z:{z}"
    if s.is_zoomed_in:
        if not s.is_widefield_data and n_pixels is not None:
            msg += f" - {_fmt_u(n_pixels)} pixels"
        return msg
    n_rois = len(s.rois[s.current_z]) if s.rois else 0
    msg += f" - {n_rois} ROI(s)"
    if s.show_original:
        msg = "ORIGINAL - " + msg
    return msg


def info_text_anchor(s: RoiQuestSession) -> tuple[float, float, str, str]:
    """Position and alignment of the info text (``changeInfoPos.m``, ``updateOverlayImage.m``).

    Returns
    -------
    tuple
        ``(x, y, horizontal_alignment, vertical_alignment)``; ``x``/``y`` in 0-based pixel
        coordinates (column, row). Zoomed in: the chosen corner of the zoom box. Otherwise
        MATLAB's ``infoPosCoord`` (``[0, 2]``, ``[0, H-2]``, ``[W, 2]``, ``[W, H-2]`` in
        1-based axes coordinates) minus 1.

    Notes
    -----
    Differences from MATLAB (display only): MATLAB stores the position computed from the
    image size when the corner is chosen and keeps it after loading an image of another
    size; here it is computed from the current image size.
    """
    pos = s.info_pos
    ha = "left" if pos[1] == "l" else "right"
    va = "top" if pos[0] == "t" else "bottom"
    if s.is_zoomed_in:
        x = s.view.xlim[0] if pos[1] == "l" else s.view.xlim[1]
        y = s.view.ylim[0] if pos[0] == "t" else s.view.ylim[1]
        return float(x), float(y), ha, va
    h, w = s.image_hw
    x = 0.0 if pos[1] == "l" else float(w)
    y = 2.0 if pos[0] == "t" else float(h - 2)
    return x - 1.0, y - 1.0, ha, va


def roi_number_labels(s: RoiQuestSession) -> list[tuple[float, float, str]]:
    """ROI numbers drawn on the overlay (``updateOverlayImage.m``, "Add ROI numbers").

    Returns
    -------
    list of tuple
        ``(x, y, label)`` per ROI of the current layer: ``x``/``y`` (0-based column, row) of
        the ROI pixel with the smallest linear index, ``label`` the 1-based ROI number as
        MATLAB shows it. Empty when numbers or ROIs are hidden.
    """
    if not (s.roi_visible and s.roi_number_visible) or not s.rois:
        return []
    h, _ = s.image_hw
    out = []
    for i, roi in enumerate(s.rois[s.current_z]):
        ind = np.asarray(roi.indices, dtype=np.float64)
        if ind.size == 0:
            continue
        k = int(ind.min()) - 1
        out.append((float(k // h), float(k % h), str(i + 1)))
    return out


def histogram_data(s: RoiQuestSession) -> tuple[np.ndarray, np.ndarray]:
    """Histogram of the visible part of the image (``createHistogram.m``, ``updateHistogram.m``).

    The image is ``s.imgs[s.img_idx][s.current_z]`` (also while the original recording is
    shown, as in MATLAB), cropped to the pixels whose centres lie inside the axes limits
    (MATLAB ``ceil``/``floor`` of ``YLim``/``XLim``).

    Returns
    -------
    counts, edges
        ``(255,)`` int64 counts and the ``(256,)`` edges ``0, 1, ..., 255`` of MATLAB's
        ``histogram(data, 0:255)`` (the last bin includes 255).
    """
    img = np.asarray(s.imgs[s.img_idx][s.current_z])
    (c0, c1), (r0, r1) = s.view.xlim, s.view.ylim
    h, w = img.shape[:2]
    rs, re = max(math.ceil(r0), 0), min(math.floor(r1), h - 1) + 1
    cs, ce = max(math.ceil(c0), 0), min(math.floor(c1), w - 1) + 1
    edges = np.arange(256, dtype=np.float64)
    counts, _ = np.histogram(img[rs:re, cs:ce].astype(np.float64).ravel(), bins=edges)
    return counts.astype(np.int64), edges
