"""RoiQuest user preferences in a JSON file.

Port of calliope ``08_RoiQuest/+utils/saveUserPrefs.m`` and ``+utils/loadUserPrefs.m``
(Leonardo Lupori, 2024). MATLAB wrote an XML file with ``writestruct``; here the same fields
are written as JSON to a file the caller chooses.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from ._types import Redraw

if TYPE_CHECKING:  # pragma: no cover
    from ._session import RoiQuestSession

__all__ = ["load_prefs", "prefs_dict", "save_prefs"]

log = logging.getLogger("calliopy.roiquest")

PREFS_FIELDS = ("cmap_R", "cmap_G", "cmap_B", "ovrlOpacity", "img_lo", "img_hi", "thr",
                "roiBg", "roiSelBg", "roiColorMode")
"""Fields of the preferences file, MATLAB's names (``saveUserPrefs.m``)."""


def prefs_dict(s: RoiQuestSession) -> dict:
    """The preferences of a session as MATLAB's ``settings`` struct (lists of floats)."""
    cmap = np.asarray(s.cmap, dtype=np.float64)
    return {
        "cmap_R": cmap[:, 0].tolist(),
        "cmap_G": cmap[:, 1].tolist(),
        "cmap_B": cmap[:, 2].tolist(),
        "ovrlOpacity": float(s.ovrl_opacity),
        "img_lo": [float(v) for v in s.img_lo],
        "img_hi": [float(v) for v in s.img_hi],
        "thr": [float(v) for v in s.thr],
        "roiBg": [float(v) for v in s.roi_bg_color],
        "roiSelBg": [float(v) for v in s.roi_sel_bg_color],
        "roiColorMode": str(s.roi_color_mode),
    }


def save_prefs(s: RoiQuestSession, path) -> None:
    """Write the session's preferences to ``path``, MATLAB ``utils.saveUserPrefs``.

    Fields (MATLAB names): ``cmap_R``/``cmap_G``/``cmap_B`` (colormap columns),
    ``ovrlOpacity``, ``img_lo``, ``img_hi``, ``thr``, ``roiBg``, ``roiSelBg`` (RGB) and
    ``roiColorMode``.

    Differences from MATLAB: JSON instead of XML; the file is an argument (MATLAB proposed
    ``<adata folder>/_UserPrefs/RoiQuest/<user>_<project>.xml`` in a dialog; see
    :func:`calliopy.roiquest.prefs_file_name`).
    """
    with open(os.fspath(path), "w", encoding="utf-8") as f:
        json.dump(prefs_dict(s), f, indent=1)
    log.info("Preferences saved.")


def load_prefs(s: RoiQuestSession, path) -> Redraw:
    """Apply a preferences file to the session, MATLAB ``utils.loadUserPrefs``.

    Sets the colormap, overlay opacity, custom contrast limits (not re-applied to the
    images, as in MATLAB), thresholds, ROI colour mode and the flat and selection colours.
    """
    with open(os.fspath(path), encoding="utf-8") as f:
        d = json.load(f)
    missing = [k for k in PREFS_FIELDS if k not in d]
    if missing:
        raise ValueError(f"preferences file lacks {missing}")
    s.cmap = np.column_stack([d["cmap_R"], d["cmap_G"], d["cmap_B"]]).astype(np.float64)
    s.ovrl_opacity = float(d["ovrlOpacity"])
    s.img_lo = [float(v) for v in d["img_lo"]]
    s.img_hi = [float(v) for v in d["img_hi"]]
    s.thr = [float(v) for v in d["thr"]]
    s.roi_color_mode = str(d["roiColorMode"])
    s.roi_sel_bg_color = tuple(float(v) for v in d["roiSelBg"])
    s.roi_bg_color = tuple(float(v) for v in d["roiBg"])
    log.info("Preferences loaded.")
    return Redraw.DISPLAY | Redraw.OVERLAY | Redraw.COLORMAP
