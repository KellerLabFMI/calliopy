"""ActiQuest without a window: activity traces of selected ROIs.

Port of calliope ``08_RoiQuest/ActiQuest/ActiQuest.m``, ``AQ_keyParser.m``,
``AQ_toggleDarkMode.m``, ``AQ_updatePlots.m`` and ``AQ_updateVisualization.m`` (Leonardo
Lupori, 2024). The data computations are :func:`calliopy.rois.actiquest_activity` and
:func:`calliopy.rois.actiquest_traces`; :class:`calliopy.roiquest.ActiQuestView` draws them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from calliopy.rois import actiquest_activity, actiquest_traces, get_labels_colormap

from ._keys import normalize_key
from ._types import Redraw

__all__ = ["ActiQuestSession"]


class ActiQuestSession:
    """State of an ActiQuest window, MATLAB ``ActiQuest`` (``08_RoiQuest/ActiQuest``).

    Parameters
    ----------
    rois
        ROIs with an ``activity`` trace each (:meth:`RoiQuestSession.activity_selection`).
    roi_indices
        0-based indices of the ROIs in their layer; the plots label them from 1, as RoiQuest
        numbers ROIs (default ``0 .. n-1``).

    Attributes
    ----------
    act_data : ndarray
        ``[nRois x nFrames]`` median-divided activity minus 1.
    raster_lo, raster_hi : float
        Colour limits of the raster (0 and 2).
    is_dark_mode : bool
        Dark colour scheme.
    """

    def __init__(self, rois: Sequence, roi_indices: Sequence[int] | None = None):
        self.raster_lo = 0.0
        self.raster_hi = 2.0
        self.is_dark_mode = False
        self.load_new_rois(rois, roi_indices)

    def load_new_rois(self, rois: Sequence, roi_indices: Sequence[int] | None = None) -> Redraw:
        """Show other ROIs, MATLAB ``ActiQuest.loadNewRois`` / ``AQ_updateVisualization``."""
        self.rois = list(rois)
        self.roi_indices = list(range(len(self.rois))) if roi_indices is None \
            else [int(i) for i in roi_indices]
        if len(self.roi_indices) != len(self.rois):
            raise ValueError("roi_indices must have one entry per ROI")
        self.act_data = actiquest_activity([np.asarray(r.activity) for r in self.rois])
        return Redraw.ALL

    @property
    def plot_data(self) -> np.ndarray:
        """Offset traces (:func:`calliopy.rois.actiquest_traces`) for ``raster_hi``."""
        return actiquest_traces(self.act_data, self.raster_hi)

    @property
    def roi_labels(self) -> list[str]:
        """Tick labels of the traces: ROI numbers from 1."""
        return [str(i + 1) for i in self.roi_indices]

    @property
    def colors(self) -> np.ndarray:
        """``(nRois, 3)`` trace colours: each ROI's random colour in RoiQuest."""
        cm = get_labels_colormap()
        return np.array([cm[int(r.rand_color_idx) - 1] for r in self.rois]).reshape(-1, 3)

    def key_press(self, key: str, modifiers: Sequence[str] = ()) -> Redraw:
        """MATLAB ``AQ_keyParser``: shift + up/down changes ``raster_hi`` by 0.2 (not below
        0.4); ``d`` toggles the dark mode."""
        k, mods = normalize_key(key, modifiers)
        if k == "uparrow" and "shift" in mods:
            self.raster_hi = self.raster_hi + 0.2
            return Redraw.DISPLAY
        if k == "downarrow" and "shift" in mods:
            self.raster_hi = max(self.raster_hi - 0.2, 0.4)
            return Redraw.DISPLAY
        if k == "d":
            self.is_dark_mode = not self.is_dark_mode
            return Redraw.COLORMAP
        return Redraw.NONE
