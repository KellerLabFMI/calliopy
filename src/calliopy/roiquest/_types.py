"""Small types shared by the RoiQuest session, its key dispatcher and its renderer."""

from __future__ import annotations

import enum
from pathlib import Path
from typing import NamedTuple

__all__ = ["KeyResult", "Redraw", "RoiQuestRefused", "SaveResult", "View"]


class RoiQuestRefused(RuntimeError):
    """An action was refused, with the message RoiQuest logged before returning.

    MATLAB's callbacks print a message and return for these cases (no ROIs selected,
    widefield data, no original recording, ...). The key dispatcher catches this exception,
    logs the message and reports the key as handled.
    """


class Redraw(enum.Flag):
    """What a window has to redraw after an action (MATLAB's ``gfx.update*`` calls).

    ``DISPLAY``: displayed image (``updateDisplayImage``); ``OVERLAY``: ROI overlay, ROI
    numbers, info text and movement vectors (``updateOverlayImage``); ``VIEW``: axes limits
    (zoom); ``COLORMAP``: image colormap; ``HISTOGRAM``: histogram data and threshold line
    (``updateHistogram``); ``HIST_THRESHOLD``: threshold line only; ``MENUS``: enabled or
    visible menu items (:attr:`RoiQuestSession.enabled`).
    """

    NONE = 0
    DISPLAY = enum.auto()
    OVERLAY = enum.auto()
    VIEW = enum.auto()
    COLORMAP = enum.auto()
    HISTOGRAM = enum.auto()
    HIST_THRESHOLD = enum.auto()
    MENUS = enum.auto()
    ALL = DISPLAY | OVERLAY | VIEW | COLORMAP | HISTOGRAM | HIST_THRESHOLD | MENUS


class View(NamedTuple):
    """Axes limits of the main image, in 0-based pixel-centre coordinates.

    ``xlim = (col_lo, col_hi)`` and ``ylim = (row_lo, row_hi)`` with ``lo < hi`` (the row
    axis points down, as in MATLAB's ``axis ij``). The full image is
    ``(-0.5, W - 0.5), (-0.5, H - 0.5)``; a zoom box has integer limits (pixel centres of the
    clicked corners), like MATLAB's ``XLim``/``YLim`` minus 1.
    """

    xlim: tuple[float, float]
    ylim: tuple[float, float]


class KeyResult(NamedTuple):
    """Outcome of :meth:`RoiQuestSession.key_press`."""

    action: str
    """Name of the action (``''`` when the key is not bound)."""
    redraw: Redraw
    """What to redraw."""
    executed: bool
    """False for requests the window must complete itself (a dialog or another window:
    ``'save'``, ``'revert'``, ``'cellpose_detect'``, ``'view_activity'``, ``'histogram'``,
    ``'confirm_delete'``)."""


class SaveResult(NamedTuple):
    """Outcome of :meth:`RoiQuestSession.save`."""

    written: dict[Path, tuple[str, ...]]
    """Adata file -> names of the variables written into it."""
    backups: dict[Path, Path]
    """Adata file -> backup copy made before the first write of this session."""
    chronic: object = None
    """For a two-photon chronic series, the :class:`calliopy.rois.ChronicRoiSave`."""
