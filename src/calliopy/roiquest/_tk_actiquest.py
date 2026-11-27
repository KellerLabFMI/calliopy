"""ActiQuest window (tkinter): activity traces of the ROIs selected in RoiQuest.

Port of the window parts of calliope ``08_RoiQuest/ActiQuest/ActiQuest.m`` and
``AQ_createMainFigure.m`` (Leonardo Lupori, 2024): a window with the raster and the traces of
:class:`~calliopy.roiquest.ActiQuestView`, the keys of ``AQ_keyParser`` and matplotlib's pan
and zoom toolbar (MATLAB ``enableDefaultInteractivity``).
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Sequence

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from ._actiquest import ActiQuestSession
from ._render import ActiQuestView

__all__ = ["ActiQuestWindow"]

TITLE = "ActiQuest - take a look at those traces! (by RoiQuest)"


class ActiQuestWindow:
    """ActiQuest as a tkinter Toplevel.

    Parameters
    ----------
    master
        Parent Tk widget (the RoiQuest window).
    rois, roi_indices
        ROIs with activity and their 0-based indices (:meth:`RoiQuestSession.activity_selection`).
    position
        ``(x, y, width, height)`` in screen pixels; default half the screen, centred
        (``AQ_createMainFigure``).

    Keys (``AQ_keyParser``): Shift + Up / Shift + Down raise / lower the raster's upper colour
    limit by 0.2 (not below 0.4); ``d`` toggles the dark mode.
    """

    def __init__(self, master: tk.Misc, rois: Sequence, roi_indices: Sequence[int] | None = None,
                 position=None):
        self.session = ActiQuestSession(rois, roi_indices)
        self.top = tk.Toplevel(master)
        self.top.title(TITLE)
        if position is None:
            sw, sh = self.top.winfo_screenwidth(), self.top.winfo_screenheight()
            w, h = sw // 2, sh // 2
            position = ((sw - w) // 2, (sh - h) // 2, w, h)
        x, y, w, h = (int(v) for v in position)
        self.top.geometry(f"{w}x{h}+{x}+{y}")
        self.figure = Figure(figsize=(w / 100, h / 100), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.top)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.top, pack_toolbar=False)
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.view = ActiQuestView(self.figure, self.session)
        self.canvas.mpl_connect("key_press_event", self._on_key)
        self.canvas.draw_idle()

    def _on_key(self, event) -> None:
        self.key_press(event.key or "")

    def key_press(self, key: str, modifiers: Sequence[str] = ()) -> None:
        """Forward a key to :meth:`ActiQuestSession.key_press` and redraw."""
        self.view.refresh(self.session.key_press(key, modifiers))
        self.canvas.draw_idle()

    def load_new_rois(self, rois: Sequence, roi_indices: Sequence[int] | None = None) -> None:
        """Show other ROIs (MATLAB ``ActiQuest.loadNewRois``)."""
        self.view.refresh(self.session.load_new_rois(rois, roi_indices))
        self.canvas.draw_idle()
        self.top.lift()

    def exists(self) -> bool:
        try:
            return bool(self.top.winfo_exists())
        except tk.TclError:
            return False

    def destroy(self) -> None:
        if self.exists():
            self.top.destroy()
