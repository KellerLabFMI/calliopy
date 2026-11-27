"""The y/n decision window of ``11_TwoPhoton/tpi_alignment_visualize.m`` and
``12_Widefield/wid_alignment_visualize.m``.
"""

from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Mapping, Sequence
from os import PathLike
from tkinter import ttk
from typing import Any

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ._alignment_check import AlignmentDecision, tpi_alignment_figures, wid_alignment_figures
from ._tk_common import ViewerWindow

__all__ = ["AlignmentDecisionWindow", "run_tpi_alignment_visualize", "run_wid_alignment_visualize"]

log = logging.getLogger("calliopy.viewers")


class AlignmentDecisionWindow(ViewerWindow):
    """Figures of an :class:`AlignmentDecision` in tabs, the check summary and two buttons.

    "Yes - set passed" (key ``y``) writes ``check_raw.passed = 1``; "No change" (key ``n``)
    and closing the window leave the file as it is. ``window.result`` / ``session.update``
    is MATLAB's output ``update`` (1 or 0).

    Differences from MATLAB: the question is asked in the window (MATLAB ``input`` in the
    Command Window); any answer other than yes is no, as in MATLAB.
    """

    def __init__(self, session: AlignmentDecision, master: tk.Misc | None = None, *, dialogs=None) -> None:
        super().__init__("Alignment check", master, dialogs=dialogs)
        self.session = session
        self.result: int | None = None
        top = self.top
        tk.Label(top, text="\n".join(session.text), justify="left", anchor="w", font=("Courier", 9)
                 ).pack(side="top", fill="x", padx=6, pady=4)
        bottom = tk.Frame(top)
        bottom.pack(side="bottom", fill="x", pady=6)
        tk.Label(bottom, text=session.question(), font=("TkDefaultFont", 11, "bold")).pack(side="left", padx=8)
        tk.Button(bottom, text="No change (n)", width=16, command=lambda: self.answer(False)).pack(side="right",
                                                                                                padx=4)
        tk.Button(bottom, text="Yes - set passed (y)", width=20, command=lambda: self.answer(True)
                  ).pack(side="right", padx=4)
        self.canvases: list[FigureCanvasTkAgg] = []
        if session.figures:
            book = ttk.Notebook(top)
            book.pack(side="top", fill="both", expand=True)
            for title, fig in session.figures:
                frame = tk.Frame(book)
                book.add(frame, text=title)
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                canvas.draw_idle()
                self.canvases.append(canvas)
        else:
            tk.Label(top, text="(nothing to show)").pack(side="top", fill="both", expand=True)
        top.bind("<KeyPress-y>", lambda e: self.answer(True))
        top.bind("<KeyPress-n>", lambda e: self.answer(False))
        top.protocol("WM_DELETE_WINDOW", lambda: self.answer(False))
        top.focus_set()

    def answer(self, passed: bool) -> int:
        """Apply the answer, close the window and return ``update``."""
        if self.result is None:
            try:
                self.result = self.session.decide(bool(passed))
            except (OSError, KeyError, ValueError) as err:
                self.dialogs.errordlg("Alignment check", f"check_raw not updated: {err}")
                self.result = 0
                self.session.update = 0
        self.close()
        return self.result


def run_tpi_alignment_visualize(adata_path: str | PathLike, *, data: Sequence[Any] | None = None,
                                site_figures: Sequence[Figure] | None = None, block: bool = True,
                                master: tk.Misc | None = None, dialogs=None) -> AlignmentDecisionWindow:
    """Inspect a failed two-photon check and decide (MATLAB ``tpi_alignment_visualize``).

    Figures: :func:`calliopy.viewers.tpi_alignment_figures`. "Yes" sets
    ``check_raw.passed = 1`` in the adata file (:func:`calliopy.viewers.set_check_raw_passed`).
    The RawDataStatus / ExpLog side is not part of calliopy.

    Returns
    -------
    AlignmentDecisionWindow
        ``window.result`` is MATLAB's ``update`` (1 or 0) once answered.
    """
    win = AlignmentDecisionWindow(tpi_alignment_figures(adata_path, data=data, site_figures=site_figures),
                                  master, dialogs=dialogs)
    if block:
        win.wait()
    return win


def run_wid_alignment_visualize(adata_path: str | PathLike, site_adata_path: str | PathLike | None = None, *,
                                exp_id: int | None = None, site_id: int | None = None,
                                tforms: Mapping[str, Any] | None = None, rng: np.random.Generator | None = None,
                                block: bool = True, master: tk.Misc | None = None,
                                dialogs=None) -> AlignmentDecisionWindow:
    """Inspect a widefield check and decide (MATLAB ``wid_alignment_visualize``).

    Figures: :func:`calliopy.viewers.wid_alignment_figures` (same arguments). "Yes" sets
    ``check_raw.passed = 1`` in the adata file. The RawDataStatus / ExpLog side is not part
    of calliopy.

    Returns
    -------
    AlignmentDecisionWindow
        ``window.result`` is MATLAB's ``update`` (1 or 0) once answered.
    """
    decision = wid_alignment_figures(adata_path, site_adata_path, exp_id=exp_id, site_id=site_id,
                                     tforms=tforms, rng=rng)
    win = AlignmentDecisionWindow(decision, master, dialogs=dialogs)
    if block:
        win.wait()
    return win
