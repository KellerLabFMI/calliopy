"""Windows of the stack aligner (``07_ViewStack/StackAligner.m``) and of the manual widefield
registration (``12_Widefield/register_widefield_exp_manual.m``).
"""

from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Sequence
from os import PathLike

import numpy as np
from matplotlib.figure import Figure

from ._manual_registration import (
    MANUAL_REGISTRATION_KEYS,
    ManualRegistrationSession,
    ManualRegistrationView,
    manual_registration_inputs,
    overlay_figure,
)
from ._stack_aligner import StackAlignerSession, StackAlignerView
from ._tk_common import ViewerWindow, split_key

__all__ = [
    "ManualRegistrationWindow",
    "StackAlignerWindow",
    "run_register_widefield_exp_manual",
    "run_stack_aligner",
]

log = logging.getLogger("calliopy.viewers")


# ---------------------------------------------------------------------------------------
# StackAligner
# ---------------------------------------------------------------------------------------
class StackAlignerWindow(ViewerWindow):
    """Window of a :class:`StackAlignerSession` ("Stack Aligner [Enter to Save]").

    Attributes
    ----------
    result : numpy.ndarray or None
        The 0-based offset ``(x, y, z)`` after Enter; None when the window was closed
        otherwise (MATLAB returns ``[]``).
    """

    def __init__(self, session: StackAlignerSession, master: tk.Misc | None = None, *, dialogs=None) -> None:
        super().__init__("Stack Aligner [Enter to Save]", master, dialogs=dialogs, figsize=(10, 5.5))
        self.session = session
        self.result: np.ndarray | None = None
        self.make_canvas(self.top)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.view = StackAlignerView(self.figure, session)
        self.canvas.mpl_connect("key_press_event", self._on_key)
        self.draw()
        self.canvas.get_tk_widget().focus_set()

    def _on_key(self, event) -> None:
        key, mods = split_key(event.key)
        if key:
            self.key_press(key, mods)

    def key_press(self, key: str, modifiers=()) -> None:
        action = self.session.key_press(key, modifiers)
        if action == "done":
            self.result = self.session.offset.copy()
            self.close()
        elif action == "redraw":
            self.view.refresh()
            self.draw()


def run_stack_aligner(ref_stack, stack, start_xyz: Sequence[float], *, block: bool = True,
                      master: tk.Misc | None = None, dialogs=None) -> StackAlignerWindow:
    """Align a reference stack in a larger stack by hand (MATLAB ``StackAligner``).

    Parameters
    ----------
    ref_stack, stack, start_xyz
        See :class:`StackAlignerSession` (``start_xyz`` is **0-based**).
    block : bool, default True
        Wait until Enter or the window is closed.
    master, dialogs
        Parent Tk widget and replacement dialogs (tests).

    Returns
    -------
    StackAlignerWindow
        ``window.result`` is the final 0-based offset ``(x, y, z)`` (MATLAB ``finalOffset``,
        1-based) or None when the window was closed without Enter.

    Notes
    -----
    Keys: see :data:`calliopy.viewers.STACK_ALIGNER_KEYS`.
    """
    win = StackAlignerWindow(StackAlignerSession(ref_stack, stack, start_xyz), master, dialogs=dialogs)
    if block:
        win.wait()
    return win


# ---------------------------------------------------------------------------------------
# register_widefield_exp_manual
# ---------------------------------------------------------------------------------------
_HELP = "\n".join(f"{k}: {v}" for k, v in MANUAL_REGISTRATION_KEYS.items())


class ManualRegistrationWindow(ViewerWindow):
    """Window of a :class:`ManualRegistrationSession`.

    Keys ``1``-``4`` arm a reference point; the next left click in the matching image sets it
    (Escape cancels). ``t`` computes the transform and opens the overlay (left/right flips
    between the origin and the registered frame, MATLAB ``view_stack``). ``s`` asks for
    confirmation and saves into the adata file (:func:`save_manual_registration`, ``tform`` as
    the 3x3 matrix ``tform.A``); ``q`` also keeps the points for the ``e`` key of a later call
    (``session.exported_refpoints``); ``e`` takes the points given as ``refpoints`` (or
    exported by ``q``).

    Differences from MATLAB: saving asks for confirmation (MATLAB saves on the key press) and
    writes ``tform`` as a matrix (MATLAB: ``simtform2d`` object); the overlay is a two-frame
    view in its own window instead of ``view_stack``.
    """

    def __init__(self, session: ManualRegistrationSession, master: tk.Misc | None = None, *,
                 dialogs=None) -> None:
        super().__init__("register_widefield_exp_manual", master, dialogs=dialogs, figsize=(11, 5.5),
                         facecolor="w")
        self.session = session
        self.armed: tuple[str, int] | None = None
        self.status = tk.StringVar(value="")
        side = tk.Frame(self.top)
        side.pack(side="right", fill="y")
        tk.Label(side, text=_HELP, justify="left", font=("TkDefaultFont", 9)).pack(padx=6, pady=6, anchor="n")
        tk.Label(self.top, textvariable=self.status, anchor="w").pack(side="bottom", fill="x")
        self.make_canvas(self.top)
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
        self.view = ManualRegistrationView(self.figure, session)
        self.canvas.mpl_connect("key_press_event", self._on_key)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.overlay_frame = 0
        self.overlay_window: tk.Toplevel | None = None
        self._overlay_fig: Figure | None = None
        self._overlay_stack: np.ndarray | None = None
        self.draw()
        self.canvas.get_tk_widget().focus_set()

    def _on_key(self, event) -> None:
        key, _mods = split_key(event.key)
        if key:
            self.key_press(key)

    def redraw(self) -> None:
        self.view.refresh()
        self.draw()

    def key_press(self, key: str) -> None:
        s = self.session
        if key == "escape":
            self.armed = None
            self.status.set("")
            return
        slot = s.point_slot(key)
        if slot is not None:
            self.armed = slot
            side = "left" if slot[0] == "fixed" else "right"
            self.status.set(f"click point {slot[1] + 1} in the {side} image")
            return
        if key == "t":
            try:
                s.compute_tform()
            except ValueError as err:
                self.dialogs.errordlg("register_widefield_exp_manual", str(err))
                return
            self.status.set(f"Similarity={s.similarity:.2f}")
            self.show_overlay()
        elif key in ("s", "q"):
            self.save(export_refpoints=key == "q")
        elif key == "e":
            if s.retrieve_previous():
                self.redraw()
            else:
                self.dialogs.msgbox("register_widefield_exp_manual", "No reference points of a previous call.",
                                    "warn")

    def _on_click(self, event) -> None:
        if self.armed is None or event.inaxes is None or event.button not in (1, None):
            return
        image, k = self.armed
        if event.inaxes is not self.view.axes_of(image):
            return
        self.session.set_point(image, k, float(event.xdata), float(event.ydata))
        self.armed = None
        self.status.set("")
        self.redraw()

    def show_overlay(self) -> None:
        self._overlay_stack = self.session.overlay_stack()
        self.overlay_frame = 0
        self._overlay_fig = overlay_figure(self._overlay_stack, 0)
        self.overlay_window = self.show_figure(self._overlay_fig, "registration overlay", key="overlay",
                                               on_key=self._overlay_key)

    def _overlay_key(self, event) -> None:
        key, _ = split_key(event.key)
        if key in ("right", "left", " ") and self._overlay_stack is not None:
            self.flip_overlay()

    def flip_overlay(self) -> None:
        """Show the other frame of the overlay."""
        if self._overlay_stack is None or self._overlay_fig is None:
            return
        self.overlay_frame = 1 - self.overlay_frame
        overlay_figure(self._overlay_stack, self.overlay_frame, self._overlay_fig)
        canvas = getattr(self.overlay_window, "_viewer_canvas", None)
        if canvas is not None:
            canvas.draw_idle()

    def save(self, *, export_refpoints: bool = False) -> bool:
        """``s`` / ``q``: confirm, then write the result into the adata file.

        ``q`` also keeps the points for the ``e`` key of a later call. Returns True when the
        file was written.
        """
        s = self.session
        if s.tform is None:
            self.dialogs.errordlg("register_widefield_exp_manual", "Compute the transform first (key t).")
            return False
        if s.adata_path is None:
            self.dialogs.errordlg("register_widefield_exp_manual", "No adata file to save to.")
            return False
        ans = self.dialogs.questdlg(
            "Save to adata",
            [(f"Write tform{s.rp_string} (3x3 matrix tform.A) and registration_parameters{s.rp_string} "
              f"into {s.adata_path.name}?"), "ROIs in the file are deleted, as in MATLAB."],
            ["Save", "Cancel"], "Save")
        if ans != "Save":
            return False
        s.save(export_refpoints=export_refpoints)
        self.status.set(f"saved to {s.adata_path.name}")
        return True


def run_register_widefield_exp_manual(adata_path: str | PathLike, origin_adata_path: str | PathLike | None = None,
                                      *, channel: str | None = None, frac_boundary: float = 0.1,
                                      normalize_template: bool = False, refpoints=None, block: bool = True,
                                      master: tk.Misc | None = None, dialogs=None) -> ManualRegistrationWindow:
    """Register a widefield experiment to its site by two point pairs (MATLAB
    ``register_widefield_exp_manual``).

    Parameters
    ----------
    adata_path
        Adata file of the experiment (written on ``s``/``q``).
    origin_adata_path
        Adata file of the site's origin (not needed for a two-colour origin's channel map).
    channel
        ``'ch525'`` or ``'ch610'`` for two-colour data; asked in a dialog when needed and
        not given (MATLAB asks in the Command Window).
    frac_boundary, normalize_template
        As MATLAB's name-value arguments (defaults 0.1 and 0).
    refpoints
        ``[4 x 2]`` points of an earlier call for the ``e`` key (MATLAB ``rp``).
    block, master, dialogs
        As :func:`run_cell_browser`.

    Returns
    -------
    ManualRegistrationWindow
        ``window.session`` holds ``tform``, ``similarity``, ``refpoints``, ``saved`` and
        ``matlab_commands()`` (MATLAB lines that save the object form, extra helper).

    Notes
    -----
    Keys: see :data:`calliopy.viewers.MANUAL_REGISTRATION_KEYS`. The ExpID lookup of the
    adata files (ExpLog) is left to the caller.
    """
    own_root = master is None
    root = tk.Tk() if own_root else master
    if own_root:
        root.withdraw()
    try:
        inputs = manual_registration_inputs(adata_path, origin_adata_path, channel=channel)
    except ValueError as err:
        if channel is not None or "channel" not in str(err):
            if own_root:
                root.destroy()
            raise
        from calliopy.roiquest._tk_dialogs import TkDialogs

        dlg = dialogs if dialogs is not None else TkDialogs(root)
        pick = dlg.listdlg("Channel", "Found 2 channels. Which channel to fix registration?", ["ch525", "ch610"])
        if pick is None:
            if own_root:
                root.destroy()
            raise ValueError("no channel chosen") from None
        inputs = manual_registration_inputs(adata_path, origin_adata_path, channel=("ch525", "ch610")[pick])
    session = ManualRegistrationSession(
        inputs.fixed, inputs.moving, frac_boundary=frac_boundary, normalize_template=normalize_template,
        rp_string=inputs.rp_string, adata_path=adata_path, fixed_exp_id=inputs.fixed_exp_id,
        moving_exp_id=inputs.moving_exp_id, previous_refpoints=refpoints)
    win = ManualRegistrationWindow(session, root, dialogs=dialogs)
    win._own_root = own_root
    if block:
        win.wait()
    return win
