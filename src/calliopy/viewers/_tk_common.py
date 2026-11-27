"""Shared parts of the viewer windows (tkinter + matplotlib's TkAgg canvas).

Only imported when a window is opened. The dialogs are RoiQuest's
(:class:`calliopy.roiquest._tk_dialogs.TkDialogs`), so tests can replace them the same way.
"""

from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Callable

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from calliopy.roiquest._tk_dialogs import TkDialogs

__all__: list[str] = []

log = logging.getLogger("calliopy.viewers")


def split_key(key: str | None) -> tuple[str, tuple[str, ...]]:
    """matplotlib key string (``'ctrl+shift+right'``) -> (``'right'``, modifiers)."""
    if not key:
        return "", ()
    if key == "+" or key.endswith("++"):
        parts = key[:-2].split("+") if key.endswith("++") else []
        return "+", tuple(p for p in parts if p)
    parts = key.split("+")
    mods = tuple("control" if p in ("ctrl", "control") else p for p in parts[:-1])
    return parts[-1], mods


class ViewerWindow:
    """A Toplevel with one matplotlib canvas; base of the viewer windows.

    Parameters
    ----------
    title
        Window title.
    master
        Parent Tk widget; a hidden ``tk.Tk()`` root is made when omitted and destroyed when
        the window closes.
    dialogs
        Object with the methods of RoiQuest's ``TkDialogs`` (tests pass a fake).
    """

    def __init__(self, title: str, master: tk.Misc | None = None, *, dialogs=None,
                 figsize=(10, 6), facecolor="k") -> None:
        self._own_root = master is None
        if master is None:
            master = tk.Tk()
            master.withdraw()
        self.root = master
        self.top = tk.Toplevel(master)
        self.top.title(title)
        self.dialogs = dialogs if dialogs is not None else TkDialogs(self.top)
        self.figure = Figure(figsize=figsize, dpi=100, facecolor=facecolor)
        self.canvas: FigureCanvasTkAgg | None = None
        self.children: list[tk.Toplevel] = []
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        self._closed = False

    def make_canvas(self, parent: tk.Misc) -> FigureCanvasTkAgg:
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        widget = self.canvas.get_tk_widget()
        widget.configure(highlightthickness=0)
        widget.bind("<Enter>", lambda e: widget.focus_set(), add="+")
        widget.bind("<ButtonPress>", lambda e: widget.focus_set(), add="+")
        return self.canvas

    def exists(self) -> bool:
        if self._closed:
            return False
        try:
            return bool(self.top.winfo_exists())
        except tk.TclError:
            return False

    def draw(self) -> None:
        if self.canvas is not None and self.exists():
            self.canvas.draw_idle()

    def show_figure(self, fig: Figure, title: str, *, key: str | None = None,
                    on_key: Callable | None = None) -> tk.Toplevel:
        """Show a figure in its own Toplevel (MATLAB ``figure(n)``; reused by ``key``)."""
        for w in list(self.children):
            try:
                if key is not None and getattr(w, "_viewer_key", None) == key and w.winfo_exists():
                    w.destroy()
                    self.children.remove(w)
            except tk.TclError:
                self.children.remove(w)
        top = tk.Toplevel(self.root)
        top.title(title)
        top._viewer_key = key  # type: ignore[attr-defined]
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        if on_key is not None:
            canvas.mpl_connect("key_press_event", on_key)
        canvas.draw_idle()
        top._viewer_canvas = canvas  # type: ignore[attr-defined]
        self.children.append(top)
        return top

    def close(self) -> None:
        """Close the window, its figure windows and (when it owns it) the hidden root."""
        if self._closed:
            return
        self._closed = True
        for w in self.children:
            try:
                if w.winfo_exists():
                    w.destroy()
            except tk.TclError:
                pass
        try:
            if self.top.winfo_exists():
                self.top.destroy()
        except tk.TclError:
            pass
        if self._own_root:
            try:
                self.root.quit()
                self.root.destroy()
            except tk.TclError:
                pass

    def wait(self) -> None:
        """Block until the window is closed (processing events meanwhile)."""
        try:
            self.top.wait_window()
        except tk.TclError:
            pass
        if self._own_root:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
