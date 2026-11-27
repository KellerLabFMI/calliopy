"""Dialogs of the RoiQuest window (tkinter replacements of MATLAB's dialog functions).

``questdlg``, ``msgbox``/``errordlg``, ``inputdlg``, ``uisetcolor``, ``listdlg``,
``uigetfile``/``uiputfile`` and the custom-contrast slider window of
``08_RoiQuest/+gfx/createContrastSelectionMenu.m``. All dialogs go through one
:class:`TkDialogs` object so that a test can replace them.

Only imported when a window is opened (tkinter is not needed to use the session).
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable, Sequence
from tkinter import colorchooser, filedialog, messagebox, simpledialog

__all__: list[str] = []

_DARK = "#333333"


def _hex(rgb) -> str:
    r, g, b = (min(max(round(float(v) * 255), 0), 255) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


class TkDialogs:
    """The dialogs a :class:`~calliopy.roiquest.RoiQuestWindow` shows.

    Every method blocks until the user answers and returns the answer (``None`` when the
    dialog is closed or cancelled, as MATLAB's empty answers).
    """

    def __init__(self, parent: tk.Misc):
        self.parent = parent

    def _transient(self, top: tk.Toplevel) -> None:
        owner = self.parent.winfo_toplevel()
        if owner.winfo_viewable():  # a transient of a hidden root would stay hidden
            top.transient(owner)

    # MATLAB questdlg -------------------------------------------------------------------
    def questdlg(self, title: str, message: str | Sequence[str], buttons: Sequence[str],
                 default: str | None = None) -> str | None:
        """Question with custom buttons (MATLAB ``questdlg``); returns the button label."""
        text = message if isinstance(message, str) else "\n".join(message)
        top = tk.Toplevel(self.parent)
        top.title(title)
        self._transient(top)
        top.resizable(False, False)
        answer: list[str | None] = [None]
        tk.Label(top, text=text, justify="left", padx=16, pady=12, wraplength=520).pack()
        row = tk.Frame(top, pady=8)
        row.pack()

        def choose(label):
            answer[0] = label
            top.destroy()

        for b in buttons:
            btn = tk.Button(row, text=b, width=max(10, len(b) + 2), command=lambda b=b: choose(b))
            btn.pack(side="left", padx=4)
            if b == default:
                btn.focus_set()
                top.bind("<Return>", lambda e, b=b: choose(b))
        top.bind("<Escape>", lambda e: top.destroy())
        top.grab_set()
        top.wait_window()
        return answer[0]

    def msgbox(self, title: str, message: str | Sequence[str], icon: str = "info") -> None:
        """MATLAB ``msgbox`` (icon ``'info'``/``'help'``, ``'warn'``, ``'error'``)."""
        text = message if isinstance(message, str) else "\n".join(message)
        if icon == "error":
            messagebox.showerror(title, text, parent=self.parent)
        elif icon == "warn":
            messagebox.showwarning(title, text, parent=self.parent)
        else:
            messagebox.showinfo(title, text, parent=self.parent)

    def errordlg(self, title: str, message: str | Sequence[str]) -> None:
        """MATLAB ``errordlg``."""
        self.msgbox(title, message, "error")

    def askstring(self, title: str, prompt: str, initial: str = "") -> str | None:
        """MATLAB ``inputdlg`` with one field."""
        return simpledialog.askstring(title, prompt, initialvalue=initial, parent=self.parent)

    def askcolor(self, initial, title: str):
        """MATLAB ``uisetcolor``: RGB in ``[0, 1]`` or ``None``."""
        rgb, _ = colorchooser.askcolor(color=_hex(initial), title=title, parent=self.parent)
        if rgb is None:
            return None
        return tuple(float(v) / 255.0 for v in rgb)

    def listdlg(self, title: str, prompt: str, items: Sequence[str]) -> int | None:
        """MATLAB ``listdlg`` (single selection); returns the 0-based index."""
        top = tk.Toplevel(self.parent)
        top.title(title)
        self._transient(top)
        tk.Label(top, text=prompt, anchor="w").pack(fill="x", padx=8, pady=(8, 2))
        frame = tk.Frame(top)
        frame.pack(fill="both", expand=True, padx=8)
        lb = tk.Listbox(frame, width=70, height=15, selectmode="browse", exportselection=False)
        sb = tk.Scrollbar(frame, command=lb.yview)
        lb.configure(yscrollcommand=sb.set)
        lb.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        for it in items:
            lb.insert("end", it)
        if items:
            lb.selection_set(0)
        answer: list[int | None] = [None]

        def ok(_e=None):
            sel = lb.curselection()
            answer[0] = int(sel[0]) if sel else None
            top.destroy()

        row = tk.Frame(top, pady=8)
        row.pack()
        tk.Button(row, text="OK", width=10, command=ok).pack(side="left", padx=4)
        tk.Button(row, text="Cancel", width=10, command=top.destroy).pack(side="left", padx=4)
        lb.bind("<Double-Button-1>", ok)
        top.bind("<Return>", ok)
        top.bind("<Escape>", lambda e: top.destroy())
        top.grab_set()
        top.wait_window()
        return answer[0]

    def askopenfilename(self, title: str, initialdir=None, filetypes=(("MAT-files", "*.mat"),
                                                                       ("All files", "*.*"))):
        """MATLAB ``uigetfile``; ``None`` when cancelled."""
        p = filedialog.askopenfilename(title=title, initialdir=initialdir, filetypes=filetypes,
                                       parent=self.parent)
        return p or None

    def asksaveasfilename(self, title: str, initialdir=None, initialfile=None,
                          filetypes=(("JSON", "*.json"),), defaultextension=".json"):
        """MATLAB ``uiputfile``; ``None`` when cancelled."""
        p = filedialog.asksaveasfilename(title=title, initialdir=initialdir,
                                         initialfile=initialfile, filetypes=filetypes,
                                         defaultextension=defaultextension, parent=self.parent)
        return p or None

    def show_text(self, title: str, text: str) -> None:
        """Read-only text window (replaces printing to MATLAB's Command Window)."""
        top = tk.Toplevel(self.parent)
        top.title(title)
        box = tk.Text(top, width=100, height=40, font=("Courier", 9))
        box.insert("1.0", text)
        box.configure(state="disabled")
        box.pack(fill="both", expand=True)


class ContrastDialog:
    """Custom contrast window with a black and a white slider.

    MATLAB ``gfx.createContrastSelectionMenu`` + ``clbk.editCustomContrast``: a 400 x 90
    window placed 120 pixels below the main window (or at 200, 200), sliders from 0 to 1.
    ``on_change(black, white, moved)`` is called when a slider is released (MATLAB
    ``ValueChangedFcn``), ``moved`` being ``'black'`` or ``'white'``.
    """

    def __init__(self, parent: tk.Misc, black: float, white: float,
                 on_change: Callable[[float, float, str], None], position=None):
        self.on_change = on_change
        self.top = tk.Toplevel(parent)
        self.top.title("Custom Contrast")
        self.top.configure(bg=_DARK)
        self.top.resizable(False, False)
        if position is not None:
            self.top.geometry(f"400x110+{int(position[0])}+{int(position[1])}")
        self.black = tk.DoubleVar(value=black)
        self.white = tk.DoubleVar(value=white)
        self.scales = {}
        for name, var in (("Black", self.black), ("White", self.white)):
            row = tk.Frame(self.top, bg=_DARK)
            row.pack(fill="x", padx=6, pady=2)
            tk.Label(row, text=name, fg="white", bg=_DARK, width=6, font=("TkDefaultFont", 11),
                     anchor="e").pack(side="left")
            sc = tk.Scale(row, variable=var, from_=0.0, to=1.0, resolution=0.01,
                          orient="horizontal", length=330, tickinterval=0.2, fg="white",
                          bg=_DARK, highlightthickness=0, troughcolor="#555555")
            sc.pack(side="left")
            moved = name.lower()
            sc.bind("<ButtonRelease-1>", lambda e, m=moved: self.apply(m))
            sc.bind("<KeyRelease>", lambda e, m=moved: self.apply(m))
            self.scales[moved] = sc

    def apply(self, moved: str) -> None:
        """Send the slider values (MATLAB ``contrastSliderChanged``)."""
        self.on_change(float(self.black.get()), float(self.white.get()), moved)

    def set_values(self, black: float, white: float) -> None:
        """Show other limits (after the session corrected them or the image changed)."""
        self.black.set(black)
        self.white.set(white)

    def exists(self) -> bool:
        try:
            return bool(self.top.winfo_exists())
        except tk.TclError:
            return False

    def destroy(self) -> None:
        if self.exists():
            self.top.destroy()
