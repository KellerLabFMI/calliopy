"""The cell browser window (tkinter): calliope ``20_MetaData/cell_browser.m``.

All state lives in :class:`~calliopy.viewers.CellBrowserSession`; this module turns key
presses, clicks, list selections and text entries into session calls and redraws.
"""

from __future__ import annotations

import logging
import tkinter as tk
from os import PathLike
from pathlib import Path

from ._cell_browser import N_CHANNEL_ROWS, CellBrowserSession, CellBrowserView
from ._tk_common import ViewerWindow, split_key

__all__ = ["CellBrowserWindow", "run_cell_browser"]

log = logging.getLogger("calliopy.viewers")

_BG = "black"
_PANEL = "#1a1a1a"
_FG = "white"
_GREEN = "#00ff00"
_HELP = ("Navigation:\n< >   btw cells\nSHIFT + < >   btw layers\nALT + < >   btw sites\n"
         "^ v   btw timepoints\nSPACE   selection\ns v m a t   plugins")


class CellBrowserWindow(ViewerWindow):
    """Window of a :class:`CellBrowserSession` (MATLAB figure ``cbh.mf``).

    Header: animal, site, cell and timepoint, the site and cell annotation fields (Return
    or leaving the field stores the text), the comment, the save and update buttons and
    the key help. Left: five channel lists (row 0 at the bottom, multiple selection).
    Right: traces and ROI thumbnail (:class:`CellBrowserView`). Keys go to the canvas (see
    :data:`CELL_BROWSER_KEYS`); plugin figures open in their own windows, one per MATLAB
    figure number.

    Differences from MATLAB: a channel list selection redraws at once (MATLAB applied it at
    the next redraw or with "update"); "save" asks for a file name when the session has none
    and does not open the file afterwards (MATLAB ``winopen``); the window size, not
    MATLAB's screen position, is kept in ``win_pos``.
    """

    def __init__(self, session: CellBrowserSession, master: tk.Misc | None = None, *, dialogs=None) -> None:
        super().__init__(session.title or "cell browser", master, dialogs=dialogs, figsize=(10, 5))
        self.session = session
        top = self.top
        top.configure(bg=_BG)
        w, h = int(session.win_pos[2]), int(session.win_pos[3])
        if w > 50 and h > 50:
            top.geometry(f"{w}x{h}")

        header = tk.Frame(top, bg=_BG)
        header.pack(side="top", fill="x")
        buttons = tk.Frame(header, bg=_BG)
        buttons.pack(side="left", padx=4)
        for text, cmd in (("save", self.save), ("update", self.redraw)):
            tk.Button(buttons, text=text, command=cmd, bg=_PANEL, fg=_GREEN, width=8).pack(pady=2)
        grid = tk.Frame(header, bg=_BG)
        grid.pack(side="left", padx=8)
        self.vars = {k: tk.StringVar() for k in ("animal", "site", "cell", "tp", "comment")}
        for r, (label, key) in enumerate((("Animal ID:", "animal"), ("Site:", "site"), ("Cell nbr:", "cell"),
                                          ("Time point:", "tp"))):
            tk.Label(grid, text=label, bg=_BG, fg=_FG, font=("TkDefaultFont", 11), anchor="e"
                     ).grid(row=r, column=0, sticky="e")
            tk.Label(grid, textvariable=self.vars[key], bg=_BG, fg=_FG, font=("TkDefaultFont", 11, "bold"),
                     anchor="w").grid(row=r, column=1, sticky="w", padx=(4, 12))
        self.site_entry = tk.Entry(grid, width=50, bg=_BG, fg=_FG, insertbackground=_FG)
        self.cell_entry = tk.Entry(grid, width=50, bg=_BG, fg=_FG, insertbackground=_FG)
        self.site_entry.grid(row=0, column=2, sticky="w")
        self.cell_entry.grid(row=1, column=2, sticky="w")
        for entry, fn in ((self.site_entry, self.commit_site_annotation),
                          (self.cell_entry, self.commit_cell_annotation)):
            entry.bind("<Return>", lambda e, fn=fn: fn())
            entry.bind("<FocusOut>", lambda e, fn=fn: fn())
        tk.Label(grid, text="Comment:", bg=_BG, fg=_FG, font=("TkDefaultFont", 11)).grid(row=2, column=2, sticky="w")
        tk.Label(grid, textvariable=self.vars["comment"], bg=_BG, fg=_FG).grid(row=3, column=2, sticky="w")
        tk.Label(header, text=_HELP, bg=_BG, fg=_GREEN, justify="left", font=("TkDefaultFont", 8)
                 ).pack(side="left", padx=12)

        body = tk.Frame(top, bg=_BG)
        body.pack(side="top", fill="both", expand=True)
        lists = tk.Frame(body, bg=_BG)
        lists.pack(side="left", fill="y")
        self.listboxes: list[tk.Listbox] = []
        for row in range(N_CHANNEL_ROWS):
            lb = tk.Listbox(lists, selectmode="multiple", exportselection=False, bg=_PANEL, fg=_GREEN,
                            width=18, height=6)
            for name in session.rdfields:
                lb.insert("end", name)
            for i in session.channels[row]:
                lb.selection_set(i)
            lb.bind("<<ListboxSelect>>", lambda e, r=row: self.on_select(r))
            self.listboxes.append(lb)
        for lb in reversed(self.listboxes):  # row 0 at the bottom, as MATLAB
            lb.pack(side="top", fill="both", expand=True, pady=1)

        self.make_canvas(body)
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
        self.view = CellBrowserView(self.figure, session)
        self.canvas.mpl_connect("key_press_event", self._on_key)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        top.bind("<Configure>", self._on_configure, add="+")
        self.redraw()
        self.canvas.get_tk_widget().focus_set()

    # --------------------------------------------------------------------------------
    def redraw(self) -> None:
        """MATLAB ``plot_all``: update the header texts, traces and thumbnail."""
        if not self.exists():
            return
        labels = self.session.labels()
        for k, var in self.vars.items():
            var.set(labels[k])
        for entry, key in ((self.site_entry, "site_annotation"), (self.cell_entry, "cell_annotation")):
            entry.delete(0, "end")
            entry.insert(0, labels[key])
        self.view.refresh()
        self.draw()

    def on_select(self, row: int) -> None:
        self.session.set_channels(row, self.listboxes[row].curselection())
        self.redraw()

    def commit_site_annotation(self) -> None:
        self.session.set_site_annotation(self.site_entry.get())

    def commit_cell_annotation(self) -> None:
        self.session.set_cell_annotation(self.cell_entry.get())

    def _on_configure(self, event) -> None:
        if event.widget is self.top:
            s = self.session
            s.win_pos = (float(s.win_pos[0]), float(s.win_pos[1]), float(event.width), float(event.height))

    def _on_key(self, event) -> None:
        key, mods = split_key(event.key)
        if key:
            self.key_press(key, mods)

    def key_press(self, key: str, modifiers=()) -> None:
        """Forward a key to the session and complete what it asks for."""
        if key == "escape" and self.session.pending_zoom is not None:
            self.session.cancel_zoom()
            self.redraw()
            return
        action = self.session.key_press(key, modifiers)
        if action is None:
            return
        if action.startswith("plugin:"):
            self.show_plugin(action.split(":", 1)[1])
            return
        if action == "pick_zoom":
            log.info("click twice on the traces to set the zoom window (Escape cancels)")
        self.redraw()

    def _on_click(self, event) -> None:
        if event.inaxes is self.view.ax and event.xdata is not None and self.session.pending_zoom is not None:
            self.session.add_zoom_click(event.xdata)
            self.redraw()

    def show_plugin(self, name: str) -> None:
        """Open the figures of a plugin for the current cell."""
        try:
            figs = self.session.plugin_figures(name)
        except (KeyError, IndexError, ValueError) as err:
            self.dialogs.errordlg("cell browser", f"{name} plugin: {err}")
            return
        for fig in figs:
            label = fig.get_label() or name
            self.show_figure(fig, label, key=label)
        self.canvas.get_tk_widget().focus_set()

    def save(self) -> Path | None:
        """MATLAB ``cba_save``: write the annotations file."""
        s = self.session
        path = s.annotations_file
        if path is None:
            chosen = self.dialogs.asksaveasfilename("Save CB annotations", filetypes=(("MAT-files", "*.mat"),),
                                                    defaultextension=".mat")
            if not chosen:
                return None
            path = Path(chosen)
        self.commit_site_annotation()
        self.commit_cell_annotation()
        return s.save_annotations(path)


def run_cell_browser(proj_meta, *, cdisp: int = 1, good_cells: bool = False,
                     annotations_file: str | PathLike | None = None, comments=None, block: bool = True,
                     master: tk.Misc | None = None, dialogs=None) -> CellBrowserWindow:
    """Open the cell browser (MATLAB ``cell_browser(proj_meta, cdisp, good_cells)``).

    Parameters
    ----------
    proj_meta : struct array or path
        The proj_meta, or a metafile path (read with :func:`calliopy.io.load_metafile`).
    cdisp, good_cells, annotations_file, comments
        See :class:`CellBrowserSession`.
    block : bool, default True
        Wait until the window is closed. ``False`` inside a running Tk application or in tests.
    master
        Parent Tk widget (default: a new hidden root).
    dialogs
        Replacement dialog object (tests).

    Returns
    -------
    CellBrowserWindow
        ``window.session`` holds the state (annotations, position) after closing.

    Notes
    -----
    Keys: see :data:`calliopy.viewers.CELL_BROWSER_KEYS`.
    """
    if isinstance(proj_meta, (str, PathLike)):
        from calliopy.io import load_metafile

        proj_meta = load_metafile(proj_meta)
    session = CellBrowserSession(proj_meta, cdisp=cdisp, good_cells=good_cells,
                                 annotations_file=annotations_file, comments=comments)
    win = CellBrowserWindow(session, master, dialogs=dialogs)
    if block:
        win.wait()
    return win
