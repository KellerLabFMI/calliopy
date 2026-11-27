"""The RoiQuest window (tkinter + matplotlib's TkAgg canvas).

Port of the window parts of calliope ``08_RoiQuest`` (Leonardo Lupori, 2024):
``+gfx/createMainfigure.m`` (figure and menus), ``+clbk/assignAllCallbacks.m`` (callback
wiring), the dialogs of the ``+clbk`` callbacks, ``+clbk/wfAddRefPoints.m`` /
``wfFigButtonDown.m`` (two clicks instead of ``ginput``), ``+clbk/editCustomContrast.m`` and
``+gfx/createContrastSelectionMenu.m`` (slider window), ``+clbk/createHistogram.m`` (histogram
window) and ``+clbk/viewActivity.m`` (ActiQuest window).

All state and logic live in :class:`~calliopy.roiquest.RoiQuestSession`; this module converts
mouse, wheel and key events into session calls, asks MATLAB's questions with tkinter dialogs
and redraws what the session's :class:`~calliopy.roiquest.Redraw` flags name.
"""

from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Callable
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from calliopy.rois import Roi, RoiSaveError, make_cross, plot_rois_to_add

from ._prefs import load_prefs, save_prefs
from ._render import HistogramView, RoiQuestView
from ._session import RoiQuestSession, cellpose_available
from ._tk_actiquest import ActiQuestWindow
from ._tk_dialogs import ContrastDialog, TkDialogs
from ._types import Redraw, RoiQuestRefused

__all__ = ["RoiQuestWindow", "run"]

log = logging.getLogger("calliopy.roiquest")

TITLE = "RoiQuest (beta) - The ROIing adventure begins now!"
_BG = "#1a1a1a"
_WHY = "Go and figure it out."

_WINDOW: RoiQuestWindow | None = None
"""The open RoiQuest window (MATLAB allows one ``fig_img`` at a time)."""


class _StatusHandler(logging.Handler):
    def __init__(self, var: tk.StringVar):
        super().__init__(logging.INFO)
        self.var = var

    def emit(self, record):  # pragma: no cover - trivial
        try:
            lines = [x for x in self.format(record).splitlines() if x.strip()]
            self.var.set(lines[0] if lines else "")
        except (tk.TclError, RuntimeError, ValueError):
            self.handleError(record)


def _mods_of(event) -> set[str]:
    mods = {str(m).lower() for m in (getattr(event, "modifiers", None) or ())}
    key = getattr(event, "key", None)
    if key:
        mods.update(p.lower() for p in str(key).split("+"))
    return {"control" if m in ("ctrl", "control") else m for m in mods}


def selection_type(button, modifiers, dblclick: bool = False) -> str:
    """MATLAB ``SelectionType`` of a mouse press.

    Left button: ``'normal'``; right button or Control + left: ``'alt'``; middle button or
    Shift + left: ``'extend'``; double click: ``'open'``.
    """
    if dblclick:
        return "open"
    b = int(getattr(button, "value", button) or 1)
    if b == 3 or (b == 1 and "control" in modifiers):
        return "alt"
    if b == 2 or (b == 1 and "shift" in modifiers):
        return "extend"
    return "normal"


class RoiQuestWindow:
    """RoiQuest's main window on a :class:`RoiQuestSession`.

    Parameters
    ----------
    session
        A loaded session.
    master
        Parent Tk widget; a hidden ``tk.Tk()`` root is created when omitted (and destroyed
        when the window closes).
    dialogs
        Object with the dialog methods of :class:`TkDialogs` (tests pass a fake).

    Notes
    -----
    Mouse (``figButtonDown``/``figButtonUp``): left drag zooms on the box and proposes a ROI;
    right click selects the ROI under the pointer, right drag selects all ROIs touching the
    box (Control + left = right, as MATLAB's ``'alt'``); Shift + left or middle click toggles
    one ROI. The wheel changes the threshold when zoomed in, the Z layer otherwise. Keys:
    see :data:`calliopy.roiquest.KEY_BINDINGS`; menu accelerators use Ctrl.

    Differences from MATLAB: accelerators are Ctrl + key on every platform (MATLAB uses Cmd on
    macOS); messages printed to MATLAB's Command Window go to the ``calliopy.roiquest`` logger
    and the status line under the image; Print Info / Print all commands open a text window;
    "But... why?" shows one fixed answer (``whyKeller`` is not ported).
    """

    def __init__(self, session: RoiQuestSession, master: tk.Misc | None = None, *, dialogs=None):
        self._own_root = master is None
        if master is None:
            master = tk.Tk()
            master.withdraw()
        self.root = master
        self.session = session
        self.top = tk.Toplevel(master)
        self.top.title(TITLE)
        self.top.configure(bg=_BG)
        sw, sh = self.top.winfo_screenwidth(), self.top.winfo_screenheight()
        w, h = sw // 2, sh // 2
        self.top.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
        self.dialogs = dialogs if dialogs is not None else TkDialogs(self.top)

        self.status = tk.StringVar(value="")
        tk.Label(self.top, textvariable=self.status, anchor="w", bg=_BG, fg="#00cccc",
                 font=("TkDefaultFont", 9)).pack(side="bottom", fill="x")
        self.figure = Figure(figsize=(w / 100, h / 100), dpi=100, facecolor=(0.1, 0.1, 0.1))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.top)
        widget = self.canvas.get_tk_widget()
        widget.configure(cursor="crosshair", bg=_BG, highlightthickness=0)
        widget.pack(side="top", fill="both", expand=True)
        # keys go to the canvas: give it the focus when the pointer enters or clicks
        widget.bind("<Enter>", lambda e: widget.focus_set(), add="+")
        widget.bind("<ButtonPress>", lambda e: widget.focus_set(), add="+")
        self.ax = self.figure.add_axes((0, 0, 1, 1))
        self.view = RoiQuestView(self.ax, session)
        for side in ("top", "bottom", "left", "right"):
            self.ax.spines[side].set_visible(False)

        self.histogram_window: tk.Toplevel | None = None
        self.histogram_canvas = None
        self.actiquest: ActiQuestWindow | None = None
        self.contrast_dialog: ContrastDialog | None = None
        self._ref_clicks: list | None = None
        self._swallow_release = False
        self._press_kind = "normal"

        self.menubar = tk.Menu(self.top)
        self.top.configure(menu=self.menubar)
        self.menus: dict[str, tk.Menu] = {}
        self.build_menus()

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("key_press_event", self._on_key)
        self.top.protocol("WM_DELETE_WINDOW", self.close)

        self._log_handler = _StatusHandler(self.status)
        log.addHandler(self._log_handler)
        if log.level == logging.NOTSET:
            log.setLevel(logging.INFO)
        self.redraw(Redraw.ALL)
        widget.focus_set()

    # ---------------------------------------------------------------------------------
    # window management
    # ---------------------------------------------------------------------------------
    def exists(self) -> bool:
        """True while the window is open."""
        try:
            return bool(self.top.winfo_exists())
        except tk.TclError:
            return False

    def raise_window(self) -> None:
        """Bring the window to the front (MATLAB ``figure(figs_img(1))``)."""
        self.top.deiconify()
        self.top.lift()
        self.canvas.get_tk_widget().focus_set()

    def close(self) -> None:
        """Close the window and its helper windows."""
        global _WINDOW
        for w in (self.actiquest, self.contrast_dialog):
            if w is not None:
                w.destroy()
        if self.histogram_window is not None and self.histogram_window.winfo_exists():
            self.histogram_window.destroy()
        log.removeHandler(self._log_handler)
        if self.exists():
            self.top.destroy()
        if _WINDOW is self:
            _WINDOW = None
        if self._own_root:
            try:
                self.root.quit()
                self.root.destroy()
            except tk.TclError:
                pass

    def _busy(self, on: bool) -> None:
        self.canvas.get_tk_widget().configure(cursor="watch" if on else "crosshair")
        self.top.update_idletasks()

    def redraw(self, redraw: Redraw) -> None:
        """Update menus, artists and helper windows named by ``redraw``."""
        if not self.exists():
            return
        if Redraw.MENUS in redraw:
            self.build_menus()
        self.view.refresh(redraw)
        if self.contrast_dialog is not None and self.contrast_dialog.exists():
            k = self.session.img_idx
            self.contrast_dialog.set_values(self.session.img_lo[k], self.session.img_hi[k])
        self.canvas.draw_idle()
        if self.histogram_canvas is not None and self.histogram_window is not None \
                and self.histogram_window.winfo_exists():
            self.histogram_canvas.draw_idle()

    def _do(self, fn: Callable[[], Redraw | None]) -> None:
        """Run a session action, show refusals and errors, redraw."""
        try:
            r = fn()
        except RoiQuestRefused as err:
            log.info("%s", err)
            return
        except Exception as err:  # keep the window alive, as MATLAB callbacks do
            log.exception("RoiQuest action failed")
            self.dialogs.errordlg("RoiQuest", str(err))
            return
        if isinstance(r, Redraw):
            self.redraw(r)

    # ---------------------------------------------------------------------------------
    # menus (createMainfigure + assignAllCallbacks)
    # ---------------------------------------------------------------------------------
    def _menu_spec(self) -> list:
        s = self.session
        en = s.enabled
        cp = cellpose_available() or s.cell_pose_model is not None

        def item(label, command, accel=None, enabled=True):
            return ("command", label, command, accel, enabled)

        sep = ("separator",)
        view = [
            ("cascade", "Colormap", [
                item("Gray", lambda: self._do(lambda: s.set_colormap("gray"))),
                item("Copper", lambda: self._do(lambda: s.set_colormap("copper"))),
                item("Red", lambda: self._do(lambda: s.set_colormap("red"))),
                item("Parula", lambda: self._do(lambda: s.set_colormap("parula"))),
                item("Hot", lambda: self._do(lambda: s.set_colormap("hot"))),
                sep,
                item("Invert colormap", lambda: self._do(s.invert_colormap)),
            ]),
            ("cascade", "Contrast", [
                item("Min/Max", lambda: self._do(lambda: s.set_contrast("minMax"))),
                item("Best Fit", lambda: self._do(lambda: s.set_contrast("bestFit"))),
                item("Custom", lambda: self._do(lambda: s.set_contrast("custom"))),
                sep,
                item("Change custom contrast", self.edit_custom_contrast),
            ]),
        ]
        if en["setFiltType"]:
            view.append(("cascade", "Change filter Type", [
                item("Updated", lambda: self._do(lambda: s.set_filter_type("updated"))),
                item("Legacy", lambda: self._do(lambda: s.set_filter_type("legacy"))),
            ]))
        if en["setFiltSize"]:
            view.append(item("Filter size", self.ask_filter_size))
        view += [
            sep,
            ("cascade", "Info position", [
                item("Top-Left", lambda: self._do(lambda: s.set_info_position("tl"))),
                item("Bottom-Left", lambda: self._do(lambda: s.set_info_position("bl"))),
                item("Top-Right", lambda: self._do(lambda: s.set_info_position("tr"))),
                item("Bottom-Right", lambda: self._do(lambda: s.set_info_position("br"))),
            ]),
            sep,
            item("Toggle ROI numbers", lambda: self._do(s.toggle_numbers)),
            item("Check image registration", lambda: None, enabled=en["checkReg"]),
        ]
        return [
            ("File", [
                item("Save to AData", self.save, "Ctrl+S"),
                item("Revert to last saved", self.revert, "Ctrl+R"),
                sep,
                item("Save settings", self.save_settings),
                item("Load settings", self.load_settings),
            ]),
            ("View", view),
            ("ROIs", [
                item("Autodetect (cellpose)", self.cellpose_detect, "Ctrl+T",
                     en["cellposeDetect"] and cp),
                item("Load Cellpose Model", self.load_cellpose_model, None,
                     cellpose_available()),
                sep,
                item("Load from AData", self.load_rois_from_adata),
                item("Load from original", lambda: self._do(
                    lambda: s.load_rois_from_adata("original")), None, en["loadRoiOrig"]),
                sep,
                item("View activity", self.view_activity, "Ctrl+V"),
                sep,
                item("Select all", lambda: self._do(s.select_all), "Ctrl+A"),
                item("Select none", lambda: self._do(s.select_none), "Ctrl+N"),
                sep,
                item("Expand selected", lambda: self._do(s.expand_selected), "Ctrl+P",
                     en["roiExpand"]),
                item("Shrink selected", lambda: self._do(s.shrink_selected), "Ctrl+L",
                     en["roiShrink"]),
                sep,
                item("Untag selected", lambda: self._do(s.untag_selected)),
                sep,
                item("Change opacity", self.ask_opacity),
                item("Change flat color", lambda: self.ask_color("roi")),
                item("Change selection color", lambda: self.ask_color("selection")),
                sep,
                ("cascade", "Color mode", [
                    item("Flat", lambda: self._do(lambda: s.set_color_mode("flat"))),
                    item("Random", lambda: self._do(lambda: s.set_color_mode("random"))),
                    item("Tags", lambda: self._do(lambda: s.set_color_mode("tags"))),
                ]),
            ]),
            ("Widefield", [
                item("Define reference points", self.define_ref_points, None,
                     en["wfAddRefPoints"]),
                item("Set up ROIs", self.set_up_wf_rois, None, en["wfAddRois"]),
            ]),
            ("Help", [
                item("Print Info", self.print_info),
                item("Print all commands", self.print_help),
                sep,
                item("But... why?", lambda: self.dialogs.msgbox("Why?", _WHY, "help")),
            ]),
        ]

    def _fill(self, menu: tk.Menu, entries: list) -> None:
        for e in entries:
            if e[0] == "separator":
                menu.add_separator()
            elif e[0] == "cascade":
                sub = tk.Menu(menu, tearoff=0)
                self._fill(sub, e[2])
                menu.add_cascade(label=e[1], menu=sub)
            else:
                _, label, command, accel, enabled = e
                menu.add_command(label=label, command=command, accelerator=accel or "",
                                 state="normal" if enabled else "disabled")

    def build_menus(self) -> None:
        """(Re)build the menubar from the session state (enabled and visible items)."""
        self.menubar.delete(0, "end")
        self.menus = {}
        for label, entries in self._menu_spec():
            m = tk.Menu(self.menubar, tearoff=0)
            self._fill(m, entries)
            self.menubar.add_cascade(label=label, menu=m)
            self.menus[label] = m

    def menu_structure(self) -> list:
        """The menubar as nested lists of ``(kind, label, accelerator, state)`` (for tests)."""
        def walk(menu: tk.Menu) -> list:
            out = []
            last = menu.index("end")
            for i in range(0 if last is None else last + 1):
                kind = menu.type(i)
                if kind == "separator":
                    out.append(("separator",))
                elif kind == "cascade":
                    sub = self.top.nametowidget(menu.entrycget(i, "menu"))
                    out.append(("cascade", menu.entrycget(i, "label"), walk(sub)))
                elif kind == "command":
                    out.append(("command", menu.entrycget(i, "label"),
                                menu.entrycget(i, "accelerator"), menu.entrycget(i, "state")))
            return out

        return [(label, walk(m)) for label, m in self.menus.items()]

    def invoke_menu(self, *path: str) -> None:
        """Invoke a menu item by its labels, e.g. ``invoke_menu('ROIs', 'Select all')``."""
        menu = self.menus[path[0]]
        for label in path[1:-1]:
            menu = self.top.nametowidget(menu.entrycget(label, "menu"))
        menu.invoke(path[-1])

    # ---------------------------------------------------------------------------------
    # mouse, wheel, keys
    # ---------------------------------------------------------------------------------
    def event_rc(self, x: float, y: float) -> tuple[float, float]:
        """Canvas pixel position -> 0-based image ``(row, col)`` (not yet rounded)."""
        col, row = self.ax.transData.inverted().transform((x, y))
        return float(row), float(col)

    def _on_press(self, event) -> None:
        if event.x is None or event.y is None:
            return
        row, col = self.event_rc(event.x, event.y)
        if self._ref_clicks is not None:
            if int(getattr(event.button, "value", event.button) or 1) == 1:
                self._ref_point_click(row, col)
            return
        self._press_kind = selection_type(event.button, _mods_of(event),
                                          bool(getattr(event, "dblclick", False)))
        self.session.button_down(row, col)

    def _on_release(self, event) -> None:
        if self._ref_clicks is not None:
            return
        if self._swallow_release:
            self._swallow_release = False
            return
        if event.x is None or event.y is None:
            return
        row, col = self.event_rc(event.x, event.y)
        self._do(lambda: self.session.button_up(row, col, self._press_kind))

    def _on_scroll(self, event) -> None:
        step = float(getattr(event, "step", 0) or 0)
        if step == 0:
            return
        count = -int(np.sign(step)) * max(1, round(abs(step)))
        self._do(lambda: self.session.scroll(count))

    def _on_key(self, event) -> None:
        if event.key:
            self.key_press(event.key)

    def key_press(self, key: str, modifiers=()) -> None:
        """Forward a key to :meth:`RoiQuestSession.key_press` and complete its requests."""
        res = self.session.key_press(key, modifiers)
        if res.executed:
            if res.action == "god_mode" and Redraw.MENUS in res.redraw:
                self._god_mode_message()
            self.redraw(res.redraw)
            return
        handler = {
            "save": self.save, "revert": self.revert, "cellpose_detect": self.cellpose_detect,
            "view_activity": self.view_activity, "histogram": self.open_histogram,
            "confirm_delete": self.confirm_delete,
        }.get(res.action)
        if handler is not None:
            handler()

    def _god_mode_message(self) -> None:
        if self.session.god_mode:
            msg = ['"With great power comes great responsibility"', "", "- Uncle Ben"]
        else:
            msg = "Looks like we're back in the groove, just the way it's meant to be."
        self.dialogs.msgbox("GodMode!!", msg, "help")

    # ---------------------------------------------------------------------------------
    # File menu
    # ---------------------------------------------------------------------------------
    def _save_call(self, **kw):
        s = self.session
        try:
            return s.save(**kw)
        except RoiSaveError as err:
            if "activity traces" not in str(err):
                raise
            ans = self.dialogs.questdlg(
                "Overwriting ROI activity traces?",
                ["WARNING!", "ROIs in the AData file have activity traces.",
                 ("Overwriting ROIs will delete all traces and you will need to recalculate "
                  "activity."), "", "Do you  want to proceed?"], ["Yes", "No"], "No")
            if ans != "Yes":
                log.info("ROI saving aborted.")
                return None
            return s.save(overwrite_activity=True, **kw)

    def ask_add_new_rois(self, res) -> bool:
        """Show the alignment of new ROIs on all timepoints, then ask (``saveRoisChronic``)."""
        log.info("\t- Visualizing results of the automatching. Close the figure to continue.")
        fig = plot_rois_to_add(res.rois_to_add, np.asarray(self.session.template[0]).shape[:2])
        self.show_figure_and_wait(fig, "NEW ROIs - Check alignment - Close to continue")
        ans = self.dialogs.questdlg(
            "Save these ROIs?", "Are you satisfied with the matching and want to proceed adding "
            "them in all timepoints?", ["Yes", "No"], "No")
        if ans != "Yes":
            log.info("\t- No new ROI(s) added")
        return ans == "Yes"

    def show_figure_and_wait(self, fig: Figure, title: str) -> None:
        """Show a matplotlib figure in a window and wait until it is closed (MATLAB ``waitfor``)."""
        top = tk.Toplevel(self.top)
        top.title(title)
        c = FigureCanvasTkAgg(fig, master=top)
        c.get_tk_widget().pack(fill="both", expand=True)
        c.draw()
        top.grab_set()
        top.wait_window()

    def save(self) -> None:
        """File > Save to AData (``clbk.saveToAdata`` with its dialogs)."""
        s = self.session
        try:
            if s.is_wf_eeg or s.is_widefield_data:
                policy = None
                if len(s.exp_group) > 1:
                    if s.curr_exp_id != s.exp_group[0]:
                        self.dialogs.errordlg(
                            "Can't save ExpID that is not first",
                            [(f"The current ExpID ({s.curr_exp_id}) is part of an experiment "
                              "group but it's not the first."), "",
                             (f"To save widefiled ROIs, please open the first ExpID "
                              f"({s.exp_group[0]}).")])
                        return
                    ans = self.dialogs.questdlg(
                        "Propagate ROIs to all ExpIDs",
                        "Do you want to propagate these ROIs to all other ExpIDs in the ExpGroup?",
                        ["Propagate without overwriting", "Propagate and overwrite"],
                        "Propagate without overwriting")
                    if ans is None:
                        return
                    policy = "keep" if ans == "Propagate without overwriting" else "overwrite"
                    log.info("Propagating ROIs to all other ExpIDs in the ExpGroup...")
                self._busy(True)
                out = self._save_call(overwrite_policy=policy)
            else:
                ans = self.dialogs.questdlg(
                    "Save ROIs to AData?",
                    ["Do you want to save these ROIs to the AData file?", "",
                     "All ROIs in the AData file will be overwritten!"], ["Save", "Cancel"], "Save")
                if ans != "Save":
                    log.info("ROI saving aborted.")
                    return
                if s.orig_adata_full_path is not None:
                    ans = self.dialogs.questdlg(
                        "Save chronic ROIs?",
                        ["This expID is part of a chronic recording.",
                         "Saving will trigger automatic ROIs matching across all timepoints",
                         "Do you want to proceed?"], ["Yes", "No"], "Yes")
                    if ans != "Yes":
                        log.info("ROI saving aborted.")
                        return
                self._busy(True)
                out = self._save_call(add_new_rois=self.ask_add_new_rois)
        except (RoiSaveError, RoiQuestRefused, ImportError) as err:
            log.info("%s", err)
            self.dialogs.errordlg("ROIs not saved", str(err))
            return
        except Exception as err:
            log.exception("Saving failed")
            self.dialogs.errordlg("ROIs not saved", str(err))
            return
        finally:
            self._busy(False)
        if out is not None:
            for path in out.written:
                log.info("New ROIs succesfully saved to:\n\t- %s", path)
            self.redraw(Redraw.OVERLAY)

    def ask_load_original(self, path, original) -> bool:
        """The "Load original ROIs?" question of ``getInfoFromOriginalAData`` (two-photon only)."""
        return ask_load_original(self.dialogs, path, original)

    def revert(self) -> None:
        """File > Revert to last saved (``clbk.revertToSaved``)."""
        ans = self.dialogs.questdlg(
            "Revert all changes?",
            ["WARNING!", "Do you want to revert all changes to the last saved AData?", "",
             "All unsaved progress will be lost!"], ["Revert", "Cancel"], "Cancel")
        if ans != "Revert":
            return
        s = self.session
        load_orig = self.ask_load_original(s.adata_full_path, s.orig_adata_full_path)
        self._do(lambda: s.revert_to_saved(load_original_rois=load_orig))

    def load(self, adata_path, original_adata_path=None) -> None:
        """Open another adata file in this window (``RoiQuest.loadNewExpid``)."""
        s = self.session
        load_orig = self.ask_load_original(adata_path, original_adata_path)
        self._do(lambda: s.load_new_adata(adata_path, original_adata_path,
                                          load_original_rois=load_orig))

    def save_settings(self) -> None:
        """File > Save settings (``utils.saveUserPrefs``, JSON)."""
        p = self.dialogs.asksaveasfilename("Save your custom RoiQuest preferences.",
                                           initialfile=self.session.prefs_default_name())
        if p:
            self._do(lambda: save_prefs(self.session, p))

    def load_settings(self) -> None:
        """File > Load settings (``utils.loadUserPrefs``, JSON)."""
        p = self.dialogs.askopenfilename("Load preferences.",
                                         filetypes=(("JSON", "*.json"), ("All files", "*.*")))
        if p:
            self._do(lambda: load_prefs(self.session, p))

    # ---------------------------------------------------------------------------------
    # View menu dialogs
    # ---------------------------------------------------------------------------------
    def edit_custom_contrast(self) -> None:
        """View > Contrast > Change custom contrast (slider window)."""
        if self.contrast_dialog is not None and self.contrast_dialog.exists():
            self.contrast_dialog.top.lift()
            return
        self.top.update_idletasks()
        x, y = self.top.winfo_rootx(), self.top.winfo_rooty() + self.top.winfo_height() + 30
        if y + 120 > self.top.winfo_screenheight():
            x, y = 200, 200
        k = self.session.img_idx
        self.contrast_dialog = ContrastDialog(self.top, self.session.img_lo[k],
                                              self.session.img_hi[k], self._contrast_changed,
                                              position=(x, y))

    def _contrast_changed(self, black: float, white: float, moved: str) -> None:
        self._do(lambda: self.session.set_custom_contrast(black, white, moved=moved))

    def ask_filter_size(self) -> None:
        """View > Filter size (``clbk.setFilterSize``)."""
        s = self.session
        default = min(np.asarray(s.template[0]).shape[:2]) / 20
        dtxt = str(int(default)) if float(default).is_integer() else f"{default:g}"
        ans = self.dialogs.askstring("Filter size", f"New filter size (default:{dtxt})",
                                     str(s.filter_win_size))
        if ans is None:
            return
        try:
            r = s.set_filter_size(ans)
        except ValueError as err:
            log.info("%s", err)
            return
        self.redraw(r)

    def ask_opacity(self) -> None:
        """ROIs > Change opacity (``clbk.changeOpacity``)."""
        s = self.session
        ans = self.dialogs.askstring("Set ROIs opacity", "New opacity (default: 0.20)",
                                     f"{s.ovrl_opacity:g}")
        if ans is None:
            return
        try:
            v = float(ans)
            r = s.set_opacity(v)
        except ValueError:
            log.info("Invalid opacity. (0-1)")
            return
        self.redraw(r)

    def ask_color(self, which: str) -> None:
        """ROIs > Change flat color / Change selection color (``uisetcolor``)."""
        s = self.session
        if which == "roi":
            rgb = self.dialogs.askcolor(s.roi_bg_color, "Select a color for ROIs")
            if rgb is not None:
                self._do(lambda: s.set_roi_color(rgb))
        else:
            rgb = self.dialogs.askcolor(s.roi_sel_bg_color, "Select a color for ROI selection")
            if rgb is not None:
                self._do(lambda: s.set_selection_color(rgb))

    def open_histogram(self) -> None:
        """Key ``h``: histogram window below the main window (``clbk.createHistogram``)."""
        if self.histogram_window is not None and self.histogram_window.winfo_exists():
            log.info("Histogram open already. Now updating it")
            self.histogram_window.lift()
            self.view.histogram.update()
            self.histogram_canvas.draw_idle()
            return
        self.top.update_idletasks()
        w = max(self.top.winfo_width(), 200)
        x = self.top.winfo_rootx()
        y = self.top.winfo_rooty() + self.top.winfo_height() + 30
        top = tk.Toplevel(self.top)
        top.title("RoiQuest - Histogram")
        top.geometry(f"{w}x100+{x}+{y}")
        fig = Figure(figsize=(w / 100, 1.0), dpi=100, facecolor=(0.1, 0.1, 0.1))
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        ax = fig.add_axes((0, 0, 1, 1))
        self.view.histogram = HistogramView(ax, self.session)
        self.histogram_window, self.histogram_canvas = top, canvas

        def closed():
            self.view.histogram = None
            top.destroy()

        top.protocol("WM_DELETE_WINDOW", closed)
        canvas.draw_idle()
        log.info("Opened image histogram.")

    # ---------------------------------------------------------------------------------
    # ROIs menu
    # ---------------------------------------------------------------------------------
    def load_cellpose_model(self) -> bool:
        """ROIs > Load Cellpose Model (``clbk.loadModel``)."""
        p = self.dialogs.askopenfilename("Select which Cellpose model to use",
                                         filetypes=(("All files", "*.*"),))
        if not p:
            log.info("No model selected.")
            return False
        self._busy(True)
        try:
            log.info("Loading CellPose model (can take long: 30-60s)...")
            self.session.load_cellpose_model(p)
        except (ImportError, OSError, RuntimeError, ValueError, TypeError) as err:
            log.info("FAILED - could not load a Cellpose model.")
            self.dialogs.errordlg("Cellpose", str(err))
            return False
        finally:
            self._busy(False)
        return True

    def cellpose_detect(self) -> None:
        """ROIs > Autodetect (cellpose) (``clbk.autodetectCellpose``)."""
        s = self.session
        if s.is_widefield_data:
            return
        if s.cell_pose_model is None:
            if not cellpose_available():
                self.dialogs.errordlg("Cellpose", "Cellpose is not installed; automatic ROI "
                                      "detection is not available.")
                return
            if not self.load_cellpose_model():
                return
        self._busy(True)
        try:
            self._do(s.cellpose_detect)
        finally:
            self._busy(False)

    def load_rois_from_adata(self) -> None:
        """ROIs > Load from AData (``clbk.loadRoisFromAdata``, ``loadRois`` tag)."""
        s = self.session
        p = self.dialogs.askopenfilename("Select AData file to load ROIs from",
                                         initialdir=None if s.adata_path is None
                                         else str(s.adata_path))
        if not p:
            log.info("Loading ROIs aborted.")
            return
        self._do(lambda: s.load_rois_from_adata(Path(p)))

    def view_activity(self) -> None:
        """ROIs > View activity (``clbk.viewActivity``): open or update ActiQuest."""
        s = self.session
        if not s.selected_indices():
            self.dialogs.msgbox("No ROIs selected",
                                ["You did not select any ROI for visualizing activity.",
                                 "Please select at least one ROI."], "error")
            return
        try:
            rois, idx = s.activity_selection()
        except RoiQuestRefused as err:
            log.info("%s", err)
            return
        nums = ",".join(str(i + 1) for i in idx)
        if self.actiquest is None or not self.actiquest.exists():
            log.info("Opening ActiQuest for ROIs: #[%s]", nums)
            self.top.update_idletasks()
            pos = (self.top.winfo_rootx() + 50, self.top.winfo_rooty(),
                   max(self.top.winfo_width(), 400), max(self.top.winfo_height(), 300))
            self.actiquest = ActiQuestWindow(self.top, rois, idx, pos)
        else:
            log.info("Updating ActiQuest with ROIs: #[%s]", nums)
            self.actiquest.load_new_rois(rois, idx)

    def confirm_delete(self) -> None:
        """MATLAB ``deleteCallback``'s dialogs for chronically tracked ROIs, then delete."""
        s = self.session
        kind = s.deletion_kind()
        if kind not in ("mixed", "tracked"):
            self._do(s.delete_selected)
            return
        if kind == "mixed":
            ans = self.dialogs.questdlg(
                "ROIs deletion", ["Some of the selected ROIs are chonically tracked.",
                                  "Do you want to delete them as well?"],
                ["Delete only new ROIs", "Delete all selected", "Cancel"], "Cancel")
            if ans in (None, "Cancel"):
                log.info("ROI deletion aborted.")
                return
            if ans == "Delete only new ROIs":
                self._do(s.delete_selected)
                return
        ans = self.dialogs.questdlg(
            "Delete ROIs?",
            ["Do you want to delete these ROIs to the AData file?",
             "This ExpID is part of a chronic series. ROIs will be deleted from all timepoints.",
             f"ExpIDs to check: {' '.join(str(e) for e in s.exp_group)}", "",
             ("To ensure consistency, proceeding will trigger auto-saving of any changes before "
              "the deletion process."), "Do you want to proceed?"], ["Delete", "Cancel"], "Cancel")
        if ans != "Delete":
            log.info("ROI deletion aborted.")
            return
        self._busy(True)
        try:
            self._do(lambda: s.delete_selected(include_tracked=True,
                                               add_new_rois=self.ask_add_new_rois))
        finally:
            self._busy(False)

    # ---------------------------------------------------------------------------------
    # Widefield menu
    # ---------------------------------------------------------------------------------
    def _ref_prompt(self, n: int) -> None:
        msg = f"Select reference point #{n} (left-click)"
        log.info("Please select reference point #%u... (left-click)", n)
        self.view.info_text.set_text(self.session.overlay_text(free_form_msg=msg))
        self.view.info_text.set_visible(True)
        self.canvas.draw_idle()

    def define_ref_points(self) -> None:
        """Widefield > Define reference points: the next two left clicks (``wfAddRefPoints``)."""
        s = self.session
        s.ref_points = np.zeros((0, 2))
        s.ref_rois = []
        self.redraw(Redraw.OVERLAY)
        self._ref_clicks = []
        self._ref_prompt(1)

    def _ref_point_click(self, row: float, col: float) -> None:
        s = self.session
        self._ref_clicks.append((row, col))
        if len(self._ref_clicks) == 1:
            hw = np.asarray(s.template[0]).shape[:2]
            r = int(min(max(round(row), 0), hw[0] - 1))
            c = int(min(max(round(col), 0), hw[1] - 1))
            s.ref_rois = [Roi(make_cross((r, c), hw), hw, rng=s.rng)]
            self.redraw(Redraw.OVERLAY)
            self._ref_prompt(2)
            return
        pts = self._ref_clicks
        self._ref_clicks = None
        self._swallow_release = True
        self._do(lambda: s.wf_add_ref_points(pts))

    def set_up_wf_rois(self) -> None:
        """Widefield > Set up ROIs (``clbk.wfAddRois``)."""
        s = self.session
        if not s.ref_rois or len(s.ref_points) < 2:
            self.dialogs.errordlg("No reference points!",
                                  ["ERROR!", "There are no reference points to create ROIs.", "",
                                   "Please define reference points first."])
            return
        if s.roi_template_dir is not None:
            files = s.roi_template_files()
            i = self.dialogs.listdlg("ROI template selection", "Select ROItemplate file", files)
            if i is None:
                log.info("ROI selection aborted.")
                return
            template = files[i]
        else:
            template = self.dialogs.askopenfilename("Select ROItemplate file")
            if not template:
                log.info("ROI selection aborted.")
                return
        self._busy(True)
        try:
            self._do(lambda: s.wf_add_rois(template))
        finally:
            self._busy(False)

    # ---------------------------------------------------------------------------------
    # Help menu
    # ---------------------------------------------------------------------------------
    def print_info(self) -> None:
        """Help > Print Info (``clbk.printInfo``)."""
        text = self.session.info_text()
        log.info("\n%s", text)
        self.dialogs.show_text("RoiQuest - Info", text)

    def print_help(self) -> None:
        """Help > Print all commands (``clbk.printHelp``)."""
        self.dialogs.show_text("RoiQuest - Manual", self.session.help_text())


def ask_load_original(dialogs, path, original) -> bool:
    """Ask "Load original ROIs?" when a two-photon file of a chronic series is opened.

    Returns True (MATLAB's default answer) when there is no original or the file is widefield.
    """
    if original is None:
        return True
    from ._load import var_names

    try:
        names = var_names(Path(path))
    except (OSError, ValueError, KeyError):
        return True
    if "first_frame" in names or "first_frame_ch525" in names:
        return True
    ans = dialogs.questdlg(
        "Load original ROIs?",
        ["Do you want to load ROIs from the orginal ExpID in the series?", "",
         "This ExpID is part of a chronic series. You can auto-load original ROIs."],
        ["Yes", "No"], "Yes")
    return ans == "Yes"


def run(adata_path=None, *, expid: int | None = None, exp_log=None, adata_dir=None,
        original_adata_path=None, roi_template_dir=None, block: bool = True, master=None,
        dialogs=None) -> RoiQuestWindow | None:
    """Open RoiQuest on an adata file or an ExpID (MATLAB ``RoiQuest(expId)``).

    Parameters
    ----------
    adata_path
        Adata file to open; when omitted (and no ``expid``), a file dialog asks for one.
    expid
        Open an ExpID instead (needs ``exp_log`` and ``adata_dir``); the original of a chronic
        series is found as in MATLAB's ``expid2adataPath``.
    exp_log, adata_dir
        ExpLog table and adata root (chronic series, ExpID lookup).
    original_adata_path
        Original recording of the chronic series of ``adata_path``.
    roi_template_dir
        Folder of widefield ROI templates (``ROItemplate_library``).
    block
        Run the Tk main loop until the window is closed. Use ``False`` inside an existing Tk
        application or in tests.
    master
        Parent Tk widget (default: a new hidden root).
    dialogs
        Replacement dialog object (tests).

    Returns
    -------
    RoiQuestWindow or None
        The window (the existing one, brought to the front, when RoiQuest is already open, as
        MATLAB does); None when the file dialog is cancelled.
    """
    global _WINDOW
    if _WINDOW is not None and _WINDOW.exists():
        _WINDOW.raise_window()
        return _WINDOW
    own_root = master is None
    root = tk.Tk() if own_root else master
    if own_root:
        root.withdraw()
    dlg = dialogs if dialogs is not None else TkDialogs(root)
    s = RoiQuestSession(exp_log=exp_log, adata_dir=adata_dir, roi_template_dir=roi_template_dir)
    if expid is not None:
        if s.exp_log is None or s.adata_dir is None:
            raise ValueError("opening by ExpID needs exp_log and adata_dir")
        from ._load import expid2adata_path

        path, orig = expid2adata_path(int(expid), s.adata_dir, s.exp_log)
    else:
        if adata_path is None:
            adata_path = dlg.askopenfilename("Select an AData file")
            if not adata_path:
                if own_root:
                    root.destroy()
                return None
        path, orig = Path(adata_path), original_adata_path
    log.info("Loading AData file:\n\t- %s", path)
    s.load_new_adata(path, orig, load_original_rois=ask_load_original(dlg, path, orig))
    win = RoiQuestWindow(s, master=root, dialogs=dialogs)
    win._own_root = own_root
    _WINDOW = win
    if block:
        root.mainloop()
    return win
