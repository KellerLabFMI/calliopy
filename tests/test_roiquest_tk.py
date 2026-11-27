"""RoiQuest and ActiQuest tkinter windows, driven by synthetic matplotlib events.

Skipped when tkinter or a Tcl/Tk display is not available. Windows are created under a hidden
root, never with a running main loop, and closed at teardown; dialogs are replaced by a fake.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")

from matplotlib.backend_bases import KeyEvent, MouseEvent
from test_roiquest_data import (
    BLOBS_Z0,
    H,
    W,
    adata_path,
    explog,
    make_2p,
    make_roi_template,
    make_wf1,
)

from calliopy.io import load_mat
from calliopy.roiquest import cellpose_available, run
from calliopy.roiquest._tk import selection_type


@pytest.fixture(scope="module")
def root():
    try:
        r = tk.Tk()
    except tk.TclError as err:
        pytest.skip(f"Tk cannot start here: {err}")
    r.withdraw()
    yield r
    r.destroy()


class FakeDialogs:
    """Records every dialog; answers come from ``answers`` keyed by the dialog title."""

    def __init__(self, answers=None):
        self.answers = dict(answers or {})
        self.calls = []

    def questdlg(self, title, message, buttons, default=None):
        self.calls.append(("questdlg", title))
        return self.answers.get(title, default)

    def msgbox(self, title, message, icon="info"):
        self.calls.append(("msgbox", title, icon))

    def errordlg(self, title, message):
        self.calls.append(("errordlg", title))

    def askstring(self, title, prompt, initial=""):
        self.calls.append(("askstring", title, initial))
        return self.answers.get(title)

    def askcolor(self, initial, title):
        self.calls.append(("askcolor", title))
        return self.answers.get(title)

    def listdlg(self, title, prompt, items):
        self.calls.append(("listdlg", title, list(items)))
        return self.answers.get(title)

    def askopenfilename(self, title, **kw):
        self.calls.append(("askopenfilename", title))
        return self.answers.get(title)

    def asksaveasfilename(self, title, **kw):
        self.calls.append(("asksaveasfilename", title))
        return self.answers.get(title)

    def show_text(self, title, text):
        self.calls.append(("show_text", title, text))


@pytest.fixture
def open_window(root):
    wins = []

    def make(path=None, answers=None, **kw):
        d = FakeDialogs(answers)
        w = run(path, block=False, master=root, dialogs=d, **kw)
        wins.append(w)
        return w

    yield make
    for w in wins:
        w.close()


def _need_update_mat():
    try:
        from calliopy.io import update_mat  # noqa: F401
    except ImportError:
        pytest.skip("calliopy.io.update_mat is not available")


def _xy(win, row, col):
    return win.ax.transData.transform((col, row))


def mouse(win, down, up=None, button=1, key=None, dblclick=False):
    """Press at ``down`` and release at ``up`` (0-based (row, col)) on the canvas."""
    x, y = _xy(win, *down)
    win.canvas.callbacks.process("button_press_event", MouseEvent(
        "button_press_event", win.canvas, x, y, button=button, key=key, dblclick=dblclick))
    x, y = _xy(win, *(up or down))
    win.canvas.callbacks.process("button_release_event", MouseEvent(
        "button_release_event", win.canvas, x, y, button=button, key=key))


def key(win, k):
    """Press and release a key (matplotlib remembers a pressed key until its release and
    reports it with mouse events, like a held modifier)."""
    win.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", win.canvas, k))
    win.canvas.callbacks.process("key_release_event",
                                 KeyEvent("key_release_event", win.canvas, k))


def scroll(win, step, at=(10, 10)):
    x, y = _xy(win, *at)
    win.canvas.callbacks.process("scroll_event", MouseEvent(
        "scroll_event", win.canvas, x, y, step=step))


def _labels(entries):
    return [e[1] if e[0] != "separator" else "-" for e in entries]


def _entry(structure, menu, label):
    for name, entries in structure:
        if name == menu:
            for e in entries:
                if e[0] != "separator" and e[1] == label:
                    return e
    raise KeyError(label)


@pytest.fixture
def two_p(tmp_path):
    return make_2p(tmp_path / "Adata-S1-T101.mat")


# ---------------------------------------------------------------------------------------
def test_selection_type():
    assert selection_type(1, set()) == "normal"
    assert selection_type(3, set()) == "alt"
    assert selection_type(1, {"control"}) == "alt"
    assert selection_type(2, set()) == "extend"
    assert selection_type(1, {"shift"}) == "extend"
    assert selection_type(1, set(), dblclick=True) == "open"


def test_window_and_menus_2p(open_window, two_p):
    w = open_window(two_p)
    assert w.top.title() == "RoiQuest (beta) - The ROIing adventure begins now!"
    assert str(w.canvas.get_tk_widget().cget("cursor")) == "crosshair"
    st = w.menu_structure()
    assert [m for m, _ in st] == ["File", "View", "ROIs", "Widefield", "Help"]
    file_ = dict(st)["File"]
    assert _labels(file_) == ["Save to AData", "Revert to last saved", "-", "Save settings",
                              "Load settings"]
    assert file_[0][2] == "Ctrl+S" and file_[1][2] == "Ctrl+R"
    assert _labels(dict(st)["View"]) == ["Colormap", "Contrast", "-", "Info position", "-",
                                         "Toggle ROI numbers", "Check image registration"]
    rois = dict(st)["ROIs"]
    assert _labels(rois) == [
        "Autodetect (cellpose)", "Load Cellpose Model", "-", "Load from AData",
        "Load from original", "-", "View activity", "-", "Select all", "Select none", "-",
        "Expand selected", "Shrink selected", "-", "Untag selected", "-", "Change opacity",
        "Change flat color", "Change selection color", "-", "Color mode"]
    acc = {e[1]: e[2] for e in rois if e[0] == "command"}
    assert (acc["Autodetect (cellpose)"], acc["View activity"], acc["Select all"],
            acc["Select none"], acc["Expand selected"], acc["Shrink selected"]) == \
        ("Ctrl+T", "Ctrl+V", "Ctrl+A", "Ctrl+N", "Ctrl+P", "Ctrl+L")
    assert _entry(st, "ROIs", "Load from original")[3] == "disabled"
    assert _entry(st, "ROIs", "Expand selected")[3] == "normal"
    assert _entry(st, "View", "Check image registration")[3] == "disabled"
    assert _entry(st, "Widefield", "Define reference points")[3] == "disabled"
    assert _entry(st, "ROIs", "Autodetect (cellpose)")[3] == (
        "normal" if cellpose_available() else "disabled")
    sub = dict(st)["View"][0][2]  # View > Colormap
    assert _labels(sub) == ["Gray", "Copper", "Red", "Parula", "Hot", "-", "Invert colormap"]
    # god mode shows the filter menus
    for _ in range(5):
        key(w, "g")
    assert ("msgbox", "GodMode!!", "help") in w.dialogs.calls
    assert "Filter size" in _labels(dict(w.menu_structure())["View"])
    assert "Change filter Type" in _labels(dict(w.menu_structure())["View"])


def test_single_instance(open_window, two_p, root):
    w = open_window(two_p)
    assert run(two_p, block=False, master=root, dialogs=FakeDialogs()) is w


def test_mouse_zoom_accept_select_scroll(open_window, two_p):
    w = open_window(two_p)
    s = w.session
    r, c = BLOBS_Z0[1]
    mouse(w, (r - 8, c - 8), (r + 8, c + 8))
    assert s.is_zoomed_in and len(s.putative_rois) == 1
    assert w.ax.get_xlim() == (c - 8, c + 8)
    scroll(w, -1, at=(r, c))  # wheel down while zoomed: threshold + 1
    assert s.thr[0] == 161.0
    key(w, "a")
    assert len(s.rois[0]) == 2 and s.rois[0][1].selected and not s.is_zoomed_in
    key(w, "right")
    assert s.rois[0][1].shift.tolist() == [[0.0, 1.0]]
    mouse(w, BLOBS_Z0[0], button=3)  # right click selects only the first ROI
    assert [x.selected for x in s.rois[0]] == [True, False]
    mouse(w, (r, c + 1), key="shift")  # shift + left toggles the second
    assert [x.selected for x in s.rois[0]] == [True, True]
    key(w, "ctrl+n")
    assert not any(x.selected for x in s.rois[0])
    key(w, "ctrl+a")
    assert all(x.selected for x in s.rois[0])
    key(w, "5")
    assert [x.type for x in s.rois[0]] == [5.0, 5.0]
    w.invoke_menu("ROIs", "Untag selected")
    assert [x.type for x in s.rois[0]] == [0.0, 0.0]
    scroll(w, -1)  # not zoomed: next layer
    assert s.current_z == 1
    key(w, "i")
    assert s.current_z == 0
    key(w, "]")
    assert s.img_idx == 1
    key(w, " ")
    assert not s.roi_visible
    # release outside the canvas is clamped to the image (MATLAB clamps at the figure edge)
    mouse(w, (H - 10, W - 10), (H + 40, W + 40))
    assert s.view.xlim == (W - 10.0, W - 1.0) and s.view.ylim == (H - 10.0, H - 1.0)


def test_save_and_revert_dialogs(open_window, two_p):
    _need_update_mat()
    w = open_window(two_p, answers={"Save ROIs to AData?": "Cancel"})
    s = w.session
    s.rois[0][0].type = 3.0
    key(w, "ctrl+s")
    assert ("questdlg", "Save ROIs to AData?") in w.dialogs.calls and not s.backups
    w.dialogs.answers["Save ROIs to AData?"] = "Save"
    key(w, "ctrl+s")
    assert s.backups
    saved = load_mat(two_p)["ROIs"][0, 0][0, 0]
    assert float(saved["type"]) == 3.0
    s.rois[0] = []
    key(w, "ctrl+r")  # default answer Cancel
    assert s.rois[0] == []
    w.dialogs.answers["Revert all changes?"] = "Revert"
    w.invoke_menu("File", "Revert to last saved")
    assert len(s.rois[0]) == 1 and s.rois[0][0].type == 3.0


def test_view_dialogs_and_contrast_window(open_window, two_p, tmp_path):
    w = open_window(two_p, answers={
        "Set ROIs opacity": "0.5", "Select a color for ROIs": (0.0, 0.0, 1.0),
        "Select a color for ROI selection": (1.0, 1.0, 0.0),
        "Save your custom RoiQuest preferences.": str(tmp_path / "p.json"),
        "Load preferences.": str(tmp_path / "p.json")})
    s = w.session
    w.invoke_menu("ROIs", "Change opacity")
    assert s.ovrl_opacity == 0.5
    w.dialogs.answers["Set ROIs opacity"] = "2"
    w.invoke_menu("ROIs", "Change opacity")
    assert s.ovrl_opacity == 0.5
    w.invoke_menu("ROIs", "Change flat color")
    w.invoke_menu("ROIs", "Change selection color")
    assert s.roi_bg_color == (0.0, 0.0, 1.0) and s.roi_sel_bg_color == (1.0, 1.0, 0.0)
    w.invoke_menu("ROIs", "Color mode", "Tags")
    assert s.roi_color_mode == "tags"
    w.invoke_menu("View", "Colormap", "Red")
    np.testing.assert_allclose(s.cmap[-1], [1, 0, 0])
    w.invoke_menu("View", "Info position", "Bottom-Right")
    assert s.info_pos == "br"
    w.invoke_menu("View", "Toggle ROI numbers")
    assert s.roi_number_visible and w.view.number_texts[0].get_text() == "1"
    w.invoke_menu("View", "Contrast", "Min/Max")
    assert s.contrast_mode == "minMax"
    w.invoke_menu("File", "Save settings")
    s.set_opacity(0.1)
    w.invoke_menu("File", "Load settings")
    assert s.ovrl_opacity == 0.5
    # custom contrast window
    w.invoke_menu("View", "Contrast", "Change custom contrast")
    d = w.contrast_dialog
    assert d is not None and d.exists() and d.top.title() == "Custom Contrast"
    d.set_values(0.6, 0.3)
    d.apply("white")
    assert (s.img_lo[0], s.img_hi[0]) == (0.6, pytest.approx(0.61))
    assert d.white.get() == pytest.approx(0.61)
    key(w, "]")
    assert d.black.get() == 0.0 and d.white.get() == 1.0  # limits of the filtered image
    # god mode filter size dialog
    for _ in range(5):
        key(w, "g")
    w.dialogs.answers["Filter size"] = "5"
    w.invoke_menu("View", "Filter size")
    assert s.filter_win_size == 5
    w.dialogs.answers["Filter size"] = "500"
    w.invoke_menu("View", "Filter size")
    assert s.filter_win_size == 5
    w.invoke_menu("View", "Change filter Type", "Updated")
    assert s.filter_type == "updated"
    w.invoke_menu("Help", "Print Info")
    w.invoke_menu("Help", "Print all commands")
    w.invoke_menu("Help", "But... why?")
    titles = [c[1] for c in w.dialogs.calls if c[0] in ("show_text", "msgbox")]
    assert {"RoiQuest - Info", "RoiQuest - Manual", "Why?"} <= set(titles)


def test_histogram_window(open_window, two_p):
    w = open_window(two_p)
    key(w, "h")
    assert w.histogram_window is not None and w.histogram_window.winfo_exists()
    assert w.histogram_window.title() == "RoiQuest - Histogram"
    hv = w.view.histogram
    assert hv.bars.get_data().values.sum() == H * W
    r, c = BLOBS_Z0[1]
    mouse(w, (r - 8, c - 8), (r + 8, c + 8))
    assert hv.bars.get_data().values.sum() == 17 * 17
    key(w, "up")
    assert hv.threshold_line.get_xdata()[0] == 159.0
    key(w, "h")  # second press updates the open window
    assert w.histogram_window.winfo_exists()


def test_actiquest_window(open_window, two_p):
    w = open_window(two_p)
    s = w.session
    key(w, "ctrl+v")
    assert ("msgbox", "No ROIs selected", "error") in w.dialogs.calls
    s.rois[0][0].activity = 10 + np.random.default_rng(0).random((40, 1))
    key(w, "ctrl+a")
    key(w, "ctrl+v")
    aq = w.actiquest
    assert aq is not None and aq.exists()
    assert aq.top.title() == "ActiQuest - take a look at those traces! (by RoiQuest)"
    aq.key_press("shift+up")
    assert aq.session.raster_hi == pytest.approx(2.2)
    aq.key_press("d")
    assert aq.session.is_dark_mode
    r, c = BLOBS_Z0[1]
    mouse(w, (r - 8, c - 8), (r + 8, c + 8))
    key(w, "a")
    s.rois[0][1].activity = 5 + np.random.default_rng(1).random((40, 1))
    key(w, "ctrl+a")
    key(w, "ctrl+v")
    assert w.actiquest is aq and aq.session.roi_labels == ["1", "2"]
    assert len(aq.view.lines) == 2


def test_cellpose_menu(open_window, two_p):
    w = open_window(two_p)
    if cellpose_available():
        pytest.skip("cellpose is installed: the model dialog would load a real model")
    key(w, "ctrl+t")  # disabled menu: nothing happens
    w.cellpose_detect()
    assert ("errordlg", "Cellpose") in w.dialogs.calls


def test_widefield_window(open_window, tmp_path):
    tdir = tmp_path / "templates"
    tdir.mkdir()
    make_roi_template(tdir / "grid.mat")
    p = make_wf1(tmp_path / "Adata-S1-T401.mat")
    w = open_window(p, roi_template_dir=tdir, answers={"ROI template selection": 0})
    s = w.session
    st = w.menu_structure()
    assert _entry(st, "Widefield", "Define reference points")[3] == "normal"
    assert _entry(st, "Widefield", "Set up ROIs")[3] == "normal"
    assert _entry(st, "ROIs", "Expand selected")[3] == "disabled"
    assert _entry(st, "ROIs", "Autodetect (cellpose)")[3] == "disabled"
    assert w.ax.get_aspect() == 1.0
    w.invoke_menu("Widefield", "Define reference points")
    assert len(s.ref_rois) == 0
    assert w.view.info_text.get_text() == \
        "ExpID: 401 - Select reference point #1 (left-click)"
    mouse(w, (10, 45))
    assert len(s.ref_rois) == 1 and len(s.ref_points) == 0
    assert "#2" in w.view.info_text.get_text()
    mouse(w, (50, 45))
    np.testing.assert_array_equal(s.ref_points, [[10, 45], [50, 45]])
    assert not s.is_zoomed_in
    w.invoke_menu("Widefield", "Set up ROIs")
    assert ("listdlg", "ROI template selection", ["grid.mat"]) in w.dialogs.calls
    assert [r.name for r in s.rois[0]] == ["bregma", "lambda", "square"]
    key(w, "d")  # deleting is refused for widefield data
    assert len(s.rois[0]) == 3
    s.ref_rois = []
    w.invoke_menu("Widefield", "Set up ROIs")
    assert ("errordlg", "No reference points!") in w.dialogs.calls


def test_chronic_open_and_tracked_delete(open_window, tmp_path):
    _need_update_mat()
    adir = tmp_path / "adata"
    for e in (201, 202):
        make_2p(adata_path(adir, e))
    el = explog([(201, 5), (202, 5)])
    w = open_window(expid=202, exp_log=el, adata_dir=adir,
                    answers={"Load original ROIs?": "No", "Delete ROIs?": "Delete"})
    assert ("questdlg", "Load original ROIs?") in w.dialogs.calls
    s = w.session
    assert s.orig_adata_full_path == adata_path(adir, 201)
    assert _entry(w.menu_structure(), "ROIs", "Load from original")[3] == "normal"
    assert len(s.rois[0]) == 1  # this file's own ROI, original ROIs not loaded
    mouse(w, BLOBS_Z0[0], button=3)
    assert s.deletion_kind() == "tracked"
    key(w, "d")
    assert ("questdlg", "Delete ROIs?") in w.dialogs.calls
    for e in (201, 202):
        assert load_mat(adata_path(adir, e))["ROIs"][0, 0].size == 0
    # the file has no ROIs left: the session holds empty layers (MATLAB kept the deleted ROI)
    assert s.rois == [[], []]


def test_command_line(tmp_path):
    from calliopy.roiquest.__main__ import main

    with pytest.raises(SystemExit):
        main(["--expid", "5"])
    r = subprocess.run([sys.executable, "-m", "calliopy.roiquest", "--help"],
                       check=False, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "--templates" in r.stdout


def _press_button(widget, text):
    for child in widget.winfo_children():
        if isinstance(child, tk.Button) and child.cget("text") == text:
            child.invoke()
            return True
        if _press_button(child, text):
            return True
    return False


def _newest_toplevel(root):
    tops = [c for c in root.winfo_children() if isinstance(c, tk.Toplevel)]
    return tops[-1]


def test_real_dialogs(root):
    from calliopy.roiquest._tk_dialogs import TkDialogs

    d = TkDialogs(root)
    root.after(200, lambda: _press_button(_newest_toplevel(root), "Delete all selected"))
    ans = d.questdlg("ROIs deletion", ["a", "b"],
                     ["Delete only new ROIs", "Delete all selected", "Cancel"], "Cancel")
    assert ans == "Delete all selected"
    root.after(200, lambda: _newest_toplevel(root).destroy())
    assert d.questdlg("Q", "closed", ["Yes", "No"], "Yes") is None

    def pick_second():
        top = _newest_toplevel(root)
        lb = next(c for f in top.winfo_children() for c in f.winfo_children()
                  if isinstance(c, tk.Listbox))
        lb.selection_clear(0, "end")
        lb.selection_set(1)
        _press_button(top, "OK")

    root.after(200, pick_second)
    assert d.listdlg("ROI template selection", "Select", ["a.mat", "b.mat"]) == 1


def test_run_with_own_root(two_p):
    w = run(two_p, block=False, dialogs=FakeDialogs())
    try:
        assert w._own_root and w.exists()
    finally:
        w.close()
    assert not w.exists()
