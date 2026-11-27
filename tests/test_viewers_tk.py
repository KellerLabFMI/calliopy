"""calliopy.viewers tkinter windows, driven by synthetic matplotlib events.

Skipped when tkinter or a Tcl/Tk display is not available. Windows are created under a
hidden root, never with a running main loop, and closed at teardown; dialogs are a fake.
"""

from __future__ import annotations

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")

from matplotlib.backend_bases import KeyEvent, MouseEvent
from test_viewers import _check_raw, _frame, _rois_cell, _stacks, make_proj_meta, make_wf_adata

from calliopy.io import load_mat, save_mat
from calliopy.viewers import (
    run_cell_browser,
    run_register_widefield_exp_manual,
    run_stack_aligner,
    run_tpi_alignment_visualize,
    run_wid_alignment_visualize,
)
from calliopy.viewers._common import read_mat_variable
from calliopy.viewers._tk_common import split_key
from calliopy.widefield import similarity_matrix, warp_image


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
        self.calls.append(("errordlg", title, message))

    def listdlg(self, title, prompt, items):
        self.calls.append(("listdlg", title, list(items)))
        return self.answers.get(title)

    def asksaveasfilename(self, title, **kw):
        self.calls.append(("asksaveasfilename", title))
        return self.answers.get(title)

    def show_text(self, title, text):
        self.calls.append(("show_text", title, text))


@pytest.fixture
def windows():
    opened = []
    yield opened
    for w in opened:
        w.close()


def key(win, k, canvas=None):
    c = canvas or win.canvas
    c.callbacks.process("key_press_event", KeyEvent("key_press_event", c, k))
    c.callbacks.process("key_release_event", KeyEvent("key_release_event", c, k))


def click(win, ax, x, y, button=1):
    px, py = ax.transData.transform((x, y))
    win.canvas.callbacks.process("button_press_event", MouseEvent(
        "button_press_event", win.canvas, px, py, button=button))
    win.canvas.callbacks.process("button_release_event", MouseEvent(
        "button_release_event", win.canvas, px, py, button=button))


def test_split_key():
    assert split_key("ctrl+shift+right") == ("right", ("control", "shift"))
    assert split_key("a") == ("a", ())
    assert split_key(None) == ("", ())


def test_cell_browser_window(root, windows, tmp_path):
    d = FakeDialogs({"Save CB annotations": str(tmp_path / "cb-cba.mat")})
    w = run_cell_browser(make_proj_meta(), block=False, master=root, dialogs=d)
    windows.append(w)
    s = w.session
    assert w.top.title() == "PRJ_A1"
    assert w.vars["cell"].get() == "1 / 3 - layer 1 - 1"
    key(w, "right")
    assert s.cell == 1 and w.vars["cell"].get().startswith("2 / 3")
    key(w, "alt+right")
    assert s.site == 1 and w.vars["animal"].get() == "M2"
    key(w, "alt+left")
    assert (s.site, s.cell) == (0, 1)  # site changes keep the closest valid cell
    w.listboxes[1].selection_clear(0, "end")
    w.listboxes[1].selection_set(0)
    w.listboxes[1].selection_set(2)
    w.on_select(1)
    assert s.channels[1] == [0, 2]
    key(w, " ")
    assert s.pending_zoom == []
    click(w, w.view.ax, 300, 2)
    click(w, w.view.ax, 120, 2)
    assert s.zoom and w.view.ax.get_xlim() == pytest.approx((120, 300), abs=1)
    key(w, "s")
    key(w, "t")
    assert len([c for c in w.children if c.winfo_exists()]) == 4  # figure 10 + three trav figures
    key(w, "s")  # figure 10 again: reused, not duplicated
    assert len([c for c in w.children if c.winfo_exists()]) == 4
    w.cell_entry.delete(0, "end")
    w.cell_entry.insert(0, "note")
    w.commit_cell_annotation()
    path = w.save()
    assert path.name == "cb-cba.mat"
    assert read_mat_variable(path, "params")["cell_annotations"][0, 0, 1] == "note"


def test_stack_aligner_window(root, windows):
    ref, big = _stacks()
    w = run_stack_aligner(ref, big, (9, 3, 2), block=False, master=root)
    windows.append(w)
    key(w, "shift+right")
    key(w, "right")
    np.testing.assert_array_equal(w.view.im_crop.get_array(), ref[:, :, 1])
    key(w, "enter")
    assert not w.exists()
    np.testing.assert_array_equal(w.result, [10, 3, 2])


def test_stack_aligner_window_closed_without_enter(root, windows):
    ref, big = _stacks()
    w = run_stack_aligner(ref, big, (10, 3, 2), block=False, master=root)
    windows.append(w)
    w.close()
    assert w.result is None


def test_manual_registration_window(root, windows, tmp_path):
    fixed = _frame()
    true = similarity_matrix(1.0, 0.0, (3.0, -2.0))
    moving = warp_image(fixed, np.linalg.inv(true), fixed.shape)
    orig = make_wf_adata(tmp_path / "Adata-S1-T100.mat", fixed)
    exp = make_wf_adata(tmp_path / "Adata-S1-T105.mat", moving, ROIs=_rois_cell())
    d = FakeDialogs({"Save to adata": "Save"})
    w = run_register_widefield_exp_manual(exp, orig, block=False, master=root, dialogs=d)
    windows.append(w)
    s = w.session
    key(w, "t")
    assert d.calls[-1][0] == "errordlg"
    pts = {"1": ("fixed", (23.0, 17.0)), "2": ("fixed", (53.0, 38.0)),
           "3": ("moving", (20.0, 19.0)), "4": ("moving", (50.0, 40.0))}
    for k, (image, (x, y)) in pts.items():
        key(w, k)
        click(w, w.view.axes_of("moving" if image == "fixed" else "fixed"), x, y)  # wrong image: ignored
        assert s.points[s.point_slot(k)] is None
        click(w, w.view.axes_of(image), x, y)
        assert s.points[s.point_slot(k)] == pytest.approx((x, y), abs=1e-6)
    key(w, "t")
    np.testing.assert_allclose(s.tform, true, atol=1e-6)
    assert w.overlay_window is not None and w.overlay_window.winfo_exists()
    w.flip_overlay()
    assert w.overlay_frame == 1
    key(w, "q")
    assert d.calls[-1][:2] == ("questdlg", "Save to adata")
    assert s.saved and s.exported_refpoints.shape == (4, 2)
    data = load_mat(exp)
    assert "ROIs" not in data and data["registration_parameters"]["registration_type"] == "manual"
    np.testing.assert_allclose(data["tform"], true, atol=1e-6)
    assert "simtform2d" in s.matlab_commands()


def test_manual_registration_asks_channel(root, windows, tmp_path):
    a = _frame((20, 24), 1)
    o = tmp_path / "Adata-S1-T300.mat"
    save_mat(o, {"first_frame_ch525": a, "first_frame_ch610": a})
    e = tmp_path / "Adata-S1-T301.mat"
    save_mat(e, {"first_frame_ch525": a, "first_frame_ch610": a})
    d = FakeDialogs({"Channel": 1})
    w = run_register_widefield_exp_manual(e, o, block=False, master=root, dialogs=d)
    windows.append(w)
    assert w.session.rp_string == "_ch610"
    d2 = FakeDialogs({"Save to adata": "Cancel"})
    w.dialogs = d2
    w.session.set_refpoints([[5, 5], [15, 10], [5, 5], [15, 10]])
    key(w, "t")
    key(w, "s")
    assert not w.session.saved and "tform_ch610" not in load_mat(e)
    d2.answers["Save to adata"] = "Save"
    key(w, "s")
    assert w.session.saved and "tform_ch610" in load_mat(e)


def test_wid_decision_window(root, windows, tmp_path):
    big = _frame((130, 140), 2)
    site = make_wf_adata(tmp_path / "Adata-S1-T10.mat", big)
    exp = make_wf_adata(tmp_path / "Adata-S1-T11.mat", big, check_raw=_check_raw(), tform=np.eye(3))
    w = run_wid_alignment_visualize(exp, site, rng=np.random.default_rng(0), block=False, master=root)
    windows.append(w)
    assert len(w.canvases) == 2
    assert w.answer(True) == 1 and not w.exists()
    assert load_mat(exp)["check_raw"]["passed"] == 1.0


def test_tpi_decision_window_no(root, windows, tmp_path):
    p = tmp_path / "Adata-S1-T42.mat"
    save_mat(p, {"check_raw": _check_raw()})
    w = run_tpi_alignment_visualize(p, block=False, master=root)
    windows.append(w)
    w.top.tk.call(w.top.protocol("WM_DELETE_WINDOW"))  # closing the window = no change
    assert not w.exists()
    assert w.result == 0 and w.session.update == 0
    assert load_mat(p)["check_raw"]["passed"] == 0.0
