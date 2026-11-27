"""calliopy.viewers without a GUI: sessions, figures and the adata writes (synthetic data).

Ports of cell_browser.m (+ plugins), StackAligner.m, register_widefield_exp_manual.m,
tpi_alignment_visualize.m and wid_alignment_visualize.m. The helpers that build synthetic
proj_meta and adata files are also used by ``test_viewers_tk.py``.
"""

from __future__ import annotations

import logging

import h5py
import numpy as np
import pytest
import scipy.io
import scipy.ndimage
from matplotlib.colors import same_color
from matplotlib.figure import Figure

from calliopy.io import MatStruct, delete_mat_variables, load_mat, save_mat
from calliopy.projmeta._struct import grid_struct_array, struct_array
from calliopy.viewers import (
    AlignmentDecision,
    CellBrowserSession,
    CellBrowserView,
    ManualRegistrationSession,
    StackAlignerSession,
    StackAlignerView,
    manual_registration_inputs,
    manual_registration_matlab_code,
    plot_cb_plugin_acx,
    plot_cb_plugin_m1,
    plot_cb_plugin_trav,
    plot_cb_plugin_vml,
    plot_cb_stim_snippets,
    save_manual_registration,
    set_check_raw_passed,
    tpi_alignment_figures,
    wid_alignment_figures,
)
from calliopy.viewers._cb_plugins import _vml_means, _vml_pages, _xcorr_unbiased
from calliopy.viewers._common import num2str, read_mat_variable
from calliopy.widefield import fit_similarity, similarity_matrix, warp_image

N = 900
H, W = 40, 30


# ---------------------------------------------------------------------------------------
# synthetic proj_meta
# ---------------------------------------------------------------------------------------
def _act(n_cells: int, rng: np.random.Generator, n: int = N) -> np.ndarray:
    a = 1 + 0.02 * rng.standard_normal((n_cells, n))
    for c in range(n_cells):  # transients at fixed places, different per cell
        for start in range(250 + 37 * c, n - 300, 260):
            a[c, start: start + 30] += 1.5 + 0.2 * c
    return a.astype(np.float32)


def _roiinfo(n_cells: int):
    items = []
    for c in range(n_cells):
        r, col = 5 + 6 * c, 4 + 5 * c
        rows, cols = np.meshgrid(np.arange(r, r + 3), np.arange(col, col + 2), indexing="ij")
        items.append({"indices": (cols.ravel() * H + rows.ravel() + 1).astype(np.float64).reshape(-1, 1),
                      "type": float(c % 2 + 1)})
    return struct_array(items, (1, n_cells))


def make_rd(n_cells: int, rng: np.random.Generator) -> MatStruct:
    t = np.arange(N)
    vrx = (t % 300) / 300 * 5.0
    stim = np.zeros(N)
    for k, s in enumerate(range(100, N - 60, 150)):
        stim[s: s + 40] = 1.0 if k % 2 == 0 else 2.04
    rd = MatStruct()
    rd.act = _act(n_cells, rng)
    rd.velM_smoothed = (-0.01 - 0.005 * np.sin(t / 40))[None, :]
    rd.velR_smoothed = (0.02 * np.cos(t / 30))[None, :]
    rd.velP_smoothed = (0.01 + 0.005 * np.sin(t / 50))[None, :]
    rd.pupil_pos = np.vstack([40 + np.sin(t / 20), 30 + np.cos(t / 20)])
    rd.stim_id = stim[None, :]
    rd.VRx = vrx[None, :]
    rd.VRrew = (t % 300 == 0).astype(float)[None, :]
    rd.GratFlash = ((t // 25) % 2).astype(float)[None, :]
    rd.airPuff = (t % 97 == 0).astype(float)[None, :]
    rd.el_stim = np.zeros((1, N))
    rd.aud_trig = (t % 50 == 0).astype(float)[None, :]
    rd.template = np.arange(H * W, dtype=np.float64).reshape(H, W, order="F")
    rd.ROIinfo = _roiinfo(n_cells)
    rd.timepoint = 1.0
    rd.piezo_layer = 1.0
    rd.fnames = "x.bin"
    rd.nbr_frames = np.array([[N / 2, N / 2]])
    return rd


def make_proj_meta(rng: np.random.Generator | None = None) -> np.ndarray:
    """Two sites: site 0 with 2 layers x 2 timepoints (3, 2 / 3, 2 cells), site 1 with 1 x 1 (4 cells)."""
    rng = rng if rng is not None else np.random.default_rng(1)
    rd0 = grid_struct_array({(0, 0): make_rd(3, rng), (1, 0): make_rd(2, rng),
                             (0, 1): make_rd(3, rng), (1, 1): make_rd(2, rng)}, 2, 2)
    rd1 = grid_struct_array({(0, 0): make_rd(4, rng)}, 1, 1)
    sites = [
        {"projID": "PRJ", "Acode": "A1", "animal": "M1", "ExpGroup": np.array([[101.0, 102.0]]),
         "nbr_piezo_layers": 2.0, "rd": rd0},
        {"projID": "PRJ", "Acode": "A1", "animal": "M2", "ExpGroup": np.array([[201.0]]),
         "nbr_piezo_layers": 1.0, "rd": rd1},
    ]
    return struct_array(sites, (1, 2))


@pytest.fixture
def pm():
    return make_proj_meta()


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())

    @property
    def text(self) -> str:
        return "\n".join(self.messages)


@pytest.fixture
def viewer_log():
    """Messages of the ``calliopy.viewers`` logger (works without pytest's logging plugin)."""
    lg = logging.getLogger("calliopy.viewers")
    h = _Records()
    old = lg.level
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)
    yield h
    lg.removeHandler(h)
    lg.setLevel(old)


# ---------------------------------------------------------------------------------------
# cell browser session
# ---------------------------------------------------------------------------------------
def test_session_defaults(pm):
    s = CellBrowserSession(pm)
    assert s.title == "PRJ_A1"
    assert "template" not in s.rdfields and "nbr_frames" not in s.rdfields
    assert s.rdfields[0] == "act"
    assert s.channels == [[0], [1], [2], [3], [4]]
    assert (s.site, s.layer, s.tp, s.cell) == (0, 0, 0, 0)
    assert s.max_nbr_cells == 4
    assert s.xlim() == (0.0, float(N))
    assert s.colors.shape == (43, 3)  # hsv(256)(1:6:end)


def test_navigation_cells_layers_sites(pm):
    s = CellBrowserSession(pm)
    for _ in range(2):
        assert s.key_press("right") == "redraw"
    assert (s.layer, s.cell) == (0, 2)
    s.key_press("right")  # past the last cell of layer 0 -> layer 1, cell 0 (sequential, as MATLAB)
    assert (s.layer, s.cell) == (1, 0)
    s.key_press("right")
    s.key_press("right")  # past the last cell of the last layer: stays
    assert (s.layer, s.cell) == (1, 1)
    s.key_press("left")
    s.key_press("left")  # before cell 0 of layer 1 -> last cell of layer 0
    assert (s.layer, s.cell) == (0, 2)
    s.key_press("leftarrow", ["shift"])
    assert (s.layer, s.cell) == (0, 2)  # first layer already: nothing changes
    s.key_press("rightarrow", ["shift"])  # layer 1 has 2 cells: closest valid cell is 1 (MATLAB: 1-based 1)
    assert (s.layer, s.cell) == (1, 1)
    s.key_press("up")
    assert (s.tp, s.cell) == (1, 1)
    s.key_press("up")
    assert s.tp == 1
    s.key_press("right", ["alt"])  # next site, first layer and timepoint, same cell index
    assert (s.site, s.layer, s.tp, s.cell) == (1, 0, 0, 1)
    s.key_press("right", ["shift"])  # site 1 has one layer (MATLAB would go to layer 2 and fail)
    assert s.layer == 0
    s.cell = 3
    s.key_press("left", ["alt"])  # site 0 layer 0 has 3 cells: clipped to 2
    assert (s.site, s.layer, s.tp, s.cell) == (0, 0, 0, 2)
    s.key_press("down")
    assert s.tp == 0


def test_navigation_two_modifiers_move_the_cell(pm):
    s = CellBrowserSession(pm)
    s.key_press("right", ["alt", "shift"])  # MATLAB strcmp on a 2-element Modifier is false
    assert (s.site, s.layer, s.cell) == (0, 0, 1)


def _good_session(pm):
    s = CellBrowserSession(pm)
    # keys (site, tp, layer), as cell_browser_good_cells returns them
    s.good = {(0, 0, 0): np.array([False, True, True]), (0, 0, 1): np.array([False, False]),
              (1, 0, 0): np.array([True, False, False, True])}
    return s


def test_navigation_good_cells_previous_and_next(pm):
    s = _good_session(pm)
    s.key_press("right")
    assert s.cell == 1
    s.key_press("right")
    assert s.cell == 2
    s.key_press("left")  # previous good cell (MATLAB r2003: the first good cell before)
    assert s.cell == 1
    s.key_press("left")  # no good cell before, no layer or site before: nothing changes
    assert (s.site, s.layer, s.cell) == (0, 0, 1)
    s.cell = 2
    s.key_press("right")  # layer 1 has no good cell: skipped; site 1 layer 0, first good cell
    assert (s.site, s.layer, s.tp, s.cell) == (1, 0, 0, 0)
    s.key_press("right")
    assert s.cell == 3
    s.key_press("right")  # nothing after: unchanged
    assert (s.site, s.cell) == (1, 3)
    s.key_press("left")
    s.key_press("left")  # back across site 0 layer 1 (no good cell) to its layer 0, last good cell
    assert (s.site, s.layer, s.cell) == (0, 0, 2)


def test_navigation_good_cells_closest_valid_cell(pm):
    s = _good_session(pm)
    s.cell = 2
    s.key_press("right", ["alt"])  # site 1 good cells 0 and 3: 3 is closest to 2
    assert (s.site, s.cell) == (1, 3)
    s.key_press("left", ["alt"])
    assert (s.site, s.cell) == (0, 2)
    s.key_press("right", ["shift"])  # layer 1 has no good cell: index clipped to its 2 cells
    assert (s.layer, s.cell) == (1, 1)
    s.good[(0, 0, 1)] = np.array([True, False])
    s.good[(0, 0, 0)] = np.array([True, False, True])
    s.key_press("left", ["shift"])
    assert (s.layer, s.cell) == (0, 0)  # tie between good cells 0 and 2 around 1 -> lower one
    assert s.closest_cell(0, 0, 0, 1) == 0
    assert s.closest_cell(1, 0, 0, 2) == 3
    s.good = None
    assert s.closest_cell(1, 0, 0, 9) == 3 and s.closest_cell(0, 1, 0, -2) == 0


def test_good_cells_flag_uses_core(pm):
    s = CellBrowserSession(pm, good_cells=True)
    assert s.good is not None and (0, 0, 0) in s.good


def test_zoom_and_keys(pm, viewer_log):
    s = CellBrowserSession(pm)
    assert s.key_press(" ") == "pick_zoom"
    assert not s.add_zoom_click(400.0)
    assert s.add_zoom_click(100.0)
    assert s.xlim() == (100.0, 400.0)
    assert s.zoom_xlim == (400.0, 100.0)  # stored as clicked, sorted for display (MATLAB)
    assert s.key_press("space") == "redraw"
    assert s.xlim() == (0.0, float(N))
    s.key_press("space")
    s.cancel_zoom()
    assert not s.zoom
    for k, name in (("s", "stim"), ("v", "vml"), ("m", "m1"), ("a", "acx"), ("t", "trav")):
        assert s.key_press(k) == f"plugin:{name}"
    assert s.key_press("g") is None
    assert "cb_plugin_screenshot" in viewer_log.text
    assert s.key_press("x") is None


def test_traces(pm):
    s = CellBrowserSession(pm)
    i_vrx = s.rdfields.index("VRx")
    i_pp = s.rdfields.index("pupil_pos")
    s.set_channels(0, [0])
    s.set_channels(1, [i_vrx, i_pp])
    s.set_channels(2, [])
    s.set_channels(3, [])
    s.set_channels(4, [])
    tr = s.traces()
    assert len(tr) == 4
    x, y, c = tr[0]
    assert c == "w" and x[0] == 1 and y.min() == 0 and y.max() == 1
    x, y, c = tr[1]
    assert y.min() == 1 and np.isclose(y.max(), 2)
    assert np.allclose(c, s.colors[2])  # row 1, first entry: MATLAB colors(min(2*2-1+1-1,10)) = colors(3)
    assert [t[2] for t in tr[2:]] == ["r", "b"]  # two-row field
    with pytest.raises(IndexError):
        s.set_channels(0, [99])


def test_thumbnail_cdisp1_asymmetric(pm):
    s = CellBrowserSession(pm)
    s.ta_win = 6
    im = s.thumbnail()
    # cell 0: rows 6..8, cols 5..6 (1-based); centre round(mean) = (7, 6) (5.5 -> 6)
    assert im.shape == (13, 13)
    tmpl = s.rd_entry()["template"]
    # window rows 1..13, cols 0..12 -> 1-based start rows 1, cols 1 with one zero column
    assert np.all(im[:, 0] == 0)
    np.testing.assert_array_equal(im[:, 1:], tmpl[0:13, 0:12])


def test_thumbnail_cdisp2_crop(pm):
    s = CellBrowserSession(pm, cdisp=2)
    s.cell = 1
    im = s.thumbnail()
    assert im.shape == (3, 2, 3)
    assert np.allclose(im[..., 0], 1) and np.allclose(im[..., 1:], 0)
    assert CellBrowserSession(pm, cdisp=0).thumbnail() is None


def test_labels(pm):
    s = CellBrowserSession(pm, comments={101: "a long comment text", 102: "short"})
    lab = s.labels()
    assert lab["cell"] == "1 / 3 - layer 1 - 1" and lab["tp"] == "1 / 2" and lab["animal"] == "M1"
    assert lab["comment"] == "a long com..."
    s.key_press("up")
    assert s.labels()["comment"] == "short..."


def test_view_draws(pm):
    s = CellBrowserSession(pm)
    fig = Figure()
    v = CellBrowserView(fig, s)
    assert len(v.ax.lines) == len(s.traces())
    s.key_press("right")
    v.refresh()
    assert v.ax.get_ylim() == (0, 5)


MATLAB_PARAMS_ORDER = ["nzlay", "win_pos", "cell_ind", "site_id", "tp", "zl", "zon", "rdfields",
                       "cell_annotations", "site_annotations", "chansToPlot", "ta_win", "colors"]


def _load_v5_params(f):
    return scipy.io.loadmat(f, struct_as_record=False, squeeze_me=False)["params"][0, 0]


def test_annotations_file_has_matlab_layout(pm, tmp_path):
    f = tmp_path / "PRJ_A1-cba.mat"
    s = CellBrowserSession(pm, annotations_file=f)
    s.set_cell_annotation("")  # visiting / unchanged text stores nothing (MATLAB edit callback)
    s.set_site_annotation("")
    assert s.cell_annotations == {} and s.site_annotations == [None, None]
    s.save_annotations()
    assert not h5py.is_hdf5(f)  # MATLAB's default save format (v7)
    assert scipy.io.whosmat(f) == [("params", (1, 1), "struct")]
    p = _load_v5_params(f)
    assert list(p._fieldnames) == MATLAB_PARAMS_ORDER
    for k in ("nzlay", "cell_ind", "site_id", "tp", "zl", "ta_win"):
        assert getattr(p, k).dtype == np.float64 and getattr(p, k).shape == (1, 1)
    assert p.nzlay[0, 0] == 4 and p.cell_ind[0, 0] == 1 and p.ta_win[0, 0] == 50
    assert p.zon.dtype == np.float64 and p.zon[0, 0] == 0  # params.zon = 0 until toggled
    assert p.win_pos.shape == (1, 4) and p.colors.shape == (43, 3)
    assert p.rdfields.shape == (len(s.rdfields), 1) and str(p.rdfields[0, 0][0]) == "act"
    assert p.cell_annotations.shape == (2, 4, 4)
    assert all(c.dtype == np.float64 and c.shape == (0, 0) for c in p.cell_annotations.ravel())
    assert p.site_annotations.shape == (2, 1) and p.site_annotations[0, 0].shape == (0, 0)
    assert p.chansToPlot.shape == (5, 1)
    np.testing.assert_array_equal(p.chansToPlot[0, 0], [[1.0]])


def test_annotations_roundtrip(pm, tmp_path):
    f = tmp_path / "PRJ_A1-cba"
    s = CellBrowserSession(pm, annotations_file=f)
    s.set_site_annotation("nice site")
    s.key_press("right")
    s.set_cell_annotation("bright cell")
    s.set_channels(2, [3, 5])
    s.set_channels(3, [])
    s.set_zoom(10, 300)
    s.key_press("up")
    saved = s.save_annotations()
    assert saved.name == "PRJ_A1-cba.mat"
    p = _load_v5_params(saved)
    assert list(p._fieldnames) == [*MATLAB_PARAMS_ORDER, "zxlim"]
    assert p.cell_ind[0, 0] == 2 and p.tp[0, 0] == 2 and p.site_id[0, 0] == 1 and p.zl[0, 0] == 1
    assert p.zon.dtype == np.uint8 and p.zon[0, 0] == 1  # logical after a toggle (scipy reads uint8)
    np.testing.assert_array_equal(p.chansToPlot[2, 0], [[4.0, 6.0]])
    assert p.chansToPlot[3, 0].shape == (0, 0)
    assert str(p.cell_annotations[0, 0, 1][0]) == "bright cell"
    assert str(p.site_annotations[0, 0][0]) == "nice site" and p.site_annotations[1, 0].shape == (0, 0)
    np.testing.assert_array_equal(p.zxlim, [[10.0, 300.0]])
    s2 = CellBrowserSession(pm, annotations_file=saved)
    assert (s2.site, s2.layer, s2.tp, s2.cell) == (0, 0, 1, 1)
    assert s2.site_annotations == ["nice site", None]
    assert s2.cell_annotations == {(0, 0, 1): "bright cell"}
    assert s2.channels[2] == [3, 5] and s2.channels[3] == []
    assert s2.zoom and s2.zoom_xlim == (10.0, 300.0)
    s2.save_annotations()
    assert _load_v5_params(saved).zon.dtype == np.uint8  # still logical


def _matlab_like_params(s):
    ca = np.empty((2, 4, 4), dtype=object)
    for idx in np.ndindex(ca.shape):
        ca[idx] = np.zeros((0, 0))
    ca[1, 0, 3] = "axon?"
    ca[0, 0, 0] = ""  # an annotation that was cleared: '' (0x0 char), not []
    sa = np.empty((2, 1), dtype=object)
    sa[0, 0] = "site one"
    sa[1, 0] = np.zeros((0, 0))
    ch = np.empty((5, 1), dtype=object)
    for i in range(5):
        ch[i, 0] = np.array([[i + 1.0]])
    rdf = np.empty((len(s.rdfields), 1), dtype=object)
    for i, n in enumerate(s.rdfields):
        rdf[i, 0] = n
    # an older field order and a field calliopy does not use
    return {"nzlay": 4.0, "win_pos": np.array([[10.0, 20.0, 800.0, 600.0]]), "cell_ind": 4.0, "site_id": 2.0,
            "tp": 1.0, "zl": 1.0, "zon": 0.0, "rdfields": rdf, "cell_annotations": ca, "site_annotations": sa,
            "chansToPlot": ch, "ta_win": 30.0, "legacy_note": "kept", "colors": np.eye(3)}


def test_annotations_from_matlab_v7_file(pm, tmp_path):
    s = CellBrowserSession(pm)
    f = tmp_path / "old-cba.mat"
    scipy.io.savemat(f, {"params": _matlab_like_params(s)}, do_compression=True)
    s2 = CellBrowserSession(pm, annotations_file=f)
    assert (s2.site, s2.cell) == (1, 3) and s2.ta_win == 30
    assert s2.cell_annotations == {(1, 0, 3): "axon?", (0, 0, 0): ""}
    assert s2.site_annotations == ["site one", None]
    assert s2.win_pos == (10.0, 20.0, 800.0, 600.0)
    np.testing.assert_array_equal(s2.colors, np.eye(3))
    assert read_mat_variable(f, "params")["cell_ind"] == 4.0
    s2.save_annotations()
    p = _load_v5_params(f)
    assert list(p._fieldnames) == ["nzlay", "win_pos", "cell_ind", "site_id", "tp", "zl", "zon", "rdfields",
                                   "cell_annotations", "site_annotations", "chansToPlot", "ta_win",
                                   "legacy_note", "colors"]
    assert str(p.legacy_note[0]) == "kept"
    assert p.cell_annotations[0, 0, 0].size == 0 and p.cell_annotations[0, 0, 0].dtype.kind == "U"
    assert p.zon.dtype == np.float64  # never toggled: stays double


def test_annotations_v73_file_stays_v73(pm, tmp_path):
    s = CellBrowserSession(pm)
    f = tmp_path / "new-cba.mat"
    params = _matlab_like_params(s)
    save_mat(f, {"params": MatStruct(params)})
    s2 = CellBrowserSession(pm, annotations_file=f)
    assert s2.cell_annotations[(1, 0, 3)] == "axon?"
    s2.set_cell_annotation("edited")
    s2.save_annotations()
    assert h5py.is_hdf5(f)
    p = load_mat(f, "params")["params"]
    assert list(p.keys())[12] == "legacy_note"
    assert p["cell_annotations"][1, 0, 3] == "edited"


def test_plugin_figures_via_session(pm):
    s = CellBrowserSession(pm)
    labels = [f.get_label() for f in s.plugin_figures("m1")]
    assert labels[0].endswith("(11)") and labels[-1].endswith("(15)")
    assert any(lb.endswith("(14)") for lb in labels)  # row 4 has a field
    s.set_channels(4, [])
    assert not any(f.get_label().endswith("(14)") for f in s.plugin_figures("m1"))
    assert len(s.plugin_figures("trav")) == 3


# ---------------------------------------------------------------------------------------
# plugins
# ---------------------------------------------------------------------------------------
@pytest.fixture
def rd():
    return make_rd(3, np.random.default_rng(5))


def _mx_onsets(act, thr, win):
    """Independent MATLAB-style onset list (1-based), straight from cb_plugin_*.m."""
    tact = act > thr
    ons = [i + 1 for i in range(tact.size - 1) if not tact[i] and tact[i + 1]]
    ons = [o for o in ons if win + 1 <= o <= tact.size - win - 1]
    return [o for o in ons if not tact[o - win // 2 - 1: o].any()]


def test_stim_snippets(rd):
    fig = plot_cb_stim_snippets(rd, 0)
    assert len(fig.axes) == 2  # conditions 1.0 and 2.0 (2.04 rounds to 2.0)
    sid = np.round(10 * rd.stim_id.ravel()) / 10
    for ax, c in zip(fig.axes, (1.0, 2.0), strict=True):
        n_on = int(np.sum(np.diff((sid == c).astype(int)) == 1))
        assert len(ax.lines) == n_on
        k = int(np.flatnonzero(np.diff((sid == c).astype(int)) == 1)[0])  # MATLAB i-1
        np.testing.assert_allclose(ax.lines[0].get_ydata(), rd.act[0, k - 20: k + 31])
        assert ax.get_ylim()[0] == 0.9


def test_acx(rd):
    act = rd.act[1].astype(np.float64)
    ons = _mx_onsets(act, 1 + 2.5 * np.std(act, ddof=1), 100)
    fig = plot_cb_plugin_acx(rd, 1)
    assert len(fig.axes) == 5 and len(ons) > 0
    assert len(fig.axes[0].lines) == len(ons)
    o = ons[0]
    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), act[o - 101: o + 100])
    np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(), rd.velM_smoothed[0, o - 101: o + 100])
    rd2 = MatStruct(rd)
    del rd2["el_stim"]
    assert "missing" in plot_cb_plugin_acx(rd2, 1).axes[3].get_title()


def test_m1_snippet_means(rd):
    from calliopy.analysis import smooth2

    fields = [["act"], ["VRx"], [], ["pupil_pos"], ["velR_smoothed"]]
    figs = plot_cb_plugin_m1(rd, 0, fields)
    by = {f.get_label()[-3:-1]: f for f in figs}
    assert set(by) == {"11", "12", "13", "14", "15"}
    a = rd.act[0].astype(np.float64)
    thr = 1 + np.std(a, ddof=1) * 2.5
    ons = _mx_onsets(smooth2(a, 20), thr, 150)
    assert ons
    f11_axes = by["11"].axes  # created for rows 0, 1, 3, 4
    act_ax = f11_axes[0]
    norm = (a - a.min()) / (a.max() - a.min())
    expected = np.mean([norm[o - 151: o + 150] for o in ons], axis=0)
    red = next(ln for ln in act_ax.lines if same_color(ln.get_color(), "r"))
    np.testing.assert_allclose(red.get_ydata(), expected, rtol=1e-6)
    pupil_ax = f11_axes[2]  # row 3 (MATLAB ind 4): pupil_pos drawn as snippets, not the -36 mean
    assert pupil_ax.get_title() == "pupil_pos"
    assert by["14"].axes[3].get_title() == f"cell mean, number of onsets: {len(ons)}"


def test_m1_pupil_pos_outside_row_3(rd):
    from calliopy.analysis import smooth2

    figs = plot_cb_plugin_m1(rd, 0, [["act"], ["pupil_pos"], [], [], []])
    ax = figs[0].axes[1]  # row 1: MATLAB plots mean(ipos1_mat, 3) - 36, no title
    assert len(ax.lines) == 1 and ax.get_title() == ""
    a = rd.act[0].astype(np.float64)
    ons = _mx_onsets(smooth2(a, 20), 1 + np.std(a, ddof=1) * 2.5, 150)
    expected = np.mean([rd.pupil_pos[0, o - 151: o + 150] for o in ons], axis=0) - 36
    np.testing.assert_allclose(ax.lines[0].get_ydata(), expected)


def test_m1_no_onsets(viewer_log):
    rd = make_rd(2, np.random.default_rng(0))
    rd.act = np.ones((2, N), dtype=np.float32)
    rd.act[:, ::2] += 0.01
    figs = plot_cb_plugin_m1(rd, 0, [["act"], [], [], [], []])
    assert len(figs) == 4 and "No act onsets" in viewer_log.text


def test_vml_pages_nan_padding():
    m = [np.ones(3), np.ones(3)]
    p = [np.full(3, 2.0)] * 3
    pages = _vml_pages(m, p, 3)
    assert pages.shape == (3, 3, 2) and np.all(np.isnan(pages[:, 2, 0]))
    mean_m, mean_p = _vml_means(pages)
    np.testing.assert_allclose(mean_m, 1.0)  # MATLAB r2003: 2/3 (zero-padded third column)
    np.testing.assert_allclose(mean_p, 2.0)
    mean_m, mean_p = _vml_means(_vml_pages([], [np.ones(3)], 3))
    assert np.all(np.isnan(mean_m)) and np.allclose(mean_p, 1.0)
    eq = _vml_pages([np.arange(3.0)], [np.arange(3.0) + 1], 3)  # equal counts: plain means
    np.testing.assert_allclose(_vml_means(eq)[0], eq[:, :, 0].mean(axis=1))


def test_vml(rd):
    figs = plot_cb_plugin_vml(rd, 2, [["act"], ["VRx"], [], [], []])
    assert [f.get_label() for f in figs] == ["cb_plugin_VML (11)", "cb_plugin_VML (12)"]
    a = rd.act[2].astype(np.float64)
    ons = _mx_onsets(a, 1.5, 200)
    assert figs[0].axes[-1].get_title() == f"Number of onsets: {len(ons)}"
    assert len(figs[1].axes[0].lines) == 1 + 2 * len(ons)


def test_xcorr_unbiased_matches_direct():
    x = np.random.default_rng(3).standard_normal(37)
    n = x.size
    direct = np.array([np.sum(x[max(0, -m): n - max(0, m)] * x[max(0, m): n - max(0, -m)]) / (n - abs(m))
                       for m in range(-(n - 1), n)])
    np.testing.assert_allclose(_xcorr_unbiased(x), direct, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(_xcorr_unbiased(x)[n - 1], np.mean(x * x))


def test_trav(rd):
    figs = plot_cb_plugin_trav(rd, 0)
    assert len(figs) == 3
    assert figs[2].axes[0].get_ylim() == (1.0, 1.2)
    ax = figs[1].axes[1]
    assert ax.get_xlim() == (N - 2000, N + 2000)


# ---------------------------------------------------------------------------------------
# StackAligner
# ---------------------------------------------------------------------------------------
def _stacks():
    big = np.arange(20 * 30 * 6, dtype=np.float64).reshape(20, 30, 6)
    ref = big[3:8, 10:14, 2:5].copy()
    return ref, big


def test_stack_aligner_crop_and_keys():
    ref, big = _stacks()
    s = StackAlignerSession(ref, big, (10, 3, 2))
    np.testing.assert_array_equal(s.crop(), ref[:, :, 0])
    s.key_press("right")
    np.testing.assert_array_equal(s.crop(), ref[:, :, 1])
    s.key_press("right")
    s.key_press("right")
    assert s.slice == 2
    s.key_press("left", ["shift"])
    assert s.offset[0] == 9
    s.key_press("right", ["ctrl", "shift"])
    assert s.offset[0] == 19
    s.key_press("down", ["shift"])
    assert s.offset[1] == 4
    s.key_press("up")
    assert s.offset[2] == 3
    s.key_press("down", ["control"])  # ctrl alone: z step stays 1
    assert s.offset[2] == 2
    s.offset[:] = (27, 3, 2)  # crop would leave the stack
    assert np.all(s.crop() == 0)
    assert s.key_press("x") is None
    assert s.key_press("enter") == "done" and s.done
    assert s.message() == "Layer: 2 | Offset -> X: 27, Y: 3, Z: 2"


def test_stack_aligner_rounding_and_grid():
    ref, big = _stacks()
    s = StackAlignerSession(ref, big, (-0.5, 1.5, 0.4))
    np.testing.assert_array_equal(s.rounded_offset(), [0, 2, 0])  # MATLAB round of 0.5, 2.5, 1.4
    pts = s.grid_points()
    assert pts.shape == (16, 2)
    np.testing.assert_allclose(pts[0], [np.linspace(1, 4, 6)[1] - 1, np.linspace(1, 5, 6)[1] - 1])
    v = StackAlignerView(Figure(), s)
    assert v.title.get_text().startswith("Layer: 0")
    with pytest.raises(ValueError):
        StackAlignerSession(ref, big, (1, 2))


# ---------------------------------------------------------------------------------------
# register_widefield_exp_manual
# ---------------------------------------------------------------------------------------
def _frame(shape=(64, 72), seed=0):
    rng = np.random.default_rng(seed)
    return 100 + 50 * scipy.ndimage.gaussian_filter(rng.standard_normal(shape), 3)


def _rois_cell():
    rois = np.empty((1, 1), dtype=object)
    rois[0, 0] = struct_array([{"indices": np.array([[1.0], [2.0]]), "name": "a"}], (1, 1))
    return rois


def make_wf_adata(path, first_frame, **extra):
    v = {"first_frame": np.asarray(first_frame), "registration_parameters":
         MatStruct(similarity_threshold=0.5, status="FAILED", registration_type="automatic")}
    v.update(extra)
    save_mat(path, v)
    return path


def test_manual_inputs_variants(tmp_path):
    ff = _frame()
    orig = make_wf_adata(tmp_path / "Adata-S1-T100.mat", ff)
    exp = make_wf_adata(tmp_path / "Adata-S1-T105.mat", ff + 1)
    inp = manual_registration_inputs(exp, orig)
    assert inp.rp_string == "" and not inp.is_2ch_origin
    assert (inp.fixed_exp_id, inp.moving_exp_id) == (100, 105)
    np.testing.assert_array_equal(inp.moving, ff + 1)
    with pytest.raises(ValueError, match="origin_adata_path"):
        manual_registration_inputs(exp)
    two = tmp_path / "Adata-S1-T200.mat"
    asym = np.arange(12.0).reshape(3, 4)
    save_mat(two, {"first_frame_ch525": asym, "first_frame_ch610": asym * 2, "tform_ch_map": np.eye(3)})
    inp = manual_registration_inputs(two)
    assert inp.is_2ch_origin and inp.rp_string == "_ch_map"
    np.testing.assert_array_equal(inp.moving, np.fliplr(asym * 2))
    o2 = tmp_path / "Adata-S1-T300.mat"
    save_mat(o2, {"first_frame_ch525": asym, "first_frame_ch610": asym + 7})
    e2 = tmp_path / "Adata-S1-T301.mat"
    save_mat(e2, {"first_frame_ch525": asym, "first_frame_ch610": asym + 9})
    with pytest.raises(ValueError, match="channel"):
        manual_registration_inputs(e2, o2)
    inp = manual_registration_inputs(e2, o2, channel="ch610")
    assert inp.rp_string == "_ch610"
    np.testing.assert_array_equal(inp.fixed, asym + 7)


def test_manual_session_tform_and_save(tmp_path):
    fixed = _frame()
    true = similarity_matrix(1.0, 3.0, (2.0, -1.5))  # moving -> fixed
    moving = warp_image(fixed, np.linalg.inv(true), fixed.shape)
    orig = make_wf_adata(tmp_path / "Adata-S1-T100.mat", fixed)
    exp = make_wf_adata(tmp_path / "Adata-S1-T105.mat", moving, ROIs=_rois_cell(), other=np.arange(4.0))
    inp = manual_registration_inputs(exp, orig)
    s = ManualRegistrationSession(inp.fixed, inp.moving, adata_path=exp, fixed_exp_id=100, moving_exp_id=105)
    assert s.titles() == ("ExpID 100", "ExpID 105")
    with pytest.raises(ValueError, match="four"):
        s.compute_tform()
    mv = np.array([[20.0, 15.0], [50.0, 40.0]])
    fx = mv @ true[:2, :2].T + true[:2, 2]
    s.set_point("fixed", 0, *fx[0])
    s.set_point("fixed", 1, *fx[1])
    s.set_point("moving", 0, *mv[0])
    s.set_point("moving", 1, *mv[1])
    tform, sim = s.compute_tform()
    np.testing.assert_allclose(tform, true, atol=1e-12)
    np.testing.assert_allclose(tform, fit_similarity(mv, fx))
    assert sim > 0.95
    assert s.overlay_stack().shape == fixed.shape + (2,)
    code = s.matlab_commands()
    assert "adata.tform = simtform2d([" in code and "adata.registration_parameters.status = 'SUCCESS';" in code
    assert "adata.save_changes;" in code and "rmfield(adata_s, 'ROIs')" in code
    written = s.save(export_refpoints=True)
    assert written == ["tform", "registration_parameters", "ROIs"]
    np.testing.assert_allclose(s.exported_refpoints, np.vstack([fx, mv]))
    d = load_mat(exp)
    assert "ROIs" not in d
    np.testing.assert_allclose(d["tform"], true, atol=1e-12)
    rp = d["registration_parameters"]
    assert rp["status"] == "SUCCESS" and rp["registration_type"] == "manual"
    assert rp["similarity_threshold"] == 0.5 and np.isclose(rp["similarity"], sim)
    assert rp["fraction_boundary"] == 0.1
    np.testing.assert_array_equal(d["first_frame"], moving)
    np.testing.assert_array_equal(d["other"], [[0.0, 1.0, 2.0, 3.0]])


def test_manual_session_refpoints_and_normalise(tmp_path):
    f = _frame()
    s = ManualRegistrationSession(f, f, normalize_template=True, previous_refpoints=np.arange(8.0).reshape(4, 2))
    assert not np.allclose(s.fixed, f)
    assert np.all(np.isnan(s.refpoints))
    assert s.retrieve_previous()
    np.testing.assert_array_equal(s.refpoints, np.arange(8.0).reshape(4, 2))
    with pytest.raises(ValueError, match="no adata file"):
        s.save()
    assert ManualRegistrationSession.point_slot("3") == ("moving", 0)
    with pytest.raises(ValueError):
        s.set_refpoints(np.zeros((3, 2)))


def test_save_manual_registration_new_struct_suffix(tmp_path):
    p = tmp_path / "Adata-S1-T7.mat"
    save_mat(p, {"first_frame_ch525": np.ones((4, 4))})
    assert save_manual_registration(p, np.eye(3), similarity=0.7, frac_boundary=0.2, rp_string="_ch525") == \
        ["tform_ch525", "registration_parameters_ch525"]
    d = load_mat(p)
    assert list(d["registration_parameters_ch525"].keys()) == \
        ["similarity", "fraction_boundary", "status", "registration_type"]


def test_manual_registration_matlab_code_exact_numbers():
    a = similarity_matrix(1.01, 2.5, (3.25, -1.0))
    code = manual_registration_matlab_code("x/Adata-S1-T5.mat", a, similarity=0.875, frac_boundary=0.1,
                                           rp_string="_ch610")
    lines = code.splitlines()
    assert lines[0] == "adata_fname = 'x/Adata-S1-T5.mat';"
    row = lines[2][lines[2].index("[") + 1: lines[2].index("]")]
    np.testing.assert_array_equal(np.array([[float(v) for v in r.split()] for r in row.split(";")]), a)
    assert lines[2].startswith("adata.tform_ch610 = simtform2d([")
    assert "adata.registration_parameters_ch610.similarity = 0.875;" in lines
    assert "adata.registration_parameters_ch610.registration_type = 'manual';" in lines


def test_delete_mat_variables_cleans_refs(tmp_path):
    p = tmp_path / "a.mat"
    keep = np.empty((1, 2), dtype=object)
    keep[0, 0] = "x"
    keep[0, 1] = np.ones(3)
    save_mat(p, {"ROIs": _rois_cell(), "keep": keep, "v": 1.0})
    with h5py.File(p, "r") as f:
        n_before = len(f["#refs#"])
    assert delete_mat_variables(p, ["ROIs", "absent"]) == ["ROIs"]
    with h5py.File(p, "r") as f:
        assert "ROIs" not in f and len(f["#refs#"]) < n_before
    d = load_mat(p)
    assert d["keep"][0, 0] == "x" and d["v"] == 1.0


# ---------------------------------------------------------------------------------------
# alignment checks
# ---------------------------------------------------------------------------------------
def _layout(path, name):
    """(field, MATLAB class, shape) of every field of a struct variable in a v7.3 file."""
    out = []
    with h5py.File(path, "r") as f:
        g = f[name]
        fields = [n.decode() if isinstance(n, bytes) else str(n)
                  for n in (b"".join(x) for x in g.attrs["MATLAB_fields"])]
        for k in fields:
            obj = g[k]
            out.append((k, obj.attrs["MATLAB_class"], obj.shape if isinstance(obj, h5py.Dataset) else None))
    return out


def test_set_check_raw_passed(tmp_path):
    p = tmp_path / "Adata-S1-T5.mat"
    cr0 = MatStruct(passed=0.0, img_similarity=0.8, act_similarity=np.array([[0.5, 0.95]]), note="x",
                    ok=np.array([[True, False]]), n=np.array([[3]], dtype=np.int32))
    save_mat(p, {"check_raw": cr0, "z": 2.0})
    before = _layout(p, "check_raw")
    cr = set_check_raw_passed(p)
    assert cr["passed"] == 1.0
    assert _layout(p, "check_raw") == before  # same fields, order, classes and sizes
    d = load_mat(p)
    assert d["check_raw"]["passed"] == 1.0 and d["check_raw"]["note"] == "x" and d["z"] == 2.0
    assert d["check_raw"]["n"].dtype == np.int32 and d["check_raw"]["ok"].dtype == bool


def test_alignment_decision(tmp_path):
    p = tmp_path / "Adata-S1-T5.mat"
    save_mat(p, {"check_raw": MatStruct(passed=0.0)})
    d = AlignmentDecision(p, [], exp_id=5)
    assert d.question() == "5 - Do you want to set the status to passed?"
    assert d.decide(False) == 0 and load_mat(p)["check_raw"]["passed"] == 0.0
    assert d.decide(True) == 1 and load_mat(p)["check_raw"]["passed"] == 1.0


def _check_raw():
    return MatStruct(passed=0.0, img_similarity=0.91234, act_similarity=np.array([[0.5, 0.95]]),
                     roi_similarity=0.97)


def test_wid_alignment_figures_single_channel(tmp_path):
    big = _frame((130, 140), 2)
    site = make_wf_adata(tmp_path / "Adata-S1-T10.mat", big)
    exp = make_wf_adata(tmp_path / "Adata-S1-T11.mat", big, check_raw=_check_raw(), tform=np.eye(3))
    dec = wid_alignment_figures(exp, site, rng=np.random.default_rng(0))
    assert [t for t, _ in dec.figures] == ["Widefield Alignment - Single Channel", "Image Similarity Snippets"]
    titles = [ax.get_title() for ax in dec.figures[0][1].axes]
    assert titles == ["siteID 10", "expID 11", "Passed: 0 - img similarity 0.91234 - act similarity 0.5  0.95"]
    assert dec.exp_id == 11 and any(line.startswith("passed: 0") for line in dec.text)
    with pytest.raises(ValueError, match="not checked"):
        wid_alignment_figures(site)


def test_wid_alignment_figures_two_channels(tmp_path):
    a = _frame((20, 24), 3)
    site = tmp_path / "Adata-S1-T10.mat"
    save_mat(site, {"first_frame_ch525": a, "first_frame_ch610": a, "check_raw": _check_raw(),
                    "tform_ch_map": np.eye(3)})
    dec = wid_alignment_figures(site)
    assert [t for t, _ in dec.figures] == ["Widefield Alignment - Channel 525 (Green)"]
    exp = tmp_path / "Adata-S1-T11.mat"
    save_mat(exp, {"first_frame_ch525": a, "first_frame_ch610": a, "check_raw": _check_raw()})
    with pytest.raises(ValueError, match="tforms="):
        wid_alignment_figures(exp, site)
    dec = wid_alignment_figures(exp, site, tforms={"tform_ch525": np.eye(3), "tform_ch610": np.eye(3)})
    assert len(dec.figures) == 2


def _tpi_file(tmp_path, roi_similarity):
    rng = np.random.default_rng(4)
    data = [rng.random((6, 5, 40)).astype(np.float32)]
    idx = [np.array([1, 2, 7]), np.array([10, 11])]
    acts = [data[0].reshape(30, 40, order="F")[i - 1].mean(axis=0) for i in idx]
    rois = np.empty((1, 1), dtype=object)
    rois[0, 0] = struct_array([{"indices": i.astype(float).reshape(-1, 1), "activity": a.reshape(-1, 1)}
                               for i, a in zip(idx, acts, strict=True)], (1, 2))
    cr = MatStruct(passed=0.0, act_similarity=np.array([[0.5]]), roi_similarity=roi_similarity)
    p = tmp_path / "Adata-S1-T42.mat"
    save_mat(p, {"check_raw": cr, "ROIs": rois})
    return p, data


def test_tpi_alignment_figures(tmp_path, viewer_log):
    p, data = _tpi_file(tmp_path, 0.99)
    site_fig = Figure()
    dec = tpi_alignment_figures(p, site_figures=[site_fig])
    assert dec.figures == [("site ROIs 1", site_fig)]
    assert "This ExpID failed activity similarity" in dec.text
    assert "This ExpID failed activity similarity" in viewer_log.text
    p, data = _tpi_file(tmp_path, 0.9)
    dec = tpi_alignment_figures(p, data=data)
    assert [t for t, _ in dec.figures] == ["activity similarity"]
    assert dec.figures[0][1].axes[0].get_title().startswith("Correlation coefficient: 1")
    assert tpi_alignment_figures(p).figures == []


def test_num2str():
    assert num2str(3.0) == "3" and num2str(0.98765432) == "0.98765" and num2str(np.pi) == "3.1416"
    assert num2str(np.array([[1.0, 10.0]])) == "1  10" and num2str("x") == "x" and num2str(np.zeros(0)) == ""
