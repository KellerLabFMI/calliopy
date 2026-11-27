"""RoiQuest session on two-photon data: loading, scripted ROI editing, saving and keys."""

from __future__ import annotations

import numpy as np
import pytest
from test_roiquest_data import (
    BLOBS_Z0,
    BLOBS_Z1,
    H,
    W,
    adata_path,
    disk,
    explog,
    find,
    make_2p,
    roi_struct,
)

from calliopy.io import load_mat, save_mat
from calliopy.roiquest import (
    BACKUP_INFIX,
    EEG_TEMPLATE_NAME,
    KEY_BINDINGS,
    Redraw,
    RoiQuestRefused,
    RoiQuestSession,
    View,
    help_text,
    info_text_anchor,
    key_press,
    load_prefs,
    normalize_key,
    save_prefs,
)
from calliopy.rois import Roi, RoiSaveError, adjust_img_contrast, filter_template


def _need_update_mat():
    try:
        from calliopy.io import update_mat  # noqa: F401
    except ImportError:
        pytest.skip("calliopy.io.update_mat is not available yet (written by another track)")


def _flat(a):
    return np.asarray(a, dtype=np.float64).ravel(order="F")


def _layer(var, z):
    c = var.ravel(order="F")[z]
    return [] if c.size == 0 else list(c.ravel(order="F"))


def _zoom_putative(s, center, half=8):
    s.button_down(center[0] - half, center[1] - half)
    return s.button_up(center[0] + half, center[1] + half)


@pytest.fixture
def two_p(tmp_path):
    return make_2p(tmp_path / "Adata-S1-T101.mat")


# ---------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------
def test_load_2p_state(two_p):
    s = RoiQuestSession.from_adata(two_p)
    assert s.curr_exp_id == 101 and s.exp_group == [101]
    assert s.num_layers == 2 and len(s.imgs) == 3
    assert all(im.dtype == np.uint8 and im.shape == (H, W) for row in s.imgs for im in row)
    assert [len(x) for x in s.rois] == [1, 0]
    assert not s.rois[0][0].is_new
    # stored indices stay 1-based
    np.testing.assert_array_equal(_flat(s.rois[0][0].indices),
                                  _flat(find(disk((H, W), BLOBS_Z0[0], 4))))
    # valid area: first layer's max |dx| = 2 rows and max |dy| = 2 columns on each side
    va = s.valid_area[:, :, 0]
    assert va.shape == (H, W) and va[2:H - 2, 2:W - 2].all() and not va[:2].any()
    assert s.filter_win_size == 3 and s.thr == [160.0, 160.0, 160.0]
    assert s.ovrl_opacity == 0.35 and s.roi_color_mode == "flat"
    assert s.view == View((-0.5, W - 0.5), (-0.5, H - 0.5)) and not s.is_zoomed_in
    # bestFit display images of the raw template
    np.testing.assert_array_equal(s.imgs[0][0], adjust_img_contrast(s.template[0]))
    np.testing.assert_array_equal(
        s.imgs[1][1], adjust_img_contrast(filter_template(s.template[1], 3, "legacy")))
    en = s.enabled
    assert en["roiExpand"] and en["cellposeDetect"] and not en["wfAddRois"]
    assert not en["loadRoiOrig"] and not en["setFiltType"]
    assert len(s.roi_trans) == 2 and s.roi_trans_cell().shape == (1, 2)


def test_rng_reset_gives_same_colours(two_p):
    a = RoiQuestSession.from_adata(two_p)
    b = RoiQuestSession.from_adata(two_p)
    assert a.rois[0][0].rand_color_idx == b.rois[0][0].rand_color_idx
    assert 2 <= a.rois[0][0].rand_color_idx <= 11


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        RoiQuestSession.from_adata(tmp_path / "nope.mat")


# ---------------------------------------------------------------------------------------
# zoom, threshold, accept, select, move, expand, tag, delete
# ---------------------------------------------------------------------------------------
def test_dead_zone_does_not_zoom(two_p):
    s = RoiQuestSession.from_adata(two_p)
    s.button_down(10, 10)
    assert s.button_up(13, 40) == Redraw.NONE  # rows within 5 px
    assert not s.is_zoomed_in and s.putative_rois == []


def test_zoom_segments_putative(two_p):
    s = RoiQuestSession.from_adata(two_p)
    rd = _zoom_putative(s, BLOBS_Z0[1])
    assert Redraw.VIEW in rd and Redraw.OVERLAY in rd
    r, c = BLOBS_Z0[1]
    assert s.is_zoomed_in and s.view == View((c - 8.0, c + 8.0), (r - 8.0, r + 8.0))
    assert s.btn_down_coord == (r - 8, c - 8) and s.btn_up_coord == (r + 8, c + 8)
    assert len(s.putative_rois) == 1
    blob = disk((H, W), BLOBS_Z0[1], 4)
    np.testing.assert_array_equal(s.putative_rois[0].to_binary_mask(), blob)
    # rounding of fractional coordinates (half away from zero)
    s.button_down(2.5, 3.49)
    assert s.btn_down_coord == (3, 3)


def test_zoom_excludes_existing_roi(two_p):
    s = RoiQuestSession.from_adata(two_p)
    _zoom_putative(s, BLOBS_Z0[0])
    assert s.putative_rois == []  # the only bright pixels belong to the saved ROI


def test_threshold_steps_and_clamp(two_p):
    s = RoiQuestSession.from_adata(two_p)
    _zoom_putative(s, BLOBS_Z0[1])
    rd = s.threshold_step("up", ())
    assert s.thr[0] == 159.0 and Redraw.HIST_THRESHOLD in rd
    s.threshold_step("up", ("shift",))
    assert s.thr[0] == 154.0
    s.threshold_step("down", ["shift"])
    assert s.thr[0] == 159.0
    s.thr[0] = 254.0
    for _ in range(3):
        s.threshold_step("down", ("shift",))
    assert s.thr[0] == 255.0
    assert s.thr[1:] == [160.0, 160.0]


def test_threshold_median_snap(two_p):
    s = RoiQuestSession.from_adata(two_p)
    s.button_down(45, 5)
    s.button_up(55, 20)  # background only
    s.thr[0] = 255.0
    s.threshold_step("down")
    assert s.putative_rois == []
    s.threshold_step("up")  # still nothing at 254 -> median snap (MATLAB crashed here)
    crop = s.imgs[0][0][45:56, 5:21].astype(float)
    assert s.thr[0] == np.floor(np.median(crop) + 0.5)
    assert len(s.putative_rois) == 1


def test_scroll(two_p):
    s = RoiQuestSession.from_adata(two_p)
    rd = s.scroll(1)  # not zoomed: next layer
    assert s.current_z == 1 and Redraw.DISPLAY in rd
    s.scroll(-3)
    assert s.current_z == 0
    _zoom_putative(s, BLOBS_Z0[1])
    s.scroll(-2)
    assert s.thr[0] == 158.0 and len(s.putative_rois) == 1


def test_scripted_session_save_reload(two_p):
    _need_update_mat()
    s = RoiQuestSession.from_adata(two_p)
    _zoom_putative(s, BLOBS_Z0[1])
    for d in ("up", "up", "down"):
        s.threshold_step(d)
    res = key_press(s, "a")
    assert res.action == "accept_putative" and res.executed
    assert not s.is_zoomed_in and s.putative_rois == []
    new = s.rois[0][1]
    assert new.is_new and new.selected and not s.rois[0][0].selected
    before = _flat(new.indices).copy()
    key_press(s, "right")
    np.testing.assert_array_equal(_flat(new.indices), before + H)
    key_press(s, "uparrow")  # not zoomed: moves the selection up
    np.testing.assert_array_equal(_flat(new.indices), before + H - 1)
    np.testing.assert_array_equal(new.shift, [[-1.0, 1.0]])
    n_before = new.indices.size
    s.expand_selected()
    assert new.indices.size > n_before
    key_press(s, "7")
    assert new.type == 7.0
    # select by right-click on the first ROI, then shift-click toggles the second
    s.button_down(*BLOBS_Z0[0])
    s.button_up(*BLOBS_Z0[0], kind="alt")
    assert [r.selected for r in s.rois[0]] == [True, False]
    r0, c0 = BLOBS_Z0[1]
    s.button_down(r0 - 1, c0 + 1)
    s.button_up(r0 - 1, c0 + 1, kind="extend")
    assert [r.selected for r in s.rois[0]] == [True, True]
    s.select_none()
    assert not any(r.selected for r in s.rois[0])

    out = s.save()
    path = two_p.resolve()
    assert set(out.written[two_p]) == {"ROIs", "ROItrans"}
    backup = out.backups[path]
    assert backup.exists() and backup.name.startswith(two_p.name + BACKUP_INFIX)
    assert load_mat(backup)["ROIs"].shape == (1, 2)
    assert not any(r.is_new for r in s.rois[0])
    # a second save makes no second backup
    assert s.save().backups == {}

    d = load_mat(two_p)
    np.testing.assert_array_equal(d["other"], [[0.0, 1.0, 2.0]])
    saved = _layer(d["ROIs"], 0)
    assert len(saved) == 2 and _layer(d["ROIs"], 1) == []
    np.testing.assert_array_equal(_flat(saved[1]["indices"]), _flat(new.indices))
    assert float(saved[1]["type"]) == 7.0
    np.testing.assert_array_equal(np.asarray(saved[1]["shift"]).ravel(), [-1.0, 1.0])

    s2 = RoiQuestSession.from_adata(two_p)
    assert [len(x) for x in s2.rois] == [2, 0]
    for a, b in zip(s.rois[0], s2.rois[0]):
        np.testing.assert_array_equal(_flat(a.indices), _flat(b.indices))
        assert float(a.type) == float(b.type)


def test_save_refuses_activity_overwrite(tmp_path):
    _need_update_mat()
    roi = roi_struct(disk((H, W), BLOBS_Z0[0], 4), activity=np.ones((5, 1)))
    p = make_2p(tmp_path / "Adata-S1-T7.mat", rois_z0=[roi])
    s = RoiQuestSession.from_adata(p)
    with pytest.raises(RoiSaveError):
        s.save()
    assert s.save(overwrite_activity=True).written


def test_save_with_fake_writer(two_p, monkeypatch):
    """Save dispatch without calliopy.io.update_mat: a writer built on load_mat/save_mat."""
    calls = []

    def fake(path, variables):
        calls.append((path, tuple(variables)))
        d = load_mat(path)
        d.update(variables)
        save_mat(path, d)

    monkeypatch.setattr("calliopy.roiquest._persist._update_mat", fake)
    s = RoiQuestSession.from_adata(two_p)
    s.rois[0][0].type = 4.0
    s.save()
    assert calls == [(two_p, ("ROIs", "ROItrans"))]
    assert float(_layer(load_mat(two_p)["ROIs"], 0)[0]["type"]) == 4.0


def test_tag_untag_select_all(two_p):
    s = RoiQuestSession.from_adata(two_p)
    s.select_all()
    s.tag_selected(3)
    assert s.rois[0][0].type == 3.0
    s.untag_selected()
    assert s.rois[0][0].type == 0.0
    s.current_z = 1
    assert s.tag_selected(2) == Redraw.NONE  # no ROIs in the layer


def test_delete_acute(two_p):
    s = RoiQuestSession.from_adata(two_p)
    assert s.deletion_kind() == "none"
    with pytest.raises(RoiQuestRefused):
        s.delete_selected()
    res = key_press(s, "d")  # refusal is logged, not raised
    assert res.executed and res.redraw == Redraw.NONE
    s.select_all()
    assert s.deletion_kind() == "acute"
    assert key_press(s, "delete").action == "delete"
    assert s.rois[0] == []


def test_shrink(two_p):
    s = RoiQuestSession.from_adata(two_p)
    s.select_all()
    n = s.rois[0][0].indices.size
    s.shrink_selected()
    assert 0 < s.rois[0][0].indices.size < n


def test_eeg_template_selection_guard(two_p):
    s = RoiQuestSession.from_adata(two_p)
    s.rois[0] = [Roi(np.array([[float(k * H + 1)]]), (H, W)) for k in range(W)] + \
        [Roi(np.array([[float(k * H + 2)]]), (H, W)) for k in range(50)]
    s.roi_template_parameters = {"name": EEG_TEMPLATE_NAME.upper()}
    s.button_down(0, 0)
    s.button_up(H - 1, W - 1, kind="alt")
    sel = [r.selected for r in s.rois[0]]
    assert not any(sel[:2]) and all(sel[2:122]) and not any(sel[122:])
    s.roi_template_parameters = None
    s.button_up(H - 1, W - 1, kind="alt")
    assert all(r.selected for r in s.rois[0])


def test_single_right_click_selects_only_hit(two_p):
    s = RoiQuestSession.from_adata(two_p)
    s.select_all()
    s.button_down(50, 70)
    s.button_up(51, 71, kind="alt")  # single click on background
    assert not s.rois[0][0].selected


# ---------------------------------------------------------------------------------------
# images, layers, contrast, filters, colormaps
# ---------------------------------------------------------------------------------------
def test_cycle_and_layers(two_p):
    s = RoiQuestSession.from_adata(two_p)
    s.select_all()
    seen = []
    for _ in range(3):
        s.cycle_image("+")
        seen.append(s.img_idx)
    assert seen == [1, 2, 0]
    s.cycle_image("-")
    assert s.img_idx == 2
    key_press(s, "rightbracket")
    assert s.img_idx == 0
    key_press(s, "j")
    assert s.current_z == 1 and not s.rois[0][0].selected
    key_press(s, "j")
    assert s.current_z == 1
    key_press(s, "i")
    assert s.current_z == 0
    np.testing.assert_array_equal(s.display_image(), s.imgs[0][0])
    assert s.curr_img_median == float(np.median(s.imgs[0][0]))


def test_contrast_modes(two_p):
    s = RoiQuestSession.from_adata(two_p)
    s.set_contrast("minMax")
    np.testing.assert_array_equal(s.imgs[0][1], adjust_img_contrast(s.template[1], "minMax"))
    s.cycle_image("+")
    s.set_custom_contrast(0.5, 0.2, moved="white")
    assert (s.img_lo[1], s.img_hi[1]) == (0.5, pytest.approx(0.51))
    s.set_custom_contrast(0.9, 0.6, moved="black")
    assert (s.img_lo[1], s.img_hi[1]) == (pytest.approx(0.59), 0.6)
    np.testing.assert_array_equal(
        s.imgs[1][0], adjust_img_contrast(filter_template(s.template[0], 3, "legacy"),
                                          "custom", (0.59, 0.6)))
    s.set_contrast("custom")
    assert s.contrast_mode == "custom"
    with pytest.raises(ValueError):
        s.set_contrast("auto")


def test_filter_settings(two_p):
    s = RoiQuestSession.from_adata(two_p)
    old = s.imgs[1][0].copy()
    s.set_filter_size("7.4")
    assert s.filter_win_size == 7 and not np.array_equal(old, s.imgs[1][0])
    for bad in (0, 101, "x"):
        with pytest.raises(ValueError):
            s.set_filter_size(bad)
    s.set_filter_type("updated")
    assert s.filter_type == "updated"
    with pytest.raises(ValueError):
        s.set_filter_type("new")


def test_display_settings(two_p):
    s = RoiQuestSession.from_adata(two_p)
    s.set_colormap("red")
    np.testing.assert_allclose(s.cmap[-1], [1.0, 0.0, 0.0])
    s.invert_colormap()
    np.testing.assert_allclose(s.cmap[0], [1.0, 0.0, 0.0])
    s.set_colormap("nonsense")
    np.testing.assert_allclose(s.cmap[255], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        s.set_opacity(1.5)
    s.set_opacity(0.6)
    with pytest.raises(ValueError):
        s.set_roi_color((2, 0, 0))
    s.set_roi_color((0, 0, 1))
    assert s.roi_bg.shape == (H, W, 3) and tuple(s.roi_bg[0, 0]) == (0.0, 0.0, 1.0)
    s.set_color_mode("tags")
    s.toggle_numbers()
    assert s.roi_number_visible
    s.set_info_position("br")
    assert info_text_anchor(s) == (W - 1.0, H - 3.0, "right", "bottom")
    with pytest.raises(ValueError):
        s.set_info_position("center")


def test_god_mode(two_p):
    s = RoiQuestSession.from_adata(two_p)
    for _ in range(4):
        assert key_press(s, "g").redraw == Redraw.NONE
    assert key_press(s, "g").redraw == Redraw.MENUS
    assert s.enabled["setFiltType"] and s.enabled["setFiltSize"]


def test_prefs_roundtrip(two_p, tmp_path):
    s = RoiQuestSession.from_adata(two_p)
    s.set_colormap("hot")
    s.invert_colormap()
    s.set_opacity(0.5)
    s.thr = [150.0, 140.0, 130.0]
    s.img_lo = [0.1, 0.2, 0.3]
    s.set_roi_color((0.1, 0.2, 0.3))
    s.set_selection_color((0.4, 0.5, 0.6))
    s.set_color_mode("random")
    f = tmp_path / "prefs.json"
    save_prefs(s, f)
    t = RoiQuestSession.from_adata(two_p)
    load_prefs(t, f)
    np.testing.assert_allclose(t.cmap, s.cmap)
    assert t.ovrl_opacity == 0.5 and t.thr == s.thr and t.img_lo == s.img_lo
    assert t.roi_bg_color == (0.1, 0.2, 0.3) and t.roi_sel_bg_color == (0.4, 0.5, 0.6)
    assert t.roi_color_mode == "random"


# ---------------------------------------------------------------------------------------
# texts, keys, Cellpose, ROI loading, revert
# ---------------------------------------------------------------------------------------
def test_texts(two_p):
    s = RoiQuestSession.from_adata(two_p)
    assert s.overlay_text() == "ExpID: 101 - Z:1 - 1 ROI(s)"
    _zoom_putative(s, BLOBS_Z0[1])
    assert s.overlay_text(49.0) == "ExpID: 101 - Z:1 - 49 pixels"
    s.escape()
    s.toggle_original()
    assert s.overlay_text().startswith("ORIGINAL - ")
    info = s.info_text()
    assert "ExpID: 101" in info and " 1              1" in info and "Filter size: 3.00" in info
    assert "RoiQuest - Manual" in help_text() and "<strong>" not in help_text()
    s.toggle_roi_visible()
    assert s.overlay_text() == ""


def test_key_table(two_p):
    s = RoiQuestSession.from_adata(two_p)
    assert normalize_key("ctrl+a") == ("a", frozenset({"control"}))
    assert normalize_key("shift+up") == ("uparrow", frozenset({"shift"}))
    assert normalize_key("]")[0] == "rightbracket" and normalize_key(" ")[0] == "space"
    assert normalize_key("C") == ("c", frozenset({"shift"}))
    for key, mods, action, executed in (
            ("s", ("control",), "save", False), ("r", ("ctrl",), "revert", False),
            ("t", ("control",), "cellpose_detect", False),
            ("v", ("control",), "view_activity", False), ("h", (), "histogram", False),
            ("a", ("control",), "select_all", True), ("n", ("control",), "select_none", True),
            ("p", ("control",), "expand", True), ("l", ("control",), "shrink", True),
            ("space", (), "toggle_roi_visible", True), ("o", (), "toggle_original", True),
            ("escape", (), "escape", True), ("q", (), "escape", True),
            ("x", (), "", True)):
        res = key_press(s, key, mods)
        assert (res.action, res.executed) == (action, executed), key
    assert {b[2] for b in KEY_BINDINGS} >= {"tag", "delete", "auto_roi_matching"}
    res = key_press(s, "c")  # not a chronic series: refusal logged
    assert res.action == "auto_roi_matching" and res.redraw == Redraw.NONE


class _FakeCellpose:
    def __init__(self, labels):
        self.labels = labels
        self.calls = []

    def eval(self, imgs, **kw):
        self.calls.append(kw)
        return [self.labels], None, None


def test_cellpose_detect_with_fake_model(two_p):
    s = RoiQuestSession.from_adata(two_p)
    lab = np.zeros((H, W), dtype=np.int32)
    lab[disk((H, W), BLOBS_Z0[1], 4)] = 1          # valid
    lab[disk((H, W), BLOBS_Z0[0], 3)] = 2          # overlaps the saved ROI
    lab[0:6, 30:36] = 3                            # touches the invalid border
    lab[50:52, 60:62] = 4                          # too small
    model = _FakeCellpose(lab)
    s.cellpose_detect(model)
    assert model.calls == [{"diameter": 25, "channels": [0, 0]}]
    assert len(s.putative_rois) == 1
    np.testing.assert_array_equal(s.putative_rois[0].to_binary_mask(), lab == 1)
    s.accept_putative()
    assert len(s.rois[0]) == 2 and s.rois[0][1].is_new
    with pytest.raises(RoiQuestRefused):
        RoiQuestSession.from_adata(two_p).cellpose_detect()


def test_load_rois_and_revert(two_p, tmp_path):
    other = make_2p(tmp_path / "other" / "Adata-S1-T5.mat",
                    rois_z0=[roi_struct(disk((H, W), BLOBS_Z0[1], 3), 2.0)],
                    rois_z1=[roi_struct(disk((H, W), BLOBS_Z1[0], 3))],
                    extra={"ROItrans": np.empty((1, 2), dtype=object)})
    s = RoiQuestSession.from_adata(two_p)
    assert s.load_rois_from_adata("original") == Redraw.NONE
    s.load_rois_from_adata(other)
    assert [len(x) for x in s.rois] == [1, 1] and s.rois[0][0].type == 2.0
    s.revert_to_saved()
    assert [len(x) for x in s.rois] == [1, 0] and s.rois[0][0].type == 0.0


def test_activity_selection(two_p):
    s = RoiQuestSession.from_adata(two_p)
    with pytest.raises(RoiQuestRefused):
        s.activity_selection()
    s.select_all()
    with pytest.raises(RoiQuestRefused, match=r"#\[1\]"):
        s.activity_selection()
    s.rois[0][0].activity = np.arange(10.0)
    rois, idx = s.activity_selection()
    assert idx == [0] and rois[0] is s.rois[0][0]


# ---------------------------------------------------------------------------------------
# chronic series (ExpLog + adata folder)
# ---------------------------------------------------------------------------------------
@pytest.fixture
def series(tmp_path):
    adir = tmp_path / "adata"
    make_2p(adata_path(adir, 201))
    make_2p(adata_path(adir, 202), rois_z0=[])
    make_2p(adata_path(adir, 203), rois_z0=[])
    make_2p(adata_path(adir, 300))
    el = explog([(201, 5), (202, 5), (203, 5), (300, 9)])
    return adir, el


def test_from_expid_acute_and_chronic(series):
    adir, el = series
    s = RoiQuestSession.from_expid(300, exp_log=el, adata_dir=adir)
    assert s.orig_adata_full_path is None and s.exp_group == [300]
    assert s.prefs_default_name() == "user_proj.json"
    s = RoiQuestSession.from_expid(201, exp_log=el, adata_dir=adir)  # original itself
    assert s.orig_adata_full_path is None and s.exp_group == [201, 202, 203]
    s = RoiQuestSession.from_expid(202, exp_log=el, adata_dir=adir)
    assert s.orig_adata_full_path == adata_path(adir, 201)
    assert s.enabled["loadRoiOrig"] and len(s.orig_imgs) == 3
    assert [len(x) for x in s.rois] == [1, 0]  # original ROIs loaded
    t = RoiQuestSession.from_expid(202, exp_log=el, adata_dir=adir, load_original_rois=False)
    assert [len(x) for x in t.rois] == [0, 0]
    with pytest.raises(FileNotFoundError):
        RoiQuestSession.from_expid(203, exp_log=explog([(203, 5)]), adata_dir=adir / "x")


def test_auto_roi_matching_identical_images(series):
    adir, el = series
    s = RoiQuestSession.from_expid(202, exp_log=el, adata_dir=adir)
    with pytest.raises(RoiQuestRefused):
        s.auto_roi_matching()
    s.select_all()
    before = _flat(s.rois[0][0].indices).copy()
    res = key_press(s, "c")
    assert res.action == "auto_roi_matching" and Redraw.OVERLAY in res.redraw
    np.testing.assert_array_equal(_flat(s.rois[0][0].indices), before)
    assert not s.rois[0][0].selected and s.show_vectors
    np.testing.assert_array_equal(s.roi_trans[0][:2].ravel(), [0.0, 0.0])
    start, _disp = s.vectors
    np.testing.assert_allclose(start[0], np.array(BLOBS_Z0[0], dtype=float), atol=0.5)


def test_chronic_save_new_rois_and_tracked_delete(series):
    _need_update_mat()
    adir, el = series
    s = RoiQuestSession.from_expid(202, exp_log=el, adata_dir=adir)
    _zoom_putative(s, BLOBS_Z0[1])
    key_press(s, "a")
    asked = []
    out = s.save(add_new_rois=lambda res: asked.append(res) or True)
    assert asked and asked[0].rois_to_add
    p201, p202 = adata_path(adir, 201), adata_path(adir, 202)
    assert set(out.written) == {p201, p202}
    assert set(out.backups) == {p201.resolve(), p202.resolve()}
    assert len(_layer(load_mat(p201)["ROIs"], 0)) == 2
    assert len(_layer(load_mat(p202)["ROIs"], 0)) == 2
    assert len(_layer(load_mat(adata_path(adir, 203))["ROIs"], 0)) == 0
    np.testing.assert_array_equal(_flat(_layer(load_mat(p201)["ROIs"], 0)[1]["indices"]),
                                  _flat(find(disk((H, W), BLOBS_Z0[1], 4))))
    assert not any(r.is_new for r in s.rois[0])

    # delete the tracked first ROI from all timepoints
    s.button_down(*BLOBS_Z0[0])
    s.button_up(*BLOBS_Z0[0], kind="alt")
    assert s.deletion_kind() == "tracked"
    res = key_press(s, "d")
    assert (res.action, res.executed) == ("confirm_delete", False)
    with pytest.raises(RoiQuestRefused):
        s.delete_selected()
    s.delete_selected(include_tracked=True)
    for p in (p201, p202):
        layer = _layer(load_mat(p)["ROIs"], 0)
        assert len(layer) == 1
        np.testing.assert_array_equal(_flat(layer[0]["indices"]),
                                      _flat(find(disk((H, W), BLOBS_Z0[1], 4))))
    assert len(s.rois[0]) == 1


def test_chronic_mixed_delete_keeps_tracked(series):
    adir, el = series
    s = RoiQuestSession.from_expid(202, exp_log=el, adata_dir=adir)
    _zoom_putative(s, BLOBS_Z0[1])
    key_press(s, "a")
    s.select_all()
    assert s.deletion_kind() == "mixed"
    s.delete_selected()  # "Delete only new ROIs"
    assert len(s.rois[0]) == 1 and not s.rois[0][0].is_new


def test_chronic_save_needs_explog(series):
    adir, _el = series
    s = RoiQuestSession.from_adata(adata_path(adir, 202),
                                   original_adata_path=adata_path(adir, 201))
    with pytest.raises(RoiQuestRefused, match="ExpLog"):
        s.save()


def test_chronic_save_rejects_fewer_rois(series):
    _need_update_mat()
    adir, el = series
    s = RoiQuestSession.from_expid(202, exp_log=el, adata_dir=adir)
    s.rois[0] = []
    with pytest.raises(RoiSaveError):
        s.save()


def test_tracked_delete_of_all_rois_leaves_session_empty(tmp_path):
    """Deleting every tracked ROI: the session holds empty layers (MATLAB kept the deleted ROIs
    on screen because loadRois aborts on a file without ROIs) and nothing comes back."""
    _need_update_mat()
    adir = tmp_path / "adata"
    for e in (201, 202):
        make_2p(adata_path(adir, e))
    el = explog([(201, 5), (202, 5)])
    s = RoiQuestSession.from_expid(202, exp_log=el, adata_dir=adir)
    s.select_all()
    assert s.deletion_kind() == "tracked"
    s.delete_selected(include_tracked=True)
    assert s.rois == [[], []] and s.putative_rois == []
    for e in (201, 202):
        assert all(c.size == 0 for c in load_mat(adata_path(adir, e))["ROIs"].ravel())
    with pytest.raises(RoiSaveError):  # the original has no ROIs any more: nothing written
        s.save()
    assert all(c.size == 0 for c in load_mat(adata_path(adir, 202))["ROIs"].ravel())
    with pytest.warns(UserWarning, match="no ROIs"):
        t = RoiQuestSession.from_expid(202, exp_log=el, adata_dir=adir)
    assert t.rois == [[], []]
    o = RoiQuestSession.from_expid(201, exp_log=el, adata_dir=adir)  # original: acute save
    assert o.rois == [[], []]
    o.save()
    assert RoiQuestSession.from_adata(adata_path(adir, 201)).rois == [[], []]
