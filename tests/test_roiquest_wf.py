"""RoiQuest session on widefield data: reference points, template ROIs, saving."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from mat73_writer import Obj, write_mat73
from test_roiquest_data import (
    TEMPLATE_HW,
    WF_H,
    WF_REF,
    WF_W,
    adata_path,
    explog,
    lin,
    make_roi_template,
    make_wf1,
    make_wf2,
    wf_frame,
)

from calliopy.io import MatStruct, load_mat
from calliopy.roiquest import (
    EEG_RING_TEMPLATE_FILE,
    Redraw,
    RoiQuestRefused,
    RoiQuestSession,
    key_press,
)

HW = (WF_H, WF_W)


def _need_update_mat():
    try:
        from calliopy.io import update_mat  # noqa: F401
    except ImportError:
        pytest.skip("calliopy.io.update_mat is not available yet (written by another track)")


def _flat(a):
    return np.asarray(a, dtype=np.float64).ravel(order="F")


def _mask(indices, hw=HW):
    m = np.zeros(hw, dtype=bool)
    m.T.flat[_flat(indices).astype(int) - 1] = True
    return m


def _first_cell(var):
    c = var.ravel(order="F")[0]
    return list(c.ravel(order="F"))


@pytest.fixture
def wf1(tmp_path):
    return make_wf1(tmp_path / "Adata-S1-T401.mat",
                    params=MatStruct(name="Synthetic grid", parameters=MatStruct(a=1.0)))


@pytest.fixture
def tdir(tmp_path):
    d = tmp_path / "templates"
    d.mkdir()
    make_roi_template(d / "grid.mat")
    make_roi_template(d / EEG_RING_TEMPLATE_FILE, name="Ring")
    return d


def test_load_wf1(wf1):
    s = RoiQuestSession.from_adata(wf1)
    assert s.is_widefield_data and not s.is_wf_two_channels and not s.can_modify_rois
    assert len(s.imgs) == 2 and s.num_layers == 1 and s.act_map == []
    assert s.valid_area.all() and s.valid_area.shape == (WF_H, WF_W, 1)
    assert len(s.rois[0]) == 4 and s.rois[0][2].name == "roi0"
    np.testing.assert_array_equal(s.ref_points, np.array(WF_REF, dtype=float))
    assert len(s.ref_rois) == 2
    assert s.ref_rois[0].to_binary_mask()[WF_REF[0]]  # cross centred on the point
    assert s.roi_template_parameters["name"] == "Synthetic grid"
    en = s.enabled
    assert en["wfAddRefPoints"] and en["wfAddRois"] and not en["roiExpand"]
    assert not en["cellposeDetect"]
    assert s.overlay_text() == "ExpID: 401 - Channel: 1 - 4 ROI(s)"
    assert s.overlay_text(free_form_msg="Select reference point #1 (left-click)") == \
        "ExpID: 401 - Select reference point #1 (left-click)"


def test_wf_refusals(wf1):
    s = RoiQuestSession.from_adata(wf1)
    s.select_all()
    assert s.deletion_kind() == "widefield"
    with pytest.raises(RoiQuestRefused):
        s.delete_selected()
    with pytest.raises(RoiQuestRefused):
        s.auto_roi_matching()
    s.button_down(30, 30)
    assert Redraw.VIEW in s.button_up(45, 50)
    assert s.is_zoomed_in and s.putative_rois == []  # no segmentation for widefield
    assert s.scroll(-1) == Redraw.NONE and s.thr[0] == 160.0
    s.threshold_step("up")
    assert s.thr[0] == 160.0
    with pytest.raises(ValueError):
        s.load_rois_from_adata(wf1)


def test_wf_move_only_eeg_anulus_and_ref_points(wf1):
    s = RoiQuestSession.from_adata(wf1)
    s.rois[0][2].name = "eeg_anulus_3"
    s.select_all()
    a2, a3 = _flat(s.rois[0][2].indices).copy(), _flat(s.rois[0][3].indices).copy()
    s.move_selected("right")
    np.testing.assert_array_equal(_flat(s.rois[0][2].indices), a2 + WF_H)
    np.testing.assert_array_equal(_flat(s.rois[0][3].indices), a3)
    # select the first reference point with a right click on it
    s.select_none()
    s.button_down(*WF_REF[0])
    s.button_up(*WF_REF[0], kind="alt")
    assert [r.selected for r in s.ref_rois] == [True, False]
    key_press(s, "left", ("shift",))
    np.testing.assert_array_equal(s.ref_points[0], [WF_REF[0][0], WF_REF[0][1] - 10])
    np.testing.assert_array_equal(s.ref_points[1], WF_REF[1])
    # shift-click toggles the reference point off again
    r, c = s.ref_points[0]
    s.button_up(r, c, kind="extend")
    assert not s.ref_rois[0].selected


def test_ref_points_and_template_rois(wf1, tdir):
    s = RoiQuestSession.from_adata(wf1, roi_template_dir=tdir)
    assert s.roi_template_files() == sorted([EEG_RING_TEMPLATE_FILE, "grid.mat"])
    s.ref_rois = []
    with pytest.raises(RoiQuestRefused):
        s.wf_add_rois("grid.mat")
    # template points (5, 30) and (45, 30); image points 5 rows and 15 columns further
    s.wf_add_ref_points([(10.2, 45.0), (50, 44.6)])
    np.testing.assert_array_equal(s.ref_points, [[10, 45], [50, 45]])
    assert s.wf_add_rois("grid.mat") == Redraw.OVERLAY
    rois = s.rois[0]
    assert [r.name for r in rois] == ["bregma", "lambda", "square"]
    assert _flat(rois[0].indices).tolist() == [lin((10, 45), HW)]
    assert _flat(rois[1].indices).tolist() == [lin((50, 45), HW)]
    sq = _mask(rois[2].indices)
    rr, cc = np.nonzero(sq)
    assert sq.sum() >= 36 and abs(rr.mean() - 27.5) <= 0.5 and abs(cc.mean() - 27.5) <= 0.5
    assert rois[2].allen_index == 7.0
    p = s.roi_template_parameters
    assert "ROIs" not in p and "img_ROIs" not in p and p["name"] == "Synthetic grid"
    assert tuple(np.ravel(p["parameters"]["SIZE_TEMPLATE"])) == TEMPLATE_HW
    assert not s.can_modify_rois


def test_wf_save_single_and_reload(wf1, tdir):
    _need_update_mat()
    s = RoiQuestSession.from_adata(wf1, roi_template_dir=tdir)
    s.wf_add_ref_points([(10, 45), (50, 45)])
    s.wf_add_rois("grid.mat")
    out = s.save()
    assert set(out.written[wf1]) == {"ROIs", "ROItemplate_parameters"}
    assert out.backups
    d = load_mat(wf1)
    saved = _first_cell(d["ROIs"])
    assert [x["name"] for x in saved] == ["bregma", "lambda", "square"]
    assert d["ROItemplate_parameters"]["name"] == "Synthetic grid"
    assert "registration_parameters" in d and "first_frame" in d
    t = RoiQuestSession.from_adata(wf1)
    np.testing.assert_array_equal(t.ref_points, [[10, 45], [50, 45]])
    for a, b in zip(s.rois[0], t.rois[0]):
        np.testing.assert_array_equal(_flat(a.indices), _flat(b.indices))


def test_eeg_ring_template_marks_file(wf1, tdir):
    _need_update_mat()
    s = RoiQuestSession.from_adata(wf1, roi_template_dir=tdir)
    s.wf_add_rois(EEG_RING_TEMPLATE_FILE)
    assert s.can_modify_rois
    assert bool(load_mat(wf1)["is_eeg"])
    assert RoiQuestSession.from_adata(wf1).can_modify_rois


def test_wf_series_save(tmp_path):
    _need_update_mat()
    adir = tmp_path / "adata"
    p1 = make_wf1(adata_path(adir, 401))
    shift = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])  # x+3, y+2
    p2 = make_wf1(adata_path(adir, 402), rois=False, extra={"tform": shift})
    el = explog([(401, 4), (402, 4)])
    s = RoiQuestSession.from_expid(401, exp_log=el, adata_dir=adir)
    assert s.exp_group == [401, 402]
    with pytest.raises(ValueError):
        s.save()
    out = s.save(overwrite_policy="keep")
    assert set(out.written) == {p2}
    moved = _first_cell(load_mat(p2)["ROIs"])
    # reference points mapped with the inverse transform: 3 columns and 2 rows back
    r0, c0 = WF_REF[0]
    assert _flat(moved[0]["indices"]).tolist() == [lin((r0 - 2, c0 - 3), HW)]
    out = s.save(overwrite_policy="overwrite")
    assert set(out.written) == {p1, p2}
    # a non-first ExpID cannot be saved
    t = RoiQuestSession.from_expid(402, exp_log=el, adata_dir=adir)
    from calliopy.rois import RoiSaveError

    with pytest.raises(RoiSaveError):
        t.save(overwrite_policy="keep")


def test_load_wf2_and_template_rois(tmp_path, tdir):
    p = make_wf2(tmp_path / "Adata-S1-T501.mat")
    s = RoiQuestSession.from_adata(p, roi_template_dir=tdir)
    assert s.is_wf_two_channels and s.num_layers == 2 and len(s.rois) == 2
    assert len(s.rois[0]) == len(s.rois[1]) == 4
    np.testing.assert_array_equal(s.tforms_2ch["ch_map"], np.eye(3))
    s.wf_add_ref_points([(10, 45), (50, 45)])
    s.wf_add_rois("grid.mat")
    green, red = s.rois
    # identity channel map: red ROIs are the mirrored green ROIs
    np.testing.assert_array_equal(_mask(red[2].indices), _mask(green[2].indices)[:, ::-1])
    key_press(s, "j")
    assert s.current_z == 1 and s.overlay_text().startswith("ExpID: 501 - Channel: 2")
    s.wf_add_rois("grid.mat")
    green, red = s.rois
    np.testing.assert_array_equal(_mask(green[2].indices), _mask(red[2].indices)[:, ::-1])


def test_wf2_transform_object_needs_matrix(tmp_path, tdir):
    p = tmp_path / "Adata-S1-T502.mat"
    write_mat73(p, {"first_frame_ch525": wf_frame(1), "first_frame_ch610": wf_frame(2),
                    "tform_ch_map": Obj("affine2d")})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = RoiQuestSession.from_adata(p, roi_template_dir=tdir)
    assert s.rois == [[], []]
    s.wf_add_ref_points([(10, 45), (50, 45)])
    with pytest.raises(RoiQuestRefused, match="tform_ch_map"):
        s.wf_add_rois("grid.mat")
    s.wf_add_rois("grid.mat", tform_ch_map=np.eye(3))
    assert len(s.rois[1]) == 3
