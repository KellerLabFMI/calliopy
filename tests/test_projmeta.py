"""Tests for calliopy.projmeta (24_ProjMeta, 20_MetaData and the aux/ROI helpers of 21_Analysis)."""

from __future__ import annotations

import struct
import warnings

import calliopy.analysis
import numpy as np
import pytest
from calliopy.io import MatStruct
from calliopy.projmeta import (
    act2mat,
    activity_onsets,
    add_uuid_to_proj_meta,
    align_aux_channels,
    assign_aux_method,
    assign_load_ftypes,
    assign_processed_aux,
    cell_browser_good_cells,
    generate_meta_info,
    get_lick_times_thrsh,
    get_smoothed_rois,
    get_vel_ind_from_adata,
    heeg_proj_meta,
    lab_note_page_get,
    plot_pm_sites,
    pm_expids,
    pm_info,
    preprocess_widefield_activity,
    proj_meta_exp_type,
    proj_meta_filename,
    proj_meta_selection,
    project_acodes,
    pulsed_to_continuous,
    read_frame_rate,
    traversal_activity,
    trig2snps,
    widefield_rois_for_proj_meta,
)
from calliopy.projmeta._generate import _nbr_frames_widefield, _parse_comment
from calliopy.projmeta._struct import grid_struct_array, struct_array
from matplotlib.figure import Figure
from test_explog_more import make_exp_log


@pytest.fixture
def fake_analysis(monkeypatch):
    """Stand-ins for the analysis-track functions, so these tests check projmeta only."""
    calls: dict = {"psmooth": []}

    def psmooth(trace, win_size=None):
        calls["psmooth"].append(win_size)
        return np.asarray(trace, dtype=float) + 10

    monkeypatch.setattr(calliopy.analysis, "psmooth", psmooth, raising=False)
    monkeypatch.setattr(calliopy.analysis, "ftfil", lambda d, f, hi, lo: np.asarray(d, float), raising=False)
    monkeypatch.setattr(calliopy.analysis, "smooth2", lambda d, w, axis=None: np.asarray(d, float) * 2,
                        raising=False)
    monkeypatch.setattr(calliopy.analysis, "extract_pulse_envelope", lambda s, **k: np.abs(np.ravel(s)),
                        raising=False)
    return calls


def col(*v):
    return np.asarray(v, dtype=float).reshape(-1, 1)


# --------------------------------------------------------------------------------------
# aux processing
# --------------------------------------------------------------------------------------
def test_assign_aux_method():
    assert assign_aux_method("Shutter", "X") == []
    assert assign_aux_method("Running", "X") == ["pos_to_vel", "pos_to_vel_smoothed"]
    assert assign_aux_method("VRx", "X") == ["do_nothing", "VR_pos_to_vel_smoothed"]
    assert assign_aux_method("Lick", "X") == ["do_nothing", "get_lick_times"]
    assert assign_aux_method("Reward", "X") == ["extend_trig_to_200ms"]
    assert assign_aux_method("nose_x", "X") == ["do_nothing"]
    assert assign_aux_method("nose_y_vel", "X") == ["pos_to_vel", "pos_to_vel_smoothed"]
    assert assign_aux_method("TonotopyID", "X") == ["decodeFrequency", "decodeLevel"]
    assert assign_aux_method("opto", "APC") == ["do_nothing", "pulsed_to_continuous"]
    assert assign_aux_method("STIM", "PLS") == ["do_nothing", "pulsed_to_continuous"]
    assert assign_aux_method("Stim", "CAT") == ["do_nothing"]
    assert assign_aux_method("stim", "YBA") == ["pulsed_to_continuous"]
    assert assign_aux_method("StimulationPower", "YBA") == ["extract_pulse_envelope"]
    assert assign_aux_method("VRmotorsensoryGain", "SNM") == ["do_nothing"]
    assert assign_aux_method("Stim", "ECT") == ["do_nothing"]
    with pytest.warns(UserWarning, match="new kind of aux channel"):
        assert assign_aux_method("weird", "X") == []


def test_pulsed_to_continuous():
    x = np.zeros(10000)
    for start in (1000, 1050, 1100, 6000):
        x[start : start + 11] = 5.0
    out = pulsed_to_continuous(x, 1000, 2)
    assert np.all(out[999:1111] == 5) and out[998] == 0 and out[1111] == 0
    assert np.all(out[5999:6011] == 5) and out[6011] == 0
    np.testing.assert_array_equal(pulsed_to_continuous(np.zeros(10)), np.zeros(10))
    y = np.r_[np.ones(5) * 5, np.zeros(5)]  # an offset before any onset only
    np.testing.assert_array_equal(pulsed_to_continuous(y), y)
    assert pulsed_to_continuous(x.reshape(1, -1)).shape == (1, 10000)


def test_lick_times_and_velocity(fake_analysis):
    np.testing.assert_array_equal(get_lick_times_thrsh([0, 0, 1, 1, 0, 1]), [2, 5])  # 1-based
    data = np.array([0, 1, 2, 9.5, -9, 0])
    res = get_vel_ind_from_adata(data)
    np.testing.assert_allclose(res.vel_raw, [1, 1, 7.5, 1.5, 9])  # wrap-around of a +/-10 V encoder
    np.testing.assert_allclose(res.vel_smoothed, 2 * res.vel)
    assert res.vel_ind.all()
    res5 = get_vel_ind_from_adata(np.array([0, 4, -4]))  # max 4 -> scale 5
    np.testing.assert_allclose(res5.vel_raw, [4, 2])
    with pytest.raises(ValueError):
        get_vel_ind_from_adata(np.array([5.0]))


def test_assign_processed_aux(fake_analysis):
    x = np.zeros(1000)
    x[100:105] = 1
    (name, ext), = assign_processed_aux(x, ["extend_trig_to_200ms"], "Reward")
    assert name == "Reward_extended"
    assert ext[100] == 1 and np.all(ext[101:301] == 0.5) and ext[301] == 0 and ext.size == 1000
    y = np.zeros(150)
    y[140:] = 1
    (_, grown), = assign_processed_aux(y, ["extend_trig_to_200ms"], "Reward")
    assert grown.size == 341 and grown[340] == 1  # MATLAB grows the trace (kept)
    out = assign_processed_aux(np.r_[0, 0, 1, 1, 0], ["do_nothing", "get_lick_times", "vis_speed"], "Lick")
    assert [o[0] for o in out] == ["Lick", "lickTimes", "vis_speed"]
    np.testing.assert_array_equal(out[1][1], [[2]])
    vr = np.array([0, 1, 5, 6, 0.5, 1.5])
    (name, v), = assign_processed_aux(vr, ["VR_pos_to_vel_smoothed"], "VRx")
    assert name == "VRx_vel_smoothed"
    np.testing.assert_allclose(v, 2 * np.array([1, 0, 1, 0, 1]))  # steps above 2 removed
    names = [o[0] for o in assign_processed_aux(np.zeros(10), ["pos_to_vel", "pos_to_vel_smoothed",
                                                                "pulsed_to_continuous", "extract_pulse_envelope"],
                                                "Running")]
    assert names == ["Running", "Running_smoothed", "Running_cont", "Running"]
    with pytest.raises(NotImplementedError):
        assign_processed_aux(np.zeros(10), ["decodeFrequency"], "TonotopyID")
    with pytest.warns(UserWarning):
        assert assign_processed_aux(np.zeros(3), ["bogus"], "x") == []


def test_assign_load_ftypes():
    lft = assign_load_ftypes([".lvd", "525.bin", "owl.avi", "ch525.ini"])
    assert lft.lvd and lft.bin and lft.owl and not lft.ini and not lft.eye
    assert list(lft) == ["ach", "eye", "eyeR", "lvd", "bin", "ini", "vrd", "vid", "vid3", "ton", "owl"]


# --------------------------------------------------------------------------------------
# ROI activity
# --------------------------------------------------------------------------------------
def test_get_smoothed_rois(fake_analysis):
    rois = struct_array([MatStruct(name="a", activity=col(1, 2, 3, 4, 5)),
                         MatStruct(name="b", activity=col(2, 2, 2, 2, 2))])
    res = get_smoothed_rois(rois)
    np.testing.assert_allclose(res.act[0], np.arange(11, 16) / 13)
    np.testing.assert_allclose(res.act[1], np.ones(5))
    assert res.act_fissa.shape == (0, 0)
    info = res.roi_info.ravel()
    assert list(info[0]) == ["name", "F0"] and info[0].F0 == 13 and res.roi_info.shape == (1, 2)
    rois_f = struct_array([MatStruct(activity=col(1, 2, 3), activity_fissa_corrected=col(0, 0, 0))])
    res = get_smoothed_rois(rois_f)
    np.testing.assert_allclose(res.act_fissa, np.ones((1, 3)))
    assert "activity_fissa_corrected" not in res.roi_info[0, 0]


def test_preprocess_widefield_activity(fake_analysis):
    a2 = np.arange(10.0)
    a2[6] = np.nan
    rois = [MatStruct(activity=col(*np.arange(10.0))), MatStruct(activity=a2.reshape(-1, 1))]
    out = preprocess_widefield_activity(rois, [4, 6])
    np.testing.assert_allclose(out[0], np.arange(10.0) + 10)
    np.testing.assert_allclose(out[1, :4], np.arange(4.0) + 10)
    np.testing.assert_array_equal(out[1, 4:], 0)  # NaN in the second stack: left at 0
    assert set(fake_analysis["psmooth"]) == {6250}


def _wf_rois():
    def r(name, ml, ind, act):
        return MatStruct(name=name, ML_index=np.zeros((0, 0)) if ml is None else np.array([[ml]]),
                         indices=np.array([ind], dtype=float) if ind else np.zeros((0, 0)),
                         activity=np.zeros((0, 0)) if act is None else col(*act))
    return [r("bregma", None, [7], None), r("lambda", None, [9], None),
            r("corner_tl", 0, [1, 2], [1, 1, 1, 1]), r("", 0, [3, 4], [2, 4, 6, 8]),
            r("V1", 1, None, None), r("S1", 2, [5, 6], [1, 2, 3, 4])]


def test_widefield_rois_for_proj_meta(fake_analysis):
    rd = widefield_rois_for_proj_meta(_wf_rois(), 100, raw_act=True)
    assert list(rd) == ["reference_points", "corners", "act", "median_act", "act_raw", "ROIinfo"]
    refs = rd.reference_points.ravel()
    assert [x.name for x in refs] == ["bregma", "lambda"]
    np.testing.assert_array_equal(refs[1].indices, [[9]])
    np.testing.assert_allclose(rd.corners.act, [[1, 1, 1, 1]])
    assert rd.corners.ROIinfo[0, 0].name == "corner_tl"
    info = rd.ROIinfo.ravel()
    assert [x.name for x in info] == ["V1", "S1"]
    np.testing.assert_array_equal(info[0].indices, [[3, 4]])  # handed over by the unnamed ROI
    assert "activity" not in info[0]
    smoothed = np.array([[12, 14, 16, 18], [11, 12, 13, 14]], dtype=float)
    med = np.median(smoothed, axis=1, keepdims=True)
    np.testing.assert_allclose(rd.act, smoothed / med, rtol=1e-6)
    assert rd.act.dtype == np.float32 and rd.median_act.shape == (2, 1)
    np.testing.assert_allclose([x.F0 for x in info], med.ravel())
    np.testing.assert_allclose(rd.act_raw, smoothed - 10)
    assert fake_analysis["psmooth"] == [6250, 6250]  # 2*round(6250*100/100/2)
    bad = _wf_rois()
    bad[1]["ML_index"] = np.array([[3]])
    with pytest.raises(ValueError, match="one reference point"):
        widefield_rois_for_proj_meta(bad, 100)


def test_widefield_two_channels(fake_analysis):
    rois = []
    for name, ml, ind in (("r1", None, [7]), ("r2", None, [8]), ("A", 1, [1, 2]), ("B", 2, [3, 4])):
        rois.append(MatStruct(name=name, ML_index=np.zeros((0, 0)) if ml is None else np.array([[ml]]),
                              indices_ch525=np.array([ind], float), indices_ch610=np.array([ind], float),
                              activity_ch525=np.zeros((0, 0)) if ml is None else col(1, 2, 3),
                              activity_ch610=np.zeros((0, 0)) if ml is None else col(4, 5, 6)))
    rd = widefield_rois_for_proj_meta(rois, [100, 100.5])
    assert {"act_ch525", "act_ch610", "median_act_ch525", "median_act_ch610"} <= set(rd)
    assert set(rd.reference_points[0, 0]) == {"name_ch525", "indices_ch525", "name_ch610", "indices_ch610"}
    info = rd.ROIinfo.ravel()
    assert info[0].F0_ch525 == 12 and info[0].F0_ch610 == 15
    assert rd.corners.act_ch525.shape == (0, 0)
    with pytest.raises(ValueError, match="frame rate"):
        widefield_rois_for_proj_meta(rois, [100, 102])


# --------------------------------------------------------------------------------------
# reading a proj_meta
# --------------------------------------------------------------------------------------
def make_proj_meta():
    site0 = MatStruct(animal="M1", ExpGroup=np.array([[10, 11]]), Acode="1", uuid="u",
                      rd=grid_struct_array({
                          (0, 0): MatStruct(act=np.arange(10.0).reshape(2, 5), nbr_frames=np.array([[5]]),
                                            velM_smoothed=np.array([[0, 0.001, 0.001, 0, 0]]),
                                            comments=np.array([["first"]], dtype=object)),
                          (1, 0): MatStruct(act=np.ones((3, 5))),
                          (0, 1): MatStruct(act=np.ones((2, 5))),
                          (1, 1): MatStruct(act=np.ones((3, 5)))}, 2, 2))
    site1 = MatStruct(animal="M2", ExpGroup=np.array([[20]]), Acode="1", uuid="u",
                      rd=grid_struct_array({(0, 0): MatStruct(act=np.zeros((0, 0)))}, 1, 1))
    return struct_array([site0, site1])


def test_pm_expids():
    pm = make_proj_meta()
    np.testing.assert_array_equal(pm_expids(pm), [[10, 11], [20, np.nan]])
    np.testing.assert_array_equal(pm_expids(pm, flat=True), [[10, 20, 11]])  # column-major
    np.testing.assert_array_equal(pm_expids(pm, tps=[0]), [[10], [20]])
    np.testing.assert_array_equal(pm_expids(pm, site_ids=[0]), pm_expids(pm))  # ignored, as MATLAB


def test_pm_info():
    res = pm_info(make_proj_meta())
    assert res.animals == ["M1", "M2"]
    np.testing.assert_array_equal(res.sites_per_animal, [1, 1])
    np.testing.assert_array_equal(res.cells_per_animal, [5, 0])
    np.testing.assert_allclose(res.perc_run, [0.4, -np.inf])
    lines = res.text.splitlines()
    assert lines[1] == "2 animals in this proj_meta file (acodes: 1)"
    assert lines[3] == "1 sites \t - 5 cells \t - M1 \t sites: 10 \t%run: 40.0\tspeed: 0.000 "
    assert lines[-2] == "u"


def test_act2mat_and_trig2snps():
    pm = make_proj_meta()
    res = act2mat(pm, 0, 0)
    assert res.actmat.shape == (5, 5)
    np.testing.assert_array_equal(res.ind_roi_in_layer, [0, 1, 0, 1, 2])
    np.testing.assert_array_equal(res.ind_layer, [0, 0, 1, 1, 1])
    assert not res.spksmat.any() and not res.fissamat.any()  # never filled in MATLAB (kept)
    assert act2mat(pm, 1, 0).actmat.shape == (0, 0)

    rd = grid_struct_array({(0, 0): MatStruct(act=np.arange(50.0).reshape(1, 50), nbr_frames=np.array([[50]]),
                                              frame_times=np.arange(1, 51).reshape(1, 50),
                                              speed=2 * np.arange(50.0).reshape(1, 50))}, 1, 1)
    pm2 = struct_array([MatStruct(ExpGroup=np.array([[1]]), rd=rd)])
    snps, arr = trig2snps(pm2, 0, 0, [20, 2, 47], (-3, 3), ["speed"])
    assert snps.shape == (1, 7, 1)
    np.testing.assert_array_equal(snps[0, :, 0], np.arange(17, 24))
    np.testing.assert_array_equal(arr[0][:, 0], 2 * np.arange(17, 24))
    snps, _ = trig2snps(pm2, 0, 0, [10, 30], (-2, 2))
    np.testing.assert_array_equal(snps[0, 2, :], [10, 30])
    assert trig2snps(pm, 1, 0, [1]).snps.shape == (0, 0)


def test_add_uuid():
    pm = make_proj_meta()
    for s in pm.ravel():
        del s["uuid"]
    out = add_uuid_to_proj_meta(pm, "abc")
    assert [s.uuid for s in out.ravel()] == ["abc", "abc"] and out.shape == (1, 2)
    assert add_uuid_to_proj_meta(out, "zzz") is out
    assert len(add_uuid_to_proj_meta(pm).ravel()[0].uuid) == 36


def test_activity_onsets_and_good_cells():
    act = np.zeros(300)
    act[150:160] = 2
    act[180:185] = 2  # preceded by activity within win/2: rejected
    act[20:25] = 2  # too close to the start
    np.testing.assert_array_equal(activity_onsets(act, 1.5, 100), [149])
    assert activity_onsets(np.zeros(300), 1.5, 100).size == 0
    with pytest.raises(ValueError):
        activity_onsets(act, 1.5, 99)
    pm = make_proj_meta()
    pm[0, 0].rd[0, 0]["act"] = np.array([[0, 1, 2, 3, 4], [0, 10, 20, 30, 40.0]])
    good, p = cell_browser_good_cells(pm)
    assert p == pytest.approx(22.0)  # 90th percentile of the ranges [4, 40, 0 x 8]
    assert good[(0, 0, 0)].tolist() == [False, True] and not good[(0, 1, 1)].any()


def test_traversal_activity():
    one = np.linspace(0.01, 5.0, 500)
    xpos = np.tile(one, 3)
    res = traversal_activity(xpos, xpos, np.ones_like(xpos), np.full(xpos.size, -0.0045) * -1)
    assert res.bins.shape == (2, 100)  # the part before the first reset is not a traversal
    finite = res.bins[0][np.isfinite(res.bins[0])]
    assert np.all(np.diff(finite) > 0)
    np.testing.assert_allclose(np.nanmax(res.grats), 1)
    assert np.isfinite(res.bin_act_speed).sum() == 1 and res.bin_std_speed.size == 14


# --------------------------------------------------------------------------------------
# generate_proj_meta steps
# --------------------------------------------------------------------------------------
def test_project_acodes_and_selection():
    el = make_exp_log()
    acodes = project_acodes(el, "PRJ")
    assert list(acodes) == [1.0, 2.0]
    np.testing.assert_array_equal(acodes[1.0], [100, 102])  # stack 105 has deleted data
    sel = proj_meta_selection(el, "PRJ", 1)
    np.testing.assert_array_equal(sel.stack_ids, [[100, 100, 100], [100, 100, 101], [100, 102, 102]])
    np.testing.assert_array_equal(sel.exp_ids, [[100, 100], [100, 102]])
    np.testing.assert_array_equal(sel.exp_index, [0, 0, 1])
    np.testing.assert_array_equal(sel.unique_stacks, [0, 2])
    assert sel.site_ids.tolist() == [100] and sel.locations == ["L2/3"] and sel.site_order.tolist() == [1]
    sel2 = proj_meta_selection(el, "PRJ", [1, 2])
    assert sel2.site_ids.tolist() == [100, 103] and sel2.site_order.tolist() == [1, 1]
    assert sel2.animal_ids == ["AA_240101_001", "AA_240101_002"]
    with pytest.raises(ValueError):
        proj_meta_selection(el, "PRJ", 5)


def test_proj_meta_filename(tmp_path):
    assert proj_meta_filename("PRJ", 1) == "PRJ_1_meta"
    assert proj_meta_filename("PRJ", [1, 2]) == "PRJ_1-2_meta"
    assert proj_meta_filename("PRJ", [12, 1]) == "PRJ_12- 1_meta"  # MATLAB num2str padding (kept)
    assert proj_meta_filename("PRJ", 1, filename="_x") == "PRJ_1_x_meta"
    assert proj_meta_filename("PRJ", 1, snob=True) == "PRJ_1_snob_meta"
    (tmp_path / "PRJ_1_snob_meta.mat").write_bytes(b"")
    (tmp_path / "PRJ_1_snob1_meta.mat").write_bytes(b"")
    assert proj_meta_filename("PRJ", 1, snob=True, meta_save_dir=tmp_path) == "PRJ_1_snob2_meta"
    assert proj_meta_filename("PRJ", 1, filename="_x", snob=True) == "PRJ_1_snob_xmeta"


def test_align_aux_channels():
    chans = [["A", "B", "C"], ["A", "B", "C"], ["A", "C", "D"]]
    res = align_aux_channels(chans, [0, 0, 1], [0, 2], {"D": None})
    assert res.total_chan_list == ["A", "B", "C", "D"]
    np.testing.assert_array_equal(res.chan_id[1], [0, 2, 3])
    assert res.chan_present == ["11", "10", "11", "01"]
    res = align_aux_channels(chans, [0, 0, 1], [0, 2], {"D": (1, "B2")})
    assert res.total_chan_list == ["A", "B2", "C"]
    np.testing.assert_array_equal(res.chan_id[1], [0, 2, 1])
    res = align_aux_channels(chans, [0, 0, 1], [0, 2], lambda name, total: 0)
    np.testing.assert_array_equal(res.chan_id[1], [0, 2, 0])
    with pytest.raises(ValueError, match="could not be found"):
        align_aux_channels(chans, [0, 0, 1], [0, 2])
    with pytest.raises(RuntimeError):
        align_aux_channels([["A", "B"], ["A"]], [0, 0], [0])
    with pytest.raises(RuntimeError):  # MATLAB's test only fires for a single non-matching channel
        align_aux_channels([["A"], ["B"]], [0, 0], [0])
    align_aux_channels([["A", "B"], ["C", "D"]], [0, 0], [0])  # passes in MATLAB too


def test_exp_type_and_local_functions(tmp_path):
    assert tuple(proj_meta_exp_type([".lvd"])) == (1, False, False)
    assert tuple(proj_meta_exp_type(["525.ini", ".lvd"])) == (0, True, False)
    assert tuple(proj_meta_exp_type(["525.ini", "610.ini"])) == (0, True, True)
    assert proj_meta_exp_type(["min.avi"]).exp_type == 3
    assert tuple(proj_meta_exp_type([".wii", ".lvd"])) == (4, True, False)
    assert tuple(proj_meta_exp_type(["EEG.dat"])) == (5, False, False)
    assert tuple(proj_meta_exp_type(["EEG.dat", "525.wii"])) == (5, True, False)
    with pytest.raises(ValueError):
        proj_meta_exp_type(["a.ini", "b.ini", "c.ini"])
    assert _parse_comment(["FB session", "nothing"]) == ["fb", "undef "]
    assert _parse_comment(["x", "y"]) == "undef undef "
    assert _parse_comment(["x", "gratings dark"]) == ["undef ", "gr", "dark"]
    meta = [MatStruct(file_id=np.array([[i]])) for i in (1, 1, 2, 2, 2)]
    np.testing.assert_array_equal(_nbr_frames_widefield(meta), [[2, 3]])
    with pytest.raises(ValueError):
        _nbr_frames_widefield([MatStruct(file_id=1), MatStruct(file_id=3)])
    (tmp_path / "a.wii").write_text("Foo = 1\nFrameRate = 30.5\n")
    assert read_frame_rate(tmp_path / "a.wii") == 30.5
    cfg = b"param.a=1;param.cameraFrameRate=100;x=2"
    (tmp_path / "b.wii").write_bytes(struct.pack("<I", len(cfg)) + cfg + b"\x00" * 20)
    assert read_frame_rate(tmp_path / "b.wii") == 100.0
    (tmp_path / "c.wii").write_bytes(struct.pack("<I", 3) + b"a=1" + b"\x00" * 5)
    assert read_frame_rate(tmp_path / "c.wii") is None


def test_lab_notes_and_meta_info(tmp_path):
    el = make_exp_log()
    page = tmp_path / "books" / "hans" / "2024" / "AA_240101_001_notes.txt"
    page.parent.mkdir(parents=True)
    page.write_text("hello")
    (tmp_path / "books" / "hans" / "AA_240101_001_dir").mkdir()
    assert lab_note_page_get("AA_240101_001", el, tmp_path / "books") == ["hello"]
    assert lab_note_page_get("AA_240101_002", el, tmp_path / "books") == ["No lab notes found"]
    table = generate_meta_info("PRJ", el)
    assert table[0] == ["ExpID", "SiteID", "AnimalID", "Date", "timepoint", "Acode", None, None, "comment"]
    assert [r[0] for r in table[1:]] == [100, 102, 103, 105]
    assert table[3][8] == "2"  # MATLAB writes the Acode into 'comment' (kept)
    old = [table[0] + ["extra"], [100, 100, "AA", "d", 0, "1", None, None, "1", "e"]]
    appended = generate_meta_info("PRJ", el, existing=old)
    assert [r[0] for r in appended[2:]] == [102, 103, 105] and appended[2][9] == "N/A"
    assert generate_meta_info("PRJ", el, existing=appended) is None


def test_heeg_proj_meta(tmp_path, monkeypatch):
    import calliopy.projmeta._generate as gen

    monkeypatch.setattr(gen, "load_heeg", lambda txt, js: MatStruct(kind="eeg", file=txt.name))
    monkeypatch.setattr(gen, "load_heyetracking", lambda js: MatStruct(kind="eye", file=js.name))
    el = make_exp_log()
    el["paradigm"] = np.array(["p0", "p1", "p2", "p3", "p4", "p5"], dtype=object)
    f1, f2 = tmp_path / "s100", tmp_path / "s103"
    f1.mkdir()
    f2.mkdir()
    (f1 / "S1-T100.ini").write_text("[main]\np.a = 1\nMM.b = 2\neye_tracking_only = 1\n")
    (f2 / "S1-T103.ini").write_text("[main]\np.a = 3\nMM.b = 4\n")
    pm = heeg_proj_meta("PRJ", "u", el, [1, 2], [100, 103], [[100, 100], [100, 102], [103, 103]], [f1, f2])
    s0, s1 = pm.ravel()
    assert s0.rd.shape == (1, 2) and s0.rd[0, 1].kind == "eye" and s0.rd[0, 1].file == "S1-T101.json"
    assert s0.paradigm.tolist() == [["p0", "p1"]] and s0.Acode == "1  2"
    assert s1.rd[0, 0].kind == "eye"  # eye-tracking-only stays on for later sites (kept)
    np.testing.assert_array_equal(s1.ExpGroup, [[103]])


def test_plot_pm_sites():
    pm = make_proj_meta()
    rd = pm[0, 0].rd
    rd[0, 0]["Running_smoothed"] = np.linspace(0, 1, 5).reshape(1, 5)
    for el_ in rd.ravel():
        el_.setdefault("comments", np.array([["c"]], dtype=object))
        el_["nbr_frames"] = np.array([[2, 3]])
    figs = plot_pm_sites(pm, site_indices=[0], downsample=2)
    assert len(figs) == 1 and isinstance(figs[0], Figure) and len(figs[0].axes) >= 2
    assert len(plot_pm_sites(pm, site_indices=[0], downsample=1, keep_clims=False)) == 1
    with pytest.raises(NotImplementedError):
        plot_pm_sites(pm, rastermap_sort=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            plot_pm_sites(pm, downsample=0)
