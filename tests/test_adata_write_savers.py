"""Tests of the adata savers of calliopy.twophoton, .widefield, .behavior and .oii."""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest

from calliopy.behavior import preprocess_owl, save_preprocess_owl_results
from calliopy.io import (
    MODIFICATION_LOG_VARIABLE,
    MatFile,
    MatStruct,
    adata_file_path,
    load_adata,
    load_mat,
    save_mat,
)
from calliopy.oii import save_analyze_oii_fourier_results
from calliopy.twophoton import (
    calc_act,
    register_bin,
    save_calc_act_results,
    save_check_raw_passed,
    save_correct_dxdy_results,
    save_detected_spines,
    save_register_bin_results,
    save_register_exp_avgplanes_results,
    save_register_exp_downsample_results,
    save_register_exp_fixoutliers_results,
    save_spine_tracking,
    save_spiny_regions,
    save_tpi_adata_check_results,
)
from calliopy.widefield import (
    PropagateTformResult,
    save_calc_widefield_act_results,
    save_propagate_tform_results,
    save_register_wid_results,
    save_wid_adata_check_results,
)
from calliopy.widefield import save_check_raw_passed as wf_save_check_raw_passed

LOG = MODIFICATION_LOG_VARIABLE
LOG_RE = re.compile(r"\d\d-[A-Z][a-z]{2}-\d{4} \d\d:\d\d:\d\d - SVN revision nbr: calliopy-\S+ : ")


def info(path):
    with MatFile(path) as mf:
        return mf.info()


def stacks(n_layers=2, shape=(16, 16, 40), seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(100, 1000, size=shape[:2])
    return [(base[:, :, None] + rng.integers(0, 50, size=shape)).astype(np.uint16)
            for _ in range(n_layers)]


# =======================================================================================
# register_bin
# =======================================================================================


def test_register_bin_saver_real_result(tmp_path):
    data = stacks()
    res = register_bin(data, np.array([[80]]))
    path = adata_file_path(tmp_path, "U1", "M1", 321)
    out = save_register_bin_results(path, res, aux_files=["a.lvd"], fnames=["S1-T321_ch525.bin"],
                                    mouse_id="M1", user_id="U1")
    assert out == path and path.is_file()
    i = info(path)
    assert i["ROIs"] == ("cell", (1, 2)) and i["np"] == ("cell", (1, 2))
    assert i["template"] == ("cell", (1, 2)) and i["dx"] == ("cell", (1, 2))
    assert i["nbr_frames"] == ("double", (1, 1))
    assert i["template_sec"] == ("cell", (0, 0)) and i["act_map_sec"] == ("cell", (0, 0))
    assert i["mouse_id"] == ("char", (1, 2)) and i["userID"] == ("char", (1, 2))
    assert i[LOG] == ("struct", (1, 1))
    ad = load_adata(path)
    assert ad.type == "2p" and ad.exp_id == 321
    assert isinstance(ad.ROIs[0, 0], MatStruct) and len(ad.ROIs[0, 0]) == 0
    np.testing.assert_array_equal(ad.dx[0, 1], res["dx"][1])
    with pytest.raises(FileExistsError):
        save_register_bin_results(path, res, aux_files=[""], fnames=["x_y"], mouse_id="M1",
                                  user_id="U1")


def test_register_bin_saver_zstack_and_missing_metadata(tmp_path):
    p = tmp_path / "Adata-S1-T1.mat"
    save_register_bin_results(p, MatStruct(mean_data=np.ones((4, 4, 3)),
                                           mean_data_sec=np.zeros((4, 4, 3))))
    assert set(info(p)) == {"mean_data", "mean_data_sec", LOG}
    with pytest.raises(ValueError, match="fnames"):
        save_register_bin_results(tmp_path / "Adata-S1-T2.mat",
                                  MatStruct(ROIs=[MatStruct()], bv=[MatStruct()], np=[MatStruct()]))


# =======================================================================================
# calc_act
# =======================================================================================


def _calc_act_file(tmp_path):
    layer0 = np.empty((1, 2), dtype=object)
    layer0[0, 0] = MatStruct(indices=np.array([[1.0], [2.0], [17.0]]))
    layer0[0, 1] = MatStruct(indices=np.array([[40.0], [41.0]]))
    rois = np.empty((1, 2), dtype=object)
    rois[0, 0], rois[0, 1] = layer0, MatStruct()
    nps = [MatStruct(), MatStruct()]  # a list: a cell of structs (see _cell_row)
    path = tmp_path / "Adata-S1-T5.mat"
    save_mat(path, {"ROIs": rois, "np": nps, "template": [np.ones((16, 16))] * 2,
                    "dx": [np.zeros((40, 1))] * 2, "fnames": ["S1-T5_ch525.bin"]})
    return path, rois


def test_calc_act_saver(tmp_path):
    path, rois = _calc_act_file(tmp_path)
    data = stacks()
    res = calc_act(data, [rois[0, 0], rois[0, 1]])
    assert save_calc_act_results(path, res) is True
    d = load_mat(path)
    layer0 = d["ROIs"][0, 0]
    assert layer0.shape == (1, 2) and layer0[0, 0].activity.shape == (40, 1)
    np.testing.assert_array_equal(layer0[0, 0].indices, [[1.0], [2.0], [17.0]])  # 1-based kept
    assert isinstance(d["ROIs"][0, 1], MatStruct) and len(d["ROIs"][0, 1]) == 0  # untouched
    np0 = d["np"][0, 0]
    assert isinstance(np0, MatStruct) and np0.indices.shape == (256 - 5, 1)
    assert np0.indices[0, 0] == 3.0 and np0.activity.shape == (40, 1)
    proj = d["projections"]
    assert info(path)["projections"] == ("struct", (1, 1))
    assert proj["median"][0, 0].dtype == np.uint16 and proj["std"][0, 0].dtype == np.float64
    log = d[LOG][0, 0]
    assert sorted(log.modified_variables[:, 0]) == ["ROIs", "np", "projections"]
    # projections exist now: MATLAB's add_variable skips them with a warning
    with pytest.warns(UserWarning, match='"projections" already exists'):
        assert save_calc_act_results(path, calc_act([x * 2 for x in data],
                                                    [rois[0, 0], rois[0, 1]])) is True
    np.testing.assert_array_equal(load_mat(path)["projections"]["mean"][0, 0], proj["mean"][0, 0])


def test_calc_act_saver_skip_act_calc_writes_np_only(tmp_path):
    path, rois = _calc_act_file(tmp_path)
    res = calc_act(stacks(), [rois[0, 0], rois[0, 1]], recalc_rois=False)
    save_calc_act_results(path, res, rois_recalculated=False)
    log = load_mat(path)[LOG][0, 0]
    assert sorted(log.modified_variables[:, 0]) == ["np", "projections"]


# =======================================================================================
# tpi_adata_check, check_raw
# =======================================================================================


def test_tpi_adata_check_saver_and_check_raw_passed(tmp_path):
    path = tmp_path / "Adata-S1-T6.mat"
    save_mat(path, {"dx": [np.zeros((3, 1))]})
    res = MatStruct(check_raw=MatStruct(passed=0.0, roi_similarity=0.5,
                                        act_similarity=np.array([[0.99, 0.2]])))
    assert save_tpi_adata_check_results(path, res) is True
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert save_tpi_adata_check_results(path, res) is False
    assert any("already exists" in str(w.message) for w in rec)
    assert save_check_raw_passed(path) is True
    cr = load_mat(path)["check_raw"]
    assert cr.passed == 1.0 and cr.roi_similarity == 0.5
    np.testing.assert_array_equal(cr.act_similarity, [[0.99, 0.2]])
    other = tmp_path / "Adata-S1-T7.mat"
    save_mat(other, {"dx": 1.0})
    with pytest.warns(UserWarning, match="not checked yet"):
        assert wf_save_check_raw_passed(other) is False


# =======================================================================================
# registration fixes
# =======================================================================================


def _reg_result(n_layers=2, with_frames=True):
    r = MatStruct(dx=[np.ones((6, 1))] * n_layers, dy=[np.zeros((6, 1))] * n_layers,
                  act_map=[np.ones((4, 4))] * n_layers, template=[np.full((4, 4), 2.0)] * n_layers,
                  template_sec=[])
    if with_frames:
        r["nbr_frames"] = np.array([[12]])
    return r


def test_downsample_saver_existing_file(tmp_path):
    path = tmp_path / "Adata-S1-T8.mat"
    old_log = np.empty((1, 1), dtype=object)
    old_log[0, 0] = "earlier entry"
    save_mat(path, {"ROIs": [MatStruct(a=1.0)], "registration_log": old_log, "dx": 0.0})
    save_register_exp_downsample_results(path, _reg_result(), fta=3, template_frames=range(10, 30))
    d = load_mat(path)
    assert d["registration_log"].shape == (1, 2) and d["registration_log"][0, 0] == "earlier entry"
    entry = d["registration_log"][0, 1]
    assert LOG_RE.match(entry)
    assert entry.endswith(" : register_exp_downsample - fta: 3, template_frames: 11:30 EOL")
    assert info(path)["nbr_frames"] == ("double", (1, 1))
    assert info(path)["template_sec"] == ("cell", (0, 0)) and info(path)["dx"] == ("cell", (1, 2))
    assert d["ROIs"][0, 0].a == 1.0 and LOG not in d  # plain save: no modification log
    with pytest.raises(ValueError, match="nbr_frames"):
        save_register_exp_downsample_results(path, _reg_result(with_frames=False))


@pytest.mark.parametrize("n_layers", [1, 2])
def test_avgplanes_saver_new_file(tmp_path, n_layers):
    path = tmp_path / "U" / "M" / "Adata-S1-T9.mat"
    with pytest.raises(FileNotFoundError):
        save_register_exp_avgplanes_results(path, _reg_result(n_layers))
    save_register_exp_avgplanes_results(path, _reg_result(n_layers), fta=2,
                                        fnames=["S1-T9_ch525.bin"], aux_files=["S1-T9.lvd"],
                                        mouse_id="M", user_id="U")
    i = info(path)
    assert set(i) == {"fnames", "aux_files", "mouse_id", "userID", "registration_log",
                      "nbr_frames", "ROIs", "bv", "np", "dx", "dy", "act_map", "template",
                      "template_sec"}
    assert i["ROIs"] == (("struct", (1, 1)) if n_layers == 1 else ("cell", (1, 2)))
    entry = load_mat(path)["registration_log"][0, 0]
    assert entry.endswith(" : register_exp_avgplanes - template_frames: 11:30 fta: 2 EOL")


def test_fixoutliers_saver(tmp_path):
    path = tmp_path / "Adata-S1-T10.mat"
    save_mat(path, {"dx": 0.0})
    assert save_register_exp_fixoutliers_results(path, MatStruct()) is False
    assert set(info(path)) == {"dx"}
    res = MatStruct(dx=np.ones((6, 1)), dy=np.ones((6, 1)), act_map=np.ones((4, 4)),
                    template=np.ones((4, 4)))
    assert save_register_exp_fixoutliers_results(path, res) is True
    i = info(path)
    assert i["dx"] == ("double", (6, 1)) and i["registration_log"] == ("cell", (1, 1))
    assert load_mat(path)["registration_log"][0, 0].endswith(" : register_exp_fixoutliers - EOL")


def test_correct_dxdy_saver_merges_layers(tmp_path):
    path = tmp_path / "Adata-S1-T11.mat"
    save_mat(path, {"dx": [np.zeros((5, 1))] * 3, "dy": [np.zeros((5, 1))] * 3,
                    "act_map": [np.full((2, 2), k) for k in (1.0, 2.0, 3.0)],
                    "template": [np.full((2, 2), k) for k in (1.0, 2.0, 3.0)]})
    res = MatStruct(dx=[np.ones((5, 1))] * 3, dy=[np.ones((5, 1))] * 3,
                    act_map=[None, np.full((2, 2), 9.0), np.full((2, 2), 8.0)],
                    template=[None, np.full((2, 2), 9.0), np.full((2, 2), 8.0)])
    save_correct_dxdy_results(path, res, z_plane=0, z_to_correct=[1, 2])
    d = load_mat(path)
    assert [d["act_map"][0, k][0, 0] for k in range(3)] == [1.0, 9.0, 8.0]
    assert d["registration_log"][0, 0].endswith(" : correct_dxdy - z_plane, 1, 2  3, : EOL")
    save_correct_dxdy_results(path, MatStruct(dx=[np.ones((5, 1))] * 3, dy=[np.ones((5, 1))] * 3))
    assert load_mat(path)["registration_log"][0, 1].endswith(" : correct_dxdy - man_corr, : EOL")


# =======================================================================================
# SpineQuest
# =======================================================================================


def test_spine_savers(tmp_path):
    path = tmp_path / "Adata-S1-T12.mat"
    save_mat(path, {"mean_data": np.zeros((8, 8, 3))})
    spines = [[MatStruct(indices=np.array([[1.0], [2.0]])), MatStruct(indices=np.array([[9.0]]))],
              []]
    assert save_detected_spines(path, spines, [np.array([3, 4]), np.zeros(0)]) is True
    d = load_mat(path)
    assert info(path)["spines"] == ("cell", (1, 2))
    assert d["spines"][0, 0].shape == (1, 2) and float(d["spines"][0, 0][0, 1].indices) == 9.0
    assert d["spines"][0, 1].size == 0  # an empty region is []
    assert d["dendrite"][0, 0].shape == (2, 1)
    # set again: replaced (MATLAB dot assignment)
    save_detected_spines(path, [[MatStruct(indices=np.array([[5.0]]))]], [np.array([1])])
    assert load_mat(path)["spines"].shape == (1, 1)
    tracking = [[MatStruct(indices=np.zeros((0, 1)), presence=np.array([[1.0, 0.0]])),
                 MatStruct(indices=np.array([[2.0]]), presence=np.array([[1.0, 1.0]]))]]
    save_spine_tracking(path, tracking)
    st = load_mat(path)["spine_tracking"][0, 0]
    assert info(path)["spine_tracking"] == ("cell", (1, 1))
    assert st[0, 0].indices.shape == (0, 0) and st[0, 1].presence.shape == (1, 2)


def test_spiny_regions_saver_overwrite_and_rename(tmp_path):
    old = tmp_path / "mean_data-S1-T13.mat"
    save_mat(old, {"mean_data": np.zeros((8, 8, 3))})
    regions = [np.array([[1, 1, 1], [4, 4, 1], [2, 2, 3]])]
    new_path = save_spiny_regions(old, regions)
    assert new_path.name == "Adata-S1-T13.mat" and not old.exists()
    assert info(new_path)["spiny_regions"] == ("cell", (1, 1))
    assert load_mat(new_path)["spiny_regions"][0, 0].dtype == np.float64
    with pytest.warns(UserWarning, match="not overwritten"):
        save_spiny_regions(new_path, [np.ones((3, 3))])
    assert load_mat(new_path)["spiny_regions"][0, 0][0, 0] == 1.0
    save_spiny_regions(new_path, [np.full((3, 3), 7.0)], overwrite=True)
    assert load_mat(new_path)["spiny_regions"][0, 0][0, 0] == 7.0
    # an old-name file whose target exists is not renamed
    other = tmp_path / "mean_data-S1-T13.mat"
    save_mat(other, {"mean_data": np.zeros((2, 2, 2))})
    assert save_spiny_regions(other, regions) == other and other.exists()


# =======================================================================================
# widefield
# =======================================================================================


def _wid_result(origin=False):
    meta = np.empty((1, 2), dtype=object)
    meta[0, 0], meta[0, 1] = MatStruct(t=1.0), MatStruct(t=2.0)
    tform = np.zeros((0, 0)) if origin else np.array([[1.0, 0, 2], [0, 1.0, -1], [0, 0, 1]])
    return MatStruct(meta_info=meta, first_frame=np.ones((4, 5), np.uint16), tform=tform,
                     registration_parameters=MatStruct(status="SUCCESS", similarity=0.9))


def test_register_wid_saver(tmp_path):
    path = tmp_path / "U" / "M" / "Adata-S1-T20.mat"
    save_register_wid_results(path, _wid_result())
    i = info(path)
    assert i["tform"] == ("double", (3, 3)) and i["meta_info"] == ("struct", (1, 2))
    assert i["first_frame"] == ("uint16", (4, 5)) and LOG in i
    np.testing.assert_array_equal(load_mat(path)["tform"], _wid_result().tform)
    assert load_adata(path).type == "wf"
    with pytest.raises(FileExistsError):
        save_register_wid_results(path, _wid_result())
    save_register_wid_results(path, _wid_result(origin=True), overwrite=True)
    assert info(path)["tform"] == ("double", (0, 0))
    bad = _wid_result()
    bad["tform"] = np.eye(2)
    with pytest.raises(ValueError, match="3x3"):
        save_register_wid_results(tmp_path / "Adata-S1-T21.mat", bad)


def test_propagate_calc_and_check_savers(tmp_path):
    path = tmp_path / "Adata-S1-T22.mat"
    save_mat(path, {"first_frame": np.ones((4, 5), np.uint16), "keep": 3.0})
    params = MatStruct(status="SUCCESS", similarity=0.4, registration_type="Copied from 20")
    save_propagate_tform_results(path, PropagateTformResult(np.eye(3), params, None))
    assert set(info(path)) == {"first_frame", "keep", "tform", "registration_parameters"}
    rois = np.empty((1, 1), dtype=object)
    rois[0, 0] = MatStruct(indices=np.array([[1.0]]), name="a")
    save_propagate_tform_results(path, PropagateTformResult(np.eye(3), params, rois))
    assert load_mat(path)["ROIs"][0, 0].name == "a"
    assert info(path)["ROIs"] == ("cell", (1, 1))  # a cell holding one struct stays a cell
    act = MatStruct(ROIs=rois, meta_info=MatStruct(t=1.0),
                    registration_parameters=MatStruct(activity_calculated=1.0))
    save_calc_widefield_act_results(path, act)
    assert load_mat(path)["registration_parameters"].activity_calculated == 1.0
    assert info(path)["ROIs"] == ("cell", (1, 1))
    assert load_mat(path)["keep"] == 3.0
    with pytest.raises(ValueError, match="does not save"):
        save_calc_widefield_act_results(path, MatStruct(first_frame=1.0))
    save_wid_adata_check_results(path, MatStruct(check_raw=MatStruct(passed=0.0,
                                                                     img_similarity=0.5,
                                                                     act_similarity=0.2)))
    save_wid_adata_check_results(path, MatStruct(check_raw=MatStruct(passed=1.0,
                                                                     img_similarity=0.9,
                                                                     act_similarity=1.0)))
    assert load_mat(path)["check_raw"].img_similarity == 0.9
    with pytest.raises(FileNotFoundError):
        save_wid_adata_check_results(tmp_path / "missing.mat", MatStruct(check_raw=MatStruct()))


# =======================================================================================
# behavior, oii
# =======================================================================================


def test_preprocess_owl_saver_real_result(tmp_path):
    rng = np.random.default_rng(1)
    frames = np.full((20, 24, 6), 200, dtype=np.uint8)
    for k in range(6):
        frames[5:12, 3 + 2 * k : 11 + 2 * k, k] = rng.integers(0, 40)
    res = preprocess_owl(frames, n_samples=6)
    path = tmp_path / "U" / "M" / "Adata-S1-T30.mat"
    save_preprocess_owl_results(path, res)
    i = info(path)
    assert i["owl"] == ("struct", (1, 1)) and set(i) == {"owl", LOG}
    owl = load_mat(path)["owl"]
    assert isinstance(owl.KP, MatStruct) and owl.KP.mouse_x.shape == (6, 1)
    assert owl.median_frame.dtype == np.uint8
    np.testing.assert_array_equal(owl.KP.mouse_y, res["owl"].KP.mouse_y)
    with pytest.raises(FileExistsError):
        save_preprocess_owl_results(path, res)


def test_oii_saver_replaces_file(tmp_path):
    path = tmp_path / "U" / "M" / "Adata-S1-T40.mat"
    path.parent.mkdir(parents=True)
    save_mat(path, {"first_frame": np.ones((2, 2))})
    specs = [np.full((3, 3), 1 + 2j, dtype=np.complex64) for _ in range(4)]
    result = MatStruct(Ibv=np.ones((3, 3)), FourierSpecs=specs, Power=np.ones((3, 3)))
    save_analyze_oii_fourier_results(path, result)
    i = info(path)
    assert set(i) == {"result"}
    r = load_mat(path)["result"]
    assert r.FourierSpecs.shape == (1, 4) and r.FourierSpecs[0, 3].dtype == np.complex64
    assert r.FourierSpecs[0, 0][0, 0] == 1 + 2j
