"""Tests of the adata writer: calliopy.io.WritableAdata, save_adata and the name rules."""

from __future__ import annotations

import datetime
import math
import warnings

import h5py
import numpy as np
import pytest
from mat73_writer import Obj, write_mat73

from calliopy.io import (
    MODIFICATION_LOG_VARIABLE,
    Adata,
    MatFile,
    MatStruct,
    WritableAdata,
    adata_file_path,
    adata_file_path_from_fnames,
    load_adata,
    load_mat,
    save_adata,
    save_mat,
)
from calliopy.io._adata_write import _datestr, _num2str_int, _struct_array

LOG = MODIFICATION_LOG_VARIABLE


def info(path):
    with MatFile(path) as mf:
        return mf.info()


def existing(tmp_path, name="Adata-S1-T123.mat"):
    path = tmp_path / name
    save_mat(path, {
        "first_frame": np.arange(12, dtype=np.uint16).reshape(3, 4),
        "dx": [np.zeros((5, 1)), np.ones((5, 1))],
        "mouse_id": "M1",
    })
    return path


# --- create / add / save ----------------------------------------------------------------


def test_create_writes_nothing_until_save(tmp_path):
    p = tmp_path / "u" / "m" / "Adata-S1-T42.mat"
    ad = WritableAdata.create(p, create_dir=True)
    assert ad.exp_id == 42 and ad.type == "unknown" and ad.variables == ()
    ad.add_variable({"first_frame": np.ones((2, 3), np.uint16)})
    assert not p.parent.exists()  # folder only made on save, as MATLAB
    assert ad.save_changes() is True
    assert p.is_file()
    i = info(p)
    assert i["first_frame"] == ("uint16", (2, 3))
    assert i[LOG] == ("struct", (1, 1))


def test_create_without_save_leaves_no_folder(tmp_path):
    p = tmp_path / "new_mouse" / "Adata-S1-T1.mat"
    WritableAdata.create(p, create_dir=True)
    assert not (tmp_path / "new_mouse").exists()


def test_create_refusals(tmp_path):
    path = existing(tmp_path)
    with pytest.raises(FileExistsError):
        WritableAdata.create(path)
    with pytest.raises(ValueError, match=".mat extension"):
        WritableAdata.create(tmp_path / "Adata-S1-T1.txt")
    with pytest.raises(ValueError, match="empty extension"):
        WritableAdata.create(tmp_path / "Adata-S1-T1")
    with pytest.raises(FileNotFoundError, match="create_dir"):
        WritableAdata.create(tmp_path / "missing" / "Adata-S1-T1.mat")
    WritableAdata.create(tmp_path / "Adata-S1-T1.MAT")  # extension check ignores case


def test_create_adata_root(tmp_path):
    root = tmp_path / "adata"
    root.mkdir()
    WritableAdata.create(root / "u" / "m" / "Adata-S1-T1.mat", create_dir=True, adata_root=root)
    sibling = tmp_path / "adata2"  # MATLAB's string prefix test accepted this one
    sibling.mkdir()
    with pytest.raises(ValueError, match="adata folder"):
        WritableAdata.create(sibling / "Adata-S1-T1.mat", adata_root=root)


def test_create_overwrite_replaces_whole_file(tmp_path):
    path = existing(tmp_path)
    ad = WritableAdata.create(path, overwrite=True)
    ad.add_variable(tform=np.zeros((0, 0)))
    ad.save_changes()
    assert set(info(path)) == {"tform", LOG}


def test_round_trip_with_read_only_adata(tmp_path):
    p = tmp_path / "Adata-S1-T7.mat"
    ad = WritableAdata.create(p)
    ad.add_variable(dx=[np.arange(3.0)], first_frame=np.zeros((2, 2), np.uint16))
    ad.save_changes()
    ro = load_adata(p)
    assert type(ro) is Adata and ro.type == "2p" and ro.exp_id == 7
    assert not hasattr(ro, "add_variable") and not hasattr(ro, "save_changes")
    np.testing.assert_array_equal(ro.dx[0, 0], np.arange(3.0).reshape(1, 3))
    rw = WritableAdata(p)
    assert isinstance(rw, Adata) and rw.type == "2p"


def test_add_variable_rules(tmp_path):
    ad = WritableAdata(existing(tmp_path))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ad.add_variable({"first_frame": 1.0, "new_var": 2.0})
        ad.add_variable({})
    msgs = [str(w.message) for w in rec]
    assert any("already exists" in m for m in msgs) and any("Empty struct" in m for m in msgs)
    assert list(ad.modified) == ["new_var"]
    assert ad.first_frame.dtype == np.uint16  # existing variable untouched
    with pytest.raises(ValueError):
        ad.add_variable({"1bad": 1.0})
    with pytest.raises(TypeError):
        ad.add_variable([("x", 1.0)])


def test_assignment_and_modification_types(tmp_path):
    ad = WritableAdata(existing(tmp_path))
    ad.mouse_id = "M2"
    ad["dx"] = [np.zeros((5, 1))]
    ad.add_variable(extra=1.0)
    ad.extra = 3.0  # an added, unsaved variable that is set again becomes 'modified'
    mods = ad.modified
    assert {k: v.mod_type for k, v in mods.items()} == {
        "mouse_id": "modified", "dx": "modified", "extra": "modified"}
    datetime.datetime.strptime(mods["dx"].last_modified, "%d.%m.%Y-%H:%M:%S")
    with pytest.raises(AttributeError, match="add_variable"):
        ad.not_a_variable = 1.0
    with pytest.raises(KeyError):
        ad["not_a_variable"] = 1.0


def test_save_changes_existing_file_keeps_other_variables(tmp_path):
    path = existing(tmp_path)
    before = load_mat(path)
    ad = WritableAdata(path)
    ad.mouse_id = "M2"
    assert ad.save_changes(caller="test") is True
    after = load_mat(path)
    assert after["mouse_id"] == "M2"
    np.testing.assert_array_equal(after["first_frame"], before["first_frame"])
    np.testing.assert_array_equal(after["dx"][0, 1], before["dx"][0, 1])
    log = after[LOG]
    assert log.shape == (1, 1)
    row = log[0, 0]
    assert row.caller == "test" and row.modified_variables[0, 0] == "mouse_id"
    assert row.mod_types[0, 0] == "modified" and "calliopy" in row.class_version
    assert "Adata 1.4.2" in row.class_version
    datetime.datetime.strptime(row.datetime, "%d-%b-%Y %H:%M:%S")
    assert abs(float(row.unix_time) - datetime.datetime.now().timestamp()) < 60
    assert ad.modified == {}
    # a second save appends a row; nothing modified -> warning, no write
    ad.add_variable(dy=[np.zeros((5, 1))])
    ad.save_changes()
    log = load_mat(path)[LOG]
    assert log.shape == (2, 1) and log[1, 0].mod_types[0, 0] == "added"
    assert "save_changes" in log[1, 0].caller
    with pytest.warns(UserWarning, match="Nothing to save"):
        assert ad.save_changes() is False


def test_new_file_without_variables_warns(tmp_path):
    ad = WritableAdata.create(tmp_path / "Adata-S1-T1.mat")
    with pytest.warns(UserWarning, match="add_variable"):
        assert ad.save_changes() is False
    assert not (tmp_path / "Adata-S1-T1.mat").exists()


@pytest.mark.parametrize("status, saved", [(math.nan, True), (0, True), (1, True), (2, False)])
def test_raw_data_lock(tmp_path, status, saved):
    path = existing(tmp_path)
    ad = WritableAdata(path)
    ad.mouse_id = "M2"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert ad.save_changes(raw_data_status=status) is saved
    assert (load_mat(path)["mouse_id"] == "M2") is saved
    assert any("locked" in str(w.message) for w in rec) is (not saved)


def test_raw_data_lock_unexpected_values_and_force(tmp_path):
    path = existing(tmp_path)
    ad = WritableAdata(path)
    ad.mouse_id = "M2"
    for status, msg in ((5, "Unexpected RawDataStatus"), ([], "not found in ExpLog")):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            assert ad.save_changes(raw_data_status=status) is False
        texts = [str(w.message) for w in rec]
        assert any(msg in t for t in texts) and any("locked" in t for t in texts)
    assert ad.save_changes(raw_data_status=2, force=True) is True
    # a new file is never locked
    new = WritableAdata.create(tmp_path / "Adata-S1-T9.mat")
    new.add_variable(x=1.0)
    assert new.save_changes(raw_data_status=2) is True


def test_matlab_modification_log_table_left_alone(tmp_path):
    path = tmp_path / "Adata-S1-T5.mat"
    write_mat73(path, {"first_frame": np.ones((2, 2), np.uint16),
                       "modification_log": Obj("table")})
    with h5py.File(path, "r+"):
        pass
    if path.read_bytes()[:19] != b"MATLAB 7.3 MAT-file":
        pytest.skip("test writer does not add the MATLAB header")
    ad = WritableAdata(path)
    ad.first_frame = np.zeros((2, 2), np.uint16)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # replacing the table would warn
        ad.save_changes()
    i = info(path)
    assert i["modification_log"][0] == "table"
    assert i[LOG] == ("struct", (1, 1))


def test_cell_of_structs_stays_a_cell(tmp_path):
    path = tmp_path / "Adata-S1-T3.mat"
    save_mat(path, {"ROIs": [MatStruct(a=1.0), MatStruct(a=2.0)]})
    assert info(path)["ROIs"] == ("cell", (1, 2))
    ad = WritableAdata(path)
    rois = ad.ROIs  # (1, 2) object array of MatStruct, ambiguous for save_mat
    assert rois.shape == (1, 2) and isinstance(rois[0, 1], MatStruct)
    rois[0, 1] = MatStruct(a=5.0)
    ad.ROIs = rois
    ad.save_changes()
    assert info(path)["ROIs"] == ("cell", (1, 2))
    assert load_mat(path)["ROIs"][0, 1].a == 5.0


def test_load_all_on_unsaved_file(tmp_path):
    ad = WritableAdata.create(tmp_path / "Adata-S1-T1.mat")
    ad.add_variable(x=1.0)
    assert ad.load_all() is ad and ad.loaded == ("x",)


# --- helpers -------------------------------------------------------------------------------


def test_struct_array_layout():
    assert _struct_array([]) is None
    one = _struct_array([{"indices": np.ones((2, 1))}])
    assert isinstance(one, MatStruct)
    many = _struct_array([{"a": 1.0}, {"a": 2.0}])
    assert many.shape == (1, 2) and many[0, 1]["a"] == 2.0


def test_num2str_int_and_datestr():
    assert _num2str_int([3, 4]) == "3  4"
    assert _num2str_int([3, 10]) == "3  10"
    assert _num2str_int(5) == "5"
    assert _num2str_int([-1, 2]) == "-1   2"
    assert _datestr(datetime.datetime(2026, 3, 4, 5, 6, 7)) == "04-Mar-2026 05:06:07"


def test_adata_file_path_rules(tmp_path):
    assert adata_file_path(tmp_path, "U", "M1", 123) == tmp_path / "U" / "M1" / "Adata-S1-T123.mat"
    fn = ["X\\Y\\S1-T55_ch525.bin", "other"]
    p = adata_file_path_from_fnames(tmp_path, "U", "M", fn)
    assert p.name == "Adata-S1-T55.mat"
    assert adata_file_path_from_fnames(tmp_path, "U", "M", ["a/b/S1-T9.wid"], wid_rule=True).name \
        == "Adata-S1-T9.mat"
    # without the wid rule a .wid name is still cut at '_'
    assert adata_file_path_from_fnames(tmp_path, "U", "M", ["S1-T9_x.wid"]).name == "Adata-S1-T9.mat"
    with pytest.raises(ValueError):
        adata_file_path_from_fnames(tmp_path, "U", "M", ["S1-T9.bin"])
    with pytest.raises(ValueError):
        adata_file_path_from_fnames(tmp_path, "U", "M", [])


# --- save_adata ---------------------------------------------------------------------------


def _save_adata_args(tmp_path, **over):
    args = {
        "adata_dir": tmp_path, "rois": [MatStruct(), MatStruct()], "bv": [MatStruct()],
        "np": [MatStruct()], "template": [np.ones((4, 4))], "dx": [np.zeros((10, 1))],
        "dy": [np.zeros((10, 1))], "aux_files": ["S1-T77.lvd"], "fnames": ["S1-T77_ch525.bin"],
        "nbr_frames": np.array([[10.0]]), "mouse_id": "M1", "user_id": "U1",
        "act_map": [np.ones((4, 4))], "template_sec": np.empty((0, 0), object),
        "act_map_sec": np.empty((0, 0), object),
    }
    args.update(over)
    return args


def test_save_adata_writes_matlab_variables(tmp_path):
    path = save_adata(**_save_adata_args(tmp_path))
    assert path == tmp_path / "U1" / "M1" / "Adata-S1-T77.mat"
    i = info(path)
    assert set(i) == {"ROIs", "bv", "np", "template", "dx", "dy", "aux_files", "fnames",
                      "mouse_id", "nbr_frames", "act_map", "template_sec", "act_map_sec"}
    assert "userID" not in i and LOG not in i
    assert i["ROIs"] == ("cell", (1, 2)) and i["template_sec"] == ("cell", (0, 0))
    d = load_mat(path)
    assert d["mouse_id"] == "M1" and d["fnames"][0, 0] == "S1-T77_ch525.bin"


def test_save_adata_if_exists(tmp_path):
    path = save_adata(**_save_adata_args(tmp_path))
    from calliopy.io import update_mat

    update_mat(path, {"extra": 1.0})
    save_adata(**_save_adata_args(tmp_path, mouse_id="M1"), if_exists="append")
    assert "extra" in info(path)
    save_adata(**_save_adata_args(tmp_path))  # default: replaced, as MATLAB effectively does
    assert "extra" not in info(path)
    with pytest.raises(FileExistsError):
        save_adata(**_save_adata_args(tmp_path), if_exists="error")
    with pytest.raises(ValueError):
        save_adata(**_save_adata_args(tmp_path), if_exists="merge")


def test_save_adata_wid_name(tmp_path):
    path = save_adata(**_save_adata_args(tmp_path, fnames=["d\\S1-T88.wid"]))
    assert path.name == "Adata-S1-T88.mat"
