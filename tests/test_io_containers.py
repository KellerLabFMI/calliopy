"""Tests for load_metafile, metafile_info and Adata/load_adata."""

from __future__ import annotations

import numpy as np
import pytest
from calliopy.io import Adata, MatStruct, load_adata, load_metafile, metafile_info
from mat73_writer import cell, struct_array, write_mat73


def _record(i: int, j: int, n_frames: int = 10) -> dict:
    return {
        "act": np.full((3, n_frames), 10.0 * i + j),
        "np": np.zeros((n_frames, 1)),
        "fnames": cell([f"S1-T{100 + j}.bin", f"S1-T{200 + j}.bin"]),
        "nbr_frames": np.array([[4.0, 6.0]]),
    }


def _site(animal: str, rd) -> dict:
    return {"projID": "TST", "animal": animal, "ExpGroup": np.array([[101, 102]], dtype=np.int32),
            "nbr_piezo_layers": 4.0, "rd": rd}


@pytest.fixture
def metafile(tmp_path):
    two_photon = struct_array([[_record(i, j) for j in range(2)] for i in range(4)])  # 4 layers x 2 ExpIDs
    single = _record(0, 0)  # stored by MATLAB as a 1x1 struct
    pm = struct_array([[_site("M1", two_photon), _site("M2", single)]])
    return write_mat73(tmp_path / "TST_1-2_meta.mat", {"proj_meta": pm, "other": 1.0})


def test_metafile_shapes(metafile):
    pm = load_metafile(metafile)
    assert pm.shape == (1, 2)
    assert pm[0, 0].animal == "M1" and pm[0, 0].ExpGroup.dtype == np.int32
    rd = pm[0, 0].rd
    assert rd.shape == (4, 2)
    for i in range(4):
        for j in range(2):
            assert rd[i, j].act.shape == (3, 10) and np.all(rd[i, j].act == 10.0 * i + j)
            assert rd[i, j].np.shape == (10, 1)
            assert list(rd[i, j].fnames.ravel()) == [f"S1-T{100 + j}.bin", f"S1-T{200 + j}.bin"]


def test_single_record_is_still_2d(metafile):
    rd = load_metafile(metafile)[0, 1].rd
    assert rd.shape == (1, 1) and isinstance(rd[0, 0], MatStruct)


def test_single_site_metafile(tmp_path):
    path = write_mat73(tmp_path / "one_meta.mat", {"proj_meta": _site("M1", _record(0, 0))})
    pm = load_metafile(path)
    assert pm.shape == (1, 1) and pm[0, 0].rd.shape == (1, 1)


def test_rd_fields_selection(metafile):
    pm = load_metafile(metafile, rd_fields=["act"])
    assert list(pm[0, 0].rd[3, 1]) == ["act"]
    assert list(pm[0, 1].rd[0, 0]) == ["act"]
    assert pm[0, 0].animal == "M1"  # site fields are always read
    pm = load_metafile(metafile, rd_fields="nbr_frames")
    assert list(pm[0, 0].rd[0, 0]) == ["nbr_frames"]


def test_metafile_info(metafile):
    info = metafile_info(metafile)
    assert [s.animal for s in info] == ["M1", "M2"]
    assert info[0].rd_size == (4, 2) and info[1].rd_size == (1, 1)
    assert info[0].rd_fields == ["act", "np", "fnames", "nbr_frames"]
    assert "rd" not in info[0]


def test_metafile_without_proj_meta(tmp_path):
    path = write_mat73(tmp_path / "x.mat", {"a": 1.0})
    with pytest.raises(KeyError, match="proj_meta"):
        load_metafile(path)


@pytest.fixture
def adata(tmp_path):
    return write_mat73(tmp_path / "GK_ABC_S1-T12345.mat", {
        "first_frame": np.ones((4, 5)), "dx": np.zeros((10, 1)), "act": np.arange(6.0).reshape(2, 3)})


def test_adata_is_lazy(adata):
    ad = load_adata(adata)
    assert isinstance(ad, Adata)
    assert ad.exp_id == 12345 and ad.type == "2p"
    assert set(ad.variables) == {"first_frame", "dx", "act"}
    assert ad.loaded == ()
    assert np.array_equal(ad.act, np.arange(6.0).reshape(2, 3))
    assert ad.loaded == ("act",)
    assert ad["dx"].shape == (10, 1)
    assert ad.info()["first_frame"] == ("double", (4, 5))


def test_adata_load_all_and_errors(adata):
    ad = load_adata(adata).load_all(exclude="act")
    assert set(ad.loaded) == {"first_frame", "dx"}
    with pytest.raises(AttributeError):
        _ = ad.not_a_variable
    with pytest.raises(KeyError):
        _ = ad["not_a_variable"]
    assert "Adata('GK_ABC_S1-T12345.mat'" in repr(ad)


@pytest.mark.parametrize("names, expected", [
    (["is_eeg", "first_frame"], "wf_eeg"),
    (["first_frame_ch525", "first_frame"], "2ch_wf"),
    (["first_frame"], "wf"),
    (["something"], "unknown"),
])
def test_adata_type(tmp_path, names, expected):
    path = write_mat73(tmp_path / "no_expid.mat", {n: 1.0 for n in names})
    ad = Adata(path)
    assert ad.type == expected and ad.exp_id is None


# -- against MATLAB (tests/matlab/ref_containers.m) ------------------------------------------
def _unwrap(a):
    if isinstance(a, np.ndarray) and a.dtype == object and a.size == 1:
        return a.flat[0]
    return a


def test_metafile_matches_matlab(sample_file, matlab_ref):
    path = sample_file("_meta.mat")
    ref = matlab_ref("load_meta")
    pm = load_metafile(path)
    assert pm.size == int(ref.out1)
    for isite in range(pm.size):
        rd = pm.flat[isite].rd
        assert rd.shape == tuple(int(x) for x in np.ravel(ref.out2.flat[isite]))
        expected = np.asarray(ref.out4.flat[isite], dtype=float).reshape(rd.shape)
        got = np.full(rd.shape, np.nan)
        for idx in np.ndindex(rd.shape):
            if "act" in rd[idx]:
                got[idx] = np.nansum(np.asarray(_unwrap(rd[idx].act), dtype=float))
        np.testing.assert_allclose(got, expected, rtol=1e-9, equal_nan=True)
    assert list(pm.flat[0].rd.flat[0]) == list(np.ravel(ref.out3))


def test_adata_matches_matlab(sample_file, matlab_ref):
    path = sample_file("S1-T*.mat")
    ref = matlab_ref("Adata")
    ad = load_adata(path)
    assert ad.type == ref.out1
    assert (ad.exp_id or 0) == int(ref.out2)
    assert set(ad.variables) == set(np.ravel(ref.out3))
