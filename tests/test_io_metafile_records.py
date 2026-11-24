"""Tests for load_metafile_records (reading selected records of a metafile)."""

from __future__ import annotations

import numpy as np
import pytest
from calliopy.io import load_metafile, load_metafile_records
from mat73_writer import struct_array, write_mat73


def _record(i: int, j: int) -> dict:
    return {"act": np.full((3, 10), 10.0 * i + j), "drug": f"drug {i} {j}", "nbr_frames": 10.0}


def _site(animal: str, rd) -> dict:
    return {"animal": animal, "rd": rd}


@pytest.fixture
def metafile(tmp_path):
    # 4 layers x 3 ExpIDs
    two_photon = struct_array([[_record(i, j) for j in range(3)] for i in range(4)])
    single = _record(0, 0)  # stored by MATLAB as a 1x1 struct
    pm = struct_array([[_site("M1", two_photon), _site("M2", single)]])
    return write_mat73(tmp_path / "TST_1-2_meta.mat", {"proj_meta": pm})


def test_records_in_requested_order(metafile):
    recs = load_metafile_records(metafile, [(0, 3, 2), (1, 0, 0), (0, 0, 1)])
    assert [float(r.act[0, 0]) for r in recs] == [32.0, 0.0, 1.0]
    assert recs[0].act.shape == (3, 10)
    assert recs[0].drug == "drug 3 2"


def test_same_values_as_load_metafile(metafile):
    pm = load_metafile(metafile)
    for isite in range(pm.shape[1]):
        rd = pm[0, isite].rd
        for idx in np.ndindex(rd.shape):
            (rec,) = load_metafile_records(metafile, [(isite, *idx)])
            assert list(rec) == list(rd[idx])
            np.testing.assert_array_equal(rec.act, rd[idx].act)
            assert rec.drug == rd[idx].drug


def test_field_selection(metafile):
    (rec,) = load_metafile_records(metafile, [(0, 1, 1)], rd_fields="act")
    assert list(rec) == ["act"]
    (rec,) = load_metafile_records(metafile, [(1, 0, 0)], rd_fields=["drug", "act"])
    assert list(rec) == ["act", "drug"]  # file order
    with pytest.raises(KeyError, match="no field"):
        load_metafile_records(metafile, [(0, 0, 0)], rd_fields=["not_a_field"])


def test_out_of_range(metafile):
    with pytest.raises(IndexError, match="outside rd"):
        load_metafile_records(metafile, [(0, 4, 0)])
    with pytest.raises(IndexError, match="outside rd"):
        load_metafile_records(metafile, [(1, 0, 1)])
    with pytest.raises(IndexError, match="sites"):
        load_metafile_records(metafile, [(2, 0, 0)])


def test_single_site_metafile(tmp_path):
    path = write_mat73(tmp_path / "one_meta.mat", {"proj_meta": _site("M1", _record(0, 0))})
    (rec,) = load_metafile_records(path, [(0, 0, 0)], rd_fields=["act"])
    assert rec.act.shape == (3, 10)


def test_no_proj_meta(tmp_path):
    path = write_mat73(tmp_path / "x.mat", {"a": 1.0})
    with pytest.raises(KeyError, match="proj_meta"):
        load_metafile_records(path, [(0, 0, 0)])
