"""Tests for the file listings and index parsers of calliopy.data."""

from __future__ import annotations

import datetime as dt
import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from calliopy.data import (
    get_adata_index,
    get_data_path_all,
    get_files,
    get_stackid,
    get_tape_index,
    list_all_adata_files,
    list_file_types,
)
from test_explog_more import make_exp_log


def test_get_stackid():
    np.testing.assert_array_equal(get_stackid(["S1-T1234.lvd", "nothing", "s1-p77_ch525.bin"]), [1234, np.nan, 77])
    assert get_stackid("Adata-S1-T55.mat") == 55.0
    assert get_stackid("S1-T0123.lvd") == 83.0  # sscanf('%i') reads a leading zero as octal (kept)
    assert get_stackid("S1-T089.lvd") == 89.0
    assert get_stackid([["S1-T1", "S1-T2"]]).shape == (1, 2)


def _touch(p: Path, size=0, mtime=None):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)
    if mtime is not None:
        ts = mtime.timestamp()
        os.utime(p, (ts, ts))


def test_get_files(tmp_path):
    d0 = dt.datetime(2024, 1, 1, 12)
    _touch(tmp_path / "hans" / "AA_240101_001" / "S1-T10.lvd", 10, d0)
    _touch(tmp_path / "hans" / "AA_240101_001" / "S1-T11.lvd", 20, d0 + dt.timedelta(days=1))
    _touch(tmp_path / "hans" / "AA_240101_001" / "S1-T11_ch525.bin", 30, d0 + dt.timedelta(days=2))
    _touch(tmp_path / "other" / "notes.txt", 5, d0 + dt.timedelta(days=3))
    res = get_files(tmp_path)
    assert sorted(res.names) == ["S1-T10.lvd", "S1-T11.lvd", "S1-T11_ch525.bin", "notes.txt"]
    assert get_files(tmp_path, ext=".lvd", just="names").file_info == ["S1-T10.lvd", "S1-T11.lvd"]
    assert get_files(tmp_path, recurse=False).names == []
    assert get_files(tmp_path, stackid=11, just="nofiles").file_info == 2
    assert get_files(tmp_path, strict=True, just="nofiles").file_info == 3
    assert get_files(tmp_path, regex="notes", just="names").file_info == ["notes.txt"]
    assert get_files(tmp_path, gthan=15, lthan=25, just="names").file_info == ["S1-T11.lvd"]
    assert get_files(tmp_path, newest=1, just="names").file_info == ["notes.txt"]
    assert get_files(tmp_path, oldest=1, just="names").file_info == ["S1-T10.lvd"]
    assert get_files(tmp_path, newest=10, just="nofiles").file_info == 4
    assert get_files(tmp_path, dayeq="02-Jan-2024", just="names").file_info == ["S1-T11.lvd"]
    assert sorted(get_files(tmp_path, newerthan="02-Jan-2024", olderthan="04-Jan-2024", just="names").file_info) \
        == ["S1-T11.lvd", "S1-T11_ch525.bin"]
    outside = get_files(tmp_path, newerthan="03-Jan-2024", olderthan="02-Jan-2024", just="names").file_info
    assert sorted(outside) == ["S1-T10.lvd", "S1-T11_ch525.bin", "notes.txt"]
    assert get_files(tmp_path, ext=[".bin", ".txt"], just="nofiles").file_info == 2
    res = get_files(tmp_path, ext=".lvd")
    np.testing.assert_array_equal(res.stackids, [10, 11])
    assert res.animals == ["AA_240101_001", "AA_240101_001"]
    np.testing.assert_array_equal(res.bytes, [10, 20])
    assert res.datestrs[0] == "01-Jan-2024 12:00:00"
    assert get_files(tmp_path, ext=".lvd", just="relpaths").file_info == ["." + os.sep + "S1-T10.lvd",
                                                                          "." + os.sep + "S1-T11.lvd"]
    folders = get_files(tmp_path, folders=True, recurse=False)
    assert sorted(folders.names) == ["hans", "other"]
    assert dict(zip(folders.names, folders.bytes, strict=True))["hans"] == 60
    multi = get_files([tmp_path / "hans", tmp_path / "other", tmp_path / "missing"], flatten=True)
    assert len(multi.names) == 4
    per = get_files([tmp_path / "hans", tmp_path / "other"])
    assert [len(n) for n in per.names] == [3, 1]


def test_index_files(tmp_path):
    idx = tmp_path / "index.txt"
    idx.write_text("./hans/AA_1_1/Adata-S1-T10.mat\t05-Jan-2024 10:00:00\t1024\n"
                   "./greta/AA_1_2/Adata-S1-T11.mat\t06-Jan-2024 10:00:00\t2048\n"
                   "header line without size\n")
    res = get_adata_index(idx)
    assert res.file_info is None
    assert res.fpath == [str(Path(tmp_path, "hans", "AA_1_1", "Adata-S1-T10.mat")),
                         str(Path(tmp_path, "greta", "AA_1_2", "Adata-S1-T11.mat"))]
    assert res.names == ["Adata-S1-T10.mat", "Adata-S1-T11.mat"]
    np.testing.assert_array_equal(res.bytes, [1024, 2048])
    np.testing.assert_array_equal(res.stackids, [10, 11])
    sub = get_adata_index(idx, tmp_path / "root", pi="GRETA")
    assert sub.fpath == [str(Path(tmp_path, "root", "greta", "AA_1_2", "Adata-S1-T11.mat"))]
    assert sub.dstr == ["06-Jan-2024 10:00:00"]

    tape = tmp_path / "tape.txt"
    tape.write_text("./hans/AA_1_1/S1-T10.lvd\t2024-03-05 10:11:12\t10\n./greta/AA_1_2/S1-T11.lvd\t"
                    "2024-11-01 00:00:00\t20\n")
    t = get_tape_index(tape, tmp_path / "arc")
    assert t.fpath[0] == str(Path(tmp_path, "arc", "hans", "AA_1_1", "S1-T10.lvd"))
    assert t.dates[0] == dt.datetime(2024, 3, 5, 10, 11, 12)
    assert t.animals == ["AA_1_1", "AA_1_2"]
    assert t.dstr == ["05-Mar-2024 10:11:12", "01-Nov-2024 00:00:00"]
    np.testing.assert_array_equal(t.stackids, [10, 11])
    assert get_tape_index(tape, tmp_path, pi="greta").names == ["S1-T11.lvd"]
    assert get_tape_index(tape, tmp_path, pi="GRETA").names == []  # case-sensitive as MATLAB regexp


def test_list_all_adata_files(tmp_path):
    _touch(tmp_path / "hans" / "M1" / "Adata-S1-T1234.mat")
    _touch(tmp_path / "hans" / "M1" / "other.mat")
    _touch(tmp_path / "greta" / "M2" / "Adata-S1-T99.mat")
    _touch(tmp_path / "_metaData" / "M3" / "Adata-S1-T7.mat")
    assert list_all_adata_files(tmp_path) == ["99", "1234"]


def test_list_file_types_and_data_path(tmp_path):
    el = make_exp_log()
    folder = tmp_path / "raw2" / "hans" / "AA_240101_001"
    for n in ("S1-T100.lvd", "S1-T100.ach", "S1-T101_ch525.bin", "S1-T100_ch525.ini", "S1-T100owl.avi",
              "S1-T200.lvd"):
        _touch(folder / n)
    assert list_file_types(100, tmp_path / "raw2", el) == [".ach", ".lvd", "525.ini", "owl.avi"]
    with pytest.raises(LookupError):
        list_file_types(999, tmp_path, el)
    roots = [tmp_path / "raw1", tmp_path / "raw2"]
    res = get_data_path_all(100, roots, el)
    assert res.data_path == tmp_path / "raw2" and res.exp_id_info.userID == "hans"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = get_data_path_all(101, roots, el)  # only the .bin exists
    assert res.data_path == tmp_path / "raw2" and any("lvd" in str(x.message) for x in w)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert get_data_path_all(102, roots, el).data_path is None
        assert get_data_path_all(999, roots, el) == (None, None)
