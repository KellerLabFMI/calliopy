"""Tests for calliopy.explog (getExpLog.m, read_info_from_ExpLog.m) on an in-memory database."""

from __future__ import annotations

import logging
import sqlite3

import numpy as np
import pytest
from calliopy.explog import (
    as_exp_log,
    clear_exp_log_cache,
    exp_log_rows,
    get_exp_log,
    read_info_from_exp_log,
)
from calliopy.io import MatStruct, load_mat, save_mat


def make_db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.executescript(
        """
        CREATE TABLE Animals (AnimalID TEXT, pi TEXT, strain TEXT, gender TEXT, DoB TEXT, DoD TEXT,
                              VivariumID INTEGER);
        CREATE TABLE Sites (siteid INTEGER, AnimalID TEXT, project TEXT, location TEXT,
                            depth REAL);
        CREATE TABLE Experiments (expid INTEGER, siteid INTEGER, analysiscode TEXT, RawDataStatus INTEGER);
        CREATE TABLE Stacks (stackid INTEGER, expid INTEGER, comment TEXT, stackdate TEXT,
                             DataDeleted INTEGER, StackTime TEXT, Paradigm TEXT);
        INSERT INTO Animals VALUES ('M1', 'userA', 'C57', 'f', '2025-01-01', NULL, 7);
        INSERT INTO Animals VALUES ('M2', 'userB', 'C57', 'm', '2025-02-01', NULL, 8);
        INSERT INTO Sites VALUES (10, 'M1', 'ECT', 'V1', 100.0);
        INSERT INTO Sites VALUES (11, 'M2', 'OTH', 'S1', 200.0);
        INSERT INTO Experiments VALUES (500, 10, NULL, 1);
        INSERT INTO Experiments VALUES (501, 10, 'A1', 1);
        INSERT INTO Experiments VALUES (600, 11, 'A2', 0);
        INSERT INTO Stacks VALUES (500, 500, 'first', '2025-03-01', 0, '10:00', 'p');
        INSERT INTO Stacks VALUES (502, 501, NULL, '2025-03-01', 0, '11:00', 'p');
        INSERT INTO Stacks VALUES (503, 501, 'second', '2025-03-01', 0, '11:05', 'p');
        INSERT INTO Stacks VALUES (600, 600, 'x', '2025-03-02', 1, '12:00', 'q');
        """
    )
    return con


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_exp_log_cache()
    yield
    clear_exp_log_cache()


def test_get_exp_log_query_and_layout():
    el = get_exp_log(connection=make_db())
    assert list(el)[:3] == ["stackid", "expid", "comment"]
    assert el.stackid.dtype == np.int64 and el.stackid.tolist() == [500, 502, 503, 600]
    assert el.comment.dtype == object and el.comment.tolist() == ["first", "", "second", "x"]
    assert el.analysiscode.tolist() == ["", "A1", "A1", "A2"]
    assert el.pi.tolist() == ["userA", "userA", "userA", "userB"]
    assert "dob" not in el


def test_get_exp_log_cache_and_extended():
    con = make_db()
    el = get_exp_log(connection=con)
    assert get_exp_log(suppress_warning=True) is el
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger("calliopy.explog._explog")
    logger.addHandler(handler)
    try:
        get_exp_log()
    finally:
        logger.removeHandler(handler)
    assert any("cache" in r.getMessage() for r in records)
    with pytest.raises(LookupError, match="extended"):
        get_exp_log(extended=True)
    ex = get_exp_log(connection=con, extended=True)
    assert "dob" in ex and "vivariumid" in ex and ex.depth.dtype == np.float64
    assert get_exp_log(extended=True, suppress_warning=True) is ex
    clear_exp_log_cache()
    with pytest.raises(LookupError, match="no ExpLog cached"):
        get_exp_log()


def test_offline_file_and_fallback(tmp_path):
    el = get_exp_log(connection=make_db())
    path = tmp_path / "ExpLog.mat"
    cells = MatStruct()
    for k, v in el.items():  # MATLAB stores the columns as n x 1 cells
        col = np.empty((v.size, 1), dtype=object)
        col[:, 0] = list(v)
        cells[k] = col
    save_mat(path, {"ExpLog": cells})
    clear_exp_log_cache()
    off = get_exp_log(explog_file=path)
    assert off.stackid.dtype == np.int64 and off.stackid.tolist() == el.stackid.tolist()
    assert off.comment.tolist() == el.comment.tolist()

    class Broken:
        def cursor(self):
            raise RuntimeError("no network")

    with pytest.warns(UserWarning, match="Couldn't query"):
        fb = get_exp_log(connection=Broken(), explog_file=path)
    assert fb.expid.tolist() == el.expid.tolist()
    with pytest.raises(RuntimeError):
        get_exp_log(connection=Broken())
    assert as_exp_log(load_mat(path)["ExpLog"]).pi.tolist() == el.pi.tolist()


def test_read_info_without_and_with_data_dir(tmp_path):
    el = get_exp_log(connection=make_db())
    info, ddir = read_info_from_exp_log(501, False, el)
    assert ddir is None
    assert info.proj == "ECT" and info.userID == "userA" and info.mouse_id == "M1"
    assert info.site_id == 10 and info.expIDs_in_siteID.tolist() == [500, 501]
    assert "fnames" not in info

    folder = tmp_path / "userA" / "M1"
    folder.mkdir(parents=True)
    for sid in (502, 503):
        (folder / f"S1-T{sid}_ch525.bin").write_bytes(b"")
    (folder / "S1-T502.lvd").write_bytes(b"")
    info, ddir = read_info_from_exp_log(501, False, el, data_dir=tmp_path)
    assert ddir == tmp_path
    assert [p.split("S1-T")[1] for p in info.fnames] == ["502_ch525.bin", "503_ch525.bin"]
    assert [p.endswith(".lvd") for p in info.aux_files] == [True, True]  # kept from MATLAB
    assert info.sec_fnames == []
    with pytest.warns(UserWarning, match="failed to find any data file"):
        read_info_from_exp_log(600, False, el, data_dir=tmp_path)
    with pytest.warns(UserWarning, match="No matching"):
        assert read_info_from_exp_log(999, False, el).exp_id_info == {}


def test_exp_log_rows():
    el = get_exp_log(connection=make_db())
    assert exp_log_rows(el, 501).tolist() == [1, 2]
