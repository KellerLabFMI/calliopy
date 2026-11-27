"""Tests for the ExpLog queries of calliopy.explog (qexp, eid, removeHTML, read-only DB tables)."""

from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from calliopy.explog import (
    eid,
    expid2adata_path,
    expid2raw_data_status,
    format_qexp_output,
    get_exp_log_metadata_files,
    get_exp_log_pis,
    get_exp_log_projects,
    get_mice_exp_log,
    get_vector_db,
    list_projects_exp_log,
    qexp,
    remove_html,
)
from calliopy.io import MatStruct


def make_exp_log() -> MatStruct:
    """Six stacks: two experiments of site 100 (hans), site 103, a deleted stack, another PI."""
    obj = lambda *v: np.array(v, dtype=object)
    return MatStruct(
        stackid=np.array([100, 101, 102, 103, 104, 105]),
        expid=np.array([100, 100, 102, 103, 104, 105]),
        comment=obj("day1 <b>gratings</b>", "day1 dark", "day2 " + "x" * 80, "ori omission", "fb", "pb"),
        stackdate=obj("05-Jan-2024", "05-Jan-2024", "07-Jan-2024", "10-Jan-2024", "11-Jan-2024", "12-Jan-2024"),
        datadeleted=np.array([0, 0, 0, 0, 0, 1]),
        paradigm=obj("", "", "", "", "", ""),
        siteid=np.array([100, 100, 100, 103, 104, 103]),
        analysiscode=obj("1", "1", "1", "2", "1", "1"),
        rawdatastatus=np.array([3, 3, 2, 1, 0, 0]),
        animalid=obj("AA_240101_001", "AA_240101_001", "AA_240101_001", "AA_240101_002", "AA_240101_003",
                     "AA_240101_002"),
        project=obj("PRJ", "PRJ", "PRJ", "PRJ", "OTH", "PRJ"),
        location=obj("L2/3", "L2/3", "L2/3", "L5", "L2/3", "L5"),
        pi=obj("hans", "hans", "hans", "hans", "greta", "hans"),
        strain=obj("C57", "C57", "C57", "C57", "C57", "C57"),
    )


@pytest.fixture
def el():
    return make_exp_log()


def test_remove_html():
    assert remove_html("a <b>bold</b> <i x='1'>it</i>") == "a bold it"
    assert remove_html(["<p>x</p>", "y"]) == ["x", "y"]


def test_eid(el):
    assert tuple(eid("hans", el)) == ("pi", 12)
    assert eid("HANS", el).field == "pi"  # case-insensitive
    assert eid(103, el).field == "siteid"  # last row first, last column first
    assert eid("101", el).field == "stackid"  # digits are searched as a number
    assert tuple(eid("nobody", el)) == (None, None)


def test_qexp_basic_filters(el):
    assert qexp(100, "comment", exp_log=el).exp_log_matched.tolist() == ["day1 <b>gratings</b>"]
    np.testing.assert_array_equal(qexp("pi", "hans", "expid", exp_log=el).exp_log_matched, [100, 102, 103, 105])
    res = qexp("hans", exp_log=el)  # auto-interpretation through eid
    np.testing.assert_array_equal(res.exp_log_matched.stackid, [100, 101, 102, 103, 105])
    assert res.idx.tolist() == [True, True, True, True, False, True]
    np.testing.assert_array_equal(qexp("regex", "day1|day2", "stackid", exp_log=el).exp_log_matched,
                                  [100, 101, 102])
    np.testing.assert_array_equal(qexp("~regex", "day", "stackid", exp_log=el).exp_log_matched, [103, 104, 105])
    np.testing.assert_array_equal(qexp("animalid", "/002/", "stackid", exp_log=el).exp_log_matched, [103, 105])
    np.testing.assert_array_equal(qexp("animalid", "/~002/", "stackid", exp_log=el).exp_log_matched,
                                  [100, 101, 102, 104])
    np.testing.assert_array_equal(qexp("acode", "2", "stackid", exp_log=el).exp_log_matched, [103])
    np.testing.assert_array_equal(qexp("acode", ["1", "2"], "stackid", exp_log=el).exp_log_matched,
                                  [100, 101, 102, 103, 104, 105])
    np.testing.assert_array_equal(qexp("stack", [101, 104], "expid", exp_log=el).exp_log_matched, [100, 104])


def test_qexp_operators_first_last_and_flags(el):
    np.testing.assert_array_equal(qexp("pi", "greta", "acode", 2, "op", "any", "stackid", exp_log=el)
                                  .exp_log_matched, [103, 104])
    np.testing.assert_array_equal(qexp("pi", "hans", "last", 2, "stackid", exp_log=el).exp_log_matched, [103, 105])
    np.testing.assert_array_equal(qexp("pi", "hans", "first", 2, "stackid", exp_log=el).exp_log_matched,
                                  [100, 101])
    np.testing.assert_array_equal(qexp("isexp", 1, "stackid", exp_log=el).exp_log_matched,
                                  [100, 102, 103, 104, 105])
    np.testing.assert_array_equal(qexp("isexp", 0, "stackid", exp_log=el).exp_log_matched, [101])
    np.testing.assert_array_equal(qexp("issite", 1, "stackid", exp_log=el).exp_log_matched, [100, 103, 104])
    np.testing.assert_array_equal(
        qexp("pi", "hans", "op", lambda m, axis: m.all(axis=axis), "stackid", exp_log=el).exp_log_matched,
        [100, 101, 102, 103, 105])


def test_qexp_dates(el):
    today = dt.date(2024, 1, 12)
    np.testing.assert_array_equal(qexp("newerthan", "10-Jan-2024", "stackid", exp_log=el).exp_log_matched,
                                  [103, 104, 105])
    np.testing.assert_array_equal(qexp("age", 1, "stackid", exp_log=el, today=today).exp_log_matched, [104, 105])
    np.testing.assert_array_equal(qexp("olderthan", 6, "stackid", exp_log=el, today=today).exp_log_matched,
                                  [100, 101])
    np.testing.assert_array_equal(qexp("before", "2024-01-07", "stackid", exp_log=el).exp_log_matched,
                                  [100, 101, 102])
    ages = qexp("pi", "greta", "age", exp_log=el, today=today).exp_log_matched
    np.testing.assert_array_equal(ages, [1.0])


def test_qexp_outputs(el, tmp_path):
    res = qexp("pi", "hans", "table", exp_log=el)
    assert res.exp_log_matched.comment[0] == "day1 gratings"
    assert res.exp_log_matched.comment[2] == "day2 " + "x" * 65 + " ..."
    assert res.exp_log_matched.stackid.tolist() == ["100", "101", "102", "103", "105"]
    assert res.idx.tolist() == [False, False, True, False, False]  # MATLAB reuses idx (kept)
    sub = qexp("pi", "greta", ["stackid", "comment"], exp_log=el).exp_log_matched
    assert list(sub) == ["stackid", "comment"]
    assert qexp("pi", "hans", "animalid", exp_log=el).exp_log_matched.tolist() == ["AA_240101_001",
                                                                                   "AA_240101_002"]
    paths = qexp("expid", 103, "adata", exp_log=el, adata_dir=tmp_path).exp_log_matched
    assert paths == [str(Path(tmp_path, "hans", "AA_240101_002", "Adata-S1-T103.mat"))]
    fn = qexp(102, "fnames", exp_log=el, raw_data_dir=tmp_path).exp_log_matched
    assert fn == [str(Path(tmp_path, "hans", "AA_240101_001", "S1-T102_ch525.bin"))]
    cells = qexp("pi", "greta", "cell", exp_log=el).exp_log_matched
    assert cells.stackid.dtype == object
    seen = []
    assert qexp("pi", "hans", lambda e, k: seen.append((e, k)), exp_log=el).exp_log_matched is None
    assert seen == []  # the callable needs two arguments: each call fails and is logged
    qexp("pi", "hans", [lambda e, k: seen.append((e, k)), "x"], exp_log=el)
    assert seen == [(100, "x"), (102, "x"), (103, "x"), (105, "x")]
    with pytest.raises(ValueError, match="interpret"):
        qexp("pi", "hans", "nonsense", exp_log=el)
    with pytest.raises(ValueError, match="adata_dir"):
        qexp("pi", "hans", "adata", exp_log=el)


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_qexp_add_callable_and_quirks(el):
    def twice(e):
        return 2 * e

    res = qexp("pi", "greta", "add", twice, exp_log=el).exp_log_matched
    assert res.twice.tolist() == [208]
    handler, logger = _Collect(), logging.getLogger("calliopy.explog._query")
    logger.addHandler(handler)
    try:
        res = qexp("mouse", "AA_240101_003", "stackid", exp_log=el)  # 'mouse' -> 'mouseid' (kept)
    finally:
        logger.removeHandler(handler)
    assert any("mouseid" in m for m in handler.messages)
    assert res.exp_log_matched.size == 6
    # 'ExpLog' inside the arguments, as in MATLAB
    np.testing.assert_array_equal(qexp("pi", "greta", "explog", el, "stackid").exp_log_matched, [104])


def test_qexp_project_fallback(el):
    meta = MatStruct(metadataname=np.array(["PRJ2_meta"], dtype=object),
                     projectid=np.array(["PRJ"], dtype=object),
                     description=np.array(["visual mismatch"], dtype=object))
    np.testing.assert_array_equal(qexp("proj", "mismatch", "stackid", exp_log=el, metadata_files=meta)
                                  .exp_log_matched, [103])
    projects = MatStruct(projectid=np.array(["OTH"], dtype=object),
                         description=np.array(["auditory cortex"], dtype=object))
    np.testing.assert_array_equal(qexp("proj", "auditory", "stackid", exp_log=el, metadata_files=meta,
                                       projects=projects).exp_log_matched, [104])
    assert qexp("proj", "auditory", "stackid", exp_log=el).exp_log_matched.size == 0


def test_format_qexp_output(el):
    q = qexp("pi", "hans", exp_log=el).exp_log_matched
    f = format_qexp_output(q)
    assert f.stackid.dtype == np.int64 and f.comment.dtype == object and len(f.stackid) == 5


def make_sqlite():
    con = sqlite3.connect(":memory:")
    con.executescript("""
        CREATE TABLE Experiments (ExpID INTEGER, SiteID INTEGER, RawDataStatus INTEGER);
        CREATE TABLE Sites (SiteID INTEGER, AnimalID TEXT);
        CREATE TABLE Animals (AnimalID TEXT, PI TEXT, Gender TEXT, DoB TEXT, DoD TEXT, Source TEXT,
                              Strain TEXT, VivariumID TEXT, PyratID TEXT);
        CREATE TABLE PIs (PI TEXT, active INTEGER);
        CREATE TABLE Projects (ProjectID TEXT, name TEXT, PIs TEXT, Description TEXT, StartDate TEXT,
                               Status TEXT);
        CREATE TABLE MetaDataFiles (MetaDataName TEXT, ProjectID TEXT, Description TEXT,
                                    RecordingLocation TEXT, StimulationLocation TEXT, Paradigm TEXT,
                                    Marker TEXT);
        INSERT INTO Experiments VALUES (10, 10, 2), (11, 10, 5);
        INSERT INTO Sites VALUES (10, 'M1');
        INSERT INTO Animals VALUES ('M1', 'hans', 'F', '2024-01-01', NULL, 'CR', 'C57', 'v', 'p'),
                                   ('M2', 'hans', 'M', '2024-01-01', '2024-06-01', 'CR', 'C57', 'v', 'p'),
                                   ('H1', 'hans', 'F', NULL, NULL, 'x', 'human', 'v', 'p'),
                                   ('M3', 'greta', 'F', NULL, NULL, 'x', 'C57', 'v', 'p');
        INSERT INTO PIs VALUES ('hans', 1), ('greta', 0);
        INSERT INTO Projects VALUES ('PRJ', 'Project', 'hans,greta', 'mismatch', '2020', 'active'),
                                    ('OTH', 'Other', 'greta', 'audio', '2021', 'done');
        INSERT INTO MetaDataFiles VALUES ('PRJ1', 'PRJ', 'd', 'V1', '', '', ''),
                                         ('OTH3', 'OTH', 'e', 'A1', '', '', '');
    """)
    return con


def test_expid_lookups(el, tmp_path):
    res = expid2adata_path(103, tmp_path, exp_log=el)
    assert res.full_path == tmp_path / "hans" / "AA_240101_002" / "Adata-S1-T103.mat"
    assert (res.user_id, res.mouse_id) == ("hans", "AA_240101_002")
    with pytest.raises(LookupError):
        expid2adata_path(999, tmp_path, exp_log=el)
    with pytest.raises(ValueError):
        expid2adata_path(0, tmp_path, exp_log=el)
    assert expid2raw_data_status(102, exp_log=el) == 2
    assert expid2raw_data_status(999, exp_log=el) is None
    con = make_sqlite()
    assert tuple(expid2adata_path(11, tmp_path, connection=con))[1:] == ("hans", "M1")
    assert expid2raw_data_status(11, connection=con) == 5
    assert expid2raw_data_status(12, connection=con) is None


def test_database_tables():
    con = make_sqlite()
    assert get_exp_log_pis(con).tolist() == ["hans"]
    assert get_exp_log_pis(con, "all").tolist() == ["hans", "greta"]
    projects, info = get_exp_log_projects(con, project="OT")
    assert projects.projectid.tolist() == ["PRJ", "OTH"] and info.name == "Other" and info.projectID == "OT"
    assert get_exp_log_projects(con, uname="greta").exp_log_projects.tolist() == ["PRJ", "OTH"]
    with pytest.raises(LookupError):
        get_exp_log_projects(con, project="ZZZ")
    assert get_exp_log_metadata_files(con).metadataname.tolist() == ["PRJ1", "OTH3"]
    assert get_mice_exp_log(con, uname="hans").animalid.tolist() == ["M1"]
    assert len(get_mice_exp_log(con).animalid) == 4
    rows = list_projects_exp_log(projects, "aud")
    assert rows.projectid.tolist() == ["OTH"]


class _Cursor:
    def __init__(self, tables):
        self.tables, self.description, self._rows = tables, None, []

    def execute(self, sql):
        key = "AliquotRequest" if "AliquotRequest" in sql else "Vector"
        names, self._rows = self.tables[key]
        self.description = [(n,) for n in names]

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class _Connection:
    def __init__(self, tables):
        self.tables = tables

    def cursor(self):
        return _Cursor(self.tables)


def test_get_vector_db_post_processing():
    tables = {"Vector": (["ID", "Sourcelab", "Comment"], [(1, "lab", "a\r\nb\t"), (2, 7, "c")]),
              "AliquotRequest": (["VectorID", "Email"], [(1, "a@b")])}
    vdb, req = get_vector_db(_Connection(tables))
    assert vdb.sourcelab.tolist() == ["lab", "7"]  # mixed text/number column -> text
    assert vdb.comment.tolist() == ["ab", "c"]
    assert req.email.tolist() == ["a@b"]
