"""Tests of calliopy.io.load_meta (keyword search over metafiles) on synthetic files."""

from __future__ import annotations

import datetime
import os
from pathlib import Path

import numpy as np
import pytest

from calliopy.io import MatStruct
from calliopy.io._meta_search import LoadMetaResult, MetafileMatchError, _datestr, load_meta
from calliopy.io._save import save_mat

# file name -> (modification date, animal); None as animal: file without proj_meta
FILES = {
    "ECT_layer23_SST_meta.mat": (datetime.datetime(2019, 5, 3, 12, 0, 0), "M1"),
    "XYZ_wf_meta.mat": (datetime.datetime(2020, 1, 2, 12, 0, 0), "M4"),
    "ECT_layer5_PV_meta.mat": (datetime.datetime(2021, 3, 5, 14, 3, 12), "M2"),
    "ABC_2p_SST_meta.mat": (datetime.datetime(2021, 7, 1, 12, 0, 0), "M3"),
    "ECT_noproj_meta.mat": (datetime.datetime(2018, 1, 1, 12, 0, 0), None),
}
DATE_ORDER = ["ECT_noproj_meta.mat", "ECT_layer23_SST_meta.mat", "XYZ_wf_meta.mat",
              "ECT_layer5_PV_meta.mat", "ABC_2p_SST_meta.mat"]


def column(values, dtype=object):
    return np.array(values, dtype=dtype)


EXP_LOG = MatStruct(
    metadatafiles=MatStruct(
        metadataname=column(["ECT_layer23_SST_meta.mat", "ECT_layer5_PV_meta.mat", "ABC_2p_SST_meta.mat"]),
        projectid=column(["ECT", "ECT", "ABC"]),
        description=column(["Layer 2/3   SST\tcells", "ab", "two-photon SST like ECT"]),
    ),
    projects=MatStruct(
        projectid=column(["ECT", "ABC", "XYZ"]),
        pis=column(["kellgeor, jdoe", "asmith", None]),
        description=column(["Electroconvulsive therapy", "Alpha project", "Widefield proj"]),
    ),
)


@pytest.fixture
def meta_dir(tmp_path):
    folder = tmp_path / "metastore"
    folder.mkdir()
    for size, (name, (when, animal)) in enumerate(FILES.items()):
        path = folder / name
        if animal is None:
            save_mat(path, {"other": 1.0})
        else:
            rd = np.empty((1, 2), dtype=object)
            rd[0, 0] = MatStruct(act=np.ones((2, 3 + size)), drug="saline")
            rd[0, 1] = MatStruct(act=np.zeros((2, 3 + size)), drug="ket")
            pm = np.empty((1, 1), dtype=object)
            pm[0, 0] = MatStruct(animal=animal, rd=rd, pad=np.zeros((1, 100 * (size + 1))))
            save_mat(path, {"proj_meta": pm})
        t = when.timestamp()
        os.utime(path, (t, t))
    (folder / "notes.txt").write_text("not a metafile")
    return folder


def candidates(excinfo):
    return excinfo.value.candidates


def test_unique_keyword_loads(meta_dir):
    res = load_meta("layer23", meta_dir=meta_dir)
    assert isinstance(res, LoadMetaResult)
    assert res.filtered_file == "ECT_layer23_SST_meta.mat"
    assert res.proj_meta.shape == (1, 1) and res.proj_meta[0, 0].animal == "M1"
    assert res.proj_meta[0, 0].rd.shape == (1, 2)


def test_words_are_anded_and_case_insensitive(meta_dir):
    assert load_meta("sst LAYER", meta_dir=meta_dir).filtered_file == "ECT_layer23_SST_meta.mat"
    assert load_meta(["SST", "2p"], meta_dir=meta_dir).filtered_file == "ABC_2p_SST_meta.mat"


def test_ambiguous_raises_with_candidates_in_date_order(meta_dir):
    with pytest.raises(MetafileMatchError) as ei:
        load_meta("SST", meta_dir=meta_dir)
    assert candidates(ei) == ["ECT_layer23_SST_meta.mat", "ABC_2p_SST_meta.mat"]
    assert "ABC_2p_SST_meta" in ei.value.table and "2 metafiles match" in str(ei.value)


def test_no_match_lists_all_files(meta_dir):
    with pytest.raises(MetafileMatchError) as ei:
        load_meta("nosuchword", meta_dir=meta_dir)
    assert candidates(ei) == DATE_ORDER  # MATLAB reverts to the full table; notes.txt ignored


def test_tilde_excludes(meta_dir):
    # keywords must not occur in the test folder path, which is searched as well
    assert load_meta("SST ~ABC_2p", meta_dir=meta_dir).filtered_file == "ECT_layer23_SST_meta.mat"
    assert load_meta("SST ~lay~er", meta_dir=meta_dir).filtered_file == "ABC_2p_SST_meta.mat"


def test_exclamation_and_minus_are_not_exclusions(meta_dir):
    # as in the MATLAB code (not its help text), only ~ excludes
    with pytest.raises(MetafileMatchError) as ei:
        load_meta("SST !ABC_2p", meta_dir=meta_dir)
    assert candidates(ei) == DATE_ORDER


def test_regex_string_is_one_keyword(meta_dir):
    with pytest.raises(MetafileMatchError) as ei:
        load_meta("layer23|XYZ_wf", meta_dir=meta_dir)
    assert candidates(ei) == ["ECT_layer23_SST_meta.mat", "XYZ_wf_meta.mat"]
    # with |, the space is part of the regex: 'PV_meta|nomatch x' is not split
    assert load_meta("nomatch x|layer5_PV", meta_dir=meta_dir).filtered_file == "ECT_layer5_PV_meta.mat"


def test_dates_are_searched(meta_dir):
    with pytest.raises(MetafileMatchError) as ei:
        load_meta("2021", meta_dir=meta_dir)
    assert candidates(ei) == ["ECT_layer5_PV_meta.mat", "ABC_2p_SST_meta.mat"]
    assert load_meta("Jul. 2021", meta_dir=meta_dir).filtered_file == "ABC_2p_SST_meta.mat"


def test_datestr_format():
    t = datetime.datetime(2021, 3, 5, 14, 3, 12).timestamp()
    assert _datestr(t) == "05. Mar. 2021 14:03:12"


def test_direct_file_name_and_path(meta_dir):
    res = load_meta("ect_layer5_pv_meta.MAT", meta_dir=meta_dir)
    assert res.filtered_file == "ECT_layer5_PV_meta.mat" and res.proj_meta[0, 0].animal == "M2"
    res = load_meta(meta_dir / "ABC_2p_SST_meta.mat", meta_dir=meta_dir)
    assert res.filtered_file == "ABC_2p_SST_meta.mat"
    res = load_meta(str(meta_dir / "XYZ_wf_meta.mat"), meta_dir=meta_dir)
    assert res.proj_meta[0, 0].animal == "M4"


def test_file_without_proj_meta_returns_none(meta_dir):
    res = load_meta("noproj", meta_dir=meta_dir)
    assert res.proj_meta is None and res.filtered_file == "ECT_noproj_meta.mat"


def test_rd_fields_passed_on(meta_dir):
    res = load_meta("layer23", meta_dir=meta_dir, rd_fields="drug")
    assert list(res.proj_meta[0, 0].rd[0, 1]) == ["drug"]


def test_sortby(meta_dir):
    def order(**kw):
        with pytest.raises(MetafileMatchError) as ei:
            load_meta("meta", meta_dir=meta_dir, **kw)
        return candidates(ei)

    assert order() == DATE_ORDER
    assert order(sortby="~date") == DATE_ORDER[::-1]
    assert order(sortby="filename") == sorted(FILES)
    assert order(sortby="!file") == sorted(FILES)[::-1]
    sizes = {p.name: p.stat().st_size for p in meta_dir.glob("*.mat")}
    assert order(sortby="size") == sorted(DATE_ORDER, key=lambda n: sizes[n])
    assert order(sortby="proj", exp_log=EXP_LOG)[:1] == ["ABC_2p_SST_meta.mat"]
    with pytest.raises(ValueError):
        load_meta("meta", meta_dir=meta_dir, sortby="colour")


def test_sortby_typed_among_keywords(meta_dir):
    with pytest.raises(MetafileMatchError) as ei:
        load_meta("SST sortby ~file", meta_dir=meta_dir)
    assert candidates(ei) == ["ECT_layer23_SST_meta.mat", "ABC_2p_SST_meta.mat"]
    with pytest.raises(MetafileMatchError) as ei:
        load_meta("SST sortby file", meta_dir=meta_dir)
    assert candidates(ei) == ["ABC_2p_SST_meta.mat", "ECT_layer23_SST_meta.mat"]


def test_exp_log_fills_descriptions_and_pis(meta_dir):
    # 'therapy' is only in the project description, filled in for short/missing descriptions
    with pytest.raises(MetafileMatchError) as ei:
        load_meta("therapy", meta_dir=meta_dir, exp_log=EXP_LOG)
    assert candidates(ei) == ["ECT_noproj_meta.mat", "ECT_layer5_PV_meta.mat"]
    assert "[Electroconvulsive therapy]" in ei.value.table
    # nocomplete: ExpLog descriptions only
    with pytest.raises(MetafileMatchError) as ei:
        load_meta("therapy", meta_dir=meta_dir, exp_log=EXP_LOG, complete=False)
    assert candidates(ei) == DATE_ORDER
    with pytest.raises(MetafileMatchError) as ei:
        load_meta("therapy nocomplete", meta_dir=meta_dir, exp_log=EXP_LOG)
    assert candidates(ei) == DATE_ORDER
    # PIs come from the Projects table
    assert load_meta("asmith", meta_dir=meta_dir, exp_log=EXP_LOG).filtered_file == "ABC_2p_SST_meta.mat"
    # tab and repeated spaces in ExpLog descriptions are normalised to single spaces
    res = load_meta("'2/3 SST cells'", meta_dir=meta_dir, exp_log=EXP_LOG)
    assert res.filtered_file == "ECT_layer23_SST_meta.mat"


def test_without_exp_log_project_comes_from_file_name(meta_dir):
    with pytest.raises(MetafileMatchError) as ei:
        load_meta("meta", meta_dir=meta_dir, proj="ect")
    assert candidates(ei) == ["ECT_noproj_meta.mat", "ECT_layer23_SST_meta.mat", "ECT_layer5_PV_meta.mat"]
    with pytest.raises(MetafileMatchError):
        load_meta("asmith", meta_dir=meta_dir)


def test_desc_pi_proj_filters(meta_dir):
    kw = {"meta_dir": meta_dir, "exp_log": EXP_LOG}
    assert load_meta("SST", pi="JDOE", **kw).filtered_file == "ECT_layer23_SST_meta.mat"
    assert load_meta("SST", proj="abc", **kw).filtered_file == "ABC_2p_SST_meta.mat"
    assert load_meta(None, desc="two-photon", **kw).filtered_file == "ABC_2p_SST_meta.mat"
    assert load_meta("", desc="widefield", **kw).filtered_file == "XYZ_wf_meta.mat"


def test_exact_project_word_narrows_unless_it_is_the_last_word(meta_dir):
    kw = {"meta_dir": meta_dir, "exp_log": EXP_LOG}
    # 'SST' and 'ECT' match layer23 and the ABC file (its description mentions ECT)
    assert load_meta("ECT SST", **kw).filtered_file == "ECT_layer23_SST_meta.mat"
    # MATLAB would ask "Did you mean proj = 'ECT'?"; without a prompt the answer is no
    with pytest.raises(MetafileMatchError) as ei:
        load_meta("SST ECT", **kw)
    assert candidates(ei) == ["ECT_layer23_SST_meta.mat", "ABC_2p_SST_meta.mat"]
    assert "proj='ECT'" in str(ei.value)
    assert load_meta("SST ECT", proj="ECT", **kw).filtered_file == "ECT_layer23_SST_meta.mat"


def test_path_is_searched_too(meta_dir):
    # MATLAB also matches the full file path: the folder name matches every file
    with pytest.raises(MetafileMatchError) as ei:
        load_meta("metastore", meta_dir=meta_dir)
    assert candidates(ei) == DATE_ORDER


def test_everything_matches_without_keywords(meta_dir, tmp_path):
    with pytest.raises(MetafileMatchError) as ei:
        load_meta(None, meta_dir=meta_dir)
    assert candidates(ei) == DATE_ORDER
    single = tmp_path / "single"
    single.mkdir()
    save_mat(single / "ONE_meta.mat", {"proj_meta": MatStruct(animal="solo")})
    res = load_meta("", meta_dir=single)
    assert res.filtered_file == "ONE_meta.mat" and res.proj_meta[0, 0].animal == "solo"


def test_errors(meta_dir, tmp_path):
    with pytest.raises(NotADirectoryError):
        load_meta("x", meta_dir=tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(MetafileMatchError):
        load_meta("x", meta_dir=empty)
    with pytest.raises(ValueError):
        load_meta("x", meta_dir=meta_dir, exp_log={"experiments": {}})
    with pytest.raises(KeyError):
        load_meta("x", meta_dir=meta_dir, exp_log={"projects": {"projectid": column(["A"])}})
    assert issubclass(MetafileMatchError, LookupError)
    assert Path(meta_dir).is_dir()
