"""Tests for calliopy.io.load_mat / MatFile on files with the MATLAB v7.3 layout."""

from __future__ import annotations

import numpy as np
import pytest
from calliopy.io import MatFile, MatlabObject, MatStruct, load_mat
from mat73_writer import Empty, Obj, cell, struct_array, write_mat73
from scipy import sparse


@pytest.fixture
def mat(tmp_path):
    """mat(**variables) -> path of a v7.3-layout file holding those variables."""

    def make(**variables):
        return write_mat73(tmp_path / "test.mat", variables)

    return make


def test_numeric_keeps_matlab_shape(mat):
    a = np.arange(6.0).reshape(2, 3)
    b = np.arange(24.0).reshape(2, 3, 4)
    d = load_mat(mat(a=a, b=b))
    assert d.a.shape == (2, 3) and np.array_equal(d.a, a)
    assert d.b.shape == (2, 3, 4) and np.array_equal(d.b, b)


def test_row_and_column_vectors_stay_2d(mat):
    d = load_mat(mat(row=np.ones((1, 5)), col=np.ones((5, 1))))
    assert d.row.shape == (1, 5)
    assert d.col.shape == (5, 1)


def test_scalars_squeezed_by_default(mat):
    path = mat(x=3.5, n=np.array([[7]], dtype=np.int16))
    d = load_mat(path)
    assert d.x == 3.5 and isinstance(d.x, np.float64)
    assert d.n == 7 and d.n.dtype == np.int16
    d = load_mat(path, squeeze_scalars=False)
    assert d.x.shape == (1, 1)


def test_dtypes_preserved(mat):
    d = load_mat(mat(i=np.array([[1, -2, 3]], dtype=np.int16), s=np.array([[1.5, 2.5]], dtype=np.float32)))
    assert d.i.dtype == np.int16 and d.s.dtype == np.float32


def test_logical(mat):
    m = np.array([[True, False], [False, True]])
    d = load_mat(mat(m=m, t=True))
    assert d.m.dtype == bool and np.array_equal(d.m, m)
    assert d.t is np.True_ or d.t == True


def test_complex(mat):
    z = np.array([[1 + 2j, 3 - 4j]])
    d = load_mat(mat(z=z))
    assert np.array_equal(d.z, z)


def test_char(mat):
    d = load_mat(mat(s="hello", m=np.array(["ab", "cd"]), u="\u00b5m"))
    assert d.s == "hello"
    assert list(d.m) == ["ab", "cd"]
    assert d.u == "\u00b5m"


def test_empty_values(mat):
    d = load_mat(mat(e=Empty("double", (0, 3)), c=Empty("char"), k=Empty("cell")))
    assert d.e.shape == (0, 3) and d.e.dtype == np.float64
    assert d.c == ""
    assert d.k.shape == (0, 0) and d.k.dtype == object


def test_cell_keeps_shape_and_mixed_content(mat):
    c = cell([1.0, "txt"], [np.arange(3.0), cell([2.0])])
    d = load_mat(mat(c=c))
    assert d.c.shape == (2, 2) and d.c.dtype == object
    assert d.c[0, 0] == 1.0 and d.c[0, 1] == "txt"
    assert np.array_equal(d.c[1, 0], np.arange(3.0).reshape(1, 3))
    assert d.c[1, 1].shape == (1, 1) and d.c[1, 1][0, 0] == 2.0


def test_struct_field_order_and_attribute_access(mat):
    d = load_mat(mat(s={"b": 1.0, "a": "x", "inner": {"value": 7.0}}))
    assert isinstance(d.s, MatStruct)
    assert list(d.s) == ["b", "a", "inner"]
    assert d.s.b == 1.0 and d.s["a"] == "x"
    assert d.s.inner.value == 7.0


def test_struct_array_keeps_layers_by_expids_shape(mat):
    # like proj_meta(isite).rd: 4 z layers x 2 ExpIDs
    elements = [[{"act": np.full((3, 5), 10.0 * i + j), "id": float(j)} for j in range(2)] for i in range(4)]
    d = load_mat(mat(rd=struct_array(elements)))
    assert d.rd.shape == (4, 2)
    for i in range(4):
        for j in range(2):
            assert d.rd[i, j].act.shape == (3, 5)
            assert np.all(d.rd[i, j].act == 10.0 * i + j)
            assert d.rd[i, j].id == j


def test_sparse(mat):
    m = sparse.csc_array(np.array([[1.0, 0, 0], [0, 0, 2.0]]))
    d = load_mat(mat(m=m))
    assert d.m.shape == (2, 3)
    assert np.array_equal(d.m.toarray(), m.toarray())


def test_objects_become_placeholders(mat):
    with pytest.warns(UserWarning, match="table"):
        d = load_mat(mat(t=Obj("table")))
    assert isinstance(d.t, MatlabObject) and d.t.class_name == "table"


def test_variable_selection(mat):
    path = mat(a=1.0, b=2.0)
    assert list(load_mat(path, "b")) == ["b"]
    assert list(load_mat(path, ["a", "b"])) == ["a", "b"]
    with pytest.raises(KeyError, match="nope"):
        load_mat(path, "nope")


def test_matfile_reads_lazily(mat):
    path = mat(a=np.zeros((2, 3)), s={"x": 1.0}, rd=struct_array([[{"v": 1.0}, {"v": 2.0}]]))
    with MatFile(path) as mf:
        assert mf.keys() == ["a", "rd", "s"] or set(mf.keys()) == {"a", "rd", "s"}
        info = mf.info()
        assert info["a"] == ("double", (2, 3))
        assert info["rd"] == ("struct", (1, 2))
        assert info["s"][0] == "struct"
        assert mf["rd"][0, 1].v == 2.0
        assert "#refs#" not in mf


def test_rejects_v5_files(tmp_path):
    p = tmp_path / "old.mat"
    p.write_bytes(b"MATLAB 5.0 MAT-file, Platform: PCWIN64".ljust(128, b" "))
    with pytest.raises(ValueError, match="v7.3"):
        load_mat(p)


def test_matstruct_attribute_errors():
    s = MatStruct(values=1, act=2)
    assert s.act == 2
    assert s["values"] == 1  # 'values' clashes with dict.values
    with pytest.raises(AttributeError):
        _ = s.missing


# -- against a file written by MATLAB (tests/matlab/ref_types.m) ----------------
def test_real_matlab_types_file(testdata):
    path = testdata / "reference" / "calliopy_types_v73.mat"
    if not path.is_file():
        pytest.skip("calliopy_types_v73.mat not generated yet")
    with pytest.warns(UserWarning):
        t = load_mat(path)
    assert t.dbl.shape == (2, 3, 4) and t.dbl[1, 2, 3] == 24.0  # reshape(1:24, 2, 3, 4)
    assert t.scalar == 3.5
    assert t.i16.dtype == np.int16 and list(t.i16.ravel()) == [1, -2, 3]
    assert t.sgl.dtype == np.float32
    assert t.lgc.dtype == bool and t.lgc.tolist() == [[True, False], [False, True]]
    assert np.array_equal(t.cpx, np.array([[1 + 2j, 3 - 4j]]))
    assert t.chr == "hello" and list(t.chrmat) == ["ab", "cd"] and t.uni == "\u00b5m"
    assert t.empty_dbl.shape == (0, 3) and t.empty_chr == "" and t.empty_cell.shape == (0, 0)
    assert t.cel.shape == (2, 2) and t.cel[0, 1] == "txt" and t.cel[1, 1][0, 0] == 2.0
    assert list(t.st) == ["b", "a"] and t.st.a == "x" and t.nested.inner.value == 7.0
    assert t.starr.shape == (4, 2)
    assert np.all(t.starr[3, 1].act == 42.0) and t.starr[3, 1].id == 8.0  # see ref_types.m
    assert t.sp.shape == (2, 3) and t.sp.toarray()[1, 2] == 2.0
    assert t.splog.dtype == bool
    assert isinstance(t.tbl, MatlabObject) and isinstance(t.str, MatlabObject)
