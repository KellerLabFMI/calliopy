"""Tests of calliopy.io.save_mat: round trips through load_mat, file layout, errors.

Run as a script to write the sample that tests/matlab/ref_io_save.m loads in MATLAB:

    python tests/test_io_save.py <CALLIOPY_TESTDATA>/samples
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy import sparse

from calliopy.io import MatStruct, load_mat, load_metafile
from calliopy.io._mat import MatlabObject
from calliopy.io._save import save_mat

SAMPLE_NAME = "save_mat_python.mat"


def assert_same(a, b, where="value"):
    """Recursive equality of load_mat outputs: types, dtypes, shapes, values, field order."""
    if isinstance(a, dict):
        assert isinstance(b, dict), where
        assert list(a.keys()) == list(b.keys()), where
        for k in a:
            assert_same(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, str):
        assert isinstance(b, str) and a == b, where
    elif sparse.issparse(a):
        assert sparse.issparse(b), where
        assert a.shape == b.shape and a.dtype == b.dtype, where
        np.testing.assert_array_equal(a.toarray(), b.toarray(), err_msg=where)
    elif isinstance(a, np.ndarray) and a.dtype == object:
        assert isinstance(b, np.ndarray) and b.dtype == object, where
        assert a.shape == b.shape, where
        for idx in np.ndindex(a.shape):
            assert_same(a[idx], b[idx], f"{where}{list(idx)}")
    elif isinstance(a, (np.ndarray, np.generic)):
        assert isinstance(b, (np.ndarray, np.generic)), where
        assert a.dtype == b.dtype, f"{where}: {a.dtype} != {b.dtype}"
        assert np.shape(a) == np.shape(b), f"{where}: {np.shape(a)} != {np.shape(b)}"
        np.testing.assert_array_equal(a, b, err_msg=where)
    else:
        assert a == b, where


def roundtrip(tmp_path, variables, **kw):
    path = tmp_path / "x.mat"
    save_mat(path, variables, **kw)
    return load_mat(path)


def obj(rows):
    """2-D object array from nested lists, without numpy guessing shapes."""
    out = np.empty((len(rows), len(rows[0])), dtype=object)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            out[i, j] = v
    return out


def matlab_sample():
    """The variables that tests/matlab/ref_io_save.m builds in MATLAB as well."""
    sa = obj([[
        MatStruct(name="x", val=np.array([[1.0, 2.0]]), tags=obj([["a", "bb"]])),
        MatStruct(name="yy", val=np.empty((0, 0)), tags=np.empty((0, 0), dtype=object)),
    ]])
    return {
        "d": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "s": np.array([[1.5, -2.25]], dtype=np.float32),
        "i8": np.array([[-3, 4]], dtype=np.int8),
        "u16": np.array([[7], [9]], dtype=np.uint16),
        "i64": np.int64(-5),
        "b": np.array([[True, False, True]]),
        "c": "h\u00e9llo",
        "e": np.zeros((0, 3)),
        "z": np.array([[1 + 2j, -3.5j]]),
        "st": MatStruct(a=np.float64(1.0), txt="abc", inner=MatStruct(v=np.arange(4.0).reshape(1, 4))),
        "sa": sa,
        "ce": obj([[np.float64(1.0), "a"], [np.array([[1.0, 2.0, 3.0]]), np.empty((0, 0), dtype=object)]]),
        "sp": sparse.csc_array(([1.5, -2.0, 3.0], ([0, 2, 1], [0, 1, 3])), shape=(3, 4)),
        # 1x1 struct arrays, as load_metafile returns proj_meta, with a nested 1x2 rd
        "sa1": obj([[MatStruct(k=np.float64(5.0))]]),
        "pm": obj([[MatStruct(animal="a1", rd=obj([[
            MatStruct(act=np.array([[1.0, 2.0]], dtype=np.float32), drug="x"),
            MatStruct(act=np.empty((0, 0), dtype=np.float32), drug=np.empty((0, 0), dtype=object)),
        ]]))]]),
    }


MATLAB_CLASSES = {
    "d": "double", "s": "single", "i8": "int8", "u16": "uint16", "i64": "int64", "b": "logical",
    "c": "char", "e": "double", "z": "double", "st": "struct", "sa": "struct", "ce": "cell",
    "sp": "double", "sa1": "struct", "pm": "struct",
}


def proj_meta_like(rng):
    """A [1 x 2] proj_meta with [nLayers x nExpIDs] rd, nested structs, cells and empties."""
    def record(il, ie):
        return MatStruct(
            ExpID=np.float64(100 * il + ie),
            act=rng.standard_normal((4, 6)).astype(np.float32),
            frame_times=np.linspace(0, 1, 6).reshape(1, 6),
            drug=obj([["saline", np.float64(ie)]]) if ie else np.empty((0, 0), dtype=object),
            stim=MatStruct(name=f"grating{ie}", ori=np.array([[0.0, 45.0, 90.0]]), ok=np.True_),
            notes="",
            roi=MatStruct(indices=np.array([[1], [5], [9]], dtype=np.int32)),
            wf=obj([[np.arange(3, dtype=np.uint16).reshape(1, 3)]]),
        )

    def rd(nl, ne):
        out = np.empty((nl, ne), dtype=object)
        for il in range(nl):
            for ie in range(ne):
                out[il, ie] = record(il, ie)
        return out

    pm = np.empty((1, 2), dtype=object)
    pm[0, 0] = MatStruct(animal="M1_200101_1", ExpGroup=obj([["ctrl", "day1"]]), rd=rd(2, 3),
                         params=MatStruct(fs=np.float64(30.0), layers=obj([["L2/3"], ["L5"]])))
    pm[0, 1] = MatStruct(animal="M2_200101_2", ExpGroup=np.empty((0, 0), dtype=object), rd=rd(1, 1),
                         params=MatStruct(fs=np.float64(15.0), layers=np.empty((0, 0), dtype=object)))
    return pm


# -- round trips ---------------------------------------------------------------------


def test_numeric_classes_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    variables = {}
    for dt in (np.float64, np.float32, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
               np.int64, np.uint64):
        variables[f"a_{np.dtype(dt).name}"] = (rng.standard_normal((3, 4)) * 50).astype(dt)
    variables["cube"] = np.arange(24.0).reshape(2, 3, 4)
    variables["fortran"] = np.asfortranarray(np.arange(6.0).reshape(2, 3))
    variables["bigendian"] = np.arange(4.0).reshape(2, 2).astype(">f8")
    out = roundtrip(tmp_path, variables)
    for k, v in variables.items():
        expect = v.astype(v.dtype.newbyteorder("=")) if k == "bigendian" else v
        assert_same(np.asarray(expect), out[k], k)


def test_scalars(tmp_path):
    out = roundtrip(tmp_path, {"f": 2.5, "i": 7, "t": True, "n32": np.float32(1.25), "u8": np.uint8(200),
                               "cz": 1 - 2j})
    assert out.f == 2.5 and out.f.dtype == np.float64
    assert out.i == 7 and out.i.dtype == np.int64
    assert out.t.dtype == bool and bool(out.t)
    assert out.n32.dtype == np.float32 and out.u8.dtype == np.uint8
    assert out.cz == 1 - 2j and out.cz.dtype == np.complex128
    raw = load_mat(tmp_path / "x.mat", squeeze_scalars=False)
    assert raw.f.shape == (1, 1)


def test_one_d_and_complex_and_logical(tmp_path):
    out = roundtrip(tmp_path, {"r": np.arange(5.0), "z": np.array([[1 + 1j, 2], [0, -1j]], np.complex64),
                               "b": np.array([[True, False], [False, True]])})
    assert out.r.shape == (1, 5)
    assert out.z.dtype == np.complex64 and out.z[1, 1] == -1j
    assert out.b.dtype == bool and out.b.tolist() == [[True, False], [False, True]]
    out = roundtrip(tmp_path, {"r": np.arange(5.0)}, oned_as="column")
    assert out.r.shape == (5, 1)


def test_char(tmp_path):
    text = "Hello w\u00f6rld \u03bb \U0001f600"
    out = roundtrip(tmp_path, {"c": text, "e": "", "u": np.str_("np")})
    assert out.c == text and out.e == "" and out.u == "np"
    with h5py.File(tmp_path / "x.mat", "r") as f:
        assert f["c"].attrs["MATLAB_class"] == b"char"
        assert f["c"].dtype == np.uint16 and f["c"].shape == (len(text.encode("utf-16-le")) // 2, 1)


def test_empties(tmp_path):
    out = roundtrip(tmp_path, {"none": None, "z03": np.zeros((0, 3)), "i20": np.zeros((2, 0), np.int16),
                               "b00": np.zeros((0, 0), bool), "cell": [], "cell00": np.empty((0, 0), object)})
    assert out.none.shape == (0, 0) and out.none.dtype == np.float64
    assert out.z03.shape == (0, 3)
    assert out.i20.shape == (2, 0) and out.i20.dtype == np.int16
    assert out.b00.dtype == bool
    assert out.cell.shape == (1, 0) and out.cell.dtype == object
    assert out.cell00.shape == (0, 0)


def test_struct_and_dict(tmp_path):
    s = MatStruct(z=np.float64(1.0), a="txt", nested=MatStruct(m=np.eye(2), c=obj([["x", np.float64(2.0)]])))
    out = roundtrip(tmp_path, {"s": s, "d": {"b": np.int32(3), "a": np.array([[1.0, 2.0]])}, "empty": MatStruct()})
    assert_same(s, out.s, "s")
    assert isinstance(out.d, MatStruct) and list(out.d) == ["b", "a"]  # field order kept
    assert out.empty == {}


def test_struct_array_and_cells(tmp_path):
    sa = obj([[MatStruct(a=np.float64(1.0), b="x")], [MatStruct(b="yy", a=np.float64(2.0))]])
    cell = obj([[np.float64(1.0), "two", None], [obj([["in"]]), MatStruct(k=np.int8(1)), [np.float64(3.0)]]])
    out = roundtrip(tmp_path, {"sa": sa, "cell": cell, "lst": ["a", np.float64(1.0)], "strs": np.array([["a", "bc"]])})
    assert out.sa.shape == (2, 1) and out.sa[1, 0].a == 2.0 and out.sa[1, 0].b == "yy"
    assert list(out.sa[1, 0]) == ["a", "b"]  # MATLAB has one field order per struct array
    assert out.cell.shape == (2, 3)
    assert out.cell[0, 1] == "two" and out.cell[0, 2].shape == (0, 0)
    assert out.cell[1, 0][0, 0] == "in" and out.cell[1, 1].k == 1
    assert out.cell[1, 2].shape == (1, 1)
    assert out.lst.shape == (1, 2) and out.lst[0, 0] == "a"
    assert out.strs.dtype == object and out.strs.tolist() == [["a", "bc"]]


def test_sparse(tmp_path):
    m = sparse.csc_array(([1.5, -2.0, 3.0], ([0, 2, 1], [0, 1, 3])), shape=(3, 4))
    out = roundtrip(tmp_path, {"m": m, "b": sparse.csr_matrix(np.eye(3, dtype=bool)),
                               "zero": sparse.csc_array((2, 5)), "i": sparse.coo_array(np.array([[0, 4], [5, 0]]))})
    np.testing.assert_array_equal(out.m.toarray(), m.toarray())
    assert out.b.dtype == bool and out.b.toarray().tolist() == np.eye(3, dtype=bool).tolist()
    assert out.zero.shape == (2, 5) and out.zero.nnz == 0
    assert out.i.dtype == np.float64 and out.i.toarray().tolist() == [[0, 4], [5, 0]]


def test_proj_meta_roundtrip(tmp_path):
    pm = proj_meta_like(np.random.default_rng(1))
    path = tmp_path / "ECT_test_meta.mat"
    save_mat(path, {"proj_meta": pm, "info": MatStruct(created="today")})
    out = load_mat(path)
    assert_same(pm, out.proj_meta, "proj_meta")
    assert out.info.created == "today"
    sites = load_metafile(path)
    assert sites.shape == (1, 2) and sites[0, 0].rd.shape == (2, 3) and sites[0, 1].rd.shape == (1, 1)
    np.testing.assert_array_equal(sites[0, 0].rd[1, 2].act, pm[0, 0].rd[1, 2].act)
    assert sites[0, 0].rd[0, 1].roi.indices.dtype == np.int32  # stored 1-based indices kept as is


def test_resave_is_identical(tmp_path):
    first = roundtrip(tmp_path, matlab_sample())
    save_mat(tmp_path / "y.mat", first)
    assert_same(first, load_mat(tmp_path / "y.mat"), "resaved")


def test_matlab_sample_roundtrip(tmp_path):
    sample = matlab_sample()
    out = roundtrip(tmp_path, sample)
    for k, v in sample.items():
        assert_same(v, out[k], k)


# -- file layout -----------------------------------------------------------------------


def test_header_and_userblock(tmp_path):
    save_mat(tmp_path / "x.mat", {"a": 1.0})
    head = (tmp_path / "x.mat").read_bytes()[:128]
    assert head.startswith(b"MATLAB 7.3 MAT-file, Platform: ")
    assert b"HDF5 schema 1.00 ." in head[:116]
    assert head[124:128] == b"\x00\x02IM"
    assert h5py.is_hdf5(tmp_path / "x.mat")
    with h5py.File(tmp_path / "x.mat", "r") as f:
        assert f.userblock_size == 512
        assert f["a"].attrs["MATLAB_class"] == b"double"


def test_struct_array_layout(tmp_path):
    sa = obj([[MatStruct(a=np.float64(1.0)), MatStruct(a=np.float64(2.0))]])
    save_mat(tmp_path / "x.mat", {"sa": sa, "c": ["x"]})
    with h5py.File(tmp_path / "x.mat", "r") as f:
        grp = f["sa"]
        assert grp.attrs["MATLAB_class"] == b"struct"
        assert [b"".join(x).decode() for x in grp.attrs["MATLAB_fields"]] == ["a"]
        assert grp["a"].shape == (2, 1)  # MATLAB 1x2, stored transposed
        assert "MATLAB_class" not in grp["a"].attrs
        assert f["c"].attrs["MATLAB_class"] == b"cell"
        assert "#refs#" in f


def test_extension_and_overwrite(tmp_path):
    save_mat(tmp_path / "noext", {"a": 1.0})
    assert (tmp_path / "noext.mat").is_file()
    save_mat(tmp_path / "noext.mat", {"b": 2.0})
    assert list(load_mat(tmp_path / "noext.mat")) == ["b"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["noext.mat"]  # no temporary file left


@pytest.mark.parametrize("version", ["7.3", "-v7.3", "v7.3"])
def test_version_spellings(tmp_path, version):
    save_mat(tmp_path / "x.mat", {"a": 1.0}, version=version)
    assert load_mat(tmp_path / "x.mat").a == 1.0


# -- errors ------------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["7", "-v7", "6", "-v4"])
def test_other_versions_not_implemented(tmp_path, version):
    with pytest.raises(NotImplementedError):
        save_mat(tmp_path / "x.mat", {"a": 1.0}, version=version)


@pytest.mark.parametrize(
    ("variables", "error"),
    [
        ({"1abc": 1.0}, ValueError),
        ({"a b": 1.0}, ValueError),
        ({"a" * 64: 1.0}, ValueError),
        ({"s": {"bad-field": 1.0}}, ValueError),
        ({"s": {1: 1.0}}, TypeError),
        ({"h": np.float16(1.0)}, TypeError),
        ({"t": np.array(["2020-01-01"], dtype="datetime64[D]")}, TypeError),
        ({"by": b"bytes"}, TypeError),
        ({"bs": np.array([b"x"])}, TypeError),
        ({"o": MatlabObject("table")}, TypeError),
        ({"f": print}, TypeError),
        ({"c": [object()]}, TypeError),
        ({"sa": obj([[MatStruct(a=1.0), MatStruct(b=1.0)]])}, ValueError),
        ({"sa": obj([[MatStruct(), MatStruct()]])}, ValueError),
    ],
)
def test_unsupported(tmp_path, variables, error):
    with pytest.raises(error):
        save_mat(tmp_path / "x.mat", variables)
    assert not (tmp_path / "x.mat").exists()
    assert list(tmp_path.iterdir()) == []


def test_bad_arguments(tmp_path):
    with pytest.raises(TypeError):
        save_mat(tmp_path / "x.mat", [("a", 1.0)])
    with pytest.raises(ValueError):
        save_mat(tmp_path / "x.mat", {"a": 1.0}, oned_as="diag")


# -- MATLAB -------------------------------------------------------------------------------


def test_matlab_reads_python_file(matlab_ref):
    """ref_io_save.m loads the file written by this module's __main__ and compares in MATLAB."""
    ref = matlab_ref("io_save")
    if ref.out5:
        pytest.skip(f"MATLAB side: {ref.out5}")
    sample = matlab_sample()
    for k, v in sample.items():  # MATLAB built the same values
        if isinstance(v, np.ndarray) and v.shape == (1, 1) and isinstance(v[0, 0], dict):
            v = v[0, 0]  # load_mat returns a MATLAB-written 1x1 struct as MatStruct
        assert_same(v, ref.out1[k], f"MATLAB {k}")
    assert set(ref.out2) == set(sample)
    assert all(bool(ref.out2[k]) for k in sample), {k: ref.out2[k] for k in sample}
    assert {k: ref.out3[k] for k in sample} == MATLAB_CLASSES


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    save_mat(out_dir / SAMPLE_NAME, matlab_sample())
