"""Tests of calliopy.io.update_mat: in-place replacement of variables in v7.3 MAT-files."""

from __future__ import annotations

import hashlib
import shutil
import warnings

import h5py
import numpy as np
import pytest
from mat73_writer import Obj, cell, struct_array, write_mat73
from scipy import io as sio
from test_io_save import assert_same, obj

from calliopy.io import MatStruct, load_mat, save_mat, update_mat

TMP_PREFIX = "#calliopy_new#"


def rois(n_per_layer, seed=0):
    """ROIs as RoiQuest saves them: 1 x nLayers cell of 1 x nRois struct arrays.

    ``indices`` are 1-based column-major linear pixel indices (Nx1 double), as MATLAB
    stores them; an empty layer is ``[]``.
    """
    rng = np.random.default_rng(seed)
    layers = np.empty((1, len(n_per_layer)), dtype=object)
    for j, n in enumerate(n_per_layer):
        if n == 0:
            layers[0, j] = np.empty((0, 0))
            continue
        elems = np.empty((1, n), dtype=object)
        for i in range(n):
            idx = np.sort(rng.choice(512 * 512, size=int(rng.integers(5, 40)), replace=False)) + 1
            elems[0, i] = MatStruct(
                indices=idx.astype(np.float64).reshape(-1, 1),
                shift=np.zeros((1, 2)),
                type=np.float64(i % 3),
            )
        layers[0, j] = elems
    return layers


def base_vars():
    return {
        "first_frame": np.arange(12, dtype=np.uint16).reshape(3, 4),
        "dx": np.float64(1.5),
        "name": "S1-T12345",
        "flag": np.array([[True, False]]),
        "meta": MatStruct(a=np.float64(1.0), txt="abc", tags=obj([["x", "yy"]])),
        "ROIs": rois([3, 2]),
        "ROItrans": np.eye(3),
    }


def make(tmp_path, variables=None, name="adata.mat"):
    path = tmp_path / name
    save_mat(path, base_vars() if variables is None else variables)
    return path


def refs_names(path):
    with h5py.File(path, "r") as f:
        return set(f["#refs#"].keys()) if "#refs#" in f else set()


def raw_snapshot(path, names):
    """Byte content, attributes and location of every HDF5 object under ``names``,
    following references into #refs#."""
    out = {}
    with h5py.File(path, "r") as f:

        def visit(obj, key):
            if key in out:
                return
            attrs = {k: np.asarray(v).tobytes() if not isinstance(v, np.ndarray) or v.dtype != object
                     else repr(v.tolist()) for k, v in obj.attrs.items()}
            if isinstance(obj, h5py.Group):
                out[key] = ("group", attrs, sorted(obj.keys()))
                for k in obj:
                    visit(obj[k], obj[k].name)
            else:
                data = obj[()]
                if h5py.check_ref_dtype(obj.dtype) is h5py.Reference:
                    live = [r for r in np.ravel(data) if r]
                    out[key] = ("refs", attrs, obj.shape, [f[r].name for r in live])
                    for r in live:
                        visit(f[r], f[r].name)
                else:
                    out[key] = ("data", attrs, obj.dtype.str, obj.shape, np.asarray(data).tobytes(),
                                obj.id.get_offset())

        for n in names:
            visit(f[n], f[n].name)
    return out


# -- replacing and adding ----------------------------------------------------------


def test_replace_scalar_array_string(tmp_path):
    path = make(tmp_path)
    update_mat(path, {"dx": np.float64(2.25), "first_frame": np.ones((5, 2), dtype=np.float32),
                      "name": "S1-T99"})
    d = load_mat(path)
    assert_same(d.dx, np.float64(2.25))
    assert_same(d.first_frame, np.ones((5, 2), dtype=np.float32))
    assert d.name == "S1-T99"
    assert_same(d.ROIs, base_vars()["ROIs"])


def test_replace_with_other_kinds(tmp_path):
    path = make(tmp_path)
    new = {
        "dx": "now a string",
        "name": np.float64(3.0),
        "flag": np.empty((0, 0), dtype=bool),
        "meta": MatStruct(b=obj([[np.float64(1.0), "z"]]), inner=MatStruct(v=np.arange(3.0).reshape(1, 3))),
        "first_frame": None,
        "c": "",
    }
    update_mat(path, new)
    d = load_mat(path)
    expected = save_and_load(tmp_path, new)
    for k in new:
        assert_same(d[k], expected[k], k)


def save_and_load(tmp_path, variables, **kw):
    ref = tmp_path / "fresh.mat"
    save_mat(ref, variables, **kw)
    return load_mat(ref)


def test_replace_rois_struct_arrays(tmp_path):
    path = make(tmp_path)
    new = rois([4, 0, 1], seed=1)
    update_mat(path, {"ROIs": new})
    d = load_mat(path)
    assert_same(d.ROIs, save_and_load(tmp_path, {"ROIs": new}).ROIs)
    # stored indices are written as given: 1-based
    assert d.ROIs[0, 0][0, 0].indices.min() >= 1
    assert d.ROIs[0, 1].shape == (0, 0)


def test_old_refs_removed(tmp_path):
    """The #refs# objects of a replaced variable are deleted, others are kept."""
    path = make(tmp_path)
    before = refs_names(path)
    with h5py.File(path, "r") as f:
        others = {f[r].name.split("/")[-1] for r in np.ravel(f["meta"]["tags"][()])}
    update_mat(path, {"ROIs": np.float64(0.0)})
    after = refs_names(path)
    assert others <= after
    assert after == others  # ROIs owned all other refs of this file
    assert len(before) > len(after)
    assert_same(load_mat(path).meta, base_vars()["meta"])


def test_add_new_variable(tmp_path):
    path = make(tmp_path)
    update_mat(path, {"newvar": np.arange(4.0), "newcell": ["a", np.float64(2.0)]})
    d = load_mat(path)
    assert_same(d.newvar, np.arange(4.0).reshape(1, 4))
    assert_same(d.newcell, obj([["a", np.float64(2.0)]]))
    for k, v in base_vars().items():
        assert_same(d[k], v, k)


def test_oned_as_column(tmp_path):
    path = make(tmp_path)
    update_mat(path, {"v": np.arange(3.0)}, oned_as="column")
    assert load_mat(path, "v").v.shape == (3, 1)


def test_untouched_variables_bitwise(tmp_path):
    path = make(tmp_path)
    keep = ["first_frame", "dx", "name", "flag", "meta", "ROItrans"]
    before = raw_snapshot(path, keep)
    update_mat(path, {"ROIs": rois([1], seed=5), "added": np.float64(1.0)})
    assert raw_snapshot(path, keep) == before


def test_header_bytes_unchanged(tmp_path):
    path = make(tmp_path)
    head = path.read_bytes()[:520]  # userblock, HDF5 signature, superblock version
    update_mat(path, {"ROIs": rois([2], seed=3)})
    update_mat(path, {"x": np.float64(1.0)})
    assert path.read_bytes()[:520] == head
    with h5py.File(path, "r") as f:
        assert f.userblock_size == 512


def test_repeated_updates(tmp_path):
    path = make(tmp_path)
    for k in range(6):
        new = rois([k + 1, 2], seed=k)
        update_mat(path, {"ROIs": new, "dx": np.float64(k)})
        d = load_mat(path)
        assert_same(d.ROIs, save_and_load(tmp_path, {"ROIs": new}).ROIs)
        assert d.dx == k
    # no leftovers: #refs# holds exactly what the current variables use
    fresh = tmp_path / "compare.mat"
    save_mat(fresh, dict(load_mat(path)))
    assert len(refs_names(path)) == len(refs_names(fresh))
    with h5py.File(path, "r") as f:
        assert not [k for k in f if k.startswith(TMP_PREFIX)]


def test_shared_refs_are_kept(tmp_path):
    """A #refs# object referenced by a kept variable survives (MATLAB shares refs)."""
    path = make(tmp_path, {"c1": obj([["a", np.float64(2.0)]]), "k": np.float64(1.0)})
    with h5py.File(path, "r+") as f:
        refs = f["c1"][()]
        ds = f.create_dataset("c2", data=refs, dtype=h5py.ref_dtype)
        ds.attrs.create("MATLAB_class", np.bytes_("cell"))
    update_mat(path, {"c1": np.float64(0.0)})
    d = load_mat(path)
    assert_same(d.c2, obj([["a", np.float64(2.0)]]))
    assert d.c1 == 0.0


def test_matlab_layout_file(tmp_path):
    """A file with MATLAB's own layout (ref names, header), including an object variable."""
    path = write_mat73(tmp_path / "m.mat", {
        "ROIs": cell([struct_array([[{"indices": np.array([[1.0], [2.0]]), "type": 1.0}]]), "x"]),
        "tbl": Obj("table"),
        "act": np.arange(6.0).reshape(2, 3),
    })
    head = path.read_bytes()[:512]
    tbl_before = raw_snapshot(path, ["tbl", "act"])
    new = rois([2], seed=9)
    update_mat(path, {"ROIs": new})
    assert path.read_bytes()[:512] == head
    assert raw_snapshot(path, ["tbl", "act"]) == tbl_before
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = load_mat(path)
    assert_same(d.ROIs, save_and_load(tmp_path, {"ROIs": new}).ROIs)
    assert refs_names(path).isdisjoint({"r1", "r2", "r3", "r4"})


def test_replacing_object_variable_warns(tmp_path):
    path = write_mat73(tmp_path / "m.mat", {"tbl": Obj("table"), "a": 1.0})
    with pytest.warns(UserWarning, match="table"):
        update_mat(path, {"tbl": np.float64(2.0)})
    assert load_mat(path, "tbl").tbl == 2.0


def test_large_dataset_not_rewritten(tmp_path):
    big = np.arange(64 * 256 * 256, dtype=np.uint16).reshape(64, 256, 256)  # 8 MB
    path = make(tmp_path, {"stack": big, "ROIs": rois([2])})
    with h5py.File(path, "r") as f:
        offset = f["stack"].id.get_offset()
    digest = hashlib.sha256(big.T.tobytes()).hexdigest()
    size = path.stat().st_size
    update_mat(path, {"ROIs": rois([30, 5], seed=2)})
    with h5py.File(path, "r") as f:
        assert f["stack"].id.get_offset() == offset
        assert hashlib.sha256(f["stack"][()].tobytes()).hexdigest() == digest
    assert path.stat().st_size - size < big.nbytes // 10
    assert_same(load_mat(path, "stack").stack, big)


def test_leftover_temporary_is_replaced(tmp_path):
    path = make(tmp_path)
    with h5py.File(path, "r+") as f:
        f.create_dataset(TMP_PREFIX + "dx", data=np.zeros(3))
    update_mat(path, {"dx": np.float64(7.0)})
    with h5py.File(path, "r") as f:
        assert TMP_PREFIX + "dx" not in f
    assert load_mat(path, "dx").dx == 7.0


def test_empty_mapping_is_noop(tmp_path):
    path = make(tmp_path)
    data = path.read_bytes()
    update_mat(path, {})
    assert path.read_bytes() == data


# -- failures ------------------------------------------------------------------------


def test_failed_value_leaves_old_content(tmp_path):
    path = make(tmp_path)
    before_refs = refs_names(path)
    with pytest.raises(TypeError):
        update_mat(path, {"ROIs": rois([5], seed=4), "dx": object()})
    d = load_mat(path)
    for k, v in base_vars().items():
        assert_same(d[k], v, k)
    assert refs_names(path) == before_refs
    with h5py.File(path, "r") as f:
        assert not [k for k in f if k.startswith("#") and k != "#refs#"]


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        update_mat(tmp_path / "nope.mat", {"a": 1.0})


def test_v5_file_rejected(tmp_path):
    path = tmp_path / "old.mat"
    sio.savemat(path, {"a": np.ones(3)})
    data = path.read_bytes()
    with pytest.raises(ValueError, match="v5/v7"):
        update_mat(path, {"a": 1.0})
    assert path.read_bytes() == data


def test_other_files_rejected(tmp_path):
    txt = tmp_path / "x.mat"
    txt.write_text("hello")
    with pytest.raises(ValueError, match="not a MATLAB v7.3"):
        update_mat(txt, {"a": 1.0})
    h5 = tmp_path / "plain.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("a", data=np.ones(2))
    with pytest.raises(ValueError, match="without the MATLAB v7.3 MAT-file header"):
        update_mat(h5, {"a": 1.0})


@pytest.mark.parametrize("bad", ["1abc", "_x", "a-b", "#refs#", "x" * 64])
def test_invalid_names(tmp_path, bad):
    path = make(tmp_path)
    data = path.read_bytes()
    with pytest.raises(ValueError, match="variable name"):
        update_mat(path, {bad: 1.0})
    assert path.read_bytes() == data


def test_bad_arguments(tmp_path):
    path = make(tmp_path)
    with pytest.raises(ValueError, match="oned_as"):
        update_mat(path, {"a": 1.0}, oned_as="diag")
    with pytest.raises(TypeError, match="mapping"):
        update_mat(path, [("a", 1.0)])
    with pytest.raises(TypeError):
        update_mat(path, {1: 1.0})


# -- real adata file (CALLIOPY_TESTDATA) ----------------------------------------------


def test_reference_adata_update_rois(sample_file, tmp_path):
    src = sample_file("S1-T*.mat")
    path = tmp_path / src.name
    shutil.copyfile(src, path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = load_mat(path)
    head = path.read_bytes()[:512]
    keep = [k for k in before if k != "ROIs"]
    raw_before = raw_snapshot(path, keep)
    new = rois([3, 0], seed=11)
    update_mat(path, {"ROIs": new})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        after = load_mat(path)
    raw_after = raw_snapshot(path, keep)
    assert path.read_bytes()[:512] == head
    assert raw_after == raw_before
    assert_same(after.ROIs, save_and_load(tmp_path, {"ROIs": new}).ROIs)
    assert set(after.keys()) == set(before.keys()) | {"ROIs"}
    for k in keep:
        assert_same(after[k], before[k], k)
