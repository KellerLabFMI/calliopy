"""Write small HDF5 files laid out like MATLAB v7.3 MAT-files (test helper).

MATLAB is not available in CI, so tests build files with MATLAB's layout:
arrays stored transposed, a MATLAB_class attribute on every variable, cells and
struct arrays as object references into '#refs#', structs as groups with a
MATLAB_fields attribute. tests/matlab/ref_types.m writes a real MATLAB file with
the same content, which test_io_mat.py checks when CALLIOPY_TESTDATA is set.

Python values map to MATLAB types as follows: numpy arrays (given in MATLAB
shape) -> numeric/logical; Python float/int -> 1x1 double; bool -> 1x1 logical;
str -> char row; numpy unicode array -> char matrix; dict -> scalar struct;
Cell / StructArray / Empty / Obj -> cell, struct array, empty value, object.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
from scipy import sparse

_CLASS = {
    np.dtype(np.float64): "double",
    np.dtype(np.float32): "single",
    np.dtype(np.int8): "int8",
    np.dtype(np.uint8): "uint8",
    np.dtype(np.int16): "int16",
    np.dtype(np.uint16): "uint16",
    np.dtype(np.int32): "int32",
    np.dtype(np.uint32): "uint32",
    np.dtype(np.int64): "int64",
    np.dtype(np.uint64): "uint64",
}


@dataclass
class Cell:
    values: np.ndarray  # object array in MATLAB shape


@dataclass
class StructArray:
    elements: np.ndarray  # object array of dicts (same fields) in MATLAB shape


@dataclass
class Empty:
    matlab_class: str
    dims: tuple = (0, 0)


@dataclass
class Obj:
    matlab_class: str


def cell(*rows) -> Cell:
    """Cell array from rows of values, e.g. cell([1.0, "a"], [2.0, "b"])."""
    arr = np.empty((len(rows), len(rows[0])), dtype=object)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            arr[i, j] = v
    return Cell(arr)


def struct_array(elements: list[list[dict]]) -> StructArray:
    """Struct array from a nested list in MATLAB shape (rows x columns)."""
    arr = np.empty((len(elements), len(elements[0])), dtype=object)
    for i, row in enumerate(elements):
        for j, v in enumerate(row):
            arr[i, j] = v
    return StructArray(arr)


class _Writer:
    def __init__(self, f: h5py.File) -> None:
        self.f = f
        self.n = 0

    def _set_class(self, obj, cls: str) -> None:
        obj.attrs["MATLAB_class"] = np.bytes_(cls)

    def _fields(self, group, names) -> None:
        arr = np.empty(len(names), dtype=object)
        for i, name in enumerate(names):
            arr[i] = np.array(list(name), dtype="S1")
        group.attrs.create("MATLAB_fields", arr, dtype=h5py.vlen_dtype(np.dtype("S1")))

    def _refs(self, values: np.ndarray) -> np.ndarray:
        g = self.f.require_group("#refs#")
        refs = np.empty(values.shape, dtype=h5py.ref_dtype)
        for idx in np.ndindex(values.shape):
            self.n += 1
            name = f"r{self.n}"
            self.write(g, name, values[idx])
            refs[idx] = g[name].ref
        return refs

    def write(self, parent, name: str, value) -> None:
        if isinstance(value, Empty):
            ds = parent.create_dataset(name, data=np.asarray(value.dims, dtype=np.uint64))
            self._set_class(ds, value.matlab_class)
            ds.attrs["MATLAB_empty"] = np.uint8(1)
        elif isinstance(value, Obj):
            data = np.array([[3707764736], [2], [1], [1], [1], [1]], dtype=np.uint32)
            ds = parent.create_dataset(name, data=data)
            self._set_class(ds, value.matlab_class)
            ds.attrs["MATLAB_object_decode"] = np.int32(3)
        elif isinstance(value, str) or (isinstance(value, np.ndarray) and value.dtype.kind == "U"):
            rows = [value] if isinstance(value, str) else [str(v) for v in np.ravel(value)]
            codes = np.array([[ord(c) for c in r] for r in rows], dtype=np.uint16)
            ds = parent.create_dataset(name, data=np.ascontiguousarray(codes.T))
            self._set_class(ds, "char")
            ds.attrs["MATLAB_int_decode"] = np.int32(2)
        elif isinstance(value, Cell):
            ds = parent.create_dataset(name, data=self._refs(value.values).T, dtype=h5py.ref_dtype)
            self._set_class(ds, "cell")
        elif isinstance(value, StructArray):
            el = value.elements
            fields = list(el.flat[0].keys())
            g = parent.create_group(name)
            self._set_class(g, "struct")
            self._fields(g, fields)
            for k in fields:
                vals = np.empty(el.shape, dtype=object)
                for idx in np.ndindex(el.shape):
                    vals[idx] = el[idx][k]
                g.create_dataset(k, data=self._refs(vals).T, dtype=h5py.ref_dtype)
        elif isinstance(value, dict):
            g = parent.create_group(name)
            self._set_class(g, "struct")
            self._fields(g, list(value))
            for k, v in value.items():
                self.write(g, k, v)
        elif sparse.issparse(value):
            csc = sparse.csc_array(value)
            g = parent.create_group(name)
            is_bool = csc.dtype == bool
            self._set_class(g, "logical" if is_bool else "double")
            g.attrs["MATLAB_sparse"] = np.uint64(csc.shape[0])
            g.create_dataset("data", data=csc.data.astype(np.uint8) if is_bool else csc.data)
            g.create_dataset("ir", data=csc.indices.astype(np.uint64))
            g.create_dataset("jc", data=csc.indptr.astype(np.uint64))
        else:
            self._write_array(parent, name, value)

    def _write_array(self, parent, name: str, value) -> None:
        if isinstance(value, bool):
            arr = np.array([[value]])
        elif isinstance(value, (int, float, complex)):
            arr = np.array([[value]], dtype=complex if isinstance(value, complex) else np.float64)
        else:
            arr = np.asarray(value)
            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
            elif arr.ndim == 1:
                arr = arr.reshape(1, -1)
        if np.iscomplexobj(arr):
            base = arr.real.dtype
            data = np.empty(arr.shape, dtype=[("real", base), ("imag", base)])
            data["real"], data["imag"] = arr.real, arr.imag
            cls = _CLASS[base]
        elif arr.dtype == bool:
            data, cls = arr.astype(np.uint8), "logical"
        else:
            data, cls = arr, _CLASS[arr.dtype]
        ds = parent.create_dataset(name, data=np.ascontiguousarray(data.T))
        self._set_class(ds, cls)


def write_mat73(path, variables: dict):
    """Write ``variables`` to ``path`` with the layout of a MATLAB v7.3 MAT-file."""
    with h5py.File(path, "w", userblock_size=512) as f:
        writer = _Writer(f)
        for name, value in variables.items():
            writer.write(f, name, value)
    text = b"MATLAB 7.3 MAT-file, Platform: PCWIN64, Created on: Mon Nov 23 12:00:00 2026 HDF5 schema 1.00 ."
    with open(path, "r+b") as fh:
        fh.write(text.ljust(116, b" ") + b"\x00" * 8 + b"\x00\x02" + b"IM")
    return path
