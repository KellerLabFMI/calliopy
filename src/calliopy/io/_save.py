"""Writing MATLAB v7.3 MAT-files (HDF5).

:func:`save_mat` is the counterpart of :func:`calliopy.io.load_mat`: it writes the
Python values that ``load_mat`` returns back into the HDF5 layout MATLAB uses for
``save(..., '-v7.3')``, so that ``save_mat(p, load_mat(q))`` writes the same
variables again.

=================================  ==================================================
Python                             MATLAB
=================================  ==================================================
numpy numeric array or scalar      numeric array of the same class (``float64`` ->
                                   double, ``float32`` -> single, ``int8`` ...
                                   ``uint64``); complex values stay complex
Python ``float`` / ``int``         1x1 double / 1x1 int64 (as ``numpy.asarray``)
``bool`` and bool arrays           logical
``str``                            char row vector (``''`` is a 0x0 char)
numpy array of ``str``             cell array of char with the same shape
:class:`MatStruct` or ``dict``     1x1 struct (keys are the field names, in order)
object array whose elements are    struct array with the array's shape (all elements
all MatStruct / dict               must have the same fields)
any other object array             cell array with the array's shape
``list`` / ``tuple``               1xN cell array, like MATLAB's ``{...}``
``None``                           ``[]`` (0x0 double)
``scipy.sparse`` array or matrix   sparse double (or sparse logical for bool)
=================================  ==================================================

Shapes: arrays with two or more dimensions keep their shape; 0-d values are 1x1;
1-D arrays become 1xN rows (``oned_as="row"``, the default, as ``scipy.io.savemat``)
or Nx1 columns (``oned_as="column"``).

Not supported (``TypeError``): ``bytes``, ``float16``, ``longdouble`` and other
dtypes MATLAB has no class for, ``datetime64``/``timedelta64``, structured
(record) arrays, numpy byte-string (``S``) arrays, :class:`MatlabObject`
placeholders (tables, strings, datetimes and classdef objects, which MATLAB writes
in an undocumented format), function handles and any other Python object. Struct
arrays without fields and struct arrays whose elements have different fields raise
``ValueError``.
"""

from __future__ import annotations

import datetime
import logging
import os
import re
import tempfile
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ._mat import MatlabObject

__all__ = ["save_mat"]

logger = logging.getLogger(__name__)

_CLASS_OF = {
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
_COMPLEX_PART = {np.dtype(np.complex128): np.float64, np.dtype(np.complex64): np.float32}
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,62}\Z")
_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_USERBLOCK = 512


def _check_name(name: object, what: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"{what} names must be str, got {type(name).__name__}")
    if not _NAME.match(name):
        raise ValueError(
            f"{name!r} is not a valid MATLAB {what} name (a letter, then up to 62 "
            "letters, digits or underscores)"
        )
    return name


def _header() -> bytes:
    """The 128-byte MAT-file header MATLAB puts at the start of the HDF5 userblock."""
    now = datetime.datetime.now()
    stamp = (
        f"{_DAYS[now.weekday()]} {_MONTHS[now.month - 1]} {now.day:2d} "
        f"{now:%H:%M:%S} {now.year}"
    )
    text = f"MATLAB 7.3 MAT-file, Platform: Python-calliopy, Created on: {stamp} HDF5 schema 1.00 ."
    raw = text.encode("ascii")[:116].ljust(116, b" ")
    # 8 bytes subsystem offset (unused), version 0x0200, endian indicator 'IM'
    return raw + b"\x00" * 8 + b"\x00\x02" + b"IM"


def _set_class(obj: h5py.HLObject, cls: str) -> None:
    obj.attrs.create("MATLAB_class", np.bytes_(cls))


def _is_sparse(value: Any) -> bool:
    if not hasattr(value, "tocsc"):
        return False
    from scipy import sparse

    return sparse.issparse(value)


class _Writer:
    """Writes Python values as MATLAB variables into one open HDF5 file."""

    def __init__(self, f: h5py.File, oned_as: str) -> None:
        self.f = f
        self.oned_as = oned_as
        self._refs: h5py.Group | None = None
        self._count = 0

    # -- helpers -----------------------------------------------------------------
    def _matlab_shaped(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            return arr.reshape(1, -1) if self.oned_as == "row" else arr.reshape(-1, 1)
        return arr

    def _new_ref_name(self) -> str:
        n, name = self._count, ""
        self._count += 1
        while True:
            name = chr(ord("a") + n % 26) + name
            n //= 26
            if n == 0:
                return name

    def _write_ref(self, value: Any) -> h5py.Reference:
        if self._refs is None:
            self._refs = self.f.create_group("#refs#")
        obj = self.write(self._refs, self._new_ref_name(), value)
        return obj.ref

    def _ref_dataset(self, parent: h5py.Group, name: str, refs: np.ndarray) -> h5py.Dataset:
        # MATLAB column-major: the HDF5 dataset holds the transposed array
        return parent.create_dataset(name, data=np.ascontiguousarray(refs.T), dtype=h5py.ref_dtype)

    # -- dispatch ----------------------------------------------------------------
    def write(self, parent: h5py.Group, name: str, value: Any) -> h5py.HLObject:
        if value is None:
            return self._empty(parent, name, "double", (0, 0))
        if isinstance(value, MatlabObject):
            raise TypeError(
                f"cannot write a MATLAB {value.class_name} object (MatlabObject placeholder)"
            )
        if isinstance(value, Mapping):
            return self._struct(parent, name, value)
        if isinstance(value, str):
            return self._char(parent, name, value)
        if isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError("bytes cannot be saved; decode to str or use a uint8 array")
        if _is_sparse(value):
            return self._sparse(parent, name, value)
        if isinstance(value, (list, tuple)):
            cell = np.empty(len(value), dtype=object)
            for i, v in enumerate(value):
                cell[i] = v
            return self._cell(parent, name, self._matlab_shaped(cell))
        if isinstance(value, (np.ndarray, np.generic, bool, int, float, complex)):
            return self._array(parent, name, np.asarray(value))
        raise TypeError(f"cannot save a value of type {type(value).__name__} in a MAT-file")

    def _array(self, parent: h5py.Group, name: str, arr: np.ndarray) -> h5py.HLObject:
        arr = self._matlab_shaped(arr)
        kind = arr.dtype.kind
        if kind == "O":
            flat = arr.ravel()
            if flat.size and all(isinstance(v, Mapping) for v in flat):
                return self._struct_array(parent, name, arr)
            return self._cell(parent, name, arr)
        if kind == "U":
            cell = np.empty(arr.shape, dtype=object)
            for idx in np.ndindex(arr.shape):
                cell[idx] = str(arr[idx])
            return self._cell(parent, name, cell)
        if kind == "b":
            if arr.size == 0:
                return self._empty(parent, name, "logical", arr.shape)
            ds = parent.create_dataset(name, data=np.ascontiguousarray(arr.T).astype(np.uint8))
            _set_class(ds, "logical")
            ds.attrs.create("MATLAB_int_decode", np.int32(1))
            return ds
        native = arr.dtype.newbyteorder("=") if arr.dtype.byteorder not in "=|" else arr.dtype
        if native in _COMPLEX_PART:
            part = _COMPLEX_PART[native]
            cls = _CLASS_OF[np.dtype(part)]
            if arr.size == 0:
                return self._empty(parent, name, cls, arr.shape)
            comp = np.dtype([("real", part), ("imag", part)])
            data = np.empty(arr.T.shape, dtype=comp)
            data["real"] = arr.T.real
            data["imag"] = arr.T.imag
            ds = parent.create_dataset(name, data=data)
            _set_class(ds, cls)
            return ds
        if native not in _CLASS_OF:
            raise TypeError(f"numpy dtype {arr.dtype} has no MATLAB class")
        cls = _CLASS_OF[native]
        if arr.size == 0:
            return self._empty(parent, name, cls, arr.shape)
        ds = parent.create_dataset(name, data=np.ascontiguousarray(arr.T).astype(native.newbyteorder("<")))
        _set_class(ds, cls)
        return ds

    # -- leaves ------------------------------------------------------------------
    @staticmethod
    def _empty(parent: h5py.Group, name: str, cls: str, shape: tuple[int, ...]) -> h5py.Dataset:
        # MATLAB stores an empty array as its dimensions, flagged with MATLAB_empty
        ds = parent.create_dataset(name, data=np.array(shape, dtype=np.uint64))
        _set_class(ds, cls)
        ds.attrs.create("MATLAB_empty", np.uint8(1))
        return ds

    def _char(self, parent: h5py.Group, name: str, text: str) -> h5py.Dataset:
        if not text:
            ds = self._empty(parent, name, "char", (0, 0))
        else:
            # MATLAB char holds UTF-16 code units; a 1xN row is stored as N x 1
            codes = np.frombuffer(text.encode("utf-16-le"), dtype="<u2")
            ds = parent.create_dataset(name, data=codes.reshape(-1, 1))
            _set_class(ds, "char")
        ds.attrs.create("MATLAB_int_decode", np.int32(2))
        return ds

    def _sparse(self, parent: h5py.Group, name: str, value: Any) -> h5py.Group:
        from scipy import sparse

        mat = sparse.csc_array(value)
        mat.sum_duplicates()
        mat.sort_indices()
        dt = mat.dtype
        if dt.kind == "b":
            cls, data = "logical", mat.data.astype(np.uint8)
        elif dt.kind == "c":
            cls = "double"
            data = np.empty(mat.nnz, dtype=np.dtype([("real", np.float64), ("imag", np.float64)]))
            data["real"], data["imag"] = mat.data.real, mat.data.imag
        elif dt.kind in "iuf":
            # MATLAB sparse matrices are double or logical only
            cls, data = "double", mat.data.astype(np.float64)
        else:
            raise TypeError(f"sparse dtype {dt} cannot be saved")
        grp = parent.create_group(name)
        _set_class(grp, cls)
        grp.attrs.create("MATLAB_sparse", np.uint64(mat.shape[0]))
        grp.create_dataset("jc", data=mat.indptr.astype(np.uint64))
        if mat.nnz:
            grp.create_dataset("ir", data=mat.indices.astype(np.uint64))
            grp.create_dataset("data", data=data)
        return grp

    # -- containers --------------------------------------------------------------
    def _cell(self, parent: h5py.Group, name: str, cell: np.ndarray) -> h5py.Dataset:
        if cell.size == 0:
            return self._empty(parent, name, "cell", cell.shape)
        refs = np.empty(cell.shape, dtype=object)
        for idx in np.ndindex(cell.shape):
            refs[idx] = self._write_ref(cell[idx])
        ds = self._ref_dataset(parent, name, refs)
        _set_class(ds, "cell")
        return ds

    @staticmethod
    def _set_fields(grp: h5py.Group, fields: list[str]) -> None:
        if not fields:
            return
        vals = np.empty(len(fields), dtype=object)
        for i, field in enumerate(fields):
            vals[i] = np.frombuffer(field.encode("ascii"), dtype="S1")
        grp.attrs.create("MATLAB_fields", vals, dtype=h5py.vlen_dtype(np.dtype("S1")))

    def _struct(self, parent: h5py.Group, name: str, value: Mapping) -> h5py.Group:
        fields = [_check_name(k, "field") for k in value]
        grp = parent.create_group(name)
        _set_class(grp, "struct")
        self._set_fields(grp, fields)
        for field in fields:
            self.write(grp, field, value[field])
        return grp

    def _struct_array(self, parent: h5py.Group, name: str, arr: np.ndarray) -> h5py.Group:
        first = arr.ravel()[0]
        fields = [_check_name(k, "field") for k in first]
        if not fields:
            raise ValueError("struct arrays without fields cannot be saved")
        for idx in np.ndindex(arr.shape):
            if set(arr[idx]) != set(fields):
                raise ValueError(
                    f"all elements of a struct array need the same fields; element {idx} has "
                    f"{sorted(arr[idx])}, element 0 has {sorted(fields)}"
                )
        grp = parent.create_group(name)
        _set_class(grp, "struct")
        self._set_fields(grp, fields)
        for field in fields:
            refs = np.empty(arr.shape, dtype=object)
            for idx in np.ndindex(arr.shape):
                refs[idx] = self._write_ref(arr[idx][field])
            # fields of a struct array are reference datasets without MATLAB_class
            self._ref_dataset(grp, field, refs)
        return grp


def save_mat(
    path: str | PathLike,
    variables: Mapping[str, Any],
    *,
    version: str = "7.3",
    oned_as: str = "row",
) -> None:
    """Save variables to a MATLAB v7.3 MAT-file.

    Python port of MATLAB's built-in ``save(filename, ..., '-v7.3')``.

    Parameters
    ----------
    path
        File to write. As in MATLAB, ``.mat`` is appended when the name has no
        extension. An existing file is replaced.
    variables
        Variable name -> value, e.g. ``{"proj_meta": proj_meta}`` or the
        :class:`MatStruct` returned by :func:`load_mat`. Names must be valid MATLAB
        variable names. See the module docstring for how values are converted.
    version
        MAT-file version; only ``"7.3"`` (also written ``"-v7.3"``) is supported.
    oned_as
        ``"row"`` (default) saves 1-D arrays and lists as 1xN, ``"column"`` as Nx1.

    Raises
    ------
    NotImplementedError
        For any ``version`` other than 7.3.
    TypeError, ValueError
        For values or names MATLAB cannot hold (see the module docstring).

    Differences from MATLAB
    -----------------------
    - Takes a mapping of variables instead of workspace variable names; there is
      no ``-append``, ``-struct``, ``-ascii``, ``-nocompression`` or wildcard option.
    - Only v7.3 files; data are written uncompressed (MATLAB compresses with gzip).
    - The file is written to a temporary file in the same folder first and then
      renamed, so an error never leaves a half-written file.
    - Python has no 1-D/2-D distinction like MATLAB: 1-D arrays follow ``oned_as``,
      and ``load_mat`` returns them 2-D.
    """
    if str(version).lstrip("-").lower().lstrip("v") != "7.3":
        raise NotImplementedError(f"only MAT-file version 7.3 can be written, not {version!r}")
    if oned_as not in ("row", "column"):
        raise ValueError(f"oned_as must be 'row' or 'column', not {oned_as!r}")
    if not isinstance(variables, Mapping):
        raise TypeError("variables must be a mapping of variable name -> value")
    names = [_check_name(k, "variable") for k in variables]
    path = Path(path)
    if path.suffix == "":
        path = path.with_name(path.name + ".mat")
    fd, tmp_name = tempfile.mkstemp(suffix=".tmp", prefix=f".{path.stem}_", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with h5py.File(tmp, "w", userblock_size=_USERBLOCK, libver="earliest") as f:
            writer = _Writer(f, oned_as)
            for name in names:
                writer.write(f, name, variables[name])
        with tmp.open("r+b") as fh:
            fh.write(_header())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    logger.debug("saved %d variable(s) to %s", len(names), path.name)
