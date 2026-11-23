"""Reading MATLAB v7.3 MAT-files (HDF5).

All lab .mat files (adata files, metafiles) are saved as v7.3, which is HDF5 with
MATLAB conventions on top. This module turns those conventions back into Python:

=====================  ==========================================================
MATLAB                 Python
=====================  ==========================================================
numeric array          ``numpy.ndarray`` with the MATLAB shape (``act`` stays
                       ``[nUnits x nFrames]``); 1x1 values become numpy scalars
                       unless ``squeeze_scalars=False``
logical                ``bool`` array
char                   ``str`` (a char matrix becomes an array of row strings)
struct                 :class:`MatStruct` (a dict whose fields are also attributes)
struct array           object array of :class:`MatStruct` with the MATLAB shape,
                       e.g. ``rd`` stays ``[nLayers x nExpIDs]``
cell                   object array with the MATLAB shape
sparse                 ``scipy.sparse.csc_array``
empty                  empty array (``''`` for char)
objects (table, ...)   :class:`MatlabObject` placeholder (cannot be decoded)
=====================  ==========================================================

Stored values are never changed: MATLAB index fields (e.g. ROI ``indices``) stay
1-based, and nothing is unwrapped or renamed.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable, Iterator
from os import PathLike
from pathlib import Path
from typing import Any, Self

import h5py
import numpy as np

__all__ = [
    "Adata",
    "MatFile",
    "MatStruct",
    "MatlabObject",
    "load_adata",
    "load_mat",
    "load_metafile",
    "metafile_info",
]

_NUMERIC = {
    "double": np.float64,
    "single": np.float32,
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "int64": np.int64,
    "uint64": np.uint64,
}


class MatStruct(dict):
    """A MATLAB struct: a dict whose fields can also be read as attributes.

    ``s.act`` and ``s["act"]`` are the same. A field whose name clashes with a dict
    method (``keys``, ``items``, ``values``, ``get``, ...) is only reachable as
    ``s["name"]``. Field order is the order MATLAB stored.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"struct has no field {name!r}") from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"struct has no field {name!r}") from None

    def __dir__(self) -> list[str]:
        fields = {k for k in self if isinstance(k, str) and k.isidentifier()}
        return sorted(set(dir(dict)) | fields)

    def __repr__(self) -> str:
        return f"MatStruct({', '.join(map(str, self.keys()))})"


class MatlabObject:
    """Placeholder for a MATLAB object that cannot be decoded from the file.

    MATLAB stores tables, strings, datetimes and classdef objects in an
    undocumented format. They are returned as this placeholder, with a warning.
    """

    __slots__ = ("class_name",)

    def __init__(self, class_name: str) -> None:
        self.class_name = class_name

    def __repr__(self) -> str:
        return f"<MATLAB {self.class_name} object (not decoded)>"


def _attr_str(obj: h5py.HLObject, name: str) -> str | None:
    value = obj.attrs.get(name)
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.ravel()[0] if value.size else b""
    if isinstance(value, bytes):
        return value.decode("ascii", "replace")
    return str(value)


def _attr_flag(obj: h5py.HLObject, name: str) -> bool:
    value = obj.attrs.get(name)
    return value is not None and bool(np.ravel(value)[0])


def _decode_chars(codes: np.ndarray) -> str:
    codes = np.asarray(codes)
    if codes.dtype.itemsize == 1:
        return codes.astype(np.uint8).tobytes().decode("latin-1")
    if codes.dtype.itemsize == 4:
        return codes.astype("<u4").tobytes().decode("utf-32-le", "replace")
    return codes.astype("<u2").tobytes().decode("utf-16-le", "replace")


class _Reader:
    """Converts HDF5 objects of one open MAT-file into Python values."""

    def __init__(self, f: h5py.File, squeeze_scalars: bool) -> None:
        self.f = f
        self.squeeze_scalars = squeeze_scalars

    def read(self, obj: h5py.HLObject) -> Any:
        cls = _attr_str(obj, "MATLAB_class")
        if isinstance(obj, h5py.Group):
            if "MATLAB_sparse" in obj.attrs:
                return self._sparse(obj, cls)
            if _attr_flag(obj, "MATLAB_empty"):
                return np.empty((0, 0), dtype=object)
            if cls in (None, "struct"):
                return self._struct(obj)
            return self._object(cls)
        if _attr_flag(obj, "MATLAB_empty"):
            return self._empty(obj, cls)
        if "MATLAB_object_decode" in obj.attrs:
            return self._object(cls)
        if cls == "char":
            return self._char(obj)
        if cls == "cell" or (cls is None and h5py.check_ref_dtype(obj.dtype) is not None):
            return self._cell(obj)
        if cls == "logical":
            return self._scalar(np.asarray(obj[()]).T.astype(bool))
        if cls is None or cls in _NUMERIC:
            return self._numeric(obj)
        return self._object(cls)

    # -- leaves ------------------------------------------------------------------
    def _scalar(self, arr: np.ndarray) -> Any:
        if self.squeeze_scalars and arr.size == 1:
            return arr[(0,) * arr.ndim]
        return arr

    def _numeric(self, obj: h5py.Dataset) -> Any:
        data = obj[()]
        names = data.dtype.names or ()
        if "real" in names and "imag" in names:
            data = data["real"] + 1j * data["imag"]
        return self._scalar(np.asarray(data).T)

    def _char(self, obj: h5py.Dataset) -> Any:
        codes = np.asarray(obj[()]).T  # MATLAB shape: rows x columns
        if codes.ndim < 2:
            return _decode_chars(codes)
        rows = [_decode_chars(row) for row in codes.reshape(-1, codes.shape[-1])]
        if len(rows) == 1:
            return rows[0]
        return np.array(rows, dtype=object).reshape(codes.shape[:-1])

    def _empty(self, obj: h5py.Dataset, cls: str | None) -> Any:
        # MATLAB stores the dimensions of an empty array as the dataset's data.
        dims = tuple(int(d) for d in np.ravel(obj[()])) if obj.size else (0, 0)
        if cls == "char":
            return ""
        if cls in ("cell", "struct"):
            return np.empty(dims, dtype=object)
        if cls == "logical":
            return np.empty(dims, dtype=bool)
        return np.empty(dims, dtype=_NUMERIC.get(cls or "double", np.float64))

    def _object(self, cls: str | None) -> MatlabObject:
        warnings.warn(
            f"MATLAB {cls} objects cannot be read from v7.3 files; "
            "returned a MatlabObject placeholder",
            stacklevel=4,
        )
        return MatlabObject(cls or "unknown")

    def _sparse(self, obj: h5py.Group, cls: str | None) -> Any:
        from scipy import sparse

        nrows = int(np.ravel(obj.attrs["MATLAB_sparse"])[0])
        jc = np.ravel(obj["jc"][()]).astype(np.int64)
        ir = np.ravel(obj["ir"][()]).astype(np.int64) if "ir" in obj else np.zeros(0, np.int64)
        if "data" in obj:
            data = obj["data"][()]
            names = data.dtype.names or ()
            if "real" in names and "imag" in names:
                data = data["real"] + 1j * data["imag"]
            data = np.ravel(data)
        else:
            data = np.zeros(0)
        if cls == "logical":
            data = data.astype(bool)
        return sparse.csc_array((data, ir, jc), shape=(nrows, len(jc) - 1))

    # -- containers --------------------------------------------------------------
    def _deref(self, ref: h5py.Reference) -> Any:
        if not ref:
            return None
        return self.read(self.f[ref])

    def _cell(self, obj: h5py.Dataset) -> np.ndarray:
        refs = obj[()]
        out = np.empty(refs.shape, dtype=object)
        for idx in np.ndindex(refs.shape):
            out[idx] = self._deref(refs[idx])
        return out.T

    @staticmethod
    def _field_names(obj: h5py.Group) -> list[str]:
        names: list[str] = []
        raw = obj.attrs.get("MATLAB_fields")
        if raw is not None:
            for item in np.ravel(raw):
                name = b"".join(np.ravel(item).tolist()).decode("ascii", "replace")
                if name in obj and name not in names:
                    names.append(name)
        names += [k for k in obj if k not in names]
        return names

    def _struct(self, obj: h5py.Group, only: Iterable[str] | None = None) -> Any:
        """Read a struct (array); with ``only``, read just those fields."""
        fields = self._field_names(obj)
        members = [obj[k] for k in fields]
        # In a struct array every field is a dataset of references without MATLAB_class.
        is_array = bool(members) and all(
            isinstance(m, h5py.Dataset)
            and "MATLAB_class" not in m.attrs
            and h5py.check_ref_dtype(m.dtype) is not None
            for m in members
        )
        if only is not None:
            only = set(only)
            keep = [(k, m) for k, m in zip(fields, members) if k in only]
        else:
            keep = list(zip(fields, members))
        if is_array:
            shapes = {m.shape for m in members}
            if len(shapes) != 1:
                raise ValueError(f"inconsistent struct array field sizes: {sorted(shapes)}")
            return self._struct_array([k for k, _ in keep], [m for _, m in keep], shapes.pop())
        return MatStruct((k, self.read(m)) for k, m in keep)

    def _struct_array(
        self, fields: list[str], members: list[h5py.Dataset], shape: tuple[int, ...]
    ) -> np.ndarray:
        refs = [m[()] for m in members]
        out = np.empty(shape, dtype=object)
        for idx in np.ndindex(shape):
            out[idx] = MatStruct((k, self._deref(r[idx])) for k, r in zip(fields, refs))
        return out.T


def _matlab_size(obj: h5py.HLObject) -> tuple[int, ...]:
    """MATLAB size of an HDF5 object, without reading its data."""
    if isinstance(obj, h5py.Group):
        if "MATLAB_sparse" in obj.attrs:
            return (int(np.ravel(obj.attrs["MATLAB_sparse"])[0]), obj["jc"].size - 1)
        members = [obj[k] for k in obj]
        if members and all(
            isinstance(m, h5py.Dataset)
            and "MATLAB_class" not in m.attrs
            and h5py.check_ref_dtype(m.dtype) is not None
            for m in members
        ):
            return tuple(members[0].shape[::-1])
        return (1, 1)
    if _attr_flag(obj, "MATLAB_empty"):
        return tuple(int(d) for d in np.ravel(obj[()])) if obj.size else (0, 0)
    return tuple(obj.shape[::-1])


def _check_v73(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if not h5py.is_hdf5(str(path)):
        with path.open("rb") as fh:
            head = fh.read(20)
        if head.startswith(b"MATLAB"):
            raise ValueError(
                f"{path.name} is a MATLAB v5/v7 MAT-file, not v7.3 (HDF5). calliopy reads "
                "v7.3 only: resave it in MATLAB with save(..., '-v7.3') or read it with "
                "scipy.io.loadmat."
            )
        raise ValueError(f"{path.name} is not a MATLAB v7.3 MAT-file")


class MatFile:
    """An open v7.3 MAT-file whose variables are read only when accessed.

    Use it when a file holds several large variables and only some are needed::

        with MatFile(path) as mf:
            print(mf.info())          # names, MATLAB classes and sizes; reads no data
            act = mf["act"]           # reads this variable only
    """

    def __init__(self, path: str | PathLike, *, squeeze_scalars: bool = True) -> None:
        self.path = Path(path)
        _check_v73(self.path)
        self._file = h5py.File(self.path, "r")
        self._reader = _Reader(self._file, squeeze_scalars)

    def keys(self) -> list[str]:
        """Variable names, in file order."""
        return [k for k in self._file if not k.startswith("#")]

    def info(self) -> dict[str, tuple[str, tuple[int, ...]]]:
        """MATLAB class and size of every variable, without reading data."""
        out = {}
        for name in self.keys():
            obj = self._file[name]
            out[name] = (_attr_str(obj, "MATLAB_class") or "unknown", _matlab_size(obj))
        return out

    def __getitem__(self, name: str) -> Any:
        if name not in self:
            raise KeyError(
                f"{name!r} is not a variable in {self.path.name}; variables: {self.keys()}"
            )
        return self._reader.read(self._file[name])

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and not name.startswith("#") and name in self._file

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        if not self._file.id.valid:
            return f"MatFile({str(self.path)!r}, closed)"
        return f"MatFile({str(self.path)!r}, variables={self.keys()})"


def load_mat(
    path: str | PathLike,
    variables: str | Iterable[str] | None = None,
    *,
    squeeze_scalars: bool = True,
) -> MatStruct:
    """Load variables from a MATLAB v7.3 MAT-file.

    Parameters
    ----------
    path
        The .mat file. Only v7.3 (HDF5) files are supported.
    variables
        One name or a list of names to load; ``None`` loads all variables.
    squeeze_scalars
        Return 1x1 numeric and logical values as numpy scalars (default). With
        ``False`` they stay 1x1 arrays, exactly as in MATLAB.

    Returns
    -------
    MatStruct
        Variable name -> value; ``d.proj_meta`` and ``d["proj_meta"]`` both work.
        See the module docstring for how MATLAB types are converted.
    """
    if isinstance(variables, str):
        variables = [variables]
    with MatFile(path, squeeze_scalars=squeeze_scalars) as mf:
        names = mf.keys() if variables is None else list(variables)
        missing = [n for n in names if n not in mf]
        if missing:
            raise KeyError(f"not in {mf.path.name}: {missing}; variables: {mf.keys()}")
        return MatStruct((n, mf[n]) for n in names)


# -- lab files -------------------------------------------------------------------


def _as_struct_array(value: Any) -> np.ndarray:
    """A struct read from a MAT-file as a 2-D object array (a 1x1 struct included)."""
    if isinstance(value, MatStruct):
        out = np.empty((1, 1), dtype=object)
        out[0, 0] = value
        return out
    return value


def _site_records(reader: _Reader, f: h5py.File, rd_fields: Iterable[str] | None) -> np.ndarray:
    top = f["proj_meta"]
    if not isinstance(top, h5py.Group):
        raise ValueError("proj_meta is not a struct")  # noqa: TRY004
    fields = reader._field_names(top)
    members = [top[k] for k in fields]

    def read_field(name: str, obj: h5py.HLObject) -> Any:
        if name != "rd":
            return reader.read(obj)
        if rd_fields is not None and isinstance(obj, h5py.Group) and not _attr_flag(obj, "MATLAB_empty"):
            return _as_struct_array(reader._struct(obj, only=rd_fields))
        return _as_struct_array(reader.read(obj))

    is_array = bool(members) and all(
        isinstance(m, h5py.Dataset)
        and "MATLAB_class" not in m.attrs
        and h5py.check_ref_dtype(m.dtype) is not None
        for m in members
    )
    if not is_array:
        site = MatStruct((k, read_field(k, m)) for k, m in zip(fields, members))
        return _as_struct_array(site)
    refs = [m[()] for m in members]
    shape = members[0].shape
    out = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        out[idx] = MatStruct(
            (k, read_field(k, f[r[idx]]) if r[idx] else None) for k, r in zip(fields, refs)
        )
    return out.T


def load_metafile(
    path: str | PathLike,
    *,
    rd_fields: str | Iterable[str] | None = None,
    squeeze_scalars: bool = True,
) -> np.ndarray:
    """Load ``proj_meta`` from a metafile.

    Parameters
    ----------
    path
        The metafile (v7.3 .mat holding a ``proj_meta`` variable).
    rd_fields
        Read only these fields of every record (e.g. ``["act", "frame_times"]``),
        which saves time and memory on large metafiles. ``None`` reads all fields.
    squeeze_scalars
        As in :func:`load_mat`.

    Returns
    -------
    numpy.ndarray
        ``proj_meta`` as a ``[1 x nSites]`` object array of :class:`MatStruct`.
        Each site's ``rd`` is a ``[nLayers x nExpIDs]`` object array of records, so
        ``proj_meta[0, isite].rd[ilayer, iexp].act`` mirrors MATLAB's
        ``proj_meta(isite).rd(ilayer, iexp).act`` (0-based).

    Differences from MATLAB
    -----------------------
    - Only reads a file given by path; the search, filter and ExpLog autocomplete of
      ``load_meta`` are not ported (they need ExpLog).
    - Returns ``proj_meta`` only and never writes into a workspace; read other
      variables of the file with :func:`load_mat`.
    - ``proj_meta`` and every ``rd`` are always 2-D object arrays, also when MATLAB
      stored a single site or a single record as a 1x1 struct.
    - Values are as stored: widefield fields wrapped in 1x1 cells stay wrapped.
    """
    if isinstance(rd_fields, str):
        rd_fields = [rd_fields]
    with MatFile(path, squeeze_scalars=squeeze_scalars) as mf:
        if "proj_meta" not in mf:
            raise KeyError(f"{mf.path.name} has no proj_meta variable; variables: {mf.keys()}")
        return _site_records(mf._reader, mf._file, rd_fields)


def metafile_info(path: str | PathLike) -> list[MatStruct]:
    """Describe a metafile without reading the records' data.

    Returns one :class:`MatStruct` per site with the site-level fields (``animal``,
    ``ExpGroup``, ...) plus ``rd_size`` (nLayers, nExpIDs) and ``rd_fields``.
    """
    sites = load_metafile(path, rd_fields=())
    with MatFile(path) as mf:
        f = mf._file
        top = f["proj_meta"]
        if "rd" not in top:
            rd_objs = [None] * sites.size
        elif isinstance(top["rd"], h5py.Dataset) and h5py.check_ref_dtype(top["rd"].dtype) is not None and "MATLAB_class" not in top["rd"].attrs:
            refs = top["rd"][()].T
            rd_objs = [f[r] for r in refs.ravel(order="F")]
        else:
            rd_objs = [top["rd"]]
        out = []
        for site, rd in zip(sites.ravel(order="F"), rd_objs):
            info = MatStruct((k, v) for k, v in site.items() if k != "rd")
            if rd is None:
                info.rd_size, info.rd_fields = (0, 0), []
            elif isinstance(rd, h5py.Group):
                info.rd_size = _matlab_size(rd)
                info.rd_fields = _Reader._field_names(rd)
            else:
                info.rd_size, info.rd_fields = _matlab_size(rd), []
            out.append(info)
    return out


_ADATA_TYPES = (("is_eeg", "wf_eeg"), ("first_frame_ch525", "2ch_wf"), ("dx", "2p"), ("first_frame", "wf"))


class Adata:
    """Read-only, lazy view of an adata file.

    Variables are read from the file the first time they are accessed, as
    attributes or items: ``ad.first_frame`` or ``ad["first_frame"]``. A variable
    whose name clashes with an attribute of this class (``path``, ``type``,
    ``exp_id``, ``variables``, ``loaded``, ``info``, ``load_all``, ``keys``) is
    only reachable as ``ad["name"]``.

    Attributes
    ----------
    path : pathlib.Path
    variables : tuple of str
        Names of the variables in the file.
    exp_id : int or None
        ExpID parsed from ``S1-T<number>`` in the file name.
    type : str
        ``"wf_eeg"``, ``"2ch_wf"``, ``"2p"``, ``"wf"`` or ``"unknown"``, decided from the
        variables present, as in MATLAB.

    Differences from MATLAB
    -----------------------
    - Read-only: no ``add_variable``/``save_changes`` (writing adata stays in MATLAB
      for now) and no ``is_raw_data_available`` (needs ExpLog).
    - Takes a path only; ``Adata(expid)`` needs ExpLog and comes later.
    - ``exp_id`` is ``None`` (MATLAB: 0 with a warning) when the name has no ``S1-T<n>``.
    """

    def __init__(self, path: str | PathLike, *, squeeze_scalars: bool = True) -> None:
        self.path = Path(path)
        _check_v73(self.path)
        with h5py.File(self.path, "r") as f:
            names = [k for k in f if not k.startswith("#")]
        self.variables: tuple[str, ...] = tuple(names)
        match = re.search(r"S1-T(\d+)", self.path.stem)
        self.exp_id: int | None = int(match.group(1)) if match else None
        self.type: str = next((t for v, t in _ADATA_TYPES if v in names), "unknown")
        self._squeeze = squeeze_scalars
        self._cache: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name not in self.__dict__.get("variables", ()):
            raise AttributeError(f"{type(self).__name__} has no attribute or variable {name!r}")
        return self[name]

    def __getitem__(self, name: str) -> Any:
        if name not in self._cache:
            if name not in self.variables:
                raise KeyError(f"{name!r} is not a variable in {self.path.name}; variables: {list(self.variables)}")
            with MatFile(self.path, squeeze_scalars=self._squeeze) as mf:
                self._cache[name] = mf[name]
        return self._cache[name]

    def __contains__(self, name: object) -> bool:
        return name in self.variables

    def __iter__(self) -> Iterator[str]:
        return iter(self.variables)

    def keys(self) -> tuple[str, ...]:
        return self.variables

    @property
    def loaded(self) -> tuple[str, ...]:
        """Variables read so far."""
        return tuple(n for n in self.variables if n in self._cache)

    def info(self) -> dict[str, tuple[str, tuple[int, ...]]]:
        """MATLAB class and size of every variable, without reading data."""
        with MatFile(self.path) as mf:
            return mf.info()

    def load_all(self, exclude: str | Iterable[str] = ()) -> Adata:
        """Read every variable not read yet, except those in ``exclude``."""
        exclude = {exclude} if isinstance(exclude, str) else set(exclude)
        todo = [n for n in self.variables if n not in self._cache and n not in exclude]
        if todo:
            with MatFile(self.path, squeeze_scalars=self._squeeze) as mf:
                for name in todo:
                    self._cache[name] = mf[name]
        return self

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | {v for v in self.variables if v.isidentifier()})

    def __repr__(self) -> str:
        return (
            f"Adata({self.path.name!r}, type={self.type!r}, exp_id={self.exp_id}, "
            f"{len(self.loaded)}/{len(self.variables)} variables loaded)"
        )


def load_adata(path: str | PathLike, *, squeeze_scalars: bool = True) -> Adata:
    """Open an adata file for lazy, read-only access; see :class:`Adata`."""
    return Adata(path, squeeze_scalars=squeeze_scalars)
