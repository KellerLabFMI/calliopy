"""Creating and writing adata files.

Port of the write side of calliope ``05_LoadData/Adata.m`` (Leonardo Lupori, class
version 1.4.2) and of ``01_DataManagement/save_adata.m``, built on
:func:`calliopy.io.save_mat` (new files) and :func:`calliopy.io.update_mat` (changes to
existing files).

- :class:`WritableAdata`: the read-only :class:`calliopy.io.Adata` view plus MATLAB's
  ``Adata.create``, ``add_variable``, assignment of variables and ``save_changes``.
- :func:`save_adata`: the old positional writer of registration results.
- :func:`adata_file_path`, :func:`adata_file_path_from_fnames`: MATLAB's file-name rules
  (``<adata_dir>/<userID>/<mouse_id>/Adata-S1-T<ExpID>.mat``).

The modification log
--------------------
MATLAB's ``save_changes`` appends one row per save to the variable ``modification_log``,
a MATLAB ``table`` whose columns hold a ``datetime`` and ``string`` values. calliopy
cannot write MATLAB objects (tables, strings, datetimes), and appending a struct to that
table would make MATLAB's next ``save_changes`` fail. :class:`WritableAdata` therefore
keeps its own log in the variable :data:`MODIFICATION_LOG_VARIABLE`
(``calliopy_modification_log``), an ``N x 1`` struct array with the same columns, and
never touches MATLAB's ``modification_log``.

Not ported here: looking up the file of an ExpID (``Adata(expid)``,
``Adata.create(expid)``) and reading the raw-data status from the ExpLog database
(``is_raw_data_available``); the status can be passed to
:meth:`WritableAdata.save_changes` instead.
"""

from __future__ import annotations

import datetime
import inspect
import logging
import math
import os
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from ._mat import Adata, MatFile, MatStruct, load_mat
from ._save import _check_name, save_mat
from ._update import update_mat

__all__ = [
    "MODIFICATION_LOG_VARIABLE",
    "WritableAdata",
    "adata_file_path",
    "adata_file_path_from_fnames",
    "save_adata",
]

logger = logging.getLogger(__name__)

MODIFICATION_LOG_VARIABLE = "calliopy_modification_log"
"""Name of the modification log :class:`WritableAdata` keeps in an adata file."""

ADATA_CLASS_VERSION = "1.4.2"
"""``Adata.VERSION`` of the MATLAB class this module ports."""

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_OWN_ATTRIBUTES = frozenset({"path", "variables", "exp_id", "type"})
_CALLER_DEPTH = 8


# ---------------------------------------------------------------------------------------
# small helpers shared with the savers of the other subpackages (private)
# ---------------------------------------------------------------------------------------


def _calliopy_version() -> str:
    from .. import __version__

    return __version__


def _datestr(now: datetime.datetime | None = None) -> str:
    """MATLAB ``datestr(now)``: ``'dd-mmm-yyyy HH:MM:SS'`` with English month names."""
    now = datetime.datetime.now() if now is None else now
    return f"{now.day:02d}-{_MONTHS[now.month - 1]}-{now.year:04d} {now:%H:%M:%S}"


def _cell_row(values: Iterable[Any], *, empty_shape: tuple[int, int] = (0, 0)) -> Any:
    """A ``1 x N`` MATLAB cell; an empty object array of ``empty_shape`` when there is nothing.

    A ``list`` is returned because :func:`calliopy.io.save_mat` writes a list as a cell
    whatever it holds, but an object array whose elements are all structs as a struct
    array: ``{struct, struct}`` must stay a cell. MATLAB's ``x = {}`` is a ``0 x 0`` cell,
    which an empty list would write as ``1 x 0``.
    """
    items = list(values)
    if not items:
        return np.empty(empty_shape, dtype=object)
    return items


def _as_cell(value: Any) -> Any:
    """A value known to be a MATLAB cell (e.g. ``ROIs``), made safe for writing.

    :func:`calliopy.io.load_mat` returns a cell of ``1 x 1`` structs as an object array of
    :class:`MatStruct`, which :func:`calliopy.io.save_mat` would write back as a struct
    array; such a cell is returned as a ``1 x N`` list (column-major order). Other values
    are returned unchanged.
    """
    if isinstance(value, Mapping):
        return [value]
    if (isinstance(value, np.ndarray) and value.dtype == object and value.size
            and all(isinstance(x, Mapping) for x in value.ravel())):
        return list(value.ravel(order="F"))
    return value


def _cells(value: Any) -> list:
    """A MATLAB cell as a list in column-major order (a non-cell value is one element)."""
    if isinstance(value, np.ndarray) and value.dtype == object:
        return list(value.ravel(order="F"))
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _struct_items(value: Any) -> list:
    """The elements of a struct array (MatStruct, object array, list), column-major."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return []
        return [x for x in value.ravel(order="F") if isinstance(x, Mapping)]
    return list(value)


def _struct_array(items: Any) -> Any:
    """A list of structs as MATLAB stores a ``1 x N`` struct array.

    One element is written as a plain struct (what MATLAB stores for a ``1 x 1`` struct),
    several as a ``(1, N)`` object array of :class:`MatStruct`. No element gives ``None``,
    written as ``[]`` (``0 x 0`` double): calliopy cannot write an empty struct array with
    fields (the same convention as RoiQuest for emptied ROI layers).
    """
    if isinstance(items, Mapping):
        return MatStruct(items)
    if (isinstance(items, np.ndarray) and items.dtype == object and items.ndim == 2
            and items.size and all(isinstance(x, Mapping) for x in items.ravel())):
        return items
    lst = _struct_items(items)
    if not lst:
        return None
    if len(lst) == 1:
        return MatStruct(lst[0])
    out = np.empty((1, len(lst)), dtype=object)
    for i, s in enumerate(lst):
        out[0, i] = MatStruct(s)
    return out


def _empty_struct() -> MatStruct:
    """MATLAB ``struct`` (``1 x 1``, no fields)."""
    return MatStruct()


def _num2str_int(values: Any) -> str:
    """MATLAB ``num2str`` of an integer-valued scalar or row vector (``[3 4]`` -> ``'3  4'``)."""
    arr = np.atleast_1d(np.asarray(values)).ravel()
    if arr.size == 0:
        return ""
    ints = [int(v) for v in arr]
    width = max(len(str(v)) for v in ints) + 2
    return "".join(f"{v:>{width}d}" for v in ints).strip()


def _float_row(values: Any) -> np.ndarray:
    """A numeric row vector as MATLAB double ``1 x N`` (e.g. ``nbr_frames``)."""
    return np.asarray(values, dtype=np.float64).reshape(1, -1)


def _raw_data_available(status: Any) -> bool:
    """MATLAB ``Adata.is_raw_data_available`` for a known ``RawDataStatus`` value.

    NaN (not checked yet), 0 (checked, failed) and 1 (checked, passed): available. 2 (raw
    files deleted): not available. An empty value (ExpID not in the ExpLog) or any other
    value: not available, with a warning.
    """
    arr = np.asarray(status, dtype=np.float64).ravel()
    if arr.size == 0:
        warnings.warn("ExpID not found in ExpLog", stacklevel=3)
        return False
    s = float(arr[0])
    if math.isnan(s) or s in (0.0, 1.0):
        return True
    if s == 2.0:
        return False
    warnings.warn(
        f"Modifications to Adata locked for safety. Unexpected RawDataStatus value: {s:g}",
        stacklevel=3,
    )
    return False


def _keep_cell(name: str, value: Any) -> Any:
    """Keep a variable the file stores as a cell a cell when it holds only structs.

    :func:`calliopy.io.load_mat` returns a cell of ``1 x 1`` structs as an object array
    of :class:`MatStruct`, which :func:`calliopy.io.save_mat` writes as a struct array. A
    ``1 x N`` one is turned into a list (written as a cell); other shapes cannot be, and
    are written as a struct array with a warning.
    """
    if not (isinstance(value, np.ndarray) and value.dtype == object and value.size
            and all(isinstance(x, Mapping) for x in value.ravel())):
        return value
    if value.ndim == 2 and value.shape[0] == 1:
        return list(value.ravel())
    warnings.warn(
        f"{name} is a cell in the file but holds only structs in a {value.shape} array; it is "
        "written as a struct array",
        stacklevel=4,
    )
    return value


def _caller_chain(depth: int = _CALLER_DEPTH) -> str:
    """``'outer->...->inner->WritableAdata.save_changes'`` for the modification log.

    MATLAB writes the whole ``dbstack`` (outermost first) up to ``Adata.save_changes``.
    Here the ``depth`` innermost frames outside this module are used.
    """
    frame = inspect.currentframe()
    names: list[str] = []
    try:
        f = frame.f_back if frame is not None else None
        while f is not None and f.f_globals.get("__name__") == __name__:
            f = f.f_back
        while f is not None and len(names) < depth:
            code = f.f_code
            names.append(getattr(code, "co_qualname", code.co_name))
            f = f.f_back
    finally:
        del frame
    names.reverse()
    names.append("WritableAdata.save_changes")
    return "->".join(names)


def _log_rows(value: Any) -> list[MatStruct]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [MatStruct(value)]
    if isinstance(value, np.ndarray) and value.dtype == object:
        rows = list(value.ravel(order="F"))
        if all(isinstance(r, Mapping) for r in rows):
            return [MatStruct(r) for r in rows]
    if isinstance(value, np.ndarray) and value.size == 0:
        return []
    raise TypeError(
        f"{MODIFICATION_LOG_VARIABLE} in this file is not a struct array written by calliopy"
    )


def _log_column(values: Sequence[str]) -> np.ndarray:
    out = np.empty((len(values), 1), dtype=object)
    for i, v in enumerate(values):
        out[i, 0] = str(v)
    return out


# ---------------------------------------------------------------------------------------
# file-name rules
# ---------------------------------------------------------------------------------------


def adata_file_path(
    adata_dir: str | PathLike, user_id: str, mouse_id: str, exp_id: int
) -> Path:
    """Path of the adata file of an experiment, as calliope builds it when creating one.

    ``<adata_dir>/<user_id>/<mouse_id>/Adata-S1-T<exp_id>.mat``: the name used by
    ``register_bin.m``, ``register_wid.m``, ``preprocess_owl.m`` and ``Adata.create``
    (via ``expid2adataPath``). Nothing is looked up or created.

    Parameters
    ----------
    adata_dir
        Root folder of the adata files (MATLAB ``set_lab_paths``; always an argument here).
    user_id, mouse_id
        ExpLog ``pi`` and ``animalid`` of the experiment.
    exp_id
        ExpID.
    """
    return Path(adata_dir) / str(user_id) / str(mouse_id) / f"Adata-S1-T{int(exp_id)}.mat"


def _stem_from_fnames(fnames: Any, *, wid_rule: bool) -> str:
    names = _cells(fnames)
    if not names or not isinstance(names[0], str) or not names[0]:
        raise ValueError("fnames must hold at least one file name")
    fn = re.split(r"[\\/]", names[0])[-1]
    cut = "." if (wid_rule and "wid" in fn) else "_"
    if cut not in fn:
        raise ValueError(f"cannot derive the adata name from {fn!r}: no {cut!r} in the name")
    return fn[: fn.index(cut)]


def adata_file_path_from_fnames(
    adata_dir: str | PathLike,
    user_id: str,
    mouse_id: str,
    fnames: Any,
    *,
    wid_rule: bool = False,
) -> Path:
    """Path of a new adata file named after the first raw-data file (older calliope rule).

    ``<adata_dir>/<user_id>/<mouse_id>/Adata-<stem>.mat``, where ``<stem>`` is the name of
    ``fnames[0]`` (folders removed, ``\\`` or ``/``) up to its first ``'_'``, as in
    ``register_exp_downsample.m`` and ``register_exp_avgplanes.m``
    (``S1-T123_ch525.bin`` -> ``Adata-S1-T123.mat``). With ``wid_rule`` (``save_adata.m``),
    a name containing ``'wid'`` is cut at its first ``'.'`` instead.

    Raises
    ------
    ValueError
        ``fnames`` is empty, or the name has no ``'_'`` (``'.'``). MATLAB then named the
        file ``Adata-.mat``.
    """
    stem = _stem_from_fnames(fnames, wid_rule=wid_rule)
    return Path(adata_dir) / str(user_id) / str(mouse_id) / f"Adata-{stem}.mat"


# ---------------------------------------------------------------------------------------
# WritableAdata
# ---------------------------------------------------------------------------------------


class WritableAdata(Adata):
    """An adata file opened for reading and writing (``05_LoadData/Adata.m``).

    Everything :class:`calliopy.io.Adata` offers (lazy loading, ``type``, ``exp_id``,
    ``load_all``) plus MATLAB's writing methods: :meth:`create` (new file),
    :meth:`add_variable`, assignment (``ad.dx = value``, ``ad["dx"] = value`` or
    :meth:`set_variable`) and :meth:`save_changes`. Changes stay in memory until
    :meth:`save_changes`.

    Parameters
    ----------
    path
        Existing v7.3 adata file. Use :meth:`create` for a new one.
    squeeze_scalars
        As for :class:`calliopy.io.Adata`.

    Notes
    -----
    A separate class, so that :class:`calliopy.io.Adata` and :func:`calliopy.io.load_adata`
    stay read-only: code that only reads cannot change a file by an accidental
    assignment. ``isinstance(w, Adata)`` is true.

    Assigning a name that is not a variable raises ``AttributeError`` / ``KeyError``, as
    MATLAB (no such property); use :meth:`add_variable`. Variables whose names clash with
    attributes (``path``, ``type``, ``exp_id``, ``variables``, methods) are set with
    ``ad["name"] = value``.

    Differences from MATLAB
    -----------------------
    - Takes a path; ``Adata(expid)`` and ``Adata.create(expid)`` need ExpLog access and are
      not ported. ``exp_id`` is ``None`` (MATLAB 0 with a warning) when the name has no
      ``S1-T<n>``.
    - The modification log is written to :data:`MODIFICATION_LOG_VARIABLE` as a struct
      array, not to MATLAB's ``modification_log`` table (see the module docstring).
    - The raw-data lock of ``save_changes`` needs the ExpLog ``RawDataStatus``; it is
      applied only when the value is passed (``raw_data_status``).
    - Existing files are always changed in place (:func:`calliopy.io.update_mat`); MATLAB
      rewrites the whole file when the changed variables exceed 10 MB
      (``size_threshold_mb``), which also compacts it. calliopy cannot do that for files
      holding MATLAB objects (tables, transforms) and writes uncompressed, so a rewrite
      could make the file larger. The file therefore grows by the size of every changed
      variable.
    - ``type`` is decided when the object is made, as in MATLAB, and not updated by
      :meth:`add_variable`.

    A variable that the file stores as a cell stays a cell when its new value is a
    ``1 x N`` object array of structs (as :func:`calliopy.io.load_mat` returns a cell of
    ``1 x 1`` structs); :func:`calliopy.io.save_mat` alone would write that as a struct
    array. For new variables, give such cells as a ``list``.
    """

    def __init__(self, path: str | PathLike, *, squeeze_scalars: bool = True) -> None:
        super().__init__(path, squeeze_scalars=squeeze_scalars)
        self._modified: dict[str, MatStruct] = {}
        self._pending_mkdir = False
        self._replace_file = False

    # -- construction of a new file ---------------------------------------------------
    @classmethod
    def create(
        cls,
        path: str | PathLike,
        *,
        create_dir: bool = False,
        adata_root: str | PathLike | None = None,
        overwrite: bool = False,
        squeeze_scalars: bool = True,
    ) -> WritableAdata:
        """A new, empty adata object for a file that :meth:`save_changes` will write.

        Port of the static method ``Adata.create(path, 'create_dir', ...)``. Nothing is
        written until :meth:`save_changes`.

        Parameters
        ----------
        path
            The new file; must end in ``.mat`` (any case).
        create_dir
            Create the missing parent folder (and its parents) in :meth:`save_changes`,
            so that no empty folder is left when nothing is saved.
        adata_root
            When given, ``path`` must lie inside this folder (MATLAB requires the
            ``set_lab_paths`` adata root; here it is an optional argument).
        overwrite
            Replace an existing file as a whole on :meth:`save_changes`. Not in MATLAB,
            which always refuses (the default here).
        squeeze_scalars
            As for :class:`calliopy.io.Adata`.

        Raises
        ------
        ValueError
            The name does not end in ``.mat``, or ``path`` is outside ``adata_root``.
        FileExistsError
            ``path`` exists and ``overwrite`` is false (MATLAB ``Adata:FileExists``).
        FileNotFoundError
            The parent folder does not exist and ``create_dir`` is false.
        IsADirectoryError
            ``path`` is a folder.

        Notes
        -----
        Differences from MATLAB: the root check compares whole path components of the
        absolute paths (MATLAB compares string prefixes, so a sibling folder whose name
        starts with the root's name passed); case is ignored on Windows, as in MATLAB.
        """
        p = Path(path)
        if p.suffix.lower() != ".mat":
            raise ValueError(f"File must have .mat extension, got: {p.suffix or 'empty extension'}")
        if p.is_dir():
            raise IsADirectoryError(f"{p} is a folder")
        exists = p.exists()
        if exists and not overwrite:
            raise FileExistsError(
                f"File already exists: {p}\nOpen existing files with WritableAdata(path)."
            )
        if adata_root is not None:
            root = os.path.normcase(os.path.abspath(adata_root))
            target = os.path.normcase(os.path.abspath(p))
            try:
                inside = os.path.commonpath([root, target]) == root and target != root
            except ValueError:  # different drives
                inside = False
            if not inside:
                raise ValueError(
                    "Adata files can only be created in the adata folder: "
                    f"{adata_root}\nRequested path: {p}"
                )
        obj = cls.__new__(cls)
        object.__setattr__(obj, "path", p)
        object.__setattr__(obj, "variables", ())
        match = re.search(r"S1-T(\d+)", p.stem)
        object.__setattr__(obj, "exp_id", int(match.group(1)) if match else None)
        object.__setattr__(obj, "type", "unknown")
        obj._squeeze = squeeze_scalars
        obj._cache = {}
        obj._modified = {}
        obj._pending_mkdir = False
        obj._replace_file = exists
        if not p.parent.is_dir():
            if create_dir:
                obj._pending_mkdir = True
            else:
                raise FileNotFoundError(
                    f"Directory does not exist: {p.parent}\nUse create_dir=True to create it."
                )
        return obj

    # -- attribute and item assignment ------------------------------------------------
    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name in _OWN_ATTRIBUTES:
            object.__setattr__(self, name, value)
            return
        if name in self.__dict__.get("variables", ()):
            self.set_variable(name, value)
            return
        raise AttributeError(
            f"{name!r} is not a variable of {self.path.name}; use add_variable to add it"
        )

    def __setitem__(self, name: str, value: Any) -> None:
        self.set_variable(name, value)

    def set_variable(self, name: str, value: Any) -> None:
        """Replace the value of an existing variable (MATLAB dot assignment).

        The variable is marked ``'modified'`` (also when it was added before and not saved
        yet, as in MATLAB) and written by :meth:`save_changes`.

        Raises
        ------
        KeyError
            ``name`` is not a variable; use :meth:`add_variable`.
        """
        if name not in self.variables:
            raise KeyError(f"{name!r} is not a variable of {self.path.name}; use add_variable")
        self._cache[name] = value
        self._mark(name, "modified")

    def add_variable(self, variables: Mapping[str, Any] | None = None, /, **kwargs: Any) -> None:
        """Add new variables (``Adata.add_variable(struct(...))``).

        Parameters
        ----------
        variables
            Mapping of new variable name -> value (a :class:`MatStruct` or ``dict``, like
            MATLAB's struct argument); keyword arguments are added after it.

        Warns
        -----
        UserWarning
            Nothing given, or a name that already exists: that variable is skipped and
            left unchanged, as in MATLAB (assign it to change it).

        Raises
        ------
        TypeError, ValueError
            Not a mapping, or not a valid MATLAB variable name.
        """
        if variables is None:
            variables = {}
        if not isinstance(variables, Mapping):
            raise TypeError("variables must be a mapping of variable name -> value")
        items = {**variables, **kwargs}
        if not items:
            warnings.warn("Empty struct provided. No variables added.", stacklevel=2)
            return
        for name, value in items.items():
            _check_name(name, "variable")
            if name in self.variables:
                warnings.warn(
                    f'Variable "{name}" already exists.\nUse assignment to modify '
                    f"(e.g., adata.{name} = new_value)",
                    stacklevel=2,
                )
                continue
            object.__setattr__(self, "variables", (*self.variables, name))
            self._cache[name] = value
            self._mark(name, "added")

    def _mark(self, name: str, mod_type: str) -> None:
        stamp = datetime.datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
        self._modified[name] = MatStruct(mod_type=mod_type, last_modified=stamp)

    @property
    def modified(self) -> dict[str, MatStruct]:
        """Variables changed since the last save: name -> ``mod_type`` (``'added'`` or
        ``'modified'``) and ``last_modified`` (``'dd.MM.yyyy-HH:mm:ss'``)."""
        return {k: MatStruct(v) for k, v in self._modified.items()}

    def load_all(self, exclude: str | Iterable[str] = ()) -> WritableAdata:
        """Read every variable not read yet (nothing to read for a file not saved yet)."""
        if self._replace_file or not self.path.is_file():
            return self
        super().load_all(exclude)
        return self

    # -- saving -----------------------------------------------------------------------
    def save_changes(
        self,
        *,
        raw_data_status: Any = None,
        force: bool = False,
        caller: str | None = None,
    ) -> bool:
        """Write the added and modified variables to the file (``Adata.save_changes``).

        A new file (from :meth:`create`) is written as a whole with
        :func:`calliopy.io.save_mat`; in an existing file only the changed variables are
        replaced or added (:func:`calliopy.io.update_mat`). Each save appends one row to
        :data:`MODIFICATION_LOG_VARIABLE` with the fields ``datetime``
        (``'dd-MMM-yyyy HH:mm:ss'``), ``unix_time``, ``machine_name`` (environment variable
        ``COMPUTERNAME``, as MATLAB; empty where it is not set), ``caller``,
        ``class_version``, ``modified_variables`` and ``mod_types`` (``N x 1`` cells).

        Parameters
        ----------
        raw_data_status
            The experiment's ExpLog ``RawDataStatus`` (NaN, 0, 1, 2). When given and the
            file exists, saving is refused for status 2 (raw data deleted) and for
            unexpected values, as MATLAB's lock. ``None``: no check (the ExpLog lookup is
            not ported).
        force
            Skip the raw-data lock (MATLAB ``'force'``).
        caller
            Text for the log's ``caller`` field; default the calling functions (the 8
            innermost, outermost first).

        Returns
        -------
        bool
            True when something was written; False when the file is locked or nothing
            was changed (both with a warning, as in MATLAB).
        """
        is_new = self._replace_file or not self.path.is_file()
        if (not is_new and not force and raw_data_status is not None
                and not _raw_data_available(raw_data_status)):
            warnings.warn(
                "This Adata file is locked for editing - the raw data are already "
                "deleted. Saving aborted.",
                stacklevel=2,
            )
            return False
        if not self._modified:
            if is_new:
                msg = "No variables to save. Use add_variable() before saving a new file."
            else:
                msg = "No variables have been modified. Nothing to save."
            warnings.warn(msg, stacklevel=2)
            return False
        if self._pending_mkdir:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Created directory: %s", self.path.parent)
            self._pending_mkdir = False

        names = list(self._modified)
        entry = MatStruct(
            datetime=_datestr(),
            unix_time=float(datetime.datetime.now().timestamp()),
            machine_name=os.environ.get("COMPUTERNAME", ""),
            caller=caller if caller is not None else _caller_chain(),
            class_version=f"calliopy {_calliopy_version()} (Adata {ADATA_CLASS_VERSION})",
            modified_variables=_log_column(names),
            mod_types=_log_column([self._modified[n].mod_type for n in names]),
        )
        rows: list[MatStruct] = []
        if not is_new and MODIFICATION_LOG_VARIABLE in self.variables:
            rows = _log_rows(self[MODIFICATION_LOG_VARIABLE])
        elif is_new and MODIFICATION_LOG_VARIABLE in self._cache:
            rows = _log_rows(self._cache[MODIFICATION_LOG_VARIABLE])
        rows.append(entry)
        log = np.empty((len(rows), 1), dtype=object)
        for i, r in enumerate(rows):
            log[i, 0] = r

        if is_new:
            data = {n: self._cache[n] for n in self.variables if n != MODIFICATION_LOG_VARIABLE}
            data[MODIFICATION_LOG_VARIABLE] = log
            logger.info("Creating new Adata file on disk: %s", self.path)
            save_mat(self.path, data)
        else:
            data = {n: self._cache[n] for n in names if n != MODIFICATION_LOG_VARIABLE}
            stored = self.info()
            for n, value in data.items():
                if stored.get(n, ("",))[0] == "cell":
                    data[n] = _keep_cell(n, value)
            data[MODIFICATION_LOG_VARIABLE] = log
            update_mat(self.path, data)
        self._cache[MODIFICATION_LOG_VARIABLE] = log
        if MODIFICATION_LOG_VARIABLE not in self.variables:
            object.__setattr__(self, "variables", (*self.variables, MODIFICATION_LOG_VARIABLE))
        self._modified = {}
        self._replace_file = False
        logger.info("Saved %d variables to Adata file: %s", len(names), self.path.name)
        return True

    def __repr__(self) -> str:
        return (
            f"WritableAdata({self.path.name!r}, type={self.type!r}, exp_id={self.exp_id}, "
            f"{len(self.loaded)}/{len(self.variables)} variables loaded, "
            f"{len(self._modified)} unsaved)"
        )


# ---------------------------------------------------------------------------------------
# helpers for the "save(..., '-append')" writers of the other subpackages
# ---------------------------------------------------------------------------------------


def _existing_v73(path: str | PathLike) -> Path:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"adata file not found: {p}")
    return p


def _read_variables(path: Path, names: Sequence[str]) -> dict[str, Any]:
    """The variables among ``names`` that the file holds (missing ones are left out)."""
    with MatFile(path) as mf:
        stored = set(mf.keys())
    present = [n for n in names if n in stored]
    if not present:
        return {}
    return dict(load_mat(path, present))


def _set_check_raw_passed(adata_path: str | PathLike) -> bool:
    """``check_raw.passed = 1`` saved with ``-append`` (``*_alignment_visualize.m``)."""
    p = _existing_v73(adata_path)
    found = _read_variables(p, ["check_raw"])
    if "check_raw" not in found:
        warnings.warn("This file is not checked yet - check first", stacklevel=3)
        return False
    check_raw = MatStruct(found["check_raw"])
    check_raw["passed"] = 1.0
    update_mat(p, {"check_raw": check_raw})
    logger.info("Updated check_raw and saved to adata")
    return True


# ---------------------------------------------------------------------------------------
# save_adata
# ---------------------------------------------------------------------------------------


def save_adata(
    adata_dir: str | PathLike,
    rois: Any,
    bv: Any,
    np: Any,
    template: Any,
    dx: Any,
    dy: Any,
    aux_files: Any,
    fnames: Any,
    nbr_frames: Any,
    mouse_id: str,
    user_id: str,
    act_map: Any,
    template_sec: Any,
    act_map_sec: Any,
    *,
    if_exists: str = "replace",
) -> Path:
    """Write registration results to a new adata file (``01_DataManagement/save_adata.m``).

    The folder ``<adata_dir>/<user_id>/<mouse_id>`` is created when missing and the file
    ``Adata-<stem>.mat`` is written there, ``<stem>`` taken from ``fnames[0]`` as in
    :func:`adata_file_path_from_fnames` with ``wid_rule=True``. The variables, in MATLAB's
    order: ``ROIs``, ``bv``, ``np``, ``template``, ``dx``, ``dy``, ``aux_files``,
    ``fnames``, ``mouse_id``, ``nbr_frames``, ``act_map``, ``template_sec``,
    ``act_map_sec`` (``userID`` is not saved, as in MATLAB). Values are written as given
    (see :func:`calliopy.io.save_mat`); lists become ``1 x N`` cells.

    Parameters
    ----------
    adata_dir
        Root folder of the adata files.
    rois, bv, np, template, dx, dy, aux_files, fnames, nbr_frames, act_map, template_sec, act_map_sec
        The MATLAB arguments of the same names (``rois`` is MATLAB ``ROIs``).
    mouse_id, user_id
        Folder names (MATLAB ``mouse_id``, ``userID``).
    if_exists
        What to do when the file exists: ``"replace"`` (default) writes a new file with
        only these variables; ``"append"`` replaces or adds these variables and keeps the
        others (:func:`calliopy.io.update_mat`); ``"error"`` raises ``FileExistsError``.

    Returns
    -------
    pathlib.Path
        The file written.

    Notes
    -----
    Kept from MATLAB (suspected bug, unverified): MATLAB intends ``-append`` when the file
    exists, but it tests ``exist(path, 'file')`` on the name *without* ``.mat``, which
    finds no ``.mat`` file, and then calls ``save(..., '')``; the existing file is
    therefore written anew (directly or through the ``-v7.3`` retry in ``catch``). The
    default ``if_exists="replace"`` reproduces that result; ``"append"`` is the written
    intent.

    Differences from MATLAB: v7.3 only (MATLAB used its default format and fell back to
    v7.3); a name without ``'_'`` (``'.'``) raises instead of writing ``Adata-.mat``.
    """
    if if_exists not in ("replace", "append", "error"):
        raise ValueError("if_exists must be 'replace', 'append' or 'error'")
    path = adata_file_path_from_fnames(adata_dir, user_id, mouse_id, fnames, wid_rule=True)
    variables = {
        "ROIs": rois,
        "bv": bv,
        "np": np,
        "template": template,
        "dx": dx,
        "dy": dy,
        "aux_files": aux_files,
        "fnames": fnames,
        "mouse_id": mouse_id,
        "nbr_frames": nbr_frames,
        "act_map": act_map,
        "template_sec": template_sec,
        "act_map_sec": act_map_sec,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if if_exists == "error":
            raise FileExistsError(f"adata file exists: {path}")
        if if_exists == "append":
            update_mat(path, variables)
            logger.info("Saved Adata (appended to %s)", path.name)
            return path
    save_mat(path, variables)
    logger.info("Saved Adata %s", path.name)
    return path
