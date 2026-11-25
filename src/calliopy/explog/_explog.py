"""The ExpLog experiment database: calliope ``02_ExpLog/getExpLog.m`` and
``02_ExpLog/read_info_from_ExpLog.m``.

An ExpLog is a :class:`calliopy.io.MatStruct` with one field per database column (lower-case
names as MATLAB's ``adodb_query`` returns them) and one entry per stack: integer columns are
int64 arrays (float64 with NaN where the database has NULL), text columns object arrays of
``str`` (``''`` for NULL), other values (dates) object arrays as the database driver returns
them.

calliopy has no database driver dependency. The caller opens a DB-API 2.0 connection to the
ExpLog database with the driver of their choice and passes it in; ``get_exp_log`` runs
MATLAB's query on it. The offline copy MATLAB falls back to (``ExpLog.mat``) is read with
:func:`calliopy.io.load_mat`.
"""

from __future__ import annotations

import contextlib
import logging
import os
import warnings
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from ..io import MatStruct, load_mat

__all__ = [
    "EXPLOG_SQL",
    "EXPLOG_SQL_EXTENDED",
    "ReadInfoResult",
    "as_exp_log",
    "clear_exp_log_cache",
    "exp_log_rows",
    "get_exp_log",
    "read_info_from_exp_log",
]

log = logging.getLogger(__name__)

EXPLOG_SQL = (
    "SELECT Stacks.stackid, Stacks.expid, Stacks.comment, Stacks.stackdate, Stacks.DataDeleted, "
    "Stacks.Paradigm, Experiments.siteid, Experiments.analysiscode, Experiments.RawDataStatus, "
    "Sites.animalid, Sites.project, Sites.location, Animals.pi, Animals.strain FROM Stacks "
    "INNER JOIN Experiments ON Stacks.expid=Experiments.expid "
    "INNER JOIN Sites ON Experiments.siteid=Sites.siteid "
    "INNER JOIN Animals ON Sites.AnimalID=Animals.AnimalID ORDER BY Stacks.stackid"
)
"""Query of ``getExpLog.m`` (default)."""

EXPLOG_SQL_EXTENDED = (
    "SELECT Stacks.stackid, Stacks.expid, Stacks.comment, Stacks.stackdate, Stacks.DataDeleted, "
    "Stacks.StackTime, Stacks.Paradigm, Experiments.siteid, Experiments.analysiscode, "
    "Experiments.RawDataStatus, Sites.animalid, Sites.project, Sites.location,Sites.depth, "
    "Animals.pi, Animals.gender, Animals.DoB, Animals.DoD, Animals.strain, Animals.VivariumID "
    "FROM Stacks INNER JOIN Experiments ON Stacks.expid=Experiments.expid "
    "INNER JOIN Sites ON Experiments.siteid=Sites.siteid "
    "INNER JOIN Animals ON Sites.AnimalID=Animals.AnimalID ORDER BY Stacks.stackid"
)
"""Query of ``getExpLog.m`` with ``'extended', 1``."""

_cache: dict[str, Any] = {"exp_log": None, "extended": False}


def clear_exp_log_cache() -> None:
    """Forget the cached ExpLog (MATLAB ``evalin('base', 'clear ExpLog')``)."""
    _cache["exp_log"] = None
    _cache["extended"] = False


def _column(values: list) -> np.ndarray:
    """One database column as an array following the ExpLog contract."""
    vals = [v.item() if isinstance(v, np.generic) else v for v in values]
    present = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if present and all(isinstance(v, (bool, int)) for v in present):
        if len(present) == len(vals):
            return np.array(vals, dtype=np.int64)
        return np.array([np.nan if v is None else float(v) for v in vals], dtype=np.float64)
    if present and all(isinstance(v, (bool, int, float)) for v in present):
        return np.array([np.nan if v is None else float(v) for v in vals], dtype=np.float64)
    if not present or all(isinstance(v, (str, bytes)) for v in present):
        out = np.empty(len(vals), dtype=object)
        for i, v in enumerate(vals):
            out[i] = v.decode() if isinstance(v, bytes) else ("" if not isinstance(v, str) else v)
        return out
    out = np.empty(len(vals), dtype=object)
    out[:] = vals
    return out


def as_exp_log(exp_log) -> MatStruct:
    """Bring an ExpLog into the calliopy layout (1-D column arrays, see the module doc).

    Accepts an ExpLog already in that layout, a mapping of columns, or the struct MATLAB
    saves in ``ExpLog.mat`` as :func:`calliopy.io.load_mat` returns it (cell columns, shape
    (n, 1)). Field names are lower-cased. The ``suppresswarning`` marker field MATLAB
    recognises is dropped.
    """
    if exp_log is None:
        raise ValueError("no ExpLog given")
    out = MatStruct()
    for name, col in dict(exp_log).items():
        key = str(name).lower()
        if "suppresswarning" in key:
            continue
        arr = np.asarray(col, dtype=object if isinstance(col, (list, tuple)) else None)
        if arr.dtype != object:
            out[key] = arr.ravel(order="F") if arr.dtype.kind in "biuf" else _column(list(arr.ravel(order="F")))
            continue
        flat = []
        for v in arr.ravel(order="F"):
            if isinstance(v, np.ndarray):
                if v.size == 0:
                    v = None
                elif v.dtype.kind in "US":
                    v = "".join(str(e) for e in v.ravel())
                elif v.size == 1:
                    v = v.ravel()[0].item()
            flat.append(v)
        col_arr = _column(flat)
        if col_arr.dtype == np.float64 and np.all(np.isfinite(col_arr)) and np.all(col_arr == np.round(col_arr)):
            col_arr = col_arr.astype(np.int64)  # MATLAB stores ids as doubles
        out[key] = col_arr
    lengths = {len(v) for v in out.values()}
    if len(lengths) > 1:
        raise ValueError(f"ExpLog columns differ in length: {sorted(lengths)}")
    return out


def _query(connection, sql: str) -> MatStruct:
    cur = connection.cursor()
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        names = [d[0].lower() for d in cur.description]
    finally:
        with contextlib.suppress(Exception):  # closing is best effort, as MATLAB's release
            cur.close()
    out = MatStruct()
    for j, name in enumerate(names):
        out[name] = _column([r[j] for r in rows])
    for name in ("comment", "analysiscode"):
        if name in out and out[name].dtype != object:
            out[name] = _column([None] * len(out[name]))
    return out


def get_exp_log(
    *,
    connection=None,
    explog_file: str | os.PathLike | None = None,
    extended: bool = False,
    force_refresh: bool = False,
    suppress_warning: bool = False,
) -> MatStruct:
    """All stacks of the ExpLog database with their experiment, site and animal fields.

    Port of calliope ``02_ExpLog/getExpLog.m``. MATLAB keeps the table in the base workspace
    and reuses it; this function keeps it in a module cache the same way.

    Parameters
    ----------
    connection : DB-API 2.0 connection, optional
        Open connection to the ExpLog database (any driver). The query is
        :data:`EXPLOG_SQL` or :data:`EXPLOG_SQL_EXTENDED`.
    explog_file : path, optional
        Offline copy ``ExpLog.mat`` (MATLAB's fallback when the database is unreachable).
        Used when no connection is given, or when the query fails.
    extended : bool, default False
        Also fetch StackTime, depth, gender, DoB, DoD and VivariumID.
    force_refresh : bool, default False
        Ignore the cache.
    suppress_warning : bool, default False
        Do not log the 'ExpLog from cache' warning.

    Returns
    -------
    MatStruct
        The ExpLog (layout in the module docstring).

    Raises
    ------
    LookupError
        Nothing cached and no source given, or an extended ExpLog is requested and only a
        plain one is cached.

    Differences from MATLAB
    -----------------------
    - The database connection (``connectToExpLog``, not part of calliope r2003) and the
      offline file are arguments; MATLAB looks for ``ExpLog.mat`` in the current folder.
    - When a source is given it is always read (and cached); MATLAB returns the base
      workspace copy unless ``forceRefresh``.
    - Columns are numpy arrays rather than cell arrays.
    """
    have_source = connection is not None or explog_file is not None
    if not have_source and not force_refresh and _cache["exp_log"] is not None:
        if extended and not _cache["extended"]:
            raise LookupError("an extended ExpLog is needed: pass connection= (or explog_file=)")
        if not suppress_warning:
            log.warning("NC Warning - getting ExpLog from cache")
        return _cache["exp_log"]
    if not have_source:
        raise LookupError("no ExpLog cached: pass connection= (an open ExpLog database connection) or explog_file=")

    exp_log = None
    if connection is not None:
        try:
            exp_log = _query(connection, EXPLOG_SQL_EXTENDED if extended else EXPLOG_SQL)
        except Exception as err:
            if explog_file is None:
                raise
            warnings.warn(f"Couldn't query the ExpLog database ({err}); loading {Path(explog_file).name}",
                          stacklevel=2)
    if exp_log is None:
        data = load_mat(explog_file)
        exp_log = as_exp_log(data.get("ExpLog", data))
        if extended and "dob" not in exp_log:
            warnings.warn("the offline ExpLog has no extended fields", stacklevel=2)
    _cache["exp_log"] = exp_log
    _cache["extended"] = bool(extended) and "dob" in exp_log
    return exp_log


def exp_log_rows(exp_log, exp_id: int) -> np.ndarray:
    """0-based rows of the ExpLog that belong to experiment ``exp_id``."""
    el = as_exp_log(exp_log)
    return np.flatnonzero(np.asarray(el["expid"]).astype(np.float64) == float(exp_id))


class ReadInfoResult(NamedTuple):
    """Outputs of :func:`read_info_from_exp_log` (MATLAB ``[ExpIDinfo, data_dirOut]``)."""

    exp_id_info: MatStruct
    data_dir_out: Path | None


def read_info_from_exp_log(
    exp_id: int,
    suppress_warning: bool = False,
    exp_log=None,
    just_behavior: bool = False,
    *,
    data_dir: str | os.PathLike | None = None,
    main_channel: str = "ch525",
    secondary_channels=(),
) -> ReadInfoResult:
    """Project, user, mouse, site and raw file names of one experiment.

    Port of calliope ``02_ExpLog/read_info_from_ExpLog.m``.

    Parameters
    ----------
    exp_id : int
        Experiment ID.
    suppress_warning : bool, default False
        Do not warn when no data file is found.
    exp_log : ExpLog, optional
        Default: :func:`get_exp_log` (cached table).
    just_behavior : bool, default False
        Only look for the aux (.lvd) files.
    data_dir : path, optional
        Raw data root holding ``<userID>/<mouse_id>/S1-T<stackid>...`` (MATLAB finds it with
        ``get_data_path`` from the lab backup paths). Without it, no file names are listed
        and ``data_dir_out`` is None.
    main_channel, secondary_channels
        Channels of the project (MATLAB reads them from ``ProjectDefaults_<project>`` via
        ``getProjDef``, default 'ch525' and none).

    Returns
    -------
    ReadInfoResult
        ``exp_id_info``: MatStruct with MATLAB's fields ``proj``, ``main_channel``,
        ``secondary_channels``, ``userID``, ``mouse_id``, ``site_id``,
        ``expIDs_in_siteID`` and, with ``data_dir``, the file lists ``fnames``,
        ``wid_fnames``, ``wid_fnames_ch525``, ``wid_fnames_ch610``, ``roi_fnames``,
        ``sec_fnames``, ``aux_files`` (lists of paths, '' for a missing aux file).
        ``data_dir_out``: the data root.

    Differences from MATLAB
    -----------------------
    - ``data_dir`` and the channels are arguments (no ``get_data_path``/``getProjDef``).
    - Unknown ``exp_id``: warning and an empty struct, as MATLAB's message and empty return.
    - Kept from MATLAB: every aux file is listed when the first stack's ``.lvd`` exists;
      the secondary-channel ``.wid`` entry uses the last stack id (MATLAB reuses the loop
      variable of an earlier loop).
    """
    el = as_exp_log(exp_log if exp_log is not None else get_exp_log(suppress_warning=True))
    rows = exp_log_rows(el, exp_id)
    info = MatStruct()
    if rows.size == 0:
        warnings.warn(f"No matching experiment ID {exp_id} in database found.", stacklevel=2)
        return ReadInfoResult(info, None)
    stack_ids = np.asarray(el["stackid"])[rows].astype(np.int64)
    info.proj = str(np.asarray(el["project"])[rows][0])
    info.main_channel = main_channel
    info.secondary_channels = list(secondary_channels)
    info.userID = str(np.asarray(el["pi"])[rows][0])
    info.mouse_id = str(np.asarray(el["animalid"])[rows][0])
    site = np.asarray(el["siteid"])
    info.site_id = int(site[rows][0])
    info.expIDs_in_siteID = np.unique(np.asarray(el["expid"])[site == info.site_id]).astype(np.int64)
    if data_dir is None:
        return ReadInfoResult(info, None)

    root = Path(data_dir)
    fn = root / info.userID / info.mouse_id

    def name(sid, suffix):
        return str(fn / f"S1-T{int(sid)}{suffix}")

    first = stack_ids[0]
    if not just_behavior:
        no_data = True
        mc = main_channel
        if os.path.exists(name(first, f"_{mc}.bin")):
            info.fnames = [name(s, f"_{mc}.bin") for s in stack_ids]
            no_data = False
        if os.path.exists(name(first, ".wii")):
            info.wid_fnames = [name(s, ".wid") for s in stack_ids]
            no_data = False
        if os.path.exists(name(first, "_ch525.wid")) or os.path.exists(name(first, "_ch525.wii")):
            info.wid_fnames_ch525 = [name(s, "_ch525.wid") for s in stack_ids]
            no_data = False
        if os.path.exists(name(first, "_ch610.wid")):
            info.wid_fnames_ch610 = [name(s, "_ch610.wid") for s in stack_ids]
            no_data = False
        if os.path.exists(name(first, f"_{mc}.wid")):
            info.wid_fnames = [name(s, f"_{mc}.wid") for s in stack_ids]
            no_data = False
        if os.path.exists(name(first, ".roi")):
            info.roi_fnames = [name(s, ".roi") for s in stack_ids]
            no_data = False
        if os.path.exists(name(first, f"_{mc}.avi")):
            info.fnames = [name(s, f"_{mc}.avi") for s in stack_ids]
            no_data = False
        if no_data and not suppress_warning:
            warnings.warn("failed to find any data file, maybe the main and secondary channels are wrong",
                          stacklevel=2)
        sec = list(secondary_channels)
        if sec:
            sc = sec[0]
            if os.path.exists(name(first, f"_{sc}.bin")):
                if no_data:
                    info.fnames = [name(s, f"_{sc}.bin") for s in stack_ids]
                    info.sec_fnames = []
                else:
                    info.sec_fnames = [name(s, f"_{sc}.bin") for s in stack_ids]
            elif os.path.exists(name(first, f"_{sc}.wid")):
                info.sec_fnames = [name(stack_ids[-1], f"_{sc}.wid")]
            else:
                if not suppress_warning:
                    log.info("failed to find secondary bin file")
                info.sec_fnames = []
        else:
            info.sec_fnames = []
    lvd = os.path.exists(name(first, ".lvd"))
    info.aux_files = [name(s, ".lvd") if lvd else "" for s in stack_ids]
    return ReadInfoResult(info, root)
