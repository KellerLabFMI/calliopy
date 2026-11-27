"""File listings and data indices: calliope ``06_Library/get_stackid.m``,
``01_DataManagement/getFiles.m``, ``get_adata_index.m``, ``list_all_adata_files.m``,
``list_file_types.m``, ``get_data_path_all.m`` and ``03_Network/get_tape_index.m``.

Every location is an argument. Paths are returned as ``str`` built with the separator of
the running system.
"""

from __future__ import annotations

import datetime as _dt
import fnmatch
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from ..explog import as_exp_log, get_exp_log
from ..io import MatStruct

__all__ = [
    "GetAdataIndexResult",
    "GetDataPathAllResult",
    "GetFilesResult",
    "GetTapeIndexResult",
    "get_adata_index",
    "get_data_path_all",
    "get_files",
    "get_stackid",
    "get_tape_index",
    "list_all_adata_files",
    "list_file_types",
]

log = logging.getLogger(__name__)

_STACKID_RX = re.compile(r"(?<=S1-[PT])\d+", re.IGNORECASE)


def _sscanf_i(s: str) -> float:
    """MATLAB ``sscanf(s, '%i')``: a leading 0 means octal (when the digits allow it)."""
    if len(s) > 1 and s[0] == "0" and all(c in "01234567" for c in s):
        return float(int(s, 8))
    return float(int(s))


def get_stackid(x):
    """Stack IDs in file names (calliope ``06_Library/get_stackid.m``).

    The number after ``S1-T`` or ``S1-P`` (case-insensitive).

    Parameters
    ----------
    x : str or sequence of str
        File names or paths.

    Returns
    -------
    float or numpy.ndarray
        The stack ID, NaN where a name has none; an array of the shape of ``x`` (float64)
        for a sequence, a float for one string.

    Differences from MATLAB
    -----------------------
    A single string is accepted (MATLAB needs a cell array). Kept from MATLAB: the number
    is read with ``sscanf('%i')``, so digits with a leading zero are read as octal
    (``S1-T0123`` gives 83); a leading zero followed by 8 or 9, where MATLAB fails, is
    read as decimal.
    """
    if isinstance(x, (str, os.PathLike)):
        m = _STACKID_RX.search(os.fspath(x))
        return _sscanf_i(m.group(0)) if m else float("nan")
    arr = np.asarray(x, dtype=object)
    out = np.full(arr.shape, np.nan)
    for i, s in np.ndenumerate(arr):
        m = _STACKID_RX.search(os.fspath(s)) if isinstance(s, (str, os.PathLike)) else None
        if m:
            out[i] = _sscanf_i(m.group(0))
    return out


# --------------------------------------------------------------------------------------
# getFiles
# --------------------------------------------------------------------------------------
class GetFilesResult(NamedTuple):
    """Outputs of :func:`get_files` (MATLAB ``[file_info, paths, names, bytes, datenums,
    datestrs, stackids, animals, folder_info]``)."""

    file_info: Any
    paths: list
    names: list
    bytes: np.ndarray
    dates: list
    datestrs: list
    stackids: np.ndarray
    animals: list
    folder_info: list


_ANIMAL_RX = re.compile(r"(?<=[\\/])[A-Za-z0-9]{1,2}_\d+_\d+", re.IGNORECASE)


def _entry(path: Path, is_dir: bool) -> MatStruct:
    st = path.stat()
    date = _dt.datetime.fromtimestamp(st.st_mtime)
    return MatStruct(name=path.name, folder=str(path.parent), date=date,
                     bytes=0 if is_dir else int(st.st_size), isdir=bool(is_dir))


def _listing(root: Path, pattern: str, recurse: bool) -> list[MatStruct]:
    """MATLAB ``dir(fullfile(root, ['**' filesep], pattern))`` without '.' and '..'."""
    folders = [root]
    if recurse:
        folders += sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: str(p).lower())
    out = []
    for folder in folders:
        try:
            children = sorted(folder.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for p in children:
            if pattern == "*.*" or fnmatch.fnmatch(p.name.lower(), pattern.lower()):
                out.append(_entry(p, p.is_dir()))
    return out


def _folder_bytes(path: str) -> int:
    return int(sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()))


def _parse_date(v) -> _dt.datetime:
    if isinstance(v, _dt.datetime):
        return v
    if isinstance(v, _dt.date):
        return _dt.datetime(v.year, v.month, v.day)
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d.%m.%Y"):
        try:
            return _dt.datetime.strptime(str(v).strip(), fmt)
        except ValueError:
            pass
    return _dt.datetime.fromisoformat(str(v).strip())


def _matches_animal(folder: str) -> str:
    m = _ANIMAL_RX.search(folder)
    return m.group(0) if m else ""


def get_files(
    dirs,
    *,
    ext="*",
    recurse: bool = True,
    folders: bool = False,
    strict: bool = False,
    stackid=0,
    lthan: float = np.inf,
    gthan: float = -np.inf,
    regex: str = "",
    olderthan="",
    newerthan="",
    dayeq="",
    newest: int = 0,
    oldest: int = 0,
    just: str = "",
    flatten: bool = False,
) -> GetFilesResult:
    """List files (or folders) below directories, with filters (``01_DataManagement/getFiles.m``).

    Parameters
    ----------
    dirs : path or sequence of paths
        Folder(s) to search. For a sequence every folder is searched on its own and every
        output is a list with one entry per folder (concatenated with ``flatten``).
    ext : str or sequence of str, default '*'
        File-name ending, e.g. ``'.lvd'`` (MATLAB ``['*' ext]``; ``'.*'`` and ``'*'`` list
        everything). A sequence keeps names that contain any of the endings.
    recurse : bool, default True
        Search sub-folders.
    folders : bool, default False
        List folders instead of files (their ``bytes`` are the summed file sizes).
    strict : bool, default False
        Keep only paths of the form ``.../<pi 4-8 chars>/<xx_123456_1...>/...``.
    stackid : int or sequence of int, default 0 (no filter)
        Keep files with these stack IDs (:func:`get_stackid`).
    lthan, gthan : float
        Keep files with ``gthan < bytes < lthan``.
    regex : str
        Keep paths that match this regular expression (case-insensitive).
    olderthan, newerthan, dayeq : str or datetime
        Date filters on the modification date (both bounds: between them, or outside
        them when ``olderthan`` is the earlier date); ``dayeq`` keeps one calendar day.
    newest, oldest : int
        Keep only the newest / oldest N files.
    just : str
        Reduce ``file_info`` to one quantity: 'paths', 'relpaths', 'bytes', 'names',
        'nofiles', 'stackids', 'dates', 'animals' (a 'folder...' prefix also sets
        ``folders``).
    flatten : bool, default False
        With several folders, concatenate the per-folder outputs.

    Returns
    -------
    GetFilesResult
        ``file_info`` (list of MatStruct with ``name``, ``folder``, ``date``, ``bytes``,
        ``isdir``, or the reduced quantity), ``paths``, ``names``, ``bytes``, ``dates``
        (datetime), ``datestrs`` (MATLAB ``dd-mmm-yyyy HH:MM:SS``), ``stackids``,
        ``animals`` (animal IDs found in the folder), ``folder_info`` (the entries that
        were not listed).

    Differences from MATLAB
    -----------------------
    - ``dirs`` is required (MATLAB: the current folder); the 4-letter rig shorthands of
      MATLAB (``s2l``) are not supported.
    - The path filters (``strict``, ``regex``) use the paths of the files left after the
      earlier filters (MATLAB uses the unfiltered path list and fails or misaligns).
    - ``newest``/``oldest`` larger than the number of files keep all files (MATLAB fails).
    - Dates are ``datetime`` (MATLAB datenums); ``folder_info`` leaves out '.' and '..'.
    - Kept from MATLAB: with both ``newest`` and ``oldest``, ``newest`` files are kept at
      each end.
    """
    if isinstance(dirs, (list, tuple)):
        parts = []
        for d in dirs:
            if not d:
                parts.append(GetFilesResult([], [], [], np.empty(0), [], [], np.empty(0), [], []))
            else:
                parts.append(get_files(d, ext=ext, recurse=recurse, folders=folders, strict=strict,
                                       stackid=stackid, lthan=lthan, gthan=gthan, regex=regex,
                                       olderthan=olderthan, newerthan=newerthan, dayeq=dayeq,
                                       newest=newest, oldest=oldest, just=just))
        if flatten:
            cols = []
            for k in range(9):
                acc: list = []
                for p in parts:
                    v = p[k]
                    if isinstance(v, np.ndarray):
                        acc.extend(v.ravel().tolist())
                    elif isinstance(v, list):
                        acc.extend(v)
                    else:
                        acc.append(v)
                cols.append(acc)
            return GetFilesResult(cols[0], cols[1], cols[2], np.asarray(cols[3]), cols[4], cols[5],
                                  np.asarray(cols[6], dtype=np.float64), cols[7], cols[8])
        return GetFilesResult(*[[p[k] for p in parts] for k in range(9)])

    exts: list[str] = []
    if isinstance(ext, (list, tuple)):
        exts = [str(e) for e in ext]
        ext = ".*"
    root = Path(dirs)
    empty = GetFilesResult([], [], [], np.empty(0), [], [], np.empty(0), [], [])
    if not root.is_dir():
        log.info("couldn't find %s", root.name)
        return empty
    pattern = "*.*" if ext in (".*", "*", "*.*") else "*" + str(ext)
    entries = _listing(root, pattern, recurse)
    want_dirs = bool(folders) or re.search("folder", just, re.IGNORECASE) is not None
    keep = [e for e in entries if e.isdir == want_dirs]
    folder_info = [e for e in entries if e.isdir != want_dirs]
    if not keep:
        return empty._replace(folder_info=folder_info)
    file_info = keep
    if exts:
        rx = re.compile("|".join(re.escape(e) for e in exts), re.IGNORECASE)
        file_info = [e for e in file_info if rx.search(e.name)]

    def paths_of(fi):
        return [os.path.join(e.folder, e.name) for e in fi]

    st = np.atleast_1d(np.asarray(stackid, dtype=np.float64))
    if st.size and np.all(st != 0):
        ids = get_stackid([e.name for e in file_info])
        file_info = [e for e, s in zip(file_info, ids, strict=True) if s in st]
    if file_info and strict:
        rx = re.compile(r"[\\/]\w{4,8}[\\/]\w{2}_\d{6}_\d+[\\/]", re.IGNORECASE)
        file_info = [e for e, p in zip(file_info, paths_of(file_info), strict=True) if rx.search(p)]
    if file_info and regex:
        rx = re.compile(regex, re.IGNORECASE)
        file_info = [e for e, p in zip(file_info, paths_of(file_info), strict=True) if rx.search(p)]
    if gthan > -np.inf or lthan < np.inf:
        file_info = [e for e in file_info if gthan < e.bytes < lthan]
    if file_info and (newest or oldest):
        order = sorted(range(len(file_info)), key=lambda i: file_info[i].date)
        nw, od = abs(int(newest)), int(oldest)
        if newest and not oldest:
            sel = order[len(order) - min(nw, len(order)):]
        elif oldest and not newest:
            sel = order[: min(od, len(order))]
        else:
            sel = order[: min(nw, len(order))] + order[len(order) - min(nw, len(order)):]
        file_info = [file_info[i] for i in sel]
    if file_info and (olderthan or newerthan or dayeq):
        if dayeq:
            d = _parse_date(dayeq).date()
            file_info = [e for e in file_info if e.date.date() == d]
        elif olderthan and not newerthan:
            o = _parse_date(olderthan)
            file_info = [e for e in file_info if e.date < o]
        elif newerthan and not olderthan:
            nt = _parse_date(newerthan)
            file_info = [e for e in file_info if e.date > nt]
        else:
            o, nt = _parse_date(olderthan), _parse_date(newerthan)
            if o < nt:
                file_info = [e for e in file_info if e.date < o or e.date > nt]
            else:
                file_info = [e for e in file_info if nt < e.date < o]

    paths = paths_of(file_info)
    names = [e.name for e in file_info]
    if want_dirs:
        nbytes = np.asarray([_folder_bytes(p) for p in paths], dtype=np.float64)
    else:
        nbytes = np.asarray([e.bytes for e in file_info], dtype=np.float64)
    dates = [e.date for e in file_info]
    datestrs = [d.strftime("%d-%b-%Y %H:%M:%S") for d in dates]
    stackids = np.asarray(get_stackid(names), dtype=np.float64) if names else np.empty(0)
    animals = [_matches_animal(e.folder + os.sep) for e in file_info]

    reduced: Any = file_info
    key = re.sub(r"folder[s]*", "", just)
    if key in ("paths", "path"):
        reduced = paths
    elif key in ("relpaths", "relpath"):
        reduced = ["." + os.sep + n for n in names]
    elif key in ("size", "bytes", "byte"):
        reduced = nbytes
    elif key in ("names", "name"):
        reduced = names
    elif key in ("number", "numel", "number_of_files", "nofiles"):
        reduced = len(file_info)
    elif key in ("stacks", "stackids", "stack_id", "stack_ids", "stackid"):
        reduced = stackids
    elif key in ("date", "dates", "mod", "modifiers"):
        reduced = datestrs
    elif key in ("animals", "animal"):
        reduced = animals
    return GetFilesResult(reduced, paths, names, nbytes, dates, datestrs, stackids, animals, folder_info)


# --------------------------------------------------------------------------------------
# index files
# --------------------------------------------------------------------------------------
def _read_index(index_file) -> list[tuple[str, str, float]]:
    """Rows ``path<TAB>date<TAB>bytes`` of an index file (MATLAB ``importdata(f, '\\t')``)."""
    rows = []
    with open(index_file, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                nbytes = float(parts[-1])
            except ValueError:
                continue
            rows.append((parts[0], parts[1], nbytes))
    return rows


def _rooted(rel: str, root) -> str:
    """Replace a leading ``./`` (or ``.\\``) of an index entry by ``root``."""
    parts = [p for p in re.split(r"[\\/]", rel) if p != ""]
    if parts and parts[0] == ".":
        return str(Path(root, *parts[1:]))
    return str(Path(*parts)) if parts else rel


class GetAdataIndexResult(NamedTuple):
    """Outputs of :func:`get_adata_index` (MATLAB ``[file_info, fpath, names, bytes, dstr,
    stackids]``)."""

    file_info: None
    fpath: list
    names: list
    bytes: np.ndarray
    dstr: list
    stackids: np.ndarray


def get_adata_index(index_file, index_root=None, pi: str = "") -> GetAdataIndexResult:
    """Parse the adata index file (``01_DataManagement/get_adata_index.m``).

    Parameters
    ----------
    index_file : path
        The tab-separated index (``index.txt``): one line per file, ``./relative/path``,
        date text, size in bytes.
    index_root : path, optional
        Folder that replaces the leading ``./`` of the entries (default: the folder of
        ``index_file``; MATLAB ``set_lab_paths``).
    pi : str, optional
        Keep only entries whose path contains this text (case-insensitive, taken
        literally; may be ``pi/animal`` with either separator).

    Returns
    -------
    GetAdataIndexResult
        ``file_info`` (None, as MATLAB's placeholder), paths, file names, sizes, date
        texts and stack IDs.

    Differences from MATLAB
    -----------------------
    The index file is an argument (MATLAB reads ``index.txt`` of ``set_lab_paths`` even
    when another index path is given). Lines without a numeric size are skipped.
    """
    rows = _read_index(index_file)
    root = Path(index_file).parent if index_root is None else Path(index_root)
    fpath = [r[0].replace("/", "\\") for r in rows]
    nbytes = [r[2] for r in rows]
    dstr = [r[1] for r in rows]
    if pi:
        rx = re.compile(re.escape(pi.replace("/", "\\")), re.IGNORECASE)
        keep = [i for i, p in enumerate(fpath) if rx.search(p)]
        fpath = [fpath[i] for i in keep]
        nbytes = [nbytes[i] for i in keep]
        dstr = [dstr[i] for i in keep]
    fpath = [_rooted(p, root) for p in fpath]
    names = [re.split(r"[\\/]", p)[-1] for p in fpath]
    return GetAdataIndexResult(None, fpath, names, np.asarray(nbytes, dtype=np.float64), dstr,
                               np.asarray(get_stackid(names), dtype=np.float64) if names else np.empty(0))


class GetTapeIndexResult(NamedTuple):
    """Outputs of :func:`get_tape_index` (MATLAB ``[fpath, names, bytes, datenums, stackids,
    animals, dstr]``)."""

    fpath: list
    names: list
    bytes: np.ndarray
    dates: list
    stackids: np.ndarray
    animals: list
    dstr: list


def get_tape_index(index_file, archive_root, pi: str = "") -> GetTapeIndexResult:
    """Parse the tape archive index (``03_Network/get_tape_index.m``).

    Parameters
    ----------
    index_file : path
        Tab-separated index: ``./relative/path``, ``yyyy-mm-dd HH:MM:SS``, size in bytes
        (MATLAB ``define_tape_index_location``).
    archive_root : path
        Archive folder replacing the leading ``./`` (MATLAB ``define_backup_paths``).
    pi : str, optional
        Keep only entries whose path contains this text (case-sensitive, literal).

    Returns
    -------
    GetTapeIndexResult
        Paths, names, sizes, dates (datetime), stack IDs, animal (the folder holding the
        file) and the date texts as ``dd-Mon-yyyy HH:MM:SS``.

    Differences from MATLAB
    -----------------------
    The date text is always converted to ``dd-Mon-yyyy`` (MATLAB only when six or more
    outputs are requested). The animal of a path with fewer than three folders is ''
    (MATLAB drops it and misaligns the list). Dates are ``datetime`` (MATLAB datenums).
    """
    rows = _read_index(index_file)
    fpath = [r[0] for r in rows]
    nbytes = [r[2] for r in rows]
    dstr = [r[1] for r in rows]
    if pi:
        rx = re.compile(re.escape(pi.replace("\\", "/")))
        keep = [i for i, p in enumerate(fpath) if rx.search(p)]
        fpath = [fpath[i] for i in keep]
        nbytes = [nbytes[i] for i in keep]
        dstr = [dstr[i] for i in keep]
    fpath = [_rooted(p, archive_root) for p in fpath]
    names = [re.split(r"[\\/]", p)[-1] for p in fpath]
    dates = [_dt.datetime.strptime(d.strip(), "%Y-%m-%d %H:%M:%S") for d in dstr]
    animals = []
    for p in fpath:
        parts = re.split(r"[\\/]", p)
        animals.append(parts[-2] if len(parts) >= 3 else "")
    dstr_fixed = [d.strftime("%d-%b-%Y %H:%M:%S") for d in dates]
    return GetTapeIndexResult(fpath, names, np.asarray(nbytes, dtype=np.float64), dates,
                              np.asarray(get_stackid(names), dtype=np.float64) if names else np.empty(0),
                              animals, dstr_fixed)


def list_all_adata_files(adata_dir) -> list[str]:
    """ExpIDs (as text) of every adata file (``01_DataManagement/list_all_adata_files.m``).

    Looks in ``<adata_dir>/<user>/<animal>/*Adata*`` for every user folder that does not
    start with '_' and returns, per file, the name from its last run of digits up to the
    4-character extension (``Adata-S1-T1234.mat`` -> ``'1234'``), in sorted folder order.

    Differences from MATLAB
    -----------------------
    Saving the list (``savefile``) is left to the caller.
    """
    out = []
    root = Path(adata_dir)
    for user in sorted(root.iterdir(), key=lambda p: p.name):
        if user.name.startswith("_") or not user.is_dir():
            continue
        for animal in sorted(user.iterdir(), key=lambda p: p.name):
            if not animal.is_dir():
                continue
            for f in sorted(animal.iterdir(), key=lambda p: p.name):
                if "adata" not in f.name.lower():
                    continue
                starts = [m.start() for m in re.finditer(r"\d+", f.name)]
                if not starts:
                    continue
                out.append(f.name[max(starts): len(f.name) - 4])
    return out


def _exp_info(exp_id: int, el) -> tuple[str, str]:
    rows = np.flatnonzero(np.asarray(el["expid"], dtype=np.float64) == float(exp_id))
    if rows.size == 0:
        raise LookupError(f"ExpID {exp_id} is not in the ExpLog")
    return (str(np.asarray(el["pi"], dtype=object)[rows[0]]),
            str(np.asarray(el["animalid"], dtype=object)[rows[0]]))


def list_file_types(exp_id: int, raw_data_dir, exp_log=None) -> list[str]:
    """File types recorded for an experiment (``01_DataManagement/list_file_types.m``).

    Lists ``<raw_data_dir>/<pi>/<animal>/*<ExpID>*`` and strips ``S1-T<digits>``, an
    optional ``_`` and an optional ``ch`` from the start of each name:
    ``S1-T1234_ch525.bin`` -> ``'525.bin'``, ``S1-T1234.lvd`` -> ``'.lvd'``.

    Returns
    -------
    list of str
        Unique types, sorted.

    Differences from MATLAB
    -----------------------
    ``raw_data_dir`` is the raw data root (MATLAB finds it with ``get_data_path``).
    """
    el = as_exp_log(exp_log if exp_log is not None else get_exp_log(suppress_warning=True))
    user, mouse = _exp_info(exp_id, el)
    folder = Path(raw_data_dir) / user / mouse
    key = str(int(exp_id))
    names = sorted(p.name for p in folder.glob(f"*{key}*")) if folder.is_dir() else []
    types = {re.sub(r"^S1-T\d*(_)?(ch)?", "", n) for n in names if key in n}
    return sorted(types)


class GetDataPathAllResult(NamedTuple):
    """Outputs of :func:`get_data_path_all` (MATLAB ``[data_path, ExpIDinfo]``)."""

    data_path: Path | None
    exp_id_info: MatStruct | None


def get_data_path_all(exp_id: int, search_dirs, exp_log=None, *, main_channel: str = "ch525",
                      secondary_channels=()) -> GetDataPathAllResult:
    """First raw data root that holds an experiment (``01_DataManagement/get_data_path_all.m``).

    Looks for ``<root>/<pi>/<animal>/S1-T<exp_id>.lvd`` in the roots in the given order,
    then for ``S1-T<exp_id>_<main_channel>.bin``.

    Parameters
    ----------
    exp_id : int
        Looked up as a **stack** ID in the ExpLog, as MATLAB does.
    search_dirs : sequence of paths
        Raw data roots, in search order (MATLAB: local RawData share, project backup
        destinations, lab backup paths, archive).
    exp_log : ExpLog, optional
        Default: the cached table.
    main_channel, secondary_channels
        Project channels (MATLAB ``getProjDef``).

    Returns
    -------
    GetDataPathAllResult
        The root (None, with a warning, when nothing is found) and the ExpID info
        (``proj``, ``userID``, ``mouse_id``, ``main_channel``, ``secondary_channels``);
        both None when the stack is not in the ExpLog (warning, as MATLAB).
    """
    el = as_exp_log(exp_log if exp_log is not None else get_exp_log(suppress_warning=True))
    rows = np.flatnonzero(np.asarray(el["stackid"], dtype=np.float64) == float(exp_id))
    if rows.size == 0:
        warnings.warn("No matching entry in database found. Aborting", stacklevel=2)
        return GetDataPathAllResult(None, None)
    info = MatStruct()
    info.proj = str(np.asarray(el["project"], dtype=object)[rows[0]])
    info.userID = str(np.asarray(el["pi"], dtype=object)[rows[0]])
    info.mouse_id = str(np.asarray(el["animalid"], dtype=object)[rows[0]])
    info.main_channel = main_channel
    info.secondary_channels = list(secondary_channels)
    roots = list(dict.fromkeys(str(d) for d in search_dirs))
    for suffix, label in ((".lvd", "lvd"), (f"_{main_channel}.bin", "bin")):
        for r in roots:
            if (Path(r) / info.userID / info.mouse_id / f"S1-T{int(exp_id)}{suffix}").exists():
                return GetDataPathAllResult(Path(r), info)
        warnings.warn(f"no {label} files found for ExpID {int(exp_id)}", stacklevel=2)
    return GetDataPathAllResult(None, info)
