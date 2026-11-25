"""Finding and loading metafiles by keywords (``load_meta``)."""

from __future__ import annotations

import datetime
import logging
import math
import re
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from ._mat import MatFile, load_metafile

__all__ = ["LoadMetaResult", "MetafileMatchError", "load_meta"]

logger = logging.getLogger(__name__)

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_FIELDS = ("projid", "descr", "pis", "fname", "fpath", "datestrs")  # MATLAB struct2cell order
_SPECIAL = re.compile("sortby|nocomplete|expand", re.IGNORECASE)
_SORT_KEY = re.compile(r"[~!\-]*(file|date|pi|proj|desc|name|size|byt)", re.IGNORECASE)
_MAX_TABLE_ROWS = 50


class LoadMetaResult(NamedTuple):
    """Outputs of :func:`load_meta`, named as in MATLAB."""

    proj_meta: Any
    filtered_file: str


class MetafileMatchError(LookupError):
    """The keywords match no metafile or more than one.

    Replaces MATLAB's interactive table. ``candidates`` holds the file names MATLAB
    would list (all files when nothing matched), in display order; ``table`` is the
    same list as text with project, date, PI and description.
    """

    def __init__(self, message: str, candidates: list[str], table: str) -> None:
        super().__init__(message)
        self.candidates = candidates
        self.table = table


# -- ExpLog tables ---------------------------------------------------------------


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, np.ndarray):
        return "" if value.size == 0 else _text(value.ravel()[0])
    return str(value)


def _get_ci(mapping: Mapping, name: str) -> Any:
    for key in mapping:
        if isinstance(key, str) and key.lower() == name:
            return mapping[key]
    return None


def _column(table: Mapping, name: str, table_name: str) -> list[str]:
    col = _get_ci(table, name)
    if col is None:
        raise KeyError(f"exp_log table {table_name!r} has no column {name!r}")
    return [_text(v) for v in np.ravel(np.asarray(col, dtype=object))]


def _exp_log_tables(exp_log: Mapping | None) -> tuple[dict | None, dict | None]:
    if exp_log is None:
        return None, None
    if not isinstance(exp_log, Mapping):
        raise TypeError("exp_log must be a mapping with tables 'metadatafiles' and/or 'projects'")
    meta_tab = _get_ci(exp_log, "metadatafiles")
    proj_tab = _get_ci(exp_log, "projects")
    if meta_tab is None and proj_tab is None:
        raise ValueError("exp_log has neither a 'metadatafiles' nor a 'projects' table")
    meta = proj = None
    if meta_tab is not None:
        meta = {
            "metadataname": _column(meta_tab, "metadataname", "metadatafiles"),
            "projectid": _column(meta_tab, "projectid", "metadatafiles"),
            # MetaLog.description = regexprep(MetaLog.description, '\n|\t|[ ]+', ' ')
            "description": [
                re.sub(r"\n|\t|[ ]+", " ", d) for d in _column(meta_tab, "description", "metadatafiles")
            ],
        }
    if proj_tab is not None:
        proj = {
            "projectid": _column(proj_tab, "projectid", "projects"),
            "pis": _column(proj_tab, "pis", "projects"),
            "description": _column(proj_tab, "description", "projects"),
        }
    return meta, proj


def _first_index(items: list[str], value: str) -> int | None:
    try:
        return items.index(value)
    except ValueError:
        return None


# -- the table of metafiles --------------------------------------------------------


def _datestr(mtime: float) -> str:
    # MATLAB dir() date 'dd-MMM-yyyy HH:mm:ss' with '-' replaced by '. '
    t = datetime.datetime.fromtimestamp(mtime)
    return f"{t.day:02d}. {_MONTHS[t.month - 1]}. {t.year} {t:%H:%M:%S}"


def _meta_descriptions(data: dict, meta: dict | None) -> list[str]:
    descr = [""] * len(data["fname"])
    if meta is not None:
        for i, name in enumerate(data["fname"]):
            j = _first_index(meta["metadataname"], name)
            if j is not None:
                descr[i] = meta["description"][j]
    return descr


def _fill_descriptions(data: dict, proj: dict | None) -> None:
    """Missing or too-short descriptions -> '[project description]' ('' without one)."""
    lower_ids = [p.lower() for p in proj["projectid"]] if proj is not None else []
    for i, d in enumerate(data["descr"]):
        if len(re.sub(r"\s+|\n|\r", "", d)) > 3:
            continue
        j = _first_index(lower_ids, data["projid"][i].lower())
        filled = "[" + (proj["description"][j] if j is not None else "") + "]"
        data["descr"][i] = re.sub(r"\[\]", "", filled)


def _collect(meta_dir: Path, meta: dict | None, proj: dict | None) -> dict:
    files = [p for p in meta_dir.iterdir() if p.is_file() and p.name.lower().endswith(".mat")]
    stats = [p.stat() for p in files]
    order = sorted(range(len(files)), key=lambda i: stats[i].st_mtime)  # sort(datenums), stable
    files = [files[i] for i in order]
    stats = [stats[i] for i in order]
    data: dict = {
        "projid": [""] * len(files),
        "descr": [""] * len(files),
        "pis": [""] * len(files),
        "fname": [p.name for p in files],
        "fpath": [str(p) for p in files],
        "datestrs": [_datestr(s.st_mtime) for s in stats],
        "bytes": [s.st_size for s in stats],
        "mtime": [int(s.st_mtime) for s in stats],
    }
    if meta is not None:
        for i, name in enumerate(data["fname"]):
            j = _first_index(meta["metadataname"], name)
            if j is not None:
                data["descr"][i] = meta["description"][j]
                data["projid"][i] = meta["projectid"][j]
    for i, pid in enumerate(data["projid"]):
        if not re.sub(r"\s+|\n|\r", "", pid):
            m = re.match(r"[a-zA-Z0-9]+(?=_)", data["fname"][i])
            data["projid"][i] = m.group(0).upper() if m else ""
    if proj is not None:
        lower_ids = [p.lower() for p in proj["projectid"]]
        for i, pid in enumerate(data["projid"]):
            j = _first_index(lower_ids, pid.lower())
            if j is not None:
                data["pis"][i] = proj["pis"][j]
    _fill_descriptions(data, proj)
    return data


def _reorder(data: dict, idx: list[int]) -> dict:
    return {k: [v[i] for i in idx] for k, v in data.items()}


def _sort(data: dict, sortby: str) -> dict:
    m = _SORT_KEY.search(sortby)
    if m is None:
        raise ValueError(
            f"sortby={sortby!r}: use date, size, file (name), pi, proj or desc, "
            "optionally prefixed by ~, ! or - for the inverse order"
        )
    key = m.group(1).lower()
    field = {"proj": "projid", "file": "fname", "name": "fname", "pi": "pis", "desc": "descr",
             "size": "bytes", "byt": "bytes"}.get(key, "mtime")
    n = len(data["fname"])
    idx = sorted(range(n), key=lambda i: data[field][i])  # stable, as MATLAB sort
    if re.search(r"[~!\-]", m.group(0)):
        idx = idx[::-1]  # fliplr(idx)
    return _reorder(data, idx)


# -- keywords ----------------------------------------------------------------------


def _request_keyword(text: str) -> list[str]:
    """MATLAB request_keyword: split the keyword string as load_meta does."""
    if re.search(r"\.mat$", text, re.IGNORECASE):
        return [text]
    is_string = re.search(r"['\"]", text) is not None
    is_regex = re.search(r"[|?=]", text) is not None
    if is_regex:
        words = [text]
    elif not is_string:
        words = re.split(" +", re.sub(r"\n|\r|\t", "", text))
    else:
        words = [w for w in re.findall(r"[^*']+", text) if w]
    if not any(_SPECIAL.search(w) for w in words):
        words = words[::-1]  # so first word stays first (MATLAB comment)
    return words


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _parse_options(words: list[str], sortby: str | None, nocomplete: float) -> tuple[list[str], str | None, float]:
    """Options typed among the keywords: 'sortby X', 'nocomplete [n]', 'expand [n]'."""
    words = list(words)
    for pat in ("sortby", "nocomplete", "expand"):
        hits = [i for i, w in enumerate(words) if re.search(pat, w, re.IGNORECASE)]
        if not hits:
            continue
        i = hits[0]
        if pat == "sortby":
            if i + 1 >= len(words):
                raise ValueError("'sortby' among the keywords needs a sort key after it")
            sortby = words[i + 1]
            del words[i : i + 2]
            continue
        has_num = i + 1 < len(words) and _is_number(words[i + 1])
        if pat == "nocomplete":
            nocomplete = float(words[i + 1]) if has_num else 1.0
        del words[i : i + (2 if has_num else 1)]
    return words, sortby, nocomplete


def _search(regex: str, text: str) -> bool:
    if regex == "":
        return False  # MATLAB regexpi(str, '') finds nothing
    return re.search(regex, text, re.IGNORECASE) is not None


def _table(data: dict, rows: list[int]) -> str:
    lines = []
    for n, i in enumerate(rows[:_MAX_TABLE_ROWS], start=1):
        lines.append(
            f"{n:3d} | {data['projid'][i]:<4.4s} | {re.sub(r'.mat', '', data['fname'][i]):<30.30s} | "
            f"{data['datestrs'][i]:<14.14s} | {data['pis'][i]:<14.14s} | {data['descr'][i]:.80s}"
        )
    if len(rows) > _MAX_TABLE_ROWS:
        lines.append(f"... and {len(rows) - _MAX_TABLE_ROWS} more")
    return "\n".join(lines)


def _pi_match(pis: str, wanted: str) -> bool:
    wanted = wanted.lower()
    return pis.lower() == wanted or wanted in {p for p in re.split(r"[,;\s]+", pis.lower()) if p}


def _load(path: str, rd_fields: Any) -> Any:
    with MatFile(path) as mf:
        has_proj_meta = "proj_meta" in mf
    if not has_proj_meta:
        logger.warning("%s has no proj_meta variable; returning None", Path(path).name)
        return None
    size = Path(path).stat().st_size
    logger.info("loading %s (%.2f GB)", Path(path).name, size / 1024**3)
    return load_metafile(path, rd_fields=rd_fields)


def load_meta(
    data: str | Sequence[str] | PathLike | None,
    *,
    meta_dir: str | PathLike,
    desc: str | None = None,
    pi: str | None = None,
    proj: str | None = None,
    exp_log: Mapping | None = None,
    sortby: str = "date",
    complete: bool = True,
    rd_fields: str | Sequence[str] | None = None,
) -> LoadMetaResult:
    """Find a metafile by keywords and load its ``proj_meta``.

    Python port of ``20_MetaData/load_meta.m`` (search and filter part; loading is
    :func:`load_metafile`).

    Parameters
    ----------
    data
        The keywords: one string as typed after ``load_meta`` (``"SST layer2/3"``)
        or a list of words (MATLAB ``varargin``; joined with spaces). Space-separated
        words must all match (AND); each word is a case-insensitive regular
        expression searched in the project id, description, PIs, file name, full
        file path and date string (``'05. Mar. 2021 14:03:12'``) of every file. A
        word containing ``~`` excludes files that match it (all ``~`` are removed
        from the word). A string containing ``|``, ``?`` or ``=`` is one regular
        expression (spaces included); text in quotes is kept together; a string
        ending in ``.mat`` is one file name. ``None`` or ``""`` matches every file.
        An existing file path is loaded directly. ``"sortby <key>"`` and
        ``"nocomplete [0|1]"`` among the words work as in MATLAB and override the
        keyword arguments; ``"expand"`` is accepted and ignored.
    meta_dir
        Folder holding the metafiles (MATLAB: the lab ``_metaData`` folder); only
        ``*.mat`` files directly in it are searched.
    desc
        Additionally require this case-insensitive regular expression in the
        description.
    pi
        Additionally require this PI (case-insensitive; one of the PIs of the
        project, as listed in the ExpLog Projects table).
    proj
        Additionally require this project id (case-insensitive). This also answers
        MATLAB's "Did you mean proj = ...?" question.
    exp_log
        ExpLog tables used to fill in project ids, PIs and descriptions: a mapping
        with ``"metadatafiles"`` (columns ``metadataname``, ``projectid``,
        ``description``; ExpLog table MetaDataFiles) and/or ``"projects"`` (columns
        ``projectid``, ``pis``, ``description``; table Projects), each a mapping of
        1-D column arrays with lower-case SQL column names. Without it, the project
        id comes from the file name prefix (``ECT_...`` -> ``ECT``) and PIs and
        descriptions stay empty.
    sortby
        Order of the candidates in :class:`MetafileMatchError`: ``"date"`` (file
        modification time, default), ``"size"``, ``"file"``/``"name"``, ``"pi"``,
        ``"proj"`` or ``"desc"``; prefix ``~``, ``!`` or ``-`` to invert.
    complete
        Fill missing or too-short (3 characters or less) descriptions with
        ``[<project description>]`` from the Projects table (MATLAB default);
        ``False`` is MATLAB's ``nocomplete``.
    rd_fields
        Passed to :func:`load_metafile` (read only these record fields).

    Returns
    -------
    LoadMetaResult
        ``proj_meta`` as returned by :func:`load_metafile` (``None`` if the file
        has no ``proj_meta`` variable) and ``filtered_file``, the file name.

    Raises
    ------
    MetafileMatchError
        When no file or more than one file matches; ``candidates`` lists them.

    Differences from MATLAB
    -----------------------
    - Never interactive: where MATLAB shows a table and asks for a number or more
      keywords, :class:`MetafileMatchError` is raised with the candidate list.
      MATLAB's yes/no questions ("Did you mean proj = 'X'?", "... PI = ...",
      "... X.mat?") are answered "no"; pass ``proj``/``pi`` or the full file name
      instead.
    - ExpLog tables come from ``exp_log`` instead of the database.
    - ``desc``, ``pi``, ``proj`` and ``rd_fields`` are new.
    - Not ported: the pseudo-command-prompt (``eval``, ``evalin``, ``debug;``,
      ``.help``, ``exit``), the ``proj_meta`` keyword (intersect with the
      ``proj_meta`` in the base workspace), ``expand``, the e-mail reminders for
      undocumented files, loading other variables into the base workspace, setting
      the window title and MATLAB-path lookup of a file name (``exist``/``which``).
    - Inverse sorting works for every key (MATLAB inverts only the date order:
      ``~size`` falls through to the date case). An unknown ``sortby`` raises
      ``ValueError`` (MATLAB silently keeps the date order).
    """
    meta_dir = Path(meta_dir)
    if not meta_dir.is_dir():
        raise NotADirectoryError(f"meta_dir is not a folder: {meta_dir}")
    if isinstance(rd_fields, str):
        rd_fields = [rd_fields]

    # explicit existing file: load directly
    first = data if isinstance(data, (str, PathLike)) or data is None else (data[0] if len(data) else None)
    if isinstance(first, PathLike) or (isinstance(first, str) and first and Path(first).is_file()):
        path = Path(first)
        if not path.is_file():
            raise FileNotFoundError(path)
        return LoadMetaResult(_load(str(path), rd_fields), path.name)

    if data is None:
        words = [".*"]
    elif isinstance(data, str):
        words = [data] if data else [".*"]
    else:
        words = [str(w) for w in data] or [".*"]
    words = list(dict.fromkeys(words))  # unique, first-occurrence order
    words = _request_keyword(" ".join(words))

    meta, proj_tab = _exp_log_tables(exp_log)
    table = _collect(meta_dir, meta, proj_tab)
    if not table["fname"]:
        raise MetafileMatchError(f"no .mat files in {meta_dir}", [], "")

    # direct file name match (case-insensitive)
    if len(words) == 1:
        hits = [i for i, f in enumerate(table["fname"]) if f.lower() == words[0].lower()]
        if len(hits) == 1:
            i = hits[0]
            return LoadMetaResult(_load(table["fpath"][i], rd_fields), table["fname"][i])

    # MATLAB nocomplete: -1 keep the filled descriptions, 1 ExpLog descriptions only, 0 fill
    words, sortby_opt, nocomplete = _parse_options(words, sortby, -1.0 if complete else 1.0)
    if nocomplete == 1:
        table["descr"] = _meta_descriptions(table, meta)
    elif nocomplete == 0:
        _fill_descriptions(table, proj_tab)
    if sortby_opt:
        table = _sort(table, sortby_opt)

    if not words or any(w == "" for w in words):
        words = [".*"]
    n = len(table["fname"])
    rows = []
    for i in range(n):
        ok = True
        for w in words:
            hit = any(_search(w.replace("~", ""), table[f][i]) for f in _FIELDS)
            if hit == ("~" in w):
                ok = False
                break
        if ok:
            rows.append(i)

    if desc is not None:
        rows = [i for i in rows if _search(desc, table["descr"][i])]
    if pi is not None:
        rows = [i for i in rows if _pi_match(table["pis"][i], pi)]
    if proj is not None:
        rows = [i for i in rows if table["projid"][i].lower() == proj.lower()]

    hints = []
    stripped = [re.sub(r"\.mat", "", w) for w in words]
    exact = (
        ("projid", words, words[0], "proj = '{}'", "pass proj='{}'"),
        ("pis", words, words[0], "PI = '{}'", "pass pi='{}'"),
        ("fname", stripped, stripped[0], "{}.mat", "pass '{}.mat' as the keyword"),
    )
    for field, wanted, first, question, hint in exact:
        values = table[field] if field != "fname" else [re.sub(r"\.mat", "", f) for f in table["fname"]]
        match_rows = [i for i in rows if values[i] in wanted]  # exact, case-sensitive
        if len(rows) > 1 and match_rows:
            if first not in values:
                rows = match_rows
            else:
                # MATLAB asks "Did you mean ...?"; without a prompt the answer is no
                hints.append(f"Did you mean {question.format(first)}? {hint.format(first)}")

    if len(rows) == 1:
        i = rows[0]
        return LoadMetaResult(_load(table["fpath"][i], rd_fields), table["fname"][i])

    shown = " && ".join(words[::-1])
    if not rows:
        all_rows = list(range(n))
        text = _table(table, all_rows)
        msg = f"no metafile in {meta_dir} matches {shown!r}; all files:\n{text}"
        raise MetafileMatchError(msg, [table["fname"][i] for i in all_rows], text)
    text = _table(table, rows)
    msg = f"{len(rows)} metafiles match {shown!r}; add keywords or pick a file name:\n{text}"
    if hints:
        msg += "\n" + "\n".join(hints)
    raise MetafileMatchError(msg, [table["fname"][i] for i in rows], text)
