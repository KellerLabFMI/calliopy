"""Querying the ExpLog table: calliope ``02_ExpLog`` (qexp, eid, removeHTML, formatQexpOutput,
expid2adataPath, expid2rawDataStatus, listProjectsExpLog) and the read-only database
queries of ``getExpLogMetaDataFiles``, ``getExpLogPIs``, ``getExpLogProjects``,
``getMiceExpLog`` and ``getVectorDB``.

Everything here reads; nothing writes to the database. Functions that query the database
take a caller-supplied DB-API 2.0 ``connection`` (as :func:`calliopy.explog.get_exp_log`);
functions that work on the ExpLog table take it as ``exp_log`` (default: the cached table).
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from ..io import MatStruct
from ._explog import _query, as_exp_log, get_exp_log

__all__ = [
    "EidResult",
    "ExpLogProjectsResult",
    "Expid2AdataPathResult",
    "QexpResult",
    "VectorDbResult",
    "eid",
    "expid2adata_path",
    "expid2raw_data_status",
    "format_qexp_output",
    "get_exp_log_metadata_files",
    "get_exp_log_pis",
    "get_exp_log_projects",
    "get_mice_exp_log",
    "get_vector_db",
    "list_projects_exp_log",
    "qexp",
    "remove_html",
]

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# small MATLAB helpers (private)
# --------------------------------------------------------------------------------------
def _num2str(v: Any) -> str:
    """MATLAB ``num2str`` for the scalar values found in ExpLog columns."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, (_dt.datetime, _dt.date)):
        return str(v)
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, (bool, int)):
        return str(int(v))
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Inf" if v > 0 else "-Inf"
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        dgt = math.floor(math.log10(abs(v)))
        prec = min(max(dgt + 5, 5), 16)
        return f"{v:.{prec}g}"
    arr = np.asarray(v)
    if arr.size == 0:
        return ""
    if arr.ndim and arr.dtype.kind in "biuf":
        flat = arr.ravel(order="F")
        if np.all(flat == np.round(flat)) and np.all(np.isfinite(flat)):
            ints = [str(int(x)) for x in flat]
            width = max(len(s) for s in ints) + 2
            return "".join(s.rjust(width) for s in ints).lstrip()
        return "  ".join(_num2str(float(x)) for x in flat)
    if arr.ndim == 0:
        return _num2str(arr.item())
    return str(v)


def _str2double(s: Any) -> float:
    """MATLAB ``str2double`` of one value (NaN for anything that is not a number text)."""
    if not isinstance(s, str):
        return math.nan
    t = s.strip()
    if not t or "_" in t:
        return math.nan
    try:
        return float(t)
    except ValueError:
        return math.nan


def _is_numeric(v: Any) -> bool:
    """MATLAB ``isnumeric`` (logical values are not numeric)."""
    if isinstance(v, (bool, np.bool_)):
        return False
    if isinstance(v, (int, float, np.integer, np.floating)):
        return True
    if isinstance(v, np.ndarray):
        return v.dtype.kind in "iuf"
    if isinstance(v, (list, tuple)) and v:
        return all(_is_numeric(x) for x in v)
    return False


def _all_digits(s: str) -> bool:
    """MATLAB ``all(isstrprop(s, 'digit'))`` (true for the empty string)."""
    return all(c.isdigit() for c in s)


_DATE_FORMATS = (
    "%d-%b-%Y %H:%M:%S",
    "%d-%b-%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y",
)


def _to_datetime(v: Any) -> _dt.datetime | None:
    """A date value of the ExpLog (driver datetime, numpy datetime64 or text) as datetime."""
    if v is None:
        return None
    if isinstance(v, _dt.datetime):
        return v.replace(tzinfo=None)
    if isinstance(v, _dt.date):
        return _dt.datetime(v.year, v.month, v.day)
    if isinstance(v, np.datetime64):
        if np.isnat(v):
            return None
        return v.astype("datetime64[us]").astype(_dt.datetime)
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, str):
        t = v.strip()
        if not t:
            return None
        for fmt in _DATE_FORMATS:
            try:
                return _dt.datetime.strptime(t, fmt)
            except ValueError:
                pass
        try:
            return _dt.datetime.fromisoformat(t).replace(tzinfo=None)
        except ValueError:
            raise ValueError(f"could not identify date {v!r}") from None
    raise ValueError(f"could not identify date variable {v!r}")


def _today(today: Any) -> _dt.datetime:
    """MATLAB ``datetime(cellstr(datetime('now','Format','dd-MMM-yyyy')))``: today at 00:00."""
    t = _dt.datetime.now() if today is None else _to_datetime(today)
    return _dt.datetime(t.year, t.month, t.day)


def _scalar(v: Any) -> Any:
    return v.item() if isinstance(v, np.generic) else v


def _regex_hits(column, pattern: str) -> np.ndarray:
    """``~cellfun(@isempty, regexpi(column, pattern, 'once'))`` (case-insensitive search)."""
    if pattern == "":
        return np.zeros(len(column), dtype=bool)  # MATLAB: an empty expression never matches
    rx = re.compile(pattern, re.IGNORECASE | re.DOTALL)
    return np.array([rx.search(_num2str(_scalar(x))) is not None for x in column], dtype=bool)


def _strcmpi(column, value: Any) -> np.ndarray:
    if not isinstance(value, str):
        return np.zeros(len(column), dtype=bool)
    lv = value.lower()
    return np.array([isinstance(x, str) and x.lower() == lv for x in column], dtype=bool)


def remove_html(txt):
    """Remove HTML tags from comment text (calliope ``02_ExpLog/removeHTML.m``).

    Parameters
    ----------
    txt : str or sequence of str
        Text, or several texts (a cell array in MATLAB).

    Returns
    -------
    str or list of str
        The text with every ``<...>`` tag removed, in the type given.
    """
    rx = re.compile(r"<[^>]*>")
    if isinstance(txt, str):
        return rx.sub("", txt)
    return [rx.sub("", t) if isinstance(t, str) else t for t in txt]


# --------------------------------------------------------------------------------------
# eid
# --------------------------------------------------------------------------------------
class EidResult(NamedTuple):
    """Outputs of :func:`eid` (MATLAB ``[field, col]``)."""

    field: str | None
    col: int | None


def eid(search_it, exp_log=None) -> EidResult:
    """Which ExpLog column holds a value (calliope ``02_ExpLog/eid.m``).

    MATLAB scans the table from the last row upwards and returns the column of the first
    hit: for a number, the last matching column of that row (``all(ismember(...))``); for a
    text, the first column whose value equals it, ignoring case.

    Parameters
    ----------
    search_it : str, number or array of numbers
        Value to look for. A text made of digits only is searched as a number.
    exp_log : ExpLog, optional
        Default: :func:`calliopy.explog.get_exp_log` (cached table).

    Returns
    -------
    EidResult
        ``field``: column name, ``col``: its 0-based position among the ExpLog columns;
        both None when nothing matches.

    Differences from MATLAB
    -----------------------
    A number is not compared with the character codes of text values (MATLAB
    ``ismember(65, 'A')`` is true).
    """
    el = as_exp_log(exp_log if exp_log is not None else get_exp_log(suppress_warning=True))
    if isinstance(search_it, str) and _all_digits(search_it):
        search_it = float(search_it)
    fields = list(el.keys())
    cols = [np.asarray(el[f]) for f in fields]
    n = len(cols[0]) if cols else 0
    if isinstance(search_it, str):
        s = search_it.lower()
        for row in range(n - 1, -1, -1):
            for j in range(len(fields)):
                x = cols[j][row]
                if isinstance(x, str) and x.lower() == s:
                    return EidResult(fields[j], j)
        return EidResult(None, None)
    needle = np.atleast_1d(np.asarray(search_it, dtype=np.float64))
    for row in range(n - 1, -1, -1):
        for j in range(len(fields) - 1, -1, -1):
            x = _scalar(cols[j][row])
            if isinstance(x, (str, bytes)) or x is None:
                continue
            try:
                hay = np.atleast_1d(np.asarray(x, dtype=np.float64))
            except (TypeError, ValueError):
                continue
            if needle.size and np.all(np.isin(needle, hay)):
                return EidResult(fields[j], j)
    return EidResult(None, None)


# --------------------------------------------------------------------------------------
# qexp
# --------------------------------------------------------------------------------------
class QexpResult(NamedTuple):
    """Outputs of :func:`qexp` (MATLAB ``[ExpLogMatched, idx]``)."""

    exp_log_matched: Any
    idx: np.ndarray


_SHORTHANDS = (
    (r"acode$", "analysiscode"),
    (r"proj$", "project"),
    (r"loc$", "location"),
    (r"^(stack|site|exp|animal|mouse)(id|ids|s)*$", r"\1id"),
    (r"^comment.*$|^cmt.*$", "comment"),
)


def _shorthand(s: str) -> str:
    for pat, rep in _SHORTHANDS:
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    return s


def _first_digits_after_word(name: str) -> str:
    """``regexpi(name, '(?<=^\\w+)\\d+', 'match', 'once')`` (variable-length look-behind)."""
    for i in range(1, len(name)):
        if name[i].isdigit() and re.fullmatch(r"\w+", name[:i]):
            j = i
            while j < len(name) and name[j].isdigit():
                j += 1
            return name[i:j]
    return ""


def _column_dates(el) -> list:
    return [_to_datetime(_scalar(v)) for v in np.asarray(el["stackdate"], dtype=object)]


def _date_mask(dates, ref, op) -> np.ndarray:
    return np.array([d is not None and op(d, ref) for d in dates], dtype=bool)


def _stacks_age_mask(lookback, el, today) -> np.ndarray:
    """The ``find_stacks_age`` sub-function of qexp.m: stacks dated on or after a date."""
    dates = _column_dates(el)
    if _is_numeric(lookback):
        ref = _today(today) - _dt.timedelta(days=float(np.ravel(lookback)[0]))
    elif isinstance(lookback, (str, _dt.date, np.datetime64)):
        ref = _to_datetime(lookback)
    else:
        raise ValueError("could not identify date variable")
    return _date_mask(dates, ref, lambda a, b: a >= b)


def qexp(
    *args,
    exp_log=None,
    metadata_files=None,
    projects=None,
    raw_data_dir: str | os.PathLike | None = None,
    adata_dir: str | os.PathLike | None = None,
    today=None,
) -> QexpResult:
    """Query the ExpLog with field/value filters (calliope ``02_ExpLog/qexp.m``).

    Arguments come in MATLAB's order: ``field, value`` pairs, optionally followed by one
    output selector. ``qexp('pi', 'hans', 'expid')``, ``qexp(1234, 'comment')`` (a number
    first means ``stackid``), ``qexp('regex', 'day1|day5', 'table')``.

    Field names and the shorthands ``acode``, ``proj``, ``loc``, ``stack``/``exp``/
    ``site``/``animal``(``s``/``id``/``ids``) and ``cmt...`` are translated as in MATLAB,
    on every text argument (values too). A single text argument is completed with the
    column :func:`eid` finds it in.

    Filters on ExpLog columns: a callable (applied to every value, must return a bool),
    ``'/regex/'`` (``'/~regex/'`` negates), numbers (``ismember`` on numeric columns),
    regular expressions on ``animalid``/``strain``/``pi``, exact codes on
    ``analysiscode``, the project (falling back to the metadata-file and project
    descriptions, see ``metadata_files``/``projects``), several texts (any equal,
    ignoring case), else equality ignoring case. Other filters: ``age``/``dateafter``/
    ``newerthan``/``bornafter``/``after`` (days or a date), ``datebefore``/``bornbefore``/
    ``before``/``olderthan``, ``regex``/``regexp``/``cregex`` and ``~regex``/``notregex``/
    ... on the comment, ``isexp``/``stackisexp`` and ``issite``/``stackissite`` (value 0
    negates), ``first``/``last`` N rows, ``add`` (a callable evaluated per ExpID and
    added as a column), ``operator``/``op`` (``'all'`` (default), ``'any'`` or a callable
    called as ``op(mask, 1)``).

    Outputs: none (the matching ExpLog: ``stackid``/``expid``/``siteid`` as integer
    arrays, other columns as arrays), a list of column names (only those columns),
    ``'stackid'``/``'expid'``/``'siteid'`` (unique sorted values), a column name (its
    values; unique sorted for text columns other than ``comment``), ``'table'``/
    ``'overview'``/``'view'`` (all columns as text, comments without HTML and cut at 70
    characters), ``'cell'``/``'forcecell'`` (all columns as object arrays), ``'age'``
    (days since the stack date), ``'fname'``/``'fnames'`` (raw ``_ch525.bin`` paths under
    ``raw_data_dir``), ``'adata'``/``'adataname'``/``'adatapath'`` (adata paths under
    ``adata_dir``), or a callable / ``[callable, *args]`` called as ``f(expid, *args)``
    for every matching ExpID.

    Parameters
    ----------
    *args
        Filters and output as described above. ``'explog', table`` inside the arguments
        is accepted as in MATLAB.
    exp_log : ExpLog, optional
        Default: :func:`calliopy.explog.get_exp_log` (cached table).
    metadata_files, projects : MatStruct, optional
        Tables of :func:`get_exp_log_metadata_files` and :func:`get_exp_log_projects`
        for the project fall-back (MATLAB queries the database). Without them the
        fall-back is skipped.
    raw_data_dir, adata_dir : path, optional
        Needed by the ``'fnames'`` and ``'adata'`` outputs (MATLAB: ``getProjDef`` backup
        destination and ``set_lab_paths``).
    today : date, optional
        Reference date of the age filters (default: today).

    Returns
    -------
    QexpResult
        ``exp_log_matched``: the output described above (None when a callable output
        was run); ``idx``: boolean mask of the matching ExpLog rows.

    Raises
    ------
    ValueError
        An output that cannot be interpreted, a needed directory missing, or an
        ``'@...'`` text that would need MATLAB's ``evalin``.

    Differences from MATLAB
    -----------------------
    - No clipboard (``'v'``) and no base workspace: the ExpLog is an argument.
    - Fixed crashes: ``datebefore`` with a number of days (MATLAB uses an undefined
      variable); ``add`` (MATLAB reads a field named after the function text and fails;
      here the column holds the function's result per ExpID); an ``analysiscode`` text
      value (MATLAB splits it into characters or fails without an output selector; here
      the text, split at ``|``, gives the codes).
    - Text columns searched with a regular expression are converted with ``num2str``
      first (MATLAB fails on non-text values).
    - Unknown outputs raise ``ValueError`` (MATLAB prints and returns nothing).
    - Kept from MATLAB: the shorthand translation applies to values too (a value ending
      in ``proj`` becomes ``...project``; ``mouse`` becomes ``mouseid``, which is no
      column and is ignored with a warning); with the ``'table'`` output ``idx`` is the
      mask of the comments longer than 70 characters among the matching rows (MATLAB
      reuses the variable).
    """
    args = list(args)
    if not args:
        raise ValueError("qexp needs at least one argument")
    args = [_shorthand(a) if isinstance(a, str) else a for a in args]
    a0 = args[0]
    if _is_numeric(a0) or (isinstance(a0, str) and _all_digits(a0)):
        args = ["stackid", *args]

    el_pos = [i for i, a in enumerate(args) if isinstance(a, str) and a.lower() == "explog"]
    if el_pos:
        i = el_pos[0]
        exp_log = args[i + 1]
        del args[i : i + 2]
    if exp_log is None:
        exp_log = get_exp_log(suppress_warning=True)
    el = MatStruct(as_exp_log(exp_log))
    if "stackid" not in el:
        el["stackid"] = el[next(iter(el))]
    n = len(np.asarray(el["stackid"]))

    if args and len(args) < 3 and isinstance(args[0], str) and not _all_digits(args[0]):
        field = eid(args[0], el).field
        if field is not None:
            args = [field, *args]

    output: Any = None
    has_output = False
    if len(args) % 2:
        output = args.pop()
        has_output = True
        if isinstance(output, str) and "@" not in output:
            output = output.lower()

    for c in args[0::2]:
        if not isinstance(c, str):
            raise TypeError(f"qexp: filter names must be text, got {c!r}")
    condis = [c.lower() for c in args[0::2]]
    vals = list(args[1::2])
    for i, v in enumerate(vals):
        if isinstance(v, str):
            d = _str2double(v)
            if not math.isnan(d):
                vals[i] = d

    op: Any = "all"
    op_idx = [i for i, c in enumerate(condis) if re.fullmatch(r"operator|op", c, re.IGNORECASE)]
    if op_idx:
        op = vals[op_idx[0]]
        for i in reversed(op_idx):
            del condis[i], vals[i]

    idx = np.zeros((n, len(condis)), dtype=bool)
    for ind, (f, v) in enumerate(zip(condis, vals, strict=True)):
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], str):
            v = "|".join(str(x) for x in v)
        if f in el:
            col = np.asarray(el[f], dtype=object) if np.asarray(el[f]).dtype == object else np.asarray(el[f])
            if callable(v):
                idx[:, ind] = [bool(v(_scalar(x))) for x in col]
            elif isinstance(v, str) and v.count("/") == 2:
                s = v.replace("/", "")
                if "~" in s:
                    idx[:, ind] = ~_regex_hits(col, s.replace("~", ""))
                else:
                    idx[:, ind] = _regex_hits(col, s)
            elif _is_numeric(v) and col.dtype.kind in "biuf":
                idx[:, ind] = np.isin(col, np.asarray(v, dtype=np.float64))
            elif f in ("animalid", "strain", "pi"):
                idx[:, ind] = _regex_hits(col, _num2str(v))
            elif f == "analysiscode":
                if isinstance(v, str):
                    codes = [c for c in v.split("|")]
                else:
                    codes = [_num2str(x) for x in np.atleast_1d(np.asarray(v, dtype=object))]
                idx[:, ind] = _regex_hits(col, "|".join(f"^{re.escape(c)}$" for c in codes))
            elif f == "project":
                tmp = _strcmpi(col, v if isinstance(v, str) else _num2str(v))
                if not tmp.any():
                    pat = _num2str(v)
                    if metadata_files is None:
                        log.warning("qexp: no project %r; pass metadata_files= (and projects=) to search the "
                                    "metafile and project descriptions", v)
                    else:
                        mf = as_exp_log(metadata_files)
                        found = _regex_hits(np.asarray(mf["description"], dtype=object), pat)
                        ids = sorted({str(x) for x in np.asarray(mf["projectid"], dtype=object)[found]})
                        tmp = _regex_hits(col, "|".join(ids))
                        acodes = [_first_digits_after_word(str(x))
                                  for x in np.asarray(mf["metadataname"], dtype=object)[found]]
                        ac = np.array([str(x).strip() for x in np.asarray(el["analysiscode"], dtype=object)])
                        tmp = tmp & np.isin(ac, acodes)
                        if not tmp.any() and projects is not None:
                            pj = as_exp_log(projects)
                            hit = _regex_hits(np.asarray(pj["description"], dtype=object), pat)
                            pids = np.asarray(pj["projectid"], dtype=object)[np.flatnonzero(hit)]
                            tmp = _regex_hits(col, "|".join(str(x) for x in pids))
                idx[:, ind] = tmp
            elif isinstance(v, (list, tuple)):
                hit = np.zeros(n, dtype=bool)
                for x in v:
                    hit |= _strcmpi(col, x)
                idx[:, ind] = hit
            else:
                idx[:, ind] = _strcmpi(col, _num2str(v))
        else:
            if f in ("age", "dateafter", "newerthan", "bornafter", "after"):
                idx[:, ind] = _stacks_age_mask(v, el, today)
            elif f in ("datebefore", "bornbefore", "before", "olderthan"):
                dates = _column_dates(el)
                if _is_numeric(v):
                    ref = _today(today) - _dt.timedelta(days=float(np.ravel(v)[0]))
                elif isinstance(v, (str, _dt.date, np.datetime64)):
                    ref = _to_datetime(v)
                else:
                    raise ValueError("could not identify date variable")
                idx[:, ind] = _date_mask(dates, ref, lambda a, b: a <= b)
            elif f in ("first", "last", "add"):
                idx[:, ind] = True
            elif f in ("cregex", "regex", "regexp", "notregex", "~regex", "!regex", "~regexp", "~cregex"):
                vv = v if isinstance(v, (list, tuple)) else [v]
                hits = _regex_hits(np.asarray(el["comment"], dtype=object), "|".join(_num2str(x) for x in vv))
                idx[:, ind] = hits if f in ("cregex", "regex", "regexp") else ~hits
            elif f in ("stackisexp", "isexp", "stackissite", "issite"):
                other = "expid" if "exp" in f else "siteid"
                m = np.asarray(el[other], dtype=np.float64) == np.asarray(el["stackid"], dtype=np.float64)
                idx[:, ind] = ~m if (_is_numeric(v) and float(np.ravel(v)[0]) == 0) else m
            else:
                idx[:, ind] = True
                log.warning("Couldn't interpret this argument: %s", f)

    if callable(op):
        mask = np.asarray(op(idx, 1), dtype=bool).ravel()
    elif op == "all":
        mask = idx.all(axis=1)
    elif op == "any":
        mask = idx.any(axis=1)
    else:
        raise ValueError(f"qexp: unknown operator {op!r} (use 'all', 'any' or a callable)")

    if "last" in condis:
        k = int(vals[condis.index("last")])
        if mask.sum() > k:
            keep = np.flatnonzero(mask)[-k:] if k > 0 else np.empty(0, dtype=int)
            mask = np.zeros_like(mask)
            mask[keep] = True
    if "first" in condis:
        k = int(vals[condis.index("first")])
        if mask.sum() > k:
            keep = np.flatnonzero(mask)[:k]
            mask = np.zeros_like(mask)
            mask[keep] = True

    expid = np.asarray(el["expid"])
    for c, f in zip(condis, vals, strict=True):
        if c != "add":
            continue
        if not callable(f):
            raise TypeError("qexp 'add' needs a callable (MATLAB function handles become Python callables)")
        name = getattr(f, "__name__", "added")
        newcol = np.empty(n, dtype=object)
        for i in range(n):
            newcol[i] = np.zeros((0, 0))
        for e in np.unique(expid[mask]):
            res = f(_scalar(e))
            for i in np.flatnonzero(expid == e):
                newcol[i] = res
        el[name] = newcol

    def _col(name):
        a = np.asarray(el[name])
        return a[mask]

    is_callable_out = callable(output) or (
        isinstance(output, (list, tuple)) and output and callable(output[0])
    )
    if not has_output or (isinstance(output, (list, tuple)) and not is_callable_out):
        out = MatStruct()
        for name in el:
            out[name] = _col(name)
        if has_output:
            keep = {str(o) for o in output}
            for name in [k for k in out if k not in keep]:
                del out[name]
        return QexpResult(out, mask)
    if is_callable_out:
        fn, extra = (output, ()) if callable(output) else (output[0], tuple(output[1:]))
        exp_ids = np.unique(expid[mask])
        log.info("executing function %s on expids %s", getattr(fn, "__name__", fn), list(exp_ids))
        for k, e in enumerate(exp_ids):
            try:
                log.info("Now processing [%s] %d/%d", e, k + 1, len(exp_ids))
                fn(_scalar(e), *extra)
            except Exception as err:  # noqa: BLE001 - MATLAB: try/catch, print and continue
                log.error("%s failed because: %s", getattr(fn, "__name__", fn), err)
        return QexpResult(None, mask)
    if not isinstance(output, str):
        raise TypeError(f"Couldn't interpret this output-argument: {output!r}")
    if output in ("stackid", "expid", "siteid"):
        return QexpResult(np.unique(_col(output)), mask)
    if output in ("fname", "fnames"):
        if raw_data_dir is None:
            raise ValueError("the 'fnames' output needs raw_data_dir= (MATLAB: backup_destination of getProjDef)")
        rows = np.flatnonzero(mask)
        pis, ans, sts = (np.asarray(el[k], dtype=object) for k in ("pi", "animalid", "stackid"))
        out = [str(Path(raw_data_dir) / str(pis[r]) / str(ans[r]) / f"S1-T{_num2str(_scalar(sts[r]))}_ch525.bin")
               for r in rows]
        return QexpResult(out, mask)
    if output == "age":
        ref = _today(today)
        ages = [np.nan if d is None else (ref - d).total_seconds() / 86400.0
                for d in (_to_datetime(_scalar(x)) for x in _col("stackdate"))]
        return QexpResult(np.asarray(ages, dtype=np.float64), mask)
    if output in ("table", "overview", "view"):
        out = MatStruct()
        for name in el:
            out[name] = np.array([_num2str(_scalar(x)) for x in _col(name)], dtype=object)
        comments = remove_html([_num2str(_scalar(x)) for x in _col("comment")])
        long = np.array([len(c) > 70 for c in comments], dtype=bool)
        comments = [c[:70] + " ..." if len(c) > 70 else c for c in comments]
        out["comment"] = np.array(comments, dtype=object)
        return QexpResult(out, long)
    if output in ("forcecell", "cell"):
        out = MatStruct()
        for name in el:
            out[name] = _col(name).astype(object)
        return QexpResult(out, mask)
    if output in el:
        vals_out = _col(output)
        if output != "comment" and vals_out.dtype == object and all(isinstance(x, str) for x in vals_out):
            vals_out = np.unique(vals_out.astype(str)).astype(object)
        return QexpResult(vals_out, mask)
    if output in ("adata", "adataname", "adatapath"):
        if adata_dir is None:
            raise ValueError("the 'adata' output needs adata_dir= (MATLAB: set_lab_paths)")
        rows = np.flatnonzero(mask)
        pis, ans = (np.asarray(el[k], dtype=object) for k in ("pi", "animalid"))
        paths = {str(Path(adata_dir) / str(pis[r]) / str(ans[r]) / f"Adata-S1-T{_num2str(_scalar(expid[r]))}.mat")
                 for r in rows}
        return QexpResult(sorted(paths), mask)
    if "@" in output:
        raise ValueError("qexp: '@...' outputs need MATLAB's base workspace; pass a callable instead")
    raise ValueError(f"Couldn't interpret this output-argument: {output}")


def format_qexp_output(qexp_output) -> MatStruct:
    """Bring a :func:`qexp` result back into the ExpLog layout (``formatQexpOutput.m``).

    MATLAB turns the row vectors ``stackid``/``expid``/``siteid`` of qexp back into cell
    columns so that the result can be passed to ``generate_proj_meta``. In calliopy both
    layouts are 1-D columns; this returns :func:`calliopy.explog.as_exp_log` of the
    result.
    """
    return as_exp_log(qexp_output)


# --------------------------------------------------------------------------------------
# ExpID lookups
# --------------------------------------------------------------------------------------
class Expid2AdataPathResult(NamedTuple):
    """Outputs of :func:`expid2adata_path` (MATLAB ``[full_path, user_id, mouse_id]``)."""

    full_path: Path
    user_id: str
    mouse_id: str


def _check_exp_id(exp_id) -> int:
    if isinstance(exp_id, bool) or float(exp_id) != int(exp_id) or int(exp_id) <= 0:
        raise ValueError("exp_id must be a positive integer")
    return int(exp_id)


def expid2adata_path(exp_id: int, adata_dir: str | os.PathLike, *, exp_log=None,
                     connection=None) -> Expid2AdataPathResult:
    """Adata file path, user and mouse of an experiment (``02_ExpLog/expid2adataPath.m``).

    Parameters
    ----------
    exp_id : int
        Positive experiment ID.
    adata_dir : path
        Adata root (MATLAB ``set_lab_paths``).
    exp_log : ExpLog, optional
        Table to look the ExpID up in (default: the cached table).
    connection : DB-API 2.0 connection, optional
        Run MATLAB's single-experiment query on the database instead.

    Returns
    -------
    Expid2AdataPathResult
        ``<adata_dir>/<pi>/<animalid>/Adata-S1-T<exp_id>.mat``, the PI and the animal.

    Raises
    ------
    LookupError
        The ExpID is not in the ExpLog.
    """
    exp_id = _check_exp_id(exp_id)
    if connection is not None:
        data = _query(connection, "SELECT Experiments.expid, Animals.pi, Sites.animalid FROM Experiments "
                                  "INNER JOIN Sites ON Experiments.siteid = Sites.siteid "
                                  "INNER JOIN Animals ON Sites.animalid = Animals.animalid "
                                  f"WHERE Experiments.expid = {exp_id}")
        rows = np.arange(len(data.get("expid", [])))
    else:
        data = as_exp_log(exp_log if exp_log is not None else get_exp_log(suppress_warning=True))
        rows = np.flatnonzero(np.asarray(data["expid"], dtype=np.float64) == exp_id)
    if rows.size == 0:
        raise LookupError(f"No data found for expID {exp_id} in ExpLog")
    user_id = str(np.asarray(data["pi"], dtype=object)[rows[0]])
    mouse_id = str(np.asarray(data["animalid"], dtype=object)[rows[0]])
    return Expid2AdataPathResult(Path(adata_dir) / user_id / mouse_id / f"Adata-S1-T{exp_id}.mat",
                                 user_id, mouse_id)


def expid2raw_data_status(exp_id: int, *, exp_log=None, connection=None):
    """RawDataStatus of an experiment (``02_ExpLog/expid2rawDataStatus.m``).

    Returns the value of the ``rawdatastatus`` column for ``exp_id`` (from the ExpLog
    table, or from the database when ``connection`` is given), or None when the ExpID
    does not exist (MATLAB ``[]``).
    """
    exp_id = _check_exp_id(exp_id)
    if connection is not None:
        data = _query(connection, "SELECT Experiments.expid, Experiments.rawdatastatus FROM Experiments "
                                  f"WHERE Experiments.expid = {exp_id}")
        rows = np.arange(len(data.get("expid", [])))
    else:
        data = as_exp_log(exp_log if exp_log is not None else get_exp_log(suppress_warning=True))
        rows = np.flatnonzero(np.asarray(data["expid"], dtype=np.float64) == exp_id)
    if rows.size == 0:
        return None
    if "rawdatastatus" not in data:
        raise LookupError("the ExpLog has no rawdatastatus column")
    return _scalar(np.asarray(data["rawdatastatus"])[rows[0]])


# --------------------------------------------------------------------------------------
# read-only database tables
# --------------------------------------------------------------------------------------
def _select_rows(table: MatStruct, rows) -> MatStruct:
    out = MatStruct()
    for k, v in table.items():
        out[k] = np.asarray(v)[rows]
    return out


SQL_METADATA_FILES = ("SELECT MetaDataFiles.MetaDataName, MetaDataFiles.ProjectID, MetaDataFiles.Description, "
                      "MetaDataFiles.RecordingLocation, MetaDataFiles.StimulationLocation, "
                      "MetaDataFiles.Paradigm, MetaDataFiles.Marker FROM MetaDataFiles")
SQL_PIS = "SELECT PIs.PI, PIs.active FROM PIs"
SQL_PROJECTS = ("SELECT Projects.ProjectID, Projects.name, Projects.PIs, Projects.Description, "
                "Projects.StartDate, Projects.Status FROM Projects")
SQL_MICE = ("SELECT Animals.AnimalID, Animals.pi, Animals.gender, Animals.DoB, Animals.DoD, Animals.Source, "
            "Animals.strain, Animals.VivariumID, Animals.PyratID FROM Animals")
SQL_VECTORS = ("SELECT Vector.ID, Sourcelab, GroupID, VirusType.TypeName as virus, Serotype.TypeName as SeroType, "
               "Promotor.PromotorName as promotor, InsertName, Aliquots, Titer, ProductionDate, Comment "
               "FROM VectorInvDB.dbo.Vector "
               "INNER JOIN VectorInvDB.dbo.SeroType ON Serotype.ID=Vector.SeroTypeID "
               "INNER JOIN VectorInvDB.dbo.Promotor ON Promotor.ID=Vector.PromotorID "
               "INNER JOIN VectorInvDB.dbo.VirusType ON VirusType.ID=Vector.VirusTypeID ORDER BY ID")
SQL_VECTORS_EXTENDED = (
    "SELECT Vector.ID, Sourcelab, Category, VirusType.TypeName as virus, Serotype.TypeName as SeroType, "
    "Promotor.PromotorName as promotor, InsertName, Aliquots, Volume, Titer, TiterMethod.MethodName as TiterMethod, "
    "ProductionDate, Comment, DNATemplate, CellLine, TemplateSource, TemplatePrepMethod, Source, FreezerBox, "
    "ClearedForPublication FROM VectorInvDB.dbo.Vector "
    "INNER JOIN VectorInvDB.dbo.SeroType ON Serotype.ID=Vector.SeroTypeID "
    "INNER JOIN VectorInvDB.dbo.Promotor ON Promotor.ID=Vector.PromotorID "
    "INNER JOIN VectorInvDB.dbo.TiterMethod ON TiterMethod.ID=Vector.TiterMethodID "
    "INNER JOIN VectorInvDB.dbo.VirusType ON VirusType.ID=Vector.VirusTypeID ORDER BY ID")
SQL_ALIQUOT_REQUESTS = "SELECT VectorID, Email, Created FROM VectorInvDB.dbo.AliquotRequest"


class ExpLogProjectsResult(NamedTuple):
    """Outputs of :func:`get_exp_log_projects` (MATLAB ``[ExpLogProjects, projectInfo]``)."""

    exp_log_projects: Any
    project_info: Any


def get_exp_log_projects(connection, *, uname: str = "", project: str = "") -> ExpLogProjectsResult:
    """All projects of the ExpLog database (``02_ExpLog/getExpLogProjects.m``).

    Parameters
    ----------
    connection : DB-API 2.0 connection
        Open connection to the ExpLog database.
    uname : str, optional
        Return only the project IDs whose PI list contains this user name.
    project : str, optional
        Also return the details of the first project whose ID contains this text
        (``'none'`` or empty: no details).

    Returns
    -------
    ExpLogProjectsResult
        ``exp_log_projects``: the Projects table (columns ``projectid``, ``name``,
        ``pis``, ``description``, ``startdate``, ``status``), or with ``uname`` an object
        array of project IDs; ``project_info``: MatStruct (``projectID``, ``name``,
        ``pis``, ``description``, ``startdate``, ``status``) or ``''``.

    Differences from MATLAB
    -----------------------
    When several project IDs contain ``project`` the first is used (MATLAB fails); no
    match raises ``LookupError`` (MATLAB fails on the empty index).
    """
    projects = _query(connection, SQL_PROJECTS)
    info: Any = ""
    if project and project != "none":
        hits = [i for i, p in enumerate(np.asarray(projects["projectid"], dtype=object)) if project in str(p)]
        if not hits:
            raise LookupError(f"no project matching {project!r}")
        i = hits[0]
        info = MatStruct(projectID=project)
        for k in ("name", "pis", "description", "startdate", "status"):
            info[k] = _scalar(np.asarray(projects[k], dtype=object)[i])
    if uname:
        pis = np.asarray(projects["pis"], dtype=object)
        rows = [i for i, p in enumerate(pis) if uname in _num2str(_scalar(p))]
        return ExpLogProjectsResult(np.asarray(projects["projectid"], dtype=object)[rows], info)
    return ExpLogProjectsResult(projects, info)


def get_exp_log_metadata_files(connection, *, uname: str = "") -> MatStruct:
    """The MetaDataFiles table of the ExpLog database (``getExpLogMetaDataFiles.m``).

    Columns ``metadataname``, ``projectid``, ``description``, ``recordinglocation``,
    ``stimulationlocation``, ``paradigm``, ``marker``. With ``uname``, only the rows of
    that user's projects (:func:`get_exp_log_projects`).
    """
    table = _query(connection, SQL_METADATA_FILES)
    if uname:
        projects = get_exp_log_projects(connection, uname=uname).exp_log_projects
        rows = np.flatnonzero(np.isin(np.asarray(table["projectid"], dtype=object).astype(str),
                                      np.asarray(projects, dtype=object).astype(str)))
        table = _select_rows(table, rows)
    return table


def get_exp_log_pis(connection, select: str | None = None) -> np.ndarray:
    """PI user names of the ExpLog database (``02_ExpLog/getExpLogPIs.m``).

    The active PIs, or all with ``select='all'``. Returned as an object array of str.
    """
    table = _query(connection, SQL_PIS)
    pis = np.asarray(table["pi"], dtype=object)
    if select is None:
        active = np.array([bool(_scalar(a)) for a in np.asarray(table["active"], dtype=object)])
        return pis[active]
    if select == "all":
        return pis
    raise ValueError("select must be None or 'all'")


def _is_null(v) -> bool:
    v = _scalar(v)
    return v is None or (isinstance(v, str) and v == "") or (isinstance(v, float) and math.isnan(v))


def get_mice_exp_log(connection, *, uname: str = "") -> MatStruct:
    """The Animals table of the ExpLog database (``02_ExpLog/getMiceExpLog.m``).

    With ``uname``: only that user's living animals (no date of death) that are not
    human subjects.
    """
    table = _query(connection, SQL_MICE)
    if uname:
        pi = np.asarray(table["pi"], dtype=object)
        dod = np.asarray(table["dod"], dtype=object)
        strain = np.asarray(table["strain"], dtype=object)
        rows = [i for i in range(len(pi)) if pi[i] == uname and _is_null(dod[i]) and strain[i] != "human"]
        table = _select_rows(table, rows)
    return table


class VectorDbResult(NamedTuple):
    """Outputs of :func:`get_vector_db` (MATLAB ``[vectorDB, aliquotRequests]``)."""

    vector_db: MatStruct
    aliquot_requests: MatStruct


def get_vector_db(connection, *, extended: bool = False) -> VectorDbResult:
    """The virus vector inventory (``02_ExpLog/getVectorDB.m``).

    Parameters
    ----------
    connection : DB-API 2.0 connection
        Open connection to the vector inventory database (MATLAB ``connectToVectorDB``).
    extended : bool, default False
        Fetch the extended column set.

    Returns
    -------
    VectorDbResult
        The vector table (columns mixing text and numbers converted to text, line breaks
        and tabs removed from ``comment``) and the aliquot requests.

    Differences from MATLAB
    -----------------------
    A failing query raises (MATLAB prints the error and then fails on the empty result).
    """
    vdb = _query(connection, SQL_VECTORS_EXTENDED if extended else SQL_VECTORS)
    requests = _query(connection, SQL_ALIQUOT_REQUESTS)
    for k, col in list(vdb.items()):
        c = np.asarray(col, dtype=object)
        kinds = {type(_scalar(x)) for x in c}
        if str in kinds and (kinds & {int, float}):
            vdb[k] = np.array([_num2str(_scalar(x)) for x in c], dtype=object)
    if "comment" in vdb:
        vdb["comment"] = np.array([re.sub(r"\r|\n|\t", "", _num2str(_scalar(x))) for x in vdb["comment"]],
                                  dtype=object)
    if extended and "sourcelab" in vdb:
        vdb["sourcelab"] = np.array([_num2str(_scalar(x)) for x in vdb["sourcelab"]], dtype=object)
    return VectorDbResult(vdb, requests)


def list_projects_exp_log(exp_log_projects, ustr: str = ".*") -> MatStruct:
    """Projects whose any column matches a regular expression (``listProjectsExpLog.m``).

    Parameters
    ----------
    exp_log_projects : MatStruct
        The Projects table (:func:`get_exp_log_projects`).
    ustr : str, default '.*'
        Regular expression, case-insensitive, searched in every column (as text).

    Returns
    -------
    MatStruct
        The matching rows. The table is also logged (MATLAB prints it).
    """
    table = as_exp_log(exp_log_projects)
    names = list(table)
    n = len(np.asarray(table[names[0]])) if names else 0
    hit = np.zeros(n, dtype=bool)
    for k in names:
        hit |= _regex_hits(np.asarray(table[k], dtype=object), ustr)
    out = _select_rows(table, np.flatnonzero(hit))
    lines = ["\t".join(names)]
    for i in range(int(hit.sum())):
        lines.append("\t".join(_num2str(_scalar(np.asarray(out[k], dtype=object)[i])) for k in names))
    log.info("\n%s", "\n".join(lines))
    return out

