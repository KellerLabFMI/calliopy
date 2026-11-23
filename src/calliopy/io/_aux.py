"""Reading auxiliary recording files: settings, channel lists, analog traces and logs.

=================================  ===================================================
Reader                             MATLAB original (calliope r2003)
=================================  ===================================================
:func:`read_ini`                   ``readini`` (``.ini`` and ``.ach`` files)
:func:`load_ach_chans`             ``load_ach_chans`` (channel labels of an ``.ach``)
:func:`parse_ach`                  ``parse_ach`` (the whole ``.ach`` and a channel table)
:func:`load_lvd`                   ``load_lvd`` with ``get_file_size`` (``.lvd``)
:func:`load_wcp`                   ``load_wcp`` (``.wcp``)
:func:`parse_vrd`                  ``parse_vrd`` (``.vrd``)
:func:`read_sidecar_csv`           ``readtable`` calls in ``load_exp`` (``*_min.csv``,
                                   ``*_beh1.csv``)
=================================  ===================================================

Every reader takes a file path and returns its data. Problems raise exceptions (MATLAB
returned ``status = -1`` or crashed), and nothing is printed (diagnostics go to the
``calliopy.io._aux`` logger). Each docstring lists its *Differences from MATLAB*.
"""

from __future__ import annotations

import csv
import io
import logging
import math
import re
import warnings
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from ._mat import MatStruct

__all__ = [
    "LvdData",
    "SidecarCsv",
    "VrdData",
    "load_ach_chans",
    "load_lvd",
    "load_wcp",
    "parse_ach",
    "parse_vrd",
    "read_ini",
    "read_sidecar_csv",
]

_log = logging.getLogger(__name__)

# A decimal literal as MATLAB writes numbers: 5, -5, 5., .5, 1.5e-3, +2E4.
_NUMBER = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_SPECIAL = re.compile(r"[+-]?(?:inf|nan)", re.IGNORECASE)


# -- shared helpers ------------------------------------------------------------------
def _file_path(path: str | PathLike) -> Path:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {p}")
    return p


def _read_text(path: Path) -> str:
    """Whole file as text, line endings untouched (MATLAB ``fscanf(fi, '%c')``)."""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _split_lines(text: str) -> list[str]:
    """Lines as MATLAB ``fgetl`` returns them: no ``\\n``, no ``\\r``, no last empty line."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [ln.removesuffix("\r") for ln in lines]


def _is_empty(value: Any) -> bool:
    if isinstance(value, str | dict):
        return len(value) == 0
    if isinstance(value, np.ndarray):
        return value.size == 0
    return value is None


def _cell_row(items: list[Any]) -> np.ndarray:
    """A 1xN MATLAB cell row: an object array of shape (1, N)."""
    out = np.empty((1, len(items)), dtype=object)
    for i, item in enumerate(items):
        out[0, i] = item
    return out


def _empty_double() -> np.ndarray:
    return np.empty((0, 0), dtype=np.float64)


def _set_nested(root: MatStruct, parts: list[str], value: Any, *, strict: bool) -> None:
    """``root.a.b.c = value`` with MATLAB's rules for creating nested structs.

    An empty ``[]`` on the way is turned into a struct (as MATLAB does). Going *through*
    a field that already holds a non-struct value raises ValueError. Overwriting an
    existing struct by a plain value raises only if ``strict`` (MATLAB allows it).
    """
    node = root
    for depth, name in enumerate(parts[:-1]):
        child = node.get(name)
        if child is None or (isinstance(child, np.ndarray) and child.size == 0):
            child = MatStruct()
            node[name] = child
        elif not isinstance(child, MatStruct):
            dotted = ".".join(parts[: depth + 1])
            raise ValueError(f"{dotted!r} already holds a value, not a struct")
        node = child
    leaf = parts[-1]
    if strict and isinstance(node.get(leaf), MatStruct):
        raise ValueError(f"{'.'.join(parts)!r} already holds nested fields")
    node[leaf] = value


def _str2double(text: str) -> float:
    """MATLAB ``str2double`` for plain numbers; NaN for anything else."""
    text = text.strip()
    if _NUMBER.fullmatch(text) or _SPECIAL.fullmatch(text):
        return float(text)
    return math.nan


# -- read_ini ------------------------------------------------------------------------
_D2 = "([0-9]{2})"
_INI_DATE_MS = re.compile(rf"([0-9]{{4}})-{_D2}-{_D2} {_D2}:{_D2}:{_D2}\.([0-9]{{3}})")
_INI_DATE_US = re.compile(rf"([0-9]{{4}})-{_D2}-{_D2}_{_D2}-{_D2}-{_D2}")


def _ini_tag(tag: str) -> str:
    tag = tag.strip().replace(" ", "_")
    tag = re.sub(r"\.([0-9])", r"_\1", tag)  # 'foo.1.bar' -> 'foo_1.bar'
    return tag.split("_[", 1)[0]  # 'foo_[1]' -> 'foo'


def _ini_date(text: str) -> datetime | None:
    m = _INI_DATE_MS.fullmatch(text)
    if m:
        y, mo, d, h, mi, s, ms = (int(g) for g in m.groups())
        us = ms * 1000
    else:
        m = _INI_DATE_US.fullmatch(text)
        if not m:
            return None
        y, mo, d, h, mi, s = (int(g) for g in m.groups())
        us = 0
    try:
        return datetime(y, mo, d, h, mi, s, us)
    except ValueError:
        return None


def _ini_value(text: str) -> Any:
    if text == "":
        return _empty_double()
    tokens = [t for t in text.split(" ") if t]
    if all(_NUMBER.fullmatch(t) for t in tokens):
        if len(tokens) == 1:
            return np.float64(tokens[0])
        return np.array(tokens, dtype=np.float64).reshape(1, -1)
    date = _ini_date(text)
    return date if date is not None else text


def read_ini(path: str | PathLike) -> MatStruct:
    """Read an ini-style file of ``tag = value`` lines into a nested struct.

    Reads ``.ini`` files and also ``.ach`` files, which have the same layout. The
    delimiter between tag and value is ``=`` or a tab; it is decided on the first data
    line (a tab if that line contains one, otherwise ``=``). A first line ``[main]``
    is skipped and blank lines are ignored.

    Tags are turned into field names like this, in this order: surrounding blanks are
    removed, blanks inside become ``_``, ``.<digit>`` becomes ``_<digit>`` (so
    ``tag.1.test`` is field ``tag_1`` with sub-field ``test``), and everything from the
    first ``_[`` on is cut (``trace [3]`` becomes ``trace``). Remaining dots build nested
    structs. A tag that occurs twice keeps its first position and its last value.

    Values are turned into Python values like this: all ``"`` are removed and blanks
    trimmed; an empty value is an empty (0, 0) array; text made only of numbers
    separated by blanks is a number (numpy float64) or a (1, N) float array; text of the
    form ``YYYY-MM-DD HH:MM:SS.mmm`` or ``YYYY-MM-DD_HH-MM-SS`` is a naive
    :class:`datetime.datetime`; everything else stays a string.

    Parameters
    ----------
    path
        The ini file.

    Returns
    -------
    MatStruct
        One field per tag, in file order; dotted tags give nested ``MatStruct``.

    Raises
    ------
    FileNotFoundError
        If ``path`` is not a file.
    ValueError
        If a line has no delimiter, a tag is empty, or two tags conflict (``a`` is a
        value on one line and ``a.b`` on another). The message gives the line number.

    Differences from MATLAB
    -----------------------
    * The ``tag`` argument is dropped; take the field from the returned struct (MATLAB
      returned ``[]`` for a missing tag, Python raises ``KeyError`` as for any dict).
    * MATLAB ran every value made of the characters ``0-9 e E + - .`` through ``eval``.
      Here only plain numbers separated by blanks are numbers, so ``2024-01-05`` stays
      a string (MATLAB: 2018), ``1+2`` stays a string (MATLAB: 3), and ``1 - 2`` stays a
      string (MATLAB: -1). A value like ``1.2.3`` is a string; MATLAB dropped the field.
    * MATLAB split a line on *every* ``=`` and kept only the text between the first and
      the second. Here the value is everything after the first ``=``. With tabs as
      delimiter the second column is still the value and further columns are ignored,
      as in MATLAB.
    * Dates are ``datetime.datetime`` (MATLAB: ``datenum`` doubles). Only the two formats
      above are recognised, with exactly these digit counts.
    * MATLAB silently dropped any line whose ``eval`` failed (tags that are not valid
      field names such as ``frame-rate``, values containing ``'``, a tag used both as
      value and as struct). Python keeps tags with any characters as plain keys (reach
      them with ``s["frame-rate"]``), keeps values containing ``'``, and raises for
      conflicting tags.
    * MATLAB crashed on blank lines and on lines without delimiter, and returned nothing
      useful for an empty file. Here blank lines are skipped, a line without delimiter
      raises ``ValueError``, and an empty file gives an empty struct.
    * ``.<digit>`` is replaced with a proper regular expression. MATLAB used the matched
      text itself as a pattern, in which ``.`` matches any character, so a tag such as
      ``x11.1`` was mangled to ``_11_1``.
    * Dotted tags give nested structs as in MATLAB; 1x1 numbers are numpy scalars and
      numeric vectors keep MATLAB's (1, N) shape (the porting guide's conventions).

    Retained from MATLAB: removing *all* ``"`` from values (so ``"123"`` becomes the
    number 123), cutting tags at ``_[``, and skipping a first line ``[main]``.
    """
    p = _file_path(path)
    lines = _split_lines(_read_text(p))
    first_no = 1
    if lines and lines[0] == "[main]":
        lines = lines[1:]
        first_no = 2
    numbered = [(i, ln) for i, ln in enumerate(lines, start=first_no) if ln.strip()]
    result = MatStruct()
    if not numbered:
        return result
    delimiter = "\t" if "\t" in numbered[0][1] else "="
    for no, line in numbered:
        if delimiter == "=":
            tag, sep, value = line.partition("=")
            if not sep:
                raise ValueError(f"{p.name}, line {no}: no '=' in {line!r}")
        else:
            cells = line.split("\t")
            if len(cells) < 2:
                raise ValueError(f"{p.name}, line {no}: no tab in {line!r}")
            tag, value = cells[0], cells[1]
        parts = _ini_tag(tag).split(".")
        if not all(parts):
            raise ValueError(f"{p.name}, line {no}: empty tag or field name in {tag.strip()!r}")
        try:
            value = value.replace('"', "").strip()
            _set_nested(result, parts, _ini_value(value), strict=True)
        except ValueError as err:
            raise ValueError(f"{p.name}, line {no}: {err}") from None
    return result


# -- load_lvd / load_wcp ---------------------------------------------------------------
_LVD_HEADER_BYTES = 32  # four big-endian doubles
_BE_DOUBLE = np.dtype(">f8")
_WCP_FIRST_LVD_EXPID = 133300  # ExpIDs above this are stored in the lvd format
_WCP_LEGACY_SCAN_RATE = 20000.0
_WCP_LEGACY_CHANNELS = 3


class LvdData(NamedTuple):
    """Result of :func:`load_lvd` and :func:`load_wcp` (MATLAB output names in brackets)."""

    data: np.ndarray  #: [num_channels x num_samples] float64 (``data``)
    scan_rate: float  #: samples per second per channel (``scanrateA``)
    num_channels: int  #: number of channels (``numchannels``)
    timestamp: float  #: header time stamp, unit as stored by the acquisition (``timestamp``)
    input_range: float  #: header input range (``inputrange``)


def _get_file_size(path: Path) -> int:
    """Size of a file in bytes (``get_file_size.m``: seek to the end and ``ftell``)."""
    with path.open("rb") as fh:
        fh.seek(0, 2)
        return fh.tell()


def _read_channels(path: Path, offset: int, num_channels: int, *, mmap: bool) -> np.ndarray:
    """Big-endian doubles from ``offset`` on as [num_channels x num_samples].

    Samples are interleaved on disk (all channels of sample 0, then sample 1, ...), which
    is what MATLAB's ``fread(fi, [numchannels inf], 'double')`` fills column by column.
    """
    n_doubles = (_get_file_size(path) - offset) // _BE_DOUBLE.itemsize
    n_samples, leftover = divmod(n_doubles, num_channels)
    if leftover:
        warnings.warn(
            f"{path.name}: {leftover} value(s) after the last complete sample "
            f"({num_channels} channels) were ignored; the file may be truncated",
            RuntimeWarning,
            stacklevel=3,
        )
    if n_samples == 0:
        return np.empty((num_channels, 0), dtype=np.float64)
    if mmap:
        shape = (n_samples, num_channels)
        raw = np.memmap(path, dtype=_BE_DOUBLE, mode="r", offset=offset, shape=shape)
    else:
        raw = np.fromfile(path, dtype=_BE_DOUBLE, count=n_samples * num_channels, offset=offset)
        raw = raw.astype(np.float64).reshape(n_samples, num_channels)
    return raw.T


def load_lvd(path: str | PathLike, *, mmap: bool = False) -> LvdData:
    """Read a ``.lvd`` file (analog aux recording written by the acquisition software).

    File layout, all numbers big-endian IEEE doubles: a 32-byte header (scan rate,
    number of channels, time stamp, input range), then the samples with all channels of
    one sample next to each other.

    Parameters
    ----------
    path
        The ``.lvd`` file.
    mmap
        Return a read-only memory map instead of reading everything into memory. The
        map keeps the on-disk byte order (dtype ``>f8``), which some compiled code
        (numba, ...) does not accept; the default copy is a native float64 array.

    Returns
    -------
    LvdData
        ``(data, scan_rate, num_channels, timestamp, input_range)``. ``data`` has shape
        ``[num_channels x num_samples]``, as in MATLAB. ``scan_rate``, ``timestamp`` and
        ``input_range`` are floats exactly as stored, ``num_channels`` is an int.

    Raises
    ------
    FileNotFoundError
        If ``path`` is not a file.
    ValueError
        If the file is shorter than the header or the stored number of channels is not a
        positive whole number.

    Warns
    -----
    RuntimeWarning
        If the data part does not hold a whole number of samples.

    Differences from MATLAB
    -----------------------
    * Returns an :class:`LvdData` named tuple without ``status``; problems raise (MATLAB:
      ``status = -1`` and empty outputs for an unreadable file; no check of the header).
    * ``fread`` padded a last incomplete sample with zeros. Here the incomplete sample is
      dropped and a ``RuntimeWarning`` is issued, so no invented values enter the data.
    * ``get_file_size`` is used to size the data. Its ``read_until`` flag (file larger
      than 20 minutes at 30 kHz) was overwritten by ``read_until = 0`` in ``load_lvd``
      and never had an effect; it is not ported.
    * ``mmap`` is new. MATLAB's ``numchannels`` is a double; here it is an int.
    * The numbers in the header are not interpreted. In particular ``timestamp`` is kept
      as the stored number, not turned into a ``datetime``.
    """
    p = _file_path(path)
    size = _get_file_size(p)
    if size < _LVD_HEADER_BYTES:
        raise ValueError(f"{p.name}: {size} bytes, shorter than the 32-byte header")
    with p.open("rb") as fh:
        header = np.frombuffer(fh.read(_LVD_HEADER_BYTES), dtype=_BE_DOUBLE)
    scan_rate, n_channels, timestamp, input_range = (float(x) for x in header)
    if not (math.isfinite(n_channels) and n_channels >= 1 and n_channels == int(n_channels)):
        raise ValueError(f"{p.name}: invalid number of channels in header: {n_channels!r}")
    num_channels = int(n_channels)
    data = _read_channels(p, _LVD_HEADER_BYTES, num_channels, mmap=mmap)
    return LvdData(data, scan_rate, num_channels, timestamp, input_range)


def load_wcp(
    path: str | PathLike, *, exp_id: int | None = None, mmap: bool = False
) -> LvdData:
    """Read a ``.wcp`` file: an ``.lvd`` file (newer) or a headerless 3-channel dump (older).

    The format depends on the ExpID, which is the number in the file name
    ``S1-T<ExpID>.wcp``. ExpIDs above 133300 are in the ``.lvd`` layout (see
    :func:`load_lvd`). Older files have no header: big-endian doubles, three channels
    interleaved, with a fixed scan rate of 20000 Hz; ``timestamp`` and ``input_range``
    are then 0.

    Parameters
    ----------
    path
        The ``.wcp`` file, named ``S1-T<ExpID>.wcp`` unless ``exp_id`` is given.
    exp_id
        ExpID to use instead of the one in the file name.
    mmap
        As in :func:`load_lvd`.

    Returns
    -------
    LvdData
        Same fields as :func:`load_lvd`; ``data`` is ``[num_channels x num_samples]``.

    Raises
    ------
    FileNotFoundError
        If ``path`` is not a file.
    ValueError
        If no ExpID is given and the file name does not contain ``S1-T<digits>.<ext>``;
        and the errors of :func:`load_lvd`.

    Warns
    -----
    RuntimeWarning
        If the data part does not hold a whole number of samples.

    Differences from MATLAB
    -----------------------
    * As for :func:`load_lvd`: named tuple without ``status``, errors raise, a last
      incomplete sample is dropped with a warning instead of zero-padded, ``mmap`` is new.
    * The ExpID is read from the file *name* only. MATLAB searched the whole path for the
      first ``S1-T`` and cut the name at ``end-4``, which also picked up directory names
      such as ``...\\S1-T5\\...`` or longer extensions.
    * A name without ``S1-T<ExpID>`` made MATLAB's ``str2num`` return ``[]``, which silently
      selected the old headerless format. Python raises ``ValueError`` unless ``exp_id`` is
      passed; ``exp_id`` is new.
    * ``num_channels`` is an int (MATLAB: double). MATLAB never closed the file in the old
      format.
    """
    p = _file_path(path)
    if exp_id is None:
        m = re.search(r"S1-T([0-9]+)\.[^.]*$", p.name)
        if m is None:
            raise ValueError(
                f"cannot tell the ExpID from the file name {p.name!r} (expected "
                "'S1-T<ExpID>.wcp'); pass exp_id= to choose the file format"
            )
        exp_id = int(m.group(1))
    if exp_id > _WCP_FIRST_LVD_EXPID:
        return load_lvd(p, mmap=mmap)
    data = _read_channels(p, 0, _WCP_LEGACY_CHANNELS, mmap=mmap)
    return LvdData(data, _WCP_LEGACY_SCAN_RATE, _WCP_LEGACY_CHANNELS, 0.0, 0.0)


# -- load_ach_chans / parse_ach --------------------------------------------------------
def load_ach_chans(path: str | PathLike) -> list[str]:
    """Read the channel labels from an ``.ach`` file.

    The file is ini-style (see :func:`read_ini`). The first line is a header and is
    skipped. Then the lines ``auxrec.channel0 = "label"``, ``auxrec.channel1 = "label"``,
    ... are looked for in this order: after the label of channel *k* has been found, the
    next matching line *below* it must be channel *k+1*. Labels are returned as written
    (see :func:`parse_ach` for the renamed ones).

    Parameters
    ----------
    path
        The ``.ach`` file.

    Returns
    -------
    list of str
        The labels of channel 0, 1, 2, ...; the list ends at the first channel that is
        not found in sequence. The list index is the 0-based channel number (MATLAB cell
        index minus one).

    Raises
    ------
    FileNotFoundError
        If ``path`` is not a file.
    ValueError
        If a channel line has no quoted label.

    Differences from MATLAB
    -----------------------
    * Returns a ``list`` of ``str`` instead of a 1xN cell array.
    * MATLAB crashed (index out of range) on any line shorter than ``auxrec.channelN ``,
      for example an empty last line; such lines are skipped.
    * MATLAB crashed on an empty label ``""``; it is returned as ``""``. A channel line
      without any quotes raises ``ValueError``.
    * Same as in MATLAB: the first line is always skipped, and a channel line that comes
      before its predecessor is not found.
    """
    p = _file_path(path)
    labels: list[str] = []
    for no, line in enumerate(_split_lines(_read_text(p))[1:], start=2):
        if not line.startswith(f"auxrec.channel{len(labels)} "):
            continue
        m = re.search(r'"([^"]*)"', line)
        if m is None:
            raise ValueError(f"{p.name}, line {no}: no quoted label in {line!r}")
        labels.append(m.group(1))
    return labels


# Labels written by the acquisition software are renamed to the names used in analysis code.
# Applied in this order and to the whole line (parse_ach.m did the same with regexprep).
_ACH_RENAMES = (
    ("Running", "velM"),
    ("VisualFlow", "velP"),
    ("PerturbationLFM", "PS"),
    ("Perturbation", "PS"),
)
_ASSIGNMENT = re.compile(r"([A-Za-z]\w*(?:\.[A-Za-z]\w*)*)\s*=\s*(.*)", re.DOTALL)


def _parse_literal(text: str) -> Any:
    """The value of the right-hand side of ``x = <text>``: string, number, array or bool."""
    text = text.strip().rstrip(";").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'" and text[0] not in text[1:-1]:
        return text[1:-1]
    if text in ("true", "false"):
        return np.bool_(text == "true")
    bracketed = text.startswith("[") and text.endswith("]")
    rows = [r.strip() for r in (text[1:-1] if bracketed else text).split(";")]
    table = [[t for t in re.split(r"[,\s]+", r) if t] for r in rows]
    if not bracketed and (len(table) != 1 or len(table[0]) != 1):
        raise ValueError(f"cannot interpret value {text!r}")
    if all(not r for r in table):
        return _empty_double()
    if len({len(r) for r in table}) != 1 or not all(_NUMBER.fullmatch(t) for r in table for t in r):
        raise ValueError(f"cannot interpret value {text!r}")
    values = np.array(table, dtype=np.float64)
    return values[0, 0] if values.size == 1 else values


def parse_ach(path: str | PathLike) -> MatStruct:
    """Parse an ``.ach`` file line by line into a struct with the channel table.

    ``.ach`` is the aux-channel file ``S1-T<ExpID>.ach`` next to the ``.lvd`` files: a
    header line, then lines like ``auxrec.scanrate = "20000"`` and
    ``auxrec.channel0 = "Running"``. Each line after the header is read as an assignment
    ``<dotted.name> = <value>``; a value is a double-quoted (or single-quoted) string, a
    number, a bracketed numeric array or ``true``/``false``. Before reading, the labels
    ``Running``, ``VisualFlow``, ``PerturbationLFM`` and ``Perturbation`` are renamed to
    ``velM``, ``velP``, ``PS`` and ``PS``. Use :func:`load_ach_chans` for the labels as
    written, or :func:`read_ini` for a plain reading without renaming (and with numbers
    in quotes converted).

    Parameters
    ----------
    path
        The ``.ach`` file.

    Returns
    -------
    MatStruct
        The fields ``auxrec`` (first; the ``auxrec.*`` lines), any other tags of the file,
        and ``aux_chans`` (last). ``aux_chans`` is an object array of shape
        ``[n_channels x 2]``: column 0 the **1-based** channel number (``channel0`` ->
        1, MATLAB's convention for this field), column 1 the (renamed) label. It is built
        from every ``auxrec`` field whose name contains digits, in file order; the digits
        are the 0-based number. Without such fields it has shape (0, 0).

    Raises
    ------
    FileNotFoundError
        If ``path`` is not a file.

    Notes
    -----
    Lines that cannot be read as an assignment are skipped with a warning on the
    ``calliopy.io._aux`` logger (blank lines are skipped silently).

    Differences from MATLAB
    -----------------------
    * MATLAB's ``parse_ach(pdef, proj, acode, ExpLog)`` also found the file by ExpID or
      by a project/analysis-code search in the ExpLog and could extend an existing
      ``pdef``. Python takes only the file path and returns a new struct; the
      lookups come later with ExpLog access.
    * MATLAB ran each line through ``eval`` as MATLAB code. Python reads only the value
      forms listed above; other MATLAB expressions are skipped like MATLAB's failing
      lines, but with a logged warning (MATLAB: silent). Quoted values stay strings
      (``auxrec.scanrate`` is ``"20000"``, as in MATLAB, where ``load_exp`` applies
      ``str2double``). Strings containing ``'`` are kept (MATLAB: failed).
    * Numbers are numpy float64 scalars or 2-D float arrays ([1 x N] for ``[1 2 3]``).
    * ``aux_chans`` is an object array in place of a cell array, and the channel number
      is an int (MATLAB: double).
    * Without any ``auxrec`` line MATLAB crashed at ``fieldnames``; Python returns an
      empty ``aux_chans`` and keeps ``auxrec`` as an empty array, as MATLAB had set it.
    """
    p = _file_path(path)
    pdef = MatStruct(auxrec=_empty_double())
    for no, line in enumerate(_split_lines(_read_text(p))[1:], start=2):
        line = line.strip()
        if not line:
            continue
        for old, new in _ACH_RENAMES:
            line = line.replace(old, new)
        m = _ASSIGNMENT.fullmatch(line)
        try:
            if m is None:
                raise ValueError("not of the form name = value")
            _set_nested(pdef, m.group(1).split("."), _parse_literal(m.group(2)), strict=False)
        except ValueError as err:
            _log.warning("%s, line %d skipped (%s): %r", p.name, no, err, line)
    rows: list[tuple[Any, Any]] = []
    auxrec = pdef["auxrec"]
    if isinstance(auxrec, MatStruct):
        for name, value in auxrec.items():
            numbers = [int(d) + 1 for d in re.findall(r"[0-9]+", name)]
            if numbers:
                rows.append((numbers[0] if len(numbers) == 1 else np.array(numbers), value))
    aux_chans = np.empty((len(rows), 2) if rows else (0, 0), dtype=object)
    for i, (number, value) in enumerate(rows):
        aux_chans[i, 0] = number
        aux_chans[i, 1] = value
    pdef["aux_chans"] = aux_chans
    return pdef


# -- parse_vrd -------------------------------------------------------------------------
_VRD_KEY = re.compile(r"([A-Za-z]\w*)(?:\(([0-9]+)\))?")
_VRD_NEVER_EXPANDED = frozenset({"m_scale", "r_scale", "x_range", "header"})
_VRD_DATETIME_FORMATS = (
    "%d-%b-%Y %H:%M:%S",
    "%d-%b-%y %H:%M:%S",
    "%d-%b-%Y %H:%M:%S.%f",
    "%d-%b-%y %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%d-%b-%Y",
    "%Y-%m-%d",
)


class VrdData(NamedTuple):
    """Result of :func:`parse_vrd` (MATLAB output names in brackets)."""

    sections: np.ndarray  #: [1 x n_sections] object array of MatStruct (``sections``)
    raw_txt: str  #: the file content, unchanged (``raw_txt``)
    vrd_processed: MatStruct | None  #: per-read table; None if not requested (``vrd_processed``)


def _vrd_key_path(key: str) -> list[tuple[str, int | None]]:
    """``a.b(2).c`` -> [('a', None), ('b', 2), ('c', None)]; indices are 1-based."""
    path = []
    for part in key.split("."):
        m = _VRD_KEY.fullmatch(part)
        if m is None or (m.group(2) is not None and int(m.group(2)) < 1):
            raise ValueError(f"cannot use {key!r} as a field name")
        path.append((m.group(1), None if m.group(2) is None else int(m.group(2))))
    return path


def _grow(cell: list, index: int) -> None:
    cell.extend([None] * (index - len(cell)))


def _vrd_assign(node: MatStruct, path: list[tuple[str, int | None]], value: Any) -> None:
    """``node.a.b{2}.c = value`` with MATLAB's growth of cells (gaps are empty)."""
    for depth, (name, index) in enumerate(path):
        last = depth == len(path) - 1
        if index is None:
            if last:
                node[name] = value
                return
            child = node.get(name)
            if child is None:
                child = node[name] = MatStruct()
        else:
            cell = node.get(name)
            if cell is None:
                cell = node[name] = []
            if not isinstance(cell, list):
                raise ValueError(f"{name!r} already holds a value, not a cell")
            _grow(cell, index)
            if last:
                cell[index - 1] = value
                return
            child = cell[index - 1]
            if child is None:
                child = cell[index - 1] = MatStruct()
        if not isinstance(child, MatStruct):
            raise ValueError(f"{name!r} already holds a value, not a struct")  # noqa: TRY004
        node = child


def _vrd_finalize(value: Any) -> Any:
    """Turn the parser's lists into (1, N) object arrays; gaps become empty arrays."""
    if isinstance(value, list):
        return _cell_row([_empty_double() if v is None else _vrd_finalize(v) for v in value])
    if isinstance(value, MatStruct):
        return MatStruct((k, _vrd_finalize(v)) for k, v in value.items())
    return value


def _vrd_sections(raw: str, path: Path) -> list[MatStruct]:
    sections: list[MatStruct] = []
    sec = 1  # MATLAB's counter: the number of the next header's section
    for no, line in enumerate(re.split(r"\r?\n", raw), start=1):
        if "=" not in line:
            if re.search(r"\[.*\]", line):
                while len(sections) < sec:
                    sections.append(MatStruct())
                sections[sec - 1]["header"] = line
                sec += 1
            continue
        key, _, value = line.partition("=")
        try:
            key_path = _vrd_key_path(key.strip().replace("read(0)", "readfirst"))
        except ValueError as err:
            raise ValueError(f"{path.name}, line {no}: {err}") from None
        index = max(sec - 1, 1)
        while len(sections) < index:
            sections.append(MatStruct())
        try:
            _vrd_assign(sections[index - 1], key_path, value.strip())
        except ValueError as err:
            raise ValueError(f"{path.name}, line {no}: {err}") from None
    return [_vrd_finalize(s) for s in sections]


def _vrd_sections_legacy(raw: str, path: Path) -> list[MatStruct]:
    sections = []
    for block in re.split(r"(?:\r?\n\r?\n)+", raw):
        if not block.strip():
            continue
        section = MatStruct()
        for row in block.split("\n"):
            if "=" not in row:
                continue
            key, _, value = row.partition("=")
            key = re.sub(r"[\r\n\t]", "", key.strip().lower()).replace("(0)", "first")
            value = re.sub(r"[\r\n\t]", "", value.strip())
            try:
                _vrd_assign(section, _vrd_key_path(key), value)
            except ValueError as err:
                _log.warning("%s: could not read line %r (%s)", path.name, row.strip(), err)
        sections.append(_vrd_finalize(section))
    return sections


def _cell_items(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray) and value.dtype == object:
        return list(value.ravel(order="F"))
    return [value]  # a plain string behaves like a 1x1 cell


def _to_datetimes(strings: list[str], fmt: str | None) -> np.ndarray:
    if not strings:
        return np.empty((0, 0), dtype=object)
    for candidate in (fmt,) if fmt else _VRD_DATETIME_FORMATS:
        try:
            parsed = [datetime.strptime(s.strip(), candidate) for s in strings]
        except ValueError:
            continue
        return _cell_row(parsed)
    raise ValueError(
        f"cannot read the 'read' time stamps, e.g. {strings[0]!r}"
        + (f" with format {fmt!r}" if fmt else f" with any of {_VRD_DATETIME_FORMATS}")
        + "; pass datetime_format=... or processed=False"
    )


def _vrd_process(sections: np.ndarray, path: Path, datetime_format: str | None) -> MatStruct:
    secs = list(sections.ravel(order="F"))
    if not secs:
        raise ValueError(f"{path.name}: no sections to process")
    vrd = MatStruct(secs[0])  # keeps the header information of the first section
    with_read = [s for s in secs if "read" in s]
    if len(with_read) < len(secs):
        _log.warning("%s: field 'read' is missing in %d of %d sections", path.name,
                     len(secs) - len(with_read), len(secs))

    def catfield(name: str) -> list[Any]:
        return [v for s in secs if name in s for v in _cell_items(s[name]) if not _is_empty(v)]

    rd = MatStruct()
    rd["read"] = _to_datetimes(catfield("read"), datetime_format)
    vrd.pop("read", None)
    times = [_str2double(v) for v in catfield("traversal_time_in_s")]
    rd["traversal_time_in_s"] = np.array(times, dtype=np.float64).reshape(1, -1)
    vrd.pop("traversal_time_in_s", None)

    template = secs[1] if len(secs) > 1 else secs[0]  # fields are chosen from section 2
    skip = _VRD_NEVER_EXPANDED | set(rd)  # 'read' and 'traversal_time_in_s' are done
    fields = sorted(k for k, v in template.items() if isinstance(v, str) and k not in skip)
    counts = [sum(not _is_empty(v) for v in _cell_items(s["read"])) for s in with_read]

    def section_value(section: MatStruct, name: str) -> str:
        value = section.get(name)
        if not isinstance(value, str):
            raise ValueError(f"{path.name}: section lacks the plain field {name!r}")  # noqa: TRY004
        return value

    for name in fields:
        if re.fullmatch(r"(?:[0-9]+|\.|-)*", template[name]):  # only digits, '.' and '-'
            values = [
                np.full(c, _str2double(section_value(s, name))) for s, c in zip(with_read, counts)
            ]
            rd[name] = np.concatenate(values).reshape(1, -1) if values else _empty_double()
        else:
            rd[name] = _cell_row(
                [section_value(s, name) for s, c in zip(with_read, counts) for _ in range(c)]
            )
        vrd.pop(name, None)
    vrd["rd"] = rd
    return vrd


def parse_vrd(
    path: str | PathLike,
    *,
    legacy: bool = False,
    processed: bool = True,
    datetime_format: str | None = None,
) -> VrdData:
    """Parse a ``.vrd`` text file into sections of ``key = value`` pairs.

    The file is made of sections, each started by a header line in square brackets,
    e.g. ``[05-Mar-19 14:22:10]``, followed by ``key = value`` lines. Keys may carry a
    1-based index, ``read(1) = ...``; all indexed keys of a section with the same name
    are collected in one field, a (1, N) object array in index order with empty arrays
    for missing indices. The key ``read(0)`` is stored as field ``readfirst`` (index 0
    does not exist in MATLAB). All values are strings, with surrounding blanks removed.
    Lines before the first header and the lines under it all belong to the first section.

    ``vrd_processed`` is the data of the first section plus a struct ``rd`` that has one
    entry per ``read`` of the whole file (all sections, empty reads removed):
    ``rd.read`` (datetimes), ``rd.traversal_time_in_s`` (floats), and for every plain
    string field of the second section (of the first if there is only one) the section's
    value repeated for each read of that section: a float array if the field holds only
    digits, ``.`` and ``-``, otherwise an array of strings. The fields ``m_scale``,
    ``r_scale``, ``x_range`` and ``header`` and all indexed fields are not expanded.
    Expanded fields are removed from the first section's data. ``rd`` is the last field.

    Parameters
    ----------
    path
        The ``.vrd`` file.
    legacy
        Old files: sections are separated by blank lines instead of headers and keys are
        lower-cased; ``(0)`` in a key becomes ``first`` for every key. ``vrd_processed``
        is not available (None). See *Differences from MATLAB*.
    processed
        Also build ``vrd_processed`` (MATLAB: only when three outputs were requested).
        Set ``False`` if the file does not have the structure described above.
    datetime_format
        :meth:`datetime.datetime.strptime` format of the ``read`` values. By default a
        list of common formats is tried (``dd-Mon-yyyy HH:MM:SS`` and ISO forms).

    Returns
    -------
    VrdData
        ``(sections, raw_txt, vrd_processed)``. ``sections`` is a (1, N) object array of
        :class:`MatStruct` (empty (0, 0) array for a file without any key); ``raw_txt``
        is the file content including its line endings.

    Raises
    ------
    FileNotFoundError
        If ``path`` is not a file.
    ValueError
        If a key cannot be a field name (not ``name``, ``name(i)`` or dotted combinations
        with i >= 1), or two lines clash; the message gives the line number. With
        ``processed=True`` also if a section lacks a field to expand, the ``read`` values
        cannot be read as datetimes, or there is no section.

    Differences from MATLAB
    -----------------------
    * Only a file path is accepted. MATLAB also took an ExpID (ExpLog lookup) or a
      directory (newest ``*.vrd`` in it); both are left to a later ``calliopy`` function.
    * The ``(`` -> ``{``, ``)`` -> ``}`` and ``read(0)`` -> ``readfirst`` replacements were
      done on the whole line in MATLAB, so values containing parentheses were changed
      too. Here they apply to the key only; values are returned as written.
    * MATLAB split on every ``=`` and kept only the text between the first and second;
      the value is now everything after the first ``=``. Values with ``'`` no longer make
      the (``eval``-based) parser fail.
    * A line with brackets *and* ``=`` is a key/value line here; MATLAB took any line
      matching ``[...]`` as a header, so a value like ``x_range = [0 1]`` started a new
      section. Lines are split at ``\\n`` or ``\\r\\n`` (MATLAB: ``\\r\\n`` only).
    * Keys that MATLAB's ``eval`` could not assign crashed MATLAB; they raise
      ``ValueError``. Only dotted names with optional single indices are supported.
    * ``vrd_processed``: ``rd.read`` is an object array of ``datetime.datetime`` (MATLAB:
      a ``datetime`` array). A missing field in a section to expand raised a MATLAB
      error and raises ``ValueError`` here; a plain string in place of an indexed field
      is read as one entry (MATLAB: error); fields of section 1 that were expanded but
      are absent from it are ignored (MATLAB: ``rmfield`` error). The missing-``read``
      warning goes to the logger instead of the console.
    * ``legacy=True``: MATLAB returned the raw text blocks as ``sections`` and
      discarded the struct it had parsed from them (variable ``res``, never assigned to
      an output); a third output crashed. Here ``sections`` holds the parsed structs,
      one per non-empty block, which is evidently what the code intended; the text is
      in ``raw_txt``. Lines that cannot be read are skipped with a logged warning, as in
      MATLAB. As in MATLAB, tabs and line breaks inside legacy keys and values are
      removed.
    """
    p = _file_path(path)
    raw = _read_text(p)
    if legacy:
        secs = _vrd_sections_legacy(raw, p)
    else:
        secs = _vrd_sections(raw, p)
    sections = _cell_row(secs) if secs else np.empty((0, 0), dtype=object)
    vrd = None
    if processed and not legacy:
        vrd = _vrd_process(sections, p, datetime_format)
    return VrdData(sections, raw, vrd)


# -- read_sidecar_csv ------------------------------------------------------------------
_MATLAB_KEYWORDS = frozenset(
    ["break", "case", "catch", "classdef", "continue", "else", "elseif", "end", "for", "function", "global", "if", "otherwise", "parfor", "persistent", "return", "spmd", "switch", "try", "while"]
)
_NAME_LENGTH_MAX = 63


class SidecarCsv(NamedTuple):
    """Result of :func:`read_sidecar_csv` (MATLAB names in brackets)."""

    names: list[str]  #: column names (``csv_names``, ``table.Properties.VariableNames``)
    values: np.ndarray  #: [n_rows x n_columns] float64 (``csv_values``, ``table2array``)


def _make_valid_name(name: str) -> str:
    """MATLAB ``matlab.lang.makeValidName`` as applied by ``readtable``."""
    if re.fullmatch(r"[A-Za-z]\w*", name, flags=re.ASCII) and name not in _MATLAB_KEYWORDS:
        return name[:_NAME_LENGTH_MAX]
    s = re.sub(r"\s+([a-z])", lambda m: m.group(1).upper(), name.strip())
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    if s in _MATLAB_KEYWORDS:
        s = "x" + s[0].upper() + s[1:]
    elif not s or not s[0].isalpha():
        s = "x" + s
    return s[:_NAME_LENGTH_MAX]


def _unique_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for name in names:
        candidate, k = name, 0
        while candidate in seen:
            k += 1
            candidate = f"{name}_{k}"
        seen.add(candidate)
        out.append(candidate)
    return out


def _is_number(cell: str) -> bool:
    cell = cell.strip()
    return bool(_NUMBER.fullmatch(cell) or _SPECIAL.fullmatch(cell))


def read_sidecar_csv(
    path: str | PathLike,
    *,
    header: bool | None = None,
    variable_names: str = "modify",
    delimiter: str = ",",
) -> SidecarCsv:
    """Read a numeric sidecar CSV (``<name>_min.csv`` or ``<name>_beh1.csv``).

    Both files are written next to the aux data by the miniscope and behaviour-camera
    software: a header row with column names (for example ``FrameNumber`` and a
    time-stamp column; the miniscope BNO sensor columns are ``Roll``, ``Pitch``,
    ``Yaw``) and numeric rows. This reads them like MATLAB's ``readtable`` followed by
    ``table2array`` in ``load_exp``.

    Parameters
    ----------
    path
        The CSV file.
    header
        Whether the first row holds column names. ``None`` (default) decides like
        ``readtable``: it is a header unless every non-empty field of the first row is a
        number. Without header the columns are named ``Var1``, ``Var2``, ...
    variable_names
        ``"modify"`` (default, as ``readtable``) turns names into valid MATLAB
        identifiers: blanks are removed and the letter after a blank is capitalised
        (``Frame Number`` -> ``FrameNumber``), other invalid characters become ``_``
        (``Time (ms)`` -> ``Time_ms_``), ``x`` is put in front of names that do not
        start with a letter, MATLAB keywords get ``x`` and a capital (``for`` ->
        ``xFor``), empty names become ``Var<column>``, names are cut at 63 characters and
        duplicates get ``_1``, ``_2``, ... ``"preserve"`` keeps the names as written
        (empty ones become ``Var<column>``).
    delimiter
        Field delimiter.

    Returns
    -------
    SidecarCsv
        ``(names, values)``: ``names`` is a list of column names; ``values`` is a float64
        array of shape ``[n_rows x n_columns]`` (MATLAB's ``table2array``: one row per
        sample, one column per variable). Empty fields are NaN.

    Raises
    ------
    FileNotFoundError
        If ``path`` is not a file.
    ValueError
        If the file has no rows, a field is not a number, or a row has more fields than
        the header.

    Differences from MATLAB
    -----------------------
    * The ``readtable``/``table2array`` pair of ``load_exp`` is one function returning a
      named tuple; ``names`` is a list instead of a 1xN cell array.
    * ``readtable`` was left to detect column types; this reader is for numeric files.
      A text column makes MATLAB's ``table2array`` return a cell array or fail; here it
      raises ``ValueError``. A row with more fields than the header made ``readtable``
      add extra variables; here it raises. A row with fewer fields is padded with NaN
      (as ``readtable``).
    * Name conversion, header detection and number syntax are re-implemented from
      MATLAB's documentation, not copied; check them against the reference test on a
      real file.
    """
    p = _file_path(path)
    rows = [r for r in csv.reader(io.StringIO(_read_text(p), newline=""), delimiter=delimiter) if r]
    if not rows:
        raise ValueError(f"{p.name}: the file is empty")
    if header is None:
        fields = [c for c in rows[0] if c.strip()]
        header = not (fields and all(_is_number(c) for c in fields))
    if header:
        raw_names, body = rows[0], rows[1:]
    else:
        raw_names, body = [""] * len(rows[0]), rows
    names = [n if n.strip() else f"Var{i}" for i, n in enumerate(raw_names, start=1)]
    if variable_names == "modify":
        names = _unique_names([_make_valid_name(n) for n in names])
    elif variable_names != "preserve":
        raise ValueError(f"variable_names must be 'modify' or 'preserve', not {variable_names!r}")
    width = len(names)
    first_data_row = 2 if header else 1
    for i, row in enumerate(body):
        if len(row) > width:
            raise ValueError(
                f"{p.name}, data row {i + first_data_row}: {len(row)} fields but {width} columns"
            )
    try:
        values = np.array([r + [""] * (width - len(r)) for r in body], dtype=np.float64)
    except ValueError:
        values = np.full((len(body), width), np.nan)
        for i, row in enumerate(body):
            for j, cell in enumerate(row):
                if cell.strip():
                    if not _is_number(cell):
                        raise ValueError(
                            f"{p.name}, data row {i + first_data_row}, column {names[j]!r}: "
                            f"{cell!r} is not a number"
                        ) from None
                    values[i, j] = float(cell)
    return SidecarCsv(names, values.reshape(len(body), width))
