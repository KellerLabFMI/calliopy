"""Score-sheet workbooks (``.xlsx``): reading as MATLAB's ``xlsread``, checks and updates.

Ports of the file input/output of calliope r2003 ``04_Calliope`` ``getweight_ScoreSheet.m``,
``getLastMouseObservation.m``, ``checkScoreSheetMouse.m``, ``checkScoreSheets.m`` and
``updateScoreSheet.m``. The rules themselves are in ``_scoresheet.py``.

How MATLAB addresses a score sheet
----------------------------------
MATLAB reads the first worksheet with ``[num, txt] = xlsread(file)`` on Windows with Excel.
``num`` and ``txt`` are two blocks of the sheet, each trimmed of leading and trailing rows and
columns that hold no number (``num``) or no text (``txt``). The MATLAB code then indexes both
blocks with fixed numbers that are only right for the lab template, where the text block
starts at cell A1 and the numeric block at cell B2:

========================  =====================  ======================================
MATLAB expression         sheet cell (template)  content
========================  =====================  ======================================
``txt{1, 3}``             C1                     start date (text)
``txt{1, k}``, k >= 4     row 1, column k        observation dates (text)
``num(1, j)``             row 2, column j + 1    day counter (0 in C2)
``num(15, j)``            row 16                 total score
``num(17, 1)``            B18                    initial weight
``num(19, j)``            row 20                 body weight
========================  =====================  ======================================

These functions reproduce ``xlsread`` (:func:`read_score_sheet`) and then use MATLAB's indices
unchanged, so they read and write the same cells as MATLAB on any sheet. They log a warning
when the blocks do not start at A1 and B2.

Formulas
--------
A score sheet's formulas (total score, weight changes, ...) are kept as formulas. calliopy
computes their results itself (``_xlsx_formula``): formula cells without stored result are
computed when reading, and every update stores the recomputed results of all formulas in the
file. No spreadsheet program is needed between updates.

openpyxl is imported inside the functions that need it, so ``import calliopy`` works without it.

None of these functions has been compared with MATLAB output.
"""

from __future__ import annotations

import datetime
import logging
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from ._scoresheet import (
    _days,
    excel_column_name,
    observation_too_late,
    score_sheet_problem,
    score_sheet_row_values,
)
from ._xlsx_formula import (
    FormulaBook,
    FormulaError,
    XlError,
    datetime_to_serial,
    serial_to_datetime,
    store_formula_results,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CheckScoreSheetsResult",
    "LastMouseObservation",
    "ScoreSheetAlert",
    "ScoreSheetUpdate",
    "ScoreSheetWeights",
    "XlsRead",
    "check_score_sheet_mouse",
    "check_score_sheets",
    "last_mouse_observation",
    "read_score_sheet",
    "score_sheet_path",
    "score_sheet_weights",
    "update_score_sheet",
]

#: Where the template puts the first element of ``txt`` and ``num`` (1-based sheet row, column).
_TEMPLATE_TXT_ORIGIN = (1, 1)
_TEMPLATE_NUM_ORIGIN = (2, 2)

_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December")
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

#: Excel shows these built-in formats with the regional short date of the computer
#: (openpyxl names them by their US spelling).
_SYSTEM_SHORT_DATE_FORMATS = {"mm-dd-yy": "", "m/d/yy h:mm": " h:mm"}

_ALERT_SUBJECT = "URGENT - check your mice"
_ALERT_BODY = ("Please check on your mice, there appears to be a problem with the score sheet "
              "of mouse: ")


# --- reading -----------------------------------------------------------------------------


class XlsRead(NamedTuple):
    """``[num, txt] = xlsread(file)`` of the first worksheet.

    Attributes
    ----------
    num
        float64 ``(rows, columns)``: numbers, NaN for text and empty cells; ``(0, 0)`` when the
        sheet holds no number.
    txt
        object array of ``str``, ``(rows, columns)``: text, ``''`` for numbers and empty
        cells; ``(0, 0)`` when the sheet holds no text.
    num_origin, txt_origin
        1-based sheet ``(row, column)`` of ``num[0, 0]`` and ``txt[0, 0]`` (not returned by
        MATLAB); ``None`` for an empty block.
    """

    num: np.ndarray
    txt: np.ndarray
    num_origin: tuple[int, int] | None
    txt_origin: tuple[int, int] | None


def _tokenize_number_format(fmt: str) -> list[str]:
    section = re.split(r';(?=(?:[^"]*"[^"]*")*[^"]*$)', fmt)[0]
    return re.findall(r'"[^"]*"|\\.|\[[^\]]*\]|_.|\*.|yyyy|yy|mmmmm|mmmm|mmm|mm|m|dddd|ddd|dd|d|'
                      r'hh|h|ss|s|AM/PM|A/P|.', section, flags=re.IGNORECASE)


def _format_excel_date(value: datetime.date | datetime.time, fmt: str,
                       short_date_format: str) -> str:
    """Text Excel shows for a date cell with number format ``fmt`` (English month names)."""
    if fmt in _SYSTEM_SHORT_DATE_FORMATS:
        fmt = short_date_format + _SYSTEM_SHORT_DATE_FORMATS[fmt]
    if isinstance(value, datetime.time):
        value = datetime.datetime.combine(datetime.date(1899, 12, 31), value)
    elif not isinstance(value, datetime.datetime):
        value = datetime.datetime(value.year, value.month, value.day)
    tokens = _tokenize_number_format(fmt)
    low = [t.lower() for t in tokens]
    twelve_hour = any(t in ("am/pm", "a/p") for t in low)
    kinds = [t if re.fullmatch(r"y+|m+|d+|h+|s+", t) else None for t in low]
    out = []
    for i, (tok, t) in enumerate(zip(tokens, low)):
        if t.startswith('"'):
            out.append(tok[1:-1])
        elif t.startswith("\\"):
            out.append(tok[1:])
        elif t.startswith("["):
            continue  # locale, colour or condition
        elif t.startswith("_"):
            out.append(" ")
        elif t.startswith("*"):
            continue
        elif t == "yyyy":
            out.append(f"{value.year:04d}")
        elif t == "yy":
            out.append(f"{value.year % 100:02d}")
        elif t in ("m", "mm"):
            prev = next((k for k in reversed(kinds[:i]) if k), None)
            nxt = next((k for k in kinds[i + 1:] if k), None)
            minute = (prev is not None and prev[0] == "h") or (nxt is not None and nxt[0] == "s")
            v = value.minute if minute else value.month
            out.append(f"{v:02d}" if t == "mm" else str(v))
        elif t == "mmm":
            out.append(_MONTHS[value.month - 1][:3])
        elif t == "mmmm":
            out.append(_MONTHS[value.month - 1])
        elif t == "mmmmm":
            out.append(_MONTHS[value.month - 1][0])
        elif t in ("d", "dd"):
            out.append(f"{value.day:02d}" if t == "dd" else str(value.day))
        elif t == "ddd":
            out.append(_WEEKDAYS[value.weekday()][:3])
        elif t == "dddd":
            out.append(_WEEKDAYS[value.weekday()])
        elif t in ("h", "hh"):
            h = (value.hour % 12 or 12) if twelve_hour else value.hour
            out.append(f"{h:02d}" if t == "hh" else str(h))
        elif t in ("s", "ss"):
            out.append(f"{value.second:02d}" if t == "ss" else str(value.second))
        elif t == "am/pm":
            out.append("AM" if value.hour < 12 else "PM")
        elif t == "a/p":
            out.append("A" if value.hour < 12 else "P")
        else:
            out.append(tok)
    return "".join(out)


def _trim_box(empty: np.ndarray) -> tuple[slice, slice] | None:
    """Rows and columns between the first and last that are not all empty (MATLAB
    ``filterDataUsingMask`` in ``xlsread``)."""
    rows = np.flatnonzero(~empty.all(axis=1))
    cols = np.flatnonzero(~empty.all(axis=0))
    if rows.size == 0:
        return None
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def read_score_sheet(path: str | os.PathLike, *, basic: bool = False,
                     short_date_format: str = "dd.mm.yyyy",
                     now: datetime.datetime | None = None) -> XlsRead:
    """Read the first worksheet of an ``.xlsx`` file as ``[num, txt] = xlsread(path)``.

    Reproduces MATLAB's ``xlsread`` (Windows with Excel, or ``'basic'`` mode) as used by the
    score-sheet functions of calliope ``04_Calliope``: the first worksheet is read, every cell
    goes either to the numeric block or to the text block, and each block is trimmed of its
    leading and trailing rows and columns that hold nothing. Indices into ``num`` and ``txt``
    are therefore relative to their own first row and column (see ``num_origin`` and
    ``txt_origin``).

    Parameters
    ----------
    path
        Workbook (``.xlsx``/``.xlsm``).
    basic
        Reproduce ``xlsread(..., 'basic')`` instead of the Windows/Excel mode MATLAB used.
    short_date_format
        Excel number format of the computer's regional short date. Excel shows dates in the
        built-in short-date format (``numFmtId`` 14) with it, and MATLAB receives that text.
    now
        Clock of ``TODAY()``/``NOW()`` when a formula has to be computed (default: the
        current time).

    Returns
    -------
    XlsRead
        ``num``, ``txt`` and their 1-based sheet origins.

    Notes
    -----
    Cell values, as MATLAB receives them:

    - numbers go to ``num``; logical cells go to ``num`` as 1/0;
    - text goes to ``txt`` (``''`` counts as empty when trimming); error values (``#DIV/0!``
      ...) become ``'#N/A'`` in ``txt``, as MATLAB's Excel mode does (kept as is in basic mode);
    - formulas give their result stored in the file. A formula cell without stored result
      (the file was saved by a program that does not calculate) is computed as Excel would
      (see ``_xlsx_formula``); an unsupported function raises ``UnsupportedFormulaError``;
    - **dates.** An Excel date is a number, the serial day count (days since 1899-12-30 in the
      1900 date system, since 1904-01-01 in the 1904 system; the fraction is the time of day).
      In the Windows/Excel mode MATLAB receives a date-formatted cell as the text Excel shows
      (``'27.11.2026'``), so it lands in ``txt``; the score-sheet code relies on this. In basic
      mode it receives the serial number, in ``num``. Both are reproduced here; serial numbers
      are converted with the workbook's own date system.

    Differences from MATLAB
    -----------------------
    - Date text is rendered from the cell's number format with English month and weekday
      names. Excel uses the names of the computer's language (and its regional short date for
      the built-in short-date format, see ``short_date_format``).
    - Formula results missing from the file are computed by calliopy, with the functions
      listed in ``_xlsx_formula``; Excel would compute them itself.
    """
    import openpyxl
    from openpyxl.styles.numbers import is_date_format
    from openpyxl.utils.datetime import CALENDAR_MAC_1904

    path = Path(path)
    wb = openpyxl.load_workbook(path, data_only=True)
    wbf = openpyxl.load_workbook(path, data_only=False)
    ws, wsf = wb.worksheets[0], wbf.worksheets[0]
    epoch1904 = wb.epoch == CALENDAR_MAC_1904
    formula_cells = {(c.row, c.column) for row in wsf.iter_rows() for c in row
                     if c.data_type == "f"}
    book = None
    nr, nc = ws.max_row, ws.max_column
    num = np.full((nr, nc), np.nan)
    txt = np.full((nr, nc), "", dtype=object)
    is_text = np.zeros((nr, nc), dtype=bool)
    for row in ws.iter_rows(min_row=1, min_col=1, max_row=nr, max_col=nc):
        for cell in row:
            r, c, v, dtype = cell.row - 1, cell.column - 1, cell.value, cell.data_type
            if v is None and (cell.row, cell.column) in formula_cells:
                if book is None:  # result not stored in the file: compute it
                    book = FormulaBook(wbf, stored=wb, now=now)
                v, dtype = _formula_cell_value(book.value(ws.title, cell.row, cell.column),
                                               cell.number_format, epoch1904, is_date_format)
            if v is None:
                continue
            if isinstance(v, (bool, int, float)):  # logical cells become 1/0
                num[r, c] = float(v)
            elif isinstance(v, (datetime.datetime, datetime.date, datetime.time)):
                if basic:
                    num[r, c] = datetime_to_serial(v, epoch1904)
                else:
                    txt[r, c] = _format_excel_date(v, cell.number_format, short_date_format)
                    is_text[r, c] = True
            elif isinstance(v, datetime.timedelta):
                num[r, c] = v.total_seconds() / 86400.0
            else:
                s = str(v)
                if dtype == "e" and not basic:
                    s = "#N/A"
                txt[r, c] = s
                is_text[r, c] = True
    wb.close()
    wbf.close()

    box = _trim_box(np.isnan(num))
    if box is None:
        num_out, num_origin = np.zeros((0, 0)), None
    else:
        num_out, num_origin = num[box].copy(), (box[0].start + 1, box[1].start + 1)
    box = _trim_box(~is_text | (txt == ""))
    if box is None:
        txt_out, txt_origin = np.empty((0, 0), dtype=object), None
    else:
        txt_out, txt_origin = txt[box].copy(), (box[0].start + 1, box[1].start + 1)
    return XlsRead(num_out, txt_out, num_origin, txt_origin)


def _formula_cell_value(x: Any, number_format: str, epoch1904: bool,
                        is_date_format: Callable[[str], bool]) -> tuple[Any, str]:
    """A computed formula result as openpyxl would have read the stored one."""
    if isinstance(x, XlError):
        return x.code, "e"
    if isinstance(x, bool):
        return x, "b"
    if isinstance(x, str):
        return x, "s"
    if is_date_format(number_format):
        return serial_to_datetime(x, epoch1904), "d"
    return x, "n"


# --- MATLAB addressing helpers -----------------------------------------------------------


def _num_at(num: np.ndarray, i: int, j: int, what: str) -> float:
    """``num(i, j)`` with MATLAB's 1-based indices."""
    if not (1 <= i <= num.shape[0] and 1 <= j <= num.shape[1]):
        raise ValueError(f"score sheet layout not as expected: {what} is element ({i}, {j}) of "
                         f"the numeric block, which has size {num.shape[0]}x{num.shape[1]}")
    return float(num[i - 1, j - 1])


def _txt_at(txt: np.ndarray, i: int, j: int, what: str) -> str:
    """``txt{i, j}`` with MATLAB's 1-based indices."""
    if not (1 <= i <= txt.shape[0] and 1 <= j <= txt.shape[1]):
        raise ValueError(f"score sheet layout not as expected: {what} is element ({i}, {j}) of "
                         f"the text block, which has size {txt.shape[0]}x{txt.shape[1]}")
    return str(txt[i - 1, j - 1])


def _check_layout(xl: XlsRead, name: str) -> None:
    if xl.txt_origin != _TEMPLATE_TXT_ORIGIN or xl.num_origin != _TEMPLATE_NUM_ORIGIN:
        logger.warning("%s: text block starts at %s and numeric block at %s (sheet row, column), "
                       "not at (1, 1) and (2, 2) as in the score-sheet template; MATLAB's fixed "
                       "indices then address other cells", name, xl.txt_origin, xl.num_origin)


def _date_format(start_text: str) -> str:
    """``dateformat`` chosen by MATLAB from the start-date text (cell C1)."""
    if re.search(r"\d{2}\.\d{2}\.\d{4}", start_text, flags=re.IGNORECASE):
        return "dd.mm.yyyy"
    if re.search(r"\d{2}-\w{3}-\d{2}", start_text, flags=re.IGNORECASE):
        return "dd-mm-yy"
    raise ValueError(f"start date {start_text!r} is neither dd.mm.yyyy nor dd-Mon-yy")


def _expand_two_digit_year(yy: int, today: datetime.date | None = None) -> int:
    """MATLAB ``datenum`` pivot: two-digit years fall in [current year - 50, + 49]."""
    base = (today or datetime.date.today()).year - 50
    year = base - base % 100 + yy
    return year + 100 if year < base else year


def _parse_sheet_date(text: str, dateformat: str) -> datetime.datetime:
    """``datenum(text, dateformat)`` for the two formats the score-sheet code uses."""
    s = text.strip()
    if dateformat == "dd.mm.yyyy":
        m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{1,4})", s)
        if m:
            day, month, year = (int(g) for g in m.groups())
            return _make_date(year, month, day, text)
    elif dateformat == "dd-mm-yy":
        m = re.fullmatch(r"(\d{1,2})-(\d{1,2}|[A-Za-z]{3,})-(\d{2})", s)
        if m:
            day, mon, yy = m.groups()
            if mon.isdigit():
                month = int(mon)
            else:
                abbr = [n[:3].lower() for n in _MONTHS]
                if mon[:3].lower() not in abbr:
                    raise ValueError(f"cannot read date {text!r} as {dateformat}")
                month = abbr.index(mon[:3].lower()) + 1
            return _make_date(_expand_two_digit_year(int(yy)), month, int(day), text)
    raise ValueError(f"cannot read date {text!r} as {dateformat}")


def _make_date(year: int, month: int, day: int, text: str) -> datetime.datetime:
    try:
        return datetime.datetime(year, month, day)
    except ValueError as err:
        raise ValueError(f"invalid date {text!r}") from err


def _format_sheet_date(date: datetime.date, dateformat: str) -> str:
    """``datestr(date, dateformat)``."""
    if dateformat == "dd.mm.yyyy":
        return f"{date.day:02d}.{date.month:02d}.{date.year:04d}"
    return f"{date.day:02d}-{date.month:02d}-{date.year % 100:02d}"


def _as_datetime(d: datetime.date) -> datetime.datetime:
    if isinstance(d, datetime.datetime):
        return d
    if isinstance(d, datetime.date):
        return datetime.datetime(d.year, d.month, d.day)
    raise TypeError(f"expected a datetime.date or datetime.datetime, got {type(d).__name__}")


def _last_data_index(num: np.ndarray) -> int:
    """``find(isnan(num(1, 2:end)), 1)``: 1-based column of ``num`` holding the last
    observation ("first NaN in the day counter")."""
    if num.shape[0] < 1 or num.shape[1] < 2:
        raise ValueError("score sheet layout not as expected: no day-counter row")
    hits = np.flatnonzero(np.isnan(num[0, 1:]))
    if hits.size == 0:
        raise ValueError("score sheet layout not as expected: the day-counter row has no empty "
                         "cell after the last observation")
    return int(hits[0]) + 1


def score_sheet_path(user_folder: str | os.PathLike, mouse: str) -> Path:
    """Score sheet of ``mouse`` in a user's animal folder: ``<user_folder>/<mouse>/ScoreSheet_<mouse>.xlsx``.

    The folder layout of calliope's score-sheet functions; the folder itself is the caller's.
    """
    return Path(user_folder) / mouse / f"ScoreSheet_{mouse}.xlsx"


# --- getweight_ScoreSheet ----------------------------------------------------------------


class ScoreSheetWeights(NamedTuple):
    """Outputs of ``getweight_ScoreSheet.m`` (grams, as in the sheet)."""

    initial_weight: float
    last_weight: float


def score_sheet_weights(path: str | os.PathLike, *, short_date_format: str = "dd.mm.yyyy",
                        now: datetime.datetime | None = None) -> ScoreSheetWeights:
    """Initial and last body weight of a score sheet (``getweight_ScoreSheet.m``, ``04_Calliope``).

    ``initial_weight`` is ``num(17, 1)`` (cell B18 in the template), ``last_weight`` is
    ``num(19, last_data_index)`` (row 20 of the last observation column), with
    ``last_data_index`` the first empty cell of the day-counter row (see
    :func:`read_score_sheet` for the addressing).

    Parameters
    ----------
    path
        Score-sheet workbook.
    short_date_format
        Passed to :func:`read_score_sheet`.
    now
        Clock of ``TODAY()``/``NOW()`` for formulas without stored result (default: now).

    Returns
    -------
    ScoreSheetWeights
        ``last_weight`` is NaN when the last observation has no weight (no replacement, as in
        MATLAB).

    Differences from MATLAB
    -----------------------
    - Takes the score-sheet path; MATLAB built it from the lab data location, a user name and
      a one-element cell with the mouse name (see :func:`score_sheet_path`).
    - An unexpected layout raises ``ValueError`` instead of MATLAB's index error.
    """
    path = Path(path)
    xl = read_score_sheet(path, short_date_format=short_date_format, now=now)
    _check_layout(xl, path.name)
    last = _last_data_index(xl.num)
    return ScoreSheetWeights(_num_at(xl.num, 17, 1, "the initial weight"),
                             _num_at(xl.num, 19, last, "the last weight"))


# --- getLastMouseObservation -------------------------------------------------------------


class LastMouseObservation(NamedTuple):
    """Outputs of ``getLastMouseObservation.m``, each float64 ``(1, n_mice)`` in days."""

    days_since_last_observation: np.ndarray
    days_in_experiment: np.ndarray


def last_mouse_observation(user_folder: str | os.PathLike, mice: Sequence[str], *,
                           now: datetime.datetime | None = None) -> LastMouseObservation:
    """Days since the last score-sheet entry and days in experiment (``getLastMouseObservation.m``).

    For each mouse with a score sheet (:func:`score_sheet_path`):
    ``days_since_last_observation = floor(now - modification time of the sheet)`` and
    ``days_in_experiment = floor(now - date)``, where the date is characters 4-9 of the mouse
    name read as ``yymmdd``. Both are ``inf`` for a mouse without score sheet. The workbook
    itself is not read (MATLAB's reading code is commented out in r2003).

    Parameters
    ----------
    user_folder
        Folder holding one subfolder per mouse.
    mice
        Mouse names.
    now
        Reference time, local (MATLAB ``now``); default the current time.

    Returns
    -------
    LastMouseObservation
        Arrays of shape ``(1, len(mice))``.

    Raises
    ------
    ValueError
        A mouse with score sheet whose characters 4-9 are not a ``yymmdd`` date.

    Differences from MATLAB
    -----------------------
    - Takes the user's animal folder instead of a user name resolved through the lab paths.
    - MATLAB reads the modification time with one-second resolution; here it is exact, which
      can change ``floor`` within one second of a whole day.
    - An empty ``mice`` returns ``(1, 0)`` arrays (MATLAB failed with unassigned outputs).
    """
    now = now or datetime.datetime.now()
    since = np.full((1, len(mice)), np.inf)
    in_exp = np.full((1, len(mice)), np.inf)
    for k, mouse in enumerate(mice):
        p = score_sheet_path(user_folder, mouse)
        if not p.is_file():
            continue
        mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime)
        since[0, k] = math.floor(_days(now, mtime))
        code = mouse[3:9]
        if not re.fullmatch(r"\d{6}", code):
            raise ValueError(f"characters 4-9 of mouse name {mouse!r} are not a yymmdd date")
        start = _make_date(_expand_two_digit_year(int(code[:2]), now.date()), int(code[2:4]),
                           int(code[4:6]), code)
        in_exp[0, k] = math.floor(_days(now, start))
    return LastMouseObservation(since, in_exp)


# --- checkScoreSheetMouse ----------------------------------------------------------------


def check_score_sheet_mouse(path: str | os.PathLike,
                            relative_to: datetime.datetime | None = None, *,
                            short_date_format: str = "dd.mm.yyyy") -> bool:
    """Whether a score sheet is out of spec (``checkScoreSheetMouse.m``, ``04_Calliope``).

    Reads the sheet as MATLAB does and applies :func:`score_sheet_problem`:

    - the start date is ``txt{1, 3}`` (cell C1); its text chooses the date format
      (``dd.mm.yyyy``, or ``dd-mm-yy`` for text like ``27-Nov-26``);
    - every non-empty ``txt{1, k}``, k >= 4, is an observation date; its total score is
      ``num(r, k - 3 + j0)``, with ``r`` the text row labelled ``'TOTAL SCORE'`` in the first
      text column minus one, and ``j0`` the first column of ``num`` whose first row is 0 (the
      day counter of the start column).

    Parameters
    ----------
    path
        Score-sheet workbook.
    relative_to
        Reference time (MATLAB ``relativeToDate``, default ``now``); also the clock of
        ``TODAY()``/``NOW()`` for formulas without stored result.
    short_date_format
        Passed to :func:`read_score_sheet`.

    Returns
    -------
    bool
        ``True`` when a problem is found (MATLAB ``problem = 1``).

    Raises
    ------
    ValueError
        No or several ``'TOTAL SCORE'`` rows, a start date in neither format, an unreadable
        date, or indices outside the sheet's blocks (MATLAB errors).

    Notes
    -----
    Kept from MATLAB (suspected bug, changes the result): gaps between observations are
    never flagged, see :func:`score_sheet_problem`.

    Differences from MATLAB
    -----------------------
    - Takes the score-sheet path instead of a user name and mouse resolved through the lab
      paths.
    - Dates written as ``dd-Mon-yy`` are read with English month abbreviations, and two-digit
      years with MATLAB's pivot (current year - 50 to + 49). Whether MATLAB's ``datenum``
      accepts a month name for the format ``dd-mm-yy`` was not checked.
    """
    path = Path(path)
    relative_to = relative_to or datetime.datetime.now()
    xl = read_score_sheet(path, short_date_format=short_date_format, now=relative_to)
    _check_layout(xl, path.name)
    num, txt = xl.num, xl.txt

    first_col = [str(t) for t in txt[:, 0]] if txt.shape[1] else []
    rows = [i + 1 for i, t in enumerate(first_col) if t == "TOTAL SCORE"]
    if len(rows) != 1:
        raise ValueError(f"{path.name}: expected one 'TOTAL SCORE' row, found {len(rows)}")
    total_score_row = rows[0] - 1

    zeros = np.flatnonzero(num[0, :] == 0) if num.shape[0] else np.array([], dtype=int)
    num_start_ind = int(zeros[0]) + 1 if zeros.size else None

    start_text = _txt_at(txt, 1, 3, "the start date")
    dateformat = _date_format(start_text)
    start = _parse_sheet_date(start_text, dateformat)

    observations: list[tuple[datetime.datetime, float]] = []
    for date_ind in range(4, txt.shape[1] + 1):
        text = str(txt[0, date_ind - 1])
        if not text:
            continue
        curr = _parse_sheet_date(text, dateformat)
        score = np.nan
        if _days(curr, start) > 0:  # MATLAB reads the score only then
            if num_start_ind is None:
                raise ValueError(f"{path.name}: no 0 in the day-counter row")
            score = _num_at(num, total_score_row, date_ind - 3 + num_start_ind, "a total score")
        observations.append((curr, score))
    return bool(score_sheet_problem(start, observations, relative_to))


# --- checkScoreSheets --------------------------------------------------------------------


class ScoreSheetAlert(NamedTuple):
    """Message ``checkScoreSheets.m`` e-mailed to the user of a mouse with a problem."""

    user: str
    mouse: str
    subject: str
    body: str


class CheckScoreSheetsResult(NamedTuple):
    """Result of :func:`check_score_sheets`.

    Attributes
    ----------
    problem_mice
        Mice with a problem (MATLAB variable ``problem_mice``, saved to an error log there).
    alerts
        One :class:`ScoreSheetAlert` per problem mouse, the e-mail MATLAB sent.
    not_in_log
        ``(user, mouse)`` of score sheets whose mouse is missing from ``dates_of_death``
        (MATLAB warned and skipped them).
    """

    problem_mice: list[str]
    alerts: list[ScoreSheetAlert]
    not_in_log: list[tuple[str, str]]


def _has_date_of_death(dod: Any) -> bool:
    """MATLAB ``~isnan(dod)``: a non-empty text or a number that is not NaN."""
    if dod is None:
        return False
    if isinstance(dod, str):
        return len(dod) > 0
    if isinstance(dod, (datetime.date, datetime.datetime)):
        return True
    if isinstance(dod, np.datetime64):
        return not np.isnat(dod)
    arr = np.asarray(dod, dtype=np.float64)
    return arr.size > 0 and bool(np.all(~np.isnan(arr)))


def _sorted_names(paths: Sequence[Path]) -> list[Path]:
    return sorted(paths, key=lambda p: (p.name.upper(), p.name))


def check_score_sheets(animal_folder: str | os.PathLike, dates_of_death: Mapping[str, Any], *,
                       now: datetime.datetime | None = None,
                       short_date_format: str = "dd.mm.yyyy") -> CheckScoreSheetsResult:
    """Check the score sheets of all living mice of all users (``checkScoreSheets.m``, ``04_Calliope``).

    ``animal_folder`` holds one folder per user, each with one folder per mouse; a mouse
    counts when it has a score sheet (:func:`score_sheet_path`). Mice with a date of death,
    and mice missing from ``dates_of_death``, are skipped. Every other score sheet is checked
    with :func:`check_score_sheet_mouse` relative to ``now``.

    Parameters
    ----------
    animal_folder
        Folder with one subfolder per user.
    dates_of_death
        Mouse name -> date of death, as in the animal table (MATLAB ``MiceExpLog.dod``):
        ``None``, NaN or ``''`` for a living mouse, a date (text or ``datetime``) otherwise.
    now
        Reference time (default: the current time, taken once).
    short_date_format
        Passed to :func:`read_score_sheet`.

    Returns
    -------
    CheckScoreSheetsResult
        Problem mice, the alert texts and the mice missing from ``dates_of_death``.

    Differences from MATLAB
    -----------------------
    - The users' folder and the dates of death are arguments; MATLAB found them through the
      lab paths and the animal database. Users and mice are taken in case-insensitive name
      order.
    - E-mails are not sent: the messages are returned in ``alerts``; finding the recipient
      address from the user name is the caller's.
    - ``problem_mice`` is returned instead of being saved to an error-log MAT-file in a lab
      location.
    - MATLAB took ``now`` anew for every mouse; here one reference time is used for all.
    """
    now = now or datetime.datetime.now()
    root = Path(animal_folder)
    problem_mice: list[str] = []
    alerts: list[ScoreSheetAlert] = []
    not_in_log: list[tuple[str, str]] = []
    for user_dir in _sorted_names([p for p in root.iterdir() if p.is_dir()]):
        user = user_dir.name
        mice = [p.name for p in _sorted_names(list(user_dir.iterdir()))
                if score_sheet_path(user_dir, p.name).is_file()]
        alive = []
        for mouse in mice:
            if mouse not in dates_of_death:
                logger.warning("Warning - mouse: %s was not found in the animal table", mouse)
                not_in_log.append((user, mouse))
            elif not _has_date_of_death(dates_of_death[mouse]):
                alive.append(mouse)
        if alive:
            logger.info("------  %s", user)
        for mouse in alive:
            if check_score_sheet_mouse(score_sheet_path(user_dir, mouse), now,
                                       short_date_format=short_date_format):
                logger.info("There is a problem with: %s", mouse)
                problem_mice.append(mouse)
                alerts.append(ScoreSheetAlert(user, mouse, _ALERT_SUBJECT, _ALERT_BODY + mouse))
    return CheckScoreSheetsResult(problem_mice, alerts, not_in_log)


# --- updateScoreSheet --------------------------------------------------------------------


class ScoreSheetUpdate(NamedTuple):
    """Result of :func:`update_score_sheet`.

    Attributes
    ----------
    success
        MATLAB's output: ``True`` once any sheet was written.
    updated
        Sheets written, with the cells written (``{'F1': '27.11.2026', 'F2': 3, ...}``).
    failed
        Sheets that could not be written (MATLAB: "it might be open").
    cancelled
        Sheet at which the update stopped because a late observation was not confirmed
        (MATLAB: the dialog answered No/Cancel); ``None`` otherwise. Later sheets are untouched.
    late_messages
        Late-observation messages that were confirmed.
    """

    success: bool
    updated: list[tuple[Path, dict[str, Any]]]
    failed: list[Path]
    cancelled: Path | None
    late_messages: list[str]


def _u_format(x: float) -> str:
    """MATLAB ``sprintf('%u', x)``."""
    return str(int(x)) if float(x).is_integer() and x >= 0 else f"{x:e}"


def _num2str(values: Sequence[float]) -> str:
    """Approximation of MATLAB ``num2str`` of a row vector."""
    parts = ["NaN" if np.isnan(v) else (str(int(v)) if float(v).is_integer() else f"{v:.4g}")
             for v in values]
    return "  ".join(parts)


def _late_message(n_days: float, last_3_scores: Sequence[float]) -> str:
    return "\n".join([
        "It looks like your observation is too late!",
        f"- {_u_format(n_days)} days since last observation",
        f"- The total score for the last observations was: {_num2str(last_3_scores)}.",
        "Are you sure you want to continue?",
    ])


def _cell_value(v: Any) -> Any:
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if math.isnan(v):
            return None  # xlswrite leaves NaN cells empty
        return int(v) if v.is_integer() else v
    if isinstance(v, np.integer):
        return int(v)
    return v


def _formula_results(wb, now: datetime.datetime | None) -> dict[str, dict[tuple[int, int], Any]]:
    """Results of all formula cells: the first sheet strictly (any error raises), the other
    sheets as far as they can be computed (their unsupported cells are left without result)."""
    book = FormulaBook(wb, now=now)
    first = wb.worksheets[0].title
    results = {first: {rc: book.value(first, *rc) for rc in book.formula_cells(first)}}
    for sheet in book.sheets[1:]:
        res = {}
        for rc in book.formula_cells(sheet):
            try:
                res[rc] = book.value(sheet, *rc)
            except FormulaError as err:
                logger.warning("formula result not stored: %s", err)
        results[sheet] = res
    return results


def _write_cells(path: Path, cells: Mapping[str, Any],
                 now: datetime.datetime | None = None) -> None:
    """Write cells of the first worksheet and the recomputed results of all formulas, keeping
    the rest of the workbook. Nothing is written when a formula cannot be computed."""
    import openpyxl

    wb = openpyxl.load_workbook(path, rich_text=True, keep_links=True)
    try:
        ws = wb.worksheets[0]
        for address, value in cells.items():
            ws[address].value = _cell_value(value)
        wb.calculation.fullCalcOnLoad = True  # a spreadsheet program recalculates on opening
        results = _formula_results(wb, now)  # raises before any file is written
        tmp = path.with_name(f".{path.name}.calliopy-tmp")
        tmp2 = path.with_name(f".{path.name}.calliopy-tmp2")
        try:
            wb.save(tmp)
            store_formula_results(tmp, tmp2, results)
            os.replace(tmp2, path)
        finally:
            for t in (tmp, tmp2):
                if t.exists():
                    t.unlink()
    finally:
        wb.close()


def update_score_sheet(score_sheets: str | os.PathLike | Sequence[str | os.PathLike],
                       observations: Mapping[str, Any], *,
                       allow_late: bool | Callable[[str], bool] = False,
                       short_date_format: str = "dd.mm.yyyy",
                       now: datetime.datetime | None = None) -> ScoreSheetUpdate:
    """Enter one observation into score sheets (``updateScoreSheet.m``, ``04_Calliope``).

    For each sheet, in order (MATLAB's ``list_of_mice``):

    1. Read it as MATLAB does (:func:`read_score_sheet`). The date format comes from the
       start date ``txt{1, 3}``; the last observation is column ``L`` of ``num``, the cell
       before the first empty cell of the day-counter row (``num(1, 2:end)``).
    2. Days since the last observation: ``Date - txt{1, L + 1}``. A negative value raises.
    3. Late check (:func:`observation_too_late`) with the total scores
       ``num(15, max(3, L - 1):L + 1)``. A late observation needs ``allow_late``; otherwise
       the update stops here (MATLAB dialog answered No/Cancel).
    4. Weights: ``initial = num(17, 1)``, ``last = num(19, L)`` (initial when empty).
    5. Write into sheet column ``L + 2`` (Excel letters): row 1 the date as text in the
       sheet's format, row 2 the days since the start date, rows 5-8, 10, 12-15 and 20-22
       the values of :func:`score_sheet_row_values`.

    Parameters
    ----------
    score_sheets
        One score-sheet path or a sequence of them.
    observations
        MATLAB's ``observations`` struct as a mapping: ``Date`` (``datetime.date`` or
        ``datetime.datetime``) and the fields of :func:`score_sheet_row_values`.
    allow_late
        Answer to MATLAB's late-observation dialog: ``True``/``False``, or a function that
        gets the dialog text and returns ``True`` to continue.
    short_date_format
        Passed to :func:`read_score_sheet`.
    now
        Clock of ``TODAY()``/``NOW()`` in the sheet's formulas (default: the current time).

    Returns
    -------
    ScoreSheetUpdate
        What was written, what failed and where the update stopped.

    Raises
    ------
    ValueError
        The observation precedes the last one in a sheet (MATLAB printed a message and
        stopped), a NaN body weight, or a sheet whose layout does not fit MATLAB's indices.
        Sheets before it in the list have already been written.
    UnsupportedFormulaError
        A formula of the first sheet uses a function calliopy cannot compute (a
        ``ValueError``); the sheet is left unchanged.

    Notes
    -----
    Kept from MATLAB (suspected bugs, they change the result):

    - The late check takes the total scores of the last two observations *and of the new,
      still empty column* (``num(15, L - 1:L + 1)``), not of three days. An empty total
      makes the sum NaN, so the weekly rule then never applies; when the template holds a
      total of 0 in the empty column, two zero scores suffice for the weekly rule.
    - The late check and the total-score row use the fixed row 15 of ``num``, while
      :func:`check_score_sheet_mouse` looks the row up by its label.

    Writing uses openpyxl: values, styles, formulas, other sheets, comments, merged cells and
    defined names are kept; the dates are written as text, as MATLAB passed them. Every formula
    is then recomputed from the new values (``_xlsx_formula``) and its result stored in the file
    next to the formula, so the workbook is complete without a spreadsheet program; it is also
    flagged so that a spreadsheet program recalculates on opening. Formulas of the first sheet
    must all be computable (otherwise ``UnsupportedFormulaError`` and nothing is written);
    other sheets keep formulas that cannot be computed without stored result.

    Differences from MATLAB
    -----------------------
    - Takes score-sheet paths instead of a user name and mouse names resolved through the
      lab paths, and the dialog answer as ``allow_late``.
    - Formula results are computed by calliopy instead of Excel (supported functions in
      ``_xlsx_formula``). Drawings and shapes in the workbook are not kept (openpyxl).
    - MATLAB wrote through Excel, which may turn the date text into a date cell depending on
      the computer's regional settings; here it stays text (both read back as the same text).
    - A write failure (``OSError``, e.g. the file is open) is logged and the next sheet is
      processed, as in MATLAB; other errors are raised. The file is written to a temporary
      file next to it and then replaced, so a failed write leaves it unchanged.
    - MATLAB's unused ``n_days_since_start_of_exp`` is not computed. The dialog text
      approximates MATLAB's number formatting.
    """
    if isinstance(score_sheets, (str, os.PathLike)):
        score_sheets = [score_sheets]
    obs_date = _as_datetime(observations["Date"])
    success = False
    updated: list[tuple[Path, dict[str, Any]]] = []
    failed: list[Path] = []
    late_messages: list[str] = []
    for sheet in score_sheets:
        path = Path(sheet)
        xl = read_score_sheet(path, short_date_format=short_date_format, now=now)
        _check_layout(xl, path.name)
        num, txt = xl.num, xl.txt
        start_text = _txt_at(txt, 1, 3, "the start date")
        dateformat = _date_format(start_text)
        last = _last_data_index(num)
        last_obs_text = _txt_at(txt, 1, last + 1, "the date of the last observation")
        n_days = _days(obs_date, _parse_sheet_date(last_obs_text, dateformat))
        if n_days < 0:
            raise ValueError(f"{path.name}: It looks like you are trying to enter an observation "
                             "that precedes an existing observation - please fix the score "
                             "sheet manually")
        last_3 = [_num_at(num, 15, j, "a total score") for j in range(max(3, last - 1), last + 2)]
        if observation_too_late(n_days, last_3):
            msg = _late_message(n_days, last_3)
            ok = allow_late(msg) if callable(allow_late) else bool(allow_late)
            if not ok:
                logger.warning("%s: late observation not confirmed, update stopped", path.name)
                return ScoreSheetUpdate(success, updated, failed, path, late_messages)
            late_messages.append(msg)
        initial_weight = _num_at(num, 17, 1, "the initial weight")
        last_weight = _num_at(num, 19, last, "the last weight")
        rows = score_sheet_row_values(observations, initial_weight, last_weight)
        col = excel_column_name(last + 2)
        start = _parse_sheet_date(start_text, dateformat)
        cells: dict[str, Any] = {f"{col}1": _format_sheet_date(obs_date, dateformat),
                                 f"{col}2": _days(obs_date, start)}
        cells.update({f"{col}{r}": v for r, v in rows.items()})
        try:
            _write_cells(path, cells, now)
        except OSError as err:
            logger.warning("Writing to score sheet %s failed - it might be open (%s)",
                           path.name, err)
            failed.append(path)
            continue
        logger.info("Score sheet %s is updated.", path.name)
        success = True
        updated.append((path, {k: _cell_value(v) for k, v in cells.items()}))
    return ScoreSheetUpdate(success, updated, failed, None, late_messages)
