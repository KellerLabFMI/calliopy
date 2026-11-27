"""Spreadsheet formulas without a spreadsheet program (private helpers of the score sheets).

:class:`FormulaBook` evaluates the formulas a score sheet can contain with Excel's rules, and
:func:`store_formula_results` writes the results into an ``.xlsx`` file as the stored values
of the formula cells. Together they let calliopy update a score sheet that stays correct for
every reader, also when no spreadsheet program ever opens it.

Supported
---------
Numbers, text, logical values, error values, cell references (``A1``, ``$A$1``, ``Sheet!A1``,
``'My sheet'!A1``), ranges (``A1:B5``, ``A:A``, ``1:1``), defined names, the operators
``+ - * / ^ & % = <> < > <= >=`` and unary minus, and the functions SUM, AVERAGE, MIN, MAX,
COUNT, COUNTA, COUNTBLANK, IF, IFERROR, AND, OR, NOT, ROUND, ROUNDUP, ROUNDDOWN, ABS, ISBLANK,
ISNUMBER, TODAY, NOW, DATE, DAYS, COUNTIF, COUNTIFS, SUMIF, SUMIFS, MAXIFS and MINIFS. Anything
else (other functions, array formulas, external references) raises
:class:`UnsupportedFormulaError` naming it.

Excel semantics reproduced: blank cells count as 0 (or ``""``) in operations; aggregate functions
skip text, logical values and blanks in ranges; error values propagate; ``IF`` and ``IFERROR``
only evaluate the branch they return; dates are serial day numbers (1900 date system with
Excel's fictitious 1900-02-29, serial 60, or the 1904 system when the workbook uses it);
``ROUND`` rounds halves away from zero on the decimal value Excel shows.
"""

from __future__ import annotations

import datetime
import io
import logging
import math
import posixpath
import re
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from decimal import ROUND_DOWN, ROUND_HALF_UP, ROUND_UP, Decimal
from typing import Any
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

__all__ = [
    "BLANK",
    "FormulaBook",
    "FormulaError",
    "UnsupportedFormulaError",
    "XlError",
    "serial_to_datetime",
    "store_formula_results",
]


class FormulaError(ValueError):
    """A spreadsheet formula cannot be evaluated (circular reference, malformed formula)."""


class UnsupportedFormulaError(FormulaError):
    """A formula uses a function or construct calliopy does not evaluate."""


class XlError:
    """A spreadsheet error value such as ``#DIV/0!``."""

    __slots__ = ("code",)

    def __init__(self, code: str):
        self.code = code

    def __eq__(self, other: object) -> bool:
        return isinstance(other, XlError) and other.code == self.code

    def __hash__(self) -> int:
        return hash(self.code)

    def __repr__(self) -> str:
        return f"XlError({self.code!r})"

    def __str__(self) -> str:
        return self.code


class _Blank:
    __slots__ = ()

    def __repr__(self) -> str:
        return "BLANK"


#: Value of an empty cell.
BLANK = _Blank()
_MISSING = object()  # omitted function argument, as in IF(A1,,1)

_DIV0 = XlError("#DIV/0!")
_VALUE = XlError("#VALUE!")
_REF = XlError("#REF!")
_NAME = XlError("#NAME?")
_NUM = XlError("#NUM!")

_FUNCTIONS = frozenset({
    "SUM", "AVERAGE", "MIN", "MAX", "COUNT", "COUNTA", "COUNTBLANK", "IF", "IFERROR", "AND", "OR",
    "NOT", "ROUND", "ROUNDUP", "ROUNDDOWN", "ABS", "ISBLANK", "ISNUMBER", "TODAY", "NOW", "DATE",
    "DAYS", "COUNTIF", "COUNTIFS", "SUMIF", "SUMIFS", "MAXIFS", "MINIFS"})
_INFIX = {":": 8, "^": 5, "*": 4, "/": 4, "+": 3, "-": 3, "&": 2,
          "=": 1, "<>": 1, "<": 1, ">": 1, "<=": 1, ">=": 1}
_PART = r"\$?[A-Za-z]{1,3}\$?\d+|\$?[A-Za-z]{1,3}|\$?\d+"
_REF_RE = re.compile(rf"(?:(?:'(?P<qs>(?:[^']|'')+)'|(?P<s>[^'!]+))!)?(?P<a>{_PART})(?::(?P<b>{_PART}))?")
_CHAIN_RE = re.compile(rf"(?:(?:'(?P<qs>(?:[^']|'')+)'|(?P<s>[^'!]+))!)?(?P<rest>(?:{_PART})(?::(?:{_PART}))+)")
_NUMBER_TEXT = re.compile(r"\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\s*%?\s*")


# --- dates -------------------------------------------------------------------------------


_DAY0_1900 = datetime.datetime(1899, 12, 31)  # serial 1 = 1900-01-01
_DAY0_1900_LATE = datetime.datetime(1899, 12, 30)  # from 1900-03-01 (serial 61) on
_DAY0_1904 = datetime.datetime(1904, 1, 1)


def datetime_to_serial(d: datetime.date | datetime.datetime | datetime.time,
                       epoch1904: bool = False) -> float:
    """Excel serial day number of a date (fraction = time of day)."""
    if isinstance(d, datetime.time):
        return (d.hour * 3600 + d.minute * 60 + d.second + d.microsecond / 1e6) / 86400.0
    if not isinstance(d, datetime.datetime):
        d = datetime.datetime(d.year, d.month, d.day)
    if epoch1904:
        base = _DAY0_1904
    else:
        base = _DAY0_1900_LATE if d >= datetime.datetime(1900, 3, 1) else _DAY0_1900
    return (d - base).total_seconds() / 86400.0


def serial_to_datetime(x: float, epoch1904: bool = False) -> datetime.datetime:
    """Date of an Excel serial day number (serial 60, Excel's 1900-02-29, gives 1900-02-28)."""
    if epoch1904:
        base = _DAY0_1904
    elif x >= 61:
        base = _DAY0_1900_LATE
    else:
        base = _DAY0_1900
        x = min(x, 59 + (x - math.floor(x)))
    return base + datetime.timedelta(days=x)


def _date_serial(year: int, month: int, epoch1904: bool) -> float:
    return datetime_to_serial(datetime.date(year, month, 1), epoch1904)


# --- parsing -----------------------------------------------------------------------------


def _col_number(letters: str) -> int:
    n = 0
    for ch in letters.upper():
        n = n * 26 + ord(ch) - 64
    return n


def _split_part(part: str) -> tuple[str, int | None, int | None]:
    """``'$B$3'`` -> ``('cell', 3, 2)``; ``'B'`` -> ``('col', None, 2)``; ``'3'`` -> ``('row', 3, None)``."""
    p = part.replace("$", "")
    m = re.fullmatch(r"([A-Za-z]{1,3})(\d+)", p)
    if m:
        return "cell", int(m.group(2)), _col_number(m.group(1))
    if p.isdigit():
        return "row", int(p), None
    return "col", None, _col_number(p)


def _parse_ref(text: str, where: str) -> tuple:
    if "[" in text:
        raise UnsupportedFormulaError(f"external workbook reference {text!r} in {where}")
    m = _REF_RE.fullmatch(text)
    if not m:
        chain = _CHAIN_RE.fullmatch(text)  # A1:A2:C5, one token: the enclosing rectangle
        if chain:
            parts = [_split_part(p) for p in chain.group("rest").split(":")]
            if all(k == "cell" for k, _, _ in parts):
                sheet = chain.group("qs").replace("''", "'") if chain.group("qs") else chain.group("s")
                rows, cols = [r for _, r, _ in parts], [c for _, _, c in parts]
                return ("ref", sheet, min(rows), min(cols), max(rows), max(cols))
        return ("name", text)
    sheet = m.group("qs").replace("''", "'") if m.group("qs") else m.group("s")
    ka, ra, ca = _split_part(m.group("a"))
    if m.group("b") is None:
        if ka != "cell":
            return ("name", text)
        return ("ref", sheet, ra, ca, ra, ca)
    kb, rb, cb = _split_part(m.group("b"))
    if ka != kb:
        raise UnsupportedFormulaError(f"mixed range {text!r} in {where}")
    if ka == "cell":
        return ("ref", sheet, min(ra, rb), min(ca, cb), max(ra, rb), max(ca, cb))
    if ka == "col":
        return ("ref", sheet, None, min(ca, cb), None, max(ca, cb))
    return ("ref", sheet, min(ra, rb), None, max(ra, rb), None)


class _Parser:
    """Precedence-climbing parser over openpyxl's formula tokens."""

    def __init__(self, formula: str, where: str):
        from openpyxl.formula.tokenizer import Token, Tokenizer

        self.T = Token
        self.where = where
        try:
            items = Tokenizer(formula).items
        except Exception as err:  # openpyxl raises its own TokenizerError
            raise FormulaError(f"cannot parse formula {formula!r} in {where}: {err}") from err
        self.items = [t for t in items if t.type != Token.WSPACE]
        self.i = 0
        self.formula = formula

    def peek(self):
        return self.items[self.i] if self.i < len(self.items) else None

    def take(self):
        t = self.peek()
        if t is None:
            raise FormulaError(f"formula {self.formula!r} in {self.where} ends unexpectedly")
        self.i += 1
        return t

    def parse(self) -> tuple:
        node = self.expr(0)
        if self.i != len(self.items):
            raise FormulaError(f"cannot parse formula {self.formula!r} in {self.where}")
        return node

    def expr(self, min_prec: int) -> tuple:
        T = self.T
        left = self.unary()
        while (t := self.peek()) is not None:
            if t.type == T.OP_POST:
                self.i += 1
                left = ("pct", left)
            elif t.type == T.OP_IN and _INFIX.get(t.value, -1) >= min_prec:
                self.i += 1
                prec = _INFIX[t.value]
                left = ("bin", t.value, left, self.expr(prec + 1))
            else:
                break
        return left

    def unary(self) -> tuple:
        t = self.peek()
        if t is not None and t.type == self.T.OP_PRE:
            self.i += 1
            operand = self.unary()
            return ("neg", operand) if t.value == "-" else operand
        return self.primary()

    def primary(self) -> tuple:
        T = self.T
        t = self.take()
        if t.type == T.OPERAND:
            if t.subtype == T.NUMBER:
                return ("const", float(t.value))
            if t.subtype == T.TEXT:
                return ("const", t.value[1:-1].replace('""', '"'))
            if t.subtype == T.LOGICAL:
                return ("const", t.value.upper() == "TRUE")
            if t.subtype == T.ERROR:
                return ("const", XlError(t.value.upper()))
            return _parse_ref(t.value, self.where)
        if t.type == T.FUNC and t.subtype == T.OPEN:
            name = t.value[:-1].upper()
            for prefix in ("_XLFN.", "_XLWS."):
                name = name.removeprefix(prefix)
            if name not in _FUNCTIONS:
                raise UnsupportedFormulaError(
                    f"unsupported spreadsheet function {name} in {self.where}")
            args: list = []
            nxt = self.peek()
            if nxt is not None and nxt.type == T.FUNC and nxt.subtype == T.CLOSE:
                self.i += 1
                return ("func", name, args)
            while True:
                nxt = self.peek()
                if nxt is not None and (nxt.type == T.SEP or
                                        (nxt.type == T.FUNC and nxt.subtype == T.CLOSE)):
                    args.append(("missing",))
                else:
                    args.append(self.expr(0))
                sep = self.take()
                if sep.type == T.FUNC and sep.subtype == T.CLOSE:
                    return ("func", name, args)
                if not (sep.type == T.SEP and sep.subtype == T.ARG):
                    raise FormulaError(f"cannot parse formula {self.formula!r} in {self.where}")
        if t.type == T.PAREN and t.subtype == T.OPEN:
            node = self.expr(0)
            close = self.take()
            if not (close.type == T.PAREN and close.subtype == T.CLOSE):
                raise FormulaError(f"cannot parse formula {self.formula!r} in {self.where}")
            return node
        if t.type == T.ARRAY:
            raise UnsupportedFormulaError(f"array constant in {self.where}")
        raise FormulaError(f"cannot parse formula {self.formula!r} in {self.where}")


# --- evaluation --------------------------------------------------------------------------


class _Range:
    __slots__ = ("book", "c1", "c2", "r1", "r2", "sheet")

    def __init__(self, book: FormulaBook, sheet: str, r1: int, c1: int, r2: int, c2: int):
        self.book, self.sheet, self.r1, self.c1, self.r2, self.c2 = book, sheet, r1, c1, r2, c2

    @property
    def shape(self) -> tuple[int, int]:
        return self.r2 - self.r1 + 1, self.c2 - self.c1 + 1

    def single(self) -> bool:
        return self.r1 == self.r2 and self.c1 == self.c2

    def values(self) -> Iterator[Any]:
        for r in range(self.r1, self.r2 + 1):
            for c in range(self.c1, self.c2 + 1):
                yield self.book.value(self.sheet, r, c)


def _coord(sheet: str, r: int, c: int) -> str:
    from openpyxl.utils import get_column_letter

    return f"'{sheet}'!{get_column_letter(c)}{r}"


def _fmt_general(x: float) -> str:
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.15g}".upper()


def _round_decimal(x: float, digits: int, mode: str) -> float:
    if digits > 15:
        return x
    if digits < -308:
        return 0.0
    q = Decimal(1).scaleb(-digits)
    return float(Decimal(repr(x)).quantize(q, rounding=mode))


class FormulaBook:
    """Formula evaluation over an openpyxl workbook loaded with its formulas.

    Parameters
    ----------
    wb
        Workbook loaded with ``data_only=False`` (formula cells hold their formula text).
    stored
        The same workbook loaded with ``data_only=True``. When given, formula cells whose
        result is stored in the file use that result; the others are computed. When ``None``,
        every formula is computed from the cell values.
    now
        Clock of ``TODAY()`` and ``NOW()`` (local time); default the current time.
    """

    def __init__(self, wb, *, stored=None, now: datetime.datetime | None = None):
        from openpyxl.utils.datetime import CALENDAR_MAC_1904

        self.epoch1904 = wb.epoch == CALENDAR_MAC_1904
        self.now = now or datetime.datetime.now()
        self._consts: dict[str, dict[tuple[int, int], Any]] = {}
        self._formulas: dict[str, dict[tuple[int, int], Any]] = {}
        self._dims: dict[str, tuple[int, int]] = {}
        self._names: dict[tuple[str | None, str], str] = {}
        for name, dn in wb.defined_names.items():
            self._names[(None, name.upper())] = dn.attr_text
        for ws in wb.worksheets:
            consts, formulas = {}, {}
            for row in ws.iter_rows():
                for cell in row:
                    v = cell.value
                    if v is None:
                        continue
                    if cell.data_type == "f":
                        formulas[(cell.row, cell.column)] = v
                    else:
                        consts[(cell.row, cell.column)] = self._convert(v, cell.data_type)
            self._consts[ws.title] = consts
            self._formulas[ws.title] = formulas
            self._dims[ws.title] = (ws.max_row, ws.max_column)
            for name, dn in ws.defined_names.items():
                self._names[(ws.title, name.upper())] = dn.attr_text
        self._lower = {t.lower(): t for t in self._consts}
        self._memo: dict[tuple[str, int, int], Any] = {}
        if stored is not None:
            for ws in stored.worksheets:
                f = self._formulas.get(ws.title, {})
                for (r, c) in f:
                    cell = ws.cell(r, c)
                    if cell.value is not None:
                        self._memo[(ws.title, r, c)] = self._convert(cell.value, cell.data_type)
        self._parsed: dict[Any, tuple] = {}

    # -- public -----------------------------------------------------------------------

    @property
    def sheets(self) -> list[str]:
        return list(self._consts)

    def formula_cells(self, sheet: str) -> list[tuple[int, int]]:
        """1-based ``(row, column)`` of the formula cells of a sheet, in row order."""
        return sorted(self._formulas[sheet])

    def is_formula(self, sheet: str, row: int, col: int) -> bool:
        return (row, col) in self._formulas[sheet]

    def value(self, sheet: str, row: int, col: int) -> Any:
        """Value of a cell (1-based): float, str, bool, :class:`XlError` or :data:`BLANK`."""
        key = (sheet, row, col)
        if key in self._memo:
            return self._memo[key]
        if (row, col) in self._formulas[sheet]:
            self._evaluate_in_order(key)
            return self._memo[key]
        return self._consts[sheet].get((row, col), BLANK)

    # -- cells ------------------------------------------------------------------------

    def _convert(self, v: Any, data_type: str) -> Any:
        if data_type == "e":
            return XlError(str(v).upper())
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, (datetime.datetime, datetime.date, datetime.time)):
            return datetime_to_serial(v, self.epoch1904)
        if isinstance(v, datetime.timedelta):
            return v.total_seconds() / 86400.0
        return str(v)

    def _parse(self, sheet: str, row: int, col: int) -> tuple:
        key = (sheet, row, col)
        if key not in self._parsed:
            raw = self._formulas[sheet][(row, col)]
            where = _coord(sheet, row, col)
            if not isinstance(raw, str):
                raise UnsupportedFormulaError(f"array or data-table formula in {where}")
            self._parsed[key] = _Parser(raw, where).parse()
        return self._parsed[key]

    def _name_node(self, sheet: str, name: str) -> tuple | None:
        text = self._names.get((sheet, name.upper()), self._names.get((None, name.upper())))
        if text is None:
            return None
        key = ("name", sheet, name.upper())
        if key not in self._parsed:
            self._parsed[key] = _Parser("=" + text, f"defined name {name}").parse()
        return self._parsed[key]

    def _sheet(self, name: str | None, current: str) -> str | None:
        if name is None:
            return current
        return self._lower.get(name.lower())

    def _deps(self, key: tuple[str, int, int]) -> Iterator[tuple[str, int, int]]:
        sheet = key[0]
        stack = [self._parse(*key)]
        seen_names: set[str] = set()
        while stack:
            node = stack.pop()
            kind = node[0]
            if kind == "ref":
                s = self._sheet(node[1], sheet)
                if s is None:
                    continue
                r1, c1, r2, c2 = self._bounds(s, node)
                for (r, c) in self._formulas[s]:
                    if r1 <= r <= r2 and c1 <= c <= c2:
                        yield (s, r, c)
            elif kind == "name":
                if node[1].upper() not in seen_names:
                    seen_names.add(node[1].upper())
                    sub = self._name_node(sheet, node[1])
                    if sub is not None:
                        stack.append(sub)
            elif kind in ("bin",):
                stack.extend(node[2:])
            elif kind in ("neg", "pct"):
                stack.append(node[1])
            elif kind == "func":
                stack.extend(node[2])

    def _evaluate_in_order(self, key: tuple[str, int, int]) -> None:
        """Evaluate ``key`` after all formula cells it refers to (iterative, so long chains of
        dependent cells need no deep recursion); circular references raise."""
        order: list[tuple[str, int, int]] = []
        state = {key: 1}
        stack = [(key, self._deps(key))]
        while stack:
            k, it = stack[-1]
            for d in it:
                if d in self._memo:
                    continue
                s = state.get(d)
                if s == 1:
                    raise FormulaError(f"circular reference: {_coord(*d)} depends on itself")
                if s is None:
                    state[d] = 1
                    stack.append((d, self._deps(d)))
                    break
            else:
                stack.pop()
                state[k] = 2
                order.append(k)
        for k in order:
            self._memo[k] = self._final(self._eval(self._parse(*k), k[0]))

    def _final(self, v: Any) -> Any:
        v = self._scalar(v)
        return 0.0 if v is BLANK else v

    def _bounds(self, sheet: str, node: tuple) -> tuple[int, int, int, int]:
        _, _, r1, c1, r2, c2 = node
        max_row, max_col = self._dims[sheet]
        return (r1 or 1, c1 or 1, r2 if r2 is not None else max_row,
                c2 if c2 is not None else max_col)

    # -- conversions ------------------------------------------------------------------

    def _scalar(self, v: Any) -> Any:
        if isinstance(v, _Range):
            return self.value(v.sheet, v.r1, v.c1) if v.single() else _VALUE
        return BLANK if v is _MISSING else v

    def _num(self, v: Any) -> Any:
        v = self._scalar(v)
        if isinstance(v, XlError):
            return v
        if v is BLANK:
            return 0.0
        if isinstance(v, bool):
            return float(v)
        if isinstance(v, float):
            return v
        if isinstance(v, str) and _NUMBER_TEXT.fullmatch(v):
            s = v.strip()
            return float(s[:-1].strip()) / 100.0 if s.endswith("%") else float(s)
        return _VALUE

    def _text(self, v: Any) -> Any:
        v = self._scalar(v)
        if isinstance(v, XlError):
            return v
        if v is BLANK:
            return ""
        if isinstance(v, bool):
            return "TRUE" if v else "FALSE"
        if isinstance(v, float):
            return _fmt_general(v)
        return v

    def _bool(self, v: Any) -> Any:
        v = self._scalar(v)
        if isinstance(v, XlError):
            return v
        if v is BLANK:
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, float):
            return v != 0
        if v.upper() in ("TRUE", "FALSE"):
            return v.upper() == "TRUE"
        return _VALUE

    @staticmethod
    def _rank(v: Any) -> int:
        return 2 if isinstance(v, bool) else (1 if isinstance(v, str) else 0)

    def _compare(self, a: Any, b: Any) -> int:
        if a is BLANK and b is BLANK:
            return 0
        if a is BLANK:
            a = False if isinstance(b, bool) else ("" if isinstance(b, str) else 0.0)
        if b is BLANK:
            b = False if isinstance(a, bool) else ("" if isinstance(a, str) else 0.0)
        ra, rb = self._rank(a), self._rank(b)
        if ra != rb:
            return -1 if ra < rb else 1
        if isinstance(a, str):
            a, b = a.casefold(), b.casefold()
        return (a > b) - (a < b)

    # -- nodes ------------------------------------------------------------------------

    def _eval(self, node: tuple, sheet: str) -> Any:
        kind = node[0]
        if kind == "const":
            return node[1]
        if kind == "ref":
            s = self._sheet(node[1], sheet)
            if s is None:
                return _REF
            return _Range(self, s, *self._bounds(s, node))
        if kind == "name":
            sub = self._name_node(sheet, node[1])
            return _NAME if sub is None else self._eval(sub, sheet)
        if kind == "missing":
            return _MISSING
        if kind == "neg":
            x = self._num(self._eval(node[1], sheet))
            return x if isinstance(x, XlError) else -x
        if kind == "pct":
            x = self._num(self._eval(node[1], sheet))
            return x if isinstance(x, XlError) else x / 100.0
        if kind == "bin":
            return self._binary(node[1], node[2], node[3], sheet)
        if kind == "func":
            return getattr(self, "_f_" + node[1].lower())(node[2], sheet)
        raise FormulaError(f"unknown formula element {kind}")  # pragma: no cover

    def _binary(self, op: str, ln: tuple, rn: tuple, sheet: str) -> Any:
        if op == ":":
            a, b = self._eval(ln, sheet), self._eval(rn, sheet)
            if not (isinstance(a, _Range) and isinstance(b, _Range) and a.sheet == b.sheet):
                return _VALUE
            return _Range(self, a.sheet, min(a.r1, b.r1), min(a.c1, b.c1), max(a.r2, b.r2),
                          max(a.c2, b.c2))
        if op == "&":
            a = self._text(self._eval(ln, sheet))
            if isinstance(a, XlError):
                return a
            b = self._text(self._eval(rn, sheet))
            return b if isinstance(b, XlError) else a + b
        if op in ("=", "<>", "<", ">", "<=", ">="):
            a = self._scalar(self._eval(ln, sheet))
            if isinstance(a, XlError):
                return a
            b = self._scalar(self._eval(rn, sheet))
            if isinstance(b, XlError):
                return b
            c = self._compare(a, b)
            return {"=": c == 0, "<>": c != 0, "<": c < 0, ">": c > 0, "<=": c <= 0,
                    ">=": c >= 0}[op]
        a = self._num(self._eval(ln, sheet))
        if isinstance(a, XlError):
            return a
        b = self._num(self._eval(rn, sheet))
        if isinstance(b, XlError):
            return b
        if op == "+":
            r = a + b
        elif op == "-":
            r = a - b
        elif op == "*":
            r = a * b
        elif op == "/":
            if b == 0:
                return _DIV0
            r = a / b
        else:  # ^
            if a == 0 and b == 0:
                return _NUM
            if a == 0 and b < 0:
                return _DIV0
            if a < 0 and b != int(b):
                return _NUM
            try:
                r = math.pow(a, b)
            except OverflowError:
                return _NUM
        return r if math.isfinite(r) else _NUM

    # -- functions --------------------------------------------------------------------

    def _args(self, nodes: Sequence[tuple], sheet: str, name: str, lo: int,
              hi: int | None) -> list:
        if len(nodes) < lo or (hi is not None and len(nodes) > hi):
            raise FormulaError(f"wrong number of arguments for {name}")
        return [self._eval(n, sheet) for n in nodes]

    def _numbers(self, values: Sequence[Any]) -> Any:
        """Numbers of an aggregate function's arguments (Excel's SUM rules)."""
        out: list[float] = []
        for v in values:
            if isinstance(v, _Range):
                for x in v.values():
                    if isinstance(x, XlError):
                        return x
                    if isinstance(x, float):
                        out.append(x)
            else:
                x = self._num(v)
                if isinstance(x, XlError):
                    return x
                out.append(x)
        return out

    def _f_sum(self, nodes, sheet):
        nums = self._numbers(self._args(nodes, sheet, "SUM", 1, None))
        return nums if isinstance(nums, XlError) else float(math.fsum(nums))

    def _f_average(self, nodes, sheet):
        nums = self._numbers(self._args(nodes, sheet, "AVERAGE", 1, None))
        if isinstance(nums, XlError):
            return nums
        return math.fsum(nums) / len(nums) if nums else _DIV0

    def _f_min(self, nodes, sheet):
        nums = self._numbers(self._args(nodes, sheet, "MIN", 1, None))
        return nums if isinstance(nums, XlError) else (min(nums) if nums else 0.0)

    def _f_max(self, nodes, sheet):
        nums = self._numbers(self._args(nodes, sheet, "MAX", 1, None))
        return nums if isinstance(nums, XlError) else (max(nums) if nums else 0.0)

    def _f_count(self, nodes, sheet):
        n = 0
        for v in self._args(nodes, sheet, "COUNT", 1, None):
            if isinstance(v, _Range):
                n += sum(isinstance(x, float) for x in v.values())
            elif v is not _MISSING and not isinstance(self._num(v), XlError):
                n += 1
        return float(n)

    def _f_counta(self, nodes, sheet):
        n = 0
        for v in self._args(nodes, sheet, "COUNTA", 1, None):
            if isinstance(v, _Range):
                n += sum(x is not BLANK for x in v.values())
            elif v is not _MISSING:
                n += 1
        return float(n)

    def _f_countblank(self, nodes, sheet):
        (v,) = self._args(nodes, sheet, "COUNTBLANK", 1, 1)
        if not isinstance(v, _Range):
            return _VALUE
        return float(sum(x is BLANK or x == "" for x in v.values()))

    def _f_if(self, nodes, sheet):
        if not 1 <= len(nodes) <= 3:
            raise FormulaError("wrong number of arguments for IF")
        cond = self._bool(self._eval(nodes[0], sheet))
        if isinstance(cond, XlError):
            return cond
        idx = 1 if cond else 2
        if idx >= len(nodes):
            return idx != 2
        if nodes[idx][0] == "missing":
            return 0.0
        return self._eval(nodes[idx], sheet)

    def _f_iferror(self, nodes, sheet):
        if len(nodes) != 2:
            raise FormulaError("wrong number of arguments for IFERROR")
        v = self._scalar(self._eval(nodes[0], sheet))
        if isinstance(v, XlError):
            return 0.0 if nodes[1][0] == "missing" else self._eval(nodes[1], sheet)
        return v

    def _logicals(self, nodes, sheet, name) -> Any:
        out: list[bool] = []
        for v in self._args(nodes, sheet, name, 1, None):
            if isinstance(v, _Range):
                for x in v.values():
                    if isinstance(x, XlError):
                        return x
                    if isinstance(x, bool):
                        out.append(x)
                    elif isinstance(x, float):
                        out.append(x != 0)
            elif v is not _MISSING:
                b = self._bool(v)
                if isinstance(b, XlError):
                    return b
                out.append(b)
        return out if out else _VALUE

    def _f_and(self, nodes, sheet):
        v = self._logicals(nodes, sheet, "AND")
        return v if isinstance(v, XlError) else all(v)

    def _f_or(self, nodes, sheet):
        v = self._logicals(nodes, sheet, "OR")
        return v if isinstance(v, XlError) else any(v)

    def _f_not(self, nodes, sheet):
        (v,) = self._args(nodes, sheet, "NOT", 1, 1)
        b = self._bool(v)
        return b if isinstance(b, XlError) else not b

    def _round(self, nodes, sheet, name, mode):
        x, d = self._args(nodes, sheet, name, 2, 2)
        x, d = self._num(x), self._num(d)
        for v in (x, d):
            if isinstance(v, XlError):
                return v
        return _round_decimal(x, int(d), mode)

    def _f_round(self, nodes, sheet):
        return self._round(nodes, sheet, "ROUND", ROUND_HALF_UP)

    def _f_roundup(self, nodes, sheet):
        return self._round(nodes, sheet, "ROUNDUP", ROUND_UP)

    def _f_rounddown(self, nodes, sheet):
        return self._round(nodes, sheet, "ROUNDDOWN", ROUND_DOWN)

    def _f_abs(self, nodes, sheet):
        (v,) = self._args(nodes, sheet, "ABS", 1, 1)
        x = self._num(v)
        return x if isinstance(x, XlError) else abs(x)

    def _f_isblank(self, nodes, sheet):
        (v,) = self._args(nodes, sheet, "ISBLANK", 1, 1)
        return isinstance(v, _Range) and v.single() and self._scalar(v) is BLANK

    def _f_isnumber(self, nodes, sheet):
        (v,) = self._args(nodes, sheet, "ISNUMBER", 1, 1)
        x = self._scalar(v)
        return isinstance(x, float) and not isinstance(x, bool)

    def _f_today(self, nodes, sheet):
        self._args(nodes, sheet, "TODAY", 0, 0)
        return float(math.floor(datetime_to_serial(self.now, self.epoch1904)))

    def _f_now(self, nodes, sheet):
        self._args(nodes, sheet, "NOW", 0, 0)
        return datetime_to_serial(self.now, self.epoch1904)

    def _f_date(self, nodes, sheet):
        vals = [self._num(v) for v in self._args(nodes, sheet, "DATE", 3, 3)]
        for v in vals:
            if isinstance(v, XlError):
                return v
        y, m, d = (math.trunc(v) for v in vals)
        if 0 <= y < 1900:
            y += 1900
        y += (m - 1) // 12
        m = (m - 1) % 12 + 1
        if not 1900 <= y <= 9999:
            return _NUM
        serial = _date_serial(y, m, self.epoch1904) + d - 1  # crosses serial 60 like Excel
        return float(serial) if serial >= 0 else _NUM

    def _f_days(self, nodes, sheet):
        end, start = (self._num(v) for v in self._args(nodes, sheet, "DAYS", 2, 2))
        for v in (end, start):
            if isinstance(v, XlError):
                return v
        return float(math.trunc(end) - math.trunc(start))

    # criteria functions

    def _criterion(self, crit: Any) -> Callable[[Any], bool]:
        crit = self._scalar(crit)
        if isinstance(crit, XlError):
            return lambda x: x == crit
        if crit is BLANK:
            crit = 0.0
        if isinstance(crit, bool):
            return lambda x: isinstance(x, bool) and x == crit
        if isinstance(crit, float):
            return lambda x: isinstance(x, float) and not isinstance(x, bool) and x == crit
        m = re.fullmatch(r"(<=|>=|<>|<|>|=)?(.*)", crit, flags=re.DOTALL)
        op, operand = m.group(1), m.group(2)
        num = None
        if _NUMBER_TEXT.fullmatch(operand):
            num = self._num(operand)

        def is_num(x):
            return isinstance(x, float) and not isinstance(x, bool)

        if op in (None, "=", "<>"):
            if operand == "":
                match = (lambda x: x is BLANK or x == "")
            elif num is not None:
                match = (lambda x: is_num(x) and x == num)
            else:
                pattern = "".join(
                    ".*" if ch == "*" else "." if ch == "?" else re.escape(ch[-1])
                    if ch in ("~*", "~?") else re.escape(ch)
                    for ch in re.split(r"(~[*?]|[*?])", operand) if ch)
                rx = re.compile(pattern, flags=re.DOTALL | re.IGNORECASE)
                match = (lambda x: isinstance(x, str) and rx.fullmatch(x) is not None)
            if op == "<>":
                if operand == "":
                    return lambda x: x is not BLANK
                return lambda x: not match(x)
            return match
        cmp = {"<": lambda a, b: a < b, ">": lambda a, b: a > b, "<=": lambda a, b: a <= b,
               ">=": lambda a, b: a >= b}[op]
        if num is not None:
            return lambda x: is_num(x) and cmp(x, num)
        low = operand.casefold()
        return lambda x: isinstance(x, str) and cmp(x.casefold(), low)

    def _ifs(self, nodes, sheet, name, *, target_first: bool, op: str):
        """Shared body of COUNTIF(S), SUMIF(S), MAXIFS, MINIFS."""
        if name == "SUMIF":
            if len(nodes) not in (2, 3):
                raise FormulaError("wrong number of arguments for SUMIF")
            vals = [self._eval(n, sheet) for n in nodes]
            rng, crit = vals[0], vals[1]
            target = vals[2] if len(vals) == 3 else rng
            if not (isinstance(rng, _Range) and isinstance(target, _Range)):
                return _VALUE
            h, w = rng.shape
            target = _Range(self, target.sheet, target.r1, target.c1, target.r1 + h - 1,
                            target.c1 + w - 1)
            pairs = [(rng, crit)]
        else:
            vals = [self._eval(n, sheet) for n in nodes]
            target = vals[0] if target_first else None
            rest = vals[1:] if target_first else vals
            if len(rest) < 2 or len(rest) % 2:
                raise FormulaError(f"wrong number of arguments for {name}")
            pairs = list(zip(rest[::2], rest[1::2]))
            if not all(isinstance(r, _Range) for r, _ in pairs):
                return _VALUE
            if target is not None and not isinstance(target, _Range):
                return _VALUE
            shapes = {r.shape for r, _ in pairs} | ({target.shape} if target else set())
            if len(shapes) != 1:
                return _VALUE
        preds = [(list(r.values()), self._criterion(c)) for r, c in pairs]
        selected = [all(p(cells[i]) for cells, p in preds) for i in range(len(preds[0][0]))]
        if op == "count":
            return float(sum(selected))
        nums = []
        for ok, x in zip(selected, target.values()):
            if ok:
                if isinstance(x, XlError):
                    return x
                if isinstance(x, float) and not isinstance(x, bool):
                    nums.append(x)
        if op == "sum":
            return float(math.fsum(nums))
        if not nums:
            return 0.0
        return max(nums) if op == "max" else min(nums)

    def _f_countif(self, nodes, sheet):
        if len(nodes) != 2:
            raise FormulaError("wrong number of arguments for COUNTIF")
        return self._ifs(nodes, sheet, "COUNTIF", target_first=False, op="count")

    def _f_countifs(self, nodes, sheet):
        return self._ifs(nodes, sheet, "COUNTIFS", target_first=False, op="count")

    def _f_sumif(self, nodes, sheet):
        return self._ifs(nodes, sheet, "SUMIF", target_first=False, op="sum")

    def _f_sumifs(self, nodes, sheet):
        return self._ifs(nodes, sheet, "SUMIFS", target_first=True, op="sum")

    def _f_maxifs(self, nodes, sheet):
        return self._ifs(nodes, sheet, "MAXIFS", target_first=True, op="max")

    def _f_minifs(self, nodes, sheet):
        return self._ifs(nodes, sheet, "MINIFS", target_first=True, op="min")


# --- storing results in the xlsx file ----------------------------------------------------


_NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_DOC_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
_M = "{" + _NS_MAIN + "}"


def _xml_number(x: float) -> str:
    return str(int(x)) if x == int(x) and abs(x) < 1e15 else repr(x)


def _rels_path(part: str) -> str:
    d, b = posixpath.split(part)
    return posixpath.join(d, "_rels", b + ".rels")


def _resolve(base_part: str, target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join(posixpath.dirname(base_part), target))


def _workbook_part(zin: zipfile.ZipFile) -> str:
    root = ET.fromstring(zin.read("_rels/.rels"))
    for rel in root.iter(f"{{{_NS_PKG_REL}}}Relationship"):
        if rel.get("Type", "").endswith("/officeDocument"):
            return _resolve("", rel.get("Target"))
    raise FormulaError("no workbook part in the xlsx package")


def _sheet_parts(zin: zipfile.ZipFile, wb_part: str) -> dict[str, str]:
    wb = ET.fromstring(zin.read(wb_part))
    rels = ET.fromstring(zin.read(_rels_path(wb_part)))
    targets = {r.get("Id"): r.get("Target") for r in rels.iter(f"{{{_NS_PKG_REL}}}Relationship")}
    out = {}
    for s in wb.iter(f"{_M}sheet"):
        rid = s.get(f"{{{_NS_DOC_REL}}}id")
        if rid in targets:
            out[s.get("name")] = _resolve(wb_part, targets[rid])
    return out


def _rewrite_xml(data: bytes, edit: Callable[[ET.Element], None]) -> bytes:
    """Parse, edit and serialise an XML part, keeping its namespace prefixes."""
    decls = [ns for _, ns in ET.iterparse(io.BytesIO(data), events=("start-ns",))]
    saved = dict(ET._namespace_map)  # the prefix registry is global: restore it afterwards
    try:
        for prefix, uri in decls:
            try:
                ET.register_namespace(prefix, uri)
            except ValueError:
                pass
        root = ET.fromstring(data)
        edit(root)
        out = ET.tostring(root, encoding="UTF-8", xml_declaration=True)
    finally:
        ET._namespace_map.clear()
        ET._namespace_map.update(saved)
    # ElementTree drops unused declarations; prefixes named in mc:Ignorable must stay declared
    text = out.decode("utf-8")
    start = text.index("<", text.index("?>") + 2)
    end = text.index(">", start)
    head = text[start:end]
    extra = "".join(f' xmlns:{p}="{u}"' for p, u in dict(decls).items()
                    if p and f"xmlns:{p}=" not in head)
    if extra:
        if head.endswith("/"):
            head = head[:-1] + extra + "/"
        else:
            head = head + extra
        text = text[:start] + head + text[end:]
    return text.encode("utf-8")


def _set_results(root: ET.Element, results: Mapping[tuple[int, int], Any]) -> None:
    from openpyxl.utils.cell import coordinate_to_tuple

    data = root.find(f"{_M}sheetData")
    if data is None:
        return
    for row in data.iter(f"{_M}row"):
        for c in row.iter(f"{_M}c"):
            ref = c.get("r")
            if ref is None:
                continue
            rc = coordinate_to_tuple(ref)
            f = c.find(f"{_M}f")
            if rc not in results or f is None:
                continue
            for child in list(c):
                if child.tag in (f"{_M}v", f"{_M}is"):
                    c.remove(child)
            value = results[rc]
            v = ET.Element(f"{_M}v")
            if isinstance(value, XlError):
                c.set("t", "e")
                v.text = value.code
            elif isinstance(value, bool):
                c.set("t", "b")
                v.text = "1" if value else "0"
            elif isinstance(value, str):
                c.set("t", "str")
                v.text = value
            else:
                c.attrib.pop("t", None)
                v.text = _xml_number(float(value))
            c.insert(list(c).index(f) + 1, v)


def _set_full_calc(root: ET.Element) -> None:
    calc = root.find(f"{_M}calcPr")
    if calc is None:
        calc = ET.Element(f"{_M}calcPr")
        children = list(root)
        after = [i for i, ch in enumerate(children)
                 if ch.tag in (f"{_M}sheets", f"{_M}functionGroups", f"{_M}externalReferences",
                               f"{_M}definedNames")]
        root.insert(after[-1] + 1 if after else len(children), calc)
    calc.set("fullCalcOnLoad", "1")


def _results_editor(res: Mapping[tuple[int, int], Any]) -> Callable[[bytes], bytes]:
    def edit(data: bytes) -> bytes:
        return _rewrite_xml(data, lambda root: _set_results(root, res))
    return edit


def store_formula_results(src, dst, results: Mapping[str, Mapping[tuple[int, int], Any]]) -> None:
    """Copy the xlsx package ``src`` to ``dst``, storing formula results.

    ``results`` maps sheet name -> ``{(row, column): value}`` (1-based) of formula cells; each
    such cell keeps its formula and gets the value with its type (number, ``t="str"``,
    ``t="b"``, ``t="e"``). The workbook is flagged ``fullCalcOnLoad`` so a spreadsheet program
    recalculates on opening. Every other package part is copied unchanged.
    """
    with zipfile.ZipFile(src) as zin:
        wb_part = _workbook_part(zin)
        parts = _sheet_parts(zin, wb_part)
        edits: dict[str, Callable[[bytes], bytes]] = {
            wb_part: lambda data: _rewrite_xml(data, _set_full_calc)}
        for sheet, res in results.items():
            if sheet in parts and res:
                edits[parts[sheet]] = _results_editor(res)
        with zipfile.ZipFile(dst, "w") as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename in edits:
                    data = edits[item.filename](data)
                zout.writestr(item, data)
