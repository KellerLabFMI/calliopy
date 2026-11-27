"""Spreadsheet formulas without a spreadsheet program: evaluator, stored results, updates."""

import datetime
import zipfile

import numpy as np
import pytest

openpyxl = pytest.importorskip("openpyxl")

from openpyxl.utils import get_column_letter
from openpyxl.utils.datetime import CALENDAR_MAC_1904
from openpyxl.workbook.defined_name import DefinedName

from calliopy.twophoton import (
    FormulaError,
    UnsupportedFormulaError,
    check_score_sheet_mouse,
    read_score_sheet,
    score_sheet_weights,
    update_score_sheet,
)
from calliopy.twophoton import _xlsx_formula as xf

NOW = datetime.datetime(2024, 3, 10, 15, 30)


def ev(cells, formula, *, now=NOW, epoch1904=False):
    """Value of ``formula`` in Z99 of a one-sheet workbook holding ``cells``."""
    wb = openpyxl.Workbook()
    if epoch1904:
        wb.epoch = CALENDAR_MAC_1904
    ws = wb.active
    ws.title = "S"
    for addr, v in cells.items():
        ws[addr] = v
    ws["Z99"] = formula
    return xf.FormulaBook(wb, now=now).value("S", 99, 26)


E = xf.XlError


# --- evaluator ---------------------------------------------------------------------------


def test_operators_and_precedence():
    assert ev({}, "=1+2*3") == 7
    assert ev({}, "=-2^2") == 4  # unary minus binds tighter than ^
    assert ev({}, "=2^3^2") == 64  # ^ is left-associative
    assert ev({}, "=2*-3^2") == 18
    assert ev({}, "=50%") == 0.5 and ev({}, "=-50%*2") == -1
    assert ev({}, '="a"&1&TRUE') == "a1TRUE"
    assert ev({}, "=0.1+0.2&\"\"") == "0.3"
    assert ev({}, "=(1+2)*3") == 9 and ev({}, "=1 + 1E+3") == 1001
    assert ev({}, "=1/0") == E("#DIV/0!") and ev({}, "=(-8)^0.5") == E("#NUM!")
    assert ev({}, '="3"+1') == 4 and ev({}, '="x"+1') == E("#VALUE!")


def test_comparisons_and_blanks():
    assert ev({}, '="a"<"B"') is True  # case-insensitive
    assert ev({}, '=1<"a"') is True and ev({}, '="z"<TRUE') is True  # numbers < text < logical
    assert ev({}, '="A"="a"') is True and ev({}, "=1<>1") is False
    assert ev({}, "=A1+1") == 1  # blank counts as 0
    assert ev({}, '=A1=""') is True and ev({}, "=A1=0") is True and ev({}, "=A1=FALSE") is True
    assert ev({}, "=A1") == 0  # a formula never returns a blank
    assert ev({"A1": "t"}, "=A1&A2") == "t"


def test_references_and_ranges():
    cells = {"A1": 1, "A2": 2, "B1": 3, "B2": "x", "C5": 10}
    assert ev(cells, "=SUM(A1:B2)") == 6  # text in a range is skipped
    assert ev(cells, "=SUM($A$1,A2)") == 3
    assert ev(cells, "=SUM(A:A)") == 3 and ev(cells, "=SUM(1:1)") == 4
    assert ev(cells, "=SUM(A1:A2:C5)") == 16  # range operator
    assert ev(cells, "=A1:A2") == E("#VALUE!")  # no implicit intersection
    assert ev(cells, "=SUM(B2)") == 0 and ev(cells, '=SUM("x")') == E("#VALUE!")
    assert ev(cells, "=SUM(TRUE,1)") == 2 and ev({"A1": True}, "=SUM(A1:A1)") == 0
    assert ev({"A1": "#N/A", "A2": 1}, "=SUM(A1:A2)") == E("#N/A")
    assert ev(cells, "=Nope*2") == E("#NAME?") and ev(cells, "='No sheet'!A1") == E("#REF!")


def test_aggregates():
    cells = {"A1": 1, "A2": 4, "A3": "t", "A4": True, "A6": ""}
    assert ev(cells, "=AVERAGE(A1:A5)") == 2.5 and ev({}, "=AVERAGE(A1:A3)") == E("#DIV/0!")
    assert ev(cells, "=MIN(A1:A5)") == 1 and ev(cells, "=MAX(A1:A5,7)") == 7
    assert ev({}, "=MAX(A1:A3)") == 0
    assert ev(cells, "=COUNT(A1:A6,1,\"2\",\"x\")") == 4
    assert ev(cells, "=COUNTA(A1:A6)") == 5 and ev(cells, "=COUNTBLANK(A1:A6)") == 2


def test_logic_and_errors():
    assert ev({}, '=IF(TRUE,1,1/0)') == 1  # only the returned branch is evaluated
    assert ev({}, "=IF(FALSE,1)") is False and ev({}, "=IF(FALSE,1,)") == 0
    assert ev({}, '=IF("x",1,2)') == E("#VALUE!") and ev({}, "=IF(1/0,1,2)") == E("#DIV/0!")
    assert ev({}, '=IFERROR(1/0,"e")') == "e" and ev({}, "=IFERROR(5,0)") == 5
    assert ev({"A1": 1, "A2": "x"}, "=AND(A1:A2,TRUE)") is True
    assert ev({}, "=OR(0,FALSE)") is False and ev({}, "=NOT(0)") is True
    assert ev({}, "=AND(A1:A3)") == E("#VALUE!")
    assert ev({}, "=ISBLANK(A1)") is True and ev({"A1": ""}, "=ISBLANK(A1)") is False
    assert ev({"A1": "3"}, "=ISNUMBER(A1)") is False and ev({}, "=ISNUMBER(1/0)") is False
    assert ev({}, "=ISNUMBER(TRUE)") is False and ev({}, "=ISNUMBER(2)") is True


def test_rounding_and_abs():
    assert ev({}, "=ROUND(2.675,2)") == 2.68  # Excel rounds the shown decimal value
    assert ev({}, "=ROUND(-2.5,0)") == -3 and ev({}, "=ROUND(1234.5,-2)") == 1200
    assert ev({}, "=ROUNDUP(1.21,1)") == 1.3 and ev({}, "=ROUNDUP(-1.21,1)") == -1.3
    assert ev({}, "=ROUNDDOWN(-1.29,1)") == -1.2 and ev({}, "=ABS(-4)") == 4


def test_dates():
    assert ev({}, "=DATE(2024,3,1)") == 45352
    assert ev({}, "=DATE(1900,2,29)") == 60 and ev({}, "=DATE(1900,3,1)") == 61  # leap-year bug
    assert ev({}, "=DATE(1900,1,1)") == 1 and ev({}, "=DATE(24,3,1)") == ev({}, "=DATE(1924,3,1)")
    assert ev({}, "=DATE(2023,15,1)") == ev({}, "=DATE(2024,3,1)")  # month overflow
    assert ev({}, "=DATE(2024,3,0)") == ev({}, "=DATE(2024,2,29)")
    assert ev({}, "=DATE(2024,3,1)", epoch1904=True) == 45352 - 1462
    assert ev({}, "=TODAY()") == 45352 + 9 and ev({}, "=NOW()") == pytest.approx(45361 + 15.5 / 24)
    assert ev({}, "=TODAY()", epoch1904=True) == 45361 - 1462
    assert ev({}, "=DAYS(DATE(2024,3,10),DATE(2024,2,28))") == 11
    assert ev({"A1": datetime.datetime(2024, 3, 1)}, "=A1+1") == 45353  # date cell -> serial
    assert xf.serial_to_datetime(45352) == datetime.datetime(2024, 3, 1)
    assert xf.serial_to_datetime(1) == datetime.datetime(1900, 1, 1)
    assert xf.datetime_to_serial(datetime.date(1900, 2, 28)) == 59


def test_criteria_functions():
    cells = {"A1": 1, "A2": 5, "A3": "apple", "A4": "*", "A6": 9,
             "B1": 10, "B2": 20, "B3": 30, "B4": 40, "B5": 50, "B6": 60}
    assert ev(cells, '=COUNTIF(A1:A6,">1")') == 2 and ev(cells, "=COUNTIF(A1:A6,5)") == 1
    assert ev(cells, '=COUNTIF(A1:A6,"a*")') == 1 and ev(cells, '=COUNTIF(A1:A6,"~*")') == 1
    assert ev(cells, '=COUNTIF(A1:A6,"")') == 1 and ev(cells, '=COUNTIF(A1:A6,"<>")') == 5
    assert ev(cells, '=COUNTIF(A1:A6,"<>5")') == 5
    assert ev(cells, '=SUMIF(A1:A6,">=5",B1:B6)') == 80 and ev(cells, '=SUMIF(A1:A6,">2")') == 14
    assert ev(cells, '=SUMIF(A1:A6,">=5",B1)') == 80  # sum range resized as Excel does
    assert ev(cells, '=_xlfn.MAXIFS(B1:B6,A1:A6,"<9")') == 20
    assert ev(cells, '=_xlfn.MINIFS(B1:B6,A1:A6,">1")') == 20
    assert ev(cells, '=MAXIFS(B1:B6,A1:A6,">100")') == 0
    assert ev(cells, '=COUNTIFS(A1:A6,">0",B1:B6,"<50")') == 2
    assert ev(cells, '=SUMIFS(B1:B6,A1:A6,">0",B1:B6,">10")') == 80
    assert ev(cells, '=MAXIFS(B1:B6,A1:A5,">0")') == E("#VALUE!")  # shapes differ
    assert ev(cells, '=COUNTIF(A1:A6,">="&DATE(1900,1,5))') == 2


def test_other_sheets_and_names():
    wb = openpyxl.Workbook()
    a = wb.active
    a.title = "Score sheet"
    a["B18"] = 20
    b = wb.create_sheet("Sum")
    b["A1"] = "='Score sheet'!B18*2"
    b["A2"] = "='score SHEET'!$B$18"  # sheet names: any case
    b["A3"] = "=Initial+1"
    b["A4"] = "=Local+1"
    wb.defined_names["Initial"] = DefinedName("Initial", attr_text="'Score sheet'!$B$18")
    b.defined_names["Local"] = DefinedName("Local", attr_text="'Score sheet'!$B$18")
    book = xf.FormulaBook(wb)
    assert [book.value("Sum", r, 1) for r in (1, 2, 3, 4)] == [40, 20, 21, 21]


def test_dependencies_cycles_and_unsupported():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "S"
    ws["A1"] = 1
    for r in range(2, 3001):  # a long chain needs no deep recursion
        ws.cell(r, 1, f"=A{r - 1}+1")
    ws["B1"] = "=C1+1"
    ws["C1"] = "=SUM(B1:B2)"
    ws["D1"] = "=VLOOKUP(1,A1:B2,2)"
    ws["E1"] = "=IF(A1,{1,2})"
    book = xf.FormulaBook(wb)
    assert book.value("S", 3000, 1) == 3000
    with pytest.raises(FormulaError, match="circular"):
        book.value("S", 1, 2)
    with pytest.raises(UnsupportedFormulaError, match="VLOOKUP"):
        book.value("S", 1, 4)
    with pytest.raises(UnsupportedFormulaError, match="array"):
        book.value("S", 1, 5)


# --- storing results ---------------------------------------------------------------------


def test_store_results_types_and_other_parts_unchanged(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "S"
    ws["A1"] = 2
    for addr, f in (("B1", "=A1*2"), ("B2", '="a"&"b"'), ("B3", "=A1>1"), ("B4", "=1/0"),
                    ("B5", '=""')):
        ws[addr] = f
    wb.create_sheet("T")["A1"] = "=S!A1"
    src, dst = tmp_path / "src.xlsx", tmp_path / "dst.xlsx"
    wb.save(src)
    book = xf.FormulaBook(openpyxl.load_workbook(src))
    results = {s: {rc: book.value(s, *rc) for rc in book.formula_cells(s)} for s in book.sheets}
    xf.store_formula_results(src, dst, results)
    v = openpyxl.load_workbook(dst, data_only=True)
    s = v["S"]
    assert [s[a].value for a in ("B1", "B2", "B3", "B4")] == [4, "ab", True, "#DIV/0!"]
    assert [s[a].data_type for a in ("B1", "B2", "B3", "B4")] == ["n", "s", "b", "e"]
    assert s["B5"].value in ("", None) and v["T"]["A1"].value == 2
    f = openpyxl.load_workbook(dst)
    assert f["S"]["B1"].value == "=A1*2" and f["T"]["A1"].value == "=S!A1"
    with zipfile.ZipFile(src) as a, zipfile.ZipFile(dst) as b:
        assert a.namelist() == b.namelist()
        changed = [n for n in a.namelist() if a.read(n) != b.read(n)]
        assert 'fullCalcOnLoad="1"' in b.read("xl/workbook.xml").decode()
    assert {"xl/worksheets/sheet1.xml", "xl/worksheets/sheet2.xml"} <= set(changed)
    assert set(changed) <= {"xl/workbook.xml", "xl/worksheets/sheet1.xml",
                            "xl/worksheets/sheet2.xml"}


def test_rewrite_keeps_ignorable_namespaces():
    data = (b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            b'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
            b'xmlns:x14ac="http://schemas.microsoft.com/office/spreadsheetml/2009/9/ac" '
            b'mc:Ignorable="x14ac"><sheetData><row r="1"><c r="A1"><f>1+1</f></c></row>'
            b'</sheetData></worksheet>')
    out = xf._rewrite_xml(data, lambda root: xf._set_results(root, {(1, 1): 2.0})).decode()
    assert 'xmlns:x14ac="' in out and 'mc:Ignorable="x14ac"' in out
    assert "<f>1+1</f><v>2</v>" in out and "<worksheet xmlns=" in out


# --- score sheets with formulas ----------------------------------------------------------

LABELS = {5: "Self-mutilation", 6: "Lack of grooming", 7: "Hunched posture",
          8: "Orbital tightening", 10: "Not responsive to touch", 12: "Swelling",
          13: "Minor bleeding", 14: "Infection or major bleeding", 15: "Body weight score",
          16: "TOTAL SCORE", 17: "Alert", 18: "Initial weight", 19: "Endpoint weight",
          20: "Body weight", 21: "Change to initial (%)", 22: "Change to last (%)"}

OBS = {"Selfmutliation": 0, "Lackofgrooming": 1, "Hunchedposture": 1, "Orbitaltightening": 2,
       "Notresponsivetotouch": 0, "Swelling": 0, "Minorbleeding": 1,
       "Infectionofmajorbleeding": 0, "BodyWeight": 18.0}
SCORE_ROWS = (5, 6, 7, 8, 10, 12, 13, 14, 15)


def make_formula_sheet(path, last_col=26):
    """Template whose totals and summaries are formulas, saved without stored results."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Score sheet"
    for r, label in LABELS.items():
        ws.cell(r, 1, label)
    ws["A1"], ws["A2"], ws["A3"] = "Date", "XYZ240301", "Day"
    ws["C1"], ws["C2"], ws["C3"] = "01.03.2024", 0, "1&2"
    ws["B18"] = 20.0
    ws["B19"] = "=B18*0.8"
    ws["D1"], ws["D2"], ws["D5"], ws["D20"] = "02.03.2024", 1, 1, 19.5
    for c in range(3, last_col + 1):
        col = get_column_letter(c)
        ws[f"{col}16"] = f"=SUM({col}5:{col}15)"
        ws[f"{col}17"] = f'=IF({col}16>=3,"ALERT",IF(ISBLANK({col}1),"",{col}16))'
    ws["C24"] = "=DATE(2024,3,1)"
    ws["C24"].number_format = "dd.mm.yyyy"
    ws["C25"] = "=DAYS(TODAY(),C24)"
    summ = wb.create_sheet("Summary")
    summ["A1"] = "='Score sheet'!B18"
    summ["A2"] = "=MAX('Score sheet'!C16:Z16)"
    summ["A3"] = "=COUNTIF('Score sheet'!D1:Z1,\"<>\")"
    summ["A4"] = "=SUMIF('Score sheet'!C16:Z16,\">0\")"
    summ["A5"] = "=Initial*2"
    summ["A6"] = "=IFERROR(1/0,-1)"
    wb.defined_names["Initial"] = DefinedName("Initial", attr_text="'Score sheet'!$B$18")
    wb.save(path)
    return path


def test_read_computes_missing_results(tmp_path):
    p = make_formula_sheet(tmp_path / "t.xlsx")
    xl = read_score_sheet(p, now=NOW)
    assert xl.num_origin == (2, 2) and xl.txt_origin == (1, 1)
    assert xl.num[14, 2] == 1 and xl.num[14, 1] == 0 and xl.num[14, 24] == 0  # D16, C16, Z16
    assert xl.num[17, 0] == 16  # B19 = B18*0.8
    assert xl.txt[23, 2] == "01.03.2024"  # date formula rendered as Excel shows it
    assert xl.num[23, 1] == 9  # C25 = DAYS(TODAY(), C24) with the injected clock
    assert read_score_sheet(p, basic=True, now=NOW).num[22, 1] == 45352
    assert tuple(score_sheet_weights(p, now=NOW)) == (20.0, 19.5)


def test_consecutive_updates_without_spreadsheet_program(tmp_path):
    p = make_formula_sheet(tmp_path / "ScoreSheet_XYZ240301.xlsx")
    totals = {4: 1.0}  # sheet column -> total score
    weights = [18.0, 17.5, 17.0]
    for k, (day, bw) in enumerate(zip((3, 4, 5), weights)):
        res = update_score_sheet(p, {**OBS, "BodyWeight": bw, "Date": datetime.date(2024, 3, day)},
                                 now=NOW)
        assert res.success and res.cancelled is None
        cells = res.updated[0][1]
        col = 5 + k
        letter = get_column_letter(col)
        assert f"{letter}1" in cells
        totals[col] = sum(cells[f"{letter}{r}"] for r in SCORE_ROWS)
        # read back exactly as MATLAB would, no spreadsheet program in between
        xl = read_score_sheet(p, now=NOW)
        for c, t in totals.items():
            assert xl.num[14, c - 2] == t, (k, c)
        assert xl.num[14, col - 1] == 0  # next, empty column: SUM of blanks
        if totals[col] >= 3:  # dependent formula on the total
            assert xl.txt[16, col - 1] == "ALERT" and np.isnan(xl.num[15, col - 2])
        else:
            assert xl.num[15, col - 2] == totals[col]
        # stored results are what any xlsx reader sees
        v = openpyxl.load_workbook(p, data_only=True)
        assert v["Score sheet"][f"{letter}16"].value == totals[col]
        assert v["Summary"]["A2"].value == max(totals.values())
        assert v["Summary"]["A3"].value == 1 + (k + 1)
        assert v["Summary"]["A4"].value == sum(totals.values())
        assert v["Summary"]["A5"].value == 40 and v["Summary"]["A6"].value == -1
        assert v["Score sheet"]["B19"].value == 16
        f = openpyxl.load_workbook(p)
        assert f["Score sheet"][f"{letter}16"].value == f"=SUM({letter}5:{letter}15)"
        assert f.calculation.fullCalcOnLoad
    assert tuple(score_sheet_weights(p)) == (20.0, 17.0)
    assert not check_score_sheet_mouse(p, datetime.datetime(2024, 3, 6))
    assert check_score_sheet_mouse(p, datetime.datetime(2024, 3, 7))
    assert not list(tmp_path.glob(".*calliopy-tmp*"))


def test_update_unsupported_function_leaves_file_unchanged(tmp_path):
    p = make_formula_sheet(tmp_path / "u.xlsx")
    wb = openpyxl.load_workbook(p)
    wb["Score sheet"]["A30"] = "=VLOOKUP(1,B18:B19,1)"
    wb.save(p)
    before = p.read_bytes()
    with pytest.raises(UnsupportedFormulaError, match="VLOOKUP"):
        update_score_sheet(p, {**OBS, "Date": datetime.date(2024, 3, 3)}, now=NOW)
    assert p.read_bytes() == before
    # the same with a stored result (as Excel would have left it): reading works, the
    # update still refuses to write a workbook whose formulas it cannot recompute
    tmp = tmp_path / "u2.xlsx"
    xf.store_formula_results(p, tmp, {"Score sheet": {(30, 1): "n/a"}})
    tmp.replace(p)
    before = p.read_bytes()
    read_score_sheet(p, now=NOW)
    with pytest.raises(UnsupportedFormulaError, match="VLOOKUP"):
        update_score_sheet(p, {**OBS, "Date": datetime.date(2024, 3, 3)}, now=NOW)
    assert p.read_bytes() == before
    assert not list(tmp_path.glob(".*calliopy-tmp*"))


def test_update_other_sheet_unsupported_is_left_without_result(tmp_path):
    p = make_formula_sheet(tmp_path / "o.xlsx")
    wb = openpyxl.load_workbook(p)
    wb["Summary"]["B1"] = "=VLOOKUP(1,A1:A2,1)"
    wb.save(p)
    res = update_score_sheet(p, {**OBS, "Date": datetime.date(2024, 3, 3)}, now=NOW)
    assert res.success
    v = openpyxl.load_workbook(p, data_only=True)
    assert v["Summary"]["B1"].value is None and v["Summary"]["A3"].value == 2
    assert openpyxl.load_workbook(p)["Summary"]["B1"].value == "=VLOOKUP(1,A1:A2,1)"


def test_dates_in_1904_workbook(tmp_path):
    wb = openpyxl.Workbook()
    wb.epoch = CALENDAR_MAC_1904
    ws = wb.active
    ws["A1"] = "x"
    ws["B1"] = "=DATE(2024,3,1)"
    ws["B1"].number_format = "dd.mm.yyyy"
    ws["C1"] = "=B1+1"
    p = tmp_path / "e.xlsx"
    wb.save(p)
    xl = read_score_sheet(p)
    assert list(xl.txt[0]) == ["x", "01.03.2024"]
    np.testing.assert_array_equal(xl.num, [[45353 - 1462]])
