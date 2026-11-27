"""Score-sheet workbooks: xlsread-like reading, checks and updates (synthetic xlsx files)."""

import datetime
import logging
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

openpyxl = pytest.importorskip("openpyxl")

from openpyxl.comments import Comment
from openpyxl.styles import Font, PatternFill
from openpyxl.workbook.defined_name import DefinedName

import calliopy
from calliopy._matlab import mround
from calliopy.twophoton import _scoresheet_xlsx as ssx
from calliopy.twophoton import (
    body_weight_score,
    check_score_sheet_mouse,
    check_score_sheets,
    last_mouse_observation,
    read_score_sheet,
    score_sheet_path,
    score_sheet_problem,
    score_sheet_weights,
    update_score_sheet,
)

LABELS = {5: "Self-mutilation", 6: "Lack of grooming", 7: "Hunched posture",
          8: "Orbital tightening", 10: "Not responsive to touch", 12: "Swelling",
          13: "Minor bleeding", 14: "Infection or major bleeding", 15: "Body weight score",
          16: "TOTAL SCORE", 18: "Initial weight", 19: "Endpoint weight", 20: "Body weight",
          21: "Change to initial (%)", 22: "Change to last (%)"}

OBS = {"Selfmutliation": 0, "Lackofgrooming": 1, "Hunchedposture": 1, "Orbitaltightening": 2,
       "Notresponsivetotouch": 0, "Swelling": 0, "Minorbleeding": 1,
       "Infectionofmajorbleeding": 0, "BodyWeight": 18.0}


def make_sheet(path, *, start="01.03.2024", obs=(), initial=20.0, prefill_to=26,
               prefill_from=None, mouse="XYZ240301", total_label="TOTAL SCORE"):
    """Score sheet in the template layout: labels in column A, start in column C, one
    observation per later column. ``obs``: ``(date_text, total_score, weight)``. Row 16 holds a
    total of 0 in every later column up to ``prefill_to`` (as a template with stored formula
    results), starting at ``prefill_from`` (default: the first empty column)."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Score sheet"
    for r, label in LABELS.items():
        ws.cell(r, 1, label)
    ws.cell(16, 1, total_label)
    ws["A1"] = "Date"
    ws["A2"] = mouse
    ws["A3"] = "Day"
    ws["C1"] = start
    ws["C2"] = 0
    ws["C3"] = "1&2"
    ws["C16"] = 0
    ws["B18"] = initial
    ws["B19"] = 0.8 * initial
    start_dt = ssx._parse_sheet_date(start, ssx._date_format(start))
    for k, (date_text, total, weight) in enumerate(obs):
        col = 4 + k
        ws.cell(1, col, date_text)
        d = ssx._parse_sheet_date(date_text, ssx._date_format(start))
        ws.cell(2, col, (d - start_dt).days)
        ws.cell(16, col, total)
        if weight is not None:
            ws.cell(20, col, weight)
    first_empty = 4 + len(obs) if prefill_from is None else prefill_from
    for col in range(first_empty, prefill_to + 1):
        ws.cell(16, col, 0)
    wb.save(path)
    return path


def daily_obs(n, total=1.0, start=datetime.date(2024, 3, 1), weight=19.5):
    return [((start + datetime.timedelta(days=k)).strftime("%d.%m.%Y"), total, weight)
            for k in range(1, n + 1)]


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


# --- reading -----------------------------------------------------------------------------


def test_read_template_blocks(tmp_path):
    p = make_sheet(tmp_path / "s.xlsx", obs=daily_obs(2))
    xl = read_score_sheet(p)
    assert xl.txt_origin == (1, 1) and xl.num_origin == (2, 2)
    assert xl.txt[0, 2] == "01.03.2024" and xl.txt[0, 4] == "03.03.2024"
    assert xl.txt[15, 0] == "TOTAL SCORE" and xl.txt[1, 0] == "XYZ240301"
    assert xl.txt[1, 2] == ""  # number cell
    assert np.isnan(xl.num[0, 0]) and xl.num[0, 1] == 0 and xl.num[0, 3] == 2  # B2, C2, E2
    assert xl.num[16, 0] == 20.0  # B18 = num(17, 1)
    assert xl.num[14, 2] == 1.0  # D16 = num(15, 3)
    assert xl.num.shape == (19, 25)  # rows 2-20, columns B-Z
    assert xl.txt.dtype == object and xl.txt.shape == (22, 5)


def test_read_trimming_and_cell_kinds(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws["B3"] = 5
    ws["C5"] = True
    ws["D4"] = "x"
    ws["E2"] = "#DIV/0!"
    ws["F6"] = ""
    p = tmp_path / "k.xlsx"
    wb.save(p)
    xl = read_score_sheet(p)
    assert xl.num_origin == (3, 2) and xl.num.shape == (3, 2)
    assert xl.num[0, 0] == 5 and xl.num[2, 1] == 1.0 and np.isnan(xl.num[1, 0])
    assert xl.txt_origin == (2, 4) and xl.txt.shape == (3, 2)  # '' in F6 does not count
    assert xl.txt[2, 0] == "x" and xl.txt[0, 1] == "#N/A"
    assert read_score_sheet(p, basic=True).txt[0, 1] == "#DIV/0!"
    empty = tmp_path / "e.xlsx"
    openpyxl.Workbook().save(empty)
    xl = read_score_sheet(empty)
    assert xl.num.shape == (0, 0) and xl.txt.shape == (0, 0) and xl.num_origin is None


def test_read_dates_excel_and_basic(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    d = datetime.datetime(2024, 3, 1)
    for addr, fmt in (("A1", "dd.mm.yyyy"), ("B1", "dd-mmm-yy"), ("C1", "mm-dd-yy")):
        ws[addr] = d
        ws[addr].number_format = fmt
    ws["D1"] = datetime.datetime(2024, 3, 1, 13, 5)
    ws["D1"].number_format = "d/m/yyyy h:mm AM/PM"
    p = tmp_path / "d.xlsx"
    wb.save(p)
    xl = read_score_sheet(p)
    assert list(xl.txt[0]) == ["01.03.2024", "01-Mar-24", "01.03.2024", "1/3/2024 1:05 PM"]
    assert xl.num.shape == (0, 0)
    assert read_score_sheet(p, short_date_format="dd/mm/yyyy").txt[0, 2] == "01/03/2024"
    # basic mode: Excel serial day numbers (1900 system: days since 1899-12-30)
    xb = read_score_sheet(p, basic=True)
    assert xb.txt.shape == (0, 0)
    np.testing.assert_allclose(xb.num[0], [45352, 45352, 45352, 45352 + (13 * 60 + 5) / 1440],
                               rtol=0, atol=1e-9)
    wb.epoch = openpyxl.utils.datetime.CALENDAR_MAC_1904
    wb.save(p)
    assert read_score_sheet(p, basic=True).num[0, 0] == 45352 - 1462
    assert read_score_sheet(p).txt[0, 0] == "01.03.2024"


def test_format_minutes_versus_months():
    t = datetime.datetime(2024, 3, 1, 9, 7, 5)
    assert ssx._format_excel_date(t, "hh:mm:ss", "dd.mm.yyyy") == "09:07:05"
    assert ssx._format_excel_date(t, "mm/dd/yyyy", "dd.mm.yyyy") == "03/01/2024"
    assert ssx._format_excel_date(t, '[$-409]dddd, d mmmm yyyy', "x") == "Friday, 1 March 2024"


def test_formula_without_stored_result_is_computed(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws["A1"] = "x"
    ws["B1"] = 1
    ws["C1"] = "=B1*2"  # saved by openpyxl: no stored result
    p = tmp_path / "f.xlsx"
    wb.save(p)
    xl = read_score_sheet(p)
    assert xl.num.shape == (1, 2) and xl.num[0, 1] == 2


def test_two_digit_year_pivot_and_text_dates():
    assert ssx._expand_two_digit_year(24, datetime.date(2026, 1, 1)) == 2024
    assert ssx._expand_two_digit_year(80, datetime.date(2026, 1, 1)) == 1980
    assert ssx._expand_two_digit_year(75, datetime.date(2026, 1, 1)) == 2075
    assert ssx._date_format("01.03.2024") == "dd.mm.yyyy"
    assert ssx._date_format("01-Mar-24") == "dd-mm-yy"
    with pytest.raises(ValueError):
        ssx._date_format("2024-03-01")
    assert ssx._parse_sheet_date("1.3.2024", "dd.mm.yyyy") == datetime.datetime(2024, 3, 1)
    assert ssx._parse_sheet_date("01-03-24", "dd-mm-yy") == datetime.datetime(2024, 3, 1)
    assert ssx._parse_sheet_date("01-Mar-24", "dd-mm-yy") == datetime.datetime(2024, 3, 1)
    with pytest.raises(ValueError):
        ssx._parse_sheet_date("31.02.2024", "dd.mm.yyyy")
    assert ssx._format_sheet_date(datetime.date(2024, 3, 4), "dd-mm-yy") == "04-03-24"


# --- getweight_ScoreSheet ----------------------------------------------------------------


def test_score_sheet_weights(tmp_path):
    p = make_sheet(tmp_path / "w.xlsx", obs=[("02.03.2024", 1, 19.5), ("03.03.2024", 1, 19.0)])
    assert tuple(score_sheet_weights(p)) == (20.0, 19.0)
    p2 = make_sheet(tmp_path / "w2.xlsx", obs=[("02.03.2024", 1, None)])
    with pytest.raises(ValueError, match="numeric block"):
        score_sheet_weights(p2)  # row 20 holds no number: xlsread trims it (MATLAB index error)
    wb = openpyxl.load_workbook(p2)
    wb.active["C20"] = 20.0  # weight on the start day
    wb.save(p2)
    w = score_sheet_weights(p2)
    assert w.initial_weight == 20.0 and np.isnan(w.last_weight)  # no replacement, as MATLAB
    p3 = make_sheet(tmp_path / "w3.xlsx", obs=daily_obs(2), prefill_to=0)
    with pytest.raises(ValueError, match="day-counter row"):
        score_sheet_weights(p3)  # no empty column after the last observation


# --- checkScoreSheetMouse ----------------------------------------------------------------


def test_check_mouse_daily(tmp_path):
    p = make_sheet(tmp_path / "c.xlsx", obs=daily_obs(3))
    assert not check_score_sheet_mouse(p, datetime.datetime(2024, 3, 5, 11))
    assert check_score_sheet_mouse(p, datetime.datetime(2024, 3, 5, 13))  # > 1.5 days


def test_check_mouse_weekly_after_three_zero_days(tmp_path):
    p = make_sheet(tmp_path / "z.xlsx", obs=daily_obs(4, total=0))
    assert not check_score_sheet_mouse(p, datetime.datetime(2024, 3, 11))
    assert check_score_sheet_mouse(p, datetime.datetime(2024, 3, 12, 1))  # > 7 days


def test_check_mouse_kept_gap_bug(tmp_path):
    # MATLAB compares start - last instead of current - last: a 5-day gap is not flagged
    p = make_sheet(tmp_path / "g.xlsx", obs=[("02.03.2024", 1, 19), ("07.03.2024", 1, 19)])
    assert not check_score_sheet_mouse(p, datetime.datetime(2024, 3, 7, 12))


def test_check_mouse_matches_core_and_month_names(tmp_path):
    obs = [("02-Mar-24", 2, 19), ("03-03-24", 0, 19), ("05-Mar-24", 1, 19)]
    p = make_sheet(tmp_path / "m.xlsx", start="01-Mar-24", obs=obs)
    for now in (datetime.datetime(2024, 3, 6), datetime.datetime(2024, 3, 7)):
        core = score_sheet_problem(
            datetime.datetime(2024, 3, 1),
            [(datetime.datetime(2024, 3, d), s) for d, s in ((2, 2), (3, 0), (5, 1))], now)
        assert check_score_sheet_mouse(p, now) == core
    assert check_score_sheet_mouse(p, datetime.datetime(2024, 3, 7))


def test_check_mouse_errors(tmp_path):
    p = make_sheet(tmp_path / "e1.xlsx", obs=daily_obs(1), total_label="TOTAL")
    with pytest.raises(ValueError, match="TOTAL SCORE"):
        check_score_sheet_mouse(p, datetime.datetime(2024, 3, 2))
    p = make_sheet(tmp_path / "e2.xlsx", obs=daily_obs(1))
    wb = openpyxl.load_workbook(p)
    wb.active["C1"] = "2024-03-01"
    wb.save(p)
    with pytest.raises(ValueError, match="neither"):
        check_score_sheet_mouse(p, datetime.datetime(2024, 3, 2))


# --- updateScoreSheet --------------------------------------------------------------------


def _decorated_sheet(path):
    make_sheet(path, obs=[("02.03.2024", 1, 19.5), ("03.03.2024", 1, 19.0)])
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    ws["A1"].font = Font(bold=True)
    ws["B18"].fill = PatternFill("solid", fgColor="FFFF00")
    ws.merge_cells("A24:C24")
    ws["A24"] = "Notes"
    ws["A24"].comment = Comment("keep me", "tester")
    other = wb.create_sheet("Other")
    other["A1"] = "=SUM(1,2)"
    wb.defined_names["Initial"] = DefinedName(
        "Initial", attr_text="'Score sheet'!$B$18")
    wb.save(path)
    return path


def test_update_round_trip(tmp_path):
    p = _decorated_sheet(tmp_path / "ScoreSheet_XYZ240301.xlsx")
    res = update_score_sheet(p, {**OBS, "Date": datetime.date(2024, 3, 4)})
    assert res.success and not res.failed and res.cancelled is None
    path, cells = res.updated[0]
    assert path == p
    bws = body_weight_score(18.0, 20.0, 19.0)
    expected = {"F1": "04.03.2024", "F2": 3, "F5": 0, "F6": 0.5, "F7": 0, "F8": 0.5, "F10": 0,
                "F12": 0, "F13": 1, "F14": 0, "F15": bws, "F20": 18, "F21": -10,
                "F22": int(mround((18 / 19 - 1) * 100))}
    assert cells == expected
    wb = openpyxl.load_workbook(p)
    ws = wb.worksheets[0]
    for addr, v in expected.items():
        assert ws[addr].value == v, addr
    assert ws["F1"].data_type == "s"  # the date is written as text, as MATLAB passed it
    assert ws["A1"].font.bold and ws["B18"].fill.fgColor.rgb.endswith("FFFF00")
    assert ws["A24"].comment.text == "keep me" and "A24:C24" in ws.merged_cells
    assert ws["E1"].value == "03.03.2024" and ws["G16"].value == 0
    assert wb["Other"]["A1"].value == "=SUM(1,2)"
    assert "Initial" in wb.defined_names
    assert wb.calculation.fullCalcOnLoad
    # the written file is read the way MATLAB reads it: the next observation goes to G
    assert tuple(score_sheet_weights(p)) == (20.0, 18.0)
    res2 = update_score_sheet(p, {**OBS, "BodyWeight": 18.5, "Date": datetime.date(2024, 3, 5)})
    assert res2.updated[0][1]["G1"] == "05.03.2024" and res2.updated[0][1]["G2"] == 4
    assert res2.updated[0][1]["G22"] == int(mround((18.5 / 18 - 1) * 100))
    assert not list(tmp_path.glob(".*calliopy-tmp"))


def test_update_dd_mm_yy_sheet_writes_numeric_month(tmp_path):
    p = make_sheet(tmp_path / "y.xlsx", start="01-Mar-24", obs=[("02-Mar-24", 1, 19.5)])
    res = update_score_sheet(p, {**OBS, "Date": datetime.date(2024, 3, 3)})
    assert res.updated[0][1]["E1"] == "03-03-24"
    assert not check_score_sheet_mouse(p, datetime.datetime(2024, 3, 4))


def test_update_precedes_raises_and_keeps_file(tmp_path):
    p = make_sheet(tmp_path / "p.xlsx", obs=daily_obs(2))
    before = p.read_bytes()
    with pytest.raises(ValueError, match="precedes"):
        update_score_sheet(p, {**OBS, "Date": datetime.date(2024, 3, 2)})
    assert p.read_bytes() == before


def test_update_late_observation(tmp_path):
    p = make_sheet(tmp_path / "l.xlsx", obs=daily_obs(2))  # totals 1: daily checks
    before = p.read_bytes()
    obs = {**OBS, "Date": datetime.date(2024, 3, 6)}
    res = update_score_sheet(p, obs)
    assert res.cancelled == p and not res.success and not res.updated
    assert p.read_bytes() == before
    seen = []
    res = update_score_sheet(p, obs, allow_late=lambda msg: seen.append(msg) or True)
    assert res.success and res.updated[0][1]["F1"] == "06.03.2024"
    assert "too late" in seen[0] and "- 3 days since last observation" in seen[0]
    assert res.late_messages == seen


def test_update_cancel_stops_later_sheets(tmp_path):
    late = make_sheet(tmp_path / "a.xlsx", obs=daily_obs(2))
    ok = make_sheet(tmp_path / "b.xlsx", obs=daily_obs(4))
    before = ok.read_bytes()
    res = update_score_sheet([late, ok], {**OBS, "Date": datetime.date(2024, 3, 6)})
    assert res.cancelled == late and ok.read_bytes() == before


def test_update_kept_late_check_window(tmp_path):
    # Kept MATLAB quirk: the late check uses the totals of the last two observations and of
    # the new, still empty column. With a stored 0 there, two zero observations already allow
    # weekly checks (6 days are not late) ...
    p = make_sheet(tmp_path / "k1.xlsx", obs=[("02.03.2024", 0, 19.5), ("03.03.2024", 0, 19.5)])
    res = update_score_sheet(p, {**OBS, "Date": datetime.date(2024, 3, 9)})
    assert res.success and not res.late_messages
    # ... and with an empty total in the new column the sum is NaN: daily rule, 2 days late
    p = make_sheet(tmp_path / "k2.xlsx", obs=daily_obs(4, total=0), prefill_from=9)
    res = update_score_sheet(p, {**OBS, "Date": datetime.date(2024, 3, 7)})
    assert res.cancelled == p


def test_update_write_failure_is_logged_and_next_sheet_written(tmp_path, monkeypatch):
    a = make_sheet(tmp_path / "a.xlsx", obs=daily_obs(2))
    b = make_sheet(tmp_path / "b.xlsx", obs=daily_obs(2))
    before = a.read_bytes()
    real_replace = os.replace

    def replace(src, dst):
        if Path(dst) == a:
            raise PermissionError("file is open")
        return real_replace(src, dst)

    monkeypatch.setattr(ssx.os, "replace", replace)
    res = update_score_sheet([a, b], {**OBS, "Date": datetime.date(2024, 3, 4)})
    assert res.failed == [a] and [u[0] for u in res.updated] == [b] and res.success
    assert a.read_bytes() == before
    assert not list(tmp_path.glob(".*calliopy-tmp"))


def test_update_nan_weight_raises(tmp_path):
    p = make_sheet(tmp_path / "n.xlsx", obs=daily_obs(1))
    with pytest.raises(ValueError):
        update_score_sheet(p, {**OBS, "BodyWeight": math.nan, "Date": datetime.date(2024, 3, 3)})


def test_layout_warning(tmp_path):
    p = make_sheet(tmp_path / "o.xlsx", obs=daily_obs(1))
    wb = openpyxl.load_workbook(p)
    wb.active["A30"] = 1  # a number in column A moves the numeric block
    wb.save(p)
    handler = _ListHandler()
    ssx.logger.addHandler(handler)
    try:
        score_sheet_weights(p)
    finally:
        ssx.logger.removeHandler(handler)
    assert any("template" in m for m in handler.messages)


# --- getLastMouseObservation -------------------------------------------------------------


def test_last_mouse_observation(tmp_path):
    now = datetime.datetime(2024, 3, 20, 12)
    sheet = score_sheet_path(tmp_path, "ABC240301_1")
    sheet.parent.mkdir()
    make_sheet(sheet)
    t = (now - datetime.timedelta(days=2.5)).timestamp()
    os.utime(sheet, (t, t))
    res = last_mouse_observation(tmp_path, ["ABC240301_1", "ABC240305"], now=now)
    assert res.days_since_last_observation.shape == (1, 2)
    assert res.days_since_last_observation[0, 0] == 2
    assert res.days_in_experiment[0, 0] == 19
    assert np.isinf(res.days_since_last_observation[0, 1]) and np.isinf(res.days_in_experiment[0, 1])
    assert last_mouse_observation(tmp_path, [], now=now).days_in_experiment.shape == (1, 0)
    bad = score_sheet_path(tmp_path, "ABCDEFGHIJ")
    bad.parent.mkdir()
    make_sheet(bad)
    with pytest.raises(ValueError, match="yymmdd"):
        last_mouse_observation(tmp_path, ["ABCDEFGHIJ"], now=now)


# --- checkScoreSheets --------------------------------------------------------------------


def test_check_score_sheets(tmp_path):
    def add(user, mouse, obs):
        p = score_sheet_path(tmp_path / user, mouse)
        p.parent.mkdir(parents=True)
        make_sheet(p, obs=obs)

    add("user1", "M1", daily_obs(5))  # last observation 06.03: fine on 06.03 evening
    add("user1", "M2", daily_obs(2))  # last 03.03: problem
    add("user1", "M3", daily_obs(1))  # dead
    add("user1", "M4", daily_obs(1))  # not in the animal table
    (tmp_path / "user1" / "notes").mkdir()
    add("user2", "M5", daily_obs(1))  # problem
    (tmp_path / "readme.txt").write_text("x")
    dod = {"M1": None, "M2": math.nan, "M3": "2024-03-04", "M5": ""}
    res = check_score_sheets(tmp_path, dod, now=datetime.datetime(2024, 3, 6, 20))
    assert res.problem_mice == ["M2", "M5"]
    assert [(a.user, a.mouse) for a in res.alerts] == [("user1", "M2"), ("user2", "M5")]
    assert res.alerts[0].subject == "URGENT - check your mice"
    assert res.alerts[0].body.endswith("score sheet of mouse: M2")
    assert res.not_in_log == [("user1", "M4")]


def test_has_date_of_death():
    assert not ssx._has_date_of_death(None)
    assert not ssx._has_date_of_death(math.nan)
    assert not ssx._has_date_of_death("")
    assert ssx._has_date_of_death("05.03.2024")
    assert ssx._has_date_of_death(datetime.date(2024, 3, 5))
    assert ssx._has_date_of_death(739000.0)


# --- lazy import -------------------------------------------------------------------------


def test_import_without_openpyxl():
    src = str(Path(calliopy.__file__).resolve().parents[1])
    code = ("import sys; sys.modules['openpyxl'] = None; import calliopy.twophoton as t; "
            "assert callable(t.update_score_sheet)")
    env = dict(os.environ, PYTHONPATH=src)
    r = subprocess.run([sys.executable, "-c", code], check=False, env=env,
                        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
