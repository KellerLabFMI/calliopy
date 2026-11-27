"""Animal score-sheet rules (computational cores of calliope ``04_Calliope`` score-sheet code).

Ports of the logic in ``checkScoreSheetMouse.m`` and ``updateScoreSheet.m``. Reading and
writing the ``.xlsx`` score sheets (``xlsread``/``xlswrite``), e-mailing and the dialogs are
not ported: calliopy has no spreadsheet reader without a new dependency. The functions take
the values already read from the sheet.

Where these functions belong (here, or with the animal/ExpLog tooling) is left to the user.

None of these functions has been compared with MATLAB output.
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .._matlab import mround

__all__ = [
    "body_weight_score",
    "excel_column_name",
    "observation_too_late",
    "score_sheet_problem",
    "score_sheet_row_values",
]


def _days(a: datetime.datetime | datetime.date, b: datetime.datetime | datetime.date) -> float:
    """``datenum(a) - datenum(b)`` in days."""
    if isinstance(a, datetime.date) and not isinstance(a, datetime.datetime):
        a = datetime.datetime(a.year, a.month, a.day)
    if isinstance(b, datetime.date) and not isinstance(b, datetime.datetime):
        b = datetime.datetime(b.year, b.month, b.day)
    return (a - b).total_seconds() / 86400.0


def score_sheet_problem(start_date: datetime.date, observations: Sequence[tuple[Any, float]],
                        relative_to: datetime.datetime) -> bool:
    """Whether a score sheet is out of spec (``checkScoreSheetMouse.m``).

    Rule: observe daily, or weekly once the total score has been 0 on the last three days
    since the start; the last observation may be at most 1.5 days (daily) or 7 days (weekly)
    before ``relative_to``.

    Parameters
    ----------
    start_date
        Start of the experiment (third column of the date row).
    observations
        ``(date, total_score)`` of every filled observation column, in sheet order.
    relative_to
        Reference time (MATLAB default ``now``).

    Notes
    -----
    Kept from MATLAB (suspected bug, changes the result): the gap between observations is
    computed as ``start - last_observation`` instead of ``current - last_observation``; it is
    never positive, so gaps between observations are never flagged, only the time since the
    last observation is checked.
    """
    last = start_date
    check_daily = True
    problem = False
    hist: list[float] = []
    for date, score in observations:
        dss = int(np.round(_days(date, start_date)))
        dso = _days(start_date, last)
        if dss > 0:
            if len(hist) < dss:
                hist.extend([0.0] * (dss - len(hist)))
            hist[dss - 1] = float(score)
        if dss > 2:
            check_daily = sum(hist[dss - 3 : dss]) != 0
        if (check_daily and dso > 1) or (not check_daily and dso > 7):
            problem = True
        last = date
    gap = _days(relative_to, last)
    if (check_daily and gap > 1.5) or (not check_daily and gap > 7):
        problem = True
    return problem


def observation_too_late(n_days_since_last_observation: float, last_3_scores: Sequence[float]) -> bool:
    """``updateScoreSheet.m``'s late-entry warning: weekly checks allowed only after three
    zero scores, else daily."""
    zero = float(np.nansum(np.asarray(last_3_scores, dtype=np.float64))) == 0 and not np.any(
        np.isnan(np.asarray(last_3_scores, dtype=np.float64)))
    return (zero and n_days_since_last_observation > 7) or (not zero and n_days_since_last_observation > 1)


def body_weight_score(body_weight: float, initial_weight: float, last_weight: float) -> int:
    """Body-weight score 0-3 from the loss relative to ``max(initial, last)`` (``updateScoreSheet.m``).

    Loss below 5 % gives 0, 5-10 % 1, 10-20 % 2, 20 % or more 3. A NaN ``last_weight`` is
    replaced by the initial weight.

    Raises
    ------
    ValueError
        NaN weight (MATLAB failed with an undefined score).
    """
    if np.isnan(last_weight):
        last_weight = initial_weight
    ch = (body_weight / max(initial_weight, last_weight) - 1) * 100
    if np.isnan(ch):
        raise ValueError("weight change is NaN")
    if ch > -5:
        return 0
    if ch > -10:
        return 1
    if ch > -20:
        return 2
    return 3


def score_sheet_row_values(observations: Mapping[str, Any], initial_weight: float,
                           last_weight: float) -> dict:
    """Values ``updateScoreSheet.m`` writes into the new observation column, by sheet row.

    ``observations`` holds MATLAB's fields ``Selfmutliation``, ``Lackofgrooming``,
    ``Hunchedposture``, ``Orbitaltightening``, ``Notresponsivetotouch``, ``Swelling``,
    ``Minorbleeding``, ``Infectionofmajorbleeding`` and ``BodyWeight``. Returns
    ``{row (1-based sheet row): value}`` for rows 5-8, 10, 12-15 and 20-22 (the date rows 1-2
    are the caller's).
    """
    if np.isnan(last_weight):
        last_weight = initial_weight
    o = observations
    bw = float(o["BodyWeight"])
    return {
        5: o["Selfmutliation"] * 3,
        6: o["Lackofgrooming"] * 0.5,
        7: (o["Hunchedposture"] - 1) * 0.5,
        8: (o["Orbitaltightening"] - 1) * 0.5,
        10: o["Notresponsivetotouch"] * 3,
        12: o["Swelling"] * 0.5,
        13: o["Minorbleeding"] * 1,
        14: o["Infectionofmajorbleeding"] * 3,
        15: body_weight_score(bw, initial_weight, last_weight),
        20: bw,
        21: float(mround((bw / initial_weight - 1) * 100)),
        22: float(mround((bw / last_weight - 1) * 100)),
    }


def excel_column_name(col: int) -> str:
    """Excel column letters of a 1-based column number (``xlC2A1`` in ``updateScoreSheet.m``)."""
    col = int(col)
    if col < 1 or col > 16384:
        raise ValueError("Column does not exist in Excel")
    out = ""
    while col > 0:
        col, rem = divmod(col - 1, 26)
        out = chr(65 + rem) + out
    return out
