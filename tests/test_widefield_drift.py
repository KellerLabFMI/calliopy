"""Tests for calliopy.widefield._drift (wid_show_stack_drift.m, compare_wid_frames.m).

Synthetic .wid files: stacks of a blob texture, shifted by whole pixels (np.roll) between
and within stacks, so register_frames returns known integer shifts.
"""

from __future__ import annotations

import gc
import math
import struct
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from calliopy.io import MatStruct
from calliopy.widefield._drift import (
    _num,
    compare_wid_frames,
    plot_stack_drift,
    plot_stack_drift_session,
    stack_drift,
    stack_drift_report,
)

H, W, N = 64, 72, 250


def blobs(h, w, seed=1):
    g = np.random.default_rng(seed)
    n = h * w // 20
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.full((h, w), 1000.0)
    for x0, y0, s, a in zip(g.uniform(0, w, n), g.uniform(0, h, n), g.uniform(1.5, 4, n),
                            g.uniform(200, 900, n)):
        img += a * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * s * s))
    return img


def bcd(value: int, nbytes: int) -> list[int]:
    digits = f"{value:0{2 * nbytes}d}"
    return [int(digits[2 * k : 2 * k + 2], 16) for k in range(nbytes)]


def write_wid(path: Path, imgs: np.ndarray) -> None:
    """(size_y, size_x, n) frames; each record uint16 size_y, size_x, pixels y fastest."""
    size_y, size_x, n = imgs.shape
    out = bytearray()
    stamp = datetime(2024, 5, 6, 7, 8, 9)
    for k in range(n):
        flat = imgs[:, :, k].astype("<u2").ravel(order="F").copy()
        flat[:14] = (bcd(k + 1, 4) + bcd(stamp.year, 2) + bcd(stamp.month, 1)
                     + bcd(stamp.day, 1) + bcd(stamp.hour, 1) + bcd(stamp.minute, 1)
                     + bcd(stamp.second, 1) + bcd(0, 3))
        out += struct.pack("<HH", size_y, size_x) + flat.tobytes()
    path.write_bytes(bytes(out))


def make_stack(base, offset, late_shift=None, late_from=None, seed=0):
    """Frames = base rolled by ``offset`` (rows, cols); from 1-based frame ``late_from`` on
    additionally by ``late_shift``; frame 1 is a bright spot (left out by the code)."""
    g = np.random.default_rng(seed)
    data = np.empty((H, W, N))
    for k in range(N):
        shift = np.array(offset)
        if late_from is not None and k + 1 >= late_from:
            shift = shift + np.array(late_shift)
        data[:, :, k] = np.roll(base, tuple(shift), axis=(0, 1)) + g.normal(0, 3, (H, W))
    data[:, :, 0] = 60000.0  # "stimulation laser spot": a useless first frame
    return np.round(data).astype(np.uint16)


@pytest.fixture(scope="module")
def three_stacks(tmp_path_factory):
    d = tmp_path_factory.mktemp("drift")
    base = blobs(H, W)
    paths = [d / "a.wid", d / "b.wid", d / "c.wid"]
    write_wid(paths[0], make_stack(base, (0, 0), late_shift=(6, 0), late_from=140, seed=1))
    write_wid(paths[1], make_stack(base, (3, -2), seed=2))
    write_wid(paths[2], make_stack(base, (7, 0), seed=3))
    info = stack_drift(paths, drift_every=50, exp_id=42, fname_field="wid_fnames_ch525")
    gc.collect()
    return info


def test_stack_drift_between_stacks(three_stacks):
    info = three_stacks
    assert list(info) == ["ExpID", "fname_field", "wid_fnames", "nbr_frames", "stack_dx",
                          "stack_dy", "stack_shifted", "drift", "t_session", "dx_vs_stack1",
                          "dy_vs_stack1"]
    assert info.ExpID == 42 and info.fname_field == "wid_fnames_ch525"
    assert info.wid_fnames.shape == (1, 3) and Path(info.wid_fnames[0, 1]).name == "b.wid"
    np.testing.assert_array_equal(info.nbr_frames, [[N, N, N]])
    np.testing.assert_array_equal(info.stack_dx, [[0.0], [3.0], [7.0]])
    np.testing.assert_array_equal(info.stack_dy, [[0.0], [-2.0], [0.0]])
    np.testing.assert_array_equal(info.stack_shifted, [[False, False, True]])


def test_stack_drift_within_stacks(three_stacks):
    info = three_stacks
    d = info.drift
    assert isinstance(d, MatStruct)
    assert list(d) == ["drift_every", "frame_rate", "max_drift", "frames", "dx", "dy",
                       "max_abs", "max_step", "step_between", "first_over", "ok"]
    assert (d.drift_every, d.frame_rate, d.max_drift) == (50, 100, 5)
    # frames 1, 51, ..., 201 sampled; frame 1 left out
    for k in range(3):
        np.testing.assert_array_equal(d.frames[0, k], [[51.0], [101.0], [151.0], [201.0]])
    np.testing.assert_array_equal(d.dx[0, 0], [[0.0], [0.0], [6.0], [6.0]])
    np.testing.assert_array_equal(d.dy[0, 0], np.zeros((4, 1)))
    for k in (1, 2):
        np.testing.assert_array_equal(d.dx[0, k], np.zeros((4, 1)))
    np.testing.assert_array_equal(d.max_abs, [[6.0, 0.0, 0.0]])
    np.testing.assert_array_equal(d.max_step, [[6.0, 0.0, 0.0]])
    # largest step: first maximum, as MATLAB max (all-zero steps -> first pair)
    np.testing.assert_array_equal(d.step_between, [[101, 151], [51, 101], [51, 101]])
    np.testing.assert_array_equal(d.first_over[0, 0], 151)
    assert np.isnan(d.first_over[0, 1:]).all()
    np.testing.assert_array_equal(d.ok, [[False, True, True]])
    # session values
    frames = np.tile([51, 101, 151, 201], 3)
    start = np.repeat([0, N, 2 * N], 4)
    np.testing.assert_allclose(info.t_session, ((start + frames) / 100).reshape(1, -1))
    expected_dx = np.concatenate([[0, 0, 6, 6], [3] * 4, [7] * 4]).reshape(-1, 1)
    np.testing.assert_array_equal(info.dx_vs_stack1, expected_dx)
    expected_dy = np.concatenate([[0] * 4, [-2] * 4, [0] * 4]).reshape(-1, 1)
    np.testing.assert_array_equal(info.dy_vs_stack1, expected_dy)


def test_stack_drift_report_text(three_stacks):
    text = stack_drift_report(three_stacks)
    lines = text.split("\n")
    assert lines[1] == ("Threshold 5 px. Offsets are frame 100 of each stack vs frame 100 of "
                        "stack 1.")
    assert lines[2] == ("Within-stack = sampled frames vs the stack's own frame 100, one sample "
                        "every 0.5 s.")
    assert lines[4].startswith("stack   offset dx")
    assert lines[5] == ("    1         0.0         0.0          6.0            6.0       "
                        "1.0 - 1.5       a.wid  <-- DRIFTS")
    assert lines[7] == ("    3         7.0         0.0          0.0            0.0       "
                        "0.5 - 1.0       c.wid  <-- SHIFTED vs stack 1")
    assert "Between stacks: stack(s) 3 are offset by more than 5 px from stack 1." in lines
    assert ("Within stack 1: first beyond 5 px at 1.5 s (frame 151); largest step 6.0 px "
            "between 1.0 and 1.5 s (max shift 6.0 px).") in lines
    assert "No stack moves" not in text


def test_number_format_like_fprintf():
    assert _num(math.nan, "7.1f") == "    NaN"
    assert _num(math.nan, "-7.1f") == "NaN    "
    assert _num(math.inf, "9.1f") == "      Inf"
    assert _num(1.25, "7.1f") == "    1.2"  # round-half-even of the binary value, as C
    assert _num(1.5, "-7.1f") == "1.5    "
    assert _num(-1.5, "-7.1f") == "-1.5   "


def test_stack_drift_single_stack_and_quiet(tmp_path):
    p = tmp_path / "s.wid"
    write_wid(p, make_stack(blobs(H, W, seed=3), (0, 0), seed=4))
    info = stack_drift(p, drift_every=200)
    gc.collect()
    assert info.stack_dx == 0.0 and info.stack_dy == 0.0
    assert np.shape(info.stack_dx) == ()
    np.testing.assert_array_equal(info.stack_shifted, [[False]])
    assert info.ExpID.shape == (0, 0)
    # frames 1 and 201 sampled; only 201 kept -> no step, NaN step_between
    np.testing.assert_array_equal(info.drift.frames[0, 0], [[201.0]])
    assert np.isnan(info.drift.step_between).all() and info.drift.max_step[0, 0] == 0
    text = stack_drift_report(info)
    assert "NaN - NaN" in text
    assert "No stack moves by more than 5 px, between or within stacks." in text


def test_stack_drift_short_file_raises(tmp_path):
    p = tmp_path / "short.wid"
    write_wid(p, np.full((8, 10, 20), 100, dtype=np.uint16))
    with pytest.raises(ValueError):
        stack_drift(p)
    gc.collect()


def test_plots(three_stacks):
    fig1 = plot_stack_drift_session(three_stacks)
    assert isinstance(fig1, Figure) and len(fig1.axes) == 2
    assert "ExpID 42" in fig1.axes[0].get_title()
    fig2 = plot_stack_drift(three_stacks)
    assert isinstance(fig2, Figure) and len(fig2.axes) == 2
    texts = [t.get_text() for t in fig2.axes[0].get_legend().get_texts()]
    assert texts == ["stack 1 (max 6.0 px) DRIFTS", "stack 2 (max 0.0 px)",
                     "stack 3 (max 0.0 px)"]
    assert "ExpID 42 wid_fnames_ch525" in fig2.axes[0].get_title()
    fig3 = plot_stack_drift(three_stacks.drift, "label")
    assert fig3.axes[0].get_title().startswith("label: within-stack drift")


def test_compare_wid_frames_clipping_and_diff():
    g = np.random.default_rng(0)
    a = g.integers(100, 1000, size=(30, 40)).astype(np.uint16)
    b = a.copy()
    a[3, 4] = 60000
    b[5, 6] = 50
    fig = compare_wid_frames(a, b, "A", "B", "C")
    assert isinstance(fig, Figure) and len(fig.axes) == 3
    lim = 2 * np.percentile(a.astype(float), 90, method="hazen")
    shown_a = np.asarray(fig.axes[0].images[0].get_array())
    assert shown_a[3, 4] == math.floor(lim + 0.5)  # stored in uint16, rounded
    np.testing.assert_array_equal(np.delete(shown_a.ravel(), 3 * 40 + 4),
                                  np.delete(a.ravel(), 3 * 40 + 4))
    diff = np.asarray(fig.axes[2].images[0].get_array())
    np.testing.assert_array_equal(diff, shown_a.astype(float) * 0 + b.astype(float)
                                  - shown_a.astype(float))
    assert fig.axes[2].images[0].norm.vcenter == 0
    assert [ax.get_title() for ax in fig.axes] == ["A", "B", "C"]
    # the given figure is cleared and reused
    again = compare_wid_frames(a, a, fig=fig)
    assert again is fig and len(fig.axes) == 3
    assert [ax.get_title() for ax in fig.axes] == ["Frame A", "Frame B", "Diff"]
    # float input keeps its class, one-signed difference still centred on zero
    f = compare_wid_frames(a.astype(np.float32), a.astype(np.float32) + 1)
    assert np.asarray(f.axes[0].images[0].get_array()).dtype == np.float32
