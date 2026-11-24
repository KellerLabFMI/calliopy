"""Synthetic tests of calliopy.widefield._activity.

roi_activity, corner_rois, calc_widefield_act (all saving modes), frame_times_widefield and
median_divide_video. Files are written under tmp_path in the layout the calliopy.io readers
parse.
"""

from __future__ import annotations

import gc
import logging
import struct
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from calliopy.io import MatStruct
from calliopy.widefield._activity import (
    CORNER_ROI_NAMES,
    calc_widefield_act,
    corner_rois,
    frame_times_widefield,
    median_divide_video,
    roi_activity,
)

RNG = np.random.default_rng(20261124)


# --- file writers (layout as parsed by calliopy.io.load_wid_data / load_roi_data) ----------


def _bcd(value: int, nbytes: int) -> list[int]:
    digits = f"{value:0{2 * nbytes}d}"
    return [int(digits[2 * k: 2 * k + 2], 16) for k in range(nbytes)]


def _pco_header(framenumber: int, stamp: datetime) -> list[int]:
    return (_bcd(framenumber, 4) + _bcd(stamp.year, 2) + _bcd(stamp.month, 1)
            + _bcd(stamp.day, 1) + _bcd(stamp.hour, 1) + _bcd(stamp.minute, 1)
            + _bcd(stamp.second, 1) + _bcd(stamp.microsecond, 3))


def _stamp(k: int) -> datetime:
    return datetime(2025, 5, 6, 10, 0, 0) + timedelta(milliseconds=50 * k)


def write_wid(path: Path, imgs: np.ndarray) -> None:
    """imgs (size_y, size_x, n); each record: uint16 size_y, size_x, pixels column-major."""
    size_y, size_x, n = imgs.shape
    out = bytearray()
    for k in range(n):
        out += struct.pack("<HH", size_y, size_x)
        out += imgs[:, :, k].astype("<u2").ravel(order="F").tobytes()
    path.write_bytes(bytes(out))


def write_roi_pair(path: Path, traces: np.ndarray) -> None:
    """traces (n_frames, n_traces) -> .roi file plus its .wii file (param.numberROIs)."""
    n_frames, n_traces = traces.shape
    text = f"version=3;param.numberROIs={n_traces};"
    one = np.zeros((1, 1), dtype="<u2").tobytes()
    path.with_suffix(".wii").write_bytes(
        struct.pack("<I", len(text)) + text.encode("latin-1")
        + struct.pack("<HH", 1, 1) + one + struct.pack("<HH", 1, 1) + one)
    out = bytearray(struct.pack("<HH", 2, 2)) + np.zeros(4, dtype="<u2").tobytes()
    for k in range(n_frames):
        out += np.array(_pco_header(k + 1, _stamp(k)), dtype="<u2").tobytes()
        out += traces[k].astype("<u2").tobytes()
    path.write_bytes(bytes(out))


def release() -> None:
    gc.collect()  # drop memory maps so Windows releases tmp files


def _rois(index_lists, field="indices", extra=True):
    out = []
    for k, ix in enumerate(index_lists):
        s = MatStruct()
        if isinstance(field, str):
            s[field] = ix
        else:
            for f in field:
                s[f] = ix
        if extra:
            s["name"] = f"roi{k + 1}"
        out.append(s)
    arr = np.empty((1, len(out)), dtype=object)
    for j, s in enumerate(out):
        arr[0, j] = s
    cell = np.empty((1, 1), dtype=object)
    cell[0, 0] = arr
    return cell


def _independent_mean(data, ix):
    """numpy route: MATLAB reshape(data, [], T)(ix, :) -> single -> mean."""
    flat = np.asarray(data).reshape(-1, data.shape[2], order="F")
    sel = flat[np.asarray(ix, dtype=np.int64).ravel(order="F") - 1].astype(np.float32)
    return sel.astype(np.float64).mean(axis=0)


# --- roi_activity ---------------------------------------------------------------------------


def test_roi_activity_hand_computed_and_column_major():
    data = np.zeros((3, 4, 2), dtype=np.uint16)
    data[:, :, 0] = np.arange(12).reshape(3, 4)          # value = 4*row + col (0-based)
    data[:, :, 1] = 100 + np.arange(12).reshape(3, 4)
    # MATLAB linear index 2 is row 2, column 1 (column-major) -> value 4
    # index 4 is row 1, column 2 -> value 1; index 12 is row 3, column 4 -> value 11
    act = roi_activity(data, [[2], [4], [12], [2, 4, 12], np.array([[2, 4], [12, 12]])])
    assert act.dtype == np.float32 and act.shape == (5, 2)
    np.testing.assert_array_equal(act[:3, 0], [4, 1, 11])
    np.testing.assert_allclose(act[3], [16 / 3, 100 + 16 / 3], rtol=1e-7)
    # duplicated index counts twice, as in MATLAB's mean of reshaped_data(idx, :)
    np.testing.assert_allclose(act[4], [(4 + 11 + 1 + 11) / 4, 100 + 27 / 4], rtol=1e-7)


def test_roi_activity_empty_and_index_one_are_nan():
    data = RNG.integers(1, 1000, size=(5, 6, 4), dtype=np.uint16)
    act = roi_activity(data, [np.empty((0, 0)), np.array(1.0), [1, 1, 1], [1, 2], [7]])
    assert np.isnan(act[0]).all()
    assert np.isnan(act[1]).all()  # genuine pixel-1 ROI is NaN, as in MATLAB
    assert np.isnan(act[2]).all()
    np.testing.assert_allclose(act[3], _independent_mean(data, [1, 2]), rtol=1e-7)
    np.testing.assert_allclose(act[4], _independent_mean(data, [7]), rtol=0)


def test_roi_activity_matches_independent_route_and_chunking(tmp_path):
    data = RNG.integers(0, 65535, size=(17, 13, 23), dtype=np.uint16)
    rois = [RNG.choice(17 * 13, size=n, replace=False) + 1 for n in (1, 5, 40, 221)]
    rois.append(np.arange(1, 17 * 13 + 1, dtype=np.float64).reshape(-1, 1))
    ref = np.stack([_independent_mean(data, ix) for ix in rois]).astype(np.float32)
    full = roi_activity(data, rois)
    np.testing.assert_allclose(full, ref, rtol=1e-6)
    for chunk in (1, 4, 23, 100):
        np.testing.assert_array_equal(roi_activity(data, rois, chunk_frames=chunk), full)
    # memory map in Fortran order, as a stack on disk
    mm = np.memmap(tmp_path / "stack.dat", dtype=np.uint16, mode="w+", shape=data.shape,
                   order="F")
    mm[:] = data
    mm.flush()
    np.testing.assert_array_equal(roi_activity(mm, rois, chunk_frames=5), full)
    del mm
    release()


def test_roi_activity_exact_float32_rounding():
    # 3 pixels with sum 100000001: mean 33333333.67 is rounded once to float32
    data = np.zeros((3, 1, 1), dtype=np.float64)
    data[:, 0, 0] = [33333333, 33333334, 33333334]
    got = roi_activity(data, [[1, 2, 3]])[0, 0]
    assert got == np.float32(100000001 / 3)


def test_roi_activity_single_frame_and_errors():
    img = np.arange(6, dtype=np.uint16).reshape(2, 3)
    np.testing.assert_array_equal(roi_activity(img, [[2, 3]]), [[(3 + 1) / 2]])
    with pytest.raises(IndexError):
        roi_activity(img, [[7]])
    with pytest.raises(IndexError):
        roi_activity(img, [[0]])
    with pytest.raises(ValueError):
        roi_activity(img, [[1.5]])
    with pytest.raises(ValueError):
        roi_activity(np.zeros(5), [[1]])
    assert roi_activity(img, []).shape == (0, 1)


# --- corner_rois ----------------------------------------------------------------------------


def _hand_corners(h, w, ell):
    """MATLAB: [x, y] = meshgrid(rows, cols); sub2ind([h w], x(:), y(:)) -> row outer."""
    def block(rows, cols):
        return [r + (c - 1) * h for r in rows for c in cols]
    lo = range(1, ell + 1)
    return [block(lo, lo), block(lo, range(w - ell + 1, w + 1)),
            block(range(h - ell + 1, h + 1), lo),
            block(range(h - ell + 1, h + 1), range(w - ell + 1, w + 1))]


def test_corner_rois_by_hand():
    h, w = 7, 6
    rois = [[1]] * 4 + [[3, 4, 5, 6]] + [[1]] * 3   # ROI 5 has 4 pixels -> side 2
    cr = corner_rois(rois, (h, w))
    assert cr.roi_len == 2 and cr.names == CORNER_ROI_NAMES
    # written out: upper left rows 1-2 cols 1-2, row outer: (1,1)=1 (1,2)=8 (2,1)=2 (2,2)=9
    assert cr.indices[0].ravel().tolist() == [1, 8, 2, 9]
    # upper right rows 1-2, cols 5-6: (1,5)=29 (1,6)=36 (2,5)=30 (2,6)=37
    assert cr.indices[1].ravel().tolist() == [29, 36, 30, 37]
    # bottom left rows 6-7, cols 1-2: (6,1)=6 (6,2)=13 (7,1)=7 (7,2)=14
    assert cr.indices[2].ravel().tolist() == [6, 13, 7, 14]
    assert cr.indices[3].ravel().tolist() == [34, 41, 35, 42]
    for got, want in zip(cr.indices, _hand_corners(h, w, 2)):
        assert got.shape == (4, 1) and got.dtype == np.float64
        assert got.ravel().tolist() == want


def test_corner_rois_reference_roi_and_rounding():
    h, w = 40, 50
    rois = [[1]] * 400
    rois[386] = list(range(1, 10))      # ROI 387: 9 pixels -> 3
    rois[4] = list(range(1, 50))        # ROI 5 ignored when >= 387 ROIs
    assert corner_rois(rois, (h, w)).roi_len == 3
    assert corner_rois(rois[:386], (h, w)).roi_len == 7  # ROI 5: sqrt(49)
    assert corner_rois([[1]] * 4 + [list(range(6))], (h, w)).roi_len == 2   # 2.449 -> 2
    assert corner_rois([[1]] * 4 + [list(range(7))], (h, w)).roi_len == 3   # 2.646 -> 3
    # 20 pixels -> 4.47 -> 4
    assert corner_rois([[1]] * 4 + [list(range(20))], (h, w)).roi_len == 4
    for got, want in zip(corner_rois(rois, (h, w)).indices, _hand_corners(h, w, 3)):
        assert got.ravel().tolist() == want
    assert corner_rois(rois, (h, w), two_channel=True).roi_len == 3


def test_corner_rois_errors_and_empty_reference():
    with pytest.raises(ValueError, match="5th"):
        corner_rois([[1]] * 4, (10, 10))
    with pytest.raises(ValueError, match="387"):
        corner_rois([[1]] * 100, (10, 10), two_channel=True)
    with pytest.raises(ValueError, match="exceeds"):
        corner_rois([[1]] * 4 + [list(range(100))], (8, 30))
    cr = corner_rois([[1]] * 4 + [[]], (8, 8))
    assert cr.roi_len == 0 and all(ix.shape == (0, 1) for ix in cr.indices)


# --- calc_widefield_act ---------------------------------------------------------------------


def _frame_rois(h, w):
    rng = np.random.default_rng(5)
    rois = [np.array([[3.0]]), np.empty((0, 0))]
    rois += [(rng.choice(h * w, size=9, replace=False) + 1.0).reshape(-1, 1) for _ in range(5)]
    rois.append(np.array([[1.0]]))
    return rois


def test_calc_widefield_act_one_channel(tmp_path):
    h, w = 12, 10
    stacks = [RNG.integers(0, 4000, size=(h, w, n), dtype=np.uint16) for n in (6, 4)]
    paths = []
    for k, s in enumerate(stacks):
        paths.append(tmp_path / f"s{k}.wid")
        write_wid(paths[-1], s)
    idx = _frame_rois(h, w)
    rois_in = _rois(idx)
    regp = MatStruct(status=np.float64(1), similarity=np.float64(0.9))
    res = calc_widefield_act(rois_in, (h, w), wid_fnames=paths, registration_parameters=regp,
                             chunk_frames=3)
    assert set(res) == {"ROIs", "registration_parameters"}
    assert res.registration_parameters.activity_calculated == 1
    assert res.registration_parameters.status == 1 and "activity_calculated" not in regp
    cell = res.ROIs
    assert cell.shape == (1, 1)
    arr = cell[0, 0]
    assert arr.shape == (1, len(idx) + 4)
    assert "activity" not in rois_in[0, 0][0, 0]  # input untouched
    keys = list(arr[0, 0].keys())
    assert keys == ["indices", "name", "activity"]
    assert all(list(s.keys()) == keys for s in arr.flat)
    corners = corner_rois(idx, (h, w))
    full = np.concatenate(stacks, axis=2)
    all_idx = idx + corners.indices
    for j, s in enumerate(arr.flat):
        assert s.activity.shape == (10, 1)
        if j == 1 or j == len(idx) - 1:  # empty ROI, pixel-1 ROI
            assert s.activity.dtype == np.float64 and np.isnan(s.activity).all()
        else:
            assert s.activity.dtype == np.float32
            np.testing.assert_allclose(s.activity[:, 0], _independent_mean(full, all_idx[j]),
                                       rtol=1e-6)
    assert [s.name for s in arr.flat][-4:] == list(CORNER_ROI_NAMES)
    np.testing.assert_array_equal(arr[0, -1].indices, corners.indices[3])
    # struct array given directly, single path, registration_parameters absent
    res2 = calc_widefield_act(rois_in[0, 0], (h, w), wid_fnames=paths[0])
    assert list(res2.registration_parameters.keys()) == ["activity_calculated"]
    assert res2.ROIs[0, 0][0, 2].activity.shape == (6, 1)
    del res, res2
    release()


def test_calc_widefield_act_two_channels(tmp_path):
    h, w = 9, 8
    rois = [[1.0]] * 386 + [list(np.arange(1.0, 5.0))] + [[float(k)] for k in range(2, 12)]
    rois[10] = []
    s525 = RNG.integers(0, 4000, size=(h, w, 5), dtype=np.uint16)
    s610 = RNG.integers(0, 4000, size=(h, w, 5), dtype=np.uint16)
    p525, p610 = tmp_path / "a.wid", tmp_path / "b.wid"
    write_wid(p525, s525)
    write_wid(p610, s610)
    cell = _rois(rois, field=("indices_ch525", "indices_ch610"), extra=False)
    res = calc_widefield_act(cell, (h, w), wid_fnames_ch525=[p525], wid_fnames_ch610=[p610],
                             frame_size_ch610=(h, w))
    arr = res.ROIs[0, 0]
    assert arr.shape == (1, len(rois) + 4)
    assert list(res.registration_parameters.keys()) == ["activity_calculated"]
    last = arr[0, -1]
    assert last.name == "bottom right corner"
    np.testing.assert_array_equal(last.indices_ch525, last.indices_ch610)
    np.testing.assert_allclose(last.activity_ch525[:, 0],
                               _independent_mean(s525, last.indices_ch525), rtol=1e-6)
    np.testing.assert_allclose(last.activity_ch610[:, 0],
                               _independent_mean(s610, last.indices_ch610), rtol=1e-6)
    assert np.isnan(arr[0, 10].activity_ch525).all() and np.isnan(arr[0, 0].activity_ch610).all()
    assert arr[0, 0].name.shape == (0, 0)  # field added to all elements as []
    with pytest.raises(ValueError, match="not identical"):
        calc_widefield_act(cell, (h, w), wid_fnames_ch525=[p525], wid_fnames_ch610=[p610],
                           frame_size_ch610=(h, w + 1))
    del res
    release()


def _trace_setup(tmp_path, n_rois=9):
    h, w = 10, 10
    idx = [np.array([[5.0]]), np.array([[50.0]])]          # reference points
    idx += [np.array([[float(10 + k)], [float(20 + k)]]) for k in range(n_rois - 2)]
    idx[3] = np.empty((0, 0))                               # empty ROI 4
    idx[4] = np.array([[33.0], [34.0], [43.0], [44.0]])     # ROI 5: corner side 2
    # (side 1 would make the upper-left corner [1], which MATLAB turns into NaN)
    n_all = n_rois + 4
    traces = []
    paths = []
    for k, n in enumerate((3, 2)):
        tr = RNG.integers(0, 1000, size=(n, n_all - 6 + 2), dtype=np.uint16)  # 2 extra traces
        traces.append(tr)
        paths.append(tmp_path / f"t{k}.roi")
        write_roi_pair(paths[-1], tr)
    idx[0] = np.array([[1.0]])  # cannot be a genuine ROI: MATLAB NaN rule
    return h, w, idx, traces, paths


def test_calc_widefield_act_roi_only(tmp_path):
    h, w, idx, traces, paths = _trace_setup(tmp_path)
    res = calc_widefield_act(_rois(idx), (h, w), roi_fnames=paths,
                             wid_fnames=[tmp_path / "missing.wid"])
    assert set(res) == {"ROIs", "meta_info", "registration_parameters"}
    arr = res.ROIs[0, 0]
    n = arr.shape[1]
    assert n == len(idx) + 4
    alltr = np.concatenate(traces, axis=0).astype(np.float32)
    for j in range(n):  # 0-based j = MATLAB i - 1
        act = arr[0, j].activity
        if j in (0, 3):                       # indices 1 / empty -> NaN (double)
            assert act.shape == (5, 1) and act.dtype == np.float64 and np.isnan(act).all()
        elif 2 <= j <= n - 5:                 # MATLAB i = 3..n-4 gets trace i-2
            assert act.dtype == np.float32
            np.testing.assert_array_equal(act[:, 0], alltr[:, j - 2])
        else:                                 # ROI 2 and the corners keep []
            assert act.shape == (0, 0)
    meta = res.meta_info
    assert meta.shape == (1, 5)
    assert "dimension" not in meta[0, 0] and meta[0, 3].file_id == 2
    assert meta[0, 1].timestamp == _stamp(1)
    del res
    release()


def test_calc_widefield_act_wid_and_roi(tmp_path):
    h, w, idx, _traces, paths = _trace_setup(tmp_path)
    stack = RNG.integers(0, 4000, size=(h, w, 7), dtype=np.uint16)
    wpath = tmp_path / "full.wid"
    write_wid(wpath, stack)
    res = calc_widefield_act(_rois(idx), (h, w), roi_fnames=paths, wid_fnames=[wpath])
    assert set(res) == {"ROIs", "ROIs_wid", "meta_info", "registration_parameters"}
    wid = res.ROIs_wid[0, 0]
    roi = res.ROIs[0, 0]
    assert wid.shape == roi.shape
    np.testing.assert_allclose(wid[0, 5].activity[:, 0], _independent_mean(stack, idx[5]),
                               rtol=1e-6)
    assert wid[0, -1].activity.shape == (7, 1) and roi[0, -1].activity.shape == (0, 0)
    assert roi[0, 5].activity.shape == (5, 1) and np.isnan(roi[0, 3].activity).all()
    del res
    release()


def test_calc_widefield_act_errors(tmp_path):
    with pytest.raises(ValueError, match="several"):
        cell = np.empty((1, 2), dtype=object)
        cell[0, 0] = _rois([[1]] * 5)[0, 0]
        cell[0, 1] = _rois([[1]] * 5)[0, 0]
        calc_widefield_act(cell, (4, 4), wid_fnames=[])
    with pytest.raises(ValueError, match="neither"):
        calc_widefield_act(_rois([[1]] * 5, field="foo"), (4, 4), wid_fnames=[])
    with pytest.raises(ValueError, match="required"):
        calc_widefield_act(_rois([[1]] * 5), (4, 4))
    h, w, idx, _traces, paths = _trace_setup(tmp_path, n_rois=12)  # too few traces now
    with pytest.raises(ValueError, match="traces"):
        calc_widefield_act(_rois(idx + [[3.0]] * 3), (h, w), roi_fnames=paths)
    release()


# --- frame_times_widefield ------------------------------------------------------------------


def _pulses(n_samples, starts, width):
    x = np.zeros(n_samples)
    for s in starts:
        x[s: s + width] = 5.0
    return x


def test_frame_times_hand_computed():
    # rising edges at 0-based samples 10, 20, ..., 90: MATLAB find(diff(bin)==1) = 10, 20, ...
    trig = _pulses(120, range(10, 100, 10), 3)
    shutter = _pulses(120, [25], 50)          # open: rise at 0-based 25 -> 1-based 25
    # close: last fall at 0-based 75 -> 1-based 75 -> keep 25 <= t <= 75
    meta = [MatStruct(timestamp=datetime(2025, 1, 1, 12, 0, 0) + timedelta(seconds=0.01 * k))
            for k in range(5)]
    res = frame_times_widefield(trig, shutter, meta, 1000.0)
    np.testing.assert_array_equal(res.frame_times, [30.0, 40.0, 50.0, 60.0, 70.0])
    assert res.frame_times.dtype == np.float64 and res.frame_times.ndim == 1
    np.testing.assert_array_equal(res.frame_times_timestamp, [[30, 40, 50, 60, 70]])
    # orientation follows trig_ch
    col = frame_times_widefield(trig.reshape(-1, 1), shutter, meta, 1000.0)
    assert col.frame_times.shape == (5, 1)
    row = frame_times_widefield(trig.reshape(1, -1), shutter, meta, 1000.0)
    assert row.frame_times.shape == (1, 5)
    # frame exactly at the opening sample is kept: shutter rises at 0-based 30
    res2 = frame_times_widefield(trig, _pulses(120, [30], 45), meta, 1000.0)
    assert res2.frame_times[0] == 30.0 and res2.frame_times[-1] == 70.0


def test_frame_times_rounding_midnight_and_plain_datetimes():
    trig = _pulses(200, range(10, 190, 20), 5)
    shutter = _pulses(200, [1], 198)
    t0 = datetime(2025, 3, 4, 23, 59, 59, 990000)
    stamps = [t0 + timedelta(microseconds=k * 20400) for k in range(9)]  # crosses midnight
    res = frame_times_widefield(trig, shutter, stamps, 1000.0)
    # dt in samples: round(k * 20.4); offset = first frame 10
    want = [10 + int(np.floor(k * 20.4 + 0.5)) for k in range(9)]
    np.testing.assert_array_equal(res.frame_times_timestamp[0], want)


def test_frame_times_month_end_kept_as_matlab():
    trig = _pulses(100, range(10, 90, 20), 5)
    shutter = _pulses(100, [1], 98)
    stamps = [datetime(2025, 1, 31, 23, 59, 59, 995000), datetime(2025, 2, 1, 0, 0, 0, 15000)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = frame_times_widefield(trig, shutter, stamps, 1000.0)
    # MATLAB: 86400 * (day 1 - day 31) + ... -> large negative offset (month end not handled)
    assert res.frame_times_timestamp[0, 1] == 10 + round((-30 * 86400 + 0.015 - 86399.995)
                                                         * 1000)


def test_frame_times_messages():
    stamps = [datetime(2025, 1, 1) + timedelta(seconds=0.02 * k) for k in range(8)]
    shutter = _pulses(300, [1], 297)
    starts = [10, 30, 50, 90, 110, 130, 150, 170]   # one dropped frame (gap 40)
    logger = logging.getLogger("calliopy.widefield._activity")
    records = []

    class H(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    hdl = H()
    logger.addHandler(hdl)
    try:
        frame_times_widefield(_pulses(300, starts, 5), shutter, stamps, 1000.0)
    finally:
        logger.removeHandler(hdl)
    assert any("1 frames were dropped" in m and "T=20 ms" in m for m in records)
    with pytest.warns(UserWarning, match="overcounting"):
        frame_times_widefield(_pulses(300, [10, 30, 35, 50, 70, 90, 110, 130], 2), shutter,
                              stamps, 1000.0)
    with pytest.warns(UserWarning, match="do not coincide"):
        frame_times_widefield(_pulses(300, starts[:3], 5), shutter, stamps, 1000.0)
    with pytest.raises(ValueError, match="shutter"):
        frame_times_widefield(_pulses(300, starts, 5), np.zeros(300), stamps, 1000.0)
    with pytest.raises(ValueError, match="inside"):
        frame_times_widefield(_pulses(300, starts, 5), _pulses(300, [200], 50), stamps, 1000.0)


# --- median_divide_video --------------------------------------------------------------------


def test_median_divide_video():
    data = np.array([1, 2, 3, 10], dtype=np.uint16).reshape(1, 1, 4) * np.ones((2, 3, 1),
                                                                                dtype=np.uint16)
    out = median_divide_video(data)
    assert out.dtype == np.float32 and out.shape == (2, 3, 4)
    np.testing.assert_allclose(out[0, 0], np.array([1, 2, 3, 10]) / 2.5, rtol=1e-7)
    d64 = RNG.normal(10, 1, size=(4, 5, 7))
    out64 = median_divide_video(d64)
    assert out64.dtype == np.float64
    np.testing.assert_allclose(out64, d64 / np.median(d64, axis=2)[:, :, None], rtol=1e-15)
    d32 = d64.astype(np.float32)
    assert median_divide_video(d32).dtype == np.float32
    z = np.zeros((1, 1, 3))
    z[0, 0, 2] = 1
    with np.errstate(all="ignore"):
        zz = median_divide_video(z)
    assert np.isnan(zz[0, 0, 0]) and np.isinf(zz[0, 0, 2])
    with pytest.raises(ValueError, match="video"):
        median_divide_video(np.ones((3, 3)))
