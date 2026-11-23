"""Tests for calliopy.io._imaging (two-photon, widefield, ROI, eye, video, tif readers).

Part 1 builds small files in exactly the layout the MATLAB readers parse (byte order,
header, dtypes and dimension order derived from the MATLAB code) and checks shapes, dtypes,
values, orientation and error cases. Part 2 compares every reader with the output of the
original MATLAB reader (tests/matlab/ref_imaging.m); those tests skip until
CALLIOPY_TESTDATA holds the samples and references.
"""

from __future__ import annotations

import gc
import logging
import os
import struct
import sys
from datetime import datetime
from pathlib import Path, PureWindowsPath

import numpy as np
import pytest
from calliopy.io._imaging import (
    AuxVidData,
    BinData,
    EyeData,
    RoiData,
    VidData,
    WidData,
    WiiData,
    load_aux_vid_data,
    load_bin,
    load_eye_monitor_data,
    load_roi_data,
    load_vid_data,
    load_wid_data,
    load_wii_data,
    read_tif,
)
from calliopy.io._mat import MatStruct

RNG = np.random.default_rng(20260101)


class capture_logs:
    """Collect the messages logged by calliopy.io._imaging (no caplog in the test shim)."""

    def __init__(self, name: str = "calliopy.io._imaging") -> None:
        self.name = name
        self.messages: list[str] = []

    def __enter__(self) -> list[str]:
        outer = self

        class Handler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                outer.messages.append(record.getMessage())

        self.handler = Handler(level=logging.DEBUG)
        self.logger = logging.getLogger(self.name)
        self.old_level = self.logger.level
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.handler)
        return self.messages

    def __exit__(self, *exc) -> None:
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self.old_level)


def release() -> None:
    """Drop memory maps so Windows lets the temporary folder go."""
    gc.collect()


# ======================================================================================
# Writers: one per file format, derived from the MATLAB readers
# ======================================================================================


def bcd(value: int, nbytes: int) -> list[int]:
    """Decimal digits of ``value`` as BCD bytes (0x23 means 23), MSB first."""
    digits = f"{value:0{2 * nbytes}d}"
    return [int(digits[2 * k : 2 * k + 2], 16) for k in range(nbytes)]


def pco_header(framenumber: int, stamp: datetime) -> list[int]:
    """The 14 header pixels of a PCO frame (see parseheader in load_wid_data.m)."""
    return (
        bcd(framenumber, 4)
        + bcd(stamp.year, 2)
        + bcd(stamp.month, 1)
        + bcd(stamp.day, 1)
        + bcd(stamp.hour, 1)
        + bcd(stamp.minute, 1)
        + bcd(stamp.second, 1)
        + bcd(stamp.microsecond, 3)
    )


def stamp_of(k: int) -> datetime:
    return datetime(2024, 3 + k % 5, 4 + k, 7 + k % 3, 10 + k, 20 + k, 123456 + 1111 * k)


def write_bin(path: Path, frames: np.ndarray) -> None:
    """``frames`` is (n, x_res, y_res): header int16 x_res, y_res, then frames, y_res fastest."""
    _, x_res, y_res = frames.shape
    path.write_bytes(struct.pack("<hh", x_res, y_res) + frames.astype("<i2").tobytes())


def write_wid(path: Path, imgs: np.ndarray, headers: list[list[int]] | None = None) -> None:
    """``imgs`` is (size_y, size_x, n) as MATLAB returns it.

    Each record: uint16 size_y, uint16 size_x, then the pixels column-major (y fastest).
    ``headers`` replaces the first 14 pixels (in file order) of every frame.
    """
    size_y, size_x, n = imgs.shape
    out = bytearray()
    for k in range(n):
        flat = imgs[:, :, k].astype("<u2").ravel(order="F").copy()
        if headers is not None:
            flat[:14] = headers[k]
        out += struct.pack("<HH", size_y, size_x) + flat.tobytes()
    path.write_bytes(bytes(out))


def wid_frames(n: int, size_y: int, size_x: int, first: int = 1) -> tuple[np.ndarray, list]:
    """Random frames plus BCD headers; returns (imgs as MATLAB sees them, headers)."""
    imgs = RNG.integers(0, 65535, size=(size_y, size_x, n), dtype=np.uint16)
    headers = [pco_header(first + k, stamp_of(k)) for k in range(n)]
    for k in range(n):  # what MATLAB gets back: the header pixels replace the first 14
        flat = imgs[:, :, k].ravel(order="F").copy()
        flat[:14] = headers[k]
        imgs[:, :, k] = flat.reshape((size_y, size_x), order="F")
    return imgs, headers


def plain_frames(n: int, size_y: int, size_x: int) -> np.ndarray:
    """Random (size_y, size_x, n) frames without a camera header (any frame size)."""
    return RNG.integers(0, 65535, size=(size_y, size_x, n), dtype=np.uint16)


def write_wii(path: Path, text: str, mean_proj: np.ndarray, frame: np.ndarray) -> None:
    """mean_proj is (m2, m1), frame is (f1, f2): MATLAB reads both column-major."""
    m2, m1 = mean_proj.shape
    f1, f2 = frame.shape
    path.write_bytes(
        struct.pack("<I", len(text))
        + text.encode("latin-1")
        + struct.pack("<HH", m1, m2)
        + mean_proj.astype("<u2").ravel(order="F").tobytes()
        + struct.pack("<HH", f1, f2)
        + frame.astype("<u2").ravel(order="F").tobytes()
    )


def write_roi(path: Path, traces: np.ndarray, size_x: int = 4, size_y: int = 2, first: int = 1):
    """``traces`` is (n_frames, nbr_ROIs): [size_x size_y] + table + (14 header + traces)."""
    out = bytearray(struct.pack("<HH", size_x, size_y))
    out += RNG.integers(0, 500, size=size_x * size_y, dtype=np.uint16).astype("<u2").tobytes()
    for k in range(traces.shape[0]):
        out += np.array(pco_header(first + k, stamp_of(k)), dtype="<u2").tobytes()
        out += traces[k].astype("<u2").tobytes()
    path.write_bytes(bytes(out))


def write_eye(path: Path, frames, byte_order: str = "<") -> None:
    """``frames``: list of (meta of 9 doubles, image (size_x, size_y)); x runs fastest."""
    out = bytearray()
    for meta, img in frames:
        out += np.asarray(meta, dtype=f"{byte_order}f8").tobytes()
        out += np.asarray(img, dtype=np.uint8).ravel(order="F").tobytes()
    path.write_bytes(bytes(out))


def eye_meta(size_x, size_y, k=0, m3=0.0, m4=0.0):
    return [100.0 + k, 7.0, m3, m4, size_x, size_y, 0.5, 1.5, 2.5 + k]


def write_vid(path: Path, metas, imgs: np.ndarray, pixel: str = "<i2", byte_order: str = "<"):
    """``imgs`` is (size_x, size_y, n); the file holds y fastest (reshape(v,y,x)')."""
    out = bytearray()
    for k in range(imgs.shape[2]):
        out += np.asarray(metas[k], dtype=f"{byte_order}f8").tobytes()
        out += imgs[:, :, k].astype(pixel).ravel(order="C").tobytes()
    path.write_bytes(bytes(out))


def vid_meta(size_y, size_x, k=0):
    return [1000.0 + k, 5.0, size_y, size_x]


def write_aux(path: Path, stamps, imgs: np.ndarray, pixel: str = "<u2") -> None:
    """``imgs`` is (size_y, size_x, n): uint64 stamp, uint16 y, uint16 x, pixels y fastest."""
    size_y, size_x, n = imgs.shape
    out = bytearray()
    for k in range(n):
        out += struct.pack("<QHH", stamps[k], size_y, size_x)
        out += imgs[:, :, k].astype(pixel).ravel(order="F").tobytes()
    path.write_bytes(bytes(out))


# ======================================================================================
# load_bin
# ======================================================================================


def matlab_load_bin(paths, n_lay):
    """Literal translation of load_bin.m: fread, reshape(..., y_res, x_res)', counters."""
    nbr = []
    for p in paths:
        with open(p, "rb") as fi:
            x_res, y_res = (int(v) for v in np.fromfile(fi, "<i2", 2))
        n = int(np.floor(os.path.getsize(p) / x_res / y_res / 2 + 0.5))  # MATLAB round()
        nbr.append(n - n % n_lay)
    total = sum(nbr)
    data = [np.zeros((x_res, y_res, total // n_lay), np.int16) for _ in range(n_lay)]
    for ind, p in enumerate(paths):
        with open(p, "rb") as fi:
            np.fromfile(fi, "<i2", 2)
            for knd in range(sum(nbr[:ind]), sum(nbr[: ind + 1])):  # 0-based knd
                v = np.fromfile(fi, "<i2", y_res * x_res)
                data[knd % n_lay][:, :, knd // n_lay] = v.reshape((y_res, x_res), order="F").T
    return data, nbr


def test_load_bin_orientation_by_hand(tmp_path):
    # MATLAB: x_res=2, y_res=3, samples 1..6 -> reshape(v, 3, 2)' = [1 2 3; 4 5 6]
    f = tmp_path / "a.bin"
    f.write_bytes(struct.pack("<hh", 2, 3) + np.arange(1, 7, dtype="<i2").tobytes())
    res = load_bin(f)
    assert isinstance(res, BinData)
    assert len(res.data) == 1
    assert res.data[0].shape == (2, 3, 1)
    assert res.data[0].dtype == np.int16
    assert np.array_equal(res.data[0][:, :, 0], [[1, 2, 3], [4, 5, 6]])
    assert res.nbr_frames.shape == (1, 1)
    assert res.nbr_frames[0, 0] == 1
    first, nbr = res  # tuple unpacking, as [data, nbr_frames] = load_bin(...)
    assert first is res.data and nbr is res.nbr_frames
    del res, first, nbr
    release()


@pytest.mark.parametrize("n_lay,n_frames", [(1, 5), (2, 6), (3, 9), (3, 7), (4, 3)])
def test_load_bin_matches_matlab_loops(tmp_path, n_lay, n_frames):
    f = tmp_path / "stack.bin"
    write_bin(f, RNG.integers(-32768, 32767, size=(n_frames, 5, 7), dtype=np.int16))
    expected, nbr = matlab_load_bin([f], n_lay)
    for copy in (False, True):
        res = load_bin(f, n_lay, copy=copy)
        assert len(res.data) == n_lay
        for got, want in zip(res.data, expected):
            assert got.shape == want.shape
            assert got.dtype == np.int16
            assert np.array_equal(got, want)
        assert res.nbr_frames.tolist() == [nbr]
        del res, got
    release()


def test_load_bin_two_files_continue_the_layer_cycle(tmp_path):
    f1, f2 = tmp_path / "a.bin", tmp_path / "b.bin"
    write_bin(f1, RNG.integers(-100, 100, size=(6, 4, 3), dtype=np.int16))
    write_bin(f2, RNG.integers(-100, 100, size=(4, 4, 3), dtype=np.int16))
    expected, nbr = matlab_load_bin([f1, f2], 2)
    res = load_bin([f1, str(f2)], 2)
    assert res.nbr_frames.shape == (1, 2)
    assert res.nbr_frames.tolist() == [[6, 4]] == [nbr]
    for got, want in zip(res.data, expected):
        assert got.shape == (4, 3, 5)
        assert np.array_equal(got, want)
        assert not isinstance(got, np.memmap)  # several files are joined in memory
    del res, got
    release()


def test_load_bin_memmap_and_copy(tmp_path):
    f = tmp_path / "a.bin"
    frames = RNG.integers(-50, 50, size=(4, 3, 3), dtype=np.int16)
    write_bin(f, frames)
    mm = load_bin(f, 2).data
    assert all(isinstance(d, np.memmap) for d in mm)
    assert not mm[0].flags.writeable  # read-only map
    cp = load_bin(f, 2, copy=True).data
    assert all(type(d) is np.ndarray and d.flags.writeable for d in cp)
    for a, b in zip(mm, cp):
        assert np.array_equal(a, b)
    # layer l holds frames l, l+2, ...; frame axis last
    assert np.array_equal(cp[1][:, :, 1], frames[3])
    del mm, cp, a, b
    release()


def test_load_bin_drops_incomplete_cycles_and_partial_frames(tmp_path):
    f = tmp_path / "a.bin"
    frames = RNG.integers(-50, 50, size=(5, 3, 4), dtype=np.int16)
    # 5 frames + 17 stray bytes (more than half a frame: MATLAB's round() would add a frame)
    f.write_bytes(
        struct.pack("<hh", 3, 4) + frames.astype("<i2").tobytes() + bytes(range(17))
    )
    with capture_logs() as logs:
        res = load_bin(f, 2)
    assert res.nbr_frames.tolist() == [[4]]
    assert res.data[0].shape == (3, 4, 2)
    assert np.array_equal(res.data[1][:, :, 1], frames[3])
    assert any("ignored" in m for m in logs)
    del res
    release()


def test_load_bin_layer_shift_is_circshift(tmp_path):
    f = tmp_path / "a.bin"
    write_bin(f, RNG.integers(-50, 50, size=(6, 2, 2), dtype=np.int16))
    base = load_bin(f, 3, copy=True).data
    minus = load_bin(f, 3, layer_shift=-1, copy=True).data  # MATLAB circshift(data, -1)
    plus = load_bin(f, 3, layer_shift=1, copy=True).data
    for j in range(3):
        assert np.array_equal(minus[j], base[(j + 1) % 3])
        assert np.array_equal(plus[j], base[(j - 1) % 3])
    same = load_bin(f, 3, layer_shift=3, copy=True).data
    assert all(np.array_equal(a, b) for a, b in zip(same, base))


def test_load_bin_empty_stack(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(struct.pack("<hh", 3, 4))
    res = load_bin(f, 2)
    assert [d.shape for d in res.data] == [(3, 4, 0), (3, 4, 0)]
    assert res.nbr_frames.tolist() == [[0]]


def test_load_bin_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="couldn't find"):
        load_bin(tmp_path / "missing.bin")
    f = tmp_path / "a.bin"
    write_bin(f, np.zeros((2, 2, 2), np.int16))
    with pytest.raises(ValueError, match="nbr_piezo_layers"):
        load_bin(f, 0)
    with pytest.raises(TypeError, match="nbr_piezo_layers"):
        load_bin(f, 1.5)
    with pytest.raises(TypeError, match="ExpID"):
        load_bin(231419)
    with pytest.raises(ValueError, match="no file"):
        load_bin([])
    short = tmp_path / "short.bin"
    short.write_bytes(b"\x02\x00")
    with pytest.raises(ValueError, match="too short"):
        load_bin(short)
    zero = tmp_path / "zero.bin"
    zero.write_bytes(struct.pack("<hh", 0, 4) + bytes(16))
    with pytest.raises(ValueError, match="resolution"):
        load_bin(zero)
    g = tmp_path / "g.bin"
    write_bin(g, np.zeros((2, 3, 2), np.int16))
    with pytest.raises(ValueError, match="changes between files"):
        load_bin([f, g])


# ======================================================================================
# load_wid_data
# ======================================================================================


def matlab_wid_frames(path, frames_to_load):
    """Literal translation of the fseek loop of load_wid_data.m (1-based frame numbers)."""
    with open(path, "rb") as fi:
        size_y, size_x = (int(v) for v in np.fromfile(fi, "<u2", 2))
    out = []
    with open(path, "rb") as fi:
        last = 0
        for f in frames_to_load:
            fi.seek(2 * (f - last - 1) * (size_x * size_y + 2), 1)
            np.fromfile(fi, "<u2", 2)  # tempdim
            out.append(np.fromfile(fi, "<u2", size_y * size_x).reshape((size_y, size_x), order="F"))
            last = f
    return np.stack(out, axis=2)


def test_load_wid_orientation_by_hand(tmp_path):
    # size_y=3, size_x=2, pixels 1..6: fread(fi, [3 2]) fills columns -> [1 4; 2 5; 3 6]
    f = tmp_path / "a.wid"
    f.write_bytes(struct.pack("<HH", 3, 2) + np.arange(1, 7, dtype="<u2").tobytes())
    res = load_wid_data(f)
    assert isinstance(res, WidData)
    assert res.wid_data.shape == (3, 2, 1)
    assert res.wid_data.dtype == np.uint16
    assert np.array_equal(res.wid_data[:, :, 0], [[1, 4], [2, 5], [3, 6]])
    assert isinstance(res.wid_meta_info, MatStruct) and len(res.wid_meta_info) == 0
    del res
    release()


def test_load_wid_roundtrip_memmap_and_copy(tmp_path):
    f = tmp_path / "a.wid"
    imgs, headers = wid_frames(4, 6, 5)
    write_wid(f, imgs, headers)
    res = load_wid_data(f)
    assert isinstance(res.wid_data, np.memmap) and not res.wid_data.flags.writeable
    assert res.wid_data.shape == (6, 5, 4)
    assert np.array_equal(res.wid_data, imgs)
    cp = load_wid_data(str(f), copy=True).wid_data
    assert type(cp) is np.ndarray and cp.flags.writeable and np.array_equal(cp, imgs)
    del res, cp
    release()


def test_load_wid_frame_selection(tmp_path):
    f = tmp_path / "a.wid"
    imgs = plain_frames(101, 2, 2)  # 4 pixels per frame: too small for a camera header
    write_wid(f, imgs)
    n = 101
    cases = {
        "first": [1],
        "firstlast": [1, n],
        "100th": [100],
        "7": list(range(1, n + 1, 7)),
        7: list(range(1, n + 1, 7)),
        "1": list(range(1, n + 1)),
    }
    for frames, numbers in cases.items():
        res = load_wid_data(f, frames=frames)
        assert np.array_equal(res.wid_data, imgs[:, :, [k - 1 for k in numbers]]), frames
        assert isinstance(res.wid_data, np.memmap) == (frames != "firstlast"), frames
        assert np.array_equal(res.wid_data, matlab_wid_frames(f, numbers)), frames
        assert isinstance(res.wid_meta_info, MatStruct)
        del res
    release()


def test_load_wid_firstlast_of_single_frame_file_returns_it_twice(tmp_path):
    f = tmp_path / "a.wid"
    imgs = plain_frames(1, 3, 3)
    write_wid(f, imgs)
    res = load_wid_data(f, frames="firstlast")
    assert res.wid_data.shape == (3, 3, 2)
    assert np.array_equal(res.wid_data[:, :, 0], res.wid_data[:, :, 1])


def test_load_wid_two_files_with_selection(tmp_path):
    f1, f2 = tmp_path / "a.wid", tmp_path / "b.wid"
    a, b = plain_frames(5, 4, 3), plain_frames(8, 4, 3)
    write_wid(f1, a)
    write_wid(f2, b)
    full = load_wid_data([f1, f2]).wid_data
    assert full.shape == (4, 3, 13)
    assert np.array_equal(full, np.concatenate([a, b], axis=2))
    fl = load_wid_data([f1, f2], frames="firstlast").wid_data
    assert np.array_equal(fl, np.stack([a[:, :, 0], a[:, :, 4], b[:, :, 0], b[:, :, 7]], axis=2))
    every = load_wid_data([f1, f2], frames=3).wid_data
    assert np.array_equal(every, np.concatenate([a[:, :, ::3], b[:, :, ::3]], axis=2))
    del full
    release()


def test_load_wid_meta_mode(tmp_path):
    f1, f2 = tmp_path / "a.wid", tmp_path / "b.wid"
    a, ha = wid_frames(3, 4, 5, first=10)
    b, hb = wid_frames(2, 4, 5, first=99999990)
    write_wid(f1, a, ha)
    write_wid(f2, b, hb)
    res = load_wid_data([f1, f2], frames="meta")
    assert res.wid_data.shape == (0, 0)
    meta = res.wid_meta_info
    assert isinstance(meta, np.ndarray) and meta.shape == (1, 5) and meta.dtype == object
    assert all(isinstance(m, MatStruct) for m in meta.ravel())
    assert list(meta[0, 0].keys()) == ["framenumber", "timestamp", "dimension", "file_id"]
    assert [m.framenumber for m in meta.ravel()] == [10, 11, 12, 99999990, 99999991]
    assert [m.file_id for m in meta.ravel()] == [1, 1, 1, 2, 2]  # 1-based, as in MATLAB
    assert [m.timestamp for m in meta.ravel()] == [stamp_of(k) for k in (0, 1, 2, 0, 1)]
    for m in meta.ravel():
        assert isinstance(m.timestamp, datetime)
        assert m.dimension.shape == (1, 2) and m.dimension.tolist() == [[4, 5]]


def test_load_wid_meta_single_frame_is_still_an_array(tmp_path):
    f = tmp_path / "a.wid"
    imgs, headers = wid_frames(1, 4, 4)
    write_wid(f, imgs, headers)
    meta = load_wid_data(f, frames="meta").wid_meta_info
    assert meta.shape == (1, 1) and meta[0, 0].framenumber == 1


def test_load_wid_non_integer_frame_count_is_floored(tmp_path):
    f = tmp_path / "a.wid"
    write_wid(f, plain_frames(3, 4, 4))
    f.write_bytes(f.read_bytes() + b"\x01\x02\x03\x04\x05")
    with capture_logs() as logs:
        res = load_wid_data(f)
    assert res.wid_data.shape == (4, 4, 3)
    assert any("not an integer" in m for m in logs)
    del res
    release()


def test_load_wid_invalid_headers(tmp_path):
    imgs, headers = wid_frames(2, 4, 4)
    bad_bcd = [list(h) for h in headers]
    bad_bcd[1][0] = 0xAB  # not decimal digits
    f = tmp_path / "bcd.wid"
    write_wid(f, imgs, bad_bcd)
    with pytest.raises(ValueError, match="frame 2 is not BCD"):
        load_wid_data(f, frames="meta")
    bad_date = [list(h) for h in headers]
    bad_date[0][6] = 0x13  # month 13
    g = tmp_path / "date.wid"
    write_wid(g, imgs, bad_date)
    with pytest.raises(ValueError, match="frame 1 has an invalid timestamp"):
        load_wid_data(g, frames="meta")
    # the images themselves do not depend on the header pixels being valid
    assert load_wid_data(f).wid_data.shape == (4, 4, 2)
    tiny = tmp_path / "tiny.wid"
    write_wid(tiny, np.zeros((2, 2, 1), np.uint16))
    with pytest.raises(ValueError, match="camera header"):
        load_wid_data(tiny, frames="meta")


def test_load_wid_errors(tmp_path):
    f = tmp_path / "a.wid"
    write_wid(f, plain_frames(3, 4, 4))
    with pytest.raises(FileNotFoundError):
        load_wid_data(tmp_path / "missing.wid")
    with pytest.raises(ValueError, match="frames="):
        load_wid_data(f, frames="last")
    with pytest.raises(ValueError, match="step must be >= 1"):
        load_wid_data(f, frames="0")
    with pytest.raises(ValueError, match="step must be >= 1"):
        load_wid_data(f, frames=-2)
    with pytest.raises(TypeError, match="frames"):
        load_wid_data(f, frames=2.5)
    with pytest.raises(ValueError, match="cannot load frame 100"):
        load_wid_data(f, frames="100th")
    with pytest.raises(TypeError, match="ExpID"):
        load_wid_data(7)
    g = tmp_path / "g.wid"
    write_wid(g, plain_frames(2, 5, 4))
    with pytest.raises(ValueError, match="changes between files"):
        load_wid_data([f, g])
    z = tmp_path / "z.wid"
    z.write_bytes(struct.pack("<HH", 0, 4))
    with pytest.raises(ValueError, match="invalid frame size"):
        load_wid_data(z)


# ======================================================================================
# load_wii_data and load_roi_data
# ======================================================================================

WII_TEXT = (
    "version=3;param.numberROIs=3;param.illumination=470nm;param.gain=1.5;param.exp=1e3;"
    "param.off=-2.5;param.empty=;param.nan=NaN;param.inf=Inf;"
    "ref.pt1.X=10;ref.pt1.Y=20;ref.pt2.X=30;"
    "roi.coord.0.X=5;roi.coord.0.Y=6;roi.coord.0.in_frame=1;roi.coord.1=7;"
    "roi.label.0.name=V1;roi.label.0.acronym=VISp;roi.label.2.name=S1;"
    "cam.fps=100;cam.lens.mm=25;"
    "extra.items.2.name=foo;extra.items.2.size=3;"
    "deep.a.b.c.d=9;"
    "last=pair"  # no closing ';'
)


def make_wii(tmp_path, text=WII_TEXT, name="a.wii"):
    mean_proj = RNG.integers(0, 65535, size=(5, 4), dtype=np.uint16)  # (m2, m1)
    frame = RNG.integers(0, 65535, size=(3, 6), dtype=np.uint16)  # (f1, f2)
    f = tmp_path / name
    write_wii(f, text, mean_proj, frame)
    return f, mean_proj, frame


def test_load_wii_arrays_by_hand(tmp_path):
    # mean projection m1=3, m2=2, values 1..6 read as [m2, m1] column-major -> [1 3 5; 2 4 6]
    f = tmp_path / "a.wii"
    text = "param.numberROIs=1;"
    f.write_bytes(
        struct.pack("<I", len(text)) + text.encode()
        + struct.pack("<HH", 3, 2) + np.arange(1, 7, dtype="<u2").tobytes()
        + struct.pack("<HH", 2, 3) + np.arange(11, 17, dtype="<u2").tobytes()
    )
    res = load_wii_data(f)
    assert isinstance(res, WiiData)
    assert res.wii_mean_proj.dtype == np.uint16
    assert np.array_equal(res.wii_mean_proj, [[1, 3, 5], [2, 4, 6]])
    assert res.frame_nonrotated.shape == (2, 3)  # [f1, f2]
    assert np.array_equal(res.frame_nonrotated, [[11, 13, 15], [12, 14, 16]])


def test_load_wii_roundtrip_arrays(tmp_path):
    f, mean_proj, frame = make_wii(tmp_path)
    res = load_wii_data(str(f))
    assert np.array_equal(res.wii_mean_proj, mean_proj)
    assert np.array_equal(res.frame_nonrotated, frame)
    assert type(res.wii_mean_proj) is np.ndarray


def test_load_wii_config_values(tmp_path):
    f, _, _ = make_wii(tmp_path)
    cfg = load_wii_data(f).wii_config
    assert isinstance(cfg, MatStruct)
    assert list(cfg.keys())[:3] == ["roi", "ref", "roi_label"]  # MATLAB field order
    assert cfg.version == 3.0 and isinstance(cfg.version, float)
    assert cfg.param.numberROIs == 3.0
    assert cfg.param.illumination == "470nm"
    assert cfg.param.gain == 1.5 and cfg.param.exp == 1000.0 and cfg.param.off == -2.5
    assert cfg.param.empty == "" and cfg.param.nan == "NaN"  # NaN stays text, as in MATLAB
    assert cfg.param.inf == float("inf")
    assert cfg.cam.fps == 100.0 and cfg.cam.lens.mm == 25.0
    assert cfg["deep.a.b.c.d"] == 9.0  # more than 4 parts: kept under the full name
    assert cfg.last == "pair"  # last pair without ';' is kept


def test_load_wii_config_struct_arrays(tmp_path):
    f, _, _ = make_wii(tmp_path)
    cfg = load_wii_data(f).wii_config
    ref = cfg.ref
    assert ref.shape == (2, 1) and ref.dtype == object
    assert ref[0, 0].X == 10.0 and ref[0, 0].Y == 20.0 and ref[1, 0].X == 30.0
    assert ref[1, 0].Y.shape == (0, 0)  # same fields in every element, [] where unset
    coord = cfg.roi.coord
    assert coord.shape == (818, 1)  # preallocated as in MATLAB
    assert list(coord[0, 0].keys()) == ["X", "Y", "in_frame"]
    assert (coord[0, 0].X, coord[0, 0].Y, coord[0, 0].in_frame) == (5.0, 6.0, 1.0)
    assert coord[1, 0].Y == 7.0  # 'roi.coord.1=7' sets Y of element 2 (MATLAB quirk)
    assert coord[1, 0].X.size == 0 and coord[817, 0].X.size == 0
    label = cfg.roi_label
    assert label.shape == (818, 1)
    assert label[0, 0].name == "V1" and label[0, 0].acronym == "VISp" and label[2, 0].name == "S1"
    assert label[1, 0].name.shape == (0, 0) and label[1, 0].acronym.shape == (0, 0)
    items = cfg.extra["items"]  # generic <a>.<b>.<i>.<field>: struct array, 1-based growth
    assert items.shape == (1, 3)
    assert items[0, 2].name == "foo" and items[0, 2].size == 3.0
    assert items[0, 0].name.shape == (0, 0)


def test_load_wii_config_grows_preallocated_arrays(tmp_path):
    f, _, _ = make_wii(tmp_path, text="roi.label.819.name=late;roi.coord.820.X=1;")
    cfg = load_wii_data(f).wii_config
    assert cfg.roi_label.shape == (820, 1) and cfg.roi_label[819, 0].name == "late"
    assert cfg.roi.coord.shape == (821, 1) and cfg.roi.coord[820, 0].X == 1.0


def test_load_wii_config_conflicts_raise(tmp_path):
    f, _, _ = make_wii(tmp_path, text="a=1;a.b=2;")
    with pytest.raises(ValueError, match="non-struct"):
        load_wii_data(f)
    g, _, _ = make_wii(tmp_path, text="ref.pt.X=1;", name="g.wii")  # 'pt' has no index digit
    with pytest.raises(ValueError, match="not an index"):
        load_wii_data(g)
    h, _, _ = make_wii(tmp_path, text="ref.x=1;", name="h.wii")  # would overwrite the array
    with pytest.raises(ValueError, match="non-struct"):
        load_wii_data(h)


def test_load_wii_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_wii_data(tmp_path / "missing.wii")
    f, _, _ = make_wii(tmp_path)
    data = f.read_bytes()
    (tmp_path / "cut.wii").write_bytes(data[:30])
    with pytest.raises(ValueError, match="too short"):
        load_wii_data(tmp_path / "cut.wii")
    (tmp_path / "cut2.wii").write_bytes(data[:-5])
    with pytest.raises(ValueError, match="too short"):
        load_wii_data(tmp_path / "cut2.wii")
    (tmp_path / "empty.wii").write_bytes(b"")
    with pytest.raises(ValueError, match="too short"):
        load_wii_data(tmp_path / "empty.wii")


def make_roi_pair(tmp_path, n_frames=6, n_roi=3, name="a", first=1):
    make_wii(tmp_path, text=f"param.numberROIs={n_roi};", name=f"{name}.wii")
    traces = RNG.integers(0, 65535, size=(n_frames, n_roi), dtype=np.uint16)
    roi = tmp_path / f"{name}.roi"
    write_roi(roi, traces, first=first)
    return roi, traces


def test_load_roi_full_by_hand(tmp_path):
    roi, traces = make_roi_pair(tmp_path, n_frames=2, n_roi=3)
    traces[:] = [[1, 2, 3], [4, 5, 6]]
    write_roi(roi, traces)
    res = load_roi_data(roi)
    assert isinstance(res, RoiData)
    assert res.roi_data.shape == (3, 1, 2)  # [nbr_ROIs 1 nFrames]
    assert res.roi_data.dtype == np.uint16
    assert res.roi_data[:, 0, 0].tolist() == [1, 2, 3]
    assert res.roi_data[:, 0, 1].tolist() == [4, 5, 6]
    assert isinstance(res.roi_meta_info, MatStruct) and len(res.roi_meta_info) == 0
    del res
    release()


def test_load_roi_memmap_copy_and_selection(tmp_path):
    roi, traces = make_roi_pair(tmp_path, n_frames=12, n_roi=4)
    res = load_roi_data(roi)
    assert isinstance(res.roi_data, np.memmap) and res.roi_data.shape == (4, 1, 12)
    assert np.array_equal(res.roi_data[:, 0, :], traces.T)
    cp = load_roi_data(str(roi), copy=True).roi_data
    assert type(cp) is np.ndarray and np.array_equal(cp, res.roi_data)
    cases = {"first": [1], "firstlast": [1, 12], "5": [1, 6, 11], 5: [1, 6, 11]}
    for frames, numbers in cases.items():
        got = load_roi_data(roi, frames=frames).roi_data
        assert np.array_equal(got[:, 0, :], traces[[k - 1 for k in numbers]].T), frames
    with pytest.raises(ValueError, match="cannot load frame 100"):
        load_roi_data(roi, frames="100th")
    del res, cp, got
    release()


def test_load_roi_meta(tmp_path):
    roi1, _ = make_roi_pair(tmp_path, n_frames=3, n_roi=2, name="a", first=5)
    roi2, _ = make_roi_pair(tmp_path, n_frames=2, n_roi=2, name="b", first=500)
    res = load_roi_data([roi1, roi2], frames="meta")
    assert res.roi_data.shape == (0, 0)
    meta = res.roi_meta_info
    assert meta.shape == (1, 5)
    assert [m.framenumber for m in meta.ravel()] == [5, 6, 7, 500, 501]
    assert [m.file_id for m in meta.ravel()] == [1, 1, 1, 2, 2]
    assert [m.timestamp for m in meta.ravel()] == [stamp_of(k) for k in (0, 1, 2, 0, 1)]
    assert all(m.dimension.tolist() == [[2, 1]] for m in meta.ravel())  # [nbr_ROIs, 1]


def test_load_roi_two_files_concatenate(tmp_path):
    roi1, t1 = make_roi_pair(tmp_path, n_frames=3, n_roi=2, name="a")
    roi2, t2 = make_roi_pair(tmp_path, n_frames=4, n_roi=2, name="b")
    got = load_roi_data([roi1, roi2]).roi_data
    assert got.shape == (2, 1, 7)
    assert np.array_equal(got[:, 0, :], np.concatenate([t1, t2]).T)


def test_load_roi_wii_path_and_odd_directory_name(tmp_path):
    odd = tmp_path / "x.roi_dir"
    odd.mkdir()
    roi, traces = make_roi_pair(odd, n_frames=2, n_roi=3)
    assert np.array_equal(load_roi_data(roi).roi_data[:, 0, :], traces.T)  # strrep would break
    other = tmp_path / "elsewhere.wii"
    other.write_bytes((odd / "a.wii").read_bytes())
    (odd / "a.wii").unlink()
    assert np.array_equal(load_roi_data(roi, wii_path=other).roi_data[:, 0, :], traces.T)
    with pytest.raises(FileNotFoundError):
        load_roi_data(roi)
    with pytest.raises(ValueError, match="1 .roi file"):
        load_roi_data(roi, wii_path=[other, other])


def test_load_roi_trailing_bytes_and_errors(tmp_path):
    roi, _ = make_roi_pair(tmp_path, n_frames=3, n_roi=2)
    roi.write_bytes(roi.read_bytes() + b"\x00\x01\x02")
    with capture_logs() as logs:
        assert load_roi_data(roi).roi_data.shape == (2, 1, 3)
    assert any("stray" in m for m in logs)
    (tmp_path / "short.roi").write_bytes(struct.pack("<HH", 50, 50))
    (tmp_path / "short.wii").write_bytes((tmp_path / "a.wii").read_bytes())
    with pytest.raises(ValueError, match="too short"):
        load_roi_data(tmp_path / "short.roi")
    with pytest.raises(FileNotFoundError):
        load_roi_data(tmp_path / "missing.roi")
    roi3, _ = make_roi_pair(tmp_path, n_frames=2, n_roi=5, name="c")
    with pytest.raises(ValueError, match="number of ROIs changes"):
        load_roi_data([roi, roi3])
    no_n, _ = make_roi_pair(tmp_path, name="d")
    make_wii(tmp_path, text="param.gain=1;", name="d.wii")
    with pytest.raises(ValueError, match="numberROIs"):
        load_roi_data(no_n)
    with pytest.raises(TypeError, match="ExpID"):
        load_roi_data(12)


# ======================================================================================
# load_eye_monitor_data
# ======================================================================================


def eye_frames(n, size_x, size_y):
    return [
        (eye_meta(size_x, size_y, k), RNG.integers(0, 255, size=(size_x, size_y), dtype=np.uint8))
        for k in range(n)
    ]


def test_load_eye_by_hand(tmp_path):
    # size_x=4, size_y=3, pixels 1..12 -> reshape(v, 4, 3) = [1 5 9; 2 6 10; 3 7 11; 4 8 12]
    f = tmp_path / "a.eye"
    f.write_bytes(
        np.array(eye_meta(4, 3), "<f8").tobytes() + np.arange(1, 13, dtype=np.uint8).tobytes()
    )
    res = load_eye_monitor_data(f)
    assert isinstance(res, EyeData)
    assert res.idata.shape == (4, 3, 1) and res.idata.dtype == np.uint8
    assert np.array_equal(
        res.idata[:, :, 0], [[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]
    )
    assert res.imeta_info.shape == (9, 1) and res.imeta_info.dtype == np.float64
    assert res.imeta_info[:, 0].tolist() == eye_meta(4, 3)
    del res
    release()


def test_load_eye_roundtrip_memmap_copy_and_no_image(tmp_path):
    f = tmp_path / "a.eye"
    frames = eye_frames(5, 6, 4)
    write_eye(f, frames)
    res = load_eye_monitor_data(f, True)
    assert isinstance(res.idata, np.memmap) and res.idata.shape == (6, 4, 5)
    for k, (meta, img) in enumerate(frames):
        assert np.array_equal(res.idata[:, :, k], img)
        assert res.imeta_info[:, k].tolist() == meta
    assert type(res.imeta_info) is np.ndarray  # meta info is always in memory
    cp = load_eye_monitor_data(str(f), copy=True).idata
    assert type(cp) is np.ndarray and np.array_equal(cp, res.idata)
    none = load_eye_monitor_data(f, load_img=False)
    assert none.idata.shape == (0, 0)
    assert np.array_equal(none.imeta_info, res.imeta_info)
    del res, cp, none
    release()


def test_load_eye_big_endian_new_format(tmp_path):
    f = tmp_path / "be.eye"
    frames = eye_frames(3, 5, 3)
    write_eye(f, frames, byte_order=">")
    with capture_logs() as logs:
        res = load_eye_monitor_data(f)
    assert any("big-endian" in m for m in logs)
    assert res.idata.shape == (5, 3, 3)
    assert all(np.array_equal(res.idata[:, :, k], frames[k][1]) for k in range(3))
    assert res.imeta_info[:, 2].tolist() == frames[2][0]
    del res
    release()


def test_load_eye_big_endian_oldest_format_subtracts_whisker_box(tmp_path):
    # meta[3], meta[4] non-zero integers (MATLAB meta_info(3), (4)): size = meta(5)-meta(3)
    f = tmp_path / "old.eye"
    img = RNG.integers(0, 255, size=(4, 3), dtype=np.uint8)
    write_eye(f, [(eye_meta(14, 23, m3=10.0, m4=20.0), img)], byte_order=">")
    res = load_eye_monitor_data(f)
    assert res.idata.shape == (4, 3, 1) and np.array_equal(res.idata[:, :, 0], img)
    del res
    release()


def test_load_eye_zero_frame_size_assumes_full_frame(tmp_path):
    f = tmp_path / "bad.eye"
    img = np.zeros((744, 480), np.uint8)
    img[3, 5] = 9
    write_eye(f, [(eye_meta(0, 0), img)])
    with pytest.warns(UserWarning, match="corrupt"):
        res = load_eye_monitor_data(f)
    assert res.idata.shape == (744, 480, 1) and res.idata[3, 5, 0] == 9
    del res
    release()


def test_load_eye_frame_size_override_replaces_filename_hack(tmp_path):
    f = tmp_path / "S1-T128658.eye"  # MATLAB forced size_y = 284 for these stacks
    frames = eye_frames(3, 4, 3)
    # header claims 4x5 although the frames are 4x3
    write_eye(f, [(eye_meta(4, 5, k), img) for k, (_, img) in enumerate(frames)])
    res = load_eye_monitor_data(f, frame_size=(4, 3))
    assert res.idata.shape == (4, 3, 3)
    assert all(np.array_equal(res.idata[:, :, k], frames[k][1]) for k in range(3))
    with pytest.raises(ValueError, match="frame_size"):
        load_eye_monitor_data(f, frame_size=(4,))
    with pytest.raises(ValueError, match="invalid frame size"):
        load_eye_monitor_data(f, frame_size=(4, -3))
    del res
    release()


def test_load_eye_frame_size_changing_mid_recording(tmp_path):
    f = tmp_path / "var.eye"
    a = RNG.integers(1, 255, size=(4, 3), dtype=np.uint8)
    b = RNG.integers(1, 255, size=(6, 2), dtype=np.uint8)
    c = RNG.integers(1, 255, size=(2, 2), dtype=np.uint8)  # smaller: file does not divide evenly
    write_eye(f, [(eye_meta(4, 3, 0), a), (eye_meta(6, 2, 1), b), (eye_meta(2, 2, 2), c)])
    with capture_logs() as logs:
        res = load_eye_monitor_data(f)
    assert any("change mid recording" in m for m in logs)
    assert res.idata.shape == (6, 3, 3)  # zero-padded to the largest frame
    assert np.array_equal(res.idata[:4, :3, 0], a) and res.idata[4:, :, 0].sum() == 0
    assert np.array_equal(res.idata[:6, :2, 1], b) and res.idata[:, 2, 1].sum() == 0
    assert np.array_equal(res.idata[:2, :2, 2], c)
    assert res.imeta_info.shape == (9, 3) and res.imeta_info[4:6, 1].tolist() == [6.0, 2.0]
    none = load_eye_monitor_data(f, load_img=False)
    assert none.imeta_info.shape == (9, 3) and none.idata.shape == (0, 0)


def test_load_eye_incomplete_last_frame_is_dropped(tmp_path):
    f = tmp_path / "cut.eye"
    frames = eye_frames(3, 4, 3)
    write_eye(f, frames)
    f.write_bytes(f.read_bytes()[:-5])  # cut into the last frame
    with pytest.warns(UserWarning, match="do not make a whole frame"):
        res = load_eye_monitor_data(f)
    assert res.idata.shape == (4, 3, 2) and res.imeta_info.shape == (9, 2)
    del res
    release()  # a file cannot be rewritten while it is memory-mapped (Windows)
    f.write_bytes(f.read_bytes() + bytes(range(3)))  # junk shorter than a meta block
    with pytest.warns(UserWarning, match="do not make a whole frame"):
        assert load_eye_monitor_data(f).idata.shape == (4, 3, 2)
    release()


def test_load_eye_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_eye_monitor_data(tmp_path / "missing.eye")
    short = tmp_path / "short.eye"
    short.write_bytes(bytes(40))
    with pytest.raises(ValueError, match="too short"):
        load_eye_monitor_data(short)
    neg = tmp_path / "neg.eye"
    neg.write_bytes(np.array(eye_meta(-4, 3), "<f8").tobytes() + bytes(20))
    with pytest.raises(ValueError, match="invalid frame size"):
        load_eye_monitor_data(neg)
    with pytest.raises(TypeError, match="ExpID"):
        load_eye_monitor_data(5, True)


def test_load_eye_empty_after_header_less_frames(tmp_path):
    f = tmp_path / "one.eye"
    f.write_bytes(np.array(eye_meta(4, 3), "<f8").tobytes())  # a meta block, no pixels
    with pytest.warns(UserWarning):
        res = load_eye_monitor_data(f)
    assert res.idata.shape == (4, 3, 0) and res.imeta_info.shape == (9, 0)


# ======================================================================================
# load_vid_data
# ======================================================================================


def test_load_vid_by_hand(tmp_path):
    # size_y=3, size_x=2, pixels 1..6: reshape(v, 3, 2)' = [1 2 3; 4 5 6]  (size_x x size_y)
    f = tmp_path / "a.run"
    f.write_bytes(
        np.array(vid_meta(3, 2), "<f8").tobytes() + np.arange(1, 7, dtype="<i2").tobytes()
    )
    res = load_vid_data(f)
    assert isinstance(res, VidData)
    assert res.vdata.shape == (2, 3, 1) and res.vdata.dtype == np.int16
    assert np.array_equal(res.vdata[:, :, 0], [[1, 2, 3], [4, 5, 6]])
    assert res.vmeta_info.shape == (4, 1) and res.vmeta_info[:, 0].tolist() == vid_meta(3, 2)
    del res
    release()


@pytest.mark.parametrize("dtype,lo,hi", [("int16", -30000, 30000), ("uint16", 0, 65535)])
def test_load_vid_roundtrip(tmp_path, dtype, lo, hi):
    f = tmp_path / "a.vid"
    imgs = RNG.integers(lo, hi, size=(5, 4, 3), dtype=np.dtype(dtype))
    write_vid(f, [vid_meta(4, 5, k) for k in range(3)], imgs, pixel="<" + dtype[0] + "2")
    res = load_vid_data(f, dtype)
    assert res.vdata.dtype == np.dtype(dtype) and res.vdata.shape == (5, 4, 3)
    assert isinstance(res.vdata, np.memmap) and np.array_equal(res.vdata, imgs)
    assert res.vmeta_info.shape == (4, 3)
    assert res.vmeta_info[:, 1].tolist() == vid_meta(4, 5, 1)
    cp = load_vid_data(str(f), dtype, copy=True).vdata
    assert type(cp) is np.ndarray and np.array_equal(cp, imgs)
    del res, cp
    release()


def test_load_vid_big_endian_old_file(tmp_path):
    f = tmp_path / "old.run"
    imgs = RNG.integers(-30000, 30000, size=(3, 2, 2), dtype=np.int16)
    write_vid(f, [vid_meta(2, 3, k) for k in range(2)], imgs, pixel=">i2", byte_order=">")
    with capture_logs() as logs:
        res = load_vid_data(f)
    assert any("big-endian" in m for m in logs)
    assert res.vdata.shape == (3, 2, 2) and np.array_equal(res.vdata, imgs)
    cp = load_vid_data(f, copy=True).vdata
    assert cp.dtype == np.int16 and cp.dtype.isnative and np.array_equal(cp, imgs)
    assert res.vmeta_info[:, 1].tolist() == vid_meta(2, 3, 1)
    del res, cp
    release()


def test_load_vid_zero_size_header_assumes_1344_by_1024(tmp_path):
    f = tmp_path / "zero.vid"
    img = np.zeros((1024, 1344, 1), np.int16)
    img[7, 11, 0] = -5
    write_vid(f, [vid_meta(0, 0)], img)
    with capture_logs() as logs:
        res = load_vid_data(f)
    assert any("assuming 1344 x 1024" in m for m in logs)
    assert res.vdata.shape == (1024, 1344, 1) and res.vdata[7, 11, 0] == -5
    del res
    release()


def test_load_vid_trailing_bytes_and_empty(tmp_path):
    f = tmp_path / "a.ton"
    imgs = RNG.integers(-9, 9, size=(2, 2, 2), dtype=np.int16)
    write_vid(f, [vid_meta(2, 2, k) for k in range(2)], imgs)
    f.write_bytes(f.read_bytes() + b"xyz")
    with capture_logs() as logs:
        res = load_vid_data(f)
    assert res.vdata.shape == (2, 2, 2) and any("stray" in m for m in logs)
    only_meta = tmp_path / "meta.ton"
    only_meta.write_bytes(np.array(vid_meta(2, 3), "<f8").tobytes())
    res2 = load_vid_data(only_meta)
    assert res2.vdata.shape == (3, 2, 0)  # (size_x, size_y, 0)
    assert res2.vmeta_info.shape == (4, 0)
    del res, res2
    release()


def test_load_vid_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_vid_data(tmp_path / "missing.run")
    f = tmp_path / "a.run"
    write_vid(f, [vid_meta(2, 2)], np.zeros((2, 2, 1), np.int16))
    with pytest.raises(ValueError, match="dtype"):
        load_vid_data(f, "int8")
    short = tmp_path / "short.run"
    short.write_bytes(bytes(10))
    with pytest.raises(ValueError, match="too short"):
        load_vid_data(short)
    neg = tmp_path / "neg.run"
    neg.write_bytes(np.array(vid_meta(-2, 2), "<f8").tobytes() + bytes(8))
    with pytest.raises(ValueError, match="invalid frame size"):
        load_vid_data(neg)
    with pytest.raises(TypeError, match="ExpID"):
        load_vid_data(3)


# ======================================================================================
# load_aux_vid_data
# ======================================================================================


def test_load_aux_vid_by_hand(tmp_path):
    # size_y=3, size_x=2, pixels 1..6 read as [size_y, size_x] -> [1 4; 2 5; 3 6]
    f = tmp_path / "a.auxvid"
    f.write_bytes(struct.pack("<QHH", 1234567890123, 3, 2) + np.arange(1, 7, dtype="<u2").tobytes())
    res = load_aux_vid_data(f)
    assert isinstance(res, AuxVidData)
    assert res.vid_data.shape == (3, 2, 1) and res.vid_data.dtype == np.uint16
    assert np.array_equal(res.vid_data[:, :, 0], [[1, 4], [2, 5], [3, 6]])
    assert isinstance(res.vid_meta_info, MatStruct) and len(res.vid_meta_info) == 0
    del res
    release()


@pytest.mark.parametrize("bit_depth,pixel", [("uint16", "<u2"), ("uint8", "u1")])
def test_load_aux_vid_selection_reads_the_right_frames(tmp_path, bit_depth, pixel):
    f = tmp_path / "a.auxvid"
    n = 12
    imgs = RNG.integers(0, 255, size=(4, 5, n), dtype=np.dtype(pixel))
    write_aux(f, [10 * k for k in range(n)], imgs, pixel=pixel)
    full = load_aux_vid_data(f, bit_depth=bit_depth)
    assert isinstance(full.vid_data, np.memmap) and full.vid_data.dtype == np.dtype(pixel)
    assert np.array_equal(full.vid_data, imgs)
    cases = {"first": [1], "firstlast": [1, n], "5": [1, 6, 11], 4: [1, 5, 9]}
    for frames, numbers in cases.items():
        got = load_aux_vid_data(f, frames, bit_depth=bit_depth).vid_data
        # MATLAB skipped with (size_x + size_y) instead of size_x * size_y and read garbage
        assert np.array_equal(got, imgs[:, :, [k - 1 for k in numbers]]), frames
    del full, got
    release()


def test_load_aux_vid_meta_and_two_files(tmp_path):
    f1, f2 = tmp_path / "a.auxvid", tmp_path / "b.auxvid"
    a = RNG.integers(0, 65535, size=(3, 4, 2), dtype=np.uint16)
    b = RNG.integers(0, 65535, size=(3, 4, 3), dtype=np.uint16)
    stamps_a = [2**60 + 1, 2**60 + 3]  # beyond 2**53: exact here, rounded by MATLAB's double
    stamps_b = [5, 6, 7]
    write_aux(f1, stamps_a, a)
    write_aux(f2, stamps_b, b)
    res = load_aux_vid_data([f1, f2], frames="meta")
    assert res.vid_data.shape == (0, 0)
    meta = res.vid_meta_info
    assert meta.shape == (1, 5)
    assert list(meta[0, 0].keys()) == ["timestamp", "dimension", "file_id"]
    assert [m.timestamp for m in meta.ravel()] == stamps_a + stamps_b
    assert [m.file_id for m in meta.ravel()] == [1, 1, 2, 2, 2]
    assert all(m.dimension.tolist() == [[3, 4]] for m in meta.ravel())  # [dim_y, dim_x]
    both = load_aux_vid_data([f1, f2]).vid_data
    assert np.array_equal(both, np.concatenate([a, b], axis=2))
    sel = load_aux_vid_data([f1, f2], frames="firstlast").vid_data
    assert np.array_equal(sel, np.stack([a[:, :, 0], a[:, :, 1], b[:, :, 0], b[:, :, 2]], axis=2))
    del both
    release()


def test_load_aux_vid_errors(tmp_path):
    f = tmp_path / "a.auxvid"
    write_aux(f, [1, 2, 3], RNG.integers(0, 255, size=(3, 3, 3), dtype=np.uint16))
    with pytest.raises(FileNotFoundError):
        load_aux_vid_data(tmp_path / "missing.auxvid")
    with pytest.raises(ValueError, match="bit_depth"):
        load_aux_vid_data(f, bit_depth="int16")
    with pytest.raises(ValueError, match="cannot load frame 100"):
        load_aux_vid_data(f, frames="100th")
    with pytest.raises(ValueError, match="frames="):
        load_aux_vid_data(f, frames="all")
    g = tmp_path / "g.auxvid"
    write_aux(g, [1], RNG.integers(0, 255, size=(4, 3, 1), dtype=np.uint16))
    with pytest.raises(ValueError, match="changes between files"):
        load_aux_vid_data([f, g])
    short = tmp_path / "short.auxvid"
    short.write_bytes(bytes(6))
    with pytest.raises(ValueError, match="too short"):
        load_aux_vid_data(short)
    with capture_logs() as logs:
        f.write_bytes(f.read_bytes() + b"1")
        assert load_aux_vid_data(f).vid_data.shape == (3, 3, 3)
    assert any("not an integer" in m for m in logs)


# ======================================================================================
# read_tif
# ======================================================================================


def test_read_tif_reports_missing_package():
    saved = sys.modules.get("tifffile", "absent")
    sys.modules["tifffile"] = None  # makes `import tifffile` raise ImportError
    try:
        with pytest.raises(ImportError, match="tifffile"):
            read_tif("anything.tif")
    finally:
        if saved == "absent":
            del sys.modules["tifffile"]
        else:
            sys.modules["tifffile"] = saved


def test_read_tif_roundtrip(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    gray = RNG.integers(0, 65535, size=(7, 9), dtype=np.uint16)
    f = tmp_path / "mean.tif"
    tifffile.imwrite(f, gray)
    got = read_tif(f)
    assert got.shape == (7, 9) and got.dtype == np.uint16 and np.array_equal(got, gray)
    rgb = RNG.integers(0, 255, size=(5, 6, 3), dtype=np.uint8)
    g = tmp_path / "rgb.tif"
    tifffile.imwrite(g, rgb, photometric="rgb")
    assert np.array_equal(read_tif(str(g)), rgb)
    stack = RNG.integers(0, 255, size=(3, 4, 5), dtype=np.uint8)
    h = tmp_path / "stack.tif"
    tifffile.imwrite(h, stack, photometric="minisblack")  # 3 grayscale pages
    assert np.array_equal(read_tif(h), stack[0])  # imread returns the first image
    assert np.array_equal(read_tif(h, page=2), stack[2])
    assert np.array_equal(read_tif(h, page=None), stack)
    with pytest.raises(FileNotFoundError):
        read_tif(tmp_path / "missing.tif")


def test_read_tif_planar_rgb_is_channels_last(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    rgb = RNG.integers(0, 255, size=(5, 6, 3), dtype=np.uint8)  # rows x columns x channels
    f = tmp_path / "planar.tif"
    tifffile.imwrite(f, np.moveaxis(rgb, -1, 0), photometric="rgb", planarconfig="separate")
    got = read_tif(f)
    assert got.shape == (5, 6, 3) and np.array_equal(got, rgb)
    assert np.array_equal(read_tif(f, page=None), rgb)


# ======================================================================================
# module-level checks
# ======================================================================================


def test_public_api_is_documented_and_exported():
    import calliopy.io._imaging as mod

    readers = [n for n in mod.__all__ if n.startswith(("load_", "read_"))]
    assert sorted(readers) == [
        "load_aux_vid_data",
        "load_bin",
        "load_eye_monitor_data",
        "load_roi_data",
        "load_vid_data",
        "load_wid_data",
        "load_wii_data",
        "read_tif",
    ]
    for name in mod.__all__:
        obj = getattr(mod, name)
        assert obj.__doc__, name
    for name in readers:
        doc = getattr(mod, name).__doc__
        assert "Differences from MATLAB" in doc or name == "read_tif", name
        assert "Parameters" in doc and "Returns" in doc, name
    assert BinData._fields == ("data", "nbr_frames")
    assert WidData._fields == ("wid_data", "wid_meta_info")
    assert RoiData._fields == ("roi_data", "roi_meta_info")
    assert WiiData._fields == ("wii_config", "wii_mean_proj", "frame_nonrotated")
    assert EyeData._fields == ("idata", "imeta_info")
    assert VidData._fields == ("vdata", "vmeta_info")
    assert AuxVidData._fields == ("vid_data", "vid_meta_info")


# ======================================================================================
# Part 2: comparison with the MATLAB r2003 readers (skip until test data exist)
# ======================================================================================


def _same_array(got, want, what):
    got = np.asarray(got)
    want = np.asarray(want)
    assert got.shape == want.shape, f"{what}: shape {got.shape} != MATLAB {want.shape}"
    assert np.array_equal(got, want.astype(got.dtype)), f"{what}: values differ"


def _ref_for(sample, ref):
    """Skip when the MATLAB reference was made from another file than the sample found here."""
    if PureWindowsPath(str(ref.input_file)).name != sample.name:
        pytest.skip(f"reference was made from {ref.input_file}, sample is {sample.name}")


def _sample_named_in_reference(testdata, ref):
    sample = Path(testdata) / "samples" / PureWindowsPath(str(ref.input_file)).name
    if not sample.is_file():
        pytest.skip(f"{sample.name} is not in the samples folder")
    return sample


def _stamp_text(stamp):
    """'yyyy-MM-dd HH:mm:ss.SSSSSS', the format of the MATLAB datetimes in the meta info."""
    return stamp.strftime("%Y-%m-%d %H:%M:%S.%f")


def _compare_meta(got, want, fields):
    """Compare a (1, N) object array of MatStruct with MATLAB's struct array."""
    want = [want] if isinstance(want, dict) else list(np.ravel(want))
    got = list(np.ravel(got))
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert sorted(g.keys()) == sorted(w.keys())
        for field in fields:
            if field == "timestamp" and isinstance(g.timestamp, datetime):
                assert _stamp_text(g.timestamp) == str(w.timestamp)  # ref_imaging.m stores text
            else:
                assert np.array_equal(np.ravel(g[field]), np.ravel(w[field])), field


def _compare_config(got, want, path="wii_config"):
    if isinstance(want, dict):
        assert isinstance(got, dict), path
        assert sorted(got.keys()) == sorted(want.keys()), path
        for key in want:
            _compare_config(got[key], want[key], f"{path}.{key}")
    elif isinstance(want, np.ndarray) and want.dtype == object:
        assert got.shape == want.shape, path
        for idx in np.ndindex(want.shape):
            _compare_config(got[idx], want[idx], f"{path}{list(idx)}")
    elif isinstance(want, str):
        assert got == want, path
    else:
        want = np.asarray(want)
        got = np.asarray(got)
        assert got.shape == want.shape or (got.size == want.size == 1), path
        assert np.array_equal(np.ravel(got), np.ravel(want)), path


def test_ref_load_bin(sample_file, matlab_ref):
    sample, ref = sample_file(".bin"), matlab_ref("load_bin")
    _ref_for(sample, ref)
    n_lay = int(ref.out3.nbr_piezo_layers)
    shift = -1 if "S1-T231419" in str(ref.input_file) else 0  # MATLAB's file-name hack
    got = load_bin(sample, n_lay, layer_shift=shift)
    want = np.atleast_2d(ref.out1)
    assert len(got.data) == want.shape[1]
    for k, layer in enumerate(got.data):
        _same_array(layer, want[0, k], f"data[{k}]")
        assert layer.dtype == np.int16
    assert np.array_equal(got.nbr_frames.ravel(), np.ravel(ref.out2))
    del got, layer
    release()


def test_ref_load_wid_data(sample_file, matlab_ref):
    sample, ref = sample_file(".wid"), matlab_ref("load_wid_data")
    _ref_for(sample, ref)
    got = load_wid_data(sample)
    _same_array(got.wid_data, ref.out1, "wid_data")
    assert got.wid_data.dtype == np.uint16
    assert isinstance(got.wid_meta_info, MatStruct) and len(got.wid_meta_info) == 0
    del got
    release()


def test_ref_load_wid_data_firstlast(sample_file, matlab_ref):
    sample, ref = sample_file(".wid"), matlab_ref("load_wid_data_firstlast")
    _ref_for(sample, ref)
    _same_array(load_wid_data(sample, frames="firstlast").wid_data, ref.out1, "firstlast")
    release()


def test_ref_load_wid_data_meta(sample_file, matlab_ref):
    sample, ref = sample_file(".wid"), matlab_ref("load_wid_data_meta")
    _ref_for(sample, ref)
    got = load_wid_data(sample, frames="meta")
    _compare_meta(got.wid_meta_info, ref.out1, ("framenumber", "timestamp", "dimension", "file_id"))


def test_ref_load_roi_data(sample_file, matlab_ref):
    sample, ref = sample_file(".roi"), matlab_ref("load_roi_data")
    _ref_for(sample, ref)
    got = load_roi_data(sample)
    _same_array(got.roi_data, ref.out1, "roi_data")
    assert got.roi_data.dtype == np.uint16
    del got
    release()


def test_ref_load_roi_data_meta(sample_file, matlab_ref):
    sample, ref = sample_file(".roi"), matlab_ref("load_roi_data_meta")
    _ref_for(sample, ref)
    got = load_roi_data(sample, frames="meta")
    _compare_meta(got.roi_meta_info, ref.out1, ("framenumber", "timestamp", "dimension", "file_id"))


def test_ref_extract_wii_file_platform(sample_file, matlab_ref):
    sample, ref = sample_file(".wii"), matlab_ref("extract_wii_file_platform")
    _ref_for(sample, ref)
    got = load_wii_data(sample)
    _compare_config(got.wii_config, ref.out1)
    _same_array(got.wii_mean_proj, ref.out2, "wii_mean_proj")
    _same_array(got.frame_nonrotated, ref.out3, "frame_nonrotated")
    assert got.wii_mean_proj.dtype == np.uint16


def test_ref_load_eye_monitor_data(sample_file, matlab_ref):
    sample, ref = sample_file(".eye"), matlab_ref("load_eye_monitor_data")
    _ref_for(sample, ref)
    got = load_eye_monitor_data(sample, True)
    _same_array(got.idata, ref.out1, "idata")
    _same_array(got.imeta_info, ref.out2, "imeta_info")
    assert got.idata.dtype == np.uint8
    del got
    release()


def test_ref_load_eye_monitor_data_no_image(sample_file, matlab_ref):
    sample, ref = sample_file(".eye"), matlab_ref("load_eye_monitor_data_noimg")
    _ref_for(sample, ref)
    got = load_eye_monitor_data(sample, False)
    assert got.idata.size == 0 and np.asarray(ref.out1).size == 0
    _same_array(got.imeta_info, ref.out2, "imeta_info")


def test_ref_load_vid_data(testdata, matlab_ref):
    ref = matlab_ref("load_vid_data")
    sample = _sample_named_in_reference(testdata, ref)  # .run, .vid, .vid2 or .ton
    got = load_vid_data(sample, "int16")
    _same_array(got.vdata, ref.out1, "vdata")
    _same_array(got.vmeta_info, ref.out2, "vmeta_info")
    assert got.vdata.dtype == np.int16
    del got
    release()


def test_ref_load_aux_vid_data(testdata, matlab_ref):
    ref = matlab_ref("load_auxVid_data")
    sample = _sample_named_in_reference(testdata, ref)
    bit_depth = str(ref.out3.bit_depth)
    got = load_aux_vid_data(sample, bit_depth=bit_depth)
    _same_array(got.vid_data, ref.out1, "vid_data")
    assert got.vid_data.dtype == np.dtype(bit_depth)
    del got
    release()


def test_ref_load_aux_vid_data_meta(testdata, matlab_ref):
    ref = matlab_ref("load_auxVid_data_meta")
    sample = _sample_named_in_reference(testdata, ref)
    got = load_aux_vid_data(sample, frames="meta", bit_depth=str(ref.out2.bit_depth))
    _compare_meta(got.vid_meta_info, ref.out1, ("timestamp", "dimension", "file_id"))


def test_ref_tif(sample_file, matlab_ref):
    pytest.importorskip("tifffile")
    sample, ref = sample_file(".tif"), matlab_ref("tif")
    _ref_for(sample, ref)
    got = read_tif(sample)
    _same_array(got, ref.out1, "tif_data")
