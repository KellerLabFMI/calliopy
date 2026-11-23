"""Tests for calliopy.io._aux: readers of auxiliary recording files.

Part 1 builds small files in the exact layout the MATLAB readers parse and always runs.
Part 2 (``test_reference_*``) compares every reader with the output of the original
MATLAB function; it is skipped until CALLIOPY_TESTDATA holds ``samples/`` and the
``reference/ref_*.mat`` files written by ``tests/matlab/ref_aux.m``.
"""

from __future__ import annotations

import contextlib
import gc
import io
import logging
import struct
import warnings
from datetime import datetime
from pathlib import Path, PureWindowsPath

import numpy as np
import pytest
from calliopy.io import _aux
from calliopy.io._aux import (
    LvdData,
    SidecarCsv,
    VrdData,
    load_ach_chans,
    load_lvd,
    load_wcp,
    parse_ach,
    parse_vrd,
    read_ini,
    read_sidecar_csv,
)
from calliopy.io._mat import MatStruct


# -- helpers -------------------------------------------------------------------------
class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def captured_log():
    """Messages logged by calliopy.io._aux while the block runs (list of strings)."""
    handler = _Records()
    logger = logging.getLogger("calliopy.io._aux")
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    messages = []
    try:
        yield messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        messages.extend(r.getMessage() for r in handler.records)


def write(path: Path, text: str) -> Path:
    """Write text without newline translation (so \r\n stays \r\n on every OS)."""
    path.write_bytes(text.encode("utf-8"))
    return path


def write_lvd(path, data, *, scan_rate=30000.0, timestamp=0.0, input_range=10.0, extra=b""):
    """Write [channels x samples] data the way LabVIEW does, one sample at a time."""
    data = np.asarray(data, dtype=float)
    n_ch, n_samples = data.shape
    out = struct.pack(">4d", scan_rate, n_ch, timestamp, input_range)
    for j in range(n_samples):
        for c in range(n_ch):
            out += struct.pack(">d", data[c, j])
    path.write_bytes(out + extra)
    return path


def channel_data(n_ch=3, n_samples=5):
    """Value 100*channel + sample, so a wrong orientation shows up immediately."""
    return 100.0 * np.arange(n_ch)[:, None] + np.arange(n_samples)[None, :]


def datenum(dt: datetime) -> float:
    """MATLAB datenum of a naive datetime."""
    seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
    return dt.toordinal() + 366 + seconds / 86400


# =============================== read_ini ===============================================
def test_read_ini_tab_delimited(tmp_path):
    text = (
        "[main]\r\nframe rate\t30\r\npiezo.nbrlayers\t5\r\nname\t\"foo bar\"\r\n"
        "gains\t1 -2 3.5\r\n"
    )
    s = read_ini(write(tmp_path / "a.ini", text))
    assert isinstance(s, MatStruct)
    assert list(s) == ["frame_rate", "piezo", "name", "gains"]
    assert s.frame_rate == 30.0 and isinstance(s.frame_rate, np.float64)
    assert isinstance(s.piezo, MatStruct) and s.piezo.nbrlayers == 5.0
    assert s.name == "foo bar"
    assert s.gains.shape == (1, 3) and s.gains.dtype == np.float64
    assert np.array_equal(s.gains, [[1.0, -2.0, 3.5]])


def test_read_ini_equals_delimited(tmp_path):
    text = 'name = "S1-T5"\nrate=20.5\nurl = "a=b=c"\n  padded   =   x y   \n'
    s = read_ini(write(tmp_path / "a.ini", text))
    assert s.name == "S1-T5"
    assert s.rate == 20.5
    assert s.url == "a=b=c"  # MATLAB kept only 'b'
    assert s.padded == "x y"


def test_read_ini_delimiter_is_decided_by_first_data_line(tmp_path):
    s = read_ini(write(tmp_path / "t.ini", "a\t1\nb\tx=y\nc\t5\tignored column\n"))
    assert s.a == 1.0 and s.b == "x=y" and s.c == 5.0
    s = read_ini(write(tmp_path / "e.ini", "[main]\na = 1\nb\t2 = 3\n"))
    assert s["b\t2"] == 3.0  # '=' file: a tab is just part of the tag


@pytest.mark.parametrize(
    "text, expected",
    [
        ("12", 12.0),
        ("-3.5", -3.5),
        ("1e3", 1000.0),
        ("1.5E-3", 0.0015),
        (".5", 0.5),
        ("5.", 5.0),
        ("+4", 4.0),
        ('"123"', 123.0),  # quotes are stripped first, as in MATLAB
    ],
)
def test_read_ini_scalar_numbers(tmp_path, text, expected):
    v = read_ini(write(tmp_path / "a.ini", f"x = {text}\n")).x
    assert isinstance(v, np.float64) and v == expected


def test_read_ini_number_vectors(tmp_path):
    s = read_ini(write(tmp_path / "a.ini", "v = 1  2   3\nw = 1 -2\nz = 1e2 .5 -.5 +1\n"))
    assert s.v.shape == (1, 3) and np.array_equal(s.v, [[1, 2, 3]])
    assert np.array_equal(s.w, [[1, -2]])
    assert np.array_equal(s.z, [[100, 0.5, -0.5, 1]])


@pytest.mark.parametrize(
    "text",
    ["2024-01-05", "1+2", "1 - 2", "1.2.3", "--", "e", "1,2", "12abc", "nan", "0x10", "5 -"],
)
def test_read_ini_values_are_not_evaluated(tmp_path, text):
    # MATLAB evaluated the first four as expressions (2018, 3, -1) or dropped the field.
    assert read_ini(write(tmp_path / "a.ini", f"x = {text}\n")).x == text


def test_read_ini_empty_value_is_empty_array(tmp_path):
    s = read_ini(write(tmp_path / "a.ini", "x =\ny = \"\"\nz =    \n"))
    for v in (s.x, s.y, s.z):
        assert isinstance(v, np.ndarray) and v.shape == (0, 0)
    s = read_ini(write(tmp_path / "t.ini", "x\t\ny\t1\n"))
    assert s.x.shape == (0, 0) and s.y == 1.0


def test_read_ini_dates(tmp_path):
    text = (
        "a = 2024-01-05 12:30:45.123\nb = 2023-12-31_23-59-58\n"
        "c = 2024-01-05 12:30:45\nd = 2024-13-45 25:00:00.000\ne = 2024-1-5_1-2-3\n"
    )
    s = read_ini(write(tmp_path / "a.ini", text))
    assert s.a == datetime(2024, 1, 5, 12, 30, 45, 123000)
    assert s.b == datetime(2023, 12, 31, 23, 59, 58)
    assert isinstance(s.a, datetime) and s.a.tzinfo is None
    assert s.c == "2024-01-05 12:30:45"  # no milliseconds: not one of the two formats
    assert s.d == "2024-13-45 25:00:00.000"  # impossible date stays text
    assert s.e == "2024-1-5_1-2-3"  # digit counts must match


def test_read_ini_tag_rules(tmp_path):
    lines = [
        "foo bar = 1",  # blanks -> _
        "tag.1.test = 2",  # .<digit> -> _<digit>
        "tag.1.other = 3",
        "x11.1 = 4",  # MATLAB's unescaped '.' turned this into _11_1
        "trace [3] = 5",  # cut at _[
        "trace [4] = 6",  # same field again: last value wins, position kept
        "a.b = 7",
        "a.c = 8",
        "foo.12.z = 9",
    ]
    s = read_ini(write(tmp_path / "a.ini", "\n".join(lines) + "\n"))
    assert list(s) == ["foo_bar", "tag_1", "x11_1", "trace", "a", "foo_12"]
    assert s.foo_bar == 1.0
    assert list(s.tag_1) == ["test", "other"] and s.tag_1.test == 2.0
    assert s.x11_1 == 4.0
    assert s.trace == 6.0
    assert list(s.a) == ["b", "c"] and s.a.c == 8.0
    assert s.foo_12.z == 9.0


def test_read_ini_keeps_what_matlab_dropped(tmp_path):
    s = read_ini(write(tmp_path / "a.ini", "frame-rate = 3\nname = it's\n"))
    assert s["frame-rate"] == 3.0
    assert s.name == "it's"


def test_read_ini_empty_file_and_blank_lines(tmp_path):
    assert read_ini(write(tmp_path / "e.ini", "")) == {}
    assert read_ini(write(tmp_path / "m.ini", "[main]\r\n")) == {}
    s = read_ini(write(tmp_path / "c.ini", "[main]\n\n a = 1 \n\n\nb = 2\n\n"))
    assert list(s) == ["a", "b"] and s.b == 2.0


def test_read_ini_errors_name_the_line(tmp_path):
    with pytest.raises(ValueError, match="line 3"):
        read_ini(write(tmp_path / "a.ini", "a = 1\nb = 2\nno delimiter here\n"))
    with pytest.raises(ValueError, match="line 2"):
        read_ini(write(tmp_path / "b.ini", "[main]\n = 5\n"))  # empty tag
    with pytest.raises(ValueError, match="line 2"):
        read_ini(write(tmp_path / "c.ini", "a = 1\na.b = 2\n"))  # value, then struct
    with pytest.raises(ValueError, match="line 2"):
        read_ini(write(tmp_path / "d.ini", "a.b = 2\na = 1\n"))  # struct, then value
    with pytest.raises(ValueError, match="line 2"):
        read_ini(write(tmp_path / "e.ini", "a\t1\nb\n"))  # tab file, line without tab
    with pytest.raises(ValueError, match="empty"):
        read_ini(write(tmp_path / "f.ini", "a..b = 1\n"))
    with pytest.raises(FileNotFoundError):
        read_ini(tmp_path / "missing.ini")


def test_read_ini_empty_array_becomes_struct(tmp_path):
    s = read_ini(write(tmp_path / "a.ini", "a =\na.b = 1\n"))  # MATLAB: [] turns into a struct
    assert s.a.b == 1.0


def test_read_ini_accepts_str_path_and_latin1(tmp_path):
    f = tmp_path / "a.ini"
    f.write_bytes("unit = \xb5m\n".encode("latin-1"))
    assert read_ini(str(f)).unit == "\xb5m"


# =============================== load_lvd ===============================================
def test_load_lvd_values_orientation_and_header(tmp_path):
    data = channel_data(3, 5)
    f = write_lvd(tmp_path / "S1-T200000.lvd", data, scan_rate=25000.0, timestamp=3.75e9,
                  input_range=5.0)
    out = load_lvd(f)
    assert isinstance(out, LvdData)
    assert out.data.shape == (3, 5)  # [channels x samples], as MATLAB
    assert np.array_equal(out.data, data)
    assert out.data[2, 4] == 204.0  # channel 2, sample 4
    assert out.scan_rate == 25000.0 and isinstance(out.scan_rate, float)
    assert out.num_channels == 3 and isinstance(out.num_channels, int)
    assert out.timestamp == 3.75e9 and out.input_range == 5.0


def test_load_lvd_is_native_float64(tmp_path):
    out = load_lvd(write_lvd(tmp_path / "a.lvd", channel_data(2, 4)))
    assert out.data.dtype == np.float64 and out.data.dtype.isnative
    assert not isinstance(out.data, np.memmap)
    assert out.data.flags.writeable


def test_load_lvd_namedtuple_fields_and_unpacking(tmp_path):
    out = load_lvd(write_lvd(tmp_path / "a.lvd", channel_data(2, 3), scan_rate=100.0))
    assert out._fields == ("data", "scan_rate", "num_channels", "timestamp", "input_range")
    data, scan_rate, *rest = out
    assert scan_rate == 100.0 and len(rest) == 3 and data is out.data


def test_load_lvd_roundtrip_random_with_nan_and_negative(tmp_path):
    rng = np.random.default_rng(1)
    data = rng.normal(size=(4, 1000)) * 1e3
    data[1, 10] = np.nan
    data[2, 20] = -0.0
    out = load_lvd(str(write_lvd(tmp_path / "a.lvd", data)))
    assert np.array_equal(out.data, data, equal_nan=True)
    assert np.signbit(out.data[2, 20])


def test_load_lvd_single_channel_and_header_only(tmp_path):
    one = load_lvd(write_lvd(tmp_path / "one.lvd", [[1.0, 2.0, 3.0]]))
    assert one.data.shape == (1, 3)
    none = load_lvd(write_lvd(tmp_path / "none.lvd", np.empty((4, 0))))
    assert none.data.shape == (4, 0) and none.num_channels == 4


def test_load_lvd_incomplete_last_sample_is_dropped_with_warning(tmp_path):
    data = channel_data(3, 4)
    f = write_lvd(tmp_path / "a.lvd", data, extra=struct.pack(">2d", 7.0, 8.0))
    with pytest.warns(RuntimeWarning, match="2 value"):
        out = load_lvd(f)
    assert out.data.shape == (3, 4) and np.array_equal(out.data, data)  # MATLAB padded a 5th column
    f = write_lvd(tmp_path / "b.lvd", data, extra=b"\x01\x02\x03")  # stray bytes < 1 double
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert load_lvd(f).data.shape == (3, 4)


def test_load_lvd_header_errors(tmp_path):
    short = tmp_path / "short.lvd"
    short.write_bytes(b"\x00" * 31)
    with pytest.raises(ValueError, match="header"):
        load_lvd(short)
    with pytest.raises(FileNotFoundError):
        load_lvd(tmp_path / "missing.lvd")


@pytest.mark.parametrize("n_channels", [0.0, -2.0, 2.5, float("nan"), float("inf")])
def test_load_lvd_invalid_channel_count(tmp_path, n_channels):
    f = tmp_path / "bad.lvd"
    f.write_bytes(struct.pack(">4d", 1000.0, n_channels, 0.0, 1.0) + struct.pack(">4d", 1, 2, 3, 4))
    with pytest.raises(ValueError, match="channels"):
        load_lvd(f)


def test_load_lvd_mmap(tmp_path):
    data = channel_data(3, 6)
    f = write_lvd(tmp_path / "a.lvd", data)
    out = load_lvd(f, mmap=True)
    assert isinstance(out.data, np.memmap)
    assert out.data.shape == (3, 6) and not out.data.flags.writeable
    assert np.array_equal(out.data, data)
    assert out.data.dtype.str == ">f8"  # on-disk byte order, documented
    del out
    gc.collect()  # release the file (Windows cannot delete mapped files)


def test_load_lvd_get_file_size_helper(tmp_path):
    f = write_lvd(tmp_path / "a.lvd", channel_data(2, 3))
    assert _aux._get_file_size(f) == 32 + 2 * 3 * 8


# =============================== load_wcp ===============================================
def test_load_wcp_new_format_is_lvd(tmp_path):
    data = channel_data(4, 6)
    f = write_lvd(tmp_path / "S1-T200000.wcp", data, scan_rate=10000.0, timestamp=5.0)
    out = load_wcp(f)
    ref = load_lvd(f)
    assert isinstance(out, LvdData)
    assert np.array_equal(out.data, data) and out.num_channels == 4
    assert out[1:] == ref[1:] and out.scan_rate == 10000.0 and out.timestamp == 5.0


def test_load_wcp_old_format_is_headerless_three_channels(tmp_path):
    data = channel_data(3, 7)
    f = tmp_path / "S1-T100000.wcp"
    f.write_bytes(b"".join(struct.pack(">d", data[c, j]) for j in range(7) for c in range(3)))
    out = load_wcp(f)
    assert out.data.shape == (3, 7) and np.array_equal(out.data, data)
    header = (out.scan_rate, out.num_channels, out.timestamp, out.input_range)
    assert header == (20000.0, 3, 0.0, 0.0)
    assert out.data.dtype == np.float64 and out.data.dtype.isnative


@pytest.mark.parametrize("exp_id, is_new", [(133300, False), (133301, True), (1, False)])
def test_load_wcp_switches_format_after_exp_id_133300(tmp_path, exp_id, is_new):
    data = channel_data(3, 4)
    f = tmp_path / f"S1-T{exp_id}.wcp"
    if is_new:
        write_lvd(f, data, scan_rate=111.0)
    else:
        f.write_bytes(data.T.astype(">f8").tobytes())
    out = load_wcp(f)
    assert np.array_equal(out.data, data)
    assert out.scan_rate == (111.0 if is_new else 20000.0)


def test_load_wcp_exp_id_is_taken_from_the_file_name_only(tmp_path):
    folder = tmp_path / "S1-T999999"  # MATLAB took the first 'S1-T' of the whole path
    folder.mkdir()
    data = channel_data(3, 3)
    f = folder / "S1-T100000.wcp"
    f.write_bytes(data.T.astype(">f8").tobytes())
    assert np.array_equal(load_wcp(f).data, data)


def test_load_wcp_name_without_exp_id(tmp_path):
    data = channel_data(3, 3)
    f = tmp_path / "recording.wcp"
    f.write_bytes(data.T.astype(">f8").tobytes())
    with pytest.raises(ValueError, match="ExpID"):
        load_wcp(f)
    assert np.array_equal(load_wcp(f, exp_id=5).data, data)  # old, headerless
    g = write_lvd(tmp_path / "other.wcp", data)
    assert np.array_equal(load_wcp(g, exp_id=200000).data, data)
    with pytest.raises(FileNotFoundError):
        load_wcp(tmp_path / "S1-T1.wcp")


def test_load_wcp_old_format_truncated_and_mmap(tmp_path):
    data = channel_data(3, 4)
    f = tmp_path / "S1-T5.wcp"
    f.write_bytes(data.T.astype(">f8").tobytes() + struct.pack(">d", 9.0))
    with pytest.warns(RuntimeWarning, match="1 value"):
        out = load_wcp(f)
    assert out.data.shape == (3, 4)
    with pytest.warns(RuntimeWarning, match="1 value"):
        mm = load_wcp(f, mmap=True)
    assert isinstance(mm.data, np.memmap) and np.array_equal(mm.data, data)
    del mm
    gc.collect()


# ============================ load_ach_chans / parse_ach ==================================
ACH = (
    "[main]\r\n"
    'auxrec.scanrate = "20000"\r\n'
    'auxrec.channel0 = "Running"\r\n'
    'auxrec.channel1 = "Perturbation"\r\n'
    'auxrec.channel2 = "EEG sync"\r\n'
    "auxrec.gain = 2.5\r\n"
    'other.name = "x"\r\n'
)


def test_load_ach_chans_returns_labels_as_written(tmp_path):
    chans = load_ach_chans(write(tmp_path / "S1-T5.ach", ACH))
    assert chans == ["Running", "Perturbation", "EEG sync"]  # parse_ach renames, this does not
    assert all(isinstance(c, str) for c in chans)


def test_load_ach_chans_skips_first_line_always(tmp_path):
    f = write(tmp_path / "a.ach", 'auxrec.channel0 = "A"\nauxrec.channel1 = "B"\n')
    assert load_ach_chans(f) == []  # channel0 is on the (header) line, as in MATLAB
    f = write(tmp_path / "b.ach", '[main]\nauxrec.channel0 = "A"\nauxrec.channel1 = "B"\n')
    assert load_ach_chans(f) == ["A", "B"]


def test_load_ach_chans_short_and_blank_lines_are_skipped(tmp_path):
    text = '[main]\n\nx\nauxrec.channel0 = "A"\n\nshort\nauxrec.channel1 = "B"\n\n\n'
    assert load_ach_chans(write(tmp_path / "a.ach", text)) == ["A", "B"]  # MATLAB: index error


def test_load_ach_chans_needs_channels_in_order(tmp_path):
    text = '[main]\nauxrec.channel0 = "A"\nauxrec.channel2 = "C"\nauxrec.channel1 = "B"\n'
    assert load_ach_chans(write(tmp_path / "a.ach", text)) == ["A", "B"]
    text = '[main]\nauxrec.channel0 = "A"\nauxrec.channel2 = "C"\n'
    assert load_ach_chans(write(tmp_path / "b.ach", text)) == ["A"]  # stops at the gap
    assert load_ach_chans(write(tmp_path / "c.ach", "[main]\nauxrec.scanrate = \"1\"\n")) == []


def test_load_ach_chans_empty_and_missing_labels(tmp_path):
    text = '[main]\nauxrec.channel0 = ""\nauxrec.channel1 = "B"\n'
    assert load_ach_chans(write(tmp_path / "a.ach", text)) == ["", "B"]
    text = "[main]\nauxrec.channel0 = A\n"
    with pytest.raises(ValueError, match="line 2"):
        load_ach_chans(write(tmp_path / "b.ach", text))
    with pytest.raises(FileNotFoundError):
        load_ach_chans(tmp_path / "missing.ach")


def test_parse_ach_fields_and_channel_table(tmp_path):
    pdef = parse_ach(write(tmp_path / "S1-T5.ach", ACH))
    assert isinstance(pdef, MatStruct)
    assert list(pdef) == ["auxrec", "other", "aux_chans"]
    assert pdef.auxrec.scanrate == "20000"  # a quoted value stays text (load_exp: str2double)
    assert pdef.auxrec.gain == 2.5 and isinstance(pdef.auxrec.gain, np.float64)
    assert pdef.other.name == "x"
    ch = pdef.aux_chans
    assert ch.dtype == object and ch.shape == (3, 2)  # 'scanrate' and 'gain' have no digit
    assert list(ch[:, 1]) == ["velM", "PS", "EEG sync"]  # Running -> velM, Perturbation -> PS
    assert list(ch[:, 0]) == [1, 2, 3]  # 1-based: channel0 is channel 1
    assert all(isinstance(n, int) for n in ch[:, 0])


def test_parse_ach_renames_labels(tmp_path):
    lines = ["[main]"] + [
        f'auxrec.channel{i} = "{name}"'
        for i, name in enumerate(
            ["Running", "VisualFlow", "PerturbationLFM", "Perturbation", "EEGsync", "RunningWheel"]
        )
    ]
    pdef = parse_ach(write(tmp_path / "a.ach", "\n".join(lines) + "\n"))
    assert list(pdef.aux_chans[:, 1]) == ["velM", "velP", "PS", "PS", "EEGsync", "velMWheel"]
    assert list(pdef.aux_chans[:, 0]) == [1, 2, 3, 4, 5, 6]
    assert pdef.auxrec.channel0 == "velM"


def test_parse_ach_value_types(tmp_path):
    text = (
        "[main]\nauxrec.scanrate = \"20000\"\nauxrec.channel0 = 'A'\nrig.gain = 2.5\n"
        "rig.range = [1 2 3]\nrig.matrix = [1 2; 3 4]\nrig.list = [1, 2,3]\nrig.flag = true\n"
        "rig.none = []\nrig.neg = -1e-3;\nrig.quote = \"it's\"\n"
    )
    rig = parse_ach(write(tmp_path / "a.ach", text)).rig
    assert rig.gain == 2.5 and isinstance(rig.gain, np.float64)
    assert rig.range.shape == (1, 3) and np.array_equal(rig.range, [[1, 2, 3]])
    assert np.array_equal(rig.matrix, [[1, 2], [3, 4]])
    assert np.array_equal(rig.list, [[1, 2, 3]])
    assert rig.flag == np.True_ and rig.none.shape == (0, 0)
    assert rig.neg == -1e-3
    assert rig.quote == "it's"  # MATLAB: eval error


def test_parse_ach_skips_unreadable_lines_with_a_warning(tmp_path):
    text = (
        "[main]\nauxrec.channel0 = \"A\"\n[a section]\nauxrec.bad = {1, 2}\nauxrec.worse = foo\n"
        "1bad.name = 3\n\nauxrec.channel1 = \"B\"\nx = 1 2\n"
    )
    with captured_log() as messages:
        pdef = parse_ach(write(tmp_path / "a.ach", text))
    assert list(pdef.aux_chans[:, 1]) == ["A", "B"]
    assert list(pdef.auxrec) == ["channel0", "channel1"]
    assert "x" not in pdef
    skipped = [m for m in messages if "skipped" in m]
    assert len(skipped) == 5  # the blank line is not reported
    assert any("line 3" in m and "[a section]" in m for m in skipped)


def test_parse_ach_without_auxrec_lines(tmp_path):
    pdef = parse_ach(write(tmp_path / "a.ach", "[main]\nfoo = 1\n"))
    assert list(pdef) == ["auxrec", "foo", "aux_chans"]
    assert pdef.auxrec.shape == (0, 0)  # as MATLAB's pdef.auxrec = []
    assert pdef.aux_chans.shape == (0, 0) and pdef.aux_chans.dtype == object
    assert parse_ach(write(tmp_path / "b.ach", "")).aux_chans.shape == (0, 0)


def test_parse_ach_channel_number_from_digits_in_field_name(tmp_path):
    text = '[main]\nauxrec.channel12 = "A"\nauxrec.gain5 = 2\nauxrec.ch1x2 = "C"\n'
    text += 'auxrec.name = "n"\n'
    ch = parse_ach(write(tmp_path / "a.ach", text)).aux_chans
    assert ch.shape == (3, 2)
    assert ch[0, 0] == 13 and ch[0, 1] == "A"
    assert ch[1, 0] == 6 and ch[1, 1] == 2.0
    assert np.array_equal(ch[2, 0], [2, 3])  # two digit groups, as MATLAB's vector


def test_parse_ach_struct_conflicts_follow_matlab(tmp_path):
    text = "[main]\na = 1\na.b = 2\nc.d = 1\nc = 5\nauxrec.channel0 = \"X\"\n"
    with captured_log() as messages:
        pdef = parse_ach(write(tmp_path / "a.ach", text))
    assert pdef.a == 1.0  # MATLAB: assigning into a number fails, the line is skipped
    assert pdef.c == 5.0  # MATLAB: a value replaces a struct
    assert len(messages) == 1 and "line 3" in messages[0]


def test_parse_ach_accepts_lf_and_missing_file(tmp_path):
    crlf = parse_ach(write(tmp_path / "a.ach", ACH))
    lf = parse_ach(write(tmp_path / "b.ach", ACH.replace("\r\n", "\n")))
    assert_same(lf, crlf)
    with pytest.raises(FileNotFoundError):
        parse_ach(tmp_path / "missing.ach")


# =============================== parse_vrd ==============================================
VRD = (
    "[05-Mar-19 14:22:10]\r\n"
    "read(0) = 05-Mar-2019 14:22:11\r\n"
    "read(1) = 05-Mar-2019 14:22:12\r\n"
    "read(2) = 05-Mar-2019 14:22:13\r\n"
    "traversal_time_in_s(1) = 1.5\r\n"
    "traversal_time_in_s(2) = 2.5\r\n"
    "speed = 3.2\r\n"
    "mode = forward (fast)\r\n"
    "m_scale = [1 2]\r\n"
    "\r\n"
    "[05-Mar-19 14:25:10]\r\n"
    "read(1) = 05-Mar-2019 14:25:12\r\n"
    "traversal_time_in_s(1) = 4\r\n"
    "speed = 3.4\r\n"
    "mode = back\r\n"
    "m_scale = [3 4]\r\n"
)


def test_parse_vrd_sections(tmp_path):
    out = parse_vrd(write(tmp_path / "S1-T5.vrd", VRD))
    assert isinstance(out, VrdData) and out._fields == ("sections", "raw_txt", "vrd_processed")
    assert out.raw_txt == VRD
    secs = out.sections
    assert secs.dtype == object and secs.shape == (1, 2)  # 1xN cell of structs
    s1, s2 = secs[0, 0], secs[0, 1]
    assert isinstance(s1, MatStruct)
    assert list(s1) == [
        "header", "readfirst", "read", "traversal_time_in_s", "speed", "mode", "m_scale",
    ]
    assert s1.header == "[05-Mar-19 14:22:10]"
    assert s1.readfirst == "05-Mar-2019 14:22:11"  # read(0)
    assert s1.read.shape == (1, 2) and s1.read.dtype == object
    assert list(s1.read[0]) == ["05-Mar-2019 14:22:12", "05-Mar-2019 14:22:13"]
    assert list(s1.traversal_time_in_s[0]) == ["1.5", "2.5"]  # values stay strings
    assert s1.speed == "3.2"
    assert s1.mode == "forward (fast)"  # MATLAB changed the parentheses to braces
    assert s1.m_scale == "[1 2]"  # brackets in a key=value line: not a header
    assert s2.header == "[05-Mar-19 14:25:10]" and s2.read.shape == (1, 1)
    assert s2.mode == "back"


def test_parse_vrd_processed(tmp_path):
    out = parse_vrd(write(tmp_path / "S1-T5.vrd", VRD))
    vrd = out.vrd_processed
    assert isinstance(vrd, MatStruct)
    # first section's data minus expanded fields; rd is added last
    assert list(vrd) == ["header", "readfirst", "m_scale", "rd"]
    assert vrd.header == "[05-Mar-19 14:22:10]" and vrd.m_scale == "[1 2]"
    rd = vrd.rd
    assert list(rd) == ["read", "traversal_time_in_s", "mode", "speed"]  # then sorted names
    assert rd.read.shape == (1, 3) and rd.read.dtype == object
    assert list(rd.read[0]) == [
        datetime(2019, 3, 5, 14, 22, 12),
        datetime(2019, 3, 5, 14, 22, 13),
        datetime(2019, 3, 5, 14, 25, 12),
    ]
    assert rd.traversal_time_in_s.shape == (1, 3)
    assert np.array_equal(rd.traversal_time_in_s, [[1.5, 2.5, 4.0]])
    assert np.array_equal(rd.speed, [[3.2, 3.2, 3.4]])  # per-section value, once per read
    assert rd.speed.dtype == np.float64
    assert rd.mode.shape == (1, 3)
    assert list(rd.mode[0]) == ["forward (fast)", "forward (fast)", "back"]


def test_parse_vrd_single_section_expands_its_own_fields(tmp_path):
    text = "[05-Mar-19 14:22:10]\nread(0) = 05-Mar-2019 14:22:11\nread(1) = 05-Mar-2019 14:22:12\n"
    text += "read(2) = 05-Mar-2019 14:22:13\ntraversal_time_in_s(1) = 1\nspeed = -2.5\n"
    vrd = parse_vrd(write(tmp_path / "a.vrd", text)).vrd_processed
    assert list(vrd) == ["header", "rd"]
    assert list(vrd.rd) == ["read", "traversal_time_in_s", "readfirst", "speed"]
    assert list(vrd.rd.readfirst[0]) == ["05-Mar-2019 14:22:11"] * 2
    assert np.array_equal(vrd.rd.speed, [[-2.5, -2.5]])


def test_parse_vrd_gaps_in_indices_are_empty_entries(tmp_path):
    out = parse_vrd(write(tmp_path / "a.vrd", "[h]\nread(3) = x\nread(1) = y\n"), processed=False)
    read = out.sections[0, 0].read
    assert read.shape == (1, 3)
    assert read[0, 0] == "y" and read[0, 2] == "x"
    assert isinstance(read[0, 1], np.ndarray) and read[0, 1].size == 0
    assert out.vrd_processed is None


def test_parse_vrd_plain_key_before_first_header_and_dotted_keys(tmp_path):
    text = "version = 2\n[h1]\na.b = 1\na.c(2) = 3\n[h2]\nx = 4\n"
    secs = parse_vrd(write(tmp_path / "a.vrd", text), processed=False).sections
    assert secs.shape == (1, 2)
    assert list(secs[0, 0]) == ["version", "header", "a"]  # lines above and under h1: section 1
    assert secs[0, 0].header == "[h1]" and secs[0, 0].a.b == "1"
    assert secs[0, 0].a.c.shape == (1, 2) and secs[0, 0].a.c[0, 1] == "3"
    assert secs[0, 1].x == "4" and secs[0, 1].header == "[h2]"


def test_parse_vrd_values_are_kept_as_written(tmp_path):
    text = "[h]\nurl = a=b=c\nnote = f(x) and read(0)\nempty =\nspaced   =   v w  \n"
    s = parse_vrd(write(tmp_path / "a.vrd", text), processed=False).sections[0, 0]
    assert s.url == "a=b=c"  # MATLAB: 'b'
    assert s.note == "f(x) and read(0)"  # MATLAB: 'f{x} and readfirst'
    assert s.empty == "" and s.spaced == "v w"


def test_parse_vrd_lf_only_file_gives_the_same_result(tmp_path):
    crlf = parse_vrd(write(tmp_path / "a.vrd", VRD))
    lf = parse_vrd(write(tmp_path / "b.vrd", VRD.replace("\r\n", "\n")))
    assert_same(lf.sections, crlf.sections)
    assert_same(lf.vrd_processed, crlf.vrd_processed)
    assert "\r" not in lf.raw_txt


def test_parse_vrd_plain_read_key_counts_as_one_entry(tmp_path):
    text = "[h]\nread = 05-Mar-2019 14:22:12\nspeed = 1\n"
    rd = parse_vrd(write(tmp_path / "a.vrd", text)).vrd_processed.rd
    assert list(rd.read[0]) == [datetime(2019, 3, 5, 14, 22, 12)]
    assert np.array_equal(rd.speed, [[1.0]])


def test_parse_vrd_non_numbers_are_nan(tmp_path):
    text = "[h]\nread(1) = 2019-03-05 14:22:12\nread(2) = 2019-03-05 14:22:13\n"
    text += "traversal_time_in_s(1) = n/a\ntraversal_time_in_s(2) = 2\nspeed =\n"
    rd = parse_vrd(write(tmp_path / "a.vrd", text)).vrd_processed.rd
    assert np.isnan(rd.traversal_time_in_s[0, 0]) and rd.traversal_time_in_s[0, 1] == 2.0
    assert rd.read[0, 1] == datetime(2019, 3, 5, 14, 22, 13)
    assert np.isnan(rd.speed).all() and rd.speed.shape == (1, 2)  # '' counts as a number


def test_parse_vrd_missing_read_in_a_section_is_logged(tmp_path):
    text = VRD.replace("read(1) = 05-Mar-2019 14:25:12\r\n", "")
    with captured_log() as messages:
        vrd = parse_vrd(write(tmp_path / "a.vrd", text)).vrd_processed
    assert any("'read'" in m and "1 of 2" in m for m in messages)
    assert vrd.rd.read.shape == (1, 2) and vrd.rd.speed.shape == (1, 2)  # only section 1 has reads


def test_parse_vrd_processing_errors(tmp_path):
    missing = VRD.replace("mode = forward (fast)\r\n", "")  # section 2 has 'mode', section 1 not
    with pytest.raises(ValueError, match="mode"):
        parse_vrd(write(tmp_path / "a.vrd", missing))
    assert parse_vrd(tmp_path / "a.vrd", processed=False).vrd_processed is None
    odd = "[h]\nread(1) = 5/3/19 14:22\nspeed = 1\n"
    with pytest.raises(ValueError, match="datetime_format"):
        parse_vrd(write(tmp_path / "b.vrd", odd))
    rd = parse_vrd(tmp_path / "b.vrd", datetime_format="%d/%m/%y %H:%M").vrd_processed.rd
    assert list(rd.read[0]) == [datetime(2019, 3, 5, 14, 22)]


@pytest.mark.parametrize(
    "line",
    ["bad key = 1", "x(0) = 1", "= 5", "x(1,2) = 1", "a..b = 1", "1x = 2", "x(-1) = 3"],
)
def test_parse_vrd_keys_that_cannot_be_fields_raise(tmp_path, line):
    with pytest.raises(ValueError, match="line 3"):
        parse_vrd(write(tmp_path / "a.vrd", f"[h]\nok = 1\n{line}\n"), processed=False)


def test_parse_vrd_clashing_keys_raise(tmp_path):
    with pytest.raises(ValueError, match="line 3.*cell"):
        parse_vrd(write(tmp_path / "a.vrd", "[h]\nx = 1\nx(2) = 2\n"), processed=False)
    with pytest.raises(ValueError, match="line 3.*struct"):
        parse_vrd(write(tmp_path / "b.vrd", "[h]\nx = 1\nx.y = 2\n"), processed=False)


def test_parse_vrd_empty_and_header_only(tmp_path):
    empty = parse_vrd(write(tmp_path / "e.vrd", ""), processed=False)
    assert empty.sections.shape == (0, 0) and empty.raw_txt == ""
    with pytest.raises(ValueError, match="no sections"):
        parse_vrd(tmp_path / "e.vrd")
    only = parse_vrd(write(tmp_path / "h.vrd", "[h]\r\n"))
    assert only.sections.shape == (1, 1) and list(only.sections[0, 0]) == ["header"]
    assert only.vrd_processed.rd.read.size == 0
    assert only.vrd_processed.rd.traversal_time_in_s.shape == (1, 0)
    with pytest.raises(FileNotFoundError):
        parse_vrd(tmp_path / "missing.vrd")


LEGACY = (
    "Session header\r\nRead(0) = a\r\nSpeed = 5\r\n\r\n\r\n"
    "Read(1) = b\r\nName = c=d\r\nbad key = 1\r\n\r\n"
)


def test_parse_vrd_legacy(tmp_path):
    with captured_log() as messages:
        out = parse_vrd(write(tmp_path / "old.vrd", LEGACY), legacy=True)
    assert out.vrd_processed is None  # not available in legacy mode
    assert out.raw_txt == LEGACY
    assert out.sections.shape == (1, 2)
    first, second = out.sections[0]
    assert list(first) == ["readfirst", "speed"]  # keys lower-cased, (0) -> first
    assert first.readfirst == "a" and first.speed == "5"
    assert second.read.shape == (1, 1) and second.read[0, 0] == "b"
    assert second.name == "c=d"  # MATLAB: 'c'
    assert len(messages) == 1 and "bad key" in messages[0]


# ============================= read_sidecar_csv =========================================
def test_read_sidecar_csv_basic(tmp_path):
    text = "Time Stamp,FrameNumber,Roll,Pitch,Yaw\r\n0.1,1,0,0,0\r\n0.2,2,1,,3\r\n\r\n"
    out = read_sidecar_csv(write(tmp_path / "S1-T5_min.csv", text))
    assert isinstance(out, SidecarCsv) and out._fields == ("names", "values")
    assert out.names == ["TimeStamp", "FrameNumber", "Roll", "Pitch", "Yaw"]
    assert out.values.shape == (2, 5) and out.values.dtype == np.float64  # [rows x columns]
    assert np.array_equal(out.values[:, out.names.index("FrameNumber")], [1.0, 2.0])
    assert out.values[1, 0] == 0.2 and out.values[1, 4] == 3.0
    assert np.isnan(out.values[1, 3])  # empty field


def test_read_sidecar_csv_header_detection(tmp_path):
    f = write(tmp_path / "a.csv", "1,2\n3,4\n")
    out = read_sidecar_csv(f)  # first row all numbers: no header, like readtable
    assert out.names == ["Var1", "Var2"] and np.array_equal(out.values, [[1, 2], [3, 4]])
    forced = read_sidecar_csv(f, header=True)
    assert forced.names == ["x1", "x2"] and np.array_equal(forced.values, [[3, 4]])
    f = write(tmp_path / "b.csv", "a,b\n1,2\n")
    assert read_sidecar_csv(f).names == ["a", "b"]
    with pytest.raises(ValueError, match="not a number"):
        read_sidecar_csv(f, header=False)


@pytest.mark.parametrize(
    "raw, valid",
    [
        ("Frame Number", "FrameNumber"),
        ("frame number", "frameNumber"),
        ("Time (ms)", "Time_ms_"),
        ("1st", "x1st"),
        ("_a", "x_a"),
        ("for", "xFor"),
        ("a-b", "a_b"),
        ("  padded  ", "padded"),
        ("ok_name1", "ok_name1"),
        ("a,b", "a_b"),
        ("x" * 70, "x" * 63),
    ],
)
def test_read_sidecar_csv_valid_names(tmp_path, raw, valid):
    f = write(tmp_path / "a.csv", f'"{raw}",other\n1,2\n')
    assert read_sidecar_csv(f).names == [valid, "other"]
    assert read_sidecar_csv(f, variable_names="preserve").names == [raw, "other"]


def test_read_sidecar_csv_duplicate_and_empty_names(tmp_path):
    f = write(tmp_path / "a.csv", "a,a,,a\n1,2,3,4\n")
    assert read_sidecar_csv(f).names == ["a", "a_1", "Var3", "a_2"]
    assert read_sidecar_csv(f, variable_names="preserve").names == ["a", "a", "Var3", "a"]
    with pytest.raises(ValueError, match="variable_names"):
        read_sidecar_csv(f, variable_names="other")


def test_read_sidecar_csv_numbers_and_rows(tmp_path):
    text = "a,b,c\n1e3,-.5,inf\nNaN, 7 ,-Inf\n5\n"
    out = read_sidecar_csv(write(tmp_path / "a.csv", text))
    assert out.values.shape == (3, 3)
    assert out.values[0, 0] == 1000.0 and out.values[0, 1] == -0.5 and out.values[0, 2] == np.inf
    assert np.isnan(out.values[1, 0]) and out.values[1, 1] == 7.0 and out.values[1, 2] == -np.inf
    assert out.values[2, 0] == 5.0 and np.isnan(out.values[2, 1:]).all()  # short row: padded


def test_read_sidecar_csv_errors(tmp_path):
    with pytest.raises(ValueError, match="column 'b'"):
        read_sidecar_csv(write(tmp_path / "a.csv", "a,b\n1,2\n3,text\n"))
    with pytest.raises(ValueError, match="data row 3"):
        read_sidecar_csv(write(tmp_path / "b.csv", "a,b\n1,2\n3,4,5\n"))
    with pytest.raises(ValueError, match="empty"):
        read_sidecar_csv(write(tmp_path / "c.csv", "\n\n"))
    with pytest.raises(FileNotFoundError):
        read_sidecar_csv(tmp_path / "missing.csv")


def test_read_sidecar_csv_header_only_bom_delimiter(tmp_path):
    out = read_sidecar_csv(write(tmp_path / "a.csv", "a,b\n"))
    assert out.names == ["a", "b"] and out.values.shape == (0, 2)
    f = tmp_path / "bom.csv"
    f.write_bytes("\ufeffa;b\r\n1.5;2\r\n".encode("utf-8"))
    out = read_sidecar_csv(f, delimiter=";")
    assert out.names == ["a", "b"] and np.array_equal(out.values, [[1.5, 2.0]])
    with pytest.raises(ValueError, match="not a number"):
        read_sidecar_csv(f)  # comma delimiter: one column named 'a;b' holding text


def test_read_sidecar_csv_many_rows(tmp_path):
    rng = np.random.default_rng(0)
    values = rng.normal(size=(20000, 4))
    rows = (",".join(repr(float(v)) for v in row) for row in values)
    text = "t,FrameNumber,Roll,Yaw\n" + "\n".join(rows)
    out = read_sidecar_csv(write(tmp_path / "a.csv", text + "\n"))
    assert out.values.shape == (20000, 4) and np.array_equal(out.values, values)


# ============================== module level ============================================
def test_public_names():
    for name in _aux.__all__:
        assert hasattr(_aux, name), name
    assert set(_aux.__all__) >= {
        "read_ini", "load_lvd", "load_wcp", "load_ach_chans", "parse_ach", "parse_vrd",
        "read_sidecar_csv",
    }


def test_readers_print_nothing(tmp_path):
    files = {
        "a.ini": "[main]\na = 1\n",
        "S1-T5.ach": ACH + "garbage line\n",
        "S1-T5.vrd": VRD,
        "a.csv": "a,b\n1,2\n",
    }
    for name, text in files.items():
        write(tmp_path / name, text)
    write_lvd(tmp_path / "S1-T9.lvd", channel_data(2, 3))
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        read_ini(tmp_path / "a.ini")
        load_ach_chans(tmp_path / "S1-T5.ach")
        parse_ach(tmp_path / "S1-T5.ach")
        parse_vrd(tmp_path / "S1-T5.vrd")
        read_sidecar_csv(tmp_path / "a.csv")
        load_lvd(tmp_path / "S1-T9.lvd")
    assert buffer.getvalue() == ""


# ===================== reference tests against MATLAB (r2003) ==========================
def assert_same(actual, expected, where="value"):
    """Deep comparison. ``expected`` may come from MATLAB: datetimes are then datenums,
    1xN cells are object arrays and numbers are float64."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{where}: expected a struct, got {type(actual)}"
        assert list(actual) == list(expected), f"{where}: fields {list(actual)} != {list(expected)}"
        for key in expected:
            assert_same(actual[key], expected[key], f"{where}.{key}")
    elif isinstance(expected, str):
        assert isinstance(actual, str), f"{where}: expected text, got {type(actual)}"
        assert actual == expected, f"{where}: {actual!r} != {expected!r}"
    elif isinstance(expected, np.ndarray) and expected.dtype == object:
        actual = np.asarray(actual, dtype=object)
        assert actual.shape == expected.shape, f"{where}: shape {actual.shape} != {expected.shape}"
        for idx in np.ndindex(expected.shape):
            assert_same(actual[idx], expected[idx], f"{where}{list(idx)}")
    elif isinstance(actual, datetime) and isinstance(expected, datetime):
        assert actual == expected, f"{where}: {actual} != {expected}"
    elif isinstance(actual, datetime):
        assert abs(datenum(actual) - float(expected)) < 1e-8, f"{where}: {actual} != {expected}"
    else:
        a, e = np.asarray(actual), np.asarray(expected)
        atol = 0.0
        if a.dtype == object and a.size and isinstance(a.flat[0], datetime):
            a = np.array([datenum(x) for x in a.ravel()]).reshape(a.shape)
            atol = 1e-8
        assert a.shape == e.shape, f"{where}: shape {a.shape} != {e.shape}"
        assert np.allclose(a, e, rtol=1e-12, atol=atol, equal_nan=True), f"{where}: {a} != {e}"


def check_input(sample, ref):
    """The MATLAB reference must have been made from the sample file the test reads."""
    name = PureWindowsPath(str(ref.input_file)).name
    assert name == sample.name, f"reference made from {name}, test reads {sample.name}"


def test_reference_read_ini(sample_file, matlab_ref):
    f, ref = sample_file(".ini"), matlab_ref("readini")
    check_input(f, ref)
    assert_same(read_ini(f), ref.out1)


def test_reference_load_lvd(sample_file, matlab_ref):
    f, ref = sample_file(".lvd"), matlab_ref("load_lvd")
    check_input(f, ref)
    out = load_lvd(f)
    assert_same(out.data, ref.out1)
    assert_same(out.scan_rate, ref.out2)
    assert_same(out.num_channels, ref.out3)
    assert_same(out.timestamp, ref.out4)
    assert_same(out.input_range, ref.out5)


def test_reference_load_wcp(sample_file, matlab_ref):
    f, ref = sample_file(".wcp"), matlab_ref("load_wcp")
    check_input(f, ref)
    out = load_wcp(f)
    assert_same(out.data, ref.out1)
    assert_same(out.scan_rate, ref.out2)
    assert_same(out.num_channels, ref.out3)
    assert_same(out.timestamp, ref.out4)
    assert_same(out.input_range, ref.out5)


def test_reference_load_ach_chans(sample_file, matlab_ref):
    f, ref = sample_file(".ach"), matlab_ref("load_ach_chans")
    check_input(f, ref)
    assert load_ach_chans(f) == [str(c) for c in np.ravel(ref.out1)]


def test_reference_parse_ach(sample_file, matlab_ref):
    f, ref = sample_file(".ach"), matlab_ref("parse_ach")
    check_input(f, ref)
    assert_same(parse_ach(f), ref.out1)


def test_reference_parse_vrd(sample_file, matlab_ref):
    # Values containing parentheses differ on purpose (MATLAB turned them into braces).
    f, ref = sample_file(".vrd"), matlab_ref("parse_vrd")
    check_input(f, ref)
    out = parse_vrd(f, processed=False)
    assert_same(out.sections, ref.out1)
    assert out.raw_txt == ref.out2
    if np.size(ref.out3) == 0:
        pytest.skip("MATLAB could not build vrd_processed for this file")
    assert_same(parse_vrd(f).vrd_processed, ref.out3)  # rd.read: datetimes vs datenums


def test_reference_read_sidecar_csv(sample_file, matlab_ref):
    try:
        f = sample_file("_min.csv")
    except pytest.skip.Exception:
        f = sample_file("_beh1.csv")  # ref_aux.m falls back to this file as well
    ref = matlab_ref("sidecar_csv")
    check_input(f, ref)
    out = read_sidecar_csv(f)
    assert out.names == [str(n) for n in np.ravel(ref.out1)]
    assert_same(out.values, ref.out2)
