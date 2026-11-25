"""Tests for calliopy.data (find_adata_file, get_adata_filename, load_exp, line shift)."""

from __future__ import annotations

import struct

import numpy as np
import pytest
import scipy.ndimage
from calliopy.data import (
    correct_line_shift,
    estimate_line_shift,
    find_adata_file,
    get_adata_filename,
    load_exp,
)
from calliopy.explog import clear_exp_log_cache, get_exp_log
from calliopy.io import save_mat
from test_explog import make_db


@pytest.fixture
def exp_log():
    clear_exp_log_cache()
    yield get_exp_log(connection=make_db())
    clear_exp_log_cache()


def test_line_shift_estimate_and_correction():
    base = scipy.ndimage.gaussian_filter(np.random.default_rng(0).normal(size=(64, 128)), (6, 3))
    for true in (3, -2, 0):
        img = base.copy()
        img[0::2] = np.roll(base[0::2], true, axis=1)
        assert estimate_line_shift(img) == -true
        res = correct_line_shift(np.repeat(img[:, :, None], 3, axis=2))
        np.testing.assert_allclose(res.data[:, :, 1], base)
    res = correct_line_shift(base.astype(np.int16), shift=2)
    assert res.shift == 2 and res.data.dtype == np.int16
    np.testing.assert_array_equal(res.data[1], base.astype(np.int16)[1])


def test_adata_file_lookup(tmp_path, exp_log):
    folder = tmp_path / "userA" / "M1"
    folder.mkdir(parents=True)
    (folder / "Adata-S1-T501.mat").write_bytes(b"")
    assert tuple(find_adata_file(501, tmp_path, exp_log)) == ("Adata-S1-T501.mat", "M1", "userA")
    assert tuple(get_adata_filename(501, tmp_path, exp_log)) == ("Adata-S1-T501.mat", "M1", "userA", "ECT")
    (folder / "sub").mkdir()
    (folder / "sub" / "old_500_x.mat").write_bytes(b"")
    assert find_adata_file(500, tmp_path, exp_log).adata_file == "old_500_x.mat"
    assert tuple(get_adata_filename(500, tmp_path, exp_log)) == ("", "", "", "ECT")
    assert tuple(find_adata_file(600, tmp_path, exp_log)) == ("", "", "")
    with pytest.raises(LookupError):
        get_adata_filename(1, tmp_path, exp_log)


def _write_lvd(path, data, scan_rate=1000.0):
    n_ch = data.shape[0]
    with open(path, "wb") as fh:
        fh.write(struct.pack(">4d", scan_rate, n_ch, 0.0, 10.0))
        fh.write(data.T.astype(">f8").tobytes())


def _galvo(n_frames, period=150, flyback=40, lead=60):
    sig = np.zeros(lead)
    for _ in range(n_frames):
        sig = np.r_[sig, np.linspace(-1, 1, period - flyback), np.ones(flyback)]
    return np.r_[sig, np.full(50, -1.0)]


def _write_bin(path, frames):
    x_res, y_res = frames.shape[1:]
    with open(path, "wb") as fh:
        fh.write(np.array([x_res, y_res], "<i2").tobytes())
        fh.write(frames.astype("<i2").tobytes())


def _experiment(tmp_path, n_frames=8):
    raw, adir = tmp_path / "raw", tmp_path / "adata"
    folder = raw / "userA" / "M1"
    folder.mkdir(parents=True)
    (adir / "userA" / "M1").mkdir(parents=True)
    rng = np.random.default_rng(0)
    frames = {}
    for sid in (502, 503):
        galvo = _galvo(n_frames)
        _write_lvd(folder / f"S1-T{sid}.lvd", np.vstack([np.zeros(galvo.size), galvo]))
        frames[sid] = rng.integers(0, 1000, size=(n_frames, 6, 5)).astype(np.int16)
        _write_bin(folder / f"S1-T{sid}_ch525.bin", frames[sid])
        (folder / f"S1-T{sid}_ch525.ini").write_text("[main]\npiezo.nbrlayers = 1\n")
    (folder / "S1-T502.ach").write_text('[main]\nauxrec.scanrate = "1000"\nauxrec.channel0 = "Running"\n'
                                        'auxrec.channel1 = "galvo"\n')
    template = np.empty((1, 1), dtype=object)
    template[0, 0] = np.zeros((6, 5))
    dx = np.empty((1, 1), dtype=object)
    dx[0, 0] = np.zeros((1, 2 * n_frames))
    save_mat(adir / "userA" / "M1" / "Adata-S1-T501.mat",
             {"template": template, "nbr_frames": np.array([[n_frames, n_frames]], dtype=float),
              "dx": dx, "dy": dx, "act_map": np.ones((6, 5))})
    return raw, adir, frames


def test_load_exp_bin_ini(tmp_path, exp_log):
    raw, adir, frames = _experiment(tmp_path)
    ws = load_exp(501, [".bin", ".ini"], exp_log, raw_data_dir=raw, adata_dir=adir, noregister=True)
    assert ws.userID == "userA" and ws.mouse_id == "M1"
    assert ws.aux_data.shape[0] == 2 and ws.aux_names == ["velM", "galvo"] and ws.aux_scanrate == 1000
    n_aux = ws.aux_data.shape[1] // 2
    assert ws.frame_times.size == 16
    assert ws.frame_times[8] == ws.frame_times[0] + n_aux  # second stack offset, 1-based samples
    assert len(ws.data) == 1 and ws.data[0].shape == (6, 5, 16) and ws.data[0].dtype == np.int16
    np.testing.assert_array_equal(ws.data[0][:, :, 3], frames[502][3])
    np.testing.assert_array_equal(ws.data[0][:, :, 8], frames[503][0])
    assert {"dx", "dy", "nbr_frames", "template", "act_map"} <= set(ws)


def test_load_exp_registration_and_errors(tmp_path, exp_log):
    raw, adir, _ = _experiment(tmp_path)
    ws = load_exp(501, [".bin"], exp_log, raw_data_dir=raw, adata_dir=adir)
    assert ws.data[0].shape == (6, 5, 16)
    with pytest.raises(NotImplementedError, match="EEG.dat"):
        load_exp(501, ["EEG.dat"], exp_log, raw_data_dir=raw, adata_dir=adir)
    with pytest.raises(ValueError, match="adata_dir"):
        load_exp(501, [".bin"], exp_log, raw_data_dir=raw)
    with pytest.raises(ValueError, match="raw_data_dir"):
        load_exp(501, [".bin"], exp_log, adata_dir=adir)
    with pytest.raises(FileNotFoundError, match="register data"):
        load_exp(500, [".bin"], exp_log, raw_data_dir=raw, adata_dir=adir)
    ach = load_exp(501, [".ach"], exp_log, raw_data_dir=raw, adata_dir=adir)
    assert "ach" in ach and "aux_data" not in ach
    wcp = np.arange(12.0).reshape(4, 3)  # 4 samples x 3 channels, old headerless layout
    (raw / "userA" / "M1" / "S1-T502.wcp").write_bytes(wcp.astype(">f8").tobytes())
    ws = load_exp(501, [".wcp"], exp_log, raw_data_dir=raw, adata_dir=adir)
    np.testing.assert_array_equal(ws.wcp_data, wcp.T)
    assert ws.wcp_scanrate == 20000
