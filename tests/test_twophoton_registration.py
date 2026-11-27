"""Synthetic tests of the two-photon registration ports (calliopy.twophoton._registration)."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure
from scipy import ndimage

from calliopy.twophoton import (
    apply_registration,
    correct_dxdy,
    expand_shifts,
    fix_registration,
    fix_shift_outliers,
    next_check_registration_version,
    plot_check_registration,
    register_bin,
    register_exp_avgplanes,
    register_exp_downsample,
    register_exp_fixoutliers,
    register_multilayer,
    save_check_registration_figure,
)
from calliopy.twophoton._registration import _bins, _movmean3

PAD = 12


@pytest.fixture(scope="module")
def base():
    rng = np.random.default_rng(0)
    return ndimage.gaussian_filter(rng.normal(size=(80 + 2 * PAD, 90 + 2 * PAD)), 2.5) * 400 + 2000


def frame(base, dr, dc):
    """Frame whose content moved dr rows down and dc columns right (register_frames sign)."""
    h, w = base.shape[0] - 2 * PAD, base.shape[1] - 2 * PAD
    return base[PAD - dr : PAD - dr + h, PAD - dc : PAD - dc + w]


def stack(base, shifts):
    return np.stack([frame(base, a, b) for a, b in shifts], axis=2).round().astype(np.int16)


def test_fix_registration_corrects_jumps(base):
    t = frame(base, 0, 0)
    d = np.stack([t, frame(base, 3, -2), frame(base, 3, -2), frame(base, 1, 1)], axis=2)
    sx, sy = fix_registration(d, t, np.array([0.0, 150, 150, 1]), np.array([0.0, 0, 0, 1]))
    np.testing.assert_array_equal(sx, [0, 3, 3, 1])
    np.testing.assert_array_equal(sy, [0, -2, -2, 1])
    sx2, _ = fix_registration(d, t, np.array([0.0, 2, 2, 1]), np.zeros(4))  # no jump: unchanged
    np.testing.assert_array_equal(sx2, [0, 2, 2, 1])


def test_fix_shift_outliers_runs_and_edges():
    dx = np.zeros((100, 1))
    dx[40:43] = 50
    dx[39], dx[43] = 1, 4
    out = fix_shift_outliers(dx, np.zeros(100))
    assert out.corrected
    np.testing.assert_array_equal(out.dx[40:43, 0], 3)  # round((1 + 4) / 2) = 3 (half away)
    first = np.zeros(100)
    first[:2] = 60
    first[2] = 7
    np.testing.assert_array_equal(fix_shift_outliers(first, np.zeros(100)).dx[:2, 0], 7)
    last = np.zeros(100)
    last[-2:] = 60
    last[-3] = -5
    np.testing.assert_array_equal(fix_shift_outliers(last, np.zeros(100)).dx[-2:, 0], -5)
    clean = fix_shift_outliers(np.zeros(10), np.zeros(10))
    assert not clean.corrected
    detect = fix_shift_outliers(dx, np.zeros(100), correct=False)
    assert detect.corrected and detect.dx[41, 0] == 50


def test_expand_shifts_and_bins():
    np.testing.assert_array_equal(expand_shifts([1, 2, 3], 2, 5)[:, 0], [1, 1, 2, 2, 3])
    d = np.arange(7.0)[None, None, :] * np.ones((2, 2, 1))
    b = _bins(d, 3, zero_gap=True)
    np.testing.assert_allclose(b[0, 0], [1, 4, 0, 5])  # MATLAB: zero block, then the tail
    np.testing.assert_allclose(_bins(d, 3, zero_gap=False)[0, 0], [1, 4, 5])


def test_movmean_matches_matlab_windows():
    x = np.array([1.0, 2, 4, 8, 16])[None, None, :]
    np.testing.assert_allclose(_movmean3(x, 3)[0, 0], [1.5, 7 / 3, 14 / 3, 28 / 3, 12])
    np.testing.assert_allclose(_movmean3(x, 2)[0, 0], [1, 1.5, 3, 6, 12])


def test_register_bin_recovers_shifts(base):
    shifts = [(0, 0)] * 30 + [(2, -1)] * 10 + [(-3, 2)] * 10
    d = stack(base, shifts)
    sec = (d.astype(np.int32) // 2).astype(np.int16)
    out = register_bin([d], np.array([[50]]), sec_data=[sec])
    np.testing.assert_array_equal(out["dx"][0][:, 0], [s[0] for s in shifts])
    np.testing.assert_array_equal(out["dy"][0][:, 0], [s[1] for s in shifts])
    assert out["template"][0].shape == d.shape[:2]
    assert out["template_sec"][0].shape == d.shape[:2] + (1,)
    assert out["act_map"][0].shape == d.shape[:2]
    assert len(out["ROIs"]) == 1 and len(out["ROIs"][0]) == 0
    # registration on channel 2 gives the same shifts here
    out2 = register_bin([d], np.array([[50]]), sec_data=[sec], reg_on_ch=2)
    np.testing.assert_array_equal(out2["dx"][0], out["dx"][0])
    # template_sec per file
    out3 = register_bin([d], np.array([[20, 30]]), sec_data=[sec])
    assert out3["template_sec"][0].shape == d.shape[:2] + (2,)


def test_register_multilayer_and_zstack(base):
    steps = [stack(base, [(0, 0)] * 30) for _ in range(10)]
    data = np.concatenate(steps, axis=2)
    res = register_multilayer(data, 30)
    assert res.mean_data.shape == data.shape[:2] + (10,)
    assert res.dx_corr.shape == (1, 10)
    np.testing.assert_allclose(res.dx_corr, 0, atol=1e-9)
    out = register_bin([data], np.array([[300]]), fpga_frames_zstep=1, fpga_frames_layer=30)
    assert set(out) == {"mean_data"}
    with pytest.raises(ValueError):
        register_multilayer(data[:, :, :90], 30)


def test_register_exp_downsample_zero_block_bug(base):
    shifts = [(3, 0)] * 10 + [(0, 0)] * 20 + [(3, 0)] * 32
    d = stack(base, shifts)
    out = register_exp_downsample([d], fta=3)
    dx = out["dx"][0][:, 0]
    np.testing.assert_array_equal(dx[30:60], 3)
    np.testing.assert_array_equal(dx[60:], 0)  # kept MATLAB bug: tail frames get shift 0
    assert set(out) == {"dx", "dy", "act_map", "template", "template_sec"}


def test_register_exp_avgplanes_shared_shifts(base):
    shifts = [(0, 0)] * 30 + [(2, 1)] * 10
    layers = [stack(base, shifts) for _ in range(2)]
    out = register_exp_avgplanes(layers, moving_average=False, nbr_frames=np.array([[80]]))
    np.testing.assert_array_equal(out["dx"][0][:, 0], [s[0] for s in shifts])
    np.testing.assert_array_equal(out["dx"][1], out["dx"][0])
    assert len(out["template"]) == 2 and "nbr_frames" in out
    out4 = register_exp_avgplanes([stack(base, shifts) for _ in range(4)], moving_average=True)
    np.testing.assert_array_equal(out4["dy"][3][:, 0], [s[1] for s in shifts])


def test_register_exp_fixoutliers(base):
    d = stack(base, [(0, 0)] * 40)
    dx = np.zeros((40, 1))
    assert register_exp_fixoutliers([dx, dx], [dx, dx]) == {}
    bad = dx.copy()
    bad[20] = 40
    with pytest.raises(ValueError):
        register_exp_fixoutliers([bad, dx], [dx, dx])
    multi = register_exp_fixoutliers([bad, dx], [dx, dx], data=[d, d])
    assert set(multi) == {"dx", "dy", "act_map"}  # kept MATLAB bug: template not stored
    assert multi["dx"][0][20, 0] == 0
    single = register_exp_fixoutliers(bad, dx, data=[d])
    assert set(single) == {"dx", "dy", "act_map", "template"}
    assert single["dx"].shape == (40, 1)
    assert register_exp_fixoutliers(bad, dx, correct=False) == {}


def test_correct_dxdy(base):
    d = stack(base, [(0, 0)] * 30)
    dx = [np.zeros((30, 1)), np.ones((30, 1)), 2 * np.ones((30, 1))]
    out = correct_dxdy(dx, dx, z_plane=0, z_to_correct=[2])
    np.testing.assert_array_equal(out["dx"][2], 0)
    np.testing.assert_array_equal(out["dx"][1], 1)
    out2 = correct_dxdy(dx, dx, man_corr=(np.zeros(30), np.zeros(30)), data=[d],
                        sec_data=[d], nbr_frames=np.array([[30]]))
    assert out2["dx"].shape == (30, 1) and out2["template"].shape == d.shape[:2]
    assert out2["template_sec"].shape == d.shape[:2] + (1,)
    with pytest.raises(ValueError):
        correct_dxdy(dx, dx)


def test_apply_registration_requires_nbr_frames(base):
    d = stack(base, [(0, 0)] * 20)
    with pytest.raises(ValueError):
        apply_registration([d], [np.zeros(20)], [np.zeros(20)], sec_data=[d])


def test_check_registration_plot_and_versions(tmp_path, base):
    d = stack(base, [(0, 0)] * 20)
    fig = plot_check_registration([np.zeros(20), np.ones(20)], [np.zeros(20)] * 2,
                                  [d[:, :, 0], d[:, :, 1]])
    assert isinstance(fig, Figure)
    assert isinstance(plot_check_registration(np.zeros(20), np.zeros(20), d[:, :, 0]), Figure)
    names = ["CheckReg-12_v0.png", "CheckReg-12_v10.png", "CheckReg-123_v99.png", "other.png"]
    assert next_check_registration_version(names, 12) == 11
    assert next_check_registration_version([], 12) == 0
    p1 = save_check_registration_figure(fig, tmp_path / "2P_Registration", 7)
    p2 = save_check_registration_figure(fig, tmp_path / "2P_Registration", 7)
    assert p1.endswith("CheckReg-7_v0.png") and p2.endswith("CheckReg-7_v1.png")
