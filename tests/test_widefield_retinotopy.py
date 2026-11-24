"""Tests for calliopy.widefield._retinotopy (online_retinotopy.m)."""

from __future__ import annotations

import struct

import numpy as np
import pytest
from matplotlib.figure import Figure

from calliopy.io import parse_ach
from calliopy.widefield._retinotopy import (
    crop_rect_from_points,
    filter_with_disk_kernel,
    online_retinotopy,
    online_retinotopy_from_files,
    plot_online_retinotopy,
    retinotopy_anatomy,
    retinotopy_frame_triggers,
    retinotopy_response_maps,
    retinotopy_stimulus_triggers,
)

THR = 0.25

# --- literal MATLAB references (1-based arithmetic, explicit loops) -----------------------


def _find_rising(x, extra=None):
    """MATLAB find(x(1:end-1) <= THR & x(2:end) > THR [& extra(1:end-1) > THR]), 1-based."""
    out = []
    for k in range(1, len(x)):  # k = 1 .. end-1
        ok = x[k - 1] <= THR and x[k] > THR
        if extra is not None:
            ok = ok and extra[k - 1] > THR
        if ok:
            out.append(k)
    return out


def _matlab_round1(x):
    y = abs(x) * 10
    r = np.floor(y + 0.5)
    return np.sign(x) * r / 10


def _ref_response_maps(data, trigs_1b, trig_type, downs, secs):
    n_avg = int(np.floor(100 * secs / downs + 0.5))
    types = sorted(set(trig_type))
    maps = np.zeros(data.shape[:2] + (len(types),))
    n_frames = data.shape[2]
    for i, tt in enumerate(types):
        these = [t for t, ty in zip(trigs_1b, trig_type, strict=True) if ty == tt]
        resp = []
        for t in these:
            frm = max(1, t - n_avg)
            pre_idx = list(range(frm, t))  # frm : t-1
            to = min(t + n_avg + 2, n_frames)
            post_idx = list(range(t + 3, to + 1))  # t+1+2 : to
            pre = (
                np.mean([data[:, :, k - 1].astype(float) for k in pre_idx], axis=0)
                if pre_idx
                else np.full(data.shape[:2], np.nan)
            )
            post = (
                np.mean([data[:, :, k - 1].astype(float) for k in post_idx], axis=0)
                if post_idx
                else np.full(data.shape[:2], np.nan)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                resp.append((post - pre) / pre)
        maps[:, :, i] = np.mean(resp, axis=0)
    return maps, np.array(types)


def _ref_disk_loop(m, radius):
    """filterWithDiskKernel as written in MATLAB: manual replicate padding and a loop."""
    rows, cols = m.shape
    ks = 2 * radius + 1
    X, Y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    disk = (np.sqrt(X**2 + Y**2) <= radius).astype(float)
    p = np.zeros((rows + 2 * radius, cols + 2 * radius))
    r = radius
    p[r : r + rows, r : r + cols] = m
    if r:
        p[:r, r : r + cols] = m[0, :]
        p[-r:, r : r + cols] = m[-1, :]
        p[r : r + rows, :r] = m[:, [0]]
        p[r : r + rows, -r:] = m[:, [-1]]
        p[:r, :r] = m[0, 0]
        p[:r, -r:] = m[0, -1]
        p[-r:, :r] = m[-1, 0]
        p[-r:, -r:] = m[-1, -1]
    out = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            out[i, j] = np.sum(p[i : i + ks, j : j + ks] * disk)
    return out


# --- synthetic recording ------------------------------------------------------------------


def _recording(n_frames_raw=400, downs=5, seed=0):
    """Aux traces at 4 samples per raw frame, stimuli of known type, and a movie."""
    rng = np.random.default_rng(seed)
    spf = 4
    n_samp = n_frames_raw * spf + 40
    cam = np.zeros(n_samp)
    for f in range(n_frames_raw):
        cam[20 + f * spf : 20 + f * spf + 2] = 3.0
    shutter = np.zeros(n_samp)
    shutter[10 : 20 + (n_frames_raw - 5) * spf - 1] = 5.0  # last 5 frames with shutter closed
    stim = np.zeros(n_samp)
    onsets = {50: 2.0, 130: 0.5, 210: 1.04, 290: 2.0}  # raw frame -> level (1.04 -> 1.0)
    for f0, level in onsets.items():
        stim[20 + f0 * spf + 1 : 20 + (f0 + 70) * spf] = level  # 14 loaded frames
    aux = np.vstack([rng.normal(0, 0.01, n_samp), cam, shutter, stim])
    labels = ["Photodiode", "CamExp", "Shutter", "StimulusID"]
    n_loaded = (n_frames_raw - 5 + downs - 1) // downs
    data = rng.integers(900, 1100, size=(6, 7, n_loaded)).astype(np.uint16)
    return data, aux, labels


# --- frame and stimulus triggers ----------------------------------------------------------


def test_frame_triggers_match_literal_find():
    _, aux, _ = _recording()
    cam, sh = aux[1], aux[2]
    ref = _find_rising(cam, sh)
    for downs in (1, 3, 5):
        got = retinotopy_frame_triggers(cam, sh, downs)
        np.testing.assert_array_equal(got + 1, ref[::downs])
    assert len(ref) == 395  # frames with open shutter


def test_frame_triggers_bad_downs_factor():
    with pytest.raises(ValueError):
        retinotopy_frame_triggers(np.zeros(5), np.zeros(5), 2.5)
    with pytest.raises(ValueError):
        retinotopy_frame_triggers(np.zeros(5), np.zeros(4), 1)


def test_stimulus_triggers_types_and_indices():
    stim = np.zeros(60)
    stim[10:30] = 2.0
    stim[40:50] = 1.04
    stim[50:53] = 0.53  # after the second trigger, inside its one-second window
    stim[58:] = 3.5  # trigger near the end: shorter window
    st = retinotopy_stimulus_triggers(stim, 5)
    np.testing.assert_array_equal(st.trigs + 1, _find_rising(stim))
    np.testing.assert_array_equal(st.trigs, [9, 39, 57])
    # window: 20 samples after the trigger (round(100/5)); median rounded to 0.1
    assert st.trig_type[0] == 2.0
    assert st.trig_type[1] == _matlab_round1(np.median(stim[40:60])) == 1.0
    assert st.trig_type[2] == 3.5
    assert st.trigger_types.shape == (1, 3)
    np.testing.assert_array_equal(st.trigger_types, [[1.0, 2.0, 3.5]])


def test_stimulus_type_rounds_half_away():
    stim = np.zeros(30)
    stim[5:] = 1.25  # median 1.25 -> round(.., 1) = 1.3 (half away from zero)
    st = retinotopy_stimulus_triggers(stim, 10)
    assert st.trig_type.tolist() == [1.3]


def test_stimulus_triggers_none():
    st = retinotopy_stimulus_triggers(np.zeros(10), 5)
    assert st.trigs.size == 0 and st.trigger_types.shape == (1, 0)


# --- response maps ------------------------------------------------------------------------


def test_response_maps_match_literal_windows():
    rng = np.random.default_rng(3)
    data = rng.integers(500, 1500, size=(3, 4, 70)).astype(np.uint16)
    trigs_1b = [1, 2, 20, 35, 64, 68, 70]  # include empty pre and empty post windows
    types = [2.0, 0.5, 2.0, 1.0, 0.5, 1.0, 3.0]
    ref, ref_types = _ref_response_maps(data, trigs_1b, types, 5, 1.5)
    got = retinotopy_response_maps(
        data, np.array(trigs_1b) - 1, types, downs_factor=5, resp_avg_seconds=1.5
    )
    np.testing.assert_array_equal(got.trigger_types, ref_types.reshape(1, -1))
    np.testing.assert_allclose(got.resp_maps, ref, rtol=1e-12, equal_nan=True)
    assert np.isnan(got.resp_maps[:, :, -1]).all()  # trigger at the last frame: no post


def test_response_maps_analytic_step():
    # baseline 100, frames from 3 after the trigger on 110: response exactly 0.1
    data = np.full((2, 2, 60), 100.0)
    t = 20  # 0-based trigger frame
    n = 30  # round(100 * 1.5 / 5)
    data[:, :, t + 3 :] = 110.0
    data[:, :, t : t + 3] = 999.0  # the trigger frame and the two after it are excluded
    got = retinotopy_response_maps(data, [t], [2.0])
    np.testing.assert_allclose(got.resp_maps[:, :, 0], 0.1, rtol=1e-12)
    # post window: n frames from t+3; frames after it do not count
    data2 = np.full((1, 1, 60), 100.0)
    data2[0, 0, t + 3 : t + 3 + n] = 50.0
    got2 = retinotopy_response_maps(data2, [t], [1.0])
    np.testing.assert_allclose(got2.resp_maps[0, 0, 0], -0.5, rtol=1e-12)


def test_response_maps_trigger_beyond_movie():
    with pytest.raises(IndexError):
        retinotopy_response_maps(np.ones((2, 2, 10)), [12], [1.0])


def test_response_maps_no_triggers():
    got = retinotopy_response_maps(np.ones((2, 3, 10)), [], [])
    assert got.resp_maps.shape == (2, 3, 0) and got.trigger_types.shape == (1, 0)


# --- anatomy and disk filter --------------------------------------------------------------


def test_anatomy():
    rng = np.random.default_rng(1)
    data = rng.integers(0, 4000, size=(5, 6, 120)).astype(np.uint16)
    m = data[:, :, 4:105].astype(float).mean(axis=2)
    expected = np.floor((m - m.min()) / (m.max() - m.min()) * 255 + 0.5).astype(np.uint8)
    got = retinotopy_anatomy(data)
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, expected)
    assert got.min() == 0 and got.max() == 255
    np.testing.assert_array_equal(retinotopy_anatomy(data[:, :, :4]), 0)  # no frame 5
    np.testing.assert_array_equal(retinotopy_anatomy(np.full((2, 2, 8), 7.0)), 0)


@pytest.mark.parametrize("radius", [0, 1, 2, 3, 6])
def test_disk_filter_equals_matlab_loop(radius):
    m = np.random.default_rng(radius).standard_normal((5, 7))
    np.testing.assert_allclose(
        filter_with_disk_kernel(m, radius), _ref_disk_loop(m, radius), rtol=1e-12, atol=1e-12
    )


def test_disk_filter_kernel_and_errors():
    impulse = np.zeros((9, 9))
    impulse[4, 4] = 1.0
    out = filter_with_disk_kernel(impulse, 2)
    assert out.sum() == 13  # lattice points with x^2 + y^2 <= 4
    np.testing.assert_array_equal(out, out.T)
    np.testing.assert_allclose(filter_with_disk_kernel(np.full((3, 4), 2.0), 2), 26.0)
    with pytest.raises(ValueError):
        filter_with_disk_kernel(impulse, 1.5)
    with pytest.raises(ValueError):
        filter_with_disk_kernel(impulse, -1)


# --- whole computation, files and figure --------------------------------------------------


def test_online_retinotopy_core():
    data, aux, labels = _recording()
    res = online_retinotopy(data, aux, labels, downs_factor=5, resp_avg_seconds=1.5)
    ref_ft = _find_rising(aux[1], aux[2])[::5]
    np.testing.assert_array_equal(res.frame_trigs + 1, ref_ft)
    stim = aux[3, np.array(ref_ft) - 1]
    st = retinotopy_stimulus_triggers(stim, 5)
    np.testing.assert_array_equal(res.trigs, st.trigs)
    np.testing.assert_array_equal(res.trigger_types, [[0.5, 1.0, 2.0]])
    ref, _ = _ref_response_maps(data, list(st.trigs + 1), list(st.trig_type), 5, 1.5)
    np.testing.assert_allclose(res.resp_maps, ref, rtol=1e-12)
    np.testing.assert_array_equal(res.anatomy, retinotopy_anatomy(data))
    # channel table as from parse_ach: [n x 2], 1-based numbers in column 0
    table = np.empty((4, 2), dtype=object)
    for i, lab in enumerate(labels):
        table[i] = (i + 1, lab.lower())  # labels are matched case-insensitively
    res2 = online_retinotopy(data, aux, table)
    np.testing.assert_array_equal(res2.resp_maps, res.resp_maps)


def test_online_retinotopy_missing_channel():
    data, aux, labels = _recording()
    with pytest.raises(ValueError, match="StimulusID"):
        online_retinotopy(data, aux, labels[:3] + ["Other"])
    with pytest.raises(ValueError, match="CamExp"):
        online_retinotopy(data, aux, ["CamExp", "CamExp", "Shutter", "StimulusID"])


def _write_wid(path, imgs):
    size_y, size_x, n = imgs.shape
    out = bytearray()
    for k in range(n):
        out += struct.pack("<HH", size_y, size_x)
        out += imgs[:, :, k].astype("<u2").ravel(order="F").tobytes()
    path.write_bytes(bytes(out))


def _write_lvd(path, data, scan_rate=4000.0):
    header = np.array([scan_rate, data.shape[0], 0.0, 10.0], dtype=">f8")
    path.write_bytes(header.tobytes() + data.T.astype(">f8").tobytes())


def _write_ach(path, labels):
    lines = ["[auxrec]", 'auxrec.scanrate = "4000"']
    lines += [f'auxrec.channel{i} = "{lab}"' for i, lab in enumerate(labels)]
    path.write_text("\n".join(lines) + "\n")


def test_online_retinotopy_from_files(tmp_path):
    data, aux, labels = _recording()
    raw = np.repeat(data, 5, axis=2)  # every 5th raw frame is the loaded movie
    _write_wid(tmp_path / "S1-T1.wid", raw)
    _write_lvd(tmp_path / "S1-T1.lvd", aux)
    _write_ach(tmp_path / "S1-T1.ach", labels)
    ach = parse_ach(tmp_path / "S1-T1.ach")
    assert [str(x) for x in ach["aux_chans"][:, 1]] == labels
    got = online_retinotopy_from_files(
        tmp_path / "S1-T1.wid", tmp_path / "S1-T1.lvd", tmp_path / "S1-T1.ach", downs_factor=5
    )
    want = online_retinotopy(data, aux, labels, downs_factor=5)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_array_equal(a, b)


def test_crop_rect_and_figure():
    # 0-based points: MATLAB clicks (3.4, 2.5), (7.6, 5.4) give pos [3, 3, 4, 3] -> minus 1
    np.testing.assert_array_equal(crop_rect_from_points(2.4, 1.5, 6.6, 4.4), [2, 2, 4, 3])
    # x, y round as MATLAB rounds the 1-based value: -0.5 (MATLAB 0.5 -> 1) gives 0
    np.testing.assert_array_equal(crop_rect_from_points(-0.5, 0.49, 1.0, 2.0), [0, 0, 2, 2])
    np.testing.assert_array_equal(crop_rect_from_points(3.0, 3.0, 0.5, 0.5), [3, 3, -3, -3])
    data, aux, labels = _recording()
    res = online_retinotopy(data, aux, labels)
    fig = plot_online_retinotopy(
        res.resp_maps, res.trigger_types, res.anatomy, [1, 0, 4, 3], sp_filter_size_px=1,
        stack_id=7,
    )
    assert isinstance(fig, Figure)
    axes = fig.get_axes()
    assert [ax.get_title() for ax in axes] == [
        "Full-field response",
        "Horizontal Retinotopy",
        "Vertical retinotopy",
    ]
    # pos [1, 0, 4, 3] (0-based) = MATLAB pos [2, 1, 4, 3]: MATLAB rows 1:4, columns 2:6
    # -> 0-based rows 0..3, columns 1..5 (4 x 5), shown transposed (5 x 4)
    shown = np.asarray(axes[0].get_images()[0].get_array())
    np.testing.assert_array_equal(shown, res.anatomy[0:4, 1:6].T)
    # full field: one overlay; horizontal: three groups; vertical: two
    assert [len(ax.get_images()) for ax in axes] == [2, 4, 3]
    # the whole image is a valid crop; one more column or row is not
    h, w = res.anatomy.shape
    plot_online_retinotopy(res.resp_maps, res.trigger_types, res.anatomy, [0, 0, w - 1, h - 1])
    for bad in ([0, 0, w, h - 1], [0, 0, w - 1, h], [-1, 0, 2, 2], [4, 0, 4, 3]):
        with pytest.raises(IndexError):
            plot_online_retinotopy(res.resp_maps, res.trigger_types, res.anatomy, bad)
    with pytest.raises(ValueError):
        plot_online_retinotopy(res.resp_maps, res.trigger_types, res.anatomy, [3, 3, -1, 2])


def test_display_orientation_is_transpose():
    from calliopy.widefield._retinotopy import _display

    a = np.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(_display(a), a.T)
