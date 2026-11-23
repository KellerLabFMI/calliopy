"""Tests for calliopy.io._eeg: load_eeg, load_heeg, load_heyetracking.

Part 1 builds small synthetic files in the formats the MATLAB readers parse:

* OpenEphys binary ``S1-T<id>_EEG.dat`` (int16 little endian, channels interleaved), the
  ``.oebin`` JSON header and the ``.npy`` timestamps (``load_EEG.m``);
* OpenBCI raw text ``S1-T<id>.txt`` (``%`` comment lines, header line, comma separated, columns
  as in the OpenBCI GUI) and the MetaQuest JSON log ``S1-T<id>.json`` (``load_hEEG.m``,
  ``load_hEyeTracking.m``).

Part 2 compares with MATLAB r2003 outputs; those tests skip until CALLIOPY_TESTDATA holds the
samples and ``tests/matlab/ref_eeg.m`` outputs (see ``docs/porting_guide.md``). Expected samples:

    samples/S1-T<id>_EEG.dat (+ .oebin, .npy)   one stack of an EEG ExpID
    samples/S1-T<id>.txt (+ same-name .json)    one hEEG stack
    samples/S1-T<id>_eye.json                   one eye-tracking stack (a copy of its .json)
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

# The package __init__ is not extended by this track; import from the private module.
from calliopy.io import _eeg
from calliopy.io._eeg import (
    EEG_CHANNEL_ORDER,
    EEG_SYNC_CHANNEL,
    EegData,
    load_eeg,
    load_heeg,
    load_heyetracking,
)

# =====================================================================================
# Synthetic file writers
# =====================================================================================

N_CHANS = 34  # >= 33 needed (channel 33 is the sync channel)


def write_eeg_stack(folder, stack_id, raw, bit_volts, rate, *, ts=None, streams=1):
    """Write S1-T<id>_EEG.{dat,oebin,npy} for ``raw`` (int16, [nChannels x nSamples])."""
    root = folder / f"S1-T{stack_id}_EEG"
    raw.T.astype("<i2").tofile(str(root) + ".dat")  # interleaved: channel index fastest
    stream = {
        "folder_name": "Record Node 101/",
        "sample_rate": float(rate),
        "stream_name": "example_data",
        "num_channels": int(raw.shape[0]),
        "channels": [
            {"channel_name": f"CH{i + 1}", "bit_volts": float(bit_volts[i]), "units": "uV"}
            for i in range(raw.shape[0])
        ],
    }
    header = {"GUID": "x", "continuous": [stream] * streams, "events": [], "spikes": []}
    with open(str(root) + ".oebin", "w", encoding="utf-8") as f:
        json.dump(header, f)
    if ts is None:
        ts = np.arange(raw.shape[1]) / rate
    np.save(str(root) + ".npy", np.asarray(ts, dtype=np.float64))
    return root


def make_raw(n_samples, n_chans=N_CHANS, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(-3000, 3000, size=(n_chans, n_samples)).astype(np.int16)


def bit_volts_for(n_chans=N_CHANS):
    return 0.195 * (1 + np.arange(n_chans) / 100.0)


def expected_eeg(raw, bit_volts):
    """What MATLAB computes per stack without resampling: single(raw) .* bit_volts."""
    return raw.astype(np.float32) * bit_volts.astype(np.float32)[:, None]


OPENBCI_COMMENTS = [
    "%OpenBCI Raw EXG Data",
    "%Number of channels = 8",
    "%Sample Rate = 250 Hz",
    "%First Column = SampleIndex",
    "%Last Column = Timestamp",
    "%Other Columns = EEG data in microvolts followed by Accel Data (in G) and Aux Data",
]
OPENBCI_COLUMNS = (
    ["Sample Index"]
    + [f"EXG Channel {i}" for i in range(8)]
    + [f"Accel Channel {i}" for i in range(3)]
    + ["Other"] * 7
    + [f"Analog Channel {i}" for i in range(3)]
    + ["Timestamp", "Marker Index", "Timestamp (Formatted)"]
)
FS_HEEG = 250.0
T0_EPOCH = 1_742_000_000.0


def write_openbci(path, exg, analog2, stamp, columns=None, extra_bad_row=False):
    """Write an OpenBCI raw text file; ``exg`` is [nSamples x 8]."""
    columns = OPENBCI_COLUMNS if columns is None else columns
    n = exg.shape[0]
    lines = list(OPENBCI_COMMENTS) + [", ".join(columns)]
    for i in range(n):
        row = [str(i % 256)]
        row += [f"{v:.17g}" for v in exg[i]]
        row += ["0.001", "0.002", "0.003"]
        row += ["0"] * 7
        row += ["0.0", "0.0", f"{analog2[i]:.17g}"]
        row += [f"{stamp[i]:.17g}", "0", "2025-03-23 14:05:12.123"]
        lines.append(", ".join(row))
        if extra_bad_row and i == n // 3:
            lines.append("1, 2, 3")  # truncated row (partial write)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def sync_pulses(n, rises, width=10, amp=3.3):
    """Analog channel: 0 with ``width``-sample pulses starting at (0-based) sample ``rises``."""
    x = np.zeros(n)
    for r in rises:
        x[r : r + width] = amp
    return x


def _xyz(values, names):
    return {n: float(v) for n, v in zip(names, values, strict=True)}


class HeegSetup:
    """A matching OpenBCI text file and VR json with known ground truth."""

    def __init__(self, folder, *, n_eeg=12000, rises=(500, 2500, 9000), vr_rate=90.0,
                 vr_rise=45, clockwise=False, reference=False, eye=False, mm_blocks=True,
                 exg=None, name="S1-T7"):
        self.n_eeg = n_eeg
        self.rises = rises
        rng = np.random.default_rng(3)
        self.exg = rng.normal(0, 20, (n_eeg, 8)) if exg is None else exg
        self.analog = sync_pulses(n_eeg, rises)
        self.stamp = T0_EPOCH + np.arange(n_eeg) / FS_HEEG
        self.txt = folder / f"{name}.txt"
        write_openbci(self.txt, self.exg, self.analog, self.stamp)

        # VR log: 90 Hz with a little timing jitter; sync pulses at records vr_rise, ...
        m = int((n_eeg / FS_HEEG + 10) * vr_rate)
        i = np.arange(m)
        self.vr_t = 5.0 + i / vr_rate + 1e-4 * np.sin(i)
        sync = np.zeros(m)
        for r in (vr_rise, vr_rise + 1000, vr_rise + 3000):
            sync[r : r + 5] = 1.0
        mm = np.zeros(m)
        if mm_blocks:
            for s, code in ((200, 1), (800, 2), (1500, 1)):
                mm[s : s + 60] = code
        w = 0.5 * (-1 if clockwise else 1)
        ang = w * i / vr_rate
        self.pos = np.stack([3 + 2 * np.cos(ang), 1.5 + 0 * ang, -1 + 2 * np.sin(ang)], axis=1)
        self.rot = np.stack([0.1 * np.sin(ang), np.degrees(ang) % 360, 0 * ang], axis=1)
        self.ref_pos = self.pos + np.array([0.0, 0.1, 0.0]) * np.cos(3 * ang)[:, None]
        self.ref_rot = self.rot + 1.0
        self.sync, self.mm = sync, mm
        self.mismatch = (i // 300) % 2 == 1
        self.audio = 0.25 * (i % 5)
        self.eye_blink = (0.5 + 0.4 * np.sin(i / 7), 0.5 + 0.4 * np.cos(i / 7))
        self.eye_rot = np.stack([np.sin(i / 11), np.cos(i / 11), 0.1 * i / m, 1.0 + 0 * i], axis=1)
        records = []
        for k in range(m):
            rec = {
                "timestamp": float(self.vr_t[k]),
                "synchroSignal": float(sync[k]),
                "mm": float(mm[k]),
                "mismatch": bool(self.mismatch[k]),
                "audioVolume": float(self.audio[k]),
                "position": _xyz(self.pos[k], "xyz"),
                "rotation": _xyz(self.rot[k], "xyz"),
            }
            if reference:
                rec["referencePosition"] = _xyz(self.ref_pos[k], "xyz")
                rec["referenceRotation"] = _xyz(self.ref_rot[k], "xyz")
            if eye:
                rec["leftEyeBlink"] = float(self.eye_blink[0][k])
                rec["rightEyeBlink"] = float(self.eye_blink[1][k])
                rec["leftEyeRotation"] = _xyz(self.eye_rot[k], "xyzw")
                rec["rightEyeRotation"] = _xyz(-self.eye_rot[k], "xyzw")
            records.append(rec)
        self.records = records
        self.json = folder / f"{name}.json"
        self.write_json()

    def write_json(self, records=None):
        with open(self.json, "w", encoding="utf-8") as f:
            json.dump({"records": self.records if records is None else records}, f)

    # --- ground truth computed independently of the module -------------------------------
    @property
    def segment(self):
        return slice(self.rises[0] - 1, self.rises[-1])  # sample before first rise .. last

    def eeg_time(self):
        s = self.stamp[self.segment]
        return s - s[0]

    def vr_time(self):
        i0 = 45 - 1  # last low record before the first VR pulse
        return self.vr_t - self.vr_t[i0]

    def nearest(self):
        vt = self.vr_time()
        return np.array([int(np.argmin(np.abs(t - vt))) for t in self.eeg_time()])

    def speed(self, pos, vt):
        d = np.sqrt(np.sum(np.diff(pos, axis=0) ** 2, axis=1))
        return np.concatenate(([0.0], d / np.diff(vt)))


# =====================================================================================
# load_eeg
# =====================================================================================


def test_eeg_channel_order_constant():
    order = list(EEG_CHANNEL_ORDER)
    assert len(order) == 30 and len(set(order)) == 30
    assert order[:7] == [25, 26, 27, 28, 29, 30, 31]  # MATLAB 26:32
    assert order[7:23] == list(range(23, 7, -1))  # MATLAB flip(9:24)
    assert order[23:] == [0, 1, 2, 3, 4, 5, 6]  # MATLAB 1:7
    assert EEG_SYNC_CHANNEL == 32
    assert not {7, 24, 32} & set(order)  # MATLAB channels 8, 25, 33 are not EEG output


def test_load_eeg_two_stacks_values_orientation_dtype(tmp_path):
    bv = bit_volts_for()
    raw1, raw2 = make_raw(50, seed=1), make_raw(70, seed=2)
    write_eeg_stack(tmp_path, 11, raw1, bv, 1000)
    write_eeg_stack(tmp_path, 12, raw2, bv, 1000)
    res = load_eeg([tmp_path / "S1-T11_EEG.dat", tmp_path / "S1-T12_EEG.dat"])

    assert isinstance(res, EegData)
    eeg, sync, samps = res  # MATLAB [EEGdata, EEGsync, SampsPerStack]
    assert eeg.shape == (30, 120) and eeg.dtype == np.float32
    assert sync.shape == (1, 120) and sync.dtype == np.float32
    assert samps.shape == (1, 2) and samps.tolist() == [[50, 70]]

    full = np.concatenate([expected_eeg(raw1, bv), expected_eeg(raw2, bv)], axis=1)
    np.testing.assert_array_equal(eeg, full[list(EEG_CHANNEL_ORDER), :])
    np.testing.assert_array_equal(sync[0], full[32, :])
    # attribute access = tuple order
    assert res.eeg_data is eeg and res.samps_per_stack is samps


def test_load_eeg_channel_mapping_and_scaling(tmp_path):
    n = 12
    raw = np.zeros((N_CHANS, n), dtype=np.int16)
    for c in range(N_CHANS):
        raw[c, :] = (c + 1) * 10  # channel c holds the constant 10 * (c + 1)
    bv = np.full(N_CHANS, 0.5)
    bv[3] = 2.0
    write_eeg_stack(tmp_path, 1, raw, bv, 1000)
    eeg, sync, _ = load_eeg(tmp_path / "S1-T1_EEG.dat")
    for row, ch in enumerate(EEG_CHANNEL_ORDER):
        assert np.all(eeg[row] == np.float32((ch + 1) * 10 * bv[ch]))
    assert np.all(sync == np.float32(330 * 0.5))  # MATLAB channel 33 = index 32


def test_load_eeg_accepts_str_single_path_and_explicit_headers(tmp_path):
    bv = bit_volts_for()
    raw = make_raw(40, seed=4)
    write_eeg_stack(tmp_path, 5, raw, bv, 1000)
    other = tmp_path / "elsewhere"
    other.mkdir()
    for ext in (".oebin", ".npy"):
        (tmp_path / f"S1-T5_EEG{ext}").rename(other / f"hdr{ext}")
    res = load_eeg(
        str(tmp_path / "S1-T5_EEG.dat"),
        header_files=str(other / "hdr.oebin"),
        timestamp_files=[other / "hdr.npy"],
    )
    assert res.eeg_data.shape == (30, 40)
    np.testing.assert_array_equal(res.eeg_sync[0], expected_eeg(raw, bv)[32])


def test_load_eeg_resamples_30khz_to_1khz(tmp_path):
    rate, n = 30000, 6000  # 0.2 s
    t = np.arange(n) / rate
    wave = 500 * np.sin(2 * np.pi * 100 * t)
    raw = np.tile(np.round(wave).astype(np.int16), (N_CHANS, 1))
    bv = np.full(N_CHANS, 0.195)
    write_eeg_stack(tmp_path, 1, raw, bv, rate)
    eeg, sync, samps = load_eeg(tmp_path / "S1-T1_EEG.dat")
    assert eeg.shape == (30, 200) and sync.shape == (1, 200)
    assert samps.tolist() == [[200]] and eeg.dtype == np.float32
    expect = 0.195 * 500 * np.sin(2 * np.pi * 100 * np.arange(200) / 1000)
    # skip the filter edges (about 10 output samples each side)
    np.testing.assert_allclose(eeg[0, 15:-15], expect[15:-15], atol=0.195 * 500 * 0.01)
    np.testing.assert_allclose(sync[0, 15:-15], expect[15:-15], atol=0.195 * 500 * 0.01)


def test_load_eeg_resampling_equals_resample_poly(tmp_path):
    from scipy.signal import resample_poly

    bv = bit_volts_for()
    raw = make_raw(3001, seed=8)
    write_eeg_stack(tmp_path, 1, raw, bv, 30000)
    res = load_eeg(tmp_path / "S1-T1_EEG.dat")
    full = expected_eeg(raw, bv).astype(np.float64)
    ref = resample_poly(full, 1, 30, axis=1)  # ceil(3001 / 30) = 101 samples
    assert res.eeg_data.shape[1] == 101
    np.testing.assert_allclose(
        res.eeg_data, ref[list(EEG_CHANNEL_ORDER)], rtol=0, atol=np.abs(ref).max() * 2e-6
    )


def test_load_eeg_2000hz_is_not_resampled_but_2500hz_is(tmp_path):
    bv = bit_volts_for()
    raw = make_raw(1000, seed=9)
    write_eeg_stack(tmp_path, 1, raw, bv, 2000)
    assert load_eeg(tmp_path / "S1-T1_EEG.dat").eeg_data.shape == (30, 1000)
    write_eeg_stack(tmp_path, 2, raw, bv, 2500)  # MATLAB resample(x, 1, 2.5) would fail
    assert load_eeg(tmp_path / "S1-T2_EEG.dat").eeg_data.shape == (30, 400)  # ceil(1000*2/5)


def test_load_eeg_sample_rate_is_robust_median(tmp_path):
    n = 3000
    steps = np.full(n, 1 / 30000.0)
    steps[::97] *= 1.4  # jitter does not move the median step
    steps[10] = 0.01  # one dropped-sample gap
    bv = bit_volts_for()
    write_eeg_stack(tmp_path, 1, make_raw(n), bv, 30000, ts=np.cumsum(steps))
    assert load_eeg(tmp_path / "S1-T1_EEG.dat").eeg_data.shape == (30, 100)


def test_load_eeg_releases_files(tmp_path):
    write_eeg_stack(tmp_path, 1, make_raw(20), bit_volts_for(), 1000)
    load_eeg(tmp_path / "S1-T1_EEG.dat")
    for ext in (".dat", ".oebin", ".npy"):  # would fail on Windows with an open memory map
        os.remove(tmp_path / f"S1-T1_EEG{ext}")


def test_load_eeg_errors(tmp_path):
    bv = bit_volts_for()
    with pytest.raises(ValueError, match="at least one stack"):
        load_eeg([])
    with pytest.raises(FileNotFoundError):
        load_eeg(tmp_path / "nothing_EEG.dat")

    write_eeg_stack(tmp_path, 1, make_raw(30), bv, 1000)
    write_eeg_stack(tmp_path, 2, make_raw(30, n_chans=40), bit_volts_for(40), 1000)
    with pytest.raises(ValueError, match="same number of channels"):
        load_eeg([tmp_path / "S1-T1_EEG.dat", tmp_path / "S1-T2_EEG.dat"])

    write_eeg_stack(tmp_path, 3, make_raw(30, n_chans=32), bit_volts_for(32), 1000)
    with pytest.raises(ValueError, match="at least 33"):
        load_eeg(tmp_path / "S1-T3_EEG.dat")

    write_eeg_stack(tmp_path, 4, make_raw(30), bv, 1000, streams=2)
    with pytest.raises(ValueError, match="exactly one entry"):
        load_eeg(tmp_path / "S1-T4_EEG.dat")

    write_eeg_stack(tmp_path, 5, make_raw(30), bv, 1000)
    with open(tmp_path / "S1-T5_EEG.dat", "ab") as f:
        f.write(b"\x00\x00")  # one int16 too many: not a whole number of samples
    with pytest.raises(ValueError, match="whole number of samples"):
        load_eeg(tmp_path / "S1-T5_EEG.dat")

    write_eeg_stack(tmp_path, 6, make_raw(30), bv, 1000, ts=[0.0])
    with pytest.raises(ValueError, match="at least two timestamps"):
        load_eeg(tmp_path / "S1-T6_EEG.dat")

    with pytest.raises(ValueError, match="same number of stacks"):
        load_eeg(
            [tmp_path / "S1-T1_EEG.dat", tmp_path / "S1-T1_EEG.dat"],
            header_files=[tmp_path / "S1-T1_EEG.oebin"],
        )


# =====================================================================================
# helpers of the VR loaders
# =====================================================================================


def test_default_decoders_rising_and_falling_edges():
    sig = np.array([0, 0, 1, 1, 0, 0, 2, 0, 1, 1], dtype=float)
    onsets, values = _eeg._decode_aux_eeg(sig, 0.5, 2)
    assert onsets.tolist() == [2, 6, 8]  # 1-based index of the last low sample
    assert values.tolist() == [1.0, 2.0, 1.0]  # value at the first sample above threshold
    assert _eeg._decode_aux_offset_eeg(sig, 0.5, 2).tolist() == [4, 7]
    onsets, values = _eeg._decode_aux_eeg(np.zeros(5), 0.5, 2)
    assert onsets.size == 0 and values.size == 0


def test_smooth_span5_matches_matlab_definition():
    rng = np.random.default_rng(0)
    y = rng.normal(size=20)
    out = _eeg._smooth_span5(y)
    assert out[0] == y[0]
    np.testing.assert_allclose(out[1], y[:3].mean())
    np.testing.assert_allclose(out[2], y[:5].mean())
    for i in range(2, 18):
        np.testing.assert_allclose(out[i], y[i - 2 : i + 3].mean())
    np.testing.assert_allclose(out[-2], y[-3:].mean())
    assert out[-1] == y[-1]
    # shorter than the span: span shrinks to the largest odd length
    np.testing.assert_allclose(_eeg._smooth_span5(y[:3]), [y[0], y[:3].mean(), y[2]])
    np.testing.assert_allclose(_eeg._smooth_span5(y[:2]), y[:2])


def test_nearest_index_matches_brute_force_including_ties():
    ref = np.array([0.0, 1.0, 1.0, 2.0, 4.0, 4.0, 7.0])
    q = np.array([-3.0, 0.0, 0.5, 1.0, 1.5, 3.0, 4.0, 5.5, 7.0, 9.0])
    brute = [int(np.argmin(np.abs(v - ref))) for v in q]  # argmin: first index on ties
    assert _eeg._nearest_index(q, ref).tolist() == brute
    unsorted = np.array([2.0, 0.0, 1.0, 1.0])
    brute = [int(np.argmin(np.abs(v - unsorted))) for v in q]
    assert _eeg._nearest_index(q, unsorted).tolist() == brute
    rng = np.random.default_rng(1)
    r = np.sort(rng.integers(0, 50, 200).astype(float)) / 10
    qq = rng.integers(-5, 600, 500) / 100
    assert _eeg._nearest_index(qq, r).tolist() == [int(np.argmin(np.abs(v - r))) for v in qq]


def test_walking_direction_sign():
    ang = np.linspace(0, 6, 60)
    ccw = np.stack([np.cos(ang), 0 * ang, np.sin(ang)], axis=1)
    assert np.all(_eeg._walking_direction(ccw) == 1)
    assert np.all(_eeg._walking_direction(ccw[::-1]) == -1)
    still = np.tile(np.array([1.0, 0.0, 2.0]), (10, 1))
    still[:, 0] += np.arange(10) * 0  # no motion: angular velocity 0 -> direction 0
    assert np.all(_eeg._walking_direction(still) == 0)


def test_smoothed_time_diff_onsets_offsets_and_edges():
    t = np.cumsum([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])  # intervals 0.2 ... 0.8
    time = np.concatenate(([0.0], t))
    dt = np.diff(time)
    out = _eeg._smoothed_time_diff(time, np.array([2]), np.array([0, 5]))
    expect = dt.copy()
    expect[1:4] = expect[1:4].mean()  # onset 2 -> intervals 2..4 (1-based)
    expect[4:7] = expect[4:7].mean()  # offset 5 (the 0 offset is dropped, as in MATLAB)
    np.testing.assert_allclose(out, expect)
    # overlapping events see the values written by the earlier one (sequential loops)
    out = _eeg._smoothed_time_diff(time, np.array([1, 2]), np.array([]))
    e = dt.copy()
    e[0:3] = e[0:3].mean()
    e[1:4] = e[1:4].mean()
    np.testing.assert_allclose(out, e)
    # onset in the last intervals: average over what exists (MATLAB would error)
    last = _eeg._smoothed_time_diff(time, np.array([dt.size]), np.array([]))
    assert last[-1] == dt[-1]
    with pytest.raises(ValueError, match="outside"):
        _eeg._smoothed_time_diff(time, np.array([dt.size + 1]), np.array([]))
    with pytest.raises(ValueError, match="outside"):
        _eeg._smoothed_time_diff(time, np.array([0]), np.array([]))


# =====================================================================================
# load_heeg
# =====================================================================================

HEEG_KEYS = [
    "activity", "timeStamps", "event", "eventType", "audioVolume", "positionX", "positionY",
    "positionZ", "rotationX", "rotationY", "rotationZ", "speed", "walkingDirection",
]


def test_load_heeg_segment_alignment_and_vr_sampling(tmp_path):
    s = HeegSetup(tmp_path)
    h = load_heeg(s.txt, s.json, filter_eeg=False)

    assert list(h) == HEEG_KEYS  # MATLAB field order
    assert h.activity is h["activity"]
    n = s.segment.stop - s.segment.start
    assert n == 8501  # samples 499 .. 8999 (0-based): before the first rise through the last one
    assert h.activity.shape == (8, n)
    np.testing.assert_array_equal(h.activity, s.exg[s.segment].T)  # first..last onset, inclusive
    for k in HEEG_KEYS[1:]:
        assert h[k].shape == (1, n), k
    assert h.timeStamps[0, 0] == 0.0
    np.testing.assert_array_equal(h.timeStamps[0], s.eeg_time())

    idx = s.nearest()
    vt = s.vr_time()
    np.testing.assert_array_equal(h.event[0], s.mm[idx])
    np.testing.assert_array_equal(h.eventType[0], s.mismatch[idx])
    assert h.eventType.dtype == bool
    np.testing.assert_array_equal(h.audioVolume[0], s.audio[idx])
    for i, axis in enumerate("XYZ"):
        np.testing.assert_array_equal(h[f"position{axis}"][0], s.pos[idx, i])
        np.testing.assert_array_equal(h[f"rotation{axis}"][0], s.rot[idx, i])
    speed = s.speed(s.pos, vt)
    np.testing.assert_allclose(h.speed[0], speed[idx], rtol=1e-12)
    assert np.all(h.walkingDirection == 1)  # counter-clockwise trajectory
    assert "speed_participant" not in h


def test_load_heeg_clockwise_walking_direction(tmp_path):
    s = HeegSetup(tmp_path, clockwise=True)
    h = load_heeg(s.txt, s.json, filter_eeg=False)
    assert np.all(h.walkingDirection == -1)


def test_load_heeg_reference_tracking_replaces_position_and_adds_speed_participant(tmp_path):
    s = HeegSetup(tmp_path, reference=True)
    sync = s.sync

    def decode(sig, thr, par):  # stands in for decodeAuxEEG
        if np.array_equal(sig, sync):
            return _eeg._decode_aux_eeg(sig, thr, par)
        return np.array([[10], [40]]), np.array([1.0, 2.0])  # stimulus onsets (1-based)

    def decode_off(sig, thr, par):  # stands in for decodeAuxOffsetEEG
        return np.array([0, 25])  # the 0 is dropped like MATLAB's stimulusOffset<1

    h = load_heeg(s.txt, s.json, filter_eeg=False, decode_aux=decode, decode_aux_offset=decode_off)
    assert list(h) == [*HEEG_KEYS, "speed_participant"]

    idx = s.nearest()
    vt = s.vr_time()
    dt = np.diff(vt)
    for k in (10, 40, 25):  # onsets first, then offsets; windows do not overlap
        dt[k - 1 : k + 2] = dt[k - 1 : k + 2].mean()
    step = np.sqrt(np.sum(np.diff(s.ref_pos, axis=0) ** 2, axis=1))
    running = np.concatenate(([0.0], step / dt))
    np.testing.assert_allclose(h.speed_participant[0], running[idx], rtol=1e-12)
    for i, axis in enumerate("XYZ"):  # position/rotation hold the reference values
        np.testing.assert_array_equal(h[f"position{axis}"][0], s.ref_pos[idx, i])
        np.testing.assert_array_equal(h[f"rotation{axis}"][0], s.ref_rot[idx, i])
    # the headset speed still comes from `position`
    np.testing.assert_allclose(h.speed[0], s.speed(s.pos, vt)[idx], rtol=1e-12)


def test_load_heeg_default_decoders_with_reference_tracking_run(tmp_path):
    s = HeegSetup(tmp_path, reference=True)
    h = load_heeg(s.txt, s.json, filter_eeg=False)  # uses the assumed edge decoders
    assert h.speed_participant.shape == h.speed.shape
    assert np.isfinite(h.speed_participant).all()


def test_load_heeg_sync_fallback_for_startup_transient(tmp_path):
    # The analog channel starts high (start-up transient). The first-half maximum is then the
    # transient, the 3.3 V pulses are cut away and MATLAB's catch branch is used.
    s = HeegSetup(tmp_path, name="S1-T8")
    analog = s.analog.copy()
    analog[:100] = 5.0
    write_openbci(s.txt, s.exg, analog, s.stamp)
    assert _eeg._sync_edges(analog, analog[4:15000].max(), analog[4:15000].max() * 0.3).size == 0
    h = load_heeg(s.txt, s.json, filter_eeg=False)
    np.testing.assert_array_equal(h.activity, s.exg[s.segment].T)


def test_load_heeg_truncated_rows_are_skipped(tmp_path):
    s = HeegSetup(tmp_path)
    write_openbci(s.txt, s.exg, s.analog, s.stamp, extra_bad_row=True)  # partial line in file
    h = load_heeg(s.txt, s.json, filter_eeg=False)
    np.testing.assert_array_equal(h.activity, s.exg[s.segment].T)


def test_load_heeg_filters_remove_dc_line_noise_and_keep_band(tmp_path):
    n = 30000
    t = np.arange(n) / FS_HEEG
    sig = 100.0 + 20 * np.sin(2 * np.pi * 10 * t) + 20 * np.sin(2 * np.pi * 50 * t)
    sig += 20 * np.sin(2 * np.pi * 110 * t)
    exg = np.tile(sig[:, None], (1, 8)) + np.arange(8)[None, :] * 0
    s = HeegSetup(tmp_path, n_eeg=n, rises=(1000, 5000, 20000), exg=exg)
    h = load_heeg(s.txt, s.json, filter_eeg=True)
    assert h.activity.shape == (8, 19001) and h.activity.dtype == np.float64
    tt = t[s.segment]
    expect = 20 * np.sin(2 * np.pi * 10 * tt)  # zero phase: the 10 Hz component is unshifted
    interior = slice(3500, -3500)  # beyond half the filter length (about 2900 samples)
    for ch in range(8):
        err = np.max(np.abs(h.activity[ch, interior] - expect[interior]))
        assert err < 0.5, (ch, err)  # 60 dB stopband: DC 100 -> 0.1, 50 Hz and 110 Hz -> 0.02


def test_load_heeg_filter_rejects_short_segments(tmp_path):
    s = HeegSetup(tmp_path)
    with pytest.raises(ValueError, match="shorter than twice the FIR order"):
        load_heeg(s.txt, s.json, filter_eeg=True)
    assert load_heeg(s.txt, s.json, filter_eeg=False).activity.shape[1] == 8501


@pytest.mark.parametrize(
    "kind, edge, passband, stopband",
    [
        ("highpass", 1.0, [3.0, 10.0, 60.0], [0.0, 0.2, 0.8]),
        ("lowpass", 100.0, [5.0, 50.0, 99.0], [105.0, 110.0, 124.0]),
        ("bandstop", (49.0, 51.0), [10.0, 48.9, 51.1, 100.0], [49.5, 50.0, 50.5]),
    ],
)
def test_fir_taps_meet_documented_specification(kind, edge, passband, stopband):
    from scipy.signal import freqz

    taps = _eeg._fir_taps(kind, edge, FS_HEEG)
    assert taps.size % 2 == 1  # odd length: integer delay
    np.testing.assert_allclose(taps, taps[::-1])  # linear phase
    _, h = freqz(taps, worN=np.array(passband), fs=FS_HEEG)
    assert np.all(np.abs(np.abs(h) - 1) < 0.02), np.abs(h)
    _, h = freqz(taps, worN=np.array(stopband), fs=FS_HEEG)
    assert np.all(20 * np.log10(np.abs(h) + 1e-12) < -50), 20 * np.log10(np.abs(h))
    with pytest.raises(ValueError, match="unknown filter kind"):
        _eeg._fir_taps("bandpass", (1, 2), FS_HEEG)


def test_load_heeg_error_cases(tmp_path):
    s = HeegSetup(tmp_path)
    # missing sync column
    cols = [c if c != "Analog Channel 2" else "Analog Channel 9" for c in OPENBCI_COLUMNS]
    write_openbci(s.txt, s.exg, s.analog, s.stamp, columns=cols)
    with pytest.raises(ValueError, match="AnalogChannel2"):
        load_heeg(s.txt, s.json, filter_eeg=False)
    # no sync pulse in the EEG
    write_openbci(s.txt, s.exg, np.zeros(12000), s.stamp)
    with pytest.raises(ValueError, match="no sync pulse found in 'Analog Channel 2'"):
        load_heeg(s.txt, s.json, filter_eeg=False)
    write_openbci(s.txt, s.exg, s.analog, s.stamp)
    # no sync pulse in the VR log
    recs = [{**r, "synchroSignal": 0.0} for r in s.records]
    s.write_json(recs)
    with pytest.raises(ValueError, match="VR 'synchroSignal'"):
        load_heeg(s.txt, s.json, filter_eeg=False)
    # records with different fields
    recs = [dict(r) for r in s.records]
    del recs[5]["audioVolume"]
    s.write_json(recs)
    with pytest.raises(ValueError, match="record 6 has different fields"):
        load_heeg(s.txt, s.json, filter_eeg=False)
    # a text file without header and a file without rows
    s.write_json()
    s.txt.write_text("%only comments\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no header"):
        load_heeg(s.txt, s.json, filter_eeg=False)
    with pytest.raises(FileNotFoundError):
        load_heeg(tmp_path / "missing.txt", s.json)


# =====================================================================================
# load_heyetracking
# =====================================================================================

EYE_KEYS = [
    "timeStamps", "event", "eventType", "audioVolume", "positionX", "positionY", "positionZ",
    "rotationX", "rotationY", "rotationZ", "speed", "walkingDirection", "leftEyeBlink",
    "rightEyeBlink", "leftEyeRotationX", "leftEyeRotationY", "leftEyeRotationZ",
    "leftEyeRotationW", "rightEyeRotationX", "rightEyeRotationY", "rightEyeRotationZ",
    "rightEyeRotationW",
]


def test_load_heyetracking_fields_shapes_and_values(tmp_path):
    s = HeegSetup(tmp_path, n_eeg=2500, rises=(100, 200, 300), eye=True, name="S1-T9")
    h = load_heyetracking(s.json)
    assert list(h) == EYE_KEYS
    m = len(s.records)
    for k in EYE_KEYS:
        assert h[k].shape == (1, m), k
    np.testing.assert_array_equal(h.timeStamps[0], s.vr_t)  # raw VR timestamps, not shifted
    np.testing.assert_array_equal(h.event[0], s.mm)
    np.testing.assert_array_equal(h.eventType[0], s.mismatch)
    np.testing.assert_array_equal(h.positionZ[0], s.pos[:, 2])
    np.testing.assert_array_equal(h.rotationY[0], s.rot[:, 1])
    np.testing.assert_allclose(h.speed[0], s.speed(s.pos, s.vr_t), rtol=1e-12)
    assert np.all(h.walkingDirection == 1)
    np.testing.assert_array_equal(h.leftEyeBlink[0], s.eye_blink[0])
    np.testing.assert_array_equal(h.rightEyeBlink[0], s.eye_blink[1])
    for i, axis in enumerate("XYZW"):
        np.testing.assert_array_equal(h[f"leftEyeRotation{axis}"][0], s.eye_rot[:, i])
        np.testing.assert_array_equal(h[f"rightEyeRotation{axis}"][0], -s.eye_rot[:, i])


def test_load_heyetracking_without_eye_fields_gives_empty_arrays(tmp_path):
    s = HeegSetup(tmp_path, n_eeg=2500, rises=(100, 200, 300), eye=False, name="S1-T9")
    h = load_heyetracking(s.json)
    assert list(h) == EYE_KEYS
    for k in EYE_KEYS[12:]:
        assert h[k].shape == (0, 0), k  # MATLAB: [] initialised and never filled
    assert h.timeStamps.shape == (1, len(s.records))


def test_load_heyetracking_reference_tracking(tmp_path):
    s = HeegSetup(tmp_path, n_eeg=2500, rises=(100, 200, 300), reference=True, eye=True,
                  name="S1-T9")

    def decode(sig, thr, par):
        return np.array([5, 50]), np.array([1.0, 1.0])

    def decode_off(sig, thr, par):
        return np.array([20])

    h = load_heyetracking(s.json, decode_aux=decode, decode_aux_offset=decode_off)
    assert list(h) == [*EYE_KEYS, "speed_participant"]
    dt = np.diff(s.vr_t)
    for k in (5, 50, 20):
        dt[k - 1 : k + 2] = dt[k - 1 : k + 2].mean()
    step = np.sqrt(np.sum(np.diff(s.ref_pos, axis=0) ** 2, axis=1))
    np.testing.assert_allclose(h.speed_participant[0], np.concatenate(([0.0], step / dt)))
    np.testing.assert_array_equal(h.positionX[0], s.ref_pos[:, 0])
    np.testing.assert_array_equal(h.rotationZ[0], s.ref_rot[:, 2])


def test_load_heyetracking_errors(tmp_path):
    s = HeegSetup(tmp_path, n_eeg=2500, rises=(100, 200, 300), name="S1-T9")
    s.write_json([s.records[0]])
    with pytest.raises(ValueError, match="at least two records"):
        load_heyetracking(s.json)
    (tmp_path / "x.json").write_text('{"nothing": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="'records'"):
        load_heyetracking(tmp_path / "x.json")
    recs = [{k: v for k, v in r.items() if k != "rotation"} for r in s.records]
    s.write_json(recs)
    with pytest.raises(ValueError, match="rotation"):
        load_heyetracking(s.json)
    with pytest.raises(FileNotFoundError):
        load_heyetracking(tmp_path / "missing.json")


# =====================================================================================
# Reference tests against MATLAB r2003 (skipped until CALLIOPY_TESTDATA has the files)
# =====================================================================================


def _flat(a):
    return np.asarray(a, dtype=np.float64).ravel()


def test_load_eeg_matches_matlab(sample_file, matlab_ref):
    dat = sample_file("_EEG.dat")  # one stack; .oebin and .npy sit next to it
    ref = matlab_ref("load_EEG")
    res = load_eeg(dat)
    want = np.asarray(ref.out1)
    assert res.eeg_data.shape == want.shape
    # single-precision data; MATLAB resamples in single, here in double: allow 1e-4 of the range
    tol = 1e-4 * np.abs(want).max()
    np.testing.assert_allclose(res.eeg_data, want, rtol=0, atol=tol)
    np.testing.assert_allclose(res.eeg_sync.ravel(), _flat(ref.out2), rtol=0, atol=tol)
    np.testing.assert_array_equal(res.samps_per_stack.ravel(), _flat(ref.out3))


def test_load_heeg_matches_matlab(sample_file, matlab_ref):
    txt = sample_file(".txt")
    js = txt.with_suffix(".json")
    if not js.is_file():
        pytest.skip(f"no {js.name} next to {txt.name}")
    ref = matlab_ref("load_hEEG").out1
    h = load_heeg(txt, js)
    assert list(h) == list(ref)  # same fields in the same order
    for key in h:
        if key == "activity":
            continue
        np.testing.assert_allclose(
            _flat(h[key]), _flat(ref[key]), rtol=1e-9, atol=0, equal_nan=True, err_msg=key
        )
        assert np.shape(h[key]) == np.shape(ref[key]), key
    # The filters are rebuilt from MATLAB's documentation, not from its code (see _fir_taps):
    # until hEEG_filters below passes, require close agreement rather than equality.
    a, b = np.asarray(h.activity), np.asarray(ref["activity"])
    assert a.shape == b.shape
    rel = np.linalg.norm(a - b) / np.linalg.norm(b)
    assert rel < 0.05, f"filtered activity differs from MATLAB by {rel:.3f} (relative RMS)"


def test_load_heeg_filter_design_matches_matlab(matlab_ref):
    ref = matlab_ref("hEEG_filters")  # out1..out3: Coefficients of highpass, lowpass, bandstop
    for name, kind, edge in (
        ("out1", "highpass", 1.0), ("out2", "lowpass", 100.0), ("out3", "bandstop", (49.0, 51.0))
    ):
        want = _flat(ref[name])
        got = _eeg._fir_taps(kind, edge, FS_HEEG)
        assert got.size == want.size, f"{kind}: {got.size} taps, MATLAB {want.size}"
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-6, err_msg=kind)


def test_decoders_match_matlab_decodeauxeeg(sample_file, matlab_ref):
    js = sample_file(".txt").with_suffix(".json")  # the VR log of the hEEG sample
    if not js.is_file():
        pytest.skip(f"no {js.name} next to the .txt sample")
    # out1: sync onsets; out2, out3: stimulus onsets and values; out4: stimulus offsets
    ref = matlab_ref("decodeAuxEEG")
    records = _eeg._read_vr_records(js)
    sync = _eeg._column(records, "synchroSignal")
    mm = _eeg._column(records, "mm")
    onsets, _ = _eeg._decode_aux_eeg(sync, 0.5, 2)
    np.testing.assert_array_equal(onsets, _flat(ref.out1), err_msg="decodeAuxEEG(sync) onsets")
    onsets, values = _eeg._decode_aux_eeg(mm, 0.5, 2)
    np.testing.assert_array_equal(onsets, _flat(ref.out2), err_msg="decodeAuxEEG(mm) onsets")
    np.testing.assert_array_equal(values, _flat(ref.out3), err_msg="decodeAuxEEG(mm) values")
    off = _eeg._decode_aux_offset_eeg(mm, 0.5, 2)
    np.testing.assert_array_equal(off, _flat(ref.out4), err_msg="decodeAuxOffsetEEG(mm)")


def test_load_heyetracking_matches_matlab(sample_file, matlab_ref):
    js = sample_file("_eye.json")
    ref = matlab_ref("load_hEyeTracking").out1
    h = load_heyetracking(js)
    assert list(h) == list(ref)
    for key in h:
        np.testing.assert_allclose(
            _flat(h[key]), _flat(ref[key]), rtol=1e-9, atol=0, equal_nan=True, err_msg=key
        )
        assert np.shape(h[key]) == np.shape(ref[key]), key
