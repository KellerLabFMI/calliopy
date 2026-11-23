"""Readers for EEG and human-recording files (OpenEphys EEG, OpenBCI hEEG, VR eye tracking).

Three MATLAB loaders of calliope r2003 are ported here. Each MATLAB function looks the
files up by ExpID in ExpLog (``read_info_from_ExpLog``); the Python readers take the files
themselves. A later ``load_experiment(expid)`` builds the paths from the patterns below, where
``<animal folder>`` is ``<data_dir>/<pi>/<animalID>`` as resolved by ``read_info_from_ExpLog``
(``<userID>/<mouse_id>`` in the human loaders).

``load_eeg``  <-  ``load_EEG(ExpID, ExpLog)``
    Per stack of the ExpID (``ExpLog.stackid``), in the animal folder:
    ``S1-T<stackID>_EEG.dat``, ``S1-T<stackID>_EEG.oebin``, ``S1-T<stackID>_EEG.npy``.
``load_heeg``  <-  ``load_hEEG(ExpID, stackID, ExpLog, ws)``
    One stack: ``S1-T<stackID>.txt`` and ``S1-T<stackID>.json``.
``load_heyetracking``  <-  ``load_hEyeTracking(ExpID, stackID, ExpLog, ws)``
    One stack: ``S1-T<stackID>.json``.

Files that are not in the calliope r2003 sources
------------------------------------------------
``load_hEEG`` and ``load_hEyeTracking`` call ``decodeAuxEEG`` and ``decodeAuxOffsetEEG``,
and ``load_EEG`` calls ``readNPY``. None of them is part of calliope r2003 (they come from
the VR / OpenEphys tool chain on the lab path). ``readNPY`` is replaced by :func:`numpy.load`.
The two decoders cannot be ported without their source: this module uses a documented
*assumption* (rising / falling edges of a thresholded signal, see :func:`_decode_aux_eeg`)
and lets the caller pass the exact decoders as ``decode_aux`` / ``decode_aux_offset``.
``tests/matlab/ref_eeg.m`` saves the outputs of the original decoders so the assumption
can be checked.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable, Sequence
from fractions import Fraction
from os import PathLike
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.signal import fftconvolve, firwin, kaiserord, resample_poly

from ._mat import MatStruct

__all__ = [
    "EEG_CHANNEL_ORDER",
    "EEG_SYNC_CHANNEL",
    "EegData",
    "load_eeg",
    "load_heeg",
    "load_heyetracking",
]

_log = logging.getLogger(__name__)

#: Rows of the raw OpenEphys recording (0-based) that form ``eeg_data``, in output order.
#: MATLAB: ``EEGdata([26:32, flip(9:24), 1:7], :)`` (1-based), "Neuronexus probe mapping".
EEG_CHANNEL_ORDER: tuple[int, ...] = (
    *range(25, 32),  # MATLAB 26:32
    *range(23, 7, -1),  # MATLAB flip(9:24)
    *range(7),  # MATLAB 1:7
)
#: Row of the raw recording (0-based) that carries the sync signal (MATLAB channel 33).
EEG_SYNC_CHANNEL = 32

_EEG_TARGET_RATE_HZ = 1000  # load_EEG resamples to 1 kHz ...
_EEG_RESAMPLE_ABOVE_HZ = 2000  # ... when the estimated sample rate is above this

_HEEG_FS = 250.0  # OpenBCI Cyton sampling rate (hard-coded in load_hEEG)
_HEEG_N_CHANNELS = 8
_AUX_THRESHOLD = 0.5  # arguments MATLAB passes to decodeAuxEEG / decodeAuxOffsetEEG
_AUX_PARAM = 2

PathLikeStr = str | PathLike[str]


class EegData(NamedTuple):
    """Outputs of :func:`load_eeg` (MATLAB ``[EEGdata, EEGsync, SampsPerStack]``).

    Attributes
    ----------
    eeg_data : numpy.ndarray
        ``[30 x nSamples]`` float32, microvolts, rows in Neuronexus probe order
        (see :data:`EEG_CHANNEL_ORDER`), stacks appended along the second dimension.
    eeg_sync : numpy.ndarray
        ``[1 x nSamples]`` float32, sync signal (raw channel 33, same scaling and sample rate).
    samps_per_stack : numpy.ndarray
        ``[1 x nStacks]`` int64, number of (resampled) samples contributed by each stack.
    """

    eeg_data: np.ndarray
    eeg_sync: np.ndarray
    samps_per_stack: np.ndarray


# =====================================================================================
# load_eeg  <-  load_EEG.m
# =====================================================================================


def load_eeg(
    eeg_files: PathLikeStr | Iterable[PathLikeStr],
    *,
    header_files: PathLikeStr | Iterable[PathLikeStr] | None = None,
    timestamp_files: PathLikeStr | Iterable[PathLikeStr] | None = None,
) -> EegData:
    """Load EEG recorded with the OpenEphys GUI (``load_EEG.m``).

    Reads the binary ``.dat`` file of each stack through a memory map, scales every
    channel to microvolts, resamples recordings above 2 kHz to 1 kHz, appends the stacks,
    splits off the sync channel and puts the EEG channels into probe order.

    MATLAB reads, for every stack ``<stackID>`` of the ExpID, in the animal folder::

        S1-T<stackID>_EEG.dat      int16, little endian, channels interleaved
        S1-T<stackID>_EEG.oebin    JSON header: continuous.num_channels,
                                   continuous.channels(i).bit_volts
        S1-T<stackID>_EEG.npy      sample timestamps in seconds (numpy file)

    Parameters
    ----------
    eeg_files : path or iterable of paths
        The ``S1-T<stackID>_EEG.dat`` file of each stack, in stack order.
    header_files : path or iterable of paths, optional
        The ``.oebin`` file of each stack. Default: the ``.dat`` path with the suffix
        replaced by ``.oebin``.
    timestamp_files : path or iterable of paths, optional
        The ``.npy`` timestamp file of each stack. Default: the ``.dat`` path with the suffix
        replaced by ``.npy``.

    Returns
    -------
    EegData
        ``eeg_data`` ``[30 x nSamples]`` float32, ``eeg_sync`` ``[1 x nSamples]`` float32,
        ``samps_per_stack`` ``[1 x nStacks]`` int64 (MATLAB: ``EEGdata``, ``EEGsync``,
        ``SampsPerStack``).

    Raises
    ------
    ValueError
        No stacks; header without exactly one continuous stream; fewer than 33 channels;
        channel counts that differ between stacks; a ``.dat`` size that is not a multiple of
        ``num_channels`` int16 values; timestamps from which no sample rate can be estimated.
    FileNotFoundError
        A ``.dat``, ``.oebin`` or ``.npy`` file does not exist.

    Notes
    -----
    Per stack: ``int16`` values are scaled in single precision by ``bit_volts`` of their
    channel. The sample rate is ``round(1 / median(diff(timestamps)))``; above 2000 Hz every
    channel is resampled to 1000 Hz with a polyphase anti-aliasing FIR filter (Kaiser window,
    ``beta = 5``, ``10 * max(p, q)`` taps per side), which is what MATLAB ``resample`` designs
    (``scipy.signal.resample_poly`` with its default Kaiser window). MATLAB resamples in
    single precision; here the filter runs in double precision and the result is cast back
    to float32, so values agree to about 1e-6 of the signal range. Only the 31 channels that
    reach the output are read and resampled; filtering is channel-wise, so the values are
    identical to resampling all channels.

    Differences from MATLAB
    -----------------------
    * Takes file paths; no ExpID/ExpLog lookup, no ``disp`` output (progress goes to
      :mod:`logging`).
    * MATLAB ``resample(x, 1, rate/1000)`` fails when the rate is not a multiple of 1 kHz
      (for example 2500 Hz); here the ratio ``1000/rate`` is reduced to lowest terms.
    * Channel counts that differ between stacks, fewer than 33 channels, a header with
      several continuous streams and a ``.dat`` of the wrong size raise a clear
      :class:`ValueError` instead of a MATLAB indexing or ``reshape`` error.
    * Only the channels that reach the output are converted (MATLAB converts all).
    * The data are a normal in-memory array; the memory map is only used while reading and
      is released before returning.
    """
    dat_files = _as_path_list(eeg_files, "eeg_files")
    if not dat_files:
        raise ValueError("eeg_files is empty: load_eeg needs at least one stack")
    if header_files is None:
        hdr_files = [p.with_suffix(".oebin") for p in dat_files]
    else:
        hdr_files = _as_path_list(header_files, "header_files")
    if timestamp_files is None:
        ts_files = [p.with_suffix(".npy") for p in dat_files]
    else:
        ts_files = _as_path_list(timestamp_files, "timestamp_files")
    if not len(dat_files) == len(hdr_files) == len(ts_files):
        raise ValueError(
            "eeg_files, header_files and timestamp_files must list the same number of stacks"
        )

    rows = [*EEG_CHANNEL_ORDER, EEG_SYNC_CHANNEL]
    parts: list[np.ndarray] = []
    n_chans_first: int | None = None
    for dat, hdr, tsf in zip(dat_files, hdr_files, ts_files, strict=True):
        num_chans, bit_volts = _read_oebin(hdr)
        if n_chans_first is None:
            n_chans_first = num_chans
        elif num_chans != n_chans_first:
            raise ValueError(
                f"{dat.name}: {num_chans} channels, but the first stack has {n_chans_first}; "
                "stacks of one ExpID must have the same number of channels"
            )
        if num_chans <= EEG_SYNC_CHANNEL:
            raise ValueError(
                f"{dat.name}: {num_chans} channels, but load_eeg needs at least "
                f"{EEG_SYNC_CHANNEL + 1} (sync is channel {EEG_SYNC_CHANNEL + 1})"
            )
        rate = _estimate_sample_rate(tsf)
        _log.info("loading %s (%d channels, %d Hz)", dat, num_chans, rate)
        parts.append(_read_eeg_stack(dat, num_chans, bit_volts, rate, rows))

    samps = np.array([[p.shape[1] for p in parts]], dtype=np.int64)
    all_rows = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
    eeg_sync = np.ascontiguousarray(all_rows[-1:, :])
    eeg_data = np.ascontiguousarray(all_rows[:-1, :])
    return EegData(eeg_data, eeg_sync, samps)


def _as_path_list(value: PathLikeStr | Iterable[PathLikeStr], what: str) -> list[Path]:
    if isinstance(value, (str, PathLike)):
        return [Path(value)]
    try:
        return [Path(v) for v in value]
    except TypeError:
        raise TypeError(f"{what} must be a path or an iterable of paths") from None


def _read_oebin(path: Path) -> tuple[int, np.ndarray]:
    """Number of channels and per-channel ``bit_volts`` (float32) from an ``.oebin`` header."""
    with open(path, encoding="utf-8-sig") as f:
        header = json.load(f)
    cont = header.get("continuous") if isinstance(header, dict) else None
    if isinstance(cont, dict):
        cont = [cont]
    if not isinstance(cont, list) or len(cont) != 1:
        n = len(cont) if isinstance(cont, list) else 0
        raise ValueError(
            f"{path.name}: expected exactly one entry in 'continuous', found {n} "
            "(load_EEG assumes a single continuous stream)"
        )
    stream = cont[0]
    try:
        num_chans = int(stream["num_channels"])
        channels = stream["channels"]
        volts = [float(channels[i]["bit_volts"]) for i in range(num_chans)]
    except (KeyError, IndexError, TypeError) as err:
        raise ValueError(
            f"{path.name}: header lacks continuous.num_channels or a bit_volts entry for "
            "each channel"
        ) from err
    if num_chans < 1:
        raise ValueError(f"{path.name}: num_channels is {num_chans}")
    # MATLAB multiplies single data by a double scalar, which is carried out in single.
    return num_chans, np.asarray(volts, dtype=np.float32)


def _estimate_sample_rate(ts_file: Path) -> int:
    """``round(1 / median(diff(timestamps)))`` of an OpenEphys timestamp file (seconds)."""
    ts = np.load(ts_file, allow_pickle=False).ravel()
    if ts.size < 2:
        raise ValueError(f"{ts_file.name}: need at least two timestamps to estimate the rate")
    step = float(np.median(np.diff(ts.astype(np.float64))))
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"{ts_file.name}: timestamps are not increasing (median step {step})")
    return int(np.floor(1.0 / step + 0.5))  # MATLAB round: halves away from zero


def _read_eeg_stack(
    dat: Path, num_chans: int, bit_volts: np.ndarray, rate: int, rows: Sequence[int]
) -> np.ndarray:
    """Scaled (and resampled) ``rows`` of one ``.dat`` file as ``[len(rows) x n]`` float32."""
    size = dat.stat().st_size  # raises FileNotFoundError like MATLAB
    frame = 2 * num_chans
    if size == 0 or size % frame:
        raise ValueError(
            f"{dat.name}: {size} bytes is not a whole number of samples of {num_chans} "
            "int16 channels"
        )
    n = size // frame
    # MATLAB: reshape(memmapfile(..., 'int16').Data, [numChans, n]) -> channels fastest
    mm = np.memmap(dat, dtype="<i2", mode="r", shape=(num_chans, n), order="F")
    ratio = Fraction(_EEG_TARGET_RATE_HZ, rate) if rate > _EEG_RESAMPLE_ABOVE_HZ else None
    out: list[np.ndarray] = []
    try:
        for ch in rows:
            x = np.asarray(mm[ch, :], dtype=np.float32) * bit_volts[ch]
            if ratio is not None:
                x = resample_poly(x.astype(np.float64), ratio.numerator, ratio.denominator)
                x = x.astype(np.float32)
            out.append(x)
    finally:
        del mm  # release the file handle (matters on Windows)
    return np.stack(out, axis=0)


# =====================================================================================
# Shared helpers of the VR (MetaQuest) JSON loaders
# =====================================================================================

Decoder = Callable[[np.ndarray, float, float], tuple[np.ndarray, np.ndarray]]
OffsetDecoder = Callable[[np.ndarray, float, float], np.ndarray]


def _decode_aux_eeg(
    signal: np.ndarray, threshold: float, param: float
) -> tuple[np.ndarray, np.ndarray]:
    """ASSUMED behaviour of MATLAB ``decodeAuxEEG(signal, threshold, param)``.

    The MATLAB source is not part of calliope r2003, so this is an assumption derived from
    how ``load_hEEG`` uses it: pulse onsets of a thresholded signal. ``onsets`` holds the
    1-based index ``k`` of the last sample at or below ``threshold`` before each rising edge
    (``find(diff(signal > threshold) == 1)``, the same convention ``load_hEEG`` uses for the
    EEG sync channel); ``values`` is the signal at the first sample above the threshold.
    ``param`` (2 in MATLAB) is not used.
    """
    above = np.asarray(signal, dtype=np.float64) > threshold
    onsets = np.flatnonzero(np.diff(above.astype(np.int8)) == 1) + 1
    values = np.asarray(signal)[onsets] if onsets.size else np.empty(0)
    return onsets.astype(np.int64), values


def _decode_aux_offset_eeg(signal: np.ndarray, threshold: float, param: float) -> np.ndarray:
    """ASSUMED behaviour of MATLAB ``decodeAuxOffsetEEG(signal, threshold, param)``.

    1-based index ``k`` of the last sample above ``threshold`` before each falling edge
    (``find(diff(signal > threshold) == -1)``). ``param`` is not used.
    """
    above = np.asarray(signal, dtype=np.float64) > threshold
    return (np.flatnonzero(np.diff(above.astype(np.int8)) == -1) + 1).astype(np.int64)


def _read_vr_records(path: PathLikeStr) -> list[dict]:
    """``jsondecode(fileread(path)).records`` as a list of dicts with identical keys."""
    path = Path(path)
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "records" not in data:
        raise ValueError(f"{path.name}: expected a JSON object with a 'records' list")
    records = data["records"]
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path.name}: 'records' is empty")
    if len(records) < 2:
        raise ValueError(f"{path.name}: at least two records are needed (speed and direction)")
    keys = set(records[0])
    for i, rec in enumerate(records):
        if not isinstance(rec, dict) or set(rec) != keys:
            # jsondecode would return a cell array here and struct2table would fail.
            raise ValueError(f"{path.name}: record {i + 1} has different fields than record 1")
    return records


def _column(records: list[dict], key: str, what: str = "") -> np.ndarray:
    """One field of every record as a 1-D array: float64, bool (all logical) or object."""
    try:
        vals = [r[key] for r in records]
    except KeyError:
        raise ValueError(f"records have no field {what or key!r}") from None
    if all(isinstance(v, bool) for v in vals):
        return np.asarray(vals, dtype=bool)
    if all(v is None or isinstance(v, (int, float)) for v in vals):
        return np.array([np.nan if v is None else float(v) for v in vals], dtype=np.float64)
    return np.asarray(vals, dtype=object)


def _components(records: list[dict], key: str, comps: str = "xyz") -> np.ndarray:
    """``[nRecords x len(comps)]`` float64 from a nested field such as ``position.{x,y,z}``."""
    try:
        return np.array(
            [[np.nan if r[key][c] is None else float(r[key][c]) for c in comps] for r in records],
            dtype=np.float64,
        )
    except (KeyError, TypeError):
        raise ValueError(f"records need a field {key!r} with components {tuple(comps)}") from None


def _row(values: np.ndarray) -> np.ndarray:
    """MATLAB row vector ``[1 x n]``."""
    return np.asarray(values).reshape(1, -1)


def _speed(pos: np.ndarray, time: np.ndarray) -> np.ndarray:
    """``[0, sqrt(dx^2+dy^2+dz^2) ./ dt]`` (length of ``time``)."""
    d = np.diff(pos, axis=0)
    step = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.concatenate(([0.0], step / np.diff(time)))


def _smooth_span5(y: np.ndarray) -> np.ndarray:
    """MATLAB ``smooth(y)``: moving average, span 5, shorter symmetric windows at the ends."""
    n = y.size
    span = 5 if n >= 5 else (n if n % 2 else n - 1)
    half = span // 2
    out = np.array(y, dtype=np.float64)
    if n >= span:
        out[half : n - half] = np.convolve(y, np.full(span, 1.0 / span), mode="valid")
    for i in range(half):  # window half-width i at the start, and likewise at the end
        out[i] = np.mean(y[: 2 * i + 1])
        out[n - 1 - i] = np.mean(y[n - 1 - 2 * i :])
    return out


def _walking_direction(pos: np.ndarray) -> np.ndarray:
    """Sign of the smoothed angular velocity around the trajectory centre (x-z plane).

    MATLAB: ``theta = atan2(Z - mean(Z), X - mean(X))``; ``unwrap``; ``smooth(diff(...))``;
    sign; first value repeated. Result is ``[nRecords]`` of -1, 0 or +1 (NaN stays NaN).
    """
    x, z = pos[:, 0], pos[:, 2]
    theta = np.unwrap(np.arctan2(z - np.mean(z), x - np.mean(x)))
    d_theta = _smooth_span5(np.diff(theta))
    d_theta = np.sign(d_theta)
    return np.concatenate(([d_theta[0]], d_theta))


def _smoothed_time_diff(
    time: np.ndarray, onsets: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """Frame intervals with the three intervals at each stimulus onset / offset averaged.

    ``onsets`` and ``offsets`` are 1-based indices into ``diff(time)`` as MATLAB uses them.
    Onsets are processed first, then offsets, one event after the other (later events see
    the values written by earlier ones), exactly as in the MATLAB loops.
    """
    dt = np.diff(time).astype(np.float64)
    n = dt.size
    offsets = np.asarray(offsets).ravel()
    offsets = offsets[offsets >= 1]  # MATLAB: stimulusOffset(stimulusOffset<1) = []
    for group in (np.asarray(onsets).ravel(), offsets.ravel()):
        for k in group:
            k = int(k)
            if k < 1 or k > n:
                raise ValueError(f"stimulus index {k} outside the {n} frame intervals")
            window = slice(k - 1, min(k + 2, n))  # MATLAB timeDiff(k:k+2)
            dt[window] = np.mean(dt[window])
    return dt


def _nearest_index(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """``[~, i] = min(abs(q - ref))`` for every ``q``: 0-based index, first one on ties."""
    query = np.asarray(query, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if not np.isnan(ref).any() and np.all(np.diff(ref) >= 0):
        hi = np.clip(np.searchsorted(ref, query, side="left"), 0, ref.size - 1)
        lo_value = ref[np.clip(hi - 1, 0, ref.size - 1)]
        lo = np.searchsorted(ref, lo_value, side="left")  # first of equal values
        use_lo = (hi > 0) & (np.abs(query - ref[lo]) <= np.abs(query - ref[hi]))
        return np.where(use_lo, lo, hi).astype(np.int64)
    out = np.empty(query.size, dtype=np.int64)
    for start in range(0, query.size, 2048):
        block = np.abs(query[start : start + 2048, None] - ref[None, :])
        out[start : start + 2048] = np.nanargmin(block, axis=1)
    return out


def _vr_kinematics(
    records: list[dict],
    time: np.ndarray,
    decode_aux: Decoder,
    decode_aux_offset: OffsetDecoder,
) -> dict[str, np.ndarray | None]:
    """Speed, walking direction and (with reference tracking) participant speed per record."""
    pos = _components(records, "position")
    speed_vr = _speed(pos, time)
    direction = _walking_direction(pos)
    speed_running = None
    if "referencePosition" in records[0]:
        ref = _components(records, "referencePosition")
        step = np.diff(ref, axis=0)
        position_diffs = np.sqrt(step[:, 0] ** 2 + step[:, 1] ** 2 + step[:, 2] ** 2)
        mm = _column(records, "mm")
        onsets, _ = decode_aux(mm, _AUX_THRESHOLD, _AUX_PARAM)
        offsets = decode_aux_offset(mm, _AUX_THRESHOLD, _AUX_PARAM)
        time_diff = _smoothed_time_diff(time, np.asarray(onsets), np.asarray(offsets))
        with np.errstate(divide="ignore", invalid="ignore"):
            speed_running = np.concatenate(([0.0], position_diffs / time_diff))
    return {"pos": pos, "speed": speed_vr, "direction": direction, "speed_running": speed_running}


def _vr_fields(
    records: list[dict], index: np.ndarray, kin: dict[str, np.ndarray | None]
) -> list[tuple[str, np.ndarray]]:
    """The common ``hEEGdata`` fields, sampled at record ``index`` (0-based), in MATLAB order."""
    rot = _components(records, "rotation")
    pos = kin["pos"]
    return [
        ("event", _row(_column(records, "mm")[index])),
        ("eventType", _row(_column(records, "mismatch")[index])),
        ("audioVolume", _row(_column(records, "audioVolume")[index])),
        ("positionX", _row(pos[index, 0])),
        ("positionY", _row(pos[index, 1])),
        ("positionZ", _row(pos[index, 2])),
        ("rotationX", _row(rot[index, 0])),
        ("rotationY", _row(rot[index, 1])),
        ("rotationZ", _row(rot[index, 2])),
        ("speed", _row(kin["speed"][index])),
        ("walkingDirection", _row(kin["direction"][index])),
    ]


def _apply_reference_fields(
    out: MatStruct, records: list[dict], index: np.ndarray, kin: dict[str, np.ndarray | None]
) -> None:
    """With reference tracking, position/rotation hold the reference values (as in MATLAB)."""
    if kin["speed_running"] is None:
        return
    ref_pos = _components(records, "referencePosition")
    ref_rot = _components(records, "referenceRotation")
    for i, axis in enumerate("XYZ"):
        out[f"position{axis}"] = _row(ref_pos[index, i])
        out[f"rotation{axis}"] = _row(ref_rot[index, i])
    out["speed_participant"] = _row(kin["speed_running"][index])


# =====================================================================================
# load_heeg  <-  load_hEEG.m
# =====================================================================================


def load_heeg(
    eeg_file: PathLikeStr,
    json_file: PathLikeStr,
    *,
    filter_eeg: bool = True,
    decode_aux: Decoder | None = None,
    decode_aux_offset: OffsetDecoder | None = None,
) -> MatStruct:
    """Load human EEG (OpenBCI) together with the VR (MetaQuest 3) log of one stack.

    Port of ``load_hEEG.m``: cuts the EEG between the first and last sync pulse, filters it,
    shifts the VR time axis so that its first sync pulse is time 0, and samples the VR
    variables at the EEG sample times (nearest VR record).

    MATLAB reads, for the stack ``<stackID>``, in the animal folder (``<userID>/<mouse_id>``)::

        S1-T<stackID>.txt     OpenBCI raw text, comma separated; columns used:
                              2-9 (EEG channels 1-8), 'Analog Channel 2' (sync), 'Timestamp'
        S1-T<stackID>.json    VR log: {"records": [...]} with fields timestamp, synchroSignal,
                              mm, mismatch, audioVolume, position{x,y,z}, rotation{x,y,z} and,
                              optionally, referencePosition{x,y,z}, referenceRotation{x,y,z}

    Parameters
    ----------
    eeg_file : path
        The ``S1-T<stackID>.txt`` file.
    json_file : path
        The ``S1-T<stackID>.json`` file.
    filter_eeg : bool, default True
        Apply MATLAB's filter chain (high-pass 1 Hz, low-pass 100 Hz, band-stop 49-51 Hz,
        fs = 250 Hz). ``False`` returns the unfiltered segment (``activity`` in the file's
        units, normally microvolts).
    decode_aux, decode_aux_offset : callable, optional
        Ports of MATLAB ``decodeAuxEEG(signal, 0.5, 2)`` -> ``(onsets, values)`` and
        ``decodeAuxOffsetEEG(signal, 0.5, 2)`` -> ``offsets``, both returning **1-based**
        indices like MATLAB. They are not part of calliope r2003; the defaults are an
        assumption (rising/falling edges of the thresholded signal, see
        :func:`_decode_aux_eeg`) and must be checked against the originals.

    Returns
    -------
    MatStruct
        MATLAB's workspace variable ``hEEGdata``. Row vectors keep MATLAB's ``[1 x n]`` shape,
        ``n`` = number of EEG samples between the first and the last sync pulse:

        ``activity`` ``[8 x n]`` float64; ``timeStamps`` seconds from the first sample;
        ``event`` (``mm``), ``eventType`` (``mismatch``), ``audioVolume``, ``positionX/Y/Z``,
        ``rotationX/Y/Z``, ``speed`` and ``walkingDirection`` (-1, 0, +1) of the VR record
        nearest in time. With reference tracking (``referencePosition`` in the log),
        ``positionX/Y/Z`` and ``rotationX/Y/Z`` are replaced by the reference values and
        ``speed_participant`` is added (field order as in MATLAB).

    Raises
    ------
    ValueError
        Missing columns or fields, no sync pulse in EEG or VR, fewer than 10 EEG samples,
        records with different fields, or (``filter_eeg=True``) a segment shorter than twice
        the FIR filter order.

    Notes
    -----
    *EEG sync.* ``AnalogChannel2`` is thresholded at 0.7 of the maximum of its first half
    (``5:end/2``), clipped to 1, and the segment runs from the sample before the first
    rising edge through the sample before the last one (MATLAB indexing
    ``onset(1):onset(end)``). If no edge is found, MATLAB's ``catch`` branch repeats this with
    the maximum of the whole signal (from sample 5) for the 30 % margin and of samples 150 and
    later for the reference; that fallback is kept.

    *Filters.* MATLAB calls the Signal Processing Toolbox ``highpass``, ``lowpass`` and
    ``bandstop`` with defaults (60 dB stopband attenuation, steepness 0.85, automatic
    FIR/IIR choice). Their design code is not public; :func:`_fir_taps` rebuilds the
    documented specification as a Kaiser-window minimum-order FIR with zero-phase delay
    compensation (what ``auto`` chooses for segments at least twice the filter order, about
    12 000 samples = 48 s here). Filtered values therefore agree closely but **not exactly**
    with MATLAB until ``tests/matlab/ref_eeg.m`` filter coefficients confirm the design.

    Differences from MATLAB
    -----------------------
    * Takes file paths; returns ``hEEGdata`` instead of ``assignin(ws, ...)``; no ExpLog.
    * ``readtable`` is replaced by a small reader that skips ``%`` comment lines and the
      header line and keeps only the 10 columns used; rows that are too short are skipped.
    * Nearest-record search is vectorised (same result, first index on ties); MATLAB's
      quadratic loop is not copied.
    * Segments shorter than twice the filter order raise :class:`ValueError`; MATLAB would
      switch to an IIR design (not ported). Use ``filter_eeg=False`` for such segments.
    * Stimulus onsets in the last two frame intervals average over the intervals that exist;
      MATLAB fails with an index error.
    * Unused MATLAB code (the ``ExpLog`` stack search, the ``ss = 5`` assignments, the
      ``try`` around ``walkingDirection``) is dropped.
    """
    decode_aux = decode_aux or _decode_aux_eeg
    decode_aux_offset = decode_aux_offset or _decode_aux_offset_eeg

    exg, analog, stamp = _read_openbci(Path(eeg_file))
    records = _read_vr_records(json_file)

    onsets = _eeg_sync_onsets(analog)
    seg = slice(int(onsets[0]), int(onsets[-1]) + 1)
    n = seg.stop - seg.start
    _log.info("EEG segment between first and last sync pulse: %d samples", n)

    activity = np.ascontiguousarray(exg[seg, :].T)
    if filter_eeg:
        activity = _matlab_filter_chain(activity, _HEEG_FS)
    eeg_time = stamp[seg] - stamp[seg][0]

    sync_vr = _column(records, "synchroSignal")
    onsets_vr, _ = decode_aux(sync_vr, _AUX_THRESHOLD, _AUX_PARAM)
    onsets_vr = np.asarray(onsets_vr).ravel()
    if onsets_vr.size == 0:
        raise ValueError("no sync pulse found in the VR 'synchroSignal'")
    vr_stamps = _column(records, "timestamp").astype(np.float64)
    vr_time = vr_stamps - vr_stamps[int(onsets_vr[0]) - 1]  # 1-based onset -> 0-based record

    kin = _vr_kinematics(records, vr_time, decode_aux, decode_aux_offset)
    index = _nearest_index(eeg_time, vr_time)

    out = MatStruct()
    out["activity"] = activity
    out["timeStamps"] = _row(eeg_time)
    for key, value in _vr_fields(records, index, kin):
        out[key] = value
    _apply_reference_fields(out, records, index, kin)
    return out


def _read_openbci(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EEG channels 1-8, ``AnalogChannel2`` and ``Timestamp`` of an OpenBCI raw text file.

    MATLAB ``readtable`` turns the header names into variable names by dropping blanks
    ("Analog Channel 2" -> ``AnalogChannel2``). The EEG channels are columns 2-9 by position.
    """
    skip = 0
    header = None
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            skip += 1
            text = line.strip()
            if text and not text.startswith("%"):
                header = text
                break
    if header is None:
        raise ValueError(f"{path.name}: no header line found")
    names = [re.sub(r"[^0-9A-Za-z]", "", h) for h in header.split(",")]
    if len(names) < 1 + _HEEG_N_CHANNELS:
        raise ValueError(f"{path.name}: header has {len(names)} columns, expected at least 9")
    cols = []
    for want in ("AnalogChannel2", "Timestamp"):
        if want not in names:
            raise ValueError(f"{path.name}: no column {want!r} in the header")
        cols.append(names.index(want))
    if not all(n.startswith("EXG") for n in names[1 : 1 + _HEEG_N_CHANNELS]):
        _log.warning("%s: columns 2-9 are not named 'EXG Channel ...'", path.name)
    use = [*range(1, 1 + _HEEG_N_CHANNELS), *cols]
    try:
        data = np.loadtxt(
            path, delimiter=",", skiprows=skip, usecols=use, dtype=np.float64,
            comments=None, ndmin=2, encoding="utf-8",
        )
    except ValueError:
        data = _read_rows_tolerant(path, skip, use)
    if data.shape[0] < 10:
        raise ValueError(f"{path.name}: {data.shape[0]} data rows, need at least 10")
    return data[:, :_HEEG_N_CHANNELS], data[:, _HEEG_N_CHANNELS], data[:, _HEEG_N_CHANNELS + 1]


def _read_rows_tolerant(path: Path, skip: int, use: list[int]) -> np.ndarray:
    """Fallback reader: skips rows that are too short or not numeric (logged)."""
    rows, bad = [], 0
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for i, line in enumerate(f):
            if i < skip or not line.strip():
                continue
            parts = line.split(",")
            try:
                rows.append([float(parts[c]) for c in use])
            except (ValueError, IndexError):
                bad += 1
    if bad:
        _log.warning("%s: skipped %d malformed rows", path.name, bad)
    return np.array(rows, dtype=np.float64).reshape(-1, len(use))


def _sync_edges(analog: np.ndarray, reference_max: float, margin: float) -> np.ndarray:
    """0-based index of the sample before each rising edge of the cleaned sync signal."""
    sync = analog.copy()
    sync[analog < reference_max - margin] = 0
    sync[sync > 1] = 1
    return np.flatnonzero(np.diff((sync > 0.5).astype(np.int8)) == 1)


def _eeg_sync_onsets(analog: np.ndarray) -> np.ndarray:
    """Sync onsets of ``load_hEEG`` (0-based = MATLAB ``find(diff(sync>0.5)==1) - 1``)."""
    n = analog.size
    first_half_max = np.nanmax(analog[4 : n // 2])  # MATLAB (5:length/2)
    onsets = _sync_edges(analog, first_half_max, first_half_max * 0.3)
    if onsets.size == 0:  # MATLAB catch branch
        if n < 150:
            raise ValueError("no sync pulse found in 'Analog Channel 2'")
        onsets = _sync_edges(analog, np.nanmax(analog[149:]), np.nanmax(analog[4:]) * 0.3)
    if onsets.size == 0:
        raise ValueError("no sync pulse found in 'Analog Channel 2'")
    return onsets


def _fir_taps(kind: str, fpass, fs: float, steepness: float = 0.85, atten_db: float = 60.0):
    """Kaiser-window minimum-order FIR approximating MATLAB ``highpass``/``lowpass``/``bandstop``.

    Specification from the Signal Processing Toolbox documentation: stopband attenuation
    ``atten_db``; transition width ``(0.99 - 0.98 * steepness)`` times ``fpass`` (high-pass),
    ``fN - fpass`` (low-pass) or the distance of each band edge to the band centre (band-stop,
    whose ``fpass`` is the *stopped* band). The window design places the cut-off in the middle
    of the transition band; an odd length (even order) keeps the delay an integer.
    """
    nyq = fs / 2.0
    k = 0.99 - 0.98 * steepness
    if kind == "highpass":
        width = k * fpass
        cutoff, pass_zero = fpass - width / 2, False
    elif kind == "lowpass":
        width = k * (nyq - fpass)
        cutoff, pass_zero = fpass + width / 2, True
    elif kind == "bandstop":
        lo, hi = fpass
        centre = (lo + hi) / 2
        width = min(k * (centre - lo), k * (hi - centre))
        cutoff, pass_zero = [lo + width / 2, hi - width / 2], True
    else:
        raise ValueError(f"unknown filter kind {kind!r}")
    numtaps, beta = kaiserord(atten_db, width / nyq)
    numtaps += 1 - numtaps % 2  # odd length
    return firwin(numtaps, cutoff, window=("kaiser", beta), pass_zero=pass_zero, fs=fs)


def _zero_phase_fir(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Filter along the last axis and remove the group delay (zero padding at both ends)."""
    order = taps.size - 1
    if x.shape[-1] < 2 * order:
        raise ValueError(
            f"segment of {x.shape[-1]} samples is shorter than twice the FIR order ({order}); "
            "MATLAB would switch to an IIR design (not ported): use filter_eeg=False"
        )
    half = order // 2
    return fftconvolve(x, taps[None, :], mode="full", axes=-1)[..., half : half + x.shape[-1]]


def _matlab_filter_chain(activity: np.ndarray, fs: float) -> np.ndarray:
    """High-pass 1 Hz, low-pass 100 Hz, band-stop [49 51] Hz, applied in this order."""
    out = activity
    for kind, edge in (("highpass", 1.0), ("lowpass", 100.0), ("bandstop", (49.0, 51.0))):
        out = _zero_phase_fir(out, _fir_taps(kind, edge, fs))
    return out


# =====================================================================================
# load_heyetracking  <-  load_hEyeTracking.m
# =====================================================================================


def load_heyetracking(
    json_file: PathLikeStr,
    *,
    decode_aux: Decoder | None = None,
    decode_aux_offset: OffsetDecoder | None = None,
) -> MatStruct:
    """Load the VR / eye-tracking log of one stack (MetaQuest 3).

    Port of ``load_hEyeTracking.m``. MATLAB reads, in the animal folder
    (``<userID>/<mouse_id>``)::

        S1-T<stackID>.json    {"records": [...]}, one record per VR frame (fields as for
                              :func:`load_heeg`, plus leftEyeBlink, rightEyeBlink,
                              leftEyeRotation{x,y,z,w}, rightEyeRotation{x,y,z,w} when
                              eye tracking was recorded)

    Parameters
    ----------
    json_file : path
        The ``S1-T<stackID>.json`` file.
    decode_aux, decode_aux_offset : callable, optional
        Ports of MATLAB ``decodeAuxEEG`` / ``decodeAuxOffsetEEG`` (1-based indices). Only
        needed when the log has reference tracking. See :func:`load_heeg` for the default
        assumption.

    Returns
    -------
    MatStruct
        MATLAB's workspace variable ``hEEGdata`` (the name MATLAB uses for eye tracking too).
        Row vectors ``[1 x nRecords]``: ``timeStamps`` (the VR ``timestamp`` as recorded, not
        shifted), ``event``, ``eventType``, ``audioVolume``, ``positionX/Y/Z``,
        ``rotationX/Y/Z``, ``speed``, ``walkingDirection``, ``leftEyeBlink``,
        ``rightEyeBlink``, ``leftEyeRotationX/Y/Z/W``, ``rightEyeRotationX/Y/Z/W``. Eye fields
        are empty ``[0 x 0]`` arrays when the log has no eye data. With reference tracking,
        position and rotation fields hold the reference values and ``speed_participant`` is
        appended, as in MATLAB.

    Raises
    ------
    ValueError
        Missing fields, fewer than two records, or records with different fields.

    Differences from MATLAB
    -----------------------
    * Takes a file path; returns the struct instead of ``assignin(ws, ...)``; no ExpLog.
    * Stimulus onsets in the last two frame intervals average over the intervals that exist
      (MATLAB: index error).
    * Unused MATLAB code (ExpLog stack search, ``try`` around ``walkingDirection``) is dropped.
    """
    decode_aux = decode_aux or _decode_aux_eeg
    decode_aux_offset = decode_aux_offset or _decode_aux_offset_eeg

    records = _read_vr_records(json_file)
    time = _column(records, "timestamp").astype(np.float64)
    kin = _vr_kinematics(records, time, decode_aux, decode_aux_offset)
    index = np.arange(len(records))

    out = MatStruct()
    out["timeStamps"] = _row(time)
    for key, value in _vr_fields(records, index, kin):
        out[key] = value

    eye_scalar = ("leftEyeBlink", "rightEyeBlink")
    eye_rot = ("leftEyeRotation", "rightEyeRotation")
    if "leftEyeBlink" in records[0]:
        for key in eye_scalar:
            out[key] = _row(_column(records, key))
        for key in eye_rot:
            comp = _components(records, key, "xyzw")
            for i, axis in enumerate("XYZW"):
                out[f"{key}{axis}"] = _row(comp[:, i])
    else:
        for key in eye_scalar:
            out[key] = np.empty((0, 0))
        for key in eye_rot:
            for axis in "XYZW":
                out[f"{key}{axis}"] = np.empty((0, 0))
    _apply_reference_fields(out, records, index, kin)
    return out
