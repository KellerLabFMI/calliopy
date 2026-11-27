"""Aux channel processing for project meta files: calliope ``24_ProjMeta/assign_aux_method.m``,
``assign_processed_aux.m``, ``pulsed_to_continuous.m``, ``20_MetaData/assign_load_ftypes.m``,
``21_Analysis/get_vel_ind_from_adata.m`` and ``21_Analysis/get_lick_times_thrsh.m``."""

from __future__ import annotations

import logging
import re
import warnings
from typing import NamedTuple

import numpy as np

from ..io import MatStruct
from ._struct import analysis, mround, ntzo

__all__ = [
    "GetVelIndResult",
    "assign_aux_method",
    "assign_load_ftypes",
    "assign_processed_aux",
    "get_lick_times_thrsh",
    "get_vel_ind_from_adata",
    "pulsed_to_continuous",
]

log = logging.getLogger(__name__)

_DO_NOTHING = {
    "ps_id", "Perturbation", "PerturbationLFM", "Perturbation_Rot", "GratID", "GratingID", "gratings", "ToneID",
    "opto", "Opto", "Control LED", "LED", "Lick", "LickL", "LickR", "VRx", "VRy", "VRangle", "pb_X", "pb_Y",
    "pb_angle", "Door", "DoorOpen", "Catapult", "ObjectPosition", "StimulusID", "Roll", "Pitch", "Yaw", "CamExp",
    "470_LED", "HR_LED", "Beams", "Frame", "PelletRetrieve", "PelletCommand", "Condition", "WaterCommand",
    "Trigger", "ObjectRemove", "SoundAmplitude", "PS", "ECTstim", "StimulationLocation", "StimulationPower",
    "EEGsync", "EEGdisconnect", "rTMSpulse", "rTMS_stim",
}
_POS_TO_VEL = {"Running", "velM", "Running_Rot", "VisualFlow", "velP", "Visual_Rot", "Roll_vel", "Pitch_vel",
               "Yaw_vel", "RotStage"}
_VR_POS = {"VRx", "VRy", "VRangle", "VRxpos", "VRypos", "pb_X", "pb_Y", "pb_angle"}
_SCREEN_PD = {"screenPD", "Screenphotodiode", "ScreenphotoDiode", "ScreenPhotodiode", "ScreenPhotoDiode",
              "ScreenPhotoDiode2"}
_TRIGGERS = {"VRrew", "Reward", "RewardTrig", "AirPuff", "AirPuffS", "RewardTrigManual", "RewardL", "RewardR"}
_LICKS = {"Lick", "LickL", "LickR"}
_OPTO = ("opto", "ctrl", "Opto_led", "ctrl_led", "Stim", "ctrlLED", "AudioID")


def _in_ci(name: str, options) -> bool:
    return name.lower() in {o.lower() for o in options}


def assign_aux_method(chan_name: str, proj_id: str) -> list[str]:
    """Processing methods of an aux channel for the project meta file (``assign_aux_method.m``).

    Parameters
    ----------
    chan_name : str
        Aux channel name as in the ``.ach`` file.
    proj_id : str
        Project ID; the projects CAT, APC, PLS, YBA, SNM, ECT and ANT override the methods
        of their opto/stimulation channels.

    Returns
    -------
    list of str
        Method names understood by :func:`assign_processed_aux`, in order.

    Differences from MATLAB
    -----------------------
    'Shutter' and 'FrameGalvo', and channels no rule knows, give an empty list with a
    log message / warning (MATLAB leaves the output undefined, so the caller fails).
    """
    if chan_name in ("Shutter", "FrameGalvo"):
        log.info("No - just no. These channels contain no usable information.")
        return []
    methods: list = []

    def add(*names):
        methods.extend(names)

    if chan_name in _DO_NOTHING:
        add("do_nothing")
    if chan_name in _POS_TO_VEL:
        add("pos_to_vel", "pos_to_vel_smoothed")
    if chan_name in _VR_POS:
        add("VR_pos_to_vel_smoothed")
    if chan_name in _SCREEN_PD:
        add("vis_speed")
    if chan_name in _TRIGGERS:
        add("extend_trig_to_200ms")
    if chan_name in _LICKS:
        add("get_lick_times")
    if chan_name == "TonotopyID":
        add("decodeFrequency", "decodeLevel")
    if re.search(r"_[xy]$", chan_name):
        add("do_nothing")
    if re.search(r"_[xy]_vel$", chan_name):
        add("pos_to_vel", "pos_to_vel_smoothed")

    def put(pos: int, name: str):  # MATLAB proc_methods{pos} = name (1-based)
        while len(methods) < pos:
            methods.append(None)
        methods[pos - 1] = name

    cnt = len(methods) + 1
    if proj_id == "CAT" and chan_name in _OPTO:
        methods, cnt = ["do_nothing"], 2
    elif proj_id in ("APC", "PLS", "ANT") and _in_ci(chan_name, _OPTO):
        put(1, "do_nothing")
        put(2, "pulsed_to_continuous")
        cnt += 1
    elif proj_id == "YBA":
        if _in_ci(chan_name, _OPTO):
            methods, cnt = ["pulsed_to_continuous"], 2
        if _in_ci(chan_name, ("StimulationPower",)):
            methods, cnt = ["extract_pulse_envelope"], 2
    elif proj_id == "SNM" and chan_name in (*_OPTO, "VRmotorsensoryGain") or proj_id == "ECT" and chan_name in ("Stim", "ctrlLED"):
        methods, cnt = ["do_nothing"], 2
    if cnt == 1:
        warnings.warn(f"It seems like a new kind of aux channel was encountered: {chan_name}. "
                      "Please define how to process it in assign_aux_method and assign_processed_aux",
                      stacklevel=2)
        return []
    return [m for m in methods if m is not None]


class GetVelIndResult(NamedTuple):
    """Outputs of :func:`get_vel_ind_from_adata` (MATLAB ``[vel, vel_ind, vel_smoothed, vel_raw]``)."""

    vel: np.ndarray
    vel_ind: np.ndarray
    vel_smoothed: np.ndarray
    vel_raw: np.ndarray


def get_vel_ind_from_adata(data, scalefactor: float | None = None, sample_rate: float | None = None
                           ) -> GetVelIndResult:
    """Running (rotary encoder) velocity from an aux position trace (``get_vel_ind_from_adata.m``).

    The trace is differentiated, wrap-arounds of the encoder voltage are undone (jumps
    larger than ``scalefactor`` are shifted by ``2*scalefactor``), the result is low-pass
    filtered below 10 Hz (``ftfil``) and smoothed over one second (``smooth2``).

    Parameters
    ----------
    data : array, shape (N,) or (1, N)
        Encoder voltage trace (one aux channel).
    scalefactor : float, optional
        Half range of the encoder voltage: default 10 when ``round(max(data)) > 7``, else 5.
    sample_rate : float, optional
        Samples per second, default 1000.

    Returns
    -------
    GetVelIndResult
        ``vel`` (filtered), ``vel_ind`` (``vel_smoothed > 0.005``), ``vel_smoothed``,
        ``vel_raw`` (unfiltered); one sample shorter than ``data``, same orientation.

    Differences from MATLAB
    -----------------------
    MATLAB also accepts a single number and loads the aux data of that experiment; here
    the trace is required (load it with :func:`calliopy.io.load_lvd`).
    """
    x = np.asarray(data, dtype=np.float64)
    if x.size == 1:
        raise ValueError("get_vel_ind_from_adata needs the aux trace (loading by ExpID is not ported)")
    if scalefactor is None:
        scalefactor = 10.0 if mround(np.max(x)) > 7 else 5.0
    if sample_rate is None:
        sample_rate = 1000.0
    vel = np.diff(x, axis=-1)
    vel = np.where(vel > scalefactor, vel - 2 * scalefactor, vel)
    vel = np.where(vel < -scalefactor, vel + 2 * scalefactor, vel)
    vel_raw = vel.copy()
    vel = np.asarray(analysis("ftfil")(vel, sample_rate, 0, 10)).reshape(vel.shape)
    vel_smoothed = np.asarray(analysis("smooth2")(vel, int(sample_rate))).reshape(vel.shape)
    return GetVelIndResult(vel, vel_smoothed > 0.005, vel_smoothed, vel_raw)


def get_lick_times_thrsh(licktrace) -> np.ndarray:
    """Lick onsets of a lick sensor trace (``21_Analysis/get_lick_times_thrsh.m``).

    Returns
    -------
    numpy.ndarray of int64
        **1-based** sample indices ``k`` (MATLAB values, stored in proj_meta as
        ``lickTimes``) where the trace, normalised to [0, 1], is <= 0.2 at ``k`` and
        > 0.2 at ``k+1``.
    """
    b = (ntzo(np.ravel(licktrace)) > 0.2).astype(np.int8)
    return np.flatnonzero(np.diff(b) == 1).astype(np.int64) + 1


def pulsed_to_continuous(aux_trace, sample_rate: float = 1000, min_inter_trigger_seconds: float = 2
                         ) -> np.ndarray:
    """Turn pulsed stimulation triggers into continuous blocks (``pulsed_to_continuous.m``).

    Threshold 0.5 V. Onsets closer than ``min_inter_trigger_seconds`` to the previous
    onset and offsets closer than that to the next offset are dropped; every remaining
    onset is joined to the nearest offset after it and the segment is set to its maximum.

    Parameters
    ----------
    aux_trace : array
        Voltage trace of one channel (any shape; treated as a vector).
    sample_rate : float, default 1000
    min_inter_trigger_seconds : float, default 2

    Returns
    -------
    numpy.ndarray
        Processed trace, same shape. Unchanged when there are no onsets or offsets.

    Differences from MATLAB
    -----------------------
    When every offset precedes the first onset the trace is returned unchanged (MATLAB
    fails on an empty index).
    """
    a = np.asarray(aux_trace)
    x = a.ravel().astype(np.float64)
    d = np.diff((x > 0.5).astype(np.int8))
    onsets = np.flatnonzero(d == 1) + 1  # 1-based as in MATLAB
    offsets = np.flatnonzero(d == -1) + 1
    out = x.copy()
    if onsets.size == 0 or offsets.size == 0:
        return out.reshape(a.shape)
    offsets = offsets[offsets >= onsets[0]]
    if offsets.size == 0:
        return out.reshape(a.shape)
    thr = min_inter_trigger_seconds * sample_rate
    onsets = onsets[np.concatenate(([True], np.diff(onsets) > thr))]
    offsets = offsets[np.concatenate((np.diff(offsets) > thr, [True]))]
    for on in onsets:
        off = offsets[int(np.argmin(np.abs(offsets - on)))]
        if off > on:
            out[on - 1 : off] = np.max(x[on - 1 : off])
    return out.reshape(a.shape)


def _extend_trig_to_200ms(x: np.ndarray) -> np.ndarray:
    b = (x > 0.2).astype(np.int8)
    ons = np.flatnonzero(np.diff(b) == 1) + 1  # 1-based
    ext = x.astype(np.float64, copy=True)
    if ons.size == 0:
        return ext
    med = np.array([np.median(x[o : min(o + 10, x.size)]) for o in ons])  # samples ons+1..ons+10
    last = int(ons.max()) + 200  # 0-based position of ons+201 (1-based)
    if last >= ext.size:
        ext = np.concatenate((ext, np.zeros(last + 1 - ext.size)))
    for k in range(1, 201):
        ext[ons + k] = med
    return ext


def assign_processed_aux(adata_ch, proc_methods, ch_name: str, scalefactor: float | None = None,
                         sample_rate: float | None = None) -> list[tuple[str, np.ndarray]]:
    """Process one aux channel with the methods of :func:`assign_aux_method` (``assign_processed_aux.m``).

    Methods and output names: ``do_nothing`` (``ch_name``, the trace), ``pos_to_vel``
    (``ch_name``, filtered velocity), ``pos_to_vel_smoothed`` (``<ch_name>_smoothed``),
    ``VR_pos_to_vel_smoothed`` (``<ch_name>_vel_smoothed``: steps above 2 removed,
    ``ftfil`` 10 Hz at 1 kHz, ``smooth2`` 1000), ``vis_speed`` (``'vis_speed'``, the
    trace), ``extend_trig_to_200ms`` (``<ch_name>_extended``: 200 samples after every
    onset above 0.2 set to the median of the 10 samples after it), ``get_lick_times``
    (``'lickTimes'``, 1-based onsets), ``pulsed_to_continuous`` (``<ch_name>_cont``),
    ``extract_pulse_envelope`` (``ch_name``).

    Parameters
    ----------
    adata_ch : array
        One aux channel.
    proc_methods : sequence of str
    ch_name : str
    scalefactor : float, optional
        Passed to :func:`get_vel_ind_from_adata`.
    sample_rate : float, optional
        Default 1000.

    Returns
    -------
    list of (str, numpy.ndarray)
        One (name, data) pair per method (MATLAB ``pdata``, n x 2 cell).

    Raises
    ------
    NotImplementedError
        ``decodeFrequency``/``decodeLevel``: ``decodeTonotopyID`` is not in calliope r2003.

    Differences from MATLAB
    -----------------------
    - ``extend_trig_to_200ms``: an onset less than 10 samples before the end uses the
      samples left (MATLAB fails). Kept from MATLAB: the trace grows (zeros) when an
      onset is less than 200 samples before the end.
    - Unknown method names are skipped with a warning (MATLAB leaves an empty row).
    """
    if sample_rate is None:
        sample_rate = 1000.0
    x = np.asarray(adata_ch, dtype=np.float64)
    out: list[tuple[str, np.ndarray]] = []
    for m in proc_methods:
        if m == "do_nothing":
            out.append((ch_name, x))
        elif m == "pos_to_vel":
            out.append((ch_name, get_vel_ind_from_adata(x, scalefactor, sample_rate).vel))
        elif m == "pos_to_vel_smoothed":
            out.append((ch_name + "_smoothed", get_vel_ind_from_adata(x, scalefactor, sample_rate).vel_smoothed))
        elif m == "VR_pos_to_vel_smoothed":
            t = np.diff(x, axis=-1)
            t = np.where((t < -2) | (t > 2), 0.0, t)
            vel = np.asarray(analysis("ftfil")(t, 1000, 0, 10)).reshape(t.shape)
            out.append((ch_name + "_vel_smoothed", np.asarray(analysis("smooth2")(vel, 1000)).reshape(t.shape)))
        elif m == "vis_speed":
            out.append(("vis_speed", x))
        elif m == "extend_trig_to_200ms":
            ext = _extend_trig_to_200ms(x.ravel())
            out.append((ch_name + "_extended", ext.reshape(1, -1) if x.ndim == 2 else ext))
        elif m == "get_lick_times":
            out.append(("lickTimes", get_lick_times_thrsh(x).reshape(1, -1)))
        elif m in ("decodeFrequency", "decodeLevel"):
            raise NotImplementedError("decodeTonotopyID is not part of calliope r2003; cannot decode TonotopyID")
        elif m == "pulsed_to_continuous":
            out.append((ch_name + "_cont", pulsed_to_continuous(x, sample_rate, 2)))
        elif m == "extract_pulse_envelope":
            env = np.asarray(analysis("extract_pulse_envelope")(x))
            out.append((ch_name, env))
        else:
            warnings.warn(f"unknown aux processing method {m!r} skipped", stacklevel=2)
    return out


def assign_load_ftypes(file_types) -> MatStruct:
    """Which file types are to be loaded (``20_MetaData/assign_load_ftypes.m``).

    Returns
    -------
    MatStruct
        Booleans ``ach`` ('.ach'), ``eye`` ('.eye'), ``eyeR`` ('.eyeR'), ``lvd`` ('.lvd'),
        ``bin`` ('525.bin'), ``ini`` ('525.ini'), ``vrd`` ('.vrd'), ``vid`` ('.vid'),
        ``vid3`` ('.vid3'), ``ton`` ('.ton'), ``owl`` ('owl.avi'): exact matches.
    """
    ft = [str(f) for f in file_types]
    lft = MatStruct()
    for key, name in (("ach", ".ach"), ("eye", ".eye"), ("eyeR", ".eyeR"), ("lvd", ".lvd"), ("bin", "525.bin"),
                      ("ini", "525.ini"), ("vrd", ".vrd"), ("vid", ".vid"), ("vid3", ".vid3"), ("ton", ".ton"),
                      ("owl", "owl.avi")):
        lft[key] = name in ft
    return lft
