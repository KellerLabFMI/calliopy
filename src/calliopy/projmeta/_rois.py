"""ROI activity for project meta files: calliope ``21_Analysis/get_smoothed_ROIs.m``,
``24_ProjMeta/preprocess_widefield_activity_for_proj_meta.m`` and the widefield ROI block of
``24_ProjMeta/generate_proj_meta.m``."""

from __future__ import annotations

import logging
import re
import warnings
from typing import NamedTuple

import numpy as np

from ..io import MatStruct
from ._struct import analysis, elements, empty, is_empty, mround, struct_array

__all__ = [
    "GetSmoothedRoisResult",
    "get_smoothed_rois",
    "preprocess_widefield_activity",
    "widefield_rois_for_proj_meta",
]

log = logging.getLogger(__name__)


class GetSmoothedRoisResult(NamedTuple):
    """Outputs of :func:`get_smoothed_rois` (MATLAB ``[act, ROIinfo, act_fissa]``)."""

    act: np.ndarray
    roi_info: np.ndarray
    act_fissa: np.ndarray


def _shape_of(rois) -> tuple:
    return rois.shape if isinstance(rois, np.ndarray) else (1, len(elements(rois)))


def get_smoothed_rois(rois, par=None) -> GetSmoothedRoisResult:
    """Baseline-corrected ROI activity for the project meta file (``get_smoothed_ROIs.m``).

    Every ROI's ``activity`` is percentile-smoothed (``psmooth``, default window) and
    divided by its median (F0); the same for ``activity_fissa_corrected`` when present.

    Parameters
    ----------
    rois : struct array
        ROIs of one layer (object array or list of structs with ``activity``).
    par : ignored
        MATLAB's ``parfor`` switch; the result does not depend on it.

    Returns
    -------
    GetSmoothedRoisResult
        ``act`` [nROIs x nFrames]; ``roi_info``: the ROIs without the activity fields and
        with ``F0`` added (same shape as ``rois``); ``act_fissa`` [nROIs x nFrames] or an
        empty (0, 0) array.
    """
    psmooth = analysis("psmooth")
    items = elements(rois)
    n = len(items)
    n_frames = np.asarray(items[0]["activity"]).size if n else 0
    act = np.zeros((n, n_frames))
    f0 = np.full(n, np.nan)
    has_fissa = n and "activity_fissa_corrected" in items[0]
    act_fissa = np.zeros((n, n_frames)) if has_fissa else empty()
    for g, r in enumerate(items):
        cur = np.ravel(np.asarray(psmooth(np.asarray(r["activity"], dtype=np.float64))))
        f0[g] = np.median(cur)
        act[g, :] = cur / f0[g]
    if has_fissa:
        for g, r in enumerate(items):
            cur = np.ravel(np.asarray(psmooth(np.asarray(r["activity_fissa_corrected"], dtype=np.float64))))
            act_fissa[g, :] = cur / np.median(cur)
    info = []
    for g, r in enumerate(items):
        s = MatStruct((k, v) for k, v in r.items() if k not in ("activity", "activity_fissa_corrected"))
        s["F0"] = f0[g]
        info.append(s)
    return GetSmoothedRoisResult(act, struct_array(info, _shape_of(rois)), act_fissa)


def _activity_rows(items, field) -> np.ndarray:
    """MATLAB ``cat(2, ROIs.(field))'``: one row per ROI with a non-empty activity."""
    rows = [np.ravel(np.asarray(r[field], dtype=np.float64)) for r in items if not is_empty(r.get(field))]
    if not rows:
        return empty()
    return np.vstack(rows)


def preprocess_widefield_activity(rois, nbr_frames, wf_names_endings: str = "") -> np.ndarray:
    """Percentile-smoothed widefield ROI activity per stack (``preprocess_widefield_activity_for_proj_meta.m``).

    Each ROI's ``activity<ending>`` is smoothed with ``psmooth`` (window 6250) separately
    in every stack (``nbr_frames`` frames each). ROIs with a NaN in a stack stay 0 there.

    Parameters
    ----------
    rois : struct array
        Widefield ROIs.
    nbr_frames : array of int
        Frames per stack.
    wf_names_endings : str, default ''
        Channel ending such as '_ch525'. With an ending, MATLAB uses one segment of
        ``min(nbr_frames, len(ROIs(3).activity_ch525), len(ROIs(3).activity_ch610))``
        frames (third ROI, both channel names hard-coded); kept.

    Returns
    -------
    numpy.ndarray
        [nROIs x nFrames] smoothed activity (not divided by F0).
    """
    psmooth = analysis("psmooth")
    items = elements(rois)
    act = _activity_rows(items, "activity" + wf_names_endings)
    nf = np.atleast_1d(np.asarray(nbr_frames, dtype=np.int64)).ravel()
    if not wf_names_endings:
        edges = np.concatenate(([0], np.cumsum(nf)))
    else:
        r3 = items[2]
        m = int(min(nf.min(), np.asarray(r3["activity_ch525"]).shape[0], np.asarray(r3["activity_ch610"]).shape[0]))
        edges = np.array([0, m])
        act = act[:, :m]
    out = np.zeros(act.shape)
    for s in range(len(edges) - 1):
        sl = slice(int(edges[s]), int(edges[s + 1]))
        for i in range(act.shape[0]):
            seg = act[i, sl]
            if np.all(~np.isnan(seg)):
                out[i, sl] = np.ravel(psmooth(seg, win_size=6250))
    return out


def _wf_endings(field_names) -> list[str]:
    """Channel endings of the activity fields (MATLAB ``regexp(name, '(?<=y).*', 'match')``)."""
    endings = []
    for f in field_names:
        if "activity" not in f:
            continue
        p = f.find("y")
        if p >= 0 and p + 1 < len(f):
            endings.append(f[p + 1 :])
    return endings or [""]


def widefield_rois_for_proj_meta(rois, frame_rates, *, raw_act: bool = False, exp_id=None) -> MatStruct:
    """Widefield ROI activity of one layer as the project meta file stores it.

    The widefield block of ``24_ProjMeta/generate_proj_meta.m``. For every channel
    (activity fields ``activity``, or ``activity_ch525``/``activity_ch610``/...):

    1. ROIs without a name (duplicated corners of two-channel recordings) hand their
       indices and activity to the next ROI and are removed.
    2. When the first two ROIs have no ``ML_index`` they are the reference points: moved
       to ``reference_points`` (``name<ending>``, ``indices<ending>``).
    3. ROIs whose name contains 'corner' are moved to ``corners`` (``act<ending>``,
       ``ROIinfo<ending>``; ``'hiccup'`` when that fails).
    4. Each activity row without NaN is percentile-smoothed (``psmooth``, window
       ``2*round(6250*frame_rate/100/2)``) and divided by its median.

    Parameters
    ----------
    rois : struct array
        ROIs of the layer (fields ``name``, ``ML_index``, ``indices<ending>``,
        ``activity<ending>``, ...).
    frame_rates : float or sequence of float
        Camera frame rate (Hz) per channel, in the order of the activity fields (MATLAB
        reads it from the ``.wii`` files; see :func:`read_frame_rate`).
    raw_act : bool, default False
        Also store the unsmoothed activity (``act_raw<ending>``).
    exp_id : int, optional
        Only used in warnings.

    Returns
    -------
    MatStruct
        Fields ``reference_points`` (when present), ``corners``, ``act<ending>`` and
        ``median_act<ending>`` (float32), ``act_raw<ending>``, and ``ROIinfo`` (the
        remaining ROIs without activity, with ``F0<ending>``).

    Raises
    ------
    ValueError
        Only one of the first two ROIs lacks an ``ML_index``, or the channel frame rates
        differ by 1 Hz or more.
    """
    psmooth = analysis("psmooth")
    orig = [MatStruct(r) for r in elements(rois)]
    field_names = list(orig[0].keys()) if orig else []
    endings = _wf_endings(field_names)
    fr = np.atleast_1d(np.asarray(frame_rates, dtype=np.float64)).ravel()
    if fr.size < len(endings):
        raise ValueError(f"need one frame rate per channel {endings}")
    if len(endings) > 1:
        diff = float(np.max(np.abs(fr[: len(endings), None] - fr[None, : len(endings)])))
        if not diff < 1:
            raise ValueError(f"!!!Max frame rate diff between ch - {diff}!!!")
    act_fields = ["activity" + e for e in endings]
    rd = MatStruct()
    refs: list[MatStruct] = [MatStruct(), MatStruct()]
    have_refs = False
    corners = MatStruct()
    f0_per_chan: list = [None] * len(endings)
    roi_info: list[MatStruct] = []
    for c, e in enumerate(endings):
        roi_info = []
        for r in orig:
            missing = [f for f in act_fields if f not in r]
            if missing:
                raise KeyError(f"ROIs have no field {missing[0]!r}")
            roi_info.append(MatStruct((k, v) for k, v in r.items() if k not in act_fields))
        ind_f, act_f = "indices" + e, "activity" + e
        empty_idx = [i for i, r in enumerate(roi_info) if is_empty(r.get("name"))]
        if empty_idx:
            for i in empty_idx:
                if not is_empty(roi_info[i].get(ind_f)):
                    roi_info[i + 1][ind_f] = roi_info[i][ind_f]
                    orig[i + 1][act_f] = orig[i][act_f]
            roi_info = [r for i, r in enumerate(roi_info) if i not in empty_idx]
            orig = [r for i, r in enumerate(orig) if i not in empty_idx]
        act_raw = _activity_rows(orig, act_f)
        ref_mask = [np.size(r.get(ind_f)) == 1 for r in roi_info]
        has_refs_act = any(not is_empty(r.get(act_f)) for r, m in zip(orig, ref_mask, strict=True) if m)
        corner_mask = [isinstance(r.get("name"), str) and "corner" in r["name"] for r in roi_info]
        has_corners_act = any(not is_empty(r.get(act_f)) for r, m in zip(orig, corner_mask, strict=True) if m)

        ml0, ml1 = is_empty(roi_info[0].get("ML_index")), is_empty(roi_info[1].get("ML_index"))
        if ml0 and ml1:
            have_refs = True
            for p in range(2):
                refs[p]["name" + e] = roi_info[p]["name"]
                refs[p][ind_f] = roi_info[p][ind_f]
            if has_refs_act:
                act_raw = act_raw[2:, :]
            roi_info = roi_info[2:]
        elif ml0 != ml1:
            raise ValueError("You have only one reference point? Something is wrong")

        try:
            names = [r.get("name") for r in roi_info]
            if not all(isinstance(nm, str) for nm in names):
                raise TypeError("ROI names must be text")
            cidx = np.array([i for i, nm in enumerate(names) if re.search("corner", nm)], dtype=np.int64)
            if has_corners_act:
                if cidx.size and cidx.max() >= act_raw.shape[0]:
                    raise IndexError("corner index exceeds the activity rows")
                corners["act" + e] = act_raw[cidx, :]
                act_raw = np.delete(act_raw, cidx, axis=0)
            else:
                corners["act" + e] = empty()
            corners["ROIinfo" + e] = struct_array([roi_info[i] for i in cidx])
            roi_info = [r for i, r in enumerate(roi_info) if i not in set(cidx.tolist())]
        except (TypeError, IndexError):
            corners["act" + e] = empty()
            corners["ROIinfo" + e] = "hiccup"

        win = int(2 * mround(6250 * fr[c] / 100 / 2))
        smoothed = np.zeros(act_raw.shape)
        nan_rois = []
        for i in range(act_raw.shape[0]):
            if np.all(~np.isnan(act_raw[i])):
                smoothed[i] = np.ravel(psmooth(act_raw[i], win_size=win))
            else:
                smoothed[i] = act_raw[i]
                nan_rois.append(i + 1)
        if nan_rois:
            warnings.warn(f"ExpID {exp_id}: ROIs numbers {', '.join(map(str, nan_rois))} contains NaN in their "
                          "activity field, therefore psmooth was not applied to those ROIs.", stacklevel=2)
        median_act = np.median(smoothed, axis=1, keepdims=True) if smoothed.size else np.zeros((smoothed.shape[0], 1))
        f0_per_chan[c] = median_act.astype(np.float32)
        if have_refs:
            rd["reference_points"] = struct_array(refs)
        rd["corners"] = corners
        rd["act" + e] = (smoothed / median_act).astype(np.float32)
        rd["median_act" + e] = median_act.astype(np.float32)
        if raw_act:
            rd["act_raw" + e] = act_raw.astype(np.float32)
    for c, e in enumerate(endings):
        f0 = f0_per_chan[c].ravel()
        for n, r in enumerate(roi_info):
            r["F0" + e] = f0[n]
    rd["ROIinfo"] = struct_array(roi_info)
    return rd
