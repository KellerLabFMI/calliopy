"""Eye-camera analysis and navigation performance (calliope ``10_Behavior``)."""

from __future__ import annotations

import logging
import math
from typing import NamedTuple

import numpy as np

from .._matlab import mround

__all__ = [
    "EyeDataAtFrames",
    "FindSaccadesResult",
    "find_saccades",
    "leinweber_performance",
    "map_eye_data_to_frame_times",
]

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- performance


def leinweber_performance(vel_m, vel_r, vr_angle, vel_trsh: float | None = 0.001) -> float:
    """Performance in the 2-D "navigate to the end of the corridor" task.

    Port of calliope ``10_Behavior/LeinweberPerformance.m``. With
    ``vel = sqrt(vel_m**2 + vel_r**2)`` and the heading ``vr_angle`` (0..5, 2.5 points to
    the target)::

        performance = nansum(vel * cos((vr_angle/2.5 - 1) * pi)) / nansum(vel)
                      * nansum(vel > vel_trsh) / len(vel)

    i.e. the distance covered towards the target over the total distance, times the
    fraction of samples in which the mouse moved.

    Parameters
    ----------
    vel_m, vel_r : array_like
        Smoothed forward and rotational velocity (MATLAB ``velM_smoothed``,
        ``velR_smoothed``), same number of elements (flattened column-major).
    vr_angle : array_like
        VR heading, same length as the velocities or one sample longer (then its first
        sample is dropped, with a logged warning, as MATLAB).
    vel_trsh : float, default 0.001
        Movement threshold (``None`` gives the default, as MATLAB's empty).

    Returns
    -------
    float

    Differences from MATLAB
    -----------------------
    - Other length mismatches raise ``ValueError`` (MATLAB fails in the element-wise
      product).
    """
    if vel_trsh is None:
        vel_trsh = 0.001
    vm = np.asarray(vel_m, dtype=np.float64).ravel(order="F")
    vr = np.asarray(vel_r, dtype=np.float64).ravel(order="F")
    ang = np.asarray(vr_angle, dtype=np.float64).ravel(order="F")
    if vm.size != vr.size:
        raise ValueError("vel_m and vel_r must have the same number of elements")
    vel = np.sqrt(vm**2 + vr**2)
    if ang.size - 1 == vel.size:
        log.warning("NC Warning: vel and VRangle are not of same length")
        ang = ang[1:]
    elif ang.size != vel.size:
        raise ValueError("vr_angle must have the length of the velocities (or one more)")
    with np.errstate(invalid="ignore", divide="ignore"):
        num = np.nansum(vel * np.cos((ang / 2.5 - 1) * np.pi))
        return float(num / np.nansum(vel) * np.sum(vel > vel_trsh) / vel.size)


# --------------------------------------------------------------------------- saccades


class FindSaccadesResult(NamedTuple):
    """Outputs of :func:`find_saccades` (MATLAB
    ``[saccade_frames, pupil_diam, pupil_pos, blink, facial_mov]``)."""

    saccade_frames: np.ndarray
    pupil_diam: np.ndarray
    pupil_pos: np.ndarray
    blink: np.ndarray
    facial_mov: np.ndarray


_OFFSETS = np.array([[0, 1, -1, 2, -2, 3, -3, 4, -4], [0, 2, -2, -1, 1, 4, -4, -3, 3]])


def _first_below(v: np.ndarray, thr: float) -> int | None:
    """1-based position of the first element below ``thr`` (MATLAB ``find(v<thr,1)``)."""
    hit = np.flatnonzero(v < thr)
    return int(hit[0]) + 1 if hit.size else None


def _mean_of_means(block: np.ndarray) -> float:
    """MATLAB ``mean(mean(block))`` (column means, then their mean)."""
    return float(block.astype(np.float64).mean(axis=0).mean())


def _saturating_frame_diff(stack: np.ndarray) -> np.ndarray:
    """MATLAB ``diff(stack, 1, 3)``: integer classes saturate at their limits."""
    d = np.diff(stack.astype(np.float64), axis=2)
    if stack.dtype.kind in "iu":
        info = np.iinfo(stack.dtype)
        d = np.clip(d, info.min, info.max)
    return d


def find_saccades(idata) -> FindSaccadesResult:
    """Pupil diameter, position, blinks, saccades and facial movement from an eye video.

    Port of calliope ``10_Behavior/find_saccades.m`` ("a bit of a hack", GK). For every
    frame ``ind = 4 .. T-2`` (1-based):

    1. a threshold: ``0.81 *`` the mean of a 21 x 21 window at the image centre until the
       pupil has been found once; afterwards the mean of ``0.81 *`` the window mean around
       the mean pupil position of the previous 10 frames and the thresholds of those frames;
    2. from 9 points around the centre (fixed offsets), the distance to the first pixel
       below the threshold downwards (``x1``), upwards (``x2``), rightwards (``y1``) and
       leftwards (``y2``), averaged over the 9 points;
    3. if all 36 searches succeed: ``pupil_diam = sqrt(((x1+x2)/2)**2 + ((y1-y2)/2)**2)`` and
       ``pupil_pos = centre + [x1-x2, y1-y2]/2``; otherwise both are copied from the
       previous frame;
    4. ``blink`` = mean intensity of a 21 x 21 window (clipped) at the rounded pupil
       position.

    Saccades are frames where the pupil moves more than 5 pixels between frames: the
    middle (rounded) of every run of such steps. ``facial_mov`` is the mean over the four
    50 x 50 image corners of the summed frame-to-frame intensity change, with a 0 appended.

    Parameters
    ----------
    idata : array_like, shape (height, width, n_frames)
        Eye-camera stack (any numeric class; at least 50 x 50 pixels).

    Returns
    -------
    FindSaccadesResult
        - ``saccade_frames``: float64 column ``(k, 1)`` of **1-based** frame numbers (the
          values MATLAB returns and stores).
        - ``pupil_diam``: float64 column ``(L, 1)``.
        - ``pupil_pos``: float64 ``(2, L)``, rows (row, column) in MATLAB's **1-based**
          pixel coordinates.
        - ``blink``: float64 column ``(L, 1)``.
        - ``facial_mov``: float64 column ``(n_frames, 1)``.

        ``L = max(idata.shape)`` (MATLAB ``length(idata)``), which equals ``n_frames`` for
        recordings with more frames than pixels per side; frames 1-3 and the last two keep
        the initial values (0 / image centre).

    Notes
    -----
    Suspected MATLAB quirks that are kept: ``length(idata)`` is the longest dimension, not
    the number of frames; for integer stacks ``diff`` saturates, so ``facial_mov`` only
    counts intensity increases (negative differences become 0 for unsigned classes).

    Differences from MATLAB
    -----------------------
    - Computation is in float64 also for ``single`` stacks (MATLAB's ``mean`` keeps
      ``single``).
    - A stack smaller than 50 x 50 pixels raises ``ValueError`` (MATLAB fails with an
      index error in the facial-movement corners or the centre window).
    - The MATLAB ``try/catch`` around the 36 edge searches is replaced by explicit checks
      (start point inside the image, a pixel below threshold found); the outcome is the
      same.
    """
    st = np.asarray(idata)
    if st.ndim != 3:
        raise ValueError("idata must be (height, width, n_frames)")
    h, w, t = st.shape
    if h < 50 or w < 50:
        raise ValueError("idata must be at least 50 x 50 pixels (facial movement corners)")
    big_l = max(st.shape)
    c0 = np.array([mround(h / 2), mround(w / 2)], dtype=np.float64)
    pupil_diam = np.zeros(big_l)
    pupil_pos = np.ones((2, big_l)) * c0[:, None]
    blink = np.zeros(big_l)
    thresh_mem = np.ones(big_l)

    nothing_yet = True
    for ind in range(4, t - 1):  # MATLAB ind = 4 : T-2 (1-based)
        tmp = st[:, :, ind - 1]
        if nothing_yet:
            cr, cc = int(mround(h / 2)), int(mround(w / 2))
            thrsh = 0.81 * _mean_of_means(tmp[cr - 11 : cr + 10, cc - 11 : cc + 10])
        else:
            lo = max(1, ind - 10)
            centre = mround(pupil_pos[:, lo - 1 : ind - 1].mean(axis=1))
            cr, cc = int(centre[0]), int(centre[1])
            r0, r1 = max(1, cr - 10), min(h, cr + 10)
            q0, q1 = max(1, cc - 10), min(w, cc + 10)
            win = 0.81 * _mean_of_means(tmp[r0 - 1 : r1, q0 - 1 : q1])
            mem = thresh_mem[max(1, ind - 10) - 1 : max(1, ind - 1)]
            thrsh = float(np.mean(np.concatenate([[win], mem])))
        thresh_mem[ind - 1] = thrsh

        found = True
        xs1, xs2, ys1, ys2 = [], [], [], []
        for j in range(9):
            r = cr + int(_OFFSETS[0, j])
            c = cc + int(_OFFSETS[1, j])
            if not (1 <= r <= h and 1 <= c <= w):
                found = False
                break
            hits = (
                _first_below(tmp[r - 1 :, c - 1], thrsh),
                _first_below(tmp[r - 1 :: -1, c - 1], thrsh),
                _first_below(tmp[r - 1, c - 1 :], thrsh),
                _first_below(tmp[r - 1, c - 1 :: -1], thrsh),
            )
            if any(v is None for v in hits):
                found = False
                break
            xs1.append(hits[0])
            xs2.append(hits[1])
            ys1.append(hits[2])
            ys2.append(hits[3])
        if found:
            x1, x2, y1, y2 = (float(np.mean(v)) for v in (xs1, xs2, ys1, ys2))
            pupil_diam[ind - 1] = math.sqrt(((x1 + x2) / 2) ** 2 + ((y1 - y2) / 2) ** 2)
            pupil_pos[:, ind - 1] = np.array([cr, cc]) + np.array([x1 - x2, y1 - y2]) / 2
            nothing_yet = False
        else:
            pupil_diam[ind - 1] = pupil_diam[ind - 2]
            pupil_pos[:, ind - 1] = pupil_pos[:, ind - 2]
        pr, pc = int(mround(pupil_pos[0, ind - 1])), int(mround(pupil_pos[1, ind - 1]))
        r0, r1 = max(1, pr - 10), min(h, pr + 10)
        q0, q1 = max(1, pc - 10), min(w, pc + 10)
        blink[ind - 1] = _mean_of_means(tmp[r0 - 1 : r1, q0 - 1 : q1])

    step = np.sqrt(np.sum(np.diff(pupil_pos.T, axis=0) ** 2, axis=1)) > 5
    ds = np.diff(step.astype(np.int8))
    on = np.flatnonzero(ds == 1) + 1
    off = np.flatnonzero(ds == -1) + 1
    try:
        sacc = mround((on + off) / 2)
    except ValueError as err:  # MATLAB fails the same way for unmatched onsets/offsets
        raise ValueError("saccade onsets and offsets do not pair up") from err
    sacc = np.asarray(sacc, dtype=np.float64).reshape(-1, 1)

    corners = (
        st[:50, :50, :],
        st[:50, w - 50 :, :],
        st[h - 50 :, :50, :],
        st[h - 50 :, w - 50 :, :],
    )
    fm = np.stack([_saturating_frame_diff(c).sum(axis=0).sum(axis=0) for c in corners], axis=1)
    facial = np.concatenate([fm.mean(axis=1), [0.0]]).reshape(-1, 1)
    return FindSaccadesResult(
        sacc, pupil_diam.reshape(-1, 1), pupil_pos, blink.reshape(-1, 1), facial
    )


# --------------------------------------------------------------------------- frame mapping


class EyeDataAtFrames(NamedTuple):
    """Outputs of :func:`map_eye_data_to_frame_times` (MATLAB
    ``[pupil_diam, pupil_pos, blink, facial_mov, whisker_pos]``)."""

    pupil_diam: np.ndarray
    pupil_pos: np.ndarray
    blink: np.ndarray
    facial_mov: np.ndarray
    whisker_pos: np.ndarray


def map_eye_data_to_frame_times(
    pupil_diam, pupil_pos, blink, facial_mov, whisker_pos, frame_times, iframe_times
) -> EyeDataAtFrames:
    """Resample eye-camera measures to imaging frames (nearest eye frame in time).

    Port of calliope ``10_Behavior/map_eye_data_to_frame_times.m``. For every imaging
    frame time the eye frame with the closest time is taken (the first one on ties, as
    MATLAB ``min``; NaN eye-frame times are ignored).

    Parameters
    ----------
    pupil_diam, blink, facial_mov : array_like, length n_eye_frames
        Per-eye-frame values (flattened column-major).
    pupil_pos, whisker_pos : array_like, shape (2, n_eye_frames)
    frame_times : array_like, length n_frames
        Imaging frame times.
    iframe_times : array_like, length n_eye_frames
        Eye-camera frame times (same unit as ``frame_times``).

    Returns
    -------
    EyeDataAtFrames
        float64 rows ``(1, n_frames)`` for ``pupil_diam``, ``blink``, ``facial_mov`` and
        ``(2, n_frames)`` for ``pupil_pos``, ``whisker_pos``, as MATLAB.
    """
    ft = np.asarray(frame_times, dtype=np.float64).ravel(order="F")
    it = np.asarray(iframe_times, dtype=np.float64).ravel(order="F")
    if it.size == 0:
        raise ValueError("iframe_times must not be empty")
    it_cmp = np.where(np.isnan(it), np.inf, it)
    idx = np.empty(ft.size, dtype=np.int64)
    chunk = max(1, int(2**22 // it.size))
    for s in range(0, ft.size, chunk):
        d = np.abs(it_cmp[None, :] - ft[s : s + chunk, None])
        d = np.where(np.isnan(d), np.inf, d)
        idx[s : s + chunk] = np.argmin(d, axis=1)

    def vec(a):
        return np.asarray(a, dtype=np.float64).ravel(order="F")[idx].reshape(1, -1)

    def two(a):
        return np.asarray(a, dtype=np.float64).reshape(2, -1, order="F")[:, idx]

    return EyeDataAtFrames(
        vec(pupil_diam), two(pupil_pos), vec(blink), vec(facial_mov), two(whisker_pos)
    )
