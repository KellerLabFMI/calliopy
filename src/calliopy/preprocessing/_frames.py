"""Frame onsets from the galvo signal: calliope ``04_Calliope/get_frame_times.m``."""

from __future__ import annotations

import warnings

import numpy as np

__all__ = ["get_frame_times"]


def get_frame_times(trig_ch) -> np.ndarray:
    """Frame onset times from the analog frame (galvo) signal.

    Port of calliope ``04_Calliope/get_frame_times.m``. Samples above 75 % of the maximum
    form one run per frame; the end of every run followed by a gap of more than 10 samples
    is a frame time. All frame times are then shifted so that the first one is the first
    sample below 50 % of the minimum (the start of the first frame), and one more frame is
    appended one median frame period after the last.

    Parameters
    ----------
    trig_ch : array_like
        Frame trigger channel (flattened in column-major order).

    Returns
    -------
    numpy.ndarray of float64
        Frame onset times as **1-based sample indices**, the value MATLAB stores in
        ``frame_times``. ``[1.]`` with a warning when no frame is found (as MATLAB).

    Differences from MATLAB
    -----------------------
    - Warnings use ``warnings.warn`` (MATLAB prints them).
    - When no sample lies below half the minimum (MATLAB would crash on an empty index),
      the start is taken one median period before the first detected frame, as in MATLAB's
      'galvo signal not zero before 1st frame' case, with that warning.
    - With a single detected frame the median period is NaN, as in MATLAB, so the appended
      frame is NaN.
    """
    t = np.asarray(trig_ch, dtype=np.float64).ravel(order="F")
    pos = np.flatnonzero(t > 0.75 * np.max(t)) + 1  # 1-based
    frame_times = pos[np.flatnonzero(np.diff(pos) > 10)].astype(np.float64)
    if frame_times.size == 0:
        warnings.warn("get_frame_times: no frames found", stacklevel=2)
        return np.array([1.0])
    d = np.diff(frame_times)
    period = float(np.median(d)) if d.size else float("nan")
    below = np.flatnonzero(t < 0.5 * np.min(t))
    if below.size == 0:
        warnings.warn("get_frame_times: galvo signal not zero before 1st frame", stacklevel=2)
        first_frame = frame_times[0] - period
    else:
        first_frame = float(below[0] + 1)
        if abs(first_frame - frame_times[0]) > period:
            warnings.warn("get_frame_times: galvo signal not zero before 1st frame", stacklevel=2)
            first_frame = frame_times[0] - period
    frame_times = frame_times + first_frame - frame_times[0]
    return np.append(frame_times, frame_times[-1] + period)
