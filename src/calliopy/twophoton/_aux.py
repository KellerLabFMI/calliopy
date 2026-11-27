"""Aux-channel checks and imaging-frame times.

Ports of calliope ``05_LoadData/get_iframe_times.m`` and of the computations in
``05_LoadData/checkAux.m`` (frame-time statistics, camera exposure intervals, channel
names). File discovery on the acquisition machines and the plots of ``checkAux.m`` are not
ported (rig-side).

None of these functions has been compared with MATLAB output.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np

from .._matlab import mround
from ..preprocessing import get_frame_times

__all__ = [
    "FrameTimeStats",
    "IframeTimes",
    "aux_channel_names",
    "aux_frame_time_stats",
    "cam_exposure_check",
    "get_iframe_times",
]


class IframeTimes(NamedTuple):
    """Result of :func:`get_iframe_times` (MATLAB ``[iframe_times, shutter_open]``)."""

    iframe_times: np.ndarray
    """``(1, nFrames)`` frame times in aux samples (MATLAB values)."""
    shutter_open: np.ndarray
    """``(1, nOpenings)`` MATLAB values of the shutter openings (see Notes)."""


def get_iframe_times(imeta_data: Any, shutter_channel: Any, nbr_iframes: Any = None) -> IframeTimes:
    """Times of the frames of an image stream in aux samples (``get_iframe_times.m``, GK).

    The shutter channel is scanned for openings (first sample > 2.5 V) and closings (first
    sample < 2.5 V after an opening). The frames of segment ``k`` get
    ``imeta_data[1, frames] - imeta_data[1, first frame of k] + shutter_open[k]``.

    Parameters
    ----------
    imeta_data
        Frame metadata ``[nRows x nFrames]``; row 1 (0-based) holds the frame time stamps.
    shutter_channel
        Aux shutter trace.
    nbr_iframes
        Frames per shutter opening. Default ``max(imeta_data.shape)`` (MATLAB
        ``length(imeta_data)``), i.e. one opening.

    Returns
    -------
    IframeTimes

    Raises
    ------
    ValueError
        The number of shutter openings differs from the number of segments (MATLAB
        printed a message and returned empty frame times).

    Notes
    -----
    As MATLAB, ``shutter_open`` is the 1-based index of the first sample above threshold
    plus one (``find(...) + start``), and that value is added to the frame times.

    Differences from MATLAB: a shutter that never closes again ends the scan (MATLAB failed
    assigning an empty ``find``).
    """
    im = np.asarray(imeta_data, dtype=np.float64)
    sh = np.asarray(shutter_channel, dtype=np.float64).ravel()
    nb = np.atleast_1d(np.asarray(max(im.shape) if nbr_iframes is None else nbr_iframes, dtype=np.int64))
    opens: list[int] = []
    close = 1
    while True:
        hit = np.flatnonzero(sh[close - 1 :] > 2.5)
        if hit.size == 0:
            break
        so = int(hit[0] + 1 + close)
        opens.append(so)
        nxt = np.flatnonzero(sh[so - 1 :] < 2.5)
        if nxt.size == 0:
            break
        close = int(nxt[0] + 1 + so)
    if len(opens) != nb.size:
        raise ValueError("Number of shutter openings does not match the eye data")
    out = np.zeros(int(nb.sum()))
    start = 0
    for k in range(nb.size):
        seg = slice(start, start + int(nb[k]))
        out[seg] = im[1, seg] + opens[k] - im[1, start]
        start += int(nb[k])
    return IframeTimes(out[None, :], np.asarray(opens, dtype=np.float64)[None, :])


class FrameTimeStats(NamedTuple):
    """Frame-interval statistics printed by ``checkAux.m``."""

    frame_intervals: np.ndarray
    mean: float
    std: float
    n_outliers: int
    """Intervals outside ``mean +/- 2 std``."""


def aux_frame_time_stats(frame_channel: Any) -> FrameTimeStats:
    """Intervals between frames of the frame (galvo) channel and their outliers (``checkAux.m``).

    Uses :func:`calliopy.preprocessing.get_frame_times`; ``std`` with N-1.
    """
    ft = np.diff(np.asarray(get_frame_times(frame_channel), dtype=np.float64).ravel())
    m = float(ft.mean())
    s = float(ft.std(ddof=1)) if ft.size > 1 else float("nan")
    n_out = int(np.sum((ft < m - 2 * s) | (ft > m + 2 * s)))
    return FrameTimeStats(ft, m, s, n_out)


def cam_exposure_check(cam_trace: Any, shutter_trace: Any, *, samples_per_ms: float = 10) -> tuple[np.ndarray, int]:
    """Camera exposure intervals and edges while the shutter is open (``checkAux.m``).

    Returns the intervals between rising edges of ``round(cam) > 1`` in ms (MATLAB divides
    the sample difference by 10, i.e. assumes 10 kHz), and the number of intervals between
    the first shutter opening and the first closing (``round(shutter) > 1`` rising,
    ``round(shutter) < 1`` rising).
    """
    cam = mround(np.asarray(cam_trace, dtype=np.float64).ravel())
    sh = mround(np.asarray(shutter_trace, dtype=np.float64).ravel())
    edges = np.flatnonzero(np.diff((cam > 1).astype(np.int8)) == 1) + 1
    intervals = np.diff(edges) / samples_per_ms
    on = np.flatnonzero(np.diff((sh > 1).astype(np.int8)) == 1) + 1
    off = np.flatnonzero(np.diff((sh < 1).astype(np.int8)) == 1) + 1
    n_edges = 0
    if on.size and off.size and off[0] >= on[0]:
        seg = cam[on[0] - 1 : off[0]]
        n_edges = max(int(np.sum(np.diff((seg > 1).astype(np.int8)) == 1)) - 1, 0)
    return intervals, n_edges


def aux_channel_names(ach: Mapping[str, Any]) -> list:
    """Channel names from a parsed ``.ach`` file (``read_ini(...)``): the values of the
    ``auxrec`` fields whose name contains ``channel``, in file order (``checkAux.m``)."""
    aux = ach["auxrec"]
    return [str(v) for k, v in aux.items() if "channel" in str(k)]
