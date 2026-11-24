"""Online retinotopy: response maps to visual stimuli (port of calliope 12_Widefield).

Port of the computation in ``online_retinotopy.m``. MATLAB found the newest (or a given)
stack in a temp-data folder, loaded the ``.wid`` movie with every ``downs_factor``-th frame,
the ``.lvd`` aux traces and the ``.ach`` channel table, computed one response map per
stimulus type, let the user click a crop rectangle and showed three overlay panels. Here:

- :func:`retinotopy_frame_triggers`, :func:`retinotopy_stimulus_triggers`,
  :func:`retinotopy_response_maps`, :func:`retinotopy_anatomy`,
  :func:`filter_with_disk_kernel`: the computation on arrays;
- :func:`online_retinotopy`: all of it on arrays (movie, aux traces, channel labels);
- :func:`online_retinotopy_from_files`: the same from the ``.wid``, ``.lvd`` and ``.ach``
  paths;
- :func:`plot_online_retinotopy`: the three panels, with the crop rectangle as an argument
  (:func:`crop_rect_from_points` turns two corner points into it, as MATLAB did with the
  two ``ginput`` clicks).

Finding the last recorded stack in a temp folder is deferred (needs the lab
configuration). None of these functions has been compared with MATLAB output.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from os import PathLike
from typing import NamedTuple

import numpy as np
import scipy.ndimage
from matplotlib.figure import Figure

from calliopy._matlab import matlab_cast, mround

__all__ = [
    "ANALOG_THR",
    "RAW_FRAMERATE",
    "OnlineRetinotopy",
    "RetinotopyMaps",
    "StimulusTriggers",
    "crop_rect_from_points",
    "filter_with_disk_kernel",
    "online_retinotopy",
    "online_retinotopy_from_files",
    "plot_online_retinotopy",
    "retinotopy_anatomy",
    "retinotopy_frame_triggers",
    "retinotopy_response_maps",
    "retinotopy_stimulus_triggers",
]

logger = logging.getLogger(__name__)

RAW_FRAMERATE = 100  #: frame rate of the raw acquisition, Hz (MATLAB constant)
ANALOG_THR = 0.25  #: threshold for binarising the aux voltage traces, V (MATLAB constant)

_PANEL_GROUPS = (
    ((2.0,),),  # full field
    ((0.5, 1.0), (1.5, 2.5), (3.0, 3.5)),  # horizontal retinotopy
    ((0.5, 1.5, 3.0), (1.0, 2.5, 3.5)),  # vertical retinotopy
)
_PANEL_COLORS = ((1.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 1.0, 0.0))  # red, cyan, yellow
_PANEL_TITLES = ("Full-field response", "Horizontal Retinotopy", "Vertical retinotopy")
_BACKGROUND = (0.15, 0.15, 0.15)


class StimulusTriggers(NamedTuple):
    """Result of :func:`retinotopy_stimulus_triggers`."""

    trigs: np.ndarray  #: 0-based downsampled-frame index of each trigger, 1-D int64
    trig_type: np.ndarray  #: stimulus type of each trigger, 1-D float64
    trigger_types: np.ndarray  #: sorted unique types, (1, n) float64 (MATLAB output)


class RetinotopyMaps(NamedTuple):
    """Result of :func:`retinotopy_response_maps` (MATLAB ``[resp_maps, trigger_types]``)."""

    resp_maps: np.ndarray
    trigger_types: np.ndarray


class OnlineRetinotopy(NamedTuple):
    """Result of :func:`online_retinotopy`; the first two are MATLAB's outputs."""

    resp_maps: np.ndarray  #: [height x width x nTypes] float64, (post - pre) / pre
    trigger_types: np.ndarray  #: (1, nTypes) float64, sorted
    anatomy: np.ndarray  #: [height x width] uint8 mean image (vasculature)
    frame_trigs: np.ndarray  #: 0-based aux sample of each loaded frame, 1-D int64
    trigs: np.ndarray  #: 0-based frame index of each stimulus trigger, 1-D int64
    trig_type: np.ndarray  #: type of each trigger, 1-D float64


def _check_downs_factor(downs_factor) -> int:
    if not (float(downs_factor) == int(downs_factor) and int(downs_factor) >= 1):
        raise ValueError(f"downs_factor must be an integer >= 1, got {downs_factor!r}")
    return int(downs_factor)


def _rising_edges(x: np.ndarray, threshold: float) -> np.ndarray:
    """0-based i with x[i] <= threshold < x[i+1] (MATLAB find(...) - 1)."""
    return np.flatnonzero((x[:-1] <= threshold) & (x[1:] > threshold))


def _rescale(x: np.ndarray) -> np.ndarray:
    """MATLAB rescale(x) to [0, 1]: NaN ignored for min/max, constant input gives 0."""
    a = np.asarray(x, dtype=np.float64)
    finite = a[~np.isnan(a)]
    if finite.size == 0:
        return np.full(a.shape, np.nan)
    lo, hi = finite.min(), finite.max()
    if hi == lo:
        return np.where(np.isnan(a), np.nan, 0.0)
    return (a - lo) / (hi - lo)


def _mean_frames(data: np.ndarray, start: int, stop: int) -> np.ndarray:
    """mean(data(:, :, start+1:stop), 3) in float64; NaN when the range is empty."""
    if stop <= start:
        return np.full(data.shape[:2], np.nan)
    return data[:, :, start:stop].mean(axis=2, dtype=np.float64)


def retinotopy_frame_triggers(
    cam_exp, shutter, downs_factor: int = 5, *, threshold: float = ANALOG_THR
) -> np.ndarray:
    """Aux samples of the loaded camera frames.

    A frame is a rising edge of the camera exposure trace (``cam_exp[i] <= threshold <
    cam_exp[i+1]``) while the shutter is open (``shutter[i] > threshold``). Every
    ``downs_factor``-th of them is kept, starting with the first, matching a movie loaded
    with ``load_wid_data(path, frames=downs_factor)``.

    Parameters
    ----------
    cam_exp, shutter
        1-D aux traces (V), same length.
    downs_factor
        Temporal down-sampling factor of the loaded movie (integer >= 1).
    threshold
        Voltage threshold (MATLAB ``ANALOG_THR``).

    Returns
    -------
    numpy.ndarray
        **0-based** sample index ``i`` (the sample before the rising edge) of each kept
        frame, 1-D int64. MATLAB's ``frame_trigs`` is this plus one.
    """
    c = np.asarray(cam_exp, dtype=np.float64).reshape(-1)
    s = np.asarray(shutter, dtype=np.float64).reshape(-1)
    if c.shape != s.shape:
        raise ValueError(f"cam_exp and shutter differ in length: {c.size} and {s.size}")
    step = _check_downs_factor(downs_factor)
    edges = np.flatnonzero((c[:-1] <= threshold) & (c[1:] > threshold) & (s[:-1] > threshold))
    return edges[::step].astype(np.int64)


def retinotopy_stimulus_triggers(
    stim_id,
    downs_factor: int = 5,
    *,
    threshold: float = ANALOG_THR,
    raw_framerate: float = RAW_FRAMERATE,
) -> StimulusTriggers:
    """Stimulus triggers and their types from the down-sampled StimulusID trace.

    A trigger is a rising edge ``stim_id[t] <= threshold < stim_id[t+1]``. Its type is the
    median of the ``round(raw_framerate / downs_factor)`` samples after ``t`` (about one
    second; fewer at the end of the trace), rounded to one decimal (MATLAB
    ``round(median(...), 1)``).

    Parameters
    ----------
    stim_id
        1-D StimulusID trace sampled at the loaded frames (``aux[:, frame_trigs]``), V.
    downs_factor
        Temporal down-sampling factor (integer >= 1).
    threshold, raw_framerate
        MATLAB ``ANALOG_THR`` and ``RAW_FRAMERATE`` (Hz).

    Returns
    -------
    StimulusTriggers
        ``trigs`` (**0-based** frame indices ``t``; MATLAB ``trigs`` minus one),
        ``trig_type`` and ``trigger_types`` (``unique(trig_type)``, shape ``(1, n)``).

    Notes
    -----
    Unverified: ``round(x, 1)`` is computed as ``round_half_away(10 * x) / 10``; MATLAB's
    implementation may differ in the last bit for some values.
    """
    sid = np.asarray(stim_id, dtype=np.float64).reshape(-1)
    step = _check_downs_factor(downs_factor)
    trigs = _rising_edges(sid, threshold).astype(np.int64)
    n_one_second = int(mround(raw_framerate / step))
    trig_type = np.zeros(trigs.size)
    for k, t in enumerate(trigs):
        window = sid[t + 1 : min(t + 1 + n_one_second, sid.size)]
        med = np.median(window) if window.size else np.nan
        trig_type[k] = mround(med * 10.0) / 10.0
    return StimulusTriggers(trigs, trig_type, np.unique(trig_type).reshape(1, -1))


def retinotopy_response_maps(
    data,
    trigs,
    trig_type,
    *,
    downs_factor: int = 5,
    resp_avg_seconds: float = 1.5,
    raw_framerate: float = RAW_FRAMERATE,
) -> RetinotopyMaps:
    """Mean ``(post - pre) / pre`` response map per stimulus type.

    With ``n = round(raw_framerate * resp_avg_seconds / downs_factor)`` and a trigger at
    0-based frame ``t``, ``pre`` is the mean of frames ``max(0, t - n) .. t - 1`` and
    ``post`` the mean of frames ``t + 3 .. min(t + n + 2, nFrames - 1)`` (0-based,
    inclusive; MATLAB ``from:this_trigs(j)-1`` and ``this_trigs(j)+1+2:to``). An empty
    window gives NaN, as MATLAB's mean over an empty range. The response maps of all
    triggers of one type are averaged.

    Parameters
    ----------
    data
        Down-sampled movie ``[height x width x nFrames]`` (any numeric dtype).
    trigs
        **0-based** trigger frames (see :func:`retinotopy_stimulus_triggers`).
    trig_type
        Type of each trigger.
    downs_factor, resp_avg_seconds, raw_framerate
        MATLAB ``opts.downs_factor``, ``opts.resp_avg_seconds`` (s), ``RAW_FRAMERATE`` (Hz).

    Returns
    -------
    RetinotopyMaps
        ``resp_maps``: float64 ``[height x width x nTypes]``, one map per entry of
        ``trigger_types`` (``unique(trig_type)``, shape ``(1, nTypes)``).

    Raises
    ------
    IndexError
        A pre-stimulus window reaches beyond the movie (MATLAB: index out of bounds).

    Notes
    -----
    Differences from MATLAB: float32 movies are averaged in float64 (MATLAB's ``mean`` of
    ``single`` stays single). ``.wid`` movies are uint16, for which MATLAB also averages in
    double.
    """
    d = np.asarray(data)
    if d.ndim != 3:
        raise ValueError(f"data must be [height x width x nFrames], got shape {d.shape}")
    t_all = np.asarray(trigs, dtype=np.int64).reshape(-1)
    types_all = np.asarray(trig_type, dtype=np.float64).reshape(-1)
    if t_all.shape != types_all.shape:
        raise ValueError("trigs and trig_type differ in length")
    step = _check_downs_factor(downs_factor)
    n_avg = int(mround(raw_framerate * resp_avg_seconds / step))
    n_frames = d.shape[2]
    trigger_types = np.unique(types_all)
    resp_maps = np.zeros(d.shape[:2] + (trigger_types.size,))
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, ttype in enumerate(trigger_types):
            these = t_all[types_all == ttype]
            responses = np.zeros(d.shape[:2] + (these.size,))
            for j, t in enumerate(these):
                start = max(0, t - n_avg)
                if t > n_frames and start < t:
                    raise IndexError(f"trigger at frame {t} (0-based) beyond {n_frames} frames")
                pre = _mean_frames(d, start, t)
                post = _mean_frames(d, t + 3, min(t + n_avg + 3, n_frames))
                responses[:, :, j] = (post - pre) / pre
            resp_maps[:, :, i] = responses.mean(axis=2)
    return RetinotopyMaps(resp_maps, trigger_types.reshape(1, -1))


def retinotopy_anatomy(data) -> np.ndarray:
    """8-bit mean image of frames 4 to 104 (0-based; MATLAB 5:105) of the down-sampled movie.

    MATLAB ``uint8(rescale(mean(data(:,:,5:min(105, end)), 3)) * 255)``: the mean is
    scaled to [0, 1] (min to max) and converted to uint8 (round half away, NaN -> 0).

    Parameters
    ----------
    data
        ``[height x width x nFrames]``.

    Returns
    -------
    numpy.ndarray
        ``[height x width]`` uint8. With fewer than 5 frames the mean is empty and the
        image is all zeros (MATLAB: NaN converted to 0). A constant mean gives zeros.

    Notes
    -----
    Unverified: MATLAB ``rescale`` may differ in the last bit from ``(x - min) / (max -
    min)``, which can change a uint8 value at an exact .5 boundary.
    """
    d = np.asarray(data)
    if d.ndim != 3:
        raise ValueError(f"data must be [height x width x nFrames], got shape {d.shape}")
    mean_img = _mean_frames(d, 4, min(105, d.shape[2]))
    return matlab_cast(_rescale(mean_img) * 255.0, np.uint8)


def filter_with_disk_kernel(input_matrix, radius: int) -> np.ndarray:
    """Sum over a disk of radius ``radius`` around every pixel, border replicated.

    Port of the local function ``filterWithDiskKernel`` in ``online_retinotopy.m``. The
    kernel is ``1`` where ``sqrt(dx^2 + dy^2) <= radius`` on the integer grid
    ``-radius .. radius`` and 0 elsewhere (not normalised: the result is a sum). MATLAB
    looped over pixels on a manually replicate-padded copy; this is
    ``scipy.ndimage.correlate(..., mode='nearest')``, which is the same operation.

    Parameters
    ----------
    input_matrix
        2-D array.
    radius
        Non-negative integer radius in pixels.

    Returns
    -------
    numpy.ndarray
        float64, same shape as ``input_matrix``.

    Notes
    -----
    Differences from MATLAB: a non-integer or negative radius raises ``ValueError``
    (MATLAB failed on the non-integer index).
    """
    a = np.asarray(input_matrix, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"input_matrix must be 2-D, got shape {a.shape}")
    if not (float(radius) == int(radius) and int(radius) >= 0):
        raise ValueError(f"radius must be a non-negative integer, got {radius!r}")
    r = int(radius)
    if a.size == 0:
        return a.copy()
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    kernel = (np.sqrt(x**2 + y**2) <= r).astype(np.float64)
    return scipy.ndimage.correlate(a, kernel, mode="nearest")


def _labels(aux_chans) -> list[str]:
    arr = np.asarray(aux_chans, dtype=object)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return [str(x) for x in arr[:, 1]]
    if arr.ndim == 1:
        return [str(x) for x in arr]
    if arr.size == 0:
        return []
    raise ValueError("aux_chans must be the [n x 2] channel table or a list of labels")


def _channel(aux: np.ndarray, labels: Sequence[str], name: str) -> np.ndarray:
    hits = [i for i, lab in enumerate(labels) if lab.lower() == name.lower()]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one aux channel {name!r}, found {len(hits)}")
    if hits[0] >= aux.shape[0]:
        raise ValueError(f"aux channel {name!r} is row {hits[0]} but aux has {aux.shape[0]}")
    return aux[hits[0], :]


def online_retinotopy(
    data, aux, aux_chans, *, downs_factor: int = 5, resp_avg_seconds: float = 1.5
) -> OnlineRetinotopy:
    """Retinotopy response maps from a down-sampled movie and its aux traces.

    Port of the computation in calliope ``12_Widefield/online_retinotopy.m``: anatomy image
    (:func:`retinotopy_anatomy`), frame triggers from the ``CamExp`` and ``Shutter``
    traces (:func:`retinotopy_frame_triggers`), the ``StimulusID`` trace at those frames,
    stimulus triggers and types (:func:`retinotopy_stimulus_triggers`) and the response
    maps (:func:`retinotopy_response_maps`).

    Parameters
    ----------
    data
        Movie ``[height x width x nFrames]`` holding every ``downs_factor``-th frame
        (``load_wid_data(path, frames=downs_factor).wid_data``).
    aux
        Aux traces ``[nChannels x nSamples]`` (``load_lvd(path).data``).
    aux_chans
        Channel table ``parse_ach(path).aux_chans`` (``[nChannels x 2]``, labels in column
        1, row ``r`` describing ``aux[r]``), or a list of labels in aux row order. Labels
        ``CamExp``, ``Shutter`` and ``StimulusID`` are matched case-insensitively.
    downs_factor
        Temporal down-sampling factor used when loading (integer >= 1).
    resp_avg_seconds
        Length of the pre- and post-stimulus windows, s.

    Returns
    -------
    OnlineRetinotopy
        ``resp_maps``, ``trigger_types`` (MATLAB outputs), and ``anatomy``,
        ``frame_trigs``, ``trigs``, ``trig_type`` (0-based indices) for plotting and checks.

    Notes
    -----
    Differences from MATLAB: a missing or repeated ``CamExp``, ``Shutter`` or
    ``StimulusID`` label raises ``ValueError`` (MATLAB went on with empty or mixed traces).
    The interactive crop and the figure are :func:`plot_online_retinotopy`; no progress
    is printed.
    """
    aux_arr = np.asarray(aux, dtype=np.float64)
    if aux_arr.ndim != 2:
        raise ValueError(f"aux must be [nChannels x nSamples], got shape {aux_arr.shape}")
    labels = _labels(aux_chans)
    anatomy = retinotopy_anatomy(data)
    frame_trigs = retinotopy_frame_triggers(
        _channel(aux_arr, labels, "CamExp"), _channel(aux_arr, labels, "Shutter"), downs_factor
    )
    stim_id = _channel(aux_arr, labels, "StimulusID")[frame_trigs]
    st = retinotopy_stimulus_triggers(stim_id, downs_factor)
    maps = retinotopy_response_maps(
        data, st.trigs, st.trig_type, downs_factor=downs_factor, resp_avg_seconds=resp_avg_seconds
    )
    return OnlineRetinotopy(
        maps.resp_maps, maps.trigger_types, anatomy, frame_trigs, st.trigs, st.trig_type
    )


def online_retinotopy_from_files(
    wid_path: str | PathLike,
    lvd_path: str | PathLike,
    ach_path: str | PathLike,
    *,
    downs_factor: int = 5,
    resp_avg_seconds: float = 1.5,
) -> OnlineRetinotopy:
    """:func:`online_retinotopy` from the ``.wid``, ``.lvd`` and ``.ach`` files of a stack.

    Replaces the file handling of calliope ``12_Widefield/online_retinotopy.m``, which
    found the files by stack number in a temp-data folder.

    Parameters
    ----------
    wid_path, lvd_path, ach_path
        The movie, the aux traces and the aux channel table of one stack.
    downs_factor, resp_avg_seconds
        As in :func:`online_retinotopy`; the movie is loaded with every ``downs_factor``-th
        frame.

    Returns
    -------
    OnlineRetinotopy

    Notes
    -----
    Deferred: choosing the last (or n-th to last) recorded stack in the temp folder
    (``stack_id <= 0``) and the default aux folder.
    """
    from calliopy.io import load_lvd, load_wid_data, parse_ach

    step = _check_downs_factor(downs_factor)
    data = load_wid_data(wid_path, frames=step).wid_data
    aux = load_lvd(lvd_path).data
    ach = parse_ach(ach_path)
    return online_retinotopy(
        data, aux, ach["aux_chans"], downs_factor=step, resp_avg_seconds=resp_avg_seconds
    )


def crop_rect_from_points(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Crop rectangle ``pos`` for :func:`plot_online_retinotopy` from two clicked corners.

    Replaces MATLAB's ``pos = round([x1, y1, x2-x1, y2-y1])`` from two ``ginput`` clicks,
    in 0-based coordinates.

    Parameters
    ----------
    x1, y1, x2, y2
        Corner coordinates in **0-based pixel coordinates**, x = column, y = row, pixel
        centres at integers: the data coordinates of a matplotlib ``imshow`` of the image
        (MATLAB ``ginput`` values minus 1). ``(x1, y1)`` is the top-left corner.

    Returns
    -------
    numpy.ndarray
        ``[x, y, w, h]`` float64: ``x``, ``y`` the **0-based** column and row of the
        first pixel (``floor(x1 + 0.5)``, ``floor(y1 + 0.5)``), ``w``, ``h`` the extents
        ``x2 - x1``, ``y2 - y1`` rounded half away from zero (MATLAB ``round``).

    Notes
    -----
    Differences from MATLAB: coordinates in and out are MATLAB's minus 1 in ``x`` and
    ``y`` (``w``, ``h`` unchanged). ``x``, ``y`` are rounded as MATLAB rounds the 1-based
    value (half up), so the result equals MATLAB's ``pos`` minus ``[1, 1, 0, 0]`` for any
    click inside the image.
    """
    xy = np.floor(np.array([x1, y1], dtype=np.float64) + 0.5)
    wh = mround(np.array([x2 - x1, y2 - y1], dtype=np.float64))
    return np.concatenate([xy, wh])


def _crop(img: np.ndarray, pos) -> np.ndarray:
    """Rows ``y .. y+h`` and columns ``x .. x+w`` (0-based, inclusive) of ``img``."""
    x, y, w, h = (int(v) for v in np.asarray(pos, dtype=np.float64).reshape(-1)[:4])
    if w < 0 or h < 0:
        raise ValueError(f"crop width and height must be >= 0, got pos={list(pos)}")
    if x < 0 or y < 0 or y + h >= img.shape[0] or x + w >= img.shape[1]:
        raise IndexError(f"crop rectangle {list(pos)} exceeds the image {img.shape[:2]}")
    return img[y : y + h + 1, x : x + w + 1]


def _display(img: np.ndarray) -> np.ndarray:
    """MATLAB fliplr(rot90(img, 3)) (numpy and MATLAB rot90 both turn counterclockwise)."""
    return np.fliplr(np.rot90(img, 3))


def plot_online_retinotopy(
    resp_maps,
    trigger_types,
    anatomy,
    pos,
    *,
    sp_filter_size_px: int = 20,
    stack_id: int | None = None,
) -> Figure:
    """The three retinotopy panels of ``online_retinotopy.m``.

    Panels: full-field response (type 2, red), horizontal retinotopy (types {0.5, 1} red,
    {1.5, 2.5} cyan, {3, 3.5} yellow) and vertical retinotopy ({0.5, 1.5, 3} red,
    {1, 2.5, 3.5} cyan). For each group the maps of its types are averaged, cropped to
    ``pos``, summed over a disk (:func:`filter_with_disk_kernel`), scaled to [0, 1] and
    shown in the group colour with opacity ``resp^2 - 0.2`` (clipped to [0, 1]) over the
    cropped anatomy. All images are shown as ``fliplr(rot90(img, 3))``, as in MATLAB.

    Parameters
    ----------
    resp_maps, trigger_types, anatomy
        From :func:`online_retinotopy`.
    pos
        Crop rectangle ``[x, y, w, h]``: **0-based** column ``x`` and row ``y`` of the
        top-left pixel and the extents; rows ``y .. y+h`` and columns ``x .. x+w``
        (inclusive, ``h+1`` rows and ``w+1`` columns, as MATLAB's
        ``pos(2):pos(2)+pos(4)``, ``pos(1):pos(1)+pos(3)``) are kept. MATLAB ``pos`` minus
        ``[1, 1, 0, 0]``; see :func:`crop_rect_from_points`. Replaces the two ``ginput``
        clicks.
    sp_filter_size_px
        Disk radius in pixels (MATLAB ``opts.sp_filter_size_px``).
    stack_id
        Only used in the figure label.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Differences from MATLAB: the crop is an argument, 0-based (MATLAB ``pos`` minus
    ``[1, 1, 0, 0]``, i.e. ``ginput`` values minus 1); a rectangle outside the image raises
    ``IndexError`` (MATLAB: index error) and a negative width or height ``ValueError``
    (MATLAB: empty panels). A group without any recorded type gives a NaN map and no
    overlay (MATLAB: NaN alpha). Figure size and fonts are matplotlib's, not
    ``screen2pos``.
    """
    maps = np.asarray(resp_maps, dtype=np.float64)
    types = np.asarray(trigger_types, dtype=np.float64).reshape(-1)
    if maps.ndim != 3 or maps.shape[2] != types.size:
        raise ValueError("resp_maps must be [height x width x numel(trigger_types)]")
    anat = _display(_crop(np.asarray(anatomy), pos))
    fig = Figure(figsize=(12, 4.5), facecolor=_BACKGROUND, layout="constrained")
    label = "Retinotopy map" if stack_id is None else f"Retinotopy map - StackID: {stack_id}"
    fig.suptitle(label, color="white")
    axes = fig.subplots(1, 3)
    for ax, groups, title in zip(axes, _PANEL_GROUPS, _PANEL_TITLES, strict=True):
        ax.imshow(anat, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        for group, color in zip(groups, _PANEL_COLORS, strict=False):
            sel = np.isin(types, group)
            if sel.any():
                this_resp = maps[:, :, sel].mean(axis=2)
            else:
                this_resp = np.full(maps.shape[:2], np.nan)
            resp = filter_with_disk_kernel(_crop(this_resp, pos), sp_filter_size_px)
            resp = _display(_rescale(resp))
            rgba = np.zeros(resp.shape + (4,))
            rgba[..., :3] = np.nan_to_num(resp)[..., None] * np.asarray(color)
            rgba[..., 3] = np.nan_to_num(np.clip(resp**2 - 0.2, 0.0, 1.0))
            ax.imshow(rgba, interpolation="nearest")
        ax.set_title(title, color="white", fontsize=16)
        ax.set_axis_off()
    return fig
