"""ROI activity of widefield stacks, frame times and median division.

Ports of ``calc_widefield_act.m``, ``get_frame_times_widefield.m`` and
``median_divide_video.m`` (calliope ``12_Widefield``).
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Sequence
from datetime import datetime
from typing import Any, NamedTuple

import numpy as np
import scipy.sparse

from .._matlab import mround
from ..io import MatStruct, load_roi_data, load_wid_data

logger = logging.getLogger(__name__)

__all__ = [
    "CORNER_ROI_NAMES",
    "CornerRois",
    "FrameTimesWidefield",
    "calc_widefield_act",
    "corner_rois",
    "frame_times_widefield",
    "median_divide_video",
    "roi_activity",
]

#: Names of the four corner ROIs, in the order ``calc_widefield_act`` appends them.
CORNER_ROI_NAMES = ("upper left corner", "upper right corner", "bottom left corner",
                    "bottom right corner")

# Memory budget (bytes) of one chunk of frames in the ROI averaging.
_CHUNK_BYTES = 256 * 2**20


def _empty() -> np.ndarray:
    """MATLAB ``[]`` as stored in a struct field."""
    return np.empty((0, 0))


# --- ROI averaging --------------------------------------------------------------------------


def _as_index_vector(ix: Any, n_pixels: int) -> np.ndarray:
    """1-based MATLAB linear indices (any shape) -> 0-based int64 vector, validated."""
    a = np.asarray(ix)
    if a.size == 0:
        return np.zeros(0, dtype=np.int64)
    if a.dtype.kind not in "biuf":
        raise ValueError(f"ROI indices must be numeric, got dtype {a.dtype}")
    a = a.ravel(order="F")
    if a.dtype.kind == "f" and not np.all(np.isfinite(a) & (a == np.trunc(a))):
        raise ValueError("ROI indices must be integer valued (MATLAB linear indices)")
    a = a.astype(np.int64)
    if a.min() < 1 or a.max() > n_pixels:
        raise IndexError(f"ROI indices must lie in 1..{n_pixels} (1-based linear indices)")
    return a - 1


def _is_marked(ix: Any) -> bool:
    """MATLAB's test for an 'empty' ROI: no indices, or all indices equal to 1."""
    a = np.asarray(ix)
    return a.size == 0 or bool(np.all(a == 1))


def _as_stack(data: Any) -> Any:
    shape = np.shape(data)
    if len(shape) == 2:
        return np.asarray(data)[:, :, np.newaxis]
    if len(shape) != 3:
        raise ValueError(f"data must be [height x width x nFrames], got shape {shape}")
    return data


def _roi_means(data: Any, indices: Sequence[Any], *,
               chunk_frames: int | None = None) -> np.ndarray:
    """Mean over the pixels of every ROI for every frame; float32 ``[nROIs x nFrames]``.

    Empty ROIs give NaN (MATLAB ``mean`` of an empty selection). ``indices`` are 1-based
    column-major linear indices. ``data`` is read in chunks of frames, so a memory map is
    never loaded whole. Sums are exact (float64); the mean is rounded once to float32.
    """
    data = _as_stack(data)
    h, w, n_frames = data.shape
    n_pix = h * w
    idx = [_as_index_vector(ix, n_pix) for ix in indices]
    n_rois = len(idx)
    out = np.full((n_rois, n_frames), np.nan, dtype=np.float32)
    counts = np.array([len(i) for i in idx], dtype=np.int64)
    if n_rois == 0 or n_frames == 0 or counts.sum() == 0:
        return out
    all_idx = np.concatenate(idx)
    uniq, inv = np.unique(all_idx, return_inverse=True)
    rows = np.repeat(np.arange(n_rois), counts)
    # summing matrix: duplicated indices within one ROI count twice, as in MATLAB
    summer = scipy.sparse.csr_array(
        (np.ones(all_idx.size), (rows, inv.ravel())), shape=(n_rois, uniq.size))
    if chunk_frames is None:
        per_frame = max(8 * uniq.size, data.dtype.itemsize * n_pix)
        chunk_frames = max(1, _CHUNK_BYTES // per_frame)
    chunk_frames = int(chunk_frames)
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be >= 1")
    nonempty = counts > 0
    for t0 in range(0, n_frames, chunk_frames):
        t1 = min(n_frames, t0 + chunk_frames)
        block = np.asarray(data[:, :, t0:t1])
        flat = block.reshape(n_pix, t1 - t0, order="F")
        sums = summer @ flat[uniq].astype(np.float64)
        out[nonempty, t0:t1] = (sums[nonempty] / counts[nonempty, None]).astype(np.float32)
    return out


def roi_activity(data: Any, indices: Sequence[Any], *,
                 chunk_frames: int | None = None) -> np.ndarray:
    """Mean activity of every ROI in every frame of a widefield stack.

    The ROI loop of ``calc_widefield_act.m`` (calliope ``12_Widefield``), written once for all
    channels and saving modes.

    Parameters
    ----------
    data
        Image stack ``[height x width x nFrames]`` (any numeric dtype, e.g. the ``uint16``
        memory map of :func:`calliopy.io.load_wid_data`); a 2-D image counts as one frame.
        The stack is read in chunks of frames and never loaded whole.
    indices
        One entry per ROI: the ROI's pixels as MATLAB linear indices, **1-based,
        column-major** (``ROIs(i).indices``), any shape; an empty entry is an empty ROI.
    chunk_frames
        Frames per chunk; default keeps a chunk near 256 MB.

    Returns
    -------
    numpy.ndarray
        float32 ``[nROIs x nFrames]``: row ``i`` is MATLAB's ``ROIs(i).activity'``, the mean
        of the ROI's pixels after conversion to single. ROIs with no indices, **or whose
        indices are all equal to 1**, are NaN (see Notes).

    Raises
    ------
    ValueError
        ``data`` not 2-D/3-D, indices not integer valued.
    IndexError
        An index outside ``1 .. height*width`` (MATLAB errors too).

    Notes
    -----
    MATLAB marks an empty ROI by setting its indices to ``1`` and afterwards overwrites the
    activity of every ROI whose indices are all ``1`` with NaN. A genuine ROI consisting of
    pixel 1 only (``indices == 1``, or ``[1; 1; ...]``) is therefore NaN as well; this is
    kept.

    Differences from MATLAB:

    - MATLAB averages in single precision (``mean`` of a single matrix); here the pixel sum
      is exact (float64) and the mean is rounded once to float32. Results can differ in the
      last bit of single precision.
    """
    out = _roi_means(data, indices, chunk_frames=chunk_frames)
    for i, ix in enumerate(indices):
        if _is_marked(ix):
            out[i] = np.nan
    return out


# --- corner ROIs ----------------------------------------------------------------------------


class CornerRois(NamedTuple):
    """Result of :func:`corner_rois`."""

    indices: list
    """Four float64 ``[ROI_len^2 x 1]`` arrays of 1-based column-major linear indices."""
    names: tuple
    """:data:`CORNER_ROI_NAMES`."""
    roi_len: int
    """Side length of the square corner ROIs (pixels)."""


def corner_rois(rois_indices: Sequence[Any], frame_size: Sequence[int], *,
                two_channel: bool = False) -> CornerRois:
    """The four corner ROIs that ``calc_widefield_act`` appends to the ROI set.

    Part of ``calc_widefield_act.m`` (calliope ``12_Widefield``).

    The side length is ``round(sqrt(numel(indices)))`` of a reference ROI: ROI number 387
    (1-based) when there are at least 387 ROIs, else ROI number 5. In two-channel data
    MATLAB always uses ROI 387 (of ``indices_ch525``).

    Parameters
    ----------
    rois_indices
        Indices of the existing ROIs, in ROI order (only the count and the reference ROI's
        number of indices are used).
    frame_size
        ``(height, width)`` of the frame (MATLAB ``size(first_frame)``).
    two_channel
        Use MATLAB's two-channel rule (always ROI 387).

    Returns
    -------
    CornerRois
        ``indices``: four float64 column vectors of **1-based, column-major** linear indices
        in the order and with the names of :data:`CORNER_ROI_NAMES`. Within a corner the
        pixels are listed row by row (row outer, column inner), as MATLAB's
        ``sub2ind(frame_size, x(:), y(:))`` with ``[x, y] = meshgrid(rows, cols)`` gives:
        MATLAB's ``x`` holds row numbers and ``y`` column numbers. The corners are rows
        ``1..L`` / ``H-L+1..H`` and columns ``1..L`` / ``W-L+1..W``.

    Raises
    ------
    ValueError
        Fewer than 5 ROIs (fewer than 387 for two channels), or a side length larger than
        the frame.

    Notes
    -----
    Differences from MATLAB: errors are :class:`ValueError` (two channels with fewer than
    387 ROIs and corners larger than the frame are MATLAB index errors).
    """
    n = len(rois_indices)
    if two_channel:
        if n < 387:
            raise ValueError(f"two-channel corner ROIs need >= 387 ROIs, got {n}")
        ref = 387
    elif n >= 387:
        ref = 387
    elif n >= 5:
        ref = 5
    else:
        raise ValueError(f"Cannot select the 5th ROI: only {n} ROIs available")
    roi_len = int(mround(np.sqrt(np.asarray(rois_indices[ref - 1]).size)))
    height, width = (int(v) for v in tuple(frame_size)[:2])
    if roi_len > min(height, width):
        raise ValueError(f"corner ROI side {roi_len} exceeds the frame size {(height, width)}")
    low = np.arange(1, roi_len + 1)
    corners = [(low, low),
               (low, np.arange(width - roi_len + 1, width + 1)),
               (np.arange(height - roi_len + 1, height + 1), low),
               (np.arange(height - roi_len + 1, height + 1),
                np.arange(width - roi_len + 1, width + 1))]
    out = []
    for rows, cols in corners:
        r, c = np.meshgrid(rows, cols, indexing="ij")  # row outer, column inner
        lin = (r.ravel() - 1) + (c.ravel() - 1) * height + 1
        out.append(lin.astype(np.float64).reshape(-1, 1))
    return CornerRois(out, CORNER_ROI_NAMES, roi_len)


# --- calc_widefield_act ---------------------------------------------------------------------


def _struct_list(rois: Any) -> list:
    """ROIs (MATLAB cell {struct array} or struct array) -> list of MatStruct copies."""
    a = rois
    if isinstance(a, np.ndarray) and a.dtype == object and a.size and isinstance(
            a.flat[0], (np.ndarray, list, tuple)):
        if a.size > 1:
            raise ValueError("Can't deal with several types of ROIs yet (ROIs has > 1 cell)")
        a = a.flat[0]
    if isinstance(a, np.ndarray):
        items = list(a.ravel(order="F"))
    elif isinstance(a, dict):
        items = [a]
    else:
        items = list(a)
    return [MatStruct(dict(s)) for s in items]


def _finalize_structs(items: list) -> np.ndarray:
    """Give every element the same fields (missing -> []) and return a (1, n) struct array."""
    keys: list = []
    for s in items:
        keys.extend(k for k in s if k not in keys)
    arr = np.empty((1, len(items)), dtype=object)
    for j, s in enumerate(items):
        arr[0, j] = MatStruct({k: s[k] if k in s else _empty() for k in keys})
    return arr


def _as_cell(struct_array: np.ndarray) -> np.ndarray:
    cell = np.empty((1, 1), dtype=object)
    cell[0, 0] = struct_array
    return cell


def _append_corners(items: list, frame_size: Sequence[int], fields: Sequence[str],
                    two_channel: bool) -> None:
    keys = list(items[0].keys()) if items else []
    cr = corner_rois([s.get(fields[0], _empty()) for s in items], frame_size,
                     two_channel=two_channel)
    for ix, name in zip(cr.indices, cr.names):
        new = MatStruct({k: _empty() for k in keys})
        for f in fields:
            new[f] = ix.copy()
        new["name"] = name
        items.append(new)


def _paths(p: Any) -> list:
    if p is None:
        return []
    if isinstance(p, (str, os.PathLike)):
        return [p]
    return list(p)


def _wid_activity(items: list, wid_fnames: list, index_field: str, act_field: str,
                  chunk_frames: int | None) -> None:
    """Full-frame mode: ROI activity per .wid stack, concatenated over stacks."""
    indices = [s.get(index_field, _empty()) for s in items]
    marked = [_is_marked(ix) for ix in indices]
    parts: list = [[] for _ in items]
    for path in wid_fnames:
        data = load_wid_data(path).wid_data
        act = _roi_means(data, indices, chunk_frames=chunk_frames)
        for i in range(len(items)):
            if marked[i]:  # MATLAB: NaN(size(data, 3), 1), class double
                parts[i].append(np.full((data.shape[2], 1), np.nan))
            else:
                parts[i].append(act[i].reshape(-1, 1))
        logger.info("Calculated activity for the stack %s", path)
        del data
    for s, p in zip(items, parts):
        s[act_field] = np.concatenate(p, axis=0) if p else _empty()


def _roi_file_activity(items: list, roi_fnames: list, wii_fnames: list | None) -> None:
    """Trace-imaging mode: trace i-2 of each .roi file -> ROI i (1-based), i = 3..n-4."""
    n = len(items)
    marked = [_is_marked(s.get("indices", _empty())) for s in items]
    parts: list = [[] for _ in items]
    for k, path in enumerate(roi_fnames):
        wii = None if wii_fnames is None else wii_fnames[k]
        data = np.asarray(load_roi_data(path, wii_path=wii).roi_data, dtype=np.float32)
        n_frames = data.shape[2]
        traces = data.reshape(-1, n_frames, order="F")
        if traces.shape[0] < n - 6:
            raise ValueError(f"{path}: {traces.shape[0]} traces for {n - 6} trace ROIs")
        for i in range(n):  # 0-based i <-> MATLAB i+1
            if marked[i]:
                parts[i].append(np.full((n_frames, 1), np.nan))
            elif 2 <= i <= n - 5:
                parts[i].append(traces[i - 2].reshape(-1, 1))
            # other ROIs (reference points, corners) keep MATLAB's [] activity
        logger.info("Calculated .roi activity for the stack %s", path)
    for s, p in zip(items, parts):
        s["activity"] = np.concatenate(p, axis=0) if p else _empty()


def calc_widefield_act(rois: Any, frame_size: Sequence[int], *,
                       wid_fnames: Any = None, roi_fnames: Any = None,
                       wid_fnames_ch525: Any = None, wid_fnames_ch610: Any = None,
                       frame_size_ch610: Sequence[int] | None = None,
                       registration_parameters: Any = None, wii_fnames: Any = None,
                       chunk_frames: int | None = None) -> MatStruct:
    """ROI activity of a widefield recording, computed stack by stack.

    Port of the computation in ``calc_widefield_act.m`` (calliope ``12_Widefield``). Appends
    the four corner ROIs (:func:`corner_rois`), computes every ROI's activity
    (:func:`roi_activity`) for each stack and concatenates the stacks.

    The saving mode follows MATLAB:

    1. ``roi_fnames`` not given: full-frame imaging. One channel (ROIs with ``indices``):
       ``activity`` from ``wid_fnames``. Two channels (ROIs with ``indices_ch525``):
       ``activity_ch525`` / ``activity_ch610`` from ``wid_fnames_ch525`` /
       ``wid_fnames_ch610`` (``indices_ch525`` / ``indices_ch610``).
    2. ``roi_fnames`` given and ``wid_fnames`` given with all files present: ``ROIs_wid``
       gets the full-frame activity from the ``.wid`` files, ``ROIs`` the trace activity
       from the ``.roi`` files.
    3. ``roi_fnames`` given, no (or missing) ``.wid`` files: ``ROIs`` gets the trace
       activity.

    In the ``.roi`` modes trace ``i-2`` of the ``.roi`` file becomes the activity of ROI
    ``i`` for ``i = 3 .. n-4`` (1-based; ``n`` counts the corner ROIs); ROIs 1-2 (reference
    points) and the corners keep ``[]``.

    Parameters
    ----------
    rois
        The adata variable ``ROIs``: a MATLAB cell holding one struct array (as returned by
        :func:`calliopy.io.load_adata`), or the struct array itself (object array or list of
        :class:`~calliopy.io.MatStruct`). Not modified.
    frame_size
        ``(height, width)``: ``size(first_frame)``, or ``size(first_frame_ch525)``.
    wid_fnames, wid_fnames_ch525, wid_fnames_ch610
        ``.wid`` file(s), one per stack, in recording order.
    roi_fnames
        ``.roi`` file(s), one per stack.
    frame_size_ch610
        ``size(first_frame_ch610)``, two-channel data only; must equal ``frame_size``.
    registration_parameters
        The adata variable ``registration_parameters`` (copied, ``activity_calculated`` is
        set to 1). ``None`` gives a new struct, as MATLAB does when the variable is absent
        (always in two-channel data, see Notes).
    wii_fnames
        ``.wii`` files for the ``.roi`` files (default: same name, see
        :func:`calliopy.io.load_roi_data`).
    chunk_frames
        Frames read at a time from a ``.wid`` file (default: about 256 MB).

    Returns
    -------
    MatStruct
        What MATLAB appends to the adata file, with MATLAB names: ``ROIs`` (a ``(1, 1)``
        object array holding the ``(1, n+4)`` struct array, with the corner ROIs and the
        activity fields), in mode 2 ``ROIs_wid``, in modes 2/3 ``meta_info`` (``.roi`` meta
        info of all files without ``dimension``), and ``registration_parameters``.
        ``indices`` are 1-based. Activities are ``[nFrames x 1]`` columns: float32, or
        float64 NaN for an empty ROI (MATLAB ``NaN(n, 1)`` is double).

    Raises
    ------
    ValueError
        Several ROI cells, ROIs without ``indices``/``indices_ch525``, missing files lists,
        different frame sizes of the two channels, ``.roi`` files with two-channel ROIs,
        too few traces in a ``.roi`` file.

    Notes
    -----
    Differences from MATLAB:

    - The ExpLog lookup, the "ROI activity already calculated" skip and the
      ``save(..., '-append')`` are left to the caller; existing activity fields are
      overwritten.
    - Mode 2 crashed in MATLAB (typo ``num_roisca`` in the ``.wid`` loop); fixed, the loop
      runs over all ROIs as in the other modes.
    - Different frame sizes of the two channels raise :class:`ValueError` (MATLAB printed a
      message and returned without saving).
    - ``.roi`` modes with two-channel ROIs raise; MATLAB computed nothing but still saved.
    - An empty ``wid_fnames`` list means mode 3; MATLAB's ``all(cellfun(...))`` of an empty
      list is true and would have chosen mode 2 with no ``.wid`` stack.
    - Means are exact sums rounded once to float32 (see :func:`roi_activity`).
    - In two-channel data MATLAB never loads ``registration_parameters`` (it loads the
      ``_ch525``/``_ch610`` variants) and saves a new ``registration_parameters`` holding only
      ``activity_calculated = 1``; with ``registration_parameters=None`` this is reproduced.
    """
    items = _struct_list(rois)
    if not items:
        raise ValueError("ROIs is empty")
    is1ch = "indices" in items[0]
    is2ch = "indices_ch525" in items[0]
    if not (is1ch or is2ch):
        raise ValueError("ROIs have neither 'indices' nor 'indices_ch525'")
    if is1ch:
        _append_corners(items, frame_size, ["indices"], two_channel=False)
    else:
        if frame_size_ch610 is not None and tuple(frame_size_ch610)[:2] != tuple(frame_size)[:2]:
            raise ValueError("frame sizes of the two channels are not identical")
        _append_corners(items, frame_size, ["indices_ch525", "indices_ch610"], two_channel=True)

    regp = MatStruct() if registration_parameters is None else MatStruct(
        dict(registration_parameters))
    regp["activity_calculated"] = np.float64(1.0)
    result = MatStruct()

    roi_list = _paths(roi_fnames)
    if not roi_list:  # mode 1
        if is1ch:
            if wid_fnames is None:
                raise ValueError("wid_fnames is required for one-channel ROIs")
            _wid_activity(items, _paths(wid_fnames), "indices", "activity", chunk_frames)
        else:
            if wid_fnames_ch525 is None or wid_fnames_ch610 is None:
                raise ValueError("wid_fnames_ch525 and wid_fnames_ch610 are required")
            _wid_activity(items, _paths(wid_fnames_ch525), "indices_ch525", "activity_ch525",
                          chunk_frames)
            _wid_activity(items, _paths(wid_fnames_ch610), "indices_ch610", "activity_ch610",
                          chunk_frames)
        result["ROIs"] = _as_cell(_finalize_structs(items))
        result["registration_parameters"] = regp
        return result

    if not is1ch:
        raise ValueError(".roi trace imaging is implemented for one-channel ROIs only")
    wii_list = None if wii_fnames is None else _paths(wii_fnames)
    meta = load_roi_data(roi_list, "meta", wii_path=wii_list).roi_meta_info
    meta_out = np.empty(meta.shape, dtype=object)
    for j, m in enumerate(meta.flat):
        meta_out.flat[j] = MatStruct({k: v for k, v in m.items() if k != "dimension"})
    wid_list = _paths(wid_fnames)
    mode2 = bool(wid_list) and all(os.path.exists(p) for p in wid_list)
    if mode2:
        items_wid = [MatStruct(dict(s)) for s in items]
        _wid_activity(items_wid, wid_list, "indices", "activity", chunk_frames)
    _roi_file_activity(items, roi_list, wii_list)
    result["ROIs"] = _as_cell(_finalize_structs(items))
    if mode2:
        result["ROIs_wid"] = _as_cell(_finalize_structs(items_wid))
    result["meta_info"] = meta_out
    result["registration_parameters"] = regp
    return result


# --- frame times ----------------------------------------------------------------------------


class FrameTimesWidefield(NamedTuple):
    """Result of :func:`frame_times_widefield` (MATLAB ``[frame_times, frame_times_timestamp]``)."""

    frame_times: np.ndarray
    frame_times_timestamp: np.ndarray


def _timestamp(m: Any) -> datetime:
    if isinstance(m, datetime):
        return m
    if isinstance(m, dict):
        return m["timestamp"]
    return m.timestamp


def frame_times_widefield(trig_ch: Any, shutter_ch: Any, meta_info: Any,
                          aux_sample_rate: float) -> FrameTimesWidefield:
    """Frame times of one widefield stack in auxiliary-recording samples.

    Port of ``get_frame_times_widefield.m`` (calliope ``12_Widefield``). Two estimates:

    1. ``frame_times``: rising edges of the binarised camera-exposure signal ``trig_ch``
       (threshold half-way between its min and max), restricted to the shutter-open period
       (from the first rising to the last falling edge of the binarised ``shutter_ch``).
    2. ``frame_times_timestamp``: the camera timestamps of the frames, converted to samples
       (``round(seconds * aux_sample_rate)``) and offset to ``frame_times[0]``.

    Parameters
    ----------
    trig_ch, shutter_ch
        Auxiliary channels (vectors of equal sampling).
    meta_info
        Per-frame meta info of the stack: ``wid_meta_info`` of
        ``calliopy.io.load_wid_data(path, 'meta')`` (structs with a ``timestamp``
        :class:`datetime.datetime`), or a sequence of datetimes.
    aux_sample_rate
        Sampling rate of the auxiliary recording (Hz).

    Returns
    -------
    FrameTimesWidefield
        ``frame_times``: float64, **1-based** sample indices as MATLAB stores them (the
        sample just before each rising edge, MATLAB ``find(diff(bin) == 1)``); same
        orientation as ``trig_ch`` (1-D input gives 1-D output).
        ``frame_times_timestamp``: float64 ``(1, nFrames_meta)`` row, 1-based samples.

    Raises
    ------
    ValueError
        No frame, no shutter edge, or no frame inside the shutter period (MATLAB errors).

    Warns
    -----
    UserWarning
        Inter-frame interval below 0.8 of the median ("overcounting"), or different numbers
        of frames in the two estimates. Dropped frames are logged (``logging`` warning).

    Notes
    -----
    The midnight branch adds ``86400 * (day(t) - day(t_first))`` with ``day`` the day of the
    month, as MATLAB does; a stack spanning the end of a month therefore gets wrong
    timestamps (kept as in MATLAB). Branching compares the first and last frame only.

    Differences from MATLAB: errors instead of MATLAB index/size errors; messages go to
    ``logging`` / :mod:`warnings`.
    """
    trig = np.asarray(trig_ch, dtype=np.float64)
    shape_in = trig.shape
    trig = trig.ravel()
    shut = np.asarray(shutter_ch, dtype=np.float64).ravel()
    tb = (trig > (trig.max() + trig.min()) / 2).astype(np.int8)
    ft = np.flatnonzero(np.diff(tb) == 1).astype(np.float64) + 1.0  # 1-based
    if ft.size == 0:
        raise ValueError("no rising edge in trig_ch")
    d = np.diff(ft)
    if d.size:
        period = float(np.median(d))
        n_dropped = int(np.sum(mround((d - period) / period)))
        if n_dropped > 0:
            logger.warning("%i frames were dropped by camera, interframe T=%i ms; aux "
                           "coordinates of the dropped frames: %s", n_dropped,
                           int(mround(period * 1000 / aux_sample_rate)),
                           ft[:-1][d > 1.2 * period])
        if d.min() < 0.8 * period:
            warnings.warn("Very weird: you have overcounting of frames!! Test "
                          "frame_times_widefield output on this stack", stacklevel=2)
    sb = (shut > (shut.max() + shut.min()) / 2).astype(np.int8)
    ds = np.diff(sb)
    opens = np.flatnonzero(ds == 1)
    closes = np.flatnonzero(ds == -1)
    if opens.size == 0 or closes.size == 0:
        raise ValueError("no shutter opening or closing edge in shutter_ch")
    ind_open, ind_close = opens[0] + 1, closes[-1] + 1  # 1-based
    ft = ft[(ft >= ind_open) & (ft <= ind_close)]
    if ft.size == 0:
        raise ValueError("no frame inside the shutter-open period")

    stamps = [_timestamp(m) for m in np.asarray(meta_info, dtype=object).ravel(order="F")]
    if not stamps:
        raise ValueError("meta_info is empty")

    def sec(t) -> float:
        return t.hour * 3600 + t.minute * 60 + (t.second + t.microsecond / 1e6)

    if stamps[0].day == stamps[-1].day:
        s = np.array([sec(t) for t in stamps])
    else:
        s = np.array([3600 * 24 * (t.day - stamps[0].day) + sec(t) for t in stamps])
    dt = mround((s - s[0]) * aux_sample_rate)
    fts = (dt + ft[0]).reshape(1, -1)
    if fts.size != ft.size:
        warnings.warn(f"Number of frames in frame_times and in frame_times_timestamp do not "
                      f"coincide. N_frame_times={ft.size}, "
                      f"N_frame_times_timestamp={fts.size}", stacklevel=2)
    if len(shape_in) == 2 and shape_in[0] == 1:
        ft = ft.reshape(1, -1)
    elif len(shape_in) == 2:
        ft = ft.reshape(-1, 1)
    return FrameTimesWidefield(ft, fts)


# --- median division ------------------------------------------------------------------------


def median_divide_video(data: Any) -> np.ndarray:
    """Divide every pixel's time course by its median over frames.

    Port of ``median_divide_video.m`` (calliope ``12_Widefield``):
    ``data ./ median(data, 3)``.

    Parameters
    ----------
    data
        Video ``[height x width x nFrames]``. Integer data is converted to float32 first
        (MATLAB ``single``); float data keeps its dtype.

    Returns
    -------
    numpy.ndarray
        Same shape, float32 (integer or float32 input) or float64. A pixel with median 0
        gives Inf/NaN; NaN in a pixel makes its median NaN (MATLAB default).

    Raises
    ------
    ValueError
        ``data`` is not 3-D.

    Notes
    -----
    Differences from MATLAB: the message about the integer conversion goes to ``logging``.
    The whole array is loaded into memory, as in MATLAB. Boolean input is divided as
    float64 (MATLAB's handling of logical input is unverified).
    """
    a = np.asarray(data)
    if a.ndim != 3:
        raise ValueError("What you provided is not a video (data must be 3-D)")
    if a.dtype.kind in "iu":
        logger.info("Integer data converted to single for the median division")
        a = a.astype(np.float32)
    elif a.dtype.kind == "b":
        a = a.astype(np.float64)
    med = np.median(a, axis=2, keepdims=True).astype(a.dtype)
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / med
