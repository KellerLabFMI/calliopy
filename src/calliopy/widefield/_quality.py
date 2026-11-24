"""Quality checks of widefield adata: activity and image similarity.

Ports of the computation in ``wid_adata_check.m`` (calliope ``12_Widefield``) and of
``image_similarty_snippets.m`` (calliope ``06_Library``).
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
from matplotlib.figure import Figure

from .._matlab import normxcorr2
from ..io import MatStruct, load_wid_data
from ._activity import _roi_means, _struct_list
from ._transform import warp_image

__all__ = [
    "ImageSimilaritySnippets",
    "activity_similarity",
    "image_similarity",
    "image_similarity_snippets",
    "plot_image_similarity_snippets",
    "quality_check",
    "wid_adata_check",
    "wid_mean_roi_trace",
]

SNIPPET_SIZE = 100
NUM_SNIPPETS = 100
SEARCH_RADIUS = 100
MAX_OFFSET = 200
ACT_THRESHOLD = 0.9999
IMG_THRESHOLD = 0.85


class ImageSimilaritySnippets(NamedTuple):
    """Result of :func:`image_similarity_snippets`."""

    average_shift: float
    """Median distance (pixels) between snippet centres and their best matches."""
    snippet_centers: np.ndarray
    """``[num_snippets x 2]`` (row, column), 1-based, in ``img1``."""
    matched_locations: np.ndarray
    """``[num_snippets x 2]`` (row, column), 1-based, centres of the matches in ``img2``."""


def image_similarity_snippets(img1: Any, img2: Any, *, rng: np.random.Generator | None = None,
                              offsets: Any = None) -> ImageSimilaritySnippets:
    """Similarity of two images from the displacement of matched snippets.

    Port of ``image_similarty_snippets.m`` (calliope ``06_Library``). 100 snippets of
    100 x 100 pixels are cut from ``img1`` around its centre (random offsets in
    ``[-200, 200]`` rows and columns, clamped to the image), each is located in ``img2``
    within +-100 pixels of its position with normalised cross-correlation
    (:func:`calliopy._matlab.normxcorr2`), and the median displacement is returned.
    ``wid_adata_check`` uses ``1 - average_shift / 100``.

    Parameters
    ----------
    img1, img2
        2-D images, each at least 100 x 100 pixels.
    rng
        Generator for the random snippet offsets (MATLAB ``randi([-200, 200])``, row offset
        then column offset per snippet); default ``numpy.random.default_rng()``.
    offsets
        Instead of random offsets, a ``[num_snippets x 2]`` integer array of (row, column)
        offsets; its length sets the number of snippets.

    Returns
    -------
    ImageSimilaritySnippets
        ``average_shift`` (MATLAB's only output) and, for plotting, ``snippet_centers`` and
        ``matched_locations`` as 1-based (row, column) float64 arrays.

    Raises
    ------
    ValueError
        Images not 2-D or smaller than a snippet; a constant snippet (MATLAB
        ``normxcorr2`` errors too).

    Notes
    -----
    The matched location is the centre of the matched snippet, defined like the snippet
    centre (top-left + ``snippet_size/2``). A normxcorr2 peak at 1-based row ``ypeak`` puts
    the template's last row on region row ``ypeak``, so its top-left row in ``img2`` is
    ``row_range(1) - 1 + ypeak - snippet_size + 1`` and the matched centre row is
    ``row_range(1) + ypeak - snippet_size/2`` (columns alike). Identical images give
    ``average_shift = 0``.

    Differences from MATLAB:

    - MATLAB uses ``row_range(1) + ypeak - snippet_size/2 - 1`` (and the same for columns),
      one pixel above/left of the matched centre: identical images give
      ``average_shift = sqrt(2)`` in MATLAB, 0 here. Changed at the user's request; every
      ``average_shift`` differs from MATLAB's accordingly.
    - When the correlation maximum is attained more than once, MATLAB's assignment
      ``matched_locations(i, :) = [...]`` fails; here the first maximum in column-major
      order (MATLAB ``find`` order) is used.
    - Random numbers come from ``rng``; MATLAB's sequence cannot be reproduced.
    - Plotting is :func:`plot_image_similarity_snippets`.
    """
    a = np.asarray(img1, dtype=np.float64)
    b = np.asarray(img2, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("img1 and img2 must be 2-D")
    half = SNIPPET_SIZE // 2
    rows1, cols1 = a.shape
    rows2, cols2 = b.shape
    if min(rows1, cols1, rows2, cols2) < SNIPPET_SIZE:
        raise ValueError(f"images must be at least {SNIPPET_SIZE} x {SNIPPET_SIZE} pixels")
    if offsets is None:
        if rng is None:
            rng = np.random.default_rng()
        offsets = rng.integers(-MAX_OFFSET, MAX_OFFSET + 1, size=(NUM_SNIPPETS, 2))
    offsets = np.asarray(offsets, dtype=np.int64).reshape(-1, 2)
    n = offsets.shape[0]
    # MATLAB round(rows/2) of a positive integer: half away from zero
    center1 = ((rows1 + 1) // 2, (cols1 + 1) // 2)
    centers = np.zeros((n, 2))
    matched = np.zeros((n, 2))
    for i in range(n):
        r0 = center1[0] - half + offsets[i, 0]
        c0 = center1[1] - half + offsets[i, 1]
        r0 = max(1, min(r0, rows1 - SNIPPET_SIZE + 1))
        c0 = max(1, min(c0, cols1 - SNIPPET_SIZE + 1))
        snippet = a[r0 - 1:r0 - 1 + SNIPPET_SIZE, c0 - 1:c0 - 1 + SNIPPET_SIZE]
        cr, cc = r0 + half, c0 + half
        centers[i] = (cr, cc)
        rr0, rr1 = max(1, cr - SEARCH_RADIUS), min(rows2, cr + SEARCH_RADIUS)
        cc0, cc1 = max(1, cc - SEARCH_RADIUS), min(cols2, cc + SEARCH_RADIUS)
        region = b[rr0 - 1:rr1, cc0 - 1:cc1]
        if region.shape[0] < SNIPPET_SIZE or region.shape[1] < SNIPPET_SIZE:
            raise ValueError("search region in img2 is smaller than a snippet")
        c = normxcorr2(snippet, region)
        flat = np.flatnonzero((c == c.max()).ravel(order="F"))[0]
        ypeak, xpeak = np.unravel_index(flat, c.shape, order="F")
        # 0-based peak -> 1-based ypeak = peak + 1; top-left in img2 (1-based) is
        # rr0 + ypeak - SNIPPET_SIZE; the centre adds half, as for snippet_centers
        top = rr0 + (ypeak + 1) - SNIPPET_SIZE
        left = cc0 + (xpeak + 1) - SNIPPET_SIZE
        matched[i] = (top + half, left + half)
    shift = float(np.median(np.sqrt(np.sum((centers - matched) ** 2, axis=1))))
    return ImageSimilaritySnippets(shift, centers, matched)


def plot_image_similarity_snippets(img1: Any, img2: Any,
                                   result: ImageSimilaritySnippets) -> Figure:
    """Figure of the snippet centres in ``img1`` and their matches in ``img2``.

    The plotting part of ``image_similarty_snippets.m`` (``fig_on``): both images with the
    grey range ``[0, 2*median]``, red circles at the snippet centres, blue circles at the
    matched locations. Points are drawn at 0-based pixel coordinates (matplotlib), i.e. the
    1-based positions minus 1.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = Figure(figsize=(10, 5))
    axes = fig.subplots(1, 2)
    for ax, img, pts, style, title in [
            (axes[0], img1, result.snippet_centers, "ro", "Snippet Centers in Image 1"),
            (axes[1], img2, result.matched_locations, "bo", "Matched Locations in Image 2")]:
        im = np.asarray(img, dtype=np.float64)
        ax.imshow(im, cmap="gray", vmin=0, vmax=2 * np.median(im))
        ax.plot(pts[:, 1] - 1, pts[:, 0] - 1, style, markersize=10, fillstyle="none")
        ax.set_title(title)
        ax.set_axis_off()
    return fig


def image_similarity(first_frame: Any, tform: Any, site_first_frame: Any, *,
                     rng: np.random.Generator | None = None) -> float:
    """Image similarity of a registered first frame with the site's first frame.

    Part of ``wid_adata_check.m`` (calliope ``12_Widefield``): ``first_frame`` is warped with
    ``tform`` onto its own grid (``imwarp(first_frame, tform, 'OutputView',
    imref2d(size(first_frame)))``) and compared with ``site_first_frame``:
    ``1 - image_similarity_snippets(warped, site_first_frame) / 100``.

    Parameters
    ----------
    first_frame, site_first_frame
        2-D images (the adata ``first_frame`` of the experiment and of its site).
    tform
        3x3 matrix ``A`` of the adata ``tform`` (``tform.A``, acting on 1-based
        ``[x; y; 1]``, see :func:`calliopy.widefield._transform.warp_image`).
    rng
        Passed to :func:`image_similarity_snippets`.

    Returns
    -------
    float
        Random (snippet positions): repeated calls differ slightly, as in MATLAB. Identical
        frames with an identity ``tform`` give exactly 1.

    Notes
    -----
    Differences from MATLAB: the matched snippet centre has no one-pixel offset (see
    :func:`image_similarity_snippets`); MATLAB gives ``1 - sqrt(2)/100 = 0.9859`` for
    identical frames, this function 1.
    """
    ff = np.asarray(first_frame)
    warped = warp_image(ff, np.asarray(tform, dtype=np.float64), ff.shape[:2])
    return 1.0 - image_similarity_snippets(warped, site_first_frame, rng=rng).average_shift / 100


def wid_mean_roi_trace(data: Any, indices: Sequence[Any], *,
                       chunk_frames: int | None = None) -> np.ndarray:
    """Mean over ROIs of the ROI traces of one ``.wid`` stack.

    Part of ``wid_adata_check.m``: per ROI the mean over its pixels (single), then
    ``nanmean`` over ROIs.

    Parameters
    ----------
    data
        Stack ``[height x width x nFrames]`` (memory map allowed; read in chunks).
    indices
        Per ROI, 1-based column-major linear indices.

    Returns
    -------
    numpy.ndarray
        float32 ``(nFrames,)``.

    Notes
    -----
    Unlike :func:`calliopy.widefield._activity.roi_activity`, an ROI with indices all equal
    to 1 is not NaN here: it contributes pixel 1, as in MATLAB. Empty ROIs are NaN and
    ignored. Differences from MATLAB: sums are exact (float64) before rounding to float32.
    """
    means = _roi_means(data, indices, chunk_frames=chunk_frames).astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames give NaN
        return np.nanmean(means, axis=0).astype(np.float32)


def activity_similarity(activities: Sequence[Any], wid_mean_trace: Any) -> float:
    """Correlation of the mean stored ROI activity with the mean recomputed from ``.wid``.

    Part of ``wid_adata_check.m``: ``corrcoef(nanmean([ROIs{1}(:).activity], 2),
    all_mean_wid_act)``.

    Parameters
    ----------
    activities
        The stored ``activity`` of every ROI (``[nFrames x 1]`` each); empty ones are
        skipped, as MATLAB's horizontal concatenation does.
    wid_mean_trace
        The recomputed mean trace (:func:`wid_mean_roi_trace`, concatenated over stacks).

    Returns
    -------
    float
        Pearson correlation; NaN when a trace contains NaN or is constant.

    Raises
    ------
    ValueError
        The stored activities differ in length from each other or from the trace.

    Notes
    -----
    Differences from MATLAB: computed in float64 (MATLAB in single when the activities are
    single).
    """
    cols = [np.asarray(a, dtype=np.float64).reshape(-1) for a in activities
            if np.size(a) > 0]
    if not cols:
        raise ValueError("no ROI activity")
    lengths = {c.size for c in cols}
    trace = np.asarray(wid_mean_trace, dtype=np.float64).reshape(-1)
    if len(lengths) != 1 or trace.size not in lengths:
        raise ValueError(f"activity lengths {sorted(lengths)} vs trace length {trace.size}")
    stacked = np.stack(cols, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames give NaN
        mean_act = np.nanmean(stacked, axis=1)
        if np.isnan(mean_act).any() or np.isnan(trace).any():
            return float("nan")
        return float(np.corrcoef(mean_act, trace)[0, 1])


def quality_check(act_similarity: float, img_similarity: float) -> MatStruct:
    """The pass rule of ``wid_adata_check.m``: ``act > 0.9999 and img > 0.85``.

    Returns
    -------
    MatStruct
        ``check_raw`` with ``passed`` (1.0 or 0.0, MATLAB double), ``img_similarity`` and
        ``act_similarity``. NaN similarities fail.
    """
    passed = bool(act_similarity > ACT_THRESHOLD and img_similarity > IMG_THRESHOLD)
    return MatStruct(passed=np.float64(passed), img_similarity=np.float64(img_similarity),
                     act_similarity=np.float64(act_similarity))


def _paths(p: Any) -> list:
    if isinstance(p, (str, os.PathLike)):
        return [p]
    return list(p)


def _channel_act_similarity(items: list, wid_fnames: Any, index_field: str, act_field: str,
                            chunk_frames: int | None) -> float:
    if not all(act_field in s for s in items):
        raise ValueError(f"Activity not calculated yet (no ROIs field {act_field!r})")
    indices = [s.get(index_field, np.empty((0, 0))) for s in items]
    traces = []
    for path in _paths(wid_fnames):
        data = load_wid_data(path).wid_data
        traces.append(wid_mean_roi_trace(data, indices, chunk_frames=chunk_frames))
        del data
    trace = np.concatenate(traces) if traces else np.zeros(0, dtype=np.float32)
    return activity_similarity([s[act_field] for s in items], trace)


def wid_adata_check(rois: Any, wid_fnames: Any, *, first_frame: Any = None, tform: Any = None,
                    site_first_frame: Any = None, is_site: bool = False,
                    img_similarity: float | None = None,
                    rng: np.random.Generator | None = None,
                    chunk_frames: int | None = None) -> MatStruct:
    """Check stored ROI activity and registration of a widefield adata file.

    Port of the computation in ``wid_adata_check.m`` (calliope ``12_Widefield``) for one
    ExpID:

    - activity similarity: correlation of the mean stored ROI ``activity`` with the mean ROI
      trace recomputed from the ``.wid`` files (:func:`activity_similarity`);
    - image similarity: :func:`image_similarity` of the warped ``first_frame`` with the
      site's first frame; 1 when the experiment is its own site (one channel);
    - pass rule ``act_similarity > 0.9999 and img_similarity > 0.85``.

    Two-channel data: give ``wid_fnames``, ``first_frame``, ``tform`` and
    ``site_first_frame`` as dicts with keys ``"ch525"`` and ``"ch610"`` (adata
    ``first_frame_ch525``, ``tform_ch525``, ...). Both similarities are the minimum over
    the channels.

    Parameters
    ----------
    rois
        The adata ``ROIs`` (cell holding one struct array, or the struct array) with
        ``indices``/``activity`` (two channels: ``indices_ch525``, ``activity_ch525``, ...).
    wid_fnames
        ``.wid`` file(s) of the experiment, or a dict per channel.
    first_frame, tform, site_first_frame
        Needed unless ``is_site`` or ``img_similarity`` is given. ``tform`` is the 3x3 matrix
        ``tform.A``.
    is_site
        The experiment is its own site (MATLAB ``ExpID == ExpIDinfo.site_id``).
    img_similarity
        Use this value instead of computing it. For two-channel data that is its own site
        MATLAB asks the user to compare the channels and sets 1 on "y" (and aborts
        otherwise); pass that answer here.
    rng
        For the random snippets of :func:`image_similarity_snippets`.
    chunk_frames
        Frames read at a time from a ``.wid`` file.

    Returns
    -------
    MatStruct
        ``check_raw`` (``passed``, ``img_similarity``, ``act_similarity``) as MATLAB saves it.

    Raises
    ------
    ValueError
        Activity not calculated, missing inputs, a two-channel site without
        ``img_similarity``.

    Notes
    -----
    Differences from MATLAB: the ExpLog lookups and skips ("already checked", no adata, no
    ROIs, old data type), the interactive question, the database update
    (``update_raw_data_status``) and the save are left to the caller; missing activity
    raises instead of being skipped. ``nanmean`` and ``corrcoef`` are computed in float64.
    The image similarity uses :func:`image_similarity_snippets` without MATLAB's one-pixel
    offset of the matched location: identical frames give 1 here, ``0.9859`` in MATLAB.
    """
    items = _struct_list(rois)
    two_ch = isinstance(wid_fnames, dict)
    if two_ch:
        act = min(_channel_act_similarity(items, wid_fnames[ch], f"indices_{ch}",
                                          f"activity_{ch}", chunk_frames)
                  for ch in ("ch525", "ch610"))
    else:
        act = _channel_act_similarity(items, wid_fnames, "indices", "activity", chunk_frames)

    if img_similarity is None:
        if is_site:
            if two_ch:
                raise ValueError("two-channel site: compare the channels and pass "
                                 "img_similarity (MATLAB asks the user)")
            img_similarity = 1.0
        else:
            if first_frame is None or tform is None or site_first_frame is None:
                raise ValueError("first_frame, tform and site_first_frame are required")
            if two_ch:
                img_similarity = min(image_similarity(first_frame[ch], tform[ch],
                                                      site_first_frame[ch], rng=rng)
                                     for ch in ("ch525", "ch610"))
            else:
                img_similarity = image_similarity(first_frame, tform, site_first_frame,
                                                  rng=rng)
    return quality_check(act, img_similarity)
