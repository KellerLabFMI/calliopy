"""Two-photon ROI activity, neuropil, projections and quick-look response maps.

Ports of calliope ``21_Analysis/ROI_activity.m``, ``ROIs2mask.m``, ``calc_act_map.m``,
``11_TwoPhoton/np_activity.m``, ``calc_act.m`` (computation), ``online_response.m``
(computation and plot), ``compareStacks.m`` (computation and plot) and the layer tiling of
``show_last_stack.m``.

ROI ``indices`` are MATLAB linear pixel indices, **1-based, column-major**, exactly as stored
in adata files. Frame numbers passed as arguments are **0-based**.

None of these functions has been compared with MATLAB output.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
from matplotlib.figure import Figure

from .._matlab import matlab_cast, prctile
from ..io import MatStruct
from ..widefield._frames import register_frames, shift_data

__all__ = [
    "CalcActResult",
    "OnlineResponse",
    "calc_act",
    "calc_act_map",
    "compare_stacks_images",
    "neuropil_indices",
    "np_activity",
    "online_response",
    "plot_compare_stacks",
    "plot_online_response",
    "roi_activity",
    "roi_traces",
    "rois_to_mask",
    "stack_projections",
    "stack_template",
    "tile_piezo_layers",
]

logger = logging.getLogger(__name__)

ONLINE_RESPONSE_LAYERS = 4
"""``online_response.m`` tiles and subsamples by 4 piezo layers, hard-coded."""


def _as_items(rois: Any) -> list:
    if isinstance(rois, MatStruct):
        return [rois]
    if isinstance(rois, np.ndarray) and rois.dtype == object:
        return list(rois.ravel(order="F"))
    return list(rois)


def _indices(item: Any, field: str = "indices") -> np.ndarray:
    ix = item[field] if isinstance(item, dict) else getattr(item, field)
    a = np.asarray(ix, dtype=np.float64).ravel(order="F")
    if a.size and (np.any(a != np.round(a)) or np.any(a < 1)):
        raise ValueError(f"{field} must be positive integers (1-based linear indices)")
    return a.astype(np.int64)


def roi_traces(data: Any, indices: Sequence[Any]) -> np.ndarray:
    """Mean of every ROI's pixels in every frame.

    Parameters
    ----------
    data
        ``[height x width x nFrames]`` stack (any numeric dtype, memory maps allowed); a 2-D
        image is one frame.
    indices
        Per ROI the pixel indices, MATLAB linear, **1-based, column-major**.

    Returns
    -------
    numpy.ndarray
        ``[nFrames x nROIs]``; float32 for float32 data, else float64 (MATLAB ``mean`` of an
        integer array is double). An ROI without indices gives NaN.

    Raises
    ------
    IndexError
        An index outside ``1 .. height*width``.
    """
    arr = np.asarray(data) if not isinstance(data, np.memmap) else data
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    if arr.ndim != 3:
        raise ValueError(f"data must be [height x width x nFrames], got {arr.shape}")
    h, w, n = arr.shape
    out = np.full((n, len(indices)), np.nan)
    for i, ix in enumerate(indices):
        a = np.asarray(ix, dtype=np.float64).ravel(order="F").astype(np.int64)
        if a.size == 0:
            continue
        if a.min() < 1 or a.max() > h * w:
            raise IndexError(f"ROI {i}: index outside 1..{h * w}")
        r, c = np.unravel_index(a - 1, (h, w), order="F")
        out[:, i] = np.asarray(arr[r, c, :], dtype=np.float64).mean(axis=0)
    if arr.dtype == np.float32:
        return out.astype(np.float32)
    return out


def roi_activity(data: Any, rois: Any) -> list:
    """Mean activity of every ROI (calliope ``21_Analysis/ROI_activity.m``).

    Parameters
    ----------
    data
        ``[height x width x nFrames]`` registered stack of one piezo layer.
    rois
        The ROI struct array of that layer (list of :class:`~calliopy.io.MatStruct` or an
        object array) with field ``indices`` (MATLAB linear, 1-based, column-major).

    Returns
    -------
    list of MatStruct
        Copies of the ROIs with ``activity`` set to the ``[nFrames x 1]`` mean trace
        (float64; float32 for float32 data).

    Notes
    -----
    Differences from MATLAB: an ROI with no indices gets an all-NaN activity (MATLAB's
    ``splitapply`` failed or misassigned traces). float32 data is averaged in float64 and
    rounded once to float32 (MATLAB averages in single; last-bit differences possible).
    """
    items = _as_items(rois)
    traces = roi_traces(data, [_indices(s) for s in items])
    out = []
    for i, s in enumerate(items):
        new = MatStruct(s)
        new["activity"] = traces[:, i : i + 1].copy()
        out.append(new)
    return out


def np_activity(data: Any, np_struct: Any) -> MatStruct:
    """Neuropil activity (calliope ``11_TwoPhoton/np_activity.m``).

    Parameters
    ----------
    data
        ``[height x width x nFrames]`` stack.
    np_struct
        Struct with ``indices`` and ``indices_middle`` (MATLAB linear, 1-based).

    Returns
    -------
    MatStruct
        Copy of ``np_struct`` with ``activity`` and ``activity_middle``, ``[nFrames x 1]``.
    """
    tr = roi_traces(data, [_indices(np_struct), _indices(np_struct, "indices_middle")])
    out = MatStruct(np_struct)
    out["activity"] = tr[:, 0:1].astype(np.float64)
    out["activity_middle"] = tr[:, 1:2].astype(np.float64)
    return out


def rois_to_mask(indices: Any, im_size: Sequence[int]) -> np.ndarray:
    """Boolean mask of an ROI (calliope ``21_Analysis/ROIs2mask.m``).

    Parameters
    ----------
    indices
        MATLAB linear pixel indices, **1-based, column-major**.
    im_size
        ``(height, width)``.

    Returns
    -------
    numpy.ndarray
        bool ``(height, width)``.
    """
    h, w = int(im_size[0]), int(im_size[1])
    mask = np.zeros(h * w, dtype=bool)
    a = np.asarray(indices, dtype=np.int64).ravel(order="F")
    if a.size:
        if a.min() < 1 or a.max() > h * w:
            raise IndexError(f"index outside 1..{h * w}")
        mask[a - 1] = True
    return mask.reshape((h, w), order="F")


def neuropil_indices(n_pixels: int, rois: Any) -> np.ndarray:
    """Neuropil pixels: every pixel not in any ROI (the ``np`` step of ``calc_act.m``).

    Returns
    -------
    numpy.ndarray
        float64 ``[nNP x 1]`` sorted MATLAB linear indices (1-based), as
        ``setdiff(1:n_pixels, vertcat(ROIs.indices))'``.
    """
    items = _as_items(rois)
    used = np.concatenate([_indices(s) for s in items]) if items else np.zeros(0, np.int64)
    return np.setdiff1d(np.arange(1, int(n_pixels) + 1), used).astype(np.float64)[:, None]


def stack_projections(data: Any) -> MatStruct:
    """Mean, median, max and std projections over frames (``calc_act.m``).

    Returns
    -------
    MatStruct
        ``mean`` (float64), ``median`` and ``max`` (class of ``data``; the median of an
        integer stack is rounded half away from zero, unverified for MATLAB's integer median
        of an even number of frames), ``std`` (float64, N-1).
    """
    arr = np.asarray(data)
    mean = arr.astype(np.float64).mean(axis=2)
    med = np.median(arr.astype(np.float64), axis=2)
    if arr.dtype.kind in "iu":
        med = matlab_cast(med, arr.dtype)
    elif arr.dtype == np.float32:
        med = med.astype(np.float32)
        mean = mean.astype(np.float32)
    mx = arr.max(axis=2)
    std = arr.astype(np.float64).std(axis=2, ddof=1) if arr.shape[2] > 1 else np.zeros(arr.shape[:2])
    return MatStruct(mean=mean, median=med, max=mx, std=std)


class CalcActResult(NamedTuple):
    """Result of :func:`calc_act` (what MATLAB writes into the adata file)."""

    ROIs: list
    """Per layer, the ROI list with ``activity`` (``adata.ROIs``)."""
    np: list
    """Per layer, a one-element list with the neuropil struct (``adata.np``)."""
    projections: MatStruct
    """``mean``, ``median``, ``max``, ``std``: lists with one image per layer."""


def calc_act(data: Sequence[Any], rois: Sequence[Any], *, sec_data: Sequence[Any] | None = None,
             calc_on_ch: int = 1, n_pixels: int | None = None,
             recalc_rois: bool = True) -> CalcActResult:
    """ROI and neuropil activity and projections of a registered 2P experiment.

    Port of the computation of calliope ``11_TwoPhoton/calc_act.m`` for two-photon data
    (``.bin``): per piezo layer ``ROI_activity`` of the ROIs, the neuropil indices (all
    pixels outside the layer's ROIs) and their activity, and the projections.

    Parameters
    ----------
    data
        One registered ``[height x width x nFrames]`` stack per layer (MATLAB ``data``, as
        :func:`calliopy.data.load_exp` returns it).
    rois
        One ROI list per layer (``adata.ROIs``) with ``indices`` (1-based linear). A layer
        whose ROIs have no ``indices`` field is skipped, as in MATLAB.
    sec_data
        Second-channel stacks, used for the ROI activity when ``calc_on_ch == 2``.
    calc_on_ch
        1 (main channel) or 2.
    n_pixels
        Number of pixels of a frame (MATLAB ``numel(adata.template{1})``); default
        ``height * width`` of the first layer.
    recalc_rois
        False reproduces MATLAB's ``skip_act_calc`` (activity present, only neuropil and
        projections missing): the ROIs are returned unchanged.

    Returns
    -------
    CalcActResult

    Notes
    -----
    As in MATLAB, the neuropil activity is always computed on ``data`` (main channel), also
    when ``calc_on_ch == 2``.

    Differences from MATLAB: the ExpLog/adata lookup, the "already calculated" checks and the
    save are left to the caller; ``ROI_fissa`` (neuropil correction with the external FISSA
    Python package) is not run, so ``activity_fissa_corrected`` is not added. The widefield
    branch of ``calc_act.m`` is not ported: it fails in MATLAB (``np_act`` undefined) and is
    covered by :func:`calliopy.widefield.calc_widefield_act`.
    """
    if calc_on_ch not in (1, 2):
        raise ValueError("calc_on_ch must be 1 or 2")
    if calc_on_ch == 2 and sec_data is None:
        raise ValueError("calc_on_ch=2 needs sec_data")
    nl = len(rois)
    if n_pixels is None:
        d0 = np.asarray(data[0]) if not isinstance(data[0], np.memmap) else data[0]
        n_pixels = int(d0.shape[0] * d0.shape[1])
    out_rois, out_np = [], []
    for q in range(nl):
        items = _as_items(rois[q])
        if not items or not all("indices" in s for s in items):
            out_rois.append(items)
            out_np.append([])
            continue
        if recalc_rois:
            src = data[q] if calc_on_ch == 1 else sec_data[q]
            items = roi_activity(src, items)
        npst = MatStruct(indices=neuropil_indices(n_pixels, rois[q]))
        npst = roi_activity(data[q], [npst])[0]
        out_rois.append(items)
        out_np.append([npst])
    proj = {k: [] for k in ("mean", "median", "max", "std")}
    for q in range(nl):
        p = stack_projections(data[q])
        for k, lst in proj.items():
            lst.append(p[k])
    return CalcActResult(out_rois, out_np, MatStruct(proj))


def calc_act_map(data: Any, step_size: int = 10) -> np.ndarray:
    """Activity map of a stack (calliope ``21_Analysis/calc_act_map.m``, BW 2014).

    Frames are averaged in consecutive blocks of ``step_size`` (block starts at 0-based
    frames ``0, step, ...`` while ``start < nFrames - step``); the overall minimum is
    subtracted and the map is ``max(blocks) / (mean(blocks) + 200)``.

    Parameters
    ----------
    data
        ``[height x width x nFrames]`` stack.
    step_size
        Frames per block (MATLAB default 10).

    Returns
    -------
    numpy.ndarray
        float64 ``(height, width)``.

    Notes
    -----
    MATLAB preallocates ``round(nFrames/step_size) - 1`` blocks; when the loop fills fewer,
    the remaining all-zero blocks stay and enter the minimum and the mean. This is
    reproduced.

    Raises
    ------
    ValueError
        ``nFrames <= step_size`` (MATLAB returned an empty or undefined map).
    """
    arr = np.asarray(data)
    if arr.ndim != 3:
        raise ValueError("data must be [height x width x nFrames]")
    h, w, n = arr.shape
    step = int(step_size)
    starts = list(range(0, n - step, step))
    if not starts:
        raise ValueError(f"need more than step_size={step} frames, got {n}")
    prealloc = int(np.floor(n / step + 0.5)) - 1
    nb = max(prealloc, len(starts))
    blocks = np.zeros((h, w, nb))
    for i, s in enumerate(starts):
        blocks[:, :, i] = arr[:, :, s : s + step].astype(np.float64).mean(axis=2)
    blocks -= blocks.min()
    return blocks.max(axis=2) / (blocks.mean(axis=2) + 200)


def tile_piezo_layers(data: Any, nbr_layers: int) -> np.ndarray:
    """Put the piezo layers of an interleaved stack side by side (``show_last_stack.m``).

    ``reshape(data, h, w * nbr_layers, nFrames / nbr_layers)`` in column-major order: frame
    ``t`` of the result holds the ``nbr_layers`` consecutive frames ``t*nbr_layers ...``
    next to each other. Trailing frames that do not fill a cycle are dropped with a
    warning, as MATLAB.
    """
    arr = np.asarray(data)
    h, w, n = arr.shape
    k = int(nbr_layers)
    if n % k:
        warnings.warn("frame numbers are not an even multiple of piezo layers", stacklevel=2)
        arr = arr[:, :, : n - n % k]
        n = arr.shape[2]
    return arr.reshape((h, w * k, n // k), order="F")


class OnlineResponse(NamedTuple):
    """Result of :func:`online_response`."""

    trigs: np.ndarray
    """0-based frame indices (in the tiled stack) of the triggers used."""
    downsamp_aux: np.ndarray
    """Aux channel sampled at the frame times (``downsampAux``)."""
    img_snp: np.ndarray
    """``(post - pre) / pre`` per trigger, ``[h x w x nTrigs]``."""
    response_map: np.ndarray
    """``mean(img_snp, 3)``."""
    resp_data: np.ndarray | None
    """Mean ``[pre; post] ./ baseline`` movie (``view_resp_stack``), else None."""
    data: np.ndarray
    """The (registered, tiled) stack the map was computed on."""


def online_response(tdata: Any, aux_data: Any, frame_times: Any, *, nbr_piezo_layers: int = 1,
                    aux_channel: int = 8, n_frames_avg: int = 20, thr: float = 1,
                    min_intertrig_samples: int = 10, register: bool = False,
                    template_frames: Sequence[int] = range(49, 80),
                    resp_stack: bool = False) -> OnlineResponse:
    """Response map to external triggers, computation of ``11_TwoPhoton/online_response.m``.

    Parameters
    ----------
    tdata
        Raw ``[h x w x nFrames]`` stack of the ``.bin`` file, all layers interleaved
        (``load_bin(...)`` with one layer).
    aux_data
        ``[nChannels x nSamples]`` aux data (:func:`calliopy.io.load_lvd`).
    frame_times
        Frame times from the frame channel (:func:`calliopy.preprocessing.get_frame_times`
        of ``aux_data[1]``), MATLAB 1-based sample numbers.
    nbr_piezo_layers
        ``piezo.nbrlayers`` of the ``.ini`` file (only used when ``register``).
    aux_channel
        **0-based** aux channel with the trigger (MATLAB default 9 = 0-based 8).
    n_frames_avg, thr, min_intertrig_samples
        As MATLAB's options.
    register
        Register before computing (MATLAB ``register_frames`` option).
    template_frames
        **0-based** frames averaged for the registration template (MATLAB 50:80 = 49..79).
    resp_stack
        Also compute the mean response movie (MATLAB ``view_resp_stack``).

    Returns
    -------
    OnlineResponse

    Raises
    ------
    ValueError
        No trigger, or no trigger far enough from the start and end (MATLAB plotted the aux
        trace and returned; see :func:`plot_online_response` for the plot).

    Notes
    -----
    As in MATLAB, the layer tiling and the subsampling of the frame times use 4 layers
    whatever ``nbr_piezo_layers`` is (:data:`ONLINE_RESPONSE_LAYERS`), except when
    registering a single-layer stack, where the stack is not tiled. This looks unintended
    for stacks without 4 layers but is kept.

    Differences from MATLAB: no file lookup in the temp folder, no viewer
    (``view_stack``), no printing; the plot is :func:`plot_online_response`.
    """
    td = np.asarray(tdata)
    h, w, n = td.shape
    k = ONLINE_RESPONSE_LAYERS
    tf = np.asarray(list(template_frames), dtype=np.int64)
    if register:
        if nbr_piezo_layers == 1:
            dx, dy = register_frames(td, td[:, :, tf].astype(np.float64).mean(axis=2))
            data = shift_data(td, dx, dy)
        else:
            sub = td[:, :, ::k]
            tfr = np.arange(0, n, k)[tf]
            dx, dy = register_frames(sub, td[:, :, tfr].astype(np.float64).mean(axis=2))
            data = shift_data(td.reshape((h, w * k, n // k), order="F"), dx, dy)
    else:
        data = td.reshape((h, w * k, n // k), order="F")
    ft = np.asarray(frame_times, dtype=np.float64).ravel()
    if ft.size > n:
        ft = ft[:n]
    ft = ft[::k].astype(np.int64)
    aux = np.asarray(aux_data, dtype=np.float64)
    down = aux[aux_channel, ft - 1]
    on0 = np.flatnonzero(np.diff((down > thr).astype(np.int8)) == 1)  # MATLAB value = on0 + 1
    if on0.size == 0:
        raise ValueError(f"No valid trigger detected in aux channel {aux_channel} (0-based)")
    valid = np.r_[True, np.diff(on0) > min_intertrig_samples]
    trig = on0[valid]  # 0-based frame index of MATLAB's trigs
    nf = data.shape[2]
    trig = trig[~((trig + 1 <= n_frames_avg) | (trig + 1 >= nf - n_frames_avg))]
    if trig.size == 0:
        raise ValueError(f"No valid trigger detected in aux channel {aux_channel} (0-based)")
    img = np.zeros((data.shape[0], data.shape[1], trig.size))
    movs = np.zeros((data.shape[0], data.shape[1], 2 * n_frames_avg, trig.size)) if resp_stack else None
    for i, t in enumerate(trig):
        pre_s = data[:, :, t - n_frames_avg : t].astype(np.float64)
        post_s = data[:, :, t + 1 : t + 1 + n_frames_avg].astype(np.float64)
        pre, post = pre_s.mean(axis=2), post_s.mean(axis=2)
        img[:, :, i] = (post - pre) / pre
        if resp_stack:
            movs[:, :, :, i] = np.concatenate([pre_s, post_s], axis=2) / pre[:, :, None]
    resp = movs.mean(axis=3) if resp_stack else None
    return OnlineResponse(trig, down, img, img.mean(axis=2), resp, data)


def plot_online_response(result: OnlineResponse, *, clims=(-0.3, 0.3), title: str = "",
                         aux_channel: int | None = None) -> Figure:
    """Response map above the downsampled aux trace with the triggers (``online_response.m``).

    MATLAB uses ``flipud(lbmap(255, 'BrownBlue'))``; the matplotlib ``BrBG`` colormap is used
    instead (not identical).
    """
    fig = Figure(figsize=(8, 6), facecolor=(0.15, 0.15, 0.15))
    gs = fig.add_gridspec(6, 1)
    ax = fig.add_subplot(gs[:5, 0])
    ax.imshow(result.response_map, cmap="BrBG", vmin=clims[0], vmax=clims[1], aspect="auto")
    ax.set_title(title, color=(0.85, 0.85, 0.85))
    ax.axis("off")
    ax2 = fig.add_subplot(gs[5, 0])
    lab = f"Aux Channel {aux_channel}" if aux_channel is not None else "Aux"
    ax2.plot(np.arange(1, result.downsamp_aux.size + 1), result.downsamp_aux,
             color=(0.85, 0.85, 0.85), lw=2, label=lab)
    for t in result.trigs:
        ax2.axvline(t + 1, color=(0.15, 0.95, 0.95), lw=1.8)
    ax2.axis("off")
    ax2.legend(frameon=False, labelcolor=(0.85, 0.85, 0.85))
    return fig


def stack_template(data: Sequence[Any], template_frames: Sequence[int] = range(49, 80)) -> np.ndarray:
    """Registered mean image of every layer, side by side (``compareStacks.m``).

    Every layer is registered (:func:`calliopy.widefield._frames.register_frames`) on the
    mean of ``template_frames`` (**0-based**; MATLAB's default ``50:80`` is 49..79), shifted
    and averaged; the layer images are concatenated horizontally (MATLAB ``cell2mat``).
    """
    tf = np.asarray(list(template_frames), dtype=np.int64)
    out = []
    for d in data:
        arr = np.asarray(d)
        dx, dy = register_frames(arr, arr[:, :, tf].astype(np.float64).mean(axis=2))
        out.append(shift_data(arr, dx, dy).astype(np.float64).mean(axis=2))
    return np.hstack(out)


def compare_stacks_images(img1: Any, img2: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rescaled images and the red/cyan merge shown by ``compareStacks.m``.

    Each image is mapped linearly so that its 0.1 and 99.9 percentiles become 0 and 1 (not
    clipped; the display clips). The merge is ``cat(3, img1, img2, img2)``.
    """
    res = []
    for img in (img1, img2):
        a = np.asarray(img, dtype=np.float64)
        lo, hi = prctile(a.ravel(order="F"), 0.1), prctile(a.ravel(order="F"), 99.9)
        res.append((a - lo) / (hi - lo))
    return res[0], res[1], np.stack([res[0], res[1], res[1]], axis=2)


def plot_compare_stacks(img1: Any, img2: Any, names: Sequence[str] = ("", "")) -> Figure:
    """Three linked panels (stack 1, stack 2, merge) as ``compareStacks.m``."""
    a, b, rgb = compare_stacks_images(img1, img2)
    fig = Figure(figsize=(10, 10), facecolor=(0.1, 0.1, 0.1))
    axes = fig.subplots(3, 1, sharex=True, sharey=True)
    labels = [names[0], names[1], "Merge"]
    for ax, im, lab in zip(axes, [a, b, rgb], labels, strict=True):
        if im.ndim == 2:
            ax.imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1, aspect="auto")
        else:
            ax.imshow(np.clip(im, 0, 1), aspect="auto")
        ax.text(10, 20, lab, color="white", fontsize=15)
        ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1, 0, 0)
    return fig
