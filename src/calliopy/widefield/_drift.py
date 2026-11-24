"""Stack drift diagnostics and frame comparison (port of calliope 12_Widefield).

Ports of ``wid_show_stack_drift.m`` (with its local ``plot_stack_drift``) and
``compare_wid_frames.m``. Nothing is registered, corrected or saved: these functions only
measure and show how the stacks of one widefield experiment move.

None of these functions has been compared with MATLAB output.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from calliopy._matlab import matlab_cast, prctile
from calliopy.io import MatStruct, load_wid_data
from calliopy.widefield._frames import register_frames

__all__ = [
    "compare_wid_frames",
    "plot_stack_drift",
    "plot_stack_drift_session",
    "stack_drift",
    "stack_drift_report",
]

logger = logging.getLogger(__name__)

# MATLAB lines(7): the default axes colour order, repeated for more stacks
_LINES = np.array(
    [
        [0.0000, 0.4470, 0.7410],
        [0.8500, 0.3250, 0.0980],
        [0.9290, 0.6940, 0.1250],
        [0.4940, 0.1840, 0.5560],
        [0.4660, 0.6740, 0.1880],
        [0.3010, 0.7450, 0.9330],
        [0.6350, 0.0780, 0.1840],
    ]
)


def _lines(n: int) -> np.ndarray:
    return _LINES[np.arange(n) % len(_LINES)]


def _empty() -> np.ndarray:
    return np.zeros((0, 0))


def _as_path_list(paths) -> list[Path]:
    if isinstance(paths, (str, os.PathLike)):
        return [Path(paths)]
    out = [Path(p) for p in paths]
    if not out:
        raise ValueError("no files given")
    return out


def stack_drift(
    wid_paths,
    *,
    drift_every: int = 1000,
    max_shift: float = 5,
    frame_rate: float = 100,
    exp_id=None,
    fname_field: str = "wid_fnames",
) -> MatStruct:
    """Shifts of the stacks of one widefield experiment, between and within stacks.

    Port of calliope ``12_Widefield/wid_show_stack_drift.m`` (computation and output
    ``info``). Diagnostic only:

    1. between stacks: frame 100 of every stack registered against frame 100 of stack 1
       (:func:`~calliopy.widefield._frames.register_frames`, template ``double(ref1)``),
       relative to stack 1;
    2. within stacks: every ``drift_every``-th frame of every stack (frames 1, 1 +
       drift_every, ...; frame 1 of each stack left out) registered against frame 100 of
       stack 1 (the reference is put first and subtracted, as in register_wid), minus the
       stack's own frame-100 offset.

    Parameters
    ----------
    wid_paths
        The experiment's ``.wid`` files of one channel, in recording order (MATLAB
        ``ExpIDinfo.wid_fnames<channel>``).
    drift_every
        Sampling step in frames (MATLAB default 1000).
    max_shift
        Threshold in pixels for flagging movement (MATLAB default 5).
    frame_rate
        Frame rate in Hz, only for the time axes (MATLAB default 100).
    exp_id
        ExpID stored in ``info.ExpID`` and used in plot titles (MATLAB: the input);
        ``None`` stores ``[]``.
    fname_field
        Name stored in ``info.fname_field`` (MATLAB: the ExpIDinfo field the files came
        from, e.g. ``'wid_fnames_ch525'``).

    Returns
    -------
    MatStruct
        MATLAB's ``info`` with the same fields:

        - ``ExpID``, ``fname_field``;
        - ``wid_fnames``: ``(1, nStacks)`` object array of the file names (str);
        - ``nbr_frames``: ``(1, nStacks)`` float64 frames per file from the file size;
        - ``stack_dx``, ``stack_dy``: ``[nStacks x 1]`` float64 shift of frame 100 of each
          stack vs stack 1, in pixels (``dx`` = rows, ``dy`` = columns, as in
          register_frames); the scalar 0.0 for a single stack, as in MATLAB;
        - ``stack_shifted``: ``(1, nStacks)`` bool, ``|dx| > max_shift | |dy| > max_shift``;
        - ``drift``: struct with ``drift_every``, ``frame_rate``, ``max_drift``
          (= ``max_shift``), ``frames``, ``dx``, ``dy`` (``(1, nStacks)`` object arrays of
          ``[nSampled x 1]`` float64 arrays; ``frames`` are 1-based frame numbers within
          the stack, ``dx``/``dy`` pixels vs the stack's own frame 100; ``[]`` for a stack
          without samples), ``max_abs``, ``max_step`` (``(1, nStacks)`` px),
          ``step_between`` (``[nStacks x 2]`` 1-based frame numbers between which the
          largest step happens, NaN if fewer than two samples), ``first_over`` (``(1,
          nStacks)``, first sampled 1-based frame beyond ``max_shift``, NaN if none) and
          ``ok`` (``(1, nStacks)`` bool, ``max_abs <= max_shift``);
        - ``t_session``: ``(1, nSamples)`` seconds, stacks concatenated without gaps;
        - ``dx_vs_stack1``, ``dy_vs_stack1``: ``[nSamples x 1]`` px vs stack 1.

    Raises
    ------
    ValueError
        A file has fewer than 100 frames, or the number of reference or sampled frames is
        not what the file sizes predict (MATLAB ``error``).

    Notes
    -----
    The printed table is :func:`stack_drift_report` (also logged at INFO level); the two
    figures are :func:`plot_stack_drift_session` and :func:`plot_stack_drift`.

    Deferred: the ExpLog lookup of the files and the channel field (``ExpLog``,
    ``channel``), and the ``view_stack`` viewers (``show_viewer``).
    """
    paths = _as_path_list(wid_paths)
    n_stacks = len(paths)
    drift_every = int(drift_every)
    if drift_every < 1:
        raise ValueError("drift_every must be a positive integer")

    # ---------------- 1. between stacks ----------------
    ref_frames = load_wid_data(paths, "100th").wid_data
    if ref_frames.shape[2] != n_stacks:
        raise ValueError(
            f"Expected {n_stacks} reference frames (one per stack), got {ref_frames.shape[2]}."
        )
    frame_h, frame_w = ref_frames.shape[:2]
    ref1 = np.array(ref_frames[:, :, 0])
    if n_stacks > 1:
        sdx, sdy = register_frames(ref_frames, ref1.astype(np.float64))
        sdx = sdx.reshape(-1, 1) - sdx[0, 0]
        sdy = sdy.reshape(-1, 1) - sdy[0, 0]
    else:
        sdx = 0.0
        sdy = 0.0
    del ref_frames
    sdx_v = np.atleast_1d(np.asarray(sdx, dtype=np.float64)).ravel()
    sdy_v = np.atleast_1d(np.asarray(sdy, dtype=np.float64)).ravel()

    # ---------------- 2. within stacks ----------------
    nbr_frames = np.array(
        [[float(os.path.getsize(p) // ((frame_h * frame_w + 2) * 2)) for p in paths]]
    )
    sampled = [np.arange(1, int(n) + 1, drift_every, dtype=np.float64) for n in nbr_frames[0]]
    n_sampled = np.array([s.size for s in sampled])
    drift_data = load_wid_data(paths, drift_every).wid_data
    if drift_data.size == 0 or drift_data.shape[2] != n_sampled.sum():
        got = 0 if drift_data.size == 0 else drift_data.shape[2]
        raise ValueError(f"Expected {n_sampled.sum()} sampled frames, load_wid_data returned {got}.")
    stack_of_sample = np.repeat(np.arange(1, n_stacks + 1), n_sampled)  # 1-based
    frame_of_sample = np.concatenate(sampled)
    keep = frame_of_sample > 1
    stack_of_sample = stack_of_sample[keep]
    frame_of_sample = frame_of_sample[keep]
    data = np.concatenate([ref1[:, :, None], np.asarray(drift_data[:, :, keep])], axis=2)
    del drift_data
    adx, ady = register_frames(data, ref1.astype(np.float64))
    del data
    adx = (adx - adx[0, 0])[1:]
    ady = (ady - ady[0, 0])[1:]
    wdx = adx - sdx_v[stack_of_sample - 1].reshape(-1, 1)
    wdy = ady - sdy_v[stack_of_sample - 1].reshape(-1, 1)
    stack_start = np.concatenate([[0.0], np.cumsum(nbr_frames[0, :-1])])
    t_session = ((stack_start[stack_of_sample - 1] + frame_of_sample) / frame_rate).reshape(1, -1)

    # ---------------- per-stack summary ----------------
    frames_c = np.empty((1, n_stacks), dtype=object)
    dx_c = np.empty((1, n_stacks), dtype=object)
    dy_c = np.empty((1, n_stacks), dtype=object)
    max_abs = np.zeros((1, n_stacks))
    max_step = np.zeros((1, n_stacks))
    step_between = np.full((n_stacks, 2), np.nan)
    first_over = np.full((1, n_stacks), np.nan)
    ok = np.ones((1, n_stacks), dtype=bool)
    for k in range(n_stacks):
        is_k = stack_of_sample == k + 1
        frames_k = frame_of_sample[is_k]
        if frames_k.size == 0:
            frames_c[0, k] = dx_c[0, k] = dy_c[0, k] = _empty()
            continue
        dx_k = wdx[is_k]
        dy_k = wdy[is_k]
        frames_c[0, k] = frames_k.reshape(-1, 1)
        dx_c[0, k] = dx_k
        dy_c[0, k] = dy_k
        max_abs[0, k] = max(np.max(np.abs(dx_k)), np.max(np.abs(dy_k)))
        ok[0, k] = max_abs[0, k] <= max_shift
        if frames_k.size > 1:
            step_k = np.sqrt(np.diff(dx_k[:, 0]) ** 2 + np.diff(dy_k[:, 0]) ** 2)
            i_step = int(np.argmax(step_k))
            max_step[0, k] = step_k[i_step]
            step_between[k] = frames_k[[i_step, i_step + 1]]
        over = np.flatnonzero((np.abs(dx_k[:, 0]) > max_shift) | (np.abs(dy_k[:, 0]) > max_shift))
        if over.size:
            first_over[0, k] = frames_k[over[0]]

    drift = MatStruct()
    drift.drift_every = float(drift_every)
    drift.frame_rate = float(frame_rate)
    drift.max_drift = float(max_shift)
    drift.frames = frames_c
    drift.dx = dx_c
    drift.dy = dy_c
    drift.max_abs = max_abs
    drift.max_step = max_step
    drift.step_between = step_between
    drift.first_over = first_over
    drift.ok = ok

    stack_shifted = ((np.abs(sdx_v) > max_shift) | (np.abs(sdy_v) > max_shift)).reshape(1, -1)
    names = np.empty((1, n_stacks), dtype=object)
    for k, p in enumerate(paths):
        names[0, k] = str(p)

    info = MatStruct()
    info.ExpID = _empty() if exp_id is None else exp_id
    info.fname_field = fname_field
    info.wid_fnames = names
    info.nbr_frames = nbr_frames
    info.stack_dx = sdx
    info.stack_dy = sdy
    info.stack_shifted = stack_shifted
    info.drift = drift
    info.t_session = t_session
    info.dx_vs_stack1 = adx
    info.dy_vs_stack1 = ady
    logger.info("%s", stack_drift_report(info))
    return info


def _num(v, spec: str) -> str:
    """``fprintf`` of one number: NaN/Inf written as MATLAB does ('NaN', 'Inf')."""
    v = float(v)
    left = spec.startswith("-")  # C/MATLAB '-' flag = left-justify (Python: '<')
    body = spec[1:] if left else spec
    if math.isfinite(v):
        return format(v, ("<" if left else "") + body)
    text = "NaN" if math.isnan(v) else ("Inf" if v > 0 else "-Inf")
    width = "".join(ch for ch in body.split(".")[0] if ch.isdigit())
    width = int(width) if width else 0
    return text.ljust(width) if left else text.rjust(width)


def _mat2str_indices(idx) -> str:
    vals = [str(int(i)) for i in idx]
    return vals[0] if len(vals) == 1 else "[" + " ".join(vals) + "]"


def stack_drift_report(info: Mapping) -> str:
    """The table and messages that ``wid_show_stack_drift.m`` prints, as text.

    Parameters
    ----------
    info
        Result of :func:`stack_drift`.

    Returns
    -------
    str
        The lines MATLAB printed after the computation (threshold, table, summary), with
        MATLAB's number formats. The header line with ExpID, SiteID and mouse (from the
        ExpLog) and the "Loading every ..." progress line are not part of it.
    """
    drift = info["drift"]
    max_shift = float(drift["max_drift"])
    frame_rate = float(drift["frame_rate"])
    names = np.asarray(info["wid_fnames"], dtype=object).ravel()
    n = names.size
    sdx = np.atleast_1d(np.asarray(info["stack_dx"], dtype=np.float64)).ravel()
    sdy = np.atleast_1d(np.asarray(info["stack_dy"], dtype=np.float64)).ravel()
    shifted = np.asarray(info["stack_shifted"], dtype=bool).ravel()
    ok = np.asarray(drift["ok"], dtype=bool).ravel()
    max_abs = np.asarray(drift["max_abs"], dtype=np.float64).ravel()
    max_step = np.asarray(drift["max_step"], dtype=np.float64).ravel()
    step_between = np.asarray(drift["step_between"], dtype=np.float64).reshape(n, 2)
    first_over = np.asarray(drift["first_over"], dtype=np.float64).ravel()
    g = format(max_shift, "g")
    lines = [
        "",
        f"Threshold {g} px. Offsets are frame 100 of each stack vs frame 100 of stack 1.",
        (
            "Within-stack = sampled frames vs the stack's own frame 100, one sample every "
            f"{float(drift['drift_every']) / frame_rate:.1f} s."
        ),
        "",
        "stack   offset dx   offset dy   within max   largest step   step between (s)   file",
    ]
    for k in range(n):
        flags = ""
        if shifted[k]:
            flags += "  <-- SHIFTED vs stack 1"
        if not ok[k]:
            flags += "  <-- DRIFTS"
        lines.append(
            f"{k + 1:5d}   {_num(sdx[k], '9.1f')}   {_num(sdy[k], '9.1f')}   "
            f"{_num(max_abs[k], '10.1f')}   {_num(max_step[k], '12.1f')}   "
            f"{_num(step_between[k, 0] / frame_rate, '7.1f')} - "
            f"{_num(step_between[k, 1] / frame_rate, '-7.1f')}   "
            f"{Path(str(names[k])).name}{flags}"
        )
    lines.append("")
    if not shifted.any() and ok.all():
        lines.append(f"No stack moves by more than {g} px, between or within stacks.")
    if shifted.any():
        lines.append(
            f"Between stacks: stack(s) {_mat2str_indices(np.flatnonzero(shifted) + 1)} are "
            f"offset by more than {g} px from stack 1."
        )
    for k in np.flatnonzero(~ok):
        lines.append(
            f"Within stack {k + 1}: first beyond {g} px at "
            f"{_num(first_over[k] / frame_rate, '.1f')} s (frame {_num(first_over[k], '.0f')}); "
            f"largest step {_num(max_step[k], '.1f')} px between "
            f"{_num(step_between[k, 0] / frame_rate, '.1f')} and "
            f"{_num(step_between[k, 1] / frame_rate, '.1f')} s "
            f"(max shift {_num(max_abs[k], '.1f')} px)."
        )
    return "\n".join(lines)


def _exp_label(info: Mapping) -> str:
    e = info.get("ExpID")
    if e is None or np.size(e) == 0:
        return ""
    return str(int(np.asarray(e).ravel()[0]))


def plot_stack_drift_session(info: Mapping):
    """Figure 1 of ``wid_show_stack_drift.m``: all sampled frames vs stack 1 over the session.

    Two panels (dx and dy vs stack 1 against session time, stacks concatenated without
    gaps), one colour per stack (MATLAB ``lines``), frame 100 of each stack as an open
    circle at its offset, dotted lines at stack boundaries, and red circles on samples that
    drifted more than ``max_shift`` from their own stack's frame 100.

    Parameters
    ----------
    info
        Result of :func:`stack_drift`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.figure import Figure

    drift = info["drift"]
    frame_rate = float(drift["frame_rate"])
    max_shift = float(drift["max_drift"])
    frames = np.asarray(drift["frames"], dtype=object).ravel()
    n = frames.size
    counts = np.array([np.size(f) for f in frames])
    stack_of_sample = np.repeat(np.arange(n), counts)
    nbr = np.asarray(info["nbr_frames"], dtype=np.float64).ravel()
    stack_start = np.concatenate([[0.0], np.cumsum(nbr[:-1])])
    sdx = np.atleast_1d(np.asarray(info["stack_dx"], dtype=np.float64)).ravel()
    sdy = np.atleast_1d(np.asarray(info["stack_dy"], dtype=np.float64)).ravel()
    t = np.asarray(info["t_session"], dtype=np.float64).ravel()
    adx = np.asarray(info["dx_vs_stack1"], dtype=np.float64).ravel()
    ady = np.asarray(info["dy_vs_stack1"], dtype=np.float64).ravel()
    wdx = adx - sdx[stack_of_sample]
    wdy = ady - sdy[stack_of_sample]
    colors = _lines(n)
    label = _exp_label(info)

    fig = Figure(figsize=(10, 6))
    fig.set_label(f"ExpID {label}: shifts vs stack 1, whole session")
    ax_dx, ax_dy = fig.subplots(2, 1)
    for k in range(n):
        is_k = stack_of_sample == k
        for ax, a, s in ((ax_dx, adx, sdx), (ax_dy, ady, sdy)):
            ax.plot(t[is_k], a[is_k], ".-", color=colors[k])
            ax.plot((stack_start[k] + 100) / frame_rate, s[k], "o", markeredgecolor="k",
                    markerfacecolor="w")
            if k < n - 1:
                ax.axvline(stack_start[k + 1] / frame_rate, color="k", linestyle=":")
    over = (np.abs(wdx) > max_shift) | (np.abs(wdy) > max_shift)
    ax_dx.plot(t[over], adx[over], "o", color="r", markerfacecolor="none", linestyle="none")
    ax_dy.plot(t[over], ady[over], "o", color="r", markerfacecolor="none", linestyle="none")
    ax_dx.set_ylabel("dx vs stack 1 (px)")
    ax_dy.set_ylabel("dy vs stack 1 (px)")
    ax_dy.set_xlabel(
        "recording time (s), stacks concatenated, gaps removed (dotted = stack boundary)"
    )
    ax_dx.set_title(
        f"ExpID {label}: open circles = frame 100 of each stack, red = drifted more than "
        f"{max_shift:g} px from own frame 100"
    )
    return fig


def plot_stack_drift(info: Mapping, label: str | None = None):
    """Figure 2 of ``wid_show_stack_drift.m`` (its local ``plot_stack_drift``): within-stack
    drift, one colour per stack, overlaid on time within the stack.

    Parameters
    ----------
    info
        Result of :func:`stack_drift`, or its ``drift`` struct (the MATLAB argument).
    label
        Text for the figure name and title (MATLAB ``'ExpID <id> <fname_field>'``, the
        default when ``info`` is the full result).

    Returns
    -------
    matplotlib.figure.Figure
        Two panels (dx, dy vs own frame 100, px) with the threshold as red dashed lines and
        a legend ``stack k (max x.x px)`` (``DRIFTS`` appended for stacks over threshold).
    """
    from matplotlib.figure import Figure

    if "drift" in info:
        drift = info["drift"]
        if label is None:
            label = f"ExpID {_exp_label(info)} {info.get('fname_field', '')}"
    else:
        drift = info
    label = label or ""
    frames = np.asarray(drift["frames"], dtype=object).ravel()
    dxs = np.asarray(drift["dx"], dtype=object).ravel()
    dys = np.asarray(drift["dy"], dtype=object).ravel()
    ok = np.asarray(drift["ok"], dtype=bool).ravel()
    max_abs = np.asarray(drift["max_abs"], dtype=np.float64).ravel()
    frame_rate = float(drift["frame_rate"])
    max_drift = float(drift["max_drift"])
    colors = _lines(frames.size)

    fig = Figure(figsize=(10, 6))
    fig.set_label(f"{label}: within-stack drift")
    ax_dx, ax_dy = fig.subplots(2, 1)
    handles, labels = [], []
    for k in range(frames.size):
        if np.size(frames[k]) == 0:
            continue
        t_k = np.asarray(frames[k], dtype=np.float64).ravel() / frame_rate
        (h,) = ax_dx.plot(t_k, np.ravel(dxs[k]), ".-", color=colors[k])
        ax_dy.plot(t_k, np.ravel(dys[k]), ".-", color=colors[k])
        handles.append(h)
        text = f"stack {k + 1} (max {max_abs[k]:.1f} px)"
        labels.append(text if ok[k] else text + " DRIFTS")
    for ax in (ax_dx, ax_dy):
        for y in (-max_drift, max_drift):
            ax.axhline(y, color="r", linestyle="--")
        if handles:
            ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax_dx.set_ylabel("dx vs own frame 100 (px)")
    ax_dy.set_ylabel("dy vs own frame 100 (px)")
    ax_dy.set_xlabel("time within stack (s)")
    ax_dx.set_title(
        f"{label}: within-stack drift, one sample every "
        f"{float(drift['drift_every']) / frame_rate:.1f} s, threshold \u00b1{max_drift:g} px"
    )
    return fig


def compare_wid_frames(
    frame_a,
    frame_b,
    title_a: str = "Frame A",
    title_b: str = "Frame B",
    title_c: str = "Diff",
    *,
    fig=None,
):
    """Two frames side by side and their difference ``B - A``.

    Port of calliope ``12_Widefield/compare_wid_frames.m``. Each frame is clipped at twice
    its 90th percentile (``frame(frame > 2*prctile(frame(:), 90)) = 2*prctile(...)``, the
    value stored in the frame's class), shown in grey (``imagesc``), and the difference
    ``double(B) - double(A)`` of the clipped frames is shown with a diverging colour map
    centred on zero. Green guide lines are drawn at the MATLAB positions.

    Parameters
    ----------
    frame_a, frame_b
        ``[height x width]`` frames (any numeric class).
    title_a, title_b, title_c
        Panel titles (MATLAB defaults ``'Frame A'``, ``'Frame B'``, ``'Diff'``).
    fig
        Figure to clear and draw into (MATLAB ``figHandle``); default a new figure.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Differences from MATLAB: the colour map of the difference is matplotlib's ``bwr``
    with zero at white (``TwoSlopeNorm`` when the difference has both signs, else a
    symmetric range) instead of calliope's ``bluewhitered``, which is not in the code
    base. Without ``fig``, a new figure is made (MATLAB reused a figure named
    ``'compare_wid_frames'``). No colour bars, as in MATLAB.
    """
    from matplotlib.colors import CenteredNorm, TwoSlopeNorm
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    def clip(frame):
        a = np.array(frame)
        lim = 2 * prctile(a, 90)
        stored = matlab_cast(lim, a.dtype) if a.dtype.kind in "iub" else a.dtype.type(lim)
        a[a > lim] = stored
        return a

    a = clip(frame_a)
    b = clip(frame_b)
    if fig is None:
        fig = Figure(figsize=(12, 4))
    else:
        fig.clear()
    fig.set_label("compare_wid_frames")
    for pos, img, title in (([0.05, 0.1, 0.25, 0.8], a, title_a),
                            ([0.35, 0.1, 0.25, 0.8], b, title_b)):
        ax = fig.add_axes(pos)
        ax.imshow(img, cmap="gray", aspect="auto", interpolation="nearest")
        ax.set_title(title)
    diff = b.astype(np.float64) - a.astype(np.float64)
    lo, hi = float(np.nanmin(diff)), float(np.nanmax(diff))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=lo, vmax=hi) if lo < 0 < hi else CenteredNorm(0.0)
    ax = fig.add_axes([0.65, 0.1, 0.25, 0.8])
    ax.imshow(diff, cmap="bwr", norm=norm, aspect="auto", interpolation="nearest")
    ax.set_title(title_c)
    green = (0.0, 1.0, 0.0)
    for x in (0.1, 0.15, 0.2, 0.25, 0.4, 0.45, 0.5, 0.55, 0.7, 0.75, 0.8, 0.85):
        fig.add_artist(Line2D([x, x], [0.1, 0.9], transform=fig.transFigure, color=green))
    for y in (0.5, 0.65, 0.35):
        fig.add_artist(Line2D([0, 1], [y, y], transform=fig.transFigure, color=green))
    return fig
