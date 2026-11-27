"""Registration of two-photon stacks: computation of calliope's ``register_*`` functions.

Ports of ``05_LoadData/register_bin.m`` and of ``21_Analysis/fix_registration.m``,
``register_multilayer.m``, ``register_exp_downsample.m``, ``register_exp_avgplanes.m``,
``register_exp_fixoutliers.m``, ``correct_dxdy.m``, ``check_registration.m`` and
``save_check_registration_figure.m``.

Every function takes the loaded stacks (one ``[height x width x nFrames]`` array per piezo
layer, as :func:`calliopy.io.load_bin` returns them) and returns what MATLAB writes into the
adata file as a :class:`~calliopy.io.MatStruct` with the MATLAB variable names. Looking up
the ExpID, reading the ``.ini`` file (except in :func:`register_bin_files`), writing the
adata file and the ``registration_log`` entry are left to the caller.

Shifts follow :func:`calliopy.widefield._frames.register_frames`: ``dx`` along rows, ``dy``
along columns, ``[nFrames x 1]`` float64 integers. Frame arguments are **0-based**.

None of these functions has been compared with MATLAB output.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
from matplotlib.figure import Figure

from .._matlab import matlab_cast, mround
from ..data import correct_line_shift
from ..io import MatStruct, load_bin, read_ini
from ..widefield._frames import register_frames, shift_data
from ._activity import calc_act_map
from ._builtins import ntzo

__all__ = [
    "MAX_SHIFT_FIX",
    "ApplyRegistrationResult",
    "FixShiftOutliersResult",
    "RegisterMultilayerResult",
    "apply_registration",
    "correct_dxdy",
    "expand_shifts",
    "fix_registration",
    "fix_shift_outliers",
    "next_check_registration_version",
    "plot_check_registration",
    "register_bin",
    "register_bin_files",
    "register_exp_avgplanes",
    "register_exp_downsample",
    "register_exp_fixoutliers",
    "register_multilayer",
    "save_check_registration_figure",
]

logger = logging.getLogger(__name__)

MAX_SHIFT_FIX = 100
"""``max_shift`` of ``fix_registration.m`` (pixels)."""


def _col(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1, 1)


def _mean3(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).mean(axis=2)


# --- fix_registration -------------------------------------------------------------------


def _band_zero(f: np.ndarray, high: int, low: int, axis: int) -> None:
    n = f.shape[axis]
    idx = [slice(None), slice(None)]
    hp = np.r_[np.arange(min(high, n)), np.arange(max(n - high + 1, 0), n)]
    idx[axis] = hp
    f[tuple(idx)] = 0
    if n - low > low + 1:
        idx[axis] = slice(low + 1, n - low)
        f[tuple(idx)] = 0


def fix_registration(data: Any, template: Any, shift_x: Any, shift_y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Re-estimate shifts that jump by more than 100 pixels between frames.

    Port of calliope ``21_Analysis/fix_registration.m``. Frames whose shift differs from
    the previous frame's by more than :data:`MAX_SHIFT_FIX` (and not from the last good
    value) are registered again by band-passed FFT cross-correlation with the template, with
    the search restricted to +/- 100 pixels around the previous frame's shift. The first
    frame and every frame with 1-based number ``1001, 2001, ...`` starting a jump are kept.

    Parameters
    ----------
    data
        ``[height x width x nFrames]`` stack.
    template
        ``[height x width]`` reference image.
    shift_x, shift_y
        Initial shifts (rows, columns), one per frame.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Corrected ``shift_x``, ``shift_y`` (float64, shape of the input).
    """
    arr = np.asarray(data)
    sx = np.array(shift_x, dtype=np.float64)
    sy = np.array(shift_y, dtype=np.float64)
    shape_x = sx.shape
    sx, sy = sx.ravel(), sy.ravel()
    h, w, n = arr.shape
    t = np.asarray(template, dtype=np.float64)
    t = t - t.mean()
    b = int(mround(0.05 * max(h, w)))
    t = t[b : h - b, b : w - b]
    th, tw = t.shape
    outsize = (th + (h - 2 * b) - 1, tw + (w - 2 * b) - 1)
    low = int(mround(min(th, tw) / 4))
    high = int(mround(min(th, tw) / 40))
    ft = np.fft.fft2(t[::-1, ::-1], s=outsize)
    for ax in (0, 1):
        _band_zero(ft, high, low, ax)
    if abs(sx[0] - np.median(sx)) > MAX_SHIFT_FIX or abs(sy[0] - np.median(sy)) > MAX_SHIFT_FIX:
        logger.info("Initial values of dx/dy already bad")
    cond1_mem = False
    mem_x = mem_y = 0.0
    for i in range(1, n):
        cond1 = abs(sx[i] - sx[i - 1]) > MAX_SHIFT_FIX or abs(sy[i] - sy[i - 1]) > MAX_SHIFT_FIX
        if (i + 1) % 1000 == 1 and not cond1_mem:
            cond1 = False
        if cond1 and not cond1_mem:
            mem_x, mem_y = sx[i - 1], sy[i - 1]
        if cond1 and abs(sx[i] - mem_x) <= MAX_SHIFT_FIX and abs(sy[i] - mem_y) <= MAX_SHIFT_FIX:
            cond1 = False
        cond1_mem = cond1
        if not cond1:
            continue
        fr = arr[:, :, i].astype(np.float64)
        fr = fr - fr.mean()
        fin = np.fft.fft2(fr[b : h - b, b : w - b], s=outsize)
        for ax in (0, 1):
            _band_zero(fin, high, low, ax)
        c = np.fft.ifft2(ft * fin).real
        c = c[b : c.shape[0] - b, b : c.shape[1] - b].copy()
        cr, cc = c.shape
        hr, hc = int(np.ceil(cr / 2)), int(np.ceil(cc / 2))
        tl, tr = int(sx[i - 1] + hr - MAX_SHIFT_FIX), int(sx[i - 1] + hr + MAX_SHIFT_FIX)
        tu, td = int(sy[i - 1] + hc - MAX_SHIFT_FIX), int(sy[i - 1] + hc + MAX_SHIFT_FIX)
        c[: max(tl, 0), :] = 0
        c[max(tr - 1, 0) :, :] = 0
        c[:, : max(tu, 0)] = 0
        c[:, max(td - 1, 0) :] = 0
        k = int(np.argmax(c.ravel(order="F")))
        sx[i] = k % cr + 1 - hr
        sy[i] = k // cr + 1 - hc
    return sx.reshape(shape_x), sy.reshape(shape_x)


# --- register_multilayer ------------------------------------------------------------------


class RegisterMultilayerResult(NamedTuple):
    """Result of :func:`register_multilayer` (MATLAB ``[mean_data, dx_corr, dy_corr]``)."""

    mean_data: np.ndarray
    dx_corr: np.ndarray
    dy_corr: np.ndarray


def _register2d(frame, template):
    dx, dy = register_frames(np.asarray(frame)[:, :, None], template)
    return float(dx[0, 0]), float(dy[0, 0])


def register_multilayer(data: Any, nbr_frames_per_layer: int) -> RegisterMultilayerResult:
    """Mean image per z step of a z-stack, aligned across steps (``register_multilayer.m``).

    For every block of ``nbr_frames_per_layer`` frames: line-shift correction
    (:func:`calliopy.data.correct_line_shift`), registration on the mean of its first 5
    frames, ``dx = round(dx - mean(dx)/2)`` (as MATLAB), and the mean of the shifted frames
    whose shift is within 25 px of the mean shift (the plain mean when 25 or fewer are).
    Consecutive mean images are then registered to each other (with
    :func:`fix_registration` for jumps above 10 px), the cumulative shifts are detrended by a
    line fitted over z steps ``round(0.1 L) .. round(0.6 L)`` (1-based) and the mean images
    shifted by the residual.

    Returns
    -------
    RegisterMultilayerResult
        ``mean_data`` ``[h x w x L]`` float64; ``dx_corr``, ``dy_corr`` ``(1, L)``.

    Raises
    ------
    ValueError
        Fewer than 5 z steps (MATLAB failed indexing ``dx_cs(0)``).
    """
    arr = np.asarray(data)
    fpl = int(nbr_frames_per_layer)
    n = arr.shape[2]
    nl = n // fpl
    if n % fpl:
        logger.warning("2p data size is not a multiple of nbr_frames_per_layer. "
                       "Recording may have been stopped early.")
    means = []
    for i in range(nl):
        cur = correct_line_shift(arr[:, :, i * fpl : (i + 1) * fpl]).data
        dx, dy = register_frames(cur, _mean3(cur[:, :, :5]))
        dx = mround(dx - dx.mean() / 2)
        dy = mround(dy - dy.mean() / 2)
        tmp = shift_data(cur, dx, dy)
        good = ((np.abs(dx - dx.mean()) < 25) & (np.abs(dy - dy.mean()) < 25)).ravel()
        means.append(_mean3(tmp[:, :, good]) if good.sum() > 25 else _mean3(cur))
    mean_data = np.stack(means, axis=2) if means else np.zeros(arr.shape[:2] + (0,))
    L = mean_data.shape[2]
    lo, hi = int(mround(0.1 * L)), int(mround(0.6 * L))
    if lo < 1 or hi <= lo:
        raise ValueError(f"need at least 5 z steps, got {L}")
    dx = np.zeros(L)
    dy = np.zeros(L)
    for i in range(1, L):
        dx[i], dy[i] = _register2d(mean_data[:, :, i], mean_data[:, :, i - 1])
        if abs(dx[i]) > 10 or abs(dy[i]) > 10:
            fx, fy = fix_registration(mean_data[:, :, i - 1 : i + 1], mean_data[:, :, i - 1],
                                      dx[i - 1 : i + 1], dy[i - 1 : i + 1])
            dx[i], dy[i] = fx[1], fy[1]
    xs = np.arange(lo, hi + 1, dtype=np.float64)
    dx_cs, dy_cs = np.cumsum(dx), np.cumsum(dy)
    allx = np.arange(1, L + 1, dtype=np.float64)
    dx_corr = dx_cs - np.polyval(np.polyfit(xs, dx_cs[lo - 1 : hi], 1), allx)
    dy_corr = dy_cs - np.polyval(np.polyfit(xs, dy_cs[lo - 1 : hi], 1), allx)
    return RegisterMultilayerResult(shift_data(mean_data, dx_corr, dy_corr),
                                    dx_corr[None, :], dy_corr[None, :])


# --- outlier correction -------------------------------------------------------------------


class FixShiftOutliersResult(NamedTuple):
    """Result of :func:`fix_shift_outliers`."""

    dx: np.ndarray
    dy: np.ndarray
    corrected: bool
    """True when outliers were found (and replaced)."""


def fix_shift_outliers(dx: Any, dy: Any, std_thresh: float = 4, *, correct: bool = True) -> FixShiftOutliersResult:
    """Replace runs of outlying shifts by the mean of their neighbours.

    The block shared by ``register_exp_downsample.m`` and ``register_exp_avgplanes.m``
    (threshold 4 SD) and ``register_exp_fixoutliers.m`` (5 SD). Frames with
    ``|dx - mean(dx)| > std_thresh * std(dx)`` (or the same for ``dy``; ``std`` with N-1)
    are grouped into runs of consecutive frames; each run takes
    ``round(mean([d(first-1), d(last+1)]))``. A run that starts at the first frame takes the
    value after the run, one that ends at the last frame the value before it.

    Parameters
    ----------
    dx, dy
        Shifts ``[nFrames x 1]``.
    std_thresh
        Threshold in standard deviations.
    correct
        False only detects (MATLAB's "Do you want to correct?" answered no).

    Returns
    -------
    FixShiftOutliersResult
        Shifts as float64 ``[nFrames x 1]`` copies.
    """
    x, y = _col(dx).copy(), _col(dy).copy()
    n = x.shape[0]
    sx = x.std(ddof=1) if n > 1 else 0.0
    sy = y.std(ddof=1) if n > 1 else 0.0
    cx = np.flatnonzero(np.abs(x[:, 0] - x.mean()) > std_thresh * sx) + 1
    cy = np.flatnonzero(np.abs(y[:, 0] - y.mean()) > std_thresh * sy) + 1
    if cx.size == 0 and cy.size == 0:
        return FixShiftOutliersResult(x, y, False)
    if not correct:
        return FixShiftOutliersResult(x, y, True)
    ftc = np.unique(np.r_[cx, cy])  # 1-based, sorted
    brk = np.flatnonzero(np.diff(ftc) > 1)
    on = np.c_[np.r_[ftc[0], ftc[brk + 1]], np.r_[ftc[brk], ftc[-1]]].astype(np.int64)
    first = last = None
    if on[0, 0] == 1:
        on[0, 0] = 2
        first = (x[on[0, 1], 0], y[on[0, 1], 0])  # 1-based on+1 -> 0-based on
    if on[-1, 1] == n:
        on[-1, 1] = n - 1
        last = (x[on[-1, 0] - 2, 0], y[on[-1, 0] - 2, 0])
    cor_x = mround((x[on[:, 0] - 2, 0] + x[on[:, 1], 0]) / 2)
    cor_y = mround((y[on[:, 0] - 2, 0] + y[on[:, 1], 0]) / 2)
    if first is not None:
        cor_x[0], cor_y[0] = first
        on[0, 0] = 1
    if last is not None:
        cor_x[-1], cor_y[-1] = last
        on[-1, 1] = n
    for k in range(on.shape[0]):
        x[on[k, 0] - 1 : on[k, 1], 0] = cor_x[k]
        y[on[k, 0] - 1 : on[k, 1], 0] = cor_y[k]
    return FixShiftOutliersResult(x, y, True)


def expand_shifts(d_ds: Any, fta: int, n_frames: int) -> np.ndarray:
    """Repeat each shift ``fta`` times and keep the first ``n_frames`` (``reshape(repmat(...))``)."""
    return np.repeat(_col(d_ds)[:, 0], int(fta))[: int(n_frames)].reshape(-1, 1)


# --- applying shifts ------------------------------------------------------------------------


class ApplyRegistrationResult(NamedTuple):
    """Result of :func:`apply_registration` (one entry per layer)."""

    act_map: list
    template: list
    act_map_sec: list
    template_sec: list
    data: list
    """The registered stacks (MATLAB keeps them in ``data``)."""


def _template_sec(sec: np.ndarray, nbr_frames, n_layers: int) -> np.ndarray:
    nf = np.asarray(nbr_frames, dtype=np.float64).ravel()
    out = []
    sta = 1.0
    sto = 0.0
    for p in range(nf.size):
        sto += nf[p] / n_layers
        out.append(_mean3(sec[:, :, int(sta) - 1 : int(sto)]))
        sta += nf[p] / n_layers
    return np.stack(out, axis=2)


def apply_registration(data: Sequence[Any], dx: Sequence[Any], dy: Sequence[Any], *,
                       sec_data: Sequence[Any] | None = None, nbr_frames: Any = None) -> ApplyRegistrationResult:
    """Shift, correct the line shift, and compute ``act_map`` and ``template`` per layer.

    The final block of every MATLAB registration function: for each layer
    ``data = shift_data(data, dx, dy)``, ``data = correct_line_shift(data, mean(data, 3))``,
    ``act_map = calc_act_map(data)``, ``template = mean(data, 3)``; the same for the second
    channel, whose template is computed per recorded file (``template_sec(:, :, file)``, with
    ``nbr_frames(file) / nLayers`` frames each).
    """
    nl = len(dx)
    if sec_data is not None and nbr_frames is None:
        raise ValueError("sec_data needs nbr_frames")
    am, tp, ams, tps, regd = [], [], [], [], []
    for r in range(nl):
        d = shift_data(np.asarray(data[r]), dx[r], dy[r])
        d = correct_line_shift(d, _mean3(d)).data
        am.append(calc_act_map(d))
        tp.append(_mean3(d))
        regd.append(d)
        if sec_data is not None:
            s = shift_data(np.asarray(sec_data[r]), dx[r], dy[r])
            s = correct_line_shift(s, _mean3(s)).data
            ams.append(calc_act_map(s))
            tps.append(_template_sec(s, nbr_frames, nl))
    return ApplyRegistrationResult(am, tp, ams, tps, regd)


def _empty_layer_structs(n: int) -> list:
    return [MatStruct() for _ in range(n)]


# --- register_bin ---------------------------------------------------------------------------


def register_bin(data: Sequence[Any], nbr_frames: Any, *, sec_data: Sequence[Any] | None = None,
                 reg_on_ch: int = 1, template_frames: Sequence[int] = range(10, 30),
                 fpga_frames_zstep: float = 0, fpga_frames_layer: int = 1) -> MatStruct:
    """Register a two-photon experiment (computation of ``05_LoadData/register_bin.m``).

    Each piezo layer is registered with ``register_frames`` on the mean of
    ``template_frames`` (**0-based**; MATLAB ``11:30`` = 10..29) of the channel
    ``reg_on_ch``; then :func:`apply_registration`. A z-stack (``fpga_frames_zstep > 0``)
    is instead reduced with :func:`register_multilayer`.

    Parameters
    ----------
    data
        One stack per layer (:func:`calliopy.io.load_bin` ``.data``).
    nbr_frames
        Frames per file (``load_bin(...).nbr_frames``).
    sec_data
        Second-channel stacks; when None, ``reg_on_ch`` is forced to 1 as in MATLAB.
    reg_on_ch
        Channel to register on (1 or 2).
    template_frames
        0-based frames of the registration template.
    fpga_frames_zstep, fpga_frames_layer
        ``FPGA.framesperzstep`` and ``FPGA.framesperlayer`` of the ``.ini`` file.

    Returns
    -------
    MatStruct
        Imaging data: ``ROIs``, ``bv``, ``np`` (one empty struct per layer), ``template``,
        ``dx``, ``dy``, ``nbr_frames``, ``act_map``, ``template_sec``, ``act_map_sec`` (lists
        per layer). Z-stack: ``mean_data`` (and ``mean_data_sec``).

    Notes
    -----
    Differences from MATLAB: ``aux_files``, ``fnames``, ``mouse_id`` and ``userID`` (ExpLog
    metadata) are added by the caller; the "Adata file already exists" check, the adata file
    creation, :func:`plot_check_registration` and :func:`save_check_registration_figure` are
    the caller's.
    """
    if sec_data is None:
        reg_on_ch = 1
    if fpga_frames_zstep > 0:
        out = MatStruct(mean_data=register_multilayer(data[0], fpga_frames_layer).mean_data)
        if sec_data is not None:
            out["mean_data_sec"] = register_multilayer(sec_data[0], fpga_frames_layer).mean_data
        return out
    tf = np.asarray(list(template_frames), dtype=np.int64)
    dx, dy = [], []
    for y in range(len(data)):
        src = np.asarray(data[y] if reg_on_ch == 1 else sec_data[y])
        a, b = register_frames(src, _mean3(src[:, :, tf]))
        dx.append(a)
        dy.append(b)
    reg = apply_registration(data, dx, dy, sec_data=sec_data, nbr_frames=nbr_frames)
    nl = len(data)
    return MatStruct(ROIs=_empty_layer_structs(nl), bv=_empty_layer_structs(nl),
                     np=_empty_layer_structs(nl), template=reg.template, dx=dx, dy=dy,
                     nbr_frames=np.asarray(nbr_frames), act_map=reg.act_map,
                     template_sec=reg.template_sec, act_map_sec=reg.act_map_sec)


def _ini_get(ini: MatStruct, section: str, key: str):
    sec = ini[section] if section in ini else ini[section.lower()]
    return sec[key] if key in sec else sec[key.lower()]


def register_bin_files(fnames: Any, *, ini_file: Any, sec_fnames: Any = None,
                       reg_on_ch: int = 1, template_frames: Sequence[int] = range(10, 30)) -> MatStruct:
    """:func:`register_bin` on ``.bin`` files, reading the layer settings from the ``.ini``.

    ``piezo.nbrlayers`` (1 when it cannot be read), ``FPGA.framesperzstep`` (when > 0 the
    layer count is set to 1) and ``FPGA.framesperlayer`` are read with
    :func:`calliopy.io.read_ini`, as MATLAB does.
    """
    try:
        ini = read_ini(ini_file)
    except (OSError, ValueError):
        ini = MatStruct()
    try:
        n_layers = int(_ini_get(ini, "piezo", "nbrlayers"))
    except (KeyError, TypeError, ValueError):
        logger.warning("could not read nbr of piezo layers from ini file")
        n_layers = 1
    try:
        zstep = float(_ini_get(ini, "FPGA", "framesperzstep"))
        if zstep > 0:
            n_layers = 1
        flayer = int(_ini_get(ini, "FPGA", "framesperlayer"))
    except (KeyError, TypeError, ValueError):
        logger.warning("could not read z step details from ini file")
        zstep, flayer = 0.0, 1
    b = load_bin(fnames, n_layers)
    sec = load_bin(sec_fnames, n_layers).data if sec_fnames else None
    return register_bin(b.data, b.nbr_frames, sec_data=sec, reg_on_ch=reg_on_ch,
                        template_frames=template_frames, fpga_frames_zstep=zstep,
                        fpga_frames_layer=flayer)


# --- register_exp_downsample / avgplanes / fixoutliers ---------------------------------------


def _bins(d: np.ndarray, fta: int, *, zero_gap: bool) -> np.ndarray:
    n = d.shape[2]
    nfull = n // fta
    out = [_mean3(d[:, :, fta * k : fta * (k + 1)]) for k in range(nfull)]
    if n % fta:
        if zero_gap:
            out.append(np.zeros(d.shape[:2]))
        out.append(_mean3(d[:, :, n - fta :]))
    return np.stack(out, axis=2)


def register_exp_downsample(data: Sequence[Any], *, fta: int = 3,
                            template_frames: Sequence[int] = range(10, 30),
                            nbr_frames: Any = None, sec_data: Sequence[Any] | None = None) -> MatStruct:
    """Register on averages of ``fta`` frames (``21_Analysis/register_exp_downsample.m``).

    Per layer the stack is averaged in blocks of ``fta`` frames, the block means are
    registered on the mean of the raw frames ``template_frames`` (**0-based**; MATLAB
    ``11:30``), each block shift is repeated for its ``fta`` frames, outliers are replaced
    (:func:`fix_shift_outliers`, 4 SD), and :func:`apply_registration` follows.

    Returns
    -------
    MatStruct
        ``dx``, ``dy``, ``act_map``, ``template``, ``template_sec`` (lists per layer, as
        MATLAB saves them; ``act_map_sec`` is computed but not saved by MATLAB and not
        returned), plus ``nbr_frames`` when given.

    Notes
    -----
    Kept from MATLAB (suspected bug, changes numbers): when the frame count is not a
    multiple of ``fta``, MATLAB preallocates ``floor(n/fta) + 1`` blocks and then appends
    the mean of the last ``fta`` frames *after* them, so block ``floor(n/fta)`` (0-based)
    stays all zero. Its registration gives shift (0, 0), which is what the last
    ``n mod fta`` frames receive; the appended tail block is cut off.
    """
    tf = np.asarray(list(template_frames), dtype=np.int64)
    dxs, dys = [], []
    for d in data:
        arr = np.asarray(d)
        n = arr.shape[2]
        ds = _bins(arr, fta, zero_gap=True)
        a, b = register_frames(ds, _mean3(arr[:, :, tf]))
        fx = fix_shift_outliers(expand_shifts(a, fta, n), expand_shifts(b, fta, n), 4)
        dxs.append(fx.dx)
        dys.append(fx.dy)
    reg = apply_registration(data, dxs, dys, sec_data=sec_data, nbr_frames=nbr_frames)
    out = MatStruct(dx=dxs, dy=dys, act_map=reg.act_map, template=reg.template,
                    template_sec=reg.template_sec)
    if nbr_frames is not None:
        out["nbr_frames"] = np.asarray(nbr_frames)
    return out


def _movmean3(d: np.ndarray, k: int) -> np.ndarray:
    """MATLAB ``movmean(d, k, 3)`` with shrinking windows at the ends."""
    n = d.shape[2]
    if k <= 1:
        return d.astype(np.float64)
    before = k // 2 if k % 2 == 0 else (k - 1) // 2
    after = k // 2 - 1 if k % 2 == 0 else (k - 1) // 2
    cs = np.concatenate([np.zeros(d.shape[:2] + (1,)), np.cumsum(d.astype(np.float64), axis=2)], axis=2)
    lo = np.clip(np.arange(n) - before, 0, n)
    hi = np.clip(np.arange(n) + after + 1, 0, n)
    return (cs[:, :, hi] - cs[:, :, lo]) / (hi - lo)


def register_exp_avgplanes(data: Sequence[Any], *, template_frames: Sequence[int] = range(10, 30),
                           fta: int = 1, moving_average: bool = True, nbr_frames: Any = None,
                           sec_data: Sequence[Any] | None = None) -> MatStruct:
    """Register all layers on the average over layers (``register_exp_avgplanes.m``).

    Steps as MATLAB: optional moving mean over ``fta`` frames per layer, stored as int16;
    the layer average; with ``fta > 1`` and no moving mean, block averages of ``fta``
    frames; registration on the mean of ``template_frames`` (**0-based**; MATLAB ``11:30``);
    the shifts are expanded, outliers replaced (4 SD) and applied to every layer
    (:func:`apply_registration`).

    Returns
    -------
    MatStruct
        ``dx``, ``dy`` (same for every layer), ``act_map``, ``template``, ``template_sec``
        and ``nbr_frames`` (when given).

    Notes
    -----
    Kept from MATLAB (suspected bugs, change numbers):

    - with 4 layers the average is ``data{1} + data{2} + data{3} + data{4}/4`` (operator
      precedence) in int16 arithmetic (saturating at each step), not the mean;
    - with ``fta > 1`` and ``moving_average`` (the default), the frame-wise shifts are
      repeated ``fta`` times and truncated, so frame ``t`` receives the shift of frame
      ``t // fta``;
    - the moving mean is stored as int16 (rounded, saturated) whatever the input class.

    Differences from MATLAB: ``manual_check`` (loading only the first frames) is the
    caller's choice of ``data``; nothing is saved.
    """
    nl = len(data)
    tf = np.asarray(list(template_frames), dtype=np.int64)
    layers = [np.asarray(d) for d in data]
    if moving_average:
        layers = [matlab_cast(_movmean3(d, fta), np.int16) for d in layers]
    n = layers[0].shape[2]
    if nl == 4:
        acc = matlab_cast(layers[0].astype(np.float64) + layers[1], layers[0].dtype)
        acc = matlab_cast(acc.astype(np.float64) + layers[2], layers[0].dtype)
        q = matlab_cast(layers[3].astype(np.float64) / 4, layers[3].dtype)
        data_ds = matlab_cast(acc.astype(np.float64) + q, layers[0].dtype)
    else:
        data_ds = sum(d.astype(np.float64) for d in layers) / nl
    if fta > 1:
        ds = _bins(data_ds, fta, zero_gap=False) if not moving_average else data_ds
        a, b = register_frames(ds, _mean3(ds[:, :, tf]))
        fx = fix_shift_outliers(expand_shifts(a, fta, n), expand_shifts(b, fta, n), 4)
        dx0, dy0 = fx.dx, fx.dy
    else:
        dx0, dy0 = register_frames(data_ds, _mean3(np.asarray(data_ds)[:, :, tf]))
    dxs = [dx0.copy() for _ in range(nl)]
    dys = [dy0.copy() for _ in range(nl)]
    reg = apply_registration(layers, dxs, dys, sec_data=sec_data, nbr_frames=nbr_frames)
    out = MatStruct(dx=dxs, dy=dys, act_map=reg.act_map, template=reg.template,
                    template_sec=reg.template_sec)
    if nbr_frames is not None:
        out["nbr_frames"] = np.asarray(nbr_frames)
    return out


def _as_layers(d) -> tuple[list, bool]:
    if isinstance(d, (list, tuple)):
        return [_col(x) for x in d], True
    if isinstance(d, np.ndarray) and d.dtype == object:
        return [_col(x) for x in d.ravel(order="F")], True
    return [_col(d)], False


def register_exp_fixoutliers(dx: Any, dy: Any, *, correct: Any = True, data: Sequence[Any] | None = None,
                             sec_data: Sequence[Any] | None = None, nbr_frames: Any = None,
                             std_thresh: float = 5) -> MatStruct:
    """Replace outlying registration shifts and re-apply them (``register_exp_fixoutliers.m``).

    Parameters
    ----------
    dx, dy
        Stored shifts (``adata.dx``, ``adata.dy``): a list per layer, or one array.
    correct
        Whether to correct a layer where outliers are found: one bool, or one per layer
        (MATLAB asks "Do you want to correct? (1/0)").
    data, sec_data
        Raw (unregistered) stacks per layer; needed to recompute ``act_map`` and
        ``template`` when something was corrected.
    nbr_frames
        Frames per file, needed with ``sec_data``.
    std_thresh
        Outlier threshold in SD (MATLAB 5).

    Returns
    -------
    MatStruct
        Empty when nothing was corrected (MATLAB saves nothing). Otherwise the fields MATLAB
        stores: one layer: ``dx``, ``dy``, ``act_map``, ``template`` as arrays, plus
        ``act_map_sec``, ``template_sec`` with a second channel; several layers: ``dx``,
        ``dy``, ``act_map``, and only with a second channel ``act_map_sec``,
        ``template_sec`` and ``template``.

    Notes
    -----
    Kept from MATLAB (suspected bug): with several layers and no second channel, MATLAB's
    save block fails at ``act_map_sec`` (caught) after storing ``dx``, ``dy`` and
    ``act_map``, so the recomputed ``template`` is not stored. This is reproduced.

    Differences from MATLAB: one-layer data whose ``dx`` is stored as an array (not a cell)
    is accepted (MATLAB failed on ``dx{1}``).
    """
    xs, _ = _as_layers(dx)
    ys, _ = _as_layers(dy)
    nl = len(xs)
    flags = [bool(correct)] * nl if np.ndim(correct) == 0 else [bool(c) for c in correct]
    go_on = []
    for j in range(nl):
        r = fix_shift_outliers(xs[j], ys[j], std_thresh, correct=flags[j])
        xs[j], ys[j] = r.dx, r.dy
        go_on.append(r.corrected and flags[j])
    if not any(go_on):
        return MatStruct()
    if data is None:
        raise ValueError("outliers were corrected: pass data to recompute act_map and template")
    reg = apply_registration(data, xs, ys, sec_data=sec_data, nbr_frames=nbr_frames)
    if nl == 1:
        out = MatStruct(dx=xs[0], dy=ys[0], act_map=reg.act_map[0], template=reg.template[0])
        if sec_data is not None:
            out["act_map_sec"] = reg.act_map_sec[0]
            out["template_sec"] = reg.template_sec[0]
        return out
    out = MatStruct(dx=xs, dy=ys, act_map=reg.act_map)
    if sec_data is not None:
        out["act_map_sec"] = reg.act_map_sec
        out["template_sec"] = reg.template_sec
        out["template"] = reg.template
    return out


def correct_dxdy(dx: Any, dy: Any, *, man_corr: tuple | None = None, z_plane: int | None = None,
                 z_to_correct: Sequence[int] = (), data: Sequence[Any] | None = None,
                 sec_data: Sequence[Any] | None = None, nbr_frames: Any = None) -> MatStruct:
    """Replace registration shifts manually or by another plane's (``21_Analysis/correct_dxdy.m``).

    Parameters
    ----------
    dx, dy
        Stored shifts (``adata.dx``, ``adata.dy``), list per layer.
    man_corr
        ``(dx_new, dy_new)``: the new shifts (lists per layer, or arrays for one layer).
    z_plane
        **0-based** layer whose shifts are copied (MATLAB ``'z_plane'`` first argument, 1-based).
    z_to_correct
        **0-based** layers that receive them.
    data, sec_data
        Raw stacks per layer; when given, ``act_map`` and ``template`` (and the second
        channel's) are recomputed (MATLAB ``recalc = 1``).
    nbr_frames
        Frames per file, needed with ``sec_data``.

    Returns
    -------
    MatStruct
        The adata fields to update (``dx``, ``dy`` and, with ``data``, ``act_map``,
        ``template``, ``act_map_sec``, ``template_sec``).

    Notes
    -----
    Differences from MATLAB (crash fixes): with ``man_corr`` and recalculation MATLAB always
    failed (``template_sec`` undefined without a second channel; ``nbr_frames`` and ``dx``
    undefined with one); with ``z_plane``, recalculation with a second channel failed
    (``nbr_frames`` undefined). Here ``nbr_frames`` is an argument and every layer of the
    second channel is processed (MATLAB's multi-layer ``man_corr`` code processed only the
    last). The ``registration_log`` entry and the save are left to the caller.
    """
    if (man_corr is None) == (z_plane is None):
        raise ValueError("give exactly one of man_corr or z_plane")
    xs, is_list = _as_layers(dx)
    ys, _ = _as_layers(dy)
    if man_corr is not None:
        nx, single = _as_layers(man_corr[0])
        ny, _ = _as_layers(man_corr[1])
        targets = list(range(len(nx)))
        xs, ys = nx, ny
        is_list = single
    else:
        targets = [int(z) for z in z_to_correct]
        for r in targets:
            xs[r] = xs[int(z_plane)].copy()
            ys[r] = ys[int(z_plane)].copy()
    out = MatStruct(dx=xs if is_list else xs[0], dy=ys if is_list else ys[0])
    if data is None:
        return out
    sub = [data[r] for r in targets]
    ssub = [sec_data[r] for r in targets] if sec_data is not None else None
    reg = apply_registration(sub, [xs[r] for r in targets], [ys[r] for r in targets],
                             sec_data=ssub, nbr_frames=nbr_frames)
    nl = len(xs)
    am, tp = [None] * nl, [None] * nl
    ams, tps = [None] * nl, [None] * nl
    for i, r in enumerate(targets):
        am[r], tp[r] = reg.act_map[i], reg.template[i]
        if ssub is not None:
            ams[r], tps[r] = reg.act_map_sec[i], reg.template_sec[i]
    out["act_map"] = am if is_list else am[0]
    out["template"] = tp if is_list else tp[0]
    if ssub is not None:
        out["act_map_sec"] = ams if is_list else ams[0]
        out["template_sec"] = tps if is_list else tps[0]
    return out


# --- check_registration ---------------------------------------------------------------------


def _clip_template(t) -> np.ndarray:
    im = ntzo(np.asarray(t))
    fac = np.nanmean(im) + 4 * np.nanstd(im, ddof=1)
    return np.where(im > fac, fac, im)


def plot_check_registration(dx: Any, dy: Any, template: Any, *, title: str = "") -> Figure:
    """Template and shift traces per layer (``21_Analysis/check_registration.m``).

    One row per layer: the template normalised to 0..1 (:func:`~._builtins.ntzo`) and
    clipped at ``mean + 4 std``, and the ``dx`` / ``dy`` traces. Pass ``template_sec``
    instead of ``template`` for MATLAB's second-channel figure.

    Differences from MATLAB: no ExpID lookup (``check_registration(ExpID)``), no workspace
    access, no fixed figure number; the registration log shown on shift-click is not
    reproduced.
    """
    xs, multi = _as_layers(dx)
    ys, _ = _as_layers(dy)
    temps = list(template) if multi else [template]
    nl = len(xs)
    fig = Figure(figsize=(6.8, 1.6 * nl))
    for i in range(nl):
        ax = fig.add_subplot(nl, 2, 2 * i + 1)
        tt = np.asarray(temps[i])
        if tt.ndim == 3:
            tt = tt[:, :, 0]
        ax.imshow(_clip_template(tt), cmap="gray", aspect="auto")
        ax.axis("off")
        ax2 = fig.add_subplot(nl, 2, 2 * i + 2)
        x = np.arange(1, xs[i].shape[0] + 1)
        ax2.plot(x, xs[i][:, 0], label="dx")
        ax2.plot(x, ys[i][:, 0], **({"color": "g"} if multi else {}), label="dy")
        if i == 0:
            ax2.legend(ncol=2, loc="upper left", frameon=False)
    if title:
        fig.suptitle(title)
    return fig


def next_check_registration_version(existing_names: Sequence[str], site_id: int) -> int:
    """Version number of the next ``CheckReg-<site_id>_v<k>.png`` file.

    Part of ``21_Analysis/save_check_registration_figure.m``: one more than the largest
    existing version, 0 when there is none.

    Differences from MATLAB: the whole version number is read (MATLAB read one character, so
    from ``_v10`` on it overwrote ``_v10``), and only files of exactly this site are counted
    (MATLAB's ``CheckReg-<site>*`` also matched sites whose number starts with it).
    """
    pat = re.compile(rf"^CheckReg-{int(site_id)}_v(\d+)")
    vers = [int(m.group(1)) for s in existing_names if (m := pat.match(os.path.basename(str(s))))]
    return max(vers) + 1 if vers else 0


def save_check_registration_figure(fig: Figure, folder: Any, site_id: int) -> str:
    """Save ``fig`` as ``<folder>/CheckReg-<site_id>_v<k>.png`` (``save_check_registration_figure.m``).

    ``folder`` is the animal's registration-figure folder (MATLAB builds it from the adata
    directory, user and mouse); it is created when missing. Returns the path written.
    """
    os.makedirs(folder, exist_ok=True)
    k = next_check_registration_version(os.listdir(folder), site_id)
    path = os.path.join(folder, f"CheckReg-{int(site_id)}_v{k}.png")
    fig.savefig(path)
    return path
