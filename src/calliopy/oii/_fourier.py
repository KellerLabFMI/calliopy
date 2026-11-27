"""Fourier (periodic-stimulus) analysis of intrinsic optical imaging (OII).

Ports of calliope ``13_OII/analyzeOII_Fourier.m``, ``showFourierMaps.m`` (computation and
plots) and the computation of ``AnalyzeWaveStimulusIOSseparate.m``. Prompts become
arguments; nothing is saved.

Image-processing built-ins (``impyramid``, ``imfilter``, ``fspecial``, ``mat2gray``,
``smooth``) are reproduced in :mod:`calliopy.twophoton._builtins` from their documented
algorithms (MathWorks code is not translated; edge behaviour unverified).

Channel arguments are **0-based**; stimulus boundaries and frame times stay MATLAB values
(1-based aux samples / ms).

None of these functions has been compared with MATLAB output.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure
from scipy import ndimage

from .._matlab import linspace, mround, prctile
from ..io import MatStruct, load_lvd, load_vid_data
from ..twophoton._builtins import (
    fspecial_average,
    fspecial_gaussian,
    imfilter,
    impyramid_reduce,
    mat2gray,
    ntzo,
)

__all__ = [
    "FourierMaps",
    "OiiFourierResult",
    "OiiStimEpochs",
    "analyze_oii_fourier",
    "analyze_oii_fourier_files",
    "fourier_maps",
    "oii_normalize",
    "oii_stim_bounds",
    "oii_stim_epochs",
    "plot_fourier_maps",
    "power_overlay",
    "smooth2",
    "wave_reference_image",
    "wave_stimulus_maps",
]

SATURATION = 4094
"""``meandat < 4094`` marks unsaturated pixels (12-bit data assumed in MATLAB)."""


def smooth2(data: Any, window_size: int, dim: int | None = None) -> np.ndarray:
    """Running average padded with the end values (calliope ``21_Analysis/smooth2.m``).

    Private copy for :func:`oii_normalize` (``smooth2`` belongs to the analysis-core track).
    The data is padded with ``ceil(w/2)`` copies of the first and ``floor(w/2)`` of the last
    value, filtered causally with ``ones(w)/w`` and the first ``w`` outputs dropped.

    Parameters
    ----------
    dim
        **0-based** axis; default (MATLAB rule): axis 0 when the array has fewer columns than
        rows (a column vector), else axis 1. A 1-D array is a row.
    """
    a = np.asarray(data, dtype=np.float64)
    one_d = a.ndim == 1
    if one_d:
        a = a[None, :]
    if dim is None:  # MATLAB: [~, I] = min(size(data)); dim = 1 if I == 2 else 2
        dim = 0 if a.shape[1] < a.shape[0] else 1
    x = a.T if dim == 0 else a
    w = int(window_size)
    pad = np.hstack([np.repeat(x[:, :1], int(np.ceil(w / 2)), axis=1), x,
                     np.repeat(x[:, -1:], w // 2, axis=1)])
    cs = np.cumsum(pad, axis=1)
    filt = np.empty_like(pad)
    filt[:, :w] = cs[:, :w] / w
    filt[:, w:] = (cs[:, w:] - cs[:, :-w]) / w
    out = filt[:, w:]
    out = out.T if dim == 0 else out
    return out[0] if one_d else out


def oii_stim_bounds(aux_data: Any, *, stim_channels: Sequence[int] = (2, 1), shutter_channel: int = 0,
                    rescue_stim_bounds: bool = False, swap_stim_channels: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Crop the aux data to the shutter-open period and find the stimulus boundaries.

    Part of ``analyzeOII_Fourier.m``. The aux data is cut from the first rising to the first
    falling edge of ``shutter > 0.5`` (to the end when it never falls). Boundaries are the
    samples where the stimulus-ID channel steps by more than 0.5 V plus the last sample,
    with the second of two adjacent boundaries removed; with ``rescue_stim_bounds`` they
    come from the low periods (``< 1.2 * min``) longer than 200 samples of aux row 2
    (0-based; MATLAB hard-codes row 3).

    Parameters
    ----------
    stim_channels
        **0-based** ``(stimID, stimPOS)`` rows (MATLAB ``StimChannels`` ``[3 2]``).
    swap_stim_channels
        Swap the two rows first (MATLAB does this for ExpIDs 37357-37397, listed in the code).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The cropped aux data and the boundaries as **1-based** sample numbers in it.
    """
    aux = np.array(aux_data, dtype=np.float64)
    sid, spos = int(stim_channels[0]), int(stim_channels[1])
    if swap_stim_channels:
        aux[[sid, spos]] = aux[[spos, sid]]
    sh = (aux[shutter_channel] > 0.5).astype(np.int8)
    up = np.flatnonzero(np.diff(sh) == 1) + 1
    down = np.flatnonzero(np.diff(sh) == -1) + 1
    if up.size == 0:
        raise ValueError("no shutter opening found")
    stop = int(down[0]) if down.size else aux.shape[1]
    aux = aux[:, int(up[0]) - 1 : stop]
    if not rescue_stim_bounds:
        sb = np.r_[np.flatnonzero(np.abs(np.diff(aux[sid])) > 0.5) + 1, aux.shape[1]].astype(np.float64)
        sb = np.delete(sb, np.flatnonzero(np.diff(sb) == 1) + 1)
        return aux, sb
    a = (aux[2] < aux[2].min() * 1.2).astype(np.int8)
    pa = np.r_[0, a, 0]
    tpos = np.flatnonzero(np.diff(pa) == 1) + 1
    lones = (np.flatnonzero(np.diff(pa) == -1) + 1) - tpos
    tidx = np.flatnonzero(lones > 200)
    sb: list[float] = []
    for j in range(1, tidx.size - 1):
        sb += [tpos[tidx[j]], tpos[tidx[j]] + lones[tidx[j]]]
    return aux, np.asarray([lones[0], *sb, tpos[-1]], dtype=np.float64)


class OiiStimEpochs(NamedTuple):
    """Frame ranges and measured frequencies of the four stimulus directions."""

    start: np.ndarray
    """0-based first frame of each direction."""
    stop: np.ndarray
    """0-based last frame (inclusive) of each direction."""
    stimfreqs: np.ndarray
    """Bar frequency measured on the stimulus-position channel (Hz, 1 kHz aux assumed)."""


def oii_stim_epochs(frame_times: Any, aux: Any, stim_bound: Any, stim_pos_channel: int = 1) -> OiiStimEpochs:
    """Frames of the four stimulus epochs and the measured bar frequencies (``analyzeOII_Fourier.m``).

    Epoch ``k`` runs from the first frame with ``frame_time >= stim_bound[2k]`` to the last
    with ``frame_time <= stim_bound[2k+1]``; its frequency is ``1 / (0.001 * mean(diff(peaks)))``
    with ``peaks`` the samples where the stimulus-position channel jumps by more than 2 V.
    ``frame_times`` is the first row of ``vmeta_info`` with its first value set to 1, as MATLAB.
    """
    ft = np.asarray(frame_times, dtype=np.float64).ravel()
    sb = np.asarray(stim_bound, dtype=np.float64).ravel()
    pos = np.asarray(aux, dtype=np.float64)[stim_pos_channel]
    st, sp, fq = [], [], []
    for k in range(4):
        lo, hi = sb[2 * k], sb[2 * k + 1]
        st.append(int(np.flatnonzero(ft >= lo)[0]))
        sp.append(int(np.flatnonzero(ft <= hi)[-1]))
        da = np.diff(pos[int(lo) - 1 : int(hi)])
        peaks = np.flatnonzero(np.abs(da) > 2)
        with np.errstate(all="ignore"):
            fq.append(1 / (0.001 * np.mean(np.diff(peaks))) if peaks.size > 1 else np.nan)
    return OiiStimEpochs(np.asarray(st), np.asarray(sp), np.asarray(fq))


def oii_normalize(dat: Any, method: str = "Mean", int_profile: Any = None) -> np.ndarray:
    """Per-frame normalisation of ``analyzeOII_Fourier.m`` (float32 in, float32 out).

    ``'Mean'``: divide by the frame mean; ``'Median'``: by the median of the column medians;
    ``'GlobalSmooth'``: by :func:`smooth2` (window 300) of the mean brightness profile;
    ``'minmax'``: see Notes; ``'None'``.

    Notes
    -----
    Kept from MATLAB (suspected bug): ``'minmax'`` computes ``x - prctile(x, 2) ./
    (prctile(x, 98) - prctile(x, 2))`` (division before subtraction) with column-wise
    percentiles, then clips the whole stack to 0..255.
    """
    d = np.asarray(dat, dtype=np.float32)
    if method == "Mean":
        return (d / d.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)).astype(np.float32)
    if method == "Median":
        med = np.median(np.median(d, axis=0), axis=0).astype(np.float32)
        return (d / med).astype(np.float32)
    if method == "GlobalSmooth":
        prof = d.mean(axis=(0, 1), dtype=np.float64) if int_profile is None else np.asarray(int_profile)
        s = smooth2(np.asarray(prof, dtype=np.float64).reshape(-1, 1), 300).ravel()
        return (d / s.astype(np.float32)[None, None, :]).astype(np.float32)
    if method == "minmax":
        out = np.empty_like(d)
        for i in range(d.shape[2]):
            x = d[:, :, i].astype(np.float64)
            p2 = prctile(x, 2, axis=0)
            p98 = prctile(x, 98, axis=0)
            out[:, :, i] = (x - p2 / (p98 - p2)).astype(np.float32)
        return np.clip(out, 0, 255).astype(np.float32)
    if method == "None":
        return d
    raise ValueError(f"unknown normalisation {method!r}")


class FourierMaps(NamedTuple):
    """Result of :func:`fourier_maps` (MATLAB ``showFourierMaps`` outputs, plus extras)."""

    Power: np.ndarray
    Phase: list
    PhaseMap: list
    PhaseMapHor: np.ndarray
    PhaseMapVer: np.ndarray
    PhaseHor: np.ndarray
    PhaseVer: np.ndarray
    power_raw: np.ndarray
    """Normalised power before the 5x5 average filter (first panel in MATLAB)."""
    power_alpha: np.ndarray
    """``mat2gray(Power)`` with values below ``1.25 * median`` set to 0 (transparency)."""


def _wrap(phase):
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def fourier_maps(fourier_specs: Sequence[Any], idx: int = 0, meandat_mask: Any = None) -> FourierMaps:
    """Phase and power maps from the four Fourier spectra (``13_OII/showFourierMaps.m``).

    Parameters
    ----------
    fourier_specs
        Four complex ``[h x w]`` (or ``[h x w x nFreq]``) spectra at the stimulus frequency:
        up, down, left, right.
    idx
        **0-based** frequency plane for 3-D spectra.
    meandat_mask
        Unsaturated-pixel mask (not flipped); its pixels outside get power 0. None: all.

    Returns
    -------
    FourierMaps
        As MATLAB, every map is flipped left-right. ``Power`` is ``sum(log|F|)``, Inf set to
        NaN, z-scored (``nanstd`` N-1), ``/6 + 0.5``, masked, 5x5 average filtered (zero
        padding) and clipped to 0..1. Phases are wrapped to ``[-pi, pi)``;
        ``PhaseHor = angle(F_down / F_up)``, ``PhaseVer = angle(F_right / F_left)``; phase
        maps are ``mat2gray(phase + pi/2)``.

    Notes
    -----
    Differences from MATLAB: the loading of maps of the ExpID selected in calliope is not
    ported; the figures are :func:`plot_fourier_maps`.
    """
    F = []
    for f in fourier_specs:
        a = np.asarray(f)
        if a.ndim == 3:
            a = a[:, :, idx]
        F.append(a[:, ::-1].astype(np.complex128))
    with np.errstate(divide="ignore", invalid="ignore"):
        phase = [_wrap(np.angle(f)) for f in F]
        power = sum(np.log(np.abs(f)) for f in F)
        ph_h = _wrap(np.angle(F[1] / F[0]))
        ph_v = _wrap(np.angle(F[3] / F[2]))
    pm_h, pm_v = mat2gray(ph_h + np.pi / 2), mat2gray(ph_v + np.pi / 2)
    power = np.where(np.isinf(power), np.nan, power)
    power = ((power - np.nanmean(power)) / np.nanstd(power, ddof=1)) / 6 + 0.5
    if meandat_mask is not None:
        power[~np.asarray(meandat_mask, dtype=bool)[:, ::-1]] = 0
    raw = power.copy()
    power = imfilter(power, fspecial_average(5))
    with np.errstate(invalid="ignore"):
        power = np.where(power < 0, 0, np.where(power > 1, 1, power))
    g = mat2gray(power)
    cut = np.median(g.ravel())
    alpha = g.copy()
    with np.errstate(invalid="ignore"):
        alpha[alpha < cut * 1.25] = 0
    pmaps = [mat2gray(p + np.pi / 2) for p in phase]
    return FourierMaps(power, phase, pmaps, pm_h, pm_v, ph_h, ph_v, raw, alpha)


def power_overlay(ibv: Any, power: Any, power_alpha: Any, min_area: int = 300) -> np.ndarray:
    """Blood-vessel image with the strong-power region tinted red (``showFourierMaps.m``).

    ``im = repmat(ntzo(Ibv), 1, 1, 3)``; green and blue are multiplied by
    ``1 - Power .* bwareaopen(PowerAlpha > 0, 300)`` (8-connected areas of at least
    ``min_area`` pixels kept).
    """
    im = np.repeat(ntzo(np.asarray(ibv, dtype=np.float64))[:, :, None], 3, axis=2)
    lab, n = ndimage.label(np.asarray(power_alpha) > 0, structure=np.ones((3, 3)))
    if n:
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        keep = np.isin(lab, np.flatnonzero(sizes >= min_area) + 1)
    else:
        keep = np.zeros(lab.shape, dtype=bool)
    fac = 1 - np.asarray(power, dtype=np.float64) * keep
    im[:, :, 1] *= fac
    im[:, :, 2] *= fac
    return im


def plot_fourier_maps(maps: FourierMaps, ibv: Any = None) -> Figure:
    """Power, overlay and phase maps (the figures of ``showFourierMaps.m`` in one Figure).

    Row 1: power (raw, filtered) and, with ``ibv``, the blood-vessel image and the power
    overlay. Rows 2-3: the four direction phase maps and the combined horizontal/vertical
    maps, HSV-coloured with the power as transparency.
    """
    fig = Figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 4)
    panels = [(maps.power_raw, "Power"), (maps.Power, "Power (filtered)")]
    for i, (im, t) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(im, cmap="jet", vmin=0, vmax=1)
        ax.set_title(t)
        ax.axis("off")
    if ibv is not None:
        ax = fig.add_subplot(gs[0, 2])
        ax.imshow(ntzo(np.asarray(ibv, dtype=np.float64)), cmap="gray")
        ax.axis("off")
        ax = fig.add_subplot(gs[0, 3])
        ax.imshow(np.clip(power_overlay(ibv, maps.Power, maps.power_alpha), 0, 1))
        ax.axis("off")
    alpha = np.nan_to_num(mat2gray(maps.Power))
    titles = ["Moving up", "Moving Down", "Moving Left", "Moving Right"]
    for i in range(4):
        ax = fig.add_subplot(gs[1 + i // 2, i % 2])
        ax.imshow(maps.PhaseMap[i], cmap="hsv", alpha=alpha)
        ax.set_title(titles[i])
        ax.axis("off")
    for j, (pm, t) in enumerate([(maps.PhaseMapHor, "Combined horizontal bar"),
                                 (maps.PhaseMapVer, "Combined Vertical Bar")]):
        ax = fig.add_subplot(gs[1 + j, 2])
        ax.imshow(pm, cmap="hsv", alpha=alpha)
        ax.set_title(t)
        ax.axis("off")
    return fig


class OiiFourierResult(NamedTuple):
    """Result of :func:`analyze_oii_fourier` (MATLAB ``[result, dat, fr]`` plus diagnostics)."""

    result: MatStruct
    dat: np.ndarray
    fr: np.ndarray
    stimfreqs: np.ndarray
    sampling_freq: float
    freq_idx: np.ndarray
    """0-based Fourier bin used per direction."""


def analyze_oii_fourier(dat: Any, fr: Any, aux_data: Any, hfreq: float, vfreq: float, *,
                        bv_data: Any = None, spatial_filter_std: float = 0,
                        spatial_filter_window: int = 15, normalize: str = "Mean",
                        frequency_select_range: int = 0, rescue_stim_bounds: bool = False,
                        stim_channels: Sequence[int] = (2, 1), swap_stim_channels: bool = False,
                        selected: int | None = None) -> OiiFourierResult:
    """Fourier analysis of an OII recording with a moving-bar stimulus (``analyzeOII_Fourier.m``).

    Steps: stimulus epochs from the aux data (:func:`oii_stim_bounds`,
    :func:`oii_stim_epochs`); mean image, pyramid-reduced, and its unsaturated mask; optional
    Gaussian filter (``fspecial('gaussian', window, std)``); pyramid reduction of the stack,
    float32; sampling rate ``1000 / mean(diff(frame_times))``; per direction the Fourier
    bin nearest the stimulus frequency (``(fs/2) * linspace(0, 1, n/2)``); normalisation
    (:func:`oii_normalize`); FFT over time; maps (:func:`fourier_maps`).

    Parameters
    ----------
    dat
        ``[h x w x nFrames]`` OII stack (``load_vid_data(..., 'uint16').vdata``).
    fr
        ``vmeta_info`` ``[4 x nFrames]``; row 0 holds the frame times in ms.
    aux_data
        ``[nChannels x nSamples]`` aux data (1 kHz).
    hfreq, vfreq
        Stimulus frequencies of the horizontal and vertical bar (Hz).
    bv_data
        Blood-vessel stack (green LED); its mean, ``mat2gray``-scaled, pyramid-reduced and
        flipped, is ``Ibv``. Default: the flipped reduced mean image.
    frequency_select_range
        Number of neighbouring bins on each side to keep.
    selected
        **0-based** index of the bin offset (in ``-range .. range``) to map. MATLAB asked
        "Use this map?" for each offset; None takes the last one (MATLAB's result when every
        answer is no; with range 0 the only one).

    Returns
    -------
    OiiFourierResult
        ``result`` holds ``Ibv``, ``FourierSpecs`` and the :class:`FourierMaps` outputs
        (``Power``, ``Phase``, ``PhaseMap``, ``PhaseMapHor``, ``PhaseMapVer``, ``PhaseHor``,
        ``PhaseVer``), as MATLAB saves it.

    Notes
    -----
    A difference of more than 0.005 Hz between measured and given stimulus frequencies
    issues a warning (MATLAB asked whether to continue). Differences from MATLAB: ExpID
    lookup, ``.ach`` channel check, workspace caching, the brightness figure and saving are
    not ported.
    """
    aux, sb = oii_stim_bounds(aux_data, stim_channels=stim_channels,
                              rescue_stim_bounds=rescue_stim_bounds,
                              swap_stim_channels=swap_stim_channels)
    frm = np.array(fr, dtype=np.float64)
    frm[0, 0] = 1
    ep = oii_stim_epochs(frm[0], aux, sb, int(stim_channels[1]))
    want = np.array([hfreq, hfreq, vfreq, vfreq], dtype=np.float64)
    if np.any(np.abs(ep.stimfreqs - want) > 0.005):
        warnings.warn(f"Calculated stimulus frequencies {ep.stimfreqs} diverge from the entered "
                      f"ones {want}", stacklevel=2)
    raw = np.asarray(dat)
    meandat = impyramid_reduce(raw.astype(np.float64).mean(axis=2))
    mask = meandat < SATURATION
    d = raw
    if spatial_filter_std > 0:
        d = imfilter(d, fspecial_gaussian(spatial_filter_window, spatial_filter_std))
    d = impyramid_reduce(d).astype(np.float32)
    sf = 1 / np.mean(np.diff(frm[0])) * 1000
    fidx = np.zeros(4, dtype=np.int64)
    for i in range(4):
        n = ep.stop[i] - ep.start[i] + 1
        allf = (sf / 2) * linspace(0, 1, n / 2)
        fidx[i] = int(np.argmin(np.abs(allf - want[i])))
    d = oii_normalize(d, normalize)
    r = int(frequency_select_range)
    offs = np.arange(-r, r + 1)
    specs = []
    for i in range(4):
        tmp = np.fft.fft(d[:, :, ep.start[i] : ep.stop[i] + 1], axis=2).astype(np.complex64)
        sel = tmp[:, :, fidx[i] + offs]
        specs.append(sel[:, :, 0] if offs.size == 1 else sel)
    if bv_data is not None:
        ibv = impyramid_reduce(mat2gray(np.asarray(bv_data).astype(np.float64).mean(axis=2)))[:, ::-1]
    else:
        ibv = meandat[:, ::-1]
    jj = offs.size - 1 if selected is None else int(selected)
    maps = fourier_maps(specs, jj, mask)
    result = MatStruct(Ibv=ibv, FourierSpecs=specs, Power=maps.Power, Phase=maps.Phase,
                       PhaseMap=maps.PhaseMap, PhaseMapHor=maps.PhaseMapHor,
                       PhaseMapVer=maps.PhaseMapVer, PhaseHor=maps.PhaseHor, PhaseVer=maps.PhaseVer)
    return OiiFourierResult(result, d, frm, ep.stimfreqs, float(sf), fidx)


def analyze_oii_fourier_files(oii_file: Any, lvd_file: Any, hfreq: float, vfreq: float, *,
                              bv_file: Any = None, **kwargs) -> OiiFourierResult:
    """:func:`analyze_oii_fourier` from files: the ``.oii`` (read as uint16; an old ``.vid``
    is read with the default int16, as MATLAB's fallback), the ``.lvd`` and optionally the
    blood-vessel ``.vid``/``.oii``."""
    p = str(oii_file)
    v = load_vid_data(p, "int16" if p.lower().endswith(".vid") else "uint16")
    aux = load_lvd(lvd_file).data
    bv = load_vid_data(bv_file).vdata if bv_file is not None else None
    return analyze_oii_fourier(v.vdata, v.vmeta_info, aux, hfreq, vfreq, bv_data=bv, **kwargs)


# --- AnalyzeWaveStimulusIOSseparate ----------------------------------------------------------


def wave_reference_image(img: Any) -> np.ndarray:
    """Blood-vessel / bolus image normalisation of ``AnalyzeWaveStimulusIOSseparate.m``.

    ``imadjust`` (contrast stretch so that 1 % of the pixels saturate at each end, here with
    :func:`calliopy._matlab.prctile` at 1 and 99 instead of MATLAB's ``stretchlim``
    histogram: approximate), then subtract the value at 10 % of the sorted pixels, divide by
    the value at 95 % and clip to 0..1.
    """
    a = np.asarray(img)
    x = a.astype(np.float64)
    if a.dtype.kind in "ui":
        x = x / np.iinfo(a.dtype).max
    lo, hi = prctile(x.ravel(), 1), prctile(x.ravel(), 99)
    x = np.clip((x - lo) / (hi - lo), 0, 1) if hi > lo else x
    v = np.sort(x.ravel(order="F"))
    x = x - v[int(mround(0.1 * v.size)) - 1]
    v = np.sort(x.ravel(order="F"))
    x = x / v[int(mround(0.95 * v.size)) - 1]
    return np.clip(x, 0, 1)


def _smooth_rows(y: np.ndarray, span: float) -> np.ndarray:
    """:func:`smooth_moving` along the last axis of a 2-D array (vectorised)."""
    n = y.shape[1]
    span = max(1, min(int(np.floor(span)), n))
    width = span - 1 + span % 2
    if width <= 1:
        return y.copy()
    half = (width - 1) // 2
    cs = np.concatenate([np.zeros((y.shape[0], 1)), np.cumsum(y, axis=1)], axis=1)
    out = np.empty_like(y)
    out[:, half : n - half] = (cs[:, width:] - cs[:, : n - width + 1]) / width
    for i in range(half):
        w = 2 * i + 1
        out[:, i] = cs[:, w] / w
        out[:, n - 1 - i] = (cs[:, n] - cs[:, n - w]) / w
    return out


def wave_stimulus_maps(data: Sequence[Any], sampling_freqs: Sequence[float],
                       stimulus_freqs: Sequence[float], *, ibv: Any = None, ibl: Any = None,
                       spatial_filter_std: float = 1.5, frequency_select_range: int = 7,
                       selected: Sequence[int] | None = None, do_high_pass: bool = True,
                       normalize_per_frame: bool = True, colormap: Any = None) -> MatStruct:
    """Retinotopic phase maps from four moving-bar recordings (``AnalyzeWaveStimulusIOSseparate.m``).

    Parameters
    ----------
    data
        The four ``BinnedData`` stacks ``[h x w x nFrames]`` in MATLAB's order: HorDown,
        VertLeft, HorUp, VertRight (``HorDown_Data.mat`` ...; old ``.mat`` files can be read
        with :func:`scipy.io.loadmat`).
    sampling_freqs
        ``SamplingFreq`` of each file.
    stimulus_freqs
        The stimulus frequency of each file: ``StimulusFreq.Hor`` for HorDown and HorUp,
        ``.Ver`` for the vertical ones.
    ibv, ibl
        Blood-vessel and bolus images normalised with :func:`wave_reference_image`; zeros
        when missing.
    frequency_select_range
        Number of bins around the stimulus bin (MATLAB 7).
    selected
        Per direction the **0-based** index into those bins (MATLAB asked the user).
        Default: the middle one (the bin nearest the stimulus frequency).
    colormap
        ``[361 x 3]`` RGB table indexed by ``floor(180 + 180 * phase / pi)`` (0-based).
        MATLAB used ``ColorMap(361, 'hsv')``, a function missing from calliope r2003; the
        default is matplotlib's ``hsv`` sampled at 361 points (substitute, not verified).

    Returns
    -------
    MatStruct
        The variables MATLAB saves in ``Wave_maps.mat``: ``PhaseMapHor1``, ``PhaseMapHor2``,
        ``PhaseMapVer1``, ``PhaseMapVer2``, ``PhaseMapHor``, ``PhaseMapVer``,
        ``ScaledPhaseMapHor``, ``ScaledPhaseMapVer``, ``ScaledPhaseMapHorBlack``,
        ``ScaledPhaseMapVerBlack``, ``HorRetinotopy``, ``VerRetinotopy``,
        ``HorVerRetinotopy`` and the six ``...BVoverlay`` / ``...BLoverlay`` images, plus
        ``Power``, ``PhaseHor``, ``PhaseVer``.

    Notes
    -----
    High-pass: ``T - smooth(T, fs * 4 / f_stim)`` per pixel (Curve Fitting Toolbox
    ``smooth``, see :func:`~calliopy.twophoton._builtins.smooth_moving`). The interactive
    directory selection, the figures and the TIFF exports are not ported.
    """
    ds = [np.asarray(d, dtype=np.float64) for d in data]
    shapes = {d.shape[:2] for d in ds}
    if len(shapes) != 1:
        raise ValueError("XY-resolution of image-stacks is not equal")
    yres, xres = ds[0].shape[:2]
    R = int(frequency_select_range)
    fidx = []
    for d, sf, f in zip(ds, sampling_freqs, stimulus_freqs, strict=True):
        allf = (sf / 2) * linspace(0, 1, d.shape[2] / 2)
        fd = np.abs(allf - f)
        fidx.append(int(np.flatnonzero(fd == fd.min())[0]))
    if spatial_filter_std > 0:
        k = fspecial_gaussian(31, spatial_filter_std)
        ds = [imfilter(d, k) for d in ds]
    if normalize_per_frame:
        ds = [d - d.sum(axis=(0, 1))[None, None, :] / (xres * yres) for d in ds]
    rng = np.arange(int(mround((R - 1) / -2)), int(mround((R - 1) / 2)) + 1)[: max(R, 1)]
    specs = []
    for d, sf, f, fi in zip(ds, sampling_freqs, stimulus_freqs, fidx, strict=True):
        t = d.reshape(-1, d.shape[2])
        if do_high_pass:
            t = t - _smooth_rows(t, sf * 4 / f)
        F = np.fft.fft(t, axis=1)
        if R > 1:
            specs.append(F[:, fi + rng].reshape(yres, xres, -1))
        else:
            specs.append(F[:, fi].reshape(yres, xres))
    if R > 1:
        sel = [R // 2] * 4 if selected is None else [int(s) for s in selected]
        specs = [s[:, :, k] for s, k in zip(specs, sel, strict=True)]
    h1, v1, h2, v2 = specs
    with np.errstate(divide="ignore", invalid="ignore"):
        ph = {"Hor1": _wrap(np.angle(h1)), "Ver1": _wrap(np.angle(v1)), "Hor2": _wrap(np.angle(h2)),
              "Ver2": _wrap(np.angle(v2)), "Hor": _wrap(np.angle(h2 / h1)), "Ver": _wrap(np.angle(v2 / v1))}
        power = sum(np.log(np.abs(s)) for s in (h1, h2, v1, v2))
    power = np.where(np.isinf(power), np.nan, power)
    power = ((power - np.nanmean(power)) / np.nanstd(power, ddof=1)) / 6 + 0.5
    power = imfilter(power, fspecial_average(5))
    with np.errstate(invalid="ignore"):
        power = np.where(power < 0, 0, np.where(power > 1, 1, power))
    if colormap is None:
        colormap = colormaps["hsv"](np.linspace(0, 1, 361))[:, :3]
    C = np.asarray(colormap, dtype=np.float64)

    def rgb(p):
        out = np.zeros(p.shape + (3,))
        ok = ~np.isnan(p)
        out[ok] = C[np.floor(180 + (p[ok] / np.pi) * 180).astype(np.int64)]
        return out

    pm = {k: rgb(v) for k, v in ph.items()}
    P3 = np.repeat(power[:, :, None], 3, axis=2)
    out = MatStruct()
    for k in ("Hor1", "Hor2", "Ver1", "Ver2"):
        out[f"PhaseMap{k}"] = (pm[k] - 0.7) * P3 + 0.7
    out["PhaseMapHor"], out["PhaseMapVer"] = pm["Hor"], pm["Ver"]
    out["ScaledPhaseMapHor"] = (pm["Hor"] - 0.7) * P3 + 0.7
    out["ScaledPhaseMapVer"] = (pm["Ver"] - 0.7) * P3 + 0.7
    out["ScaledPhaseMapHorBlack"] = pm["Hor"] * P3
    out["ScaledPhaseMapVerBlack"] = pm["Ver"] * P3
    retino = {}
    for k in ("Hor", "Ver"):
        r = out[f"ScaledPhaseMap{k}Black"].copy()
        with np.errstate(invalid="ignore"):
            r[np.repeat((r.max(axis=2) < 0.2)[:, :, None], 3, axis=2)] = 0
            lines = np.mod(mround(ph[k] * 10), mround((np.pi / 4) * 10)) != 0
        r[np.repeat(lines[:, :, None], 3, axis=2)] = 0
        retino[k] = r
    hv = retino["Ver"].copy()
    ix = np.repeat((retino["Hor"].sum(axis=2) > 0)[:, :, None], 3, axis=2)
    hv[ix] = retino["Hor"][ix]
    out["HorRetinotopy"], out["VerRetinotopy"], out["HorVerRetinotopy"] = retino["Hor"], retino["Ver"], hv
    for name, ref in (("BV", ibv), ("BL", ibl)):
        base = np.zeros((yres, xres)) if ref is None else np.asarray(ref, dtype=np.float64)
        for key, src in (("Hor", retino["Hor"]), ("Ver", retino["Ver"]), ("HorVer", hv)):
            o = np.repeat(base[:, :, None], 3, axis=2)
            m = np.repeat((src.sum(axis=2) > 0)[:, :, None], 3, axis=2)
            o[m] = src[m]
            out[f"{key}Retinotopy{name}overlay"] = o
    out["Power"], out["PhaseHor"], out["PhaseVer"] = power, ph["Hor"], ph["Ver"]
    return out
