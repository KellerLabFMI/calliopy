"""EEG alignment to aux time, event-related potentials and quick-look spectra.

Ports of calliope ``04_Calliope/get_eeg_times.m`` and ``05_LoadData/create_erp_auxaligned.m``,
``myResample.m``, ``myWelch.m``, ``plotERP.m`` and of the computation in
``EEG_check_spectrum.m`` and ``EEG_check_gratings.m`` (their search for the newest files on
the acquisition machine is not ported: pass the files).

EEG arrays are ``[nChannels x nSamples]``. Sample numbers stored or returned as MATLAB
values (trigger samples, ``eeg_times``) are **1-based**; channel and sample arguments that
select something are **0-based**.

None of these functions has been compared with MATLAB output.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
from matplotlib.figure import Figure

from .._matlab import mround
from ..io import EEG_CHANNEL_ORDER, EEG_SYNC_CHANNEL, load_lvd, parse_ach
from ..plotting import plot_bstrp
from ..twophoton._builtins import matlab_colon
from ..widefield import zero_shift_notch

__all__ = [
    "EegGratingErps",
    "EegSpectrum",
    "ErpResult",
    "TriggerAlignmentError",
    "create_erp_auxaligned",
    "eeg_average_spectrum",
    "eeg_average_spectrum_files",
    "eeg_grating_erps",
    "eeg_grating_erps_files",
    "get_eeg_times",
    "my_resample",
    "my_welch",
    "plot_eeg_spectrum",
    "plot_erp",
    "read_eeg_dat",
    "rising_edges",
]


class TriggerAlignmentError(ValueError):
    """Aux and EEG sync triggers do not line up (``get_eeg_times`` returned ``[]``)."""


def rising_edges(x: Any, thr: float = 1) -> np.ndarray:
    """MATLAB ``find(diff(x > thr) == 1)``: **1-based** index of the last sample at or below
    ``thr`` before each crossing (float64)."""
    v = np.asarray(x, dtype=np.float64).ravel()
    return (np.flatnonzero(np.diff((v > thr).astype(np.int8)) == 1) + 1).astype(np.float64)


def get_eeg_times(aux_sync_ch: Any, eeg_sync_ch: Any) -> np.ndarray:
    """Time stamp of every EEG sample in aux samples (``04_Calliope/get_eeg_times.m``, GK/HL).

    Rising edges (> 1 V) of both sync channels are normalised so that the first trigger is 0
    and the trigger at 80 % of the aux triggers is 1. Two checks as MATLAB: the normalised
    trigger sequences may not differ by more than 100 aux samples (any trigger), and the
    linear fit ``aux = a + b * eeg`` (least squares on the normalised triggers) must leave a
    mean residual of at most 100 aux samples. The fit then maps every EEG sample to aux
    time, rounded half away from zero.

    Parameters
    ----------
    aux_sync_ch
        EEG sync channel recorded by the aux recorder.
    eeg_sync_ch
        Sync channel recorded with the EEG.

    Returns
    -------
    numpy.ndarray
        ``(1, nEegSamples)`` float64, **1-based** aux sample numbers (MATLAB values).

    Raises
    ------
    TriggerAlignmentError
        One of the two checks fails (MATLAB printed a message and returned ``[]``).

    Notes
    -----
    Differences from MATLAB: when the aux channel has more triggers than the EEG channel,
    the residual check uses the first ``n`` aux triggers matching the fitted ones (MATLAB
    failed with a size mismatch).
    """
    a_tr = rising_edges(aux_sync_ch)
    e_tr = rising_edges(eeg_sync_ch)
    idx = int(mround(a_tr.size * 0.8)) - 1
    if idx < 1 or idx >= e_tr.size:
        raise TriggerAlignmentError("not enough sync triggers")
    a_span = a_tr[idx] - a_tr[0]
    e_span = e_tr[idx] - e_tr[0]
    na = (a_tr - a_tr[0]) / a_span
    ne = (e_tr - e_tr[0]) / e_span
    m = min(na.size, ne.size)
    a, b = na[:m], ne[:m]
    nd = np.abs(a - b)
    if np.mean(nd * a_span > 100) != 0:
        raise TriggerAlignmentError("Looks like your aux trigger is misaligned, investigate !")
    X = np.c_[np.ones(m), b]
    theta, *_ = np.linalg.lstsq(X, a, rcond=None)
    an, bn = theta
    pred = an + bn * b
    res = np.abs(a_tr[:m] - (a_tr[0] + a_span * pred))
    if res.mean() > 100:
        raise TriggerAlignmentError("Looks like your aux trigger is offset from your eeg trigger, investigate !")
    n = np.asarray(eeg_sync_ch).size
    eidx = np.arange(1, n + 1, dtype=np.float64)
    aux_idx = a_tr[0] + a_span * (an + bn * (eidx - e_tr[0]) / e_span)
    return mround(aux_idx)[None, :]


class ErpResult(NamedTuple):
    """Result of :func:`create_erp_auxaligned` (MATLAB ``[erp, time, epochsIds]``)."""

    erp: np.ndarray
    """``[nChannels x nTime]`` mean over triggers."""
    time: np.ndarray
    """``(1, nTime)`` seconds, ``-sec_before : 1/fs : sec_after``."""
    epochs_ids: np.ndarray
    """``[nTriggers x nTime]`` **0-based** sample indices of the epochs (MATLAB: 1-based)."""


def create_erp_auxaligned(eeg_data: Any, eeg_times: Any, trigger_times: Any, fs: float,
                          sec_before: float = 0.5, sec_after: float = 3) -> ErpResult:
    """Event-related potential with triggers given in aux time (``create_erp_auxaligned.m``).

    Each trigger is placed on the EEG sample whose ``eeg_times`` value is nearest (first one
    on ties); the epoch covers ``floor(time * fs)`` samples around it.

    Parameters
    ----------
    eeg_data
        ``[nChannels x nSamples]``. As MATLAB, ``min(size(eeg_data))`` channels are used.
    eeg_times
        Aux time of every EEG sample (:func:`get_eeg_times`).
    trigger_times
        Trigger times in aux samples.
    fs
        EEG sampling rate (Hz).
    sec_before, sec_after
        Window (MATLAB defaults 0.5 s and 3 s).

    Raises
    ------
    IndexError
        An epoch reaches outside the recording (MATLAB errors as well).
    """
    d = np.asarray(eeg_data, dtype=np.float64)
    et = np.asarray(eeg_times, dtype=np.float64).ravel()
    tt = np.asarray(trigger_times, dtype=np.float64).ravel()
    time = matlab_colon(-sec_before, 1.0 / fs, sec_after)
    win = np.floor(time * fs).astype(np.int64)
    trig0 = np.array([int(np.argmin(np.abs(et - t))) for t in tt], dtype=np.int64)
    ids = win[None, :] + trig0[:, None]
    nch = min(d.shape)
    if ids.size and (ids.min() < 0 or ids.max() >= d.shape[1]):
        raise IndexError("an epoch reaches outside the recording")
    erp = np.full((nch, win.size), np.nan)
    if tt.size:
        for c in range(nch):
            erp[c] = d[c][ids].mean(axis=0)
    return ErpResult(erp, time[None, :], ids)


def my_resample(x: Any, p: float, q: float, dim: int | None = None) -> np.ndarray:
    """Resample by ``p/q`` with a windowed-sinc FIR (``05_LoadData/myResample.m``, ES 2025).

    Zero-stuffing by ``p``, filtering with a Hann-windowed sinc of ``10*max(p, q) + 1`` taps,
    cutoff ``1/(2 max(p, q))`` of the sampling rate, normalised to unit sum (``conv2``
    ``'same'``), and keeping every ``q``-th sample starting with the first.

    Parameters
    ----------
    x
        Vector or 2-D array.
    p, q
        Integer factors.
    dim
        **0-based** axis (default: first non-singleton).

    Notes
    -----
    Kept from MATLAB: the filter is normalised to unit sum, so for ``p > 1`` the output is
    scaled by ``1/p`` (MATLAB's ``resample`` compensates); the calliope caller uses ``p = 1``.
    This differs from :func:`scipy.signal.resample_poly`.
    """
    if abs(p - round(p)) > np.finfo(float).eps or abs(q - round(q)) > np.finfo(float).eps:
        raise ValueError("p and q must be integers.")
    p, q = int(np.round(p)), int(np.round(q))
    a = np.asarray(x, dtype=np.float64)
    if a.ndim > 2:
        raise ValueError("x must be a vector or a 2-D array")
    one_d = a.ndim == 1
    if one_d:
        a = a[:, None]
        dim = 0
    if dim is None:
        nz = [i for i, s in enumerate(a.shape) if s > 1]
        dim = nz[0] if nz else 0
    a = np.moveaxis(a, dim, 0)
    N = 10 * max(p, q)
    fc = 1 / (2 * max(p, q))
    n = np.arange(-(N // 2), N // 2 + 1, dtype=np.float64)
    h = np.sinc(fc * n)
    L = h.size
    h = h * 0.5 * (1 - np.cos(2 * np.pi * np.arange(L) / (L - 1)))
    h = h / h.sum()
    xu = np.zeros((p * a.shape[0], a.shape[1]))
    xu[::p] = a
    half = (L - 1) // 2
    yf = np.empty_like(xu)
    for c in range(xu.shape[1]):
        yf[:, c] = np.convolve(xu[:, c], h, mode="full")[half : half + xu.shape[0]]
    y = np.moveaxis(yf[::q], 0, dim)
    return y[:, 0] if one_d else y


def my_welch(x: Any, window: Any, noverlap: float, nfft: int, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectrum without toolboxes (``05_LoadData/myWelch.m``, ES 2025).

    Segments of the window length, step ``L - noverlap``, each multiplied by the window,
    ``|fft(seg, nfft)|^2 / (L * fs)``, averaged over segments; the first ``nfft/2 + 1``
    (even) or ``(nfft+1)/2`` (odd) bins are returned with ``f = k * fs / nfft``.

    Parameters
    ----------
    window
        Length (a Hamming window ``0.54 - 0.46 cos(2 pi n/(N-1))`` is used) or the window.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``Pxx`` and ``f``, column vectors ``[nBins x 1]``.

    Notes
    -----
    Kept from MATLAB: no factor 2 for the one-sided spectrum and no normalisation by the
    window power, so the values differ from :func:`scipy.signal.welch` and MATLAB
    ``pwelch``.
    """
    v = np.asarray(x, dtype=np.float64).ravel(order="F")
    w = np.asarray(window, dtype=np.float64)
    if w.size == 1:
        N = int(w)
        win = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
    else:
        win = w.ravel()
        N = win.size
    step = N - noverlap
    nseg = int(np.floor((v.size - N) / step)) + 1
    pxx = np.zeros(int(nfft))
    for k in range(max(nseg, 0)):
        start = k * step
        if start != int(start):
            raise ValueError("segment starts must be integers (window - noverlap)")
        start = int(start)
        if start + N > v.size:
            break
        seg = v[start : start + N] * win
        X = np.fft.fft(seg, int(nfft))
        pxx += np.abs(X) ** 2 / (N * fs)
    with np.errstate(invalid="ignore", divide="ignore"):
        pxx = pxx / nseg
    nb = nfft // 2 + 1 if nfft % 2 == 0 else (nfft + 1) // 2
    pxx = pxx[:nb]
    f = np.arange(nb, dtype=np.float64) * fs / nfft
    return pxx[:, None], f[:, None]


def plot_erp(eeg_erp: Any, time: Any, chan_nums: Sequence[int] | None = None, *,
             chan_diff: float = 15, scale_val: float = 5, mult_fak: float = 10,
             chan_shift: float = 5, ax=None) -> Figure:
    """Stacked ERP traces with a scale bar (``05_LoadData/plotERP.m``).

    Channel ``i`` (0-based) is drawn as ``erp[i] * mult_fak - i * chan_diff``; the y tick
    labels are ``chan_nums`` flipped (default ``1..30``); a ``scale_val`` uV bar is drawn
    below the last channel.

    Returns the Figure (MATLAB returned the last line handle).
    """
    e = np.asarray(eeg_erp, dtype=np.float64)
    t = np.asarray(time, dtype=np.float64).ravel()
    cn = list(range(1, 31)) if chan_nums is None else list(chan_nums)
    if ax is None:
        fig = Figure(figsize=(6, 8))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.figure
    n = e.shape[0]
    for i in range(n):
        ax.plot(t, e[i] * mult_fak - i * chan_diff)
    ax.axvline(0, color="r", ls="--")
    xpos = t[-1] - t[-1] * 0.15
    y0 = -(n + chan_shift) * chan_diff
    ax.plot([xpos, xpos], [y0, y0 + scale_val * mult_fak], color="k", lw=1)
    ax.text(xpos + t[-1] * 0.01, y0 + scale_val * mult_fak / 2, f"{scale_val:g}\u00b5V", fontsize=8)
    ticks = np.arange(-(n - 1) * chan_diff, chan_diff / 2, chan_diff)
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(c) for c in cn[::-1]][: ticks.size])
    ax.set_ylabel("Channel")
    ax.set_xlabel("Time [s]")
    return fig


def read_eeg_dat(dat_file: Any, oebin_file: Any = None) -> tuple[np.ndarray, float]:
    """Read an OpenEphys ``EEG.dat`` file scaled to microvolts, as the EEG_check functions.

    Returns ``[numChannels x nSamples]`` float32 (``single(int16) * bit_volts`` per channel)
    and ``continuous.sample_rate`` from the ``.oebin`` header (default: same path with suffix
    ``.oebin``).
    """
    if oebin_file is None:
        oebin_file = os.path.splitext(os.fspath(dat_file))[0] + ".oebin"
    with open(oebin_file, encoding="utf-8-sig") as f:
        s = json.load(f)
    cont = s["continuous"]
    cont = cont[0] if isinstance(cont, list) else cont
    nch = int(cont["num_channels"])
    raw = np.fromfile(dat_file, dtype="<i2")
    data = raw.reshape((-1, nch)).T.astype(np.float32)
    bv = np.array([float(cont["channels"][i]["bit_volts"]) for i in range(nch)])
    data = (data * bv[:, None]).astype(np.float32)
    return data, float(cont["sample_rate"])


def _reorder(e: np.ndarray) -> np.ndarray:
    return e[list(EEG_CHANNEL_ORDER), :]


class EegSpectrum(NamedTuple):
    """Result of :func:`eeg_average_spectrum`."""

    freq_spec: np.ndarray
    """Frequencies within the limits, ``[nFreq x 1]``."""
    rec_means: np.ndarray
    """``[nFreq x nRecordings]`` power averaged over channels."""


def eeg_average_spectrum(recordings: Sequence[Any], fs: float, *, common_avg: bool = True,
                         freq_lim: Sequence[float] = (0, 100), window_size_sec: float = 10,
                         exclude_chan: Sequence[int] = ()) -> EegSpectrum:
    """Channel-averaged power spectrum per recording (``05_LoadData/EEG_check_spectrum.m``).

    Per recording: channels into probe order (:data:`calliopy.io.EEG_CHANNEL_ORDER`),
    ``exclude_chan`` rows removed, common average subtracted, cut to the shortest recording;
    :func:`my_welch` with a ``round(window_size_sec * fs)`` Hamming window, half overlap and
    ``nfft`` equal to the window; mean over channels within ``freq_lim``.

    Parameters
    ----------
    recordings
        ``[numChannels x nSamples]`` scaled data per stack (:func:`read_eeg_dat`).
    fs
        Sampling rate of the first recording (MATLAB uses it for all).
    exclude_chan
        **0-based** rows in probe order to drop (MATLAB: 1-based).
    """
    excl = np.asarray(list(exclude_chan), dtype=np.int64)
    recs = []
    for r in recordings:
        e = _reorder(np.asarray(r, dtype=np.float32))
        if excl.size:
            e = np.delete(e, excl, axis=0)
        if common_avg:
            e = (e - e.mean(axis=0, dtype=np.float64).astype(np.float32)).astype(np.float32)
        recs.append(e)
    rec_len = min(e.shape[1] for e in recs)
    window = int(mround(window_size_sec * fs))
    _, f = my_welch(np.zeros(window), window, window / 2, window, fs)
    f = f[:, 0]
    mask = (f >= freq_lim[0]) & (f <= freq_lim[1])
    means = np.zeros((int(mask.sum()), len(recs)))
    for j, e in enumerate(recs):
        p = np.stack([my_welch(e[c, :rec_len], window, window / 2, window, fs)[0][:, 0]
                      for c in range(e.shape[0])])
        means[:, j] = p[:, mask].mean(axis=0)
    return EegSpectrum(f[mask][:, None], means)


def eeg_average_spectrum_files(dat_files: Sequence[Any], **kwargs) -> EegSpectrum:
    """:func:`eeg_average_spectrum` of ``EEG.dat`` files (headers next to them)."""
    data = [read_eeg_dat(p) for p in dat_files]
    return eeg_average_spectrum([d for d, _ in data], data[0][1], **kwargs)


def plot_eeg_spectrum(spec: EegSpectrum, *, rng: np.random.Generator, nbss: int = 1000,
                      title: str = "") -> Figure:
    """Bootstrapped mean spectrum on a log axis (``EEG_check_spectrum.m``, ``plotBSTRP``)."""
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    plot_bstrp(spec.rec_means, np.ones(spec.rec_means.shape[1]), nbss,
               x_data=spec.freq_spec[:, 0], rng=rng, ax=ax, label="Frequency Spectrum")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power [\u00b5V\u00b2]")
    if title:
        ax.set_title(title)
    return fig


class EegGratingErps(NamedTuple):
    """Result of :func:`eeg_grating_erps`."""

    erp_on: np.ndarray
    time_on: np.ndarray
    erp_off: np.ndarray
    time_off: np.ndarray


def eeg_grating_erps(stacks: Sequence[dict], *, common_avg: bool = True, pre_erp_sec: float = 0.5,
                     post_erp_sec: float = 1, down_samp: float = 1000, f_aux: float = 10000,
                     exclude_chan: Sequence[int] = ()) -> EegGratingErps:
    """Visual evoked potentials at grating on- and offsets (``05_LoadData/EEG_check_gratings.m``).

    Per stack (in stack order): resample to ``down_samp`` (:func:`my_resample`, ``q = fs /
    down_samp``), take the sync channel (row :data:`calliopy.io.EEG_SYNC_CHANNEL`), put the
    channels in probe order, set ``exclude_chan`` rows to NaN, subtract the NaN-ignoring
    common average, notch 50 Hz (2 Hz wide,
    :func:`calliopy.widefield.zero_shift_notch`); align with :func:`get_eeg_times`;
    grating on/offsets are rising/falling edges of ``grating > 1`` at least ``pre`` / ``post``
    seconds (in aux samples, ``f_aux``) inside the EEG. Stacks are concatenated with their
    aux times offset by the previous stack's last EEG time; then
    :func:`create_erp_auxaligned`.

    Parameters
    ----------
    stacks
        One dict per stack with ``eeg`` (``[numChannels x nSamples]`` scaled,
        :func:`read_eeg_dat`), ``fs``, ``grating`` (aux ``StimulusID`` channel) and
        ``aux_sync`` (aux ``EEGsync`` channel).
    exclude_chan
        **0-based** rows in probe order set to NaN.
    """
    excl = list(exclude_chan)
    all_eeg, all_times, all_on, all_off = [], [], [], []
    last = None
    for s in stacks:
        e = my_resample(np.asarray(s["eeg"]), 1, s["fs"] / down_samp, 1)
        sync = e[EEG_SYNC_CHANNEL]
        e = _reorder(e)
        e[excl, :] = np.nan
        if common_avg:
            with np.errstate(all="ignore"):
                e = e - np.nanmean(e, axis=0)
        e = zero_shift_notch(e, down_samp, 50, 2)
        g = np.asarray(s["grating"], dtype=np.float64).ravel()
        et = get_eeg_times(s["aux_sync"], sync)[0]
        gb = (g > 1).astype(np.int8)
        on = (np.flatnonzero(np.diff(gb) == 1) + 1).astype(np.float64)
        off = (np.flatnonzero(np.diff(gb) == -1) + 1).astype(np.float64)
        on = on[(on - pre_erp_sec * f_aux > et[0]) & (on + post_erp_sec * f_aux < et[-1])]
        off = off[(off - pre_erp_sec * f_aux > et[0]) & (off + post_erp_sec * f_aux < et[-1])]
        if last is not None:
            et, on, off = et + last, on + last, off + last
        all_eeg.append(e)
        all_times.append(et)
        all_on.append(on)
        all_off.append(off)
        last = et[-1]
    eeg = np.hstack(all_eeg)
    times = np.concatenate(all_times)
    r_on = create_erp_auxaligned(eeg, times, np.concatenate(all_on), down_samp, pre_erp_sec, post_erp_sec)
    r_off = create_erp_auxaligned(eeg, times, np.concatenate(all_off), down_samp, pre_erp_sec, post_erp_sec)
    return EegGratingErps(r_on.erp, r_on.time, r_off.erp, r_off.time)


def eeg_grating_erps_files(dat_files: Sequence[Any], lvd_files: Sequence[Any],
                           ach_files: Sequence[Any], **kwargs) -> EegGratingErps:
    """:func:`eeg_grating_erps` from files: ``EEG.dat`` (+ ``.oebin``), ``.lvd`` and ``.ach``.

    The aux rows are the positions of the ``StimulusID`` and ``EEGsync`` labels in the
    ``.ach`` channel table, as MATLAB.
    """
    stacks = []
    for dat, lvd, ach in zip(dat_files, lvd_files, ach_files, strict=True):
        eeg, fs = read_eeg_dat(dat)
        aux = np.asarray(load_lvd(lvd).data)
        labels = [str(v) for v in np.asarray(parse_ach(ach)["aux_chans"], dtype=object)[:, 1]]
        stacks.append({"eeg": eeg, "fs": fs, "grating": aux[labels.index("StimulusID")],
                       "aux_sync": aux[labels.index("EEGsync")]})
    return eeg_grating_erps(stacks, **kwargs)
