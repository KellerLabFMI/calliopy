"""Overview figure of the sites of a proj_meta: calliope ``24_ProjMeta/TP_checkPmSites.m``."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from .._matlab import filter2, prctile
from ._access import _get, _rd, _sites

__all__ = ["plot_pm_sites"]

_TRACES = (("Running_smoothed", "Running"), ("VisualFlow_smoothed", "VisFlow"), ("GratingID", "Gratings"),
           ("Stim", "Opto"), ("ctrlLED", "LED"))
_FG = (0.85, 0.85, 0.85)


def _gauss_row(width: int, sigma: float) -> np.ndarray:
    """MATLAB ``fspecial('gaussian', [1, width], sigma)``."""
    x = np.arange(width) - (width - 1) / 2
    k = np.exp(-(x**2) / (2 * sigma**2))
    k[k < np.finfo(float).eps * k.max()] = 0
    return (k / k.sum()).reshape(1, -1)


def _imfilter(a: np.ndarray, k: np.ndarray, replicate: bool) -> np.ndarray:
    """MATLAB ``imfilter(a, k)`` (correlation, 'same'), zero or replicate padding along rows."""
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    if not replicate:
        return filter2(k, a, "same")
    w = k.shape[1]
    padded = np.pad(a, ((0, 0), (w, w)), mode="edge")
    return filter2(k, padded, "same")[:, w:-w]


def _rescale(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """MATLAB ``rescale(v, 'InputMin', lo, 'InputMax', hi)`` into [0, 1]."""
    c = np.clip(v, lo, hi)
    return (c - lo) / (hi - lo) if hi > lo else np.zeros_like(c)


def _collect_traces(rd_col, rescale_to: float, downsample: int):
    traces, names = [], []
    first = rd_col[0]
    for field, label in _TRACES:
        if field not in first:
            continue
        v = np.asarray(first[field], dtype=np.float64).ravel()
        if v.size == 0:
            continue
        t = _imfilter(v, _gauss_row(downsample, downsample / 2), replicate=False).ravel()[::downsample]
        t = _rescale(t, float(prctile(t, 0.1)), float(prctile(t, 99.9))) * rescale_to
        traces.append(t)
        names.append(label)
    return traces, names


def plot_pm_sites(proj_meta, *, site_indices=None, downsample: int = 10, plot_fissa: bool = False,
                  keep_clims: bool = True, rastermap_sort: bool = False) -> list[Figure]:
    """Activity rasters with behaviour traces, one figure per site (``TP_checkPmSites.m``).

    One row of axes per timepoint: all cells of all layers (``act``, or ``act_fissa``),
    Gaussian-filtered and downsampled, with the running, visual flow, grating, opto and
    LED traces (when present) drawn in the band below the cells, dotted lines between
    stacks and the stack comment as title.

    Parameters
    ----------
    proj_meta : struct array
    site_indices : sequence of int, optional
        **0-based** sites to plot (default all).
    downsample : int, default 10
        Downsampling factor (>= 1); activity is filtered with a Gaussian of width
        ``4*downsample + 1`` and sigma ``downsample`` first (edges replicated).
    plot_fissa : bool, default False
        Plot the FISSA-corrected activity.
    keep_clims : bool, default True
        Use common colour limits (0.1th to 99.5th percentile over all timepoints).
    rastermap_sort : bool, default False
        Not available (rastermap is a third-party package); ``True`` raises.

    Returns
    -------
    list of matplotlib.figure.Figure
        One figure per site.

    Differences from MATLAB
    -----------------------
    MATLAB's ``prctiles_rescale`` option is not used by MATLAB either and is left out;
    figure placement on the screen is not reproduced.
    """
    if downsample < 1 or int(downsample) != downsample:
        raise ValueError("downsample must be an integer >= 1")
    if rastermap_sort:
        raise NotImplementedError("rastermap sorting needs the third-party rastermap package")
    downsample = int(downsample)
    sites = _sites(proj_meta)
    if site_indices is None:
        site_indices = range(len(sites))
    figs = []
    for i, s in enumerate(site_indices):
        site = sites[int(s)]
        rd = _rd(site)
        n_tp = rd.shape[1]
        fig = Figure(facecolor=(0.15, 0.15, 0.15), figsize=(12, 2.5 * max(n_tp, 1)))
        fig.suptitle(f"SiteID: {int(np.ravel(site['ExpGroup'])[0])} ({int(s) + 1}) - MouseID: {site['animal']}",
                     color=_FG)
        axs = fig.subplots(max(n_tp, 1), 1, squeeze=False)[:, 0]
        clims, images = [], []
        for tp in range(n_tp):
            ax = axs[tp]
            field = "act_fissa" if plot_fissa else "act"
            parts = [np.asarray(_get(rd[z, tp], field), dtype=np.float64) for z in range(rd.shape[0])]
            parts = [p for p in parts if p.ndim == 2 and p.size]
            act = np.vstack(parts) if parts else np.zeros((0, 1))
            if downsample > 1 and act.size:
                width = downsample * 4 + 1
                act = _imfilter(act, _gauss_row(width, downsample), replicate=True)[:, ::downsample]
            n_neurons = act.shape[0]
            act = np.vstack((np.full((int(np.ceil(n_neurons / 5)), act.shape[1]), np.nan), act))
            traces, names = _collect_traces(rd[:, tp], n_neurons / 5, downsample)
            finite = act[np.isfinite(act)]
            lims = (float(prctile(finite, 0.1)), float(prctile(finite, 99.5))) if finite.size else (0.0, 1.0)
            im = ax.imshow(act, aspect="auto", origin="lower", cmap="magma", interpolation="nearest",
                           vmin=None if keep_clims else lims[0], vmax=None if keep_clims else lims[1])
            images.append(im)
            clims.append(lims)
            cb = fig.colorbar(im, ax=ax)
            cb.ax.tick_params(colors=_FG)
            ax.set_ylabel("Neuron #", color=_FG)
            ax.set_xticks([])
            ax.set_yticks([])
            for t, nm in zip(traces, names, strict=True):
                ax.plot(np.arange(t.size), t, label=nm)
            nbf = np.ravel(np.asarray(_get(rd[0, tp], "nbr_frames"), dtype=np.float64))
            for x in (np.cumsum(nbf) / downsample)[:-1]:
                ax.axvline(x, color=(0.5, 0.5, 0.5), linewidth=0.5, linestyle=":")
            if tp == n_tp - 1:
                ax.set_xlabel("Time [Frames]", color=_FG)
                if traces:
                    ax.legend(loc="lower right", ncol=len(traces))
            comments = _get(rd[0, tp], "comments")
            first = np.ravel(np.asarray(comments, dtype=object))[0] if np.size(comments) else ""
            ax.set_title(f'Timepoint: {tp + 1} - "{first}"', color=_FG)
        if keep_clims and clims:
            lo, hi = min(c[0] for c in clims), max(c[1] for c in clims)
            for im in images:
                im.set_clim(lo, hi)
        figs.append(fig)
    return figs
