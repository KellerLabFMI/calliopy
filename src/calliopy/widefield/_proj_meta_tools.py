"""proj_meta struct arrays: site selection, preprocessing and per-ROI power maps.

Ports of calliope ``21_Analysis`` functions that take and return a ``proj_meta`` struct
array (``WF_bandpass``, ``WF_trimStackFrames``, ``WF_velocityField``, ``WF_commodore``) and
of ``22_Plotting/WF_imageROIvelocityField``.

proj_meta in Python
-------------------
A metafile read with :func:`calliopy.io.load_mat` gives ``proj_meta`` as an object array of
:class:`~calliopy.io.MatStruct` with MATLAB's shape (``(1, nSites)``), or a single
``MatStruct`` when MATLAB stored one site. Each site's ``rd`` is an object array
``(nLayers, nTimepoints)`` (or a single ``MatStruct``). Every function here accepts any of
these forms, and also a plain list of sites (dicts); internally the sites are taken in
MATLAB's linear (column-major) order. Functions that return a proj_meta return a new object
array of shape ``(1, nSites)`` holding copies of the sites; the caller's data are never
changed.

Index conventions: every argument that selects timepoints or ROIs is **0-based**
(``tp_list=[0, 1]`` is MATLAB's ``tp_list=[1 2]``). ``rd(1, tp)`` of MATLAB is
``rd[0, tp]``. Values stored in proj_meta (``frame_times``, ROI ``indices``, ``AP_index``,
``ML_index``) keep their MATLAB values. Site IDs (``ExpGroup(1)``) are lab labels, not
indices, and are compared as they are.

What is not ported: the temp-file cache of ``WF_bandpass``/``WF_velocityField`` (they saved
and re-loaded the result under ``set_lab_paths('TempFile_dir')``): nothing is written or
read here, but the ``uuid`` tag is still updated so that ``WF_corrmat``'s "already
bandpassed" check keeps working. Nothing has been compared with MATLAB output.
"""

from __future__ import annotations

import copy
import logging
import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import scipy.signal

from calliopy._matlab import mround
from calliopy.io import MatStruct

from ._flow import uv_img_to_linear, uv_linear_to_img, velocity_field_horn_schunck, wf_grid2vid

__all__ = [
    "BANDPASS_PAD",
    "N_GRID_ROIS",
    "plot_roi_velocity_field",
    "plot_wf_commodore",
    "select_sites",
    "wf_bandpass",
    "wf_commodore",
    "wf_trim_stack_frames",
    "wf_velocity_field",
]

logger = logging.getLogger(__name__)

N_GRID_ROIS = 818  #: brain ROIs of the standard grid (824 = 2 reference points + 818 + 4 corners)
BANDPASS_PAD = 3000  #: samples mirrored at each end before band-pass filtering (MATLAB l_pad)
_TIME_SERIES_FIELDS = (
    "act", "frame_times", "PerturbationLFM", "VisualFlow", "VisualFlow_smoothed", "Running",
    "Running_smoothed", "AirPuff_extended", "StimulusID", "vis_speed", "ECTstim",
)


# --- struct helpers (shared with _network and _onsets) -------------------------------------


def _sites(proj_meta) -> list:
    """Sites of a proj_meta struct array as a list, in MATLAB linear order."""
    if isinstance(proj_meta, dict):
        return [proj_meta]
    if isinstance(proj_meta, np.ndarray):
        return list(proj_meta.ravel(order="F"))
    if isinstance(proj_meta, Sequence) and not isinstance(proj_meta, str):
        return list(proj_meta)
    raise TypeError("proj_meta must be a struct, a struct array or a list of structs")


def _as_struct_array(sites: list) -> np.ndarray:
    out = np.empty((1, len(sites)), dtype=object)
    for i, s in enumerate(sites):
        out[0, i] = s
    return out


def _rd_grid(site) -> np.ndarray:
    """``site.rd`` as a 2-D object array ``(nLayers, nTimepoints)``."""
    rd = site.get("rd")
    if rd is None:
        return np.empty((0, 0), dtype=object)
    if isinstance(rd, dict):
        g = np.empty((1, 1), dtype=object)
        g[0, 0] = rd
        return g
    if isinstance(rd, np.ndarray):
        if rd.size == 0:
            return np.empty((0, 0), dtype=object)
        return rd if rd.ndim == 2 else rd.reshape(1, -1)
    rd = list(rd)
    g = np.empty((1, len(rd)), dtype=object)
    for i, r in enumerate(rd):
        g[0, i] = r
    return g


def _n_tp(site) -> int:
    """MATLAB ``size(site.rd, 2)``."""
    return _rd_grid(site).shape[1]


def _rd_row(site, tp: int):
    """MATLAB ``site.rd(1, tp+1)`` (``tp`` 0-based)."""
    g = _rd_grid(site)
    if not 0 <= tp < g.shape[1]:
        raise IndexError(f"timepoint {tp} (0-based) out of range for rd of size {g.shape}")
    return g[0, tp]


def _rd_linear(site, k: int):
    """MATLAB ``site.rd(k+1)`` (linear, column-major; ``k`` 0-based)."""
    g = _rd_grid(site).ravel(order="F")
    if not 0 <= k < g.size:
        raise IndexError(f"rd index {k} (0-based) out of range for {g.size} entries")
    return g[k]


def _isempty(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v == ""
    if isinstance(v, dict):
        return False
    return np.size(v) == 0


def _field(s, name: str, default=None):
    return s[name] if (isinstance(s, dict) and name in s) else default


def _texts(v) -> list[str]:
    """A char array or a cell array of char as a list of strings (MATLAB linear order)."""
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, np.ndarray):
        return ["" if x is None or (isinstance(x, np.ndarray) and x.size == 0) else str(x)
                for x in v.ravel(order="F")]
    if isinstance(v, Sequence):
        return [str(x) for x in v]
    return [str(v)]


def _str_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return [str(x) for x in v]


def _contains(text, patterns) -> np.ndarray:
    """MATLAB ``contains(text, patterns)``: 1-D bool, one element per string of ``text``."""
    pats = _str_list(patterns)
    return np.array([any(p in t for p in pats) for t in _texts(text)], dtype=bool)


def _vec(v) -> np.ndarray:
    """A MATLAB vector (any orientation) as a 1-D array in linear order."""
    return np.asarray(v).ravel(order="F")


def _matlab_logical(mask, n: int) -> np.ndarray:
    """MATLAB logical indexing ``x(mask)`` of a dimension of length ``n``.

    A shorter mask is padded with ``False`` (as MATLAB does); a longer mask with a ``True``
    beyond ``n`` raises ``IndexError`` (MATLAB: index exceeds the array bounds).
    """
    m = np.asarray(mask).ravel(order="F").astype(bool)
    if m.size > n:
        if np.any(m[n:]):
            raise IndexError(f"logical index has a true element beyond position {n}")
        return m[:n]
    out = np.zeros(n, dtype=bool)
    out[: m.size] = m
    return out


def _concat_field(structs, name: str) -> np.ndarray:
    """MATLAB ``[s.name]`` of numeric scalars: empty fields are skipped (positions shift)."""
    vals = []
    for s in _sites(structs):
        v = _field(s, name)
        if not _isempty(v):
            vals.extend(np.asarray(v, dtype=np.float64).ravel(order="F").tolist())
    return np.array(vals, dtype=np.float64)


def _cell_field(structs, name: str) -> list[str]:
    """MATLAB ``{s.name}`` of strings: one entry per struct (empty fields are ``''``)."""
    out = []
    for s in _sites(structs):
        v = _field(s, name)
        out.append("" if _isempty(v) else str(v))
    return out


def _num2str(x: float) -> str:
    """MATLAB ``num2str`` of a real scalar (integers exactly, else ``%.<d>g``)."""
    x = float(x)
    if math.isfinite(x) and x == round(x):
        return str(int(x))
    if not math.isfinite(x):
        return "NaN" if math.isnan(x) else ("Inf" if x > 0 else "-Inf")
    d = max(math.floor(math.log10(abs(x))) + 5, 5) if x != 0 else 5
    return f"{x:.{min(d, 16)}g}"


def _copy_proj_meta(proj_meta) -> list:
    """Copies of the sites and of every rd entry (arrays inside are shared, not copied)."""
    out = []
    for s in _sites(proj_meta):
        s2 = MatStruct(s) if isinstance(s, MatStruct) else MatStruct(dict(s))
        if "rd" in s:
            g = _rd_grid(s)
            g2 = np.empty(g.shape, dtype=object)
            for idx in np.ndindex(g.shape):
                r = g[idx]
                g2[idx] = MatStruct(r) if isinstance(r, dict) else copy.copy(r)
            s2["rd"] = g2
        out.append(s2)
    return out


def select_sites(proj_meta, site_ids) -> np.ndarray:
    """Sites whose site ID (first element of ``ExpGroup``) is in ``site_ids``.

    The ``'siteID'`` filter shared by calliope ``21_Analysis/WF_corrmat.m``,
    ``WF_corrmat_matrix.m``, ``WF_LesioningROIs.m`` and ``WF_commodore.m``.

    Parameters
    ----------
    proj_meta
        proj_meta struct array (see the module docstring).
    site_ids
        Site IDs to keep: values compared with ``ExpGroup(1)`` (lab labels, not indices).

    Returns
    -------
    numpy.ndarray
        Object array ``(1, nKept)`` of the kept sites (the same objects, not copies), in
        their original order. Sites without ``ExpGroup`` or with an empty one are dropped.
    """
    ids = np.atleast_1d(np.asarray(site_ids, dtype=np.float64)).ravel()
    kept = []
    for s in _sites(proj_meta):
        eg = _field(s, "ExpGroup")
        if not _isempty(eg) and np.any(ids == float(_vec(eg)[0])):
            kept.append(s)
    return _as_struct_array(kept)


def _filter_sites(sites: list, site_ids) -> list:
    if site_ids is None or np.size(site_ids) == 0:
        return sites
    kept = _sites(select_sites(sites, site_ids))
    if not kept:
        warnings.warn(f"No sites found matching the specified siteID(s): {site_ids}",
                      stacklevel=3)
    else:
        logger.info("Processing %d site(s) matching siteID(s): %s", len(kept), site_ids)
    return kept


# --- WF_bandpass ------------------------------------------------------------------------------


def _bandpass_fir(x: np.ndarray, fs: float, wpass, steepness: float = 0.85,
                  attenuation: float = 60.0) -> np.ndarray:
    """Zero-delay band-pass of each row with a minimum-order Kaiser FIR (approximation).

    Follows the documented behaviour of MATLAB ``bandpass(x', wpass, fs)`` (FIR path):
    stopband attenuation 60 dB, transition widths ``(1 - steepness)`` of the distance from
    each pass-band edge to 0 and to the Nyquist frequency, delay compensated. The filter
    design (``scipy.signal.kaiserord``/``firwin``) is NOT MATLAB's, so values differ.
    """
    nyq = fs / 2.0
    lo, hi = float(wpass[0]), float(wpass[1])
    if not 0 < lo < hi < nyq:
        raise ValueError(f"bandpass_range must satisfy 0 < low < high < {nyq} Hz, got {wpass}")
    w_lo = (1 - steepness) * lo
    w_hi = (1 - steepness) * (nyq - hi)
    width = min(w_lo, w_hi)
    numtaps, beta = scipy.signal.kaiserord(attenuation, width / nyq)
    numtaps += 1 - numtaps % 2  # type I (odd length) for a band-pass
    cut = [lo - w_lo / 2.0, hi + w_hi / 2.0]
    b = scipy.signal.firwin(numtaps, cut, window=("kaiser", beta), pass_zero=False, fs=fs)
    delay = (numtaps - 1) // 2
    full = scipy.signal.oaconvolve(x, b[np.newaxis, :], mode="full", axes=1)
    return full[:, delay : delay + x.shape[1]]


def wf_bandpass(
    proj_meta,
    bandpass_range=(0.2, 8.0),
    framerate: float | None = None,
    *,
    filter_fn: Callable[[np.ndarray, float, tuple[float, float]], np.ndarray] | None = None,
) -> np.ndarray:
    """Median-normalise and band-pass filter the ROI activity of every site and timepoint.

    Port of calliope ``21_Analysis/WF_bandpass.m``. For every site, every timepoint
    ``rd[0, tp]`` and every activity field (``act``, or ``act_ch525`` and ``act_ch610`` when
    the first rd entry has ``act_ch525``):

    1. ``act = act / median(act, 2)`` (per-ROI median over time);
    2. rows with any NaN or Inf are set to 0;
    3. ``BANDPASS_PAD`` (3000) samples are mirrored at both ends
       (``[fliplr(act(:,1:3000)) act fliplr(act(:,end-2999:end))]``);
    4. every row is band-pass filtered and the padding is removed;
    5. ``act = 1 + filtered``; the bad rows become NaN again.

    Parameters
    ----------
    proj_meta
        proj_meta struct array (module docstring). Not changed.
    bandpass_range
        ``(low, high)`` pass band in Hz (MATLAB default ``[0.2 8]``).
    framerate
        Frame rate in Hz. Default (MATLAB): ``round(10000 / median(diff(frame_times)))`` of
        the first site's ``rd[0, 0]`` (frame times in aux samples at 10 kHz).
    filter_fn
        ``filter_fn(x, fs, (low, high))`` filtering every row of ``x`` (``[nROI x nT]``)
        and returning an array of the same shape. Default: a Kaiser-window FIR that follows
        MATLAB's documented ``bandpass`` defaults (see Notes).

    Returns
    -------
    numpy.ndarray
        New proj_meta, object array ``(1, nSites)``, with the filtered activity (float64).
        When the sites have a ``uuid`` field, the first site's ``uuid`` gets the suffix
        ``'-WF_bandpass_<low>-<high>'`` (MATLAB ``num2str`` formatting), as in MATLAB.

    Raises
    ------
    ValueError
        A recording shorter than 3000 frames (MATLAB failed indexing the padding).

    Notes
    -----
    Differences from MATLAB:

    - **Filter.** MATLAB's ``bandpass`` (Signal Processing Toolbox) designs its own
      minimum-order filter (FIR when the signal is long enough, else IIR with
      ``filtfilt``); that design cannot be reproduced with scipy. The default here is an
      approximation with the same documented specification (60 dB stopband, steepness
      0.85, delay compensation). Results differ from MATLAB in the transition bands. Pass
      ``filter_fn`` to use another filter (e.g. a wrapper of :func:`zero_shift_bandpass`).
    - No temp file is searched or saved (MATLAB: ``set_lab_paths('TempFile_dir')``); no
      progress is printed.
    """
    sites = _copy_proj_meta(proj_meta)
    if not sites:
        return _as_struct_array(sites)
    lo, hi = float(bandpass_range[0]), float(bandpass_range[1])
    if framerate is None:
        ft = _vec(_field(_rd_row(sites[0], 0), "frame_times")).astype(np.float64)
        framerate = float(mround(10000.0 / np.median(np.diff(ft))))
    fn = filter_fn or (lambda x, fs, band: _bandpass_fir(x, fs, band))
    first_rd = _rd_linear(sites[0], 0)
    act_names = ("act_ch525", "act_ch610") if "act_ch525" in first_rd else ("act",)
    pad = BANDPASS_PAD
    for k, site in enumerate(sites):
        logger.info("Now processing site %d of %d", k + 1, len(sites))
        g = site["rd"]
        for tp in range(g.shape[1]):
            rd = g[0, tp]
            for name in act_names:
                act = np.asarray(rd[name], dtype=np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):
                    act = act / np.median(act, axis=1, keepdims=True)
                bad = np.any(~np.isfinite(act), axis=1)
                act[bad, :] = 0.0
                if act.shape[1] < pad:
                    raise ValueError(
                        f"recording has {act.shape[1]} frames; WF_bandpass needs at least {pad}"
                    )
                padded = np.concatenate(
                    [act[:, :pad][:, ::-1], act, act[:, -pad:][:, ::-1]], axis=1
                )
                filt = np.asarray(fn(padded, framerate, (lo, hi)), dtype=np.float64)
                act = 1.0 + filt[:, pad:-pad]
                act[bad, :] = np.nan
                rd[name] = act
    if "uuid" in sites[0]:
        sites[0]["uuid"] = f"{sites[0]['uuid']}-WF_bandpass_{_num2str(lo)}-{_num2str(hi)}"
    return _as_struct_array(sites)


# --- WF_trimStackFrames -----------------------------------------------------------------------


def _subexp_frames(rd, sub: int) -> tuple[int, int]:
    """1-based inclusive frame range of sub-experiment ``sub`` (0-based)."""
    nf = rd["nbr_frames"]
    counts = [int(np.asarray(c).ravel()[0]) for c in nf.ravel(order="F")] if (
        isinstance(nf, np.ndarray) and nf.dtype == object
    ) else [int(c) for c in _vec(nf)]
    start = 1 + sum(counts[:sub])
    return start, start + counts[sub] - 1


def _cols(v, a: int, b: int) -> np.ndarray:
    """MATLAB ``v(:, a:b)`` with 1-based inclusive ``a``, ``b``."""
    arr = np.asarray(v)
    arr2 = arr.reshape(1, -1) if arr.ndim < 2 else arr
    if b > arr2.shape[1] or a < 1:
        raise IndexError(f"frames {a}:{b} exceed a field with {arr2.shape[1]} columns")
    return arr2[:, a - 1 : b]


def _hcat(pieces: list) -> np.ndarray:
    """MATLAB ``cat(2, pieces{:})``: 0x0 pieces are ignored."""
    keep = [np.asarray(p) for p in pieces if np.asarray(p).shape != (0, 0)]
    if not keep:
        return np.zeros((0, 0))
    keep = [p.reshape(1, -1) if p.ndim < 2 else p for p in keep]
    return np.concatenate(keep, axis=1)


def _trim_roiinfo(roiinfo, act):
    if _isempty(roiinfo):
        return None
    if isinstance(roiinfo, dict):
        return roiinfo
    arr = np.asarray(roiinfo)
    if arr.dtype == object:  # struct array
        if arr.size > N_GRID_ROIS:  # MATLAB ROIinfo(1:818) keeps a row's orientation
            first = arr.ravel(order="F")[:N_GRID_ROIS]
            return first.reshape(-1, 1) if arr.ndim == 2 and arr.shape[1] == 1 else \
                first.reshape(1, -1)
        return roiinfo
    if arr.dtype.kind in "biuf":
        rows = cols = False
        if not _isempty(act):
            n = np.asarray(act).shape[0]
            if arr.shape[0] == n:
                rows = True
            elif arr.ndim > 1 and arr.shape[1] == n:
                cols = True
        if not rows and not cols:
            a2 = arr.reshape(-1, 1) if arr.ndim < 2 else arr
            if a2.shape[0] > 1 and a2.shape[1] == 1:
                rows = True
            elif (a2.shape[1] > 1 and a2.shape[0] == 1) or a2.shape[1] >= a2.shape[0]:
                cols = True
            else:
                rows = True
        if rows and arr.shape[0] > N_GRID_ROIS:
            return arr[:N_GRID_ROIS, ...]
        if cols and arr.ndim > 1 and arr.shape[1] > N_GRID_ROIS:
            return arr[:, :N_GRID_ROIS]
    return roiinfo


def wf_trim_stack_frames(proj_meta, frames_to_remove: int) -> np.ndarray:
    """Remove frames at the start and end of every stack (sub-experiment).

    Port of calliope ``21_Analysis/WF_trimStackFrames.m`` (its function line names it
    ``trim_proj_meta_frames``; MATLAB calls it by the file name). For every rd entry (MATLAB
    linear order): the sub-experiments are the cells of ``comments`` (one when ``comments``
    is a char array), their frame ranges come from ``nbr_frames``. From each one
    ``frames_to_remove`` frames are removed at both ends; a sub-experiment with
    ``<= 2 * frames_to_remove`` frames becomes empty. The time-series fields ``act,
    frame_times, PerturbationLFM, VisualFlow, VisualFlow_smoothed, Running,
    Running_smoothed, AirPuff_extended, StimulusID, vis_speed, ECTstim`` are sliced
    (columns) and concatenated again; ``act`` and ``ROIinfo`` are cut to the first 818
    ROIs; ``nbr_frames`` becomes the remaining frame counts.

    Parameters
    ----------
    proj_meta
        proj_meta struct array (module docstring). Not changed.
    frames_to_remove
        Frames removed at each end of every sub-experiment.

    Returns
    -------
    numpy.ndarray
        New proj_meta ``(1, nSites)``. As in MATLAB, a time-series field that was missing
        or empty is created: ``act`` as ``zeros(nROI, 0)``, the others as ``[]``
        (``(0, 0)`` arrays). ``nbr_frames`` is a float64 ``(1, nSub)`` array when
        ``comments`` is a cell array, else a scalar.

    Notes
    -----
    Differences from MATLAB: none intended. The unused local function ``slice_timeseries``
    of the MATLAB file is not ported. ``frame_times`` keep their (1-based) values.
    """
    n_rm = int(frames_to_remove)
    sites = _copy_proj_meta(proj_meta)
    for site in sites:
        if "rd" not in site:
            continue
        g = site["rd"]
        flat = list(g.ravel(order="F"))
        for rd in flat:
            original = dict(rd)
            roiinfo = _trim_roiinfo(original.get("ROIinfo"), original.get("act"))
            if roiinfo is not None:
                rd["ROIinfo"] = roiinfo
            elif "ROIinfo" in rd:
                rd["ROIinfo"] = np.zeros((0, 0))
            comments = original.get("comments")
            is_cell = isinstance(comments, np.ndarray) and comments.dtype == object
            n_sub = comments.size if is_cell else 1

            def empty_act(orig=original, ri=roiinfo):
                act = orig.get("act")
                if not _isempty(act):
                    return np.zeros((min(np.asarray(act).shape[0], N_GRID_ROIS), 0))
                n = N_GRID_ROIS
                if ri is not None:
                    n = min(np.asarray(ri).size if np.asarray(ri).dtype == object
                            else np.asarray(ri).shape[0], N_GRID_ROIS)
                return np.zeros((n, 0))

            segs = {f: [] for f in _TIME_SERIES_FIELDS}
            counts = []
            for sub in range(n_sub):
                s0, e0 = _subexp_frames(original, sub)
                valid = (e0 - s0 + 1) > 2 * n_rm
                s1, e1 = s0 + n_rm, e0 - n_rm
                counts.append(e1 - s1 + 1 if valid else 0)
                for f in _TIME_SERIES_FIELDS:
                    v = original.get(f)
                    if not _isempty(v):
                        if valid:
                            seg = _cols(v, s1, e1)
                            if f == "act" and seg.shape[0] > N_GRID_ROIS:
                                seg = seg[:N_GRID_ROIS, :]
                        elif f == "act":
                            seg = np.zeros((min(np.asarray(v).shape[0], N_GRID_ROIS), 0))
                        else:
                            seg = np.zeros((0, 0))
                    else:
                        seg = empty_act() if f == "act" else np.zeros((0, 0))
                    segs[f].append(seg)
            if is_cell:
                rd["nbr_frames"] = np.array(counts, dtype=np.float64).reshape(1, -1)
                for f in _TIME_SERIES_FIELDS:
                    rd[f] = _hcat(segs[f])
            else:
                rd["nbr_frames"] = np.float64(counts[0])
                for f in _TIME_SERIES_FIELDS:
                    rd[f] = segs[f][0]
    return _as_struct_array(sites)


# --- WF_velocityField -------------------------------------------------------------------------


def wf_velocity_field(
    proj_meta, roi_info, alpha: float = 0.005, max_iter: int = 100, sigma: float = 1.0
) -> np.ndarray:
    """Horn-Schunck velocity field of the ROI-grid activity of every site and timepoint.

    Port of calliope ``21_Analysis/WF_velocityField.m``. For each site, each timepoint
    ``rd[0, tp]`` and each activity field (``act``, or ``act_ch525``/``act_ch610``):
    :func:`wf_grid2vid` places the ROIs on the grid, :func:`velocity_field_horn_schunck`
    computes the flow, and :func:`uv_img_to_linear` takes it back to ``[nROI x nFrames]``.
    The results are stored as ``U``/``V`` (``U_ch525``, ``V_ch525``, ... for two channels).

    Parameters
    ----------
    proj_meta
        proj_meta struct array (module docstring). Not changed.
    roi_info
        ROI grid information (``ROIs_transformed`` of the 818-ROI grid template; MATLAB
        loaded it from the lab folder). See :mod:`calliopy.widefield._flow`.
    alpha, max_iter, sigma
        Smoothness weight, iterations and Gaussian sigma (pixels), MATLAB defaults.

    Returns
    -------
    numpy.ndarray
        New proj_meta ``(1, nSites)`` with ``U``/``V`` fields (float64 ``[nROI x nFrames]``,
        ROI-grid pixels per frame). When the sites have ``uuid``, the first one gets the
        suffix ``'-WF_velocityField_alpha=%.3f_max_iter=%i_sigma=%.1f'``.

    Notes
    -----
    Differences from MATLAB: ``roi_info`` is an argument; no temp file is read or saved.
    """
    sites = _copy_proj_meta(proj_meta)
    if not sites:
        return _as_struct_array(sites)
    act_names = ("act_ch525", "act_ch610") if "act_ch525" in _rd_linear(sites[0], 0) else (
        "act",)
    for k, site in enumerate(sites):
        logger.info("Now processing site %d of %d", k + 1, len(sites))
        g = site["rd"]
        for tp in range(g.shape[1]):
            rd = g[0, tp]
            for name in act_names:
                suffix = name[name.index("_"):] if "_" in name else ""
                vid = wf_grid2vid(rd[name], roi_info)
                U, V = velocity_field_horn_schunck(vid, alpha=alpha, max_iter=max_iter,
                                                   sigma=sigma)
                u_lin, v_lin = uv_img_to_linear(U, V, roi_info)
                rd["U" + suffix] = u_lin
                rd["V" + suffix] = v_lin
    if "uuid" in sites[0]:
        tag = f"alpha={alpha:.3f}_max_iter={int(max_iter)}_sigma={sigma:.1f}"
        sites[0]["uuid"] = f"{sites[0]['uuid']}-WF_velocityField_{tag}"
    return _as_struct_array(sites)


def plot_roi_velocity_field(
    U,
    V,
    scale_factor: float,
    roi_info,
    *,
    linewidth: float = 1.0,
    cmap: Any = "turbo",
    ax=None,
    imin: float | None = None,
    imax: float | None = None,
):
    """Velocity vectors of one frame drawn on the ROI grid.

    Port of calliope ``22_Plotting/WF_imageROIvelocityField.m``. The ROI grid
    (:func:`wf_grid2vid` of ``-1`` per ROI) is shown in grey on white; the arrows start at
    the ROI positions (MATLAB pixel coordinates, 1-based, ``x`` = column, ``y`` = row, y
    pointing down) and are coloured by their length.

    Parameters
    ----------
    U, V
        One value per ROI (vectors of length nROI): ``U`` positive to the right, ``V``
        positive from anterior to posterior (downwards).
    scale_factor
        Arrow length in grid pixels per unit of velocity (required, as in MATLAB).
    roi_info
        ROI grid information (MATLAB: ``ROIs_transformed`` of the template in the lab
        folder); its length must equal the length of ``U`` and ``V``.
    linewidth, cmap, ax
        Arrow line width, arrow colormap, axes to draw into (default: a new figure).
    imin, imax
        Colour limits of the arrow length (default: data range).

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Differences from MATLAB: ``quiverC2D`` (third-party, not ported) is replaced by
    :meth:`matplotlib.axes.Axes.quiver` coloured by vector length with
    ``scale_factor`` as the length scale (``quiverC2D``'s own automatic scaling is not
    reproduced, so arrow lengths differ). NaN velocities are drawn as zero-length arrows
    (quiverC2D sets NaN to 0). One axes is used instead of two overlaid ones.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if scale_factor is None:
        raise ValueError("provide scale_factor")
    u = np.asarray(U, dtype=np.float64)
    v = np.asarray(V, dtype=np.float64)
    if u.ndim != 1 and min(u.shape) != 1 or v.ndim != 1 and min(v.shape) != 1:
        raise ValueError("U and V must be vectors")
    u, v = u.ravel(), v.ravel()
    n_info = len(_sites(roi_info))
    if n_info != u.size or n_info != v.size:
        raise ValueError("Length of ROIinfo must match length of U and V")
    grid = wf_grid2vid(-np.ones((u.size, 1)), roi_info)[:, :, 0]
    u_img, v_img = uv_linear_to_img(u, v, roi_info)
    u_img, v_img = u_img[:, :, 0], v_img[:, :, 0]
    h, w = grid.shape
    X, Y = np.meshgrid(np.arange(1, w + 1), np.arange(1, h + 1))
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    a = 0.8
    ax.imshow(grid, cmap=ListedColormap([[a, a, a], [1, 1, 1]]), vmin=-1, vmax=0,
              extent=(0.5, w + 0.5, h + 0.5, 0.5), interpolation="nearest")
    uu = np.nan_to_num(u_img, nan=0.0, posinf=0.0, neginf=0.0)
    vv = np.nan_to_num(v_img, nan=0.0, posinf=0.0, neginf=0.0)
    mag = np.hypot(uu, vv)
    q = ax.quiver(X, Y, uu * scale_factor, vv * scale_factor, mag, angles="xy",
                  scale_units="xy", scale=1, cmap=cmap, linewidth=linewidth)
    if imin is not None or imax is not None:
        q.set_clim(imin if imin is not None else mag.min(),
                   imax if imax is not None else mag.max())
    fig.colorbar(q, ax=ax)
    bd = 4
    ax.set_xlim(-bd, w + bd)
    ax.set_ylim(h + bd, -bd)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig


# --- WF_commodore -----------------------------------------------------------------------------


def _avg_psd_in_range(signal, fs: float, freq_range, use_db: bool = False, axis: int = 0):
    """Adapter over :func:`calliopy.analysis.avg_psd_in_range` (``21_Analysis/AvgPsdInRange.m``).

    Keeps this module's conventions: ``axis`` (0-based) keeps length 1 also for 1-D input, and
    ``nan`` (a float) is returned for empty/too short input or an empty frequency range.
    """
    from calliopy.analysis import avg_psd_in_range

    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0 or x.shape[axis] < 2:
        return math.nan
    n = x.shape[axis]
    freqs = np.arange(n // 2 + 1) * (fs / n)
    if not np.any((freqs >= freq_range[0]) & (freqs <= freq_range[1])):
        warnings.warn(f"No frequencies found within the specified range {list(freq_range)} Hz",
                      stacklevel=2)
        return math.nan
    out = avg_psd_in_range(x, fs, freq_range, use_db_averaging=use_db, axis=axis)
    return np.reshape(out, x.shape[:axis] + (1,) + x.shape[axis + 1:])


def _msqueeze(a: np.ndarray) -> np.ndarray:
    """MATLAB ``squeeze``: drop singleton dimensions, keep at least 2 (column for a vector)."""
    a = np.asarray(a)
    shape = list(a.shape)
    while len(shape) > 2 and shape[-1] == 1:  # MATLAB has no trailing singleton dimensions
        shape.pop()
    if len(shape) <= 2:
        return a.reshape(shape, order="F")
    shape = [d for d in shape if d != 1]
    if len(shape) == 0:
        shape = [1, 1]
    elif len(shape) == 1:
        shape = [shape[0], 1]
    return a.reshape(shape, order="F")


def _mmean_omitnan(a: np.ndarray, axis: int) -> np.ndarray:
    """MATLAB ``mean(a, axis+1, 'omitnan')``; an axis beyond ``ndim`` leaves ``a`` as is."""
    a = np.asarray(a, dtype=np.float64)
    if axis >= a.ndim:
        return a
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis, keepdims=True)


def wf_commodore(
    proj_meta,
    *,
    site_id=None,
    tp_list=None,
    frame_rate: float = 100.0,
    frequency_range=(1.0, 2.0),
    analysis_window: float = 60.0,
    average_over_psd: bool = True,
    average_over_time: bool = False,
) -> list[np.ndarray]:
    """Mean power in a frequency band for every ROI and time window, averaged over sites.

    Computation of calliope ``21_Analysis/WF_commodore.m``. Per timepoint:

    1. For every site, the frames of every stack except 200 at each end
       (``frms = [1, cumsum(nbr_frames)]``, 1-based ``frms(k)+200 : frms(k+1)-200``) are
       concatenated; the common length is the minimum total length over sites that have
       data. ROIs 3:820 (1-based) are used when ``act`` has 824 rows, else ROIs 1:818.
    2. The data ``[818 x T x nSites]`` are cut into windows of
       ``round(analysis_window * frame_rate)`` frames (the incomplete last part is dropped).
    3. ``average_over_psd`` (default): the mean PSD in ``frequency_range``
       (``AvgPsdInRange``, linear averaging) per ROI, window and site, then the mean over
       sites (NaN ignored). Otherwise the activity is averaged over sites first.
    4. ``average_over_time``: the mean over windows (NaN ignored).

    Parameters
    ----------
    proj_meta
        proj_meta struct array (module docstring).
    site_id
        Keep only sites whose ``ExpGroup(1)`` is in this list (MATLAB ``'siteID'``).
    tp_list
        **0-based** timepoints (columns of ``rd``); default all of the first site.
    frame_rate, frequency_range, analysis_window, average_over_psd, average_over_time
        MATLAB ``'FrameRate'``, ``'FrequencyRange'`` (Hz), ``'AnalysisWindow'`` (s),
        ``'AverageOverPsd'``, ``'AverageOverTime'``.

    Returns
    -------
    list of numpy.ndarray
        One entry per timepoint (MATLAB ``max_amplitudes`` cell): ``[818 x nWindows]``
        float64 power (signal units squared per Hz); ``[818 x 1]`` with
        ``average_over_time``; all-NaN ``[818 x 1]`` when the window is longer than the
        data; ``(0, 0)`` when no site has data. An empty list when ``site_id`` matches no
        site (MATLAB: ``[]`` with a warning).

    Notes
    -----
    Suspected MATLAB bugs kept (they change numbers): with exactly one window and several
    sites, MATLAB's ``squeeze`` turns ``[818 x 1 x 1 x nSites]`` into ``[818 x nSites]`` and
    ``mean(..., 3)`` then averages nothing, so the result has one column **per site**, not
    per window. The per-site lengths are kept across timepoints, so a site skipped at a
    later timepoint still contributes its earlier length to the common length. A stack of
    exactly 401 frames adds 1 to the length but no data (MATLAB's ``s_idx < e_idx``).

    Differences from MATLAB: plotting (:func:`plot_wf_commodore`) is separate; the video
    output (``'SaveVideo'``) is not ported (it failed in MATLAB on the undefined
    ``global_min``). The Allen-template loading is replaced by the ``template`` argument of
    the plot function. A first timepoint at which no site has data gives the "no valid
    data" result instead of MATLAB's undefined-variable error.
    """
    sites = _sites(proj_meta)
    if site_id is not None and np.size(site_id):
        sites = _filter_sites(sites, site_id)
        if not sites:
            return []
    if tp_list is None:
        tp_list = list(range(_n_tp(sites[0])))
    tps = [int(t) for t in np.atleast_1d(tp_list).ravel()] if not isinstance(
        tp_list, list) else [int(np.asarray(t).ravel()[0]) for t in tp_list]
    ns = len(sites)
    lengths = np.zeros(ns, dtype=np.int64)
    results: list[np.ndarray] = []
    for tp in tps:
        for s, site in enumerate(sites):
            g = _rd_grid(site)
            if g.size == 0 or g.size < tp + 1:
                continue
            frms = np.concatenate([[1], np.cumsum(_vec(_rd_row(site, tp)["nbr_frames"]))])
            total = 0
            for k in range(frms.size - 1):
                seg = (frms[k + 1] - 200) - (frms[k] + 200) + 1
                if seg > 0:
                    total += int(seg)
            lengths[s] = total
        valid = lengths[lengths > 0]
        if valid.size == 0:
            warnings.warn(f"No valid data found for Timepoint {tp}. Skipping.", stacklevel=2)
            results.append(np.zeros((0, 0)))
            continue
        common = int(valid.min())
        m = np.full((N_GRID_ROIS, common, ns), np.nan)
        for s, site in enumerate(sites):
            act = np.asarray(_rd_linear(site, tp)["act"], dtype=np.float64)
            rows = np.arange(2, 820) if act.shape[0] == 824 else np.arange(N_GRID_ROIS)
            frms = np.concatenate([[1], np.cumsum(_vec(_rd_row(site, tp)["nbr_frames"]))])
            cols = []
            for k in range(frms.size - 1):
                a, b = int(frms[k] + 200), int(frms[k + 1] - 200)
                if a < b <= act.shape[1]:
                    cols.extend(range(a - 1, b))
            take = min(len(cols), common)
            if take > 0:
                if act.shape[0] < rows[-1] + 1:
                    raise IndexError(f"act has {act.shape[0]} rows; 818 ROIs are needed")
                m[:, :take, s] = act[np.ix_(rows, np.asarray(cols[:take]))]
        wf = int(mround(analysis_window * frame_rate))
        if wf > common:
            warnings.warn(
                f"Analysis window ({wf} frames) is larger than recording length "
                f"({common} frames). Skipping TP.", stacklevel=2)
            results.append(np.full((N_GRID_ROIS, 1), np.nan))
            continue
        nw = common // wf
        m4 = m[:, : nw * wf, :].reshape((N_GRID_ROIS, wf, nw, ns), order="F")
        if average_over_psd:
            psd = _avg_psd_in_range(m4, frame_rate, frequency_range, False, axis=1)
            if np.ndim(psd) == 0:
                amp = np.full((1, 1), psd)
            else:
                amp = _mmean_omitnan(_msqueeze(psd), axis=2)
                amp = amp.reshape(amp.shape[:2], order="F") if amp.ndim == 3 else amp
        else:
            avg = _mmean_omitnan(m4, axis=3)
            avg = avg.reshape(avg.shape[:3], order="F") if avg.ndim == 4 else avg
            psd = _avg_psd_in_range(avg, frame_rate, frequency_range, False, axis=1)
            amp = np.full((1, 1), psd) if np.ndim(psd) == 0 else _msqueeze(psd)
        results.append(np.asarray(amp, dtype=np.float64))
    if average_over_time:
        out = []
        for r in results:
            if r.size == 0:
                out.append(np.zeros((r.shape[0], 1)))
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    out.append(np.nanmean(r, axis=1, keepdims=True))
        results = out
    return results


def _window_for_subplot(s: int, n_avail: int, n_cols: int) -> int:
    """1-based window shown in column ``s`` (0-based) of the WF_commodore figure."""
    if s == 0:
        return 1
    usable = max(1, n_avail - 1)
    spacing = (usable - 1) / (n_cols - 1)
    return int(min(1 + mround(s * spacing), usable))


def _roi_range(n_rois: int) -> np.ndarray:
    """0-based brain ROIs: 2..819 for 824 ROIs, else 0..817 (MATLAB 3:820 / 1:818)."""
    return np.arange(2, 820) if n_rois == 824 else np.arange(N_GRID_ROIS)


def _value_image(values, template, background=np.nan) -> np.ndarray:
    """Values of the first ``min(nROI, len(values))`` template ROIs painted into the atlas
    image (NaN values skipped), flipped left-right (MATLAB ``flip(img, 2)``)."""
    vals = np.asarray(values, dtype=np.float64).ravel()
    img = np.full(template.image_shape, background, dtype=np.float64)
    for r in range(min(template.n_roi, vals.size)):
        if not np.isnan(vals[r]):
            rr, cc = np.unravel_index(template.indices[r] - 1, template.image_shape, order="F")
            img[rr, cc] = vals[r]
    return img[:, ::-1]


def _draw_roi_map(ax, values, *, template=None, roi_info=None, clim, cmap, border_color="w",
                  show_borders=True, background=np.nan, marker_size=2):
    """Atlas image of ROI values with Allen borders, or a griddata map when no template."""
    if template is not None:
        img = _value_image(values, template, background)
        im = ax.imshow(img, cmap=cmap, vmin=clim[0], vmax=clim[1], interpolation="nearest")
        ax.set_facecolor("k")
        if show_borders:
            rows, cols = np.nonzero(template.area_borders[:, ::-1])
            ax.scatter(cols, rows, s=marker_size, c=border_color, marker=".", linewidths=0)
    else:
        if roi_info is None:
            raise ValueError("give template (GridTemplate) or roi_info")
        import scipy.interpolate

        rr = _roi_range(len(_sites(roi_info)))
        ml = _concat_field(roi_info, "ML_index")[rr]
        ap = _concat_field(roi_info, "AP_index")[rr]
        vals = np.asarray(values, dtype=np.float64).ravel()[: rr.size]
        gx, gy = np.unique(ml), np.unique(ap)
        GX, GY = np.meshgrid(gx, gy)
        ok = np.isfinite(vals)
        img = scipy.interpolate.griddata((ml[ok], ap[ok]), vals[ok], (GX, GY), method="linear")
        im = ax.imshow(img, cmap=cmap, vmin=clim[0], vmax=clim[1], origin="lower",
                       extent=(gx[0] - 0.5, gx[-1] + 0.5, gy[0] - 0.5, gy[-1] + 0.5))
    ax.set_axis_off()
    ax.set_aspect("equal")
    return im


def plot_wf_commodore(
    max_amplitudes: list,
    tp_list=None,
    *,
    template=None,
    roi_info=None,
    frequency_range=(1.0, 2.0),
    analysis_window: float = 60.0,
    average_over_time: bool = False,
    number_of_windows: int = 5,
    color_lim=None,
    show_atlas_borders: bool = True,
    border_color: Any = "w",
):
    """Maps of the band power of :func:`wf_commodore` (static figure of ``WF_commodore``).

    Rows are timepoints, columns windows (first window, then windows spread evenly over the
    complete ones); with ``average_over_time`` one map per timepoint in a single row. Colour
    limits are shared (``color_lim`` or the overall min/max), colormap magma.

    Parameters
    ----------
    max_amplitudes
        Output of :func:`wf_commodore`.
    tp_list
        **0-based** timepoints used (titles show them 1-based, as MATLAB did); default
        ``0..len-1``.
    template
        :class:`GridTemplate` (``818_ROIs_grid_template.mat``): values are painted into the
        atlas image and the Allen borders are drawn. Without it, ``roi_info`` (ROIinfo with
        ``ML_index``/``AP_index``) is required and the values are interpolated on the ML/AP
        grid.
    frequency_range, analysis_window, average_over_time, number_of_windows, color_lim,
    show_atlas_borders, border_color
        As MATLAB ``'FrequencyRange'``, ``'AnalysisWindow'``, ``'AverageOverTime'``,
        ``'NumberOfWindows'``, ``'ColorLim'``, ``'ShowAtlasBorders'``, ``'BorderColor'``.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Differences from MATLAB: the template is an argument (MATLAB loaded it from the lab
    folder); the fallback interpolation is linear (``scipy.interpolate.griddata``) instead
    of MATLAB's ``'natural'`` neighbour, which scipy does not provide.
    """
    import matplotlib.pyplot as plt

    n_tp = len(max_amplitudes)
    tps = list(range(n_tp)) if tp_list is None else [int(t) for t in np.ravel(tp_list)]
    if color_lim is not None:
        vmin, vmax = float(color_lim[0]), float(color_lim[1])
    else:
        vals = [np.asarray(r, dtype=float) for r in max_amplitudes if np.size(r)]
        finite = np.concatenate([v[np.isfinite(v)] for v in vals]) if vals else np.zeros(0)
        vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    n_cols = n_tp if average_over_time else int(number_of_windows)
    n_rows = 1 if average_over_time else n_tp
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                             figsize=(3 * n_cols, 3 * n_rows if not average_over_time else 3))
    im = None
    for t in range(n_tp):
        data = np.asarray(max_amplitudes[t], dtype=float)
        data2 = data.reshape(data.shape[0], -1) if data.ndim else data.reshape(1, 1)
        n_av = data2.shape[1] if data.size else 0
        loops = 1 if average_over_time else n_cols
        for s in range(loops):
            ax = axes[0, t] if average_over_time else axes[t, s]
            w = 1 if average_over_time else _window_for_subplot(s, n_av, n_cols)
            win = data2[:, w - 1] if w <= n_av else np.full(N_GRID_ROIS, np.nan)
            im = _draw_roi_map(ax, win, template=template, roi_info=roi_info,
                               clim=(vmin, vmax), cmap="magma", border_color=border_color,
                               show_borders=show_atlas_borders)
            if average_over_time:
                ax.set_title(f"TP {tps[t] + 1} (Avg {frequency_range[0]:g}-"
                             f"{frequency_range[1]:g} Hz)\nEntire Rec", fontsize=12)
            else:
                ax.set_title(f"TP {tps[t] + 1} | Win {w}\n{(w - 1) * analysis_window:.0f}-"
                             f"{w * analysis_window:.0f} s", fontsize=12)
    if im is not None:
        cb = fig.colorbar(im, ax=axes.ravel().tolist())
        cb.set_label(f"Mean PSD ({frequency_range[0]:g}-{frequency_range[1]:g} Hz)",
                     fontsize=14, fontweight="bold")
    return fig
