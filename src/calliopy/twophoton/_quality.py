"""Quality checks and inspection views of two-photon adata.

Ports of the computation in calliope ``11_TwoPhoton/tpi_adata_check.m``,
``tpi_alignment_visualize.m``, ``04_Calliope/check_siteROIs.m``,
``check_registration_widefield.m`` and ``kalliope_callback_list_failed_registration.m``.
ExpLog lookups, database updates, prompts and saving are left to the caller.

ROI ``indices`` are MATLAB linear indices, **1-based, column-major**.

None of these functions has been compared with MATLAB output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform

from ..io import MatStruct
from ..widefield import warp_image
from ._activity import _as_items, _indices, roi_traces

__all__ = [
    "TPI_ACT_THRESHOLD",
    "TPI_ROI_THRESHOLD",
    "align_site_first_frames",
    "align_site_first_frames_2ch",
    "plot_site_rois",
    "plot_tpi_activity_similarity",
    "registration_failed",
    "roi_perimeter_image",
    "roi_positions",
    "roi_similarity",
    "site_grid_shape",
    "site_roi_label_image",
    "tpi_act_similarity",
    "tpi_adata_check",
    "tpi_adata_incomplete",
    "tpi_inspection_flags",
    "tpi_mean_roi_traces",
]

TPI_ACT_THRESHOLD = 0.9999
TPI_ROI_THRESHOLD = 0.98


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _empty(x: Any) -> bool:
    return x is None or (hasattr(x, "size") and np.asarray(x).size == 0) or (
        isinstance(x, (list, tuple)) and len(x) == 0)


def tpi_adata_incomplete(adata: Any) -> str | None:
    """Reason why ``tpi_adata_check.m`` would skip this adata, or None when complete.

    Checks, in MATLAB's order: ``ROIs`` present, ``activity`` in the first layer's ROIs,
    ``template{1}`` and ``act_map{1}`` non-empty and not all zero, ``np{1}`` with fields,
    then ``projections.max{1}`` non-empty and not all zero. The first failing check is
    returned.
    """
    rois = _get(adata, "ROIs")
    if rois is None:
        return "Found no ROIs in adata file"
    first = _as_items(rois[0])
    if not first or not all("activity" in s for s in first):
        return "Activity not calculated yet"
    for name in ("template", "act_map"):
        v = _get(adata, name)
        if v is None or _empty(v[0]) or np.nanmax(np.asarray(v[0])) == 0:
            return f"{name} does not exist or is empty"
    npl = _get(adata, "np")
    np1 = _as_items(npl[0]) if npl is not None else []
    if not np1 or not any(len(s) for s in np1):
        return "np does not exist or is empty"
    proj = _get(adata, "projections")
    if proj is None or _empty(_get(proj, "max")) or np.nanmax(np.asarray(_get(proj, "max")[0])) == 0:
        return "Projections do not exist or are empty"
    return None


def roi_positions(rois: Sequence[Any], template_shapes: Sequence[Sequence[int]]) -> np.ndarray:
    """ROI centroids as ``tpi_adata_check.m`` collects them.

    For each layer and ROI ``k`` (0-based), column ``k`` gets the mean (row, column) of the
    ROI's pixels, **1-based** (MATLAB ``ind2sub``). As in MATLAB, all layers write into the
    same columns, so column ``k`` holds the ROI ``k`` of the *last* layer that has more
    than ``k`` ROIs.

    Returns
    -------
    numpy.ndarray
        ``[2 x maxROIs]`` (row 0: mean row, row 1: mean column); an ROI without indices
        gives NaN.
    """
    cols: dict[int, tuple[float, float]] = {}
    for layer, shape in zip(rois, template_shapes, strict=True):
        for k, s in enumerate(_as_items(layer)):
            ix = _indices(s)
            if ix.size == 0:
                cols[k] = (np.nan, np.nan)
                continue
            r, c = np.unravel_index(ix - 1, tuple(int(v) for v in shape[:2]), order="F")
            cols[k] = (r.mean() + 1, c.mean() + 1)
    n = max(cols) + 1 if cols else 0
    out = np.full((2, n), np.nan)
    for k, v in cols.items():
        out[:, k] = v
    return out


def roi_similarity(site_rois: Sequence[Any], site_template_shapes: Sequence[Sequence[int]],
                   rois: Sequence[Any], template_shapes: Sequence[Sequence[int]]) -> float:
    """ROI-layout similarity of an experiment to its site (``tpi_adata_check.m``).

    The pairwise-distance matrices of the ROI centroids (:func:`roi_positions`) of the site
    and of the experiment are compared column by column with the correlation coefficient;
    the result is the minimum over columns.

    Raises
    ------
    ValueError
        The two experiments have different numbers of ROI columns (MATLAB failed in
        ``corrcoef``).

    Notes
    -----
    Kept from MATLAB (suspected bug): the layers overwrite each other's ROI positions (see
    :func:`roi_positions`), so for multi-layer data mostly the last layer is compared.
    Differences from MATLAB: positions and correlations are not carried over from the
    previously checked ExpID (MATLAB did not clear ``ROIpos`` and ``dist`` between ExpIDs).
    """
    p1 = roi_positions(site_rois, site_template_shapes)
    p2 = roi_positions(rois, template_shapes)
    if p1.shape != p2.shape:
        raise ValueError(f"ROI counts differ: site {p1.shape[1]}, experiment {p2.shape[1]}")
    d1 = squareform(pdist(p1.T))
    d2 = squareform(pdist(p2.T))
    vals = [np.corrcoef(d1[:, k], d2[:, k])[0, 1] for k in range(d1.shape[1])]
    return float(np.min(vals)) if vals else float("nan")


def tpi_mean_roi_traces(data: Sequence[Any], rois: Sequence[Any]) -> list:
    """Per layer, the mean over ROIs of the ROI traces recomputed from the raw data.

    As ``tpi_adata_check.m``: each ROI trace is the single-precision mean of its pixels,
    then ``nanmean`` over ROIs. Returns float32 ``[nFrames x 1]`` arrays (computed in
    float64, rounded to float32: last-bit differences to MATLAB possible).
    """
    out = []
    for d, layer in zip(data, rois, strict=True):
        tr = roi_traces(d, [_indices(s) for s in _as_items(layer)]).astype(np.float64)
        with np.errstate(all="ignore"):
            out.append(np.nanmean(tr, axis=1).astype(np.float32)[:, None])
    return out


def _stored_mean(layer) -> np.ndarray:
    acts = [np.asarray(s["activity"], dtype=np.float64).ravel() for s in _as_items(layer)]
    with np.errstate(all="ignore"):
        return np.nanmean(np.stack(acts, axis=1), axis=1)


def tpi_act_similarity(rois: Sequence[Any], mean_traces: Sequence[Any]) -> np.ndarray:
    """Correlation of the stored and the recomputed mean ROI activity, per layer.

    Returns
    -------
    numpy.ndarray
        ``(1, nLayers)`` (MATLAB ``act_similarity``).
    """
    vals = []
    for layer, mt in zip(rois, mean_traces, strict=True):
        vals.append(np.corrcoef(_stored_mean(layer), np.asarray(mt, dtype=np.float64).ravel())[0, 1])
    return np.asarray(vals, dtype=np.float64)[None, :]


def tpi_adata_check(rois: Sequence[Any], data: Sequence[Any], *,
                    template_shapes: Sequence[Sequence[int]] | None = None,
                    site_rois: Sequence[Any] | None = None,
                    site_template_shapes: Sequence[Sequence[int]] | None = None,
                    is_site: bool = False) -> MatStruct:
    """Check stored ROI activity and ROI alignment of a two-photon adata file.

    Port of the computation in ``11_TwoPhoton/tpi_adata_check.m`` for one ExpID, in the
    style of :func:`calliopy.widefield.wid_adata_check`.

    Parameters
    ----------
    rois
        ``adata.ROIs``: one ROI list per layer, with ``indices`` and ``activity``.
    data
        Registered stacks per layer (``load_exp(ExpID, {'525.bin'})``).
    template_shapes
        ``size(adata.template{layer})`` per layer; default the frame size of ``data``.
    site_rois, site_template_shapes
        The same for the site's adata (needed unless ``is_site``).
    is_site
        The experiment is its own site (``roi_similarity = 1``).

    Returns
    -------
    MatStruct
        ``check_raw`` with ``passed`` (1.0/0.0), ``roi_similarity`` and ``act_similarity``
        (``(1, nLayers)``). Pass rule ``min(act) > 0.9999 and roi > 0.98``.

    Notes
    -----
    Differences from MATLAB: the "already checked" and completeness skips
    (:func:`tpi_adata_incomplete`), the ExpLog/database update and the save are the
    caller's.
    """
    if template_shapes is None:
        template_shapes = [np.asarray(d).shape[:2] for d in data]
    if is_site:
        roi_sim = 1.0
    else:
        if site_rois is None or site_template_shapes is None:
            raise ValueError("site_rois and site_template_shapes are required unless is_site")
        roi_sim = roi_similarity(site_rois, site_template_shapes, rois, template_shapes)
    act = tpi_act_similarity(rois, tpi_mean_roi_traces(data, rois))
    passed = bool(np.min(act) > TPI_ACT_THRESHOLD and roi_sim > TPI_ROI_THRESHOLD)
    return MatStruct(passed=np.float64(passed), roi_similarity=np.float64(roi_sim),
                     act_similarity=act)


def tpi_inspection_flags(check_raw: Any) -> tuple[bool, bool]:
    """What ``tpi_alignment_visualize.m`` shows for a failed check.

    Returns ``(show_site_rois, show_activity)``. As in MATLAB, the site ROI views
    (``check_siteROIs``) are shown when the *activity* similarity is below 0.9, and the
    activity comparison when the *ROI* similarity is below 0.98 (the flags look swapped in
    MATLAB; kept and reported).
    """
    act = np.asarray(_get(check_raw, "act_similarity"), dtype=np.float64)
    roi = float(np.asarray(_get(check_raw, "roi_similarity")))
    return bool(np.min(act) < 0.9), bool(roi < TPI_ROI_THRESHOLD)


def plot_tpi_activity_similarity(rois: Sequence[Any], mean_traces: Sequence[Any]) -> Figure:
    """Stored vs recomputed mean ROI activity per layer (``tpi_alignment_visualize.m``)."""
    act = tpi_act_similarity(rois, mean_traces)[0]
    nl = len(rois)
    fig = Figure(figsize=(8, 2 * nl))
    for i in range(nl):
        ax = fig.add_subplot(nl, 1, i + 1)
        ax.plot(_stored_mean(rois[i]))
        ax.plot(np.asarray(mean_traces[i]).ravel())
        ax.set_title(f"Correlation coefficient: {act[i]:.4g}")
    return fig


def registration_failed(adata: Any) -> bool:
    """True when any ``registration_parameters*`` field has ``status == 'FAILED'``.

    Port of the per-ExpID test in ``04_Calliope/kalliope_callback_list_failed_registration.m``
    (the loop over ExpIDs and the adata lookup are the caller's; MATLAB returns 0 for an
    ExpID whose adata cannot be opened).
    """
    keys = adata.keys() if isinstance(adata, Mapping) else [k for k in dir(adata) if not k.startswith("_")]
    for k in keys:
        if str(k).startswith("registration_parameters"):
            st = _get(_get(adata, k), "status")
            if isinstance(st, str) and st == "FAILED":
                return True
    return False


def align_site_first_frames(site_first_frame: Any, first_frames: Sequence[Any],
                            tforms: Sequence[Any]) -> np.ndarray:
    """Stack of a site's first frame and the other experiments' registered first frames.

    Computation of ``04_Calliope/check_registration_widefield.m`` (single channel): each
    experiment's ``first_frame`` is warped with its ``tform`` (``tform.A``, 3x3) onto its own
    frame size (:func:`calliopy.widefield.warp_image`); an empty ``tform`` keeps the frame;
    a missing experiment (``None``) gives a black frame of the site's size.

    Returns
    -------
    numpy.ndarray
        float64 ``[h x w x (1 + nExps)]``, the site's frame first.
    """
    site = np.asarray(site_first_frame, dtype=np.float64)
    out = [site]
    for ff, tf in zip(first_frames, tforms, strict=True):
        if ff is None:
            out.append(np.zeros(site.shape[:2]))
            continue
        f = np.asarray(ff, dtype=np.float64)
        if tf is None or np.asarray(tf).size == 0:
            out.append(f)
        else:
            out.append(warp_image(f, np.asarray(tf, dtype=np.float64), f.shape))
    return np.stack(out, axis=2)


def align_site_first_frames_2ch(site: Mapping[str, Any], exps: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Two-channel version of :func:`align_site_first_frames`.

    ``site`` holds ``first_frame_ch525``, ``first_frame_ch610`` and ``tform_ch_map``; each
    entry of ``exps`` holds ``first_frame_ch525/610`` and ``tform_ch525/610``.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``[h x w x nFrames x 2]`` aligned frames, and the site's ch610 frame flipped left-right
        and warped with ``tform_ch_map`` (compare it with the ch525 frame, e.g. with
        :func:`calliopy.widefield.compare_wid_frames`).
    """
    f525 = np.asarray(site["first_frame_ch525"], dtype=np.float64)
    f610 = np.asarray(site["first_frame_ch610"], dtype=np.float64)
    planes = [np.stack([f525, f610], axis=2)]
    for e in exps:
        a = warp_image(np.asarray(e["first_frame_ch525"], dtype=np.float64),
                       np.asarray(e["tform_ch525"], dtype=np.float64), f525.shape)
        b = warp_image(np.asarray(e["first_frame_ch610"], dtype=np.float64),
                       np.asarray(e["tform_ch610"], dtype=np.float64), f610.shape)
        planes.append(np.stack([a, b], axis=2))
    stack = np.stack(planes, axis=2)
    mapped = warp_image(f610[:, ::-1], np.asarray(site["tform_ch_map"], dtype=np.float64), f610.shape)
    return stack, mapped


def site_grid_shape(n_exps: int) -> tuple[int, int]:
    """Rows and columns of the ``check_siteROIs.m`` panel grid."""
    rows = 1 if n_exps < 3 else (2 if n_exps < 9 else 3)
    return rows, int(np.ceil(n_exps / rows))


def site_roi_label_image(rois: Any, shape: Sequence[int]) -> np.ndarray:
    """Label image of a layer's ROIs, ROI ``k`` (0-based) drawn with value ``k + 1``.

    ``show_figs == 2`` panel of ``check_siteROIs.m``; later ROIs overwrite earlier ones.
    """
    out = np.zeros(int(shape[0]) * int(shape[1]))
    for k, s in enumerate(_as_items(rois)):
        ix = _indices(s)
        out[ix - 1] = k + 1
    return out.reshape((int(shape[0]), int(shape[1])), order="F")


def roi_perimeter_image(rois: Any, image: Any, color=(1.0, 0.0, 0.0)) -> np.ndarray:
    """RGB image of ``image`` (normalised to 0..1) with the ROI outlines in ``color``.

    Stands in for ``ROIs2image(..., 'type', 'perim', 'singlecolor', [1 0 0])`` used by
    ``check_siteROIs.m`` (``ROIs2image`` belongs to the RoiQuest track; the outline here is
    the ROI pixels with a 4-neighbour outside the ROI, as ``bwperim``; unverified against
    ``ROIs2image``).
    """
    img = np.asarray(image, dtype=np.float64)
    lo, hi = np.nanmin(img), np.nanmax(img)
    g = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
    rgb = np.stack([g, g, g], axis=2)
    h, w = img.shape
    edge = np.zeros((h, w), dtype=bool)
    st = ndimage.generate_binary_structure(2, 1)
    for s in _as_items(rois):
        ix = _indices(s)
        if ix.size == 0:
            continue
        m = np.zeros(h * w, dtype=bool)
        m[ix - 1] = True
        m = m.reshape((h, w), order="F")
        edge |= m & ~ndimage.binary_erosion(m, st, border_value=0)
    rgb[edge] = color
    return rgb


def plot_site_rois(images: Sequence[Any], rois: Sequence[Any], *, labels: Sequence[str] = (),
                   title: str = "") -> Figure:
    """Panels of one layer across a site's experiments with the ROI outlines (``check_siteROIs.m``).

    Parameters
    ----------
    images
        Per experiment the background image (``template{layer}`` or ``act_map{layer}``).
    rois
        Per experiment the ROI list of that layer.
    labels
        Per experiment a label (MATLAB prints the stack date and ExpID).
    """
    n = len(images)
    rows, cols = site_grid_shape(n)
    fig = Figure(figsize=(3 * cols, 3 * rows), facecolor="k")
    for i in range(n):
        r, c = i // cols, i % cols
        ax = fig.add_axes((c / cols, 1 - (r + 1) / rows, 1 / cols, 1 / rows))
        ax.imshow(roi_perimeter_image(rois[i], images[i]), aspect="auto")
        if i < len(labels):
            ax.text(0.1, 0.95, labels[i], transform=ax.transAxes, color="w", fontweight="bold",
                    va="top")
        if i == 0 and title:
            ax.text(0.1, 0.05, title, transform=ax.transAxes, color="w", fontweight="bold")
        ax.axis("off")
    return fig
