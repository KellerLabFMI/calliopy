"""Inspection of a failed raw-data check and the pass/no-change decision: calliope
``11_TwoPhoton/tpi_alignment_visualize.m`` and ``12_Widefield/wid_alignment_visualize.m``.

- :func:`tpi_alignment_figures` / :func:`wid_alignment_figures` build what MATLAB shows;
- :class:`AlignmentDecision` holds those figures and applies the answer;
- :func:`set_check_raw_passed` is the adata write MATLAB does on "y".

The tkinter window is in ``_tk_alignment.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from calliopy.io import Adata, MatStruct, load_mat, update_mat
from calliopy.twophoton import (
    plot_tpi_activity_similarity,
    tpi_inspection_flags,
    tpi_mean_roi_traces,
)
from calliopy.widefield import (
    compare_wid_frames,
    image_similarity_snippets,
    plot_image_similarity_snippets,
    warp_image,
)

from ._common import num2str, tform_from

__all__ = [
    "AlignmentDecision",
    "set_check_raw_passed",
    "tpi_alignment_figures",
    "wid_alignment_figures",
]

log = logging.getLogger("calliopy.viewers")

SEP = "-" * 50


def set_check_raw_passed(adata_path: str | PathLike) -> MatStruct:
    """Set ``check_raw.passed = 1`` in an adata file (the "y" answer of the MATLAB tools).

    MATLAB: ``check_raw = adata.check_raw; check_raw.passed = 1; save(file, 'check_raw',
    '-append')``. Here the stored struct is read (1x1 values kept 1x1), ``passed`` set to
    1.0 (1x1 double, as MATLAB) and written back with :func:`calliopy.io.update_mat`; every
    other field keeps its value, class and position, so the variable has MATLAB's layout. A
    ``check_raw`` holding MATLAB objects (string, datetime) cannot be written back: the
    write then fails with ``TypeError`` and the file is left unchanged.

    Returns
    -------
    MatStruct
        The ``check_raw`` written.

    Raises
    ------
    KeyError
        The file has no ``check_raw``.

    Notes
    -----
    The RawDataStatus / ExpLog update that goes with a passed check is not part of calliopy
    and is not done here.
    """
    path = Path(adata_path)
    data = load_mat(path, "check_raw", squeeze_scalars=False)
    check_raw = data["check_raw"]
    if not isinstance(check_raw, dict):
        raise TypeError("check_raw in the adata file is not a struct")
    check_raw = MatStruct(check_raw)
    check_raw.passed = 1.0
    update_mat(path, {"check_raw": check_raw})
    log.info("Updated check_raw and saved to adata")
    return check_raw


def _summary(check_raw: Mapping) -> list[str]:
    """MATLAB ``disp(adata.check_raw)``: one ``name: value`` line per field."""
    out = []
    for k, v in check_raw.items():
        a = np.asarray(v) if not isinstance(v, str) else None
        if isinstance(v, str):
            txt = v
        elif a is not None and a.dtype != object and a.size <= 10:
            txt = num2str(a)
        else:
            txt = f"[{'x'.join(str(n) for n in np.shape(v))} {type(v).__name__}]"
        out.append(f"{k}: {txt}")
    return out


class AlignmentDecision:
    """Figures of a failed check and the y/n decision (``*_alignment_visualize.m``).

    Parameters
    ----------
    adata_path
        Adata file whose ``check_raw.passed`` is set on "yes".
    figures
        ``(title, Figure)`` pairs to show.
    text
        Lines shown above the question (MATLAB prints them to the Command Window).
    exp_id
        For the question (MATLAB ``'<ExpID> - Do you want to set the status to passed?'``).

    Attributes
    ----------
    update : int or None
        1 after "yes" (file written), 0 after "no" or closing, None before (MATLAB's
        output ``update``).
    """

    def __init__(self, adata_path: str | PathLike, figures: Sequence[tuple[str, Figure]],
                 text: Sequence[str] = (), exp_id: int | None = None) -> None:
        self.adata_path = Path(adata_path)
        self.figures = list(figures)
        self.text = list(text)
        self.exp_id = exp_id
        self.update: int | None = None

    def question(self) -> str:
        head = str(self.exp_id) if self.exp_id is not None else self.adata_path.name
        return f"{head} - Do you want to set the status to passed?"

    def decide(self, passed: bool) -> int:
        """Apply the answer: ``True`` writes ``check_raw.passed = 1``; returns ``update``."""
        if passed:
            set_check_raw_passed(self.adata_path)
            self.update = 1
        else:
            log.info("No change made")
            self.update = 0
        return self.update


# ---------------------------------------------------------------------------------------
# tpi_alignment_visualize.m
# ---------------------------------------------------------------------------------------
def tpi_alignment_figures(adata_path: str | PathLike, *, data: Sequence[Any] | None = None,
                          site_figures: Sequence[Figure] | None = None) -> AlignmentDecision:
    """What ``tpi_alignment_visualize.m`` shows for a two-photon experiment.

    Reads ``check_raw`` (and ``ROIs``) from the adata file and decides with
    :func:`calliopy.twophoton.tpi_inspection_flags`:

    - minimum ``act_similarity`` below 0.9: the site's ROI views (MATLAB runs
      ``check_siteROIs`` unless its figures are open); pass them as ``site_figures``
      (e.g. from :func:`calliopy.twophoton.plot_site_rois`);
    - ``roi_similarity`` below 0.98: the stored mean ROI activity against the mean
      recomputed from the raw data, per layer, with their correlation
      (:func:`calliopy.twophoton.plot_tpi_activity_similarity`); needs ``data``.

    Parameters
    ----------
    adata_path
        Two-photon adata file with ``check_raw`` and ``ROIs``.
    data
        Raw data per layer, ``(rows, cols, frames)`` arrays (MATLAB ``load_exp(ExpID,
        {'525.bin'})``), e.g. from :func:`calliopy.io.load_bin`.
    site_figures
        Figures of the site's ROIs.

    Returns
    -------
    AlignmentDecision
        With the figures and the ``check_raw`` lines.

    Notes
    -----
    Kept from MATLAB (reported earlier in ``tpi_inspection_flags``): the site ROIs are
    shown on low *activity* similarity and the activity on low *ROI* similarity.

    Differences from MATLAB
    -----------------------
    Finding the adata file, the site's experiments and the raw data by ExpID is left to the
    caller (ExpLog); when the inputs a view needs are not given, it is skipped with a log
    message.
    """
    ad = Adata(adata_path)
    if "check_raw" not in ad:
        raise ValueError("This file is not checked yet - check first")
    check_raw = ad["check_raw"]
    text = [SEP, f"Inspecting ExpID: {ad.exp_id if ad.exp_id is not None else ad.path.name}", SEP]
    text += _summary(check_raw)
    show_site, show_act = tpi_inspection_flags(check_raw)
    if show_site:
        text.append("This ExpID failed activity similarity")
    if show_act:
        text.append("This ExpID failed ROI similarity")
    text.append(SEP)
    for line in text:
        log.info(line)
    figures: list[tuple[str, Figure]] = []
    if show_site:
        if site_figures:
            figures += [(f"site ROIs {i + 1}", f) for i, f in enumerate(site_figures)]
        else:
            log.info("site ROI figures not given (site_figures=); skipped")
    if show_act:
        if data is None:
            log.info("raw data not given (data=); activity comparison skipped")
        else:
            from calliopy.projmeta._struct import as_cell_list

            rois = as_cell_list(ad["ROIs"])
            log.info("Done loading all the .bin files for this ExpID, computing corrcoef...")
            traces = tpi_mean_roi_traces(list(data), rois)
            figures.append(("activity similarity", plot_tpi_activity_similarity(rois, traces)))
    return AlignmentDecision(adata_path, figures, text, ad.exp_id)


# ---------------------------------------------------------------------------------------
# wid_alignment_visualize.m
# ---------------------------------------------------------------------------------------
def wid_alignment_figures(adata_path: str | PathLike, site_adata_path: str | PathLike | None = None, *,
                          exp_id: int | None = None, site_id: int | None = None,
                          tforms: Mapping[str, Any] | None = None,
                          rng: np.random.Generator | None = None) -> AlignmentDecision:
    """What ``wid_alignment_visualize.m`` shows for a widefield experiment.

    - one channel: the site's ``first_frame`` next to the experiment's ``first_frame``
      warped with ``tform`` (:func:`calliopy.widefield.compare_wid_frames`), and the
      snippet matching of the two (:func:`calliopy.widefield.plot_image_similarity_snippets`);
    - two channels, the site's origin: ``first_frame_ch525`` next to
      ``fliplr(first_frame_ch610)`` warped with ``tform_ch_map``;
    - two channels, other experiments: per channel the site's frame next to the
      experiment's frame warped with ``tform_ch525`` / ``tform_ch610``.

    The third panel title is ``'Passed: <passed> - img similarity <..> - act similarity
    <..>'`` from ``check_raw``.

    Parameters
    ----------
    adata_path
        Widefield adata file with ``check_raw``.
    site_adata_path
        Adata file of the site's origin experiment; ``None`` when ``adata_path`` is the
        origin itself.
    exp_id, site_id
        For titles and to decide whether the experiment is its site's origin (default: the
        ``S1-T<n>`` number of the file names).
    tforms
        ``{variable name: 3x3 tform.A}`` for transforms the file holds as MATLAB objects
        (which calliopy cannot decode).
    rng
        For the random snippet positions (:func:`calliopy.widefield.image_similarity_snippets`).

    Returns
    -------
    AlignmentDecision

    Raises
    ------
    ValueError
        The file has no ``check_raw`` (MATLAB prints "This file is not checked yet - check
        first" and returns 0), or a transform is not available as a matrix.

    Differences from MATLAB
    -----------------------
    - Paths and ExpIDs are arguments (MATLAB looks them up in ExpLog); the figures are new
      every call (MATLAB reuses persistent figure windows).
    - The snippet similarity has no one-pixel offset (see
      :func:`calliopy.widefield.image_similarity_snippets`).
    """
    ad = Adata(adata_path)
    if "check_raw" not in ad:
        raise ValueError("This file is not checked yet - check first")
    site = Adata(site_adata_path) if site_adata_path is not None else ad
    exp_id = exp_id if exp_id is not None else ad.exp_id
    site_id = site_id if site_id is not None else site.exp_id
    is_site = site_adata_path is None or (exp_id is not None and exp_id == site_id) or \
        Path(site_adata_path).resolve() == Path(adata_path).resolve()
    cr = ad["check_raw"]
    title_c = (f"Passed: {num2str(cr.get('passed', ''))} - img similarity {num2str(cr.get('img_similarity', ''))}"
               f" - act similarity {num2str(cr.get('act_similarity', ''))}")
    t_site, t_exp = f"siteID {num2str(site_id) if site_id is not None else ''}", \
        f"expID {num2str(exp_id) if exp_id is not None else ''}"

    def warped(frame, name):
        f = np.asarray(frame)
        stored = ad[name] if name in ad.variables else None
        a = tform_from(stored, name, tforms)
        return warp_image(f, a, f.shape)

    figures: list[tuple[str, Figure]] = []
    if "first_frame_ch525" not in ad:
        ff_trans = warped(ad["first_frame"], "tform")
        site_ff = np.asarray(site["first_frame"])
        figures.append(("Widefield Alignment - Single Channel",
                        compare_wid_frames(site_ff, ff_trans, t_site, t_exp, title_c)))
        res = image_similarity_snippets(ff_trans, site_ff, rng=rng)
        figures.append(("Image Similarity Snippets", plot_image_similarity_snippets(ff_trans, site_ff, res)))
    elif is_site:
        ff_trans = warped(np.fliplr(np.asarray(ad["first_frame_ch610"])), "tform_ch_map")
        figures.append(("Widefield Alignment - Channel 525 (Green)",
                        compare_wid_frames(ad["first_frame_ch525"], ff_trans, t_site, t_exp, title_c)))
    else:
        for ch, label in (("ch525", "Channel 525 (Green)"), ("ch610", "Channel 610 (Red)")):
            ff_trans = warped(ad[f"first_frame_{ch}"], f"tform_{ch}")
            figures.append((f"Widefield Alignment - {label}",
                            compare_wid_frames(site[f"first_frame_{ch}"], ff_trans, t_site, t_exp, title_c)))
    text = _summary(cr)
    return AlignmentDecision(adata_path, figures, text, exp_id)
