"""Loading adata files into a RoiQuest session.

Port of calliope ``08_RoiQuest/+clbk/getInfoFromAData.m``, ``+clbk/getInfoFromOriginalAData.m``,
``+utils/loadRois.m``, ``+utils/expid2adataPath.m`` and the image preparation of
``+gfx/createDisplayImage.m`` (Leonardo Lupori, 2024). The functions fill the attributes of a
:class:`~calliopy.roiquest.RoiQuestSession`; files are only read here.
"""

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from calliopy.io import MatFile, MatlabObject, load_mat
from calliopy.rois import (
    Roi,
    adjust_img_contrast,
    default_filter_size,
    exp_group,
    filter_template,
    make_cross,
    struct2rois,
    struct2wf_rois,
    struct2wf_rois_2ch,
    valid_area,
)

if TYPE_CHECKING:  # pragma: no cover
    from ._session import RoiQuestSession

__all__: list[str] = []

log = logging.getLogger("calliopy.roiquest")

_ACTIVITY = ("activity", "activity_fissa_corrected")


# ---------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------
def _cells(c) -> list:
    """MATLAB cell (object array, list or a single value) -> list in column-major order."""
    if isinstance(c, np.ndarray) and c.dtype == object:
        return list(c.ravel(order="F"))
    if isinstance(c, (list, tuple)):
        return list(c)
    return [c]


def _struct_items(s) -> list:
    if s is None:
        return []
    if isinstance(s, Mapping):
        return [s] if len(s) else []
    if isinstance(s, np.ndarray):
        if s.dtype != object:
            return []
        return [x for x in s.ravel(order="F") if isinstance(x, Mapping)]
    return list(s)


def _layer_has_rois(layer) -> bool:
    """``~isempty(x) && ~isempty(fieldnames(x))`` of an adata ROI layer."""
    items = _struct_items(layer)
    return bool(items) and len(items[0]) > 0


def _strip_activity(layer):
    items = _struct_items(layer)
    out = []
    for s in items:
        t = type(s)((k, v) for k, v in s.items() if k not in _ACTIVITY)
        out.append(t)
    return out


def var_names(path: Path) -> set[str]:
    """Names of the variables in a v7.3 MAT-file (MATLAB ``who('-file', ...)``)."""
    with MatFile(path) as mf:
        return set(mf.keys())


def _has_var(names: set[str], name: str) -> bool:
    """``utils.mIsField``: case-insensitive test."""
    return any(n.lower() == name.lower() for n in names)


def _var(names: set[str], name: str) -> str:
    for n in names:
        if n.lower() == name.lower():
            return n
    return name


def parse_exp_id(path: Path) -> int | None:
    """ExpID from an adata file name, ``str2double(strrep(f, 'Adata-S1-T', ''))``.

    Returns ``None`` where MATLAB gets NaN (the rest of the name is not a number).
    """
    rest = path.stem.replace("Adata-S1-T", "")
    try:
        v = float(rest)
    except ValueError:
        return None
    if not np.isfinite(v) or v != int(v):
        return None
    return int(v)


def tform_matrix(value) -> np.ndarray | None:
    """3x3 transform matrix from an adata ``tform*`` variable, or ``None``.

    calliopy.io cannot decode MATLAB transform objects (``affine2d``/``affinetform2d``) and
    returns :class:`calliopy.io.MatlabObject` placeholders for them; only a numeric 3x3 array
    (taken to be MATLAB's ``tform.A``, see :mod:`calliopy.widefield`) is accepted.
    """
    if value is None or isinstance(value, MatlabObject):
        return None
    a = np.asarray(value)
    if a.dtype == object or a.shape != (3, 3):
        return None
    return a.astype(np.float64)


# ---------------------------------------------------------------------------------------
# getInfoFromAData
# ---------------------------------------------------------------------------------------
def get_info_from_adata(s: RoiQuestSession, path: Path) -> None:
    """Fill the session from an adata file, MATLAB ``clbk.getInfoFromAData``.

    Widefield files (``first_frame`` or ``first_frame_ch525``) give one image per channel
    (``template``), the ROIs (:class:`calliopy.rois.RoiWf`), the reference points (the first
    two ROIs) and ``ROItemplate_parameters``; two-photon files give ``template`` and
    ``act_map`` per Z layer, the ROIs (:class:`calliopy.rois.Roi`), ``ROItrans`` and the
    valid area from ``dx``/``dy``. The images are kept raw here (``imgs``); see
    :func:`prepare_display_images`.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"adata file not found: {path}")
    s.adata_full_path = path
    s.adata_path = path.parent
    s.adata_file_name = path.name

    names = var_names(path)
    is_wf1 = "first_frame" in names
    is_wf2 = "first_frame_ch525" in names
    is_eeg = "is_eeg" in names
    s.can_modify_rois = not ((is_wf1 or is_wf2) and not is_eeg)
    s.is_widefield_data = bool(is_wf1 or is_wf2)
    s.is_wf_two_channels = bool(is_wf2)
    s.tforms_2ch = {}

    if s.is_widefield_data:
        wanted = [n for n in ("first_frame", "first_frame_ch525", "first_frame_ch610",
                              "tform_ch_map", "tform_ch525", "tform_ch610", "ROIs",
                              "ROItemplate_parameters") if n in names]
        m = load_mat(path, wanted)
        if is_wf1:
            s.template = [np.asarray(m["first_frame"], dtype=np.float64)]
        else:
            s.template = [np.asarray(m["first_frame_ch525"], dtype=np.float64),
                          np.asarray(m["first_frame_ch610"], dtype=np.float64)]
            if "tform_ch_map" in m:
                s.tforms_2ch["ch_map"] = m["tform_ch_map"]
            else:
                s.tforms_2ch["ch525"] = m.get("tform_ch525")
                s.tforms_2ch["ch610"] = m.get("tform_ch610")
        s.act_map = []
        nz = len(s.template)
        hw = s.template[0].shape[:2]
        s.valid_area = np.ones((hw[0], hw[1], nz), dtype=bool)
        s.rois = [[] for _ in range(nz)]
        if "ROIs" in m:
            first = _cells(m["ROIs"])[0] if _cells(m["ROIs"]) else None
            if _layer_has_rois(first):
                if s.is_wf_two_channels:
                    g, r = struct2wf_rois_2ch(first, hw, rng=s.rng)
                    s.rois = [g, r]
                else:
                    s.rois[0] = struct2wf_rois(first, hw, rng=s.rng)
                pts = []
                for k in (0, 1):
                    ind = int(np.ravel(s.rois[0][k].indices)[0]) - 1
                    pts.append((ind % hw[0], ind // hw[0]))
                s.ref_points = np.asarray(pts, dtype=np.float64)
                s.ref_rois = [Roi(make_cross(p, hw), hw, rng=s.rng) for p in pts]
        if "ROItemplate_parameters" in m:
            s.roi_template_parameters = m["ROItemplate_parameters"]
        s.filter_win_size = default_filter_size(hw)
        s.imgs = [[], []]
        for t in s.template:
            s.imgs[0].append(t)
            s.imgs[1].append(filter_template(t, s.filter_win_size, s.filter_type))
    else:
        wanted = [_var(names, n) for n in ("ROIs", "ROItrans") if _has_var(names, n)]
        m = load_mat(path, ["template", "act_map", "dx", "dy", *wanted])
        s.template = _cells(m["template"])
        s.act_map = _cells(m["act_map"])
        nz = len(s.template)
        hw = np.asarray(s.template[0]).shape[:2]
        s.rois = [[] for _ in range(nz)]
        if _has_var(names, "ROIs"):
            layers = _cells(m[_var(names, "ROIs")])
            for i in range(min(nz, len(layers))):
                if _layer_has_rois(layers[i]):
                    s.rois[i] = struct2rois(layers[i], hw, rng=s.rng)
        if _has_var(names, "ROItrans"):
            s.set_roi_trans(m[_var(names, "ROItrans")])
        else:
            s.set_roi_trans([np.zeros((3, 1)) for _ in range(nz)])
        s.valid_area = valid_area(m["dx"], m["dy"], hw)
        s.filter_win_size = default_filter_size(hw)
        s.imgs = [[], [], []]
        for t, a in zip(s.template, s.act_map):
            s.imgs[0].append(t)
            s.imgs[1].append(filter_template(t, s.filter_win_size, s.filter_type))
            s.imgs[2].append(a)

    s.curr_exp_id = parse_exp_id(path)
    if s.exp_log is not None and s.curr_exp_id is not None:
        el = s.exp_log
        s.exp_group = [int(v) for v in exp_group(s.curr_exp_id, el["expid"], el["siteid"])]
    else:
        s.exp_group = [] if s.curr_exp_id is None else [s.curr_exp_id]
    s.roi_bg_color = (0.93, 0.13, 0.55)
    s.roi_sel_bg_color = (0.0, 0.94, 0.14)


# ---------------------------------------------------------------------------------------
# getInfoFromOriginalAData
# ---------------------------------------------------------------------------------------
def get_info_from_original_adata(s: RoiQuestSession, path: Path | None,
                                 load_original_rois: bool = True) -> None:
    """Load the original recording of a chronic series, MATLAB ``clbk.getInfoFromOriginalAData``.

    Parameters
    ----------
    s
        The session (already filled by :func:`get_info_from_adata`).
    path
        Adata file of the first ExpID of the series; ``None`` does nothing.
    load_original_rois
        Two-photon only: the answer to MATLAB's dialog "Load original ROIs?" (default
        Yes). The original's ROIs then replace the ROIs of every layer that has ROIs in the
        original file.

    Notes
    -----
    Differences from MATLAB: the warning "The original ExpID has no ROIs saved!" (a modal
    message box in MATLAB) is a :class:`UserWarning`.
    """
    if path is None or str(path) == "":
        return
    path = Path(path)
    log.info("Part of a chronic series! Loading also from the original exp in:\n\t- %s", path)
    s.orig_adata_full_path = path
    names = var_names(path)
    if s.is_widefield_data:
        if _has_var(names, "first_frame"):
            m = load_mat(path, [_var(names, "first_frame")])
            s.orig_template = [np.asarray(m[_var(names, "first_frame")], dtype=np.float64)]
        elif _has_var(names, "first_frame_ch525"):
            m = load_mat(path, ["first_frame_ch525", "first_frame_ch610", "tform_ch_map"])
            s.orig_template = [np.asarray(m["first_frame_ch525"], dtype=np.float64),
                               np.asarray(m["first_frame_ch610"], dtype=np.float64)]
            s.tforms_2ch["ch_map"] = m["tform_ch_map"]
        s.orig_imgs = [[], []]
        for t in s.orig_template:
            s.orig_imgs[0].append(t)
            s.orig_imgs[1].append(filter_template(t, s.filter_win_size, s.filter_type))
        return

    wanted = [_var(names, "ROIs")] if _has_var(names, "ROIs") else []
    m = load_mat(path, ["template", "act_map", *wanted])
    s.orig_template = _cells(m["template"])
    s.orig_act_map = _cells(m["act_map"])
    s.orig_imgs = [[], [], []]
    for t, a in zip(s.orig_template, s.orig_act_map):
        s.orig_imgs[0].append(t)
        s.orig_imgs[1].append(filter_template(t, s.filter_win_size, s.filter_type))
        s.orig_imgs[2].append(a)
    if not load_original_rois:
        log.info("Original ROIs not loaded.")
        return
    layers = _cells(m[_var(names, "ROIs")]) if wanted else []
    filled = [_layer_has_rois(x) for x in layers]
    if not any(filled):
        warnings.warn("The original ExpID has no ROIs saved! The currently loaded expID is "
                      "part of a chronic series. Auto loading of original ROIs aborted. "
                      "Please first ROI the original ExpID of this series.", stacklevel=3)
        return
    hw = np.asarray(s.orig_imgs[1][-1]).shape[:2]
    for i in range(min(len(s.orig_template), len(layers))):
        if filled[i]:
            s.rois[i] = struct2rois(_strip_activity(layers[i]), hw, rng=s.rng)


def prepare_display_images(s: RoiQuestSession) -> None:
    """Convert every image to uint8 with the ``'bestFit'`` contrast (``gfx.createDisplayImage``)."""
    s.imgs = [[adjust_img_contrast(im) for im in row] for row in s.imgs]
    s.orig_imgs = [[adjust_img_contrast(im) for im in row] for row in s.orig_imgs]


# ---------------------------------------------------------------------------------------
# loadRois
# ---------------------------------------------------------------------------------------
def load_rois(s: RoiQuestSession, path: Path) -> bool:
    """Replace the ROIs with those of an adata file, MATLAB ``utils.loadRois``.

    Resets the random-colour generator (MATLAB ``rng(42)``), drops the putative ROIs, replaces
    the ROIs of all layers (layers empty in the file become empty) and takes ``ROItrans`` from
    the file when present. Activity traces are not loaded.

    Returns
    -------
    bool
        False when the file has no ``ROIs`` variable or no ROIs at all (MATLAB logged a
        message and returned).

    Raises
    ------
    ValueError
        For widefield data (MATLAB's ``loadRois`` reads the two-photon ``template`` and
        creates two-photon ROI objects, which fails on widefield files).
    """
    if s.is_widefield_data:
        raise ValueError("Loading ROIs from another AData file is only possible for "
                         "two-photon data.")
    path = Path(path)
    s.reset_rng()
    log.info("Loading ROIs from:\n\t- %s", path)
    names = var_names(path)
    if not _has_var(names, "ROIs"):
        log.info("Selected AData file does not contain a ROI field. Loading ROIs aborted.")
        return False
    wanted = [_var(names, "ROIs"), "template"]
    if _has_var(names, "ROItrans"):
        wanted.append(_var(names, "ROItrans"))
    m = load_mat(path, wanted)
    layers = _cells(m[_var(names, "ROIs")])
    empty = [len(_struct_items(x)) == 0 for x in layers]
    if all(empty):
        log.info("Selected AData file contains no ROIs. Loading ROIs aborted.")
        return False
    s.putative_rois = []
    hw = np.asarray(_cells(m["template"])[0]).shape[:2]
    s.rois = [[] for _ in layers]
    for i, layer in enumerate(layers):
        if not empty[i]:
            s.rois[i] = struct2rois(_strip_activity(layer), hw, rng=s.rng)
    if _has_var(names, "ROItrans"):
        s.set_roi_trans(m[_var(names, "ROItrans")])
    return True


# ---------------------------------------------------------------------------------------
# series files (expid2adataPath, get_adata_filename)
# ---------------------------------------------------------------------------------------
def adata_file_of(exp_id: int, adata_dir, exp_log) -> Path | None:
    """Adata file of an ExpID as RoiQuest finds it: ``<adata_dir>/<pi>/<animal>/<file>`` with
    the file name from :func:`calliopy.data.get_adata_filename`; ``None`` when there is none."""
    from calliopy.data import get_adata_filename

    res = get_adata_filename(int(exp_id), adata_dir, exp_log)
    if not res.adata_file:
        return None
    return Path(adata_dir) / res.user_id / res.mouse_id / res.adata_file


def expid2adata_path(exp_id: int, adata_dir, exp_log) -> tuple[Path, Path | None]:
    """Adata file of an ExpID and of the original of its chronic series, MATLAB
    ``utils.expid2adataPath`` (08_RoiQuest).

    Returns
    -------
    tuple
        ``(adata_path, original_adata_path)``; the original is ``None`` when the ExpID is
        alone at its site or is itself the first ExpID of the series.

    Raises
    ------
    FileNotFoundError
        No adata file for ``exp_id`` (MATLAB built a folder path and failed later).
    """
    path = adata_file_of(exp_id, adata_dir, exp_log)
    if path is None:
        raise FileNotFoundError(f"ExpID {exp_id} has no adata file in the given adata folder")
    group = [int(v) for v in exp_group(exp_id, exp_log["expid"], exp_log["siteid"])]
    if len(group) == 1 or exp_id == group[0]:
        return path, None
    orig = adata_file_of(group[0], adata_dir, exp_log)
    return path, orig


def is_zstack_name(path: Path | None) -> bool:
    """``contains(adataFile, 'mean_data')``: adata files of z-stacks are skipped."""
    return path is not None and "mean_data" in path.name


_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def prefs_file_name(user: str, project: str) -> str:
    """Default preferences file name ``<user>_<project>.json`` (MATLAB used ``.xml``)."""
    return _NAME_RE.sub("_", f"{user}_{project}") + ".json"
