"""RoiQuest without a window: state and one method per MATLAB callback.

Port of calliope ``08_RoiQuest/RoiQuest.m`` and the callbacks in ``08_RoiQuest/+clbk``
(Leonardo Lupori, 2024). The session holds everything the MATLAB app object holds, except
graphics handles; a window (or a test) calls its methods with explicit pixel coordinates,
keys and answers to MATLAB's dialogs, then redraws what the returned :class:`Redraw` flags
name, with :class:`calliopy.roiquest.RoiQuestView` for example.

Conventions
-----------
- Pixel coordinates passed to methods are **0-based** ``(row, col)`` (matplotlib's
  ``(ydata, xdata)``); ``current_z`` and ``img_idx`` are 0-based. Coordinates are rounded
  half away from zero, like MATLAB's ``round(CurrentPoint)``.
- ROI ``indices`` keep the stored MATLAB values (1-based column-major linear indices), so
  saved files are interchangeable with MATLAB's.
- Texts shown to the user (info text, log messages, ROI numbers) number Z layers and ROIs
  from 1, as RoiQuest does.
- Nothing prints; messages go to the ``calliopy.roiquest`` logger. Refusals that MATLAB
  logged before returning raise :class:`RoiQuestRefused`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from calliopy._matlab import mround
from calliopy.explog import as_exp_log
from calliopy.rois import (
    ChronicRoiSave,
    Roi,
    adjust_img_contrast,
    auto_roi_matching,
    filter_cellpose_labels,
    filter_template,
    get_labels_colormap,
    make_cross,
    rois_binary_mask,
    segment_neuron,
    struct2wf_rois,
    uniform_rgb_image,
)
from calliopy.rois._builtins import mat2gray

from . import _load, _persist
from ._display import colormap
from ._types import Redraw, RoiQuestRefused, SaveResult, View

__all__ = ["RoiQuestSession", "cellpose_available"]

log = logging.getLogger("calliopy.roiquest")

DEAD_ZONE = 5
"""Mouse releases closer than this (pixels, per coordinate) to the press do not zoom
(``figButtonUp.m``)."""

EEG_TEMPLATE_NAME = "Grid whole cortex plus Ring Rois around electrode positions based on " \
                    "lambda-bregma"
"""``ROItemplate_parameters.name`` of the widefield + EEG template (``figButtonUp.m``)."""

EEG_RING_TEMPLATE_FILE = "Ring_Rois_around_electrode_positions_based_on_lambda-bregma.mat"
"""Template file that marks a recording as widefield + EEG (``wfAddRois.m``)."""

IMG_NAMES = ("Template", "Filtered Template", "Activity Map")

_CONTRAST_TEXT = {"minMax": "Min/Max", "bestFit": "Best Fit", "custom": "Custom"}

_DEFAULT_ROI_COLOR = (0.93, 0.13, 0.55)
_DEFAULT_SEL_COLOR = (0.0, 0.94, 0.14)


def cellpose_available() -> bool:
    """True when the optional ``cellpose`` package can be imported (menu ROIs > Autodetect)."""
    import importlib.util

    return importlib.util.find_spec("cellpose") is not None


def _find(mask: np.ndarray) -> np.ndarray:
    """MATLAB ``find(mask)``: 1-based column-major linear indices, ``(N, 1)`` float64."""
    return (np.flatnonzero(np.asarray(mask).ravel(order="F")) + 1).astype(np.float64)[:, None]


def _lin(indices) -> np.ndarray:
    return np.asarray(indices, dtype=np.float64).ravel(order="F").astype(np.int64)


def _mods(modifiers) -> set[str]:
    if modifiers is None:
        return set()
    if isinstance(modifiers, str):
        modifiers = [modifiers]
    out = set()
    for m in modifiers:
        m = str(m).lower()
        out.add({"ctrl": "control", "cmd": "command"}.get(m, m))
    return out


class RoiQuestSession:
    """State and actions of RoiQuest, MATLAB ``RoiQuest`` (``08_RoiQuest/RoiQuest.m``).

    Create a session with :meth:`from_adata` (an adata file, optionally the original file of
    its chronic series) or :meth:`from_expid` (ExpLog table and adata folder needed). The
    attributes are the MATLAB properties in snake_case (graphics handles excepted); the
    methods are the MATLAB callbacks with explicit arguments instead of figure state, and
    return :class:`Redraw` flags.

    Parameters
    ----------
    exp_log
        ExpLog table (any form :func:`calliopy.explog.as_exp_log` accepts). Needed for
        chronic series (ExpIDs recorded at the same site) and for :meth:`from_expid`.
    adata_dir
        Root of the adata folders (``<adata_dir>/<pi>/<animal>/<file>``), MATLAB
        ``set_lab_paths``. Needed with ``exp_log``.
    roi_template_dir
        Folder of widefield ROI templates (MATLAB ``12_Widefield/ROItemplate_library``),
        used by :meth:`roi_template_files` and :meth:`wf_add_rois`.
    seed
        Seed of the generator of random ROI colours (MATLAB ``rng(42)``).

    Attributes
    ----------
    adata_full_path, adata_path, adata_file_name : Path, Path, str
        Current adata file, its folder and its name.
    curr_exp_id : int or None
        ExpID from the file name (``Adata-S1-T<ExpID>``).
    template, act_map : list of ndarray
        Per Z layer (widefield: per channel) the template and the activity map (two-photon).
    valid_area : ndarray
        Boolean ``(H, W, nZ)``: where ROIs may be created.
    rois : list of list of Roi
        Per layer, the ROIs (:class:`calliopy.rois.Roi`, widefield
        :class:`calliopy.rois.RoiWf`).
    putative_rois : list of Roi
        ROIs proposed by the zoom selection or Cellpose, not yet accepted.
    roi_trans : list of ndarray
        Per layer, ``ROItrans`` (global translation; first two values ``[dx, dy]``).
    imgs, orig_imgs : list of list of ndarray
        ``imgs[k][z]``: uint8 display image ``k`` (0 template, 1 filtered template, 2 activity
        map; widefield has 0 and 1) of layer ``z``; ``orig_imgs`` for the original recording.
    img_idx, current_z : int
        Displayed image kind and layer, 0-based.
    thr, img_lo, img_hi : list of float
        Per image kind: segmentation threshold (0-255, default 160) and custom contrast limits.
    contrast_mode, filter_type, filter_win_size
        ``'bestFit'``, ``'legacy'``, ``round(min(H, W)/20)``.
    cmap : ndarray
        ``(256, 3)`` colormap of the image.
    lab_cmap : ndarray
        ``(12, 3)`` colormap of ROI labels (:func:`calliopy.rois.get_labels_colormap`).
    ovrl_opacity, roi_bg_color, roi_sel_bg_color, roi_color_mode
        Overlay opacity (0.35), flat ROI colour, selection colour, ``'flat'`` / ``'random'`` /
        ``'tags'``.
    roi_visible, roi_number_visible, info_pos, show_original
        Display toggles; ``info_pos`` is ``'tl'``, ``'tr'``, ``'bl'`` or ``'br'``.
    is_zoomed_in, view
        Zoom state and axes limits (:class:`View`) shared with the renderer.
    btn_down_coord, btn_up_coord : tuple or None
        Last mouse press/release, 0-based ``(row, col)``.
    exp_group : list of int
        ExpIDs of the chronic series (the current one alone without an ExpLog).
    orig_adata_full_path, orig_template, orig_act_map
        Original recording of a chronic series (``None`` / empty otherwise).
    is_widefield_data, is_wf_two_channels, is_wf_eeg, can_modify_rois
        Recording kind; ``is_wf_eeg`` is never set by RoiQuest (kept for ``saveToAdata``).
    ref_points : ndarray
        ``(n, 2)`` widefield reference points, 0-based ``(row, col)``.
    ref_rois : list of Roi
        Crosses drawn at the reference points.
    roi_template_parameters
        ``ROItemplate_parameters`` of the widefield ROI template.
    tforms_2ch : dict
        Two-channel transforms as read (``'ch_map'``, ``'ch525'``, ``'ch610'``).
    cell_pose_model
        Cellpose model used by :meth:`cellpose_detect`.
    vectors, show_vectors
        Movement vectors of the last automatic ROI matching (renderer shows them once).
    backups : dict
        Adata file -> backup copy made by this session.

    Notes
    -----
    Differences from MATLAB:

    - every load starts from the defaults for data-related state (reference points, ROI
      template parameters, transforms, zoom), where MATLAB kept some of them from the
      previously loaded file;
    - ``ref_points`` is empty until reference points exist (MATLAB: ``zeros(2,2)``) and
      0-based;
    - only one instance per figure is enforced by the window, not by the session.
    """

    def __init__(self, *, exp_log=None, adata_dir=None, roi_template_dir=None, seed: int = 42):
        self.exp_log = None if exp_log is None else as_exp_log(exp_log)
        self.adata_dir = None if adata_dir is None else Path(adata_dir)
        self.roi_template_dir = None if roi_template_dir is None else Path(roi_template_dir)
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        # data of the current recording
        self.adata_full_path: Path | None = None
        self.adata_path: Path | None = None
        self.adata_file_name: str | None = None
        self.curr_exp_id: int | None = None
        self.template: list = []
        self.act_map: list = []
        self.valid_area: np.ndarray | None = None
        self.rois: list[list[Roi]] = []
        self.putative_rois: list[Roi] = []
        self.roi_trans: list[np.ndarray] = []
        self._roi_trans_shape: tuple[int, ...] = (1, 0)
        self.imgs: list[list[np.ndarray]] = []
        self.img_names = list(IMG_NAMES)
        self.curr_img_median: float | None = None
        self.curr_img_std: float | None = None
        self.ovrl_opacity = 0.35
        # chronic imaging
        self.exp_group: list[int] = []
        self.orig_adata_full_path: Path | None = None
        self.orig_template: list = []
        self.orig_act_map: list = []
        self.orig_imgs: list[list[np.ndarray]] = []
        self.show_original = False
        # GUI parameters and settings
        self.can_modify_rois = True
        self.img_idx = 0
        self.current_z = 0
        self.contrast_mode = "bestFit"
        self.img_lo = [0.0, 0.0, 0.0]
        self.img_hi = [1.0, 1.0, 1.0]
        self.thr = [160.0, 160.0, 160.0]
        self.filter_win_size: int | None = None
        self.filter_type = "legacy"
        self.cmap = colormap("gray")
        self.btn_down_coord: tuple[int, int] | None = None
        self.btn_up_coord: tuple[int, int] | None = None
        self.lab_cmap = get_labels_colormap()
        self.min_roi_pix = 20
        self.is_zoomed_in = False
        self.view = View((-0.5, 0.5), (-0.5, 0.5))
        self.roi_bg_color = _DEFAULT_ROI_COLOR
        self.roi_sel_bg_color = _DEFAULT_SEL_COLOR
        self.roi_color_mode = "flat"
        self.cell_pose_model = None
        self.roi_visible = True
        self.roi_number_visible = False
        self.god_counter = 0
        self.god_mode = False
        self.show_vectors = False
        self.vectors: tuple[np.ndarray, np.ndarray] | None = None
        self.info_pos = "tl"
        # widefield
        self.is_widefield_data = False
        self.is_wf_two_channels = False
        self.is_wf_eeg = False
        self.ref_points = np.zeros((0, 2))
        self.ref_rois: list[Roi] = []
        self.roi_template_parameters = None
        self.tforms_2ch: dict = {}
        self.backups: dict[Path, Path] = {}

    # ---------------------------------------------------------------------------------
    # construction and loading
    # ---------------------------------------------------------------------------------
    @classmethod
    def from_adata(cls, adata_path, *, original_adata_path=None, exp_log=None, adata_dir=None,
                   roi_template_dir=None, load_original_rois: bool = True,
                   seed: int = 42) -> RoiQuestSession:
        """Open an adata file, MATLAB ``RoiQuest(expId)`` once the file is known.

        Parameters
        ----------
        adata_path
            The adata file (MATLAB v7.3).
        original_adata_path
            Adata file of the original (first) recording of the chronic series, if the file
            belongs to one and is not the original itself.
        exp_log, adata_dir, roi_template_dir, seed
            See :class:`RoiQuestSession`.
        load_original_rois
            Answer to "Load original ROIs?" (two-photon chronic series; default Yes).
        """
        s = cls(exp_log=exp_log, adata_dir=adata_dir, roi_template_dir=roi_template_dir,
                seed=seed)
        s.load_new_adata(adata_path, original_adata_path=original_adata_path,
                         load_original_rois=load_original_rois)
        return s

    @classmethod
    def from_expid(cls, exp_id: int, *, exp_log, adata_dir, roi_template_dir=None,
                   load_original_rois: bool = True, seed: int = 42) -> RoiQuestSession:
        """Open the adata file of an ExpID, MATLAB ``RoiQuest(expId)``.

        The file is ``<adata_dir>/<pi>/<animal>/<name>`` with the name found by
        :func:`calliopy.data.get_adata_filename`; for a chronic series (several ExpIDs at the
        same site) whose first ExpID is not ``exp_id``, the first ExpID's file is opened as
        the original (MATLAB ``utils.expid2adataPath``).
        """
        s = cls(exp_log=exp_log, adata_dir=adata_dir, roi_template_dir=roi_template_dir,
                seed=seed)
        s.load_new_expid(exp_id, load_original_rois=load_original_rois)
        return s

    def prefs_default_name(self) -> str | None:
        """Default preferences file name ``<pi>_<project>.json`` from the ExpLog (MATLAB:
        ``<pi>_<project>.xml``), or None without ExpLog or ExpID."""
        if self.exp_log is None or self.curr_exp_id is None:
            return None
        el = self.exp_log
        rows = np.flatnonzero(np.asarray(el["expid"], dtype=np.float64) == self.curr_exp_id)
        if rows.size == 0 or "pi" not in el or "project" not in el:
            return None
        i = int(rows[0])
        return _load.prefs_file_name(str(np.asarray(el["pi"], dtype=object)[i]),
                                     str(np.asarray(el["project"], dtype=object)[i]))

    def reset_rng(self) -> None:
        """Restart the random-colour generator (MATLAB ``rng(42)``)."""
        self.rng = np.random.default_rng(self._seed)

    def _reset_for_load(self) -> None:
        self.putative_rois = []
        self.img_idx = 0
        self.current_z = 0
        self.contrast_mode = "bestFit"
        self.img_lo = [0.0, 0.0, 0.0]
        self.img_hi = [1.0, 1.0, 1.0]
        self.thr = [160.0, 160.0, 160.0]
        self.orig_adata_full_path = None
        self.orig_template = []
        self.orig_act_map = []
        self.orig_imgs = []
        self.ref_points = np.zeros((0, 2))
        self.ref_rois = []
        self.roi_template_parameters = None
        self.tforms_2ch = {}
        self.vectors = None
        self.show_vectors = False
        self.reset_rng()

    def load_new_adata(self, adata_path, original_adata_path=None, *,
                       load_original_rois: bool = True) -> Redraw:
        """Load another adata file, MATLAB ``RoiQuest.loadNewExpid`` once the files are known.

        Resets the putative ROIs, image kind, layer, contrast and thresholds, reads the file
        (``getInfoFromAData``) and the original of its series (``getInfoFromOriginalAData``)
        and prepares the uint8 display images (``createDisplayImage``).
        """
        self._reset_for_load()
        _load.get_info_from_adata(self, Path(adata_path))
        _load.get_info_from_original_adata(
            self, None if original_adata_path is None else Path(original_adata_path),
            load_original_rois)
        _load.prepare_display_images(self)
        self.reset_view()
        self.display_image()
        log.info("Done loading. Happy ROIing :)")
        return Redraw.ALL

    def load_new_expid(self, exp_id: int, *, load_original_rois: bool = True) -> Redraw:
        """Load the adata file of another ExpID, MATLAB ``RoiQuest.loadNewExpid``."""
        if self.exp_log is None or self.adata_dir is None:
            raise ValueError("loading by ExpID needs exp_log and adata_dir")
        path, orig = _load.expid2adata_path(int(exp_id), self.adata_dir, self.exp_log)
        log.info("Loading new AData file:\n\t- %s", path)
        return self.load_new_adata(path, orig, load_original_rois=load_original_rois)

    def revert_to_saved(self, *, load_original_rois: bool = True) -> Redraw:
        """Reload the current files, dropping all unsaved changes (``clbk.revertToSaved``).

        MATLAB asks for confirmation first; the window does that before calling.
        """
        if self.adata_full_path is None:
            raise RoiQuestRefused("No AData file loaded.")
        return self.load_new_adata(self.adata_full_path, self.orig_adata_full_path,
                                   load_original_rois=load_original_rois)

    def load_rois_from_adata(self, source) -> Redraw:
        """Replace the ROIs with those of an adata file (``clbk.loadRoisFromAdata``).

        Parameters
        ----------
        source
            Path of an adata file (menu "Load from AData") or ``'original'`` for the original
            recording of the series (menu "Load from original"; nothing happens without one).
        """
        if isinstance(source, str) and source == "original":
            if self.orig_adata_full_path is None:
                return Redraw.NONE
            path = self.orig_adata_full_path
        else:
            path = Path(source)
        _load.load_rois(self, path)
        return Redraw.OVERLAY

    # ---------------------------------------------------------------------------------
    # derived state
    # ---------------------------------------------------------------------------------
    @property
    def num_layers(self) -> int:
        """Number of Z layers (widefield: channels), MATLAB ``size(app.template, 2)``."""
        return len(self.template)

    @property
    def image_hw(self) -> tuple[int, int]:
        """``(H, W)`` of the images (MATLAB ``size(app.imgs{app.imgIdx, 1})``)."""
        if not self.imgs or not self.imgs[self.img_idx]:
            return (0, 0)
        h, w = np.asarray(self.imgs[self.img_idx][0]).shape[:2]
        return int(h), int(w)

    @property
    def roi_bg(self) -> np.ndarray:
        """``(H, W, 3)`` flat ROI colour image (MATLAB ``roiBg``)."""
        return uniform_rgb_image(self.image_hw, self.roi_bg_color)

    @property
    def roi_sel_bg(self) -> np.ndarray:
        """``(H, W, 3)`` selection colour image (MATLAB ``roiSelBg``)."""
        return uniform_rgb_image(self.image_hw, self.roi_sel_bg_color)

    @property
    def enabled(self) -> dict[str, bool]:
        """Menu items a window enables or shows, keyed by MATLAB's handle names.

        ``utils.enableFunctionality`` (widefield vs two-photon items), ``loadRoiOrig``
        (chronic series only), ``setFiltType``/``setFiltSize`` (visible in "god mode", see
        :meth:`god_mode_key`) and ``checkReg`` (always disabled in r2003).

        Differences from MATLAB: MATLAB only called ``enableFunctionality`` for widefield
        files, so loading a two-photon file after a widefield one kept the widefield
        settings; here the items follow the current file.
        """
        wf = self.is_widefield_data
        return {
            "roiExpand": not wf,
            "roiShrink": not wf,
            "cellposeDetect": not wf,
            "wfAddRefPoints": wf,
            "wfAddRois": wf,
            "wfPropagate": wf,
            "wfCheckAlignment": wf,
            "loadRoiOrig": self.orig_adata_full_path is not None,
            "setFiltType": self.god_mode,
            "setFiltSize": self.god_mode,
            "checkReg": False,
        }

    def set_roi_trans(self, value) -> None:
        """Set ``roi_trans`` from an adata ``ROItrans`` cell (object array or list)."""
        if isinstance(value, np.ndarray) and value.dtype == object:
            self._roi_trans_shape = value.shape
            items = list(value.ravel(order="F"))
        else:
            items = list(value)
            self._roi_trans_shape = (1, len(items))
        self.roi_trans = [np.array(v, dtype=np.float64) for v in items]

    def roi_trans_cell(self) -> np.ndarray:
        """``roi_trans`` as the MATLAB cell array it was read from (object array)."""
        shape = self._roi_trans_shape
        if int(np.prod(shape)) != len(self.roi_trans):
            shape = (1, len(self.roi_trans))
        out = np.empty(shape, dtype=object)
        for k, v in enumerate(self.roi_trans):
            out[np.unravel_index(k, shape, order="F")] = v
        return out

    def reset_view(self) -> Redraw:
        """Show the whole image (MATLAB ``axis(ax, 'tight')``) and leave the zoom mode."""
        h, w = self.image_hw
        self.view = View((-0.5, w - 0.5), (-0.5, h - 0.5))
        self.is_zoomed_in = False
        return Redraw.VIEW

    def display_image(self) -> np.ndarray:
        """Image to display, MATLAB ``gfx.updateDisplayImage`` (data part).

        The original recording's image when ``show_original`` is set and there is one,
        otherwise ``imgs[img_idx][current_z]``. Updates ``curr_img_median`` and
        ``curr_img_std`` (NaNs ignored, ``std`` with N-1).
        """
        if self.show_original and self.orig_adata_full_path is not None and self.orig_imgs:
            im = self.orig_imgs[self.img_idx][self.current_z]
        else:
            im = self.imgs[self.img_idx][self.current_z]
        d = np.asarray(im, dtype=np.float64)
        self.curr_img_median = float(np.nanmedian(d))
        self.curr_img_std = float(np.nanstd(d, ddof=1)) if d.size > 1 else float("nan")
        return np.asarray(im)

    # ---------------------------------------------------------------------------------
    # mouse: figButtonDown / figButtonUp
    # ---------------------------------------------------------------------------------
    def _clamp(self, row, col) -> tuple[int, int]:
        h, w = self.image_hw
        r = int(mround(float(row)))
        c = int(mround(float(col)))
        return min(max(r, 0), h - 1), min(max(c, 0), w - 1)

    def button_down(self, row: float, col: float) -> Redraw:
        """Mouse press at 0-based ``(row, col)``, MATLAB ``clbk.figButtonDown``.

        The point is rounded (``round(CurrentPoint)``) and stored in ``btn_down_coord``.
        """
        self.btn_down_coord = (int(mround(float(row))), int(mround(float(col))))
        return Redraw.NONE

    def button_up(self, row: float, col: float, kind: str = "normal") -> Redraw:
        """Mouse release at 0-based ``(row, col)``, MATLAB ``clbk.figButtonUp``.

        Parameters
        ----------
        row, col
            Pointer position (rounded, clamped to the image).
        kind
            MATLAB's ``SelectionType``: ``'normal'`` (left button: zoom on the box from the
            press to the release, then for two-photon data segment the zoomed region with
            :func:`calliopy.rois.segment_neuron` into one putative ROI; nothing when the
            release is within :data:`DEAD_ZONE` pixels of the press in either coordinate);
            ``'alt'`` (right button: a click selects the ROI under the pointer and deselects
            all others, a drag selects the ROIs touching the box; also selects widefield
            reference points); ``'extend'`` (shift + left or middle button: toggle the
            selection of the ROI and reference point under the pointer).

        Notes
        -----
        With the widefield + EEG template (``ROItemplate_parameters.name`` equal to
        :data:`EEG_TEMPLATE_NAME`), the right-button selection skips ROIs 0, 1 (the reference
        points) and 122 onwards (0-based; MATLAB ``i < 3 || i > 122``).

        Differences from MATLAB: both corners are clamped to the image (MATLAB clamped only
        a release outside the figure, so a release in the figure but beside the image grew
        the selection mask); without ``ROItemplate_parameters`` (two-photon data) the EEG
        test is false (MATLAB evaluated ``[].name``, an error).
        """
        up = self._clamp(row, col)
        self.btn_up_coord = up
        down = self._clamp(*(self.btn_down_coord if self.btn_down_coord is not None else up))
        dr, dc = abs(down[0] - up[0]), abs(down[1] - up[1])
        r0, r1 = sorted((down[0], up[0]))
        c0, c1 = sorted((down[1], up[1]))
        if kind == "normal":
            if dr < DEAD_ZONE or dc < DEAD_ZONE:
                return Redraw.NONE
            self.view = View((float(c0), float(c1)), (float(r0), float(r1)))
            self.is_zoomed_in = True
            if not self.is_widefield_data:
                mask = self._segment()
                if mask.any():
                    self.putative_rois = [Roi(_find(mask), mask.shape, rng=self.rng)]
            return Redraw.VIEW | Redraw.OVERLAY | Redraw.HISTOGRAM
        rois = self.rois[self.current_z] if self.rois else []
        h, w = self.image_hw
        if kind == "alt":
            if not rois:
                return Redraw.NONE
            box = np.zeros((h, w), dtype=bool)
            box[r0:r1 + 1, c0:c1 + 1] = True
            flat = box.ravel(order="F")
            eeg = self._eeg_template_selected()
            for i, roi in enumerate(rois):
                if eeg and (i < 2 or i > 121):
                    continue
                roi.selected = bool(flat[_lin(roi.indices) - 1].any())
            for ref in self.ref_rois[:2]:
                ref.selected = bool(flat[_lin(ref.indices) - 1].any())
            return Redraw.OVERLAY
        if kind == "extend":
            clicked = up[0] + up[1] * h + 1
            for roi in rois:
                if clicked in _lin(roi.indices):
                    roi.selected = not roi.selected
            for ref in self.ref_rois[:2]:
                if clicked in _lin(ref.indices):
                    ref.selected = not ref.selected
            return Redraw.OVERLAY
        return Redraw.OVERLAY

    def _eeg_template_selected(self) -> bool:
        p = self.roi_template_parameters
        name = p.get("name") if isinstance(p, Mapping) else None
        return isinstance(name, str) and name.lower() == EEG_TEMPLATE_NAME.lower()

    def _zoom_crop(self) -> tuple[int, int, int, int]:
        (c0, c1), (r0, r1) = self.view.xlim, self.view.ylim
        return int(r0), int(r1) + 1, int(c0), int(c1) + 1

    def _segment(self) -> np.ndarray:
        """``gfx.segmentNeuron(app)`` on the zoomed region of the current image."""
        z = self.current_z
        rois = self.rois[z]
        return segment_neuron(
            self.imgs[self.img_idx][z], self.thr[self.img_idx], self._zoom_crop(),
            min_roi_pix=self.min_roi_pix,
            valid_area=None if self.valid_area is None else self.valid_area[:, :, z],
            existing_mask=rois_binary_mask(rois) if rois else None,
            close_and_despur=self.img_idx == 2)

    # ---------------------------------------------------------------------------------
    # threshold: scrollWheelCallback / upDownArrow
    # ---------------------------------------------------------------------------------
    def _threshold_change(self, amount: float) -> None:
        k = self.img_idx
        self.thr[k] = float(min(max(self.thr[k] + amount, 0.0), 255.0))
        mask = self._segment()
        if not mask.any() and amount < 0:
            r0, r1, c0, c1 = self._zoom_crop()
            crop = np.asarray(self.imgs[k][self.current_z])[r0:r1, c0:c1]
            self.thr[k] = float(mround(np.median(crop.astype(np.float64))))
            mask = self._segment()
        if mask.any():
            self.putative_rois = [Roi(_find(mask), mask.shape, rng=self.rng)]
        else:
            self.putative_rois = []

    def scroll(self, count: int) -> Redraw:
        """Mouse wheel, MATLAB ``clbk.scrollWheelCallback``.

        Zoomed in: the threshold of the current image kind changes by ``count``
        (MATLAB ``VerticalScrollCount``; negative = wheel up = lower threshold, more pixels),
        clamped to 0-255, and the zoomed region is segmented again. When nothing is segmented
        after scrolling up, the threshold snaps to the median of the zoomed region. Not zoomed
        in: wheel up shows the previous layer, wheel down the next.

        Differences from MATLAB: for widefield data the wheel does nothing while zoomed in
        (MATLAB segmented two-photon-style putative ROIs that could then be added to the
        widefield ROIs; ``upDownArrow`` already refused this).
        """
        if self.is_zoomed_in:
            if self.is_widefield_data:
                return Redraw.NONE
            self._threshold_change(float(count))
            return Redraw.OVERLAY | Redraw.HIST_THRESHOLD
        self.change_layer("-" if count < 0 else "+")
        return Redraw.DISPLAY | Redraw.OVERLAY | Redraw.HISTOGRAM

    def threshold_step(self, direction: str, modifiers=()) -> Redraw:
        """Up/down arrow keys, MATLAB ``clbk.upDownArrow``.

        Zoomed in on two-photon data: ``'up'`` lowers and ``'down'`` raises the threshold by 1
        (5 with ``'shift'``) and segments again, snapping to the zoomed region's median when
        ``'up'`` leaves nothing segmented. Otherwise, without putative ROIs, the selected
        ROIs move up/down by 1 (10) pixel (:meth:`move_selected`).

        Differences from MATLAB: the median snap tested ``event.VerticalScrollCount``, which
        does not exist in this callback, so MATLAB stopped with an error whenever the
        threshold left nothing segmented; here the snap applies to ``'up'``, the key that
        corresponds to scrolling up.
        """
        mods = _mods(modifiers)
        amount = 5.0 if "shift" in mods else 1.0
        if direction == "up":
            amount = -amount
        elif direction != "down":
            raise ValueError("direction must be 'up' or 'down'")
        if self.is_zoomed_in and not self.is_widefield_data:
            self._threshold_change(amount)
            return Redraw.OVERLAY | Redraw.HIST_THRESHOLD
        if not self.putative_rois:
            self.move_selected(direction, modifiers)
        return Redraw.OVERLAY

    # ---------------------------------------------------------------------------------
    # ROI editing
    # ---------------------------------------------------------------------------------
    def move_selected(self, direction: str, modifiers=()) -> Redraw:
        """Move the selected ROIs and reference points, MATLAB ``clbk.moveSelectedRois``.

        One pixel (10 with ``'shift'``) ``'up'``, ``'down'``, ``'left'`` or ``'right'``
        (:meth:`calliopy.rois.Roi.move`; pixels leaving the image are dropped and ``shift``
        records the movement). When ROIs may not be modified (widefield without EEG), only
        selected ROIs whose name contains ``'eeg_anulus'`` move. Selected reference points
        move with ``ref_points``. Nothing happens when the layer has no ROIs.
        """
        step = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
        if direction not in step:
            raise ValueError("direction must be 'up', 'down', 'left' or 'right'")
        rois = self.rois[self.current_z] if self.rois else []
        ref_sel = [r.selected for r in self.ref_rois]
        if not rois or not (any(r.selected for r in rois) or any(ref_sel)):
            return Redraw.NONE
        dx, dy = step[direction]
        if "shift" in _mods(modifiers):
            dx, dy = dx * 10, dy * 10
        if self.can_modify_rois:
            to_move = [r for r in rois if r.selected]
        else:
            names = [str(getattr(r, "name", "")) for r in self.rois[0]]
            to_move = [r for i, r in enumerate(rois)
                       if r.selected and i < len(names) and "eeg_anulus" in names[i]]
        for r in to_move:
            r.move(dx, dy)
        if self.ref_rois and any(ref_sel):
            for i, ref in enumerate(self.ref_rois):
                if ref.selected:
                    ref.move(dx, dy)
                    if i < len(self.ref_points):
                        self.ref_points[i] = self.ref_points[i] + (dx, dy)
        return Redraw.OVERLAY

    def expand_selected(self, amount: int = 1) -> Redraw:
        """Grow the selected ROIs by ``amount`` (``bwmorph 'thicken'``), ``clbk.contractExpandRois``."""
        rois = self.rois[self.current_z] if self.rois else []
        if not rois:
            return Redraw.NONE
        for r in rois:
            if r.selected:
                r.expand(amount)
        return Redraw.OVERLAY

    def shrink_selected(self, amount: int = 1) -> Redraw:
        """Shrink the selected ROIs by ``amount`` (``bwmorph 'thin'``), ``clbk.contractExpandRois``."""
        rois = self.rois[self.current_z] if self.rois else []
        if not rois:
            return Redraw.NONE
        for r in rois:
            if r.selected:
                r.shrink(amount)
        return Redraw.OVERLAY

    def selected_indices(self) -> list[int]:
        """0-based indices of the selected ROIs of the current layer."""
        rois = self.rois[self.current_z] if self.rois else []
        return [i for i, r in enumerate(rois) if r.selected]

    def deletion_kind(self) -> str:
        """What deleting the selection means, so a window can ask MATLAB's questions first.

        Returns
        -------
        str
            ``'widefield'`` (refused), ``'none'`` (no ROIs or none selected), ``'acute'``
            (not a chronic series), ``'new_only'`` (only ROIs not saved yet are selected),
            ``'mixed'`` (new and chronically tracked ROIs) or ``'tracked'`` (only tracked
            ROIs).
        """
        if self.is_widefield_data:
            return "widefield"
        rois = self.rois[self.current_z] if self.rois else []
        sel = [r for r in rois if r.selected]
        if not sel:
            return "none"
        if self.orig_adata_full_path is None:
            return "acute"
        new = [r.is_new for r in sel]
        if all(new):
            return "new_only"
        return "mixed" if any(new) else "tracked"

    def delete_selected(self, *, include_tracked: bool = False,
                        add_new_rois: bool | Callable[[ChronicRoiSave], bool] = False) -> Redraw:
        """Delete the selected ROIs, MATLAB ``clbk.deleteCallback``.

        Parameters
        ----------
        include_tracked
            Chronic series only: the user chose "Delete all selected" and then "Delete" in
            MATLAB's two dialogs. The current state is saved first
            (``saveRoisChronic`` without confirmation, ``add_new_rois`` answers its question
            about new ROIs), the selected ROIs are removed from the adata file of every
            ExpID of the series whose layer has as many ROIs as the current layer (written
            with :func:`calliopy.io.update_mat`, after a backup), and the ROIs are reloaded
            from the current file. Without it, a mixed selection deletes only the new ROIs
            ("Delete only new ROIs") and a selection of tracked ROIs only is refused
            (MATLAB's "Cancel").
        add_new_rois
            See :meth:`save`.

        Raises
        ------
        RoiQuestRefused
            Widefield data, no ROIs or none selected, or tracked ROIs without
            ``include_tracked``.

        Notes
        -----
        Differences from MATLAB:

        - a layer whose ROIs are all deleted from a series file is written as ``[]`` (MATLAB
          wrote a 1x0 struct array);
        - unintended behaviour fixed: after deleting tracked ROIs MATLAB reloads the ROIs with
          ``utils.loadRois``, which aborts when the file has no ROIs left, so the deleted ROIs
          stayed on screen and a later save could write them back. Here the session then holds
          empty layers, matching the file.
        """
        kind = self.deletion_kind()
        rois = self.rois[self.current_z] if self.rois else []
        if kind == "widefield":
            raise RoiQuestRefused("You can't delete ROIs for widefield data")
        if kind == "none":
            raise RoiQuestRefused("No ROIs to delete." if not rois
                                  else "No ROIs selected. Not deleting anything.")
        if kind in ("acute", "new_only"):
            self.rois[self.current_z] = [r for r in rois if not r.selected]
            return Redraw.OVERLAY
        if not include_tracked:
            if kind == "mixed":
                self.rois[self.current_z] = [r for r in rois if not (r.selected and r.is_new)]
                return Redraw.OVERLAY
            raise RoiQuestRefused("ROI deletion aborted (the selected ROIs are chronically "
                                  "tracked; pass include_tracked=True to delete them from all "
                                  "timepoints).")
        _persist.delete_tracked(self, self.selected_indices(), add_new_rois)
        return Redraw.OVERLAY

    def tag_selected(self, tag: int = 0) -> Redraw:
        """Set the type (tag 1-9; 0 = untag) of the selected ROIs, ``clbk.tagSelectedRois``."""
        rois = self.rois[self.current_z] if self.rois else []
        if not rois:
            return Redraw.NONE
        idx = self.selected_indices()
        for i in idx:
            rois[i].type = float(tag)
        log.info("Tag for ROI(s) [%s] changed to: %s", ",".join(str(i + 1) for i in idx), tag)
        return Redraw.OVERLAY

    def untag_selected(self) -> Redraw:
        """Remove the tags of the selected ROIs (menu "Untag selected")."""
        return self.tag_selected(0)

    def accept_putative(self) -> Redraw:
        """Add the putative ROIs to the layer, MATLAB ``clbk.acceptPutativeRois``.

        Accepted ROIs are marked new; when exactly one ROI is added it becomes the only
        selected ROI.
        """
        if not self.putative_rois:
            return Redraw.NONE
        for r in self.putative_rois:
            r.is_new = True
        z = self.current_z
        self.rois[z] = list(self.rois[z]) + list(self.putative_rois)
        if len(self.putative_rois) == 1:
            for r in self.rois[z]:
                r.selected = False
            self.rois[z][-1].selected = True
        n = len(self.putative_rois)
        self.putative_rois = []
        log.info("Added %u ROI(s).", n)
        return Redraw.OVERLAY

    def escape(self) -> Redraw:
        """Escape / q key: show the whole image and drop the putative ROIs (``keyParser``)."""
        self.reset_view()
        self.putative_rois = []
        return Redraw.VIEW | Redraw.OVERLAY | Redraw.HISTOGRAM

    def select_all(self) -> Redraw:
        """Select all ROIs of the layer, ``clbk.selectAllNoneRois`` (``roiSelAll``)."""
        for r in (self.rois[self.current_z] if self.rois else []):
            r.selected = True
        return Redraw.OVERLAY

    def select_none(self) -> Redraw:
        """Deselect all ROIs of the layer, ``clbk.selectAllNoneRois`` (``roiSelNone``)."""
        for r in (self.rois[self.current_z] if self.rois else []):
            r.selected = False
        return Redraw.OVERLAY

    def auto_roi_matching(self, modifiers=()) -> Redraw:
        """Move the selected ROIs onto this recording, MATLAB ``clbk.autoRoiMatching``.

        Uses :func:`calliopy.rois.auto_roi_matching` on the filtered display images of this
        and the original recording (``imgs[1][z]``, ``orig_imgs[1][z]``). ``'shift'`` matches
        locally only, ``'control'`` applies only the global shift. The global shift is stored
        in ``roi_trans[z][:2]``, all ROIs are deselected and the movement vectors are kept in
        ``vectors`` (shown once by the renderer).

        Raises
        ------
        RoiQuestRefused
            Widefield data, no ROIs, no original recording, or no ROI selected.
        """
        if self.is_widefield_data:
            raise RoiQuestRefused("You can't auto-match ROIs for widefield data")
        z = self.current_z
        rois = self.rois[z] if self.rois else []
        if not rois:
            raise RoiQuestRefused("No ROIs to correct. ROI correction aborted.")
        if not self.orig_imgs or not self.orig_imgs[0]:
            raise RoiQuestRefused("Not a chronic series. There's no original site loaded. "
                                  "ROI correction aborted.")
        sel = np.array([r.selected for r in rois])
        if not sel.any():
            raise RoiQuestRefused("No ROIs selected for correction. ROI correction aborted.")
        mods = _mods(modifiers)
        res = auto_roi_matching(rois, self.imgs[1][z], self.orig_imgs[1][z], sel,
                                local_only="shift" in mods, global_only="control" in mods)
        for k, oob in enumerate(np.ravel(res.is_out_of_bounds)):
            if oob:
                log.warning("\t- ROI# %u not adjusted. Auto-adjust location was out-of-bounds.",
                            k + 1)
        tr = self.roi_trans[z].copy()
        flat = tr.reshape(-1, order="F")
        flat[:2] = np.ravel(res.roi_trans)[:2]
        self.roi_trans[z] = flat.reshape(tr.shape, order="F")
        self.vectors = (np.asarray(res.start_centroids) - 1.0, np.asarray(res.displacement))
        self.show_vectors = True
        log.info("correction done.")
        return Redraw.OVERLAY

    # ---------------------------------------------------------------------------------
    # navigation
    # ---------------------------------------------------------------------------------
    def cycle_image(self, sign: str) -> Redraw:
        """Next (``'+'``) or previous (``'-'``) image kind, MATLAB ``clbk.cycleThroughImgs``."""
        n = len(self.imgs)
        if sign == "+":
            self.img_idx = (self.img_idx + 1) % n
        elif sign == "-":
            self.img_idx = (self.img_idx - 1) % n
        else:
            raise ValueError("sign must be '+' or '-'")
        return Redraw.DISPLAY | Redraw.HISTOGRAM

    def change_layer(self, direction: str) -> Redraw:
        """Next (``'+'``) or previous (``'-'``) Z layer, MATLAB ``clbk.changePiezoLayer``.

        The ROIs of the layer being left are deselected first (two-photon data), also when
        there is no layer to move to.
        """
        if not self.is_widefield_data and self.rois:
            for r in self.rois[self.current_z]:
                r.selected = False
        if direction == "+":
            if self.current_z < self.num_layers - 1:
                self.current_z += 1
        elif direction == "-":
            if self.current_z > 0:
                self.current_z -= 1
        else:
            raise ValueError("direction must be '+' or '-'")
        return Redraw.DISPLAY | Redraw.OVERLAY | Redraw.HISTOGRAM

    def toggle_original(self) -> Redraw:
        """Show or hide the original recording's image (key ``o``)."""
        self.show_original = not self.show_original
        return Redraw.DISPLAY | Redraw.OVERLAY | Redraw.HISTOGRAM

    def toggle_roi_visible(self) -> Redraw:
        """Show or hide the ROI overlay (space bar)."""
        self.roi_visible = not self.roi_visible
        return Redraw.OVERLAY

    def toggle_numbers(self) -> Redraw:
        """Show or hide the ROI numbers, MATLAB ``clbk.toggleNumbers``."""
        self.roi_number_visible = not self.roi_number_visible
        return Redraw.OVERLAY

    def god_mode_key(self) -> Redraw:
        """Key ``g``, MATLAB ``clbk.godModeCounter``: every fifth press shows or hides the
        filter type and size menus (``enabled['setFiltType']``/``['setFiltSize']``)."""
        self.god_counter += 1
        if self.god_counter == 5:
            self.god_mode = not self.god_mode
            self.god_counter = 0
            log.info("GodMode %s.", "on" if self.god_mode else "off")
            return Redraw.MENUS
        return Redraw.NONE

    # ---------------------------------------------------------------------------------
    # display settings
    # ---------------------------------------------------------------------------------
    def _source_image(self, k: int, z: int, original: bool = False, filter_type=None):
        tmpl = self.orig_template if original else self.template
        if k == 0:
            return tmpl[z]
        if k == 1:
            ft = self.filter_type if filter_type is None else filter_type
            return filter_template(tmpl[z], self.filter_win_size, ft)
        return (self.orig_act_map if original else self.act_map)[z]

    def set_contrast(self, mode: str) -> Redraw:
        """Contrast of the current image kind in all layers, MATLAB ``clbk.setContrast``.

        ``mode`` is ``'minMax'``, ``'bestFit'`` or ``'custom'`` (limits ``img_lo``/``img_hi``
        of the image kind), applied with :func:`calliopy.rois.adjust_img_contrast` to the
        template, the filtered template or the activity map, and to the original
        recording's images.

        Differences from MATLAB: ``contrast_mode`` records the mode (MATLAB never updated it).
        """
        if mode not in _CONTRAST_TEXT:
            raise ValueError("mode must be 'minMax', 'bestFit' or 'custom'")
        k = self.img_idx
        clims = (self.img_lo[k], self.img_hi[k])
        for z in range(self.num_layers):
            self.imgs[k][z] = adjust_img_contrast(self._source_image(k, z), mode, clims)
        if self.orig_adata_full_path is not None:
            for z in range(len(self.orig_template)):
                self.orig_imgs[k][z] = adjust_img_contrast(
                    self._source_image(k, z, original=True), mode, clims)
        self.contrast_mode = mode
        msg = f"Contrast mode for {self.img_names[k]} set to {_CONTRAST_TEXT[mode]}."
        if mode == "custom":
            msg += f" Limits: [{clims[0]:.2f}, {clims[1]:.2f}]"
        log.info(msg)
        return Redraw.DISPLAY | Redraw.HISTOGRAM

    def set_custom_contrast(self, black: float, white: float, *, moved: str = "white") -> Redraw:
        """Custom contrast sliders, MATLAB ``clbk.contrastSliderChanged``.

        Parameters
        ----------
        black, white
            Slider values in ``[0, 1]``.
        moved
            Which slider the user moved. Moving ``'white'`` below black sets white to
            ``min(black + 0.01, 1)``; moving ``'black'`` above white sets black to
            ``max(white - 0.01, 0)``.

        Notes
        -----
        The limits are applied to the current image kind of every layer and stored in
        ``img_lo``/``img_hi``. Kept as in MATLAB (suspected bugs, reported): the filtered
        template is recomputed with the ``'legacy'`` filter whatever ``filter_type`` is, and
        the original recording's images are not updated.
        """
        black, white = float(black), float(white)
        if moved == "white":
            if white < black:
                white = min(black + 0.01, 1.0)
        elif moved == "black":
            if black > white:
                black = max(white - 0.01, 0.0)
        else:
            raise ValueError("moved must be 'white' or 'black'")
        k = self.img_idx
        for z in range(self.num_layers):
            src = self._source_image(k, z, filter_type="legacy")
            self.imgs[k][z] = adjust_img_contrast(src, "custom", (black, white))
        self.img_lo[k] = black
        self.img_hi[k] = white
        log.info("Contrast mode for %s set to Custom. Limits:[%0.2f, %0.2f]",
                 self.img_names[k], black, white)
        return Redraw.DISPLAY | Redraw.HISTOGRAM

    def set_colormap(self, name: str) -> Redraw:
        """Image colormap (``clbk.setImageColormap``); see :func:`calliopy.roiquest.colormap`."""
        self.cmap = colormap(name)
        log.info("Selected colormap: %s", name)
        return Redraw.COLORMAP

    def invert_colormap(self) -> Redraw:
        """Reverse the image colormap (menu "Invert colormap")."""
        self.cmap = self.cmap[::-1].copy()
        log.info("Inverted colormap.")
        return Redraw.COLORMAP

    def _refilter(self) -> None:
        for z in range(self.num_layers):
            f = filter_template(self.template[z], self.filter_win_size, self.filter_type)
            self.imgs[1][z] = adjust_img_contrast(mat2gray(f), "bestFit")
        if self.orig_adata_full_path is not None:
            for z in range(len(self.orig_template)):
                f = filter_template(self.orig_template[z], self.filter_win_size,
                                    self.filter_type)
                self.orig_imgs[1][z] = adjust_img_contrast(mat2gray(f), "bestFit")

    def set_filter_type(self, filter_type: str) -> Redraw:
        """``'legacy'`` or ``'updated'`` template filter, MATLAB ``clbk.setFilterType``.

        The filtered images (also the original's) are recomputed with ``'bestFit'`` contrast.
        """
        if filter_type not in ("legacy", "updated"):
            raise ValueError("filter_type must be 'legacy' or 'updated'")
        self.filter_type = filter_type
        self._refilter()
        log.info('Filter type changed to %s. Contrast set to "Best Fit"', filter_type)
        return Redraw.DISPLAY | Redraw.HISTOGRAM

    def set_filter_size(self, size) -> Redraw:
        """Template filter radius, MATLAB ``clbk.setFilterSize``.

        ``size`` is rounded (MATLAB ``round(str2num(answer))``) and must be within 1-100
        (``ValueError`` otherwise, MATLAB logged "Invalid filter size"). The filtered images
        are recomputed with ``'bestFit'`` contrast.
        """
        try:
            v = float(np.ravel(np.asarray(size, dtype=np.float64))[0])
        except (TypeError, ValueError, IndexError):
            raise ValueError("Input filter size not valid.") from None
        n = mround(v)
        if not (1 <= n <= 100):
            raise ValueError("Invalid filter size. (1-100)")
        self.filter_win_size = int(n)
        self._refilter()
        log.info('Filter size changed to %u. Contrast set to "Best Fit"', self.filter_win_size)
        return Redraw.DISPLAY | Redraw.HISTOGRAM

    def set_opacity(self, value: float) -> Redraw:
        """Overlay opacity in ``[0, 1]``, MATLAB ``clbk.changeOpacity``."""
        v = float(value)
        if not (0.0 <= v <= 1.0):
            raise ValueError("Invalid opacity. (0-1)")
        self.ovrl_opacity = v
        log.info("ROIs opacity changed to %.2f.", v)
        return Redraw.OVERLAY

    @staticmethod
    def _rgb(color) -> tuple[float, float, float]:
        c = np.asarray(color, dtype=np.float64).ravel()
        if c.size != 3 or np.any(c < 0) or np.any(c > 1):
            raise ValueError("color must be three values in [0, 1]")
        return float(c[0]), float(c[1]), float(c[2])

    def set_roi_color(self, color) -> Redraw:
        """Flat ROI colour (``clbk.changeRoiColor``, ``chngRoiCol``)."""
        self.roi_bg_color = self._rgb(color)
        return Redraw.OVERLAY

    def set_selection_color(self, color) -> Redraw:
        """Colour of putative ROIs (``clbk.changeRoiColor``, ``chngRoiSelCol``)."""
        self.roi_sel_bg_color = self._rgb(color)
        return Redraw.OVERLAY

    def set_color_mode(self, mode: str) -> Redraw:
        """``'flat'``, ``'random'`` or ``'tags'``, MATLAB ``clbk.changeColorMode``."""
        if mode not in ("flat", "random", "tags"):
            raise ValueError("mode must be 'flat', 'random' or 'tags'")
        self.roi_color_mode = mode
        log.info("Roi color mode set to: %s", mode)
        return Redraw.OVERLAY

    def set_info_position(self, pos: str) -> Redraw:
        """Corner of the info text, ``'tl'``, ``'bl'``, ``'tr'`` or ``'br'`` (``clbk.changeInfoPos``)."""
        if pos not in ("tl", "bl", "tr", "br"):
            raise ValueError("pos must be 'tl', 'bl', 'tr' or 'br'")
        self.info_pos = pos
        return Redraw.OVERLAY

    # ---------------------------------------------------------------------------------
    # widefield
    # ---------------------------------------------------------------------------------
    def wf_add_ref_points(self, points) -> Redraw:
        """Define the two reference points (e.g. lambda, bregma), MATLAB ``clbk.wfAddRefPoints``.

        Parameters
        ----------
        points
            Two 0-based ``(row, col)`` positions (MATLAB took them with two left clicks),
            rounded half away from zero.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape != (2, 2):
            raise ValueError("points must be two (row, col) pairs")
        hw = np.asarray(self.template[0]).shape[:2]
        pts = np.array([[mround(v) for v in p] for p in pts], dtype=np.float64)
        self.ref_points = pts
        self.ref_rois = [Roi(make_cross(p.astype(int), hw), hw, rng=self.rng) for p in pts]
        log.info("Reference points selected.")
        return Redraw.OVERLAY

    def roi_template_files(self) -> list[str]:
        """File names in ``roi_template_dir`` (the list of MATLAB's template dialog)."""
        if self.roi_template_dir is None:
            raise ValueError("no roi_template_dir given")
        return sorted(p.name for p in self.roi_template_dir.iterdir() if p.is_file())

    def _tform_ch_map(self, tform_ch_map=None) -> np.ndarray:
        if tform_ch_map is not None:
            return np.asarray(tform_ch_map, dtype=np.float64)
        mat = _load.tform_matrix(self.tforms_2ch.get("ch_map"))
        if mat is None:
            raise RoiQuestRefused("The channel map transform (tform_ch_map) of this recording is "
                                  "a MATLAB object calliopy cannot read; pass it as a 3x3 "
                                  "matrix (tform_ch_map=...)")
        return mat

    def wf_add_rois(self, template, *, tform_ch_map=None) -> Redraw:
        """Create the widefield ROIs from a ROI template, MATLAB ``clbk.wfAddRois``.

        The template's ROIs are mapped onto the image with the reference points
        (:func:`calliopy.widefield.transform_rois_from_template`) and replace all ROIs; for
        two-channel data the ROIs are drawn on the current channel and mapped onto the other
        with the channel map (:meth:`calliopy.rois.RoiWf.transform_green2red` /
        ``transform_red2green``). ``ROItemplate_parameters`` becomes the template without
        ``ROIs`` and ``img_ROIs``. The ring template
        (:data:`EEG_RING_TEMPLATE_FILE`) allows moving its ROIs and writes ``is_eeg = true``
        into the adata file (after a backup), as MATLAB does.

        Parameters
        ----------
        template
            File name in ``roi_template_dir`` or path of a template file.
        tform_ch_map
            Two-channel data: 3x3 channel map matrix (MATLAB ``tform.A``) when the adata
            transform object cannot be read.

        Raises
        ------
        RoiQuestRefused
            No reference points.

        Notes
        -----
        Differences from MATLAB: the ROIs are replaced only when the transformation
        succeeds (MATLAB deleted them first and kept none after an error).
        """
        from calliopy.widefield import load_roi_template, transform_rois_from_template

        if not self.ref_rois or len(self.ref_points) < 2:
            raise RoiQuestRefused("There are no reference points to create ROIs. Please define "
                                  "reference points first.")
        path = Path(template)
        if self.roi_template_dir is not None and (self.roi_template_dir / path).is_file():
            path = self.roi_template_dir / path
        struct = load_roi_template(path)
        params = type(struct)((k, v) for k, v in struct.items() if k not in ("img_ROIs", "ROIs"))
        self.roi_template_parameters = params
        hw = np.asarray(self.template[0]).shape[:2]
        size_t = np.ravel(np.asarray(struct["parameters"]["SIZE_TEMPLATE"], dtype=np.float64))
        p1 = (self.ref_points[0, 1] + 1, self.ref_points[0, 0] + 1)
        p2 = (self.ref_points[1, 1] + 1, self.ref_points[1, 0] + 1)
        transformed = transform_rois_from_template(struct["ROIs"], tuple(int(v) for v in size_t),
                                                   hw, p1, p2)
        if self.is_wf_two_channels:
            a = self._tform_ch_map(tform_ch_map)
            same = struct2wf_rois(transformed, hw, rng=self.rng)
            other = struct2wf_rois(transformed, hw, rng=self.rng)
            for r in other:
                if self.current_z == 0:
                    r.transform_green2red(a)
                else:
                    r.transform_red2green(a)
            new = [same, other] if self.current_z == 0 else [other, same]
        else:
            new = [struct2wf_rois(transformed, hw, rng=self.rng)]
        self.rois = new
        if path.name == EEG_RING_TEMPLATE_FILE:
            self.can_modify_rois = True
            _persist.write(self, self.adata_full_path, {"is_eeg": True})
            log.info('A variable "is_eeg" was saved to the Adata file:\n\t- %s',
                     self.adata_full_path)
        log.info("ROIs selected.")
        return Redraw.OVERLAY

    # ---------------------------------------------------------------------------------
    # Cellpose
    # ---------------------------------------------------------------------------------
    def load_cellpose_model(self, model_path) -> None:
        """Load a Cellpose model file, MATLAB ``clbk.loadModel`` / ``utils.loadCellPose``.

        Cellpose is optional: when the ``cellpose`` package is not installed this raises
        ``ImportError`` and the window disables the menu (see :func:`cellpose_available`).
        """
        try:
            from cellpose import models
        except ImportError as err:
            raise ImportError("Cellpose is not installed; automatic ROI detection is not "
                              "available") from err
        self.cell_pose_model = models.CellposeModel(pretrained_model=os.fspath(model_path))

    def cellpose_detect(self, model=None, *, diameter: float = 25) -> Redraw:
        """Detect ROIs with Cellpose, MATLAB ``clbk.autodetectCellpose`` + ``gfx.cellposeDetect``.

        The current display image is segmented (``model.eval([img], diameter=25,
        channels=[0, 0])``); labels touching the invalid area, overlapping existing ROIs or
        smaller than ``min_roi_pix`` are rejected (:func:`calliopy.rois.filter_cellpose_labels`)
        and the others are appended to the putative ROIs (accept them with
        :meth:`accept_putative`).

        Parameters
        ----------
        model
            Any object with Cellpose's ``eval`` method; default ``cell_pose_model``.
        diameter
            Expected cell diameter in pixels.
        """
        model = self.cell_pose_model if model is None else model
        if model is None:
            raise RoiQuestRefused("No Cellpose model loaded (load_cellpose_model).")
        log.info("Autodetecting ROIs...")
        z = self.current_z
        img = np.asarray(self.imgs[self.img_idx][z], dtype=np.uint8)
        out = model.eval([img], diameter=diameter, channels=[0, 0])
        labels = np.asarray(out[0][0])
        labels = np.clip(np.round(labels), 0, 65535).astype(np.uint16)
        rois = self.rois[z]
        res = filter_cellpose_labels(
            labels, None if self.valid_area is None else self.valid_area[:, :, z],
            rois_binary_mask(rois) if rois else None, self.min_roi_pix)
        hw = labels.shape
        for ind in res.indices:
            self.putative_rois.append(Roi(ind, hw, rng=self.rng))
        log.info("Done. \n\t- %u Total ROIs detected by Cellpose\n\t- %u ROIs rejected (too "
                 "small, on borders, overlapping existing ones)\n\t- %u valid putative ROIs. "
                 "Press A to add them.", res.num_detected, res.num_removed,
                 len(self.putative_rois))
        return Redraw.OVERLAY

    # ---------------------------------------------------------------------------------
    # activity, saving, texts, keys
    # ---------------------------------------------------------------------------------
    def activity_selection(self) -> tuple[list[Roi], list[int]]:
        """Selected ROIs and their 0-based indices for ActiQuest, MATLAB ``clbk.viewActivity``.

        Raises
        ------
        RoiQuestRefused
            No ROI selected, or some selected ROIs have no activity trace (numbered from 1 in
            the message, as in MATLAB).
        """
        idx = self.selected_indices()
        if not idx:
            raise RoiQuestRefused("You did not select any ROI for visualizing activity. Please "
                                  "select at least one ROI.")
        rois = self.rois[self.current_z]
        lacking = [i for i in idx if rois[i].activity is None or np.size(rois[i].activity) == 0]
        if lacking:
            raise RoiQuestRefused(f"ROIs #[{','.join(str(i + 1) for i in lacking)}] do not have "
                                  "activity traces! Visualization Aborted.")
        return [rois[i] for i in idx], idx

    def save(self, *, overwrite_activity: bool = False,
             add_new_rois: bool | Callable[[ChronicRoiSave], bool] = False,
             overwrite_policy: str | None = None, tforms: Mapping | None = None) -> SaveResult:
        """Save the ROIs into the adata file(s), MATLAB ``clbk.saveToAdata``.

        Before a file is written for the first time in this session, a backup copy is made
        next to it (``<name>.roiquest_backup_<date-time>``). Values are written with
        :func:`calliopy.io.update_mat`, which keeps all other variables of the file.

        - Two-photon, not a chronic series (or the original of one): ``ROIs`` and
          ``ROItrans`` (:func:`calliopy.rois.save_rois_acute`). The ROIs are then marked as
          not new.
        - Two-photon chronic series: :func:`calliopy.rois.save_rois_chronic` matches the
          ROIs against the original and every analysed timepoint (needs ``exp_log`` and
          ``adata_dir``); ``ROIs``/``ROItrans`` of this file are written; new ROIs are added
          to all timepoints when ``add_new_rois`` is true or returns true.
        - Widefield: :func:`calliopy.rois.wf_save_rois` (single recording, or propagation to
          the series with ``overwrite_policy``).

        Parameters
        ----------
        overwrite_activity
            Answer to "Overwriting ROI activity traces?" (default No).
        add_new_rois
            Chronic series with new ROIs: the answer to "Save these ROIs?", or a callable that
            receives the :class:`calliopy.rois.ChronicRoiSave` (e.g. to show
            :func:`calliopy.rois.plot_rois_to_add`) and returns the answer. Default No.
        overwrite_policy
            Widefield series: ``'keep'`` ("Propagate without overwriting") or
            ``'overwrite'`` ("Propagate and overwrite").
        tforms
            Widefield series: ExpID -> 3x3 ``tform`` matrix, or mapping with
            ``'tform_ch525'``/``'tform_ch610'``, for files whose transform objects calliopy
            cannot read.

        Returns
        -------
        SaveResult

        Raises
        ------
        calliopy.rois.RoiSaveError
            MATLAB refused to save (activity traces without ``overwrite_activity``, ROI
            mismatch in a chronic series, ...).
        RoiQuestRefused
            A chronic series without ``exp_log``/``adata_dir``.
        ImportError
            :func:`calliopy.io.update_mat` is not available.

        Notes
        -----
        The confirmation dialog MATLAB shows first is the window's job.
        """
        if self.adata_full_path is None:
            raise RoiQuestRefused("No AData file loaded.")
        return _persist.save(self, overwrite_activity=overwrite_activity,
                             add_new_rois=add_new_rois, overwrite_policy=overwrite_policy,
                             tforms=tforms)

    def histogram_data(self):
        """Histogram of the visible image region; see :func:`calliopy.roiquest.histogram_data`."""
        from ._display import histogram_data

        return histogram_data(self)

    def overlay_text(self, n_pixels=None, free_form_msg=None) -> str:
        """Info text on the image; see :func:`calliopy.roiquest.overlay_text`."""
        from ._display import overlay_text

        return overlay_text(self, n_pixels, free_form_msg)

    def info_text(self) -> str:
        """State report of the Help > Print Info menu, MATLAB ``clbk.printInfo``."""
        from ._text import info_text

        return info_text(self)

    @staticmethod
    def help_text() -> str:
        """Key and mouse help of the Help > Print all commands menu (``clbk.printHelp``)."""
        from ._text import help_text

        return help_text()

    def key_press(self, key: str, modifiers: Sequence[str] = ()):
        """Handle a key press as MATLAB's ``clbk.keyParser`` and the menu accelerators;
        see :func:`calliopy.roiquest.key_press`."""
        from ._keys import key_press

        return key_press(self, key, modifiers)
