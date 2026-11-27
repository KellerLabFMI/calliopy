"""Cell browser for project meta files: calliope ``20_MetaData/cell_browser.m``.

:class:`CellBrowserSession` holds the state of MATLAB's ``params`` (position in the
proj_meta, selected channels, zoom, annotations) and has one method per callback;
:class:`CellBrowserView` draws it on a matplotlib figure; the tkinter window is in
``_tk_cell_browser.py``.

Positions (site, layer, timepoint, cell) are **0-based** in the session; the annotations
file keeps MATLAB's 1-based values, so files written by either language are
interchangeable.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from matplotlib.figure import Figure

from calliopy._matlab import mround
from calliopy.io import MatStruct, save_mat
from calliopy.projmeta import cell_browser_good_cells

from ._cb_plugins import PLUGINS
from ._common import hsv, num2str, read_mat_variable

__all__ = ["CELL_BROWSER_KEYS", "CellBrowserSession", "CellBrowserView"]

log = logging.getLogger("calliopy.viewers")

HIDDEN_RD_FIELDS = ("template", "act_map", "ROIinfo", "timepoint", "piezo_layer", "fnames", "nbr_frames")
"""``rd`` fields MATLAB leaves out of the channel lists."""

N_CHANNEL_ROWS = 5
MATLAB_NZLAY = 4
DEFAULT_TA_WIN = 50
DEFAULT_WIN_POS = (100.0, 100.0, 1200.0, 700.0)

CELL_BROWSER_KEYS = {
    "right / left": "next / previous cell, or good cell with good_cells (continues in the neighbouring layer)",
    "shift + right / left": "next / previous layer (closest valid cell)",
    "alt + right / left": "next / previous site (closest valid cell)",
    "up / down": "next / previous timepoint (closest valid cell)",
    "space": "zoom: click twice on the traces to set the time window; again to zoom out",
    "s": "activity around stimulus onsets (figure 10)",
    "v": "VML plugin (cb_plugin_VML)",
    "m": "M1 plugin (cb_plugin_M1)",
    "a": "ACX plugin (cb_plugin_ACX)",
    "t": "traversal plugin (cb_plugin_trav)",
}
"""Keys of the cell browser (MATLAB help text plus the plugin keys)."""

_KEY_ALIASES = {"right": "rightarrow", "left": "leftarrow", "up": "uparrow", "down": "downarrow",
                " ": "space"}
_PLUGIN_KEYS = {"s": "stim", "v": "vml", "m": "m1", "a": "acx", "t": "trav"}
_MISSING_PLUGINS = {"g": "cb_plugin_screenshot", "l": "cb_plugin_LFM"}


def _sites(proj_meta) -> list:
    from calliopy.projmeta._access import _sites as sites

    return sites(proj_meta)


def _rd(site) -> np.ndarray:
    from calliopy.projmeta._access import _rd as rd

    return rd(site)


def _elements(sa) -> list:
    from calliopy.projmeta._struct import elements

    return elements(sa)


def _scalar(v, default=None):
    a = np.asarray(v)
    if a.size == 0 or a.dtype == object:
        return default
    return float(np.ravel(a)[0])


def _str(v) -> str | None:
    if isinstance(v, str):
        return v
    a = np.asarray(v)
    if a.dtype.kind in "US":
        return "".join(str(s) for s in np.ravel(a))
    return None


def _normalise_modifiers(modifiers) -> frozenset[str]:
    out = set()
    for m in modifiers or ():
        m = str(m).lower()
        out.add("control" if m in ("ctrl", "control") else m)
    return frozenset(out)


class CellBrowserSession:
    """State and callbacks of the cell browser (``cell_browser.m``).

    Parameters
    ----------
    proj_meta : struct array
        The proj_meta (``(1, nSites)`` object array of :class:`~calliopy.io.MatStruct` as
        :func:`calliopy.io.load_metafile` returns it, or a list of structs).
    cdisp : {0, 1, 2}, default 1
        Thumbnail of the ROI: 0 off, 1 the template around the ROI centre (cells), 2 the ROI
        filled red on the template, cropped to its bounding box (axons).
    good_cells : bool, default False
        Browse only the cells with the largest dynamic range
        (:func:`calliopy.projmeta.cell_browser_good_cells`); MATLAB's third argument.
    annotations_file : path, optional
        Annotations file (MATLAB ``<adata_dir>_CBannotations/<projID>_<acode>-cba.mat``).
        When it exists, the browser resumes from it; :meth:`save_annotations` writes it.
    comments : mapping, optional
        ExpID -> comment shown next to "Comment:" (MATLAB looks it up in ExpLog).

    Attributes
    ----------
    site, layer, tp, cell : int
        **0-based** position (MATLAB ``params.site_id``, ``zl``, ``tp``, ``cell_ind``).
    channels : list of 5 lists of int
        **0-based** indices into :attr:`rdfields` per channel row; row 0 is drawn at the
        bottom (MATLAB ``chansToPlot``).
    zoom : bool
        Zoomed in on :attr:`zoom_xlim` (MATLAB ``zon``, ``zxlim``; x in samples, 1-based
        plot coordinates as MATLAB draws them).
    site_annotations : list of str or None
        Per site; None where nothing was entered (``[]`` in the file).
    cell_annotations : dict
        ``{(site, layer, cell): str}`` with **0-based** keys; cells without an entry are
        ``[]`` in the file.

    Differences from MATLAB
    -----------------------
    - Layers run over the rows of ``rd`` of the current site, so every layer of the file is
      reachable (MATLAB: always 4, and moving to a layer or timepoint a site does not have
      stopped the browser with an error); after a move, layer and timepoint are clamped to
      the site's ``rd``.
    - Navigation follows one rule (MATLAB's behaviour differed, see Notes):

      * right / left with ``good_cells``: the next / previous good cell of the layer; past
        the last / first one, the first / last good cell of the next / previous layer that
        has one (layers of the site, then the following / preceding sites at the same
        timepoint, clamped); nothing changes when there is none;
      * right / left without ``good_cells``: as MATLAB (next / previous cell, continuing
        at the first / last cell of the neighbouring layer of the same site);
      * shift + arrows (layer), alt + arrows (site; layer and timepoint go to the first, as
        in MATLAB) and up / down (timepoint) keep the closest valid cell:
        :meth:`closest_cell` (nearest good cell, the lower one on a tie; without good
        cells, or where a position has none, the same cell index clipped to the cells
        there). MATLAB went to cell 1 on layer and site changes and kept the index,
        possibly invalid, on timepoint changes.
    - Every trace is drawn on its own: one failing channel does not stop the others
      (MATLAB's single ``try`` skips all channels after the first error).
    - The ``g`` (``cb_plugin_screenshot``) and ``l`` (``cb_plugin_LFM``) keys log that the
      plugins are not in calliope r2003 (MATLAB fails); ``m`` runs ``cb_plugin_M1`` (see
      :func:`plot_cb_plugin_m1`).
    - The proj_meta is passed in (no fetching from the base workspace); the ExpLog lookups
      (project, analysis code, comment, annotations folder) are replaced by
      ``annotations_file`` and ``comments``.

    The annotations file is unchanged: same variable, fields, classes and 1-based values as
    MATLAB writes (see :meth:`params_struct`), MATLAB's default v7 format for new files.

    Notes
    -----
    MATLAB r2003 with ``good_cells``: the left key jumped to the *first* good cell before
    the current one (``next(1)``), and a layer or site change landed on cell 1
    (``max([1 good_cells{...}])`` is always 1), which need not be a good cell.
    """

    def __init__(self, proj_meta, *, cdisp: int = 1, good_cells: bool = False,
                 annotations_file: str | PathLike | None = None,
                 comments: Mapping[Any, str] | None = None) -> None:
        self.sites = _sites(proj_meta)
        if not self.sites:
            raise ValueError("proj_meta has no sites")
        if int(cdisp) not in (0, 1, 2):
            raise ValueError("cdisp must be 0, 1 or 2")
        self.cdisp = int(cdisp)
        self.good: dict | None = None
        if good_cells:
            log.info("finding cells with largest dynamic range (>90th prcentile)")
            self.good = cell_browser_good_cells(proj_meta)[0]
        self.comments = dict(comments) if comments is not None else {}
        first = self.sites[0]
        self.title = f"{num2str(first.get('projID', ''))}_{num2str(first.get('Acode', ''))}"
        self.annotations_file = Path(annotations_file) if annotations_file is not None else None

        self.max_nbr_cells = 0
        for site in self.sites:
            rd = _rd(site)
            nl = int(_scalar(site.get("nbr_piezo_layers", rd.shape[0]), rd.shape[0]))
            for z in range(min(nl, rd.shape[0])):
                if rd.shape[1]:
                    self.max_nbr_cells = max(self.max_nbr_cells, self._n_rows(rd[z, 0]))

        self.site = self.layer = self.tp = self.cell = 0
        self.zoom = False
        self.zoom_xlim: tuple[float, float] | None = None
        self.pending_zoom: list[float] | None = None
        rd0 = _rd(self.sites[0])
        names = list(rd0[0, 0].keys()) if rd0.size and isinstance(rd0[0, 0], Mapping) else []
        self.rdfields: list[str] = [n for n in names if n not in HIDDEN_RD_FIELDS]
        self.channels: list[list[int]] = [[i] if i < len(self.rdfields) else [] for i in range(N_CHANNEL_ROWS)]
        self.site_annotations: list[str | None] = [None] * len(self.sites)
        self.cell_annotations: dict[tuple[int, int, int], str] = {}
        self._loaded_params: MatStruct | None = None
        self._ca_shape: tuple[int, int, int] = (0, 0, 0)
        self._zon_logical = False
        self.ta_win = DEFAULT_TA_WIN
        self.colors = hsv(256)[::6]
        self.win_pos: tuple[float, ...] = DEFAULT_WIN_POS
        if self.annotations_file is not None and self.annotations_file.is_file():
            self.load_annotations(self.annotations_file)

    # ---------------------------------------------------------------------------------
    # proj_meta access
    # ---------------------------------------------------------------------------------
    @staticmethod
    def _n_rows(entry) -> int:
        if not isinstance(entry, Mapping) or "act" not in entry:
            return 0
        a = np.asarray(entry["act"])
        return int(a.shape[0]) if a.ndim == 2 else 0

    def n_sites(self) -> int:
        return len(self.sites)

    def n_layers(self, site: int | None = None) -> int:
        """Rows of ``rd`` of a site (current site by default)."""
        return int(_rd(self.sites[self.site if site is None else site]).shape[0])

    def n_timepoints(self, site: int | None = None) -> int:
        """Columns of ``rd`` of a site (current site by default)."""
        return int(_rd(self.sites[self.site if site is None else site]).shape[1])

    def rd_entry(self, site: int | None = None, layer: int | None = None, tp: int | None = None):
        """``proj_meta(site).rd(layer, tp)`` (0-based; current position by default) or None."""
        s = self.site if site is None else site
        z = self.layer if layer is None else layer
        t = self.tp if tp is None else tp
        if not 0 <= s < len(self.sites):
            return None
        rd = _rd(self.sites[s])
        if not (0 <= z < rd.shape[0] and 0 <= t < rd.shape[1]):
            return None
        e = rd[z, t]
        return e if isinstance(e, Mapping) else None

    def n_cells(self, site: int | None = None, layer: int | None = None, tp: int | None = None) -> int:
        """Rows of ``act`` at a position (0 when there is none)."""
        return self._n_rows(self.rd_entry(site, layer, tp))

    def _good(self, site: int, tp: int, layer: int) -> np.ndarray:
        g = self.good.get((site, tp, layer)) if self.good is not None else None
        return np.flatnonzero(np.asarray(g, dtype=bool)) if g is not None else np.zeros(0, np.intp)

    def _clamp(self) -> None:
        self.tp = max(0, min(self.tp, self.n_timepoints() - 1))
        self.layer = max(0, min(self.layer, self.n_layers() - 1))
        self.cell = max(self.cell, 0)

    # ---------------------------------------------------------------------------------
    # callbacks
    # ---------------------------------------------------------------------------------
    def key_press(self, key: str, modifiers: Sequence[str] = ()) -> str | None:
        """MATLAB ``cb_kpf``: apply a key press.

        Parameters
        ----------
        key
            MATLAB key name (``'rightarrow'``, ``'space'``, ``'s'``, ...) or matplotlib's
            (``'right'``, ``' '``).
        modifiers
            Held modifiers (``'shift'``, ``'alt'``, ``'control'``/``'ctrl'``). As in MATLAB,
            ``alt`` and ``shift`` act only when they are the only modifier.

        Returns
        -------
        str or None
            ``'redraw'``; ``'pick_zoom'`` (collect two clicks with :meth:`add_zoom_click`);
            ``'plugin:<name>'`` (``stim``, ``vml``, ``m1``, ``acx``, ``trav``: show
            :meth:`plugin_figures`); None when nothing changed.
        """
        k = _KEY_ALIASES.get(key, key)
        mods = _normalise_modifiers(modifiers)
        if k == "rightarrow":
            self._right(mods)
            return "redraw"
        if k == "leftarrow":
            self._left(mods)
            return "redraw"
        if k == "uparrow":
            if self.tp < self.n_timepoints() - 1:
                self._move_to(self.site, self.layer, self.tp + 1)
            return "redraw"
        if k == "downarrow":
            if self.tp > 0:
                self._move_to(self.site, self.layer, self.tp - 1)
            return "redraw"
        if k == "space":
            return self.toggle_zoom()
        if k in _PLUGIN_KEYS:
            return f"plugin:{_PLUGIN_KEYS[k]}"
        if k in _MISSING_PLUGINS:
            log.warning("%s is not part of calliope r2003; nothing to show", _MISSING_PLUGINS[k])
        return None

    def closest_cell(self, site: int, layer: int, tp: int, cell: int) -> int:
        """The valid cell nearest to ``cell`` at a position (all **0-based**).

        With ``good_cells``: the good cell with the smallest ``|good - cell|`` (the lower one on
        a tie); when the position has no good cell, or without ``good_cells``: ``cell`` clipped
        to ``0 .. nCells-1`` (0 when the position has no cells).
        """
        n = self.n_cells(site, layer, tp)
        if self.good is not None:
            g = self._good(site, tp, layer)
            g = g[g < n]
            if g.size:
                return int(g[np.argmin(np.abs(g - cell))])
        return int(min(max(cell, 0), max(n - 1, 0)))

    def _positions(self, forward: bool):
        """(site, layer, tp) after (forward) or before the current one: layers, then sites."""
        tp = self.tp
        if forward:
            for z in range(self.layer + 1, self.n_layers()):
                yield self.site, z, tp
            for s in range(self.site + 1, self.n_sites()):
                t = min(tp, self.n_timepoints(s) - 1)
                for z in range(self.n_layers(s)):
                    yield s, z, t
        else:
            for z in range(self.layer - 1, -1, -1):
                yield self.site, z, tp
            for s in range(self.site - 1, -1, -1):
                t = min(tp, self.n_timepoints(s) - 1)
                for z in range(self.n_layers(s) - 1, -1, -1):
                    yield s, z, t

    def _step_good(self, forward: bool) -> None:
        """Next (previous) good cell; past the end of a layer the first (last) good cell of the
        next (previous) layer that has one, across sites. Nothing changes when there is none."""
        g = self._good(self.site, self.tp, self.layer)
        cand = g[g > self.cell] if forward else g[g < self.cell]
        if cand.size:
            self.cell = int(cand[0] if forward else cand[-1])
            return
        for s, z, t in self._positions(forward):
            g = self._good(s, t, z)
            g = g[g < self.n_cells(s, z, t)]
            if g.size:
                self.site, self.layer, self.tp = s, z, t
                self.cell = int(g[0] if forward else g[-1])
                return

    def _move_to(self, site: int, layer: int, tp: int) -> None:
        """Change position and keep the closest valid cell."""
        self.site, self.layer, self.tp = site, layer, tp
        self._clamp()
        self.cell = self.closest_cell(self.site, self.layer, self.tp, self.cell)

    def _right(self, mods) -> None:
        if mods == {"alt"}:
            if self.site < self.n_sites() - 1:
                self._move_to(self.site + 1, 0, 0)
        elif mods == {"shift"}:
            if self.layer < self.n_layers() - 1:
                self._move_to(self.site, self.layer + 1, self.tp)
        elif self.good is not None:
            self._step_good(True)
        else:
            self.cell += 1
            if self.cell > self.n_cells() - 1:
                if self.layer < self.n_layers() - 1:
                    self.cell = 0
                    self.layer += 1
                else:
                    self.cell -= 1
        self._clamp()

    def _left(self, mods) -> None:
        if mods == {"alt"}:
            if self.site > 0:
                self._move_to(self.site - 1, 0, 0)
        elif mods == {"shift"}:
            if self.layer > 0:
                self._move_to(self.site, self.layer - 1, self.tp)
        elif self.good is not None:
            self._step_good(False)
        else:
            self.cell -= 1
            if self.cell < 0:
                if self.layer > 0:
                    self.layer -= 1
                    self.cell = max(self.n_cells() - 1, 0)
                else:
                    self.cell += 1
        self._clamp()

    def toggle_zoom(self) -> str:
        """Space key: zoom out, or start collecting the two clicks of the zoom window."""
        self.zoom = not self.zoom
        self._zon_logical = True  # MATLAB params.zon = ~params.zon makes it logical
        if self.zoom:
            self.pending_zoom = []
            return "pick_zoom"
        self.pending_zoom = None
        return "redraw"

    def add_zoom_click(self, x: float) -> bool:
        """One click of the zoom window (x in plot coordinates); True after the second."""
        if self.pending_zoom is None:
            return False
        self.pending_zoom.append(float(x))
        if len(self.pending_zoom) < 2:
            return False
        self.zoom_xlim = (self.pending_zoom[0], self.pending_zoom[1])
        self.pending_zoom = None
        return True

    def cancel_zoom(self) -> None:
        """Abandon a zoom selection (MATLAB's ``ginputc`` ended early fails)."""
        self.pending_zoom = None
        self.zoom = False

    def set_zoom(self, x0: float, x1: float) -> None:
        """Zoom on samples ``x0 .. x1`` (plot coordinates) without clicking."""
        self.zoom = True
        self._zon_logical = True
        self.pending_zoom = None
        self.zoom_xlim = (float(x0), float(x1))

    def set_channels(self, row: int, field_indices: Sequence[int]) -> None:
        """Selection of channel row ``row`` (0-based, 0 = bottom): 0-based field indices."""
        if not 0 <= row < N_CHANNEL_ROWS:
            raise IndexError("channel row must be 0..4")
        idx = [int(i) for i in field_indices]
        if any(not 0 <= i < len(self.rdfields) for i in idx):
            raise IndexError("field index outside rdfields")
        self.channels[row] = idx

    def channel_fields(self) -> list[list[str]]:
        """Field names of the five channel rows."""
        return [[self.rdfields[i] for i in row] for row in self.channels]

    def set_site_annotation(self, text: str) -> None:
        """MATLAB ``site_anno_cb``: store the site annotation when the text changed.

        As MATLAB's edit-box callback, nothing is stored when the text equals what is shown,
        so sites that were only visited keep ``[]`` in the file.
        """
        if str(text) != self.site_annotation():
            self.site_annotations[self.site] = str(text)

    def set_cell_annotation(self, text: str) -> None:
        """MATLAB ``cell_anno_cb``: store the cell annotation when the text changed."""
        if str(text) != self.cell_annotation():
            self.cell_annotations[(self.site, self.layer, self.cell)] = str(text)

    def site_annotation(self) -> str:
        return self.site_annotations[self.site] or ""

    def cell_annotation(self) -> str:
        return self.cell_annotations.get((self.site, self.layer, self.cell), "")

    def plugin_figures(self, name: str) -> list[Figure]:
        """Figures of a plugin for the current cell (``stim``, ``acx``, ``m1``, ``vml``, ``trav``)."""
        rd = self.rd_entry()
        if rd is None:
            raise IndexError("no rd entry at the current position")
        fn = PLUGINS[name]
        out = fn(rd, self.cell, self.channel_fields()) if name in ("m1", "vml") else fn(rd, self.cell)
        return out if isinstance(out, list) else [out]

    # ---------------------------------------------------------------------------------
    # what the view shows (MATLAB plot_all)
    # ---------------------------------------------------------------------------------
    def xlim(self) -> tuple[float, float]:
        """Time axis: the zoom window, else ``[0, sum(nbr_frames)]``."""
        if self.zoom and self.zoom_xlim is not None:
            a, b = sorted(self.zoom_xlim)
            return a, b
        e = self.rd_entry()
        nbf = np.ravel(np.asarray(e.get("nbr_frames", 0), dtype=np.float64)) if e is not None else np.zeros(0)
        total = float(nbf.sum()) if nbf.size else 0.0
        return 0.0, total if total > 0 else 1.0

    def traces(self) -> list[tuple[np.ndarray, np.ndarray, Any]]:
        """``(x, y, colour)`` of every trace: each normalised to [0, 1] and offset by its row.

        ``act`` of the cell is white; a field with one row (after transposing fields with
        more than 10 rows) gets colour ``colors[min(2*row + k, 9)]`` (row and position ``k``
        in the row, 0-based, MATLAB ``min(ind*2-1+knd-1, 10)``); a field with two rows is
        drawn red and blue; other fields are not drawn. ``x`` = 1, 2, ... as MATLAB's
        ``plot(y)``.
        """
        from calliopy.projmeta._struct import ntzo

        e = self.rd_entry()
        out: list = []
        if e is None:
            return out
        for ind, row in enumerate(self.channels):
            for knd, fi in enumerate(row):
                name = self.rdfields[fi] if 0 <= fi < len(self.rdfields) else None
                try:
                    if name == "act":
                        y = ntzo(np.asarray(e["act"])[self.cell, :]) + ind
                        out.append((np.arange(1, y.size + 1), y, "w"))
                        continue
                    t = np.asarray(e[name])
                    if t.dtype == object or t.size == 0:
                        continue
                    t = np.atleast_2d(t)
                    if t.shape[0] > 10:
                        t = t.T
                    if t.shape[0] == 1:
                        y = ntzo(t[0]) + ind
                        c = tuple(self.colors[min(ind * 2 + knd, 9)])
                        out.append((np.arange(1, y.size + 1), y, c))
                    elif t.shape[0] == 2:
                        for r, c in ((0, "r"), (1, "b")):
                            y = ntzo(t[r]) + ind
                            out.append((np.arange(1, y.size + 1), y, c))
                except (KeyError, IndexError, TypeError, ValueError) as err:
                    log.debug("channel %s not drawn: %s", name, err)
        return out

    def thumbnail(self) -> np.ndarray | None:
        """ROI thumbnail (MATLAB ``cbh.ta``): see ``cdisp``; None when it cannot be made."""
        if self.cdisp == 0:
            return None
        e = self.rd_entry()
        try:
            tmpl = np.asarray(e["template"], dtype=np.float64)
            rois = _elements(e["ROIinfo"])
            idx = np.ravel(np.asarray(rois[self.cell]["indices"])).astype(np.int64)
            ts = tmpl.shape
            rows, cols = np.unravel_index(idx - 1, ts, order="F")
            rows, cols = rows + 1, cols + 1  # MATLAB ind2sub, 1-based
            if self.cdisp == 1:
                cx = int(mround(rows.mean()))
                cy = int(mround(cols.mean()))
                w = int(self.ta_win)
                out = np.zeros((2 * w + 1, 2 * w + 1))
                r0, r1 = max(1, cx - w), min(ts[0], cx + w)
                c0, c1 = max(1, cy - w), min(ts[1], cy + w)
                dr = 1 + max(0, 1 - (cx - w))
                dc = 1 + max(0, 1 - (cy - w))
                out[dr - 1: dr - 1 + r1 - r0 + 1, dc - 1: dc - 1 + c1 - c0 + 1] = tmpl[r0 - 1: r1, c0 - 1: c1]
                return out
            from calliopy.plotting import rois_to_image

            im = rois_to_image([{"indices": idx}], ts, template=tmpl, type="fill", singlecolor=[1, 0, 0])
            return im[rows.min() - 1: rows.max(), cols.min() - 1: cols.max(), :]
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as err:
            log.debug("no thumbnail: %s", err)
            return None

    def labels(self) -> dict[str, str]:
        """Texts of the header: animal, site, cell, timepoint, comment, annotations."""
        site = self.sites[self.site]
        e = self.rd_entry()
        n = self.n_cells()
        cell = f"{self.cell + 1} / {n} - layer {self.layer + 1}"
        try:
            info = _elements(e["ROIinfo"])[self.cell] if e is not None else None
            if isinstance(info, Mapping) and "type" in info:
                cell += f" - {num2str(info['type'])}"
        except (KeyError, IndexError, TypeError):
            pass
        comment = ""
        try:
            exp_id = np.ravel(np.asarray(site["ExpGroup"], dtype=np.float64))[self.tp]
            c = self.comments.get(int(exp_id), self.comments.get(float(exp_id)))
            if c:
                comment = f"{c[:10]}..."
        except (KeyError, IndexError, TypeError, ValueError):
            pass
        return {
            "animal": num2str(site.get("animal", "")),
            "site": str(self.site + 1),
            "cell": cell,
            "tp": f"{self.tp + 1} / {self.n_timepoints()}",
            "comment": comment,
            "site_annotation": self.site_annotation(),
            "cell_annotation": self.cell_annotation(),
        }

    # ---------------------------------------------------------------------------------
    # annotations file (MATLAB cba_save / load)
    # ---------------------------------------------------------------------------------
    def params_struct(self) -> MatStruct:
        """MATLAB ``params`` as ``cell_browser.m`` saves it (1-based values, MATLAB classes).

        Fields in MATLAB's order (``nzlay``, ``win_pos``, ``cell_ind``, ``site_id``, ``tp``,
        ``zl``, ``zon``, ``rdfields``, ``cell_annotations``, ``site_annotations``,
        ``chansToPlot``, ``ta_win``, ``colors``, then ``zxlim`` once a zoom window was set).
        After :meth:`load_annotations` the loaded field order is kept, fields calliopy does
        not use are written back unchanged, and new fields are appended (as MATLAB does).

        Classes as in MATLAB: every number double; ``zon`` double 0 until the zoom was
        toggled, then logical (MATLAB ``~params.zon``); ``rdfields`` an N x 1 cell of char;
        ``cell_annotations`` an nSites x 4 x maxCells cell (grown when needed), ``[]`` where
        nothing was entered; ``site_annotations`` an nSites x 1 cell; ``chansToPlot`` a
        5 x 1 cell of 1 x k rows (``[]`` when nothing is selected).
        """
        p = MatStruct(self._loaded_params) if self._loaded_params is not None else MatStruct()
        p["nzlay"] = float(MATLAB_NZLAY)
        p["win_pos"] = np.asarray(self.win_pos, dtype=np.float64).reshape(1, 4)
        p["cell_ind"] = float(self.cell + 1)
        p["site_id"] = float(self.site + 1)
        p["tp"] = float(self.tp + 1)
        p["zl"] = float(self.layer + 1)
        p["zon"] = bool(self.zoom) if self._zon_logical else float(self.zoom)
        rdf = np.empty((len(self.rdfields), 1), dtype=object)
        for i, n in enumerate(self.rdfields):
            rdf[i, 0] = n
        p["rdfields"] = rdf
        shape = [max(len(self.sites), self._ca_shape[0]), max(MATLAB_NZLAY, self._ca_shape[1]),
                 max(self.max_nbr_cells, self._ca_shape[2])]
        for s, z, c in self.cell_annotations:
            shape = [max(shape[0], s + 1), max(shape[1], z + 1), max(shape[2], c + 1)]
        ca = np.empty(shape, dtype=object)
        for idx in np.ndindex(ca.shape):
            ca[idx] = np.zeros((0, 0))
        for (s, z, c), text in self.cell_annotations.items():
            ca[s, z, c] = text
        p["cell_annotations"] = ca
        sa = np.empty((len(self.sites), 1), dtype=object)
        for s, text in enumerate(self.site_annotations):
            sa[s, 0] = text if text is not None else np.zeros((0, 0))
        p["site_annotations"] = sa
        ch = np.empty((N_CHANNEL_ROWS, 1), dtype=object)
        for i, row in enumerate(self.channels):
            ch[i, 0] = (np.asarray(row, dtype=np.float64).reshape(1, -1) + 1) if row else np.zeros((0, 0))
        p["chansToPlot"] = ch
        p["ta_win"] = float(self.ta_win)
        p["colors"] = np.asarray(self.colors, dtype=np.float64)
        if self.zoom_xlim is not None:
            p["zxlim"] = np.asarray(self.zoom_xlim, dtype=np.float64).reshape(1, 2)
        return p

    def save_annotations(self, path: str | PathLike | None = None) -> Path:
        """Write ``params`` to the annotations file (MATLAB ``cba_save``); returns the path.

        One variable ``params`` (:meth:`params_struct`). A new file, or one MATLAB wrote with
        its default ``save``, is written as a MATLAB v7 MAT-file (compressed v5 format, as
        MATLAB's ``save(file, 'params')``); an existing v7.3 file stays v7.3. ``.mat`` is
        appended to a name without extension (MATLAB ``save``).
        """
        target = Path(path) if path is not None else self.annotations_file
        if target is None:
            raise ValueError("no annotations file given")
        if not target.suffix:
            target = target.with_name(target.name + ".mat")
        params = self.params_struct()
        if target.is_file() and h5py.is_hdf5(target):
            save_mat(target, {"params": params})
        else:
            import scipy.io

            scipy.io.savemat(target, {"params": _v5_value(params)}, do_compression=True, oned_as="row")
        self.annotations_file = target
        log.info("Saved params to CB annotations file %s", target.name)
        return target

    def load_annotations(self, path: str | PathLike) -> None:
        """Resume from an annotations file written by MATLAB (v7 or v7.3) or calliopy."""
        p = read_mat_variable(path, "params")
        if not isinstance(p, Mapping):
            raise TypeError("params in the annotations file is not a struct")
        self._loaded_params = MatStruct(p)
        if "rdfields" in p:
            names = [_str(v) for v in np.ravel(np.asarray(p["rdfields"], dtype=object), order="F")]
            self.rdfields = [n for n in names if n is not None]
        if "chansToPlot" in p:
            rows = list(np.ravel(np.asarray(p["chansToPlot"], dtype=object), order="F"))
            ch: list[list[int]] = []
            for r in rows[:N_CHANNEL_ROWS]:
                vals = np.ravel(np.asarray(r, dtype=np.float64)) if np.size(r) else np.zeros(0)
                ch.append([int(v) - 1 for v in vals if 1 <= v <= len(self.rdfields)])
            self.channels = ch + [[] for _ in range(N_CHANNEL_ROWS - len(ch))]
        if "site_annotations" in p:
            vals = list(np.ravel(np.asarray(p["site_annotations"], dtype=object), order="F"))
            for s in range(min(len(vals), len(self.sites))):
                self.site_annotations[s] = _str(vals[s])
        if "cell_annotations" in p:
            ca = np.asarray(p["cell_annotations"], dtype=object)
            while ca.ndim < 3:
                ca = ca[..., None]
            self._ca_shape = tuple(int(n) for n in ca.shape[:3])
            for idx in np.ndindex(ca.shape):
                text = _str(ca[idx])
                if text is not None:
                    self.cell_annotations[(int(idx[0]), int(idx[1]), int(idx[2]))] = text
        if "colors" in p and np.size(p["colors"]):
            self.colors = np.asarray(p["colors"], dtype=np.float64)
        if "ta_win" in p:
            self.ta_win = int(_scalar(p["ta_win"], DEFAULT_TA_WIN))
        if "win_pos" in p and np.size(p["win_pos"]) == 4:
            self.win_pos = tuple(float(v) for v in np.ravel(p["win_pos"]))
        if "zxlim" in p and np.size(p["zxlim"]) == 2:
            v = np.ravel(np.asarray(p["zxlim"], dtype=np.float64))
            self.zoom_xlim = (float(v[0]), float(v[1]))
        zon = np.asarray(p.get("zon", 0.0))
        self._zon_logical = zon.dtype.kind in "bu"  # logical (scipy reads v5 logical as uint8)
        self.zoom = bool(_scalar(zon, 0)) and self.zoom_xlim is not None
        self.site = int(min(max(_scalar(p.get("site_id", 1), 1) - 1, 0), len(self.sites) - 1))
        self.layer = int(max(_scalar(p.get("zl", 1), 1) - 1, 0))
        self.tp = int(max(_scalar(p.get("tp", 1), 1) - 1, 0))
        self.cell = int(max(_scalar(p.get("cell_ind", 1), 1) - 1, 0))
        self._clamp()


def _v5_value(v: Any) -> Any:
    """A :class:`MatStruct` tree as ``scipy.io.savemat`` writes it in MATLAB's layout.

    Structs become dicts (field order kept), cells object arrays, ``str`` char rows
    (``''`` a 0 x 0 char), ``bool`` logical, Python numbers 1 x 1 double.
    """
    if isinstance(v, Mapping):
        return {k: _v5_value(x) for k, x in v.items()}
    if isinstance(v, np.ndarray) and v.dtype == object:
        out = np.empty(v.shape, dtype=object)
        for idx in np.ndindex(v.shape):
            out[idx] = _v5_value(v[idx])
        return out
    if isinstance(v, (bool, np.bool_)):
        return np.array([[bool(v)]])
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return np.array([[float(v)]])
    if v is None:
        return np.zeros((0, 0))
    return v


class CellBrowserView:
    """Traces and ROI thumbnail of a :class:`CellBrowserSession` on a matplotlib figure.

    The figure gets a black main axes for the traces (y from 0 to 5, one band per channel
    row) and a small axes for the thumbnail in the top-right corner (MATLAB ``cbh.ma`` and
    ``cbh.ta``). Call :meth:`refresh` after every session change.
    """

    def __init__(self, figure: Figure, session: CellBrowserSession) -> None:
        self.figure = figure
        self.session = session
        figure.set_facecolor("k")
        self.ax = figure.add_axes((0.03, 0.04, 0.86, 0.93), facecolor="k")
        self.thumb_ax = figure.add_axes((0.9, 0.74, 0.095, 0.24), facecolor="k")
        self.thumb_ax.set_axis_off()
        for side in self.ax.spines.values():
            side.set_color((0.5, 0.5, 0.5))
        self.ax.tick_params(colors=(0.7, 0.7, 0.7))
        self.thumb = None
        self.refresh()

    def refresh(self) -> None:
        s = self.session
        ax = self.ax
        for artist in list(ax.lines) + list(ax.collections):
            artist.remove()
        for x, y, c in s.traces():
            ax.plot(x, y, color=c, linewidth=0.8)
        ax.set_ylim(0, 5)
        ax.set_xlim(*s.xlim())
        if s.pending_zoom:
            for x in s.pending_zoom:
                ax.axvline(x, color="w", linestyle="--", linewidth=0.8)
        im = s.thumbnail() if s.cdisp else None
        if s.cdisp:
            if im is None:
                w = 2 * int(s.ta_win) + 1
                im = np.zeros((w, w))
            if self.thumb is None or np.ndim(self.thumb.get_array()) != im.ndim:
                self.thumb_ax.clear()
                self.thumb_ax.set_axis_off()
                self.thumb = self.thumb_ax.imshow(im, cmap="gray", interpolation="nearest")
            else:
                self.thumb.set_data(im)
                self.thumb.set_extent((-0.5, im.shape[1] - 0.5, im.shape[0] - 0.5, -0.5))
                if im.ndim == 2:
                    self.thumb.autoscale()
