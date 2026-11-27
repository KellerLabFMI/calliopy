"""Matplotlib drawing of RoiQuest and ActiQuest on any Axes/Figure (no window toolkit).

Port of the drawing parts of calliope ``08_RoiQuest/+gfx/createDisplayImage.m``,
``+gfx/updateDisplayImage.m``, ``+gfx/updateOverlayImage.m``, ``+clbk/createHistogram.m``,
``+gfx/updateHistogram.m``, ``ActiQuest/AQ_createMainFigure.m``,
``AQ_createVisualization.m``, ``AQ_updatePlots.m``, ``AQ_updateVisualization.m``,
``AQ_toggleDarkMode.m`` and ``AQ_colorTickLabels.m`` (Leonardo Lupori, 2024).

Artists are created once and updated with ``set_data``; works with the Agg backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.colors import ListedColormap

from ._display import (
    histogram_data,
    info_text_anchor,
    overlay_image,
    overlay_text,
    roi_number_labels,
)
from ._types import Redraw

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ._actiquest import ActiQuestSession
    from ._session import RoiQuestSession

__all__ = ["ActiQuestView", "HistogramView", "RoiQuestView"]

_BG = (0.1, 0.1, 0.1)
_TEXT_COLOR = (0.0, 0.8, 0.8)
_NUM_COLOR = (0.0, 0.8, 0.8)
_HIST_FACE = (0.0, 0.8, 0.8)
_THR_COLOR = (0.93, 0.13, 0.55)
ASPECT_2P = 1.5
"""Axes aspect of two-photon images: MATLAB ``DataAspectRatio = [3, 2, 1]`` (a pixel is 1.5
times as high as wide, so cells do not look squished)."""


def _extent(hw):
    h, w = hw
    return (-0.5, w - 0.5, h - 0.5, -0.5)


class RoiQuestView:
    """Main image of RoiQuest drawn on a matplotlib Axes.

    Creates the image, the RGBA ROI overlay, the info text, the ROI number texts and the
    movement vectors (MATLAB ``gfx.createDisplayImage``); :meth:`refresh` updates them from
    the session (``updateDisplayImage``, ``updateOverlayImage``) according to the
    :class:`Redraw` flags an action returned.

    Parameters
    ----------
    ax
        Any matplotlib Axes (e.g. ``Figure().add_subplot()`` with the Agg backend, or the
        axes of an embedded canvas).
    session
        The :class:`RoiQuestSession` to show.
    histogram
        Optional :class:`HistogramView` refreshed with the main view (MATLAB updates the
        histogram window whenever it is open).

    Attributes
    ----------
    image, overlay, info_text, quiver
        The artists (``quiver`` is None until vectors are shown).

    Notes
    -----
    Differences from MATLAB (display only): the info text uses matplotlib's default font
    (MATLAB: Myriad Pro); ROI number texts are reused instead of deleted and recreated.
    """

    def __init__(self, ax: Axes, session: RoiQuestSession, histogram: HistogramView | None = None):
        self.ax = ax
        self.session = session
        self.histogram = histogram
        s = session
        img = s.display_image()
        ax.set_facecolor(_BG)
        ax.set_xticks([])
        ax.set_yticks([])
        self.image = ax.imshow(img, cmap=ListedColormap(s.cmap), vmin=0, vmax=255,
                               interpolation="nearest", extent=_extent(img.shape[:2]), zorder=1)
        h, w = img.shape[:2]
        self.overlay = ax.imshow(np.zeros((h, w, 4)), interpolation="nearest",
                                 extent=_extent((h, w)), zorder=2)
        self.info_text = ax.text(0, 0, "RoiQuest", color=_TEXT_COLOR, fontsize=12,
                                 va="top", ha="left", zorder=4,
                                 bbox={"facecolor": _BG, "edgecolor": "none", "pad": 2})
        self.number_texts: list = []
        self.quiver = None
        self.n_pixels = 0.0
        self.refresh(Redraw.ALL)

    # -------------------------------------------------------------------------------
    def refresh(self, redraw: Redraw = Redraw.ALL) -> None:
        """Update what ``redraw`` names (does not call ``canvas.draw``)."""
        if Redraw.COLORMAP in redraw:
            self.update_colormap()
        if Redraw.DISPLAY in redraw:
            self.update_display()
        if Redraw.VIEW in redraw:
            self.update_view()
        if Redraw.OVERLAY in redraw:
            self.update_overlay()
        if self.histogram is not None:
            if Redraw.HISTOGRAM in redraw or Redraw.DISPLAY in redraw:
                self.histogram.update()
            elif Redraw.HIST_THRESHOLD in redraw:
                self.histogram.update(only_threshold=True)

    def update_colormap(self) -> None:
        """Apply ``session.cmap`` (``setImageColormap``)."""
        self.image.set_cmap(ListedColormap(self.session.cmap))

    def update_display(self) -> None:
        """Show the current image, MATLAB ``gfx.updateDisplayImage``."""
        s = self.session
        img = s.display_image()
        self.image.set_data(img)
        self.image.set_extent(_extent(img.shape[:2]))
        self.ax.set_aspect("equal" if s.is_widefield_data else ASPECT_2P)

    def update_view(self) -> None:
        """Apply the zoom limits of ``session.view`` (row axis pointing down)."""
        v = self.session.view
        self.ax.set_xlim(v.xlim[0], v.xlim[1])
        self.ax.set_ylim(v.ylim[1], v.ylim[0])

    def update_overlay(self) -> None:
        """Redraw ROIs, info text, ROI numbers and vectors, MATLAB ``gfx.updateOverlayImage``."""
        s = self.session
        ov = overlay_image(s)
        h, w = ov.rgba.shape[:2]
        self.overlay.set_data(ov.rgba)
        self.overlay.set_extent(_extent((h, w)))
        self.n_pixels = ov.n_pixels
        if s.roi_visible:
            x, y, ha, va = info_text_anchor(s)
            self.info_text.set_position((x, y))
            self.info_text.set_ha(ha)
            self.info_text.set_va(va)
        self.info_text.set_text(overlay_text(s, ov.n_pixels))
        self.info_text.set_visible(bool(self.info_text.get_text()))
        labels = roi_number_labels(s)
        while len(self.number_texts) < len(labels):
            self.number_texts.append(self.ax.text(0, 0, "", color=_NUM_COLOR, fontsize=18,
                                                  zorder=3, clip_on=True))
        for t, (x, y, lab) in zip(self.number_texts, labels):
            t.set_position((x, y))
            t.set_text(lab)
            t.set_visible(True)
        for t in self.number_texts[len(labels):]:
            t.set_visible(False)
        if s.show_vectors and s.vectors is not None:
            start, disp = s.vectors
            if self.quiver is not None:
                self.quiver.remove()
            self.quiver = self.ax.quiver(start[:, 1], start[:, 0], disp[:, 1], disp[:, 0],
                                         color="cyan", angles="xy", scale_units="xy", scale=1,
                                         width=0.004, zorder=5)
            s.show_vectors = False
        elif self.quiver is not None:
            self.quiver.set_visible(False)


class HistogramView:
    """Histogram of the visible image region with the threshold line.

    MATLAB ``clbk.createHistogram`` / ``gfx.updateHistogram``: counts of the uint8 values
    0..255 on a log scale, the threshold of the current image kind as a vertical line.
    """

    def __init__(self, ax: Axes, session: RoiQuestSession):
        self.ax = ax
        self.session = session
        ax.set_facecolor(_BG)
        counts, edges = histogram_data(session)
        self.bars = ax.stairs(counts, edges, fill=True, color=_HIST_FACE, baseline=0)
        thr = session.thr[session.img_idx]
        self.threshold_line = ax.axvline(thr, color=_THR_COLOR, linewidth=2)
        ax.set_xlim(0, 255)
        ax.set_yscale("log", nonpositive="clip")
        ax.tick_params(length=0)
        self._autoscale(counts)

    def _autoscale(self, counts) -> None:
        top = max(float(np.max(counts)) if counts.size else 1.0, 1.0)
        self.ax.set_ylim(0.8, top * 1.5)

    def update(self, only_threshold: bool = False) -> None:
        """MATLAB ``gfx.updateHistogram(app, 'onlyThreshold', only_threshold)``."""
        s = self.session
        thr = s.thr[s.img_idx]
        self.threshold_line.set_xdata([thr, thr])
        if only_threshold:
            return
        counts, edges = histogram_data(s)
        self.bars.set_data(counts, edges)
        self._autoscale(counts)


class ActiQuestView:
    """ActiQuest figure: raster of the activity and offset traces (``AQ_createMainFigure``).

    Parameters
    ----------
    fig
        A matplotlib Figure (two axes are added at MATLAB's positions).
    session
        The :class:`ActiQuestSession`.

    Notes
    -----
    Differences from MATLAB (display only): the axes share x and y limits, set as MATLAB sets
    the traces axes (``[1, nFrames]``, ``[0.5, nRois + 1]``); tick label colours are set
    with matplotlib instead of TeX ``\\color`` markup (``AQ_colorTickLabels``).
    """

    RASTER_HEIGHT = 0.25

    def __init__(self, fig: Figure, session: ActiQuestSession):
        self.fig = fig
        self.session = session
        rh = self.RASTER_HEIGHT
        self.ax_raster = fig.add_axes((0.05, 1 - rh, 0.9, rh))
        self.ax_traces = fig.add_axes((0.05, 0.08, 0.9, 1 - rh - 0.08),
                                      sharex=self.ax_raster, sharey=self.ax_raster)
        self.raster = None
        self.lines: list = []
        self.update_visualization()
        self.apply_style()

    def update_visualization(self) -> None:
        """Draw new ROIs, MATLAB ``AQ_createVisualization`` / ``AQ_updateVisualization``."""
        s = self.session
        a = s.act_data
        n, nf = a.shape
        if self.raster is None:
            self.raster = self.ax_raster.imshow(a, aspect="auto", origin="lower",
                                                interpolation="nearest",
                                                extent=(0.5, nf + 0.5, 0.5, n + 0.5))
        else:
            self.raster.set_data(a)
            self.raster.set_extent((0.5, nf + 0.5, 0.5, n + 0.5))
        self.raster.set_clim(s.raster_lo, s.raster_hi)
        self.ax_raster.set_yticks([])
        for ln in self.lines:
            ln.remove()
        x = np.arange(1, nf + 1)
        cols = s.colors
        self.lines = [self.ax_traces.plot(x, y, color=cols[i], linewidth=0.8)[0]
                      for i, y in enumerate(s.plot_data)]
        self.ax_traces.set_xlim(1, max(nf, 2))
        self.ax_traces.set_ylim(0.5, n + 1)
        self.ax_traces.set_yticks(np.arange(1, n + 1))
        self.ax_traces.set_yticklabels(s.roi_labels)
        self.ax_traces.yaxis.grid(True)
        self.ax_traces.set_xlabel("Time (frames)")
        self.ax_traces.set_ylabel("Normalized Activity for each ROI number")
        self.apply_style()

    def update_plots(self) -> None:
        """New ``raster_hi``, MATLAB ``AQ_updatePlots``."""
        s = self.session
        for ln, y in zip(self.lines, s.plot_data):
            ln.set_ydata(y)
        self.raster.set_clim(s.raster_lo, s.raster_hi)

    def apply_style(self) -> None:
        """Light or dark colours, MATLAB ``AQ_toggleDarkMode``."""
        dark = self.session.is_dark_mode
        bg = (0.1, 0.1, 0.1) if dark else (1.0, 1.0, 1.0)
        fg = (0.85, 0.85, 0.85) if dark else (0.15, 0.15, 0.15)
        self.fig.set_facecolor(bg)
        for ax in (self.ax_raster, self.ax_traces):
            ax.set_facecolor(bg)
            ax.tick_params(colors=fg, length=0)
        label = (1.0, 1.0, 1.0) if dark else (0.15, 0.15, 0.15)
        self.ax_traces.xaxis.label.set_color(label)
        self.ax_traces.yaxis.label.set_color(label)
        if self.raster is not None:
            self.raster.set_cmap("hot" if dark else "gray_r")

    def refresh(self, redraw: Redraw = Redraw.ALL) -> None:
        """Update after :meth:`ActiQuestSession.key_press` or ``load_new_rois``."""
        if Redraw.ALL in redraw:
            self.update_visualization()
            return
        if Redraw.DISPLAY in redraw:
            self.update_plots()
        if Redraw.COLORMAP in redraw:
            self.apply_style()
