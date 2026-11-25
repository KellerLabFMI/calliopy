"""Interactive stack and movie viewer: calliope ``07_ViewStack/view_stack.m`` (matplotlib)."""

from __future__ import annotations

import logging

import numpy as np

from .._matlab import mround, prctile
from ._rot import get_rot_ind

__all__ = ["StackViewer", "view_stack"]

log = logging.getLogger(__name__)

_HELP = """view_stack keys:
  right/left            next/previous frame
  shift+right/left      lower colour limit x1.1 / x0.9
  shift+up/down         upper colour limit x1.1 / x0.9
  ctrl+arrows           tilt the cut through the stack (ctrl+up/down: x, ctrl+right/left: y)
  up/down               faster/slower playback
  space                 play/pause
  s                     scale the figure to the image aspect ratio
  t                     average image of the stack in a new figure
  c                     clear area selections and redraw
  i                     toggle linear/sigmoid grey colormap
  g                     invert the colormap
  x                     toggle keeping the displayed frame in viewer.curr_frame
  h                     log this help
Methods replace the keys that asked for mouse or console input in MATLAB:
  select_area (a), plot_area_traces (p), mark_line (f), add_mark (w), delete_mark (W),
  set_frames_to_average (v)."""


def _ntzo(x: np.ndarray) -> np.ndarray:
    """Normalise to [0, 1] (calliope ``ntzo``)."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


class StackViewer:
    """Viewer created by :func:`view_stack`. All frame numbers are 0-based.

    Attributes
    ----------
    fig, image_ax, image, slider_ax, slider_bar, text
        matplotlib objects (MATLAB ``hf``, ``h_f_ax``, ``h_im_data``, ``h_sl_ax``,
        ``h_sl_bar``, ``h_txt``).
    cp : float
        Slider position in [0, 1]; the frame shown is ``round(cp * (n_frames - 1))``.
    clim : list of float
        Colour limits.
    marks : list of int
        Marked voxels as 0-based column-major linear indices into the stack.
    raw_traces : list of numpy.ndarray
        Mean traces of the selected areas (MATLAB ``raw_traces`` in the base workspace).
    curr_frame : numpy.ndarray or None
        Last frame shown while export is on (MATLAB ``curr_frame`` in the base workspace).
    """

    def __init__(self, data, *, marks=None, adata=None, frame_rate: float = 0, frame_zero: float | None = None,
                 map="gray", nbr_of_frames_to_average: float = 1, set_movie_to_frame: int = 0,
                 small_figure_window: float = 0, check_reg_exp_ids=None, stack_lims=None) -> None:
        import matplotlib.pyplot as plt

        d = np.asarray(data)
        if d.ndim == 2:
            d = d[:, :, None]
        if d.ndim != 3:
            raise ValueError("data must be a (rows, cols, frames) stack")
        if d.size == 1:
            raise ValueError("loading a stack by ExpID is not supported; pass the data")
        self.data = d
        self.n_frames = d.shape[2]
        self.frame_rate = float(frame_rate)
        self.frame_zero = -1.0 if frame_zero is None else float(frame_zero)
        self.nbr_avg_frames = max(1, int(np.floor(float(nbr_of_frames_to_average))))
        self.check_reg_exp_ids = list(check_reg_exp_ids) if check_reg_exp_ids is not None else None
        self.frame_pause = 0.01
        self.playback_spacing = 1.0
        self.movie_status = False
        self.tx = 0
        self.ty = 0
        self.export_frame = False
        self.sigmoid_colormap = False
        self.curr_frame = None
        self.raw_traces: list[np.ndarray] = []
        self.area_rows: list[tuple[int, int]] = []
        self.area_cols: list[tuple[int, int]] = []
        self.marks = [int(m) for m in np.ravel(marks)] if marks is not None else []
        self._mark_artists: list = []
        self._button_down = False
        self._timer = None
        self._aux_fig = None
        self._avg_fig = None
        self.map = map

        if adata is not None and np.size(adata):
            a = np.asarray(adata, dtype=np.float64).ravel().copy()
            a[a < 0] = 0
            self.adata = _ntzo(a)
        else:
            self.adata = None

        mid = int(mround(self.n_frames / 2)) - 1 if self.n_frames > 1 else 0
        template = d[:, :, max(mid, 0)].astype(np.float64).ravel()
        if stack_lims is None or np.size(stack_lims) == 0:
            lo = float(np.ravel(prctile(template, 10))[0])
            hi = float(np.ravel(prctile(template, 99.99))[0])
            if lo < hi:
                self.clim = [lo, hi]
            else:
                log.info("set clim to default values")
                self.clim = [0.0, 255.0]
        else:
            self.clim = [float(v) for v in np.ravel(stack_lims)[:2]]

        rows, cols = d.shape[:2]
        if small_figure_window == 1:
            size = (3.47, 3.04)
        elif small_figure_window > 1:
            size = (3.47 * small_figure_window, 3.04)
        else:
            size = (max(cols, 100) / 100, 1.03 * max(rows, 100) / 100)
        self.fig = plt.figure(figsize=size, facecolor="k")
        lim1 = 0.975 if self.adata is not None else 1.0
        self.image_ax = self.fig.add_axes((0, 0.03, lim1, 0.97), facecolor="k")
        f0 = int(set_movie_to_frame)
        first = d[:, :, f0 : f0 + self.nbr_avg_frames].astype(np.float64).mean(axis=2)
        self.image = self.image_ax.imshow(first, cmap=map, vmin=self.clim[0], vmax=self.clim[1],
                                          interpolation="nearest", aspect="auto")
        self.image_ax.set_axis_off()
        for m in self.marks:
            r, c, _ = np.unravel_index(m, d.shape, order="F")
            (h,) = self.image_ax.plot(c, r, "o", mfc="none", mec="r", ms=9)
            self._mark_artists.append(h)
        self.text = self.image_ax.text(cols / 20, rows / 20, str(f0), fontsize=12, color="w",
                                       fontweight="bold", va="top")
        if self.adata is not None:
            self.aux_ax = self.fig.add_axes((0.975, 0.03, 0.025, 0.97), facecolor="k")
            (self.aux_bar,) = self.aux_ax.plot([0, 0], [0, self.adata[0]], linewidth=10, color="g")
            self.aux_ax.set_ylim(0, 1)
            self.aux_ax.set_axis_off()
        self.slider_ax = self.fig.add_axes((0, 0, 1, 0.03), facecolor="k")
        (self.slider_bar,) = self.slider_ax.plot([0, 0], [0, 1], "r", linewidth=20)
        self.slider_ax.set_xlim(0, 1)
        self.slider_ax.set_ylim(0, 1)
        self.slider_ax.set_axis_off()
        self.cp = 0.0

        canvas = self.fig.canvas
        manager = getattr(canvas, "manager", None)
        if manager is not None and getattr(manager, "key_press_handler_id", None) is not None:
            canvas.mpl_disconnect(manager.key_press_handler_id)  # keys belong to the viewer
        canvas.mpl_connect("key_press_event", self._on_key)
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("button_release_event", self._on_release)
        canvas.mpl_connect("motion_notify_event", self._on_motion)
        canvas.mpl_connect("close_event", self._on_close)

    # ----------------------------------------------------------------- display
    @property
    def frame_index(self) -> int:
        """0-based index of the frame at the slider position."""
        return int(mround(self.cp * (self.n_frames - 1)))

    def _cut(self, f: int) -> np.ndarray:
        idx = get_rot_ind(self.data.shape, f, self.tx, self.ty)
        return self.data.ravel(order="F")[idx].reshape(self.data.shape[:2], order="F")

    def frame_image(self, f: int) -> np.ndarray:
        """The image :meth:`draw_frame` shows for 0-based frame ``f``."""
        n = self.nbr_avg_frames
        tilted = self.tx != 0 or self.ty != 0
        if n == 1:
            return self._cut(f) if tilted else self.data[:, :, f]
        if tilted:
            return np.max(np.stack([self._cut(f + k) for k in range(n)], axis=2), axis=2)
        return self.data[:, :, f : min(f + n, self.n_frames)].astype(np.float64).mean(axis=2)

    def label(self, f: int) -> str:
        """Frame label: number, time relative to ``frame_zero``, tilt or ExpID."""
        if self.check_reg_exp_ids is not None:
            return f"ExpID:  {self.check_reg_exp_ids[f]}"
        if self.tx != 0 or self.ty != 0:
            return f"{f} - Tilt X: {self.tx} Tilt Y: {self.ty}"
        if self.frame_rate == 0:
            return str(f)
        return f"{int(np.floor((f - self.frame_zero) / self.frame_rate))} s"

    def draw_frame(self, cp: float) -> None:
        """Show the frame at slider position ``cp`` in [0, 1] (MATLAB ``draw_frame``)."""
        self.cp = float(min(max(cp, 0.0), 1.0))
        f = self.frame_index
        self.slider_bar.set_xdata([self.cp, self.cp])
        frame = self.frame_image(f)
        self.image.set_data(frame)
        if self.nbr_avg_frames == 1:
            for m, h in zip(self.marks, self._mark_artists):
                z = np.unravel_index(m, self.data.shape, order="F")[2]
                h.set_markeredgecolor((1, 0, 0) if z == f else (1, 1, 0))
        if self.adata is not None:
            self.aux_bar.set_ydata([0, self.adata[f]])
        if self.export_frame:
            self.curr_frame = np.asarray(frame)
        self.text.set_text(self.label(f))
        self.fig.canvas.draw_idle()

    def set_frame(self, f: int) -> None:
        """Show 0-based frame ``f``."""
        self.draw_frame(f / (self.n_frames - 1) if self.n_frames > 1 else 0.0)

    def set_clim(self, lo: float, hi: float) -> None:
        self.clim = [float(lo), float(hi)]
        self.image.set_clim(*self.clim)
        self.fig.canvas.draw_idle()

    def set_frames_to_average(self, n: float) -> None:
        """Number of frames averaged per displayed image (MATLAB key 'v')."""
        self.nbr_avg_frames = max(1, int(np.floor(float(n))))
        self.draw_frame(self.cp)

    # ----------------------------------------------------------------- playback
    def step(self) -> None:
        """Advance one playback step (wrapping to the start), as one loop of MATLAB 'space'."""
        spacing = 1.0 / self.n_frames
        cp = self.cp + spacing * self.playback_spacing
        self.draw_frame(0.0 if cp > 1 else cp)

    def toggle_play(self) -> None:
        self.movie_status = not self.movie_status
        if self.movie_status:
            self._timer = self.fig.canvas.new_timer(interval=max(1, int(self.frame_pause * 1000)))
            self._timer.add_callback(self.step)
            self._timer.start()
        elif self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _retime(self) -> None:
        if self._timer is not None:
            self._timer.interval = max(1, int(self.frame_pause * 1000))

    # ----------------------------------------------------------------- tools
    def select_area(self, rows, cols) -> np.ndarray:
        """Mean trace of a rectangle (MATLAB key 'a'); ``rows``/``cols`` are 0-based inclusive
        (first, last) pairs. Draws the rectangle and returns the trace (also kept in
        ``raw_traces``)."""
        r0, r1 = sorted(int(v) for v in rows)
        c0, c1 = sorted(int(v) for v in cols)
        k = len(self.raw_traces)
        color = "mgycrbw"[k % 7]
        self.image_ax.plot([c1, c1, c0, c0, c1], [r1, r0, r0, r1, r1], color=color)
        trace = self.data[r0 : r1 + 1, c0 : c1 + 1, :].astype(np.float64).mean(axis=(0, 1))
        self.raw_traces.append(trace)
        self.area_rows.append((r0, r1))
        self.area_cols.append((c0, c1))
        return trace

    def plot_area_traces(self):
        """Figure with the selected areas' traces, colour-matched (MATLAB key 'p')."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(facecolor="k")
        for k, tr in enumerate(self.raw_traces):
            ax.plot(tr, color="mgycrbw"[k % 7], linewidth=2)
        ax.set_facecolor("k")
        ax.tick_params(colors="w", direction="out")
        ax.autoscale(tight=True)
        return fig

    def mark_line(self, p0, p1) -> None:
        """Red line between two (x, y) image points (MATLAB key 'f')."""
        self.image_ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "r", linewidth=2)

    def add_mark(self, row: int, col: int) -> int:
        """Mark a voxel of the current frame (MATLAB key 'w'); returns its 0-based linear index."""
        f = self.frame_index
        r, c = int(row), int(col)
        m = int(np.ravel_multi_index((r, c, f), self.data.shape, order="F"))
        self.marks.append(m)
        (h,) = self.image_ax.plot(c, r, "o", mfc="none", mec="r", ms=8)
        self._mark_artists.append(h)
        return m

    def delete_mark(self, row: int, col: int) -> bool:
        """Delete the first mark within 5 pixels and 5 frames (MATLAB key 'W')."""
        f = self.frame_index
        for k, m in enumerate(self.marks):
            r, c, z = np.unravel_index(m, self.data.shape, order="F")
            if abs(r - row) < 5 and abs(c - col) < 5 and abs(z - f) < 5:
                self._mark_artists.pop(k).remove()
                self.marks.pop(k)
                return True
        return False

    def show_template(self):
        """Average image of the stack in a new grey figure (MATLAB key 't')."""
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(self.data.astype(np.float64).mean(axis=2), cmap="gray", aspect="auto")
        ax.set_axis_off()
        self._avg_fig = fig
        return fig

    def clear(self) -> None:
        """Remove area selections and lines, show the single frame (MATLAB key 'c')."""
        for art in list(self.image_ax.lines):
            if art not in self._mark_artists:
                art.remove()
        self.raw_traces.clear()
        self.area_rows.clear()
        self.area_cols.clear()
        self.image.set_cmap(self.map)
        self.draw_frame(self.cp)

    def _grey_map(self, sigmoid: bool):
        from matplotlib.colors import ListedColormap

        g = np.arange(256) / 255.0
        if sigmoid:
            g = 1.0 / (1.0 + np.exp(-10 * (g - 0.5)))
        return ListedColormap(np.column_stack([g, g, g]))

    # ----------------------------------------------------------------- events
    def handle_key(self, key: str) -> None:
        """Apply one key (matplotlib key name, e.g. 'right', 'shift+up', ' ')."""
        spacing = 1.0 / self.n_frames
        if key == "s":
            w = self.fig.get_figwidth()
            self.fig.set_size_inches(w, 1.03 * w * self.data.shape[0] / self.data.shape[1])
        elif key == "h":
            log.info(_HELP)
        elif key == "t":
            self.show_template()
        elif key == "c":
            self.clear()
        elif key == "i":
            self.image.set_cmap(self._grey_map(not self.sigmoid_colormap))
            self.sigmoid_colormap = not self.sigmoid_colormap
        elif key == "g":
            self.image.set_cmap(self.image.get_cmap().reversed())
        elif key == "x":
            self.export_frame = not self.export_frame
            log.info("Export frames is turned %s", "ON" if self.export_frame else "OFF")
        elif key == "p":
            self.plot_area_traces()
        elif key in ("right", "left"):
            self.draw_frame(self.cp + (spacing if key == "right" else -spacing))
        elif key in ("shift+right", "shift+left"):
            self.set_clim(self.clim[0] * (1.1 if key.endswith("right") else 0.9), self.clim[1])
        elif key in ("shift+up", "shift+down"):
            self.set_clim(self.clim[0], self.clim[1] * (1.1 if key.endswith("up") else 0.9))
        elif key in ("ctrl+right", "ctrl+left"):
            self.ty += 1 if key.endswith("right") else -1
            self.draw_frame(self.cp)
        elif key in ("ctrl+up", "ctrl+down"):
            self.tx += 1 if key.endswith("up") else -1
            self.draw_frame(self.cp)
        elif key == "up":
            if self.frame_pause <= 0.01:
                self.playback_spacing *= 2
            else:
                self.frame_pause *= 0.5
            self._retime()
        elif key == "down":
            if self.playback_spacing > 1:
                self.playback_spacing /= 2
            else:
                self.frame_pause *= 2
            self._retime()
        elif key in (" ", "space"):
            self.toggle_play()
        self.fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if event.key:
            self.handle_key(event.key)

    def _on_press(self, event) -> None:
        self._button_down = True
        self._on_motion(event)

    def _on_release(self, event) -> None:
        self._button_down = False

    def _on_motion(self, event) -> None:
        if self._button_down and event.inaxes is self.slider_ax and event.xdata is not None:
            self.draw_frame(min(max(float(event.xdata), 0.0), 1.0))

    def _on_close(self, event) -> None:
        import matplotlib.pyplot as plt

        if self._timer is not None:
            self._timer.stop()
        for f in (self._aux_fig, self._avg_fig):
            if f is not None:
                plt.close(f)


def view_stack(data, *, marks=None, adata=None, frame_rate: float = 0, frame_zero: float | None = None,
               map="gray", nbr_of_frames_to_average: float = 1, set_movie_to_frame: int = 0,
               small_figure_window: float = 0, check_reg_exp_ids=None, stack_lims=None,
               show: bool = True) -> StackViewer:
    """Interactive viewer for image stacks and movies (calliope ``07_ViewStack/view_stack.m``).

    Opens a figure showing one frame of ``data`` with a slider bar along the bottom (click
    or drag to move), a frame number or time label, and keyboard controls (press 'h' or
    see :data:`StackViewer` methods). The viewer is not blocking.

    Parameters
    ----------
    data : array_like, shape (rows, cols, frames)
    marks : array_like of int, optional
        Voxels to mark, 0-based column-major linear indices into ``data``.
    adata : array_like, optional
        One value per frame shown as a bar on the right (negative values set to 0, then
        scaled to [0, 1]).
    frame_rate : float, default 0
        Frames per second; when non-zero the label shows seconds relative to ``frame_zero``.
    frame_zero : float, optional
        0-based frame at time zero. Default: one frame before the first frame, which is
        MATLAB's default ``FrameZero = 0`` with 1-based frames.
    map : str or matplotlib colormap or (n, 3) array, default 'gray'
    nbr_of_frames_to_average : float, default 1
        Frames averaged per image (non-integers are floored, as MATLAB's colon does).
    set_movie_to_frame : int, default 0
        0-based first frame of the initial image.
    small_figure_window : float, default 0
        1: small window; > 1: small window that many times wider.
    check_reg_exp_ids : sequence of str, optional
        One label per frame (``'ExpID:  <label>'``).
    stack_lims : (float, float), optional
        Colour limits (default: 10th and 99.99th percentile of the middle frame, or
        (0, 255) if those are equal).
    show : bool, default True
        Show the figure without blocking when the backend is interactive.

    Returns
    -------
    StackViewer

    Differences from MATLAB
    -----------------------
    - Frames, ``frame_zero``, ``set_movie_to_frame`` and ``marks`` are 0-based.
    - Keys that asked for mouse clicks or console input are methods: ``select_area`` ('a'),
      ``plot_area_traces`` ('p'), ``mark_line`` ('f'), ``add_mark``/``delete_mark``
      ('w'/'W'), ``set_frames_to_average`` ('v'). Results stay on the viewer
      (``raw_traces``, ``marks``, ``curr_frame``) instead of the base workspace.
    - Not ported: movie export ('m', needs an external video encoder), the first/last
      frame registration check ('r'), the blocking mode that returns marks on 'E', and
      loading a z-layer when ``data`` is an ExpID.
    - Playback uses a matplotlib timer instead of a ``pause`` loop.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    v = StackViewer(data, marks=marks, adata=adata, frame_rate=frame_rate, frame_zero=frame_zero, map=map,
                    nbr_of_frames_to_average=nbr_of_frames_to_average, set_movie_to_frame=set_movie_to_frame,
                    small_figure_window=small_figure_window, check_reg_exp_ids=check_reg_exp_ids,
                    stack_lims=stack_lims)
    if show and matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps", "cairo", "template"):
        plt.show(block=False)
    return v
