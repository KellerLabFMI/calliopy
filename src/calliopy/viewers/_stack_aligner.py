"""Interactive alignment of a reference stack in a larger stack: calliope
``07_ViewStack/StackAligner.m``.

:class:`StackAlignerSession` holds the offset and the slice and has the key callback;
:class:`StackAlignerView` draws both images on a matplotlib figure; the tkinter window is
in ``_tk_registration.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.figure import Figure

from calliopy._matlab import linspace, mround

__all__ = ["STACK_ALIGNER_KEYS", "StackAlignerSession", "StackAlignerView"]

STACK_ALIGNER_KEYS = {
    "right / left": "next / previous slice of the reference stack",
    "up / down": "one layer deeper / shallower in the stack (z offset +1 / -1)",
    "shift + right / left": "shift the crop right / left (x offset)",
    "shift + up / down": "shift the crop up / down (y offset)",
    "ctrl (with shift)": "steps of 10 pixels instead of 1",
    "enter": "accept the offset and close",
}
"""Keys of the stack aligner (the help MATLAB's caller prints)."""

_KEY_ALIASES = {"right": "rightarrow", "left": "leftarrow", "up": "uparrow", "down": "downarrow",
                "enter": "return"}


class StackAlignerSession:
    """State and key callback of ``StackAligner.m``.

    The reference stack (``refStack``, the region of the site's origin) is compared with a
    crop of the same size of ``stack``, starting at ``offset`` = (x, y, z) = (column, row,
    layer) of ``stack``. Slice ``k`` of the reference is shown next to layer
    ``k + offset[2]`` of the crop.

    Parameters
    ----------
    ref_stack : array (rows, cols, layers)
        Reference stack (any numeric class; converted to float64 as MATLAB ``double``).
    stack : array (rows, cols, layers)
        Stack in which the reference is searched.
    start_xyz : sequence of 3 numbers
        Start offset ``(x, y, z)``, **0-based** (column, row, layer of ``stack`` where the
        crop starts). MATLAB's ``startXYZ`` is 1-based: pass ``startXYZ - 1``.

    Attributes
    ----------
    offset : numpy.ndarray
        Current offset ``(x, y, z)``, **0-based**, float (not rounded, as MATLAB returns it).
    slice : int
        **0-based** slice of the reference stack shown.
    done : bool
        Set by the Enter key.

    Differences from MATLAB
    -----------------------
    Offsets and slices are 0-based (MATLAB 1-based): the 1-based ``spiny_regions`` corner
    of the caller is ``offset + 1``. The crop uses MATLAB's rounding of the 1-based offset.
    """

    def __init__(self, ref_stack, stack, start_xyz: Sequence[float]) -> None:
        ref = np.asarray(ref_stack, dtype=np.float64)
        big = np.asarray(stack, dtype=np.float64)
        if ref.ndim == 2:
            ref = ref[:, :, None]
        if big.ndim == 2:
            big = big[:, :, None]
        if ref.ndim != 3 or big.ndim != 3:
            raise ValueError("ref_stack and stack must be (rows, cols, layers) arrays")
        start = np.asarray(start_xyz, dtype=np.float64).ravel()
        if start.size != 3:
            raise ValueError("start_xyz must have three elements (x, y, z)")
        self.ref_stack = ref
        self.stack = big
        self.clim_ref = (float(ref.min()), float(ref.max()))
        self.clim_stack = (float(big.min()), float(big.max()))
        self.offset = start.copy()
        self.slice = 0
        self.done = False

    def rounded_offset(self) -> np.ndarray:
        """``round`` of the 1-based offset, minus 1 (MATLAB ``round(d.offset)``), int."""
        return (mround(self.offset + 1) - 1).astype(np.int64)

    def key_press(self, key: str, modifiers: Sequence[str] = ()) -> str | None:
        """MATLAB ``keyHandler``; returns ``'redraw'``, ``'done'`` (Enter) or None."""
        k = _KEY_ALIASES.get(key, key)
        mods = {("control" if str(m).lower() in ("ctrl", "control") else str(m).lower()) for m in modifiers}
        step = 10 if "control" in mods else 1
        shift = "shift" in mods
        if k == "rightarrow":
            if shift:
                self.offset[0] += step
            else:
                self.slice = min(self.slice + 1, self.ref_stack.shape[2] - 1)
        elif k == "leftarrow":
            if shift:
                self.offset[0] -= step
            else:
                self.slice = max(self.slice - 1, 0)
        elif k == "uparrow":
            if shift:
                self.offset[1] -= step
            else:
                self.offset[2] += 1
        elif k == "downarrow":
            if shift:
                self.offset[1] += step
            else:
                self.offset[2] -= 1
        elif k == "return":
            self.done = True
            return "done"
        else:
            return None
        return "redraw"

    def reference_image(self) -> np.ndarray:
        """Slice :attr:`slice` of the reference stack."""
        return self.ref_stack[:, :, self.slice]

    def crop(self) -> np.ndarray:
        """The crop of ``stack`` compared with the reference slice; zeros when outside."""
        x, y, z = (int(v) for v in self.rounded_offset())
        h, w = self.ref_stack.shape[:2]
        zl = self.slice + z
        H, W, Z = self.stack.shape
        if 0 <= zl < Z and y >= 0 and y + h <= H and x >= 0 and x + w <= W:
            return self.stack[y: y + h, x: x + w, zl]
        return np.zeros((h, w))

    def message(self) -> str:
        """Title line (MATLAB ``sgtitle``), with 0-based layer and offsets."""
        x, y, z = (int(v) for v in self.rounded_offset())
        return f"Layer: {self.slice} | Offset -> X: {x}, Y: {y}, Z: {z}"

    def grid_points(self) -> np.ndarray:
        """The 16 green crosses: inner points of a 5 x 5 grid, ``[16 x 2]`` 0-based (x, y)."""
        h, w = self.ref_stack.shape[:2]
        xs = linspace(1, w, 6)[1:5]
        ys = linspace(1, h, 6)[1:5]
        gx, gy = np.meshgrid(xs, ys)
        return np.column_stack([gx.ravel(order="F"), gy.ravel(order="F")]) - 1


class StackAlignerView:
    """Reference slice (left) and crop (right) of a :class:`StackAlignerSession`.

    Black background, grey images with the stacks' global colour limits, green crosses at
    the same positions on both, and the offset line as figure title (MATLAB layout).
    """

    def __init__(self, figure: Figure, session: StackAlignerSession) -> None:
        self.figure = figure
        self.session = session
        figure.set_facecolor("k")
        self.ax_ref = figure.add_axes((0.0, 0.0, 0.5, 0.9), facecolor="k")
        self.ax_crop = figure.add_axes((0.5, 0.0, 0.5, 0.9), facecolor="k")
        s = session
        self.im_ref = self.ax_ref.imshow(s.reference_image(), cmap="gray", vmin=s.clim_ref[0],
                                         vmax=s.clim_ref[1], interpolation="nearest")
        self.im_crop = self.ax_crop.imshow(s.crop(), cmap="gray", vmin=s.clim_stack[0],
                                           vmax=s.clim_stack[1], interpolation="nearest")
        self.ax_ref.set_title("Site origin", color="w")
        self.ax_crop.set_title("This ExpID", color="w")
        pts = s.grid_points()
        for ax in (self.ax_ref, self.ax_crop):
            ax.plot(pts[:, 0], pts[:, 1], "+", color=(0, 1, 0), markersize=8, markeredgewidth=1.5)
            ax.set_xticks([])
            ax.set_yticks([])
        self.title = figure.suptitle("", color="w", fontsize=12, fontweight="bold")
        self.refresh()

    def refresh(self) -> None:
        s = self.session
        self.im_ref.set_data(s.reference_image())
        self.im_crop.set_data(s.crop())
        self.title.set_text(s.message())
