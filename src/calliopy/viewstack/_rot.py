"""Tilted cut through a stack: calliope ``07_ViewStack/get_rot_ind.m``."""

from __future__ import annotations

import numpy as np

from .._matlab import mround

__all__ = ["get_rot_ind"]


def get_rot_ind(dims, cf: int, tx: int, ty: int) -> np.ndarray:
    """Linear indices of a frame cut obliquely through a (rows, cols, frames) stack.

    Port of calliope ``07_ViewStack/get_rot_ind.m``. The image is split into
    ``|tx| + 1`` bands of rows and ``|ty| + 1`` bands of columns; each band takes its pixels
    from a frame shifted by one per band, so the cut is tilted by ``tx`` frames along the
    rows and ``ty`` along the columns, centred on frame ``cf``.

    Parameters
    ----------
    dims : sequence of int
        Stack shape ``(rows, cols, frames)``.
    cf : int
        Centre frame, **0-based**.
    tx, ty : int
        Tilt along rows / columns, in frames (0: no tilt).

    Returns
    -------
    numpy.ndarray of int64, shape (rows * cols,)
        **0-based** linear indices into the stack in column-major (``order="F"``) layout,
        clipped to the stack as MATLAB does; ``data.ravel(order="F")[idx].reshape(rows,
        cols, order="F")`` is the tilted frame.

    Differences from MATLAB
    -----------------------
    ``cf`` and the returned indices are 0-based (MATLAB: 1-based ``cf`` and indices).
    """
    rows, cols = int(dims[0]), int(dims[1])
    nfr = int(dims[2]) if len(dims) > 2 else 1
    tx, ty = int(tx), int(ty)
    shift = np.zeros((rows, cols), dtype=np.int64)
    xdist = int(np.ceil(rows / (abs(tx) + 1)))
    for ind in range(1, abs(tx) + 2):
        shift[(ind - 1) * xdist : min(rows, ind * xdist), :] = ind * int(np.sign(tx))
    ydist = int(np.ceil(cols / (abs(ty) + 1)))
    for ind in range(1, abs(ty) + 2):
        shift[:, (ind - 1) * ydist : min(cols, ind * ydist)] += ind * int(np.sign(ty))
    shift = shift - int(mround((tx + ty) / 2))
    shift = shift * rows * cols
    npix = rows * cols
    idx = np.arange(int(cf) * npix, (int(cf) + 1) * npix, dtype=np.int64) + shift.ravel(order="F")
    return np.clip(idx, 0, npix * nfr - 1)
