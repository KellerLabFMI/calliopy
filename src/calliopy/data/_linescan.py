"""Line-shift correction of bidirectionally scanned images: calliope
``05_LoadData/estimate_line_shift.m`` and ``05_LoadData/correct_line_shift.m``."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .._matlab import mround

__all__ = ["CorrectLineShiftResult", "correct_line_shift", "estimate_line_shift"]


def _xcorr_unbiased(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MATLAB ``xcorr(a, b, 'unbiased')`` for equal-length real vectors (lags -(n-1)..n-1)."""
    n = a.size
    c = np.correlate(a, b, mode="full")
    lags = np.arange(-(n - 1), n)
    return c / (n - np.abs(lags))


def estimate_line_shift(template) -> int:
    """Shift (in columns) between odd and even scan lines of an image.

    Port of calliope ``05_LoadData/estimate_line_shift.m``: the unbiased cross-correlation of
    every pair of neighbouring (mean-subtracted) rows, flipped for every second pair, is
    averaged over the pairs; the peak within the central half of the lags gives the shift.

    Returns
    -------
    int
        The shift in columns, as MATLAB returns it (``correct_line_shift`` circularly
        shifts the odd rows, MATLAB rows 1, 3, ..., by this amount).
    """
    t = np.asarray(template, dtype=np.float64)
    rows, cols = t.shape
    if rows < 2:
        raise ValueError("template needs at least two rows")
    xc = np.empty((rows - 1, 2 * cols - 1))
    for i in range(rows - 1):  # i is MATLAB ind - 1
        r = _xcorr_unbiased(t[i] - t[i].mean(), t[i + 1] - t[i + 1].mean())
        xc[i] = r if i % 2 == 0 else r[::-1]
    boundary = cols // 2
    m = xc[:, boundary : xc.shape[1] - boundary].mean(axis=0)
    b = int(np.argmax(m)) + 1  # MATLAB 1-based position of the maximum
    return int(mround(xc.shape[1] / 2)) - b - boundary


class CorrectLineShiftResult(NamedTuple):
    """Outputs of :func:`correct_line_shift` (MATLAB ``[data, shift]``)."""

    data: np.ndarray
    shift: int


def correct_line_shift(data, template=None, *, shift: int | None = None) -> CorrectLineShiftResult:
    """Undo the shift between scan lines (calliope ``05_LoadData/correct_line_shift.m``).

    Parameters
    ----------
    data : array_like, shape (rows, cols) or (rows, cols, frames)
    template : array_like, optional
        Image to estimate the shift on (default: the mean over frames when ``shift`` is not
        given).
    shift : int, optional
        Shift to apply (default: :func:`estimate_line_shift` of the template).

    Returns
    -------
    CorrectLineShiftResult
        ``data`` with rows 0, 2, 4, ... (MATLAB's odd rows) circularly shifted by
        ``shift`` columns, same dtype; ``shift``.
    """
    d = np.array(data, copy=True)
    if shift is None:
        if template is None:
            template = d.astype(np.float64).mean(axis=2) if d.ndim == 3 else d
        shift = estimate_line_shift(template)
    shift = int(shift)
    if shift != 0:
        d[0::2] = np.roll(d[0::2], shift, axis=1)
    return CorrectLineShiftResult(d, shift)
