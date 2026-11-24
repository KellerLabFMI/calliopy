"""Rigid frame registration by FFT cross-correlation (port of calliope 05_LoadData).

Ports of ``register_frames.m``, ``fftreg.m`` and ``shift_data.m``, used by the widefield
registration and drift code. They are private to :mod:`calliopy.widefield`.

Naming and sign convention (taken from the MATLAB code, see the tests): in these three
functions ``shift_x`` is the shift along the first array dimension (rows) and ``shift_y``
along the second (columns), even though elsewhere x means column. ``register_frames``
returns, per frame, the integer displacement of the frame relative to the template: if
``frame[r, c] == template[r - dx, c - dy]`` (content moved ``dx`` rows down and ``dy``
columns right), it returns ``shift_x = dx``, ``shift_y = dy``; ``shift_data(frame, dx, dy)``
then gives ``out[r, c] = frame[r + dx, c + dy]``, which re-aligns the frame with the
template.

None of these functions has been compared with MATLAB output.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np

from calliopy._matlab import matlab_cast, mround, prctile

__all__ = ["FftregResult", "RegisterFramesResult", "fftreg", "register_frames", "shift_data"]

logger = logging.getLogger(__name__)

_NBIM_TEMP = 10  # template refresh: mean over the last 10 aligned frames
_NBIM_REFRESH = 50  # ... every 50 frames
_HIGH_PASS = 3  # FFT bins removed around zero frequency
_LOW_PASS = 60  # FFT bins kept on each side of zero frequency


class FftregResult(NamedTuple):
    """Result of :func:`fftreg` (MATLAB ``[row_shift, col_shift]``)."""

    row_shift: float
    col_shift: float


class RegisterFramesResult(NamedTuple):
    """Result of :func:`register_frames` (MATLAB ``[shift_x, shift_y]``)."""

    shift_x: np.ndarray
    shift_y: np.ndarray


def _as_class(x, dtype):
    """Result of MATLAB arithmetic stored in ``dtype``'s class (identity for double)."""
    if dtype == np.float64:
        return np.asarray(x, dtype=np.float64)
    return matlab_cast(x, dtype)


def fftreg(buf1ft, buf2ft) -> FftregResult:
    """Integer shift between two images from their 2-D FFTs.

    Port of calliope ``05_LoadData/fftreg.m``. The cross-correlation
    ``CC = ifft2(buf1ft .* conj(buf2ft))`` is maximal at the displacement of image 1
    relative to image 2: if ``image1[r, c] == image2[r - dr, c - dc]`` (circularly), the
    result is ``(dr, dc)``.

    Parameters
    ----------
    buf1ft, buf2ft
        ``numpy.fft.fft2`` of two real images of the same shape ``(m, n)``.

    Returns
    -------
    FftregResult
        ``row_shift``, ``col_shift`` (floats holding integers, as MATLAB doubles). Peaks
        in the upper half are wrapped to negative shifts: ``row_shift`` lies in
        ``[fix(m/2) - m, fix(m/2) - 1]``, ``col_shift`` likewise with ``n``.

    Notes
    -----
    The peak is the first maximum in column-major order, as MATLAB's ``max(CC)`` then
    ``max(max1)`` pick it. ``CC`` is taken as real (``.real``): for the transforms of real
    images MATLAB's ``ifft2`` detects the conjugate symmetry and returns a real array; that
    MATLAB returns real here is unverified (if it did not, ``max`` would compare
    magnitudes).
    """
    b1 = np.asarray(buf1ft)
    b2 = np.asarray(buf2ft)
    if b1.ndim != 2 or b1.shape != b2.shape:
        raise ValueError(f"buf1ft and buf2ft must be 2-D of equal shape: {b1.shape}, {b2.shape}")
    m, n = b1.shape
    cc = np.fft.ifft2(b1 * np.conj(b2)).real
    k = int(np.argmax(cc.ravel(order="F")))
    rloc, cloc = k % m + 1, k // m + 1  # 1-based, as in MATLAB
    row_shift = rloc - m - 1 if rloc > m // 2 else rloc - 1
    col_shift = cloc - n - 1 if cloc > n // 2 else cloc - 1
    return FftregResult(float(row_shift), float(col_shift))


def shift_data(data, shift_x, shift_y) -> np.ndarray:
    """Shift every frame of a stack by an integer number of rows and columns.

    Port of calliope ``05_LoadData/shift_data.m``. Frame ``i`` of the output is
    ``out[r, c, i] = data[r + sx[i], c + sy[i], i]`` with ``sx = round(shift_x)``,
    ``sy = round(shift_y)`` (MATLAB ``round``); positions that fall outside the frame take
    the median of the top-left pixel over all frames, ``median(data[0, 0, :])``, converted to
    the class of ``data``.

    Parameters
    ----------
    data
        ``[height x width x nFrames]`` stack, or one ``[height x width]`` frame; any numeric
        dtype.
    shift_x
        Shift along rows (first dimension), in pixels: a scalar (same for all frames) or one
        value per frame. Positive values move the content up (towards row 0); the
        displacement returned by :func:`register_frames` re-aligns a frame.
    shift_y
        Shift along columns (second dimension), likewise; positive moves content left.

    Returns
    -------
    numpy.ndarray
        Same shape and dtype as ``data``.

    Raises
    ------
    ValueError
        The number of shifts is neither 1 nor nFrames, or a shift is not finite.

    Notes
    -----
    Differences from MATLAB: the stack is not padded as a whole (the output is the same,
    with less memory); the message MATLAB prints when ``padarray`` is missing is dropped.
    A shift vector whose length differs from nFrames raises; MATLAB used its first nFrames
    values when it was longer and failed when it was shorter. Non-finite shifts raise
    (MATLAB failed with an indexing error).
    """
    arr = np.asarray(data)
    squeeze = arr.ndim == 2
    if squeeze:
        arr = arr[:, :, np.newaxis]
    if arr.ndim != 3:
        raise ValueError(f"data must be [height x width x nFrames], got shape {arr.shape}")
    h, w, n = arr.shape
    sx = _per_frame(shift_x, n, "shift_x")
    sy = _per_frame(shift_y, n, "shift_y")
    padval = matlab_cast(np.median(arr[0, 0, :].astype(np.float64)), arr.dtype)
    out = np.empty_like(arr)
    for i in range(n):
        out[:, :, i] = padval
        dr, dc = int(sx[i]), int(sy[i])
        r0, r1 = max(0, -dr), min(h, h - dr)  # output rows whose source row is inside
        c0, c1 = max(0, -dc), min(w, w - dc)
        if r0 < r1 and c0 < c1:
            out[r0:r1, c0:c1, i] = arr[r0 + dr : r1 + dr, c0 + dc : c1 + dc, i]
    return out[:, :, 0] if squeeze else out


def _per_frame(shift, n, name):
    s = np.asarray(shift, dtype=np.float64).ravel(order="F")
    if s.size == 1:
        s = np.repeat(s, n)
    if s.size != n:
        raise ValueError(f"{name} must have 1 or {n} elements, got {s.size}")
    if not np.all(np.isfinite(s)):
        raise ValueError(f"{name} must be finite")
    return mround(s).astype(np.int64)


def _clip_and_filter(template):
    """Clip at round(prctile(template(:), 95)) and band-pass the FFT, as register_frames.m."""
    upper_lim = mround(prctile(template, 95))
    template = template.copy()
    template[template > upper_lim] = upper_lim
    f = np.fft.fft2(template.astype(np.float64))
    for axis in (0, 1):
        m = f.shape[axis]
        idx = [slice(None), slice(None)]
        # MATLAB 1-based rows [1:3, m-1:m] and [62 : m-60]
        hp = np.r_[np.arange(min(_HIGH_PASS, m)), np.arange(max(m - _HIGH_PASS + 1, 0), m)]
        idx[axis] = hp
        f[tuple(idx)] = 0
        lo, hi = _LOW_PASS + 1, m - _LOW_PASS
        if hi > lo:
            idx[axis] = slice(lo, hi)
            f[tuple(idx)] = 0
    return template, f


def register_frames(data, template=None, perc_boundary: float = 0.1) -> RegisterFramesResult:
    """Rigid (integer-pixel) displacement of every frame relative to a template.

    Port of calliope ``05_LoadData/register_frames.m`` (GK 2011). Steps, as in MATLAB:

    1. ``template`` minus its mean, stored in the template's class (an integer template
       saturates: negative values become 0 for unsigned types, see Notes);
    2. a border of ``boundary = round(perc_boundary * max(height, width))`` pixels is cut
       from template and frames (``round(perc_boundary * min(height, width))`` when the first
       value is not smaller than the height);
    3. the template is clipped at ``round(prctile(template(:), 95))``; in its 2-D FFT the bins
       0-2 and the last two of each axis (high-pass) and the bins from 61 to ``m - 61``
       (0-based, low-pass) are set to zero;
    4. each frame's displacement is found with :func:`fftreg` (frame FFT, template FFT);
    5. at every 50th frame (1-based frame numbers 50, 100, ...) the template is renewed:
       the previous 10 frames are aligned with :func:`shift_data` using their shifts, their
       mean is averaged with the clipped original template (``(mean + template_orig) / 2``, in
       the template's class), and step 3 is repeated on the result.

    Parameters
    ----------
    data
        ``[height x width x nFrames]`` stack (any numeric dtype; memory maps are read frame
        by frame).
    template
        ``[height x width]`` reference image. Default: the mean over frames,
        ``mean(data, 3)`` (float64; float32 for float32 data).
    perc_boundary
        Fraction of the image size cut from each border (MATLAB default 0.1).

    Returns
    -------
    RegisterFramesResult
        ``shift_x`` (rows) and ``shift_y`` (columns), float64 ``[nFrames x 1]`` arrays of
        integer displacements of each frame relative to the template, as returned by
        :func:`fftreg` (see the module docstring for the sign). Pass them to
        :func:`shift_data` to align the stack.

    Raises
    ------
    ValueError
        ``data`` is not 3-D, ``template`` has a different frame size, or fewer than 3 rows or
        columns remain after cutting the border.

    Notes
    -----
    The template keeps its numeric class through the computation, as in MATLAB: the
    calliope callers pass ``wid_data(:,:,1)`` (uint16), for which ``template - mean`` is
    rounded and saturated at 0, and the renewed template is rounded twice
    (``uint16(mean + t)``, then ``uint16(./2)``). This is reproduced deliberately. For an
    integer template, ``prctile`` is evaluated in float64 and rounded, which equals MATLAB's
    integer-class interpolation up to ties at exactly .5 for signed types (unverified).
    For a float32 template the arithmetic is done in float64 and rounded to float32
    (MATLAB computes in single; unverified, may differ in the last bit).

    Differences from MATLAB: a frame cropped to fewer than 3 rows or columns raises
    (MATLAB grew the FFT array when zeroing rows 1-3 and returned meaningless shifts, or
    failed); a template of the wrong size raises before the loop (MATLAB failed in
    :func:`fftreg`).
    """
    arr = np.asarray(data)
    if arr.ndim != 3:
        raise ValueError(f"data must be [height x width x nFrames], got shape {arr.shape}")
    h, w, n = arr.shape
    if template is None:
        if arr.dtype == np.float32:
            template = np.mean(arr, axis=2, dtype=np.float64).astype(np.float32)
        else:
            template = np.mean(arr, axis=2, dtype=np.float64)
    template = np.asarray(template)
    if template.shape != (h, w):
        raise ValueError(f"template must be {(h, w)}, got {template.shape}")
    # class in which MATLAB stores the template arithmetic (logical -> double)
    tclass = template.dtype if template.dtype.kind in "iu" else np.dtype(np.float64)
    if template.dtype == np.float32:
        tclass = np.dtype(np.float32)

    tmean = np.mean(template, dtype=np.float64)
    template = _as_class(template.astype(np.float64) - tmean, tclass)
    boundary = int(mround(perc_boundary * max(h, w)))
    if boundary >= h:
        boundary = int(mround(perc_boundary * min(h, w)))
    rows = slice(boundary, h - boundary)
    cols = slice(boundary, w - boundary)
    hc, wc = h - 2 * boundary, w - 2 * boundary
    if hc < 3 or wc < 3:
        raise ValueError(
            f"only {max(hc, 0)} x {max(wc, 0)} pixels remain after cutting a border of "
            f"{boundary}; at least 3 x 3 are needed"
        )
    template = template[rows, cols]
    template, fft2_template = _clip_and_filter(template)
    template_orig = template

    shifts = np.zeros((n, 2))
    for ind in range(1, n + 1):  # 1-based frame number, as in MATLAB
        if ind % _NBIM_REFRESH == 0:
            prev = slice(ind - _NBIM_TEMP - 1, ind - 1)  # 1-based frames ind-10 .. ind-1
            temp_ali = shift_data(
                np.asarray(arr[rows, cols, prev], dtype=np.float64),
                shifts[prev, 0],
                shifts[prev, 1],
            )
            mean_ali = np.mean(temp_ali, axis=2)
            summed = _as_class(mean_ali + template_orig.astype(np.float64), tclass)
            template = _as_class(summed.astype(np.float64) / 2, tclass)
            template, fft2_template = _clip_and_filter(template)
        frame = np.asarray(arr[rows, cols, ind - 1], dtype=np.float64)
        shifts[ind - 1] = fftreg(np.fft.fft2(frame), fft2_template)
    logger.debug("register_frames: %d frames, border %d px", n, boundary)
    return RegisterFramesResult(shifts[:, 0:1].copy(), shifts[:, 1:2].copy())
