"""Mouse position tracking in overhead ("owl") camera videos: calliope
``10_Behavior/preprocess_owl.m``."""

from __future__ import annotations

import math

import numpy as np
import scipy.ndimage

from .._matlab import linspace, matlab_cast, mround

__all__ = ["preprocess_owl", "rgb2gray_uint8"]

# MATLAB rgb2gray weights (rgb2ntsc first row, as rgb2gray applies them)
_GRAY_W = np.array([0.298936021293775, 0.587043074451121, 0.114020904255103])
_CONN8 = np.ones((3, 3), dtype=bool)


def rgb2gray_uint8(frame) -> np.ndarray:
    """MATLAB ``rgb2gray`` for a ``uint8`` RGB image ``(height, width, 3)``.

    ``round(0.298936021293775*R + 0.587043074451121*G + 0.114020904255103*B)`` saturated to
    ``uint8``. Not compared with MATLAB output: the rounding of exact halves is assumed to
    follow MATLAB's ``round`` (half away from zero).
    """
    f = np.asarray(frame)
    if f.ndim != 3 or f.shape[2] != 3:
        raise ValueError("frame must be (height, width, 3)")
    g = f.astype(np.float64) @ _GRAY_W
    return matlab_cast(g, np.uint8)


def _largest_blob_centroid(mask: np.ndarray, min_area: int):
    """``bwareaopen(mask, min_area)`` then the centroid of the largest component.

    8-connectivity as MATLAB's defaults. Components are ordered as MATLAB's
    ``bwconncomp`` (by the column-major position of their first pixel) so that the first of
    several equally large components wins, as ``max`` does in MATLAB. Returns
    ``(x, y)`` in MATLAB's 1-based pixel coordinates, or ``None``.
    """
    lab, n = scipy.ndimage.label(mask, structure=_CONN8)
    if n == 0:
        return None
    flat = lab.ravel(order="F")
    areas = np.bincount(flat, minlength=n + 1)[1:]
    pos = np.flatnonzero(flat)
    first = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first, flat[pos] - 1, pos)
    keep = np.flatnonzero(areas >= min_area)
    if keep.size == 0:
        return None
    order = keep[np.argsort(first[keep], kind="stable")]
    best = order[np.argmax(areas[order])] + 1
    rows, cols = np.nonzero(lab == best)
    return float(cols.mean() + 1), float(rows.mean() + 1)


def preprocess_owl(frames, *, threshold: float = 20, min_area: int = 50, n_samples: int = 50):
    """Track the mouse in an overhead video by background subtraction.

    Computational core of calliope ``10_Behavior/preprocess_owl.m``. The background is the
    median of ``n_samples`` frames taken at ``floor(linspace(1, n_frames, n_samples))``
    (1-based). In every frame, pixels darker than the background by more than ``threshold``
    (``uint8`` subtraction, saturating at 0) form a mask; components smaller than
    ``min_area`` pixels are removed (``bwareaopen``) and the centroid of the largest
    remaining component (``regionprops``) is the mouse position.

    Parameters
    ----------
    frames : array_like
        The video as ``uint8``: ``(height, width, 3, n_frames)`` RGB (converted with
        MATLAB's ``rgb2gray`` weights, :func:`rgb2gray_uint8`) or ``(height, width,
        n_frames)`` already grey. A ``numpy.memmap`` is read one frame at a time.
    threshold : float, default 20
        MATLAB ``thresholdValue``.
    min_area : int, default 50
        Minimum component size in pixels (MATLAB ``bwareaopen(..., 50)``).
    n_samples : int, default 50
        Number of frames for the median background (MATLAB ``numSamples``).

    Returns
    -------
    calliopy.io.MatStruct
        What MATLAB adds to the adata file: a MatStruct with the single variable ``owl``,
        itself a MatStruct with fields

        - ``KP``: MatStruct with ``mouse_x``, ``mouse_y``, float64 columns
          ``(n_frames, 1)`` in MATLAB's **1-based** pixel coordinates (x = column,
          y = row), NaN where no blob was found (MATLAB stores a ``table`` with these two
          variables);
        - ``median_frame``: ``uint8`` ``(height, width)`` background;
        - ``first_frame``: the first frame as given (RGB for RGB input);
        - ``last_frame``: the last frame, **grey** (MATLAB overwrites ``frame`` with its
          grey version before storing it).

    Differences from MATLAB
    -----------------------
    - Takes the decoded frames instead of an ExpID: reading the ``.avi`` (``VideoReader``)
      and writing ``owl`` into the adata file are deferred (no video reader and no adata
      writing in calliopy). The MATLAB ``table`` ``KP`` is returned as a MatStruct.
    - Background frames are taken by index; MATLAB seeks by time
      (``CurrentTime = (k-1)/FrameRate``), which normally selects the same frame.
    - The unused parameters ``frameRate`` and ``gridSize`` are dropped.
    """
    from ..io import MatStruct

    v = frames
    shape = np.shape(v)
    if len(shape) == 4 and shape[2] == 3:
        rgb = True
        n_frames = shape[3]
    elif len(shape) == 3:
        rgb = False
        n_frames = shape[2]
    else:
        raise ValueError("frames must be (h, w, 3, n_frames) or (h, w, n_frames)")
    if n_frames == 0:
        raise ValueError("the video has no frames")

    def raw(k: int):
        return np.asarray(v[:, :, :, k] if rgb else v[:, :, k])

    def gray(k: int) -> np.ndarray:
        f = raw(k)
        g = rgb2gray_uint8(f) if rgb else f
        if g.dtype != np.uint8:
            g = matlab_cast(g, np.uint8)
        return g

    idx = np.floor(linspace(1, n_frames, n_samples)).astype(np.int64) - 1
    stack = np.stack([gray(int(k)) for k in idx], axis=2)
    background = matlab_cast(mround(np.median(stack.astype(np.float64), axis=2)), np.uint8)

    xs = np.full(n_frames, np.nan)
    ys = np.full(n_frames, np.nan)
    bg = background.astype(np.int16)
    last = None
    for k in range(n_frames):
        g = gray(k)
        diff = np.maximum(bg - g.astype(np.int16), 0)
        c = _largest_blob_centroid(diff > threshold, math.ceil(min_area))
        if c is not None:
            xs[k], ys[k] = c
        last = g
    kp = MatStruct(mouse_x=xs.reshape(-1, 1), mouse_y=ys.reshape(-1, 1))
    return MatStruct(
        owl=MatStruct(KP=kp, median_frame=background, first_frame=raw(0), last_frame=last)
    )
