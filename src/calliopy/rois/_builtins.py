"""Private equivalents of the MATLAB Image Processing Toolbox functions RoiQuest uses.

``fspecial('disk')``, ``mat2gray``, ``quantile``, ``strel('disk', r, 0)``, ``imtophat``,
``imclose``, ``bwareaopen``, ``bwperim`` and ``bwmorph`` (``'thin'``, ``'thicken'``,
``'spur'``). They live here, not in :mod:`calliopy._matlab`, because this track may not edit
that module; they are candidates for consolidation there.

None of these functions has been compared with MATLAB output (see each docstring for what is
known to differ).
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage

from calliopy._matlab import prctile

__all__: list[str] = []

_CONN8 = np.ones((3, 3), dtype=bool)


def fspecial_disk(radius: float) -> np.ndarray:
    """Averaging disk kernel, MATLAB ``fspecial('disk', radius)``.

    Each weight is the area of the pixel (unit square centred on the integer grid
    ``-crad .. crad``, ``crad = ceil(radius - 0.5)``) covered by a disk of ``radius``, with
    MATLAB's corrections for the centre pixel and the four axis pixels at the rim; the kernel
    is normalised to sum 1. This reproduces the formula of MATLAB's ``fspecial`` (the brief
    for this port asked for the exact MATLAB kernel). The radius-1 kernel equals the values
    printed in MATLAB's documentation to 4 decimals (corner 0.0251, edge 0.1453, centre
    0.3183); the radius-2 values used in the tests are recalled MATLAB output, not a checked
    reference. Unverified against MATLAB until ``ref_rq_fspecial`` passes.

    Parameters
    ----------
    radius
        Positive radius in pixels (non-integer allowed, as in MATLAB).

    Returns
    -------
    numpy.ndarray
        float64 ``(2*crad + 1, 2*crad + 1)``.
    """
    rad = float(radius)
    if not rad > 0:
        raise ValueError(f"radius must be positive, got {radius!r}")
    crad = int(np.ceil(rad - 0.5))
    x, y = np.meshgrid(np.arange(-crad, crad + 1), np.arange(-crad, crad + 1))
    maxxy = np.maximum(np.abs(x), np.abs(y)).astype(np.float64)
    minxy = np.minimum(np.abs(x), np.abs(y)).astype(np.float64)
    r2 = rad**2

    def sq(v):
        return np.sqrt(np.asarray(v, dtype=np.complex128))

    def asin(v):
        return np.arcsin(np.asarray(v, dtype=np.complex128))

    c1 = r2 < (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2
    m1 = c1 * (minxy - 0.5) + (~c1) * sq(r2 - (maxxy + 0.5) ** 2)
    c2 = r2 > (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2
    m2 = c2 * (minxy + 0.5) + (~c2) * sq(r2 - (maxxy - 0.5) ** 2)
    sel = ((r2 < (maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2)
           & (r2 > (maxxy - 0.5) ** 2 + (minxy - 0.5) ** 2)) | (
        (minxy == 0) & (maxxy - 0.5 < rad) & (maxxy + 0.5 >= rad))
    sgrid = (r2 * (0.5 * (asin(m2 / rad) - asin(m1 / rad))
                   + 0.25 * (np.sin(2 * asin(m2 / rad)) - np.sin(2 * asin(m1 / rad))))
             - (maxxy - 0.5) * (m2 - m1) + (m1 - minxy + 0.5)) * sel
    sgrid = sgrid + ((maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2 < r2)
    sgrid[crad, crad] = min(np.pi * r2, np.pi / 2)
    if crad > 0 and rad > crad - 0.5 and r2 < (crad - 0.5) ** 2 + 0.25:
        m1s = np.sqrt(r2 - (crad - 0.5) ** 2)
        m1n = m1s / rad
        sg0 = 2 * (r2 * (0.5 * np.arcsin(m1n) + 0.25 * np.sin(2 * np.arcsin(m1n)))
                   - m1s * (crad - 0.5))
        for a, b in ((2 * crad, crad), (crad, 2 * crad), (crad, 0), (0, crad)):
            sgrid[a, b] = sg0
        for a, b in ((2 * crad - 1, crad), (crad, 2 * crad - 1), (crad, 1), (1, crad)):
            sgrid[a, b] = sgrid[a, b] - sg0
    sgrid[crad, crad] = min(sgrid[crad, crad].real, 1.0)
    sgrid = sgrid.real
    return sgrid / sgrid.sum()


def mat2gray(a, limits=None) -> np.ndarray:
    """Scale to ``[0, 1]``, MATLAB ``mat2gray(A, [amin amax])``.

    ``limits`` defaults to ``[min(A(:)) max(A(:))]`` (NaNs ignored). The result is
    ``A*delta - amin*delta`` with ``delta = 1/(amax - amin)``, clipped to ``[0, 1]``; for equal
    limits the input itself is clipped (unverified MATLAB behaviour). NaN stays NaN.
    """
    arr = np.asarray(a, dtype=np.float64)
    if limits is None:
        if arr.size == 0 or np.all(np.isnan(arr)):
            lo = hi = np.nan
        else:
            lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    else:
        lo, hi = (float(v) for v in np.ravel(limits)[:2])
    if hi == lo:
        out = arr.copy()
    else:
        delta = 1.0 / (hi - lo)
        out = arr * delta + (-lo * delta)
    return np.clip(out, 0.0, 1.0)


def quantile(x, p) -> np.ndarray | float:
    """MATLAB ``quantile(x(:), p)`` for ``p`` in ``[0, 1]``: ``prctile(x(:), 100*p)``."""
    return prctile(np.asarray(x, dtype=np.float64).ravel(), np.asarray(p, dtype=np.float64) * 100)


def strel_disk(radius: int) -> np.ndarray:
    """Flat disk footprint, MATLAB ``strel('disk', radius, 0)``: ``x^2 + y^2 <= radius^2``.

    Differences from MATLAB: ``strel('disk', r)`` without ``N`` uses ``N = 4`` and approximates
    the disk by periodic lines; for small radii (``r <= 2``) the shapes agree, for larger radii
    MATLAB's footprint is a different (octagon-like) shape. Unverified.
    """
    r = int(radius)
    if r < 0 or r != radius:
        raise ValueError(f"radius must be a non-negative integer, got {radius!r}")
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    return (x**2 + y**2) <= r**2


def imerode_gray(img, footprint) -> np.ndarray:
    """Flat grey erosion, MATLAB ``imerode`` (outside the image counts as ``+Inf``)."""
    a = np.asarray(img, dtype=np.float64)
    return scipy.ndimage.grey_erosion(a, footprint=footprint, mode="constant", cval=np.inf)


def imdilate_gray(img, footprint) -> np.ndarray:
    """Flat grey dilation, MATLAB ``imdilate`` (outside the image counts as ``-Inf``)."""
    a = np.asarray(img, dtype=np.float64)
    fp = np.asarray(footprint, dtype=bool)  # only symmetric footprints are used here
    return scipy.ndimage.grey_dilation(a, footprint=fp, mode="constant", cval=-np.inf)


def imtophat(img, footprint) -> np.ndarray:
    """White top-hat, MATLAB ``imtophat(I, se)``: ``I - imopen(I, se)`` (float64)."""
    a = np.asarray(img, dtype=np.float64)
    return a - imdilate_gray(imerode_gray(a, footprint), footprint)


def imclose_binary(mask, footprint) -> np.ndarray:
    """Binary closing, MATLAB ``imclose(BW, se)``: dilation (outside = 0), then erosion
    (outside = 1). Only symmetric footprints (disks) are used in this package."""
    bw = np.asarray(mask, dtype=bool)
    fp = np.asarray(footprint, dtype=bool)
    dil = scipy.ndimage.binary_dilation(bw, structure=fp, border_value=0)
    return scipy.ndimage.binary_erosion(dil, structure=fp, border_value=1)


def bwareaopen(mask, p: int) -> np.ndarray:
    """Remove 8-connected objects with fewer than ``p`` pixels, MATLAB ``bwareaopen(BW, P)``."""
    bw = np.asarray(mask, dtype=bool)
    if bw.size == 0:
        return bw.copy()
    lab, n = scipy.ndimage.label(bw, structure=_CONN8)
    if n == 0:
        return bw.copy()
    sizes = np.bincount(lab.ravel())
    keep = sizes >= p
    keep[0] = False
    return keep[lab]


def bwperim(mask) -> np.ndarray:
    """Perimeter pixels, MATLAB ``bwperim(BW)`` (4-connected): foreground pixels with a
    background 4-neighbour; pixels outside the image count as background."""
    bw = np.asarray(mask, dtype=bool)
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    return bw & ~scipy.ndimage.binary_erosion(bw, structure=cross, border_value=0)


def _neighbours(bw: np.ndarray) -> list[np.ndarray]:
    """x1..x8 of every pixel (x1 = east, counter-clockwise), outside = False."""
    p = np.pad(bw, 1, constant_values=False)
    h, w = bw.shape

    def at(dr, dc):
        return p[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]

    return [at(0, 1), at(-1, 1), at(-1, 0), at(-1, -1),
            at(0, -1), at(1, -1), at(1, 0), at(1, 1)]


def _thin_subiteration(bw: np.ndarray, first: bool) -> np.ndarray:
    x = _neighbours(bw)
    x = [None] + x + [x[0]]  # 1-based, x[9] = x[1]
    xh = sum(((~x[2 * i - 1]) & (x[2 * i] | x[2 * i + 1])).astype(np.int8) for i in range(1, 5))
    n1 = sum((x[2 * k - 1] | x[2 * k]).astype(np.int8) for k in range(1, 5))
    n2 = sum((x[2 * k] | x[2 * k + 1]).astype(np.int8) for k in range(1, 5))
    mn = np.minimum(n1, n2)
    g12 = (xh == 1) & (mn >= 2) & (mn <= 3)
    if first:
        g3 = ~((x[2] | x[3] | ~x[8]) & x[1])
    else:
        g3 = ~((x[6] | x[7] | ~x[4]) & x[5])
    return bw & ~(g12 & g3)


def _iterations(n) -> int:
    if n is None or (isinstance(n, float) and np.isinf(n)):
        return -1
    n = int(n)
    if n < 0:
        raise ValueError("number of iterations must be non-negative or inf")
    return n


def bwmorph_thin(mask, n=1) -> np.ndarray:
    """Thinning, MATLAB ``bwmorph(BW, 'thin', n)``.

    Implements the published two-subiteration algorithm (Lam, Lee and Suen 1992, the variant
    described in MATLAB's documentation of ``bwmorph``): a foreground pixel is removed in the
    first subiteration when conditions G1, G2 and G3 hold, in the second when G1, G2 and G3'
    hold. Pixels outside the image count as background. ``n = inf`` repeats until nothing
    changes. Unverified against MATLAB.
    """
    bw = np.asarray(mask, dtype=bool).copy()
    if bw.ndim != 2:
        raise ValueError("mask must be 2-D")
    k = _iterations(n)
    it = 0
    while k < 0 or it < k:
        new = _thin_subiteration(_thin_subiteration(bw, True), False)
        it += 1
        if np.array_equal(new, bw):
            break
        bw = new
    return bw


def bwmorph_thicken(mask, n=1) -> np.ndarray:
    """Thickening, MATLAB ``bwmorph(BW, 'thicken', n)``, as thinning of the background.

    Each iteration thins the complement of the image (padded with background, so that the
    outer background is connected beyond the border) by one iteration of
    :func:`bwmorph_thin`. MATLAB documents thickening as adding pixels to the exterior of
    objects without making unconnected objects 8-connected; its implementation details
    (e.g. extra handling of isolated pixels) are not documented, so results may differ.
    Known limitation of this construction: an isolated single pixel (and the corners of
    objects) do not grow, because condition G2 protects background pixels surrounded by
    background. Unverified against MATLAB.
    """
    bw = np.asarray(mask, dtype=bool)
    if bw.ndim != 2:
        raise ValueError("mask must be 2-D")
    k = _iterations(n)
    comp = np.pad(~bw, 1, constant_values=True)
    it = 0
    while k < 0 or it < k:
        new = _thin_subiteration(_thin_subiteration(comp, True), False)
        new[0, :] = new[-1, :] = True
        new[:, 0] = new[:, -1] = True
        it += 1
        if np.array_equal(new, comp):
            break
        comp = new
    return ~comp[1:-1, 1:-1]


def bwmorph_spur(mask, n=1) -> np.ndarray:
    """Spur removal, MATLAB ``bwmorph(BW, 'spur', n)``: each iteration removes, all at once,
    the foreground pixels that have exactly one 8-connected foreground neighbour (isolated
    pixels are kept). Matches the example in MATLAB's documentation; otherwise unverified."""
    bw = np.asarray(mask, dtype=bool).copy()
    k = _iterations(n)
    it = 0
    while k < 0 or it < k:
        cnt = sum(v.astype(np.int8) for v in _neighbours(bw))
        new = bw & ~(cnt == 1)
        it += 1
        if np.array_equal(new, bw):
            break
        bw = new
    return bw
