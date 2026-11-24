"""2-D geometric transforms in MATLAB's convention (simtform2d, fitgeotform2d, imwarp).

A transform is a 3x3 float64 matrix ``A`` (the ``A`` property of MATLAB's ``simtform2d`` /
``affinetform2d``, ``tform.A``) acting on column vectors::

    [x'; y'; 1] = A @ [x; y; 1]

with ``x`` = column and ``y`` = row in MATLAB's 1-based pixel coordinates: the centre of the
top-left pixel is ``(x, y) = (1, 1)``, the centre of pixel ``image[r, c]`` (0-based numpy
indices) is ``(x, y) = (c + 1, r + 1)``. Point arrays are ``[N x 2]`` with columns
``(x, y)``. The conversion to scipy's 0-based ``(row, col)`` happens only inside
:func:`warp_image`.

The legacy MATLAB objects ``affine2d`` / ``fitgeotrans`` store the transposed matrix
(``tform.T == tform.A.'``, row-vector convention): pass ``T.T`` to these functions.

None of these functions has been compared with MATLAB output.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import scipy.ndimage

from calliopy._matlab import matlab_cast

__all__ = [
    "SimilarityParams",
    "fit_similarity",
    "similarity_matrix",
    "similarity_params",
    "transform_points",
    "warp_image",
]


class SimilarityParams(NamedTuple):
    """Result of :func:`similarity_params` (MATLAB ``simtform2d`` properties)."""

    scale: float
    rotation_angle: float
    translation: np.ndarray


def _sincosd(deg: float) -> tuple[float, float]:
    """sin and cos of an angle in degrees, exact at multiples of 90 (MATLAB sind/cosd)."""
    q = round(deg / 90.0)
    r = math.radians(deg - 90.0 * q)
    s, c = math.sin(r) + 0.0, math.cos(r) + 0.0
    q %= 4
    if q == 0:
        return s, c
    if q == 1:
        return c, 0.0 - s
    if q == 2:
        return 0.0 - s, 0.0 - c
    return 0.0 - c, s


def _as_tform(tform) -> np.ndarray:
    a = np.asarray(tform, dtype=np.float64)
    if a.shape != (3, 3):
        raise ValueError(f"tform must be a 3x3 matrix, got shape {a.shape}")
    if not np.array_equal(a[2], [0.0, 0.0, 1.0]):
        raise ValueError("tform must be affine: last row [0, 0, 1]")
    return a


def similarity_matrix(scale: float, rotation_deg: float, translation) -> np.ndarray:
    """Matrix of a similarity transform, MATLAB ``simtform2d(scale, rotationAngle, translation).A``.

    Parameters
    ----------
    scale
        Isotropic scale factor (> 0).
    rotation_deg
        Rotation angle in degrees, counter-clockwise in (x, y) axes. Because y (the row)
        points down in an image, a positive angle turns the image clockwise on screen.
    translation
        ``(tx, ty)`` in pixels, added after scaling and rotation.

    Returns
    -------
    numpy.ndarray
        ``[[s*cos, -s*sin, tx], [s*sin, s*cos, ty], [0, 0, 1]]``, float64 ``(3, 3)``. Sine
        and cosine are exact at multiples of 90 degrees (MATLAB ``sind``/``cosd``).
    """
    if not scale > 0:
        raise ValueError(f"scale must be positive, got {scale!r}")
    t = np.asarray(translation, dtype=np.float64).reshape(-1)
    if t.size != 2:
        raise ValueError("translation must have two elements (tx, ty)")
    s, c = _sincosd(float(rotation_deg))
    return np.array(
        [[scale * c, -scale * s, t[0]], [scale * s, scale * c, t[1]], [0.0, 0.0, 1.0]]
    )


def similarity_params(tform, *, rtol: float = 1e-9) -> SimilarityParams:
    """Scale, rotation and translation of a similarity matrix (inverse of
    :func:`similarity_matrix`).

    Parameters
    ----------
    tform
        3x3 matrix ``A`` of a non-reflective similarity (``A[0,0] == A[1,1]``,
        ``A[0,1] == -A[1,0]`` within ``rtol`` of the scale).
    rtol
        Relative tolerance of that check.

    Returns
    -------
    SimilarityParams
        ``scale`` (float), ``rotation_angle`` in degrees in ``(-180, 180]`` (float) and
        ``translation`` ``(tx, ty)`` as a float64 ``(2,)`` array; the names of the MATLAB
        ``simtform2d`` properties ``Scale``, ``RotationAngle``, ``Translation``.

    Raises
    ------
    ValueError
        ``tform`` is not a similarity (shear, anisotropic scale or reflection).
    """
    a = _as_tform(tform)
    scale = math.hypot(a[0, 0], a[1, 0])
    if scale == 0 or (
        abs(a[0, 0] - a[1, 1]) > rtol * scale or abs(a[0, 1] + a[1, 0]) > rtol * scale
    ):
        raise ValueError("tform is not a non-reflective similarity transform")
    angle = math.degrees(math.atan2(a[1, 0], a[0, 0]))
    if angle == -180.0:  # atan2(-0.0, -1) gives -180; the range is (-180, 180]
        angle = 180.0
    return SimilarityParams(scale, angle, a[:2, 2].copy())


def fit_similarity(moving_points, fixed_points) -> np.ndarray:
    """Least-squares similarity transform, MATLAB ``fitgeotform2d(moving, fixed, "similarity")``.

    Finds the non-reflective similarity (scale, rotation, translation) that maps
    ``moving_points`` onto ``fixed_points`` with the smallest sum of squared distances. With
    two point pairs the fit is exact.

    Parameters
    ----------
    moving_points, fixed_points
        ``[N x 2]`` arrays of ``(x, y)`` points (x = column, y = row, MATLAB 1-based pixel
        coordinates), ``N >= 2``; row ``i`` of both is one pair.

    Returns
    -------
    numpy.ndarray
        float64 ``(3, 3)`` matrix ``A`` with ``fixed ~ A @ [moving; 1]`` (``tform.A``).

    Raises
    ------
    ValueError
        Fewer than two pairs, shapes that differ, non-finite points, or moving points that
        are all the same.

    Notes
    -----
    Closed-form least squares on centred points, written with complex numbers
    ``z = x + i*y``: with ``zc = z - mean(z)`` and ``wc = w - mean(w)`` (moving, fixed), the
    minimiser of ``sum(|a*z + b - w|**2)`` is ``a = sum(wc * conj(zc)) / sum(|zc|**2)`` and
    ``b = mean(w) - a*mean(z)``; then ``A = [[Re a, -Im a, Re b], [Im a, Re a, Im b]]``. For
    two pairs this is ``a = (w2 - w1) / (z2 - z1)``, so axis-aligned point pairs give exact
    zeros and exact translations (no solve on raw pixel coordinates, whose rounding left
    errors of about 1e-11). MATLAB normalises the points and solves a linear system; both are
    the same least-squares minimiser, so results should agree up to rounding (unverified, in
    particular for N > 2).
    """
    mv = np.asarray(moving_points, dtype=np.float64)
    fx = np.asarray(fixed_points, dtype=np.float64)
    if mv.ndim != 2 or mv.shape[1] != 2 or mv.shape != fx.shape:
        raise ValueError(
            f"moving_points and fixed_points must both be [N x 2], got {mv.shape}, {fx.shape}"
        )
    n = mv.shape[0]
    if n < 2:
        raise ValueError("at least two point pairs are needed")
    if not (np.all(np.isfinite(mv)) and np.all(np.isfinite(fx))):
        raise ValueError("points must be finite")
    mz = mv.mean(axis=0)  # centroids (x, y)
    mw = fx.mean(axis=0)
    zx, zy = mv[:, 0] - mz[0], mv[:, 1] - mz[1]
    wx, wy = fx[:, 0] - mw[0], fx[:, 1] - mw[1]
    den = float(np.sum(zx * zx + zy * zy))
    if den == 0:
        raise ValueError("moving points are degenerate (all the same)")
    # a = sum(wc * conj(zc)) / sum(|zc|^2), in real arithmetic
    a_re = float(np.sum(wx * zx + wy * zy)) / den
    a_im = float(np.sum(wy * zx - wx * zy)) / den
    tx = mw[0] - (a_re * mz[0] - a_im * mz[1])
    ty = mw[1] - (a_im * mz[0] + a_re * mz[1])
    return np.array([[a_re, -a_im, tx], [a_im, a_re, ty], [0.0, 0.0, 1.0]])


def transform_points(tform, points) -> np.ndarray:
    """Apply a transform to points, MATLAB ``transformPointsForward(tform, points)``.

    Parameters
    ----------
    tform
        3x3 matrix ``A`` (see the module docstring).
    points
        ``[N x 2]`` (or a single ``(2,)``) array of ``(x, y)`` points.

    Returns
    -------
    numpy.ndarray
        float64 array with the shape of ``points``: ``(x', y')``. For the inverse mapping
        (``transformPointsInverse``) pass ``numpy.linalg.inv(tform)``.
    """
    a = _as_tform(tform)
    p = np.asarray(points, dtype=np.float64)
    if p.shape[-1] != 2 or p.ndim > 2:
        raise ValueError(f"points must be [N x 2] or (2,), got shape {p.shape}")
    return p @ a[:2, :2].T + a[:2, 2]


def warp_image(image, tform, output_shape, *, order: int = 1, fill_value=0) -> np.ndarray:
    """Warp an image, MATLAB ``imwarp(image, tform, 'OutputView', imref2d(output_shape))``.

    Output pixel ``(x', y')`` (1-based, x = column, y = row) takes the input value at
    ``(x, y) = A^-1 @ [x'; y'; 1]``, interpolated in the input image, which has MATLAB's
    default spatial referencing (``imref2d(size(image))``: pixel centres at integer world
    coordinates 1..n).

    Parameters
    ----------
    image
        ``(height, width)`` image, or ``(height, width, ...)``: every 2-D plane is warped with
        the same transform (MATLAB warps the planes of an RGB image or stack the same way).
    tform
        3x3 matrix ``A`` mapping input to output coordinates (``tform.A``).
    output_shape
        ``(height, width)`` of the output (MATLAB ``size(fixed)``; extra elements ignored).
    order
        0 = MATLAB ``'nearest'``, 1 = ``'linear'`` (default, as MATLAB). ``'cubic'`` is not
        provided.
    fill_value
        Value of output pixels that map outside the input (MATLAB 'FillValues', default 0).

    Returns
    -------
    numpy.ndarray
        ``(height, width) + image.shape[2:]``. Same dtype as ``image``: integer results are
        rounded and saturated as MATLAB does (:func:`~calliopy._matlab.matlab_cast`); boolean
        images are warped as float64 and returned as float64.

    Notes
    -----
    An output pixel is filled when its source point lies outside the input pixel centres,
    i.e. outside ``1 <= x <= width``, ``1 <= y <= height`` (with a tolerance of ``1e-9`` px for
    rounding); inside, the value is interpolated from the input only. Linear interpolation is
    computed as ``a + t*(b - a)`` along columns, then rows, so a region of constant value
    (e.g. a label area) is reproduced exactly; MATLAB's own formula is not documented
    (unverified in the last bit for non-constant neighbourhoods).

    Differences from MATLAB / unverified: MATLAB's handling of source points between the
    outermost pixel centre and the image edge (up to half a pixel outside) is not
    documented precisely and may differ from the rule above, so output pixels on the border
    of the valid region are unverified. Nearest-neighbour rounding of a source coordinate that
    falls exactly half-way between two pixels is unverified.
    """
    if order not in (0, 1):
        raise ValueError("order must be 0 (nearest) or 1 (linear)")
    a = _as_tform(tform)
    img = np.asarray(image)
    if img.ndim < 2:
        raise ValueError(f"image must be at least 2-D, got shape {img.shape}")
    h_out, w_out = (int(v) for v in tuple(output_shape)[:2])
    ainv = np.linalg.inv(a)
    yy, xx = np.mgrid[1 : h_out + 1, 1 : w_out + 1].astype(np.float64)
    src_c = ainv[0, 0] * xx + ainv[0, 1] * yy + ainv[0, 2] - 1.0  # 0-based column
    src_r = ainv[1, 0] * xx + ainv[1, 1] * yy + ainv[1, 2] - 1.0  # 0-based row
    h_in, w_in = img.shape[:2]
    tol = 1e-9
    inside = (src_r >= -tol) & (src_r <= h_in - 1 + tol)
    inside &= (src_c >= -tol) & (src_c <= w_in - 1 + tol)
    coords = np.stack([np.clip(src_r, 0, h_in - 1), np.clip(src_c, 0, w_in - 1)])
    work = np.float32 if img.dtype == np.float32 else np.float64
    planes = img.reshape(h_in, w_in, -1).astype(work)
    out = np.empty((h_out, w_out, planes.shape[2]), dtype=work)
    if order == 1:
        # bilinear as a + t*(b - a) in both directions: a constant neighbourhood is returned
        # exactly ((1-t)*a + t*b is not, e.g. 182305689 -> 182305688.99999997)
        r0 = np.clip(np.floor(coords[0]), 0, max(h_in - 2, 0)).astype(np.intp)
        c0 = np.clip(np.floor(coords[1]), 0, max(w_in - 2, 0)).astype(np.intp)
        r1 = np.minimum(r0 + 1, h_in - 1)
        c1 = np.minimum(c0 + 1, w_in - 1)
        tr = (coords[0] - r0).astype(work)
        tc = (coords[1] - c0).astype(work)
    for i in range(planes.shape[2]):
        plane = planes[:, :, i]
        if order == 1:
            v00, v01 = plane[r0, c0], plane[r0, c1]
            v10, v11 = plane[r1, c0], plane[r1, c1]
            top = v00 + tc * (v01 - v00)
            bottom = v10 + tc * (v11 - v10)
            vals = top + tr * (bottom - top)
        else:
            vals = scipy.ndimage.map_coordinates(
                plane, coords, order=0, mode="nearest", prefilter=False
            )
        out[:, :, i] = np.where(inside, vals, fill_value)
    out = out.reshape((h_out, w_out) + img.shape[2:])
    if img.dtype.kind in "iu":
        return matlab_cast(out, img.dtype)
    return out
