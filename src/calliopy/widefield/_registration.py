"""Registration of widefield experiments to their site (port of calliope 12_Widefield).

Ports of ``register_wid.m`` (main function and its local ``compute_tform``), the
computational core of ``propagate_tform_and_ROIs_from_expid.m`` and the local
``compute_tform`` of ``register_widefield_exp_manual.m``.

MATLAB estimates the similarity transform with Image Processing Toolbox ``imregtform``
(``'similarity'``, ``OnePlusOneEvolutionary`` optimiser, ``MattesMutualInformation`` metric),
which has no numpy/scipy equivalent. :func:`imregtform_similarity` implements the same
method (Mattes mutual information, (1+1) evolutionary strategy, coarse-to-fine pyramid) with
numpy/scipy. Its result is not expected to equal MATLAB's digit for digit; the tests check
that known similarity transforms are recovered to a small fraction of a pixel.

Transforms are 3x3 matrices in the convention of :mod:`calliopy.widefield._transform`
(MATLAB ``tform.A``, acting on 1-based ``[x; y; 1]`` with x = column, y = row).

None of these functions has been compared with MATLAB output.
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np
import scipy.ndimage

from calliopy._matlab import corr2, filter2, mround
from calliopy.io import MatStruct, load_roi_data, load_wid_data, load_wii_data
from calliopy.widefield._frames import register_frames
from calliopy.widefield._transform import fit_similarity, warp_image

__all__ = [
    "SIMILARITY_THRESHOLD",
    "ManualTformResult",
    "PropagateTformResult",
    "StackAlignmentError",
    "TformResult",
    "compute_tform",
    "compute_tform_manual",
    "crop_margins",
    "imregtform_similarity",
    "plot_registration_check",
    "propagate_tform",
    "register_wid",
    "registration_similarity",
    "uncrop_tform",
]

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.5
"""corr2 below which register_wid's compute_tform reports status 'FAILED'."""

_MAX_STACK_SHIFT = 5  # px, register_wid's between-stack consistency limit
_PARZEN_PAD = 2  # histogram bins kept free at each end for the B-spline Parzen window


class TformResult(NamedTuple):
    """Result of :func:`compute_tform` (MATLAB ``[tform, registration_parameters]``)."""

    tform: np.ndarray
    registration_parameters: MatStruct


class ManualTformResult(NamedTuple):
    """Result of :func:`compute_tform_manual` (``tform`` and ``similarity``)."""

    tform: np.ndarray
    similarity: float


class PropagateTformResult(NamedTuple):
    """Result of :func:`propagate_tform` (what MATLAB saved into the inheriting adata)."""

    tform: np.ndarray
    registration_parameters: MatStruct
    ROIs: object


class StackAlignmentError(RuntimeError):
    """The stacks of one experiment are shifted against each other by more than 5 px."""


# ======================================================================================
# imregtform replacement
# ======================================================================================


def _reduce(img: np.ndarray) -> np.ndarray:
    """One pyramid level down: Gaussian smoothing (sigma 1 px) and every second pixel."""
    return scipy.ndimage.gaussian_filter(img, 1.0, mode="nearest")[::2, ::2]


def _bin_range(img: np.ndarray, n_bins: int) -> tuple[float, float]:
    vals = img[np.isfinite(img)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    size = (hi - lo) / (n_bins - 2 * _PARZEN_PAD)
    return lo, size if size > 0 else 1.0


class _MattesLevel:
    """Mattes mutual information between a fixed and a moving image at one pyramid level.

    All fixed pixels are samples (MATLAB ``UseAllPixels = true``). Fixed intensities are
    binned with a zero-order (box) window, moving intensities with a cubic B-spline Parzen
    window (Mattes et al. 2003); samples that map outside the moving image or are not finite
    are left out. Coordinates are 0-based (x = column, y = row) of the full-resolution image.
    """

    def __init__(self, fixed, moving, factor, centre, length, n_bins):
        self.n_bins = n_bins
        self.moving = moving
        self.factor = factor
        self.centre = centre
        self.length = length
        flo, fsize = _bin_range(fixed, n_bins)
        self.mlo, self.msize = _bin_range(moving, n_bins)
        fvals = fixed.ravel()
        good = np.isfinite(fvals)
        fbin = np.floor((np.where(good, fvals, flo) - flo) / fsize + _PARZEN_PAD)
        fbin = np.clip(fbin, _PARZEN_PAD, n_bins - _PARZEN_PAD - 1).astype(np.int64)
        gy, gx = np.mgrid[0 : fixed.shape[0], 0 : fixed.shape[1]]
        self.fbase = (fbin * n_bins)[good]
        self.gx = (gx.ravel()[good] * factor).astype(np.float64)
        self.gy = (gy.ravel()[good] * factor).astype(np.float64)

    def __call__(self, p: np.ndarray) -> float:
        """Negative mutual information (nats) for parameters ``p`` (see _params_to_a0)."""
        theta, dscale, tx, ty = p
        s = 1.0 + dscale
        if s <= 0:
            return 0.0
        co, si = math.cos(theta), math.sin(theta)
        cx, cy = self.centre
        dx = self.gx - cx - tx * self.length
        dy = self.gy - cy - ty * self.length
        xm = (cx + (co * dx + si * dy) / s) / self.factor
        ym = (cy + (-si * dx + co * dy) / s) / self.factor
        h, w = self.moving.shape
        ok = (xm >= 0) & (xm <= w - 1) & (ym >= 0) & (ym <= h - 1)
        if np.count_nonzero(ok) < 16:
            return 0.0
        vm = scipy.ndimage.map_coordinates(
            self.moving, [ym[ok], xm[ok]], order=1, prefilter=False
        )
        fin = np.isfinite(vm)
        vm = vm[fin]
        fbase = self.fbase[ok][fin]
        nb = self.n_bins
        cm = (vm - self.mlo) / self.msize + _PARZEN_PAD
        k0 = np.floor(cm).astype(np.int64)
        u = cm - k0
        u2 = u * u
        u3 = u2 * u
        weights = ((1 - u) ** 3 / 6, (3 * u3 - 6 * u2 + 4) / 6, (-3 * u3 + 3 * u2 + 3 * u + 1) / 6,
                   u3 / 6)
        idx = np.concatenate([fbase + np.clip(k0 + j, 0, nb - 1) for j in (-1, 0, 1, 2)])
        joint = np.bincount(idx, weights=np.concatenate(weights), minlength=nb * nb)
        total = joint.sum()
        if total <= 0:
            return 0.0
        pj = joint.reshape(nb, nb) / total
        pf = pj.sum(axis=1, keepdims=True)
        pm = pj.sum(axis=0, keepdims=True)
        nz = pj > 0
        return -float(np.sum(pj[nz] * np.log(pj[nz] / (pf * pm)[nz])))


def _one_plus_one(fun, p0, rng, growth, radius, epsilon, max_iter):
    """(1+1) evolutionary strategy (Styner et al. 2000), as ITK's OnePlusOneEvolutionary.

    The search distribution ``A`` starts as ``radius * I``; after each trial it is updated
    with ``A += (c - 1) * (A z) z' / |z|^2`` where ``c`` is ``growth`` after an improvement
    and ``growth ** -0.25`` otherwise. Stops when ``||A||_F <= epsilon`` or after
    ``max_iter`` trials.
    """
    n = len(p0)
    a = radius * np.eye(n)
    shrink = growth**-0.25
    parent = np.asarray(p0, dtype=np.float64).copy()
    f_parent = fun(parent)
    for _ in range(int(max_iter)):
        z = rng.standard_normal(n)
        step = a @ z
        child = parent + step
        f_child = fun(child)
        if f_child < f_parent:
            parent, f_parent, adjust = child, f_child, growth
        else:
            adjust = shrink
        a = a + (adjust - 1.0) * np.outer(step, z) / float(z @ z)
        if np.linalg.norm(a) <= epsilon:
            break
    return parent


def _phase_correlation(moving: np.ndarray, fixed: np.ndarray) -> tuple[int, int]:
    """Integer (row, col) displacement of ``moving`` relative to ``fixed`` (Hann-windowed)."""
    a = np.nan_to_num(moving - np.nanmean(moving))
    b = np.nan_to_num(fixed - np.nanmean(fixed))
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    cross = np.fft.fft2(a * win) * np.conj(np.fft.fft2(b * win))
    cross /= np.abs(cross) + np.finfo(np.float64).tiny
    cc = np.fft.ifft2(cross).real
    r, c = np.unravel_index(int(np.argmax(cc)), cc.shape)
    m, n = cc.shape
    return (r - m if r > m // 2 else r), (c - n if c > n // 2 else c)


def imregtform_similarity(
    moving,
    fixed,
    *,
    rng=None,
    pyramid_levels: int = 3,
    number_of_histogram_bins: int = 50,
    growth_factor: float = 1.03,
    initial_radius: float = 0.003,
    epsilon: float = 1.5e-4,
    maximum_iterations: int = 300,
    initialize_translation: bool = False,
) -> np.ndarray:
    """Similarity transform maximising Mattes mutual information, as MATLAB ``imregtform``.

    Replaces ``imregtform(moving, fixed, 'similarity', optimizer, metric)`` with
    ``registration.optimizer.OnePlusOneEvolutionary`` and
    ``registration.metric.MattesMutualInformation`` (Image Processing Toolbox), written with
    numpy/scipy:

    - metric: Mattes mutual information with ``number_of_histogram_bins`` bins (MATLAB
      default 50), all fixed pixels as samples (MATLAB default ``UseAllPixels``), cubic
      B-spline Parzen window for the moving and box window for the fixed intensities;
      linear interpolation of the moving image; samples mapping outside it are left out;
    - optimiser: (1+1) evolutionary strategy with ``growth_factor``, shrink factor
      ``growth_factor ** -0.25``, ``initial_radius``, ``epsilon`` (stop when the Frobenius
      norm of the search matrix falls to it) and ``maximum_iterations`` per pyramid level;
      defaults are the settings of ``register_wid.m``;
    - pyramid: ``pyramid_levels`` levels (MATLAB default 3), each a Gaussian smoothing
      (sigma 1 px) and 2x decimation of the previous; coarse to fine, each level starting
      from the previous result and with a new search radius.

    Parameters
    ----------
    moving, fixed
        2-D images (any real dtype, used as float64).
    rng
        ``numpy.random.Generator`` or seed for the optimiser's random steps. ``None`` uses
        the fixed seed 0, so repeated calls give identical results.
    pyramid_levels, number_of_histogram_bins, growth_factor, initial_radius, epsilon, \
maximum_iterations
        See above.
    initialize_translation
        Opt-in, default ``False``: start from the integer translation found by FFT phase
        correlation (only when both images have the same shape). By default the optimiser
        starts from the identity, as MATLAB's ``imregtform``.

    Returns
    -------
    numpy.ndarray
        float64 ``(3, 3)`` matrix ``A`` (MATLAB ``tform.A``) mapping moving to fixed
        coordinates, 1-based (x = column, y = row), so that
        ``warp_image(moving, A, fixed.shape)`` aligns ``moving`` with ``fixed``.

    Notes
    -----
    Parameterisation: rotation angle (radians) and scale about the centre of ``fixed``,
    and translation; scale (as ``scale - 1``) and translation (as a fraction of half the
    larger image side) are normalised so that one unit of every parameter moves the image
    border by about half the image size. ``initial_radius`` and ``epsilon`` apply in these
    units.

    Differences from MATLAB (the result will not equal MATLAB's digit for digit):

    - MATLAB's internal parameter scales, pyramid filter, sample handling at the image
      border and random number generator are not documented and are not reproduced;
      the choices above are this implementation's own.
    - The default start is the identity, as in MATLAB. The phase-correlation start
      (``initialize_translation=True``) is an optional extra that MATLAB does not have; it is
      off by default.
    - Non-finite pixels are left out of the histogram.
    """
    mov = np.asarray(moving, dtype=np.float64)
    fix = np.asarray(fixed, dtype=np.float64)
    if mov.ndim != 2 or fix.ndim != 2:
        raise ValueError("moving and fixed must be 2-D images")
    if min(mov.shape) < 4 or min(fix.shape) < 4:
        raise ValueError("images must be at least 4 x 4 pixels")
    if int(pyramid_levels) < 1:
        raise ValueError("pyramid_levels must be >= 1")
    if int(number_of_histogram_bins) < 2 * _PARZEN_PAD + 1:
        raise ValueError("number_of_histogram_bins is too small")
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(0 if rng is None else rng)
    h, w = fix.shape
    centre = ((w - 1) / 2.0, (h - 1) / 2.0)  # 0-based
    length = max(h, w) / 2.0

    pyramid = [(fix, mov)]
    for _ in range(int(pyramid_levels) - 1):
        f_prev, m_prev = pyramid[-1]
        if min(f_prev.shape) < 8 or min(m_prev.shape) < 8:
            break
        pyramid.append((_reduce(f_prev), _reduce(m_prev)))

    p = np.zeros(4)
    if initialize_translation and mov.shape == fix.shape:
        dr, dc = _phase_correlation(mov, fix)
        p[2], p[3] = -dc / length, -dr / length
    for level in range(len(pyramid) - 1, -1, -1):
        f_l, m_l = pyramid[level]
        metric = _MattesLevel(f_l, m_l, 2**level, centre, length, int(number_of_histogram_bins))
        p = _one_plus_one(metric, p, rng, growth_factor, initial_radius, epsilon,
                          maximum_iterations)
    return _params_to_matrix(p, centre, length)


def _params_to_matrix(p, centre, length) -> np.ndarray:
    """Parameters (theta, scale - 1, tx/L, ty/L) about 0-based ``centre`` -> 1-based tform.A."""
    theta, dscale, tx, ty = (float(v) for v in p)
    s = 1.0 + dscale
    rot = s * np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    c = np.asarray(centre, dtype=np.float64)
    a0 = np.eye(3)
    a0[:2, :2] = rot
    a0[:2, 2] = c + np.array([tx, ty]) * length - rot @ c
    # 0-based -> 1-based pixel coordinates: A1 = T(+1) A0 T(-1)
    a1 = a0.copy()
    a1[:2, 2] = a0[:2, 2] + 1.0 - rot @ np.ones(2)
    return a1


# ======================================================================================
# register_wid.m: compute_tform
# ======================================================================================


def crop_margins(shape, frac_boundary: float) -> tuple[int, int]:
    """Crop margins ``(dw, dh)`` of register_wid: ``round(frac_boundary * [width, height])``.

    Parameters
    ----------
    shape
        ``(height, width)`` of the moving image (MATLAB ``size(moving)``).
    frac_boundary
        Fraction of the image removed on each side.

    Returns
    -------
    tuple of int
        ``dw`` (columns) and ``dh`` (rows) removed on each side, MATLAB ``round``.
    """
    fb = float(frac_boundary)
    if not math.isfinite(fb):
        raise ValueError(f"frac_boundary must be finite, got {frac_boundary!r}")
    return int(mround(fb * shape[1])), int(mround(fb * shape[0]))


def _crop(img: np.ndarray, dw: int, dh: int) -> np.ndarray:
    """MATLAB ``img(1+dh:end-dh, 1+dw:end-dw)``."""
    h, w = img.shape[:2]
    return img[dh : h - dh, dw : w - dw]


def uncrop_tform(tform_cropped, dw: int, dh: int) -> np.ndarray:
    """Translation correction of register_wid for a transform estimated on cropped images.

    Reproduces ``dshift = T - ((A(1:2,1:2) - eye(2)) * [dw; dh])'`` and writes ``dshift``
    into ``A(1:2, 3)``: the transform of the uncropped images is the cropped one conjugated
    with a shift by the crop margin.

    Parameters
    ----------
    tform_cropped
        3x3 ``tform.A`` estimated on ``img(1+dh:end-dh, 1+dw:end-dw)``.
    dw, dh
        Columns and rows removed on each side (see :func:`crop_margins`).

    Returns
    -------
    numpy.ndarray
        float64 ``(3, 3)`` matrix for the uncropped images.
    """
    a = np.array(tform_cropped, dtype=np.float64)
    if a.shape != (3, 3):
        raise ValueError("tform_cropped must be 3x3")
    lin = a[:2, :2]
    a[:2, 2] = a[:2, 2] - (lin - np.eye(2)) @ np.array([dw, dh], dtype=np.float64)
    return a


def registration_similarity(moving, fixed, tform, frac_boundary: float) -> float:
    """Similarity score of register_wid: corr2 of the warped moving and the fixed image,
    both cropped by the margins of :func:`crop_margins`.

    ``imwarp(moving, tform, 'OutputView', imref2d(size(fixed)))`` (linear, fill 0, class of
    ``moving`` kept), then ``corr2(warped(1+dh:end-dh, 1+dw:end-dw), fixed(...))`` with
    ``dw, dh`` computed from the size of ``moving`` (as in MATLAB).

    Returns
    -------
    float
        Correlation coefficient; NaN for constant input.
    """
    mv = np.asarray(moving)
    fx = np.asarray(fixed)
    dw, dh = crop_margins(mv.shape, frac_boundary)
    registered = warp_image(mv, tform, fx.shape)
    return corr2(_crop(registered, dw, dh), _crop(fx, dw, dh))


def _local_normalise(img: np.ndarray, win_size: int) -> np.ndarray:
    """``double(img) ./ filter2(ones(win)/win^2, img)`` (zero-padded local mean)."""
    kernel = np.ones((win_size, win_size)) / win_size**2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(img, dtype=np.float64) / filter2(kernel, img)


def compute_tform(
    moving,
    fixed,
    *,
    frac_boundary: float = 0.1,
    normalize_template: bool = True,
    rng=None,
    initialize_translation: bool = False,
) -> TformResult:
    """Similarity transform between an experiment's first frame and its site's first frame.

    Port of the local function ``compute_tform`` of calliope ``12_Widefield/register_wid.m``:

    1. ``dw = round(frac_boundary * width)``, ``dh = round(frac_boundary * height)`` of
       ``moving``;
    2. if ``normalize_template``: both images divided by their local mean,
       ``filter2(ones(win)/win^2, img)`` with ``win = round(min(size(fixed))/20)``;
    3. both cropped to ``img(1+dh:end-dh, 1+dw:end-dw)`` and registered with
       :func:`imregtform_similarity` (MATLAB ``imregtform``, similarity, Mattes MI,
       OnePlusOneEvolutionary with GrowthFactor 1.03, InitialRadius 0.003, Epsilon 1.5e-4,
       MaximumIterations 300);
    4. the translation corrected for the crop margin (:func:`uncrop_tform`);
    5. the *raw* moving image warped onto the fixed grid and the similarity computed as
       ``corr2`` on the cropped region (:func:`registration_similarity`); status ``'FAILED'``
       when it is below 0.5, else ``'SUCCESS'``.

    Parameters
    ----------
    moving
        ``[height x width]`` first frame of the experiment to register (any numeric class).
    fixed
        ``[height x width]`` first frame of the site's origin experiment.
    frac_boundary
        Fraction of the image cut on each side (MATLAB default 0.1).
    normalize_template
        Divide by the local average before registering (MATLAB default 1).
    rng
        ``numpy.random.Generator`` or seed for the optimiser; ``None`` = seed 0.
    initialize_translation
        Opt-in, default ``False`` (identity start, as MATLAB): start the optimiser at the
        phase-correlation translation, an optional extra not in MATLAB (see
        :func:`imregtform_similarity`).

    Returns
    -------
    TformResult
        ``tform``: float64 ``(3, 3)`` ``tform.A`` mapping moving to fixed, 1-based (x =
        column, y = row). ``registration_parameters``: :class:`~calliopy.io.MatStruct` with
        the MATLAB fields ``status`` (``'SUCCESS'``/``'FAILED'``), ``similarity``,
        ``similarity_threshold`` (0.5), ``fraction_boundary``, ``registration_type``
        (``'automatic'``) and ``activity_calculated`` (0.0).

    Notes
    -----
    Differences from MATLAB:

    - The transform comes from :func:`imregtform_similarity`, not ``imregtform``; it will
      not equal MATLAB's digit for digit (see that function). By default it starts from the
      identity, as MATLAB; the phase-correlation start is an optional extra, off by default.
      Everything around it is ported as written.
    - Capture range of the identity start (synthetic 96 x 112 images, local normalisation
      with a 5 px window): rotation + scale, translations up to about 4 px and small
      combinations are recovered; translations of about 6 px, and 3-4 px translations
      combined with a 2-3 degree rotation and 2 % scale, are not. Pass
      ``initialize_translation=True`` for such cases. Without normalisation the identity
      start recovered translations up to 8 px. Whether MATLAB's imregtform has the same
      limits is unverified.
    - The ``fprintf`` status line is logged (INFO). The ``verbose`` figures are
      :func:`plot_registration_check`. ``ExpID`` and ``SiteID`` (only used for printing)
      are not arguments.
    - ``tform`` is a 3x3 matrix, not a ``simtform2d`` object.
    """
    mv = np.asarray(moving)
    fx = np.asarray(fixed)
    if mv.ndim != 2 or fx.ndim != 2:
        raise ValueError("moving and fixed must be 2-D images")
    dw, dh = crop_margins(mv.shape, frac_boundary)
    mv_n, fx_n = mv, fx
    if normalize_template:
        win = int(mround(min(fx.shape) / 20))
        if win < 1:
            raise ValueError("fixed is too small for the local normalisation window")
        mv_n = _local_normalise(mv, win)
        fx_n = _local_normalise(fx, win)
    mv_c = _crop(mv_n, dw, dh)
    fx_c = _crop(fx_n, dw, dh)
    if min(mv_c.shape) < 4 or min(fx_c.shape) < 4:
        raise ValueError("too few pixels remain after cropping by frac_boundary")
    tform_cropped = imregtform_similarity(
        mv_c, fx_c, rng=rng, initialize_translation=initialize_translation
    )
    tform = uncrop_tform(tform_cropped, dw, dh)
    similarity = registration_similarity(mv, fx, tform, frac_boundary)
    status = "FAILED" if similarity < SIMILARITY_THRESHOLD else "SUCCESS"
    logger.info("Registration status: %s. Similarity = %.2f", status, similarity)
    params = MatStruct()
    params.status = status
    params.similarity = float(similarity)
    params.similarity_threshold = SIMILARITY_THRESHOLD
    params.fraction_boundary = float(frac_boundary)
    params.registration_type = "automatic"
    params.activity_calculated = 0.0
    return TformResult(tform, params)


def plot_registration_check(
    moving,
    fixed,
    tform,
    *,
    frac_boundary: float = 0.1,
    normalize_template: bool = True,
    title: str | None = None,
):
    """Figure of the ``verbose`` branch of register_wid's ``compute_tform``.

    Three panels: ``imshowpair(fixed, movingRegistered, 'falsecolor')`` with the crop
    rectangle in red (``fixed`` is the locally normalised image when
    ``normalize_template``, as in MATLAB), then ``imagesc`` of the cropped raw fixed and
    of the cropped registered moving image.

    Parameters
    ----------
    moving, fixed
        Raw first frames, as passed to :func:`compute_tform`.
    tform
        3x3 transform from :func:`compute_tform`.
    frac_boundary, normalize_template
        As in :func:`compute_tform`.
    title
        Title of the overlay panel (MATLAB: status, ExpID, SiteID, similarity).

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Differences from MATLAB: one figure with three axes instead of three figures. The
    false-colour overlay puts each image, scaled to [0, 1] independently, into red+blue
    (fixed) and green (registered moving), as ``imshowpair``'s default
    ``'ColorChannels', 'green-magenta'`` (unverified in detail).
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    mv = np.asarray(moving)
    fx = np.asarray(fixed)
    dw, dh = crop_margins(mv.shape, frac_boundary)
    registered = warp_image(mv, tform, fx.shape)
    shown_fixed = fx
    if normalize_template:
        shown_fixed = _local_normalise(fx, int(mround(min(fx.shape) / 20)))

    def scale01(img):
        a = np.asarray(img, dtype=np.float64)
        lo, hi = np.nanmin(a), np.nanmax(a)
        return np.zeros_like(a) if hi <= lo else (a - lo) / (hi - lo)

    f01, m01 = scale01(shown_fixed), scale01(registered)
    rgb = np.clip(np.stack([f01, m01, f01], axis=2), 0, 1)
    fig = Figure(figsize=(12, 4))
    axs = fig.subplots(1, 3)
    axs[0].imshow(rgb, interpolation="nearest")
    h, w = mv.shape
    axs[0].add_patch(Rectangle((dw - 0.5, dh - 0.5), w - 2 * dw, h - 2 * dh,
                               edgecolor="red", facecolor="none", linewidth=1.5))
    if title:
        axs[0].set_title(title)
    axs[1].imshow(_crop(fx, dw, dh), aspect="auto", interpolation="nearest")
    axs[2].imshow(_crop(registered, dw, dh), aspect="auto", interpolation="nearest")
    return fig


# ======================================================================================
# register_widefield_exp_manual.m: compute_tform (two reference point pairs)
# ======================================================================================


def compute_tform_manual(
    moving, fixed, moving_points, fixed_points, *, frac_boundary: float = 0.1
) -> ManualTformResult:
    """Transform from reference point pairs and its similarity score.

    Port of the local function ``compute_tform`` of calliope
    ``12_Widefield/register_widefield_exp_manual.m`` (the interactive tool itself is
    deferred): ``fitgeotform2d([current refpoint1; refpoint2], [orig refpoint1; refpoint2],
    'similarity')``, then the similarity as in register_wid (corr2 on the region cropped by
    ``frac_boundary``).

    Parameters
    ----------
    moving
        ``[height x width]`` image of the current experiment.
    fixed
        ``[height x width]`` image of the origin experiment.
    moving_points, fixed_points
        ``[N x 2]`` reference points ``(x, y)`` = (column, row), 1-based as MATLAB
        ``ginput`` returns them; MATLAB uses exactly two pairs, any ``N >= 2`` works.
    frac_boundary
        Crop fraction for the similarity (MATLAB ``params.frac_boundary``).

    Returns
    -------
    ManualTformResult
        ``tform`` (3x3 ``tform.A``, moving to fixed) and ``similarity`` (corr2).

    Notes
    -----
    Differences from MATLAB: no figure (``view_stack`` of the overlay) and no printing;
    the result is returned instead of stored in the figure's ``UserData``. Beyond
    :func:`~calliopy.widefield._transform.fit_similarity` this adds only the similarity
    score; no status is set (the manual tool sets ``'SUCCESS'`` when saving).
    """
    tform = fit_similarity(moving_points, fixed_points)
    return ManualTformResult(tform, registration_similarity(moving, fixed, tform, frac_boundary))


# ======================================================================================
# propagate_tform_and_ROIs_from_expid.m
# ======================================================================================


def propagate_tform(
    first_frame,
    site_first_frame,
    tform,
    registration_parameters,
    *,
    exp_id_origin: int,
    exp_id: int | None = None,
    rois=None,
) -> PropagateTformResult:
    """Copy one experiment's transform to another experiment of the same site and session.

    Computational core of calliope ``12_Widefield/propagate_tform_and_ROIs_from_expid.m``
    for one inheriting experiment: the inheriting experiment's ``first_frame`` is warped
    with the origin experiment's ``tform`` onto the site's first frame, the similarity is
    recomputed on the region cropped by ``registration_parameters.fraction_boundary``, and
    the copied ``registration_parameters`` are updated: ``registration_type = 'Copied from
    <exp_id_origin>'``, ``status = 'SUCCESS'`` (always, as in MATLAB), ``similarity``, and
    ``similarity_threshold`` (unchanged). When the similarity is below the threshold, the
    MATLAB warning is issued (:class:`UserWarning`).

    Parameters
    ----------
    first_frame
        ``[height x width]`` ``first_frame`` of the inheriting experiment's adata.
    site_first_frame
        ``[height x width]`` ``first_frame`` of the site's origin experiment.
    tform
        3x3 ``tform`` (``tform.A``) of the experiment whose registration is copied.
    registration_parameters
        That experiment's ``registration_parameters`` (mapping with at least
        ``fraction_boundary`` and ``similarity_threshold``); not modified.
    exp_id_origin
        ExpID whose transform is copied (for ``registration_type``).
    exp_id
        ExpID of the inheriting experiment (only for the warning text).
    rois
        The origin's ``ROIs`` (MATLAB cell array; here a sequence or object array whose
        first element is a struct), or ``None`` when the origin adata has none.

    Returns
    -------
    PropagateTformResult
        ``tform`` (the input, as float64 copy), ``registration_parameters`` (new
        :class:`~calliopy.io.MatStruct`), and ``ROIs``: ``None`` when ``rois`` is
        ``None``; otherwise a copy whose first element has no ``activity`` field, and then
        ``registration_parameters.activity_calculated`` is 0. Without ROIs,
        ``activity_calculated`` keeps the origin's value, as in MATLAB.

    Raises
    ------
    ValueError
        ``tform`` is empty (an ORIGIN experiment has ``tform = []``) or
        ``fraction_boundary`` is not finite (MATLAB failed in ``imwarp`` / indexing).

    Notes
    -----
    Deferred: the ExpLog/adata lookups (``find_adata_file``, ``read_info_from_ExpLog``), the
    same-mouse / same-user checks and saving into the inheriting adata file; the loop over
    several inheriting experiments is the caller's.
    """
    tf = np.asarray(tform, dtype=np.float64)
    if tf.size == 0:
        raise ValueError("tform is empty: an ORIGIN experiment has no transform to copy")
    frac = float(registration_parameters["fraction_boundary"])
    threshold = registration_parameters["similarity_threshold"]
    similarity = registration_similarity(first_frame, site_first_frame, tf, frac)
    params = MatStruct(registration_parameters)
    params.registration_type = f"Copied from {int(exp_id_origin)}"
    label = "?" if exp_id is None else str(int(exp_id))
    if similarity < threshold:
        warnings.warn(
            f"ExpID {label} is registered as success even though similarity = "
            f"{similarity:.2f}. \nCheck that this ExpID is indeed aligned by running "
            "check_widefield_alignment",
            UserWarning,
            stacklevel=2,
        )
    else:
        logger.info("Similarity for ExpID %s is %.2f", label, similarity)
    params.status = "SUCCESS"
    params.similarity = float(similarity)
    params.similarity_threshold = threshold
    out_rois = None
    if rois is not None:
        out_rois = _drop_activity(rois)
        params.activity_calculated = 0.0
    return PropagateTformResult(tf.copy(), params, out_rois)


def _drop_activity(rois):
    """Copy of a ROIs cell array with the ``activity`` field removed from element 1."""
    if isinstance(rois, np.ndarray):
        out = rois.copy()
        flat = out.reshape(-1, order="F")
    else:
        out = list(rois)
        flat = out
    if not len(flat):
        return out
    first = flat[0]
    if isinstance(first, Mapping) and "activity" in first:
        first = MatStruct(first)
        del first["activity"]
        flat[0] = first
    elif isinstance(first, np.ndarray) and first.dtype == object:
        # ROIs{1} is a struct array (the usual widefield case): MATLAB's rmfield removes
        # the field from every element.
        elems = first.reshape(-1, order="F")
        if len(elems) and all(isinstance(e, Mapping) for e in elems) and any(
            "activity" in e for e in elems
        ):
            stripped = np.empty(first.shape, dtype=object)
            sflat = stripped.reshape(-1, order="F")
            for k, e in enumerate(elems):
                s = MatStruct(e)
                s.pop("activity", None)
                sflat[k] = s
            flat[0] = stripped
    return out


# ======================================================================================
# register_wid.m: main function
# ======================================================================================


def _as_path_list(paths) -> list[Path]:
    if isinstance(paths, (str, os.PathLike)):
        return [Path(paths)]
    out = [Path(p) for p in paths]
    if not out:
        raise ValueError("no files given")
    return out


def _without_dimension(meta) -> np.ndarray:
    """MATLAB ``rmfield(meta_info, 'dimension')`` on a struct array."""
    arr = np.asarray(meta, dtype=object)
    out = np.empty(arr.shape, dtype=object)
    for idx in np.ndindex(arr.shape):
        s = MatStruct(arr[idx])
        s.pop("dimension", None)
        out[idx] = s
    return out


def _empty() -> np.ndarray:
    return np.zeros((0, 0))


def _origin_parameters() -> MatStruct:
    params = MatStruct()
    params.status = "ORIGIN"
    params.similarity = math.nan
    params.similarity_threshold = math.nan
    params.fraction_boundary = math.nan
    params.registration_type = math.nan
    params.activity_calculated = 0.0
    return params


def register_wid(
    wid_paths,
    *,
    site_first_frame=None,
    frac_boundary: float = 0.1,
    normalize_template: bool = True,
    check_stacks: bool = True,
    rng=None,
    initialize_translation: bool = False,
) -> MatStruct:
    """Register one widefield experiment to its site; return the adata variables.

    Port of the main function of calliope ``12_Widefield/register_wid.m`` for one
    experiment whose files are given directly:

    1. ``meta_info`` = per-frame meta info of the (first channel's) files,
       ``load_wid_data(..., 'meta')`` without the field ``dimension``;
    2. per channel: ``first_frame<ending>`` = frame 100 of the first stack
       (``load_wid_data(..., '100th')``); with ``check_stacks`` and more than one stack, the
       frame-100 of every stack is registered against that of stack 1 with
       :func:`~calliopy.widefield._frames.register_frames` (template in its stored class,
       uint16, as in MATLAB) and a shift of more than 5 px raises
       :class:`StackAlignmentError` (MATLAB printed and returned without saving);
    3. per channel: for the site's origin experiment (``site_first_frame is None``)
       ``tform<ending> = []`` and ``registration_parameters<ending>`` with status
       ``'ORIGIN'``; for two-colour data the channel map ``tform_ch_map`` /
       ``registration_parameters_ch_map`` = :func:`compute_tform` of ``first_frame_ch525``
       against ``fliplr(first_frame_ch610)``. Otherwise ``tform<ending>`` and
       ``registration_parameters<ending>`` = :func:`compute_tform` of this experiment's
       first frame against the origin's.

    Parameters
    ----------
    wid_paths
        The experiment's files in recording order: a path or sequence of ``.wid`` paths
        (single channel, ending ``''``); a mapping from channel ending to paths for two-colour
        data, e.g. ``{'_ch525': [...], '_ch610': [...]}`` (MATLAB ``wid_fnames_ch525``,
        ``wid_fnames_ch610``), processed in the mapping's order; or ``.roi`` paths for trace
        imaging (see Notes).
    site_first_frame
        ``None`` when this experiment is the site's origin. Otherwise the origin's
        ``first_frame`` (single channel or trace imaging) or a mapping from channel ending
        to the origin's ``first_frame<ending>`` (two colours).
    frac_boundary, normalize_template, rng, initialize_translation
        Passed to :func:`compute_tform` (MATLAB defaults 0.1 and 1; ``initialize_translation``
        defaults to ``False``, the identity start of MATLAB). One generator is used for all
        channels in order.
    check_stacks
        Check the between-stack alignment (MATLAB default 1).

    Returns
    -------
    MatStruct
        The variables MATLAB adds to ``Adata-S1-T<ExpID>.mat``, in MATLAB's order:
        ``meta_info`` ((1, nFrames) object array of structs), then per channel
        ``first_frame<ending>`` (uint16 ``[height x width]``), ``tform<ending>`` (3x3, or
        ``(0, 0)`` for the origin), ``registration_parameters<ending>``, and for a
        two-colour origin ``tform_ch_map`` and ``registration_parameters_ch_map``.

    Raises
    ------
    StackAlignmentError
        Stacks shifted by more than 5 px against stack 1.
    ValueError
        Unknown channel layout, missing ``site_first_frame`` entries, or a two-colour
        origin whose ``'_ch610'`` comes before ``'_ch525'``.

    Notes
    -----
    Trace imaging (only ``.roi`` files): ``meta_info`` is read with
    :func:`~calliopy.io.load_roi_data` (``'meta'``), and ``first_frame`` is the mean
    projection image of the ``.wii`` file of the first ``.roi`` file
    (:func:`~calliopy.io.load_wii_data`). MATLAB reads ``meta_info(1).img_mean``, a field
    that the current ``load_roi_data.m`` no longer sets (the line is commented out), so the
    MATLAB branch fails; the mean projection is what the commented-out code stored there.

    Deferred: the ExpLog lookup of files and SiteID, the file-type checks
    (``list_file_types``), the ``overwrite_flag`` logic that reads an existing adata file,
    creating folders and saving the adata file, and the recursion over several ExpIDs.

    Differences from MATLAB: the stack misalignment raises instead of returning silently;
    the trace-imaging branch is fixed as described; printing is logging; the ``verbose``
    figures are :func:`plot_registration_check`.
    """
    if isinstance(wid_paths, Mapping):
        channels = {str(k): _as_path_list(v) for k, v in wid_paths.items()}
        if not channels:
            raise ValueError("wid_paths is an empty mapping")
    else:
        channels = {"": _as_path_list(wid_paths)}
    suffixes = {p.suffix.lower() for paths in channels.values() for p in paths}
    trace = suffixes == {".roi"}
    if not trace and suffixes != {".wid"}:
        raise ValueError(f"expected only .wid or only .roi files, got {sorted(suffixes)}")
    if trace and list(channels) != [""]:
        raise ValueError("trace imaging (.roi) is single channel: pass a path or a sequence")
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(0 if rng is None else rng)
    is_origin = site_first_frame is None
    if not is_origin and len(channels) > 1 and not isinstance(site_first_frame, Mapping):
        raise ValueError("two-colour data: site_first_frame must map channel ending to frame")

    adata = MatStruct()
    first_paths = next(iter(channels.values()))
    if trace:
        meta = load_roi_data(first_paths, "meta").roi_meta_info
    else:
        meta = load_wid_data(first_paths, "meta").wid_meta_info
    adata.meta_info = _without_dimension(meta)

    for ending, paths in channels.items():
        if trace:
            first = np.array(load_wii_data(paths[0].with_suffix(".wii")).wii_mean_proj)
        else:
            wid = load_wid_data(paths, "100th").wid_data
            first = np.array(wid[:, :, 0])
            if check_stacks and wid.shape[2] > 1:
                dx, dy = register_frames(wid, wid[:, :, 0])
                dx = dx - dx[0]
                dy = dy - dy[0]
                if np.max(np.abs(dx)) > _MAX_STACK_SHIFT or np.max(np.abs(dy)) > _MAX_STACK_SHIFT:
                    del wid
                    raise StackAlignmentError(
                        "stackIDs out of alignment - there is no fix for this yet "
                        f"(channel '{ending}': max |shift| {np.max(np.abs(dx)):g} rows, "
                        f"{np.max(np.abs(dy)):g} columns)"
                    )
            del wid
        adata["first_frame" + ending] = first

        if is_origin:
            adata["tform" + ending] = _empty()
            adata["registration_parameters" + ending] = _origin_parameters()
            if ending in ("_ch525", ""):
                logger.info("Registration status: SUCCESS. It is the first ExpID of this SiteID")
            if ending == "_ch610":
                if "first_frame_ch525" not in adata:
                    raise ValueError("channel '_ch525' must come before '_ch610'")
                res = compute_tform(
                    adata["first_frame_ch525"],
                    np.fliplr(first),
                    frac_boundary=frac_boundary,
                    normalize_template=normalize_template,
                    rng=rng,
                    initialize_translation=initialize_translation,
                )
                adata.tform_ch_map = res.tform
                adata.registration_parameters_ch_map = res.registration_parameters
        else:
            if isinstance(site_first_frame, Mapping):
                if ending not in site_first_frame:
                    raise ValueError(f"site_first_frame has no entry for channel '{ending}'")
                fixed = site_first_frame[ending]
            else:
                fixed = site_first_frame
            res = compute_tform(
                first,
                np.asarray(fixed),
                frac_boundary=frac_boundary,
                normalize_template=normalize_template,
                rng=rng,
                initialize_translation=initialize_translation,
            )
            adata["tform" + ending] = res.tform
            adata["registration_parameters" + ending] = res.registration_parameters
    return adata
