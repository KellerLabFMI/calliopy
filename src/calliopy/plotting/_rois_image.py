"""ROI label / overlay images: calliope ``22_Plotting/ROIs2image.m``."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import scipy.ndimage

from .._matlab import mround, prctile

__all__ = ["rois_to_image"]

_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def _jet(m: int) -> np.ndarray:
    """MATLAB ``jet(m)`` colormap, ``(m, 3)`` (MATLAB's own construction)."""
    if m <= 0:
        return np.zeros((0, 3))
    n = math.ceil(m / 4)
    u = np.concatenate([np.arange(1, n + 1) / n, np.ones(n - 1), np.arange(n, 0, -1) / n])
    g = math.ceil(n / 2) - (1 if m % 4 == 1 else 0) + np.arange(1, u.size + 1)
    r = g + n
    b = g - n
    g = g[g <= m]
    r = r[r <= m]
    b = b[b >= 1]
    j = np.zeros((m, 3))
    j[r - 1, 0] = u[: r.size]
    j[g - 1, 1] = u[: g.size]
    j[b - 1, 2] = u[u.size - b.size :]
    return j


def _bwperim(mask: np.ndarray) -> np.ndarray:
    """MATLAB ``bwperim`` (4-connected perimeter; outside the image counts as background)."""
    inner = scipy.ndimage.binary_erosion(mask, structure=_CROSS, border_value=0)
    return mask & ~inner


def _roi_indices(roi) -> np.ndarray:
    if isinstance(roi, Mapping):
        if "indices" in roi:
            idx = roi["indices"]
        elif "PixelIdxList" in roi:
            idx = roi["PixelIdxList"]
        else:
            raise ValueError("ROIs do not contain field 'PixelIdxList' or 'indices'!")
    else:
        idx = roi
    return np.asarray(idx, dtype=np.int64).ravel(order="F")


def rois_to_image(
    rois,
    dim,
    *,
    randcolor: bool = True,
    singlecolor=None,
    template=None,
    type: str = "full",
    sel=None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Paint ROIs into an image, as labels or as a coloured overlay on a template.

    Port of calliope ``22_Plotting/ROIs2image.m`` (Marcus, 2012).

    Parameters
    ----------
    rois : sequence
        One entry per ROI: a mapping (MatStruct / dict) with ``indices`` or
        ``PixelIdxList``, or directly the index array. Indices are MATLAB **1-based**
        column-major linear indices into an image of size ``dim`` (the values stored in
        the ROI files).
    dim : tuple of two ints
        Image size ``(height, width)``.
    randcolor : bool, default True
        Random colours (MATLAB ``'randcolor'``); needs ``rng``.
    singlecolor : RGB triple, optional
        One colour for all ROIs (MATLAB ``'singlecolor'``).
    template : array_like (height, width), optional
        Background image; the result is then an RGB overlay.
    type : {'full', 'perim'}, default 'full'
        Fill the ROI or draw its 4-connected perimeter (``bwperim``). Without a template
        anything other than ``'full'`` draws perimeters, as MATLAB.
    sel : array_like, optional
        Selected ROIs and their colour groups. **0-based** ROI positions in ``rois``:
        a 1-D array (all in colour group 1), or ``(k, 2)`` rows ``[roi, group]`` where
        ``group`` is a colour-group number ``1..G`` (a label, as in MATLAB). A ``(2, k)``
        array is transposed, as MATLAB does whenever the first dimension is the smaller one
        (this includes ``(2, 2)``). Default: every ROI is its own group
        (MATLAB ``expmat((1:n)', 2)``).
    rng : numpy.random.Generator, optional
        Needed for random colours (``randcolor`` without ``singlecolor``).

    Returns
    -------
    numpy.ndarray
        Without template: float64 ``(height, width)`` label image. Selected ROIs get the
        value ``colors[group]`` with ``colors`` a random permutation of ``2..G+1``
        (``randcolor``), ``2..G+1`` in order (``randcolor=False``) or
        ``10 * round(sum(singlecolor))``; unselected ROIs get 1, background 0.
        With template: float64 ``(height, width, 3)`` in ``[0, 1]``: the template scaled
        between its 1st and 99th percentiles in all three channels, selected ROIs in their
        group colour (random RGB, MATLAB ``jet(G)``, or ``singlecolor``), unselected ROIs
        grey 0.5.

    Notes
    -----
    Suspected MATLAB bug that is kept: without a template the loop runs over
    ``length(sel)`` (the number of selection rows, or 2 for a single row), not over the
    ROIs, so ROIs after that position are not drawn at all (instead of in grey 1). With a
    template all ROIs are drawn.

    Differences from MATLAB
    -----------------------
    - ``sel`` ROI numbers are 0-based.
    - ROIs that have only ``PixelIdxList`` are accepted (MATLAB's renaming line assigns in
      the wrong direction and fails for them).
    - A single-row ``sel`` (or a single ROI by default) no longer reads past the last ROI
      (MATLAB's loop over ``length(sel) == 2`` fails with one ROI); the loop is capped at
      the number of ROIs.
    - A column ``sel`` gets colour group 1 (MATLAB fails on ``sel(:,2)``); an empty
      ``sel`` and an ROI listed twice in ``sel`` raise ``ValueError`` (MATLAB fails).
    - Random colours come from ``rng`` and differ from MATLAB's.
    - Integer templates are scaled in float64 (MATLAB keeps the integer class, which
      rounds the overlay to 0/1).
    - ``'display'`` is not provided: show the result with ``ax.imshow``. MATLAB uses the
      ``colorcube`` colormap, which matplotlib does not have.
    """
    rois = list(rois)
    n_rois = len(rois)
    dim = tuple(int(d) for d in dim)
    if len(dim) != 2:
        raise ValueError("dim must be (height, width)")
    if sel is None:
        s = np.repeat(np.arange(1, n_rois + 1)[:, None], 2, axis=1)
    else:
        s = np.asarray(sel)
        if s.size == 0:
            raise ValueError("sel must not be empty")
        if s.ndim == 1:
            s = np.stack([s + 1, np.ones(s.size, dtype=s.dtype)], axis=1)
        elif s.ndim == 2:
            if s.shape[0] <= s.shape[1]:
                s = s.T
            if s.shape[1] == 1:
                s = np.concatenate([s + 1, np.ones_like(s)], axis=1)
            else:
                s = s.copy()
                s[:, 0] = s[:, 0] + 1
        else:
            raise ValueError("sel must be 1-D or 2-D")
    s = s.astype(np.int64)
    if np.unique(s[:, 0]).size != s.shape[0]:
        raise ValueError("an ROI is listed twice in sel")
    n_groups = int(s[:, 1].max())
    if s[:, 1].min() < 1:
        raise ValueError("colour groups must be numbered from 1")

    if randcolor and singlecolor is None:
        if rng is None:
            raise ValueError("random colours need rng (or pass randcolor=False)")
        colors = rng.permutation(n_groups) + 2.0
        colors2 = rng.random((3, n_groups))
    elif singlecolor is None:
        colors = np.arange(2, n_groups + 2, dtype=np.float64)
        colors2 = _jet(n_groups).T
    else:
        sc = np.asarray(singlecolor, dtype=np.float64).ravel()
        val = 10.0 * float(mround(np.sum(sc)))
        colors = np.full(n_groups, val)
        colors2 = np.repeat(sc[:, None], n_groups, axis=1)

    group_of = {int(r): int(g) for r, g in s}
    npix = dim[0] * dim[1]

    def mask_of(i: int) -> np.ndarray:
        idx = _roi_indices(rois[i])
        if idx.size and (idx.min() < 1 or idx.max() > npix):
            raise IndexError(f"ROI {i} has indices outside an image of size {dim}")
        m = np.zeros(npix, dtype=bool)
        m[idx - 1] = True
        return m.reshape(dim, order="F")

    if template is None:
        im = np.zeros(dim)
        n_loop = min(max(s.shape) if s.ndim == 2 else s.size, n_rois)
        for i in range(n_loop):
            m = mask_of(i)
            if type != "full":
                m = _bwperim(m)
            g = group_of.get(i + 1)
            im[m] = colors[g - 1] if g is not None else 1.0
        return im

    t = np.asarray(template, dtype=np.float64)
    if t.shape != dim:
        raise ValueError("template must have the shape dim")
    hi = float(prctile(t, 99))
    lo = float(prctile(t, 1))
    im = np.repeat(((t - lo) / (hi - lo))[:, :, None], 3, axis=2)
    for i in range(n_rois):
        m = mask_of(i)
        if type == "perim":
            m = _bwperim(m)
        g = group_of.get(i + 1)
        for ch in range(3):
            im[:, :, ch][m] = colors2[ch, g - 1] if g is not None else 0.5
    return np.clip(im, 0.0, 1.0)
