"""White-centred diverging colormaps: calliope ``22_Plotting/cmap_centered.m``."""

from __future__ import annotations

import numpy as np

from .._matlab import mround

__all__ = ["cmap_centered"]

# 15-row ColorBrewer tables (0-255) as listed in cmap_centered.m; row 8 is the centre.
_TABLES = {
    "Spectral": [158,1,66,213,62,79,215,25,28,244,109,67,252,141,89,253,174,97,254,224,139,255,255,191,230,245,152,171,221,164,153,213,148,102,194,165,43,131,186,50,136,189,94,79,162],
    "RdYlGn": [165,0,38,215,48,39,215,25,28,244,109,67,252,141,89,253,174,97,254,224,139,255,255,191,217,239,139,166,217,106,145,207,96,102,189,99,26,150,65,26,152,80,0,104,55],
    "RdYlBu": [165,0,38,215,48,39,215,25,28,244,109,67,252,141,89,253,174,97,254,224,144,255,255,191,224,243,248,171,217,233,145,191,219,116,173,209,44,123,182,69,117,180,49,54,149],
    "RdGy": [103,0,31,178,24,43,202,0,32,214,96,77,239,138,98,244,165,130,253,219,199,255,255,255,224,224,224,186,186,186,153,153,153,135,135,135,64,64,64,77,77,77,26,26,26],
    "RdBu": [103,0,31,178,24,43,202,0,32,214,96,77,239,138,98,244,165,130,253,219,199,255,255,255,209,229,240,146,197,222,103,169,207,67,147,195,5,113,176,33,102,172,5,48,97],
    "PuOr": [127,59,8,179,88,6,230,97,1,224,130,20,241,163,64,253,184,99,254,224,182,255,255,255,216,218,235,178,171,210,153,142,195,128,115,172,94,60,153,84,39,136,45,0,75],
    "PrGn": [64,0,75,118,42,131,123,50,148,153,112,171,175,141,195,194,165,207,231,212,232,255,255,255,217,240,211,166,219,160,127,191,123,90,174,97,0,136,55,27,120,55,0,68,27],
    "PiYG": [142,1,82,197,27,125,208,28,139,222,119,174,233,163,201,241,182,218,253,224,239,255,255,255,230,245,208,184,225,134,161,215,106,127,188,65,77,172,38,77,146,33,39,100,25],
    "BrBg": [84,48,5,140,81,10,166,97,26,191,129,45,216,179,101,223,194,125,246,232,195,255,255,255,199,234,229,128,205,193,90,180,172,53,151,143,1,133,113,1,102,94,0,60,48],
}


def _half(m: int, name: str, option: str) -> np.ndarray:
    """``get_map`` of cmap_centered.m: one half of the table resampled to ``m`` rows."""
    table = np.asarray(_TABLES[name], dtype=np.float64).reshape(15, 3)
    half = table[7::-1] if option == "positive" else table[14:6:-1]
    if m <= 0:
        return np.empty((0, 3))
    xq = np.linspace(0.0, 1.0, m) if m > 1 else np.array([1.0])
    xp = np.linspace(0.0, 1.0, half.shape[0])
    return np.column_stack([np.interp(xq, xp, half[:, j] / 256.0) for j in range(3)])


def _current_clim():
    """CLim of the current axes as MATLAB ``get(gca, 'CLim')`` would give it, or (0, 1)."""
    import matplotlib.pyplot as plt

    if not plt.get_fignums():
        return (0.0, 1.0)  # MATLAB's gca would create new axes with CLim [0 1]
    ax = plt.gcf().gca()
    for art in reversed(ax.images + ax.collections):
        if hasattr(art, "get_clim"):
            lo, hi = art.get_clim()
            if lo is not None and hi is not None:
                return (float(lo), float(hi))
    return (0.0, 1.0)


def cmap_centered(name: str = "RdBu", m: int | None = None, center: float | None = None, *, clim=None) -> np.ndarray:
    """Diverging colormap whose white entry sits at ``center`` of the colour limits.

    Re-implementation of calliope ``22_Plotting/cmap_centered.m`` (third-party code by
    M. Courthoud). The colour limits ``clim`` minus ``center`` are split into a negative
    and a positive part; ``round(m * neg / (neg + pos))`` rows go to the negative half of
    the table (dark end to white) and the rest to the positive half (white to dark end).
    Limits that are all >= center give the positive half only, all < center the negative
    half only.

    Parameters
    ----------
    name : str, default 'RdBu'
        One of 'Spectral', 'RdYlGn', 'RdYlBu', 'RdGy', 'RdBu', 'PuOr', 'PrGn', 'PiYG',
        'BrBg' (MATLAB's names and tables).
    m : int, optional
        Number of rows (default 256, the length of MATLAB's default figure colormap).
    center : float, optional
        Value that gets white (default 0).
    clim : (float, float), optional
        Colour limits the map will be used with. MATLAB always reads them from the current
        axes (``get(gca, 'CLim')``); when ``clim`` is not given this function does the same:
        the colour limits of the last image or collection of the current matplotlib axes,
        else (0, 1), MATLAB's default CLim.

    Returns
    -------
    numpy.ndarray, shape (m, 3)
        RGB rows in [0, 1); wrap with ``matplotlib.colors.ListedColormap``.

    Differences from MATLAB
    -----------------------
    - ``clim`` can be given explicitly; without it, the current matplotlib axes stand in
      for MATLAB's ``gca`` and no figure is created.
    - Default ``m`` is 256 rather than the row count of the current figure's colormap.
    - An unknown ``name`` raises ``ValueError`` (MATLAB fails on an undefined variable).

    Kept from MATLAB (suspected quirks): the table is divided by 256, so white is
    (255/256, 255/256, 255/256); the centre row appears in both halves.
    """
    if name not in _TABLES:
        raise ValueError(f"unknown colormap {name!r}; one of {sorted(_TABLES)}")
    m = 256 if m is None else int(m)
    center = 0.0 if center is None else float(center)
    lims = _current_clim() if clim is None else tuple(float(v) for v in np.ravel(clim)[:2])
    lo, hi = lims[0] - center, lims[1] - center
    if lo < 0 and hi > 0:
        ratio = abs(lo) / (abs(lo) + hi)
        neglen = int(mround(m * ratio))
        return np.vstack([_half(neglen, name, "negative"), _half(m - neglen, name, "positive")])
    if lo >= 0:
        return _half(m, name, "positive")
    return _half(m, name, "negative")
