"""Private helpers for MATLAB struct arrays as calliopy represents them (object arrays of
:class:`calliopy.io.MatStruct`), and a few MATLAB built-ins used by the projmeta ports."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from .._matlab import mround
from ..io import MatStruct

__all__ = ["mround"]  # re-exported for the projmeta modules


def empty() -> np.ndarray:
    """MATLAB ``[]``."""
    return np.zeros((0, 0))


def is_empty(v: Any) -> bool:
    """MATLAB ``isempty``."""
    if v is None:
        return True
    if isinstance(v, (str, list, tuple, dict)):
        return len(v) == 0
    if isinstance(v, np.ndarray):
        return v.size == 0
    return False


def elements(sa) -> list:
    """Elements of a struct array (object array, list or a single struct), column-major."""
    if sa is None:
        return []
    if isinstance(sa, dict):
        return [sa]
    if isinstance(sa, np.ndarray):
        return list(sa.ravel(order="F"))
    return list(sa)


def struct_array(items, shape=None) -> np.ndarray:
    """Object array of structs with the same fields (MATLAB fills new fields with ``[]``).

    Field order is the order in which fields first appear, element by element.
    """
    items = list(items)
    order: list = []
    seen = set()
    for it in items:
        for k in it:
            if k not in seen:
                seen.add(k)
                order.append(k)
    out = np.empty(len(items), dtype=object)
    for i, it in enumerate(items):
        s = MatStruct()
        for k in order:
            s[k] = it[k] if k in it else empty()
        out[i] = s
    if shape is None:
        shape = (1, len(items))
    return out.reshape(shape, order="F")


def grid_struct_array(cells: dict, n_rows: int, n_cols: int) -> np.ndarray:
    """A 2-D struct array from ``{(row, col): struct}`` (0-based), unset elements empty."""
    items = []
    for c in range(n_cols):
        for r in range(n_rows):
            items.append(cells.get((r, c), MatStruct()))
    return struct_array(items, (n_rows, n_cols))


def as_cell_list(v) -> list:
    """A MATLAB cell array (object array or list) as a list, column-major."""
    if isinstance(v, np.ndarray) and v.dtype == object:
        return list(v.ravel(order="F"))
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def ntzo(x) -> np.ndarray:
    """MATLAB ``06_Library/ntzo.m``: shift to minimum 0 and scale to maximum 1."""
    a = np.asarray(x)
    a = a.astype(np.float32) if a.dtype.kind in "iu" else a.astype(np.float64, copy=True)
    a = a - np.min(a)
    if np.max(a) > np.min(a):
        a = a / np.max(a)
    return a


def analysis(name: str):
    """A function of :mod:`calliopy.analysis`, imported when first needed.

    psmooth, ftfil, smooth2, act2spks and extract_pulse_envelope are ported by the
    analysis track; this keeps calliopy.projmeta importable while they are missing.
    """
    mod = importlib.import_module("calliopy.analysis")
    try:
        return getattr(mod, name)
    except AttributeError:
        raise ImportError(f"calliopy.analysis.{name} is not available (needed here; port of 21_Analysis/"
                          f"{name}.m)") from None
