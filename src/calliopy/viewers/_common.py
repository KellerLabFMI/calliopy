"""Private helpers shared by the viewer sessions (no tkinter here).

- reading a MAT-file of either format (v7.3 through :mod:`calliopy.io`, v5/v7 through
  ``scipy.io.loadmat``; MATLAB's plain ``save`` writes v7 by default);
- MATLAB ``num2str`` for figure titles and labels, MATLAB ``hsv``;
- 3x3 transform matrices from adata variables.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import h5py
import matplotlib.colors
import numpy as np

from calliopy.io import MatlabObject, MatStruct, load_mat

__all__: list[str] = []

log = logging.getLogger("calliopy.viewers")

_REFS = "#refs#"


# ---------------------------------------------------------------------------------------
# MAT-files
# ---------------------------------------------------------------------------------------
def _from_v5(value: Any) -> Any:
    """Values of ``scipy.io.loadmat(..., struct_as_record=False)`` as :func:`load_mat` gives them."""
    from scipy.io.matlab import mat_struct

    if isinstance(value, mat_struct):
        return MatStruct((k, _from_v5(getattr(value, k))) for k in value._fieldnames)
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            if value.size == 1 and isinstance(value.flat[0], mat_struct):
                return _from_v5(value.flat[0])
            out = np.empty(value.shape, dtype=object)
            for idx in np.ndindex(value.shape):
                out[idx] = _from_v5(value[idx])
            return out
        if value.dtype.kind == "U":
            return "".join(str(s) for s in np.ravel(value))
    return value


def read_mat_variable(path: str | PathLike, name: str) -> Any:
    """One variable of a MAT-file written by MATLAB (v5, v7 or v7.3) or calliopy."""
    path = Path(path)
    if h5py.is_hdf5(path):
        return load_mat(path, name)[name]
    import scipy.io

    data = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=False, chars_as_strings=True,
                            variable_names=[name])
    if name not in data:
        raise KeyError(f"{name!r} is not a variable in {path.name}")
    return _from_v5(data[name])


def mat_variable_names(path: str | PathLike) -> tuple[str, ...]:
    """Names of the variables of a v7.3 MAT-file (HDF5 objects not starting with ``#``)."""
    with h5py.File(Path(path), "r") as f:
        return tuple(k for k in f if not k.startswith("#"))


def tform_from(value: Any, name: str, given: Mapping[str, Any] | None) -> np.ndarray:
    """3x3 ``tform.A`` of adata variable ``name``: from ``given`` first, else the stored value.

    Raises ``ValueError`` when neither is a numeric 3x3 matrix (MATLAB transform objects
    are read as :class:`calliopy.io.MatlabObject` placeholders and cannot be used).
    """
    if given is not None and name in given and given[name] is not None:
        a = np.asarray(given[name], dtype=np.float64)
        if a.shape != (3, 3):
            raise ValueError(f"tforms[{name!r}] must be a 3x3 matrix (tform.A)")
        return a
    if value is not None and not isinstance(value, MatlabObject):
        a = np.asarray(value)
        if a.dtype != object and a.shape == (3, 3):
            return a.astype(np.float64)
    raise ValueError(
        f"{name} is not a numeric 3x3 matrix in the adata file (MATLAB transform objects cannot "
        f"be decoded); pass tforms={{{name!r}: A}} with the 3x3 matrix tform.A")


# ---------------------------------------------------------------------------------------
# MATLAB built-ins for display
# ---------------------------------------------------------------------------------------
def num2str(v: Any) -> str:
    """MATLAB ``num2str`` of a number, a numeric vector or a string (titles and labels)."""
    if isinstance(v, str):
        return v
    a = np.asarray(v)
    if a.dtype.kind in "US":
        return "".join(str(s) for s in np.ravel(a))
    if a.size == 0:
        return ""
    flat = a.ravel(order="F").astype(np.float64)
    if a.size > 1:
        if np.all(np.isfinite(flat)) and np.all(flat == np.round(flat)):
            ints = [str(int(x)) for x in flat]
            width = max(len(s) for s in ints) + 2
            return "".join(s.rjust(width) for s in ints).lstrip()
        return "  ".join(num2str(float(x)) for x in flat)
    x = float(flat[0])
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if x == round(x) and abs(x) < 1e15:
        return str(int(x))
    d = max(math.floor(math.log10(abs(x))) + 5, 5)
    return f"{x:.{min(d, 16)}g}"


def hsv(n: int) -> np.ndarray:
    """MATLAB ``hsv(n)``: ``hsv2rgb([(0:n-1)'/n, 1, 1])``."""
    h = np.arange(n) / max(n, 1)
    return matplotlib.colors.hsv_to_rgb(np.column_stack([h, np.ones(n), np.ones(n)]))
