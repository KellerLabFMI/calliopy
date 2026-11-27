"""Small array and container utilities of calliope ``06_Library``."""

from __future__ import annotations

import numbers

import numpy as np

__all__ = ["expmat", "flatten_cell", "is_number"]


def expmat(b, m: int) -> np.ndarray:
    """Repeat every element of a matrix ``m`` times along its rows.

    Port of calliope ``06_Library/expmat.m``:
    ``[1 2; 3 4]`` with ``m = 3`` gives ``[1 1 1 2 2 2; 3 3 3 4 4 4]``.

    Parameters
    ----------
    b : array_like
        Matrix ``(r, c)``; a 1-D array is treated as a row ``(1, c)`` and a scalar as
        ``(1, 1)``.
    m : int
        Number of repetitions per element (non-negative).

    Returns
    -------
    numpy.ndarray
        ``(r, m * c)``, dtype of ``b``. Equal to ``numpy.repeat(b, m, axis=1)``, which is
        what MATLAB's ``reshape(repmat(b',1,m)', r, m*c)`` computes.
    """
    a = np.asarray(b)
    if a.ndim == 0:
        a = a.reshape(1, 1)
    elif a.ndim == 1:
        a = a[np.newaxis, :]
    elif a.ndim != 2:
        raise ValueError(f"b must be 2-D, got shape {a.shape}")
    m = int(m)
    if m < 0:
        raise ValueError("m must be non-negative")
    return np.repeat(a, m, axis=1)


def flatten_cell(cell) -> list:
    """Flatten nested cell arrays into one flat list.

    Port of calliope ``06_Library/flattenCell.m``. Elements that are themselves cells are
    flattened recursively; every other element is kept as it is.

    Parameters
    ----------
    cell : list, tuple or object ndarray
        A MATLAB cell array as calliopy represents it: a list/tuple, or a numpy array of
        dtype object (as :func:`calliopy.io.load_mat` returns cells). Object arrays are
        traversed in column-major order, like MATLAB's linear indexing ``in{i}``.

    Returns
    -------
    list
        The leaves in order (MATLAB returns a 1 x n cell row).
    """
    out: list = []
    items = cell.ravel(order="F") if isinstance(cell, np.ndarray) else cell
    for el in items:
        if isinstance(el, (list, tuple)) or (
            isinstance(el, np.ndarray) and el.dtype == object
        ):
            out.extend(flatten_cell(el))
        else:
            out.append(el)
    return out


def _str2double(s: str) -> float | complex:
    t = s.strip().replace(",", "")
    if not t:
        return float("nan")
    try:
        return float(t)
    except ValueError:
        pass
    try:
        c = complex(t.replace("i", "j").replace("I", "j"))
    except ValueError:
        return float("nan")
    return c


def is_number(x) -> bool:
    """True for numeric values and for strings that read as a number.

    Port of calliope ``06_Library/isnumber.m``:
    ``isnumeric(x) || ischar(x) && ~isnan(str2double(x))``.

    Parameters
    ----------
    x
        Any value.

    Returns
    -------
    bool
        ``True`` for Python/numpy numbers (int, float, complex) and numeric ndarrays (also
        empty ones, as MATLAB ``isnumeric([])``); ``True`` for a string that parses as a
        number (``'3'``, ``' -1e3 '``, ``'Inf'``, ``'1,000'``, ``'1+2i'``); ``False`` for
        booleans (MATLAB ``logical`` is not numeric), ``'NaN'``, other strings and everything
        else.

    Differences from MATLAB
    -----------------------
    - ``str2double`` is approximated with ``float``/``complex`` after removing commas
      (MATLAB's thousands separators): exotic forms that MATLAB accepts (``'1e3i'`` written
      with ``*``, ``'+i'``) or rejects may differ.
    """
    if isinstance(x, (bool, np.bool_)):
        return False
    if isinstance(x, numbers.Number):
        return True
    if isinstance(x, np.ndarray):
        return x.dtype.kind in "iufc"
    if isinstance(x, str):
        v = _str2double(x)
        return not (isinstance(v, float) and np.isnan(v)) and not (
            isinstance(v, complex) and (np.isnan(v.real) or np.isnan(v.imag))
        )
    return False
