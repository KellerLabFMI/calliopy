"""Hierarchical (nested) bootstrap: calliope ``21_Analysis/nbstrp.m`` and ``nbstrp_get_indices.m``.

Groups (for example animals) are resampled with replacement, then the observations of each
selected group (for example neurons) are resampled with replacement within that group.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = ["NbstrpResult", "find_groups", "nbstrp", "nbstrp_get_indices"]


class NbstrpResult(NamedTuple):
    """Outputs of :func:`nbstrp` (MATLAB ``[mean_est, median_est]``)."""

    mean_est: np.ndarray
    median_est: np.ndarray


def find_groups(groups) -> np.ndarray:
    """MATLAB ``findgroups`` for a vector: group numbers 1..N in sorted order of the labels.

    Returns 1-based group numbers as int64 (stored-value convention of MATLAB). NaN labels
    get group NaN in MATLAB; here they raise ``ValueError`` (nbstrp never passes them on
    meaningfully).
    """
    g = np.asarray(groups).ravel()
    if g.dtype.kind == "f" and np.isnan(g).any():
        raise ValueError("group labels must not be NaN")
    _, inv = np.unique(g, return_inverse=True)
    return inv.astype(np.int64).ravel() + 1


def nbstrp_get_indices(groups, nbss: int, *, rng: np.random.Generator) -> list[np.ndarray]:
    """Hierarchical bootstrap resampling indices (calliope ``21_Analysis/nbstrp_get_indices.m``).

    Parameters
    ----------
    groups : array_like of int
        Group number of every observation, 1..N as returned by :func:`find_groups`
        (MATLAB convention: the function needs normalised groups).
    nbss : int
        Number of bootstrap iterations.
    rng : numpy.random.Generator
        Source of the random numbers.

    Returns
    -------
    list of numpy.ndarray
        ``nbss`` arrays of **0-based** indices into the observations. Each array has as many
        entries as the observations of the selected groups together; group ``g`` of the draw
        contributes ``n_g`` indices drawn with replacement from its own members.

    Differences from MATLAB
    -----------------------
    - Indices are 0-based (MATLAB: 1-based).
    - ``randi`` cannot be reproduced; the draws come from ``rng`` and differ from MATLAB's
      for the same seed. The order of draws is groups first (all iterations), then members.
    """
    g = np.asarray(groups).ravel().astype(np.int64)
    nbss = int(nbss)
    if g.size == 0:
        raise ValueError("groups must not be empty")
    num_groups = int(g.max())
    if g.min() < 1:
        raise ValueError("groups must be numbered from 1 (use find_groups)")
    order = np.argsort(g, kind="stable")  # members of each group, in original order
    sizes = np.bincount(g, minlength=num_groups + 1)[1:]
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    # [num_groups x nbss] selected groups, as MATLAB's randi(num_groups, num_groups, nbss)
    group_bss = rng.integers(0, num_groups, size=(num_groups, nbss))
    out = []
    for ib in range(nbss):
        sel = group_bss[:, ib]
        sel = sel[sizes[sel] > 0]  # a group number without members contributes nothing
        n_sel = sizes[sel]
        rep_sel = np.repeat(sel, n_sel)
        within = rng.integers(0, np.repeat(n_sel, n_sel))
        out.append(order[starts[rep_sel] + within])
    return out


def nbstrp(
    samples,
    groups,
    nbss: int = 1000,
    *,
    omit_nans: bool = True,
    omit_infs: bool = True,
    rng: np.random.Generator,
) -> NbstrpResult:
    """Hierarchical bootstrap of the mean and median (calliope ``21_Analysis/nbstrp.m``).

    Parameters
    ----------
    samples : array_like
        All observations (any shape; flattened in column-major order as MATLAB ``(:)``).
    groups : array_like
        Group label of every observation (any numeric or string labels), same length.
    nbss : int, default 1000
        Number of bootstrap samples.
    omit_nans, omit_infs : bool, default True
        Remove NaN / Inf observations (and their labels) before resampling.
    rng : numpy.random.Generator
        Source of the random numbers.

    Returns
    -------
    NbstrpResult
        ``mean_est`` and ``median_est``, float64 arrays of length ``nbss``.

    Notes
    -----
    calliope's ``nbstrp`` is the function earlier called ``nbstrp_clean`` (its error
    identifiers still carry that name).

    Differences from MATLAB
    -----------------------
    - Random draws differ from MATLAB's (see :func:`nbstrp_get_indices`).
    - ``median_est`` is always computed (MATLAB computes it only when requested).
    """
    s = np.asarray(samples, dtype=np.float64).ravel(order="F")
    gr = np.asarray(groups).ravel(order="F")
    if s.size != gr.size:
        raise ValueError("samples and groups must have the same length")
    if s.size == 0:
        raise ValueError("samples and groups cannot be empty")
    keep = np.ones(s.shape, dtype=bool)
    if omit_nans:
        keep &= ~np.isnan(s)
    if omit_infs:
        keep &= ~np.isinf(s)
    s, gr = s[keep], gr[keep]
    if s.size == 0:
        raise ValueError("no valid data remaining after cleaning NaNs and Infs")
    idx = nbstrp_get_indices(find_groups(gr), nbss, rng=rng)
    mean_est = np.array([s[i].mean() for i in idx], dtype=np.float64)
    median_est = np.array([np.median(s[i]) for i in idx], dtype=np.float64)
    return NbstrpResult(mean_est, median_est)
