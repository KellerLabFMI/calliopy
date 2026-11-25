"""Tests for calliopy.stats (hierarchical bootstrap, calliope 21_Analysis/nbstrp.m)."""

from __future__ import annotations

import numpy as np
import pytest
from calliopy.stats import find_groups, nbstrp, nbstrp_get_indices


def test_find_groups_numbers_sorted_labels():
    assert find_groups(["b", "a", "b", "c"]).tolist() == [2, 1, 2, 3]
    assert find_groups([10, 3, 3]).tolist() == [2, 1, 1]


def test_indices_resample_groups_then_members():
    groups = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3])
    rng = np.random.default_rng(0)
    idx = nbstrp_get_indices(groups, 500, rng=rng)
    assert len(idx) == 500
    sizes = np.bincount(groups)[1:]
    members = {g: set(np.flatnonzero(groups == g)) for g in (1, 2, 3)}
    for ii in idx:
        # every draw is made of whole groups: its length is a sum of 3 group sizes
        assert ii.size in {a + b + c for a in sizes for b in sizes for c in sizes}
        assert ii.min() >= 0 and ii.max() < groups.size
    # all members of every group are reachable
    seen = set(np.concatenate(idx).tolist())
    assert seen == set().union(*members.values())


def test_indices_within_group_blocks():
    groups = np.array([1, 1, 2, 2, 2])
    idx = nbstrp_get_indices(groups, 200, rng=np.random.default_rng(3))
    for ii in idx:
        g = groups[ii]
        # blocks of one group appear contiguously with that group's size
        pos = 0
        while pos < ii.size:
            size = int(np.sum(groups == g[pos]))
            assert np.all(g[pos : pos + size] == g[pos])
            pos += size


def test_nbstrp_centres_on_group_structure():
    rng = np.random.default_rng(1)
    s = np.r_[rng.normal(0, 0.1, 50), rng.normal(10, 0.1, 50)]
    g = np.r_[np.zeros(50), np.ones(50)]
    res = nbstrp(s, g, 4000, rng=rng)
    assert res.mean_est.shape == (4000,) and res.median_est.shape == (4000,)
    # only 3 outcomes for the group draw: both group 0, mixed, both group 1
    assert abs(res.mean_est.mean() - 5) < 0.5
    assert np.mean(res.mean_est < 1) == pytest.approx(0.25, abs=0.03)


def test_nbstrp_single_group_is_ordinary_bootstrap():
    rng = np.random.default_rng(2)
    s = rng.normal(3, 1, 400)
    res = nbstrp(s, np.ones(400), 3000, rng=rng)
    assert res.mean_est.std() == pytest.approx(s.std(ddof=1) / 20, rel=0.1)


def test_nbstrp_nan_inf_handling():
    s = np.array([1.0, np.nan, 2.0, np.inf, 3.0])
    g = np.array([1, 1, 2, 2, 2])
    res = nbstrp(s, g, 50, rng=np.random.default_rng(0))
    assert np.all(np.isfinite(res.mean_est))
    res = nbstrp(s, g, 50, omit_nans=False, rng=np.random.default_rng(0))
    assert np.isnan(res.mean_est).any() or np.isinf(res.mean_est).any()
    with pytest.raises(ValueError, match="no valid data"):
        nbstrp([np.nan], [1], 5, rng=np.random.default_rng(0))


def test_nbstrp_input_errors():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="same length"):
        nbstrp([1, 2], [1], rng=rng)
    with pytest.raises(ValueError, match="empty"):
        nbstrp([], [], rng=rng)


def test_nbstrp_reproducible_with_seed():
    s, g = np.arange(20.0), np.repeat([1, 2, 3, 4], 5)
    a = nbstrp(s, g, 100, rng=np.random.default_rng(7)).mean_est
    b = nbstrp(s, g, 100, rng=np.random.default_rng(7)).mean_est
    np.testing.assert_array_equal(a, b)
