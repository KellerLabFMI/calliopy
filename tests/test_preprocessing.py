"""Tests for calliopy.preprocessing.get_frame_times (calliope 04_Calliope/get_frame_times.m)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from calliopy.preprocessing import get_frame_times


def _galvo(n_frames=8, period=150, flyback=40, lead=60, rest=0.0):
    """Galvo-like trace: ``rest`` before the first frame, then a ramp from -1 to 1 per frame
    whose top (flyback) part exceeds 75 % of the maximum."""
    sig = np.full(lead, rest)
    for _ in range(n_frames):
        sig = np.r_[sig, np.linspace(-1, 1, period - flyback), np.ones(flyback)]
    return np.r_[sig, np.full(50, -1.0)]


def test_frame_times_1_based_and_regular():
    sig = _galvo()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ft = get_frame_times(sig)
    assert ft.dtype == np.float64
    assert ft[0] == 61.0  # start of the first ramp: first sample below half the minimum, 1-based
    np.testing.assert_allclose(np.diff(ft), 150)
    assert ft.size == 8  # 7 run ends with a following gap + the appended last frame


def test_no_frames_returns_one_with_warning():
    with pytest.warns(UserWarning, match="no frames"):
        ft = get_frame_times(np.ones(100))
    np.testing.assert_array_equal(ft, [1.0])


def test_signal_not_zero_before_first_frame():
    sig = _galvo(rest=-1.0)  # already at the minimum long before the first frame
    with pytest.warns(UserWarning, match="not zero before 1st frame"):
        ft = get_frame_times(sig)
    np.testing.assert_allclose(np.diff(ft), 150)


def test_matches_hand_computed_matlab_steps():
    sig = _galvo(n_frames=3, period=100, flyback=20, lead=10)
    pos = np.flatnonzero(sig > 0.75 * sig.max()) + 1
    ends = pos[np.flatnonzero(np.diff(pos) > 10)]
    first = np.flatnonzero(sig < 0.5 * sig.min())[0] + 1
    expected = ends + first - ends[0]
    expected = np.r_[expected, expected[-1] + np.median(np.diff(ends))]
    np.testing.assert_allclose(get_frame_times(sig), expected)
