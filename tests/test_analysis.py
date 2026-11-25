"""Tests for calliopy.analysis.ampd (calliope 21_Analysis/ampd.m)."""

from __future__ import annotations

import numpy as np
import pytest
from calliopy.analysis import AmpdAdaptiveDebug, AmpdExtendedDebug, AmpdOriginalDebug, ampd


def _sine(n=1000, periods=10, phase=0.0):
    return np.sin(np.linspace(0, 2 * np.pi * periods, n) + phase)


def _maxima(n=1000, periods=10):
    """Samples nearest to the sine maxima (0-based)."""
    return np.round((0.25 + np.arange(periods)) * (n - 1) / periods).astype(int)


def test_extended_finds_sine_peaks_0_based():
    x = _sine()
    pks = ampd(x)
    expected = _maxima()
    assert pks.dtype == np.int64
    # interior peaks at the sine maxima (0-based samples; a maximum half-way between two
    # samples may land on either after detrending)
    assert pks.size >= expected.size
    assert np.max(np.abs(pks[: expected.size] - expected)) <= 1


def test_original_excludes_edges():
    x = _sine()
    pks = ampd(x, mode="original")
    assert pks.min() > 0 and pks.max() < x.size - 1
    # interior maxima only: the first lies closer to the edge than the selected scale
    assert all(np.min(np.abs(_maxima() - p)) <= 1 for p in pks)
    assert pks.size >= 8


def test_adaptive_and_noise_robustness():
    rng = np.random.default_rng(0)
    x = _sine(2000, 20) + 0.2 * rng.normal(size=2000) + np.linspace(0, 5, 2000)  # trend removed
    for mode, kw in (("extended", {}), ("original", {}), ("adaptive", {"window": 400})):
        pks = ampd(x, mode=mode, **kw)
        if mode == "adaptive":  # ASS-AMPD gives spurious detections within window/2 of the ends
            pks = pks[(pks >= 200) & (pks < 1800)]
        true = _maxima(2000, 20)
        if mode == "adaptive":
            true = true[(true >= 200) & (true < 1800)]
        # every true peak has a detection within 10 samples, and few extras
        near = [np.min(np.abs(pks - t)) for t in true[1:-1]]
        assert max(near) <= 10, mode
        assert pks.size <= 1.3 * true.size, mode


def test_debug_outputs():
    x = _sine(200, 4)
    d = ampd(x, mode="original", debug=True)
    assert isinstance(d, AmpdOriginalDebug) and d.lsm.shape == (100, 200) and d.l_scale >= 1
    d = ampd(x, mode="extended", debug=True, scale=30)
    assert isinstance(d, AmpdExtendedDebug) and d.lsm.shape == (30, 200) and d.g.shape == (30,)
    d = ampd(x, mode="adaptive", debug=True, window=50)
    assert isinstance(d, AmpdAdaptiveDebug) and d.ass_lsm.shape == (25, 200)
    assert d.adaptive_scale.shape == (200,) and d.adaptive_scale.min() >= 0


def test_short_input_and_bad_mode():
    assert ampd([1.0, 2.0, 1.0], mode="original").size == 0
    assert ampd([1.0], mode="extended").size == 0
    with pytest.raises(ValueError, match="Unknown mode"):
        ampd([1.0, 2.0], mode="ampd")
