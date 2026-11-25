"""Tests for calliopy.plotting (plotBSTRP, cmap_centered, Violin/violinplot)."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from calliopy.plotting import cmap_centered, linespec_kwargs, plot_bstrp, violin, violinplot


@pytest.fixture(autouse=True)
def _close():
    plt.switch_backend("Agg")
    yield
    plt.close("all")


def test_linespec():
    assert linespec_kwargs("-k") == {"linestyle": "-", "color": "k"}
    assert linespec_kwargs("--r") == {"linestyle": "--", "color": "r"}
    assert linespec_kwargs("o") == {"marker": "o", "linestyle": "none"}
    assert linespec_kwargs("-.bs") == {"linestyle": "-.", "color": "b", "marker": "s"}
    assert linespec_kwargs(None) == {}
    with pytest.raises(ValueError):
        linespec_kwargs("-q")


def test_plot_bstrp_mean_and_sd_band():
    rng = np.random.default_rng(0)
    data = np.vstack([np.full(12, 1.0), np.linspace(0, 1, 12)])  # 2 time points, 12 traces
    _, ax = plt.subplots()
    res = plot_bstrp(data, np.repeat([1, 2, 3], 4), 300, "-k", rng=rng, ax=ax, linewidth=3)
    assert res.bss_all.shape == (2, 300)
    np.testing.assert_allclose(res.bss_all[0], 1.0)
    np.testing.assert_allclose(res.h.get_ydata(), res.bss_all.mean(axis=1))
    np.testing.assert_allclose(res.h.get_xdata(), [1, 2])
    assert res.h.get_linewidth() == 3 and res.h.get_color() == "k"
    band = ax.patches[0]
    xy = band.get_xy()
    y = xy[xy[:, 0] == 2, 1]
    sd = res.bss_all[1].std(ddof=1)
    assert y.max() == pytest.approx(res.bss_all[1].mean() + sd)
    assert y.min() == pytest.approx(res.bss_all[1].mean() - sd)
    assert band.get_label() == "_nolegend_"


def test_plot_bstrp_percentiles_baseline_and_kwargs_colour():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(120, 20)) + 5
    _, ax = plt.subplots()
    res = plot_bstrp(data, np.arange(20) // 4, 200, "-b", ptil_range=[2.5, 97.5], ms=True,
                     baseline=np.arange(10), rng=rng, ax=ax, color=(1, 0, 0))
    assert np.nanmean(res.bss_all[:10]) == pytest.approx(0, abs=1e-12)
    assert matplotlib.colors.to_rgb(res.h.get_color()) == (1, 0, 0)
    # default baseline is MATLAB's 93:100 (0-based 92..99)
    res2 = plot_bstrp(data, np.arange(20) // 4, 50, ms=True, rng=rng, ax=ax)
    assert np.nanmean(res2.bss_all[92:100]) == pytest.approx(0, abs=1e-12)


def test_plot_bstrp_omit_nan_and_errors():
    rng = np.random.default_rng(2)
    data = np.ones((5, 6))
    data[2, 1] = np.nan
    _, ax = plt.subplots()
    res = plot_bstrp(data, [1, 1, 2, 2, 3, 3], 20, bstrp_omit_nan=True, rng=rng, ax=ax)
    assert np.all(np.isfinite(res.bss_all))
    with pytest.raises(ValueError, match="group_index"):
        plot_bstrp(data, [1, 2], 5, rng=rng, ax=ax)
    with pytest.raises(ValueError, match="x_data"):
        plot_bstrp(data, np.ones(6), 5, x_data=[1, 2], rng=rng, ax=ax)


def test_cmap_centered_symmetric_and_asymmetric():
    cm = cmap_centered("RdBu", 11, 0, clim=(-1, 1))
    assert cm.shape == (11, 3)
    np.testing.assert_allclose(cm[5], 255 / 256)  # white (MATLAB divides by 256)
    np.testing.assert_allclose(cm[0], np.array([5, 48, 97]) / 256)  # dark blue for negative
    np.testing.assert_allclose(cm[-1], np.array([103, 0, 31]) / 256)
    cm = cmap_centered("RdBu", 100, 0, clim=(-1, 3))
    # round(100 * 1/4) = 25 negative rows; white is the last of them and the first positive
    np.testing.assert_allclose(cm[24], 255 / 256)
    np.testing.assert_allclose(cm[25], 255 / 256)
    cm = cmap_centered("RdBu", 10, 0, clim=(1, 3))
    np.testing.assert_allclose(cm[0], 255 / 256)  # positive half only, white first
    cm = cmap_centered("RdBu", 10, 5, clim=(1, 3))
    np.testing.assert_allclose(cm[-1], 255 / 256)  # negative half only, white last


def test_cmap_centered_reads_current_axes_and_defaults():
    assert cmap_centered().shape == (256, 3)  # no figure: CLim [0 1], positive half
    _, ax = plt.subplots()
    ax.imshow(np.zeros((2, 2)), vmin=-2, vmax=2)
    cm = cmap_centered("PuOr", 21)
    np.testing.assert_allclose(cm[10], 255 / 256)
    with pytest.raises(ValueError, match="unknown"):
        cmap_centered("viridis", 10, clim=(0, 1))


def test_violin_children_and_options():
    rng = np.random.default_rng(0)
    _, ax = plt.subplots()
    v = violin(rng.normal(size=100), 2, ax=ax, rng=rng, show_mean=True, show_notches=True)
    assert v.parent is ax and v.orientation == "vertical"
    for name in ("violin_plot", "scatter_plot", "box_plot", "whisker_plot", "median_plot", "mean_plot"):
        assert getattr(v, name) is not None, name
    assert len(v.notch_plots) == 2
    xy = v.violin_plot.get_xy()
    assert xy[:, 0].max() == pytest.approx(2.3) and xy[:, 0].min() == pytest.approx(1.7)
    pts = v.scatter_plot.get_offsets()
    assert np.all(np.abs(pts[:, 0] - 2) <= 0.3 + 1e-12)


def test_violin_split_half_horizontal_and_styles():
    rng = np.random.default_rng(1)
    _, ax = plt.subplots()
    v = violin([rng.normal(size=40), rng.normal(2, 1, 40)], 1, ax=ax, rng=rng,
               violin_color=[(1, 0, 0), (0, 0, 1)], violin_alpha=[0.5, 0.2])
    assert v.violin_plot2 is not None and v.scatter_plot2 is not None
    assert v.violin_plot.get_xy()[:, 0].max() <= 1 + 1e-12  # first set on the left
    assert v.violin_plot2.get_xy()[:, 0].min() >= 1 - 1e-12
    h = violin(rng.normal(size=30), 3, ax=ax, rng=rng, orientation="horizontal", half_violin="right",
               quartile_style="shadow", data_style="histogram")
    assert h.violin_plot.get_xy()[:, 1].min() >= 3 - 1e-12
    assert h.violin_plot_q is not None and h.histogram_plot is not None and h.box_plot is None
    with pytest.raises(ValueError):
        violin([1.0, 2.0], 1, ax=ax, quartile_style="bad")


def test_violinplot_layouts():
    rng = np.random.default_rng(2)
    _, ax = plt.subplots()
    vs = violinplot(rng.normal(size=(30, 3)), ["a", "b", "c"], ax=ax, rng=rng)
    assert [v.pos for v in vs] == [1, 2, 3]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]
    _, ax = plt.subplots()
    vals = rng.normal(size=60)
    cats = np.repeat(["y", "x"], 30)
    vs = violinplot(vals, cats, ax=ax, rng=rng)
    assert len(vs) == 2 and [t.get_text() for t in ax.get_xticklabels()] == ["x", "y"]
    _, ax = plt.subplots()
    vs = violinplot({"b": rng.normal(size=10), "a": rng.normal(size=10), "s": ["no"] * 10}, ax=ax, rng=rng)
    assert len(vs) == 2
    _, ax = plt.subplots()
    vs = violinplot([rng.normal(size=(20, 2)), rng.normal(size=(20, 2))], ax=ax, rng=rng)
    assert len(vs) == 2 and all(v.violin_plot2 is not None for v in vs)
