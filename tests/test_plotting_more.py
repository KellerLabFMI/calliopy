"""Synthetic tests for ROIs2image, plotSEM, specTraces and plotBSTRP_multi ports."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from calliopy.io import MatStruct
from calliopy.plotting import (
    bootstrap_overlap_pvalues,
    plot_bstrp_multi,
    plot_sem,
    rois_to_image,
    spec_traces,
    spec_traces_data,
)
from calliopy.plotting._rois_image import _bwperim, _jet


@pytest.fixture(autouse=True)
def _close():
    plt.switch_backend("Agg")
    yield
    plt.close("all")


def _lin(rows, cols, dim):
    """1-based column-major linear indices of the given 0-based (row, col) pairs."""
    return np.ravel_multi_index((np.asarray(rows), np.asarray(cols)), dim, order="F") + 1


def _square_roi(r0, c0, n, dim):
    rr, cc = np.mgrid[r0 : r0 + n, c0 : c0 + n]
    return MatStruct(indices=_lin(rr.ravel(), cc.ravel(), dim))


def test_jet_matches_matlab_values():
    np.testing.assert_array_equal(_jet(4), [[0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0]])
    np.testing.assert_allclose(
        _jet(5), [[0, 0.5, 1], [0, 1, 1], [0.5, 1, 0.5], [1, 1, 0], [1, 0.5, 0]]
    )
    np.testing.assert_allclose(_jet(64)[0], [0, 0, 0.5625])


def test_bwperim_square_and_border():
    m = np.zeros((5, 5), dtype=bool)
    m[1:4, 1:4] = True
    p = _bwperim(m)
    assert p.sum() == 8 and not p[2, 2]
    assert _bwperim(np.ones((3, 3), dtype=bool)).sum() == 8  # image border is background


def test_rois_to_image_labels():
    dim = (10, 12)
    rois = [_square_roi(0, 0, 2, dim), _square_roi(5, 5, 3, dim), _square_roi(7, 0, 2, dim)]
    im = rois_to_image(rois, dim, randcolor=False)
    assert im.shape == dim
    assert im[0, 0] == 2 and im[6, 6] == 3 and im[8, 1] == 4 and im[3, 3] == 0
    # selection: ROI 1 in group 1 only; loop over length(sel)=2 positions (MATLAB quirk)
    im2 = rois_to_image(rois, dim, randcolor=False, sel=[1])
    assert im2[0, 0] == 1 and im2[6, 6] == 2 and im2[8, 1] == 0
    per = rois_to_image(rois, dim, randcolor=False, type="perim")
    assert per[6, 6] == 0 and per[5, 5] == 3
    single = rois_to_image(rois, dim, singlecolor=[0, 1, 1])
    assert single[6, 6] == 20
    rng = np.random.default_rng(0)
    rnd = rois_to_image(rois, dim, rng=rng)
    assert sorted({rnd[0, 0], rnd[6, 6], rnd[8, 1]}) == [2, 3, 4]
    with pytest.raises(ValueError):
        rois_to_image(rois, dim)  # random colours need rng


def test_rois_to_image_pixelidxlist_single_roi_and_groups():
    dim = (6, 6)
    roi = MatStruct(PixelIdxList=_lin([1, 2], [1, 1], dim))
    im = rois_to_image([roi], dim, randcolor=False)  # MATLAB fails on both counts
    assert im[1, 1] == 2 and im.sum() == 4
    rois = [_square_roi(0, 0, 1, dim), _square_roi(3, 3, 1, dim)]
    # a (2, 2) sel is transposed, as MATLAB: rows [0, 2], [1, 1] become [0, 1], [2, 1]
    # (ROI 0 and ROI 2 in group 1), so ROI 1 is drawn as unselected
    im2 = rois_to_image(rois, dim, randcolor=False, sel=[[0, 2], [1, 1]])
    assert im2[0, 0] == 2 and im2[3, 3] == 1


def test_rois_to_image_template_overlay():
    dim = (10, 10)
    template = np.arange(100, dtype=float).reshape(dim)
    rois = [_square_roi(0, 0, 2, dim), _square_roi(5, 5, 2, dim)]
    im = rois_to_image(rois, dim, randcolor=False, template=template, sel=[1])
    assert im.shape == (10, 10, 3)
    np.testing.assert_allclose(im[0, 0], 0.5)  # unselected -> grey
    np.testing.assert_allclose(im[5, 5], _jet(1)[0])  # group 1 colour
    lo, hi = np.percentile(template, [1, 99], method="hazen")
    np.testing.assert_allclose(im[3, 3, 0], (template[3, 3] - lo) / (hi - lo))
    assert im.min() >= 0 and im.max() <= 1


def test_plot_sem_values():
    d = np.array([[1.0, 3.0, np.nan], [2.0, 2.0, 2.0], [0.0, 4.0, 8.0]])
    _, ax = plt.subplots()
    h = plot_sem(d, "-r", ax=ax, linewidth=2)
    np.testing.assert_allclose(h.get_ydata(), [2.0, 2.0, 4.0])
    np.testing.assert_allclose(h.get_xdata(), [1, 2, 3])
    assert h.get_color() == "r" and h.get_linewidth() == 2
    xy = ax.patches[0].get_xy()
    top = xy[:3, 1]
    np.testing.assert_allclose(top, [2 + 1.0, 2.0, 4 + 4 / np.sqrt(3)])


def test_spec_traces():
    d = np.vstack([np.linspace(0, 2, 50), np.linspace(5, 1, 50), np.ones(50)])
    out = spec_traces_data(d)  # 3 x 50 -> transposed to 50 x 3
    assert out.shape == (50, 3)
    np.testing.assert_allclose(out[:, 0], np.linspace(0, 1, 50) + 2.2)
    np.testing.assert_allclose(out[:, 1], np.linspace(1, 0, 50) + 1.1)
    assert np.isnan(out[:, 2]).all()  # constant trace: 0/0
    res = spec_traces(d, legend_names=["a", "b", "c"], title="t")
    ax = res.h.axes[0]
    assert ax.get_ylim() == pytest.approx((0, 3.3))
    assert [t.get_text() for t in ax.get_yticklabels()] == ["3", "2", "1"]
    assert spec_traces(d, plot=False).h is None


def test_bootstrap_overlap_pvalues():
    b1 = np.array([[1.0, 2.0, 3.0, 4.0], [0, 0, 0, 0]])
    b2 = np.array([[0.0, 0.0, 0.0, 5.0], [0, 0, 0, 0]])
    np.testing.assert_allclose(bootstrap_overlap_pvalues(b1, b2, "both"), [0.5, 0.0])
    np.testing.assert_allclose(bootstrap_overlap_pvalues(b1, b2, "right"), [0.25, 0.0])
    np.testing.assert_allclose(bootstrap_overlap_pvalues(b1, b2, "left"), [0.75, 0.0])
    with pytest.raises(ValueError):
        bootstrap_overlap_pvalues(b1, b2, "up")


def test_plot_bstrp_multi():
    rng = np.random.default_rng(1)
    n_time = 20
    a = rng.normal(0, 0.1, size=(n_time, 12))
    b = rng.normal(5, 0.1, size=(n_time, 12))
    c = rng.normal(0, 0.1, size=(n_time, 12))
    g = np.repeat([1, 2, 3], 4)
    _, ax = plt.subplots()
    res = plot_bstrp_multi([a, b, c], [g, g, g], 200, rng=rng, ax=ax)
    r = res.results
    np.testing.assert_array_equal(r.pairs, [[0, 1], [0, 2], [1, 2]])
    assert r.pvals.shape == (n_time, 3) and r.sig_mask.shape == (n_time, 3)
    assert r.sig_mask[:, 0].all() and r.sig_mask[:, 2].all()
    assert len(res.h_lines) == 3 and len(r.bss_all) == 3
    assert ax.get_xlim() == pytest.approx((1, n_time))
    _, ax2 = plt.subplots()
    res2 = plot_bstrp_multi([a, b], [g, g], 50, pairs=None, rng=rng, ax=ax2, ylim=(-1, 6))
    assert res2.results.pvals.shape == (n_time, 0)
    assert ax2.get_ylim() == pytest.approx((-1, 6))
    with pytest.raises(ValueError):
        plot_bstrp_multi([a, b], [g, g], 10, pairs=[[0, 2]], rng=rng, ax=ax2)
