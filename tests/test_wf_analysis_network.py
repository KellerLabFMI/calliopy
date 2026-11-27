"""Synthetic tests: correlation analyses of calliopy.widefield (ports of calliope
21_Analysis WF_corrmat*, WF_corrmatBoxplot, WF_findLowCorrcoef, WF_plotROICountsByLocation,
WF_LesioningROIs, WF_seedcorr, WF_plotCorrChangeInteractive, WF_maskROIs)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.stats
from test_wf_analysis import make_rd, make_roiinfo, make_site, make_template

from calliopy import widefield as wf
from calliopy.widefield import _network as nw


def _pm(n_sites=2, n_tp=2, n_roi=818, nbr=(60, 40), seed=0, roiinfo=None):
    rng = np.random.default_rng(seed)
    sites = []
    for s in range(n_sites):
        rds = []
        for _ in range(n_tp):
            act = 5 + rng.standard_normal((n_roi, sum(nbr)))
            run = np.zeros((1, sum(nbr)))
            run[0, ::3] = 1.0
            rds.append(make_rd(act, list(nbr), Running=run, Running_smoothed=run.copy(),
                               ROIinfo=make_roiinfo(n_roi) if roiinfo is None else roiinfo))
        sites.append(make_site(rds, site_id=s + 1))
    return sites


def _small_pm(n_roi=12, n_tp=2, n_sites=2, nbr=(30, 20), seed=0):
    info = make_roiinfo(n_roi)
    return _pm(n_sites, n_tp, n_roi, nbr, seed, roiinfo=info)


# --- wf_corrmat ----------------------------------------------------------------------------


def test_wf_corrmat_matches_corrcoef():
    pm = _small_pm()
    cm, used = wf.wf_corrmat(pm)
    assert cm.shape == (12, 12, 2, 2) and used.shape == (1, 2)
    np.testing.assert_allclose(cm[:, :, 1, 0], np.corrcoef(pm[1].rd[0, 0].act), rtol=1e-12)


def test_wf_corrmat_filters():
    pm = _small_pm()
    act = pm[0].rd[0, 0].act
    cm, _ = wf.wf_corrmat(pm, comment="PB", tp_list=[0])  # second stack only
    np.testing.assert_allclose(cm[:, :, 0, 0], np.corrcoef(act[:, 30:]), rtol=1e-12)
    cm, _ = wf.wf_corrmat(pm, comment="FB", paradigm="PB", logic="or", tp_list=[0])
    np.testing.assert_allclose(cm[:, :, 0, 0], np.corrcoef(act), rtol=1e-12)
    cm, _ = wf.wf_corrmat(pm, running_state="running", tp_list=[0])
    np.testing.assert_allclose(cm[:, :, 0, 0], np.corrcoef(act[:, ::3]), rtol=1e-12)
    mask = np.zeros(12, bool)
    mask[:5] = True
    cm, _ = wf.wf_corrmat(pm, roi_mask=mask, tp_list=[1])
    assert cm.shape == (5, 5, 2, 1)
    cm, used = wf.wf_corrmat(pm, site_id=[2], tp_list=[[0, 1]])
    assert used.shape == (1, 1) and used[0, 0] is pm[1]
    with pytest.warns(UserWarning, match="No sites"):
        cm, used = wf.wf_corrmat(pm, site_id=[9])
    assert cm.shape == (0, 0) and used.shape == (1, 0)


def test_wf_corrmat_per_site_tp_list_zero_fill():
    pm = _small_pm()
    cm, _ = wf.wf_corrmat(pm, tp_list=[[0, 1], [1]])
    assert cm.shape == (12, 12, 2, 2)
    assert np.all(cm[:, :, 1, 1] == 0)  # never written: zero, as MATLAB growth


def test_median_division_matlab_stack_offsets():
    pm = _small_pm()
    rd = pm[0].rd[0, 0]
    rd.median_act = np.full((12, 1), 2.0)
    cm, _ = wf.wf_corrmat(pm, median_div_on=True, tp_list=[0])
    a = rd.act * 2.0
    s1 = a[:, 0:29] / np.median(a[:, 0:29], axis=1, keepdims=True)  # frms(1):frms(2)-1
    s2 = a[:, 29:49] / np.median(a[:, 29:49], axis=1, keepdims=True)  # 30:49, last dropped
    np.testing.assert_allclose(cm[:, :, 0, 0], np.corrcoef(np.hstack([s1, s2])), rtol=1e-12)


def test_wf_corrmat_bandpass_uuid_skip(monkeypatch):
    pm = _small_pm()
    pm[0].uuid = "u-WF_bandpass_0.2-8"
    called = []
    monkeypatch.setattr(nw, "wf_bandpass", lambda *a, **k: called.append(1))
    wf.wf_corrmat(pm, bandpass_on=True)
    assert not called


# --- corr_distance_histogram / plots -------------------------------------------------------


def test_corr_distance_histogram_bins():
    pm = _small_pm()
    c = np.full((12, 12), 0.5)
    np.fill_diagonal(c, 1)
    h = wf.corr_distance_histogram(c, pm)
    assert h.normalized_data.shape == (40, 40)
    np.testing.assert_allclose(h.normalized_data.sum(), 1.0)
    np.testing.assert_allclose(h.normalized_data[20].sum(), 1.0)  # 0.5 -> bin [0.5, 0.525)
    info = pm[0].rd[0, 0].ROIinfo.ravel()
    ml = np.array([r.ML_index for r in info])
    ap = np.array([r.AP_index for r in info])
    d = np.hypot(ml[:, None] - ml[None], ap[:, None] - ap[None])[np.tril_indices(12, -1)]
    cnt, _ = np.histogram(d / 16, bins=h.x_edges)
    np.testing.assert_allclose(h.normalized_data[20], cnt / cnt.sum())


def test_corr_distance_histogram_zero_corr_raises():
    pm = _small_pm()
    c = np.full((12, 12), 0.5)
    c[3, 1] = 0
    with pytest.raises(ValueError):
        wf.corr_distance_histogram(c, pm)


def test_corrmat_plots():
    pm = _small_pm()
    cm, used = wf.wf_corrmat(pm)
    contour, norm, _ = wf.wf_corrmat_plot(cm[:, :, :, 0], used)
    assert isinstance(contour, wf.CorrContour) and norm.shape == (40, 40)
    assert isinstance(wf.plot_wf_corrmat(cm, used, comment="FB"), plt.Figure)
    plt.close("all")


# --- WF_corrmat_ROI_filter / WF_corrmat_matrix --------------------------------------------


def test_wf_corrmat_roi_filter():
    pm = _small_pm(n_roi=300)  # VISp (AP 0-5) and RSP (AP 6-7)
    res = wf.wf_corrmat_roi_filter(pm, exclude_rois=["VISp"], tp_list=[0])
    acr = [r.Allen_acronym for r in pm[0].rd[0, 0].ROIinfo.ravel()]
    keep = np.array([a != "VISp" for a in acr])
    assert 0 < keep.sum() < 300
    np.testing.assert_array_equal(res.keep_mask, keep)
    np.testing.assert_allclose(res.corrmat_all_filtered,
                               res.corrmat_all[np.ix_(keep, keep, [0, 1], [0])])
    fig = wf.plot_wf_corrmat_roi_filter(res, exclude_rois=["VISp"])
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_wf_corrmat_matrix_regions():
    pm = _pm(n_sites=1, n_tp=1, nbr=(50,))
    act = pm[0].rd[0, 0].act
    info = pm[0].rd[0, 0].ROIinfo.ravel()
    res = wf.wf_corrmat_matrix(pm, allen_regions=["VISp", "RSP"])
    assert res.region_labels == ["VISp-L", "VISp-R", "RSP-L", "RSP-R"]

    def idx(a, h):
        return [i for i, r in enumerate(info) if a in r.Allen_acronym and h in r.name]

    g = np.vstack([act[idx("VISp", "left")].mean(0), act[idx("VISp", "right")].mean(0),
                   act[idx("RSP", "left")].mean(0), act[idx("RSP", "right")].mean(0)])
    np.testing.assert_allclose(res.corrmat_all[:, :, 0], np.corrcoef(g), rtol=1e-12)
    res2 = wf.wf_corrmat_matrix(pm, allen_regions=["VISp", "RSP"], hemisphere_organization=2)
    assert res2.region_labels == ["VISp-L", "RSP-L", "VISp-R", "RSP-R"]
    res3 = wf.wf_corrmat_matrix(pm, allen_regions=["VISp", "RSP"], hemisphere_organization=3)
    assert sorted(res3.region_labels) == sorted(res.region_labels)
    res4 = wf.wf_corrmat_matrix(pm, allen_regions=["VISp"], separate_hemispheres=False)
    assert res4.corrmat_all.shape == (1, 1, 1)
    with pytest.warns(UserWarning):
        r5 = wf.wf_corrmat_matrix(pm, allen_regions=["VISp", "NOPE"], separate_hemispheres=False)
    assert np.isnan(r5.corrmat_all).all()  # 'rows','complete' drops every frame
    assert isinstance(wf.plot_wf_corrmat_matrix(res.corrmat_all, res.region_labels,
                                                proj_meta=pm), plt.Figure)
    plt.close("all")


# --- WF_corrmatBoxplot ---------------------------------------------------------------------


def test_wf_corrmat_boxplot_ttest_and_bootstrap():
    pm = _small_pm()
    cm, used = wf.wf_corrmat(pm)
    res = wf.wf_corrmat_boxplot(cm, used, bregma_lambda_distance=2)
    assert len(res) == 1
    r = res[0]
    assert r.short_range.size + r.long_range.size == 2 * 66
    assert scipy.stats.ttest_ind(r.short_range, r.long_range).pvalue == pytest.approx(
        r.p_short_vs_long)
    assert scipy.stats.ttest_1samp(r.long_range, 0).pvalue == pytest.approx(
        r.p_long_vs_no_change)
    info = used[0, 0].rd[0, 0].ROIinfo.ravel()
    ml = np.array([x.ML_index for x in info])
    ap = np.array([x.AP_index for x in info])
    i, j = nw._tril_pairs(12)
    d = np.hypot(ml[i] - ml[j], ap[i] - ap[j])
    delta = (cm[i, j, 0, 1] - cm[i, j, 0, 0]) / (cm[i, j, 0, 0] + 1e-6) * 100
    np.testing.assert_allclose(r.short_by_site[0], delta[d <= 1.8])
    b = wf.wf_corrmat_boxplot(cm, used, bregma_lambda_distance=2, use_bootstrap=True,
                              num_bootstrap_samples=200, rng=np.random.default_rng(0))
    assert 0 <= b[0].p_short_vs_long <= 1 and b[0].ci_short_vs_long.shape == (2,)
    with pytest.raises(ValueError):
        wf.wf_corrmat_boxplot(cm, used, use_bootstrap=True)
    assert wf.wf_corrmat_boxplot(cm[:, :, :, :1], used) == []
    figs = wf.plot_wf_corrmat_boxplot(res)
    assert len(figs) == 1
    plt.close("all")


def test_significance_stars():
    assert [nw._significance_stars(p) for p in (0.0005, 0.005, 0.04, 0.2)] == [
        "***", "**", "*", "n.s."]


# --- WF_findLowCorrcoef --------------------------------------------------------------------


def test_wf_find_low_corrcoef_matlab_and_fixed():
    c = np.array([[1, 0.05, 0.5], [0.05, 1, 0.02], [0.5, 0.02, 1]])
    cm = c[:, :, None, None]
    res = wf.wf_find_low_corrcoef(cm)
    np.testing.assert_array_equal(res.idx_counts_sort_all[0], [[1, 6], [0, 5], [2, 5]])
    fixed = wf.wf_find_low_corrcoef(cm, lower_triangle_only=True)
    np.testing.assert_array_equal(fixed.idx_counts_sort_all[0], [[1, 2], [0, 1], [2, 1]])
    none = wf.wf_find_low_corrcoef(cm, cc_range=(0.9, 0.95), lower_triangle_only=True)
    assert none.idx_counts_sort_all[0].shape == (0, 0)
    with pytest.warns(DeprecationWarning):
        wf.wf_find_low_corrcoef(cm, cc_th=0.1)
    # tp_list values are ignored, only its length counts (MATLAB loop counter)
    cm2 = np.concatenate([cm, np.eye(3)[:, :, None, None]], axis=3)
    a = wf.wf_find_low_corrcoef(cm2, tp_list=[1], lower_triangle_only=True)
    np.testing.assert_array_equal(a.idx_counts_sort_all[0], fixed.idx_counts_sort_all[0])


def test_plot_roi_counts_by_location():
    pm = _small_pm()
    res = wf.wf_find_low_corrcoef(np.eye(12)[:, :, None, None] * 0 + 0.05)
    figs = wf.plot_roi_counts_by_location(pm, res.idx_counts_sort_all, res.site_id)
    assert len(figs) == 1
    figs = wf.plot_roi_counts_by_location(pm, [np.zeros((0, 0))], [0])
    assert len(figs) == 1
    plt.close("all")


# --- WF_LesioningROIs ----------------------------------------------------------------------


def _brute_lesion(c, remove):
    keep = ~remove
    s = c[np.ix_(keep, keep)]
    return s[np.triu_indices(s.shape[0], 1)].mean()


def test_wf_lesioning_rois_metrics():
    pm = _pm(n_sites=2, n_tp=1, nbr=(40,))
    res = wf.wf_lesioning_rois(pm)
    sc = res.lesioning_scores["TP1"]
    c = res.corrmat_all[:, :, :, 0].mean(axis=2)
    n = c.shape[0]
    assert n == 818
    base = c[~np.eye(n, dtype=bool)].mean()
    assert sc.baseline == pytest.approx(base, rel=1e-12)
    for i in (0, 400, 817):
        m = np.zeros(n, bool)
        m[i] = True
        assert sc.lesion_change[i] == pytest.approx(_brute_lesion(c, m) - base, abs=1e-12)
        assert sc.roi_avg_corr[i] == pytest.approx(np.delete(c[i], i).mean(), rel=1e-12)
    acr = [r.Allen_acronym for r in pm[0].rd[0, 0].ROIinfo.ravel()]
    assert sc.region_names == sorted(set(acr))
    k = sc.region_names.index("RSP")
    m = np.array([a == "RSP" for a in acr])
    assert sc.region_values[k] == pytest.approx(_brute_lesion(c, m) - base, abs=1e-12)
    grp = wf.wf_lesioning_rois(pm, tp_list=[[0, 0]])
    assert list(grp.lesioning_scores) == ["TP[1 1]"]
    figs = wf.plot_wf_lesioning_rois(res.lesioning_scores, pm, template=make_template())
    assert len(figs) == 3
    figs = wf.plot_wf_lesioning_rois(res.lesioning_scores, pm)
    assert len(figs) == 3
    plt.close("all")


def test_lesion_nan_path_equals_vectorised():
    pm = _pm(n_sites=1, n_tp=1, nbr=(40,))
    res = wf.wf_lesioning_rois(pm)
    c = res.corrmat_all[:, :, 0, 0]
    sc = res.lesioning_scores["TP1"]
    n = c.shape[0]
    looped = np.array([nw._upper_mean_without(c, np.eye(1, n, i, dtype=bool)[0])
                       for i in (3, 77)]) - sc.baseline
    np.testing.assert_allclose(sc.lesion_change[[3, 77]], looped, atol=1e-13)


# --- WF_seedcorr ---------------------------------------------------------------------------


def test_wf_seedcorr_manual_and_nan():
    pm = _pm(n_sites=1, n_tp=1, nbr=(40,))
    res = wf.wf_seedcorr(pm, seed_pattern="manual", seed_rois=[0, 5])
    act = pm[0].rd[0, 0].act
    full = np.corrcoef(act)
    np.testing.assert_allclose(res.seed_corr[:, :, 0, 0], full[[0, 5]], rtol=1e-10)
    act[3, 7] = np.nan
    res = wf.wf_seedcorr(pm, seed_pattern="manual", seed_rois=[3])
    ok = ~np.isnan(act[3])
    assert res.seed_corr[0, 10, 0, 0] == pytest.approx(
        np.corrcoef(act[3, ok], act[10, ok])[0, 1], rel=1e-10)
    with pytest.raises(ValueError):
        wf.wf_seedcorr(pm, seed_pattern="manual", seed_rois=[818])
    with pytest.raises(ValueError):
        wf.wf_seedcorr(pm, seed_pattern="manual")


def test_wf_seedcorr_patterns_and_running():
    pm = _pm(n_sites=1, n_tp=1, nbr=(40,))
    std = wf.wf_seedcorr(pm)
    info = pm[0].rd[0, 0].ROIinfo.ravel()
    exp = [i for m in range(-20, 1, 4) for a in range(0, 20, 5) for i, r in enumerate(info)
           if r.ML_index == m and r.AP_index == a]
    np.testing.assert_array_equal(std.inds_roi_seed, exp)
    allen = wf.wf_seedcorr(pm, seed_pattern="allen")
    assert allen.inds_roi_seed.size >= 1
    run = wf.wf_seedcorr(pm, seed_pattern="manual", seed_rois=[0], running_state="running")
    act = pm[0].rd[0, 0].act[:, ::3]
    assert run.seed_corr[0, 1, 0, 0] == pytest.approx(np.corrcoef(act[0], act[1])[0, 1])
    pm[0].rd[0, 0].Running_smoothed = np.zeros((1, 40))
    none = wf.wf_seedcorr(pm, seed_pattern="manual", seed_rois=[0], running_state="running")
    assert np.isnan(none.seed_corr).all()
    figs = wf.plot_wf_seedcorr(std.seed_corr, std.inds_roi_seed[:3], make_template())
    assert len(figs) == 1
    plt.close("all")


# --- WF_plotCorrChangeInteractive ----------------------------------------------------------


def test_corr_change_pairs_and_density():
    pm = _small_pm()
    rng = np.random.default_rng(3)
    a = rng.uniform(0.1, 0.9, (12, 12))
    a = (a + a.T) / 2
    b = np.clip(a + 0.05, 0, 1)
    p = wf.corr_change_pairs(np.stack([a, a], axis=2), b, pm)
    assert p.naive.size == 2 * 66 and p.roi_pairs.shape == (132, 2)
    i, j = p.roi_pairs[0]
    assert (i, j) == (1, 0) and p.naive[0] == a[1, 0] and p.treated[0] == b[1, 0]
    d = wf.corr_change_roi_density(p, pm, a[1, 0], b[1, 0], radius=1e-9)
    assert d.n_pairs == 2 and d.roi_counts[0] == 2 and d.roi_counts[1] == 2
    assert d.image.sum() == 4
    assert wf.corr_change_roi_density(p, pm, 1.5, 0.5) is None
    assert isinstance(wf.plot_corr_change(p, pm, click=(a[1, 0], b[1, 0])), plt.Figure)
    assert isinstance(wf.plot_corr_change(p, pm), plt.Figure)
    plt.close("all")


# --- WF_maskROIs core ----------------------------------------------------------------------


def test_mask_rois_core():
    frame = np.arange(36, dtype=float).reshape(6, 6)
    # ROI 0: rows 0-2, cols 0-2 (1-based column-major linear indices); ROI 1: empty
    roi0 = np.array([r + 1 + 6 * c for c in range(3) for r in range(3)], float)
    rois = [roi0, np.zeros(0), np.array([36.0])]
    means = wf.roi_mean_values(frame, rois)
    assert means[0] == pytest.approx(frame[:3, :3].mean()) and np.isinf(means[1])
    assert means[2] == frame[5, 5]
    borders = wf.roi_border_indices(rois, frame.shape)
    assert set(borders[0]) == set(roi0) - {8}  # centre pixel (row 1, col 1) is interior
    mask = wf.roi_threshold_mask(means, 20.0, roi_mask=[True, True, False],
                                 has_border=[b.size > 0 for b in borders])
    np.testing.assert_array_equal(mask, [False, True, True])
    assert isinstance(wf.plot_mask_rois(frame, rois, mask), plt.Figure)
    plt.close("all")


def test_tril_helpers():
    m = np.arange(16, dtype=float).reshape(4, 4) + 1
    i, j = nw._tril_pairs(4)
    np.testing.assert_array_equal(nw._tril_vector(m), m[i, j])
    np.testing.assert_array_equal(i[:3], [1, 2, 3])  # column-major

