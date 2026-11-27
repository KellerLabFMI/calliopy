"""Synthetic tests: proj_meta tools, power maps and onsets of calliopy.widefield
(ports of calliope 21_Analysis WF_bandpass, WF_trimStackFrames, WF_velocityField,
WF_commodore, WF_onsets and 22_Plotting WF_imageROIvelocityField)."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.signal

from calliopy import widefield as wf
from calliopy.io import MatStruct
from calliopy.widefield import _proj_meta_tools as pmt

# --- synthetic proj_meta -------------------------------------------------------------------


def make_roiinfo(n=818, extra_ref=False):
    """ROIinfo struct array (1, n): grid ML -20..20, AP 0..; acronyms by AP band."""
    rois = []
    if extra_ref:
        rois += [MatStruct(ML_index=0.0, AP_index=0.0, Allen_acronym="", name="",
                           label="Bregma"),
                 MatStruct(ML_index=0.0, AP_index=30.0, Allen_acronym="", name="",
                           label="Lambda")]
    k = 0
    for ap in range(40):
        for ml in range(-20, 21):
            if k == n - (2 if extra_ref else 0):
                break
            acr = ["VISp", "RSP", "MOp", "SSp"][min(ap // 6, 3)]
            name = "left hemisphere" if ml < 0 else "right hemisphere"
            rois.append(MatStruct(ML_index=float(ml), AP_index=float(ap), Allen_acronym=acr,
                                  name=name, label=""))
            k += 1
    arr = np.empty((1, len(rois)), dtype=object)
    for i, r in enumerate(rois):
        arr[0, i] = r
    return arr


def make_rd(act, nbr_frames, comments=("FB one", "PB two"), **extra):
    c = np.empty((1, len(comments)), dtype=object)
    for i, s in enumerate(comments):
        c[0, i] = s
    rd = MatStruct(act=np.asarray(act, dtype=float),
                   nbr_frames=np.asarray(nbr_frames, dtype=float).reshape(1, -1),
                   comments=c, session=c.copy(), ROIinfo=extra.pop("ROIinfo", make_roiinfo()))
    rd.update(extra)
    return rd


def make_site(rds, site_id=1, uuid="abc"):
    arr = np.empty((1, len(rds)), dtype=object)
    for i, r in enumerate(rds):
        arr[0, i] = r
    return MatStruct(ExpGroup=np.array([[site_id, 1.0]]), projID="PRJ", Acode="A1",
                     animal=f"m{site_id}", uuid=uuid, rd=arr)


def as_pm(sites):
    arr = np.empty((1, len(sites)), dtype=object)
    for i, s in enumerate(sites):
        arr[0, i] = s
    return arr


# --- helpers -------------------------------------------------------------------------------


def test_struct_forms_and_select_sites():
    s1, s2 = make_site([make_rd(np.ones((818, 5)), [5])], 3), make_site(
        [make_rd(np.ones((818, 5)), [5])], 7)
    assert len(pmt._sites(s1)) == 1
    assert len(pmt._sites([s1, s2])) == 2
    assert len(pmt._sites(as_pm([s1, s2]))) == 2
    kept = wf.select_sites(as_pm([s1, s2]), [7])
    assert kept.shape == (1, 1) and kept[0, 0] is s2
    assert wf.select_sites([s1, s2], [99]).shape == (1, 0)
    assert pmt._rd_grid(MatStruct(rd=MatStruct(a=1))).shape == (1, 1)


@pytest.mark.parametrize(("x", "s"), [(0.2, "0.2"), (8, "8"), (math.pi, "3.1416"),
                                      (0.01, "0.01"), (123.456, "123.456")])
def test_num2str(x, s):
    assert pmt._num2str(x) == s


def test_msqueeze_and_logical():
    assert pmt._msqueeze(np.zeros((818, 1, 4, 3))).shape == (818, 4, 3)
    assert pmt._msqueeze(np.zeros((818, 1, 1, 3))).shape == (818, 3)
    assert pmt._msqueeze(np.zeros((818, 1, 4, 1))).shape == (818, 4)
    assert pmt._msqueeze(np.zeros((1, 1, 5))).shape == (5, 1)
    assert pmt._msqueeze(np.zeros((1, 5))).shape == (1, 5)
    np.testing.assert_array_equal(pmt._matlab_logical([True, False], 4),
                                  [True, False, False, False])
    np.testing.assert_array_equal(pmt._matlab_logical([1, 0, 1, 0, 0], 3), [True, False, True])
    with pytest.raises(IndexError):
        pmt._matlab_logical([0, 0, 0, 1], 3)


def test_concat_field_skips_empty():
    rois = [MatStruct(ML_index=1.0), MatStruct(ML_index=np.zeros((0, 0))),
            MatStruct(ML_index=3.0)]
    np.testing.assert_array_equal(pmt._concat_field(rois, "ML_index"), [1.0, 3.0])
    assert pmt._cell_field([MatStruct(a="x"), MatStruct(a="")], "a") == ["x", ""]


# --- wf_bandpass ---------------------------------------------------------------------------


def _bp_site(n_frames=8000, fs=100.0, n_roi=818):
    t = np.arange(n_frames) / fs
    rng = np.random.default_rng(0)
    act = 10 + 0.5 * np.sin(2 * np.pi * 2.0 * t)[None, :] + 3 * np.sin(
        2 * np.pi * 0.01 * t)[None, :] + 0.01 * rng.standard_normal((n_roi, n_frames))
    act[5, 100] = np.nan
    ft = (np.arange(n_frames) * 100.0).reshape(1, -1)  # 10 kHz aux samples, 100 Hz frames
    return make_site([make_rd(act, [n_frames], comments=("x",), frame_times=ft)])


def test_wf_bandpass_identity_filter_and_bookkeeping():
    site = _bp_site(4000)
    act0 = site.rd[0, 0].act.copy()
    out = wf.wf_bandpass([site], (0.2, 8), filter_fn=lambda x, fs, band: x)
    a = out[0, 0].rd[0, 0].act
    med = np.median(act0, axis=1, keepdims=True)
    good = ~np.isnan(act0).any(axis=1)
    np.testing.assert_allclose(a[good], 1 + act0[good] / med[good], rtol=1e-12)
    assert np.isnan(a[5]).all()
    assert out[0, 0].uuid == "abc-WF_bandpass_0.2-8"
    np.testing.assert_array_equal(site.rd[0, 0].act, act0, strict=True)  # input unchanged
    assert site.uuid == "abc"


def test_wf_bandpass_default_filter_passes_band_and_removes_drift():
    site = _bp_site()
    med = np.median(site.rd[0, 0].act[0])
    out = wf.wf_bandpass([site], (0.2, 8))
    a = out[0, 0].rd[0, 0].act[0, 2000:6000] - 1
    t = np.arange(2000, 6000) / 100.0
    rel = 0.5 / med * np.sin(2 * np.pi * 2.0 * t)  # the in-band part after median division
    # the 0.01 Hz drift (amplitude 0.3 after median division) is removed; the remaining
    # error comes from the noise and the filter ripple
    assert np.max(np.abs(a - rel)) < 0.01
    assert abs(np.std(a) / np.std(rel) - 1) < 0.05


def test_wf_bandpass_framerate_default_and_errors():
    site = _bp_site(2000)
    with pytest.raises(ValueError, match="3000"):
        wf.wf_bandpass([site], (0.2, 8))
    calls = {}

    def fn(x, fs, band):
        calls["fs"] = fs
        return x

    wf.wf_bandpass([_bp_site(3500)], (0.2, 8), filter_fn=fn)
    assert calls["fs"] == 100.0


def test_bandpass_fir_matches_ideal_response():
    fs = 100.0
    x = np.zeros((1, 20001))
    x[0, 10000] = 1.0
    y = pmt._bandpass_fir(x, fs, (1.0, 8.0))
    _, h = scipy.signal.freqz(y[0], worN=[2.0, 4.0, 0.05, 30.0], fs=fs)
    assert np.all(np.abs(np.abs(h[:2]) - 1) < 0.01)
    assert np.all(np.abs(h[2:]) < 0.01)
    np.testing.assert_allclose(y[0], y[0][::-1], atol=1e-15)  # zero phase (symmetric)


# --- wf_trim_stack_frames ------------------------------------------------------------------


def test_wf_trim_stack_frames_cell_comments():
    n = 14
    act = np.arange(824 * n, dtype=float).reshape(824, n)
    rd = make_rd(act, [10, 4], frame_times=np.arange(1, n + 1, dtype=float).reshape(1, -1),
                 ROIinfo=make_roiinfo(824, extra_ref=True))
    out = wf.wf_trim_stack_frames([make_site([rd])], 2)
    r = out[0, 0].rd[0, 0]
    np.testing.assert_array_equal(r.nbr_frames, [[6, 0]])
    assert r.act.shape == (818, 6)
    np.testing.assert_array_equal(r.act, act[:818, 2:8])
    np.testing.assert_array_equal(r.frame_times, [[3, 4, 5, 6, 7, 8]])
    assert r.Running.shape == (0, 0)  # missing field created, as MATLAB
    assert r.ROIinfo.shape == (1, 818)
    assert rd.act.shape == (824, n)  # input unchanged


def test_wf_trim_stack_frames_char_comments():
    rd = make_rd(np.ones((818, 10)), [10])
    rd.comments = "single"
    rd.Running = np.arange(10.0).reshape(1, -1)
    r = wf.wf_trim_stack_frames([make_site([rd])], 3)[0, 0].rd[0, 0]
    assert r.nbr_frames == 4
    np.testing.assert_array_equal(r.Running, [[3, 4, 5, 6]])
    r = wf.wf_trim_stack_frames([make_site([rd])], 5)[0, 0].rd[0, 0]
    assert r.nbr_frames == 0 and r.act.shape == (818, 0)


# --- wf_velocity_field / plot_roi_velocity_field -------------------------------------------


def _small_roiinfo():
    return [MatStruct(ML_index=float(m), AP_index=float(a)) for a in range(4) for m in range(5)]


def test_wf_velocity_field_moving_blob():
    info = _small_roiinfo()
    n = len(info)
    act = np.zeros((n, 6))
    for t in range(6):
        for i, r in enumerate(info):
            act[i, t] = math.exp(-((r.ML_index - 1 - 0.3 * t) ** 2 + (r.AP_index - 1.5) ** 2))
    site = make_site([make_rd(act, [6], comments=("x",), ROIinfo=np.array(info, dtype=object))])
    out = wf.wf_velocity_field([site], info, alpha=0.1, max_iter=200)
    r = out[0, 0].rd[0, 0]
    assert r.U.shape == (n, 6) and r.V.shape == (n, 6)
    vid = wf.wf_grid2vid(act, info)
    U, _ = wf.velocity_field_horn_schunck(vid, alpha=0.1, max_iter=200)
    np.testing.assert_allclose(r.U, wf.uv_img_to_linear(U, U, info).U_selected)
    assert out[0, 0].uuid == "abc-WF_velocityField_alpha=0.100_max_iter=200_sigma=1.0"


def test_plot_roi_velocity_field():
    info = _small_roiinfo()
    fig = wf.plot_roi_velocity_field(np.ones(20), np.zeros(20), 2.0, info)
    assert isinstance(fig, plt.Figure)
    with pytest.raises(ValueError):
        wf.plot_roi_velocity_field(np.ones(19), np.zeros(19), 2.0, info)
    with pytest.raises(ValueError):
        wf.plot_roi_velocity_field(np.ones(20), np.zeros(20), None, info)
    plt.close("all")


# --- AvgPsdInRange / wf_commodore ----------------------------------------------------------


@pytest.mark.parametrize("n", [200, 201])
def test_avg_psd_matches_periodogram(n):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, n))
    f, p = scipy.signal.periodogram(x, fs=50.0, window="boxcar", detrend=False, axis=1)
    sel = (f >= 2) & (f <= 7)
    got = pmt._avg_psd_in_range(x, 50.0, (2, 7), axis=1)
    np.testing.assert_allclose(got[:, 0], p[:, sel].mean(axis=1), rtol=1e-12)
    assert math.isnan(pmt._avg_psd_in_range(x[:, :1], 50.0, (2, 7), axis=1))


def _commodore_pm(n_sites=2, n_roi=818, stacks=(1000, 900)):
    rng = np.random.default_rng(2)
    sites = []
    for s in range(n_sites):
        act = rng.standard_normal((n_roi, sum(stacks)))
        sites.append(make_site([make_rd(act, list(stacks))], site_id=s + 1))
    return sites


def test_wf_commodore_shapes_and_values():
    sites = _commodore_pm()
    res = wf.wf_commodore(sites, frame_rate=10.0, analysis_window=10.0,
                          frequency_range=(1, 2))
    # lengths: (1000-400)+(900-400) = 1100 -> 11 windows of 100
    assert len(res) == 1 and res[0].shape == (818, 11)
    act = sites[0].rd[0, 0].act
    cols = np.r_[200:800, 1200:1700]
    seg = act[:, cols][:, :100]
    p0 = pmt._avg_psd_in_range(seg, 10.0, (1, 2), axis=1)[:, 0]
    act1 = sites[1].rd[0, 0].act
    p1 = pmt._avg_psd_in_range(act1[:, cols][:, :100], 10.0, (1, 2), axis=1)[:, 0]
    np.testing.assert_allclose(res[0][:, 0], (p0 + p1) / 2, rtol=1e-12)
    avg = wf.wf_commodore(sites, frame_rate=10.0, analysis_window=10.0,
                          average_over_time=True)
    assert avg[0].shape == (818, 1)
    m2 = wf.wf_commodore(sites, frame_rate=10.0, analysis_window=10.0,
                         average_over_psd=False)
    assert m2[0].shape == (818, 11)


def test_wf_commodore_matlab_quirks():
    sites = _commodore_pm(stacks=(600,))  # 200 usable frames
    one = wf.wf_commodore(sites, frame_rate=10.0, analysis_window=15.0)  # one window
    assert one[0].shape == (818, 2)  # MATLAB squeeze: one column per site (kept)
    with pytest.warns(UserWarning, match="larger than recording"):
        long_win = wf.wf_commodore(sites, frame_rate=10.0, analysis_window=100.0)
    assert long_win[0].shape == (818, 1) and np.isnan(long_win[0]).all()
    sites824 = _commodore_pm(n_roi=824, stacks=(600,))
    r = wf.wf_commodore(sites824, frame_rate=10.0, analysis_window=10.0)
    act = sites824[0].rd[0, 0].act
    p = pmt._avg_psd_in_range(act[2:820, 200:300], 10.0, (1, 2), axis=1)[:, 0]
    p1 = pmt._avg_psd_in_range(sites824[1].rd[0, 0].act[2:820, 200:300], 10.0, (1, 2),
                               axis=1)[:, 0]
    np.testing.assert_allclose(r[0][:, 0], (p + p1) / 2, rtol=1e-12)
    with pytest.warns(UserWarning):
        assert wf.wf_commodore(sites, site_id=[99]) == []


def test_plot_wf_commodore_fallback_and_template():
    sites = _commodore_pm()
    res = wf.wf_commodore(sites, frame_rate=10.0, analysis_window=10.0)
    fig = wf.plot_wf_commodore(res, [0], roi_info=sites[0].rd[0, 0].ROIinfo)
    assert isinstance(fig, plt.Figure)
    tmpl = make_template()
    fig = wf.plot_wf_commodore(res, [0], template=tmpl, average_over_time=True)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def make_template(n=818):
    """GridTemplate with one atlas pixel per ROI on a 30 x 41 image."""
    shape = (30, 41)
    idx = [np.array([1 + (i % 30) + 30 * (i // 30)], dtype=np.int64) for i in range(n)]
    borders = np.zeros(shape, bool)
    borders[:, 20] = True
    return wf.GridTemplate(np.arange(n, dtype=float), np.arange(n, dtype=float), idx,
                           ["r"] * n, ["VISp"] * n, borders, None)


# --- onsets --------------------------------------------------------------------------------


def test_onset_triggers():
    rd = MatStruct(StimulusID=np.array([[0, 0, 1, 1, 0, 1, 0]], float))
    np.testing.assert_array_equal(wf.onset_triggers(rd, "gratings"), [3, 6])
    run = np.zeros(20)
    run[[2, 12, 13]] = 1
    rd = MatStruct(Running_smoothed=run)
    np.testing.assert_array_equal(wf.onset_triggers(rd, "running", min_still_duration=5), [13])
    rd = MatStruct(PerturbationLFM=np.array([0, 1, 0, 0, 1, 1.0]),
                   Running_smoothed=np.array([0, 0, 0, 0, 0.5, 0.5]))
    np.testing.assert_array_equal(wf.onset_triggers(rd, "mismatch", thresh_run=0.1), [5])
    with pytest.raises(ValueError):
        wf.onset_triggers(rd, "nope")


def _onset_site(stim_frames, n=1200, nbr=(600, 600), site_id=1, response=2.0):
    info = make_roiinfo()
    n_roi = 818
    act = np.ones((n_roi, n))
    stim = np.zeros((1, n))
    vis = np.array([i for i, r in enumerate(info.ravel()) if r.Allen_acronym == "VISp"])
    for f in stim_frames:  # 1-based frame numbers
        stim[0, f - 1 : f + 19] = 1
        act[np.ix_(vis, np.arange(f - 1, f + 49))] += response
    rd = make_rd(act, list(nbr), StimulusID=stim, ROIinfo=info)
    return make_site([rd], site_id=site_id), vis


def test_wf_onsets_gratings_values_and_edges():
    site, vis = _onset_site([300, 620, 1150])  # 620 is near the stack boundary 600
    s = wf.wf_onsets([site], "gratings", window=(-10, 60), baseline_window=(-30, -20),
                     normalization_method="none")
    assert s.shape == (vis.size, 71, 1, 1)
    trace = s[0, :, 0, 0]
    np.testing.assert_allclose(trace[:10], 0)
    np.testing.assert_allclose(trace[10:60], 2.0)  # only the onset at 300 is kept
    np.testing.assert_allclose(trace[60:], 0)
    med = wf.wf_onsets([site], "gratings", window=(-10, 60), baseline_window=(-30, -20))
    np.testing.assert_allclose(med[0, 10:60, 0, 0], 2.0)  # median is 1
    none = wf.wf_onsets([site], "gratings", comment=["nothing"])
    assert none.shape == (0, 0)


def test_wf_onsets_groups_sites_and_session_type():
    a, _ = _onset_site([300], site_id=1)
    b, _ = _onset_site([300], site_id=2, response=4.0)
    s = wf.wf_onsets([a, b], "gratings", window=(-10, 60), baseline_window=(-30, -20),
                     normalization_method="none", tp_list=[[0], [5]])
    assert s.shape[2:] == (2, 2)
    np.testing.assert_allclose(s[0, 20, :, 0], [2.0, 4.0])
    assert np.isnan(s[:, :, :, 1]).all()
    pb = wf.wf_onsets([a], "gratings", session_type="PB", window=(-10, 60),
                      baseline_window=(-30, -20))
    assert pb.shape[0] > 0  # comments contain 'PB' (any of the cell)
    with pytest.raises(ValueError):
        wf.wf_onsets([a], "gratings", baseline_window=(-10, 5))


def test_wf_onsets_running_state_filter():
    site, _ = _onset_site([300])
    run = np.zeros(1200)
    site.rd[0, 0].Running_smoothed = run
    stat = wf.wf_onsets([site], "gratings", running_state="stationary", window=(-10, 60),
                        baseline_window=(-30, -20))
    assert stat.shape[0] > 0
    runn = wf.wf_onsets([site], "gratings", running_state="running", window=(-10, 60),
                        baseline_window=(-30, -20))
    assert runn.shape == (0, 0)


def test_plot_wf_onsets():
    site, _ = _onset_site([300])
    s = wf.wf_onsets([site], "gratings", window=(-10, 60), baseline_window=(-30, -20),
                     tp_list=[0, 0])
    assert isinstance(wf.plot_wf_onsets(s, [0, 0], window=(-10, 60)), plt.Figure)
    assert isinstance(wf.plot_wf_onsets(s, overlay_plots=True, window=(-10, 60)), plt.Figure)
    fig = wf.plot_wf_onsets(s, window=(-10, 60), bootstrap_plot=True,
                            rng=np.random.default_rng(0))
    assert isinstance(fig, plt.Figure)
    assert wf.plot_wf_onsets(np.zeros((0, 0))) is None
    plt.close("all")
