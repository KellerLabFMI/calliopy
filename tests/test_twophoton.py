"""Synthetic tests of calliopy.twophoton (activity, quality, aux, score sheets) and the
private MATLAB built-in helpers of the 2P/EEG/OII track."""

from __future__ import annotations

import datetime

import numpy as np
import pytest
from matplotlib.figure import Figure
from scipy import ndimage

from calliopy.io import MatStruct
from calliopy.twophoton import (
    align_site_first_frames,
    aux_channel_names,
    aux_frame_time_stats,
    body_weight_score,
    calc_act,
    calc_act_map,
    cam_exposure_check,
    compare_stacks_images,
    excel_column_name,
    get_iframe_times,
    neuropil_indices,
    np_activity,
    observation_too_late,
    online_response,
    plot_compare_stacks,
    plot_online_response,
    plot_site_rois,
    plot_tpi_activity_similarity,
    registration_failed,
    roi_activity,
    roi_perimeter_image,
    roi_positions,
    roi_similarity,
    roi_traces,
    rois_to_mask,
    score_sheet_problem,
    score_sheet_row_values,
    site_grid_shape,
    site_roi_label_image,
    stack_projections,
    stack_template,
    tile_piezo_layers,
    tpi_adata_check,
    tpi_adata_incomplete,
    tpi_inspection_flags,
    tpi_mean_roi_traces,
)
from calliopy.twophoton._builtins import (
    fspecial_average,
    fspecial_gaussian,
    imfilter,
    impyramid_reduce,
    mat2gray,
    matchpairs,
    matlab_colon,
    ntzo,
    smooth_moving,
    sub2ind_f,
)


def _lin(rows, cols, shape):
    """1-based column-major linear indices of (row, col) pairs (0-based inputs)."""
    return np.ravel_multi_index((np.asarray(rows), np.asarray(cols)), shape, order="F") + 1


@pytest.fixture
def stack():
    rng = np.random.default_rng(1)
    return rng.integers(0, 1000, size=(6, 5, 7)).astype(np.int16)


# --- builtins ------------------------------------------------------------------------------


def test_matlab_colon_endpoints_and_length():
    t = matlab_colon(-0.5, 1 / 1000, 3)
    assert t.size == 3501
    assert t[0] == -0.5 and t[-1] == 3.0
    np.testing.assert_allclose(np.diff(t), 1e-3, atol=1e-12)
    assert matlab_colon(1, 1, 0).size == 0
    np.testing.assert_array_equal(matlab_colon(1, 2, 6), [1, 3, 5])


def test_sub2ind_f_matches_ravel():
    assert sub2ind_f((3, 4), [1, 3], [1, 4]).tolist() == [1, 12]


def test_ntzo_and_mat2gray():
    x = np.array([[2, 4], [6, 10]], dtype=np.uint16)
    out = ntzo(x)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[0, 0.25], [0.5, 1]])
    np.testing.assert_array_equal(ntzo(np.full((2, 2), 3.0)), np.zeros((2, 2)))
    g = mat2gray(np.array([1.0, np.nan, 3.0, 5.0]))
    np.testing.assert_allclose(g, [0, np.nan, 0.5, 1])
    np.testing.assert_allclose(mat2gray(np.array([0.0, 5, 10]), (2, 8)), [0, 0.5, 1])


def test_fspecial():
    h = fspecial_gaussian(5, 1.0)
    assert h.shape == (5, 5) and np.isclose(h.sum(), 1)
    np.testing.assert_allclose(h, h.T)
    np.testing.assert_allclose(h, h[::-1, ::-1])
    assert h[2, 2] == h.max()
    np.testing.assert_allclose(fspecial_average(3), np.full((3, 3), 1 / 9))


def test_imfilter_matches_scipy_correlate_and_keeps_class():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(9, 11))
    k = rng.normal(size=(3, 5))
    ref = ndimage.correlate(a, k, mode="constant", cval=0)
    np.testing.assert_allclose(imfilter(a, k), ref, atol=1e-12)
    u = np.array([[0, 10], [20, 65535]], dtype=np.uint16)
    out = imfilter(u, np.full((1, 1), 1.5))
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, [[0, 15], [30, 65535]])
    st = rng.normal(size=(9, 11, 3))
    np.testing.assert_allclose(imfilter(st, k)[:, :, 2], ndimage.correlate(st[:, :, 2], k, mode="constant"),
                               atol=1e-12)


def test_impyramid_reduce_shape_and_constant():
    a = np.full((7, 10, 2), 3.0)
    out = impyramid_reduce(a)
    assert out.shape == (4, 5, 2)
    np.testing.assert_allclose(out, 3.0)
    ramp = np.tile(np.arange(9.0), (5, 1))
    r = impyramid_reduce(ramp)
    np.testing.assert_allclose(r[:, 1:-1], ramp[::2, ::2][:, 1:-1])  # linear interior preserved
    assert impyramid_reduce(np.ones((4, 4), dtype=np.uint16)).dtype == np.uint16


def test_smooth_moving_ends_and_interior():
    y = np.array([1.0, 5, 2, 8, 3, 9, 4, 0])
    out = smooth_moving(y, 5)
    assert out[0] == 1
    assert out[1] == pytest.approx((1 + 5 + 2) / 3)
    assert out[2] == pytest.approx(y[:5].mean())
    assert out[-1] == 0 and out[-2] == pytest.approx(y[-3:].mean())
    np.testing.assert_allclose(smooth_moving(y, 4), smooth_moving(y, 3))  # even span -> odd


def test_matchpairs_against_brute_force():
    rng = np.random.default_rng(3)
    import itertools

    for _ in range(20):
        c = rng.uniform(0, 3, size=(3, 4))
        cu = 1.0
        pairs = matchpairs(c, cu)
        tot = c[pairs[:, 0], pairs[:, 1]].sum() + cu * (3 + 4 - 2 * len(pairs))
        best = np.inf
        for k in range(4):
            for rows in itertools.combinations(range(3), k):
                for cols in itertools.permutations(range(4), k):
                    best = min(best, sum(c[r, q] for r, q in zip(rows, cols)) + cu * (7 - 2 * k))
        assert tot == pytest.approx(best)
    assert matchpairs(np.zeros((0, 3)), 1).shape == (0, 2)
    assert matchpairs(np.array([[5.0]]), 1).shape == (0, 2)  # above 2 * cost_unmatched


# --- activity ------------------------------------------------------------------------------


def test_roi_traces_and_roi_activity(stack):
    h, w, n = stack.shape
    ix = _lin([0, 1, 5], [0, 2, 4], (h, w))
    tr = roi_traces(stack, [ix, []])
    flat = stack.reshape(h * w, n, order="F").astype(np.float64)
    np.testing.assert_allclose(tr[:, 0], flat[ix - 1].mean(axis=0))
    assert np.all(np.isnan(tr[:, 1]))
    rois = [MatStruct(indices=ix.astype(float)[:, None], name="a")]
    out = roi_activity(stack, rois)
    assert out[0]["activity"].shape == (n, 1) and out[0]["name"] == "a"
    assert "activity" not in rois[0]
    with pytest.raises(IndexError):
        roi_traces(stack, [[h * w + 1]])
    assert roi_traces(stack.astype(np.float32), [ix]).dtype == np.float32


def test_np_activity_and_masks(stack):
    h, w, _ = stack.shape
    npst = MatStruct(indices=_lin([0], [0], (h, w)), indices_middle=_lin([1, 2], [1, 1], (h, w)))
    out = np_activity(stack, npst)
    np.testing.assert_allclose(out["activity"][:, 0], stack[0, 0].astype(float))
    np.testing.assert_allclose(out["activity_middle"][:, 0], stack[1:3, 1].astype(float).mean(axis=0))
    m = rois_to_mask(_lin([2, 3], [1, 4], (h, w)), (h, w))
    assert m.dtype == bool and m.sum() == 2 and m[2, 1] and m[3, 4]
    nps = neuropil_indices(h * w, [MatStruct(indices=[1, 2]), MatStruct(indices=[5])])
    assert nps.shape == (h * w - 3, 1) and nps[0, 0] == 3 and 5 not in nps


def test_calc_act(stack):
    h, w, n = stack.shape
    sec = (stack.astype(np.int32) * 2).astype(np.int16)
    rois = [[MatStruct(indices=_lin([0, 1], [0, 0], (h, w)))]]
    res = calc_act([stack], rois, sec_data=[sec], calc_on_ch=2)
    np.testing.assert_allclose(res.ROIs[0][0]["activity"][:, 0], sec[0:2, 0].astype(float).mean(axis=0))
    npst = res.np[0][0]
    assert npst["indices"].shape == (h * w - 2, 1)
    flat = stack.reshape(h * w, n, order="F").astype(float)
    np.testing.assert_allclose(npst["activity"][:, 0], flat[2:].mean(axis=0))  # main channel
    p = res.projections
    np.testing.assert_allclose(p["mean"][0], stack.astype(float).mean(axis=2))
    assert p["max"][0].dtype == np.int16 and p["median"][0].dtype == np.int16
    np.testing.assert_allclose(p["std"][0], stack.astype(float).std(axis=2, ddof=1))
    skip = calc_act([stack], rois, recalc_rois=False)
    assert "activity" not in skip.ROIs[0][0]
    empty = calc_act([stack], [[MatStruct()]])
    assert empty.np[0] == []


def test_stack_projections_median_rounding():
    d = np.array([1, 2, 3, 4], dtype=np.int16).reshape(1, 1, 4)
    assert stack_projections(d)["median"][0, 0] == 3  # 2.5 rounds away from zero


def test_calc_act_map_formula():
    rng = np.random.default_rng(2)
    d = rng.integers(0, 50, size=(3, 4, 27)).astype(np.int16)
    blocks = np.stack([d[:, :, s : s + 10].astype(float).mean(axis=2) for s in (0, 10)], axis=2)
    blocks -= blocks.min()
    np.testing.assert_allclose(calc_act_map(d), blocks.max(axis=2) / (blocks.mean(axis=2) + 200))
    with pytest.raises(ValueError):
        calc_act_map(d[:, :, :10])


def test_tile_piezo_layers():
    d = np.arange(2 * 3 * 9).reshape(2, 3, 9)
    with pytest.warns(UserWarning):
        t = tile_piezo_layers(d, 4)
    assert t.shape == (2, 12, 2)
    np.testing.assert_array_equal(t[:, 3:6, 1], d[:, :, 5])


def test_online_response_recovers_response():
    h, w, nt = 4, 3, 4 * 60
    td = np.full((h, w, nt), 100, dtype=np.int16)
    trig_tile = 30  # 0-based tiled frame
    td[:, :, 4 * (trig_tile + 1) :] = 110
    frame_times = 10 * (np.arange(nt) + 1)
    aux = np.zeros((10, frame_times[-1] + 10))
    aux[8, frame_times[4 * (trig_tile + 1)] - 1 :] = 5
    res = online_response(td, aux, frame_times, n_frames_avg=5)
    assert res.trigs.tolist() == [trig_tile]
    np.testing.assert_allclose(res.response_map, 0.1)
    assert res.data.shape == (h, 4 * w, 60)
    assert isinstance(plot_online_response(res), Figure)
    with pytest.raises(ValueError):
        online_response(td, aux * 0, frame_times)


def test_compare_stacks():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(20, 30))
    r1, r2, rgb = compare_stacks_images(a, 2 * a + 1)
    np.testing.assert_allclose(r1, r2)
    assert rgb.shape == (20, 30, 3)
    base = ndimage.gaussian_filter(rng.normal(size=(40, 40)), 2) * 100 + 500
    data = np.repeat(base[:, :, None], 90, axis=2)
    t = stack_template([data, data])
    assert t.shape == (40, 80)
    np.testing.assert_allclose(t[:, :40], base)
    assert isinstance(plot_compare_stacks(a, a, ("x", "y")), Figure)


# --- quality --------------------------------------------------------------------------------


def _rois(shape, centers):
    return [MatStruct(indices=_lin([r, r + 1], [c, c], shape).astype(float)[:, None]) for r, c in centers]


def test_roi_positions_overwrite_across_layers():
    shape = (10, 10)
    l1 = _rois(shape, [(0, 0), (4, 4)])
    l2 = _rois(shape, [(8, 8)])
    pos = roi_positions([l1, l2], [shape, shape])
    np.testing.assert_allclose(pos[:, 0], [9.5, 9])  # layer 2 overwrote column 0 (1-based)
    np.testing.assert_allclose(pos[:, 1], [5.5, 5])


def test_roi_similarity_and_tpi_adata_check(stack):
    shape = stack.shape[:2]
    rois = roi_activity(stack, _rois(shape, [(0, 0), (3, 2), (4, 4)]))
    assert roi_similarity([rois], [shape], [rois], [shape]) == pytest.approx(1)
    chk = tpi_adata_check([rois], [stack], is_site=True)
    assert chk["passed"] == 1.0 and chk["act_similarity"].shape == (1, 1)
    assert chk["act_similarity"][0, 0] == pytest.approx(1, abs=1e-6)
    chk2 = tpi_adata_check([rois], [stack], site_rois=[rois], site_template_shapes=[shape])
    assert chk2["roi_similarity"] == pytest.approx(1)
    other = roi_activity(stack, _rois(shape, [(0, 0)]))
    with pytest.raises(ValueError):
        roi_similarity([rois], [shape], [other], [shape])
    mt = tpi_mean_roi_traces([stack], [rois])
    assert mt[0].dtype == np.float32
    assert isinstance(plot_tpi_activity_similarity([rois], mt), Figure)


def test_tpi_adata_incomplete_and_flags(stack):
    shape = stack.shape[:2]
    rois = roi_activity(stack, _rois(shape, [(0, 0)]))
    good = {"ROIs": [rois], "template": [np.ones(shape)], "act_map": [np.ones(shape)],
            "np": [[MatStruct(indices=[1])]], "projections": MatStruct(max=[np.ones(shape)])}
    assert tpi_adata_incomplete(good) is None
    assert "Activity" in tpi_adata_incomplete({**good, "ROIs": [_rois(shape, [(0, 0)])]})
    assert "template" in tpi_adata_incomplete({**good, "template": [np.zeros(shape)]})
    assert "np" in tpi_adata_incomplete({**good, "np": [[MatStruct()]]})
    assert "Projections" in tpi_adata_incomplete({**good, "projections": None})
    assert tpi_inspection_flags({"act_similarity": [[0.5, 1]], "roi_similarity": 0.99}) == (True, False)
    assert tpi_inspection_flags({"act_similarity": [[1]], "roi_similarity": 0.5}) == (False, True)


def test_registration_failed():
    ok = {"registration_parameters": MatStruct(status="OK"), "x": 1}
    assert not registration_failed(ok)
    assert registration_failed({**ok, "registration_parameters_ch2": MatStruct(status="FAILED")})


def test_site_helpers():
    assert site_grid_shape(2) == (1, 2) and site_grid_shape(5) == (2, 3) and site_grid_shape(10) == (3, 4)
    shape = (6, 5)
    rois = [MatStruct(indices=[1, 2]), MatStruct(indices=[2, 7])]
    lab = site_roi_label_image(rois, shape)
    assert lab[0, 0] == 1 and lab[1, 0] == 2 and lab[0, 1] == 2
    img = np.arange(30.0).reshape(shape)
    sq = _lin(*np.meshgrid(range(1, 4), range(1, 4), indexing="ij"), shape).ravel()
    rgb = roi_perimeter_image([MatStruct(indices=sq)], img)
    assert rgb.shape == (6, 5, 3)
    assert tuple(rgb[1, 1]) == (1.0, 0.0, 0.0) and rgb[2, 2, 1] == rgb[2, 2, 0]  # centre not outline
    assert isinstance(plot_site_rois([img, img], [rois, rois], labels=["a", "b"], title="t"), Figure)


def test_align_site_first_frames():
    a = np.arange(12.0).reshape(3, 4)
    out = align_site_first_frames(a, [a, None, a], [np.eye(3), None, []])
    assert out.shape == (3, 4, 4)
    np.testing.assert_allclose(out[:, :, 1], a)
    np.testing.assert_array_equal(out[:, :, 2], 0)
    np.testing.assert_allclose(out[:, :, 3], a)


# --- aux ------------------------------------------------------------------------------------


def test_get_iframe_times():
    sh = np.zeros(200)
    sh[20:80] = 5
    sh[120:190] = 5
    im = np.zeros((4, 5))
    im[1] = [100, 110, 120, 500, 530]
    out = get_iframe_times(im, sh, [3, 2])
    # first sample above 2.5: 1-based 21 (and 121); MATLAB adds one more
    assert out.shutter_open.tolist() == [[22, 122]]
    np.testing.assert_allclose(out.iframe_times, [[22, 32, 42, 122, 152]])
    with pytest.raises(ValueError):
        get_iframe_times(im, sh, [5])


@pytest.mark.filterwarnings("ignore:get_frame_times")
def test_aux_frame_time_stats_and_cam():
    tr = np.zeros(1000)
    for s in range(50, 950, 40):
        tr[s : s + 20] = 5
    st = aux_frame_time_stats(tr)
    assert st.mean == pytest.approx(40) and st.n_outliers == 0
    cam = np.zeros(400)
    cam[10::20] = 5
    shutter = np.zeros(400)
    shutter[100:300] = 5
    iv, n = cam_exposure_check(cam, shutter)
    np.testing.assert_allclose(iv, 2.0)
    assert n == 9
    ach = {"auxrec": {"scanrate": "1000", "channel0": "Shutter", "channel1": "FrameGalvo"}}
    assert aux_channel_names(ach) == ["Shutter", "FrameGalvo"]


# --- score sheets --------------------------------------------------------------------------


def test_body_weight_score_and_rows():
    assert body_weight_score(20, 20, np.nan) == 0
    assert body_weight_score(18.9, 20, 19) == 1
    assert body_weight_score(17.5, 20, 19) == 2
    assert body_weight_score(15, 20, 19) == 3
    obs = {"Selfmutliation": 1, "Lackofgrooming": 1, "Hunchedposture": 2, "Orbitaltightening": 1,
           "Notresponsivetotouch": 0, "Swelling": 1, "Minorbleeding": 0,
           "Infectionofmajorbleeding": 0, "BodyWeight": 19.0}
    rows = score_sheet_row_values(obs, 20, 19.5)
    assert rows[5] == 3 and rows[7] == 0.5 and rows[8] == 0 and rows[15] == 1  # exactly -5 % scores 1
    assert rows[21] == -5 and rows[22] == -3


def test_excel_column_name():
    assert [excel_column_name(c) for c in (1, 26, 27, 52, 702, 703, 16384)] == [
        "A", "Z", "AA", "AZ", "ZZ", "AAA", "XFD"]
    with pytest.raises(ValueError):
        excel_column_name(0)


def test_score_sheet_problem():
    d0 = datetime.date(2024, 1, 1)
    obs = [(d0 + datetime.timedelta(days=k), 1.0) for k in range(1, 4)]
    now = datetime.datetime(2024, 1, 4, 12)
    assert not score_sheet_problem(d0, obs, now)
    assert score_sheet_problem(d0, obs, datetime.datetime(2024, 1, 6))
    zeros = [(d0 + datetime.timedelta(days=k), 0.0) for k in range(1, 5)]
    assert not score_sheet_problem(d0, zeros, datetime.datetime(2024, 1, 10))  # weekly mode
    # kept MATLAB bug: a 5-day gap between observations is not flagged
    gap = [(d0 + datetime.timedelta(days=1), 1.0), (d0 + datetime.timedelta(days=6), 1.0)]
    assert not score_sheet_problem(d0, gap, datetime.datetime(2024, 1, 7, 12))
    assert observation_too_late(2, [1, 0, 0]) and not observation_too_late(5, [0, 0, 0])
