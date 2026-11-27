"""Synthetic tests for calliopy.rois (port of calliope 08_RoiQuest). No lab data needed."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.ndimage
from matplotlib.figure import Figure

from calliopy import _matlab as ml
from calliopy import rois as rq
from calliopy.io import MatStruct, load_mat, save_mat
from calliopy.rois import _builtins as bi
from calliopy.rois._matching import _num2str


def lin1(rows, cols, hw):
    """0-based (rows, cols) -> MATLAB 1-based column-major linear indices, (N, 1) float."""
    r = np.atleast_1d(rows)
    c = np.atleast_1d(cols)
    return (r + c * hw[0] + 1).astype(float).reshape(-1, 1)


def smooth_noise(shape, seed=0, sigma=2.0):
    rng = np.random.default_rng(seed)
    return scipy.ndimage.gaussian_filter(rng.standard_normal(shape), sigma) * 100 + 500


# --------------------------------------------------------------------------- builtins
def test_fspecial_disk_known_values():
    h1 = bi.fspecial_disk(1)
    exp1 = np.array([[0.0251, 0.1453, 0.0251], [0.1453, 0.3183, 0.1453],
                     [0.0251, 0.1453, 0.0251]])
    np.testing.assert_allclose(h1, exp1, atol=5e-5)
    # radius 2: values recalled from MATLAB output, not a sourced reference (see ref test)
    h2 = bi.fspecial_disk(2)
    exp2_row0 = [0, 0.0170, 0.0381, 0.0170, 0]
    exp2_row1 = [0.0170, 0.0784, 0.0796, 0.0784, 0.0170]
    np.testing.assert_allclose(h2[0], exp2_row0, atol=5e-5)
    np.testing.assert_allclose(h2[1], exp2_row1, atol=5e-5)
    for r in (1, 2.5, 7, 26):
        h = bi.fspecial_disk(r)
        assert h.shape == (2 * int(np.ceil(r - 0.5)) + 1,) * 2
        assert h.sum() == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(h, h.T, atol=1e-15)
        np.testing.assert_allclose(h, h[::-1, ::-1], atol=1e-15)


def test_fspecial_disk_close_to_pixel_disk_overlap_area():
    # independent route: supersampled area of each pixel inside the disk
    r = 7
    n = 64
    crad = int(np.ceil(r - 0.5))
    sub = (np.arange(n) + 0.5) / n - 0.5
    area = np.zeros((2 * crad + 1,) * 2)
    for i, y in enumerate(range(-crad, crad + 1)):
        for j, x in enumerate(range(-crad, crad + 1)):
            yy, xx = np.meshgrid(y + sub, x + sub, indexing="ij")
            area[i, j] = np.mean(xx**2 + yy**2 <= r**2)
    area[crad, crad] = 1
    np.testing.assert_allclose(bi.fspecial_disk(r), area / area.sum(), atol=2e-4)


def test_fspecial_disk_rejects_non_positive():
    with pytest.raises(ValueError):
        bi.fspecial_disk(0)


def test_mat2gray_and_quantile():
    a = np.array([[0.0, 5.0], [10.0, np.nan]])
    np.testing.assert_allclose(bi.mat2gray(a), [[0, 0.5], [1, np.nan]], equal_nan=True)
    np.testing.assert_allclose(bi.mat2gray(a, [2, 4]), [[0, 1], [1, np.nan]], equal_nan=True)
    np.testing.assert_allclose(bi.mat2gray(np.full((2, 2), 0.3), [0.5, 0.5]), 0.3)
    x = np.arange(1.0, 11.0)
    assert bi.quantile(x, 0.5) == pytest.approx(np.percentile(x, 50, method="hazen"))
    assert bi.quantile(x, 0.995) == pytest.approx(10.0)


def test_strel_disk_small_radii_and_tophat():
    np.testing.assert_array_equal(bi.strel_disk(1), [[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    d2 = bi.strel_disk(2)
    assert d2.sum() == 13 and d2[2].all() and not d2[0, 1]
    img = np.full((15, 15), 10.0)
    img[7, 7] = 30.0
    th = bi.imtophat(img, bi.strel_disk(2))
    assert th[7, 7] == pytest.approx(20.0)
    assert np.count_nonzero(th) == 1


def test_bwareaopen_8_connectivity():
    m = np.zeros((6, 6), bool)
    m[0, 0] = m[1, 1] = True  # diagonal pair: one 8-connected object of 2
    m[4, 4] = True
    out = bi.bwareaopen(m, 2)
    assert out[0, 0] and out[1, 1] and not out[4, 4]


def test_bwmorph_spur_documentation_example():
    bw = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [1, 1, 0, 0]], bool)
    exp = bw.copy()
    exp[2, 2] = False
    np.testing.assert_array_equal(bi.bwmorph_spur(bw), exp)
    iso = np.zeros((3, 3), bool)
    iso[1, 1] = True
    np.testing.assert_array_equal(bi.bwmorph_spur(iso), iso)


def test_bwmorph_thin_bar_and_idempotence():
    bw = np.zeros((9, 20), bool)
    bw[3:6, 2:18] = True
    th = bi.bwmorph_thin(bw, np.inf)
    assert (th.sum(axis=0)[4:16] == 1).all()  # one pixel thick in the middle
    assert (th <= bw).all()
    np.testing.assert_array_equal(bi.bwmorph_thin(th, 1), th)
    _, n = scipy.ndimage.label(th, np.ones((3, 3)))
    assert n == 1


def test_bwmorph_thicken_grows_without_merging():
    bw = np.zeros((20, 20), bool)
    bw[4:7, 3:6] = True
    bw[4:7, 9:12] = True
    once = bi.bwmorph_thicken(bw, 1)
    assert once.sum() > bw.sum() and (once >= bw).all()
    many = bi.bwmorph_thicken(bw, 10)
    _, n = scipy.ndimage.label(many, np.ones((3, 3)))
    assert n == 2


# --------------------------------------------------------------------------- Roi
def test_roi_mask_uses_column_major_one_based_indices():
    hw = (3, 4)
    r = rq.Roi([2.0], hw, rng=np.random.default_rng(0))
    m = r.to_binary_mask()
    assert m.shape == hw and m[1, 0] and m.sum() == 1
    assert 2 <= r.rand_color_idx <= 11
    assert r.indices.shape == (1, 1)


def test_roi_move_drops_pixels_and_accumulates_shift():
    hw = (5, 7)
    idx = lin1([0, 4, 2], [0, 6, 3], hw)
    r = rq.Roi(idx, hw)
    r.move(1, 0)
    # (0,0)->(1,0); (4,6) leaves the image; (2,3)->(3,3)
    np.testing.assert_array_equal(r.indices, lin1([1, 3], [0, 3], hw))
    np.testing.assert_array_equal(r.shift, [[1, 0]])
    r.move(0, -1, global_xy=(1, 0))
    # (1,0) -> (2,-1) leaves the image; (3,3) -> (4,2)
    np.testing.assert_array_equal(r.indices, lin1([4], [2], hw))
    np.testing.assert_array_equal(r.shift, [[1, -1]])  # global part not stored
    row = rq.Roi(lin1([1, 2], [1, 1], hw).T, hw)
    row.move(0, 1)
    assert row.indices.shape == (1, 2)


def test_roi_label_images_and_type_quirk():
    hw = (4, 4)
    a = rq.Roi(lin1([0], [0], hw), hw, type=3)
    b = rq.Roi(lin1([1], [1], hw), hw, type=None)
    assert a.to_label_type_image()[0, 0] == 4
    assert b.to_label_type_image()[1, 1] == 13  # single ROI, empty type
    stack = rq.rois_label_type_image([a, b])
    assert stack.shape == (4, 4, 2) and stack.dtype == np.uint8
    assert stack[0, 0, 0] == 4 and stack[1, 1, 1] == 12  # array: empty type -> 12
    lab = rq.rois_label_image([a, b])
    assert lab[0, 0, 0] == a.rand_color_idx and lab[1, 1, 1] == b.rand_color_idx
    m = rq.rois_binary_mask([a, b])
    assert m.sum() == 2 and m[0, 0] and m[1, 1]
    with pytest.raises(ValueError):
        rq.rois_binary_mask([])


def test_roi_expand_and_shrink():
    hw = (12, 12)
    rr, cc = np.meshgrid(range(4, 8), range(4, 8), indexing="ij")
    r = rq.Roi(lin1(rr.ravel(), cc.ravel(), hw), hw)
    before = r.to_binary_mask()
    r.expand(1)
    assert (r.to_binary_mask() >= before).all() and r.to_binary_mask().sum() > 16
    s = rq.Roi(lin1(rr.ravel(), cc.ravel(), hw), hw)
    s.shrink(1)
    assert 0 < s.to_binary_mask().sum() < 16 and (s.to_binary_mask() <= before).all()
    assert s.indices.shape[1] == 1


def test_roiwf_struct_equals_flip_and_rank_image():
    hw = (4, 5)
    r = rq.RoiWf(lin1([0, 1], [0, 0], hw), hw, 1, 2.0, 3.0, "A", "area", "acr", 7.0)
    s = r.to_struct()
    assert list(s) == ["indices", "shift", "type", "ML_index", "AP_index", "name",
                       "Allen_area", "Allen_acronym", "Allen_index"]
    assert next(iter(r.to_struct(rename_indices_as="indices_ch525"))) == "indices_ch525"
    other = rq.RoiWf(np.array([[1.0], [2.0]]), hw)
    assert r.equals(other)
    odd = rq.RoiWf(np.array([[0.0], [3.0]]), hw)  # sum of differences is 0 -> "equal"
    assert r.equals(odd)
    r.fliplr()
    np.testing.assert_array_equal(r.indices, lin1([0, 1], [4, 4], hw))
    a = rq.RoiWf(lin1([0], [0], hw), hw, 1, allen_index=50)
    b = rq.RoiWf(lin1([3], [4], hw), hw, 1, allen_index=9)
    img = rq.rois_label_type_image([a, b])
    assert img[0, 0] == 2 and img[3, 4] == 1 and img[1, 1] == 0
    c = rq.RoiWf(lin1([0], [0], hw), hw, 1, allen_index=None)
    assert c.to_label_type_image().max() == 0


def test_roiwf_channel_transforms_with_translation():
    hw = (10, 12)
    a = np.array([[1, 0, 2], [0, 1, 0], [0, 0, 1]], float)  # x -> x + 2
    rr, cc = np.meshgrid(range(3, 5), range(5, 8), indexing="ij")
    roi = rq.RoiWf(lin1(rr.ravel(), cc.ravel(), hw), hw)
    roi.transform_green2red(a)
    # warp with inv(A): content moves 2 columns left, then flipped
    exp = np.zeros(hw, bool)
    exp[3:5, 3:6] = True
    np.testing.assert_array_equal(roi.to_binary_mask(), exp[:, ::-1])
    pt = rq.RoiWf(lin1([3], [6], hw), hw)
    pt.transform_green2red(a)
    np.testing.assert_array_equal(pt.indices, lin1([3], [hw[1] - 1 - 4], hw))
    # red2green: point branch uses the forward transform, mask branch the inverse
    pt2 = rq.RoiWf(lin1([3], [6], hw), hw)
    pt2.transform_red2green(a)
    np.testing.assert_array_equal(pt2.indices, lin1([3], [hw[1] - 1 - 6 + 2], hw))
    roi2 = rq.RoiWf(lin1(rr.ravel(), cc.ravel(), hw), hw)
    roi2.transform_red2green(a)
    exp2 = np.zeros(hw, bool)
    exp2[3:5, 12 - 8 - 2 : 12 - 5 - 2] = True
    np.testing.assert_array_equal(roi2.to_binary_mask(), exp2)


# --------------------------------------------------------------------------- structs
def test_rois2struct_struct2rois_round_trip_through_mat_file(tmp_path):
    hw = (8, 9)
    layer1 = [rq.Roi(lin1([1, 2], [3, 3], hw), hw, type=2), rq.Roi(lin1([5], [5], hw), hw)]
    layer1[0].shift = np.array([[1.0, -2.0]])
    layer2 = [rq.Roi(lin1([0], [8], hw), hw, type=1)]
    acute = rq.save_rois_acute([layer1, layer2, []], [np.zeros((3, 1))] * 3)
    assert acute.ROIs.shape == (1, 3)
    assert acute.ROIs[0, 2].shape == (0, 0)
    save_mat(tmp_path / "a.mat", acute)
    back = load_mat(tmp_path / "a.mat")
    rois1 = rq.struct2rois(back.ROIs[0, 0], hw)
    assert len(rois1) == 2
    np.testing.assert_array_equal(rois1[0].indices, layer1[0].indices)
    np.testing.assert_array_equal(rois1[0].shift, [[1, -2]])
    assert float(rois1[0].type) == 2
    rois2 = rq.struct2rois(back.ROIs[0, 1], hw)  # 1x1 struct loads as a MatStruct
    assert len(rois2) == 1 and float(rois2[0].type) == 1
    assert rq.struct2rois(back.ROIs[0, 2], hw) == []


def test_rois2struct_empty_and_activity_removed():
    assert rq.rois2struct([]) == MatStruct() and len(rq.rois2struct([])) == 0
    s = rq.rois2struct([MatStruct(indices=np.ones((1, 1)), shift=np.zeros((1, 2)), type=0.0,
                                  activity=np.ones((5, 1)))])
    assert "activity" not in s[0, 0]


def test_struct2rois_defaults_and_activity():
    s = [MatStruct(indices=np.array([[1.0], [2.0]]), activity=np.arange(3.0))]
    r = rq.struct2rois(s, (4, 4))[0]
    assert r.type == 0 and np.array_equal(r.shift, [[0, 0]])
    np.testing.assert_array_equal(r.activity, np.arange(3.0))
    assert rq.struct2rois(MatStruct(), (4, 4)) == []


def wf_struct(idx, **extra):
    s = MatStruct(indices=np.asarray(idx, float), shift=np.zeros((1, 2)), type=0.0,
                  ML_index=1.0, AP_index=2.0, name="r", Allen_area="a", Allen_acronym="b",
                  Allen_index=np.empty((0, 0)))
    s.update(extra)
    return s


def test_struct2wf_rois_and_2ch():
    hw = (6, 6)
    s = wf_struct([[3.0]], activity=np.ones((4, 1)))
    rois = rq.struct2wf_rois([s], hw)
    assert isinstance(rois[0], rq.RoiWf) and rois[0].allen_index is None
    assert rois[0].ml_index == 1.0 and rois[0].activity.shape == (4, 1)
    s2 = MatStruct(indices_ch525=np.array([[1.0]]), indices_ch610=np.array([[5.0]]),
                   type=0.0, ML_index=1.0, AP_index=2.0, name="r", Allen_area="a",
                   Allen_acronym="b", Allen_index=4.0, activity_ch610=np.zeros((2, 1)))
    g, r = rq.struct2wf_rois_2ch(np.array([[s2]], dtype=object), hw)
    assert g[0].indices[0, 0] == 1 and r[0].indices[0, 0] == 5
    assert g[0].activity is None and r[0].activity.shape == (2, 1)


def test_move_roi_structs():
    hw = (5, 5)
    s = MatStruct(indices=lin1([0, 4], [0, 0], hw), shift=np.array([[1.0, 1.0]]), type=0.0)
    out = rq.move_roi_structs([s], 1, 2, [1, 0], hw)
    np.testing.assert_array_equal(out[0, 0]["indices"], lin1([2], [2], hw))
    np.testing.assert_array_equal(out[0, 0]["shift"], [[2, 3]])
    np.testing.assert_array_equal(s["indices"], lin1([0, 4], [0, 0], hw))


# --------------------------------------------------------------------------- images
def test_filter_template_legacy_against_padded_filter2():
    img = smooth_noise((40, 37))
    out = rq.filter_template(img, 3)
    k = bi.fspecial_disk(3)
    pad = k.shape[0] // 2
    den = ml.filter2(k, np.pad(img, pad, mode="edge"), "valid")
    np.testing.assert_allclose(out, img / den, rtol=1e-12)
    np.testing.assert_allclose(rq.filter_template(np.full((20, 20), 7.0), 4), 1.0)
    f32 = rq.filter_template(img.astype(np.float32), 3)
    assert f32.dtype == np.float32
    np.testing.assert_allclose(f32, out, rtol=1e-5)
    u16 = rq.filter_template(ml.matlab_cast(img, np.uint16), 3)
    assert u16.dtype == np.uint16 and set(np.unique(u16)) <= {0, 1, 2}


def test_filter_template_updated_and_errors():
    img = np.full((30, 30), 100.0)
    img[10:13, 10:13] = 200.0
    out = rq.filter_template(img, 2, "updated")
    assert out.shape == img.shape and np.nanmax(out) > 1
    with pytest.raises(ValueError):
        rq.filter_template(img, 2, "other")


def test_default_filter_size():
    assert rq.default_filter_size((512, 796)) == 26
    assert rq.default_filter_size((30, 50)) == 2  # round(1.5) = 2, half away from zero


def test_adjust_img_contrast_modes():
    img = np.linspace(0, 1, 101).reshape(1, -1) * 10
    mm = rq.adjust_img_contrast(img, "minMax")
    assert mm.dtype == np.uint8 and mm[0, 0] == 0 and mm[0, -1] == 255
    assert mm[0, 50] == 128  # 127.5 rounds half away from zero
    bf = rq.adjust_img_contrast(img)
    lo, hi = np.percentile(img.ravel(), [0.5, 99.5], method="hazen")
    exp = ml.matlab_cast(np.clip((img - lo) / (hi - lo), 0, 1) * 255, np.uint8)
    assert np.abs(bf.astype(int) - exp.astype(int)).max() <= 1
    cu = rq.adjust_img_contrast(img, "custom", (0.1, 0.5))
    # limits: m + range*0.1 = 1, M*0.5 = 5 (MATLAB's upper limit, kept)
    assert cu[0, 10] == 0 and cu[0, 50] == 255 and cu[0, 30] == 128
    with pytest.warns(UserWarning):
        c2 = rq.adjust_img_contrast(img, "custom", (0.6, 0.2))
    np.testing.assert_array_equal(c2, mm)
    nan = img.copy()
    nan[0, 3] = np.nan
    assert rq.adjust_img_contrast(nan, "minMax")[0, 3] == 0
    with pytest.raises(ValueError):
        rq.adjust_img_contrast(img, "custom", (0, 2))
    with pytest.raises(ValueError):
        rq.adjust_img_contrast(img, "auto")


def test_labels_colormap_and_uniform_rgb():
    cm = rq.get_labels_colormap()
    assert cm.shape == (12, 3)
    np.testing.assert_allclose(cm[0], 0)
    np.testing.assert_allclose(cm[1], [1, 0, 0])
    np.testing.assert_allclose(cm[2], [1, 0.6, 0], atol=1e-12)  # MATLAB hsv(10)(2,:)
    np.testing.assert_allclose(cm[-1], [0.22, 0.9, 0.19])
    rgb = rq.uniform_rgb_image((3, 4))
    assert rgb.shape == (3, 4, 3)
    np.testing.assert_allclose(rgb[2, 3], [0.93, 0.13, 0.55])


def test_make_cross():
    hw = (30, 40)
    idx = rq.make_cross((10, 20), hw, 10)
    m = np.zeros(hw, bool)
    m.T.flat[idx.ravel().astype(int) - 1] = True
    assert idx.shape == (85, 1) and m.sum() == 85
    assert m[5:16, 18:23].all() and m[8:13, 15:26].all()
    edge = rq.make_cross((0, 0), hw, 10)
    assert edge.size == 6 * 3 + 3 * 6 - 9
    assert np.all(np.diff(idx.ravel()) > 0)
    with pytest.raises(ValueError):
        rq.make_cross((1.5, 2), hw)


def test_segment_neuron():
    img = np.zeros((20, 20), np.uint8)
    img[2:8, 2:8] = 200   # 36 px
    img[12, 12] = 200     # 1 px, removed by bwareaopen
    img[2:8, 14:19] = 200  # outside crop
    m = rq.segment_neuron(img, 160, (0, 15, 0, 13), min_roi_pix=20)
    assert m.sum() == 36 and m[2:8, 2:8].all()
    valid = np.ones((20, 20), bool)
    valid[:3] = False
    existing = np.zeros((20, 20), bool)
    existing[7] = True
    m2 = rq.segment_neuron(img, 160, (0, 20, 0, 20), min_roi_pix=20, valid_area=valid,
                           existing_mask=existing)
    assert not m2[:3].any() and not m2[7].any() and m2[3:7, 2:8].all()
    m3 = rq.segment_neuron(img, 160, (0, 20, 0, 20), min_roi_pix=20, close_and_despur=True)
    assert m3[2:8, 2:8].all()


def test_valid_area_uses_first_layer_maxima():
    dx = [np.array([1.0, -2.0, np.nan]), np.array([5.0])]
    dy = [np.array([0.0, 3.0]), np.array([0.0])]
    va = rq.valid_area(dx, dy, (10, 12))
    assert va.shape == (10, 12, 2)
    np.testing.assert_array_equal(va[:, :, 0], va[:, :, 1])  # MATLAB quirk kept
    assert va[2:8, 3:9, 0].all() and va[:, :, 0].sum() == 6 * 6
    none = rq.valid_area([np.array([np.nan])], [np.array([0.0])], (5, 5))
    assert not none.any()


def test_filter_cellpose_labels():
    lab = np.zeros((10, 10), int)
    lab[0:5, 0:5] = 1     # touches invalid border
    lab[6:9, 6:9] = 2     # 9 px, overlaps existing
    lab[1:4, 6:9] = 3     # 9 px ok when min 5
    lab[8, 0] = 5         # label 4 missing -> removed as too small
    valid = np.ones((10, 10), bool)
    valid[0] = False
    existing = np.zeros((10, 10), bool)
    existing[7, 7] = True
    res = rq.filter_cellpose_labels(lab, valid, existing, 5)
    assert res.num_detected == 5 and res.num_removed == 4 and len(res.indices) == 1
    expected = np.flatnonzero((lab == 3).ravel(order="F")) + 1.0
    np.testing.assert_array_equal(res.indices[0], expected.reshape(-1, 1))


def test_actiquest_activity_traces_and_plot():
    acts = [np.array([[1.0], [2.0], [3.0]]), np.array([[2.0], [2.0], [4.0]])]
    a = rq.actiquest_activity(acts)
    np.testing.assert_allclose(a, [[-0.5, 0, 0.5], [0, 0, 1]])
    np.testing.assert_allclose(rq.actiquest_traces(a, 2), a / 2.5 + [[1], [2]])
    fig = rq.plot_actiquest(a, roi_numbers=[4, 7], rand_color_idx=[2, 3])
    assert isinstance(fig, Figure)


# --------------------------------------------------------------------------- matching
def test_check_roi_matching_cases():
    hw = (20, 20)
    o = [MatStruct(indices=lin1([5, 5, 6], [5, 6, 5], hw), shift=np.zeros((1, 2)), type=1.0),
         MatStruct(indices=lin1([10], [10], hw), shift=np.zeros((1, 2)), type=0.0)]
    t = [MatStruct(indices=lin1([7, 7, 8], [4, 5, 4], hw), shift=np.array([[0.0, -1.0]]),
                   type=1.0),
         MatStruct(indices=lin1([12], [10], hw), shift=np.zeros((1, 2)), type=2.0)]
    res = rq.check_roi_matching([o], [t], [np.array([[2.0], [0.0], [0.0]])], hw)
    assert not res.matched
    np.testing.assert_array_equal(res.is_overlapping[0], [[True, True]])
    np.testing.assert_array_equal(res.is_type_matched[0], [[True, False]])
    t[1]["type"] = 0.0
    assert rq.check_roi_matching([o], [t], [np.array([2.0, 0, 0])], hw).matched
    short = rq.check_roi_matching([o], [t[:1]], [np.array([2.0, 0, 0])], hw)
    np.testing.assert_array_equal(short.is_overlapping[0], [[True, False]])
    far = rq.check_roi_matching([o], [t], [np.array([30.0, 0, 0])], hw)  # outside: no crash
    assert not far.matched
    assert not rq.check_roi_matching([o, o], [t], [np.zeros(3)], hw).matched


def test_roi_match_message_and_num2str():
    assert _num2str([1, 2, 3]) == "1  2  3"
    assert _num2str([1, 10]) == "1  10"
    msg = rq.roi_match_message([np.array([[True, False, False]]), np.array([[True]])],
                               [np.array([[True, True, True]]), np.array([[False]])])
    assert msg.startswith("ERROR. ROI mismatch detected!\n")
    assert "Z-Layer(s) [1] fail(s) ROI overlap criterion." in msg
    assert "ROIs(s) [2  3]." in msg
    assert "Z-Layer(s) 2 fail(s) ROI type matching." in msg
    assert "\t\t- ROIs(s) [1]." in msg


def make_chronic_images(shift=(3, -2)):
    ref = smooth_noise((128, 128), seed=1)
    cur = np.roll(ref, shift, axis=(0, 1))
    return ref, cur


def test_auto_adjust_and_auto_roi_matching():
    ref, cur = make_chronic_images()
    hw = ref.shape
    r0 = rq.Roi(lin1([60, 61, 60], [60, 60, 61], hw), hw)
    r1 = rq.Roi(lin1([30], [90], hw), hw)
    res = rq.auto_adjust_rois([r0, r1], cur, ref)
    np.testing.assert_array_equal(res.global_dxdy, [[3, -2]])
    np.testing.assert_array_equal(res.dxdy, 0)
    np.testing.assert_array_equal(res.displacement, [[3, -2], [3, -2]])
    np.testing.assert_allclose(res.start_centroids[1], [31, 91])
    assert not res.is_out_of_bounds.any()
    orig = [r0.to_struct(), r1.to_struct()]
    out = rq.auto_roi_matching([r0, r1], cur, ref, [True, True])
    np.testing.assert_array_equal(out.roi_trans, [[3, -2]])
    np.testing.assert_array_equal(r1.indices, lin1([33], [88], hw))
    np.testing.assert_array_equal(r1.shift, [[0, 0]])
    assert not r0.selected
    chk = rq.check_roi_matching([orig], [[r0, r1]], [np.array([3.0, -2.0, 0])], hw)
    assert chk.matched
    with pytest.raises(ValueError):
        rq.auto_roi_matching([r0], cur, ref, [])


def test_save_rois_acute_activity_check():
    hw = (5, 5)
    old = np.empty((1, 1), dtype=object)
    old[0, 0] = MatStruct(indices=np.ones((1, 1)), activity=np.ones((3, 1)))
    with pytest.raises(rq.RoiSaveError):
        rq.save_rois_acute([[rq.Roi([[1.0]], hw)]], [np.zeros(3)], old_rois=old)
    ok = rq.save_rois_acute([[rq.Roi([[1.0]], hw)]], [np.zeros(3)], old_rois=old,
                            overwrite_activity=True)
    assert ok.ROIs.shape == (1, 1)


def test_save_rois_chronic_with_new_roi():
    ref, cur = make_chronic_images()
    hw = ref.shape
    orig_structs = [MatStruct(indices=lin1([60, 61], [60, 60], hw), shift=np.zeros((1, 2)),
                              type=0.0),
                    MatStruct(indices=lin1([30], [90], hw), shift=np.zeros((1, 2)), type=0.0)]
    orig_cell = np.empty((1, 1), dtype=object)
    orig_cell[0, 0] = np.array([orig_structs], dtype=object)
    cur_rois = rq.struct2rois(orig_cell[0, 0], hw)
    for r in cur_rois:
        r.move(0, 0, (3, -2))
    new = rq.Roi(lin1([90, 91], [40, 40], hw), hw)
    cur_rois.append(new)
    roi_trans = [np.array([3.0, -2.0, 0.0])]
    tps = {10: {"ROIs": orig_cell, "ROItrans": [np.zeros(3)], "template": [ref]},
           11: None, 12: {"ROIs": [[]], "ROItrans": [np.zeros(3)], "template": [ref]}}
    res = rq.save_rois_chronic([cur_rois], roi_trans, [cur], 13, [10, 11, 12, 13], tps,
                               filter_win_size=3)
    assert res.ROIs[0, 0].shape == (1, 2)
    assert [int(e["expID"]) for e in res.rois_to_add] == [10, 13]
    added_orig = rq.struct2rois(res.updated_rois[10][0, 0], hw)
    assert len(added_orig) == 3
    np.testing.assert_array_equal(added_orig[2].indices, lin1([87, 88], [42, 42], hw))
    np.testing.assert_array_equal(added_orig[2].shift, [[0, 0]])
    added_cur = rq.struct2rois(res.updated_rois[13][0, 0], hw)
    np.testing.assert_array_equal(added_cur[2].indices, new.indices)
    assert res.rois_to_add[0]["titles"][0, 0] == "ORIGINAL ExpID: 13 - Z:1"
    assert res.rois_to_add[1]["titles"][0, 0] == "CURRENT ExpID: 13 - Z:1"
    assert res.rois_to_add[0]["imgs"][0, 0].dtype == np.uint8
    assert isinstance(rq.plot_rois_to_add(res.rois_to_add, hw), Figure)
    # fewer ROIs than the original
    with pytest.raises(rq.RoiSaveError, match="less ROIs"):
        rq.save_rois_chronic([cur_rois[:1]], roi_trans, [cur], 13, [10, 13], tps,
                             filter_win_size=3)
    # mismatch: ROI not translated back correctly
    with pytest.raises(rq.RoiSaveError, match="ROI mismatch"):
        rq.save_rois_chronic([cur_rois[:2]], [np.array([0.0, 0.0, 0.0])], [cur], 13, [10, 13],
                             tps, filter_win_size=3)
    plain = rq.save_rois_chronic([cur_rois[:2]], roi_trans, [cur], 13, [10, 13], tps,
                                 filter_win_size=3)
    assert plain.rois_to_add == [] and plain.updated_rois == {}
    with pytest.raises(rq.RoiSaveError, match="no AData"):
        rq.save_rois_chronic([cur_rois], roi_trans, [cur], 13, [10, 13], {10: None},
                             filter_win_size=3)


def test_exp_group():
    np.testing.assert_array_equal(rq.exp_group(5, [7, 5, 6, 9], [1, 2, 2, 1]), [5, 6])
    with pytest.raises(ValueError):
        rq.exp_group(4, [7, 5], [1, 2])


# --------------------------------------------------------------------------- widefield
def wf_rois(hw):
    return [wf_struct(lin1([2], [3], hw)[:1], name="lambda"),
            wf_struct(lin1([8], [3], hw)[:1], name="bregma"),
            wf_struct(lin1([4, 4, 5], [6, 7, 6], hw), name="V1"),
            wf_struct(lin1([0], [0], hw), name="edge")]


def test_wf_transform_rois_translation():
    hw = (10, 12)
    tform = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], float)  # current -> first: x + 1
    with pytest.warns(UserWarning, match="beyond template"):
        out = rq.wf_transform_rois(np.array([wf_rois(hw)], dtype=object), hw, tform)
    assert out.shape == (1, 4)
    np.testing.assert_array_equal(out[0, 0]["indices"], lin1([2], [2], hw))
    np.testing.assert_array_equal(out[0, 2]["indices"], lin1([4, 5, 4], [5, 5, 6], hw))
    assert out[0, 3]["indices"].shape == (0, 0)
    assert out[0, 2]["name"] == "V1"
    bad = np.array([[1, 0, 9], [0, 1, 0], [0, 0, 1]], float)
    with pytest.raises(ValueError, match="reference points"):
        rq.wf_transform_rois(wf_rois(hw), hw, bad)


def test_wf_transform_rois_2ch_and_struct():
    hw = (10, 12)
    s = []
    for x in wf_rois(hw)[:3]:
        t = MatStruct(indices_ch525=x["indices"], indices_ch610=x["indices"])
        t.update((k, v) for k, v in x.items() if k != "indices")
        s.append(t)
    eye = np.eye(3)
    shift = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]], float)  # y + 1
    out = rq.wf_transform_rois_2ch(s, hw, eye, shift)
    np.testing.assert_array_equal(out[0, 2]["indices_ch525"].ravel(),
                                  np.sort(s[2]["indices_ch525"].ravel()))
    np.testing.assert_array_equal(out[0, 2]["indices_ch610"], lin1([3, 4, 3], [6, 6, 7], hw))
    g = [rq.RoiWf(lin1([1], [1], hw), hw, name="a"),
         rq.RoiWf(lin1([2], [2], hw), hw, name="upper left corner")]
    r = [rq.RoiWf(lin1([3], [3], hw), hw, name="a"),
         rq.RoiWf(lin1([4], [4], hw), hw, name="upper left corner")]
    st = rq.wf_rois_struct(g, r)
    assert st.shape == (1, 1)
    assert list(st[0, 0])[:3] == ["indices_ch525", "indices_ch610", "shift"]
    assert rq.wf_rois_struct(g).shape == (1, 1)


def test_wf_save_rois_policies():
    hw = (10, 12)
    rois = rq.struct2wf_rois(wf_rois(hw)[:3], hw)
    new = rq.wf_rois_struct(rois)
    rp_ok = MatStruct(status="OK", activity_calculated=1.0)
    tform = np.eye(3)
    first_cell = np.empty((1, 1), dtype=object)
    first_cell[0, 0] = new
    tps = {1: {"type": "wf", "registration_parameters": rp_ok, "ROIs": first_cell},
           2: {"type": "wf", "registration_parameters": MatStruct(status="FAILED",
                                                                   activity_calculated=0.0),
               "tform": tform},
           3: {"type": "wf", "registration_parameters": MatStruct(status="OK",
                                                                   activity_calculated=0.0),
               "tform": tform},
           4: None}
    keep = rq.wf_save_rois(rois, exp_group_ids=[1, 2, 3, 4], current_exp_id=1,
                           template_hw=hw, roi_template_parameters=MatStruct(name="t"),
                           timepoints=tps, overwrite_policy="keep")
    assert set(keep) == {3}
    assert keep[3].ROIs.shape == (1, 1) and keep[3].ROItemplate_parameters.name == "t"
    over = rq.wf_save_rois(rois, exp_group_ids=[1, 2, 3, 4], current_exp_id=1,
                           template_hw=hw, roi_template_parameters=MatStruct(name="t"),
                           timepoints=tps, overwrite_policy="overwrite")
    assert set(over) == {1, 3}
    assert over[1].registration_parameters.activity_calculated == 0
    assert tps[1]["registration_parameters"].activity_calculated == 1  # input untouched
    moved = rois[:2] + [rq.RoiWf(lin1([1], [1], hw), hw)]
    with pytest.raises(rq.RoiSaveError, match="different"):
        rq.wf_save_rois(moved, exp_group_ids=[1, 3], current_exp_id=1, template_hw=hw,
                        roi_template_parameters=MatStruct(), timepoints=tps,
                        overwrite_policy="keep")
    with pytest.raises(rq.RoiSaveError, match="not the first"):
        rq.wf_save_rois(rois, exp_group_ids=[1, 3], current_exp_id=3, template_hw=hw,
                        roi_template_parameters=MatStruct(), timepoints=tps,
                        overwrite_policy="keep")
    with pytest.raises(ValueError):
        rq.wf_save_rois(rois, exp_group_ids=[1, 3], current_exp_id=1, template_hw=hw,
                        roi_template_parameters=MatStruct(), timepoints=tps)
    single = rq.wf_save_rois(rois, exp_group_ids=[1], current_exp_id=1, template_hw=hw,
                             roi_template_parameters=MatStruct(), timepoints={})
    assert list(single) == [1]


def test_wf_save_rois_two_channel_reset_names():
    hw = (10, 12)
    base = wf_rois(hw)[:3]
    g = rq.struct2wf_rois(base, hw)
    r = rq.struct2wf_rois(base, hw)
    tps = {1: {"type": "2ch_wf",
               "registration_parameters_ch525": MatStruct(status="OK", activity_calculated=1.0),
               "registration_parameters_ch610": MatStruct(status="OK", activity_calculated=1.0),
               "ROIs": [rq.wf_rois_struct(g, r)]},
           2: {"type": "2ch_wf",
               "registration_parameters_ch525": MatStruct(status="OK", activity_calculated=0.0),
               "registration_parameters_ch610": MatStruct(status="OK", activity_calculated=0.0),
               "tform_ch525": np.eye(3), "tform_ch610": np.eye(3)}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = rq.wf_save_rois(g, rois_ch610=r, exp_group_ids=[1, 2], current_exp_id=1,
                              template_hw=hw, roi_template_parameters=MatStruct(),
                              timepoints=tps, overwrite_policy="overwrite")
    assert "registration_parameters_525" in out[1] and "registration_parameters_610" in out[1]
    assert list(out[2].ROIs[0, 0][0, 0])[:2] == ["indices_ch525", "indices_ch610"]
