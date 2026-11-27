"""Reference tests: calliopy.rois against MATLAB output of calliope r2003 (08_RoiQuest).

The references are written by ``tests/matlab/ref_rois.m`` (``ref_rq_<name>.mat``; inputs
followed by the outputs as ``out1, out2, ...`` in the order given in each block of that
script). All tests are skipped unless ``CALLIOPY_TESTDATA`` holds the reference files.
Tests marked ``xfail`` compare functions documented as approximations of MATLAB (see their
docstrings); a pass there means the approximation is exact for the test input.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from calliopy import rois as rq
from calliopy.rois import _builtins as bi

F64 = {"rtol": 1e-12, "atol": 1e-12}


def check(actual, expected, **tol):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape and actual.size == expected.size and (
            min(actual.squeeze().ndim, expected.squeeze().ndim) <= 1):
        actual = actual.ravel(order="F")
        expected = expected.ravel(order="F")
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual.astype(float), expected.astype(float), equal_nan=True,
                               **(tol or F64))


def same_set(actual, expected):
    np.testing.assert_array_equal(np.sort(np.ravel(actual)), np.sort(np.ravel(expected)))


def hw_of(v):
    return tuple(int(x) for x in np.ravel(v)[:2])


def test_ref_fspecial(matlab_ref):
    r = matlab_ref("rq_fspecial")
    for k, rad in enumerate(np.ravel(r.out1)):
        check(bi.fspecial_disk(rad), r[f"out{k + 2}"])


def test_ref_bwmorph_exactish(matlab_ref):
    r = matlab_ref("rq_bwmorph")
    bw = np.asarray(r.out1, bool)
    check(bi.bwmorph_thin(bw, 1), r.out2, rtol=0, atol=0)
    check(bi.bwmorph_thin(bw, np.inf), r.out3, rtol=0, atol=0)
    check(bi.bwmorph_spur(bw, 1), r.out6, rtol=0, atol=0)
    check(bi.bwareaopen(bw, 5), r.out7, rtol=0, atol=0)
    check(bi.imclose_binary(bw, bi.strel_disk(2)), r.out8, rtol=0, atol=0)
    check(bi.bwperim(bw), r.out9, rtol=0, atol=0)


@pytest.mark.xfail(reason="bwmorph 'thicken' is implemented as thinning of the background; "
                          "MATLAB's implementation is not documented", strict=False)
def test_ref_bwmorph_thicken(matlab_ref):
    r = matlab_ref("rq_bwmorph")
    bw = np.asarray(r.out1, bool)
    check(bi.bwmorph_thicken(bw, 1), r.out4, rtol=0, atol=0)
    check(bi.bwmorph_thicken(bw, 3), r.out5, rtol=0, atol=0)


def test_ref_tophat_exact_disks(matlab_ref):
    r = matlab_ref("rq_tophat")
    check(bi.imtophat(r.out1, bi.strel_disk(2)), r.out2)
    check(bi.imtophat(r.out1, bi.strel_disk(5)), r.out4)


@pytest.mark.xfail(reason="strel('disk', 5) with N = 4 is a periodic-line approximation",
                   strict=False)
def test_ref_tophat_default_strel(matlab_ref):
    r = matlab_ref("rq_tophat")
    check(bi.imtophat(r.out1, bi.strel_disk(5)), r.out3)


def test_ref_mat2gray_quantile(matlab_ref):
    r = matlab_ref("rq_mat2gray")
    check(bi.mat2gray(r.out1), r.out2)
    check(bi.mat2gray(r.out1, [0.2, 0.7]), r.out3)
    check(bi.quantile(r.out1, [0.005, 0.995]), r.out4)
    check(bi.mat2gray(np.ones((3, 3)) * 0.4, [0.5, 0.5]), r.out5)


def test_ref_roi_move(matlab_ref):
    r = matlab_ref("rq_roi_move")
    hw = (12, 15)
    roi = rq.Roi(r.out1, hw)
    roi.move(2, -3)
    check(roi.indices, r.out2, rtol=0, atol=0)
    check(roi.shift, r.out3, rtol=0, atol=0)
    roi.move(-1, 0, [1, 1])
    check(roi.indices, r.out4, rtol=0, atol=0)
    check(roi.shift, r.out5, rtol=0, atol=0)


def test_ref_roi_morph(matlab_ref):
    r = matlab_ref("rq_roi_morph")
    hw = (12, 15)
    s = rq.Roi(r.out1, hw)
    s.shrink(1)
    check(s.indices, r.out4, rtol=0, atol=0)


@pytest.mark.xfail(reason="Roi.expand uses the approximate bwmorph 'thicken'", strict=False)
def test_ref_roi_expand(matlab_ref):
    r = matlab_ref("rq_roi_morph")
    hw = (12, 15)
    for amount, out in ((1, r.out2), (2, r.out3)):
        e = rq.Roi(r.out1, hw)
        e.expand(amount)
        check(e.indices, out, rtol=0, atol=0)


def test_ref_roi_labels(matlab_ref):
    r = matlab_ref("rq_roi_labels")
    hw = (12, 15)
    a = rq.Roi(r.out1, hw, float(r.out3))
    b = rq.Roi(r.out2, hw, None)
    check(a.to_label_type_image(), r.out4, rtol=0, atol=0)
    check(b.to_label_type_image(), r.out5, rtol=0, atol=0)
    check(rq.rois_label_type_image([a, b]), r.out6, rtol=0, atol=0)


def test_ref_roi_struct(matlab_ref):
    r = matlab_ref("rq_roi_struct")
    back = rq.rois2struct(rq.struct2rois(r.out1, (12, 15)))
    for got, exp in zip(back.ravel(), np.ravel(r.out2)):
        assert list(got) == list(exp)
        for f in got:
            check(got[f], exp[f], rtol=0, atol=0)


def test_ref_roiwf_transforms(matlab_ref):
    r = matlab_ref("rq_roiwf")
    hw = (20, 24)
    a = np.asarray(r.out2, float)
    for method, out in (("transform_green2red", r.out3), ("transform_red2green", r.out4),):
        roi = rq.RoiWf(r.out1, hw)
        getattr(roi, method)(a)
        same_set(roi.indices, out)
    for method, out in (("transform_green2red", r.out6), ("transform_red2green", r.out7)):
        roi = rq.RoiWf(np.atleast_2d(r.out5), hw)
        getattr(roi, method)(a)
        check(roi.indices, out, rtol=0, atol=0)
    p = rq.RoiWf(r.out1, hw, 1, allen_index=50)
    q = rq.RoiWf(np.array([[8 + 4 * 20], [15 + 19 * 20]], float), hw, 1, allen_index=9)
    check(rq.rois_label_type_image([p, q]), r.out8, rtol=0, atol=0)


def test_ref_filter_template_legacy(matlab_ref):
    r = matlab_ref("rq_filter")
    check(rq.filter_template(r.out1, 4, "legacy"), r.out2)
    check(rq.filter_template(r.out1, 2, "updated"), r.out4)


@pytest.mark.xfail(reason="'updated' uses strel('disk', r, 0); MATLAB's default N = 4 differs "
                          "for r > 2", strict=False)
def test_ref_filter_template_updated_r4(matlab_ref):
    r = matlab_ref("rq_filter")
    check(rq.filter_template(r.out1, 4, "updated"), r.out3)


def test_ref_adjust_contrast(matlab_ref):
    r = matlab_ref("rq_contrast")
    check(rq.adjust_img_contrast(r.out1, "minMax"), r.out2, rtol=0, atol=0)
    check(rq.adjust_img_contrast(r.out1, "bestFit"), r.out3, rtol=0, atol=0)
    check(rq.adjust_img_contrast(r.out1, "custom", (0.1, 0.6)), r.out4, rtol=0, atol=0)


def test_ref_cross_colormap_rgb(matlab_ref):
    r = matlab_ref("rq_cross")
    c = np.ravel(r.out1) - 1  # MATLAB centre is 1-based
    hw = hw_of(r.out2)
    check(rq.make_cross(c, hw, 10), r.out3, rtol=0, atol=0)
    check(rq.make_cross((0, 0), hw, 10), r.out4, rtol=0, atol=0)
    check(rq.get_labels_colormap(), r.out5)
    check(rq.uniform_rgb_image((3, 4)), r.out6)


def test_ref_segment_neuron(matlab_ref):
    r = matlab_ref("rq_segment")
    xl = np.ravel(r.out2).astype(int)
    yl = np.ravel(r.out3).astype(int)
    crop = (yl[0] - 1, yl[1], xl[0] - 1, xl[1])
    m1 = rq.segment_neuron(r.out1, 160, crop, min_roi_pix=20, valid_area=r.out4)
    check(m1, r.out5, rtol=0, atol=0)
    m3 = rq.segment_neuron(r.out1, 160, crop, min_roi_pix=20, valid_area=r.out4,
                           close_and_despur=True)
    check(m3, r.out6, rtol=0, atol=0)


def test_ref_auto_adjust(matlab_ref):
    r = matlab_ref("rq_autoadjust")
    hw = np.asarray(r.out1).shape
    rois = [rq.Roi(np.atleast_2d(r.out3), hw), rq.Roi(np.atleast_2d(r.out4), hw)]
    for ignore, base in ((False, 5), (True, 10)):
        res = rq.auto_adjust_rois(rois, r.out2, r.out1, ignore)
        for k, name in enumerate(rq.AutoAdjustResult._fields):
            check(getattr(res, name), r[f"out{base + k}"])


def test_ref_check_matching(matlab_ref):
    r = matlab_ref("rq_checkmatch")
    res = rq.check_roi_matching(r.out1, r.out2, r.out3, hw_of(r.out4))
    assert res.matched == bool(r.out5)
    check(res.is_overlapping[0], np.ravel(r.out6)[0], rtol=0, atol=0)
    check(res.is_type_matched[0], np.ravel(r.out7)[0], rtol=0, atol=0)


def test_ref_match_message_overlap_only(matlab_ref):
    r = matlab_ref("rq_matchmsg")
    ov = list(np.ravel(r.out1))
    ty = list(np.ravel(r.out2))
    assert rq.roi_match_message(ov, ty) == str(r.out3)


def test_ref_wf_transform(matlab_ref):
    r = matlab_ref("rq_wftransform")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = rq.wf_transform_rois(r.out1, hw_of(r.out2), r.out3)
    exp = np.ravel(r.out4)
    for got, e in zip(out.ravel(), exp):
        if np.size(e["indices"]) == 0:
            assert np.size(got["indices"]) == 0
        else:
            check(got["indices"], e["indices"], rtol=0, atol=0)
