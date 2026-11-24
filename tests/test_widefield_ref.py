"""Reference tests: calliopy.widefield against MATLAB output.

The references are written by ``tests/matlab/ref_widefield.m`` (run through
``make_reference.m``). Every file holds the inputs followed by the MATLAB outputs as
``out1, out2, ...``; the order is given in the comment of each block of that script.
All tests are skipped unless ``CALLIOPY_TESTDATA`` holds the reference files
(see docs/porting_guide.md).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from calliopy import _matlab as ml
from calliopy import widefield as wf
from calliopy.widefield import _frames

F64 = {"rtol": 1e-12, "atol": 1e-12}


def check(actual, expected, **tol):
    """Compare with MATLAB; vectors are compared regardless of orientation."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    is_vector = min(actual.squeeze().ndim, expected.squeeze().ndim) <= 1
    if actual.shape != expected.shape and actual.size == expected.size and is_vector:
        actual = actual.ravel(order="F")
        expected = expected.ravel(order="F")
    assert actual.shape == expected.shape
    np.testing.assert_allclose(
        actual.astype(float), expected.astype(float), equal_nan=True, **(tol or F64)
    )


def test_ref_round(matlab_ref):
    r = matlab_ref("wf_round")
    check(ml.mround(r.out1), r.out2, rtol=0, atol=0)


def test_ref_prctile(matlab_ref):
    r = matlab_ref("wf_prctile")
    check(ml.prctile(r.out1, np.ravel(r.out2)), r.out3)
    check(ml.prctile(r.out1, [10, 90], axis=1), r.out4)


def test_ref_corr2(matlab_ref):
    r = matlab_ref("wf_corr2")
    check(ml.corr2(r.out1, r.out2), r.out3)


def test_ref_conv2(matlab_ref):
    r = matlab_ref("wf_conv2")
    a, k2, k3 = r.out1, r.out2, r.out3
    check(ml.conv2(a, k2, "full"), r.out4)
    check(ml.conv2(a, k2, "same"), r.out5)
    check(ml.conv2(a, k2, "valid"), r.out6)
    check(ml.conv2(a, k3, "same"), r.out7)


def test_ref_filter2(matlab_ref):
    r = matlab_ref("wf_filter2")
    a, k2, k3 = r.out1, r.out2, r.out3
    check(ml.filter2(k2, a), r.out4)
    check(ml.filter2(k3, a, "valid"), r.out5)
    check(ml.filter2(np.ones((4, 4)) / 16, a), r.out6)


def test_ref_imgaussfilt(matlab_ref):
    r = matlab_ref("wf_imgaussfilt")
    a = r.out1
    check(ml.imgaussfilt(a, 1.3), r.out2)
    check(ml.imgaussfilt(a, (0.8, 2.2)), r.out3)
    check(ml.imgaussfilt(ml.matlab_cast(a * 100, np.uint16), 1), r.out4, rtol=0, atol=0)


def test_ref_normxcorr2(matlab_ref):
    r = matlab_ref("wf_normxcorr2")
    check(ml.normxcorr2(r.out2, r.out1), r.out3, rtol=1e-9, atol=1e-9)


def test_ref_linspace(matlab_ref):
    r = matlab_ref("wf_linspace")
    check(ml.linspace(0, 1, 10), r.out1, rtol=0, atol=0)
    check(ml.linspace(-3.3, 7.9, 13), r.out2, rtol=0, atol=0)


def test_ref_simtform2d(matlab_ref):
    r = matlab_ref("wf_simtform2d")
    scale, angle, tx, ty = np.ravel(r.out1)
    check(wf.similarity_matrix(scale, angle, (tx, ty)), r.out2)


def test_ref_fitgeotform2d(matlab_ref):
    r = matlab_ref("wf_fitgeotform2d")
    check(wf.fit_similarity(r.out1, r.out2), r.out3)
    check(wf.fit_similarity(r.out4, r.out5), r.out6, rtol=1e-10, atol=1e-10)


def test_ref_transform_points(matlab_ref):
    r = matlab_ref("wf_transform_points")
    check(wf.transform_points(r.out1, r.out2), r.out3)


def test_ref_imwarp(matlab_ref):
    r = matlab_ref("wf_imwarp")
    img = r.out1
    check(wf.warp_image(img, r.out2, (7, 9)), r.out3)
    check(wf.warp_image(img, r.out4, (8, 10)), r.out5)
    check(wf.warp_image(img, r.out4, (8, 10), order=0), r.out6)
    check(wf.warp_image(img.astype(np.uint16), r.out4, (7, 9)), r.out7, rtol=0, atol=0)


def test_ref_imwarp_labels(matlab_ref):
    r = matlab_ref("wf_imwarp_labels")
    check(wf.warp_image(r.out1, r.out2, (30, 40)), r.out3)


def test_ref_fftreg(matlab_ref):
    r = matlab_ref("wf_fftreg")
    img = r.out1
    for (dr, dc), expected in zip(np.asarray(r.out2, dtype=int), np.asarray(r.out3)):
        got = _frames.fftreg(np.fft.fft2(np.roll(img, (dr, dc), axis=(0, 1))), np.fft.fft2(img))
        check([got.row_shift, got.col_shift], expected, rtol=0, atol=0)


def test_ref_shift_data(matlab_ref):
    r = matlab_ref("wf_shift_data")
    stack = r.out1
    check(_frames.shift_data(stack, [1, 0], [0, -1]), r.out2, rtol=0, atol=0)
    check(_frames.shift_data(stack.astype(np.uint16), 0.5, -0.5), r.out3, rtol=0, atol=0)


def test_ref_register_frames(matlab_ref):
    r = matlab_ref("wf_register_frames")
    stack = r.out1
    got = _frames.register_frames(stack, stack[:, :, 0])
    check(got.shift_x, r.out3, rtol=0, atol=0)
    check(got.shift_y, r.out4, rtol=0, atol=0)
    stack16 = ml.matlab_cast(stack, np.uint16)
    got = _frames.register_frames(stack16, stack16[:, :, 0])
    check(got.shift_x, r.out5, rtol=0, atol=0)
    check(got.shift_y, r.out6, rtol=0, atol=0)
    got = _frames.register_frames(stack)
    check(got.shift_x, r.out7, rtol=0, atol=0)
    check(got.shift_y, r.out8, rtol=0, atol=0)


def test_ref_zero_shift_bandpass(matlab_ref):
    r = matlab_ref("wf_zeroShiftBandPass")
    for k in range(1, 13, 3):
        fs, lb, ub = np.ravel(r[f"out{k + 1}"])
        check(wf.zero_shift_bandpass(r[f"out{k}"], fs, lb, ub), r[f"out{k + 2}"])


def test_ref_zero_shift_notch(matlab_ref):
    r = matlab_ref("wf_zeroShiftNotch")
    for k in range(1, 10, 3):
        fs, nf, bw = np.ravel(r[f"out{k + 1}"])
        check(wf.zero_shift_notch(r[f"out{k}"], fs, nf, bw), r[f"out{k + 2}"])


def test_ref_velocity_field(matlab_ref):
    r = matlab_ref("wf_velocity_field")
    alpha, max_iter, sigma = np.ravel(r.out2)
    got = wf.velocity_field_horn_schunck(r.out1, alpha=alpha, max_iter=int(max_iter), sigma=sigma)
    # 50 iterations accumulate rounding differences between the two convolution routines
    check(got.U, r.out3, rtol=1e-8, atol=1e-10)
    check(got.V, r.out4, rtol=1e-8, atol=1e-10)


def test_ref_get_frame_times_widefield(matlab_ref):
    r = matlab_ref("wf_get_frame_times_widefield")
    stamps = [
        datetime(int(y), int(mo), int(d), int(h), int(mi)) + timedelta(seconds=float(s))
        for y, mo, d, h, mi, s in np.asarray(r.out4)
    ]
    got = wf.frame_times_widefield(np.ravel(r.out1), np.ravel(r.out2), stamps, float(r.out3))
    check(got.frame_times, r.out5, rtol=0, atol=0)
    check(got.frame_times_timestamp, r.out6, rtol=0, atol=0)


def test_ref_median_divide_video(matlab_ref):
    r = matlab_ref("wf_median_divide_video")
    check(wf.median_divide_video(r.out1), r.out2, rtol=1e-6, atol=0)


def test_ref_angles_from_top_left(matlab_ref):
    r = matlab_ref("wf_angles_from_top_left")
    check(wf.angles_from_top_left(r.out1, np.ravel(r.out2), np.ravel(r.out3)), r.out4)
