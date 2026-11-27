"""Reference tests against MATLAB r2003 output for the analysis-core ports.

The references are written by ``tests/matlab/ref_analysis_core.m`` (run through
``make_reference.m``); each file holds the inputs followed by the MATLAB outputs as
``out1, out2, ...`` in the order given in that script. Skipped unless ``CALLIOPY_TESTDATA``
holds the reference files.
"""

from __future__ import annotations

import numpy as np
from calliopy.analysis import (
    avg_psd_in_range,
    check_vis_speed,
    expmat,
    extract_pulse_envelope,
    ftfil,
    ngsc,
    psmooth,
    smooth2,
)
from calliopy.behavior import find_saccades, leinweber_performance, map_eye_data_to_frame_times
from calliopy.plotting import rois_to_image, spec_traces_data
from calliopy.stats import nan_sem

F64 = {"rtol": 1e-12, "atol": 1e-12}


def check(actual, expected, **tol):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if actual.shape != expected.shape and actual.size == expected.size:
        actual, expected = actual.ravel(order="F"), expected.ravel(order="F")
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, equal_nan=True, **(tol or F64))


def test_ref_psmooth(matlab_ref):
    r = matlab_ref("analysis_core_psmooth")
    x = np.asarray(r.out1)
    check(psmooth(x, 100), r.out2)
    check(psmooth(x), r.out3)
    check(psmooth(np.asarray(r.out4), 60), r.out5)


def test_ref_smooth2(matlab_ref):
    r = matlab_ref("analysis_core_smooth2")
    a = np.asarray(r.out1)
    check(smooth2(a, 5), r.out2)
    check(smooth2(a, 4, axis=1), r.out3)
    check(smooth2(np.asarray(r.out4), 7), r.out5)


def test_ref_ftfil(matlab_ref):
    r = matlab_ref("analysis_core_ftfil")
    x = np.asarray(r.out1)
    check(ftfil(x, 100, 1, 20), r.out2)
    check(ftfil(x.T, 100, 0.5, 30), r.out3)


def test_ref_ngsc(matlab_ref):
    r = matlab_ref("analysis_core_ngsc")
    check(ngsc(r.out1), r.out2, rtol=1e-10, atol=0)  # eigenvalues via SVD vs MATLAB pca


def test_ref_avg_psd(matlab_ref):
    r = matlab_ref("analysis_core_psd")
    x = np.asarray(r.out1)
    check(avg_psd_in_range(x, 100, [5, 20]), r.out2)
    check(avg_psd_in_range(x, 100, [5, 20], True), r.out3)
    m = np.asarray(r.out4)
    check(avg_psd_in_range(m, 50, [1, 10], False, axis=1), r.out5)
    check(avg_psd_in_range(m.T, 50, [1, 10], True), r.out6)


def test_ref_envelope(matlab_ref):
    r = matlab_ref("analysis_core_envelope")
    check(extract_pulse_envelope(r.out1, dt=4), r.out2)


def test_ref_check_vis_speed(matlab_ref):
    r = matlab_ref("analysis_core_vis_speed")
    check(check_vis_speed(r.out1), r.out2)


def test_ref_library(matlab_ref):
    r = matlab_ref("analysis_core_library")
    check(expmat(r.out1, 3), r.out2, rtol=0, atol=0)
    x = np.asarray(r.out3)
    check(nan_sem(x), r.out4)
    check(nan_sem(x, axis=1), r.out5)


def test_ref_behavior(matlab_ref):
    r = matlab_ref("analysis_core_behavior")
    vm, vr, ang = r.out1, r.out2, np.asarray(r.out3)
    check(leinweber_performance(vm, vr, ang), r.out4)
    check(leinweber_performance(vm, vr, np.r_[0.0, ang.ravel()]), r.out5)
    res = map_eye_data_to_frame_times(r.out13, r.out14, r.out15, r.out16, r.out17, r.out6, r.out7)
    for k, got in enumerate(res):
        check(got, r[f"out{8 + k}"], rtol=0, atol=0)


def test_ref_find_saccades(matlab_ref):
    r = matlab_ref("analysis_core_find_saccades")
    res = find_saccades(np.asarray(r.out1))
    check(res.saccade_frames, r.out2, rtol=0, atol=0)
    for k, got in enumerate(res[1:]):
        check(got, r[f"out{3 + k}"])


def _cells(c):
    a = np.asarray(c, dtype=object)
    return [np.asarray(v).ravel() for v in a.ravel(order="F")]


def test_ref_rois2image(matlab_ref):
    r = matlab_ref("analysis_core_rois2image")
    dim = tuple(np.ravel(r.out1).astype(int))
    rois = _cells(r.out2)
    check(rois_to_image(rois, dim, randcolor=False), r.out3, rtol=0, atol=0)
    check(rois_to_image(rois, dim, randcolor=False, type="perim"), r.out4, rtol=0, atol=0)
    check(rois_to_image(rois, dim, randcolor=False, sel=[1, 2]), r.out5, rtol=0, atol=0)
    tmpl = np.asarray(r.out6)
    check(rois_to_image(rois, dim, randcolor=False, template=tmpl, sel=[1, 0]), r.out7)
    sel3 = [[1, 1], [2, 2], [0, 1]]
    check(rois_to_image(rois, dim, randcolor=False, sel=sel3), r.out8, rtol=0, atol=0)


def test_ref_spec_traces(matlab_ref):
    r = matlab_ref("analysis_core_spec_traces")
    check(spec_traces_data(r.out1), r.out2)
