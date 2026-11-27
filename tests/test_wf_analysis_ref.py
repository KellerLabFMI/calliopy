"""Reference tests: 21_Analysis WF_* ports against MATLAB output.

The references are written by ``tests/matlab/ref_wf_analysis.m``. Every file holds the
inputs followed by the MATLAB outputs as ``out1, out2, ...`` (order in the comment of each
block of that script). All tests are skipped unless ``CALLIOPY_TESTDATA`` holds the
reference files (see docs/porting_guide.md).
"""

from __future__ import annotations

import numpy as np

from calliopy import widefield as wf
from calliopy.widefield import _proj_meta_tools as pmt

F64 = {"rtol": 1e-12, "atol": 1e-12}
# FFT-based results: MATLAB's FFTW and numpy's pocketfft round differently
FFT = {"rtol": 1e-9, "atol": 1e-12}


def check(actual, expected, **tol):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if actual.shape != expected.shape and actual.size == expected.size:
        actual, expected = actual.ravel(order="F"), expected.ravel(order="F")
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, equal_nan=True, **(tol or F64))


def cells(c):
    return list(np.asarray(c, dtype=object).ravel(order="F"))


def test_ref_avgpsd(matlab_ref):
    r = matlab_ref("wfa_avgpsd")
    check(pmt._avg_psd_in_range(r.out1, 50, (2, 7), axis=1), r.out3, **FFT)
    check(pmt._avg_psd_in_range(r.out2, 50, (2, 7), axis=1), r.out4, **FFT)
    check(pmt._avg_psd_in_range(r.out1, 50, (2, 7), True, axis=1), r.out5, **FFT)


def test_ref_corrmat(matlab_ref):
    r = matlab_ref("wfa_corrmat")
    pm = r.out1
    check(wf.wf_corrmat(pm).corrmat_all, r.out2)
    check(wf.wf_corrmat(pm, comment="PB", tp_list=[0]).corrmat_all, r.out3)
    check(wf.wf_corrmat(pm, median_div_on=True).corrmat_all, r.out4)
    check(wf.wf_corrmat(pm, running_state="running").corrmat_all, r.out5)
    check(wf.corr_distance_histogram(r.out2[:, :, :, 0], pm).normalized_data, r.out6)


def test_ref_corrmat_matrix(matlab_ref):
    r = matlab_ref("wfa_corrmat_matrix")
    pm = r.out1
    check(wf.wf_corrmat_matrix(pm).corrmat_all, r.out2)
    check(wf.wf_corrmat_matrix(pm, hemisphere_organization=2).corrmat_all, r.out3)
    check(wf.wf_corrmat_matrix(pm, separate_hemispheres=False).corrmat_all, r.out4)


def test_ref_find_low_corrcoef(matlab_ref):
    r = matlab_ref("wfa_findlow")
    res = wf.wf_find_low_corrcoef(r.out1, cc_range=(0, 0.05))
    for got, exp in zip(res.idx_counts_sort_all, cells(r.out3), strict=True):
        got = np.asarray(got)
        exp = np.asarray(exp)
        np.testing.assert_array_equal(got[:, 0] + 1, exp[:, 0])  # MATLAB ROIs are 1-based
        np.testing.assert_array_equal(got[:, 1], exp[:, 1])


def test_ref_lesioning(matlab_ref):
    r = matlab_ref("wfa_lesion")
    res = wf.wf_lesioning_rois(r.out1)
    for key, exp in r.out2.items():
        got = res.lesioning_scores[key]
        check(got.roi_avg_corr, exp["roi_avg_corr"])
        check(got.lesion_change, exp["lesion_change"], rtol=1e-10, atol=1e-13)
        check(got.region_values, exp["region_values"])
        assert got.region_names == [str(x) for x in cells(exp["region_names"])]
        check(got.baseline, exp["baseline"])


def test_ref_trim(matlab_ref):
    r = matlab_ref("wfa_trim")
    got = wf.wf_trim_stack_frames(r.out1, 5)
    exp = pmt._sites(r.out2)
    for gs, es in zip(pmt._sites(got), exp, strict=True):
        for gr, er in zip(gs["rd"].ravel(order="F"), pmt._rd_grid(es).ravel(order="F"),
                          strict=True):
            check(gr["act"], er["act"])
            check(gr["nbr_frames"], er["nbr_frames"])
            check(gr["Running"], er["Running"])


def test_ref_commodore(matlab_ref):
    r = matlab_ref("wfa_commodore")
    a = wf.wf_commodore(r.out1, frame_rate=10, analysis_window=10)
    b = wf.wf_commodore(r.out1, frame_rate=10, analysis_window=10, average_over_psd=False)
    for got, exp in zip(a, cells(r.out2), strict=True):
        check(got, exp, **FFT)
    for got, exp in zip(b, cells(r.out3), strict=True):
        check(got, exp, **FFT)


def test_ref_onsets(matlab_ref):
    r = matlab_ref("wfa_onsets")
    got = wf.wf_onsets(r.out1, "gratings", window=(-10, 60), baseline_window=(-30, -20))
    check(got, r.out2)


def test_ref_roi_filter(matlab_ref):
    r = matlab_ref("wfa_roi_filter")
    got = wf.wf_corrmat_roi_filter(r.out1, exclude_rois=["RSP"])
    check(got.corrmat_all_filtered, r.out2)
