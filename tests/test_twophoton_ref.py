"""Comparison with MATLAB r2003 outputs saved by tests/matlab/ref_twophoton.m.

Skipped unless CALLIOPY_TESTDATA holds ``reference/ref_twophoton_<name>.mat``. Tolerances
follow CLAUDE.md (float64 1e-12, exact for integers); looser ones are explained inline.
"""

from __future__ import annotations

import numpy as np
import pytest

from calliopy.eeg import create_erp_auxaligned, get_eeg_times, my_resample, my_welch
from calliopy.io import MatStruct
from calliopy.oii import fourier_maps
from calliopy.twophoton import (
    calc_act_map,
    fix_registration,
    get_iframe_times,
    np_activity,
    register_multilayer,
    roi_activity,
)
from calliopy.twophoton._builtins import (
    fspecial_gaussian,
    imfilter,
    impyramid_reduce,
    mat2gray,
    matchpairs,
    smooth_moving,
)
from calliopy.twophoton._registration import _movmean3


def a(x):
    return np.asarray(x)


def test_ref_roi_activity(matlab_ref):
    r = matlab_ref("twophoton_roi_activity")
    rois = [MatStruct(indices=r["out2"]), MatStruct(indices=r["out3"])]
    out = roi_activity(a(r["out1"]), rois)
    np.testing.assert_allclose(np.hstack([o["activity"] for o in out]), a(r["out4"]), rtol=1e-12)


def test_ref_np_activity(matlab_ref):
    r = matlab_ref("twophoton_np_activity")
    out = np_activity(a(r["out1"]), MatStruct(indices=r["out2"], indices_middle=r["out3"]))
    np.testing.assert_allclose(out["activity"], a(r["out4"]).reshape(-1, 1), rtol=1e-12)
    np.testing.assert_allclose(out["activity_middle"], a(r["out5"]).reshape(-1, 1), rtol=1e-12)


def test_ref_calc_act_map(matlab_ref):
    r = matlab_ref("twophoton_calc_act_map")
    np.testing.assert_allclose(calc_act_map(a(r["out1"])), a(r["out2"]), rtol=1e-12)
    np.testing.assert_allclose(calc_act_map(a(r["out1"]), 4), a(r["out3"]), rtol=1e-12)


def test_ref_fix_registration(matlab_ref):
    r = matlab_ref("twophoton_fix_registration")
    sx, sy = fix_registration(a(r["out1"]), a(r["out2"]), a(r["out3"]), a(r["out4"]))
    np.testing.assert_array_equal(sx.ravel(), a(r["out5"]).ravel())
    np.testing.assert_array_equal(sy.ravel(), a(r["out6"]).ravel())


def test_ref_register_multilayer(matlab_ref):
    r = matlab_ref("twophoton_register_multilayer")
    res = register_multilayer(a(r["out1"]), int(r["out2"]))
    np.testing.assert_allclose(res.mean_data, a(r["out3"]), rtol=1e-12)
    np.testing.assert_allclose(res.dx_corr.ravel(), a(r["out4"]).ravel(), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(res.dy_corr.ravel(), a(r["out5"]).ravel(), rtol=1e-12, atol=1e-9)


def test_ref_get_eeg_times(matlab_ref):
    r = matlab_ref("twophoton_get_eeg_times")
    np.testing.assert_array_equal(get_eeg_times(r["out1"], r["out2"]).ravel(), a(r["out3"]).ravel())


def test_ref_erp(matlab_ref):
    r = matlab_ref("twophoton_erp")
    res = create_erp_auxaligned(r["out1"], r["out2"], r["out3"], 1000, 0.2, 0.5)
    np.testing.assert_allclose(res.erp, a(r["out4"]), rtol=1e-12)
    np.testing.assert_allclose(res.time.ravel(), a(r["out5"]).ravel(), rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(res.epochs_ids + 1, a(r["out6"]))


def test_ref_resample(matlab_ref):
    r = matlab_ref("twophoton_resample")
    np.testing.assert_allclose(my_resample(a(r["out1"]), 1, 2, 1), a(r["out2"]), rtol=1e-12, atol=1e-14)
    v = a(r["out3"]).reshape(-1, 1)
    np.testing.assert_allclose(my_resample(v, 2, 3), a(r["out4"]).reshape(-1, 1), rtol=1e-12, atol=1e-14)


def test_ref_welch(matlab_ref):
    r = matlab_ref("twophoton_welch")
    p, f = my_welch(r["out1"], 128, 64, 128, 1000)
    np.testing.assert_allclose(p.ravel(), a(r["out2"]).ravel(), rtol=1e-12)
    np.testing.assert_allclose(f.ravel(), a(r["out3"]).ravel(), rtol=1e-12)
    p2, _ = my_welch(r["out1"], 128, 64, 101, 1000)
    np.testing.assert_allclose(p2.ravel(), a(r["out4"]).ravel(), rtol=1e-12)


def test_ref_iframe_times(matlab_ref):
    r = matlab_ref("twophoton_iframe_times")
    out = get_iframe_times(r["out1"], r["out2"], r["out3"])
    np.testing.assert_array_equal(out.iframe_times.ravel(), a(r["out4"]).ravel())
    np.testing.assert_array_equal(out.shutter_open.ravel(), a(r["out5"]).ravel())


def test_ref_fourier_maps(matlab_ref):
    r = matlab_ref("twophoton_fourier_maps")
    m = fourier_maps([r["out1"], r["out2"], r["out3"], r["out4"]], 0, a(r["out5"]).astype(bool))
    # imfilter / std summation order: allow 1e-10
    np.testing.assert_allclose(m.Power, a(r["out6"]), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(m.PhaseHor, a(r["out7"]), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(m.PhaseVer, a(r["out8"]), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(m.PhaseMapHor, a(r["out9"]), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(m.Phase[0], a(r["out10"]), rtol=1e-12, atol=1e-14)


def test_ref_builtins(matlab_ref):
    r = matlab_ref("twophoton_builtins")
    # impyramid / imfilter are reimplemented from the documented algorithms; this test is
    # what decides whether they match MATLAB (border handling unverified until it passes)
    np.testing.assert_allclose(impyramid_reduce(a(r["out1"])), a(r["out2"]), rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(impyramid_reduce(a(r["out3"])), a(r["out4"]))
    np.testing.assert_allclose(fspecial_gaussian(15, 2), a(r["out5"]), rtol=1e-12)
    np.testing.assert_array_equal(imfilter(a(r["out3"]), a(r["out5"])), a(r["out6"]))
    np.testing.assert_allclose(mat2gray(a(r["out1"])), a(r["out7"]), rtol=1e-12)
    y = a(r["out8"]).ravel()
    np.testing.assert_allclose(smooth_moving(y, 81), a(r["out9"]).ravel(), rtol=1e-12)
    np.testing.assert_allclose(smooth_moving(y, 6), a(r["out10"]).ravel(), rtol=1e-12)
    np.testing.assert_allclose(_movmean3(a(r["out11"]), 4), a(r["out12"]), rtol=1e-12)
    pairs = matchpairs(a(r["out13"]), 0.6) + 1
    ref = a(r["out14"]).reshape(-1, 2)
    np.testing.assert_array_equal(pairs[np.lexsort(pairs.T[::-1])], ref[np.lexsort(ref.T[::-1])])


@pytest.mark.parametrize("name", ["twophoton_roi_activity"])
def test_ref_files_are_synthetic(matlab_ref, name):
    assert str(matlab_ref(name)["input_file"]) == "synthetic"
