"""Reference tests against MATLAB output for ampd, get_frame_times, get_rot_ind,
cmap_centered and the line-shift helpers.

The references are written by ``tests/matlab/ref_core.m`` (run through
``make_reference.m``); each file holds the inputs followed by the MATLAB outputs as
``out1, out2, ...`` in the order given in that script. Skipped unless ``CALLIOPY_TESTDATA``
holds the reference files.
"""

from __future__ import annotations

import numpy as np
from calliopy.analysis import ampd
from calliopy.data import correct_line_shift, estimate_line_shift
from calliopy.plotting import cmap_centered
from calliopy.preprocessing import get_frame_times
from calliopy.viewstack import get_rot_ind

F64 = {"rtol": 1e-12, "atol": 1e-12}


def check(actual, expected, **tol):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if actual.shape != expected.shape and actual.size == expected.size:
        actual, expected = actual.ravel(order="F"), expected.ravel(order="F")
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, equal_nan=True, **(tol or F64))


def test_ref_ampd(matlab_ref):
    r = matlab_ref("core_ampd")
    x = np.ravel(r.out1)
    check(ampd(x, mode="original") + 1, np.ravel(r.out2), rtol=0, atol=0)
    check(ampd(x, mode="extended") + 1, np.ravel(r.out3), rtol=0, atol=0)
    check(ampd(x, mode="adaptive", window=120) + 1, np.ravel(r.out4), rtol=0, atol=0)
    assert ampd(x, mode="extended", debug=True).l_scale == int(np.ravel(r.out5)[0])
    check(ampd(x, mode="adaptive", window=120, debug=True).adaptive_scale, np.ravel(r.out6), rtol=0, atol=0)


def test_ref_get_frame_times(matlab_ref):
    import warnings

    r = matlab_ref("core_frame_times")
    check(get_frame_times(r.out1), np.ravel(r.out2), rtol=0, atol=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check(get_frame_times(r.out3), np.ravel(r.out4), rtol=0, atol=0)


def test_ref_get_rot_ind(matlab_ref):
    r = matlab_ref("core_rot_ind")
    dims = np.ravel(r.out1).astype(int)
    cases = np.atleast_2d(r.out2).astype(int)
    for k, (cf, tx, ty) in enumerate(cases):
        expected = np.ravel(r[f"out{k + 3}"])
        check(get_rot_ind(dims, cf - 1, tx, ty) + 1, expected, rtol=0, atol=0)


def test_ref_line_shift(matlab_ref):
    r = matlab_ref("core_line_shift")
    data = np.asarray(r.out1)
    assert estimate_line_shift(data.mean(axis=2)) == int(np.ravel(r.out2)[0])
    res = correct_line_shift(data)
    check(res.data, r.out3)
    assert res.shift == int(np.ravel(r.out4)[0])


def test_ref_cmap_centered(matlab_ref):
    r = matlab_ref("core_cmap")
    names = ["RdBu", "RdBu", "PuOr", "Spectral", "RdBu", "BrBg"]
    clims = np.atleast_2d(r.out1)
    centers, ms = np.ravel(r.out2), np.ravel(r.out3).astype(int)
    for k, name in enumerate(names):
        check(cmap_centered(name, ms[k], centers[k], clim=clims[k]), r[f"out{k + 4}"])
