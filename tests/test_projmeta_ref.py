"""Reference tests of calliopy.projmeta / explog / data against MATLAB output.

The references are written by ``tests/matlab/ref_projmeta.m`` (run through
``make_reference.m``); each file holds the inputs followed by the MATLAB outputs as
``out1, out2, ...`` in the order given in that script. Skipped unless ``CALLIOPY_TESTDATA``
holds the reference files. Functions that need psmooth/ftfil/smooth2 are skipped until
:mod:`calliopy.analysis` provides them.
"""

from __future__ import annotations

import numpy as np
import pytest
from calliopy.data import get_stackid
from calliopy.explog import as_exp_log, eid, qexp, remove_html
from calliopy.io import MatStruct
from calliopy.projmeta import (
    act2mat,
    assign_aux_method,
    assign_processed_aux,
    get_lick_times_thrsh,
    get_smoothed_rois,
    get_vel_ind_from_adata,
    pm_expids,
    preprocess_widefield_activity,
    pulsed_to_continuous,
    trig2snps,
)
from calliopy.projmeta._struct import grid_struct_array, struct_array

F64 = {"rtol": 1e-12, "atol": 1e-12}


def check(actual, expected, **tol):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if actual.shape != expected.shape and actual.size == expected.size:
        actual, expected = actual.ravel(order="F"), expected.ravel(order="F")
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, equal_nan=True, **(tol or F64))


def need(*names):
    import calliopy.analysis

    missing = [n for n in names if not hasattr(calliopy.analysis, n)]
    if missing:
        pytest.skip(f"calliopy.analysis lacks {missing}")


def test_ref_stackid_html(matlab_ref):
    r = matlab_ref("projmeta_stackid")
    check(get_stackid(["S1-T1234.lvd", "nothing", "s1-p77_ch525.bin", "S1-T0123.lvd"]), r.out1, rtol=0, atol=0)
    assert remove_html("a <b>bold</b> <i x='1'>it</i>") == matlab_ref("projmeta_remove_html").out1


def test_ref_aux(matlab_ref):
    r = matlab_ref("projmeta_pulsed")
    check(pulsed_to_continuous(r.out1, 1000, 2), r.out2, rtol=0, atol=0)
    r = matlab_ref("projmeta_lick")
    check(get_lick_times_thrsh(r.out1), r.out2, rtol=0, atol=0)
    r = matlab_ref("projmeta_extend_trig")
    check(assign_processed_aux(np.ravel(r.out1), ["extend_trig_to_200ms"], "Reward")[0][1], r.out2, rtol=0, atol=0)
    r = matlab_ref("projmeta_aux_method")
    pairs = np.asarray(r.out1, dtype=object)
    expected = [str(x) for x in np.ravel(np.asarray(r.out2, dtype=object), order="F")]
    got = [",".join(assign_aux_method(str(pairs[k, 0]), str(pairs[k, 1]))) for k in range(pairs.shape[0])]
    assert got == expected


def test_ref_velocity(matlab_ref):
    need("ftfil", "smooth2")
    r = matlab_ref("projmeta_vel")
    res = get_vel_ind_from_adata(np.ravel(r.out1))
    check(res.vel, r.out2)
    check(res.vel_ind, r.out3, rtol=0, atol=0)
    check(res.vel_smoothed, r.out4)
    check(res.vel_raw, r.out5)
    r = matlab_ref("projmeta_vr_vel")
    check(assign_processed_aux(np.ravel(r.out1), ["VR_pos_to_vel_smoothed"], "VRx")[0][1], r.out2)


def test_ref_act2mat_trig2snps(matlab_ref):
    r = matlab_ref("projmeta_act2mat")
    rd = grid_struct_array({(0, 0): MatStruct(act=np.asarray(r.out1, float), nbr_frames=np.array([[25, 25]]),
                                              frame_times=np.arange(1, 51).reshape(1, 50),
                                              speed=2 * np.arange(1, 51.0).reshape(1, 50)),
                            (1, 0): MatStruct(act=np.asarray(r.out2, float))}, 2, 1)
    pm = struct_array([MatStruct(rd=rd)])
    res = act2mat(pm, 0, 0)
    check(res.actmat, r.out3)
    check(res.ind_roi_in_layer + 1, r.out4, rtol=0, atol=0)
    check(res.ind_layer + 1, r.out5, rtol=0, atol=0)
    snps, arr = trig2snps(pm, 0, 0, [9, 29, 47, 25], (-3, 3), ["speed"])
    check(snps, r.out6)
    check(arr[0], r.out7)


def test_ref_pm_expids(matlab_ref):
    r = matlab_ref("projmeta_pm_expids")
    rd1 = grid_struct_array({(0, 1): MatStruct(act=np.zeros((0, 0)))}, 1, 2)
    rd2 = grid_struct_array({(0, 0): MatStruct(act=np.zeros((0, 0)))}, 1, 1)
    pm = struct_array([MatStruct(ExpGroup=np.array([[10, 11]]), rd=rd1), MatStruct(ExpGroup=np.array([[20]]), rd=rd2)])
    check(pm_expids(pm), r.out1)
    check(pm_expids(pm, flat=True), r.out2)
    check(pm_expids(pm, tps=[0]), r.out3)


def test_ref_qexp(matlab_ref):
    r = matlab_ref("projmeta_qexp")
    el = as_exp_log(r.out1)
    check(qexp("pi", "hans", "expid", exp_log=el).exp_log_matched, r.out2, rtol=0, atol=0)
    check(qexp("regex", "day1|day2", "stackid", exp_log=el).exp_log_matched, r.out3, rtol=0, atol=0)
    check(qexp("acode", "2", "stackid", exp_log=el).exp_log_matched, r.out4, rtol=0, atol=0)
    check(qexp("pi", "hans", "last", 2, "stackid", exp_log=el).exp_log_matched, r.out5, rtol=0, atol=0)
    check(qexp("animalid", "/002/", "stackid", exp_log=el).exp_log_matched, r.out6, rtol=0, atol=0)
    check(qexp("isexp", 0, "stackid", exp_log=el).exp_log_matched, r.out7, rtol=0, atol=0)
    assert eid("hans", el).field == r.out8
    assert eid(103, el).field == r.out9
    assert eid("101", el).field == r.out10


def test_ref_smoothed_rois(matlab_ref):
    need("psmooth")
    r = matlab_ref("projmeta_smoothed_rois")
    a = np.asarray(r.out1, float)
    rois = struct_array([MatStruct(indices=np.array([[1.0, 2.0]]), activity=a[:, [k]]) for k in range(3)])
    res = get_smoothed_rois(rois)
    check(res.act, r.out2)
    check([x.F0 for x in res.roi_info.ravel()], r.out3)
    r = matlab_ref("projmeta_preprocess_wf")
    a = np.asarray(r.out1, float)
    check(preprocess_widefield_activity([MatStruct(activity=a[:, [k]]) for k in range(2)], [7000, 7000]), r.out2)
