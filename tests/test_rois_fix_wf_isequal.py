"""wf_save_rois: "Propagate without overwriting" compares indices like MATLAB isequal."""

from __future__ import annotations

import numpy as np
import pytest

from calliopy.io import MatStruct, load_mat, save_mat
from calliopy.rois import RoiSaveError, RoiWf, wf_rois_struct, wf_save_rois
from calliopy.rois._wf import _isequal

HW = (20, 30)


def _rois():
    ref1 = RoiWf(np.array([[5.0]]), HW, 0.0, None, None, "bregma", "", "", 0.0)
    ref2 = RoiWf(np.array([[25.0]]), HW, 0.0, None, None, "lambda", "", "", 0.0)
    area = RoiWf(np.array([[41.0], [42.0], [61.0]]), HW, 0.0, 1.0, 2.0, "a", "A", "AA", 3.0)
    return [ref1, ref2, area]


def _timepoints(tmp_path, rois, *, squeeze=True):
    p = tmp_path / "first.mat"
    cell = np.empty((1, 1), dtype=object)
    cell[0, 0] = wf_rois_struct(rois)
    save_mat(p, {"ROIs": cell,
                 "registration_parameters": MatStruct(status="OK", activity_calculated=0.0)})
    tp = load_mat(p, squeeze_scalars=squeeze)
    tp["type"] = "wf"
    return {1: tp, 2: None}


def test_isequal_matlab_semantics():
    assert _isequal(np.float64(5.0), np.array([[5.0]]))
    assert _isequal([1.0, 2.0], np.array([[1.0, 2.0]]))
    assert _isequal(np.array([[1], [2]], dtype=np.int64), np.array([[1.0], [2.0]]))
    assert not _isequal(np.array([[1.0, 2.0]]), np.array([[1.0], [2.0]]))  # row vs column
    assert not _isequal(np.array([[np.nan]]), np.array([[np.nan]]))
    assert not _isequal(np.array([[1.0]]), np.array([[1.0, 1.0]]))


@pytest.mark.parametrize("squeeze", [True, False])
def test_keep_policy_accepts_reloaded_rois(tmp_path, squeeze):
    rois = _rois()
    tps = _timepoints(tmp_path, rois, squeeze=squeeze)
    if squeeze:  # the one-pixel reference ROIs come back as numpy scalars
        assert np.ndim(tps[1]["ROIs"][0, 0][0, 0]["indices"]) == 0
    out = wf_save_rois(rois, exp_group_ids=[1, 2], current_exp_id=1, template_hw=HW,
                       roi_template_parameters=None, timepoints=tps, overwrite_policy="keep")
    assert out == {}  # first file has the same ROIs (skipped), second file has no adata


def test_keep_policy_still_detects_changes(tmp_path):
    rois = _rois()
    tps = _timepoints(tmp_path, rois)
    rois[2].indices = np.array([[41.0], [42.0], [62.0]])
    with pytest.raises(RoiSaveError):
        wf_save_rois(rois, exp_group_ids=[1, 2], current_exp_id=1, template_hw=HW,
                     roi_template_parameters=None, timepoints=tps, overwrite_policy="keep")
