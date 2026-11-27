"""Regression test: propagate_tform drops ``activity`` from a struct-array ROIs{1} like rmfield."""

from __future__ import annotations

import numpy as np

from calliopy.io import MatStruct
from calliopy.widefield._registration import _drop_activity


def _struct_array():
    a = np.empty((1, 2), dtype=object)
    a[0, 0] = MatStruct(indices=np.array([[1]]), activity=np.ones((1, 3)))
    a[0, 1] = MatStruct(indices=np.array([[2]]), activity=np.ones((1, 3)))
    return a


def test_struct_array_first_layer_loses_activity_in_every_element():
    first, second = _struct_array(), _struct_array()
    rois = np.empty((1, 2), dtype=object)
    rois[0, 0], rois[0, 1] = first, second
    out = _drop_activity(rois)
    assert out[0, 0].shape == (1, 2)
    assert not any("activity" in e for e in out[0, 0].ravel())
    assert all("activity" in e for e in out[0, 1].ravel())   # MATLAB touches ROIs{1} only
    assert all("activity" in e for e in first.ravel())       # input not modified


def test_scalar_struct_first_layer_still_handled():
    s = MatStruct(indices=1, activity=2)
    out = _drop_activity([s, s])
    assert "activity" not in out[0] and "activity" in out[1]
