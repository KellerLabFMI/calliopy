"""Synthetic tests of the SpineQuest computational cores (calliopy.twophoton._spines)."""

from __future__ import annotations

import datetime

import numpy as np
import pytest
import tifffile

from calliopy.io import MatStruct
from calliopy.twophoton import (
    crop_spiny_region,
    match_timepoint_to_master,
    preceding_expid_in_site,
    propagate_spiny_regions,
    seed_label_map,
    spine_positions,
    spine_tracking_export,
    spines_from_roi_map,
    spiny_regions_from_marks,
    track_spines,
    write_label_tiff,
    write_spine_region_tiff,
)

SHAPE = (20, 30, 6)


def lin(r, c, z, shape=SHAPE):
    return np.ravel_multi_index((np.atleast_1d(r) - 1, np.atleast_1d(c) - 1, np.atleast_1d(z) - 1),
                                shape, order="F") + 1


def test_spiny_regions_from_marks():
    marks = np.r_[lin(2, 3, 1), lin(10, 12, 1), lin(5, 5, 4)]
    reg = spiny_regions_from_marks(marks, SHAPE)
    np.testing.assert_array_equal(reg[0], [[2, 3, 1], [10, 12, 1], [5, 5, 4]])
    with pytest.raises(ValueError):
        spiny_regions_from_marks(marks[:2], SHAPE)
    with pytest.raises(ValueError):  # corners swapped
        spiny_regions_from_marks(np.r_[lin(10, 12, 1), lin(2, 3, 1), lin(5, 5, 4)], SHAPE)
    with pytest.raises(ValueError):  # point 3 not deeper
        spiny_regions_from_marks(np.r_[lin(2, 3, 2), lin(10, 12, 2), lin(5, 5, 1)], SHAPE)


def test_crop_and_propagate():
    md = np.arange(np.prod(SHAPE)).reshape(SHAPE)
    reg = np.array([[2, 3, 1], [10, 12, 1], [5, 5, 4]])
    crop = crop_spiny_region(md, reg)
    assert crop.shape == (9, 10, 4)
    assert crop[0, 0, 0] == md[1, 2, 0]
    new = propagate_spiny_regions([reg], [[4, 6, 2]])[0]  # x=4 (col), y=6 (row), z=2
    np.testing.assert_array_equal(new, [[6, 4, 2], [14, 13, 2], [5, 5, 5]])


def test_tiff_roundtrip_and_seed(tmp_path):
    vol = np.random.default_rng(0).normal(size=(5, 7, 3))
    p = tmp_path / "r.tif"
    write_spine_region_tiff(p, vol)
    back = tifffile.imread(p)
    np.testing.assert_allclose(np.moveaxis(back, 0, 2), vol)
    spines = [MatStruct(indices=[1, 2]), MatStruct(indices=[105])]
    lab = seed_label_map(spines, (5, 7, 3))
    assert lab[0, 0, 0] == 1 and lab[1, 0, 0] == 1 and lab.ravel(order="F")[104] == 2
    assert seed_label_map([MatStruct(indices=[200])], (5, 7, 3)) is None
    assert seed_label_map([], (5, 7, 3)) is None
    write_label_tiff(tmp_path / "s.tif", lab)
    assert tifffile.imread(tmp_path / "s.tif").dtype == np.uint16


def test_spines_from_roi_map():
    lab = np.zeros((4, 5, 2))
    lab[0, 0, 0] = 3
    lab[1, 1, 1] = 7
    lab[2, 1, 1] = 7
    dend = np.zeros((4, 5, 2))
    dend[3, 4, 1] = 1
    spines, den = spines_from_roi_map(np.concatenate([lab, dend], axis=2))
    assert len(spines) == 2
    assert spines[0]["indices"][:, 0].tolist() == [1]
    assert spines[1]["indices"][:, 0].tolist() == [26, 27]
    assert den[:, 0].tolist() == [40]


def test_preceding_expid():
    e = [5, 6, 7, 9, 9]
    s = [1, 1, 1, 2, 2]
    d = ["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-01", "2024-01-01"]
    assert preceding_expid_in_site(5, e, s, d) == 7
    assert preceding_expid_in_site(6, e, s, d) is None
    assert preceding_expid_in_site(9, e, s, d) is None
    assert preceding_expid_in_site(42, e, s, d) is None
    assert preceding_expid_in_site(7, e, s, [datetime.datetime(2024, 1, k) for k in (3, 1, 2, 1, 1)]) == 6


def test_spine_positions():
    size = (10, 12, 4)
    sp = [MatStruct(indices=lin([2, 4], [3, 3], [1, 1], size))]
    np.testing.assert_allclose(spine_positions(sp, size), [[3, 3, 1]])


def test_match_timepoint_to_master():
    pos = np.array([[10.0, 10, 1], [30, 30, 2]])
    pres = np.array([[1.0, 0, 0], [1, 0, 0]])
    cur = np.array([[10.5, 10, 1], [50, 50, 3]])
    m = match_timepoint_to_master(pres, pos, [0, 1], cur, 1, reg_first_positions=pos.copy())
    assert m.assignment.tolist() == [0, 2]
    np.testing.assert_array_equal(m.reg_presence[:, 1], [1, 0, 1])
    np.testing.assert_array_equal(m.reg_presence[2], [0, 1, 0])
    assert m.active_rows.tolist() == [0, 2]
    np.testing.assert_allclose(m.reg_positions[0], [10.5, 10, 1])
    np.testing.assert_allclose(m.reg_first_positions[0], [10, 10, 1])  # never updated
    # manual override: spine 1 is row 1; then spine 0 forced new
    m2 = match_timepoint_to_master(pres, pos, [0, 1], cur, 1, overrides={1: 1, 0: None},
                                   reg_first_positions=pos.copy())
    assert m2.assignment.tolist() == [2, 1]
    # conflict: both claim row 0 -> the later keeps it
    m3 = match_timepoint_to_master(pres, pos, [0, 1], cur, 1, overrides={1: 0},
                                   reg_first_positions=pos.copy())
    assert m3.assignment.tolist() == [2, 0]


def _sp(size, centers):
    return [MatStruct(indices=lin([r, r + 1], [c, c], [z, z], size).astype(float)[:, None])
            for r, c, z in centers]


def test_track_spines_lost_never_returns_and_export():
    region = np.array([[1, 1, 1], [20, 20, 1], [10, 10, 3]])
    size = (20, 20, 3)
    tp0 = [_sp(size, [(2, 2, 1), (10, 10, 2)])]
    tp1 = [_sp(size, [(2, 2, 1)])]  # spine 1 lost
    tp2 = [_sp(size, [(2, 2, 1), (10, 10, 2)])]  # reappears: new row
    tr = track_spines([tp0, tp1, tp2], [region])
    np.testing.assert_array_equal(tr.presence, [[1, 1, 1], [1, 0, 0], [0, 0, 1]])
    assert tr.region_idx[:, 0].tolist() == [0, 0, 0]
    exp = spine_tracking_export(tr, [tp0, tp1, tp2], [[region]] * 3)
    assert len(exp[0]) == 3
    np.testing.assert_array_equal(exp[0][2]["indices"], tp2[0][1]["indices"])
    assert exp[0][1]["presence"].shape == (1, 3)
    with pytest.raises(ValueError):
        track_spines([tp0], [region])
