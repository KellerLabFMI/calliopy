"""Tests of calliopy.widefield._templates and _template_builders (ROI templates).

Synthetic tests always run. The reference tests compare with the template files that MATLAB
wrote (calliope 12_Widefield/ROItemplate_library/*.mat and Plotting_templates/*.mat); they
need CALLIOPY_TESTDATA with a sub-folder ``widefield_templates`` holding those files plus the
Allen data of Build_code (``annotation_25_mid_ctxOnly.mat``, ``structures.json``), all in one
folder, and are skipped otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.io

from calliopy.io import MatStruct
from calliopy.widefield import _template_builders as tb
from calliopy.widefield import _templates as tp
from calliopy.widefield._transform import fit_similarity, warp_image

# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _flat(v):
    return np.asarray(v, dtype=np.float64).ravel(order="F")


def _rois(s):
    return list(np.asarray(s["ROIs"], dtype=object).ravel())


def _const_allen(label=1.0):
    flat = np.full((528, 456), label)
    onto = [{"id": int(label), "name": "area one", "acronym": "A1"}]
    return flat, onto


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------


def test_square_roi_indices_hand_checked():
    # rows 2..4 (l_w - D_BORDER = 3 rows), columns 3..4 (2 columns), column-major, 1-based
    idx = tp._square_roi_indices(2, 3, 5, 2, 10)
    assert idx.tolist() == [22.0, 23.0, 24.0, 32.0, 33.0, 34.0]


_SMALL = {"w_hem": 1.0, "l_back": 0.2, "l_front": 0.6, "n_w": 10, "tan_ant": 1.35,
          "tan_post": 0.5, "l_corner": 0.18, "d_border": 3}


def test_cortex_grid_small_hand_checked():
    # lbd = 100, l_w = 20, N_l = 18; x ticks 50:20:230 (rows), y ticks 20:20:360 (columns)
    size = (400, 400)
    rois = tp.cortex_grid(size, (40, 150), (140, 150), **_SMALL)
    assert rois
    ml = np.array([float(r["ML_index"]) for r in rois])
    ap = np.array([float(r["AP_index"]) for r in rois])
    for r, m, a in zip(rois, ml, ap):
        rr, cc = tp._ind2sub(size, _flat(r["indices"]))
        assert rr.size == 17 * 16
        assert rr.min() == 150 - 20 * m and cc.min() == 40 + 20 * a
        # independent evaluation of the three accepting conditions at the corner (x, y)
        x, y = 150 - 20 * m, 40 + 20 * a
        d = abs(x - 150 + 10)
        assert y - 40 <= -1.35 * d + np.ceil(100 * (1 + 1.35 * 1.0))
        assert y >= 40 or y - 40 >= -0.5 * d
        assert y - 40 >= 100 * 0.18 - 1.35 * d
    # MATLAB order: grid row (ML index decreasing) outer, grid column (AP increasing) inner
    assert np.all(np.diff(-ml) >= 0)
    assert all(np.all(np.diff(ap[ml == m]) > 0) for m in np.unique(ml))
    # the accepted set is symmetric about the midline (ML m <-> 1 - m, abs shifted by l_w/2)
    pairs = {(m, a) for m, a in zip(ml, ap)}
    assert all((1 - m, a) in pairs for m, a in pairs)


def test_cortex_grid_kept_shift_bug_gives_empty_ap_index():
    # lambda at y = 40, ticks 28, 39, 50, ...: shift = 1 moves them to 27, 38, ... (away from
    # lambda), no tick equals lambda and AP_index becomes MATLAB's 1x0 empty
    rois = tp.cortex_grid((200, 200), (40, 100), (100, 100), w_hem=0.9, l_back=0.2,
                          l_front=0.6, n_w=10, tan_ant=1.35, tan_post=0.5, l_corner=0.18,
                          d_border=3)
    assert rois and all(np.asarray(r["AP_index"]).shape == (1, 0) for r in rois)
    assert all(np.asarray(r["ML_index"]).ndim == 0 for r in rois)


def test_cortex_grid_boundary_rois_get_empty_indices():
    rois = tp.cortex_grid((120, 120), (20, 60), (80, 60), w_hem=0.95, l_back=0.25,
                          l_front=0.75, n_w=8, tan_ant=1.35, tan_post=0.5, l_corner=0.18,
                          d_border=3)
    empties = [r for r in rois if np.asarray(r["indices"]).size == 0]
    assert empties and all(np.asarray(r["indices"]).shape == (0, 0) for r in empties)


def test_cranial_window_grid_inside_circle():
    rois = tp.cranial_window_grid((100, 100), (10, 50), (90, 50), n_w=8, d_border=2)
    # diam 80, centre (50, 50), l_w 10
    for r in rois:
        rr, cc = tp._ind2sub((100, 100), _flat(r["indices"]))
        x, y = rr.min(), cc.min()
        assert np.hypot(y + 5 - 50, x + 5 - 50) <= 40
    assert any(float(r["ML_index"]) == 0 and float(r["AP_index"]) == 0 for r in rois)


def test_angles_from_top_left_analytic():
    size = (5, 5)
    pts = np.array([[3, 4], [2, 3], [3, 2], [4, 3], [2, 4], [4, 2]])  # [row, col]
    ang = tp.angles_from_top_left(pts, size, (3, 3))
    np.testing.assert_allclose(ang, [0, 90, 180, -90, 45, -135])
    lin = (pts[:, 1] - 1) * 5 + pts[:, 0]
    np.testing.assert_array_equal(tp.angles_from_top_left(lin, size, (3, 3)), ang)
    np.testing.assert_array_equal(tp.angles_from_top_left(lin[:, None], size, (3, 3)), ang)
    with pytest.raises(ValueError):
        tp.angles_from_top_left(np.ones((2, 3)), size, (3, 3))


def test_eeg_discs_and_rings():
    discs, centres = tp.eeg_electrode_discs((60, 60), (30, 30), r_inner=None, r_outer=5,
                                            el_x=(0.1,), el_y=(-0.05,))
    assert centres == [(20.0, 25.0)]  # x_loc = round(-10 + 30), y_loc = round(-5 + 30)
    rr, cc = tp._ind2sub((60, 60), discs[0])
    assert np.all((rr - 20) ** 2 + (cc - 25) ** 2 <= 25)
    assert discs[0].size == sum(
        1 for a in range(-5, 6) for b in range(-5, 6) if a * a + b * b <= 25
    )
    assert np.all(np.diff(discs[0]) > 0)  # MATLAB find order
    ring, _ = tp.eeg_electrode_discs((60, 60), (30, 30), r_inner=3, r_outer=5,
                                     el_x=(0.1,), el_y=(-0.05,))
    theta = tp.angles_from_top_left(ring[0], (60, 60), centres[0])
    sectors = [ring[0][(theta >= lb) & (theta <= ub)] for lb, ub in tp.EEG_ANGLES]
    allsec = np.concatenate(sectors)
    assert np.unique(allsec).size == allsec.size
    # pixels on the two axes through the centre belong to no sector
    r, c = tp._ind2sub((60, 60), np.setdiff1d(ring[0], allsec))
    assert np.all((r == 20) | (c == 25))


def test_eeg_ring_rois_layout():
    rois = tp.eeg_ring_rois((1000, 1000), (632, 500))
    assert len(rois) == 120
    assert [float(r["ML_index"]) for r in rois[:5]] == [1, 1, 1, 1, 2]
    assert [float(r["AP_index"]) for r in rois[:5]] == [1, 2, 3, 4, 1]
    assert rois[59]["name"] == "right hemisphere" and rois[60]["name"] == "left hemisphere"


# --------------------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------------------


def test_warp_allen_map_keeps_uniform_labels_exact():
    # the label count in assign_allen_areas needs exact labels inside uniform areas
    lab = np.full((40, 50), 182305689.0)
    lab[:, 25:] = 182305690.0
    out = tp.warp_allen_map(lab, [(5, 7), (30, 20)], [(8, 3), (40, 35)], (60, 70))
    vals = np.unique(out)
    assert {182305689.0, 182305690.0} <= set(vals)
    inside = out[(out != 0)]
    frac = inside[(inside != 182305689.0) & (inside != 182305690.0)]
    assert frac.size < 0.1 * inside.size  # only the border column is mixed


def test_transform_mask_matches_warp_image():
    rng = np.random.default_rng(3)
    size_o, size_t = (60, 50), (90, 80)
    a = fit_similarity(np.array([[10.0, 12], [40, 45]]), np.array([[20.0, 70], [60, 15]]))
    for _ in range(5):
        r0, c0 = rng.integers(5, 40), rng.integers(5, 30)
        mask = np.zeros(size_o)
        mask[r0 : r0 + 6, c0 : c0 + 4] = 1
        mask[r0 + 2, c0 + 5] = 1
        lin0 = np.flatnonzero(mask.ravel(order="F"))
        got = tp._warp_mask_nonzero(lin0, size_o, a, size_t)
        ref = np.flatnonzero(warp_image(mask, a, size_t).ravel(order="F") != 0)
        np.testing.assert_array_equal(got, ref)


def _toy_template(size=(40, 40)):
    rois = tp.reference_rois(size, [(5, 5), (30, 5)], ["P", "A"])
    idx = np.array([r + (c - 1) * size[0] for c in (10, 11) for r in (12, 13, 14)], float)
    rois.append(tp.new_roi(idx.reshape(-1, 1)))
    return tp._struct_array(rois)


def test_transform_rois_identity_and_references():
    rois = _toy_template()
    out = tp.transform_rois_from_template(rois, (40, 40), (40, 40), (5, 5), (30, 5))
    out = list(out.ravel())
    assert float(out[0]["indices"]) == 5 + 4 * 40 and float(out[1]["indices"]) == 5 + 29 * 40
    np.testing.assert_array_equal(_flat(out[2]["indices"]), _flat(list(rois.ravel())[2]["indices"]))


def test_transform_rois_translation_and_padding():
    rois = _toy_template()
    out = list(tp.transform_rois_from_template(rois, (40, 40), (60, 70), (25, 15), (50, 15)).ravel())
    r, c = tp._ind2sub((60, 70), _flat(out[2]["indices"]))
    assert (r.min(), r.max(), c.min(), c.max()) == (22, 24, 30, 31)


def test_transform_rois_outside_warns_empty():
    # translation by (0, -4): the ROI (rows 12-14) lands on rows 8-10, outside 5 rows
    rois = _toy_template()
    with pytest.warns(UserWarning):
        out = list(tp.transform_rois_from_template(rois, (40, 40), (5, 40), (5, 1),
                                                   (30, 1)).ravel())
    assert np.asarray(out[2]["indices"]).shape == (0, 0)
    with pytest.raises(ValueError):  # MATLAB sub2ind fails for a target point outside
        tp.transform_rois_from_template(rois, (40, 40), (20, 20), (500, 5), (525, 5))


def test_area_borders_step_and_independent_sobel():
    img = np.zeros((20, 20))
    img[:, 10:] = 5.0
    b = tp.allen_area_borders(img)
    assert b[:, 9:11].all() and not b[:, :8].any() and not b[:, 12:].any()
    rng = np.random.default_rng(4)
    img = rng.integers(0, 3, (15, 17)).astype(float)
    p = np.pad(img, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    # sigma 0.1: the Gaussian side taps are ~2e-22, so the result is the plain magnitude
    np.testing.assert_array_equal(tp.allen_area_borders(img), np.hypot(gx, gy) > 0.1)


# --------------------------------------------------------------------------------------
# Allen
# --------------------------------------------------------------------------------------


def test_allen_flat_map_first_nonzero_and_fix():
    vol = np.zeros((4, 3, 5), dtype=np.uint32)
    vol[0, 2, 0] = 7
    vol[0, 1, 0] = 9  # first along axis 1 wins
    vol[3, 0, 4] = 2
    flat = tp.allen_flat_map(vol)
    assert flat.shape == (4, 5) and flat.dtype == np.float64
    assert flat[0, 0] == 9 and flat[3, 4] == 2 and flat.sum() == 11
    big = np.zeros((200, 2, 300), dtype=np.uint8)
    assert tp.allen_flat_map(big)[178, 258] == 993
    assert tp.allen_flat_map(big, fix_pixel=False)[178, 258] == 0


def test_assign_allen_areas_majority_and_ties():
    flat = np.zeros((50, 50))
    flat[:, :25] = 10
    flat[:, 25:] = 20
    onto = [{"id": 10, "name": "ten", "acronym": "T"}, {"id": 20, "name": "twenty",
                                                        "acronym": "W"}]
    size = (50, 50)
    refs = tp.reference_rois(size, [(5, 5), (40, 5)], ["a", "b"])
    def roi(cols, ml):
        idx = [r + (c - 1) * 50 for c in cols for r in range(20, 24)]
        return tp.new_roi(np.array(idx, float).reshape(-1, 1), ml_index=ml)
    # columns 1-25 carry 10, 26-50 carry 20; the third ROI (columns 24-27) is a 2:2 tie
    rois = refs + [roi(range(20, 24), 1), roi(range(25, 30), -1), roi(range(24, 28), 0)]
    out = list(tp.assign_allen_areas(rois, flat, onto, [(5, 5), (40, 5)], [(5, 5), (40, 5)],
                                     size, name_hemisphere=True).ravel())
    # tie: MATLAB groupcounts sorts ascending and max takes the first -> 10 ("T")
    assert [o["Allen_acronym"] for o in out[2:]] == ["T", "W", "T"]
    assert out[4]["Allen_index"] == 10.0 and out[0]["Allen_area"] == ""
    assert [o["name"] for o in out[2:]] == ["right hemisphere", "left hemisphere",
                                           "left hemisphere"]


def test_first_is_zero_based():
    flat = np.full((50, 50), 10.0)
    onto = [{"id": 10, "name": "ten", "acronym": "T"}]
    refs = tp.reference_rois((50, 50), [(5, 5), (40, 5)], ["a", "b"])
    rois = refs + [tp.new_roi(np.array([[300.0], [301.0]]))]
    pts = [(5, 5), (40, 5)]
    out = list(tp.assign_allen_areas(rois, flat, onto, pts, pts, (50, 50), first=1).ravel())
    assert out[0]["Allen_acronym"] == "" and out[1]["Allen_acronym"] == "T"
    assert out[2]["Allen_acronym"] == "T"
    out = list(tp.assign_allen_areas(rois, flat, onto, pts, pts, (50, 50)).ravel())
    assert [o["Allen_acronym"] for o in out] == ["", "", "T"]  # default first=2
    img = tp.roi_template_colour_image(out, flat, (50, 50), first=2)
    assert img.reshape(-1, 3, order="F").any(axis=1).sum() == 2  # only ROI 2's pixels
    img0 = tp.roi_template_colour_image(out, flat, (50, 50), first=0)
    assert img0.reshape(-1, 3, order="F").any(axis=1).sum() == 2  # refs have no Allen_index


def test_assign_allen_areas_outside_raises():
    flat = np.zeros((50, 50))
    refs = tp.reference_rois((50, 50), [(5, 5), (40, 5)], ["a", "b"])
    rois = refs + [tp.new_roi(np.array([[300.0]]))]
    with pytest.raises(ValueError, match="outside of the Allen map"):
        tp.assign_allen_areas(rois, flat, [], [(5, 5), (40, 5)], [(5, 5), (40, 5)], (50, 50))


def test_colour_image_partition():
    flat, onto = _const_allen()
    s = tb.generate_template_eeg_circles_lambda_bregma(flat, onto, rng=np.random.default_rng(0))
    img = s["img_ROIs"]
    assert img.shape == (1000, 1000, 3)
    mask = np.zeros(10**6, bool)
    for r in _rois(s)[2:]:
        mask[(_flat(r["indices"]) - 1).astype(int)] = True
    np.testing.assert_array_equal(img.any(axis=2), mask.reshape(1000, 1000, order="F"))
    assert len(np.unique(img[mask.reshape(1000, 1000, order="F")], axis=0)) == 1


# --------------------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------------------


def test_grid_whole_cortex_lambda_bregma_struct():
    flat, onto = _const_allen()
    s = tb.generate_template_grid_whole_cortex_lambda_bregma(flat, onto)
    assert list(s.keys()) == ["name", "description", "comments", "code_to_generate",
                              "parameters", "ROIs", "img_ROIs"]
    rois = _rois(s)
    assert s["ROIs"].shape == (1, len(rois))
    assert rois[0]["name"] == "Lambda" and float(rois[0]["indices"]) == 500 + 199 * 1000
    assert float(rois[1]["indices"]) == 500 + 631 * 1000
    for r in rois[2:]:
        assert r["Allen_acronym"] == "A1" and r["Allen_index"] == 1.0
        right = float(r["ML_index"]) >= 1
        assert r["name"] == ("right hemisphere" if right else "left hemisphere")
        assert np.asarray(r["indices"]).shape[1] == 1
        # l_w = 27: square corner = (500 - 27*ML, 200 + 27*AP) (row, column)
        rr, cc = tp._ind2sub((1000, 1000), _flat(r["indices"]))
        assert rr.min() == 500 - 27 * float(r["ML_index"])
        assert cc.min() == 200 + 27 * float(r["AP_index"])
    p = s["parameters"]
    assert p["SIZE_TEMPLATE"].shape == (1, 2) and p["N_W"] == 30.0
    assert "dir_ROItemplate" not in s


def test_grid_icv_fake_bregma_same_grid():
    flat, onto = _const_allen()
    a = tb.generate_template_grid_whole_cortex_lambda_bregma(flat, onto)
    b = tb.generate_template_grid_whole_cortex_lambda_icv(flat, onto)
    assert _rois(b)[1]["name"] == "Inferior Cerebral Vein"
    ra, rb = _rois(a)[2:], _rois(b)[2:]
    assert len(ra) == len(rb)
    assert all(np.array_equal(_flat(x["indices"]), _flat(y["indices"])) for x, y in zip(ra, rb))
    assert "VEC_ICV" in b["parameters"] and "VEC_BREGMA" not in b["parameters"]


def test_grid_plus_eeg_order():
    flat, onto = _const_allen()
    s = tb.generate_template_grid_plus_eeg_circles_lambda_bregma(flat, onto)
    rois = _rois(s)
    assert len(rois) == 2 + 120 + len(tp.cortex_grid((1000, 1000), (200, 500), (632, 500),
                                                     **tb._GRID_DEFAULTS))
    assert rois[2]["name"] == "right hemisphere" and rois[122]["name"] == ""


def test_only_eeg_type_bug_kept():
    flat, onto = _const_allen()
    s = tb.generate_template_only_eeg_electrode_positions_lambda_bregma(flat, onto)
    rois = _rois(s)
    assert len(rois) == 32
    assert rois[2]["type"] == 0.0 and np.asarray(rois[3]["type"]).shape == (0, 0)
    assert all(np.asarray(r["AP_index"]).size == 0 for r in rois)


def test_cranial_window_generator():
    s = tb.generate_template_grid_cranial_window()
    rois = _rois(s)
    assert rois[0]["name"] == "Posterior Reference Point"
    img = s["img_ROIs"]
    assert img.shape == (500, 500) and set(np.unique(img)) == {0.0, 1.0}
    union = np.unique(np.concatenate([_flat(r["indices"]) for r in rois]))
    assert img.sum() == union.size
    np.testing.assert_array_equal(np.flatnonzero(img.ravel(order="F")) + 1, union)
    grid = np.concatenate([_flat(r["indices"]) for r in rois[2:]])
    assert np.unique(grid).size == grid.size  # grid squares do not overlap
    sizes = {_flat(r["indices"]).size for r in rois[2:]}
    assert sizes == {18 * 17}


def test_14rois_lambda_bregma_square_and_circle():
    s = tb.generate_template_14rois_lambda_bregma()
    rois = _rois(s)
    assert len(rois) == 16 and rois[2]["name"] == "leftV1"
    assert rois[2]["Allen_area"] == "left primary visual area"
    assert rois[3]["Allen_acronym"] == "right VISp"
    idx = _flat(rois[2]["indices"])
    assert idx.size == 21 * 21 and idx[0] == (238 - 1) * 1107 + 916
    assert np.all(np.diff(idx[:21]) == 1)
    assert set(np.unique(s["img_ROIs"])) == {0.0, 0.5}
    assert "W_HEM" not in s["parameters"]
    c = tb.generate_template_14rois_lambda_bregma(shape="circle")
    assert _flat(_rois(c)[2]["indices"]).size == 20 * 20 - 2 * (6 + 4 + 2)
    with pytest.raises(ValueError):
        tb.generate_template_14rois_lambda_bregma(shape="hexagon")


def test_14rois_cortex_mirror_and_size():
    s = tb.generate_template_14rois_cortex_lambda_bregma()
    p = s["parameters"]
    assert p["SIZE_TEMPLATE"].tolist() == [[814.0, 814.0]]
    rois = _rois(s)
    assert list(rois[0].keys())[-1] == "Allen_desc" and np.asarray(rois[0]["Allen_desc"]).size == 0
    size = (814, 814)
    bregma_row = p["VEC_BREGMA"][0, 1]
    for left, right in zip(rois[2::2], rois[3::2]):
        rl, cl = tp._ind2sub(size, _flat(left["indices"]))
        rr, cr = tp._ind2sub(size, _flat(right["indices"]))
        assert np.array_equal(np.unique(cl), np.unique(cr))  # same AP columns
        assert rl.size == rr.size == 441
        # mirror about bregma's row: centre_right = 2*VEC_BREGMA(2) - centre_left
        assert (rl.mean() + rr.mean()) / 2 == bregma_row


def test_plot_roi_template_returns_figure():
    from matplotlib.figure import Figure

    fig = tp.plot_roi_template(tb.generate_template_grid_cranial_window())
    assert isinstance(fig, Figure)


def test_make_plotting_grid_synthetic():
    flat = np.full((528, 456), 3.0)
    tmpl = tb.generate_template_grid_cranial_window()
    out = tp.make_plotting_grid_rois_template(tmpl, flat, "cranial window")
    assert list(out.keys()) == ["areaBorders", "ROIs_transformed"]
    assert out["areaBorders"].dtype == bool and out["areaBorders"].shape == (1500, 1500)
    raw = list(tp.transform_rois_from_template(tmpl["ROIs"], (500, 500), (1500, 1500),
                                               (750, 1200), (750, 200)).ravel())[2:]
    got = _rois({"ROIs": out["ROIs_transformed"]})
    assert len(got) == len(raw) == 298
    d, n = 5, 1500
    for a, b in zip(raw[:5], got[:5]):
        ia = _flat(a["indices"])
        exp = np.unique(np.concatenate([ia + s for s in (0, d * n, d, d * n + d, -d * n, -d,
                                                         -d * n - d, -d * n + d, d * n - d)]))
        np.testing.assert_array_equal(_flat(b["indices"]), exp)
    with pytest.raises(ValueError):
        tp.make_plotting_grid_rois_template(tmpl, flat, "both")


def test_load_roi_template_roundtrip(tmp_path):
    rois = np.zeros((1, 2), dtype=[("indices", object), ("name", object)])
    rois[0, 0] = (np.array([[5.0]]), "Lambda")
    rois[0, 1] = (np.array([[1.0], [2.0], [3.0]]), "x")
    scipy.io.savemat(tmp_path / "t.mat", {"struct_ROIs": {
        "name": "toy", "parameters": {"SIZE_TEMPLATE": np.array([[10.0, 10.0]])},
        "ROIs": rois}})
    s = tp.load_roi_template(tmp_path / "t.mat")
    assert isinstance(s, MatStruct) and s["name"] == "toy"
    r = _rois(s)
    assert r[0]["name"] == "Lambda" and r[0]["indices"] == 5.0
    assert _flat(r[1]["indices"]).tolist() == [1, 2, 3]
    assert s["parameters"]["SIZE_TEMPLATE"].shape == (1, 2)


def test_load_allen_ontology(tmp_path):
    (tmp_path / "s.json").write_text('[{"id": 1, "name": "a", "acronym": "A"}]')
    assert tp.load_allen_ontology(tmp_path / "s.json")[0]["acronym"] == "A"


# --------------------------------------------------------------------------------------
# reference: MATLAB-made template files
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tdir(testdata):
    d = testdata / "widefield_templates"
    if not d.is_dir():
        pytest.skip("no widefield_templates folder in CALLIOPY_TESTDATA")
    return d


@pytest.fixture(scope="module")
def allen(tdir):
    a, s = tdir / "annotation_25_mid_ctxOnly.mat", tdir / "structures.json"
    if not (a.is_file() and s.is_file()):
        pytest.skip("Allen annotation / structures.json not in widefield_templates")
    return tp.load_allen_flat_map(a), tp.load_allen_ontology(s)


def _stored(tdir, name):
    p = tdir / name
    if not p.is_file():
        pytest.skip(f"{name} not in widefield_templates")
    return tp.load_roi_template(p)


def _same(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return np.array_equal(_flat(a), _flat(b))


def _assert_template_equal(gen, st):
    for k in ("name", "description", "comments", "code_to_generate"):
        assert gen[k] == st[k], k
    assert set(gen["parameters"]) == set(st["parameters"])
    for k in st["parameters"]:
        assert _same(gen["parameters"][k], st["parameters"][k]), k
    g, s = _rois(gen), _rois(st)
    assert len(g) == len(s)
    for i, (a, b) in enumerate(zip(g, s), start=1):
        assert list(a.keys()) == list(b.keys()), i
        for f in b:
            assert _same(a[f], b[f]), (i, f)


_ALLEN_GENERATORS = {
    "Grid_whole_cortex_based_on_lambda-bregma.mat":
        tb.generate_template_grid_whole_cortex_lambda_bregma,
    "Grid_whole_cortex_based_on_lambda-inferior_cerebral_vein.mat":
        tb.generate_template_grid_whole_cortex_lambda_icv,
    "Ring_Rois_around_electrode_positions_based_on_lambda-bregma.mat":
        tb.generate_template_eeg_circles_lambda_bregma,
    "EEG_electrode_positions_based_on_lambda-bregma.mat":
        tb.generate_template_only_eeg_electrode_positions_lambda_bregma,
    "Grid_whole_cortex_plus_Ring_Rois_around_electrode_positions_based_on_lambda-bregma.mat":
        tb.generate_template_grid_plus_eeg_circles_lambda_bregma,
}


@pytest.mark.parametrize("name", sorted(_ALLEN_GENERATORS))
def test_reference_allen_generators(tdir, allen, name):
    st = _stored(tdir, name)
    gen = _ALLEN_GENERATORS[name](*allen, rng=np.random.default_rng(0))
    _assert_template_equal(gen, st)
    assert np.array_equal(gen["img_ROIs"].any(axis=2), np.asarray(st["img_ROIs"]).any(axis=2))


def test_reference_cranial_window(tdir):
    st = _stored(tdir, "Grid_for_circular_cranial_window.mat")
    gen = tb.generate_template_grid_cranial_window()
    _assert_template_equal(gen, st)
    np.testing.assert_array_equal(gen["img_ROIs"], np.asarray(st["img_ROIs"], float))


def test_reference_14rois_cortex(tdir):
    st = _stored(tdir, "14_cortical_regions_of_interest_based_on_lambda-bregma.mat")
    gen = tb.generate_template_14rois_cortex_lambda_bregma()
    _assert_template_equal(gen, st)
    np.testing.assert_array_equal(gen["img_ROIs"], st["img_ROIs"])


def test_reference_stim_points_file_starts_with_icv_grid(tdir, allen):
    st = _stored(
        tdir,
        "Grid_whole_cortex_based_on_lambda-inferior_cerebral_vein_stim_points_"
        "128_676_338_177_374_384_193_667_623.mat",
    )
    gen = tb.generate_template_grid_whole_cortex_lambda_icv(*allen)
    s = _rois(st)
    assert len(s) == 829 and all(r["name"].startswith("stim_target") for r in s[820:])
    _assert_template_equal(gen, MatStruct({**st, "ROIs": st["ROIs"][:, :820]}))


def _plotting(tdir, name):
    p = tdir / name
    if not p.is_file():
        pytest.skip(f"{name} not in widefield_templates")
    return tp.load_plotting_grid_template(p)


def test_reference_plotting_grid_whole_cortex(tdir, allen):
    st = _plotting(tdir, "818_ROIs_grid_template.mat")
    tmpl = _stored(tdir, "Grid_whole_cortex_based_on_lambda-inferior_cerebral_vein.mat")
    gen = tp.make_plotting_grid_rois_template(tmpl, allen[0], "whole cortex")
    np.testing.assert_array_equal(gen["areaBorders"], st["areaBorders"].astype(bool))
    g, s = _rois({"ROIs": gen["ROIs_transformed"]}), _rois({"ROIs": st["ROIs_transformed"]})
    assert len(g) == len(s) == 818
    for a, b in zip(g, s):
        for f in b:
            assert _same(a[f], b[f]), f


@pytest.fixture(scope="module")
def cranial_plotting(tdir, allen):
    st = _plotting(tdir, "cranial_window_grid_template.mat")
    tmpl = _stored(tdir, "Grid_for_circular_cranial_window.mat")
    return tp.make_plotting_grid_rois_template(tmpl, allen[0], "cranial window"), st


def test_reference_plotting_grid_cranial_window_borders_and_subset(cranial_plotting):
    gen, st = cranial_plotting
    np.testing.assert_array_equal(gen["areaBorders"], st["areaBorders"].astype(bool))
    g, s = _rois({"ROIs": gen["ROIs_transformed"]}), _rois({"ROIs": st["ROIs_transformed"]})
    assert len(g) == len(s) == 298
    for a, b in zip(g, s):
        ia, ib = _flat(a["indices"]), _flat(b["indices"])
        assert np.isin(ia, ib).all()  # MATLAB keeps at most one more boundary row/column
        ra, ca = tp._ind2sub((1500, 1500), ia)
        rb, cb = tp._ind2sub((1500, 1500), ib)
        assert ra.min() - rb.min() in (0, 1) and cb.max() - ca.max() in (0, 1)


@pytest.mark.xfail(strict=True, reason="knife-edge pixels at scale 2.5: MATLAB's imwarp keeps "
                   "one more boundary row/column; not reproduced (see report)")
def test_reference_plotting_grid_cranial_window_exact(cranial_plotting):
    gen, st = cranial_plotting
    g, s = _rois({"ROIs": gen["ROIs_transformed"]}), _rois({"ROIs": st["ROIs_transformed"]})
    assert all(_same(a["indices"], b["indices"]) for a, b in zip(g, s))
