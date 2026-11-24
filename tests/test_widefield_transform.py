"""Synthetic tests of calliopy.widefield._transform (MATLAB simtform2d / fitgeotform2d / imwarp)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.ndimage
import scipy.optimize

from calliopy.widefield._transform import (
    fit_similarity,
    similarity_matrix,
    similarity_params,
    transform_points,
    warp_image,
)


def _asym(h=7, w=9):
    """Image whose every pixel value is distinct (row*100 + col + 1), so any flip shows."""
    r, c = np.mgrid[0:h, 0:w]
    return (r * 100 + c + 1).astype(np.float64)


# --- matrices -------------------------------------------------------------------------------


def test_similarity_matrix_and_params_round_trip():
    a = similarity_matrix(2.0, 30.0, (5.0, -3.0))
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    np.testing.assert_allclose(a, [[2 * c, -2 * s, 5], [2 * s, 2 * c, -3], [0, 0, 1]], rtol=1e-15)
    p = similarity_params(a)
    assert p.scale == pytest.approx(2.0)
    assert p.rotation_angle == pytest.approx(30.0)
    np.testing.assert_array_equal(p.translation, [5.0, -3.0])
    for ang in (-179.0, -90.0, 0.0, 45.0, 135.0, 180.0):
        assert similarity_params(similarity_matrix(0.7, ang, (0, 0))).rotation_angle == (
            pytest.approx(ang))


def test_similarity_matrix_exact_at_right_angles():
    np.testing.assert_array_equal(similarity_matrix(1, 90, (0, 0)), [[0, -1, 0], [1, 0, 0],
                                                                      [0, 0, 1]])
    np.testing.assert_array_equal(similarity_matrix(1, 180, (0, 0))[:2, :2], [[-1, 0], [0, -1]])
    np.testing.assert_array_equal(similarity_matrix(1, -90, (0, 0))[:2, :2], [[0, 1], [-1, 0]])
    np.testing.assert_array_equal(similarity_matrix(1, 450, (0, 0)), similarity_matrix(1, 90, (0, 0)))


def test_similarity_params_rejects_non_similarity():
    with pytest.raises(ValueError):
        similarity_params([[1, 0, 0], [0, -1, 0], [0, 0, 1]])  # reflection
    with pytest.raises(ValueError):
        similarity_params([[1, 0.2, 0], [0, 1, 0], [0, 0, 1]])  # shear
    with pytest.raises(ValueError):
        similarity_params(np.eye(2))
    with pytest.raises(ValueError):
        similarity_matrix(0, 0, (0, 0))


def test_transform_points():
    a = similarity_matrix(1.5, 90, (10, 20))
    pts = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    # x' = -1.5*y + 10, y' = 1.5*x + 20
    np.testing.assert_array_equal(transform_points(a, pts), [[10, 21.5], [7, 20], [4, 24.5]])
    np.testing.assert_array_equal(transform_points(a, [1.0, 0.0]), [10, 21.5])
    back = transform_points(np.linalg.inv(a), transform_points(a, pts))
    np.testing.assert_allclose(back, pts, atol=1e-12)
    with pytest.raises(ValueError):
        transform_points(a, np.zeros((3, 3)))
    with pytest.raises(ValueError):
        transform_points([[1, 0, 0], [0, 1, 0], [0.1, 0, 1]], pts)  # projective


# --- fitting --------------------------------------------------------------------------------


def test_fit_similarity_two_points_exact():
    true = similarity_matrix(1.3, -25.0, (40.0, -7.5))
    moving = np.array([[230.0, 372.0], [229.0, 78.0]])
    fixed = transform_points(true, moving)
    np.testing.assert_allclose(fit_similarity(moving, fixed), true, atol=1e-10)
    # the call in make_plotting_grid_ROIs_template.m: two anatomical points onto a target frame
    a = fit_similarity([[230, 372], [229, 78]], [[750, 1200], [750, 200]])
    np.testing.assert_allclose(transform_points(a, [[230, 372], [229, 78]]),
                               [[750, 1200], [750, 200]], atol=1e-9)
    p = similarity_params(a)
    assert p.scale == pytest.approx(1000 / math.hypot(1, 294))


def test_fit_similarity_least_squares_against_optimizer():
    rng = np.random.default_rng(0)
    true = similarity_matrix(0.8, 12.0, (3.0, 9.0))
    moving = rng.uniform(1, 200, size=(8, 2))
    fixed = transform_points(true, moving) + rng.normal(scale=0.5, size=(8, 2))
    a = fit_similarity(moving, fixed)

    def resid(q):
        return (transform_points(similarity_matrix(q[0], q[1], q[2:]), moving) - fixed).ravel()

    q = scipy.optimize.least_squares(resid, [1.0, 0.0, 0.0, 0.0], xtol=1e-15, ftol=1e-15).x
    np.testing.assert_allclose(a, similarity_matrix(q[0], q[1], q[2:]), rtol=1e-7, atol=1e-7)
    # noiseless points of a mirror image still give a non-reflective similarity
    mirrored = moving * [-1, 1]
    assert np.linalg.det(fit_similarity(moving, mirrored)[:2, :2]) >= 0


def test_fit_similarity_axis_aligned_pairs_are_exact():
    # case from the ROI templates: rotation by -90 degrees, scale 2.5, integer translation
    a = fit_similarity([[50, 250], [450, 250]], [[750, 1200], [750, 200]])
    assert a[0, 0] == 0.0 and a[1, 1] == 0.0
    assert a[0, 1] == 2.5 and a[1, 0] == -2.5
    assert a[0, 2] == 125.0 and a[1, 2] == 1325.0
    # pure scale + translation and pure translation, exact as well
    a = fit_similarity([[10, 20], [30, 20]], [[5, 7], [65, 7]])
    np.testing.assert_array_equal(a, [[3, 0, -25], [0, 3, -53], [0, 0, 1]])
    # N = 3: centroids such as 4/3 are not representable, so allow 1e-13
    a = fit_similarity([[1, 2], [3, 5], [8, 1]], [[4, 0], [6, 3], [11, -1]])
    np.testing.assert_allclose(a, [[1, 0, 3], [0, 1, -2], [0, 0, 1]], rtol=0, atol=1e-13)
    # N > 2 with noise: within 1e-13 of the closed-form least-squares route via lstsq
    rng = np.random.default_rng(3)
    mv = rng.uniform(1, 500, size=(6, 2))
    fx = transform_points(similarity_matrix(1.7, 33.0, (12.0, -4.0)), mv) + rng.normal(size=(6, 2))
    x, y = mv[:, 0], mv[:, 1]
    one, zero = np.ones(6), np.zeros(6)
    m = np.vstack([np.column_stack([x, -y, one, zero]), np.column_stack([y, x, zero, one])])
    s = np.linalg.lstsq(m, np.r_[fx[:, 0], fx[:, 1]], rcond=None)[0]
    ref = np.array([[s[0], -s[1], s[2]], [s[1], s[0], s[3]], [0, 0, 1]])
    np.testing.assert_allclose(fit_similarity(mv, fx), ref, rtol=1e-9, atol=1e-9)


def test_fit_similarity_errors():
    with pytest.raises(ValueError):
        fit_similarity([[1, 2]], [[3, 4]])
    with pytest.raises(ValueError):
        fit_similarity([[1, 2], [1, 2]], [[3, 4], [5, 6]])
    with pytest.raises(ValueError):
        fit_similarity(np.zeros((3, 2)) + [[0, 0], [1, 0], [0, 1]], np.zeros((2, 2)))


# --- warping --------------------------------------------------------------------------------


def test_warp_integer_translation_moves_pixels_exactly():
    img = _asym()
    a = similarity_matrix(1, 0, (2, 1))  # x (column) +2, y (row) +1
    out = warp_image(img, a, img.shape)
    expected = np.zeros_like(img)
    expected[1:, 2:] = img[:-1, :-2]
    np.testing.assert_array_equal(out, expected)
    out0 = warp_image(img, a, img.shape, order=0, fill_value=-1)
    expected[0, :] = -1
    expected[:, :2] = -1
    np.testing.assert_array_equal(out0, expected)
    # larger output view: the image lands at the translated place
    big = warp_image(img, a, (10, 12))
    np.testing.assert_array_equal(big[1:8, 2:11], img)
    assert big[0].sum() == 0 and big[8:].sum() == 0


def test_warp_rotation_90_equals_rot90():
    sq = _asym(7, 7)
    centre = (4.0, 4.0)  # (x, y) of the centre pixel, 1-based
    rot = similarity_matrix(1, 90, (0, 0))
    a = np.array([[1, 0, centre[0]], [0, 1, centre[1]], [0, 0, 1]]) @ rot @ np.array(
        [[1, 0, -centre[0]], [0, 1, -centre[1]], [0, 0, 1]])
    np.testing.assert_array_equal(warp_image(sq, a, sq.shape), np.rot90(sq, -1))
    # rectangular image, rotation about pixel (x0, y0): rot90 shifted by (x0-y0, H+1-x0-y0)
    img = _asym(5, 8)
    h, w = img.shape
    x0, y0 = 3, 2
    a = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]]) @ rot @ np.array(
        [[1, 0, -x0], [0, 1, -y0], [0, 0, 1]])
    out = warp_image(img, a, (12, 12), fill_value=np.nan)
    rr = np.rot90(img, -1)  # (w, h)
    dr, dc = x0 - y0, h + 1 - x0 - y0
    for r in range(12):
        for c in range(12):
            i, j = r + dr, c + dc
            if 0 <= i < w and 0 <= j < h:
                assert out[r, c] == rr[i, j]
            else:
                assert np.isnan(out[r, c])


def test_warp_then_inverse_returns_interior():
    yy, xx = np.mgrid[0:60, 0:70]
    img = np.sin(xx / 7.0) + np.cos(yy / 5.0) + 0.01 * xx  # smooth: linear interpolation is close
    a = similarity_matrix(1.05, 17.0, (4.3, -2.1))
    back = warp_image(warp_image(img, a, img.shape, fill_value=np.nan), np.linalg.inv(a), img.shape,
                      fill_value=np.nan)
    interior = (slice(15, 45), slice(15, 55))
    assert not np.isnan(back[interior]).any()
    # two linear interpolations of a function with curvature <= 1/25 per px^2
    np.testing.assert_allclose(back[interior], img[interior], atol=0.03)
    # exact case: 90-degree rotation plus integer translation and back
    img2 = _asym(9, 9)
    a = similarity_matrix(1, 90, (11, 0))  # maps the 9x9 image into the 9x12 view
    back = warp_image(warp_image(img2, a, (9, 12)), np.linalg.inv(a), img2.shape)
    np.testing.assert_array_equal(back, img2)


def test_warp_cross_check_with_affine_transform():
    rng = np.random.default_rng(1)
    img = rng.normal(size=(30, 40))
    a = similarity_matrix(0.9, -33.0, (6.0, 8.0))
    out = warp_image(img, a, (35, 30), fill_value=np.nan)
    ai = np.linalg.inv(a)
    # independent route: 0-based (row, col) matrix for scipy
    m = np.array([[ai[1, 1], ai[1, 0]], [ai[0, 1], ai[0, 0]]])
    off = np.array([ai[1, 0] + ai[1, 1] + ai[1, 2] - 1, ai[0, 0] + ai[0, 1] + ai[0, 2] - 1])
    ref = scipy.ndimage.affine_transform(img, m, off, output_shape=(35, 30), order=1,
                                         mode="constant", cval=np.nan)
    both = ~np.isnan(out) & ~np.isnan(ref)
    assert both.sum() > 300
    np.testing.assert_allclose(out[both], ref[both], atol=1e-12)


def test_warp_linear_reproduces_constant_regions_exactly():
    a = similarity_matrix(1.07, 17.0, (2.3, -1.1))
    img = np.full((20, 25), 182305689.0)
    out = warp_image(img, a, img.shape, fill_value=np.nan)
    vals = out[~np.isnan(out)]
    assert vals.size > 200
    assert np.all(vals == 182305689.0)
    # label image: two large labels; output pixels whose four source neighbours carry one
    # label must hold that label exactly (majority counts on label images depend on it)
    lab = np.full((30, 40), 182305689.0)
    lab[:, 20:] = 182305701.0
    out = warp_image(lab, a, lab.shape, fill_value=np.nan)
    ai = np.linalg.inv(a)
    yy, xx = np.mgrid[1:31, 1:41].astype(float)
    src_c = ai[0, 0] * xx + ai[0, 1] * yy + ai[0, 2] - 1
    left = ~np.isnan(out) & (np.floor(src_c) + 1 <= 19)
    right = ~np.isnan(out) & (np.floor(src_c) >= 20)
    assert left.sum() > 100 and right.sum() > 100
    assert np.all(out[left] == 182305689.0)
    assert np.all(out[right] == 182305701.0)
    mixed = ~np.isnan(out) & ~left & ~right
    assert np.all((out[mixed] >= 182305689.0) & (out[mixed] <= 182305701.0))
    assert warp_image(lab.astype(np.float32), a, lab.shape)[left].dtype == np.float32


def test_warp_types_and_stacks():
    img = _asym()
    stack = np.stack([img, 2 * img, -img], axis=2)
    a = similarity_matrix(1.2, 10.0, (1.5, 0.5))
    out = warp_image(stack, a, (8, 10, 3))
    assert out.shape == (8, 10, 3)
    for i in range(3):
        np.testing.assert_array_equal(out[:, :, i], warp_image(stack[:, :, i], a, (8, 10)))
    u = img.astype(np.uint16)
    uo = warp_image(u, a, img.shape)
    assert uo.dtype == np.uint16
    f = warp_image(img, a, img.shape)
    np.testing.assert_array_equal(uo, np.where(f >= 0, np.floor(f + 0.5), 0).astype(np.uint16))
    assert warp_image(img.astype(np.float32), a, img.shape).dtype == np.float32
    with pytest.raises(ValueError):
        warp_image(img, a, img.shape, order=3)
    with pytest.raises(ValueError):
        warp_image(img, np.eye(2), img.shape)
