"""Synthetic tests of the private MATLAB built-in equivalents in calliopy._matlab."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.signal

from calliopy._matlab import (
    conv2,
    corr2,
    filter2,
    imgaussfilt,
    linspace,
    matlab_cast,
    mround,
    normxcorr2,
    prctile,
)

A3 = np.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]])


# --- mround / matlab_cast -----------------------------------------------------------------


def test_mround_halves_away_from_zero():
    x = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.4999, -2.4999, 0.49999999999999994])
    np.testing.assert_array_equal(mround(x), [1, 2, 3, -1, -2, -3, 2, -2, 0])
    assert mround(2.5) == 3.0
    assert np.isnan(mround(np.nan))
    assert mround(np.inf) == np.inf
    assert mround(np.float32(0.5)).dtype == np.float32
    np.testing.assert_array_equal(mround(np.array([3, -4])), [3, -4])


def test_matlab_cast_rounds_and_saturates():
    np.testing.assert_array_equal(
        matlab_cast([3 - 5.4, 70000.4, np.nan, 2.5, 3.49], np.uint16), [0, 65535, 0, 3, 3]
    )
    assert matlab_cast(np.array([300], np.uint16), np.uint8)[0] == 255
    assert matlab_cast(np.array([-3], np.int16), np.uint8)[0] == 0
    assert matlab_cast(np.array([200], np.uint8), np.uint64)[0] == 200
    assert matlab_cast(-2.5, np.int16) == -3
    assert matlab_cast(np.array([1.25]), np.float32).dtype == np.float32
    np.testing.assert_array_equal(matlab_cast([0.0, 0.2, -1], bool), [False, True, True])
    with pytest.raises(ValueError):
        matlab_cast([np.nan], bool)


# --- linspace -------------------------------------------------------------------------------


def test_linspace_matlab_arithmetic():
    y = linspace(0, 1, 10)
    k = np.arange(10.0)
    np.testing.assert_array_equal(y, k * 1.0 / 9)  # product first, then division
    # numpy's d1 + k*step differs in the last bit here (7/9 vs 7*(1/9))
    assert y[7] == 7.0 / 9 and y[7] != np.linspace(0, 1, 10)[7]
    y = linspace(-3.3, 7.9, 13)
    np.testing.assert_array_equal(y[1:-1], (-3.3 + np.arange(13.0) * (7.9 + 3.3) / 12)[1:-1])
    assert y[0] == -3.3 and y[-1] == 7.9  # end points exact
    np.testing.assert_array_equal(linspace(2, 2, 4), [2, 2, 2, 2])
    np.testing.assert_array_equal(linspace(1, 5, 1), [5.0])
    assert linspace(1, 5, 0).shape == (0,) and linspace(1, 5, -3).shape == (0,)
    np.testing.assert_array_equal(linspace(0, 4, 5.7), [0, 1, 2, 3, 4])  # n floored
    assert linspace(0, 1).size == 100
    big = np.finfo(float).max
    y = linspace(-big, big, 5)  # d2 - d1 overflows: divide first
    assert np.all(np.isfinite(y)) and y[2] == 0.0 and y[0] == -big and y[-1] == big
    y = linspace(0, big / 2, 5)  # (d2 - d1)*(n - 2) overflows
    assert np.all(np.isfinite(y)) and y[-1] == big / 2


# --- prctile --------------------------------------------------------------------------------


def _prctile_by_definition(v, p):
    """MATLAB's documented rule: sorted values at 100*(0.5:n-0.5)/n, clamp outside."""
    v = np.sort(v[~np.isnan(v)])
    n = v.size
    q = np.r_[0, 100 * (np.arange(n) + 0.5) / n, 100]
    xx = np.r_[v[0], v, v[-1]]
    return np.interp(p, q, xx)


def test_prctile_matches_definition():
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 7, 50):
        v = rng.normal(size=n)
        for p in (0, 1, 5, 25, 50, 75, 95, 99, 100):
            assert prctile(v, p) == pytest.approx(_prctile_by_definition(v, p), rel=1e-12)
    # hand computed: n=4 values 1..4 sit at 12.5, 37.5, 62.5, 87.5 %
    assert prctile([4, 1, 3, 2], 50) == pytest.approx(2.5)
    assert prctile([4, 1, 3, 2], 25) == pytest.approx(1.5)
    assert prctile([4, 1, 3, 2], 95) == 4.0
    assert prctile([4, 1, 3, 2], 10) == 1.0


def test_prctile_ignores_nan_and_shapes():
    v = np.array([np.nan, 3.0, 1.0, np.nan, 2.0])
    assert prctile(v, 50) == 2.0
    assert np.isnan(prctile([np.nan, np.nan], 50))
    assert np.isnan(prctile(np.zeros(0), 50))
    np.testing.assert_allclose(prctile(v, [0, 50, 100]), [1, 2, 3])
    x = np.arange(24.0).reshape(2, 3, 4)
    assert prctile(x, 50, axis=1).shape == (2, 1, 4)
    assert prctile(x, [10, 50, 90], axis=2).shape == (2, 3, 3)
    np.testing.assert_allclose(prctile(x, [10, 90], axis=2)[1, 2], _prctile_by_definition(
        x[1, 2], np.array([10, 90])))
    # axis=None works over x(:) (column-major order does not change percentiles)
    assert prctile(x, 30) == pytest.approx(_prctile_by_definition(x.ravel(), 30))
    assert prctile(np.ones(3, np.float32), 50).dtype == np.float32
    assert prctile(np.array([1, 2, 3, 4], np.uint16), 50) == 2.5
    with pytest.raises(ValueError):
        prctile(v, 101)
    with pytest.raises(ValueError):
        prctile(v, [])


# --- corr2 ----------------------------------------------------------------------------------


def test_corr2():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(6, 5))
    b = 0.3 * a + rng.normal(size=(6, 5))
    assert corr2(a, b) == pytest.approx(np.corrcoef(a.ravel(), b.ravel())[0, 1], rel=1e-12)
    assert corr2(a, -2 * a + 7) == pytest.approx(-1.0)
    assert corr2(np.arange(6, dtype=np.uint8).reshape(2, 3), np.arange(6).reshape(2, 3)) == (
        pytest.approx(1.0))
    assert math.isnan(corr2(np.ones((3, 3)), a[:3, :3]))
    with pytest.raises(ValueError):
        corr2(a, b.T)


# --- conv2 / filter2 ------------------------------------------------------------------------


def test_conv2_2x2_kernel_hand_computed():
    a = np.array([[1.0, 2], [3, 4]])
    k = np.ones((2, 2))
    np.testing.assert_array_equal(conv2(a, k), [[1, 3, 2], [4, 10, 6], [3, 7, 4]])
    np.testing.assert_array_equal(conv2(a, k, "same"), [[10, 6], [7, 4]])
    np.testing.assert_array_equal(conv2(a, k, "valid"), [[10]])
    # asymmetric 2x2 kernel: the 'same' window starts at row/column 1 of the full result
    k = np.array([[1.0, 0], [0, 0]])
    np.testing.assert_array_equal(conv2(A3, k, "same"), [[5, 6, 0], [8, 9, 0], [0, 0, 0]])
    # scipy's "same" is one sample earlier for even kernels: this is what conv2 must not do
    assert not np.array_equal(scipy.signal.convolve2d(A3, k, "same"), conv2(A3, k, "same"))


def test_conv2_3x3_kernel_hand_computed():
    k = np.zeros((3, 3))
    k[1, 2] = 1.0  # convolution moves the image one column to the right
    np.testing.assert_array_equal(conv2(A3, k, "same"), [[0, 1, 2], [0, 4, 5], [0, 7, 8]])
    full = conv2(A3, k)
    assert full.shape == (5, 5)
    np.testing.assert_array_equal(full[1:4, 2:5], A3)
    np.testing.assert_array_equal(conv2(A3, k, "valid"), [[4]])
    k2 = np.arange(1.0, 10).reshape(3, 3)
    # odd kernel: MATLAB and scipy 'same' agree
    np.testing.assert_allclose(conv2(A3, k2, "same"), scipy.signal.convolve2d(A3, k2, "same"))
    # centre value by hand: sum(a .* rot180(k))
    assert conv2(A3, k2, "same")[1, 1] == np.sum(A3 * k2[::-1, ::-1])


def test_conv2_sizes_types_and_cross_check():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(7, 9))
    for kshape in [(1, 1), (2, 3), (4, 4), (3, 6), (9, 2)]:
        k = rng.normal(size=kshape)
        full = conv2(a, k)
        np.testing.assert_allclose(full, scipy.signal.fftconvolve(a, k), atol=1e-12)
        r0, c0 = kshape[0] // 2, kshape[1] // 2
        np.testing.assert_array_equal(conv2(a, k, "same"), full[r0 : r0 + 7, c0 : c0 + 9])
        assert conv2(a, k, "valid").shape == (max(7 - kshape[0] + 1, 0), max(9 - kshape[1] + 1, 0))
    assert conv2(a[:2, :2], np.ones((3, 3)), "valid").shape == (0, 0)
    assert conv2(a.astype(np.float32), np.ones((2, 2)), "same").dtype == np.float32
    assert conv2(np.arange(6, dtype=np.uint16).reshape(2, 3), [[1, 1]]).dtype == np.float64
    np.testing.assert_array_equal(conv2([1, 2, 3], [1, 1]), [[1, 3, 5, 3]])  # row vectors
    with pytest.raises(ValueError):
        conv2(a, np.ones(2), "middle")


def test_filter2_is_correlation():
    k = np.array([[1.0, 0], [0, 0]])
    np.testing.assert_array_equal(filter2(k, A3), A3)
    np.testing.assert_array_equal(filter2(k[::-1, ::-1], A3), [[5, 6, 0], [8, 9, 0], [0, 0, 0]])
    k = np.zeros((3, 3))
    k[1, 2] = 1.0  # correlation reads the right neighbour
    np.testing.assert_array_equal(filter2(k, A3), [[2, 3, 0], [5, 6, 0], [8, 9, 0]])
    rng = np.random.default_rng(3)
    a = rng.normal(size=(8, 6))
    k = rng.normal(size=(3, 3))
    np.testing.assert_allclose(filter2(k, a), scipy.signal.correlate2d(a, k, "same"), atol=1e-12)
    np.testing.assert_allclose(filter2(k, a, "valid"), scipy.signal.correlate2d(a, k, "valid"),
                               atol=1e-12)
    # local mean as in register_widefield_exp: ones(w)/w^2 on an integer image
    img = np.full((10, 10), 4, dtype=np.uint16)
    out = filter2(np.ones((4, 4)) / 16, img)
    assert out.dtype == np.float64
    assert out[4, 4] == pytest.approx(4.0)
    assert out[0, 0] == pytest.approx(4.0 * 9 / 16)  # even kernel: 3x3 of the 4x4 overlaps


# --- imgaussfilt ----------------------------------------------------------------------------


def _gauss_reference(img, sr, sc):
    rr, rc = math.ceil(2 * sr), math.ceil(2 * sc)
    gr = np.exp(-0.5 * (np.arange(-rr, rr + 1) / sr) ** 2)
    gc = np.exp(-0.5 * (np.arange(-rc, rc + 1) / sc) ** 2)
    k = np.outer(gr / gr.sum(), gc / gc.sum())
    padded = np.pad(img, ((rr, rr), (rc, rc)), mode="edge")
    return conv2(padded, k, "valid")


def test_imgaussfilt_against_explicit_kernel():
    rng = np.random.default_rng(4)
    img = rng.normal(size=(15, 12))
    for s in (0.1, 0.5, 1.3, 2.0):
        np.testing.assert_allclose(imgaussfilt(img, s), _gauss_reference(img, s, s), atol=1e-12)
    np.testing.assert_allclose(imgaussfilt(img, (0.8, 2.2)), _gauss_reference(img, 0.8, 2.2),
                               atol=1e-12)
    # kernel support is 2*ceil(2*sigma)+1 taps: an impulse spreads over exactly that many
    imp = np.zeros((21, 21))
    imp[10, 10] = 1.0
    out = imgaussfilt(imp, 1.3)
    rows = np.flatnonzero(out[:, 10] > 0)
    assert rows.min() == 10 - 3 and rows.max() == 10 + 3
    assert out.sum() == pytest.approx(1.0)


def test_imgaussfilt_stack_types_and_replicate_border():
    rng = np.random.default_rng(5)
    stack = rng.normal(size=(9, 8, 3))
    out = imgaussfilt(stack, 1.0)
    for i in range(3):
        np.testing.assert_allclose(out[:, :, i], imgaussfilt(stack[:, :, i], 1.0), atol=1e-12)
    np.testing.assert_allclose(imgaussfilt(np.full((5, 5), 7.0), 2.0), 7.0)  # replicate border
    u = (rng.random((10, 10)) * 1000).astype(np.uint16)
    uo = imgaussfilt(u, 1.0)
    assert uo.dtype == np.uint16
    np.testing.assert_array_equal(uo, matlab_cast(imgaussfilt(u.astype(float), 1.0), np.uint16))
    assert imgaussfilt(stack.astype(np.float32), 1.0).dtype == np.float32
    with pytest.raises(ValueError):
        imgaussfilt(stack, 0)


# --- normxcorr2 -----------------------------------------------------------------------------


def _normxcorr2_brute(t, a):
    m, n = t.shape
    big = np.pad(a, ((m - 1, m - 1), (n - 1, n - 1)))
    t0 = t - t.mean()
    out = np.zeros((a.shape[0] + m - 1, a.shape[1] + n - 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            w = big[i : i + m, j : j + n]
            w0 = w - w.mean()
            d = math.sqrt(np.sum(w0 * w0) * np.sum(t0 * t0))
            out[i, j] = np.sum(w0 * t0) / d if d > 1e-9 else 0.0
    return out


def test_normxcorr2_against_brute_force():
    rng = np.random.default_rng(6)
    for (m, n), (mm, nn) in [((3, 4), (9, 7)), ((2, 2), (5, 6)), ((5, 1), (8, 3)), ((6, 7), (6, 7))]:
        t = rng.normal(size=(m, n))
        a = rng.normal(size=(mm, nn)) + 3.0
        c = normxcorr2(t, a)
        assert c.shape == (mm + m - 1, nn + n - 1)
        np.testing.assert_allclose(c, _normxcorr2_brute(t, a), atol=1e-9)


def test_normxcorr2_peak_and_flat_regions():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(20, 25))
    t = a[5:11, 8:12].copy()  # top-left corner at 0-based (5, 8)
    c = normxcorr2(t, a)
    i, j = np.unravel_index(np.argmax(c), c.shape)
    assert (i - t.shape[0] + 1, j - t.shape[1] + 1) == (5, 8)
    assert c[i, j] == pytest.approx(1.0, abs=1e-9)
    flat = np.zeros((10, 10))
    flat[6:, 6:] = rng.normal(size=(4, 4))
    c = normxcorr2(rng.normal(size=(3, 3)), flat)
    assert np.all(c[:5, :5] == 0)  # windows that only see zeros
    with pytest.raises(ValueError):
        normxcorr2(np.ones((2, 2)), a)
    with pytest.raises(ValueError):
        normxcorr2(a, t)
