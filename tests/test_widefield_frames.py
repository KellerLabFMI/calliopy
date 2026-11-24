"""Synthetic tests of calliopy.widefield._frames (register_frames, fftreg, shift_data)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.ndimage

from calliopy._matlab import matlab_cast
from calliopy.widefield._frames import fftreg, register_frames, shift_data

# displacement (rows, cols) of each frame relative to the first; 72 frames, so the template
# is renewed once (frame 50)
OFFSETS = [(0, 0), (3, -2), (-5, 4), (7, 7), (-1, -6), (2, 0)] * 12


def _scene(seed=0, shape=(200, 220)):
    rng = np.random.default_rng(seed)
    return scipy.ndimage.gaussian_filter(rng.normal(size=shape), 2.0) * 1000 + 5000


def _shifted_stack(offsets, h=120, w=130, seed=0):
    """frame[r, c] = template[r - dr, c - dc]: content moved dr rows down, dc columns right."""
    scene = _scene(seed)
    r0, c0 = 40, 45
    return np.stack([scene[r0 - dr : r0 - dr + h, c0 - dc : c0 - dc + w] for dr, dc in offsets],
                    axis=2)


# --- fftreg ---------------------------------------------------------------------------------


def test_fftreg_sign_and_wrap():
    rng = np.random.default_rng(1)
    img = rng.normal(size=(10, 12))
    f2 = np.fft.fft2(img)
    for dr, dc, exp in [(0, 0, (0, 0)), (2, -3, (2, -3)), (-4, 5, (-4, 5)), (4, 7, (4, -5)),
                        (5, 6, (-5, -6)), (-5, -6, (-5, -6)), (6, 1, (-4, 1))]:
        moved = np.roll(img, (dr, dc), axis=(0, 1))  # moved[r] = img[r - dr]
        res = fftreg(np.fft.fft2(moved), f2)
        assert (res.row_shift, res.col_shift) == exp
        assert isinstance(res.row_shift, float)
    # rows 0..fix(m/2)-1 are positive, the rest negative: m = 10 -> [-5, 4]; n = 12 -> [-6, 5]


def test_fftreg_ties_resolved_in_column_major_order():
    ref = np.zeros((6, 6))
    ref[0, 0] = 1.0
    img = np.zeros((6, 6))
    img[0, 1] = 1.0  # displacement (0, 1)
    img[1, 0] = 1.0  # displacement (1, 0): first in column-major order
    res = fftreg(np.fft.fft2(img), np.fft.fft2(ref))
    assert (res.row_shift, res.col_shift) == (1.0, 0.0)
    with pytest.raises(ValueError):
        fftreg(np.zeros((4, 4)), np.zeros((4, 5)))


# --- shift_data -----------------------------------------------------------------------------


def test_shift_data_hand_computed():
    a = np.arange(1, 13, dtype=np.float64).reshape(3, 4)  # [[1..4],[5..8],[9..12]]
    stack = np.stack([a, a + 100], axis=2)
    out = shift_data(stack, [1, 0], [0, -1])
    pad = np.median(stack[0, 0, :])  # 51
    # frame 0: out[r, c] = a[r + 1, c]
    np.testing.assert_array_equal(out[:, :, 0], [[5, 6, 7, 8], [9, 10, 11, 12], [pad] * 4])
    # frame 1: out[r, c] = a[r, c - 1] + 100
    np.testing.assert_array_equal(out[:, :, 1], [[pad, 101, 102, 103], [pad, 105, 106, 107],
                                                 [pad, 109, 110, 111]])
    assert out.dtype == np.float64


def test_shift_data_rounding_types_and_shapes():
    a = np.arange(1, 13, dtype=np.uint16).reshape(3, 4)
    stack = np.stack([a, a + 1], axis=2)  # corner values 1 and 2 -> median 1.5 -> uint16 2
    out = shift_data(stack, 0.5, -0.5)  # scalar, rounded away from zero to (1, -1)
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out[:, :, 0], [[2, 5, 6, 7], [2, 9, 10, 11], [2, 2, 2, 2]])
    # a single 2-D frame stays 2-D; shifts larger than the frame give only padding
    np.testing.assert_array_equal(shift_data(a, 0, 0), a)
    assert np.all(shift_data(a, 5, 0) == 1)
    np.testing.assert_array_equal(shift_data(stack, np.array([[0], [0]]), [0, 0]), stack)
    with pytest.raises(ValueError):
        shift_data(stack, [1, 2, 3], 0)
    with pytest.raises(ValueError):
        shift_data(stack, [np.nan, 0], 0)


def test_shift_data_matches_scipy_shift():
    rng = np.random.default_rng(2)
    stack = rng.normal(size=(9, 11, 4))
    sx, sy = [2, -3, 0, 4], [-1, 5, 0, 2]
    out = shift_data(stack, sx, sy)
    pad = np.median(stack[0, 0, :])
    for i in range(4):
        ref = scipy.ndimage.shift(stack[:, :, i], (-sx[i], -sy[i]), order=0, mode="constant",
                                  cval=pad)
        np.testing.assert_array_equal(out[:, :, i], ref)


# --- register_frames ------------------------------------------------------------------------


@pytest.mark.parametrize("as_uint16", [False, True])
def test_register_frames_recovers_known_offsets(as_uint16):
    stack = _shifted_stack(OFFSETS)
    if as_uint16:
        stack = matlab_cast(stack, np.uint16)  # wid_data class; template wid_data(:,:,1)
    res = register_frames(stack, stack[:, :, 0])
    assert res.shift_x.shape == (len(OFFSETS), 1) and res.shift_y.shape == (len(OFFSETS), 1)
    assert res.shift_x.dtype == np.float64
    np.testing.assert_array_equal(np.c_[res.shift_x, res.shift_y], np.array(OFFSETS, float))
    # shift_data with the returned shifts re-aligns the stack (away from the padded border)
    aligned = shift_data(stack, res.shift_x, res.shift_y)
    ref = np.repeat(stack[10:-10, 10:-10, :1], len(OFFSETS), axis=2)
    np.testing.assert_array_equal(aligned[10:-10, 10:-10, :], ref)


def test_register_frames_default_template_is_the_mean():
    stack = _shifted_stack(OFFSETS[:6])
    res = register_frames(stack)
    explicit = register_frames(stack, stack.mean(axis=2))
    np.testing.assert_array_equal(res.shift_x, explicit.shift_x)
    np.testing.assert_array_equal(res.shift_y, explicit.shift_y)
    # relative to the mean the offsets are recovered up to one common shift
    d = np.c_[res.shift_x, res.shift_y] - np.array(OFFSETS[:6], float)
    np.testing.assert_array_equal(d, np.repeat(d[:1], 6, axis=0))


def test_register_frames_refresh_and_single_frame():
    # a stack of exactly 50 frames runs the refresh at the last frame
    stack = _shifted_stack(OFFSETS[:50], seed=3)
    res = register_frames(stack, stack[:, :, 0])
    np.testing.assert_array_equal(np.c_[res.shift_x, res.shift_y], np.array(OFFSETS[:50], float))
    one = register_frames(stack[:, :, 3:4], stack[:, :, 0])
    assert (one.shift_x[0, 0], one.shift_y[0, 0]) == OFFSETS[3]
    # float32 data: default template is float32, shifts unchanged
    res32 = register_frames(stack[:, :, :6].astype(np.float32), stack[:, :, 0].astype(np.float32))
    np.testing.assert_array_equal(np.c_[res32.shift_x, res32.shift_y], np.array(OFFSETS[:6], float))


def test_register_frames_boundary_and_errors():
    # wide frames: round(0.1*200) = 20 >= 10 rows, so the border is round(0.1*10) = 1
    rng = np.random.default_rng(4)
    flat = rng.normal(size=(10, 200, 2))
    assert register_frames(flat).shift_x.shape == (2, 1)
    with pytest.raises(ValueError):
        register_frames(rng.normal(size=(4, 4, 2)), perc_boundary=0.4)  # nothing left
    with pytest.raises(ValueError):
        register_frames(flat, np.zeros((10, 199)))
    with pytest.raises(ValueError):
        register_frames(flat[:, :, 0])
