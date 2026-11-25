"""Tests for calliopy.viewstack (get_rot_ind, view_stack)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from calliopy.viewstack import get_rot_ind, view_stack
from matplotlib.backend_bases import KeyEvent


@pytest.fixture(autouse=True)
def _close():
    plt.switch_backend("Agg")
    yield
    plt.close("all")


def test_get_rot_ind_no_tilt_is_the_frame():
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4, order="F")
    idx = get_rot_ind(data.shape, 2, 0, 0)
    frame = data.ravel(order="F")[idx].reshape(2, 3, order="F")
    np.testing.assert_array_equal(frame, data[:, :, 2])


def test_get_rot_ind_hand_computed_tilt():
    # dims (4, 2, 5), cf 2 (MATLAB 3), tx 1: rows 0-1 from frame cf+0, rows 2-3 from cf+1
    idx = get_rot_ind((4, 2, 5), 2, 1, 0)
    np.testing.assert_array_equal(idx, [16, 17, 26, 27, 20, 21, 30, 31])
    # ty -1 on (2, 4, 3), cf 0: columns 0-1 shift -1 -(-1) = 0 ... clipped at 0
    idx = get_rot_ind((2, 4, 3), 0, 0, -1)
    assert idx.min() >= 0 and idx.max() < 24


def test_get_rot_ind_clips_to_stack():
    idx = get_rot_ind((4, 4, 2), 1, 3, 0)
    assert idx.max() == 4 * 4 * 2 - 1 and idx.min() >= 0


def _press(v, key):
    v.fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", v.fig.canvas, key=key))


def test_viewer_frames_labels_and_keys():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(8, 10, 21))
    v = view_stack(data, frame_rate=10, frame_zero=5, stack_lims=(-1, 1))
    assert v.clim == [-1, 1] and v.frame_index == 0
    _press(v, "right")
    assert v.frame_index == 1 and v.text.get_text() == "-1 s"
    v.set_frame(20)
    assert v.text.get_text() == "1 s"
    np.testing.assert_array_equal(v.image.get_array(), data[:, :, 20])
    _press(v, "shift+up")
    assert v.clim[1] == pytest.approx(1.1)
    _press(v, "ctrl+up")
    assert v.tx == 1 and "Tilt X: 1" in v.text.get_text()
    _press(v, "ctrl+down")
    _press(v, "up")
    assert v.playback_spacing == 2
    _press(v, "g")
    _press(v, "x")
    v.set_frame(3)
    np.testing.assert_array_equal(v.curr_frame, data[:, :, 3])


def test_viewer_default_label_and_averaging():
    data = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
    v = view_stack(data, nbr_of_frames_to_average=2.5)  # floored to 2
    assert v.nbr_avg_frames == 2
    v.set_frame(5)
    np.testing.assert_allclose(v.image.get_array(), data[:, :, 5])  # only one frame left
    v.set_frame(2)
    np.testing.assert_allclose(v.image.get_array(), data[:, :, 2:4].mean(axis=2))
    assert v.text.get_text() == "2"
    v2 = view_stack(data, frame_rate=1)
    v2.set_frame(0)
    assert v2.text.get_text() == "1 s"  # MATLAB default FrameZero = 0 with 1-based frames


def test_viewer_tools_and_marks():
    data = np.zeros((10, 10, 5))
    data[2:4, 3:6, :] = np.arange(5)
    v = view_stack(data, adata=np.linspace(-1, 1, 5))
    assert v.adata.min() == 0 and v.adata.max() == 1
    tr = v.select_area((2, 3), (3, 5))
    np.testing.assert_allclose(tr, np.arange(5))
    assert len(v.raw_traces) == 1
    v.set_frame(2)
    m = v.add_mark(4, 7)
    assert m == np.ravel_multi_index((4, 7, 2), data.shape, order="F")
    assert v.delete_mark(5, 8) and v.marks == []
    v.clear()
    assert v.raw_traces == []
    assert v.show_template() is not None
    with pytest.raises(ValueError):
        view_stack(np.zeros((2, 2, 2, 2)))


def test_viewer_playback_step_wraps():
    v = view_stack(np.zeros((3, 3, 4)))
    v.set_frame(3)
    v.step()
    assert v.frame_index == 0
