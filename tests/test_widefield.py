"""Tests for calliopy.widefield (grid template, grid2vid, roi_image, plot_roi_values)."""

from __future__ import annotations

import numpy as np
import pytest
from calliopy.widefield import GridTemplate, grid2vid, plot_roi_values, roi_image
from scipy.io import savemat


@pytest.fixture
def template(tmp_path):
    # 3 ROIs: AP 2 (anterior) .. 0, ML -1 .. 1, in a 4 x 5 atlas image
    rois = np.zeros((1, 3), dtype=[("AP_index", "O"), ("ML_index", "O"), ("indices", "O"),
                                   ("name", "O"), ("Allen_acronym", "O")])
    rois[0, 0] = (2.0, -1.0, np.array([1.0, 2.0]), "left", "MOs")
    rois[0, 1] = (1.0, 0.0, np.array([6.0]), "left", "SSp")
    rois[0, 2] = (0.0, 1.0, np.array([20.0]), "right", "VISp")
    borders = np.zeros((4, 5), dtype=bool)
    borders[0, 4] = True
    path = tmp_path / "grid_template.mat"
    savemat(path, {"ROIs_transformed": rois, "areaBorders": borders})
    from calliopy.widefield import load_grid_template

    return load_grid_template(path)


def test_load_grid_template(template):
    assert isinstance(template, GridTemplate)
    assert template.n_roi == 3 and template.image_shape == (4, 5)
    assert np.array_equal(template.ap_index, [2, 1, 0]) and np.array_equal(template.ml_index, [-1, 0, 1])
    assert [list(i) for i in template.indices] == [[1, 2], [6], [20]]
    assert template.allen_acronym == ["MOs", "SSp", "VISp"]


def test_grid2vid_places_rois_anterior_on_top(template):
    v = grid2vid(np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]), template)
    assert v.shape == (3, 3, 2)
    expected = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]])
    assert np.array_equal(v[:, :, 0], expected) and np.array_equal(v[:, :, 1], 10 * expected)
    assert grid2vid(np.array([1.0, 2.0, 3.0]), template, resize_factor=2).shape == (6, 6, 1)
    with pytest.raises(ValueError, match="rows"):
        grid2vid(np.ones((2, 1)), template)


def test_roi_image_uses_column_major_indices_and_flips(template):
    img = roi_image([1.0, 2.0, 3.0], template)
    unflipped = img[:, ::-1]
    # MATLAB linear index 1, 2 -> (row 0, col 0), (row 1, col 0); 6 -> (1, 1); 20 -> (3, 4)
    assert unflipped[0, 0] == 1 and unflipped[1, 0] == 1 and unflipped[1, 1] == 2
    assert unflipped[3, 4] == 3
    assert np.isnan(unflipped[0, 1]) and np.sum(np.isfinite(img)) == 4


def test_plot_roi_values_runs(template):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    im = plot_roi_values([1.0, np.nan, 3.0], (0, 3), template=template, ax=ax,
                         cmap=[[0, 0, 0], [1, 1, 1]], border_color=(0.15, 0.15, 0.15), seed=0)
    assert im.get_clim() == (0, 3)
    plt.close(fig)
