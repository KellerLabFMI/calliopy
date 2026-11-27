"""RoiQuest and ActiQuest drawing on matplotlib Axes with the Agg backend."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from test_roiquest_data import BLOBS_Z0, WF_REF, H, W, disk, make_2p, make_wf1

from calliopy.roiquest import (
    ActiQuestSession,
    ActiQuestView,
    HistogramView,
    Redraw,
    RoiQuestSession,
    RoiQuestView,
    colormap,
    histogram_data,
    overlay_image,
    roi_number_labels,
)

FLAT = np.array([0.93, 0.13, 0.55])


@pytest.fixture
def two_p(tmp_path):
    return make_2p(tmp_path / "Adata-S1-T101.mat")


def _view(session, size=(W, H), with_hist=False):
    fig = Figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    FigureCanvasAgg(fig)
    hist = None
    if with_hist:
        hfig = Figure()
        FigureCanvasAgg(hfig)
        hist = HistogramView(hfig.add_subplot(), session)
    ax = fig.add_axes((0, 0, 1, 1))
    v = RoiQuestView(ax, session, histogram=hist)
    return fig, v


def test_overlay_flat_and_selection(two_p):
    s = RoiQuestSession.from_adata(two_p)
    fig, v = _view(s)
    fig.canvas.draw()
    ov = v.overlay.get_array()
    roi = disk((H, W), BLOBS_Z0[0], 4)
    np.testing.assert_allclose(ov[roi][:, :3], np.tile(FLAT, (roi.sum(), 1)))
    np.testing.assert_allclose(ov[roi][:, 3], 0.35)
    assert np.all(ov[~roi][:, 3] == 0)
    s.select_all()
    v.refresh(Redraw.OVERLAY)
    np.testing.assert_allclose(v.overlay.get_array()[roi][:, 3], 0.9)
    s.toggle_roi_visible()
    v.refresh(s.toggle_numbers())
    assert np.all(v.overlay.get_array()[..., 3] == 0) and not v.info_text.get_visible()


def test_rendered_pixel_colour(two_p):
    """On the rendered canvas, an ROI pixel is the flat colour blended over the image."""
    s = RoiQuestSession.from_adata(two_p)
    s.set_opacity(1.0)
    fig, v = _view(s)
    v.ax.set_aspect("auto")
    v.info_text.set_visible(False)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3] / 255.0
    assert buf.shape == (H, W, 3)
    r, c = BLOBS_Z0[0]
    np.testing.assert_allclose(buf[r, c], FLAT, atol=2 / 255)
    s.set_opacity(0.0)
    v.refresh(Redraw.OVERLAY)
    v.info_text.set_visible(False)  # its dark box covers the top rows of this tiny figure
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3] / 255.0
    g = s.imgs[0][0][r, c] / 255.0  # gray colormap: value k -> k/255
    np.testing.assert_allclose(buf[r, c], [g, g, g], atol=2 / 255)


def test_overlay_random_and_tags(two_p):
    s = RoiQuestSession.from_adata(two_p)
    roi = disk((H, W), BLOBS_Z0[0], 4)
    s.set_color_mode("random")
    rgba = overlay_image(s).rgba
    k = s.rois[0][0].rand_color_idx
    np.testing.assert_allclose(rgba[roi][0, :3], s.lab_cmap[k - 1])
    s.set_color_mode("tags")
    s.select_all()
    s.tag_selected(3)
    rgba = overlay_image(s).rgba
    np.testing.assert_allclose(rgba[roi][0, :3], s.lab_cmap[4])  # type 3 -> labCmap(5,:)
    assert rgba[0, 0, 3] == 0


def test_overlay_zoom_and_putative(two_p):
    s = RoiQuestSession.from_adata(two_p)
    fig, v = _view(s, with_hist=True)
    r, c = BLOBS_Z0[1]
    s.button_down(r - 8, c - 8)
    v.refresh(s.button_up(r + 8, c + 8))
    assert v.ax.get_xlim() == (c - 8, c + 8) and v.ax.get_ylim() == (r + 8, r - 8)
    ov = overlay_image(s)
    blob = disk((H, W), BLOBS_Z0[1], 4)
    np.testing.assert_allclose(ov.rgba[blob][0, :3], [0.0, 0.94, 0.14])
    assert ov.rgba[disk((H, W), BLOBS_Z0[0], 4)][:, 3].max() == 0  # saved ROI hidden
    assert v.info_text.get_text() == f"ExpID: 101 - Z:1 - {int(blob.sum())} pixels"
    assert v.info_text.get_position() == (c - 8, r - 8)
    counts, edges = histogram_data(s)
    assert counts.sum() == 17 * 17 and edges[-1] == 255
    np.testing.assert_array_equal(v.histogram.bars.get_data().values, counts)
    s.threshold_step("up")
    v.refresh(Redraw.HIST_THRESHOLD)
    assert v.histogram.threshold_line.get_xdata()[0] == 159.0
    v.refresh(s.escape())
    assert v.ax.get_xlim() == (-0.5, W - 0.5)
    assert v.histogram.bars.get_data().values.sum() == H * W
    fig.canvas.draw()


def test_numbers_vectors_colormap_display(two_p):
    s = RoiQuestSession.from_adata(two_p)
    fig, v = _view(s)
    v.refresh(s.toggle_numbers())
    labels = roi_number_labels(s)
    r, c = BLOBS_Z0[0]
    assert labels == [(float(c - 4), float(r), "1")]  # leftmost column, smallest index
    assert v.number_texts[0].get_text() == "1" and v.number_texts[0].get_visible()
    s.vectors = (np.array([[20.0, 20.0]]), np.array([[1.0, 2.0]]))
    s.show_vectors = True
    v.refresh(Redraw.OVERLAY)
    assert v.quiver is not None and not s.show_vectors and v.quiver.get_visible()
    v.refresh(Redraw.OVERLAY)
    assert not v.quiver.get_visible()
    v.refresh(s.set_colormap("hot"))
    np.testing.assert_allclose(v.image.get_cmap()(1.0)[:3], colormap("hot")[-1])
    v.refresh(s.cycle_image("+"))
    np.testing.assert_array_equal(v.image.get_array(), s.imgs[1][0])
    assert v.ax.get_aspect() == 1.5
    fig.canvas.draw()


def test_widefield_ref_cross_colours(tmp_path):
    s = RoiQuestSession.from_adata(make_wf1(tmp_path / "Adata-S1-T401.mat"))
    fig, v = _view(s, size=(90, 70))
    assert v.ax.get_aspect() == 1.0
    centre = WF_REF[0]
    rgba = v.overlay.get_array()
    np.testing.assert_allclose(rgba[centre][:3], [1.0, 1.0, 0.0])  # MATLAB double image clip
    s.rois[0] = []
    np.testing.assert_allclose(overlay_image(s).rgba[centre][:3], [0.8, 1.0, 0.0])
    s.rois = [[]]
    s.button_down(*centre)
    # selected reference point drawn opaque
    s.ref_rois[0].selected = True
    assert overlay_image(s).rgba[centre][3] == 0.9
    s.set_color_mode("tags")
    s2 = RoiQuestSession.from_adata(tmp_path / "Adata-S1-T401.mat")
    s2.set_color_mode("tags")
    rgba = overlay_image(s2).rgba
    a, b = s2.rois[0][2], s2.rois[0][3]
    ca = rgba[a.to_binary_mask()][0, :3]
    cb = rgba[b.to_binary_mask()][0, :3]
    assert not np.allclose(ca, cb)
    fig.canvas.draw()


def test_actiquest(two_p):
    s = RoiQuestSession.from_adata(two_p)
    g = np.random.default_rng(3)
    r0 = s.rois[0][0]
    r0.activity = 10 + g.random((50, 1))
    s.select_all()
    rois, idx = s.activity_selection()
    aq = ActiQuestSession(rois, idx)
    assert aq.act_data.shape == (1, 50) and aq.roi_labels == ["1"]
    np.testing.assert_allclose(np.median(aq.act_data + 1, axis=1), 1.0)
    fig = Figure()
    FigureCanvasAgg(fig)
    view = ActiQuestView(fig, aq)
    view.refresh(aq.key_press("up", ("shift",)))
    assert aq.raster_hi == pytest.approx(2.2)
    assert view.raster.get_clim() == (0.0, pytest.approx(2.2))
    np.testing.assert_allclose(view.lines[0].get_ydata(), aq.plot_data[0])
    for _ in range(20):
        aq.key_press("downarrow", ("shift",))
    assert aq.raster_hi == 0.4
    assert aq.key_press("up") == Redraw.NONE
    view.refresh(aq.key_press("d"))
    assert aq.is_dark_mode and view.raster.get_cmap().name == "hot"
    np.testing.assert_allclose(view.lines[0].get_color(), aq.colors[0])
    aq2 = [r0, r0]
    view.refresh(aq.load_new_rois(aq2, [0, 0]))
    assert len(view.lines) == 2 and view.ax_traces.get_ylim() == (0.5, 3.0)
    fig.canvas.draw()
