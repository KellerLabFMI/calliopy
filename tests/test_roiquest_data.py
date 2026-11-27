"""Synthetic adata files for the RoiQuest tests (builders only, no tests here).

Two-photon files hold ``template``/``act_map``/``dx``/``dy``/``ROIs``/``ROItrans`` as 1 x nZ
cells, widefield files ``first_frame`` (one channel) or ``first_frame_ch525``/``_ch610`` with
``tform_ch_map`` (two channels) and ``ROIs`` as a 1x1 cell whose first two ROIs are the
reference points. ROI ``indices`` are 1-based column-major linear indices, as MATLAB stores
them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from calliopy.io import MatStruct, save_mat

H, W = 60, 80
"""Two-photon image size."""
BLOBS_Z0 = ((20, 20), (40, 55), (15, 62))
"""0-based (row, col) centres of bright cells in layer 0 (the first carries a saved ROI)."""
BLOBS_Z1 = ((30, 30),)
WF_H, WF_W = 70, 90
"""Widefield image size."""


def cell(items) -> np.ndarray:
    a = np.empty((1, len(items)), dtype=object)
    for i, x in enumerate(items):
        a[0, i] = x
    return a


def disk(hw, center, radius) -> np.ndarray:
    yy, xx = np.mgrid[: hw[0], : hw[1]]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2


def find(mask) -> np.ndarray:
    """MATLAB find: (N, 1) float64 1-based column-major indices."""
    return (np.flatnonzero(np.asarray(mask).ravel(order="F")) + 1).astype(np.float64)[:, None]


def lin(rc, hw) -> float:
    """1-based linear index of a 0-based (row, col)."""
    return float(rc[0] + rc[1] * hw[0] + 1)


def templates(seed=0):
    g = np.random.default_rng(seed)
    t0 = 100 + g.normal(0, 3, (H, W))
    for c in BLOBS_Z0:
        t0[disk((H, W), c, 4)] = 300.0
    t1 = 100 + g.normal(0, 3, (H, W))
    for c in BLOBS_Z1:
        t1[disk((H, W), c, 4)] = 300.0
    return t0, t1


def roi_struct(mask, type_=0.0, **extra) -> MatStruct:
    s = MatStruct(indices=find(mask), shift=np.zeros((1, 2)), type=np.float64(type_))
    s.update(extra)
    return s


def make_2p(path: Path, *, rois_z0=None, rois_z1=None, seed=0, extra=None) -> Path:
    """Two-photon adata with two layers.

    ``rois_z0``/``rois_z1``: lists of ROI structs (default: one ROI on the first blob of layer
    0, nothing in layer 1). ``dx``/``dy`` give a 2-pixel border.
    """
    t0, t1 = templates(seed)
    if rois_z0 is None:
        rois_z0 = [roi_struct(disk((H, W), BLOBS_Z0[0], 4))]
    rois_z1 = rois_z1 or []
    variables = {
        "template": cell([t0, t1]),
        "act_map": cell([t0 * 1.0, t1 * 1.0]),
        "dx": cell([np.array([[0.0, 1.0, -2.0]]), np.array([[1.0]])]),
        "dy": cell([np.array([[0.0, 2.0]]), np.array([[0.0]])]),
        "ROIs": cell([cell(rois_z0) if rois_z0 else np.empty((0, 0)),
                      cell(rois_z1) if rois_z1 else np.empty((0, 0))]),
        "ROItrans": cell([np.zeros((3, 1)), np.zeros((3, 1))]),
        "other": np.arange(3.0).reshape(1, 3),
    }
    if extra:
        variables.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_mat(path, variables)
    return path


WF_REF = ((10, 45), (60, 45))
"""0-based (row, col) of the widefield reference points (bregma-like, lambda-like)."""


def wf_roi(indices, name, allen_index, type_=0.0, ml=1.0, ap=2.0) -> MatStruct:
    return MatStruct(indices=indices, shift=np.zeros((1, 2)), type=np.float64(type_),
                     ML_index=np.float64(ml), AP_index=np.float64(ap), name=name,
                     Allen_area="area " + name, Allen_acronym=name.upper()[:3],
                     Allen_index=np.float64(allen_index))


def wf_rois(hw=(WF_H, WF_W), n_extra=2):
    out = [wf_roi(np.array([[lin(WF_REF[0], hw)]]), "bregma", 0),
           wf_roi(np.array([[lin(WF_REF[1], hw)]]), "lambda", 0)]
    for k in range(n_extra):
        m = np.zeros(hw, dtype=bool)
        m[20 + 10 * k: 26 + 10 * k, 20:30] = True
        out.append(wf_roi(find(m), f"roi{k}", 10 + k))
    return out


def wf_frame(seed=1, hw=(WF_H, WF_W)):
    g = np.random.default_rng(seed)
    f = 1000 + g.normal(0, 20, hw)
    f[30:40, 30:50] += 400
    return f


def make_wf1(path: Path, *, rois=True, params=None, extra=None) -> Path:
    """One-channel widefield adata (``first_frame``), optionally with ROIs and
    ``ROItemplate_parameters``."""
    v = {"first_frame": wf_frame(), "registration_parameters": MatStruct(
        status="OK", activity_calculated=np.float64(0))}
    if rois:
        v["ROIs"] = cell([cell(wf_rois())])
    if params is not None:
        v["ROItemplate_parameters"] = params
    if extra:
        v.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_mat(path, v)
    return path


def make_wf2(path: Path, *, tform_ch_map=None, rois=True) -> Path:
    """Two-channel widefield adata with a numeric ``tform_ch_map`` (identity by default)."""
    hw = (WF_H, WF_W)
    v = {"first_frame_ch525": wf_frame(1), "first_frame_ch610": wf_frame(2),
         "tform_ch_map": np.eye(3) if tform_ch_map is None else tform_ch_map}
    if rois:
        items = []
        for r in wf_rois(hw):
            t = MatStruct(indices_ch525=r["indices"], indices_ch610=r["indices"])
            t.update((k, x) for k, x in r.items() if k != "indices")
            items.append(t)
        v["ROIs"] = cell([cell(items)])
    path.parent.mkdir(parents=True, exist_ok=True)
    save_mat(path, v)
    return path


TEMPLATE_HW = (50, 60)
TEMPLATE_REF = ((5, 30), (45, 30))
"""0-based (row, col) reference points of the synthetic ROI template."""


def make_roi_template(path: Path, name="Synthetic grid") -> Path:
    """ROI template file of the ``ROItemplate_library`` kind (MATLAB v5, ``struct_ROIs``)."""
    hw = TEMPLATE_HW
    dtype = [("indices", object), ("type", object), ("ML_index", object), ("AP_index", object),
             ("name", object), ("Allen_area", object), ("Allen_acronym", object),
             ("Allen_index", object)]
    rois = np.zeros((1, 3), dtype=dtype)
    rois[0, 0] = (np.array([[lin(TEMPLATE_REF[0], hw)]]), 0.0, 0.0, 0.0, "bregma", "", "", 0.0)
    rois[0, 1] = (np.array([[lin(TEMPLATE_REF[1], hw)]]), 0.0, 0.0, 0.0, "lambda", "", "", 0.0)
    m = np.zeros(hw, dtype=bool)
    m[20:26, 10:16] = True
    rois[0, 2] = (find(m), 0.0, 1.0, 2.0, "square", "Area", "SQ", 7.0)
    scipy.io.savemat(path, {"struct_ROIs": {
        "name": name, "description": "test template",
        "parameters": {"SIZE_TEMPLATE": np.array([[float(hw[0]), float(hw[1])]])},
        "img_ROIs": np.zeros(hw), "ROIs": rois}})
    return path


def explog(rows):
    """ExpLog columns from ``(expid, siteid)`` pairs (pi 'user', animal 'mouse')."""
    return {
        "expid": np.array([r[0] for r in rows], dtype=np.int64),
        "siteid": np.array([r[1] for r in rows], dtype=np.int64),
        "stackid": np.array([r[0] for r in rows], dtype=np.int64),
        "pi": np.array(["user"] * len(rows), dtype=object),
        "animalid": np.array(["mouse"] * len(rows), dtype=object),
        "project": np.array(["proj"] * len(rows), dtype=object),
    }


def adata_path(adata_dir: Path, exp_id: int) -> Path:
    return Path(adata_dir) / "user" / "mouse" / f"Adata-S1-T{exp_id}.mat"
