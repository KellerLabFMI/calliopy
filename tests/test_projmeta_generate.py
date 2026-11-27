"""End-to-end tests of calliopy.projmeta.generate_proj_meta on small synthetic experiments."""

from __future__ import annotations

import struct

import calliopy.analysis
import numpy as np
import pytest
from calliopy.io import MatStruct, save_mat
from calliopy.projmeta import generate_proj_meta
from calliopy.projmeta._struct import struct_array

PI, MOUSE = "hans", "AA_240101_009"


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(calliopy.analysis, "psmooth", lambda t, win_size=None: np.asarray(t, float) + 10,
                        raising=False)
    monkeypatch.setattr(calliopy.analysis, "ftfil", lambda d, f, hi, lo: np.asarray(d, float), raising=False)
    monkeypatch.setattr(calliopy.analysis, "smooth2", lambda d, w, axis=None: np.asarray(d, float) * 2,
                        raising=False)

    def no_spikes(act):
        raise RuntimeError("no cascade here")

    monkeypatch.setattr(calliopy.analysis, "act2spks", no_spikes, raising=False)


def exp_log(stacks, exps, acode="1", comments=None):
    n = len(stacks)
    obj = lambda v: np.array(v, dtype=object)
    return MatStruct(stackid=np.array(stacks), expid=np.array(exps),
                     comment=obj(comments or ["fb day1"] * n), stackdate=obj(["05-Jan-2024"] * n),
                     datadeleted=np.zeros(n, int), paradigm=obj([""] * n), siteid=np.full(n, exps[0]),
                     analysiscode=obj([acode] * n), animalid=obj([MOUSE] * n), project=obj(["PRJ"] * n),
                     location=obj(["L2/3"] * n), pi=obj([PI] * n), strain=obj(["C57"] * n))


def write_lvd(path, data, scan_rate=1000.0):
    with open(path, "wb") as fh:
        fh.write(struct.pack(">4d", scan_rate, data.shape[0], 0.0, 10.0))
        fh.write(data.T.astype(">f8").tobytes())


def write_ach(path, names):
    lines = ["[main]", 'auxrec.scanrate = "1000"'] + [f'auxrec.channel{i} = "{n}"' for i, n in enumerate(names)]
    path.write_text("\n".join(lines) + "\n")


def galvo(n_frames, period=150, flyback=40, lead=60):
    sig = np.zeros(lead)
    for _ in range(n_frames):
        sig = np.r_[sig, np.linspace(-1, 1, period - flyback), np.ones(flyback)]
    return np.r_[sig, np.full(50, -1.0)]


def test_behaviour_only(tmp_path, fakes):
    raw, adir = tmp_path / "raw", tmp_path / "adata"
    folder = raw / PI / MOUSE
    folder.mkdir(parents=True)
    adir.mkdir()
    n = 400
    running = np.linspace(0, 4, n)
    lick = np.zeros(n)
    lick[[100, 101, 250]] = 5
    write_lvd(folder / "S1-T700.lvd", np.vstack((running, lick)))
    write_ach(folder / "S1-T700.ach", ["Running", "Lick"])
    el = exp_log([700], [700], comments=["ori omission"])
    pm = generate_proj_meta("PRJ", 1, exp_log=el, adata_dir=adir, raw_data_dir=raw, file_types=[".lvd", ".ach"],
                            chans_to_include=["Running", "Lick"], uuid="u1")
    assert pm.shape == (1, 1)
    site = pm[0, 0]
    assert (site.projID, site.uuid, site.Acode, site.animal, site.animalNbr, site.siteNbr) == \
        ("PRJ", "u1", "1", MOUSE, 1, 1)
    np.testing.assert_array_equal(site.ExpGroup, [[700]])
    assert site.lab_notes.tolist() == [["No lab notes found"]]
    rd = site.rd[0, 0]
    assert list(rd)[:4] == ["timepoint", "piezo_layer", "comments", "Running"]
    assert rd.timepoint == 1 and rd.piezo_layer == 1 and rd.session == ["ori", "omission"]
    np.testing.assert_allclose(rd.Running, np.diff(running))  # fake ftfil: identity
    np.testing.assert_allclose(rd.Running_smoothed, 2 * np.diff(running))
    np.testing.assert_array_equal(rd.Lick, lick)
    np.testing.assert_array_equal(rd.lickTimes, [[100, 250]])  # 1-based onsets

    ds = generate_proj_meta("PRJ", 1, exp_log=el, adata_dir=adir, raw_data_dir=raw, file_types=[".lvd", ".ach"],
                            chans_to_include=[1], downsample_behavior=True, filtered_exp_log=True)
    rd = ds[0, 0].rd[0, 0]
    np.testing.assert_array_equal(rd.Lick, lick[::10])
    assert "Running" not in rd and "ExpLog" in ds[0, 0]
    with pytest.raises(ValueError, match="chans_to_include"):
        generate_proj_meta("PRJ", 1, exp_log=el, adata_dir=adir, raw_data_dir=raw, file_types=[".lvd"])
    with pytest.raises(NotImplementedError):
        generate_proj_meta("PRJ", 1, exp_log=el, adata_dir=adir, raw_data_dir=raw, file_types=[".lvd", ".eye"],
                           chans_to_include=[0])


def _cell(*items):
    c = np.empty((1, len(items)), dtype=object)
    for i, x in enumerate(items):
        c[0, i] = x
    return c


def test_two_photon(tmp_path, fakes):
    raw, adir = tmp_path / "raw", tmp_path / "adata"
    folder = raw / PI / MOUSE
    folder.mkdir(parents=True)
    (adir / PI / MOUSE).mkdir(parents=True)
    n_frames = 6
    stacks = {800: [800, 801], 802: [802]}
    for exp, sids in stacks.items():
        for sid in sids:
            g = galvo(n_frames)
            write_lvd(folder / f"S1-T{sid}.lvd", np.vstack((np.arange(g.size) * 0.01, g)))
            write_ach(folder / f"S1-T{sid}.ach", ["Running", "FrameGalvo"])
            with open(folder / f"S1-T{sid}_ch525.bin", "wb") as fh:
                fh.write(np.array([4, 3], "<i2").tobytes())
                fh.write(np.zeros(n_frames * 12, "<i2").tobytes())
        (folder / f"S1-T{exp}_ch525.ini").write_text("[main]\npiezo.nbrlayers = 1\n")
        total = n_frames * len(sids)
        rois = struct_array([MatStruct(indices=np.array([[1.0, 2.0]]), activity=np.arange(total, dtype=float)
                                       .reshape(-1, 1) + k) for k in (1, 5)])
        save_mat(adir / PI / MOUSE / f"Adata-S1-T{exp}.mat",
                 {"ROIs": _cell(rois), "template": _cell(np.ones((4, 3))), "act_map": _cell(np.zeros((4, 3))),
                  "dx": _cell(np.zeros((1, total))), "dy": _cell(np.zeros((1, total))),
                  "nbr_frames": np.full((1, len(sids)), float(n_frames)),
                  "fnames": _cell(*[f"data\\sub\\S1-T{s}_ch525.bin" for s in sids])})
    el = exp_log([800, 801, 802], [800, 800, 802])
    pm = generate_proj_meta("PRJ", 1, exp_log=el, adata_dir=adir, raw_data_dir=raw,
                            file_types=["525.ini", ".lvd", ".ach"], chans_to_include=[0], uuid="u")
    site = pm[0, 0]
    assert site.rd.shape == (1, 2) and site.nbr_piezo_layers == 1
    np.testing.assert_array_equal(site.ExpGroup, [[800, 802]])
    rd = site.rd[0, 0]
    assert rd.fnames.tolist() == [["S1-T800_ch525.bin", "S1-T801_ch525.bin"]]
    assert rd.act.shape == (2, 12) and "spks" not in rd
    cur = np.arange(12.0) + 1 + 10
    np.testing.assert_allclose(rd.act[0], cur / np.median(cur))
    assert rd.ROIinfo[0, 0].F0 == np.median(cur)
    np.testing.assert_array_equal(rd.nbr_frames, [[6, 6]])
    ft = rd.frame_times
    assert ft.shape == (1, 12) and ft.min() >= 1
    vel = np.diff(np.arange(galvo(n_frames).size) * 0.01)
    assert rd.Running.dtype == np.float32 and rd.Running.shape == (1, 12)
    np.testing.assert_allclose(rd.Running[0, :6], vel[ft[0, :6].astype(int) - 1], rtol=1e-6)
    assert rd.session == ["fb", "fb"] and site.rd[0, 1].timepoint == 2
    with pytest.raises(ValueError, match="raw_data_dir"):
        generate_proj_meta("PRJ", 1, exp_log=el, adata_dir=adir, file_types=[".lvd"])


def test_spines(tmp_path):
    adir = tmp_path / "adata"
    (adir / PI / MOUSE).mkdir(parents=True)
    for exp in (900, 901):
        save_mat(adir / PI / MOUSE / f"Adata-S1-T{exp}.mat",
                 {"spiny_regions": _cell(np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]])),
                  "spines": _cell(np.array([[5.0]]), np.zeros((0, 0))),
                  "dendrite": _cell(np.array([[6.0]]), np.array([[7.0]])),
                  "spine_tracking": _cell(np.array([[1.0, 0.0]]), np.zeros((0, 0))),
                  "dendrite_length": _cell(np.array([[12.5]]), np.array([[3.0]]))})
    el = exp_log([900, 901], [900, 901])
    meta_info = [["ExpID", "timepoint"], [900, "-48h"], [901, "+24h"]]
    pm = generate_proj_meta("PRJ", 1, exp_log=el, adata_dir=adir, meta_info=meta_info,
                            meta_info_columns=["timepoint", "missing"], uuid="u")
    site = pm[0, 0]
    assert site.exp_type == 6 and site.nbr_spiny_regions == 2 and site.rd.shape == (2, 2)
    r = site.rd[0, 0]
    assert r.timepoint == "-48h"  # the MetaInfo column overwrites the timepoint number, as in MATLAB
    np.testing.assert_array_equal(r.spines, [[5.0]])
    np.testing.assert_array_equal(r.spine_tracking, [[1.0, 0.0]])
    assert site.rd[1, 0].spines.shape == (0, 0)
    assert site.rd[0, 1].spine_tracking.shape == (0, 0) and site.rd[0, 1].timepoint == "+24h"
