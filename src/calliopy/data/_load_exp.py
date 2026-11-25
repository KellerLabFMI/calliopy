"""Load the files of one experiment: calliope ``05_LoadData/load_exp.m``."""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

import numpy as np

from ..explog import as_exp_log, get_exp_log, read_info_from_exp_log
from ..io import Adata, MatStruct, load_lvd, load_wcp, load_wid_data, parse_ach, read_ini, read_tif
from ..preprocessing import get_frame_times
from ..widefield import frame_times_widefield
from ..widefield._frames import shift_data
from ._files import get_adata_filename
from ._linescan import correct_line_shift

__all__ = ["LOAD_EXP_DEFERRED_FTYPES", "load_exp"]

log = logging.getLogger(__name__)

LOAD_EXP_DEFERRED_FTYPES = (
    "EEG.dat", ".run", ".vid", ".vid2", ".avi", "min.avi", "min.csv", ".eye", "sec.eye", ".eyeR", ".ton",
)
"""File types of ``load_exp.m`` whose branches are not ported yet."""


def _adata_get(adata, name):
    try:
        return adata[name]
    except (KeyError, AttributeError):
        return None


def _has(adata, name) -> bool:
    return name in tuple(getattr(adata, "variables", ()))


def _read_bin(fnames, nbr_frames, nbr_piezo_layers: int, shape) -> list[np.ndarray]:
    """The inline ``.bin`` reading of load_exp.m: ``nbr_frames[i]`` frames from file ``i``."""
    total = int(np.sum(nbr_frames))
    out = [np.zeros((shape[0], shape[1], total // nbr_piezo_layers), dtype=np.int16)
           for _ in range(nbr_piezo_layers)]
    k = 0
    for fname, nf in zip(fnames, nbr_frames):
        log.info("loading file %s from data directory", Path(fname).name)
        with open(fname, "rb") as fh:
            x_res, y_res = (int(v) for v in np.fromfile(fh, dtype="<i2", count=2))
            frames = np.fromfile(fh, dtype="<i2", count=int(nf) * x_res * y_res)
        frames = frames.reshape(int(nf), x_res, y_res)
        for f in frames:
            lay, pos = k % nbr_piezo_layers, k // nbr_piezo_layers
            if pos < out[lay].shape[2]:
                out[lay][:, :, pos] = f
            k += 1
    return out


def load_exp(
    exp_id: int,
    load_ftypes,
    exp_log=None,
    *,
    raw_data_dir: str | os.PathLike | None = None,
    adata_dir: str | os.PathLike | None = None,
    main_channel: str = "ch525",
    secondary_channels=(),
    noregister: bool = False,
    wid_name_shutter: str = "Shutter",
    wid_name_trig: str = "CamExp",
) -> MatStruct:
    """Load the requested file types of one experiment.

    Port of calliope ``05_LoadData/load_exp.m``. MATLAB assigns the loaded variables into the
    base or caller workspace; here they are returned in a :class:`MatStruct` under the same
    names (``data``, ``sec_data``, ``frame_times``, ``aux_data``, ``aux_names``,
    ``aux_scanrate``, ``ach``, ``wcp_data``, ``wcp_scanrate``, ``tif_data``,
    ``mean_data``, ``mouse_id``, ``userID``, ``adata_dir`` and the adata fields ``dx``,
    ``dy``, ``nbr_frames``, ``template``, ``act_map``, ``template_sec``, ``ROIs``, ``owl``).

    Parameters
    ----------
    exp_id : int
        Experiment ID.
    load_ftypes : sequence of str
        File types as in MATLAB: '.bin' (or '<channel number>.bin'), '.ini', '.lvd', '.ach',
        '.wcp', '.wii', '.wid', '.tif' suffixes, 'owl.avi'. The types in
        :data:`LOAD_EXP_DEFERRED_FTYPES` raise ``NotImplementedError``.
    exp_log : ExpLog, optional
        Default: :func:`calliopy.explog.get_exp_log` (cached table).
    raw_data_dir : path
        Raw data root holding ``<userID>/<mouse_id>/S1-T...`` files (MATLAB:
        ``get_data_path``).
    adata_dir : path
        Adata root holding ``<userID>/<mouse_id>/Adata-S1-T....mat`` (MATLAB:
        ``set_lab_paths``).
    main_channel, secondary_channels
        Channels of the project (MATLAB: ``ProjectDefaults_<project>``).
    noregister : bool, default False
        Load two-photon data without the dx/dy shift and line-shift correction (MATLAB reads
        ``load_noregister`` from the base workspace).
    wid_name_shutter, wid_name_trig : str
        Aux channel names of the shutter and the camera exposure (MATLAB:
        ``pdef.wid.name_shutter``/``name_trig``, falling back to 'Shutter'/'CamExp').

    Returns
    -------
    MatStruct
        The loaded variables. ``frame_times`` are 1-based sample indices into ``aux_data``
        (stored-value convention).

    Raises
    ------
    ValueError
        A directory that the requested types need is missing.
    FileNotFoundError
        Two-photon or widefield data are requested and there is no adata file (MATLAB
        prints 'register data before loading' and returns nothing).
    RuntimeError
        Aux frames and ``.bin`` frames cannot be matched (MATLAB prints a critical error and
        returns nothing).
    NotImplementedError
        A deferred file type is requested.

    Differences from MATLAB
    -----------------------
    - Directories, channels and ``noregister`` are arguments; nothing is read from the base
      workspace and nothing is written to an error log (critical warnings are
      ``warnings.warn``).
    - Requesting '.ini' also loads the aux data, which its frame-time step needs (MATLAB
      crashes when '.ini' is requested alone).
    - Kept from MATLAB: the list of adata fields to return is replaced, not extended, by
      the two-photon branch ('dx', 'dy', 'nbr_frames', 'template', 'act_map') and by the
      secondary-channel branch ('template_sec'), so ``ROIs``/``owl`` are only returned when
      no ``.bin`` is loaded.
    """
    ft = [str(f) for f in load_ftypes]
    deferred = [f for f in ft if f in LOAD_EXP_DEFERRED_FTYPES]
    if deferred:
        raise NotImplementedError(f"load_exp: file types not ported yet: {deferred}")
    if adata_dir is None:
        raise ValueError("adata_dir is required (MATLAB set_lab_paths)")
    el = as_exp_log(exp_log if exp_log is not None else get_exp_log(suppress_warning=True))
    sec = list(secondary_channels)

    load_2p = ".bin" in ft or f"{main_channel[2:]}.bin" in ft
    load_2p_sec = bool(sec) and f"{sec[0][2:]}.bin" in ft
    load_ini = any(f in ft for f in (".ini", "525.ini", "610.ini"))
    load_aux = ".lvd" in ft
    load_ach = ".ach" in ft
    load_wid = ".wid" in ft
    load_wii = any(f in ft for f in (".wii", "525.wii", "610.wii"))
    want_wcp = ".wcp" in ft
    load_tif = any(".tif" in f for f in ft)
    load_owl = any("owl.avi" in f for f in ft)
    if load_wid:
        load_wii = True
    if load_2p or load_2p_sec or load_wii or load_ini:
        load_aux = True
    if load_2p or load_2p_sec:
        load_ini = True
    if (load_aux or load_ach or want_wcp or load_2p or load_2p_sec or load_wid or load_tif) and raw_data_dir is None:
        raise ValueError("raw_data_dir is required for the requested file types (MATLAB get_data_path)")

    info, data_dir = read_info_from_exp_log(exp_id, True, el, data_dir=raw_data_dir,
                                            main_channel=main_channel, secondary_channels=sec)
    adata_file, mouse_id, user_id, _ = get_adata_filename(exp_id, adata_dir, el)

    out = MatStruct()
    adata_fields: list[str] = []
    adata = None
    if not adata_file:
        if load_2p or load_2p_sec or load_wii:
            raise FileNotFoundError(f"found no adata file for ExpID {exp_id} - register data before loading")
    else:
        adata = Adata(Path(adata_dir) / user_id / mouse_id / adata_file)
        if _has(adata, "mean_data"):
            load_2p = load_2p_sec = load_aux = False
            out.mean_data = adata["mean_data"]
        out.mouse_id = mouse_id
        out.userID = user_id
        out.adata_dir = str(adata_dir)
        if _has(adata, "ROIs"):
            adata_fields = ["ROIs"]
        if load_owl and _has(adata, "owl"):
            adata_fields = ["owl"]

    aux_files = list(info.get("aux_files", []))
    aux_base = aux_files[0][:-4] if aux_files and aux_files[0] else None
    if load_ach:
        out.ach = read_ini(aux_base + ".ach")

    aux_length: list[int] = []
    if load_aux:
        ach_parsed = parse_ach(aux_base + ".ach")
        aux_names = [str(n) for n in np.asarray(ach_parsed["aux_chans"], dtype=object)[:, 1]]
        aux_scanrate = float(ach_parsed["auxrec"]["scanrate"])
        chunks = []
        for f in aux_files:
            log.info("loading file %s from data directory", Path(f).name)
            d = load_lvd(f).data
            chunks.append(d)
            aux_length.append(d.shape[1])
        n_ch = max(c.shape[0] for c in chunks)
        aux_data = np.zeros((n_ch, sum(aux_length)))
        o = 0
        for c in chunks:
            aux_data[: c.shape[0], o : o + c.shape[1]] = c
            o += c.shape[1]
        out.aux_data, out.aux_names, out.aux_scanrate = aux_data, aux_names, aux_scanrate

    nbr_piezo_layers = 1
    if load_ini:
        fnames = info.get("fnames", [])
        if not fnames:
            raise FileNotFoundError(f"no .bin file of ExpID {exp_id} for the .ini")
        ini = read_ini(fnames[0][:-3] + "ini")
        nbr_piezo_layers = int(np.ravel(ini["piezo"]["nbrlayers"])[0])
        nbr_frames = np.ravel(np.asarray(adata["nbr_frames"])).astype(np.int64)
        frame_times = []
        for ind, n_aux in enumerate(aux_length):
            start = sum(aux_length[:ind])
            tmp = get_frame_times(out.aux_data[1, start : start + n_aux])
            nf = int(nbr_frames[ind])
            if nf != tmp.size:
                dt = np.diff(tmp)
                regular = dt.size and np.max(dt) / np.median(dt) < 1.5
                if regular and tmp.size > nf:
                    log.warning("NC WARNING! Probably scanning too fast for FPGA to stop in the same frame "
                                "--- %d frame(s) too many ---", tmp.size - nf)
                    tmp = tmp[:nf]
                elif regular and tmp.size < nf:
                    warnings.warn("CRITICAL WARNING! looks like a crashed stack where line galvo stopped running "
                                  f"--- {nf - tmp.size} frame(s) too few ---", stacklevel=2)
                    keep = nbr_piezo_layers * (tmp.size // nbr_piezo_layers)
                    nbr_frames[ind] = keep
                    tmp = tmp[:keep]
                elif tmp.size > nf:
                    warnings.warn("CRITICAL WARNING! This looks like a HD full crash file - careful this is a hack "
                                  f"that only rarely works --- {tmp.size - nf} frame(s) too many ---", stacklevel=2)
                    tmp = tmp[:nf]
                else:
                    raise RuntimeError("Critical error - aux frames and bin frames do not match "
                                       "(aux data longer than it should be)")
            frame_times.append(tmp + start)
        out.frame_times = np.concatenate(frame_times) if frame_times else np.empty(0)
        adata_nbr_frames = nbr_frames
    else:
        adata_nbr_frames = None

    if want_wcp:
        wcp = load_wcp(aux_base + ".wcp")
        out.wcp_data, out.wcp_scanrate = wcp.data, wcp.scan_rate

    def two_photon(fnames):
        template0 = np.asarray(adata["template"], dtype=object).ravel()[0]
        data = _read_bin(fnames, adata_nbr_frames, nbr_piezo_layers, np.shape(template0))
        if not noregister:
            dx = np.asarray(adata["dx"], dtype=object).ravel()
            dy = np.asarray(adata["dy"], dtype=object).ravel()
            for r in range(len(dx)):
                data[r] = shift_data(data[r], np.ravel(dx[r]), np.ravel(dy[r]))
                data[r] = correct_line_shift(data[r], data[r].astype(np.float64).mean(axis=2)).data
        return data

    if load_2p:
        out.data = two_photon(info.fnames)
        adata_fields = ["dx", "dy", "nbr_frames", "template", "act_map"]
    if load_2p_sec:
        out.sec_data = two_photon(info.sec_fnames)
        adata_fields = ["template_sec"]

    if load_wii:
        meta = np.asarray(adata["meta_info"], dtype=object).ravel()
        file_ids = np.array([int(np.ravel(m["file_id"])[0]) for m in meta])
        aux_names = out.aux_names
        if wid_name_shutter in aux_names and wid_name_trig in aux_names:
            sh_i, tr_i = aux_names.index(wid_name_shutter), aux_names.index(wid_name_trig)
        else:
            log.info("Making educated guess about the names of the shutter and frame signal..")
            sh_i, tr_i = aux_names.index("Shutter"), aux_names.index("CamExp")
        frame_times = []
        for ind, n_aux in enumerate(aux_length):
            start = sum(aux_length[:ind])
            seg = out.aux_data[:, start : start + n_aux]
            ft_res = frame_times_widefield(seg[tr_i], seg[sh_i], list(meta[file_ids == ind + 1]), out.aux_scanrate)
            frame_times.append(np.ravel(ft_res.frame_times_timestamp) + start)
        out.frame_times = np.concatenate(frame_times) if frame_times else np.empty(0)

    if load_wid:
        wid = load_wid_data(info.wid_fnames)
        out.data = [wid.wid_data]
        if np.shape(wid.wid_data)[2] != np.size(out.frame_times):
            log.warning("Nbr of frames in data and in lvd don't match - investigate")

    if load_tif:
        suffix = next(f for f in ft if ".tif" in f)
        out.tif_data = read_tif(Path(data_dir) / info.userID / info.mouse_id / f"S1-T{int(exp_id)}_{suffix}")

    for name in adata_fields:
        if name == "nbr_frames" and adata_nbr_frames is not None:
            out[name] = adata_nbr_frames.reshape(1, -1)
        else:
            val = _adata_get(adata, name)
            if val is not None:
                out[name] = val
    log.info("----Done loading Exp!----")
    return out
