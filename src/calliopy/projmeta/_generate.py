"""Assembling a project meta file (``proj_meta``): calliope ``24_ProjMeta/generate_proj_meta.m``
with its helpers ``hEEG_proj_meta.m``, ``generateMetaInfo.m`` and ``lab_note_page_get.m``.

``generate_proj_meta.m`` asks the user several questions (Acodes, file types, aux channel
mapping, channels to include, downsampling) and reads the project defaults, the lab paths
and the MetaInfo Excel sheet. Here each answer and each location is an argument, and the
steps that only compute are public functions of their own:

- :func:`project_acodes`, :func:`proj_meta_selection`: which stacks, experiments and sites
- :func:`proj_meta_filename`: the file name MATLAB would save under
- :func:`align_aux_channels`: one channel list for all experiments
- :func:`proj_meta_exp_type`: experiment type from the file types
- :func:`read_frame_rate`: camera frame rate of a ``.wii`` file

Nothing is saved: the proj_meta struct array is returned.
"""

from __future__ import annotations

import logging
import os
import re
import struct as _struct
import uuid as _uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from ..data import get_adata_filename, list_file_types, load_exp
from ..explog import as_exp_log
from ..explog._query import _num2str
from ..io import Adata, MatStruct, load_ach_chans, load_heeg, load_heyetracking, parse_vrd, read_ini
from ._aux import assign_aux_method, assign_load_ftypes, assign_processed_aux
from ._rois import _wf_endings, get_smoothed_rois, widefield_rois_for_proj_meta
from ._struct import (
    analysis,
    as_cell_list,
    elements,
    empty,
    grid_struct_array,
    is_empty,
    struct_array,
)

__all__ = [
    "AlignAuxChannelsResult",
    "ExpTypeResult",
    "ProjMetaSelection",
    "align_aux_channels",
    "generate_meta_info",
    "generate_proj_meta",
    "heeg_proj_meta",
    "lab_note_page_get",
    "proj_meta_exp_type",
    "proj_meta_filename",
    "proj_meta_selection",
    "project_acodes",
    "read_frame_rate",
]

log = logging.getLogger(__name__)

_DEFERRED_LFT = ("eye", "eyeR", "ton", "vid", "vid3")


# --------------------------------------------------------------------------------------
# selection of stacks
# --------------------------------------------------------------------------------------
def _str2num(s) -> list[float]:
    """MATLAB ``str2num`` for analysis codes ('3', '3 4', '' -> [])."""
    if not isinstance(s, str):
        try:
            v = float(s)
        except (TypeError, ValueError):
            return []
        return [] if np.isnan(v) else [v]
    parts = [p for p in re.split(r"[\s,;]+", s.strip()) if p]
    try:
        return [float(p) for p in parts]
    except ValueError:
        return []


def _deleted(el) -> np.ndarray:
    n = len(np.asarray(el["expid"]))
    if "datadeleted" not in el:
        return np.zeros(n, dtype=bool)
    d = np.asarray(el["datadeleted"], dtype=object)
    out = np.zeros(n, dtype=bool)
    for i, v in enumerate(d):
        if v is None or (isinstance(v, str) and v == ""):
            continue
        fv = float(v)
        out[i] = fv != 0  # NaN counts as deleted, as MATLAB's find()
    return out


def _project_rows(el, proj_id) -> np.ndarray:
    proj = np.asarray(el["project"], dtype=object)
    return np.array([p == proj_id for p in proj], dtype=bool) & ~_deleted(el)


def project_acodes(exp_log, proj_id: str) -> dict:
    """Analysis codes of a project with their ExpIDs (the list ``generate_proj_meta`` shows).

    Returns
    -------
    dict
        ``{acode (float): sorted ExpIDs (int array)}`` over the stacks of the project whose
        data are not deleted.
    """
    el = as_exp_log(exp_log)
    rows = _project_rows(el, proj_id)
    ac = np.asarray(el["analysiscode"], dtype=object)
    expid = np.asarray(el["expid"])
    codes = sorted({c for i in np.flatnonzero(rows) for c in _str2num(ac[i])})
    out = {}
    for c in codes:
        sel = rows & np.array([a == _num2str(c) for a in ac], dtype=bool)
        out[c] = np.unique(expid[sel]).astype(np.int64)
    return out


class ProjMetaSelection(NamedTuple):
    """Stacks, experiments and sites of a project and its Acodes (:func:`proj_meta_selection`)."""

    rows: np.ndarray
    stack_ids: np.ndarray
    exp_ids: np.ndarray
    exp_index: np.ndarray
    unique_stacks: np.ndarray
    site_ids: np.ndarray
    comments: np.ndarray
    paradigms: np.ndarray | None
    animal_ids: list
    site_animals: list
    locations: list
    site_order: np.ndarray


def proj_meta_selection(exp_log, proj_id: str, acodes) -> ProjMetaSelection:
    """The stacks of a project with the given Acodes, grouped as ``generate_proj_meta`` does.

    Parameters
    ----------
    exp_log : ExpLog
    proj_id : str
    acodes : number or sequence of numbers
        Analysis codes; each must occur in the project (:func:`project_acodes`).

    Returns
    -------
    ProjMetaSelection
        ``rows``: 0-based ExpLog rows (stacks) selected, in ExpLog order;
        ``stack_ids`` [nStacks x 3] (siteid, expid, stackid); ``exp_ids`` [nExps x 2]
        (siteid, expid), sorted by ExpID; ``exp_index``: 0-based row of ``exp_ids`` of
        every stack; ``unique_stacks``: 0-based row of ``stack_ids`` of the first stack
        of every experiment; ``site_ids`` sorted; ``comments`` and ``paradigms`` (None
        without that column) per stack; ``animal_ids`` sorted; per site its animal and
        location (first ExpLog row of the site); ``site_order``: 1-based number of each
        site among the sites of its animal (stored as ``siteNbr``).

    Raises
    ------
    ValueError
        An Acode that the project does not have (MATLAB asks again).
    """
    el = as_exp_log(exp_log)
    acodes = [float(a) for a in np.atleast_1d(np.asarray(acodes, dtype=np.float64))]
    have = project_acodes(el, proj_id)
    if len(set(have) & set(acodes)) != len(acodes):
        raise ValueError(f"Acodes {acodes} are not all codes of project {proj_id}: {sorted(have)}")
    proj = np.asarray(el["project"], dtype=object)
    ac = np.asarray(el["analysiscode"], dtype=object)
    sel = np.zeros(len(proj), dtype=bool)
    for a in acodes:
        sel |= np.array([p == proj_id for p in proj], dtype=bool) & np.array([c == _num2str(a) for c in ac])
    sel &= ~_deleted(el)
    rows = np.flatnonzero(sel)
    site = np.asarray(el["siteid"], dtype=np.int64)
    expid = np.asarray(el["expid"], dtype=np.int64)
    stack = np.asarray(el["stackid"], dtype=np.int64)
    stack_ids = np.column_stack((site[rows], expid[rows], stack[rows])) if rows.size else np.zeros((0, 3), np.int64)
    _, unique_stacks, exp_index = np.unique(stack_ids[:, 1], return_index=True, return_inverse=True)
    exp_ids = stack_ids[unique_stacks, :2]
    site_ids = np.unique(exp_ids[:, 0])
    comments = np.asarray(el["comment"], dtype=object)[rows].reshape(-1, 1)
    paradigms = np.asarray(el["paradigm"], dtype=object)[rows].reshape(-1, 1) if "paradigm" in el else None
    animal = np.asarray(el["animalid"], dtype=object)
    animal_ids = sorted({str(a) for a in animal[rows]})
    loc = np.asarray(el["location"], dtype=object) if "location" in el else np.full(len(animal), "", object)
    site_animals, locations = [], []
    for s in site_ids:
        i = int(np.flatnonzero(site == s)[0])
        site_animals.append(animal[i])
        locations.append(loc[i])
    site_order = np.zeros(len(site_ids), dtype=np.int64)
    for a in animal_ids:
        cur = np.array([sa == a for sa in site_animals], dtype=bool)
        site_order[cur] = np.arange(1, cur.sum() + 1)
    return ProjMetaSelection(rows, stack_ids, exp_ids, exp_index.astype(np.int64), unique_stacks.astype(np.int64),
                             site_ids, comments, paradigms, animal_ids, site_animals, locations, site_order)


def _acode_str(acodes) -> str:
    """MATLAB ``num2str(Acodes)`` for a row of integer codes."""
    return _num2str(np.atleast_1d(np.asarray(acodes, dtype=np.float64)))


def proj_meta_filename(proj_id: str, acodes, *, filename: str | None = None, snob: bool = False,
                       meta_save_dir: str | os.PathLike | None = None) -> str:
    """File name (without ``.mat``) ``generate_proj_meta`` saves under.

    ``<proj>_<acodes>[<filename>]_meta``, Acodes joined as MATLAB's ``num2str`` with each
    double blank replaced by '-'. With ``snob`` (a filtered ExpLog was given) the name
    becomes ``<proj>_<acodes>_snob[<k>]_<middle>meta``, ``k`` counting up while that file
    exists in ``meta_save_dir``; ``middle`` is the part of a custom name between
    ``<proj>_<acodes>_`` and ``_meta`` (MATLAB's index arithmetic, kept: a custom name
    not starting with '_' is dropped).
    """
    acodefn = _acode_str(acodes).replace("  ", "-")
    base = f"{proj_id}_{acodefn}"
    name = f"{base}{filename}_meta" if filename is not None else f"{base}_meta"
    if not snob:
        return name

    def candidate(k: str) -> str:
        m_end = re.search(re.escape(base), name)
        head = name[: m_end.end()] if m_end else ""
        m_mid = re.search(re.escape(base + "_"), name)
        m_meta = re.search("_meta", name)
        mid = name[m_mid.end(): m_meta.start()] if (m_mid and m_meta) else ""
        return f"{head}_snob{k}_{mid}meta"

    k = ""
    if meta_save_dir is not None:
        while (Path(meta_save_dir) / (candidate(k) + ".mat")).is_file():
            k = "1" if k == "" else str(int(k) + 1)
    return candidate(k)


# --------------------------------------------------------------------------------------
# aux channels
# --------------------------------------------------------------------------------------
class AlignAuxChannelsResult(NamedTuple):
    """Outputs of :func:`align_aux_channels`."""

    total_chan_list: list
    chan_id: list
    chan_present: list


def align_aux_channels(chans, exp_index, unique_stacks, resolution: Mapping | Callable | None = None
                       ) -> AlignAuxChannelsResult:
    """One aux channel list for all experiments of a proj_meta (the ACH step of ``generate_proj_meta.m``).

    Parameters
    ----------
    chans : sequence of sequence of str
        Channel names of every stack (``.ach`` files), in the order of
        :attr:`ProjMetaSelection.stack_ids`.
    exp_index, unique_stacks : array of int
        From :func:`proj_meta_selection`.
    resolution : mapping or callable, optional
        Answer for a channel name that is not in the list so far (MATLAB asks): a 0-based
        index into the list (same channel), ``None`` (a new channel), or ``(index,
        new_name)`` (same channel, renamed in the list). A callable is called as
        ``resolution(name, total_chan_list)``.

    Returns
    -------
    AlignAuxChannelsResult
        ``total_chan_list``; ``chan_id``: per experiment, the 0-based position in
        ``total_chan_list`` of each of its channels; ``chan_present``: per channel a text
        of '0'/'1' per experiment.

    Raises
    ------
    RuntimeError
        Channels differ in number between the stacks of one experiment (MATLAB: 'Fatal
        problem').
    ValueError
        A new channel name without an answer in ``resolution``.

    Notes
    -----
    Kept from MATLAB: within an experiment, stacks with the same number of channels are
    only rejected when a single-channel list matches nowhere (``~sum(...) == length``).
    """
    exp_index = np.asarray(exp_index, dtype=np.int64)
    unique_stacks = np.asarray(unique_stacks, dtype=np.int64)
    chans = [list(c) for c in chans]
    for e in range(len(unique_stacks)):
        stacks = np.flatnonzero(exp_index == e)
        ref = chans[stacks[0]]
        for k in stacks[1:]:
            if len(ref) == len(chans[k]):
                n_eq = sum(a == b for a, b in zip(ref, chans[k], strict=True))
                if int(n_eq == 0) == len(ref):
                    raise RuntimeError("Fatal problem - channels are not consistent within an ExpID - panic")
            else:
                raise RuntimeError("Fatal problem - channels are not consistent within an ExpID - "
                                   "now is a good time to panic")
    total = list(chans[unique_stacks[0]])
    chan_id: list = [np.arange(len(total), dtype=np.int64)]
    curr_list, curr_id = list(total), np.arange(len(total), dtype=np.int64)
    for knd in range(1, len(unique_stacks)):
        this = chans[unique_stacks[knd]]
        consistent = len(curr_list) == len(this) and all(a == b for a, b in zip(curr_list, this, strict=True))
        if consistent:
            chan_id.append(curr_id.copy())
            continue
        ids: list[int] = []
        for name in this:
            if name in total:
                ids.append(total.index(name))
                continue
            if callable(resolution):
                ans = resolution(name, list(total))
            elif resolution is not None and name in resolution:
                ans = resolution[name]
            else:
                raise ValueError(f'Channel: "{name}" could not be found in list {total}; give its answer in '
                                 "resolution= (0-based index, None for a new channel, or (index, new_name))")
            if ans is None:
                total.append(name)
                ids.append(len(total) - 1)
            elif isinstance(ans, (tuple, list)):
                idx, new_name = int(ans[0]), str(ans[1])
                if not 0 <= idx < len(total):
                    raise ValueError(f"channel index {idx} out of range")
                total[idx] = new_name
                ids.append(idx)
            else:
                idx = int(ans)
                if not 0 <= idx < len(total):
                    raise ValueError(f"channel index {idx} out of range")
                ids.append(idx)
        chan_id.append(np.asarray(ids, dtype=np.int64))
        curr_list, curr_id = list(this), chan_id[-1]
    present = ["".join("1" if np.any(cid == i) else "0" for cid in chan_id) for i in range(len(total))]
    return AlignAuxChannelsResult(total, chan_id, present)


# --------------------------------------------------------------------------------------
# experiment type, small local functions of generate_proj_meta.m
# --------------------------------------------------------------------------------------
class ExpTypeResult(NamedTuple):
    """Outputs of :func:`proj_meta_exp_type`."""

    exp_type: int
    has_image_data: bool
    has_secondary_2p: bool


def proj_meta_exp_type(file_types) -> ExpTypeResult:
    """Experiment type of a proj_meta from its file types (``generate_proj_meta.m``).

    0: two-photon (one '.ini' type; two: with a secondary channel), 1: behaviour only,
    3: miniscope ('min.avi'), 4: widefield (a type matching ``.wii``), 5: EEG ('EEG.dat';
    with widefield data when a ``.wii`` type is present). Later rules win, in this order.

    Raises
    ------
    ValueError
        More than two '.ini' types (MATLAB fails later).
    """
    ft = [str(f) for f in file_types]
    n_ini = sum(f.endswith(".ini") for f in ft)
    if n_ini > 2:
        raise ValueError("more than two .ini file types")
    exp_type, has_image, has_sec = 1, True, n_ini == 2
    if n_ini:
        exp_type = 0
    if "min.avi" in ft:
        exp_type = 3
    has_wii = any(re.search(r".wii", f) for f in ft)
    if has_wii:
        exp_type = 4
    if "EEG.dat" in ft:
        exp_type = 5
        has_image = has_wii
    if exp_type == 1:
        has_image = False
    return ExpTypeResult(exp_type, has_image, has_sec)


def _parse_comment(comments):
    """``parse_comment`` of generate_proj_meta.m (MATLAB's concatenation of chars and cells)."""
    session: Any = None
    for c in comments:
        found = re.findall(r"fb|pb|dk|gr|gratings|dark|ori|omission|flip", str(c).lower())
        temp: Any = found if found else "undef "
        if session is None:
            session = temp
        elif isinstance(session, str) and isinstance(temp, str):
            session = session + temp
        elif isinstance(session, str):
            session = [session, *temp]
        elif isinstance(temp, str):
            session = [*session, temp]
        else:
            session = [*session, *temp]
    return empty() if session is None else session


def _nbr_frames_widefield(meta_info_wid) -> np.ndarray:
    """``generate_nbr_frames_for_widefield``: frames per file from the widefield meta_info."""
    ids = np.array([int(np.ravel(m["file_id"])[0]) for m in elements(meta_info_wid)])
    u = np.unique(ids)
    if not np.array_equal(u, np.arange(1, u.size + 1)):
        raise ValueError("Something is wrong with the file_id assignment of the widefield meta_info field in Adata")
    return np.array([[int(np.sum(ids == i)) for i in u]])


def read_frame_rate(filename: str | os.PathLike) -> float | None:
    """Camera frame rate of a ``.wii`` file (``readFrameRate`` in ``generate_proj_meta.m``).

    Tries a text line ``FrameRate = <value>``, then the platform format: a 4-byte
    little-endian length followed by a ``label=value;...`` string holding
    ``param.cameraFrameRate``.

    Returns
    -------
    float or None
        None when neither format gives a frame rate.
    """
    raw = Path(filename).read_bytes()
    for line in raw.decode("latin-1").split("\n"):
        m = re.match(r"^\s*FrameRate\s*=\s*(\S+)", line.rstrip("\r"))
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return float("nan")
    if len(raw) < 4:
        return None
    n = _struct.unpack("<I", raw[:4])[0]
    n = min(n, len(raw) - 4)
    cfg = raw[4 : 4 + n].decode("latin-1")
    m = re.search(r"(?:^|;)\s*param\.cameraFrameRate\s*=\s*([^;]+)", cfg)
    if m:
        try:
            return float(m.group(1).strip())
        except ValueError:
            return float("nan")
    return None


def lab_note_page_get(mouse_id: str, exp_log, lab_books_dir) -> list[str]:
    """Contents of the lab note pages of an animal (``24_ProjMeta/lab_note_page_get.m``).

    Every file below ``<lab_books_dir>/<pi>/`` (any depth) whose name contains the animal
    ID, for each PI of the animal in the ExpLog.

    Returns
    -------
    list of str
        File contents, or ``['No lab notes found']``.

    Differences from MATLAB
    -----------------------
    ``lab_books_dir`` is an argument (MATLAB ``set_lab_paths('LabBooks_dir')``); folders
    whose name contains the ID are skipped (MATLAB tries to read them and fails).
    """
    el = as_exp_log(exp_log)
    animal = np.asarray(el["animalid"], dtype=object)
    pis = sorted({str(p) for p, a in zip(np.asarray(el["pi"], dtype=object), animal, strict=True) if a == mouse_id})
    paths = []
    for pi in pis:
        folder = Path(lab_books_dir) / pi
        if folder.is_dir():
            paths += sorted(p for p in folder.rglob(f"*{mouse_id}*") if p.is_file())
    if not paths:
        return ["No lab notes found"]
    return [p.read_text(encoding="utf-8", errors="replace") for p in paths]


def generate_meta_info(proj_id: str, exp_log, existing=None):
    """Rows of the MetaInfo sheet of a project (``24_ProjMeta/generateMetaInfo.m``).

    Header ``ExpID, SiteID, AnimalID, Date, timepoint, Acode, <empty>, <empty>,
    comment`` and one row per experiment of the project (the ExpLog row whose stack ID
    equals the ExpID), timepoint 0.

    Parameters
    ----------
    proj_id : str
    exp_log : ExpLog
    existing : sequence of rows, optional
        The current sheet (header first). Only experiments not yet in its first column
        are added, padded with 'N/A' to its width (MATLAB's append mode).

    Returns
    -------
    list of list or None
        The table (header first; empty cells are None), or None when there is nothing to
        write.

    Differences from MATLAB
    -----------------------
    The table is returned; writing the Excel file is left to the caller. Kept from
    MATLAB: the 'comment' column holds the analysis code.
    """
    el = as_exp_log(exp_log)
    proj = np.asarray(el["project"], dtype=object)
    expid = np.asarray(el["expid"], dtype=np.int64)
    stack = np.asarray(el["stackid"], dtype=np.int64)
    exps = np.unique(expid[np.array([p == proj_id for p in proj], dtype=bool)])
    if existing is not None:
        old = {int(r[0]) for r in list(existing)[1:] if r and r[0] is not None}
        exps = np.array([e for e in exps if int(e) not in old], dtype=np.int64)
        if exps.size == 0:
            log.info("No new ExpID found to append.")
            return None
    inds = [int(np.flatnonzero(stack == e)[0]) for e in exps if np.any(stack == e)]
    if not inds:
        return None
    header = ["ExpID", "SiteID", "AnimalID", "Date", "timepoint", "Acode", None, None, "comment"]
    cols = {k: np.asarray(el[k], dtype=object) for k in ("expid", "siteid", "animalid", "stackdate", "analysiscode")}
    body = [[cols["expid"][i], cols["siteid"][i], cols["animalid"][i], cols["stackdate"][i], 0,
             cols["analysiscode"][i], None, None, cols["analysiscode"][i]] for i in inds]
    if existing is None:
        return [header, *body]
    existing = [list(r) for r in existing]
    width = max(len(r) for r in existing)
    body = [r + ["N/A"] * (width - len(r)) for r in body]
    return existing + body


def _meta_info_values(meta_info, columns, exp_id) -> dict:
    out = {}
    if not columns:
        return out
    if meta_info is None:
        raise ValueError("meta_info_columns need meta_info= (the rows of MetaInfo-<proj>.xlsx, header first)")
    table = [list(r) for r in meta_info]
    header = table[0]
    rows = [r for r in table[1:] if r and r[0] is not None and _num2str(r[0]) == _num2str(exp_id)]
    for col in columns:
        if col not in header:
            log.warning("It looks like your meta_info.xls file does not contain a column specified in the "
                        "ProjectDefaults_XXX.m file")
            continue
        j = header.index(col)
        out[col] = rows[0][j] if rows and j < len(rows[0]) else empty()
    return out


# --------------------------------------------------------------------------------------
# hEEG and spine variants
# --------------------------------------------------------------------------------------
def heeg_proj_meta(proj_id: str, uuid: str, exp_log, acodes, site_ids, exp_ids, file_paths) -> np.ndarray:
    """Human EEG proj_meta (``24_ProjMeta/hEEG_proj_meta.m``).

    Per site: ``projID``, ``uuid``, ``Acode``, ``animal``, ``ExpGroup``, ``p`` and ``MM``
    from ``S1-T<first ExpID>.ini``, ``paradigm`` per stack and ``rd`` (one entry per
    stack: :func:`calliopy.io.load_heeg` of ``S1-T<stack>.txt``/``.json``, or
    :func:`calliopy.io.load_heyetracking` when the ini sets ``eye_tracking_only = 1``).

    Parameters
    ----------
    proj_id, uuid : str
    exp_log : ExpLog
    acodes : number or sequence
    site_ids : array
        Site IDs (:attr:`ProjMetaSelection.site_ids`).
    exp_ids : array [nExps x 2]
        (siteid, expid) (:attr:`ProjMetaSelection.exp_ids`).
    file_paths : sequence of paths
        Data folder (``<raw>/<pi>/<animal>``) per experiment.

    Returns
    -------
    numpy.ndarray
        proj_meta struct array (1, nSites); ``rd`` is a (1, nStacks) object array.

    Notes
    -----
    Kept from MATLAB: the folder of site ``i`` is ``file_paths[i]`` (an experiment
    index used for a site; equal when every site has one experiment), and once a site
    is eye-tracking only, all later sites are too.
    """
    el = as_exp_log(exp_log)
    expid = np.asarray(el["expid"], dtype=np.int64)
    animal = np.asarray(el["animalid"], dtype=object)
    stack = np.asarray(el["stackid"], dtype=np.int64)
    paradigm = np.asarray(el["paradigm"], dtype=object) if "paradigm" in el else np.full(len(expid), "", object)
    exp_ids = np.asarray(exp_ids, dtype=np.int64).reshape(-1, 2)
    eye_only = False
    sites = []
    for ind, s in enumerate(np.ravel(site_ids)):
        cur = exp_ids[exp_ids[:, 0] == s, 1]
        rows = np.flatnonzero(expid == cur[0])
        folder = Path(file_paths[ind])
        pm = MatStruct(projID=proj_id, uuid=uuid, Acode=_acode_str(acodes), animal=animal[rows[0]],
                       ExpGroup=cur.reshape(1, -1))
        ini = read_ini(folder / f"S1-T{int(cur[0])}.ini")
        pm["p"] = ini["p"]
        pm["MM"] = ini["MM"]
        if "eye_tracking_only" in ini and float(np.ravel(ini["eye_tracking_only"])[0]) == 1:
            eye_only = True
        para, rd = [], []
        for st in stack[rows]:
            para.append(paradigm[rows][len(para)])
            if eye_only:
                rd.append(load_heyetracking(folder / f"S1-T{int(st)}.json"))
            else:
                rd.append(load_heeg(folder / f"S1-T{int(st)}.txt", folder / f"S1-T{int(st)}.json"))
        pm["paradigm"] = np.array(para, dtype=object).reshape(1, -1)
        rd_arr = np.empty(len(rd), dtype=object)
        rd_arr[:] = rd
        pm["rd"] = rd_arr.reshape(1, -1)
        sites.append(pm)
    return struct_array(sites)


def _adata_path(exp_id, adata_dir, el) -> tuple[Path, str, str]:
    name, mouse, user, _ = get_adata_filename(int(exp_id), adata_dir, el)
    if not name:
        raise FileNotFoundError(f"found no adata file for ExpID {int(exp_id)}")
    return Path(adata_dir) / user / mouse / name, mouse, user


def _cells(v) -> list:
    return as_cell_list(v) if not isinstance(v, dict) else [v]


def _spines_proj_meta(proj_id, uid, acodes, sel, el, adata_dir, meta_info, meta_info_columns, lab_books_dir):
    sites = []
    for knd, cur_site in enumerate(sel.site_ids):
        cur_exps = sel.exp_ids[sel.exp_ids[:, 0] == cur_site, 1]
        log.info("************ Adding site %d to meta file ************", cur_site)
        pm = MatStruct()
        cells: dict = {}
        for tp, e in enumerate(cur_exps):
            path, mouse, _ = _adata_path(e, adata_dir, el)
            orig = Adata(path)
            regions = _cells(orig["spiny_regions"])
            if tp == 0:
                pm.update(projID=proj_id, uuid=uid, Acode=_acode_str(acodes), exp_type=6, animal=mouse,
                          animalNbr=sel.animal_ids.index(mouse) + 1, siteNbr=int(sel.site_order[knd]),
                          cortical_layer=sel.locations[knd], ExpGroup=cur_exps.reshape(1, -1),
                          lab_notes=_lab_notes(mouse, el, lab_books_dir), nbr_spiny_regions=len(regions))
            spines, dendrite = _cells(orig["spines"]), _cells(orig["dendrite"])
            first = e == cur_site
            tracking = _cells(orig["spine_tracking"]) if first else None
            dlen = _cells(orig["dendrite_length"]) if first else None
            for sr in range(len(regions)):
                r = MatStruct(timepoint=tp + 1, comments=sel.comments[sel.stack_ids[:, 1] == e])
                r.update(_meta_info_values(meta_info, meta_info_columns, e))
                r["spiny_region"] = regions[sr]
                r["spines"] = empty() if is_empty(spines[sr]) else spines[sr]
                r["dendrite"] = empty() if is_empty(dendrite[sr]) else dendrite[sr]
                if first:
                    r["spine_tracking"] = empty() if is_empty(tracking[sr]) else tracking[sr]
                    r["dendrite_length"] = empty() if is_empty(dlen[sr]) else dlen[sr]
                else:
                    r["spine_tracking"] = empty()
                    r["dendrite_length"] = empty()
                cells[(sr, tp)] = r
        n_rows = max((k[0] for k in cells), default=-1) + 1
        pm["rd"] = grid_struct_array(cells, n_rows, len(cur_exps))
        sites.append(pm)
    return sites


def _lab_notes(mouse, el, lab_books_dir):
    if lab_books_dir is None:
        log.info("no lab_books_dir given: lab_notes not searched")
        notes = ["No lab notes found"]
    else:
        notes = lab_note_page_get(mouse, el, lab_books_dir)
    arr = np.empty(len(notes), dtype=object)
    arr[:] = notes
    return arr.reshape(1, -1)


# --------------------------------------------------------------------------------------
# generate_proj_meta
# --------------------------------------------------------------------------------------
def _is_cell(v) -> bool:
    if not isinstance(v, np.ndarray) or v.dtype != object:
        return False
    return v.size == 0 or not isinstance(v.ravel(order="F")[0], dict)


def _matlab_length(v) -> int:
    if isinstance(v, dict):
        return 1
    a = np.asarray(v, dtype=object) if not isinstance(v, np.ndarray) else v
    return int(max(a.shape)) if a.size else 0


def _n_cols(v) -> int:
    if isinstance(v, dict):
        return 1
    if isinstance(v, np.ndarray):
        return v.shape[1] if v.ndim == 2 else v.size
    return len(v)


def _basename(p) -> str:
    return re.split(r"[\\/]", str(p))[-1]


def generate_proj_meta(
    proj_id: str,
    acodes,
    *,
    exp_log,
    adata_dir: str | os.PathLike,
    raw_data_dir: str | os.PathLike | None = None,
    file_types=None,
    chans_to_include=None,
    channel_resolution: Mapping | Callable | None = None,
    meta_info=None,
    meta_info_columns=(),
    raw_act: bool = False,
    downsample_behavior: bool = False,
    uuid: str | None = None,
    main_channel: str = "ch525",
    secondary_channels=(),
    lab_books_dir: str | os.PathLike | None = None,
    heeg: bool = False,
    filtered_exp_log: bool = False,
) -> np.ndarray:
    """Assemble the project meta file of a project (calliope ``24_ProjMeta/generate_proj_meta.m``).

    For every site (all stacks of ``proj_id`` with the given Acodes, data not deleted)
    and every timepoint (experiment of the site), the experiment's files are loaded with
    :func:`calliopy.data.load_exp`, the aux channels are processed
    (:func:`assign_aux_method`, :func:`assign_processed_aux`) and sampled at the frame
    times, and the ROI activity is smoothed and normalised (two-photon:
    :func:`get_smoothed_rois`; widefield: :func:`widefield_rois_for_proj_meta`).
    Spine-imaging projects (adata with ``spiny_regions``) and human EEG projects
    (``heeg``) have their own layout.

    Parameters
    ----------
    proj_id : str
        Project ID.
    acodes : number or sequence of numbers
        Analysis codes to include (MATLAB asks; see :func:`project_acodes`).
    exp_log : ExpLog
        The ExpLog, or a filtered one (:func:`calliopy.explog.qexp` +
        :func:`calliopy.explog.format_qexp_output`).
    adata_dir : path
        Adata root (MATLAB ``set_lab_paths``).
    raw_data_dir : path
        Raw data root ``<raw>/<pi>/<animal>/S1-T...`` (MATLAB ``get_data_path``); needed
        except for spine projects.
    file_types : sequence of str
        File types to load (MATLAB asks among :func:`calliopy.data.list_file_types`; its
        advice: all except '.wid' and '.bin'). Required except for spine and hEEG
        projects.
    chans_to_include : sequence of int or str
        Aux channels to include: 0-based indices into the aligned channel list
        (:func:`align_aux_channels`) or channel names (MATLAB asks; not Shutter or
        FrameGalvo). Required when the project has ``.lvd`` files.
    channel_resolution : mapping or callable, optional
        Answers for channels that differ between experiments (:func:`align_aux_channels`).
    meta_info : sequence of rows, optional
        The MetaInfo sheet (header first, ExpID in the first column); needed with
        ``meta_info_columns``.
    meta_info_columns : sequence of str
        MetaInfo columns copied into every ``rd`` entry (MATLAB
        ``pdef.meta_info_columns``).
    raw_act : bool, default False
        Also store the unsmoothed activity (``act_raw``).
    downsample_behavior : bool, default False
        Without image data, keep every 10th aux sample (MATLAB asks).
    uuid : str, optional
        Default: a new random UUID.
    main_channel, secondary_channels
        Channels of the project (MATLAB ``ProjectDefaults_<proj>``).
    lab_books_dir : path, optional
        Lab books root for ``lab_notes`` (:func:`lab_note_page_get`); without it
        ``lab_notes`` is ``['No lab notes found']``.
    heeg : bool, default False
        Build the human EEG variant (:func:`heeg_proj_meta`).
    filtered_exp_log : bool, default False
        ``exp_log`` is a filtered ExpLog: it is stored in ``proj_meta[0].ExpLog`` (MATLAB
        does so when the 'explog' argument is given; the file name gets '_snob', see
        :func:`proj_meta_filename`).

    Returns
    -------
    numpy.ndarray
        The proj_meta struct array, shape (1, nSites). Site fields ``projID``, ``uuid``,
        ``Acode``, ``nbr_piezo_layers``, ``animal``, ``animalNbr`` and ``siteNbr``
        (1-based numbers), ``cortical_layer``, ``ExpGroup``, ``lab_notes``, ``rd``
        ([nLayers x nTimepoints]). ``rd`` fields as MATLAB (``timepoint`` and
        ``piezo_layer`` 1-based, ``frame_times`` and ``lickTimes`` 1-based sample indices,
        ``y_resets`` 1-based).

    Raises
    ------
    NotImplementedError
        File types whose processing needs functions that are not ported: eye tracking
        ('.eye', '.eyeR'), tongue video ('.ton'), body video ('.vid', '.vid3'), EEG
        ('EEG.dat'); also everything :func:`calliopy.data.load_exp` defers.
    ValueError
        Missing answers (``file_types``, ``chans_to_include``, ``raw_data_dir``) or bad
        Acodes.
    ImportError
        ``psmooth``/``ftfil``/``smooth2`` of :mod:`calliopy.analysis` are not available.

    Differences from MATLAB
    -----------------------
    - Every question is an argument; nothing is saved and ``pm_info`` is not run (call
      :func:`proj_meta_filename`, :func:`pm_info` and a writer yourself).
    - The MetaInfo sheet and the project defaults are arguments (MATLAB reads
      ``MetaInfo-<proj>.xlsx`` and ``ProjectDefaults_<proj>``).
    - ``act2spks`` (spike estimation) is used when :mod:`calliopy.analysis` provides it;
      otherwise ``spks`` is left out with a log message, as MATLAB does when it fails.
    - Without an adata file for the first experiment the project is not treated as spine
      imaging (MATLAB fails on the missing file).
    - Kept from MATLAB: variables loaded for one experiment stay available for the next
      ones (MATLAB loads into its workspace), the number of layers of non-two-photon data
      is ``length(ROIs)``, and the 'No ROIs' test uses the ROI counts of earlier
      experiments for layers the current one does not have.
    """
    el = as_exp_log(exp_log)
    uid = str(_uuid.uuid4()) if uuid is None else str(uuid)
    sel = proj_meta_selection(el, proj_id, acodes)
    if sel.exp_ids.shape[0] == 0:
        raise ValueError(f"no experiments for project {proj_id} with Acodes {acodes}")
    acode_s = _acode_str(acodes)
    animal_col = np.asarray(el["animalid"], dtype=object)
    expid_col = np.asarray(el["expid"], dtype=np.int64)
    pi_col = np.asarray(el["pi"], dtype=object)

    def folder_of(e) -> Path:
        i = int(np.flatnonzero(expid_col == e)[0])
        return Path(raw_data_dir) / str(pi_col[i]) / str(animal_col[i])

    if heeg:
        if raw_data_dir is None:
            raise ValueError("raw_data_dir is required (MATLAB get_data_path)")
        paths = [folder_of(e) for e in sel.exp_ids[:, 1]]
        out = heeg_proj_meta(proj_id, uid, el, acodes, sel.site_ids, sel.exp_ids, paths)
        return out

    try:
        first_path, _, _ = _adata_path(sel.exp_ids[0, 1], adata_dir, el)
        spines_flag = "spiny_regions" in Adata(first_path).variables
    except FileNotFoundError:
        spines_flag = False
    if spines_flag:
        sites = _spines_proj_meta(proj_id, uid, acodes, sel, el, adata_dir, meta_info, meta_info_columns,
                                  lab_books_dir)
        if filtered_exp_log:
            sites[0]["ExpLog"] = el
        return struct_array(sites)

    if raw_data_dir is None:
        raise ValueError("raw_data_dir is required (MATLAB get_data_path)")
    if file_types is None:
        raise ValueError("file_types is required (MATLAB asks which file types to include)")
    file_types = [str(f) for f in file_types]
    all_types = sorted({t for e in sel.exp_ids[:, 1] for t in list_file_types(int(e), raw_data_dir, el)})
    has_aux = any(".lvd" in t for t in all_types)
    lft = assign_load_ftypes(file_types)
    deferred = [k for k in _DEFERRED_LFT if lft[k]] + (["EEG.dat"] if "EEG.dat" in file_types else [])
    if deferred:
        raise NotImplementedError(f"generate_proj_meta: file types not ported yet: {deferred}")

    total_chan_list: list = []
    chan_id: list = []
    include: list[int] = []
    if has_aux:
        chans = [load_ach_chans(folder_of(e) / f"S1-T{int(st)}.ach")
                 for e, st in zip(sel.stack_ids[:, 1], sel.stack_ids[:, 2], strict=True)]
        total_chan_list, chan_id, _ = align_aux_channels(chans, sel.exp_index, sel.unique_stacks,
                                                         channel_resolution)
        if chans_to_include is None:
            raise ValueError(f"chans_to_include is required; channels: {total_chan_list}")
        for c in chans_to_include:
            include.append(total_chan_list.index(c) if isinstance(c, str) else int(c))

    exp_type, has_image_data, has_secondary_2p = proj_meta_exp_type(file_types)
    nbr_piezo_layers = 1
    ws: dict = {"frame_times": np.zeros((0, 0))}
    n_rois_hist: list[int] = []
    paradigms = sel.paradigms
    nan_paradigm = paradigms is None or all(
        (x is None) or (isinstance(x, str) and x == "") or (isinstance(x, float) and np.isnan(x))
        for x in np.ravel(paradigms))
    sites = []
    for knd, cur_site in enumerate(sel.site_ids):
        cur_exps = sel.exp_ids[sel.exp_ids[:, 0] == cur_site, 1]
        log.info("************ Adding site %d to meta file ************", cur_site)
        pm = MatStruct()
        cells: dict = {}
        for tp, e in enumerate(cur_exps):
            e = int(e)
            stack_sel = sel.stack_ids[:, 1] == e
            folder = folder_of(e)
            vrd = parse_vrd(folder / f"S1-T{e}.vrd")[0] if lft.vrd else None
            ws.update(load_exp(e, file_types, el, raw_data_dir=raw_data_dir, adata_dir=adata_dir,
                               main_channel=main_channel, secondary_channels=secondary_channels))
            orig: dict = {}
            no_rois: list[int] = []
            if has_image_data:
                path, mouse_id, _ = _adata_path(e, adata_dir, el)
                ad = Adata(path)
                for k in ("ROIs", "template", "act_map", "dx", "dy", "nbr_frames", "fnames", "meta_info"):
                    if k in ad.variables:
                        orig[k] = ad[k]
                sec_template = ad["template_sec"] if (has_secondary_2p and "template_sec" in ad.variables) else None
                if exp_type == 0:
                    if tp == 0:
                        ini = read_ini(folder / f"S1-T{e}_ch525.ini")
                        try:
                            nbr_piezo_layers = int(np.ravel(ini["piezo"]["nbrlayers"])[0])
                        except (KeyError, IndexError, TypeError):
                            nbr_piezo_layers = 1
                else:
                    if "ROIs" not in ws:
                        raise KeyError("non-two-photon data need ROIs in the adata file (MATLAB length(ROIs))")
                    nbr_piezo_layers = _matlab_length(ws["ROIs"])
                if _is_cell(orig.get("ROIs")):
                    for k in ("ROIs", "template", "act_map", "dx", "dy"):
                        if k in orig:
                            orig[k] = as_cell_list(orig[k])
                else:
                    for k in ("ROIs", "template", "act_map", "dx", "dy"):
                        if k in orig:
                            orig[k] = [orig[k]]
                rois_l = orig.get("ROIs", [])
                for y, r in enumerate(rois_l):
                    if y < len(n_rois_hist):
                        n_rois_hist[y] = _n_cols(r)
                    else:
                        n_rois_hist.append(_n_cols(r))
                no_rois = [0] * nbr_piezo_layers
                if n_rois_hist and all(n == 1 for n in n_rois_hist):
                    log.info("No ROIs selected")
                    no_rois = [1] * nbr_piezo_layers
                else:
                    for z in range(nbr_piezo_layers):
                        try:
                            if "activity" not in elements(rois_l[z])[0]:
                                log.info("Activity has not been calculated for layer %d", z + 1)
                                no_rois[z] = 1
                        except (IndexError, TypeError, KeyError):
                            log.info("There is a problem with layer %d", z + 1)
                            no_rois[z] = 1
            else:
                mouse_id = str(animal_col[np.flatnonzero(expid_col == e)[0]])
                sec_template = None

            if nan_paradigm:
                session = _parse_comment(np.ravel(sel.comments[stack_sel]))
            else:
                session = paradigms[stack_sel]

            analysed: list = []
            if "aux_data" in ws and has_aux:
                aux_data = np.asarray(ws["aux_data"], dtype=np.float64)
                uexp = int(np.flatnonzero(sel.exp_ids[:, 1] == e)[0])
                for c in include:
                    name = total_chan_list[c]
                    methods = assign_aux_method(name, proj_id)
                    hit = np.flatnonzero(chan_id[uexp] == c)
                    if hit.size:
                        pdata = assign_processed_aux(aux_data[hit[0], :], methods, name, None,
                                                     float(ws.get("aux_scanrate", 1000)))
                    else:
                        pdata = [(name, np.full((1, aux_data.shape[1]), np.nan))]
                    analysed.extend([list(p) for p in pdata])

            frame_times = np.ravel(np.asarray(ws.get("frame_times", np.zeros(0)), dtype=np.float64))
            for zl in range(nbr_piezo_layers):
                if tp == 0:
                    pm.update(projID=proj_id, uuid=uid, Acode=acode_s, nbr_piezo_layers=nbr_piezo_layers,
                              animal=mouse_id, animalNbr=sel.animal_ids.index(mouse_id) + 1,
                              siteNbr=int(sel.site_order[knd]), cortical_layer=sel.locations[knd],
                              ExpGroup=cur_exps.reshape(1, -1), lab_notes=_lab_notes(mouse_id, el, lab_books_dir))
                r = MatStruct(timepoint=tp + 1)
                r.update(_meta_info_values(meta_info, meta_info_columns, e))
                r["piezo_layer"] = zl + 1
                r["comments"] = sel.comments[stack_sel]
                for p in analysed:
                    if p[0][:1].isdigit():
                        p[0] = "Z" + p[0]
                if lft.owl:
                    r["owl"] = ws.get("owl", empty())
                ft_index = frame_times[zl::nbr_piezo_layers]
                if has_image_data:
                    if exp_type == 0:
                        r["fnames"] = np.array([_basename(p) for p in as_cell_list(orig.get("fnames", []))],
                                               dtype=object).reshape(1, -1)
                        if no_rois[zl] == 0:
                            res = get_smoothed_rois(orig["ROIs"][zl])
                            r["act"], r["ROIinfo"] = res.act, res.roi_info
                            if res.act_fissa.size:
                                r["act_fissa"] = res.act_fissa
                            if raw_act:
                                r["act_raw"] = np.vstack([np.ravel(x["activity"]) for x in elements(orig["ROIs"][zl])])
                            try:
                                r["spks"] = analysis("act2spks")(res.act - 1)
                            except Exception as err:  # noqa: BLE001 - MATLAB: try/catch with a message
                                log.info("Spike estimation failed - see act2spks for documentation (%s)", err)
                        r["dx"] = orig["dx"][zl]
                        r["dy"] = orig["dy"][zl]
                        r["nbr_frames"] = np.asarray(orig["nbr_frames"], dtype=np.float64) / nbr_piezo_layers
                        r["template"] = orig["template"][zl]
                        r["act_map"] = orig["act_map"][zl]
                    if exp_type == 4:
                        r["nbr_frames"] = _nbr_frames_widefield(orig["meta_info"]) / nbr_piezo_layers
                        layer = orig["ROIs"][zl]
                        endings = _wf_endings(list(elements(layer)[0].keys()))
                        if len(endings) > 1:
                            frs = [read_frame_rate(folder / f"S1-T{e}{x}.wii") for x in endings]
                        else:
                            frs = [read_frame_rate(folder / f"S1-T{e}.wii")]
                        if any(f is None for f in frs):
                            raise ValueError(f"no frame rate in the .wii file(s) of ExpID {e}")
                        r.update(widefield_rois_for_proj_meta(layer, frs, raw_act=raw_act, exp_id=e))
                    r["frame_times"] = ft_index.reshape(1, -1)
                    if sec_template is not None and not is_empty(sec_template):
                        r["template_sec"] = as_cell_list(sec_template)[zl] if _is_cell(sec_template) else sec_template
                    idx = ft_index.astype(np.int64) - 1
                    for name, data in analysed:
                        if name in ("lickTimes", "rTMS_stim"):
                            r[name] = data
                        elif not is_empty(data):
                            r[re.sub(r"\s+", "_", name)] = np.asarray(data, dtype=np.float64).ravel()[idx] \
                                .astype(np.float32).reshape(1, -1)
                        else:
                            r[name] = data
                else:
                    for name, data in analysed:
                        key = name.replace(" ", "_")
                        d = np.asarray(data)
                        if downsample_behavior:
                            d = d.ravel()[::10].reshape(1, -1) if d.ndim == 2 and d.shape[0] == 1 else d.ravel()[::10]
                        r[key] = d
                r["session"] = session
                if lft.vrd:
                    r["vrd"] = vrd
                    try:
                        yp = np.ravel(np.asarray(r["VRypos"], dtype=np.float64))
                        r["y_resets"] = (np.flatnonzero(np.diff(yp) < -4.5) + 2).reshape(1, -1)
                    except KeyError:
                        pass
                cells[(zl, tp)] = r
        n_rows = max((k[0] for k in cells), default=-1) + 1
        pm["rd"] = grid_struct_array(cells, n_rows, len(cur_exps))
        sites.append(pm)
    if filtered_exp_log:
        sites[0]["ExpLog"] = el
    return struct_array(sites)
