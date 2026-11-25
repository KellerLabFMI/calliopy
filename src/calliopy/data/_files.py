"""Adata file lookup: calliope ``01_DataManagement/find_adata_file.m`` and
``01_DataManagement/get_adata_filename.m``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ..explog import as_exp_log, exp_log_rows, get_exp_log, read_info_from_exp_log

__all__ = ["FindAdataResult", "GetAdataFilenameResult", "find_adata_file", "get_adata_filename"]


class FindAdataResult(NamedTuple):
    """Outputs of :func:`find_adata_file` (MATLAB ``[adata_file, mouse_id, userID]``)."""

    adata_file: str
    mouse_id: str
    user_id: str


class GetAdataFilenameResult(NamedTuple):
    """Outputs of :func:`get_adata_filename` (MATLAB ``[adata_file, mouse_id, userID, projID]``)."""

    adata_file: str
    mouse_id: str
    user_id: str
    proj_id: str


def find_adata_file(exp_id: int, adata_dir: str | os.PathLike, exp_log=None) -> FindAdataResult:
    """Name of the adata file of an experiment (calliope ``01_DataManagement/find_adata_file.m``).

    Looks for ``<adata_dir>/<userID>/<mouse_id>/Adata-S1-T<exp_id>.mat``; otherwise the first
    file below that folder (recursively, sorted) whose path contains the ExpID.

    calliope marks this function obsolete in favour of :func:`get_adata_filename`; it is
    kept because existing analysis code calls it.

    Returns
    -------
    FindAdataResult
        File name (not the path), mouse ID and user ID; three empty strings when nothing
        is found.

    Differences from MATLAB
    -----------------------
    The recursive search goes through sorted paths (MATLAB ``dirrec`` order is that of the
    file system).
    """
    el = as_exp_log(exp_log if exp_log is not None else get_exp_log(suppress_warning=True))
    info = read_info_from_exp_log(exp_id, True, el).exp_id_info
    if "userID" not in info:
        return FindAdataResult("", "", "")
    folder = Path(adata_dir) / info.userID / info.mouse_id
    name = f"Adata-S1-T{int(exp_id)}.mat"
    if (folder / name).exists():
        return FindAdataResult(name, info.mouse_id, info.userID)
    if folder.is_dir():
        for p in sorted(folder.rglob("*")):
            if p.is_file() and str(int(exp_id)) in str(p):
                return FindAdataResult(p.name, info.mouse_id, info.userID)
    return FindAdataResult("", "", "")


def get_adata_filename(exp_id: int, adata_dir: str | os.PathLike, exp_log=None) -> GetAdataFilenameResult:
    """Name of the adata file of an experiment (calliope ``01_DataManagement/get_adata_filename.m``).

    The file is the one in ``<adata_dir>/<userID>/<mouse_id>/`` whose name between the first
    'T' and the 4-character extension is the ExpID.

    Returns
    -------
    GetAdataFilenameResult
        File name, mouse ID, user ID and project; the first three are empty strings when no
        file matches (the project is kept, as in MATLAB).

    Raises
    ------
    LookupError
        ``exp_id`` is not in the ExpLog (MATLAB fails on an empty index).

    Differences from MATLAB
    -----------------------
    ``adata_dir`` is required (MATLAB: ``set_lab_paths`` when called with one argument).
    """
    el = as_exp_log(exp_log if exp_log is not None else get_exp_log(suppress_warning=True))
    rows = exp_log_rows(el, exp_id)
    if rows.size == 0:
        raise LookupError(f"ExpID {exp_id} is not in the ExpLog")
    i = int(rows[0])
    user_id = str(np.asarray(el["pi"])[i])
    proj_id = str(np.asarray(el["project"])[i])
    mouse_id = str(np.asarray(el["animalid"])[i])
    folder = Path(adata_dir) / user_id / mouse_id
    if folder.is_dir():
        for p in sorted(folder.iterdir()):
            n = p.name
            if "T" in n and n[n.index("T") + 1 : len(n) - 4] == str(int(exp_id)):
                return GetAdataFilenameResult(n, mouse_id, user_id, proj_id)
    return GetAdataFilenameResult("", "", "", proj_id)
