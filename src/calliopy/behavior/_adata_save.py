"""Saving behaviour results into adata files: the save part of calliope
``10_Behavior/preprocess_owl.m``."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

from ..io import MatStruct, WritableAdata

__all__ = ["save_preprocess_owl_results"]

logger = logging.getLogger(__name__)


def save_preprocess_owl_results(
    adata_path: str | PathLike,
    result: Mapping[str, Any],
    *,
    create_dir: bool = True,
    adata_root: str | PathLike | None = None,
) -> Path:
    """Create the adata file of an overhead-camera recording (``preprocess_owl.m``).

    A new file is made with :meth:`calliopy.io.WritableAdata.create` (folders created as
    MATLAB) holding the single variable ``owl`` (``KP``, ``median_frame``, ``first_frame``,
    ``last_frame``) from :func:`calliopy.behavior.preprocess_owl`, and saved.

    Parameters
    ----------
    adata_path
        The new file (MATLAB: :func:`calliopy.io.adata_file_path`).
    result
        :func:`calliopy.behavior.preprocess_owl` output (``MatStruct(owl=...)``) or the
        ``owl`` struct itself.
    create_dir, adata_root
        Passed to :meth:`calliopy.io.WritableAdata.create`.

    Returns
    -------
    pathlib.Path

    Raises
    ------
    FileExistsError
        The file exists (MATLAB skipped the ExpID: "Adata file already exists").

    Notes
    -----
    Differences from MATLAB: ``owl.KP`` is written as a struct with the columns
    ``mouse_x`` and ``mouse_y`` (``[nFrames x 1]`` double) instead of a MATLAB ``table``,
    which calliopy cannot write. ``owl.KP.mouse_x`` reads the same in MATLAB;
    table-only syntax (``height(owl.KP)``, ``owl.KP{:, 1}``) does not. Finding whether an
    adata file exists for the ExpID under another name (``find_adata_file``) is the
    caller's.
    """
    owl = MatStruct(result.get("owl", result))
    if isinstance(owl.get("KP"), Mapping):
        owl["KP"] = MatStruct(owl["KP"])
    ad = WritableAdata.create(adata_path, create_dir=create_dir, adata_root=adata_root)
    ad.add_variable(owl=owl)
    ad.save_changes()
    return ad.path
