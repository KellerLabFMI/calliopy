"""Saving OII results into adata files: the ``savemaps`` part of calliope
``13_OII/analyzeOII_Fourier.m``."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

from ..io import MatStruct, save_mat

__all__ = ["save_analyze_oii_fourier_results"]

logger = logging.getLogger(__name__)


def save_analyze_oii_fourier_results(adata_path: str | PathLike, result: Any) -> Path:
    """Save the Fourier maps of an OII recording (``analyzeOII_Fourier(..., 'savemaps', 1)``).

    MATLAB creates the folder and writes ``Adata-S1-T<ExpID>.mat`` there with the single
    variable ``result`` (``Ibv``, ``FourierSpecs``, ``Power``, ``Phase``, ``PhaseMap``,
    ``PhaseMapHor``, ``PhaseMapVer``, ``PhaseHor``, ``PhaseVer``), with a plain ``save``:
    an existing file is replaced as a whole. Here the same.

    Parameters
    ----------
    adata_path
        The file (MATLAB: ``<adata_dir>/<userID>/<mouse_id>/Adata-S1-T<ExpID>.mat``,
        :func:`calliopy.io.adata_file_path`).
    result
        :class:`calliopy.oii.OiiFourierResult` or its ``result`` struct.

    Returns
    -------
    pathlib.Path

    Notes
    -----
    Kept from MATLAB: any other variable of an existing file at that path is lost (MATLAB
    ``save`` without ``-append``). v7.3 only (MATLAB used its default format).
    """
    res = result.result if hasattr(result, "result") and not isinstance(result, Mapping) else result
    path = Path(adata_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_mat(path, {"result": MatStruct(res)})
    logger.info(" * Saved maps!")
    return path
