"""Changing variables of an existing MATLAB v7.3 MAT-file in place.

:func:`update_mat` replaces or adds variables of a MAT-file that MATLAB (or
:func:`calliopy.io.save_mat`) wrote, without rewriting the other variables. It is
what MATLAB does with ``save(filename, ..., '-append')`` on a v7.3 file and what
calliope's ``adata.save_changes`` relies on.

Values are encoded exactly as :func:`calliopy.io.save_mat` encodes them (the same
private writer is used); see the module docstring of ``calliopy.io._save``.

How the file is changed
-----------------------
In a v7.3 file, the elements of cell arrays and the field values of struct arrays
are separate HDF5 objects in the group ``#refs#``, reached through object
references. Replacing a variable therefore

1. finds the ``#refs#`` objects reachable from the old value, and those reachable
   from every other top-level object (other variables, ``#subsystem#``); only the
   ones reachable from the old value alone are deleted afterwards, so references
   shared with other variables (MATLAB shares, for example, one canonical empty
   value) stay valid;
2. writes the new value under a temporary top-level name (starting with ``#``,
   which readers skip), its cell and struct-array elements as new objects in
   ``#refs#``;
3. deletes the old variable, renames the temporary one to the variable name and
   deletes the exclusively owned ``#refs#`` objects.

HDF5 never returns freed space to the operating system: the file does not shrink
and grows by the size of the new values on every update. Re-save the file
(``save_mat(path, load_mat(path))`` or ``h5repack``) to compact it.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ._mat import _attr_str, _check_v73
from ._save import _check_name, _Writer

__all__ = ["delete_mat_variables", "update_mat"]

logger = logging.getLogger(__name__)

_REFS = "#refs#"
_TMP_PREFIX = "#calliopy_new#"
_MAT73_HEADER = b"MATLAB 7.3 MAT-file"


class _UpdateWriter(_Writer):
    """The writer of :func:`save_mat`, appending to an existing ``#refs#`` group.

    ``save_mat`` names the objects of ``#refs#`` ``a``, ``b``, ... from scratch; here the
    names already present in the file are skipped. Every name handed out is recorded
    so that a failed update can remove what it wrote.
    """

    def __init__(self, f: h5py.File, oned_as: str) -> None:
        super().__init__(f, oned_as)
        refs = f.get(_REFS)
        if refs is not None:
            if not isinstance(refs, h5py.Group):
                raise ValueError(f"{_REFS} in this file is not a group; not a MATLAB v7.3 layout")
            self._refs = refs
            self._taken: set[str] = set(refs.keys())
        else:
            self._taken = set()
        self.created: list[str] = []

    def _new_ref_name(self) -> str:
        while True:
            name = super()._new_ref_name()
            if name not in self._taken:
                self._taken.add(name)
                self.created.append(f"/{_REFS}/{name}")
                return name


def _check_mat73_header(path: Path) -> None:
    """Refuse HDF5 files without the MATLAB v7.3 header in their userblock."""
    with path.open("rb") as fh:
        head = fh.read(len(_MAT73_HEADER))
    if head != _MAT73_HEADER:
        raise ValueError(
            f"{path.name} is an HDF5 file without the MATLAB v7.3 MAT-file header; "
            "update_mat only changes MAT-files written by MATLAB or save_mat"
        )


def _ref_targets(f: h5py.File, ds: h5py.Dataset) -> Iterable[h5py.HLObject]:
    """Objects referenced by a dataset of object references (null and dangling skipped)."""
    refs = ds[()]
    for ref in np.ravel(refs):
        if not ref:
            continue
        try:
            yield f[ref]
        except (KeyError, ValueError, RuntimeError):
            # a reference to an object deleted earlier (garbage left in #refs#)
            continue


def _reachable_refs(
    f: h5py.File, roots: Iterable[h5py.HLObject], stop_when: set[str] | None = None
) -> set[str]:
    """Names of the ``#refs#`` objects reachable from ``roots``.

    Follows group membership and object references in datasets (MATLAB stores
    references nowhere else). With ``stop_when``, the walk ends as soon as every
    name in it has been found.
    """
    prefix = f"/{_REFS}/"
    found: set[str] = set()
    visited: set[str] = set()
    stack = list(roots)
    while stack:
        obj = stack.pop()
        if isinstance(obj, h5py.Group):
            stack.extend(obj.values())
        elif isinstance(obj, h5py.Dataset) and h5py.check_ref_dtype(obj.dtype) is h5py.Reference:
            for target in _ref_targets(f, obj):
                name = target.name
                if name is None or name in visited:
                    continue
                visited.add(name)
                if name.startswith(prefix):
                    found.add(name)
                stack.append(target)
        if stop_when is not None and stop_when <= found:
            break
    return found


def _is_object(obj: h5py.HLObject) -> bool:
    return isinstance(obj, h5py.Dataset) and "MATLAB_object_decode" in obj.attrs


def update_mat(
    path: str | PathLike,
    variables: Mapping[str, Any],
    *,
    oned_as: str = "row",
) -> None:
    """Replace or add variables in an existing MATLAB v7.3 MAT-file, in place.

    Python counterpart of MATLAB's built-in ``save(filename, ..., '-append')`` for v7.3
    files, as used by ``adata.save_changes`` in calliope (e.g. RoiQuest saving
    ``ROIs`` into an adata file).

    Parameters
    ----------
    path
        Existing v7.3 MAT-file, written by MATLAB or :func:`save_mat`. No ``.mat`` is
        appended to the name.
    variables
        Variable name -> value. A name already in the file is replaced, a new name is
        added; every other variable is left as it is, byte for byte. Values are
        encoded as :func:`save_mat` encodes them (``MatStruct``, struct arrays as
        object arrays of ``MatStruct``, cells, ``str``, logicals, empty values, ...).
        Stored index values (e.g. ROI ``indices``) are written as given; keep them
        1-based, as MATLAB stores them.
    oned_as
        ``"row"`` (default) saves 1-D arrays and lists as 1xN, ``"column"`` as Nx1.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If ``path`` is not a v7.3 MAT-file (a v5/v7 MAT-file, another file, or an
        HDF5 file without the MATLAB header), for invalid variable names, or for
        ``oned_as`` other than ``"row"``/``"column"``.
    TypeError
        For values MATLAB cannot hold (as :func:`save_mat`). The file is then left
        with its old content.

    Warns
    -----
    UserWarning
        When a replaced variable was a MATLAB object (table, string, datetime,
        classdef): its entries in ``#subsystem#`` are left in place.

    Notes
    -----
    The update is not atomic. New values are written completely before the old
    ones are removed, and the swap itself only renames HDF5 links, so the window
    in which an interruption leaves a variable missing is short; a crash or power
    loss while HDF5 writes its metadata can nevertheless corrupt the whole file.
    Keep a copy of files that matter before updating them.

    The file never shrinks: space of the replaced values stays allocated (see the
    module docstring). The 512-byte MATLAB header, including its creation date, is
    left unchanged.

    Differences from MATLAB
    -----------------------
    - Takes a mapping of variables instead of workspace variable names; v7.3 only.
    - New values are written uncompressed (MATLAB compresses with gzip).
    - MATLAB's header is not updated (MATLAB rewrites it on ``-append``; unverified).
    - ``#refs#`` objects that were already unreachable before the call (garbage left
      by earlier MATLAB or calliopy updates) are left in place.
    """
    if oned_as not in ("row", "column"):
        raise ValueError(f"oned_as must be 'row' or 'column', not {oned_as!r}")
    if not isinstance(variables, Mapping):
        raise TypeError("variables must be a mapping of variable name -> value")
    names = [_check_name(k, "variable") for k in variables]
    path = Path(path)
    _check_v73(path)
    _check_mat73_header(path)
    if not names:
        return

    with h5py.File(path, "r+") as f:
        replaced = [n for n in names if n in f]
        for name in replaced:
            if _is_object(f[name]):
                warnings.warn(
                    f"{name} is a MATLAB {_attr_str(f[name], 'MATLAB_class')} object; "
                    "its data in #subsystem# are left in the file",
                    stacklevel=2,
                )

        # 1. which #refs# objects only the replaced values use (read-only pass)
        doomed: set[str] = set()
        if replaced:
            doomed = _reachable_refs(f, [f[n] for n in replaced])
            if doomed:
                keep_roots = [f[k] for k in f if k != _REFS and k not in replaced]
                doomed -= _reachable_refs(f, keep_roots, stop_when=doomed)

        # 2. write the new values under temporary names
        writer = _UpdateWriter(f, oned_as)
        tmp_names = [f"{_TMP_PREFIX}{n}" for n in names]
        try:
            for name, tmp in zip(names, tmp_names):
                if tmp in f:
                    logger.warning("removing %s left over by an interrupted update", tmp)
                    del f[tmp]
                writer.write(f, tmp, variables[name])
        except BaseException:
            for obj_name in [*tmp_names, *writer.created]:
                if obj_name in f:
                    del f[obj_name]
            raise

        # 3. swap: delete the old variables, rename the new ones, drop orphaned refs
        for name, tmp in zip(names, tmp_names):
            if name in f:
                del f[name]
            f.move(tmp, name)
        for obj_name in sorted(doomed, key=len, reverse=True):
            if obj_name in f:
                del f[obj_name]
    logger.debug(
        "updated %s: %d replaced, %d added, %d orphaned #refs# object(s) removed",
        path.name,
        len(replaced),
        len(names) - len(replaced),
        len(doomed),
    )

def delete_mat_variables(path: str | PathLike, names: Iterable[str]) -> list[str]:
    """Remove variables from an existing MATLAB v7.3 MAT-file in place.

    The counterpart of MATLAB's ``rmvar`` / ``rmfield`` + re-save on an adata file. The
    ``#refs#`` objects (cell elements, struct-array fields) reachable only from the
    removed variables are deleted too, exactly as :func:`update_mat` does for replaced
    values; objects shared with a kept variable are left alone.

    Parameters
    ----------
    path
        Existing v7.3 MAT-file (with the MATLAB header).
    names
        Variable names to remove. Names not present, and ``#``-prefixed HDF5 internals,
        are ignored.

    Returns
    -------
    list of str
        The names that were removed, in the order given.

    Raises
    ------
    FileNotFoundError, ValueError
        As :func:`update_mat` for a missing, non-v7.3 or headerless file.

    Differences from MATLAB
    -----------------------
    - The file does not shrink (HDF5 keeps the freed space); MATLAB rewrites the file.
    - Not atomic: back the file up first, as for :func:`update_mat`.
    """
    path = Path(path)
    _check_v73(path)
    _check_mat73_header(path)
    with h5py.File(path, "r+") as f:
        present = [n for n in names if n in f and not n.startswith("#")]
        if not present:
            return []
        doomed = _reachable_refs(f, [f[n] for n in present])
        if doomed:
            keep = [f[k] for k in f if k != _REFS and k not in present]
            doomed -= _reachable_refs(f, keep, stop_when=doomed)
        for n in present:
            del f[n]
        for obj in sorted(doomed, key=len, reverse=True):
            if obj in f:
                del f[obj]
    return present
