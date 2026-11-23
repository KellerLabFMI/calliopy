"""Reading imaging and video files (two-photon, widefield, ROI traces, eye monitor, ...).

Every reader takes a file path (several readers also take a list of paths, like the cell
arrays of file names in MATLAB) and returns values; nothing is written to the caller's
workspace and nothing is printed. MATLAB's ``disp('NC WARNING ...')`` messages go to
:mod:`logging` (logger ``calliopy.io._imaging``); MATLAB's ``warning(...)`` calls are
:func:`warnings.warn`.

=============================  ===============================  ==============================
Python                         MATLAB (calliope r2003)          Files
=============================  ===============================  ==============================
:func:`load_bin`               ``load_bin``                     two-photon ``.bin`` stacks
:func:`load_wid_data`          ``load_wid_data``                widefield ``.wid``
:func:`load_roi_data`          ``load_roi_data``                widefield ROI traces ``.roi``
:func:`load_wii_data`          ``extract_wii_file_platform``    ``.wii`` (ROI geometry, image)
:func:`load_eye_monitor_data`  ``load_eye_monitor_data``        eye-monitor video ``.eye``
:func:`load_vid_data`          ``load_vid_data`` (99_Archive)   ``.run`` ``.vid`` ``.vid2`` ``.ton``
:func:`load_aux_vid_data`      ``load_auxVid_data``             auxiliary video recorders
:func:`read_tif`               ``imread`` (in ``load_exp``)     ``.tif``
=============================  ===============================  ==============================

Array orientation is the one MATLAB returns, e.g. ``load_bin`` gives ``[x_res x y_res x
nFrames]`` and ``load_wid_data`` gives ``[size_y x size_x x nFrames]``. Raw image data is
returned as a read-only :class:`numpy.memmap` (a view on the file, nothing is read until
it is used) when a single file is read and all frames are wanted; pass ``copy=True`` for an
ordinary in-memory array. When several files are concatenated, or only some frames are
selected with a list of indices, the result is an in-memory array. A memory-mapped file
stays open (and, on Windows, cannot be deleted) until the array and all views of it are
garbage-collected.

The frame axis is last, as in MATLAB, but the files store frame after frame, so the
returned arrays are views with unusual strides: ``data[..., t]`` is one contiguous block
of the file, ``np.moveaxis(data, -1, 0)`` gives a frame-first view without copying.

Stored values are never changed: frame numbers, ``file_id`` and the other MATLAB index
values stay 1-based where MATLAB stored them 1-based.
"""

from __future__ import annotations

import logging
import re
import struct
import warnings
from collections.abc import Sequence
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from ._mat import MatStruct

__all__ = [
    "AuxVidData",
    "BinData",
    "EyeData",
    "RoiData",
    "VidData",
    "WidData",
    "WiiData",
    "load_aux_vid_data",
    "load_bin",
    "load_eye_monitor_data",
    "load_roi_data",
    "load_vid_data",
    "load_wid_data",
    "load_wii_data",
    "read_tif",
]

logger = logging.getLogger(__name__)

_PathArg = str | PathLike

# Number of leading pixels of every widefield / ROI frame that hold the camera header
# (BCD-coded frame number and timestamp). Fixed in the MATLAB code ("always 14").
_PCO_META_SIZE = 14


# ======================================================================================
# Result types
# ======================================================================================


class BinData(NamedTuple):
    """Result of :func:`load_bin` (MATLAB ``[data, nbr_frames]``)."""

    data: list[np.ndarray]
    nbr_frames: np.ndarray


class WidData(NamedTuple):
    """Result of :func:`load_wid_data` (MATLAB ``[wid_data, wid_meta_info]``)."""

    wid_data: np.ndarray
    wid_meta_info: MatStruct | np.ndarray


class RoiData(NamedTuple):
    """Result of :func:`load_roi_data` (MATLAB ``[roi_data, roi_meta_info]``)."""

    roi_data: np.ndarray
    roi_meta_info: MatStruct | np.ndarray


class WiiData(NamedTuple):
    """Result of :func:`load_wii_data` (MATLAB ``[wii_config, wii_mean_proj, frame_nonrotated]``)
    """

    wii_config: MatStruct
    wii_mean_proj: np.ndarray
    frame_nonrotated: np.ndarray


class EyeData(NamedTuple):
    """Result of :func:`load_eye_monitor_data` (MATLAB ``[idata, imeta_info]``)."""

    idata: np.ndarray
    imeta_info: np.ndarray


class VidData(NamedTuple):
    """Result of :func:`load_vid_data` (MATLAB ``[vdata, vmeta_info]``)."""

    vdata: np.ndarray
    vmeta_info: np.ndarray


class AuxVidData(NamedTuple):
    """Result of :func:`load_aux_vid_data` (MATLAB ``[vid_data, vid_meta_info]``)."""

    vid_data: np.ndarray
    vid_meta_info: MatStruct | np.ndarray


# ======================================================================================
# Shared helpers
# ======================================================================================


def _as_paths(paths: Any) -> list[Path]:
    """One path or a sequence of paths -> list of :class:`~pathlib.Path`."""
    if isinstance(paths, (str, PathLike)):
        items = [paths]
    else:
        try:
            items = list(paths)
        except TypeError:
            raise TypeError(
                f"expected a file path or a sequence of file paths, got {type(paths).__name__}; "
                "calliopy readers take paths, looking files up by ExpID is not supported"
            ) from None
    if not items:
        raise ValueError("no file given")
    out = []
    for item in items:
        if not isinstance(item, (str, PathLike)):
            raise TypeError(
                f"expected a file path, got {type(item).__name__}; "
                "calliopy readers take paths, looking files up by ExpID is not supported"
            )
        out.append(Path(item))
    return out


def _as_path(path: Any) -> Path:
    """A single file path (a sequence of paths is not accepted here)."""
    if not isinstance(path, (str, PathLike)):
        raise TypeError(
            f"expected a file path, got {type(path).__name__}; "
            "calliopy readers take paths, looking files up by ExpID is not supported"
        )
    return Path(path)


def _check_file(path: Path) -> int:
    """Raise if ``path`` is not a file; return its size in bytes."""
    if not path.is_file():
        raise FileNotFoundError(f"couldn't find {path}")
    return path.stat().st_size


def _read_head(path: Path, nbytes: int, what: str) -> bytes:
    with path.open("rb") as fh:
        head = fh.read(nbytes)
    if len(head) < nbytes:
        raise ValueError(f"{path.name} is too short ({len(head)} bytes) to hold {what}")
    return head


def _is_whole(x: float) -> bool:
    """True for finite numbers without a fractional part."""
    return bool(np.isfinite(x)) and float(x) == float(np.floor(x))


def _as_int(value: Any, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _memmap(path: Path, dtype: Any, offset: int, shape: tuple[int, ...]) -> np.ndarray:
    """Read-only memory map; an empty in-memory array when there is nothing to map."""
    if 0 in shape:
        return np.empty(shape, dtype=dtype)
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)


def _native(arr: np.ndarray) -> np.ndarray:
    """In-memory copy in native byte order (a no-copy pass-through is not wanted here)."""
    return np.array(arr, dtype=arr.dtype.newbyteorder("="))


def _finish(view: np.ndarray, copy: bool) -> np.ndarray:
    return _native(view) if copy else view


def _concat(parts: list[np.ndarray], copy: bool) -> np.ndarray:
    """Join per-file arrays along the last (frame) axis; one part stays a view."""
    if len(parts) == 1:
        return _finish(parts[0], copy)
    out = np.concatenate(parts, axis=-1)
    return out if out.dtype.isnative else _native(out)


def _empty_matlab() -> np.ndarray:
    """MATLAB's ``[]``."""
    return np.empty((0, 0))


def _struct_row(structs: list[MatStruct]) -> MatStruct | np.ndarray:
    """A MATLAB 1xN struct array as a ``(1, N)`` object array; ``struct()`` when empty."""
    if not structs:
        return MatStruct()
    out = np.empty((1, len(structs)), dtype=object)
    for k, s in enumerate(structs):
        out[0, k] = s
    return out


# -- frame selection -------------------------------------------------------------------


def _parse_frames(
    frames: str | int | None, nbr_frames: Sequence[int], paths: Sequence[Path]
) -> tuple[bool, list[slice | np.ndarray] | None]:
    """Translate the ``frames`` option of the widefield / ROI / video readers.

    Returns ``(load_img, selection)``. ``selection`` is ``None`` when every frame is
    wanted, otherwise one zero-based ``slice`` or index array per file.
    """
    if frames is None or (isinstance(frames, str) and frames == ""):
        return True, None
    step = None
    if isinstance(frames, str):
        if frames == "meta":
            return False, None
        if frames in ("first", "firstlast", "100th"):
            sel: list[slice | np.ndarray] = []
            for n, p in zip(nbr_frames, paths):
                wanted = {"first": [1], "firstlast": [1, n], "100th": [100]}[frames]
                bad = [f for f in wanted if f < 1 or f > n]
                if bad:
                    raise ValueError(
                        f"{p.name} has {n} frame(s), cannot load frame {bad[0]} (frames={frames!r})"
                    )
                idx = np.array(wanted, dtype=np.intp) - 1
                # a single frame stays a slice, so one file can remain a memory map
                sel.append(slice(int(idx[0]), int(idx[0]) + 1) if idx.size == 1 else idx)
            return True, sel
        if re.fullmatch(r"\s*\+?\d+\s*", frames):
            step = int(frames)
        else:
            raise ValueError(
                f"frames={frames!r}: use 'first', 'firstlast', '100th', 'meta' or a number "
                "(load every n-th frame)"
            )
    elif isinstance(frames, (int, np.integer)) and not isinstance(frames, bool):
        step = int(frames)
    else:
        raise TypeError(f"frames must be a string, an integer or None, got {frames!r}")
    if step < 1:
        raise ValueError(f"frames={frames!r}: the step must be >= 1")
    return True, [slice(0, n, step) for n in nbr_frames]


def _select(view: np.ndarray, sel: slice | np.ndarray | None) -> np.ndarray:
    """Select frames (last axis) of a per-file view."""
    return view if sel is None else view[..., sel]


# -- camera header ---------------------------------------------------------------------


def _decode_pco_headers(hdr: np.ndarray, where: str) -> tuple[np.ndarray, list[datetime]]:
    """Decode the BCD camera header of ``n`` frames (an ``(n, 14)`` array).

    Pixels 1-4 hold the frame number as 8 decimal digits, pixels 5-6 the year, 7-11 month,
    day, hour, minute and second, and pixels 12-14 six digits of microseconds; each pixel
    holds two BCD digits (``0x23`` means 23).
    """
    hdr = np.asarray(hdr, dtype=np.int64)
    hi, lo = hdr >> 4, hdr & 0xF
    bad = (hdr > 0xFF) | (hi > 9) | (lo > 9)
    if bad.any():
        k = int(np.flatnonzero(bad.any(axis=1))[0])
        raise ValueError(
            f"{where}: the header of frame {k + 1} is not BCD-coded: {hdr[k].tolist()}"
        )
    dec = hi * 10 + lo
    framenumber = ((dec[:, 0] * 100 + dec[:, 1]) * 100 + dec[:, 2]) * 100 + dec[:, 3]
    year = dec[:, 4] * 100 + dec[:, 5]
    micro = (dec[:, 11] * 100 + dec[:, 12]) * 100 + dec[:, 13]
    stamps = []
    for k in range(len(dec)):
        try:
            stamps.append(
                datetime(
                    int(year[k]), int(dec[k, 6]), int(dec[k, 7]), int(dec[k, 8]),
                    int(dec[k, 9]), int(dec[k, 10]), int(micro[k]),
                )
            )
        except ValueError as exc:
            raise ValueError(f"{where}: frame {k + 1} has an invalid timestamp ({exc})") from None
    return framenumber, stamps


def _pco_structs(
    headers: np.ndarray, dimension: np.ndarray, file_id: int, where: str
) -> list[MatStruct]:
    """Per-frame meta structs (framenumber, timestamp, dimension, file_id) of one file."""
    framenumber, stamps = _decode_pco_headers(headers, where)
    out = []
    for k, stamp in enumerate(stamps):
        out.append(
            MatStruct(
                [
                    ("framenumber", int(framenumber[k])),
                    ("timestamp", stamp),
                    ("dimension", np.array(dimension[k], dtype=np.int64)[None, :]),
                    ("file_id", file_id),
                ]
            )
        )
    return out


# ======================================================================================
# load_bin
# ======================================================================================


def load_bin(
    path: _PathArg | Sequence[_PathArg],
    nbr_piezo_layers: int = 1,
    *,
    layer_shift: int = 0,
    copy: bool = False,
) -> BinData:
    """Load two-photon ``.bin`` stacks, split by piezo layer.

    Port of ``load_bin.m``; also covers the inline ``.bin`` reading of ``load_exp.m``
    (lines 233-310), which parses the same format.

    File format (little-endian, as MATLAB's default ``fopen`` on the lab's Windows/x86
    machines): an ``int16`` ``x_res`` and an ``int16`` ``y_res``, followed by the frames
    back to back, ``int16`` each. A frame is ``x_res`` rows of ``y_res`` samples (the
    sample index along ``y_res`` runs fastest in the file), so ``data[l][:, :, t]`` has
    shape ``(x_res, y_res)`` as in MATLAB. Frame ``k`` (counting over all files, from 0)
    belongs to piezo layer ``k % nbr_piezo_layers``.

    Parameters
    ----------
    path
        One ``.bin`` file or a sequence of files. Frames of several files are joined in
        the order given.
    nbr_piezo_layers
        Number of piezo layers (``piezo.nbrlayers`` in the experiment's ``.ini`` file;
        see :func:`calliopy.io.read_ini`). Trailing frames that do not fill a whole
        layer cycle are dropped from every file.
    layer_shift
        Rotate the layer list like MATLAB's ``circshift(data, layer_shift)``. 0 (default)
        keeps the file order; see *Differences from MATLAB*.
    copy
        Return in-memory arrays instead of memory maps.

    Returns
    -------
    BinData
        ``data``: list of ``nbr_piezo_layers`` ``int16`` arrays of shape
        ``(x_res, y_res, nbr_frames_total // nbr_piezo_layers)`` (MATLAB: a ``1 x
        nbr_piezo_layers`` cell). For one file with ``copy=False`` they are read-only
        memory maps; several files are concatenated in memory.
        ``nbr_frames``: ``(1, n_files)`` integer array, frames per file after dropping
        the incomplete trailing layer cycle.

    Raises
    ------
    FileNotFoundError
        A file does not exist.
    ValueError
        A file is too short for its header, has a non-positive resolution, or the files
        differ in resolution.

    Notes
    -----
    Differences from MATLAB:

    - ``load_exp.m`` reads the same format inline but takes the number of frames per file
      and the frame size from the adata object (``adata.nbr_frames``,
      ``adata.template``) instead of from the file; for consistent data the result is the
      same. Use this function for ``sec_fnames`` (second channel) files as well.
    - Takes file paths only. MATLAB also accepts a numeric ExpID, in which case it
      looks the files up in the ExpLog and reads ``nbr_piezo_layers`` from the ``.ini``;
      here pass the paths and ``nbr_piezo_layers`` explicitly.
    - Memory-mapped result (zero-copy) instead of a fully loaded array; layers are
      strided views of the file.
    - The frame count is ``(bytes - 4) // (2 * x_res * y_res)``. MATLAB used
      ``round(bytes / x_res / y_res / 2)``, which rounds a partial last frame up and
      then fails in ``reshape``. An incomplete trailing frame is dropped (logged).
    - MATLAB never printed its "More frames in bin file than expected" warning (the
      check ran after the remainder was removed); here dropped frames are logged.
    - Files with different resolutions raise instead of printing nothing and failing
      later.
    - The "horrible hack" of ``load_bin.m`` (``circshift(data, -1)`` when a file name
      contains ``S1-T231419``, a stack whose z planes were misaligned) is not applied
      automatically, because it depends on the file name. Pass ``layer_shift=-1`` to
      reproduce it for that stack.
    - No ``disp`` output (progress is logged at INFO level).
    """
    paths = _as_paths(path)
    n_lay = _as_int(nbr_piezo_layers, "nbr_piezo_layers", minimum=1)
    layer_shift = _as_int(layer_shift, "layer_shift")

    infos = []  # (path, x_res, y_res, usable frames)
    for p in paths:
        size = _check_file(p)
        x_res, y_res = (int(v) for v in np.frombuffer(_read_head(p, 4, "x_res and y_res"), "<i2"))
        if x_res < 1 or y_res < 1:
            raise ValueError(f"{p.name}: invalid resolution x_res={x_res}, y_res={y_res}")
        frame_bytes = 2 * x_res * y_res
        n_all, extra = divmod(size - 4, frame_bytes)
        n = n_all - n_all % n_lay
        if extra or n != n_all:
            logger.warning(
                "%s: %d complete frame(s) and %d stray byte(s) at the end are ignored "
                "(frames per file must fill whole cycles of %d piezo layer(s))",
                p.name, n_all - n, extra, n_lay,
            )
        infos.append((p, x_res, y_res, n))
    if len({(i[1], i[2]) for i in infos}) > 1:
        raise ValueError(
            "frame size changes between files: "
            + ", ".join(f"{i[0].name}: {i[1]}x{i[2]}" for i in infos)
        )
    x_res, y_res = infos[0][1], infos[0][2]

    layers: list[list[np.ndarray]] = [[] for _ in range(n_lay)]
    for p, _, _, n in infos:
        logger.info("loading file %s", p)
        raw = _memmap(p, "<i2", 4, (n, x_res, y_res))  # frame-major, y_res fastest
        view = raw.transpose(1, 2, 0)  # (x_res, y_res, n) as in MATLAB
        for lay in range(n_lay):
            layers[lay].append(view[:, :, lay::n_lay])
    data = [_concat(parts, copy) for parts in layers]
    if layer_shift % n_lay:
        s = layer_shift % n_lay
        data = [data[(j - s) % n_lay] for j in range(n_lay)]  # np.roll semantics
    nbr_frames = np.array([[i[3] for i in infos]], dtype=np.int64)
    return BinData(data, nbr_frames)


# ======================================================================================
# load_wid_data
# ======================================================================================


def load_wid_data(
    path: _PathArg | Sequence[_PathArg],
    frames: str | int | None = None,
    *,
    copy: bool = False,
) -> WidData:
    """Load widefield imaging data (``.wid`` files, PCO camera via LabVIEW).

    Port of ``load_wid_data.m``.

    File format (little-endian): ``nbr_frames`` records, each ``uint16 dim_y, uint16
    dim_x`` followed by ``size_y * size_x`` ``uint16`` pixels with ``y`` running fastest
    (MATLAB ``fread(fi, [size_y, size_x])``). The frame size is taken from the first
    record's two header values (``size_y``, ``size_x``). The first 14 pixels of
    each frame hold the camera header: frame number and timestamp as BCD digits.

    Parameters
    ----------
    path
        One ``.wid`` file or a sequence of files (MATLAB: cell array of file names).
    frames
        Which frames of each file to load; frame numbers are 1-based as in MATLAB.

        ===============  ==========================================================
        ``None``         all frames
        ``'first'``      the first frame of each file
        ``'firstlast'``  the first and the last frame of each file
        ``'100th'``      the 100th frame of each file
        ``'meta'``       no images, only the per-frame meta info of all frames
        ``n`` / ``'n'``  every ``n``-th frame, starting with the first (1, 1+n, ...)
        ===============  ==========================================================
    copy
        Return an in-memory array instead of a memory map.

    Returns
    -------
    WidData
        ``wid_data``: ``uint16`` array ``(size_y, size_x, n_frames)``; with
        ``frames='meta'`` an empty ``(0, 0)`` array (MATLAB ``[]``). For a single file
        with ``frames=None``, ``'first'``, ``'100th'`` or an every-``n``-th selection this
        is a read-only memory map unless ``copy=True``; ``'firstlast'`` and several files
        give an in-memory array.

        ``wid_meta_info``: with ``frames='meta'`` a ``(1, n_frames)`` object array of
        :class:`~calliopy.io.MatStruct` (MATLAB: 1 x N struct array) with fields
        ``framenumber`` (int), ``timestamp`` (:class:`datetime.datetime`), ``dimension``
        (``(1, 2)`` int array, the two header values of the frame) and ``file_id`` (1-based
        index into ``path``). In every other mode an empty ``MatStruct`` (MATLAB
        ``struct()``), as in MATLAB.

    Raises
    ------
    FileNotFoundError
        A file does not exist.
    ValueError
        Unknown ``frames`` option, a requested frame beyond the end of a file, files with
        different frame sizes, or a camera header that is not valid BCD / a valid date.

    Notes
    -----
    Differences from MATLAB:

    - ``verbose`` and ``pco_meta_size`` (documented as "not meant to be changed") are
      not parameters; there is no progress bar or ``disp`` output.
    - ``timestamp`` is a :class:`datetime.datetime` (microsecond resolution, no time
      zone), not a MATLAB ``datetime``. ``framenumber`` is parsed from the BCD digits
      directly; MATLAB evaluated the hex string with ``str2num``.
    - Memory-mapped result for a single file. The MATLAB check that aborts with empty
      outputs when the data would exceed the physical memory is dropped (a memory map
      needs no memory; ``copy=True`` raises :class:`MemoryError` as numpy does).
    - Files with different frame sizes raise a :class:`ValueError`; MATLAB printed
      "there is no fix for this - panic" and carried on.
    - An unknown ``frames`` string raises. MATLAB evaluated it with ``str2num`` and loaded
      zero frames when it was not a number. ``frames='100th'`` on a shorter file raises
      (MATLAB failed in the assignment).
    - The per-frame ``dim`` header values are only read for ``frames='meta'``; they are
      not compared with the frame size (neither does MATLAB).
    - A file size that is not a whole number of frames is floored (logged), as in MATLAB.
    """
    paths = _as_paths(path)
    infos = []  # (path, size_y, size_x, nbr_frames)
    for p in paths:
        size = _check_file(p)
        size_y, size_x = (int(v) for v in np.frombuffer(_read_head(p, 4, "the frame size"), "<u2"))
        if size_x < 1 or size_y < 1:
            raise ValueError(f"{p.name}: invalid frame size {size_y}x{size_x}")
        n, extra = divmod(size, (size_x * size_y + 2) * 2)
        if extra:
            logger.warning(
                "NC WARNING - nbr frames is not an integer - attemping to floor (%s)", p.name
            )
        infos.append((p, size_y, size_x, n))
    if len({(i[1], i[2]) for i in infos}) > 1:
        raise ValueError(
            "frame size changes between files: "
            + ", ".join(f"{i[0].name}: {i[1]}x{i[2]}" for i in infos)
        )
    size_y, size_x = infos[0][1], infos[0][2]
    load_img, sel = _parse_frames(frames, [i[3] for i in infos], paths)
    if not load_img and size_x * size_y < _PCO_META_SIZE:
        raise ValueError(f"frames of {size_y}x{size_x} pixels cannot hold the camera header")

    dtype = np.dtype([("dim", "<u2", (2,)), ("img", "<u2", (size_x, size_y))])
    parts: list[np.ndarray] = []
    metas: list[MatStruct] = []
    for k, (p, _, _, n) in enumerate(infos):
        logger.info("loading %s: %d frame(s)", p, n)
        rec = _memmap(p, dtype, 0, (n,))
        if load_img:
            view = rec["img"].transpose(2, 1, 0)  # (size_y, size_x, n), y fastest per frame
            parts.append(_select(view, None if sel is None else sel[k]))
        else:
            header = np.array(rec["img"].reshape(n, size_x * size_y)[:, :_PCO_META_SIZE])
            metas += _pco_structs(header, np.array(rec["dim"]), k + 1, p.name)
        del rec
    if not load_img:
        return WidData(_empty_matlab(), _struct_row(metas))
    return WidData(_concat(parts, copy), MatStruct())


# ======================================================================================
# load_roi_data and load_wii_data
# ======================================================================================


def load_roi_data(
    path: _PathArg | Sequence[_PathArg],
    frames: str | int | None = None,
    *,
    wii_path: _PathArg | Sequence[_PathArg] | None = None,
    copy: bool = False,
) -> RoiData:
    """Load widefield ROI traces (``.roi`` files) of the imaging platform.

    Port of ``load_roi_data.m``. The number of ROIs per frame is not stored in the
    ``.roi`` file but in the companion ``.wii`` file (``param.numberROIs``, read with
    :func:`load_wii_data`).

    File format (little-endian ``uint16`` throughout): two values ``size_x, size_y``;
    then a coordinate table of ``size_x * size_y`` values (ROI and reference-point
    coordinates; the MATLAB reader skips it and does not return it); then the frames,
    each a 14-value camera header (BCD frame number and timestamp, see
    :func:`load_wid_data`) followed by ``nbr_ROIs`` trace values.

    Parameters
    ----------
    path
        One ``.roi`` file or a sequence of files.
    frames
        Frame selection, same options as in :func:`load_wid_data` (``None``, ``'first'``,
        ``'firstlast'``, ``'100th'``, ``'meta'``, every ``n``-th).
    wii_path
        The ``.wii`` file(s) to take ``param.numberROIs`` from; default is the ``.roi``
        path with the suffix changed to ``.wii``.
    copy
        Return an in-memory array instead of a memory map.

    Returns
    -------
    RoiData
        ``roi_data``: ``uint16`` array ``(nbr_ROIs, 1, n_frames)``; with
        ``frames='meta'`` an empty ``(0, 0)`` array. ``roi_meta_info``: as in
        :func:`load_wid_data`, with ``dimension`` ``= [nbr_ROIs, 1]``; an empty
        ``MatStruct`` unless ``frames='meta'``.

    Raises
    ------
    FileNotFoundError
        A ``.roi`` or ``.wii`` file does not exist.
    ValueError
        Files with different coordinate tables or ROI counts, a file shorter than its
        header, ``param.numberROIs`` missing in the ``.wii`` file, or the errors listed in
        :func:`load_wid_data`.

    Notes
    -----
    Differences from MATLAB:

    - ``verbose`` and ``pco_meta_size`` are not parameters; no progress output.
    - The ``.wii`` path is the ``.roi`` path with its *suffix* replaced. MATLAB used
      ``strrep(path, '.roi', '.wii')``, which also changed a ``.roi`` anywhere else in
      the path.
    - The frame count is computed exactly, ``(bytes - 4 - 2 * size_x * size_y) //
      (2 * (nbr_ROIs + 14))``. MATLAB left out the 4 header bytes and relied on
      flooring (and logged a spurious "not an integer" warning for every file).
    - Different coordinate-table sizes or ROI counts between files raise instead of
      printing "panic".
    - The coordinate table read by MATLAB is a local variable that is never returned;
      ROI coordinates and labels are in the ``.wii`` file (:func:`load_wii_data`).
    - Other differences are those listed for :func:`load_wid_data` (memory map, no memory
      check, ``datetime`` timestamps, errors for bad ``frames`` options).
    """
    paths = _as_paths(path)
    if wii_path is None:
        wii_paths = [p.with_suffix(".wii") for p in paths]
    else:
        wii_paths = _as_paths(wii_path)
        if len(wii_paths) != len(paths):
            raise ValueError(f"{len(paths)} .roi file(s) but {len(wii_paths)} .wii file(s)")

    infos = []  # (path, nbr_ROIs, (size_x, size_y), nbr_frames, data offset)
    for p, wii in zip(paths, wii_paths):
        config = load_wii_data(wii).wii_config
        param = config.get("param")
        n_roi = param.get("numberROIs") if isinstance(param, dict) else None
        if not isinstance(n_roi, (int, float, np.floating)) or not _is_whole(n_roi) or n_roi < 1:
            raise ValueError(f"{wii.name}: no valid param.numberROIs entry (got {n_roi!r})")
        n_roi = int(n_roi)
        size = _check_file(p)
        size_x, size_y = (
            int(v) for v in np.frombuffer(_read_head(p, 4, "the coordinate table size"), "<u2")
        )
        offset = 4 + 2 * size_x * size_y
        if size < offset:
            raise ValueError(f"{p.name} is too short ({size} bytes) for its coordinate table")
        n, extra = divmod(size - offset, 2 * (n_roi + _PCO_META_SIZE))
        if extra:
            logger.warning("%s: %d stray byte(s) after the last frame are ignored", p.name, extra)
        infos.append((p, n_roi, (size_x, size_y), n, offset))
    if len({i[2] for i in infos}) > 1:
        raise ValueError("size of the coordinate table changes between files")
    if len({i[1] for i in infos}) > 1:
        raise ValueError(
            "number of ROIs changes between files: "
            + ", ".join(f"{i[0].name}: {i[1]}" for i in infos)
        )
    n_roi = infos[0][1]
    load_data, sel = _parse_frames(frames, [i[3] for i in infos], paths)

    parts: list[np.ndarray] = []
    metas: list[MatStruct] = []
    for k, (p, _, _, n, offset) in enumerate(infos):
        logger.info("loading %s: %d frame(s)", p, n)
        raw = _memmap(p, "<u2", offset, (n, _PCO_META_SIZE + n_roi))
        if load_data:
            view = np.expand_dims(raw[:, _PCO_META_SIZE:].T, 1)  # (nbr_ROIs, 1, n)
            parts.append(_select(view, None if sel is None else sel[k]))
        else:
            dims = np.tile(np.array([n_roi, 1]), (n, 1))
            metas += _pco_structs(np.array(raw[:, :_PCO_META_SIZE]), dims, k + 1, p.name)
        del raw
    if not load_data:
        return RoiData(_empty_matlab(), _struct_row(metas))
    return RoiData(_concat(parts, copy), MatStruct())


def load_wii_data(path: _PathArg) -> WiiData:
    """Read a ``.wii`` file of the widefield imaging platform.

    Port of ``extract_wii_file_platform.m``. The file holds the acquisition settings and
    ROI definitions as a text string, the mean projection image and one non-rotated
    frame.

    File format (little-endian): ``uint32 n_chars``; ``n_chars`` ASCII characters of
    settings text; ``uint16 m1, uint16 m2`` and ``m1 * m2`` ``uint16`` values (mean
    projection, MATLAB shape ``(m2, m1)``, column-major); ``uint16 f1, uint16 f2`` and
    ``f1 * f2`` ``uint16`` values (frame, shape ``(f1, f2)``, column-major).

    The settings text is a sequence of ``label=value;`` pairs. A value that parses as a
    number (MATLAB ``str2double``) becomes a float, anything else stays a string. Dotted
    labels become nested structs: ``param.numberROIs=12;`` gives
    ``wii_config.param.numberROIs``. Labels with three or four parts are special-cased
    as in the MATLAB code, with preallocated struct arrays ``wii_config.ref`` (2 x 1),
    ``wii_config.roi.coord`` (818 x 1, fields ``X``, ``Y``, ``in_frame``) and
    ``wii_config.roi_label`` (818 x 1), indexed 1-based in MATLAB, 0-based here:

    ================================  ===================================================
    label                             stored at
    ================================  ===================================================
    ``ref.<name><k>.<field>``         ``ref[k - 1].<field>`` (``k`` = last character)
    ``roi.coord.<i>``                 ``roi.coord[i].Y`` (as in MATLAB)
    ``roi.coord.<i>.<field>``         ``roi.coord[i].<field>``
    ``<a>.label.<i>.<field>``         ``roi_label[i].<field>``
    ``<a>.<b>.<c>``                   ``<a>.<b>.<c>`` (nested)
    ``<a>.<b>.<i>.<field>``           ``<a>.<b>[i].<field>`` (struct array)
    ================================  ===================================================

    where ``i`` in a label is the ROI index as written in the file (0-based, LabVIEW).
    Struct arrays are ``(n, 1)`` object arrays of :class:`~calliopy.io.MatStruct`
    that, as in MATLAB, share the same fields in every element (``[]`` as an empty
    ``(0, 0)`` array where unset); the preallocated ones keep their MATLAB length.

    Parameters
    ----------
    path
        The ``.wii`` file.

    Returns
    -------
    WiiData
        ``wii_config``: settings struct (see above). ``wii_mean_proj``: ``uint16``
        ``(m2, m1)``. ``frame_nonrotated``: ``uint16`` ``(f1, f2)``.

    Raises
    ------
    FileNotFoundError
        The file does not exist.
    ValueError
        The file is shorter than its header says, or a label cannot be stored (an
        ``roi.coord`` / ``ref`` index that is not a number).

    Notes
    -----
    This function is public (the MATLAB function is a stand-alone entry point of the
    widefield folder): :func:`load_roi_data` needs it, and it is the only way to get
    the ROI coordinates, labels, reference points and mean image that belong to ``.roi``
    traces. It is named ``load_wii_data`` after its siblings.

    Differences from MATLAB:

    - No ``view_mean_proj`` option (it opened a GUI viewer); plot ``wii_mean_proj``
      instead.
    - A last ``label=value`` pair without a closing ``;`` is kept; MATLAB dropped it.
    - Labels with five or more parts are kept under their full dotted name (MATLAB
      ignored them). Labels whose parent struct does not exist yet are created (MATLAB
      raised "Reference to non-existent field"); the ``<a>.<b>.<i>.<field>`` case builds
      the struct array the code was evidently meant to (MATLAB's version always failed).
    - ``str2double`` extras (complex numbers, thousands separators) are not parsed; such
      values stay strings.
    - Arrays are copies in native byte order.
    """
    p = _as_path(path)
    _check_file(p)
    data = p.read_bytes()

    def need(pos: int, nbytes: int, what: str) -> None:
        if pos + nbytes > len(data):
            raise ValueError(f"{p.name} is too short: cannot read {what} at byte {pos}")

    need(0, 4, "the settings text length")
    (n_chars,) = struct.unpack_from("<I", data, 0)
    pos = 4
    need(pos, n_chars, "the settings text")
    text = data[pos : pos + n_chars].decode("latin-1")
    pos += n_chars
    arrays = []
    for what in ("mean projection", "non-rotated frame"):
        need(pos, 4, f"the size of the {what}")
        d1, d2 = struct.unpack_from("<2H", data, pos)
        pos += 4
        need(pos, 2 * d1 * d2, f"the {what}")
        flat = np.frombuffer(data, dtype="<u2", count=d1 * d2, offset=pos)
        pos += 2 * d1 * d2
        # MATLAB reads [d2, d1] for the mean projection and [d1, d2] for the frame
        shape = (d2, d1) if what == "mean projection" else (d1, d2)
        arrays.append(flat.reshape(shape, order="F").astype(np.uint16))
    return WiiData(_parse_wii_config(text), arrays[0], arrays[1])


_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?")


def _str2double(text: str) -> float:
    """The part of MATLAB ``str2double`` that settings values need; NaN if not a number."""
    t = text.strip()
    if _NUMBER.fullmatch(t):
        return float(t.replace("d", "e").replace("D", "e"))
    if t.lower().lstrip("+-") in ("inf", "infinity"):
        return float("-inf") if t.startswith("-") else float("inf")
    return float("nan")


class _StructArray:
    """A MATLAB struct array under construction (same fields in every element)."""

    def __init__(self, n: int, fields: Sequence[str] = (), column: bool = True) -> None:
        self.fields = list(fields)
        self.column = column
        self.items = [self._new() for _ in range(n)]

    def _new(self) -> MatStruct:
        return MatStruct((f, _empty_matlab()) for f in self.fields)

    def set(self, idx: int, field: str, value: Any) -> None:
        """``s(idx).field = value`` with 1-based ``idx``; grows the array like MATLAB."""
        if idx < 1:
            raise ValueError(f"struct array index must be a positive integer, got {idx}")
        while len(self.items) < idx:
            self.items.append(self._new())
        if field not in self.fields:
            self.fields.append(field)
            for item in self.items:
                item[field] = _empty_matlab()
        self.items[idx - 1][field] = value

    def to_numpy(self) -> np.ndarray:
        n = len(self.items)
        out = np.empty((n, 1) if self.column else (1, n), dtype=object)
        for k, item in enumerate(self.items):
            out[(k, 0) if self.column else (0, k)] = item
        return out


def _index(text: str, label: str, offset: int = 0) -> int:
    """MATLAB ``str2double(text) + offset`` as an integer index."""
    value = _str2double(text)
    if not _is_whole(value):
        raise ValueError(f"cannot store {label!r}: {text!r} is not an index")
    return int(value) + offset


def _substruct(parent: MatStruct, key: str, label: str) -> MatStruct:
    """The struct field ``parent[key]``, created when missing."""
    current = parent.get(key)
    if current is None:
        current = parent[key] = MatStruct()
    elif not isinstance(current, MatStruct):
        raise ValueError(f"cannot store {label!r}: {key!r} is already set to a non-struct value")
    return current


def _struct_array(parent: MatStruct, key: str, label: str, create: bool = False) -> _StructArray:
    """The struct array ``parent[key]`` (created when ``create``)."""
    current = parent.get(key)
    if current is None and create:
        current = parent[key] = _StructArray(0, column=False)
    if not isinstance(current, _StructArray):
        raise ValueError(f"cannot store {label!r}: {key!r} is not a struct array")  # noqa: TRY004
    return current


def _update_wii_config(config: MatStruct, label: str, raw_value: str) -> None:
    """Port of ``update_config`` in ``extract_wii_file_platform.m``."""
    parts = label.split(".")
    num = _str2double(raw_value)
    value: Any = raw_value if np.isnan(num) else np.float64(num)

    if len(parts) == 1:
        config[label] = value
    elif len(parts) == 2:
        _substruct(config, parts[0], label)[parts[1]] = value
    elif len(parts) == 3:
        if parts[0] == "ref":
            _struct_array(config, "ref", label).set(_index(parts[1][-1:], label), parts[2], value)
        elif parts[0] == "roi" and parts[1] == "coord":
            coord = _struct_array(_substruct(config, "roi", label), "coord", label)
            coord.set(_index(parts[2], label, 1), "Y", value)  # sets Y, as in MATLAB
        else:
            _substruct(_substruct(config, parts[0], label), parts[1], label)[parts[2]] = value
    elif len(parts) == 4:
        idx = _index(parts[2], label, 1)
        if parts[1] == "label":
            _struct_array(config, "roi_label", label).set(idx, parts[3], value)
        elif parts[0] == "roi" and parts[1] == "coord":
            coord = _struct_array(_substruct(config, "roi", label), "coord", label)
            coord.set(idx, parts[3], value)
        else:
            parent = _substruct(config, parts[0], label)
            _struct_array(parent, parts[1], label, create=True).set(idx, parts[3], value)
    else:
        config[label] = value


def _finalize_struct_arrays(obj: Any) -> Any:
    if isinstance(obj, _StructArray):
        return obj.to_numpy()
    if isinstance(obj, MatStruct):
        for key in list(obj.keys()):
            obj[key] = _finalize_struct_arrays(obj[key])
    return obj


def _parse_wii_config(text: str) -> MatStruct:
    """Parse the ``label=value;`` settings text of a ``.wii`` file."""
    config = MatStruct()
    config["roi"] = MatStruct(coord=_StructArray(818, ("X", "Y", "in_frame")))
    config["ref"] = _StructArray(2)
    config["roi_label"] = _StructArray(818)

    label = value = ""
    in_label = True
    for ch in text:
        if ch == "=":
            in_label = False
            value = ""
        elif ch == ";":
            if label:
                _update_wii_config(config, label, value)
            in_label = True
            label = ""
        elif in_label:
            label += ch
        else:
            value += ch
    if label and not in_label:  # last pair without ';' (MATLAB drops it)
        _update_wii_config(config, label, value)
    return _finalize_struct_arrays(config)


# ======================================================================================
# load_eye_monitor_data
# ======================================================================================

_EYE_META = 9  # doubles of meta info in front of every frame


def load_eye_monitor_data(
    path: _PathArg,
    load_img: bool = True,
    *,
    frame_size: tuple[int, int] | None = None,
    copy: bool = False,
) -> EyeData:
    """Load an eye-monitor video file (``.eye``, ``.eyeR``).

    Port of ``load_eye_monitor_data.m``.

    File format: frame after frame, each ``9`` ``float64`` meta values followed by
    ``size_x * size_y`` ``uint8`` pixels, the ``x`` index running fastest (MATLAB
    ``reshape(fread(...), size_x, size_y)``). The frame size is meta values 5 and 6
    (``size_x``, ``size_y``) of the first frame. Files are little-endian; very old files
    are big-endian and are recognised by a non-integer frame size when read as
    little-endian (then, if meta value 3 is a non-zero integer, the frame size is
    ``meta[5] - meta[3]`` by ``meta[6] - meta[4]``, as in MATLAB, for the oldest format
    that stored the whisker region first).

    Parameters
    ----------
    path
        The ``.eye`` file.
    load_img
        ``False`` reads only the meta info and returns an empty ``idata``.
    frame_size
        ``(size_x, size_y)`` to use instead of the header values; see *Differences from
        MATLAB*. The size is then fixed for the whole file.
    copy
        Return ``idata`` as an in-memory array instead of a memory map.

    Returns
    -------
    EyeData
        ``idata``: ``uint8`` array ``(size_x, size_y, n_frames)``, a read-only memory map
        unless ``copy=True`` (or the frame size changes within the file, see below);
        an empty ``(0, 0)`` array when ``load_img`` is false. ``imeta_info``: ``float64``
        ``(9, n_frames)`` meta values, one column per frame (always in memory).

    Raises
    ------
    FileNotFoundError
        The file does not exist.
    ValueError
        The file is shorter than one meta block, or the frame size is not a positive
        integer.

    Warns
    -----
    UserWarning
        The frame size in the header is 0 (corrupt file; 744 x 480 is assumed, as in
        MATLAB), or the file ends in the middle of a frame (the incomplete frame is
        dropped).

    Notes
    -----
    If the frame size recorded in the individual frames changes during a little-endian
    recording, MATLAB (and this function) follow the changes ("attempting to fix"):
    frames are read one by one, ``idata`` is zero-padded to the largest frame, and the
    result is an in-memory array.

    Differences from MATLAB:

    - MATLAB hard-coded ``size_y = 284`` for any file whose *path* contains one of the
      stack numbers 128658, 128659, 128661, 128662 or 128664 ("I have no idea how this
      happened"). That rule is not applied; pass ``frame_size=(size_x, 284)`` for those
      files.
    - If the file does not exist MATLAB tried ``sec.eye`` instead of ``.eye`` (with a
      warning) and otherwise returned silently without outputs; here a missing file
      raises :class:`FileNotFoundError` and the caller picks the file name.
    - An incomplete last frame is dropped with a warning. MATLAB counted it as a frame
      (an all-zero image and a partly or fully zero meta column) and warned about the
      error.
    - The big-endian message says what happens (MATLAB's text said "little endian").
    - Memory-mapped ``idata`` (see the module docstring); no ``disp`` output.
    """
    p = _as_path(path)
    file_size = _check_file(p)
    head = _read_head(p, 8 * _EYE_META, "the first meta block")
    byte_order = "<"
    meta = np.frombuffer(head, "<f8")
    size_x, size_y = float(meta[4]), float(meta[5])
    update_size = True
    if not (_is_whole(size_x) and _is_whole(size_y)):
        logger.warning(
            "NC Warning: %s looks like an old big-endian file; reading it as big-endian", p.name
        )
        byte_order = ">"
        meta = np.frombuffer(head, ">f8")
        if meta[2] == 0 or not _is_whole(meta[2]):
            size_x, size_y = float(meta[4]), float(meta[5])
        else:  # very old data, where meta values 3 and 4 were not the whisker position
            size_x, size_y = float(meta[4] - meta[2]), float(meta[5] - meta[3])
        update_size = False
    if frame_size is not None:
        if len(frame_size) != 2:
            raise ValueError(f"frame_size must be (size_x, size_y), got {frame_size!r}")
        size_x, size_y = float(frame_size[0]), float(frame_size[1])
        update_size = False
    elif size_x == 0 or size_y == 0:
        warnings.warn(
            f"{p.name}: the file is probably corrupt (frame size {size_x:g} x {size_y:g}); "
            "assuming the whole 744 x 480 frame was recorded, this may well fail",
            stacklevel=2,
        )
        size_x, size_y = 744.0, 480.0
        update_size = False
    if not (_is_whole(size_x) and _is_whole(size_y) and size_x >= 1 and size_y >= 1):
        raise ValueError(f"{p.name}: invalid frame size {size_x:g} x {size_y:g}")
    sx, sy = int(size_x), int(size_y)

    rec_bytes = sx * sy + 8 * _EYE_META
    n_frames, extra = divmod(file_size, rec_bytes)
    if update_size:
        # The frame size may change mid recording ("attempting to fix" in MATLAB): the
        # fast path below is only valid if every record has the size of the first.
        records, partial = _scan_eye_records(p, byte_order, sx, sy)
        if len({(r[1], r[2]) for r in records}) > 1:
            return _read_eye_variable_size(p, byte_order, records, load_img)
        n_frames, extra = len(records), partial
    if extra:
        warnings.warn(
            f"{p.name}: the last {extra} byte(s) do not make a whole frame and are ignored",
            stacklevel=2,
        )
    dtype = np.dtype([("meta", f"{byte_order}f8", (_EYE_META,)), ("img", "u1", (sy, sx))])
    rec = _memmap(p, dtype, 0, (n_frames,))
    imeta_info = np.array(rec["meta"].T, dtype=np.float64)
    if not load_img:
        return EyeData(_empty_matlab(), imeta_info)
    idata = rec["img"].transpose(2, 1, 0)  # (size_x, size_y, n), x fastest per frame
    return EyeData(_finish(idata, copy), imeta_info)


def _scan_eye_records(
    p: Path, byte_order: str, size_x: int, size_y: int
) -> tuple[list[tuple[int, int, int]], int]:
    """Walk the headers of a little-endian ``.eye`` file.

    Returns ``([(byte offset, size_x, size_y), ...], stray bytes)`` for the complete
    records, following frame size changes announced in the headers like the MATLAB loop.
    """
    buf = np.memmap(p, dtype=np.uint8, mode="r")
    total = int(buf.size)
    head = 8 * _EYE_META
    records: list[tuple[int, int, int]] = []
    pos = 0
    while pos < total:
        if pos + head > total:
            break
        meta = np.frombuffer(buf[pos : pos + head].tobytes(), f"{byte_order}f8")
        if (size_x, size_y) != (meta[4], meta[5]):
            logger.warning(
                "NC WARNING: frame size appears to change mid recording - attempting to fix"
            )
            if not (_is_whole(meta[4]) and _is_whole(meta[5]) and meta[4] >= 1 and meta[5] >= 1):
                raise ValueError(f"{p.name}: invalid frame size {meta[4]:g} x {meta[5]:g}")
            size_x, size_y = int(meta[4]), int(meta[5])
        if pos + head + size_x * size_y > total:
            break
        records.append((pos, size_x, size_y))
        pos += head + size_x * size_y
    del buf
    return records, total - pos


def _read_eye_variable_size(
    p: Path, byte_order: str, records: list[tuple[int, int, int]], load_img: bool
) -> EyeData:
    """Read a ``.eye`` file whose frame size changes; ``idata`` is zero-padded."""
    buf = np.memmap(p, dtype=np.uint8, mode="r")
    head = 8 * _EYE_META
    imeta_info = np.empty((_EYE_META, len(records)))
    frames = []
    for k, (pos, size_x, size_y) in enumerate(records):
        imeta_info[:, k] = np.frombuffer(buf[pos : pos + head].tobytes(), f"{byte_order}f8")
        if load_img:
            pix = np.array(buf[pos + head : pos + head + size_x * size_y])
            frames.append(pix.reshape(size_y, size_x).T)  # (size_x, size_y), x fastest
    del buf
    if not load_img:
        return EyeData(_empty_matlab(), imeta_info)
    max_x = max(r[1] for r in records)
    max_y = max(r[2] for r in records)
    idata = np.zeros((max_x, max_y, len(records)), dtype=np.uint8)
    for k, f in enumerate(frames):
        idata[: f.shape[0], : f.shape[1], k] = f
    return EyeData(idata, imeta_info)


# ======================================================================================
# load_vid_data
# ======================================================================================

_VID_META = 4  # doubles of meta info in front of every frame
_VID_DTYPES = {"int16": "i2", "uint16": "u2"}


def load_vid_data(
    path: _PathArg,
    dtype: str = "int16",
    *,
    copy: bool = False,
) -> VidData:
    """Load a video file written by the LabVIEW video recorders (``.run``, ``.vid``, ...).

    Port of ``load_vid_data.m`` (``99_Archive/LoadSaveView``). ``load_exp.m`` uses it for
    the ``.run``, ``.vid``, ``.vid2`` (all with ``'int16'``) and ``.ton`` (default format)
    files.

    File format: frame after frame, each ``4`` ``float64`` meta values followed by
    ``size_x * size_y`` 16-bit pixels, the ``y`` index running fastest (MATLAB
    ``reshape(fread(...), size_y, size_x)'``). The frame size is meta values 3 and 4
    (``size_y``, ``size_x``) of the first frame. Files are little-endian; if the values
    read that way are not integers the file is read as big-endian (old files).

    Parameters
    ----------
    path
        The video file.
    dtype
        Pixel type, ``'int16'`` (default, what all calliope callers use) or ``'uint16'``
        (MATLAB argument ``format``).
    copy
        Return ``vdata`` as an in-memory array instead of a memory map.

    Returns
    -------
    VidData
        ``vdata``: array ``(size_x, size_y, n_frames)`` of the requested type, a
        read-only memory map unless ``copy=True``. For a big-endian file the memory map
        has big-endian dtype (``>i2``); ``copy=True`` returns native byte order.
        ``vmeta_info``: ``float64`` ``(4, n_frames)`` meta values, one column per frame.

    Raises
    ------
    FileNotFoundError
        The file does not exist.
    ValueError
        Unsupported ``dtype``, a file shorter than one meta block, or a frame size that is
        not a positive integer.

    Notes
    -----
    Differences from MATLAB:

    - Only ``'int16'`` and ``'uint16'`` are accepted. MATLAB's third branch (any other
      ``format``, in practice ``'int8'``) read one byte per pixel but computed the
      frame count and offsets for two bytes per pixel, so it could never read a file
      correctly.
    - A file with frame size 0 in the header is read as 1344 x 1024 (``size_y`` x
      ``size_x``) as in MATLAB, but this is logged as a warning.
    - Trailing bytes that do not make a whole frame are ignored (floored in MATLAB, silently)
      and logged.
    - Memory-mapped ``vdata``; no ``disp`` output.
    """
    if dtype not in _VID_DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_VID_DTYPES)}, got {dtype!r}")
    p = _as_path(path)
    file_size = _check_file(p)
    head = _read_head(p, 8 * _VID_META, "the first meta block")
    byte_order = "<"
    meta = np.frombuffer(head, "<f8")
    size_y, size_x = float(meta[2]), float(meta[3])
    if not (_is_whole(size_y) and _is_whole(size_x)):
        logger.warning("%s looks like an old big-endian file; reading it as big-endian", p.name)
        byte_order = ">"
        meta = np.frombuffer(head, ">f8")
        size_y, size_x = float(meta[2]), float(meta[3])
    if size_x == 0 or size_y == 0:
        logger.warning(
            "%s: frame size 0 in the header, assuming 1344 x 1024 (size_y x size_x)", p.name
        )
        size_x, size_y = 1024.0, 1344.0
    if not (_is_whole(size_x) and _is_whole(size_y) and size_x >= 1 and size_y >= 1):
        raise ValueError(f"{p.name}: invalid frame size {size_y:g} x {size_x:g} (size_y x size_x)")
    sx, sy = int(size_x), int(size_y)

    n_frames, extra = divmod(file_size, sx * sy * 2 + 8 * _VID_META)
    if extra:
        logger.warning("%s: %d stray byte(s) after the last frame are ignored", p.name, extra)
    pixel = byte_order + _VID_DTYPES[dtype]
    rec_dtype = np.dtype([("meta", f"{byte_order}f8", (_VID_META,)), ("img", pixel, (sx, sy))])
    rec = _memmap(p, rec_dtype, 0, (n_frames,))
    vmeta_info = np.array(rec["meta"].T, dtype=np.float64)
    vdata = rec["img"].transpose(1, 2, 0)  # (size_x, size_y, n), y fastest per frame
    return VidData(_finish(vdata, copy), vmeta_info)


# ======================================================================================
# load_aux_vid_data
# ======================================================================================

_AUX_PIXEL = {"uint8": "u1", "uint16": "<u2"}


def load_aux_vid_data(
    path: _PathArg | Sequence[_PathArg],
    frames: str | int | None = None,
    *,
    bit_depth: str = "uint16",
    copy: bool = False,
) -> AuxVidData:
    """Load video from the auxiliary video recorders.

    Port of ``load_auxVid_data.m``.

    File format (little-endian): frame after frame, each ``uint64`` timestamp
    (microseconds, system time), ``uint16 dim_y, uint16 dim_x`` and ``size_y * size_x``
    pixels of ``bit_depth`` with the ``y`` index running fastest (MATLAB ``fread(fi,
    [size_y, size_x])``). The frame size is the dimension pair of the first frame of each
    file.

    Parameters
    ----------
    path
        One file or a sequence of files.
    frames
        Frame selection, same options as in :func:`load_wid_data` (``None``, ``'first'``,
        ``'firstlast'``, ``'100th'``, ``'meta'``, every ``n``-th).
    bit_depth
        ``'uint16'`` (default) or ``'uint8'``: the pixel type of the file.
    copy
        Return an in-memory array instead of a memory map.

    Returns
    -------
    AuxVidData
        ``vid_data``: array ``(size_y, size_x, n_frames)`` of ``bit_depth``; with
        ``frames='meta'`` an empty ``(0, 0)`` array. ``vid_meta_info``: with
        ``frames='meta'`` a ``(1, n_frames)`` object array of
        :class:`~calliopy.io.MatStruct` with fields ``timestamp`` (int, microseconds),
        ``dimension`` (``(1, 2)`` int array, the frame's ``[dim_y, dim_x]`` header) and
        ``file_id`` (1-based index into ``path``); otherwise an empty ``MatStruct``.

    Raises
    ------
    FileNotFoundError
        A file does not exist.
    ValueError
        Unsupported ``bit_depth``, unknown ``frames`` option, a requested frame beyond the
        end of a file, or files with different frame sizes.

    Notes
    -----
    Differences from MATLAB:

    - MATLAB's frame skipping for ``frames`` selections computed the byte offset with
      ``(size_x + size_y)`` instead of ``size_x * size_y`` pixels, so every frame but
      the first was read from the wrong place. Here frames are indexed directly.
    - ``timestamp`` is an exact integer. MATLAB's ``fread`` returned a double, which is
      exact up to 2^53 (about 285 years of microseconds, so the same values in practice).
    - Other ``bit_depth`` values raise. MATLAB warned "Bit depth undefined. Set to
      uint16", but kept using the given name for the array type and the read, so only
      the offsets assumed 16 bits.
    - Files with different frame sizes raise; no memory check, no progress output, no
      ``verbose`` (see :func:`load_wid_data`).
    - A file size that is not a whole number of frames is floored (logged), as in MATLAB.
    """
    if bit_depth not in _AUX_PIXEL:
        raise ValueError(f"bit_depth must be one of {sorted(_AUX_PIXEL)}, got {bit_depth!r}")
    pixel = np.dtype(_AUX_PIXEL[bit_depth])
    paths = _as_paths(path)
    infos = []  # (path, size_y, size_x, nbr_frames)
    for p in paths:
        size = _check_file(p)
        head = _read_head(p, 12, "the first timestamp and frame size")
        size_y, size_x = (int(v) for v in np.frombuffer(head[8:12], "<u2"))
        if size_x < 1 or size_y < 1:
            raise ValueError(f"{p.name}: invalid frame size {size_y}x{size_x}")
        n, extra = divmod(size, 12 + size_x * size_y * pixel.itemsize)
        if extra:
            logger.warning(
                "NC WARNING - nbr frames is not an integer - attemping to floor (%s)", p.name
            )
        infos.append((p, size_y, size_x, n))
    if len({(i[1], i[2]) for i in infos}) > 1:
        raise ValueError(
            "frame size changes between files: "
            + ", ".join(f"{i[0].name}: {i[1]}x{i[2]}" for i in infos)
        )
    size_y, size_x = infos[0][1], infos[0][2]
    load_img, sel = _parse_frames(frames, [i[3] for i in infos], paths)

    dtype = np.dtype(
        [("ts", "<u8"), ("dim", "<u2", (2,)), ("img", pixel, (size_x, size_y))]
    )
    parts: list[np.ndarray] = []
    metas: list[MatStruct] = []
    for k, (p, _, _, n) in enumerate(infos):
        logger.info("loading %s: %d frame(s)", p, n)
        rec = _memmap(p, dtype, 0, (n,))
        if load_img:
            view = rec["img"].transpose(2, 1, 0)  # (size_y, size_x, n), y fastest per frame
            parts.append(_select(view, None if sel is None else sel[k]))
        else:
            ts, dims = np.array(rec["ts"]), np.array(rec["dim"], dtype=np.int64)
            for j in range(n):
                metas.append(
                    MatStruct(
                        [
                            ("timestamp", int(ts[j])),
                            ("dimension", dims[j][None, :].copy()),
                            ("file_id", k + 1),
                        ]
                    )
                )
        del rec
    if not load_img:
        return AuxVidData(_empty_matlab(), _struct_row(metas))
    return AuxVidData(_concat(parts, copy), MatStruct())


# ======================================================================================
# read_tif
# ======================================================================================


def read_tif(path: _PathArg, *, page: int | None = 0) -> np.ndarray:
    """Read a ``.tif`` image, as MATLAB ``imread`` does in ``load_exp.m`` (``tif_data``).

    Uses ``tifffile`` (a calliopy dependency), imported only when the function is called so
    that ``import calliopy.io`` stays fast. Compressed TIFFs (LZW, JPEG, ...) also need
    ``imagecodecs``, which is not installed by default.

    Parameters
    ----------
    path
        The TIFF file.
    page
        Page (image) to read, 0-based; default 0, the first image, which is what
        ``imread(path)`` returns. ``None`` reads all pages as one array.

    Returns
    -------
    numpy.ndarray
        The pixel data with its stored type (e.g. ``uint16``) and MATLAB's orientation:
        ``(rows, columns)`` or ``(rows, columns, channels)``. For a palette (indexed)
        image the index image is returned, without the colour map.

    Raises
    ------
    ImportError
        ``tifffile`` is not installed (the environment is out of date).
    FileNotFoundError
        The file does not exist.

    Notes
    -----
    Differences from MATLAB: none in the pixel data of grayscale and RGB images, also
    when the channels are stored as separate planes (returned channels-last, as by imread);
    ``imread`` of an indexed image also returns the colour map as a second output, which
    ``load_exp`` ignores.
    """
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError(
            "read_tif needs the 'tifffile' package, a calliopy dependency that is missing "
            "from this environment: run `uv sync` to install it"
        ) from exc
    p = _as_path(path)
    _check_file(p)
    with tifffile.TiffFile(p) as tif:
        if page is None:
            info = tif.pages[0]
            data = tif.asarray()
        else:
            info = tif.pages[_as_int(page, "page", minimum=0)]
            data = info.asarray()
    if info.samplesperpixel > 1 and info.planarconfig == tifffile.PLANARCONFIG.SEPARATE:
        # planar (one plane per channel) -> channels last, as MATLAB imread returns
        data = np.moveaxis(data, -3, -1)
    return np.asarray(data)
