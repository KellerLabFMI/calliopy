"""ROI objects of RoiQuest and their conversion to and from the adata ROI struct.

Port of calliope ``08_RoiQuest/+utils/Roi.m``, ``Roi_wf.m``, ``rois2struct.m``,
``struct2rois.m``, ``struct2wfRois.m`` and ``struct2wfRois_2ch.m`` (Leonardo Lupori, 2024).

Index conventions
-----------------
``Roi.indices`` holds the values MATLAB stores in the adata ``ROIs`` struct: **1-based linear
indices in column-major (MATLAB) order** into an ``img_h x img_w`` image, as float64 with
MATLAB's vector orientation (a row ``(1, N)`` stays a row, everything else is a column
``(N, 1)``; ``find`` results are columns). Use
``np.unravel_index(indices.ravel() - 1, (img_h, img_w), order="F")`` for 0-based numpy
subscripts. ``shift`` is the stored ``[dx, dy]`` (rows, columns) displacement in pixels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from calliopy._matlab import mround
from calliopy.io import MatStruct

from ._builtins import bwmorph_thicken, bwmorph_thin

__all__ = [
    "Roi",
    "RoiWf",
    "move_roi_structs",
    "rois2struct",
    "rois_binary_mask",
    "rois_label_image",
    "rois_label_type_image",
    "struct2rois",
    "struct2wf_rois",
    "struct2wf_rois_2ch",
]

_EMPTY = np.empty((0, 0))


# ---------------------------------------------------------------------------------------
# helpers shared by the package
# ---------------------------------------------------------------------------------------
def _is_empty(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v == ""
    if isinstance(v, Mapping):
        return False
    return np.size(v) == 0


def _struct_list(s) -> list:
    """MATLAB struct array (as read by calliopy.io, or a list) -> list of elements.

    ``None``, ``[]`` and a struct without fields (``struct()``) give an empty list.
    """
    if s is None:
        return []
    if isinstance(s, Mapping):
        return [s] if len(s) else []
    if isinstance(s, np.ndarray):
        if s.dtype != object:
            if s.size == 0:
                return []
            raise TypeError(f"expected a struct array, got a {s.dtype} array")
        return list(s.ravel(order="F"))
    return list(s)


def _struct_array(items: Sequence) -> np.ndarray:
    """List of MatStruct -> ``(1, N)`` object array (a MATLAB 1xN struct array)."""
    out = np.empty((1, len(items)), dtype=object)
    for i, it in enumerate(items):
        out[0, i] = it
    return out


def _cell_list(c) -> list:
    """MATLAB cell array (object array or sequence) -> list, column-major order."""
    if isinstance(c, np.ndarray) and c.dtype == object:
        return list(c.ravel(order="F"))
    if isinstance(c, (list, tuple)):
        return list(c)
    raise TypeError(f"expected a cell array (object array or list), got {type(c).__name__}")


def _cell_row(items: Sequence) -> np.ndarray:
    """List -> ``(1, N)`` object array (a MATLAB 1xN cell array)."""
    out = np.empty((1, len(items)), dtype=object)
    for i, it in enumerate(items):
        out[0, i] = it
    return out


def _get(roi, name: str, default=None):
    if isinstance(roi, Mapping):
        return roi.get(name, default)
    return getattr(roi, name, default)


def _lin(indices) -> np.ndarray:
    """1-based linear indices (any shape) -> 1-D int64 array (column-major order)."""
    a = np.asarray(indices, dtype=np.float64)
    if a.size and not np.all(a == np.round(a)):
        raise ValueError("ROI indices must be integers")
    return a.ravel(order="F").astype(np.int64)


def _is_row(indices) -> bool:
    a = np.asarray(indices)
    return a.ndim == 2 and a.shape[0] == 1 and a.size > 0


def _vector(values, row: bool) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).ravel()
    return v.reshape(1, -1) if row else v.reshape(-1, 1)


def _find(mask) -> np.ndarray:
    """MATLAB ``find(mask)``: 1-based column-major linear indices as an ``(N, 1)`` column."""
    m = np.asarray(mask)
    return (np.flatnonzero(m.ravel(order="F")) + 1).astype(np.float64).reshape(-1, 1)


def _ind2sub(hw, indices):
    """1-based linear -> 1-based (rows, cols), MATLAB ``ind2sub``."""
    lin = _lin(indices)
    h, w = int(hw[0]), int(hw[1])
    if lin.size and (lin.min() < 1 or lin.max() > h * w):
        raise ValueError("index exceeds the image size")
    r, c = np.unravel_index(lin - 1, (h, w), order="F")
    return r + 1, c + 1


def _sub2ind(hw, rows, cols) -> np.ndarray:
    """1-based (rows, cols) -> 1-based linear indices, MATLAB ``sub2ind`` (errors when out
    of range)."""
    h, w = int(hw[0]), int(hw[1])
    r = np.asarray(rows, dtype=np.float64)
    c = np.asarray(cols, dtype=np.float64)
    if r.size and (r.min() < 1 or r.max() > h or c.min() < 1 or c.max() > w
                   or not np.all(r == np.round(r)) or not np.all(c == np.round(c))):
        raise ValueError("Out of range subscript")
    return (r - 1) + (c - 1) * h + 1


def _shift_vec(shift) -> np.ndarray:
    if _is_empty(shift):
        return np.zeros((1, 2))
    return np.asarray(shift, dtype=np.float64).reshape(1, 2)


def _move_indices(indices, hw, dx, dy, global_xy):
    """Translate 1-based linear indices; pixels leaving the image are dropped."""
    row = _is_row(indices)
    if np.size(indices) == 0:
        return np.asarray(indices, dtype=np.float64).copy()
    r, c = _ind2sub(hw, indices)
    g = np.ravel(global_xy).astype(np.float64)
    r = r + dx + g[0]
    c = c + dy + g[1]
    keep = ~((r < 1) | (r > hw[0]) | (c < 1) | (c > hw[1]))
    return _vector(_sub2ind(hw, r[keep], c[keep]), row)


# ---------------------------------------------------------------------------------------
# classes
# ---------------------------------------------------------------------------------------
class Roi:
    """A region of interest, MATLAB ``utils.Roi`` (``08_RoiQuest/+utils/Roi.m``).

    Parameters
    ----------
    indices
        1-based column-major linear indices of the ROI pixels (stored adata values).
    img_hw
        ``(height, width)`` of the image (MATLAB default ``[NaN, NaN]``).
    type
        ROI tag, 0 (no type) to 9; ``None`` stands for MATLAB ``[]``.
    rng
        ``numpy.random.Generator`` for ``rand_color_idx`` (MATLAB ``randi([2, 11])``).

    Attributes
    ----------
    img_h, img_w, indices, shift, type, activity, activity_fissa_corrected, rand_color_idx,
    selected, to_delete, is_new
        The MATLAB properties in snake_case. ``shift`` is a ``(1, 2)`` float64 array.

    Notes
    -----
    MATLAB's methods can be called on object arrays; here methods act on one ROI and the
    array versions are the module functions :func:`rois_binary_mask`,
    :func:`rois_label_image` and :func:`rois_label_type_image` (``move``, ``expand`` and
    ``shrink`` on an array are loops over the ROIs).

    Differences from MATLAB: the random colour index cannot reproduce MATLAB's ``rng(42)``
    sequence.
    """

    def __init__(self, indices, img_hw=(np.nan, np.nan), type=0, *, rng=None):
        gen = np.random.default_rng() if rng is None else rng
        self.rand_color_idx = int(gen.integers(2, 12))
        ind = np.asarray(indices, dtype=np.float64)
        if ind.ndim > 2:
            raise ValueError("indices must be a vector")
        self.indices = ind if ind.ndim == 2 else ind.reshape(-1, 1)
        self.img_h = img_hw[0]
        self.img_w = img_hw[1]
        self.type = type
        self.shift = np.zeros((1, 2))
        self.activity = None
        self.activity_fissa_corrected = None
        self.selected = False
        self.to_delete = False
        self.is_new = False

    @property
    def _hw(self):
        return (int(self.img_h), int(self.img_w))

    def move(self, dx, dy, global_xy=(0, 0)) -> None:
        """Translate the ROI in place, MATLAB ``roi.move(dx, dy, globalXY)``.

        ``dx`` (rows) and ``dy`` (columns) are added to the pixel subscripts together with
        ``global_xy = [dx, dy]``; pixels that leave the image are removed. Only ``[dx, dy]``
        (not ``global_xy``) is accumulated in ``shift``, as in MATLAB.
        """
        self.indices = _move_indices(self.indices, self._hw, dx, dy, global_xy)
        self.shift = _shift_vec(self.shift) + np.array([[dx, dy]], dtype=np.float64)

    def to_binary_mask(self) -> np.ndarray:
        """``(img_h, img_w)`` boolean mask, MATLAB ``roi.toBinaryMask()``."""
        mask = np.zeros(self._hw, dtype=bool)
        mask.T.flat[_lin(self.indices) - 1] = True  # flat index of .T = column-major index
        return mask

    def to_label_image(self) -> np.ndarray:
        """``uint8`` image with ``rand_color_idx`` on the ROI, MATLAB ``roi.toLabelImage()``."""
        img = np.zeros(self._hw, dtype=np.uint8)
        img.T.flat[_lin(self.indices) - 1] = self.rand_color_idx
        return img

    def to_label_type_image(self) -> np.ndarray:
        """``uint8`` image with ``type + 1`` on the ROI (13 for an empty type), MATLAB
        ``roi.toLabelTypeImage()`` called on a single ROI."""
        lab = 13 if _is_empty(self.type) else float(np.ravel(self.type)[0]) + 1
        img = np.zeros(self._hw, dtype=np.uint8)
        img.T.flat[_lin(self.indices) - 1] = _u8(lab)
        return img

    def expand(self, amount=1) -> None:
        """Grow the ROI in place, MATLAB ``roi.expand(amount)`` (``bwmorph 'thicken'``)."""
        self.indices = _find(bwmorph_thicken(self.to_binary_mask(), amount))

    def shrink(self, amount=1) -> None:
        """Shrink the ROI in place, MATLAB ``roi.shrink(amount)`` (``bwmorph 'thin'``)."""
        self.indices = _find(bwmorph_thin(self.to_binary_mask(), amount))

    def to_struct(self) -> MatStruct:
        """``MatStruct`` with fields ``indices``, ``shift``, ``type`` (MATLAB ``toStruct``).

        Activity is not exported, as in MATLAB (saving ROIs deletes calculated activity).
        """
        return MatStruct(
            indices=self.indices,
            shift=_shift_vec(self.shift),
            type=0.0 if _is_empty(self.type) else self.type,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({np.size(self.indices)} px, type={self.type!r})"


def _u8(v) -> np.uint8:
    return np.uint8(min(max(float(mround(v)), 0), 255))


class RoiWf(Roi):
    """Widefield ROI, MATLAB ``utils.Roi_wf`` (``08_RoiQuest/+utils/Roi_wf.m``).

    Adds ``ml_index``, ``ap_index`` (MATLAB ``ML_index``/``AP_index``, stored values),
    ``name``, ``allen_area``, ``allen_acronym`` and ``allen_index`` to :class:`Roi`.
    ``None`` stands for MATLAB ``[]``.

    Differences from MATLAB: MATLAB overloads ``==`` (``eq``); here that comparison is
    :meth:`equals`, and ``==`` keeps Python's identity semantics.
    """

    def __init__(self, indices, img_hw=(np.nan, np.nan), type=0, ml_index=None,
                 ap_index=None, name="", allen_area="", allen_acronym="", allen_index=None,
                 *, rng=None):
        super().__init__(indices, img_hw, type, rng=rng)
        self.ml_index = ml_index
        self.ap_index = ap_index
        self.name = name
        self.allen_area = allen_area
        self.allen_acronym = allen_acronym
        self.allen_index = allen_index

    def to_struct(self, *, rename_indices_as: str | None = None) -> MatStruct:
        """``MatStruct`` with the widefield ROI fields in MATLAB's order (``toStruct``).

        Fields: ``indices`` (or ``rename_indices_as``), ``shift``, ``type``, ``ML_index``,
        ``AP_index``, ``name``, ``Allen_area``, ``Allen_acronym``, ``Allen_index``; ``None``
        becomes ``[]`` (a ``(0, 0)`` array).
        """
        def mat(v):
            return _EMPTY.copy() if v is None else v

        s = MatStruct()
        s[rename_indices_as or "indices"] = self.indices
        s["shift"] = _shift_vec(self.shift)
        s["type"] = 0.0 if _is_empty(self.type) else self.type
        s["ML_index"] = mat(self.ml_index)
        s["AP_index"] = mat(self.ap_index)
        s["name"] = self.name
        s["Allen_area"] = self.allen_area
        s["Allen_acronym"] = self.allen_acronym
        s["Allen_index"] = mat(self.allen_index)
        return s

    def equals(self, other) -> bool:
        """MATLAB ``Roi_wf.eq``: same number of indices and ``sum(a - b) == 0``.

        Kept as in MATLAB (suspected bug, reported): the test is on the *sum* of the index
        differences, so ``[1, 3]`` and ``[2, 2]`` count as equal.
        """
        a = _lin(self.indices)
        b = _lin(_get(other, "indices"))
        if a.size != b.size:
            return False
        return bool(np.sum(a - b) == 0)

    def fliplr(self) -> RoiWf:
        """Mirror the ROI horizontally in place (``cols = img_w + 1 - cols``); returns self."""
        if np.size(self.indices):
            row = _is_row(self.indices)
            r, c = _ind2sub(self._hw, self.indices)
            self.indices = _vector(_sub2ind(self._hw, r, int(self.img_w) + 1 - c), row)
        return self

    def transform_red2green(self, tform_ch_map) -> RoiWf:
        """Map the ROI from the red (610) to the green (525) channel, in place.

        MATLAB ``Roi_wf.transform_red2green(tform_ch_map)``. ``tform_ch_map`` is the 3x3
        ``A`` matrix (MATLAB ``tform.A``, column-vector convention of
        :mod:`calliopy.widefield`; for a legacy ``affine2d`` pass ``T.T``).

        One pixel: flip horizontally, then ``transformPointsForward(tform_ch_map)``, rounded.
        Several pixels: the mask is flipped and warped with ``invert(tform_ch_map)`` onto the
        same grid (``imwarp`` with ``'SameAsInput'``), and every non-zero output pixel is kept.

        Kept as in MATLAB (suspected bug, reported): the one-pixel branch applies the forward
        transform, the mask branch the inverse.

        Unverified: MATLAB's interpolation of a logical image in ``imwarp``; here the mask is
        interpolated bilinearly as float and every output pixel > 0 is kept.
        """
        return _transform_ch(self, tform_ch_map, red2green=True)

    def transform_green2red(self, tform_ch_map) -> RoiWf:
        """Map the ROI from the green (525) to the red (610) channel, in place.

        MATLAB ``Roi_wf.transform_green2red``: with ``tform = invert(tform_ch_map)``, one pixel
        is transformed with ``tform`` and then flipped; a mask is warped with ``tform``
        and flipped. See :meth:`transform_red2green` for the matrix convention and what is
        unverified.
        """
        return _transform_ch(self, tform_ch_map, red2green=False)

    def to_label_type_image(self) -> np.ndarray:
        """Dense-rank image of ``allen_index``, MATLAB ``toLabelTypeImage`` on one ROI.

        The ROI pixels get ``allen_index`` (0 when ``type`` is empty), then the image values
        are replaced by their 0-based rank among the unique values (float64).

        Differences from MATLAB: an empty ``allen_index`` with a non-empty ``type`` crashed in
        MATLAB (null assignment); it is treated as 0.
        """
        return _rank_image(_wf_label(self)[None, ...])


def _wf_label(roi: RoiWf, single: bool = True) -> np.ndarray:
    if single and _is_empty(roi.type):
        lab = 0
    else:
        lab = 0 if _is_empty(roi.allen_index) else float(np.ravel(roi.allen_index)[0])
    img = np.zeros(roi._hw, dtype=np.float64)
    img.T.flat[_lin(roi.indices) - 1] = min(max(float(mround(lab)), 0), 65535)  # uint16
    return img


def _rank_image(stack: np.ndarray) -> np.ndarray:
    summed = stack.sum(axis=0)  # sum of uint16 in MATLAB returns double
    _, inv = np.unique(summed.ravel(), return_inverse=True)
    return inv.reshape(summed.shape).astype(np.float64)


def _transform_ch(roi: RoiWf, tform_ch_map, *, red2green: bool) -> RoiWf:
    from calliopy.widefield import transform_points, warp_image

    if np.size(roi.indices) == 0:
        return roi
    a = np.asarray(tform_ch_map, dtype=np.float64)
    a_inv = np.linalg.inv(a)
    hw = roi._hw
    if np.size(roi.indices) == 1:
        r, c = _ind2sub(hw, roi.indices)
        if red2green:
            c = hw[1] + 1 - c
            x, y = transform_points(a, np.column_stack([c, r]).astype(np.float64))[0]
        else:
            x, y = transform_points(a_inv, np.column_stack([c, r]).astype(np.float64))[0]
            x = hw[1] + 1 - x
        roi.indices = np.asarray(_sub2ind(hw, mround(y), mround(x)), dtype=np.float64).reshape(1, 1)
        return roi
    mask = roi.to_binary_mask().astype(np.float64)
    if red2green:
        out = warp_image(mask[:, ::-1], a_inv, hw, order=1)
    else:
        out = warp_image(mask, a_inv, hw, order=1)[:, ::-1]
    roi.indices = _find(out != 0)
    return roi


# ---------------------------------------------------------------------------------------
# array versions of the methods
# ---------------------------------------------------------------------------------------
def _seq(rois) -> list:
    if isinstance(rois, Roi):
        return [rois]
    out = list(rois)
    if not out:
        raise ValueError("no ROIs")
    return out


def rois_binary_mask(rois) -> np.ndarray:
    """Union mask of several ROIs, MATLAB ``rois.toBinaryMask()`` on an object array.

    Parameters
    ----------
    rois
        A :class:`Roi` or a non-empty sequence of them (image size from the first).

    Returns
    -------
    numpy.ndarray
        Boolean ``(img_h, img_w)``.
    """
    seq = _seq(rois)
    mask = np.zeros(seq[0]._hw, dtype=bool)
    for r in seq:
        mask.T.flat[_lin(r.indices) - 1] = True
    return mask


def rois_label_image(rois) -> np.ndarray:
    """``rois.toLabelImage()``: ``(H, W)`` for one ROI, ``(H, W, N)`` uint8 for N >= 2."""
    seq = _seq(rois)
    if len(seq) < 2:
        return seq[0].to_label_image()
    return np.stack([r.to_label_image() for r in seq], axis=2)


def rois_label_type_image(rois) -> np.ndarray:
    """``rois.toLabelTypeImage()`` on an object array, as MATLAB dispatches it.

    :class:`Roi`: ``(H, W)`` uint8 for one ROI (``type + 1``, 13 for an empty type); for
    N >= 2 ROIs ``(H, W, N)`` uint8 with ``type + 1`` per plane and **12** for an empty type.
    :class:`RoiWf`: ``(H, W)`` float64 dense ranks of the summed ``allen_index`` images (see
    :meth:`RoiWf.to_label_type_image`).

    Kept as in MATLAB (suspected bug, reported): for :class:`Roi` an empty type gives 13 on a
    single ROI but 12 in an array.
    """
    seq = _seq(rois)
    if isinstance(seq[0], RoiWf):
        if len(seq) < 2:
            return seq[0].to_label_type_image()
        return _rank_image(np.stack([_wf_label(r, single=False) for r in seq]))
    if len(seq) < 2:
        return seq[0].to_label_type_image()
    planes = []
    for r in seq:
        lab = 12 if _is_empty(r.type) else float(np.ravel(r.type)[0]) + 1
        img = np.zeros(r._hw, dtype=np.uint8)
        img.T.flat[_lin(r.indices) - 1] = _u8(lab)
        planes.append(img)
    return np.stack(planes, axis=2)


def move_roi_structs(rois, dx, dy, global_dxdy, img_hw) -> np.ndarray:
    """Translate ROI structs, local function ``moveROIs`` of ``saveRoisChronic.m``.

    Same rule as :meth:`Roi.move` on structs: subscripts move by ``dx + global_dxdy[0]``
    (rows) and ``dy + global_dxdy[1]`` (columns), pixels leaving the image are dropped,
    ``shift`` accumulates ``[dx, dy]``.

    Parameters
    ----------
    rois
        Struct array (``(1, N)`` object array of MatStruct, a MatStruct or a list).
    dx, dy
        Displacement in pixels (rows, columns).
    global_dxdy
        ``[dx, dy]`` global displacement.
    img_hw
        ``(height, width)``.

    Returns
    -------
    numpy.ndarray
        ``(1, N)`` object array of new MatStruct (the input is not modified).
    """
    out = []
    for s in _struct_list(rois):
        t = MatStruct(s)
        t["indices"] = _move_indices(s["indices"], img_hw, dx, dy, global_dxdy)
        t["shift"] = _shift_vec(s.get("shift")) + np.array([[dx, dy]], dtype=np.float64)
        out.append(t)
    return _struct_array(out)


# ---------------------------------------------------------------------------------------
# struct conversion
# ---------------------------------------------------------------------------------------
_ACTIVITY_FIELDS = ("activity", "activity_fissa_corrected", "activity_ch525", "activity_ch610")


def rois2struct(rois):
    """ROI objects -> adata ROI struct array, MATLAB ``utils.rois2struct``.

    Parameters
    ----------
    rois
        Sequence of :class:`Roi` / :class:`RoiWf` (``to_struct`` is called) or of ROI
        structs (copied).

    Returns
    -------
    numpy.ndarray or MatStruct
        ``(1, N)`` object array of MatStruct with MATLAB's field names (stored 1-based
        ``indices``); a MatStruct without fields (MATLAB ``struct()``) for no ROIs. The fields
        ``activity``, ``activity_fissa_corrected``, ``activity_ch525`` and ``activity_ch610``
        are removed.
    """
    items = [] if rois is None else (rois if isinstance(rois, (list, tuple))
                                     else _struct_list(rois) if not isinstance(rois, Roi)
                                     else [rois])
    if len(items) == 0:
        return MatStruct()
    out = []
    for r in items:
        s = r.to_struct() if isinstance(r, Roi) else MatStruct(r)
        for f in _ACTIVITY_FIELDS:
            s.pop(f, None)
        out.append(s)
    return _struct_array(out)


def struct2rois(roi_struct, img_hw, *, rng=None) -> list[Roi]:
    """Adata ROI struct array -> list of :class:`Roi`, MATLAB ``utils.struct2rois``.

    Parameters
    ----------
    roi_struct
        Struct array as returned by :func:`calliopy.io.load_adata` / ``load_mat`` (``(1, N)``
        object array of MatStruct, a single MatStruct, or a list of mappings).
    img_hw
        ``(height, width)`` of the image (MATLAB's argument is named ``imgWH`` but is passed
        on as ``[height, width]``, which is what all callers give).
    rng
        Generator for the random colour indices.

    Returns
    -------
    list of Roi
        ``type`` and ``shift`` default to 0 and ``[0, 0]`` when the field is missing;
        ``activity`` and ``activity_fissa_corrected`` are copied when present. Indices stay
        the stored 1-based values.

    Notes
    -----
    Differences from MATLAB: a struct without fields gives an empty list (MATLAB failed on
    the missing ``indices`` field).
    """
    out = []
    for s in _struct_list(roi_struct):
        r = Roi(s["indices"], img_hw, rng=rng)
        r.type = s.get("type", 0)
        r.shift = np.asarray(s["shift"], dtype=np.float64) if "shift" in s else np.zeros((1, 2))
        if "activity" in s:
            r.activity = s["activity"]
        if "activity_fissa_corrected" in s:
            r.activity_fissa_corrected = s["activity_fissa_corrected"]
        out.append(r)
    return out


def _wf_from_struct(s, field, img_hw, rng):
    def val(name):
        v = s[name]
        return None if (isinstance(v, np.ndarray) and v.dtype != object and v.size == 0) else v

    return RoiWf(s[field], img_hw, s["type"], val("ML_index"), val("AP_index"), s["name"],
                 s["Allen_area"], s["Allen_acronym"], val("Allen_index"), rng=rng)


def struct2wf_rois(roi_struct, img_hw, *, rng=None) -> list[RoiWf]:
    """Widefield ROI struct array -> list of :class:`RoiWf`, MATLAB ``utils.struct2wfRois``.

    The struct must have the fields ``indices``, ``type``, ``ML_index``, ``AP_index``,
    ``name``, ``Allen_area``, ``Allen_acronym`` and ``Allen_index`` (``activity`` is copied
    when present). ``img_hw`` is ``(height, width)``. Empty numeric fields (``[]``) become
    ``None``.
    """
    out = []
    for s in _struct_list(roi_struct):
        r = _wf_from_struct(s, "indices", img_hw, rng)
        if "activity" in s:
            r.activity = s["activity"]
        out.append(r)
    return out


def struct2wf_rois_2ch(roi_struct, img_hw, *, rng=None) -> tuple[list[RoiWf], list[RoiWf]]:
    """Two-channel widefield ROI struct -> ``(green_rois, red_rois)``, MATLAB
    ``utils.struct2wfRois_2ch`` (which returns the cell ``{roiArray_g, roiArray_r}``).

    Green ROIs use ``indices_ch525`` and ``activity_ch525``, red ROIs ``indices_ch610`` and
    ``activity_ch610``; the other fields as in :func:`struct2wf_rois`.
    """
    green, red = [], []
    for s in _struct_list(roi_struct):
        g = _wf_from_struct(s, "indices_ch525", img_hw, rng)
        r = _wf_from_struct(s, "indices_ch610", img_hw, rng)
        if "activity_ch525" in s:
            g.activity = s["activity_ch525"]
        if "activity_ch610" in s:
            r.activity = s["activity_ch610"]
        green.append(g)
        red.append(r)
    return green, red

