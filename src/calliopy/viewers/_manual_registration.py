"""Manual registration of a widefield experiment by two point pairs: calliope
``12_Widefield/register_widefield_exp_manual.m``.

- :func:`manual_registration_inputs` picks the two frames from the adata files as MATLAB does;
- :class:`ManualRegistrationSession` holds the reference points and the transform and has
  one method per key (``1``-``4``, ``t``, ``s``, ``e``, ``q``);
- :func:`save_manual_registration` writes the transform (3x3 matrix) into the adata file;
  :func:`manual_registration_matlab_code` gives MATLAB lines that save it as MATLAB's
  ``simtform2d`` object instead (extra helper);
- :class:`ManualRegistrationView` draws both frames with the points.

The tkinter window is in ``_tk_registration.py``.
"""

from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path
from typing import NamedTuple

import numpy as np
from matplotlib.figure import Figure

from calliopy._matlab import mround, prctile
from calliopy.io import Adata, MatStruct, delete_mat_variables, load_mat, update_mat
from calliopy.widefield import compute_tform_manual, warp_image

__all__ = [
    "MANUAL_REGISTRATION_KEYS",
    "ManualRegistrationInputs",
    "ManualRegistrationSession",
    "ManualRegistrationView",
    "manual_registration_inputs",
    "manual_registration_matlab_code",
    "save_manual_registration",
]

log = logging.getLogger("calliopy.viewers")

MANUAL_REGISTRATION_KEYS = {
    "1": "Mark point 1 in left image",
    "2": "Mark point 2 in left image",
    "3": "Mark point 1 in right image",
    "4": "Mark point 2 in right image",
    "t": "Compute and display t-form",
    "s": "Save to adata",
    "q": "Save to adata and export ref points",
    "e": "Retrieve ref points of previous function call",
}
"""Keys of the manual registration window (MATLAB's help box)."""

_CHANNELS = ("_ch525", "_ch610")


class ManualRegistrationInputs(NamedTuple):
    """Frames and target variables chosen by :func:`manual_registration_inputs`."""

    fixed: np.ndarray
    """Frame A (MATLAB ``frameA``, the origin's frame; left image)."""
    moving: np.ndarray
    """Frame B (MATLAB ``frameB``, the experiment's frame; right image)."""
    rp_string: str
    """Suffix of the variables written: ``''``, ``'_ch_map'``, ``'_ch525'`` or ``'_ch610'``."""
    is_2ch_origin: bool
    """The adata file has ``tform_ch_map``: the channel map of a two-colour origin is fixed."""
    fixed_exp_id: int | None
    moving_exp_id: int | None


def _channel_suffix(channel) -> str:
    c = str(channel)
    c = c if c.startswith("_") else f"_{c}"
    if c not in _CHANNELS:
        raise ValueError("channel must be 'ch525' or 'ch610'")
    return c


def manual_registration_inputs(adata_path: str | PathLike, origin_adata_path: str | PathLike | None = None,
                               *, channel: str | None = None) -> ManualRegistrationInputs:
    """The two frames MATLAB registers, and where the result goes.

    - The adata file has ``tform_ch_map`` (two-colour origin): ``first_frame_ch525`` is
      fixed, ``fliplr(first_frame_ch610)`` moving; the result goes to ``tform_ch_map``.
    - Otherwise the origin's ``first_frame`` is fixed and the experiment's moving (result in
      ``tform``); for two-colour data ``first_frame<channel>`` of both (result in
      ``tform<channel>``).

    Parameters
    ----------
    adata_path
        Adata file of the experiment to register.
    origin_adata_path
        Adata file of the site's origin experiment (MATLAB ``Adata-S1-T<site_id>``);
        not needed for a two-colour origin.
    channel
        ``'ch525'`` or ``'ch610'`` for two-colour data (MATLAB asks for 1 or 2 in the
        Command Window and takes the suffix from the ExpLog column names).

    Raises
    ------
    ValueError
        The origin file is needed but not given, has no first frame, or the channel is
        needed but not given.
    """
    ad = Adata(adata_path)
    if "tform_ch_map" in ad:
        log.info("This is a 2ch origin - assuming you want to fix the channel alignment")
        fixed = np.asarray(ad["first_frame_ch525"])
        moving = np.fliplr(np.asarray(ad["first_frame_ch610"]))
        return ManualRegistrationInputs(fixed, moving, "_ch_map", True, ad.exp_id, ad.exp_id)
    if origin_adata_path is None:
        raise ValueError("origin_adata_path is needed (the adata file of the site's origin experiment)")
    orig = Adata(origin_adata_path)
    if "first_frame" in orig:
        return ManualRegistrationInputs(np.asarray(orig["first_frame"]), np.asarray(ad["first_frame"]), "",
                                        False, orig.exp_id, ad.exp_id)
    if "first_frame_ch525" in orig or "first_frame_ch610" in orig:
        if channel is None:
            raise ValueError("two-colour data: pass channel='ch525' or 'ch610'")
        rp = _channel_suffix(channel)
        log.info("Thus shall be working on %s.", rp)
        return ManualRegistrationInputs(np.asarray(orig[f"first_frame{rp}"]), np.asarray(ad[f"first_frame{rp}"]),
                                        rp, False, orig.exp_id, ad.exp_id)
    raise ValueError("the origin adata file has no first_frame, first_frame_ch525 or first_frame_ch610")


def _normalised(img: np.ndarray, win: int) -> np.ndarray:
    from calliopy.widefield._registration import _local_normalise

    return _local_normalise(img, win)


class ManualRegistrationSession:
    """Reference points and transform of ``register_widefield_exp_manual.m``.

    Parameters
    ----------
    fixed, moving : array (rows, cols)
        Left (origin, MATLAB ``frameA``) and right (experiment, ``frameB``) image.
    frac_boundary : float, default 0.1
        Fraction of the border left out of the similarity score.
    normalize_template : bool, default False
        Divide both images by their local mean (window ``round(min(size(fixed))/20)``)
        before display, point picking and the similarity score (MATLAB default 0).
    rp_string : str, default ''
        Suffix of the variables :meth:`save` writes (see :func:`manual_registration_inputs`).
    adata_path : path, optional
        Adata file :meth:`save` writes into.
    fixed_exp_id, moving_exp_id : int, optional
        For the panel titles.
    previous_refpoints : array (4, 2), optional
        Points of an earlier call for the ``e`` key (MATLAB ``rp`` in the base workspace).

    Attributes
    ----------
    points : dict
        ``{('fixed', 0), ('fixed', 1), ('moving', 0), ('moving', 1)}`` -> ``(x, y)`` or None.
        Point coordinates are MATLAB's: x = column, y = row, **1-based**, pixel centres at
        integers (what ``drawcrosshair`` returns and ``fitgeotform2d`` takes).
    tform : numpy.ndarray or None
        3x3 ``tform.A`` mapping moving to fixed, after :meth:`compute_tform`.
    similarity : float or None
    exported_refpoints : numpy.ndarray or None
        Set by the ``q`` key.
    """

    def __init__(self, fixed, moving, *, frac_boundary: float = 0.1, normalize_template: bool = False,
                 rp_string: str = "", adata_path: str | PathLike | None = None,
                 fixed_exp_id: int | None = None, moving_exp_id: int | None = None,
                 previous_refpoints=None) -> None:
        f = np.asarray(fixed)
        m = np.asarray(moving)
        if f.ndim != 2 or m.ndim != 2:
            raise ValueError("fixed and moving must be 2-D images")
        if normalize_template:
            win = int(mround(min(f.shape) / 20))
            f, m = _normalised(f, win), _normalised(m, win)
        self.fixed = f.astype(np.float64)
        self.moving = m.astype(np.float64)
        self.frac_boundary = float(frac_boundary)
        self.normalize_template = bool(normalize_template)
        self.rp_string = str(rp_string)
        self.adata_path = Path(adata_path) if adata_path is not None else None
        self.fixed_exp_id = fixed_exp_id
        self.moving_exp_id = moving_exp_id
        self.previous_refpoints = None if previous_refpoints is None else np.asarray(previous_refpoints, float)
        self.points: dict[tuple[str, int], tuple[float, float] | None] = {
            ("fixed", 0): None, ("fixed", 1): None, ("moving", 0): None, ("moving", 1): None}
        self.tform: np.ndarray | None = None
        self.similarity: float | None = None
        self.exported_refpoints: np.ndarray | None = None
        self.saved = False

    # keys 1-4 ---------------------------------------------------------------------------
    @staticmethod
    def point_slot(key: str) -> tuple[str, int] | None:
        """Which point a key marks: ``'1'`` -> ('fixed', 0), ..., ``'4'`` -> ('moving', 1)."""
        return {"1": ("fixed", 0), "2": ("fixed", 1), "3": ("moving", 0), "4": ("moving", 1)}.get(key)

    def set_point(self, image: str, k: int, x: float, y: float) -> None:
        """Set reference point ``k`` (0 or 1) of ``image`` (``'fixed'``/``'moving'``), 1-based (x, y)."""
        if (image, k) not in self.points:
            raise KeyError("image must be 'fixed' or 'moving' and k 0 or 1")
        self.points[(image, k)] = (float(x), float(y))

    @property
    def refpoints(self) -> np.ndarray:
        """``[4 x 2]``: fixed 1, fixed 2, moving 1, moving 2 (MATLAB ``rp``); NaN when unset."""
        order = (("fixed", 0), ("fixed", 1), ("moving", 0), ("moving", 1))
        return np.array([self.points[o] if self.points[o] is not None else (np.nan, np.nan) for o in order])

    def set_refpoints(self, rp) -> None:
        """Set all four points from a ``[4 x 2]`` array (MATLAB ``e`` key)."""
        a = np.asarray(rp, dtype=np.float64)
        if a.shape != (4, 2):
            raise ValueError("refpoints must be a 4 x 2 array")
        for row, slot in zip(a, (("fixed", 0), ("fixed", 1), ("moving", 0), ("moving", 1)), strict=True):
            self.points[slot] = (float(row[0]), float(row[1]))

    def retrieve_previous(self) -> bool:
        """``e`` key: take the points of the previous call; False when there are none."""
        rp = self.previous_refpoints if self.previous_refpoints is not None else self.exported_refpoints
        if rp is None:
            log.warning("no reference points of a previous call")
            return False
        log.info("Retrieving previous reference points...")
        self.set_refpoints(rp)
        return True

    # key t ------------------------------------------------------------------------------
    def compute_tform(self) -> tuple[np.ndarray, float]:
        """``t`` key: similarity transform from the two point pairs and its similarity score.

        :func:`calliopy.widefield.compute_tform_manual` (``fitgeotform2d(..., 'similarity')``
        and ``corr2`` on the region inside ``frac_boundary``).
        """
        missing = [k for k, v in self.points.items() if v is None]
        if missing:
            raise ValueError("mark all four reference points first (keys 1-4)")
        mv = np.array([self.points[("moving", 0)], self.points[("moving", 1)]])
        fx = np.array([self.points[("fixed", 0)], self.points[("fixed", 1)]])
        res = compute_tform_manual(self.moving, self.fixed, mv, fx, frac_boundary=self.frac_boundary)
        self.tform = res.tform
        self.similarity = float(res.similarity)
        log.info("Similarity=%.2f", self.similarity)
        return self.tform, self.similarity

    def overlay_stack(self) -> np.ndarray:
        """``[rows x cols x 2]``: fixed and registered moving image, each divided by its median.

        What MATLAB shows with ``view_stack(cat(3, ...), 'small_figure_window', 1)``.
        """
        if self.tform is None:
            raise ValueError("compute the transform first (key t)")
        reg = warp_image(self.moving, self.tform, self.fixed.shape)
        reg = reg / float(prctile(reg, 50))
        fx = self.fixed / float(prctile(self.fixed, 50))
        return np.stack([fx, reg], axis=2)

    # keys s and q -----------------------------------------------------------------------
    def save(self, *, export_refpoints: bool = False) -> list[str]:
        """``s`` key (``q`` with ``export_refpoints``): write the result into the adata file.

        See :func:`save_manual_registration` (``tform`` as the 3x3 matrix ``tform.A``).
        Returns the names of the variables written or removed.
        """
        if export_refpoints:
            self.exported_refpoints = self.refpoints
            log.info("Exporting reference points and saving tform to adata...")
        if self.adata_path is None:
            raise ValueError("no adata file to save to")
        if self.tform is None or self.similarity is None:
            raise ValueError("compute the transform first (key t)")
        written = save_manual_registration(self.adata_path, self.tform, similarity=self.similarity,
                                           frac_boundary=self.frac_boundary, rp_string=self.rp_string)
        self.saved = True
        return written

    def matlab_commands(self) -> str:
        """MATLAB code that saves this result as the MATLAB tool does (``simtform2d`` object).

        See :func:`manual_registration_matlab_code`.
        """
        if self.tform is None or self.similarity is None:
            raise ValueError("compute the transform first (key t)")
        if self.adata_path is None:
            raise ValueError("no adata file")
        return manual_registration_matlab_code(self.adata_path, self.tform, similarity=self.similarity,
                                               frac_boundary=self.frac_boundary, rp_string=self.rp_string)

    def titles(self) -> tuple[str, str]:
        def t(i, default):
            return f"ExpID {int(i)}" if i is not None else default

        return t(self.fixed_exp_id, "origin"), t(self.moving_exp_id, "current")


def _matlab_num(x: float) -> str:
    return f"{float(x):.17g}"


def manual_registration_matlab_code(adata_path: str | PathLike, tform, *, similarity: float,
                                    frac_boundary: float, rp_string: str = "") -> str:
    """MATLAB code that writes a manual registration as ``register_widefield_exp_manual.m`` does.

    The lines set ``tform<rp_string>`` to a ``simtform2d`` built from the 3x3 matrix
    ``tform.A`` (all digits), add ``similarity``, ``fraction_boundary``, ``status =
    'SUCCESS'`` and ``registration_type = 'manual'`` to ``registration_parameters<rp_string>``,
    call ``save_changes`` and remove ``ROIs`` when the file has them, i.e. the MATLAB tool's
    ``compute_tform`` + ``save_to_adata``. Run them in MATLAB with calliope on the path.

    Parameters
    ----------
    adata_path
        Adata file (written into the code as given).
    tform
        3x3 similarity ``tform.A`` (moving to fixed).
    similarity, frac_boundary, rp_string
        As :func:`save_manual_registration`.

    Returns
    -------
    str
        The MATLAB lines.

    Differences from MATLAB
    -----------------------
    The ``ROIs`` removal saves with ``'-v7.3'`` (the MATLAB tool's ``save(..., '-struct',
    'adata')`` uses MATLAB's default format, which changes the adata file's format).
    """
    a = np.asarray(tform, dtype=np.float64)
    if a.shape != (3, 3):
        raise ValueError("tform must be a 3x3 matrix")
    rows = "; ".join(" ".join(_matlab_num(v) for v in row) for row in a)
    p = f"registration_parameters{rp_string}"
    f = str(adata_path).replace("'", "''")
    lines = [
        f"adata_fname = '{f}';",
        "adata = Adata(adata_fname);",
        f"adata.tform{rp_string} = simtform2d([{rows}]);",
        f"adata.{p}.similarity = {_matlab_num(similarity)};",
        f"adata.{p}.fraction_boundary = {_matlab_num(frac_boundary)};",
        f"adata.{p}.status = 'SUCCESS';",
        f"adata.{p}.registration_type = 'manual';",
        "adata.save_changes;",
        "if isprop(adata, 'ROIs')",
        "    adata_s = load(adata_fname);",
        "    adata_s = rmfield(adata_s, 'ROIs');",
        "    save(adata_fname, '-struct', 'adata_s', '-v7.3');",
        "end",
    ]
    return "\n".join(lines)


def save_manual_registration(adata_path: str | PathLike, tform, *, similarity: float, frac_boundary: float,
                             rp_string: str = "") -> list[str]:
    """Write a manual registration into an adata file (``save_to_adata`` of the MATLAB tool).

    Writes ``tform<rp_string>`` as the numeric 3x3 matrix ``tform.A`` (as the other calliopy
    registration savers) and ``registration_parameters<rp_string>`` (the stored struct, or a
    new one, with ``similarity``, ``fraction_boundary``, ``status = 'SUCCESS'`` and
    ``registration_type = 'manual'``) with :func:`calliopy.io.update_mat`, then removes the
    variable ``ROIs`` when the file has one with :func:`calliopy.io.delete_mat_variables`
    (MATLAB: "Detected ROIs in this Adata file. Deleting them...").

    Parameters
    ----------
    adata_path
        Adata file (MATLAB v7.3).
    tform
        3x3 ``tform.A`` mapping the experiment's frame onto the origin's.
    similarity, frac_boundary
        Values set by the ``t`` key (MATLAB stores them in the same struct).
    rp_string
        ``''``, ``'_ch_map'``, ``'_ch525'`` or ``'_ch610'``.

    Returns
    -------
    list of str
        Variables written or removed.

    Differences from MATLAB
    -----------------------
    - ``tform`` is stored as the numeric 3x3 matrix ``tform.A``; MATLAB stores a
      ``simtform2d`` object (calliopy cannot write MATLAB objects). MATLAB code that reads the
      file gets a matrix and wraps it (``simtform2d(A)`` or ``affinetform2d(A)``) before
      ``imwarp``. :func:`manual_registration_matlab_code` gives MATLAB lines that save the
      object form instead, if wanted.
    - ``registration_parameters`` keeps the stored fields, their order and classes.
    - ``ROIs`` is removed in place, keeping the v7.3 format (MATLAB loads the whole file and
      saves it again with ``save(..., '-struct')`` in its default format).
    - The ExpLog/RawDataStatus side of the tool is not part of this function.
    """
    path = Path(adata_path)
    a = np.asarray(tform, dtype=np.float64)
    if a.shape != (3, 3):
        raise ValueError("tform must be a 3x3 matrix")
    t_name = f"tform{rp_string}"
    p_name = f"registration_parameters{rp_string}"
    ad = Adata(path)
    params = MatStruct()
    if p_name in ad:
        old = load_mat(path, p_name, squeeze_scalars=False)[p_name]
        if isinstance(old, dict):
            params = MatStruct(old)
    params.similarity = float(similarity)
    params.fraction_boundary = float(frac_boundary)
    params.status = "SUCCESS"
    params.registration_type = "manual"
    update_mat(path, {t_name: a, p_name: params})
    written = [t_name, p_name]
    if "ROIs" in ad:
        log.info("Detected ROIs in this Adata file. Deleting them...")
        written += delete_mat_variables(path, ["ROIs"])
    log.info("Saved %s to Adata", ad.exp_id if ad.exp_id is not None else path.name)
    return written


class ManualRegistrationView:
    """Both frames of a :class:`ManualRegistrationSession` with the reference points.

    Axes use MATLAB's pixel coordinates (1-based, x = column, y = row), so a click's data
    coordinates are the point MATLAB's ``drawcrosshair`` would return. Colour range: 1st to
    99th percentile (``viridis``; MATLAB's default ``parula`` is not in matplotlib); point 1
    red, point 2 blue (``+``); square axes without ticks.
    """

    def __init__(self, figure: Figure, session: ManualRegistrationSession) -> None:
        self.figure = figure
        self.session = session
        figure.set_facecolor("w")
        self.ax_fixed = figure.add_subplot(1, 2, 1)
        self.ax_moving = figure.add_subplot(1, 2, 2)
        self.refresh()

    def axes_of(self, image: str):
        return self.ax_fixed if image == "fixed" else self.ax_moving

    def refresh(self) -> None:
        s = self.session
        for name, img, title in (("fixed", s.fixed, s.titles()[0]), ("moving", s.moving, s.titles()[1])):
            ax = self.axes_of(name)
            ax.clear()
            h, w = img.shape
            ax.imshow(img, cmap="viridis", vmin=float(prctile(img, 1)), vmax=float(prctile(img, 99)),
                      extent=(0.5, w + 0.5, h + 0.5, 0.5), interpolation="nearest", aspect="auto")
            for k, style in ((0, "r+"), (1, "b+")):
                p = s.points[(name, k)]
                if p is not None:
                    ax.plot(p[0], p[1], style, markersize=10)
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.set_box_aspect(1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title)


def overlay_figure(stack: np.ndarray, frame: int = 0, fig: Figure | None = None) -> Figure:
    """One frame of :meth:`ManualRegistrationSession.overlay_stack` (grey).

    Colour limits as ``view_stack`` sets them: 10th and 99.99th percentile of its middle
    frame (``round(2/2)`` = the first frame of a two-frame stack).
    """
    fig = fig if fig is not None else Figure(figsize=(4, 4))
    fig.clear()
    ax = fig.add_axes((0, 0, 1, 0.94))
    img = stack[:, :, frame]
    ref = stack[:, :, 0]
    ax.imshow(img, cmap="gray", vmin=float(prctile(ref, 10)), vmax=float(prctile(ref, 99.99)),
              interpolation="nearest")
    ax.set_axis_off()
    fig.suptitle(["origin", "registered"][frame] + f" ({frame + 1}/2, left/right to flip)", fontsize=9)
    return fig
