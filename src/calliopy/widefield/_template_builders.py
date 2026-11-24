"""The eight ROI-template generators of calliope ``12_Widefield/ROItemplate_library/Build_code``.

Each function returns the ``struct_ROIs`` that the MATLAB ``ROItemplate_generate_*`` function
saves, as a :class:`~calliopy.io.MatStruct` with the fields ``name``, ``description``,
``comments``, ``code_to_generate``, ``parameters``, ``ROIs``, ``img_ROIs`` (in that order).
Nothing is saved and no dialog is shown: the caller saves.

Common differences from MATLAB (also listed per function):

- ``dir_ROItemplate`` (a lab folder hard-coded in MATLAB) is not a field of the result.
- No figure: draw it with :func:`calliopy.widefield._templates.plot_roi_template`.
- The Allen annotation and ontology are arguments (``allen_map``: the annotation volume
  ``midAnnotation`` or its flat map, :func:`~calliopy.widefield._templates.load_allen_flat_map`;
  ``ontology``: :func:`~calliopy.widefield._templates.load_allen_ontology`).
- ``img_ROIs`` of the Allen-assigning generators is coloured with a random permutation of
  ``hsv`` colours that cannot be MATLAB's (``rng(7)``); pass ``rng`` to fix it.

ROI ``indices`` are MATLAB linear indices, 1-based and column-major, into ``SIZE_TEMPLATE``.
"""

from __future__ import annotations

import numpy as np

from calliopy.io import MatStruct
from calliopy.widefield._templates import (
    ALLEN_REF_POINTS,
    _empty,
    _mat_col,
    _mat_num,
    _row,
    _struct_array,
    _sub2ind,
    assign_allen_areas,
    binary_roi_image,
    cortex_grid,
    cranial_window_grid,
    eeg_electrode_discs,
    eeg_ring_rois,
    reference_rois,
    roi_template_colour_image,
)

__all__ = [
    "generate_template_14rois_cortex_lambda_bregma",
    "generate_template_14rois_lambda_bregma",
    "generate_template_eeg_circles_lambda_bregma",
    "generate_template_grid_cranial_window",
    "generate_template_grid_plus_eeg_circles_lambda_bregma",
    "generate_template_grid_whole_cortex_lambda_bregma",
    "generate_template_grid_whole_cortex_lambda_icv",
    "generate_template_only_eeg_electrode_positions_lambda_bregma",
]

_MIRROR_COMMENT = (
    "ROIs that are on the left from the medial line have positive ML_index to indicate that "
    "they belong to the right hemisphere - because image after recording is mirrored - "
    '"left hemisphere" on the recording is actually right'
)
_CORTEX_DESCRIPTION = (
    "Grid which covers the whole cortex with acrylic glass window (which is larger than "
    "crystal skull coverslip). Alignment is done via lambda and bregma"
)

# Grid parameters that the EEG generators store although they do not use them.
_GRID_DEFAULTS = {"w_hem": 0.95, "l_back": 0.25, "l_front": 0.75, "n_w": 30, "tan_ant": 1.35,
                  "tan_post": 0.5, "l_corner": 0.18, "d_border": 3}


def _grid_params(**kw) -> dict:
    """The grid parameters of a generator call, keyed as in :func:`cortex_grid`."""
    return {k: kw[k] for k in _GRID_DEFAULTS}


def _struct_rois(name, description, comments, code, parameters, rois, img) -> MatStruct:
    return MatStruct(
        name=name,
        description=description,
        comments=comments,
        code_to_generate=code,
        parameters=parameters,
        ROIs=rois if isinstance(rois, np.ndarray) else _struct_array(rois),
        img_ROIs=img,
    )


def _cortex_parameters(size_template, vec_lambda, second_name, second, p) -> MatStruct:
    return MatStruct(
        SIZE_TEMPLATE=_row(size_template),
        VEC_LAMBDA=_row(vec_lambda),
        **{second_name: _row(second)},
        W_HEM=_mat_num(p["w_hem"]),
        L_BACK=_mat_num(p["l_back"]),
        L_FRONT=_mat_num(p["l_front"]),
        N_W=_mat_num(p["n_w"]),
        TAN_ANT=_mat_num(p["tan_ant"]),
        TAN_POST=_mat_num(p["tan_post"]),
        D_BORDER=_mat_num(p["d_border"]),
        L_CORNER=_mat_num(p["l_corner"]),
    )


def _allen_finish(rois, allen_map, ontology, ref2_allen, targets, size_template, rng,
                  name_hemisphere):
    rois = assign_allen_areas(
        rois, allen_map, ontology, (ALLEN_REF_POINTS["lambda"], ref2_allen), targets,
        size_template, name_hemisphere=name_hemisphere,
    )
    img = roi_template_colour_image(rois, allen_map, size_template, rng=rng)
    return rois, img


def generate_template_grid_whole_cortex_lambda_bregma(
    allen_map,
    ontology,
    *,
    size_template=(1000, 1000),
    vec_lambda=(200, 500),
    vec_bregma=(632, 500),
    w_hem=0.95,
    l_back=0.25,
    l_front=0.75,
    n_w=30,
    tan_ant=1.35,
    tan_post=0.5,
    l_corner=0.18,
    d_border=3,
    rng=None,
) -> MatStruct:
    """Square grid over the whole dorsal cortex, aligned by lambda and bregma.

    Port of ``12_Widefield/ROItemplate_library/Build_code/
    ROItemplate_generate_grid_whole_cortex_lambda_bregma.m``.

    Parameters
    ----------
    allen_map
        Allen annotation volume (``midAnnotation``) or its flat map.
    ontology
        Parsed ``structures.json``.
    size_template, vec_lambda, vec_bregma, w_hem, l_back, l_front, n_w, tan_ant, tan_post, \
l_corner, d_border
        MATLAB's fixed parameters (``SIZE_TEMPLATE`` rows x cols; ``VEC_*`` in MATLAB axes
        coordinates ``(x, y)``, 1-based); the defaults are the MATLAB values.
    rng
        :class:`numpy.random.Generator` for the colours of ``img_ROIs``.

    Returns
    -------
    MatStruct
        ``struct_ROIs``. ``ROIs`` ``(1, N)``: Lambda, Bregma (1-based scalar index each), then
        the grid squares (``indices`` ``(n, 1)``, ``ML_index``, ``AP_index``, ``name``
        ``'right hemisphere'``/``'left hemisphere'``, ``Allen_area``/``_acronym``/``_index``).
        ``img_ROIs``: float64 ``(rows, cols, 3)``.

    Notes
    -----
    Differences from MATLAB: no ``dir_ROItemplate`` field, no figure, nothing saved;
    ``img_ROIs`` colours differ (random permutation). Allen areas rest on
    :func:`~calliopy.widefield._transform.warp_image` reproducing ``imwarp``.
    """
    p = _grid_params(w_hem=w_hem, l_back=l_back, l_front=l_front, n_w=n_w, tan_ant=tan_ant,
                     tan_post=tan_post, l_corner=l_corner, d_border=d_border)
    grid = cortex_grid(size_template, vec_lambda, vec_bregma, **p)
    rois = reference_rois(size_template, [vec_lambda, vec_bregma], ["Lambda", "Bregma"]) + grid
    rois, img = _allen_finish(rois, allen_map, ontology, ALLEN_REF_POINTS["bregma"],
                              (vec_lambda, vec_bregma), size_template, rng, True)
    return _struct_rois(
        "Grid whole cortex based on lambda-bregma", _CORTEX_DESCRIPTION, _MIRROR_COMMENT,
        "ROItemplate_generate_grid_whole_cortex_lambda_bregma",
        _cortex_parameters(size_template, vec_lambda, "VEC_BREGMA", vec_bregma, p), rois, img,
    )


def generate_template_grid_whole_cortex_lambda_icv(
    allen_map,
    ontology,
    *,
    size_template=(1000, 1000),
    vec_lambda=(200, 500),
    vec_icv=(956, 500),
    w_hem=0.95,
    l_back=0.25,
    l_front=0.75,
    n_w=30,
    tan_ant=1.35,
    tan_post=0.5,
    l_corner=0.18,
    d_border=3,
    rng=None,
) -> MatStruct:
    """Square grid over the whole dorsal cortex, aligned by lambda and the inferior
    cerebral vein (ICV).

    Port of ``12_Widefield/ROItemplate_library/Build_code/
    ROItemplate_generate_grid_whole_cortex_lambda_ICV.m``. A "fake bregma"
    ``VEC_LAMBDA*L_FRONT/(1+L_FRONT) + VEC_ICV*1/(1+L_FRONT)`` sets the grid as in
    :func:`generate_template_grid_whole_cortex_lambda_bregma`; the second reference ROI is
    the ICV (name ``'Inferior Cerebral Vein'``) and the Allen map is aligned by lambda and
    the ICV.

    Parameters and returns as :func:`generate_template_grid_whole_cortex_lambda_bregma`,
    with ``vec_icv`` (``VEC_ICV``) instead of ``vec_bregma``.

    Notes
    -----
    Differences from MATLAB: as :func:`generate_template_grid_whole_cortex_lambda_bregma`.
    """
    p = _grid_params(w_hem=w_hem, l_back=l_back, l_front=l_front, n_w=n_w, tan_ant=tan_ant,
                     tan_post=tan_post, l_corner=l_corner, d_border=d_border)
    lam = np.asarray(vec_lambda, dtype=np.float64)
    icv = np.asarray(vec_icv, dtype=np.float64)
    fake_bregma = lam * l_front / (1 + l_front) + icv * 1 / (1 + l_front)
    grid = cortex_grid(size_template, vec_lambda, fake_bregma, **p)
    rois = reference_rois(
        size_template, [vec_lambda, vec_icv], ["Lambda", "Inferior Cerebral Vein"]
    ) + grid
    rois, img = _allen_finish(rois, allen_map, ontology, ALLEN_REF_POINTS["icv"],
                              (vec_lambda, vec_icv), size_template, rng, True)
    return _struct_rois(
        "Grid whole cortex based on lambda-inferior cerebral vein",
        "Grid which covers the whole cortex with acrylic glass window (which is larger than "
        "crystal skull coverslip). Alignment is done via lambda and inferior cerebral vein "
        "instead of bregma (suitable for thin-skull+acrylic glass)",
        _MIRROR_COMMENT.replace("ML_index", "ML_ind"),
        "ROItemplate_generate_grid_whole_cortex_lambda_ICV",
        _cortex_parameters(size_template, vec_lambda, "VEC_ICV", vec_icv, p), rois, img,
    )


def generate_template_eeg_circles_lambda_bregma(
    allen_map,
    ontology,
    *,
    size_template=(1000, 1000),
    vec_lambda=(200, 500),
    vec_bregma=(632, 500),
    r_inner=30,
    r_outer=40,
    rng=None,
) -> MatStruct:
    """Rings around the 30 electrodes of the NeuroNexus mouse EEG probe, each split into
    four 90-degree sectors.

    Port of ``12_Widefield/ROItemplate_library/Build_code/
    ROItemplate_generate_eeg_circles_lambda_bregma.m`` (with
    ``12_Widefield/angles_from_top_left.m``).

    Parameters
    ----------
    allen_map, ontology, rng
        As :func:`generate_template_grid_whole_cortex_lambda_bregma`.
    size_template, vec_lambda, vec_bregma
        MATLAB's ``SIZE_TEMPLATE``, ``VEC_LAMBDA``, ``VEC_BREGMA``.
    r_inner, r_outer
        ``ElR``, ``ElRo``: ring pixels satisfy ``ElR < distance < ElRo``.

    Returns
    -------
    MatStruct
        ``struct_ROIs`` with 2 + 120 ROIs (electrode outer, sector inner;
        ``ML_index`` = channel, ``AP_index`` = sector 1..4, name ``'right hemisphere'`` for
        channels 1-15). ``parameters`` holds the grid parameters MATLAB stores although the
        rings do not use them (``W_HEM`` ... ``L_CORNER``).

    Notes
    -----
    Differences from MATLAB: as :func:`generate_template_grid_whole_cortex_lambda_bregma`.
    ``ElX``, ``ElY`` and the sector angles are not stored in ``parameters`` (as in MATLAB).
    """
    rings = eeg_ring_rois(size_template, vec_bregma, r_inner=r_inner, r_outer=r_outer)
    rois = reference_rois(size_template, [vec_lambda, vec_bregma], ["Lambda", "Bregma"]) + rings
    rois, img = _allen_finish(rois, allen_map, ontology, ALLEN_REF_POINTS["bregma"],
                              (vec_lambda, vec_bregma), size_template, rng, False)
    return _struct_rois(
        "Ring Rois around electrode positions based on lambda-bregma",
        "Rings, divided into four parts, around each of the 30 electrodes of the neuronexus "
        "mouse EEG probe. Positions are fixed to bregma",
        _MIRROR_COMMENT,
        "ROItemplate_generate_eeg_circles_lambda_bregma",
        _cortex_parameters(size_template, vec_lambda, "VEC_BREGMA", vec_bregma, _GRID_DEFAULTS),
        rois, img,
    )


def generate_template_only_eeg_electrode_positions_lambda_bregma(
    allen_map,
    ontology,
    *,
    size_template=(1000, 1000),
    vec_lambda=(200, 500),
    vec_bregma=(632, 500),
    radius=40,
    rng=None,
) -> MatStruct:
    """Discs at the positions of the 30 EEG electrodes.

    Port of ``12_Widefield/ROItemplate_library/Build_code/
    ROItemplate_generate_only_eeg_electrode_positions_lambda_bregma.m``.

    Parameters
    ----------
    allen_map, ontology, rng, size_template, vec_lambda, vec_bregma
        As :func:`generate_template_eeg_circles_lambda_bregma`.
    radius
        ``ElR``: disc pixels satisfy ``distance <= ElR``.

    Returns
    -------
    MatStruct
        ``struct_ROIs`` with 2 + 30 ROIs (``ML_index`` = channel, ``AP_index`` ``[]``).

    Notes
    -----
    Suspected MATLAB bug, kept: the loop sets ``potROIs(ind_ROI).type = 0`` with
    ``ind_ROI`` fixed at 1, so only the first electrode ROI has ``type = 0``; the others
    have ``type = []``. Other differences: as
    :func:`generate_template_grid_whole_cortex_lambda_bregma`.
    """
    discs, _ = eeg_electrode_discs(size_template, vec_bregma, r_inner=None, r_outer=radius)
    elec = []
    for chan, idx in enumerate(discs, start=1):
        hem = "right" if chan <= 15 else "left"
        roi = MatStruct(
            indices=_mat_col(idx) if idx.size else np.empty((0, 1)),
            type=np.float64(0.0) if chan == 1 else _empty(),
            shift=_row([0, 0]),
            ML_index=np.float64(chan),
            AP_index=_empty(),
            name=f"{hem} hemisphere",
            Allen_area="",
            Allen_acronym="",
            Allen_index=_empty(),
        )
        elec.append(roi)
    rois = reference_rois(size_template, [vec_lambda, vec_bregma], ["Lambda", "Bregma"]) + elec
    rois, img = _allen_finish(rois, allen_map, ontology, ALLEN_REF_POINTS["bregma"],
                              (vec_lambda, vec_bregma), size_template, rng, False)
    return _struct_rois(
        "EEG electrode positions based on lambda-bregma",
        "Circles, at the positions of the 30 electrodes of the neuronexus mouse EEG probe. "
        "Positions are fixed to bregma",
        _MIRROR_COMMENT,
        "ROItemplate_generate_only_eeg_electrode_positions_lambda_bregma",
        _cortex_parameters(size_template, vec_lambda, "VEC_BREGMA", vec_bregma, _GRID_DEFAULTS),
        rois, img,
    )


def generate_template_grid_plus_eeg_circles_lambda_bregma(
    allen_map,
    ontology,
    *,
    size_template=(1000, 1000),
    vec_lambda=(200, 500),
    vec_bregma=(632, 500),
    w_hem=0.95,
    l_back=0.25,
    l_front=0.75,
    n_w=30,
    tan_ant=1.35,
    tan_post=0.5,
    l_corner=0.18,
    d_border=3,
    r_inner=30,
    r_outer=40,
    rng=None,
) -> MatStruct:
    """EEG ring sectors followed by the whole-cortex grid, in one template.

    Port of ``12_Widefield/ROItemplate_library/Build_code/
    ROItemplate_generate_grid_plus_eeg_circles_lambda_bregma.m``.

    Returns
    -------
    MatStruct
        ``struct_ROIs`` with ROIs: Lambda, Bregma, the 120 ring sectors of
        :func:`generate_template_eeg_circles_lambda_bregma`, then the grid of
        :func:`generate_template_grid_whole_cortex_lambda_bregma`. Grid ROIs keep an empty
        ``name`` (this generator does not name hemispheres).

    Notes
    -----
    MATLAB's ``description`` (``'Combines ROItemplate_generate_grid_whole_cortex_lambda_bregma
    and '``) and ``code_to_generate`` (``'ROItemplate_generate_eeg_circles_lambda_bregma'``)
    are copied as they are, although unfinished / naming another generator. Other
    differences: as :func:`generate_template_grid_whole_cortex_lambda_bregma`.
    """
    p = _grid_params(w_hem=w_hem, l_back=l_back, l_front=l_front, n_w=n_w, tan_ant=tan_ant,
                     tan_post=tan_post, l_corner=l_corner, d_border=d_border)
    rings = eeg_ring_rois(size_template, vec_bregma, r_inner=r_inner, r_outer=r_outer)
    grid = cortex_grid(size_template, vec_lambda, vec_bregma, **p)
    rois = reference_rois(size_template, [vec_lambda, vec_bregma], ["Lambda", "Bregma"])
    rois = rois + rings + grid
    rois, img = _allen_finish(rois, allen_map, ontology, ALLEN_REF_POINTS["bregma"],
                              (vec_lambda, vec_bregma), size_template, rng, False)
    return _struct_rois(
        "Grid whole cortex plus Ring Rois around electrode positions based on lambda-bregma",
        "Combines ROItemplate_generate_grid_whole_cortex_lambda_bregma and ",
        _MIRROR_COMMENT,
        "ROItemplate_generate_eeg_circles_lambda_bregma",
        _cortex_parameters(size_template, vec_lambda, "VEC_BREGMA", vec_bregma, p), rois, img,
    )


def generate_template_grid_cranial_window(
    *,
    size_template=(500, 500),
    vec_posterior=(50, 250),
    vec_anterior=(450, 250),
    d_border=2,
    n_w=20,
) -> MatStruct:
    """Square grid inside a circular cranial window, aligned by a posterior and an anterior
    reference point.

    Port of ``12_Widefield/ROItemplate_library/Build_code/
    ROItemplate_generate_grid_cranial_window.m``. No Allen assignment.

    Parameters
    ----------
    size_template, vec_posterior, vec_anterior, d_border, n_w
        MATLAB's ``SIZE_TEMPLATE``, ``VEC_POSTERIOR``, ``VEC_ANTERIOR`` (axes coordinates
        ``(x, y)``), ``D_BORDER``, ``N_W``.

    Returns
    -------
    MatStruct
        ``struct_ROIs``; ROIs: 'Posterior Reference Point', 'Anterior Reference Point',
        then the grid squares. ``img_ROIs``: float64 0/1 image of all ROIs (reference pixels
        included).

    Notes
    -----
    Differences from MATLAB: no ``dir_ROItemplate`` field, no figure, nothing saved.
    """
    grid = cranial_window_grid(
        size_template, vec_posterior, vec_anterior, n_w=n_w, d_border=d_border
    )
    rois = reference_rois(
        size_template, [vec_posterior, vec_anterior],
        ["Posterior Reference Point", "Anterior Reference Point"],
    ) + grid
    params = MatStruct(
        SIZE_TEMPLATE=_row(size_template),
        VEC_POSTERIOR=_row(vec_posterior),
        VEC_ANTERIOR=_row(vec_anterior),
        N_W=_mat_num(n_w),
        D_BORDER=_mat_num(d_border),
    )
    return _struct_rois(
        "Grid for circular cranial window",
        "Grid which covers the circular window. Alignment is done via posterior and anterior "
        "reference points",
        "",
        "ROItemplate_generate_grid_cranial_window",
        params, rois, binary_roi_image(rois, size_template),
    )


_14_MASTER = (
    (238, 916), (922, 916), (368, 834), (792, 834), (540, 874), (620, 874), (366, 614),
    (794, 614), (374, 450), (786, 450), (528, 442), (632, 442), (538, 266), (622, 266),
)  # fmt: skip
_14_NAMES = ("leftV1", "rightV1", "leftV2am", "rightV2am", "leftRSC", "rightRSC", "leftS1",
             "rightS1", "leftM1", "rightM1", "leftACC", "rightACC", "leftM2", "rightM2")
_14_ACRONYM = {"V1": "VISp", "V2am": "VISam", "RSC": "RSP", "S1": "SSp", "M1": "MOp",
               "ACC": "ACA", "OFC": "MOs", "M2": "MOs"}
_14_AREA = {"V1": "primary visual area", "V2am": "anteromedial visual area",
            "RSC": "retrosplenial area", "S1": "primary somatosensory area",
            "M1": "primary motor area", "ACC": "anterior cingulate area",
            "OFC": "secondary motor area", "M2": "secondary motor area"}


def generate_template_14rois_lambda_bregma(
    *,
    diameter=20,
    shape: str = "square",
    size_template=(1107, 1200),
    vec_lambda=(488, 590),
    vec_bregma=(918, 590),
    rois_master=_14_MASTER,
) -> MatStruct:
    """Fourteen square (or rounded) ROIs at fixed positions over the dorsal cortex.

    Port of ``12_Widefield/ROItemplate_library/Build_code/
    ROItemplate_generate_14ROIs_lambda_bregma.m``.

    Parameters
    ----------
    diameter
        Side of the ROI in pixels (MATLAB ``'diameter'``, default 20).
    shape
        ``'square'``: ``(diameter+1) x (diameter+1)`` pixels from the top-left corner;
        ``'circle'``: ``diameter x diameter`` with 3/2/1 pixels cut at the first and last
        three rows (MATLAB ``'shape'``).
    size_template, vec_lambda, vec_bregma
        ``SIZE_TEMPLATE`` (rows x cols) and the reference points (axes ``(x, y)``).
    rois_master
        Top-left corners ``(x, y)`` = (column, row), 1-based, one per ROI, in the order of
        the names leftV1, rightV1, ..., rightM2.

    Returns
    -------
    MatStruct
        ``struct_ROIs``; ROIs: Lambda, Bregma, then the 14 ROIs (``name`` e.g. ``'leftV1'``,
        ``Allen_area`` ``'left primary visual area'``, ``Allen_acronym`` ``'left VISp'``).
        ``img_ROIs`` = 0.5 at ROI pixels, 0 elsewhere (float64).

    Notes
    -----
    The MATLAB function cannot run as written; the crashes are fixed here:

    - ``parameters`` refers to ``W_HEM``, ``L_BACK``, ``L_FRONT``, ``N_W``, ``TAN_ANT``,
      ``TAN_POST``, ``D_BORDER``, which are undefined (commented out): ``parameters`` holds
      ``SIZE_TEMPLATE``, ``VEC_LAMBDA``, ``VEC_BREGMA`` and ``ROI_DIAMETER`` (= ``diameter``;
      MATLAB never stored it).
    - ``[ROIs_bl potROIs]`` concatenates structs with different fields (the reference ROIs
      lack ``Allen_acronym``): the reference ROIs get ``Allen_acronym = ''``; all ROIs use the
      field order of the reference ROIs (indices, type, shift, ML_index, AP_index, name,
      Allen_area, Allen_acronym, Allen_index).
    - The ``'shape'`` validator (``@isnumeric``) rejects every string, so MATLAB can only
      use the default ``'square'``; here ``'circle'`` is accepted.

    The unused MATLAB parameter ``'size_img'`` is dropped. MATLAB saves only when the
    file already exists (no ``else`` branch); nothing is saved here. No MATLAB-made file
    of this generator exists (``14_cortical_regions_...mat`` was made by
    ``ROItemplate_generate_14ROIs_cortex_lambda_bregma``), so the output is unverified.
    """
    if shape not in ("square", "circle"):
        raise ValueError("shape must be 'square' or 'circle'")
    d = float(diameter)
    rois = []
    for (x0, y0), nm in zip(rois_master, _14_NAMES):
        x0, y0 = float(x0), float(y0)
        coords = []
        if shape == "square":
            for x in np.arange(x0, x0 + d + 1):
                for y in np.arange(y0, y0 + d + 1):
                    coords.append((x, y))
        else:
            for x in np.arange(x0, x0 + d):
                off = x - x0
                if off in (0, d - 1):
                    ign = 3
                elif off in (1, d - 2):
                    ign = 2
                elif off in (2, d - 3):
                    ign = 1
                else:
                    ign = 0
                for y in np.arange(y0 + ign, y0 + d - ign):
                    coords.append((x, y))
        c = np.array(coords, dtype=np.float64).reshape(-1, 2)
        key = nm.replace("left", "").replace("right", "")
        side = nm[: len(nm) - len(key)]
        rois.append(
            MatStruct(
                indices=_mat_col(_sub2ind(size_template, c[:, 1], c[:, 0])),
                type=np.float64(0.0),
                shift=_row([0, 0]),
                ML_index=_empty(),
                AP_index=_empty(),
                name=nm,
                Allen_area=f"{side} {_14_AREA[key]}",
                Allen_acronym=f"{side} {_14_ACRONYM[key]}",
                Allen_index=_empty(),
            )
        )
    rois = reference_rois(size_template, [vec_lambda, vec_bregma], ["Lambda", "Bregma"]) + rois
    params = MatStruct(
        SIZE_TEMPLATE=_row(size_template),
        VEC_LAMBDA=_row(vec_lambda),
        VEC_BREGMA=_row(vec_bregma),
        ROI_DIAMETER=_mat_num(diameter),
    )
    return _struct_rois(
        "14 cortical regions of interest based on lambda-bregma",
        "Template of 14 regions of interest across dorsal cortex. Alignment is done via "
        "lambda and bregma",
        "",
        "ROItemplate_generate_14ROIs_lambda_bregma",
        params, rois, binary_roi_image(rois, size_template, value=0.5),
    )


_14C_MASTER = (
    (916, 238), (914, 888), (834, 368), (838, 792), (874, 540), (874, 645), (614, 366),
    (614, 790), (450, 374), (458, 784), (442, 528), (444, 632), (266, 538), (266, 636),
)  # fmt: skip
_14C_INFO = (
    ("lV1", "lVISp", "left primary visual area"),
    ("rV1", "rVISp", "right primary visual area"),
    ("lV2", "lVISam", "left anteromedial visual area"),
    ("rV2", "rVISam", "right anteromedial visual area"),
    ("lRSC", "lRSP", "left retrosplenial area"),
    ("rRSC", "rRSP", "right retrosplenial area"),
    ("lS1", "lSSp", "left primary somatosensory area"),
    ("rS1", "rSSp", "right primary somatosensory area"),
    ("lM1", "lMOp", "left primary motor area"),
    ("rM1", "rMOp", "right primary motor area"),
    ("lACC", "lACA", "left anterior cingulate area"),
    ("rACC", "rACA", "left anterior cingulate area"),
    ("lM2", "lMOs", "left secondary motor area"),
    ("rM2", "rMOs", "right secondary motor area"),
)


def generate_template_14rois_cortex_lambda_bregma(
    *,
    vec_bregma=(488, 590),
    vec_lambda=(918, 590),
    roi_diameter=20,
    rois_master=_14C_MASTER,
) -> MatStruct:
    """Fourteen square ROIs, mirrored left/right about bregma, in a template cropped to
    them ("offline" mode).

    Port of the ``'offline'`` branch of ``12_Widefield/ROItemplate_library/Build_code/
    ROItemplate_generate_14ROIs_cortex_lambda_bregma.m``: every odd ROI of ``rois_master``
    (left hemisphere) is moved to its centre; every even ROI is its mirror image
    ``(x_left, |2*VEC_BREGMA(2) - y_left|)``. The template is then cropped to a margin of
    2.5 ROI diameters and made square (``max(coordinates) + 4*ROI_DIAMETER``).

    Parameters
    ----------
    vec_bregma, vec_lambda
        Reference points as in MATLAB (axes coordinates; note that MATLAB's values put
        "bregma" at the smaller first coordinate).
    roi_diameter
        ``ROI_DIAMETER``; each ROI has ``(ROI_DIAMETER+1)^2`` pixels.
    rois_master
        Corner points ``(a, b)`` of the 14 ROIs (lV1, rV1, ..., rM2); only the odd ones are
        used (the even ones are mirrored).

    Returns
    -------
    MatStruct
        ``struct_ROIs``; ROI fields ``indices``, ``type``, ``shift``, ``ML_index``,
        ``AP_index``, ``name`` (``'lV1'``...), ``Allen_area`` (``'lVISp'``...),
        ``Allen_desc``; reference ROIs have ``Allen_desc = []``. ``parameters``:
        ``SIZE_TEMPLATE``, ``VEC_LAMBDA``, ``VEC_BREGMA`` (shifted), ``ROI_DIAMETER``.
        ``img_ROIs`` = 0.5 at ROI pixels (float64).

    Notes
    -----
    Not ported: the ``'online'`` branch (``ExpID``/``ExpLog`` given), which reads the ROI
    positions from the experiment's ``.wii`` file through ExpLog lookups and an undefined
    ``ROI_mask_daq``/``ExpIDinfo`` (it cannot run as written).

    Kept from MATLAB: ``code_to_generate`` is ``'ROItemplate_generate_14ROIs_lambda_bregma'``
    (another generator's name) and the description of ``rACC`` reads ``'left anterior
    cingulate area'``. Differences: no ``dir_ROItemplate`` field, no figure, nothing saved.
    """
    d = float(roi_diameter)
    master = np.asarray(rois_master, dtype=np.float64)
    centre = master + d / 2
    sym = np.zeros_like(centre)
    for i in range(centre.shape[0]):
        if i % 2 == 0:
            x_left, y_left = centre[i]
            sym[i] = (x_left, y_left)
        else:
            sym[i] = (x_left, abs(2 * float(vec_bregma[1]) - y_left))
    shift = np.array([sym[:, 0].min() - 5 * d / 2, sym[:, 1].min() - 5 * d / 2])
    sym = sym - shift
    vb = np.asarray(vec_bregma, dtype=np.float64) - shift
    vl = np.asarray(vec_lambda, dtype=np.float64) - shift
    n = float(sym.max() + 4 * d)
    size_template = (n, n)
    rois = []
    for i, (nm, area, desc) in enumerate(_14C_INFO[: sym.shape[0]]):
        xs = np.arange(sym[i, 0] - d / 2, sym[i, 0] + d / 2 + 0.5)
        ys = np.arange(sym[i, 1] - d / 2, sym[i, 1] + d / 2 + 0.5)
        cx = np.repeat(xs, ys.size)
        cy = np.tile(ys, xs.size)
        rois.append(
            MatStruct(
                indices=_mat_col(_sub2ind(size_template, cy, cx)),
                type=np.float64(0.0),
                shift=_row([0, 0]),
                ML_index=_empty(),
                AP_index=_empty(),
                name=nm,
                Allen_area=area,
                Allen_desc=desc,
            )
        )
    fields = ["indices", "type", "shift", "ML_index", "AP_index", "name", "Allen_area",
              "Allen_desc"]
    rois = reference_rois(size_template, [vl, vb], ["Lambda", "Bregma"], fields=fields) + rois
    params = MatStruct(
        SIZE_TEMPLATE=_row(size_template),
        VEC_LAMBDA=_row(vl),
        VEC_BREGMA=_row(vb),
        ROI_DIAMETER=_mat_num(roi_diameter),
    )
    return _struct_rois(
        "14 cortical regions of interest based on lambda-bregma",
        "Template of 14 regions of interest across dorsal cortex. Alignment is done via "
        "lambda and bregma",
        "",
        "ROItemplate_generate_14ROIs_lambda_bregma",
        params, _struct_array(rois), binary_roi_image(rois, size_template, value=0.5),
    )

