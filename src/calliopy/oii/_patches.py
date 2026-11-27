"""Classical patch-stimulus OII analysis (computation of ``13_OII/analyzeOII_classical_patches.m``).

The MATLAB file is a script with a hard-coded stack; its computation is ported as functions
taking the arrays. Its filtered maps use ``Filter2Modified``, which is not part of calliope
r2003, and are not ported.

None of these functions has been compared with MATLAB output.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
from matplotlib.figure import Figure

from .._matlab import mround

__all__ = ["PatchConditions", "oii_patch_condition_images", "oii_stim_conditions", "plot_patch_montage"]


def oii_stim_conditions(stim_trace: Any, min_count: int = 20) -> np.ndarray:
    """Stimulus condition of every aux sample (``analyzeOII_classical_patches.m``).

    Conditions are the ranks of the distinct values of ``round(stim * 10)`` (1-based, as
    MATLAB's ``unique`` third output); conditions with fewer than ``min_count`` samples are
    merged into condition 1 and the higher ones renumbered, one at a time as in MATLAB;
    finally 1 is subtracted (0 = background).

    Returns
    -------
    numpy.ndarray
        int64, one value per sample.
    """
    v = mround(np.asarray(stim_trace, dtype=np.float64).ravel() * 10)
    _, cond = np.unique(v, return_inverse=True)
    cond = cond.astype(np.int64) + 1
    nn = np.bincount(cond, minlength=cond.max() + 1)[1:]
    wrong = list(np.flatnonzero(nn < min_count) + 1)
    for i in range(len(wrong)):
        w = wrong[i]
        cond[cond == w] = 1
        cond[cond > w] -= 1
        wrong = [x - 1 for x in wrong]
    return cond - 1


class PatchConditions(NamedTuple):
    """Result of :func:`oii_patch_condition_images`."""

    condimg: np.ndarray
    """``[h x w x nPatches]`` summed (frames - pre-onset) images, flipped left-right."""
    frame_id: np.ndarray
    """Condition of every frame."""
    frame_id_reps: np.ndarray
    """Repetition (1-based) of every frame."""


def oii_patch_condition_images(vdata: Any, frame_times: Any, stim_trace: Any,
                               patches: Sequence[int] = (3, 6)) -> PatchConditions:
    """Response image of every patch, summed over repetitions.

    Each frame gets the condition of the aux sample at its time (``frame_times[0]`` set to
    1, MATLAB 1-based samples). A repetition ends where the last condition is followed by
    0. For patch ``p`` and repetition ``r`` the response is ``mean(frames of p in r) -
    mean(5 frames ending at its first frame)``, flipped left-right; the condition image is
    the sum over repetitions.

    Parameters
    ----------
    vdata
        ``[h x w x nFrames]`` stack.
    frame_times
        Row 0 of ``vmeta_info`` (aux sample numbers).
    stim_trace
        Aux stimulus channel (MATLAB row 3, ``stim_ch = 3``).
    patches
        Grid of patches (MATLAB ``[3 6]``).

    Notes
    -----
    Differences from MATLAB: when a patch's first frame is earlier than frame 4 (0-based),
    the pre-onset mean uses the frames available (MATLAB failed indexing frame 0 or less);
    a patch missing in a repetition raises ``ValueError`` (MATLAB failed on ``aa(1)``).
    """
    vd = np.asarray(vdata)
    cond = oii_stim_conditions(stim_trace)
    ft = np.asarray(frame_times, dtype=np.float64).ravel().copy()
    ft[0] = 1
    fid = cond[ft.astype(np.int64) - 1]
    _, first = np.unique(fid, return_index=True)
    last_id = fid[np.sort(first)][-1]
    rep = list(np.flatnonzero((fid[:-1] == last_id) & (fid[1:] == 0)) + 1)
    rep = [0, *rep, fid.size] if fid[-1] == last_id else [0, *rep]
    reps = np.zeros(fid.size, dtype=np.int64)
    for k in range(len(rep) - 1):
        reps[rep[k] : rep[k + 1]] = k + 1
    npatch = int(np.prod(patches))
    out = np.zeros(vd.shape[:2] + (npatch,))
    for p in range(1, npatch + 1):
        acc = np.zeros(vd.shape[:2])
        for r in range(1, int(reps.max()) + 1):
            aa = np.flatnonzero((fid == p) & (reps == r))
            if aa.size == 0:
                raise ValueError(f"patch {p} missing in repetition {r}")
            a1 = vd[:, :, max(aa[0] - 4, 0) : aa[0] + 1].astype(np.float64).mean(axis=2)[:, ::-1]
            a2 = vd[:, :, aa].astype(np.float64).mean(axis=2)[:, ::-1]
            acc += a2 - a1
        out[:, :, p - 1] = acc
    return PatchConditions(out, fid, reps)


def plot_patch_montage(condimg: Any, patches: Sequence[int] = (3, 6), *, clim_scale=(0.4, 0.2)) -> Figure:
    """Montage of the condition images (``montage(..., 'Size', patches)``), colour limits
    ``[min * clim_scale[0], max * clim_scale[1]]`` as the script's second figure."""
    c = np.asarray(condimg, dtype=np.float64)
    rows, cols = int(patches[0]), int(patches[1])
    lo, hi = c.min() * clim_scale[0], c.max() * clim_scale[1]
    fig = Figure(figsize=(2 * cols, 2 * rows))
    for i in range(c.shape[2]):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(c[:, :, i], cmap="gray", vmin=lo, vmax=hi)
        ax.axis("off")
    return fig
