"""RoiQuest: drawing and curating ROIs on two-photon and widefield recordings.

Port of calliope ``08_RoiQuest`` (the RoiQuest app and ActiQuest), built on the
computational cores of :mod:`calliopy.rois`:

- :class:`RoiQuestSession`: all of RoiQuest's state and one method per MATLAB callback,
  with explicit 0-based ``(row, col)`` pixel coordinates, keys and dialog answers; opens an
  adata file (:meth:`RoiQuestSession.from_adata`) or an ExpID
  (:meth:`RoiQuestSession.from_expid`, needs the ExpLog table and the adata folder) and
  saves ROIs back with :func:`calliopy.io.update_mat` after a backup copy;
- :func:`key_press`: MATLAB's key table (``keyParser`` and the menu accelerators);
- :class:`RoiQuestView`, :class:`HistogramView`: matplotlib drawing on any Axes (Agg works);
- :class:`ActiQuestSession`, :class:`ActiQuestView`: activity traces of selected ROIs;
- :func:`save_prefs`, :func:`load_prefs`: user preferences as JSON;
- :func:`overlay_image`, :func:`overlay_text`, :func:`histogram_data`, :func:`colormap`:
  what is drawn, as arrays and strings;
- :func:`run`, :class:`RoiQuestWindow`, :class:`ActiQuestWindow`: the tkinter windows
  (``python -m calliopy.roiquest <adata file>``).

tkinter is imported only when a window is opened, so the session works without Tk. Stored ROI ``indices`` stay MATLAB's 1-based values;
every coordinate or index passed to a method is 0-based. Cellpose is optional
(:func:`cellpose_available`).
"""

from ._actiquest import ActiQuestSession
from ._display import (
    COLORMAP_NAMES,
    OverlayImage,
    colormap,
    histogram_data,
    info_text_anchor,
    overlay_image,
    overlay_text,
    roi_number_labels,
)
from ._keys import KEY_BINDINGS, key_press, normalize_key
from ._load import prefs_file_name
from ._persist import BACKUP_INFIX
from ._prefs import load_prefs, prefs_dict, save_prefs
from ._render import ActiQuestView, HistogramView, RoiQuestView
from ._session import (
    DEAD_ZONE,
    EEG_RING_TEMPLATE_FILE,
    EEG_TEMPLATE_NAME,
    RoiQuestSession,
    cellpose_available,
)
from ._text import help_text, info_text
from ._types import KeyResult, Redraw, RoiQuestRefused, SaveResult, View
from ._why import DEFAULT_WHY_SUBJECTS, why_keller

__all__ = [
    "BACKUP_INFIX",
    "COLORMAP_NAMES",
    "DEAD_ZONE",
    "DEFAULT_WHY_SUBJECTS",
    "EEG_RING_TEMPLATE_FILE",
    "EEG_TEMPLATE_NAME",
    "KEY_BINDINGS",
    "ActiQuestSession",
    "ActiQuestView",
    "ActiQuestWindow",
    "HistogramView",
    "KeyResult",
    "OverlayImage",
    "Redraw",
    "RoiQuestRefused",
    "RoiQuestSession",
    "RoiQuestView",
    "RoiQuestWindow",
    "SaveResult",
    "View",
    "cellpose_available",
    "colormap",
    "help_text",
    "histogram_data",
    "info_text",
    "info_text_anchor",
    "key_press",
    "load_prefs",
    "normalize_key",
    "overlay_image",
    "overlay_text",
    "prefs_dict",
    "prefs_file_name",
    "roi_number_labels",
    "run",
    "save_prefs",
    "why_keller",
]


def run(adata_path=None, *, expid=None, exp_log=None, adata_dir=None, original_adata_path=None,
        roi_template_dir=None, block=True, master=None, dialogs=None):
    """Open the RoiQuest window, MATLAB ``RoiQuest(expId)``.

    See :func:`calliopy.roiquest._tk.run` for the parameters: an adata file (or, with
    ``exp_log`` and ``adata_dir``, an ExpID), the original file of a chronic series, the
    widefield ROI template folder, and ``block=False`` to return without running the Tk
    main loop. Returns the :class:`RoiQuestWindow` (the open one if RoiQuest is already open).
    """
    from ._tk import run as _run

    return _run(adata_path, expid=expid, exp_log=exp_log, adata_dir=adata_dir,
                original_adata_path=original_adata_path, roi_template_dir=roi_template_dir,
                block=block, master=master, dialogs=dialogs)


def __getattr__(name):
    if name == "RoiQuestWindow":
        from ._tk import RoiQuestWindow

        return RoiQuestWindow
    if name == "ActiQuestWindow":
        from ._tk_actiquest import ActiQuestWindow

        return ActiQuestWindow
    raise AttributeError(f"module 'calliopy.roiquest' has no attribute {name!r}")
