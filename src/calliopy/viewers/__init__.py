"""Interactive viewers: sessions, matplotlib views and tkinter windows.

Ports of calliope r2003 interactive tools, each split as in :mod:`calliopy.roiquest`: a
session class without any GUI toolkit (state and one method per callback, 0-based
positions), a view that draws it on matplotlib axes, and a thin tkinter window opened by
``run_<name>(..., block=True)``.

- cell browser (``20_MetaData/cell_browser.m`` and its plugins ``cb_plugin_ACX``, ``_M1``,
  ``_VML``, ``_trav``): run_cell_browser, CellBrowserSession, CellBrowserView,
  plot_cb_stim_snippets, plot_cb_plugin_acx, plot_cb_plugin_m1, plot_cb_plugin_vml,
  plot_cb_plugin_trav
- stack alignment (``07_ViewStack/StackAligner.m``): run_stack_aligner,
  StackAlignerSession, StackAlignerView
- manual widefield registration (``12_Widefield/register_widefield_exp_manual.m``):
  run_register_widefield_exp_manual, ManualRegistrationSession, ManualRegistrationView,
  manual_registration_inputs, manual_registration_matlab_code, save_manual_registration
- alignment checks (``11_TwoPhoton/tpi_alignment_visualize.m``,
  ``12_Widefield/wid_alignment_visualize.m``): run_tpi_alignment_visualize,
  run_wid_alignment_visualize, tpi_alignment_figures, wid_alignment_figures,
  AlignmentDecision, set_check_raw_passed

Nothing here looks up an ExpID: adata files, metafiles and annotation files are passed in
by path. tkinter is imported only when a window is opened.

Backwards compatibility: the cell browser's annotations file (MATLAB's default v7 format)
and ``check_raw.passed`` keep MATLAB's layout (variables, fields, classes, 1-based values).
The manual registration writes ``tform`` as the 3x3 matrix ``tform.A``, as the other
calliopy registration savers (MATLAB stores a ``simtform2d`` object);
:func:`manual_registration_matlab_code` gives MATLAB lines for the object form.
"""

from ._alignment_check import (
    AlignmentDecision,
    set_check_raw_passed,
    tpi_alignment_figures,
    wid_alignment_figures,
)
from ._cb_plugins import (
    plot_cb_plugin_acx,
    plot_cb_plugin_m1,
    plot_cb_plugin_trav,
    plot_cb_plugin_vml,
    plot_cb_stim_snippets,
)
from ._cell_browser import CELL_BROWSER_KEYS, CellBrowserSession, CellBrowserView
from ._manual_registration import (
    MANUAL_REGISTRATION_KEYS,
    ManualRegistrationInputs,
    ManualRegistrationSession,
    ManualRegistrationView,
    manual_registration_inputs,
    manual_registration_matlab_code,
    save_manual_registration,
)
from ._stack_aligner import STACK_ALIGNER_KEYS, StackAlignerSession, StackAlignerView

__all__ = [
    "CELL_BROWSER_KEYS",
    "MANUAL_REGISTRATION_KEYS",
    "STACK_ALIGNER_KEYS",
    "AlignmentDecision",
    "CellBrowserSession",
    "CellBrowserView",
    "ManualRegistrationInputs",
    "ManualRegistrationSession",
    "ManualRegistrationView",
    "StackAlignerSession",
    "StackAlignerView",
    "manual_registration_inputs",
    "manual_registration_matlab_code",
    "plot_cb_plugin_acx",
    "plot_cb_plugin_m1",
    "plot_cb_plugin_trav",
    "plot_cb_plugin_vml",
    "plot_cb_stim_snippets",
    "run_cell_browser",
    "run_register_widefield_exp_manual",
    "run_stack_aligner",
    "run_tpi_alignment_visualize",
    "run_wid_alignment_visualize",
    "save_manual_registration",
    "set_check_raw_passed",
    "tpi_alignment_figures",
    "wid_alignment_figures",
]

_RUNNERS = {
    "run_cell_browser": "._tk_cell_browser",
    "run_stack_aligner": "._tk_registration",
    "run_register_widefield_exp_manual": "._tk_registration",
    "run_tpi_alignment_visualize": "._tk_alignment",
    "run_wid_alignment_visualize": "._tk_alignment",
}


def __getattr__(name: str):
    """Import the tkinter windows only when a ``run_*`` function is first used."""
    if name in _RUNNERS:
        import importlib

        return getattr(importlib.import_module(_RUNNERS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
