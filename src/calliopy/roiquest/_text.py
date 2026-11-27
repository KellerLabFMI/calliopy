"""Texts of RoiQuest's Help menu.

Port of calliope ``08_RoiQuest/+clbk/printHelp.m`` with ``08_RoiQuest/+utils/helpRoiQuest.txt``
and ``08_RoiQuest/+clbk/printInfo.m`` (Leonardo Lupori, 2024). MATLAB printed them in the
Command Window; here they are returned as strings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ._session import RoiQuestSession

__all__ = ["help_text", "info_text"]

# helpRoiQuest.txt with MATLAB's <strong> markup removed and its \t escapes expanded (fprintf).
HELP_TEXT = (
    '###########################################################################\n'
    '#####################  RoiQuest - Manual ##################################\n'
    '###########################################################################\n'
    '\n'
    'RoiQuest in a software for drawing ROIs (Region Of Interest) in 2P data.\n'
    'It works by semi-automatically detecting ROIs for each expID.\n'
    '\n'
    '### QUICK START GUIDE\n'
    '---------------------------------------------------------------------------\n'
    '- Use the scroll wheel to move between Z layers (up/down)\n'
    '    \t- Alternatively use the I and J keys to move up or down\n'
    '- Cycle through visualizations with the "[" and "]" keys\n'
    '    \tYou can draw ROIs on the template image (mean), the filtered template or the activity map\n'
    '- To draw a ROI click and drag a square around a cell\n'
    '    \t- now you are selecting the ROI, \n'
    '    \t- scroll up if you want more pixels or down if you want less\n'
    '    \t- press A to add the ROI\n'
    '    \t- press Esc or Q to cancel\n'
    '- You can select existing ROIs with right click\n'
    '    \t- select a single ROI by right clicking on it\n'
    '    \t- select multiple by click a dragging a selection square\n'
    '    \t- toggle selection of ROIs with middle-click or shift + left-click\n'
    '    \t- select all with Ctrl+A\n'
    '    \t- select none with Ctrl+N\n'
    '- Toggle ROIs visibility with spacebar\n'
    '- Save the ROIs to AData with Ctrl+S\n'
    '- To delete ROIs select the ones you want to delete and hit "Delete" \n'
    '- To move selected ROIs use the arrow keys\n'
    '    \t- hold shift to move faster\n'
    '- Open the image histogram with H. It automatically updates when changing images\n'
    '\n'
    '### ADVANCED\n'
    '---------------------------------------------------------------------------\n'
    '- You can Expand and Contract selected ROIs with Ctrl+P and Ctrl+L respectively\n'
    '- You can Tag ROIs with a type (0-9, 0 is default) by selecting them and hit a number\n'
    '- You can customize your visualization\n'
    '    \t- Change the colormap of the image\n'
    '    \t- Change the colormode of the ROIs\n'
    '        \t\t- Flat: All ROIs are of the same colors\n'
    '        \t\t- Random: ROIs are of the random colors\n'
    '        \t\t- Tag: ROIs are colored according to their tag\n'
    '    \t- Change the color of the ROIs\n'
    '        \t\t- Change the color of ROIs in Flat colormode\n'
    '        \t\t- Change the color of ROIs during selection mode\n'
    '    \t- Change the image contrast\n'
    '        \t\t- Min/Max: Contrast is scaled to fit the entire range of values in the image\n'
    '        \t\t- Best Fit: The top and bottom 0.5 percentiles are saturated\n'
    '        \t\t- Custom: Custom contrast adjustment (change the values in the "Change custom contrast" menu)\n'
    '- You can save/load your personalized settings in the File menu\n'
    '- You can automatically detect and segment ROIs (powered by cellpose) by clicking ROIs/Autodetect or Ctrl+T\n'
    '\n'
    '### CHRONIC IMAGING\n'
    '---------------------------------------------------------------------------\n'
    '- Always start to draw ROIs from the original ExpID (expId == siteId)\n'
    '- You can load ROIs from the original file with the menu ROIs/Load from Original\n'
    '- You can auto-adjust ROI positions by selecting all ROIs (Ctrl+A) and hitting C\n'
    '- From any visualization (template, filtered template or activity) and any z-layer \n'
    '\tyou can toggle the original expID by pressing o\n'
    '- When saving ROIs, RoiQuest will automatically match ROI identity in all timepoints\n'
    '- Deleting ROIs: \n'
    '    \t- Before deleting ROIs in a chronic experiment save first!\n'
    '    \t- After deletion RoiQuest will reload ROIs from AData file. Unsaved modifications will be lost \n'
    '    \t- RoiQuest will delete the ROIs from all timepoints'
)


def help_text() -> str:
    """Manual of RoiQuest (keys, mouse, chronic imaging), MATLAB ``clbk.printHelp``.

    Returns
    -------
    str
        The text of ``helpRoiQuest.txt`` as MATLAB prints it, without the ``<strong>``
        markup of the Command Window.
    """
    return HELP_TEXT


def info_text(s: RoiQuestSession) -> str:
    """State report, MATLAB ``clbk.printInfo``: adata file, ExpID, layers, thresholds,
    opacity, filter size, custom contrast limits and the number of ROIs per Z layer
    (layers numbered from 1)."""
    from ._display import _fmt_u

    out = ["#" * 50, "########## ROIQUEST - INFO", "#" * 50]
    out.append(f" - AData loaded: {'' if s.adata_full_path is None else s.adata_full_path}")
    out.append(f" - ExpID: {_fmt_u(s.curr_exp_id)}")
    out.append(f" - This exp has {len(s.template)} piezo layers. Now visualizing "
               f"#{s.current_z + 1}")
    out.append(" - Thresholds:")
    for name, k in (("Template", 0), ("Filtered", 1), ("Activity", 2)):
        out.append(f"\t - {name} image: {s.thr[k]:.2f}")
    out.append(f" - ROI Opacity: {s.ovrl_opacity:.2f}")
    fws = float("nan") if s.filter_win_size is None else float(s.filter_win_size)
    out.append(f" - Filter size: {fws:.2f}")
    out.append(" - Custom contrast thresholds:")
    for name, k in (("Template", 0), ("Filtered", 1), ("Activity", 2)):
        out.append(f"\t - {name} image: [{s.img_lo[k]:.2f} {s.img_hi[k]:.2f}]")
    out.append("")
    out.append(" ------------ ROIs ------------")
    out.append(f" {'Z-Layer':<15}{'# ROIs':<15}")
    out.append(" " + "-" * 30)
    for i in range(len(s.template)):
        n = len(s.rois[i]) if i < len(s.rois) else 0
        out.append(f" {i + 1:<15}{n:<15}")
    out.append(" " + "-" * 30)
    return "\n".join(out) + "\n"
