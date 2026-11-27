"""ROI definition and matching (port of calliope ``08_RoiQuest``).

RoiQuest is the lab's interactive MATLAB GUI for drawing ROIs on two-photon and widefield
templates and keeping them matched across chronic recordings. This subpackage ports its
computational layer; the GUI itself (figures, callbacks, key handling, preferences,
ActiQuest windows, Cellpose loading) is deferred until calliopy has an interactive viewer.
Nothing here looks up an ExpID, prints, or writes into an adata file: what MATLAB saved is
returned as ``MatStruct`` values with MATLAB's variable names.

ROI ``indices`` are always the stored MATLAB values: 1-based column-major linear indices.
Arguments that select or locate something (which ROIs, crop rectangles, cross centres) are
0-based. See docs/matlab_map.csv for the MATLAB function each name replaces.

- ROI objects: Roi, RoiWf, rois_binary_mask, rois_label_image, rois_label_type_image
- adata ROI structs: rois2struct, struct2rois, struct2wf_rois, struct2wf_rois_2ch,
  move_roi_structs
- images: filter_template, default_filter_size, adjust_img_contrast, get_labels_colormap,
  uniform_rgb_image, make_cross, segment_neuron, valid_area, filter_cellpose_labels
- chronic matching and saving: auto_adjust_rois, auto_roi_matching, check_roi_matching,
  roi_match_message, save_rois_acute, save_rois_chronic, plot_rois_to_add, exp_group
- widefield: wf_transform_rois, wf_transform_rois_2ch, wf_rois_struct, wf_save_rois
- ActiQuest: actiquest_activity, actiquest_traces, plot_actiquest
"""

from ._images import (
    CellposeRois,
    actiquest_activity,
    actiquest_traces,
    adjust_img_contrast,
    default_filter_size,
    filter_cellpose_labels,
    filter_template,
    get_labels_colormap,
    make_cross,
    plot_actiquest,
    plot_rois_to_add,
    segment_neuron,
    uniform_rgb_image,
    valid_area,
)
from ._matching import (
    AutoAdjustResult,
    AutoRoiMatchingResult,
    ChronicRoiSave,
    RoiMatchResult,
    RoiSaveError,
    auto_adjust_rois,
    auto_roi_matching,
    check_roi_matching,
    exp_group,
    roi_match_message,
    save_rois_acute,
    save_rois_chronic,
)
from ._roi import (
    Roi,
    RoiWf,
    move_roi_structs,
    rois2struct,
    rois_binary_mask,
    rois_label_image,
    rois_label_type_image,
    struct2rois,
    struct2wf_rois,
    struct2wf_rois_2ch,
)
from ._wf import (
    CORNER_ROI_NAMES_RQ,
    wf_rois_struct,
    wf_save_rois,
    wf_transform_rois,
    wf_transform_rois_2ch,
)

__all__ = [
    "CORNER_ROI_NAMES_RQ",
    "AutoAdjustResult",
    "AutoRoiMatchingResult",
    "CellposeRois",
    "ChronicRoiSave",
    "Roi",
    "RoiMatchResult",
    "RoiSaveError",
    "RoiWf",
    "actiquest_activity",
    "actiquest_traces",
    "adjust_img_contrast",
    "auto_adjust_rois",
    "auto_roi_matching",
    "check_roi_matching",
    "default_filter_size",
    "exp_group",
    "filter_cellpose_labels",
    "filter_template",
    "get_labels_colormap",
    "make_cross",
    "move_roi_structs",
    "plot_actiquest",
    "plot_rois_to_add",
    "roi_match_message",
    "rois2struct",
    "rois_binary_mask",
    "rois_label_image",
    "rois_label_type_image",
    "save_rois_acute",
    "save_rois_chronic",
    "segment_neuron",
    "struct2rois",
    "struct2wf_rois",
    "struct2wf_rois_2ch",
    "uniform_rgb_image",
    "valid_area",
    "wf_rois_struct",
    "wf_save_rois",
    "wf_transform_rois",
    "wf_transform_rois_2ch",
]
