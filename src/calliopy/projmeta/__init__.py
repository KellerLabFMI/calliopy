"""Project meta files (calliope ``24_ProjMeta`` and ``20_MetaData``).

- assembling a proj_meta: generate_proj_meta and its steps project_acodes,
  proj_meta_selection, proj_meta_filename, align_aux_channels, proj_meta_exp_type,
  read_frame_rate; heeg_proj_meta, generate_meta_info, lab_note_page_get
- aux channel processing: assign_aux_method, assign_processed_aux, assign_load_ftypes,
  pulsed_to_continuous, get_vel_ind_from_adata, get_lick_times_thrsh
- ROI activity: get_smoothed_rois, preprocess_widefield_activity,
  widefield_rois_for_proj_meta
- reading a proj_meta: pm_expids, pm_info, act2mat, trig2snps, add_uuid_to_proj_meta,
  activity_onsets, traversal_activity, cell_browser_good_cells, plot_pm_sites
"""

from ._access import (
    Act2MatResult,
    PmInfoResult,
    TraversalActivityResult,
    Trig2SnpsResult,
    act2mat,
    activity_onsets,
    add_uuid_to_proj_meta,
    cell_browser_good_cells,
    pm_expids,
    pm_info,
    traversal_activity,
    trig2snps,
)
from ._aux import (
    GetVelIndResult,
    assign_aux_method,
    assign_load_ftypes,
    assign_processed_aux,
    get_lick_times_thrsh,
    get_vel_ind_from_adata,
    pulsed_to_continuous,
)
from ._generate import (
    AlignAuxChannelsResult,
    ExpTypeResult,
    ProjMetaSelection,
    align_aux_channels,
    generate_meta_info,
    generate_proj_meta,
    heeg_proj_meta,
    lab_note_page_get,
    proj_meta_exp_type,
    proj_meta_filename,
    proj_meta_selection,
    project_acodes,
    read_frame_rate,
)
from ._plot import plot_pm_sites
from ._rois import (
    GetSmoothedRoisResult,
    get_smoothed_rois,
    preprocess_widefield_activity,
    widefield_rois_for_proj_meta,
)

__all__ = [
    "Act2MatResult",
    "AlignAuxChannelsResult",
    "ExpTypeResult",
    "GetSmoothedRoisResult",
    "GetVelIndResult",
    "PmInfoResult",
    "ProjMetaSelection",
    "TraversalActivityResult",
    "Trig2SnpsResult",
    "act2mat",
    "activity_onsets",
    "add_uuid_to_proj_meta",
    "align_aux_channels",
    "assign_aux_method",
    "assign_load_ftypes",
    "assign_processed_aux",
    "cell_browser_good_cells",
    "generate_meta_info",
    "generate_proj_meta",
    "get_lick_times_thrsh",
    "get_smoothed_rois",
    "get_vel_ind_from_adata",
    "heeg_proj_meta",
    "lab_note_page_get",
    "plot_pm_sites",
    "pm_expids",
    "pm_info",
    "preprocess_widefield_activity",
    "proj_meta_exp_type",
    "proj_meta_filename",
    "proj_meta_selection",
    "project_acodes",
    "pulsed_to_continuous",
    "read_frame_rate",
    "traversal_activity",
    "trig2snps",
    "widefield_rois_for_proj_meta",
]
