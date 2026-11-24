"""Reading lab data files.

Every reader takes a file path; looking files up by ExpID comes later, once the lab
configuration and ExpLog access exist. See docs/porting_guide.md for the conventions
and docs/matlab_map.csv for the MATLAB function each reader replaces.

- MAT-files (v7.3): load_mat, MatFile, load_metafile, load_metafile_records, metafile_info,
  load_adata
- auxiliary files: read_ini, load_lvd, load_wcp, load_ach_chans, parse_ach, parse_vrd,
  read_sidecar_csv
- imaging and video: load_bin, load_wid_data, load_roi_data, load_wii_data,
  load_eye_monitor_data, load_vid_data, load_aux_vid_data, read_tif
- EEG and human recordings: load_eeg, load_heeg, load_heyetracking
"""

from ._aux import (
    LvdData,
    SidecarCsv,
    VrdData,
    load_ach_chans,
    load_lvd,
    load_wcp,
    parse_ach,
    parse_vrd,
    read_ini,
    read_sidecar_csv,
)
from ._eeg import (
    EEG_CHANNEL_ORDER,
    EEG_SYNC_CHANNEL,
    EegData,
    load_eeg,
    load_heeg,
    load_heyetracking,
)
from ._imaging import (
    AuxVidData,
    BinData,
    EyeData,
    RoiData,
    VidData,
    WidData,
    WiiData,
    load_aux_vid_data,
    load_bin,
    load_eye_monitor_data,
    load_roi_data,
    load_vid_data,
    load_wid_data,
    load_wii_data,
    read_tif,
)
from ._mat import (
    Adata,
    MatFile,
    MatlabObject,
    MatStruct,
    load_adata,
    load_mat,
    load_metafile,
    load_metafile_records,
    metafile_info,
)

__all__ = [
    "EEG_CHANNEL_ORDER",
    "EEG_SYNC_CHANNEL",
    "Adata",
    "AuxVidData",
    "BinData",
    "EegData",
    "EyeData",
    "LvdData",
    "MatFile",
    "MatStruct",
    "MatlabObject",
    "RoiData",
    "SidecarCsv",
    "VidData",
    "VrdData",
    "WidData",
    "WiiData",
    "load_ach_chans",
    "load_adata",
    "load_aux_vid_data",
    "load_bin",
    "load_eeg",
    "load_eye_monitor_data",
    "load_heeg",
    "load_heyetracking",
    "load_lvd",
    "load_mat",
    "load_metafile",
    "load_metafile_records",
    "load_roi_data",
    "load_vid_data",
    "load_wcp",
    "load_wid_data",
    "load_wii_data",
    "metafile_info",
    "parse_ach",
    "parse_vrd",
    "read_ini",
    "read_sidecar_csv",
    "read_tif",
]
