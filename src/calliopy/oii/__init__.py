"""Intrinsic optical imaging (OII): Fourier retinotopy and patch-stimulus maps
(calliope r2003 ``13_OII``)."""

from ._adata_save import save_analyze_oii_fourier_results
from ._fourier import (
    FourierMaps,
    OiiFourierResult,
    OiiStimEpochs,
    analyze_oii_fourier,
    analyze_oii_fourier_files,
    fourier_maps,
    oii_normalize,
    oii_stim_bounds,
    oii_stim_epochs,
    plot_fourier_maps,
    power_overlay,
    smooth2,
    wave_reference_image,
    wave_stimulus_maps,
)
from ._patches import (
    PatchConditions,
    oii_patch_condition_images,
    oii_stim_conditions,
    plot_patch_montage,
)

__all__ = [
    "FourierMaps",
    "OiiFourierResult",
    "OiiStimEpochs",
    "PatchConditions",
    "analyze_oii_fourier",
    "analyze_oii_fourier_files",
    "fourier_maps",
    "oii_normalize",
    "oii_patch_condition_images",
    "oii_stim_bounds",
    "oii_stim_conditions",
    "oii_stim_epochs",
    "plot_fourier_maps",
    "plot_patch_montage",
    "power_overlay",
    "save_analyze_oii_fourier_results",
    "smooth2",
    "wave_reference_image",
    "wave_stimulus_maps",
]
