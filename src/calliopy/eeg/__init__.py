"""EEG: alignment to aux time, ERPs, resampling and spectra (calliope r2003 ``05_LoadData``,
``04_Calliope/get_eeg_times.m``). Loading EEG files is in :mod:`calliopy.io`
(:func:`calliopy.io.load_eeg`)."""

from ._eeg import (
    EegGratingErps,
    EegSpectrum,
    ErpResult,
    TriggerAlignmentError,
    create_erp_auxaligned,
    eeg_average_spectrum,
    eeg_average_spectrum_files,
    eeg_grating_erps,
    eeg_grating_erps_files,
    get_eeg_times,
    my_resample,
    my_welch,
    plot_eeg_spectrum,
    plot_erp,
    read_eeg_dat,
    rising_edges,
)

__all__ = [
    "EegGratingErps",
    "EegSpectrum",
    "ErpResult",
    "TriggerAlignmentError",
    "create_erp_auxaligned",
    "eeg_average_spectrum",
    "eeg_average_spectrum_files",
    "eeg_grating_erps",
    "eeg_grating_erps_files",
    "get_eeg_times",
    "my_resample",
    "my_welch",
    "plot_eeg_spectrum",
    "plot_erp",
    "read_eeg_dat",
    "rising_edges",
]
