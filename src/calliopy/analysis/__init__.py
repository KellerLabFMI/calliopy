"""Signal analysis (calliope ``21_Analysis``, ``06_Library``): peak detection, baseline and
running-mean smoothing, Fourier filtering, band power, pulse envelopes, spatial complexity,
small array/cell utilities."""

from ._ampd import AmpdAdaptiveDebug, AmpdExtendedDebug, AmpdOriginalDebug, ampd
from ._jokes import WTF_ANSWERS, wtf
from ._library import expmat, flatten_cell, is_number
from ._signal import (
    avg_psd_in_range,
    check_vis_speed,
    extract_pulse_envelope,
    ftfil,
    ngsc,
    psmooth,
    smooth2,
)

__all__ = [
    "WTF_ANSWERS",
    "AmpdAdaptiveDebug",
    "AmpdExtendedDebug",
    "AmpdOriginalDebug",
    "ampd",
    "avg_psd_in_range",
    "check_vis_speed",
    "expmat",
    "extract_pulse_envelope",
    "flatten_cell",
    "ftfil",
    "is_number",
    "ngsc",
    "psmooth",
    "smooth2",
    "wtf",
]
