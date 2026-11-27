"""Plotting (calliope ``22_Plotting``): bootstrapped traces and comparison strips, centred
colormaps, violins, mean +/- SEM, stacked traces, ROI images."""

from ._bstrp import PlotBstrpResult, plot_bstrp
from ._bstrp_multi import PlotBstrpMultiResult, bootstrap_overlap_pvalues, plot_bstrp_multi
from ._cmap import cmap_centered
from ._linespec import linespec_kwargs
from ._rois_image import rois_to_image
from ._traces import SpecTracesResult, plot_sem, spec_traces, spec_traces_data
from ._violin import Violin, violin, violinplot

__all__ = [
    "PlotBstrpMultiResult",
    "PlotBstrpResult",
    "SpecTracesResult",
    "Violin",
    "bootstrap_overlap_pvalues",
    "cmap_centered",
    "linespec_kwargs",
    "plot_bstrp",
    "plot_bstrp_multi",
    "plot_sem",
    "rois_to_image",
    "spec_traces",
    "spec_traces_data",
    "violin",
    "violinplot",
]
