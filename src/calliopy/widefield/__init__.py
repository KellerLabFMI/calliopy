"""Widefield imaging on the standard ROI grid (818 ROIs).

- grid template: load_grid_template, GridTemplate
- ROI values as images: grid2vid (one pixel per ROI), roi_image (atlas-space image)
- plotting: plot_roi_values (ROI values with the Allen area borders)

The template file (e.g. ``818_ROIs_grid_template.mat``) is passed in by path; it is
lab data and is not part of calliopy.
"""

from ._grid import GridTemplate, grid2vid, load_grid_template, plot_roi_values, roi_image

__all__ = ["GridTemplate", "grid2vid", "load_grid_template", "plot_roi_values", "roi_image"]
