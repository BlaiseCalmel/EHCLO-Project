import math
import numpy as np
from plot_functions.plot_common import *

def lineplot(gdf, ds, indicator, path_result, row_name=None, row_headers=None, col_name=None, col_headers=None,
             cbar_title=None, title=None, dict_shapefiles=None, percent=True, bounds=None, discretize=7,
             vmin=None, vmax=None, palette='BrBG', fontsize=14, font='sans-serif'):
    col_keys = [None]
    col_values = None
    len_cols = 1
    row_keys = [None]
    row_values = None
    len_rows = 1
    if isinstance(col_headers, dict) and len(col_headers) > 0:
        col_keys = list(col_headers.keys())
        col_values = list(col_headers.values())
        len_cols = len(col_keys)
    if isinstance(row_headers, dict) and len(row_headers) > 0:
        row_keys = list(row_headers.keys())
        row_values = list(row_headers.values())
        len_rows = len(row_keys)

    if vmax is None:
        vmax = math.ceil(abs(ds.variables[indicator]).max() / 5) * 5
    if vmin is None:
        vmin = -vmax

    bounds_cmap = np.linspace(vmin, vmax, discretize+1)
    # cmap = mpl.cm.get_cmap(palette, discretize)
    cmap = plt.get_cmap(palette, discretize)
    norm = mpl.colors.BoundaryNorm(bounds_cmap, cmap.N)

    plt.rcParams['font.family'] = font
    plt.rcParams['font.size'] = fontsize
    text_kwargs ={'weight': 'bold'}

    fig, axes = plt.subplots(len_rows, len_cols, figsize=(1 + 4 * len_cols, len_rows * 4), constrained_layout=True)
    # Main title
    if title is not None:
        fig.suptitle(title, fontsize=plt.rcParams['font.size'] + 2)

    axes_flatten = axes.flatten()

    for col_idx, col in enumerate(col_keys):
        for row_idx, row in enumerate(row_keys):
            idx = len_cols * row_idx + col_idx
            ax = axes_flatten[idx]

            temp_dict = {}
            if col_name is not None and col is not None:
                temp_dict |= {col_name: col}
            if row_name is not None and row is not None:
                temp_dict |= {row_name: row}

            row_data = ds.sel(temp_dict)[indicator].values
            gdf[indicator] = row_data

            # Background shapefiles
            # if dict_shapefiles is not None:
            #     for key, subdict in dict_shapefiles.items():
            #         shp_kwargs = {k: subdict[k] for k in subdict.keys() if k != 'shp'}
            #         subdict['shp'].plot(ax=ax, figsize=(18, 18), **shp_kwargs)
            #
            # gdf.plot(column=indicator, cmap=cmap, norm=norm, ax=ax, legend=False, markersize=100, edgecolor='k',
            #          alpha=0.9, zorder=10)

            if bounds is not None:
                ax.set_xlim(bounds[0], bounds[2])
                ax.set_ylim(bounds[1], bounds[3])
            ax.set_axis_off()

    # Headers
    add_headers(fig, col_headers=col_values, row_headers=row_values, row_pad=0, col_pad=5, **text_kwargs)

    # Colorbar
    define_cbar(fig, axes_flatten, cmap, bounds_cmap, cbar_title=cbar_title, percent=percent, **text_kwargs)

    plt.savefig(path_result, bbox_inches='tight')

