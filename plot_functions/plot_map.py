import math
import copy
from plot_functions.plot_common import *

def mapplot(gdf, ds, indicator_plot, path_result, cols, rows,
             cbar_title=None, title=None, dict_shapefiles=None, percent=True, bounds=None, discretize=7,
             vmin=None, vmax=None, palette='BrBG', fontsize=14, font='sans-serif'):

    ds_plot = copy.deepcopy(ds)
    len_cols, cols, ds_plot = init_grid(cols, ds_plot)
    len_rows, rows, ds_plot = init_grid(rows, ds_plot)

    if percent:
        if vmax is None:
            vmax = math.ceil(abs(ds_plot.variables[indicator_plot]).max() / 5) * 5
        if vmin is None:
            vmin = -vmax
    else:
        if vmax is None:
            vmax = int((ds_plot.variables[indicator_plot].max()))
        if vmin is None:
            vmin = int(ds_plot.variables[indicator_plot].min()) - 1

    bounds_cmap = np.linspace(vmin, vmax, discretize+1)
    cmap = plt.get_cmap(palette, discretize)
    norm = mpl.colors.BoundaryNorm(bounds_cmap, cmap.N)

    plt.rcParams['font.family'] = font
    plt.rcParams['font.size'] = fontsize
    text_kwargs ={'weight': 'bold'}

    if bounds is not None :
        x_y_ratio = abs((bounds[2] - bounds[0]) / (bounds[3] - bounds[1]))
        if x_y_ratio > 1:
            figsize = (4 * len_cols , len_rows * 4 / x_y_ratio)
        else:
            figsize = (4 * len_cols * x_y_ratio , len_rows * 4)
    else:
        figsize = (4 * len_cols, len_rows * 4)

    fig, axes = plt.subplots(len_rows, len_cols, figsize=figsize, constrained_layout=True)
    # Main title
    if title is not None:
        fig.suptitle(title, fontsize=plt.rcParams['font.size'] + 2)

    axes_flatten = axes.flatten()

    for col_idx, col in enumerate(cols['values_var']):
        for row_idx, row in enumerate(rows['values_var']):
            idx = len_cols * row_idx + col_idx
            ax = axes_flatten[idx]

            temp_dict = {}
            if cols['names_var'] is not None and col is not None:
                temp_dict |= {cols['names_var']: col}
            if rows['names_var'] is not None and row is not None:
                temp_dict |= {rows['names_var']: row}

            row_data = ds_plot.sel(temp_dict)[indicator_plot].values
            gdf[indicator_plot] = row_data

            # Background shapefiles
            if dict_shapefiles is not None:
                for key, subdict in dict_shapefiles.items():
                    shp_kwargs = {k: subdict[k] for k in subdict.keys() if k != 'shp'}
                    subdict['shp'].plot(ax=ax, figsize=(18, 18), **shp_kwargs)

            gdf.plot(column=indicator_plot, cmap=cmap, norm=norm, ax=ax, legend=False, markersize=100, edgecolor='k',
                     alpha=0.9, zorder=10)

            if bounds is not None:
                ax.set_xlim(bounds[0], bounds[2])
                ax.set_ylim(bounds[1], bounds[3])
            ax.set_axis_off()

    # Headers
    add_headers(fig, col_headers=cols['names_plot'], row_headers=rows['names_plot'], row_pad=5, col_pad=5, **text_kwargs)

    # Colorbar
    define_cbar(fig, axes_flatten, cmap, bounds_cmap, cbar_title=cbar_title, percent=percent, **text_kwargs)

    plt.savefig(path_result, bbox_inches='tight')

