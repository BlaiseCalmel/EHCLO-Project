import math
import copy

import numpy as np
from matplotlib.colors import from_levels_and_colors
from plot_functions.plot_common import *
import matplotlib.cm as cm

def format_significant(lst, n):
    # formatted_list = [f"{x:.{n}g}" for x in lst]
    formatted_list = [str(np.round(x, n)) for x in lst]
    return formatted_list

def mapplot(gdf, indicator_plot, path_result, cols=None, rows=None, ds=None,
            cbar_title=None, title=None, dict_shapefiles=None, percent=True, bounds=None, discretize=7,
            vmin=None, vmax=None, palette='BrBG', cbar_midpoint=None, cbar_ticks='border',
            fontsize=14, edgecolor='k', cbar_values=None,
            font='sans-serif', references=None, markersize=50):

    ds_plot = copy.deepcopy(ds)
    len_cols, cols_plot, ds_plot = init_grid(cols, ds_plot)
    len_rows, rows_plot, ds_plot = init_grid(rows, ds_plot)
    subplot_titles = None
    if isinstance(rows, int):
        len_cols = int(len_cols / len_rows)
        subplot_titles = cols['names_plot']
        cols_plot['names_plot'] = [None]
    if isinstance(cols, int):
        len_rows = int(len_rows / len_cols)
        subplot_titles = rows['names_plot']
        rows_plot['names_plot'] = [None]


    if percent:
        if vmax is None:
            if np.logical_not(isinstance(indicator_plot, list)) and indicator_plot in gdf.columns:
                vmax = math.ceil(abs(gdf[indicator_plot]).max() / 5) * 5
            else:
                if np.logical_not(isinstance(indicator_plot, list)):
                    vmax = math.ceil(abs(ds[indicator_plot]).max() / 5) * 5
                else:
                    vmax = math.ceil(abs(ds[indicator_plot].to_array()).max() / 5) * 5

        if vmin is None:
            vmin = -vmax
    else:
        if vmax is None:
            if np.logical_not(isinstance(indicator_plot, list)) and indicator_plot in gdf.columns:
                vmax = gdf[indicator_plot].max().values
            else:
                if np.logical_not(isinstance(indicator_plot, list)):
                    vmax = ds[indicator_plot].max().values
                else:
                    vmax = ds[indicator_plot].to_array().max().values
        if vmin is None:
            if np.logical_not(isinstance(indicator_plot, list)) and indicator_plot in gdf.columns:
                vmin = gdf[indicator_plot].min().values
            else:
                if np.logical_not(isinstance(indicator_plot, list)):
                    vmin = ds[indicator_plot].min().values
                else:
                    vmin = ds[indicator_plot].to_array().min().values

    # if vmax - vmin > 5:
    #     bounds_cmap = np.linspace(vmin, vmax, discretize+1, dtype=int)
    # else:
    #     bounds_cmap = np.linspace(vmin, vmax, discretize+1)

    if cbar_midpoint == 'min':
        midpoint = vmin
    elif cbar_midpoint == 'zero':
        midpoint = 0
    else:
        midpoint = None

    if midpoint is not None:
        levels = np.linspace(vmin, vmax, discretize+1)
        midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
        vals = np.interp(midp, [vmin, midpoint, vmax], [0, 0.5, 1])
        colormap = getattr(plt.cm, palette)
        colors = colormap(vals)
        cmap, norm = from_levels_and_colors(levels, colors)

        # cmap_temp = plt.get_cmap(palette, 256)
        # cmap = mpl.colors.LinearSegmentedColormap.from_list(palette, cmap_temp(np.linspace(0.5, 1, 128)),
        #                                                     N=discretize)
    else:
        levels = np.linspace(vmin, vmax, discretize+1)
        cmap = plt.get_cmap(palette, discretize)
        norm = mpl.colors.BoundaryNorm(boundaries=levels, ncolors=discretize)

    plt.rcParams['font.family'] = font
    plt.rcParams['font.size'] = fontsize
    text_kwargs ={'weight': 'bold'}

    fig_dim = 4
    # fig_dim = 8
    if bounds is not None:
        x_y_ratio = abs((bounds[2] - bounds[0]) / (bounds[3] - bounds[1]))
        if x_y_ratio > 1:
            figsize = (fig_dim * len_cols , len_rows * fig_dim / x_y_ratio)
        else:
            figsize = (fig_dim * len_cols * x_y_ratio , len_rows * fig_dim)
    else:
        figsize = (fig_dim * len_cols, len_rows * fig_dim)

    fig, axes = plt.subplots(len_rows, len_cols, figsize=figsize, constrained_layout=True)
    # Main title
    if title is not None:
        fig.suptitle(title, fontsize=plt.rcParams['font.size'] + 2, weight='bold')

    if len_rows > 1 or len_cols > 1:
        axes_flatten = axes.flatten()
    else:
        axes_flatten = [axes]

    # Save axes position
    # for ax in axes_flatten:
    #     initial_position = ax.get_position()
    #     ax.set_position(initial_position)

    # Iterate over subplots
    idx = -1
    for col_idx, col in enumerate(cols_plot['values_var']):
        for row_idx, row in enumerate(rows_plot['values_var']):
            idx += 1
            # idx = len_cols * row_idx + col_idx
            subplot_title = None
            ax = axes_flatten[idx]
            if cols_plot['names_coord'] == 'indicator':
                current_indicator = col
            elif rows_plot['names_coord'] == 'indicator':
                current_indicator = row
            elif isinstance(indicator_plot, list):
                current_indicator = indicator_plot[idx]
            else:
                current_indicator = indicator_plot

            if subplot_titles:
                subplot_title = subplot_titles[idx]

            gdf_plot = copy.deepcopy(gdf)
            # If not in gdf_plot, add a col from selection
            if current_indicator not in gdf.columns:
                temp_dict = {}
                if cols_plot['names_coord'] is not None and col is not None and cols_plot['names_coord'] != 'indicator':
                    temp_dict |= {cols_plot['names_coord']: col}
                if rows_plot['names_coord'] is not None and row is not None and rows_plot['names_coord'] != 'indicator':
                    temp_dict |= {rows_plot['names_coord']: row}

                row_data = ds_plot.sel(temp_dict)[current_indicator].values
                gdf_plot[current_indicator] = row_data

            # Background shapefiles
            if dict_shapefiles is not None:
                for key, subdict in dict_shapefiles.items():
                    shp_kwargs = {k: subdict[k] for k in subdict.keys() if k != 'shp'}
                    subdict['shp'].plot(ax=ax, figsize=(18, 18), **shp_kwargs)

            # Plot
            gdf_plot.plot(column=current_indicator, cmap=cmap, norm=norm, ax=ax, legend=False, markersize=markersize,
                          edgecolor=edgecolor, alpha=0.9, zorder=10, )


            if references is not None:
                for key, values in references.items():
                    scatter_kwarg = {i: j for i, j in values.items() if i != 'text'}
                    ax.scatter(**scatter_kwarg)
                    if 'text' in values.keys():
                        text_kwarg = values['text']
                        ax.annotate(xycoords='data',
                                    weight='bold',
                                    fontsize=plt.rcParams['font.size'] - 2, **text_kwarg)

            if subplot_title:
                ax.set_title(subplot_title)

            # Headers and axes label
            add_header(ax, rows_plot, cols_plot, ylabel='', xlabel='')

            if bounds is not None:
                ax.set_xlim(bounds[0], bounds[2])
                ax.set_ylim(bounds[1], bounds[3])
            # ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])
            plt.setp(ax.spines.values(), color=None)

    # Colorbar
    cbar = define_cbar(fig, axes_flatten, len_rows, len_cols, cmap, bounds_cmap=levels,
                       cbar_title=cbar_title, percent=percent, **text_kwargs)
    if cbar_ticks == 'mid':
        cbar.set_ticks((levels[1:] + levels[:-1])/2)
        cbar.ax.tick_params(size=0)
        if cbar_values is None:
            cbar.set_ticklabels((levels[1:] + levels[:-1])/2)
        else:
            if isinstance(cbar_values, int):
                cbar.set_ticklabels(format_significant((levels[1:] + levels[:-1])/2, cbar_values))
            else:
                cbar.set_ticklabels(cbar_values)
    elif cbar_values is not None:
        tick_values = cbar.get_ticks()
        cbar.set_ticklabels(format_significant(tick_values, cbar_values))

    plt.savefig(path_result, bbox_inches='tight')

