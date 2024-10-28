import math
import numpy as np
from plot_functions.plot_common import *

def lineplot(ds, x_axis, y_axis, cols, rows, path_result, xmin=None, xmax=None, ymin=None, ymax=None,
             title=None, percent=True, palette='BrBG', fontsize=14, font='sans-serif', ):
    # col_keys = [None]
    # col_values = None
    # len_cols = 1
    # row_keys = [None]
    # row_values = None
    # len_rows = 1
    # if isinstance(col_headers, dict) and len(col_headers) > 0:
    #     col_keys = list(col_headers.keys())
    #     col_values = list(col_headers.values())
    #     len_cols = len(col_keys)
    # if isinstance(row_headers, dict) and len(row_headers) > 0:
    #     row_keys = list(row_headers.keys())
    #     row_values = list(row_headers.values())
    #     len_rows = len(row_keys)

    ds_plot = ds.sel({cols['name_var']: cols['values_var'],
                      rows['name_var']: rows['values_var']})

    if isinstance(x_axis['names_var'], str):
        x_axis['names_var'] = [x_axis['names_var']]
    if isinstance(x_axis['names_var'], str):
        y_axis['names_var'] = [y_axis['names_var']]


    if xmin is None:
        x_min_temp = [ds_plot.variables[i].min() for i in x_axis['names_var']]
        try:
            xmin = np.nanmin(x_min_temp)
        except ValueError:
            xmin = min(x_min_temp)
    if xmax is None:
        x_max_temp = max([ds_plot.variables[i].max() for i in x_axis['names_var']])
        try:
            xmax = np.nanmin(x_max_temp)
        except ValueError:
            xmax = max(x_max_temp)

    if ymin is None:
        y_min_temp = [ds_plot.variables[i].min() for i in y_axis['names_var']]
        try:
            ymin = np.nanmin(y_min_temp)
        except ValueError:
            ymin = min(y_min_temp)

    if ymax is None:
        y_max_temp = np.nanmax([ds_plot.variables[i].max() for i in y_axis['names_var']])
        try:
            ymax = np.nanmax(y_max_temp)
        except ValueError:
            ymax = max(y_min_temp)


    plt.rcParams['font.family'] = font
    plt.rcParams['font.size'] = fontsize
    text_kwargs ={'weight': 'bold'}

    len_rows = len(rows['values_var'])
    len_cols = len(cols['values_var'])

    fig, axes = plt.subplots(len_rows, len_cols, figsize=(1 + 6 * len_cols, len_rows * 4), constrained_layout=True)
    # Main title
    if title is not None:
        fig.suptitle(title, fontsize=plt.rcParams['font.size'] + 2)

    axes_flatten = axes.flatten()

    for col_idx, col in enumerate(cols['values_var']):
        for row_idx, row in enumerate(rows['values_var']):
            idx = len_cols * row_idx + col_idx
            ax = axes_flatten[idx]

            temp_dict = {}
            if cols['name_var'] is not None and col is not None:
                temp_dict |= {cols['name_var']: col}
            if rows['name_var'] is not None and row is not None:
                temp_dict |= {rows['name_var']: row}

            for y_var in y_axis['names_var']:
                for x_var in x_axis['names_var']:
                    row_data = ds_plot.sel(temp_dict)[y_var]
                    # ds.sel(id_geometry=1)[var_name].plot(ax=ax, color='lightgrey')
                    ax.plot(row_data[x_var], row_data.values, color='lightgrey', alpha=0.8)

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            # ax.set_axis_off()

    # Headers
    add_headers(fig, col_headers=cols['name_plot'], row_headers=rows['name_plot'], row_pad=25, col_pad=5, **text_kwargs)

    # Colorbar
    # define_cbar(fig, axes_flatten, cmap, bounds_cmap, cbar_title=cbar_title, percent=percent, **text_kwargs)

    plt.savefig(path_result, bbox_inches='tight')

