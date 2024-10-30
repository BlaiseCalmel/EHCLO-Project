import copy
import numpy as np
import matplotlib.ticker as mtick
from plot_functions.plot_common import *

def lineplot(ds, x_axis, y_axis, path_result, cols, rows, xmin=None, xmax=None, ymin=None, ymax=None,
             title=None, percent=True, fontsize=14, font='sans-serif', ):

    ds_plot = copy.deepcopy(ds)
    if cols is not None:
        len_cols = len(cols['values_var'])
        if cols['names_var'] != 'indicator':
            ds_plot = ds_plot.sel({cols['names_var']: cols['values_var']})
    else:
        len_cols = 1
        cols = {'values_var': [None], 'names_plot': [None]}

    if rows is not None:
        len_rows = len(rows['values_var'])
        if rows['names_var'] != 'indicator':
            ds_plot = ds_plot.sel({rows['names_var']: rows['values_var']})
    else:
        len_rows = 1
        rows = {'values_var': [None], 'names_plot': [None]}

    # Find extrema
    xmin, xmax, ymin, ymax = find_extrema(ds_plot, x_axis, y_axis, xmin, xmax, ymin, ymax)

    # Font parameters
    plt.rcParams['font.family'] = font
    plt.rcParams['font.size'] = fontsize
    text_kwargs ={'weight': 'bold'}


    fig, axes = plt.subplots(len_rows, len_cols, figsize=(1 + 6 * len_cols, len_rows * 4), constrained_layout=True)
    if hasattr(axes, 'flatten'):
        axes_flatten = axes.flatten()
    else:
        axes_flatten = [axes]

    # Main title
    if title is not None:
        fig.suptitle(title, fontsize=plt.rcParams['font.size'] + 2)

    for col_idx, col in enumerate(cols['values_var']):
        for row_idx, row in enumerate(rows['values_var']):
            idx = len_cols * row_idx + col_idx
            ax = axes_flatten[idx]

            ds_selection = copy.deepcopy(ds_plot)
            if col is not None and cols['names_var'] is not None:
                ds_selection = plot_selection(ds_selection, cols['names_var'], col)
            if row is not None and rows['names_var'] is not None:
                ds_selection = plot_selection(ds_selection, rows['names_var'], row)

            for y_var in y_axis['values_var']:
                data = plot_selection(ds_selection, y_axis['names_var'], y_var)
                ax.plot(data[x_axis['values_var']], data.values, color='lightgrey', alpha=0.8)

            ax.spines[['right', 'top']].set_visible(False)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            if percent:
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            # ax.set_axis_off()

            sbs = ax.get_subplotspec()
            if sbs.is_first_col():
                ax.set_ylabel(y_axis['name_axis'])
            if sbs.is_last_row():
                ax.set_xlabel(x_axis['name_axis'])

    # Headers
    add_headers(fig, col_headers=cols['names_plot'], row_headers=rows['names_plot'], row_pad=25, col_pad=5, **text_kwargs)

    plt.savefig(path_result, bbox_inches='tight')

