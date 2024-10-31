import copy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from plot_functions.plot_common import *

def boxplot(ds, x_axis, y_axis, path_result, cols=None, rows=None, ymin=None, ymax=None,
             title=None, percent=False, palette='BrBG', fontsize=14, font='sans-serif', ):

    ds_plot = copy.deepcopy(ds)
    len_cols, cols, ds_plot = init_grid(cols, ds_plot)
    len_rows, rows, ds_plot = init_grid(rows, ds_plot)
    # if cols is not None:
    #     len_cols = len(cols['values_var'])
    #     if cols['names_var'] != 'indicator':
    #         ds_plot = ds_plot.sel({cols['names_var']: cols['values_var']})
    # else:
    #     len_cols = 1
    #     cols = {'values_var': [None], 'names_plot': [None]}
    #
    # if rows is not None:
    #     len_rows = len(rows['values_var'])
    #     if rows['names_var'] != 'indicator':
    #         ds_plot = ds_plot.sel({rows['names_var']: rows['values_var']})
    # else:
    #     len_rows = 1
    #     rows = {'values_var': [None], 'names_plot': [None]}

    # Find extrema
    # _, _, ymin, ymax = find_extrema(ds_plot, x_axis, y_axis, 0, 0, ymin, ymax)

    plt.rcParams['font.family'] = font
    plt.rcParams['font.size'] = fontsize
    text_kwargs ={'weight': 'bold'}

    cmap = plt.get_cmap(palette)
    colors = cmap(np.linspace(0, 1, len(y_axis['values_var'])))

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

            position_main = np.arange(len(x_axis['values_var'])) * 2
            bp_legend= []
            j = -1
            i = -(len(y_axis['values_var'])/2) - 0.5
            for y_var in y_axis['values_var']:
                cell_boxplots = []
                cell_data = plot_selection(ds_selection, y_axis['names_var'], y_var)
                j += 1
                for x_var in x_axis['values_var']:
                    # cell_data = ds_selection[y_var]
                    boxplot_values = plot_selection(cell_data, x_axis['names_var'], x_var).values
                    mask = ~np.isnan(boxplot_values)
                    cell_boxplots.append(boxplot_values[mask])
                i += 1
                x_position = position_main + 0.35 * i


                # Plot by sub box categories
                bp = ax.boxplot(cell_boxplots, positions=x_position, widths=0.3, patch_artist=True,
                                boxprops=dict(facecolor=colors[j], alpha=0.3), medianprops=dict(color="black"))
                bp_legend.append(bp["boxes"][0])

            # Set ticks
            ax.set_xticks(position_main)
            ax.set_xticklabels(x_axis['names_plot'])
            ax.legend(bp_legend, y_axis['names_plot'], loc="upper right")

            # ax.set_ylim(ymin, ymax)
            ax.spines[['right', 'top']].set_visible(False)

            if percent:
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())

            sbs = ax.get_subplotspec()
            if sbs.is_first_col():
                ax.set_ylabel(y_axis['name_axis'])
            if sbs.is_last_row():
                ax.set_xlabel(x_axis['name_axis'])


    # Headers
    add_headers(fig, col_headers=cols['names_plot'], row_headers=rows['names_plot'], row_pad=35, col_pad=5, **text_kwargs)

    plt.savefig(path_result, bbox_inches='tight')

