import copy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from plot_functions.plot_common import *

def boxplot(ds, x_axis, x2_axis, y_axis, path_result, cols=None, rows=None, ymin=None, ymax=None,
             title=None, percent=False, palette='BrBG', fontsize=14, font='sans-serif', ):

    ds_plot = copy.deepcopy(ds)
    len_cols, cols_plot, ds_plot = init_grid(cols, ds_plot)
    len_rows, rows_plot, ds_plot = init_grid(rows, ds_plot)

    if 'name_axis' in x_axis:
        x_title = x_axis['name_axis']
        del x_axis['name_axis']
    else:
        x_title = None
    if 'name_axis' in y_axis:
        y_title = y_axis['name_axis']
        del y_axis['name_axis']
    else:
        y_title = None

    # x_flatten = flatten_to_strings(x_axis.keys())
    # y_flatten = flatten_to_strings(y_axis.keys())
    # indicator_plot = variables['simulation_horizon_deviation_by_sims']

    ds_plot = ds_plot[y_axis.keys()]

    # Find extrema
    # xmin, xmax, ymin, ymax = find_extrema(ds_plot, x_axis, y_axis, xmin, xmax, ymin, ymax)

    plt.rcParams['font.family'] = font
    plt.rcParams['font.size'] = fontsize
    text_kwargs ={'weight': 'bold'}

    cmap = plt.get_cmap(palette)
    colors = cmap(np.linspace(0, 1, len(x2_axis['values_var'])))

    legend_items = []
    fig, axes = plt.subplots(len_rows, len_cols, figsize=(1 + 6 * len_cols, len_rows * 4), constrained_layout=True)
    if hasattr(axes, 'flatten'):
        axes_flatten = axes.flatten()
    else:
        axes_flatten = [axes]
    # Main title
    if title is not None:
        fig.suptitle(title, fontsize=plt.rcParams['font.size'] + 2)

    for col_idx, col in enumerate(cols_plot['values_var']):
        for row_idx, row in enumerate(rows_plot['values_var']):
            idx = len_cols * row_idx + col_idx
            ax = axes_flatten[idx]

            ds_selection = copy.deepcopy(ds_plot)
            if col is not None and cols['names_var'] is not None:
                ds_selection = plot_selection(ds_selection, cols['names_var'], col)
            if row is not None and rows['names_var'] is not None:
                ds_selection = plot_selection(ds_selection, rows['names_var'], row)

            position_main = np.arange(len(x_axis['values_var'])) * 2
            j = -1
            i = -(len(x2_axis['values_var'])/2) - 0.5
            for x2_var in x2_axis['values_var']:
                cell_boxplots = []
                cell_data = plot_selection(ds_selection, x2_axis['names_coord'], x2_var)

                j += 1
                for x_var in x_axis['values_var']:
                    boxplot_values = plot_selection(cell_data, x_axis['names_coord'], x_var)
                    data_list = []
                    for data_name in boxplot_values.data_vars:
                        data_list.append(boxplot_values[data_name].values.flatten())
                    all_values = np.concatenate(data_list).flatten()
                    mask = ~np.isnan(all_values)
                    cell_boxplots.append(all_values[mask])
                i += 1
                x_position = position_main + 0.35 * i

                # Plot by sub box categories
                bp = ax.boxplot(cell_boxplots, positions=x_position, widths=0.3, patch_artist=True,
                                boxprops=dict(facecolor=colors[j], alpha=0.3), medianprops=dict(color="black"))

                if bp["boxes"][0] not in legend_items:
                    legend_items.append(bp["boxes"][0])

            # Set ticks
            ax.set_xticks(position_main)
            ax.set_xticklabels(x_axis['names_plot'])

            ax.spines[['right', 'top']].set_visible(False)

            if percent:
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())

            # Headers and axes label
            add_header(ax, rows_plot, cols_plot, ylabel=y_title, xlabel=x_title)

    plt.legend(legend_items, x2_axis['names_plot'], loc='upper left', bbox_to_anchor=(1, 1), fancybox=False, shadow=False,
               ncol=1)

    plt.savefig(path_result, bbox_inches='tight')

