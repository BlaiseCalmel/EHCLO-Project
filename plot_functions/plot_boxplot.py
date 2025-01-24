import copy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from plot_functions.plot_common import *

def boxplot(ds, x_axis, y_axis, path_result, cols=None, rows=None, ymin=None, ymax=None, vlines=None,
             title=None, percent=False, fontsize=14, font='sans-serif', blank_space=1):

    ds_plot = copy.deepcopy(ds)
    len_cols, cols_plot, ds_plot = init_grid(cols, ds_plot)
    len_rows, rows_plot, ds_plot = init_grid(rows, ds_plot)

    subplot_titles = None
    del_axes = None
    if isinstance(rows, int):
        len_cols, mod = divmod(len_cols, len_rows)
        if mod > 0:
            len_cols += 1
            del_axes = len_rows - mod
        subplot_titles = cols['names_plot']
        cols_plot['names_plot'] = [None]
    if isinstance(cols, int):
        len_rows, mod = divmod(len_rows, len_cols)
        if mod > 0:
            len_rows += 1
            del_axes = len_cols - mod
        subplot_titles = rows['names_plot']
        rows_plot['names_plot'] = [None]
    # if isinstance(rows, int):
    #     len_cols = int(len_cols / len_rows)
    #     subplot_titles = cols['names_plot']
    #     cols_plot['names_plot'] = [None]
    # if isinstance(cols, int):
    #     len_rows = int(len_rows / len_cols)
    #     subplot_titles = rows['names_plot']
    #     rows_plot['names_plot'] = [None]

    if 'name_axis' in x_axis:
        x_title = x_axis['name_axis']
    else:
        x_title = None
    if 'name_axis' in y_axis:
        y_title = y_axis['name_axis']
    else:
        y_title = None

    list_of_sims = [subdict['values'] for subdict in y_axis['values_var'][0].values()]
    all_sims = set([i for j in list_of_sims for i in j])
    ymax = ds[all_sims].to_array().max()
    ymin = ds[all_sims].to_array().min()
    xmin = - blank_space / 2 - 1
    xmax = (len(x_axis['names_plot']) * len(y_axis['names_plot']) + 2 * blank_space * (len(x_axis['names_plot'])) -
            2 * blank_space)
    # -3-2-1 0123 456
    # 789 10111213 141516
    # 171819 202122223 242526

    legend_items = []
    legend_labels = []
    # centers = [((len(y_axis['names_plot']) + 1) / 2) - 1 + 5 * j for j in range(len(x_axis['names_plot']))]
    init_center = (len(y_axis['names_plot']) - 1) / 2
    centers = [init_center + k * (2 * blank_space + len(y_axis['names_plot'])) for k in
               range(len(x_axis['names_plot']))]

    if vlines:
        vlines = [(centers[i] + centers[i + 1]) / 2 for i in range(len(centers) - 1)]

    # Font parameters
    plt.rcParams['font.family'] = font
    plt.rcParams['font.size'] = fontsize

    fig, axes = plt.subplots(len_rows, len_cols, figsize=(1 + 10 * len_cols, len_rows * 4), constrained_layout=True)
    if del_axes:
        for i in range(del_axes):
            fig.delaxes(fig.axes[-1])
            axes = fig.axes

    if hasattr(axes, 'flatten'):
        axes_flatten = axes.flatten()
    elif isinstance(axes, list):
        axes_flatten = axes
    else:
        axes_flatten = [axes]

    # Main title
    if title is not None:
        fig.suptitle(title, fontsize=plt.rcParams['font.size'] + 2)

    idx = -1
    for row_idx, row in enumerate(rows_plot['values_var']):
        for col_idx, col in enumerate(cols_plot['values_var']):
            # idx = len_cols * row_idx + col_idx
            idx += 1
            ax = axes_flatten[idx]

            temp_dict = {}
            if cols_plot['names_coord'] is not None and col is not None and cols_plot['names_coord'] != 'indicator':
                temp_dict |= {cols_plot['names_coord']: col}
            if rows_plot['names_coord'] is not None and row is not None and rows_plot['names_coord'] != 'indicator':
                temp_dict |= {rows_plot['names_coord']: row}

            # Select station
            ds_selection = ds_plot.sel(temp_dict)

            subplot_title = None
            if subplot_titles:
                subplot_title = subplot_titles[idx]

            i = -1
            for y_idx, y_values in enumerate(y_axis['values_var']):
                for name, value in y_values.items():
                    name_sims = value['values']
                    kwargs = value['kwargs']
                    cell_data = plot_selection(ds_selection, y_axis['names_coord'], name_sims)

                    cell_boxplots = []
                    for x_var in x_axis['values_var']:
                        boxplot_values = plot_selection(cell_data, x_axis['names_coord'], x_var)
                        data_list = []
                        for data_name in boxplot_values.data_vars:
                            data_list.append(boxplot_values[data_name].values.flatten())
                        all_values = np.concatenate(data_list).flatten()
                        mask = ~np.isnan(all_values)
                        cell_boxplots.append(all_values[mask])
                    i += 1
                    current_position = [i + (len(y_axis['names_plot']) + 2 * blank_space) * j for j in
                                        range(len(x_axis['names_plot']))]

                    # Plot by sub box categories
                    bp = ax.boxplot(cell_boxplots, positions=current_position,
                                    **kwargs)

                    if 'label' in kwargs:
                        label = kwargs['label']
                    else:
                        label = y_axis['names_plot'][i]


                    if label not in legend_labels:
                        legend_items.append(bp["boxes"][0])
                        legend_labels.append(label)
                        # if 'label' in kwargs:
                        #     legend_labels.append(kwargs['label'])
                        # else:
                        #     legend_labels.append(y_axis['names_plot'][i])

            # Set ticks
            ax.set_xticks(centers)
            ax.set_xticklabels(x_axis['names_plot'])

            ax.plot([xmin, xmax], [0, 0], color='k', linestyle='--', linewidth=0.5, dashes=(10,10),
                    zorder=1000)

            if vlines is not None:
                ax.vlines(x=vlines,
                          ymin=ymin, ymax=ymax,
                          color='lightgray', linewidth=1, alpha=0.5)

            # plt.rc('grid', linestyle="dashed", color='lightgray', linewidth=0.1, alpha=0.4)
            # ax.grid(True)

            ax.spines[['right', 'top']].set_visible(False)

            if percent:
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())

            if subplot_title:
                ax.set_title(subplot_title)

            # Headers and axes label
            add_header(ax, rows_plot, cols_plot, ylabel=y_title, xlabel=x_title)

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

    fig.legend(legend_items, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0), fancybox=False, shadow=False,
               ncol=2, fontsize=fontsize-2)

    plt.savefig(path_result, bbox_inches='tight')

