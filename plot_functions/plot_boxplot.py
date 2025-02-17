import copy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from plot_functions.plot_common import *

def boxplot(ds, x_axis, y_axis, path_result, cols=None, rows=None, ymin=None, ymax=None, vlines=None,
             title=None, percent=False, fontsize=14, font='sans-serif', blank_space=1, common_yaxes=True):

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

    if ymin is None:
        # ymin = ds.to_array().quantile(0.01).item()
        ymin = ds.to_array().min().item()
    if ymax is None:
        # ymax = ds.to_array().quantile(0.99).item()
        ymax = ds.to_array().max().item()

    # if y_axis['names_coord'] == 'indicator':
    #     list_of_sims = [subdict['values'] for subdict in y_axis['values_var'].values()]
    #     all_sims = set([i for j in list_of_sims for i in j])
    #     ymax = ds[all_sims].to_array().max()
    #     ymin = ds[all_sims].to_array().min()

    xmin = - blank_space / 2 - 1
    xmax = (len(x_axis['names_plot']) * len(y_axis['names_plot']) + 2 * blank_space * (len(x_axis['names_plot'])) -
            2 * blank_space)

    legend_items = []
    legend_labels = []
    # centers = [((len(y_axis['names_plot']) + 1) / 2) - 1 + 5 * j for j in range(len(x_axis['names_plot']))]
    init_center = (len(y_axis['names_plot']) - 1) / 2
    centers = [init_center + k * (2 * blank_space + len(y_axis['names_plot'])) for k in
               range(len(x_axis['names_plot']))]

    if vlines is not None:
        vlines = [(centers[i] + centers[i + 1]) / 2 for i in range(len(centers) - 1)]

    # Font parameters
    plt.rcParams['font.family'] = font
    plt.rcParams['font.size'] = fontsize

    # fig_dim = 4
    fig_dim = 3

    fig, axes = plt.subplots(len_rows, len_cols, figsize=(1 + 2.5 * fig_dim * len_cols, 1 + len_rows * fig_dim), constrained_layout=True)
    max_values = []
    min_values = []
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
        fig.suptitle(title)

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
            # for y_idx, y_values in enumerate(y_axis['values_var']):
            #     print(y_values)
            #     for name, value in y_values.items():
            #         print(name)
            y_temp_max = []
            y_temp_min = []
            for name, y_var in y_axis['values_var'].items():
                if y_axis['names_coord'] == 'indicator':
                    name_sims = y_var['values']
                    cell_data = plot_selection(ds_selection, y_axis['names_coord'], name_sims)
                else:
                    cell_data = plot_selection(ds_selection, y_axis['names_coord'], name)

                if 'kwargs' in  y_var.keys():
                    kwargs = y_var['kwargs']
                else:
                    kwargs = {}

                cell_boxplots = []
                for x_var in x_axis['values_var']:
                    boxplot_values = plot_selection(ds_selection=cell_data, names_var=x_axis['names_coord'], value=x_var)
                    data_list = []
                    for data_name in boxplot_values.data_vars:
                        data_list.append(boxplot_values[data_name].values.flatten())
                    all_values = np.concatenate(data_list).flatten()
                    mask = ~np.isnan(all_values)
                    if any(mask):
                        cell_boxplots.append(all_values[mask])
                    else:
                        cell_boxplots.append([np.nan])
                i += 1
                current_position = [i + (len(y_axis['names_plot']) + 2 * blank_space) * j for j in
                                    range(len(x_axis['names_plot']))]

                # Plot by sub box categories
                bp = ax.boxplot(cell_boxplots, positions=current_position, vert=True,
                                whiskerprops=dict(linewidth=0.4), **kwargs)
                y_temp_max.append(np.nanmax(cell_boxplots))
                y_temp_min.append(np.nanmin(cell_boxplots))

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
                          color='lightgray', linewidth=2, alpha=0.6)

            # plt.rc('grid', linestyle="dashed", color='lightgray', linewidth=0.1, alpha=0.4)
            # ax.grid(True)
            ax.yaxis.grid(True, linestyle="--", color='lightgray', linewidth=0.1, alpha=0.4)

            ax.spines[['right', 'top']].set_visible(False)

            if percent:
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())

            if subplot_title:
                ax.set_title(subplot_title)

            # Headers and axes label
            add_header(ax, rows_plot, cols_plot, ylabel=y_title, xlabel=x_title)
            ax.set_xlim(xmin, xmax)
            max_values.append(np.nanmax(y_temp_max))
            min_values.append(np.nanmin(y_temp_min))

    if ymin is None:
        ymin = np.nanmin(min_values)
    if ymax is None:
        ymax = np.nanmax(max_values)

    abs_max = max([ymax, -ymin])
    n = 2*abs_max / 4
    exponent = round(math.log10(n))
    step = np.round(n, -exponent+1)
    if step == 0:
        step = n
    ticks = mirrored(abs_max, inc=step, val_center=0)
    for ax in axes_flatten:
        ax.set_yticks(ticks)

    if common_yaxes:
        for ax in axes_flatten:
            ax.set_ylim(ymin, ymax)
    else:
        for ax_idx, ax in enumerate(axes_flatten):
            ax.set_ylim(min_values[ax_idx], max_values[ax_idx])

    imported_labels = [y_axis['values_var'][i]['kwargs']['label'] for i in y_axis['names_plot']]
    if set(imported_labels) == set(legend_labels):
        imported_order = [imported_labels.index(x) for x in legend_labels]
    else:
        imported_order = np.arange(len(legend_labels))

    # Estimate necessary width for each legend's column
    fig_width = fig.get_size_inches()[0]
    avg_label_length = np.median([len(l) for l in legend_labels])  # Longueur moyenne des labels

    # Déterminer dynamiquement le nombre de colonnes
    ncol = max(1, int(fig_width * 5 / avg_label_length))

    fig.legend(np.array(legend_items)[imported_order], np.array(legend_labels)[imported_order], loc='upper center', bbox_to_anchor=(0.5, 0),
               fancybox=False, shadow=False, ncol=ncol)

    plt.savefig(path_result, bbox_inches='tight')

