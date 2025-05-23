"""
    EHCLO Project
    Copyright (C) 2025  Blaise CALMEL (INRAE)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from plot_functions.plot_common import *

def boxplot(ds, x_axis, y_axis, path_result, references=None, cols=None, rows=None, ymin=None, ymax=None, vlines=None,
             title=None, percent=False, fontsize=14, font='sans-serif', blank_space=1, common_yaxes=True, strip=False):

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

    # if ymin is None:
    #     if 'values_flatten' in y_axis.keys():
    #         _temp_y_values = ds[y_axis['values_flatten']].to_array()
    #     else:
    #         _temp_y_values = ds.to_array()
    #     ymin_vline = _temp_y_values.min().item()
    #     # yquant = _temp_y_values.quantile(0.05) - 2 * _temp_y_values.std()
    #     # if yquant > ymin_vline:
    #     #   ymin_vline =  _temp_y_values.quantile(0.05)
    #     ymin = ymin_vline
    # else:
    #     ymin_vline = ymin
    # if ymax is None:
    #     if 'values_flatten' in y_axis.keys():
    #         _temp_y_values = ds[y_axis['values_flatten']] #.sel(time=getattr(ds, 'horizon2'), gid=rows['values_var']).to_array()
    #         # ds['IPSL-CM5A-MR_RCA4_ADAMONT_MORDOR-SD_deviation'].sel(time=getattr(ds, 'horizon2'), gid='K055001010').values
    #     else:
    #         _temp_y_values = ds.to_array()
    #     ymax_vline = _temp_y_values.max().item()
    #     # yquant = _temp_y_values.quantile(0.95) + 2 * _temp_y_values.std()
    #     # if yquant < ymax_vline:
    #     #   ymax_vline =  _temp_y_values.quantile(0.95)
    #     ymax = ymax_vline
    # else:
    #     ymax_vline = ymax

    # if y_axis['names_coord'] == 'indicator':
    #     list_of_sims = [subdict['values'] for subdict in y_axis['values_var'].values()]
    #     all_sims = set([i for j in list_of_sims for i in j])
    #     ymax = ds[all_sims].to_array().max()
    #     ymin = ds[all_sims].to_array().min()

    ignore = 0
    for key, value in y_axis['values_var'].items():
        if 'type' in value.keys():
            ignore += 1

    xmin = - blank_space / 2 - 0.5
    xmax = (len(x_axis['names_plot']) * (len(y_axis['names_plot']) - ignore) + 2 * blank_space *
            (len(x_axis['names_plot'])) - 2 * blank_space) - 0.5

    legend_items = []
    legend_labels = []
    # centers = [((len(y_axis['names_plot']) + 1) / 2) - 1 + 5 * j for j in range(len(x_axis['names_plot']))]
    init_center = (len(y_axis['names_plot']) - ignore - 1) / 2
    centers = [init_center + k * (2 * blank_space + len(y_axis['names_plot']) - ignore) for k in
               range(len(x_axis['names_plot']))]

    if vlines is not None:
        vlines = [(centers[i] + centers[i + 1]) / 2 for i in range(len(centers) - 1)]

    # Font parameters
    plt.rcParams['font.family'] = font
    plt.rcParams['font.size'] = fontsize

    # fig_dim = 4
    fig_dim = 3

    # extend_rows = 1 + fontsize * (1 + max(s.count('\n') for s in rows['names_plot'])) / 200

    # fig, axes = plt.subplots(len_rows, len_cols, figsize=(1 + 2.5 * fig_dim * len_cols, 1 + extend_rows * len_rows * fig_dim), constrained_layout=True)
    fig, axes = plt.subplots(len_rows, len_cols, figsize=(16, 1 + len_rows * fig_dim), constrained_layout=True)
    max_values = []
    min_values = []
    _y_values = []
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
        fig.suptitle(title, fontsize=plt.rcParams['font.size'], weight='bold')

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
            # y_temp_max = []
            # y_temp_min = []

            for name, y_var in y_axis['values_var'].items():
                if y_axis['names_coord'] == 'indicator':
                    name_sims = y_var['values']
                    cell_data = plot_selection(ds_selection, y_axis['names_coord'], name_sims)
                else:
                    cell_data = plot_selection(ds_selection, y_axis['names_coord'], name)

                if 'kwargs' in y_var.keys():
                    kwargs = y_var['kwargs']
                else:
                    kwargs = {}

                cell_boxplots = []
                for x_var in x_axis['values_var']:
                    if x_var is not None:
                        boxplot_values = plot_selection(ds_selection=cell_data, names_var=x_axis['names_coord'], value=x_var)
                    else:
                        boxplot_values = cell_data
                    data_list = []
                    for data_name in boxplot_values.data_vars:
                        data_list.append(boxplot_values[data_name].values.flatten())
                    all_values = np.concatenate(data_list).flatten()
                    mask = ~np.isnan(all_values)
                    if any(mask):
                        cell_boxplots.append(all_values[mask])
                    else:
                        cell_boxplots.append([np.nan])

                # Plot by sub box categories
                if 'type' in y_var.keys():
                    if y_var['type'] == 'histo':
                        for cell_idx, cell in enumerate(cell_boxplots):
                            bp = ax.fill_between(x=[centers[cell_idx] - (len(y_axis['names_plot']) - ignore) / 2 - 0.2,
                                                 centers[cell_idx] + (len(y_axis['names_plot']) - ignore) / 2 + 0.2],
                                                 y1=[0],
                                                 y2=[np.nanmedian(cell)], **kwargs)
                elif strip:
                    i += 1
                    current_position = [i + (len(y_axis['names_plot']) - ignore + 2 * blank_space) * j for j in
                                        range(len(x_axis['names_plot']))]
                    if len(current_position) > 1:
                        width = 0.5 * (current_position[1] - current_position[0])
                    else: 
                        width = 0.8
                    for cell_idx, cell in enumerate(cell_boxplots):
                        bp = ax.hlines(y=cell, xmin=[current_position[cell_idx] - width/2] * len(cell), 
                                       xmax=[current_position[cell_idx] + width/2] * len(cell), 
                                       **kwargs)
                else:
                    i += 1
                    current_position = [i + (len(y_axis['names_plot']) - ignore + 2 * blank_space) * j for j in
                                        range(len(x_axis['names_plot']))]
                    bp = ax.boxplot(cell_boxplots, positions=current_position, vert=True,
                                    whiskerprops=dict(linewidth=0.4),
                                    flierprops=dict(marker='.', markersize=4, markerfacecolor='k'), **kwargs)
                
                if references is not None:
                    if len(current_position) > 1:
                        width = 0.5 * (current_position[1] - current_position[0])
                    else:
                        width = 0.9
                    is_dict_of_dicts = (isinstance(references, dict) and all(isinstance(v, dict) for v in references.values()))
                    if is_dict_of_dicts:
                        if name in references.keys():
                            ref_data = references[name]
                            for ref, ref_args in ref_data.items():
                                # hline_values = ds_selection[ref].values
                                hline_values = plot_selection(ds_selection=cell_data, names_var=x_axis['names_coord'], value=x_axis['values_var'])
                                if not np.isnan(hline_values[ref]).all():
                                    if 'function' in ref_args.keys():
                                        str_to_func = {
                                                        "mean": np.nanmean,
                                                        "median": np.nanmedian,
                                                        "sum": np.nansum,
                                                        "min": np.nanmin,
                                                        "max": np.nanmax,
                                                    }
                                        hline_plot = str_to_func[ref_args['function']](hline_values[ref])
                                        ref_args = {key: value for key, value in ref_args.items() if key != "function"}
                                    else:
                                        hline_plot = hline_values[ref]

                                    ax.hlines(y=hline_plot.values, xmin=[val - width/2 for val in current_position], 
                                            xmax=[val + width/1.75 for val in current_position], **ref_args)
                                    ax.scatter(x=[val + width/1.75 for val in current_position], y=hline_plot.values, s=25, **ref_args)
                    else:
                        hline_values = plot_selection(ds_selection=cell_data, names_var=x_axis['names_coord'], value=x_axis['values_var'])
                        for ref, ref_args in references.items():
                            ax.hlines(y=hline_values[ref].values, xmin=[val - width/2 for val in current_position], 
                                        xmax=[val + width/1.75 for val in current_position], **ref_args)
                            ax.scatter(x=[val + width/1.75 for val in current_position], y=hline_values[ref].values, s=25, **ref_args)
                            # ax.hlines(y=hline_values[ref].values, xmin=[j+w/1.5 for j in current_position], xmax=[j+w+blank_space for j in current_position], **ref_args)
                            # ax.scatter(x=[j+w+blank_space for j in current_position], y=hline_values[ref].values, s=25, **ref_args)

                if any(mask):
                    _y_values.append(cell_boxplots)
                    # y_temp_max.append(np.nanmax(cell_boxplots))
                    # y_temp_min.append(np.nanmin(cell_boxplots))

                if 'label' in kwargs:
                    label = kwargs['label']
                else:
                    label = y_axis['names_plot'][i]

                if label not in legend_labels:
                    if isinstance(bp, dict) and "boxes" in bp.keys():
                        legend_items.append(bp["boxes"][0])
                    elif isinstance(bp, LineCollection):
                        handle = Line2D([0], [0], color=bp._original_edgecolor, linewidth=5, alpha=1)
                        legend_items.append(handle)
                    else:
                        legend_items.append(bp)
                        
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

            if vlines is not None and len(vlines) > 0:
                ax.axvline(x=vlines,
                        #   ymin=ymin_vline, ymax=ymax_vline,
                          color='lightgray', linewidth=2, alpha=0.6)

            # plt.rc('grid', linestyle="dashed", color='lightgray', linewidth=0.1, alpha=0.4)
            # ax.grid(True)
            ax.yaxis.grid(True, linestyle="--", color='lightgray', linewidth=0.1, alpha=0.4)

            ax.spines[['right', 'top']].set_visible(False)

            if x_axis['names_coord'] is None or ('names_plot' in x_axis.keys() and all(x is None for x in x_axis['names_plot'])):
                ax.set_xticks([])

            if percent:
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())

            if subplot_title:
                ax.set_title(subplot_title)

            # Headers and axes label
            add_header(ax, rows_plot, cols_plot, ylabel=y_title, xlabel=x_title)
            ax.set_xlim(xmin, xmax)
            # max_values.append(np.nanmax(y_temp_max))
            # min_values.append(np.nanmin(y_temp_min))

    _y_flatten = np.concatenate([arr for sublist in _y_values for arr in sublist])
    if ymin is None:
        ymin = np.nanmin(_y_flatten)
    if ymax is None:
        ymax = np.nanmax(_y_flatten)

    if common_yaxes:
        abs_max = max([ymax, -ymin])
        n = abs_max / 3
        exponent = round(math.log10(n))
        step = np.round(n, -exponent+1)
        if step == 0:
            step = n
        ticks = mirrored(abs_max, inc=step, val_center=0)
        for ax in axes_flatten:
            ax.set_yticks(ticks)
        
        for ax_idx, ax in enumerate(axes_flatten):
            ax.set_ylim(np.round(ymin, -exponent+1), np.round(ymax, -exponent+1))
            sbs = ax.get_subplotspec()
            if not sbs.is_first_col():
                ax.set_yticklabels([])
                # ax.set_yticks([])

    else:
        for ax_idx, ax in enumerate(axes_flatten):
            ax.set_ylim(min_values[ax_idx], max_values[ax_idx])
   
    if references is not None and not isinstance(references, dict):
        for key, value in references.items():
            handle = Line2D([0], [0], color=value['color'], linewidth=5, alpha=1)
            legend_items.append(handle)
            legend_labels.append(value['label'])
    
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
    # ncol = np.round(len(legend_items) / ncol) + 1

    fig.legend(np.array(legend_items)[imported_order], np.array(legend_labels)[imported_order], loc='upper center', bbox_to_anchor=(0.5, 0),
               fancybox=False, shadow=False, ncol=ncol)

    plt.savefig(path_result, bbox_inches='tight')