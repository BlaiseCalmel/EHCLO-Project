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
from plot_functions.plot_common import *

def lineplot(ds, indicator_plot, x_axis, y_axis, path_result, cols, rows, vlines=None, legend_items= None,
             xmin=None, xmax=None, ymin=None, ymax=None, common_axes=True, remove_xticks=True,
             title=None, percent=True, fontsize=14, font='sans-serif', references=None):

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
    # ds_plot = ds_plot[x_flatten + y_flatten]

    if 'names_sorted' in x_axis:
        ds_plot = ds_plot.sel(month=sorted(ds_plot.month.values, key=lambda m: x_axis['names_sorted'].index(m)))

    # Find extrema
    xmin, xmax, ymin, ymax = find_extrema(ds_plot, x_axis, y_axis, indicator_plot, xmin, xmax, ymin, ymax)

    # Font parameters
    plt.rcParams['font.family'] = font
    plt.rcParams['font.size'] = fontsize

    # fig_dim = 4
    fig_dim = 3
    # fig, axes = plt.subplots(len_rows, len_cols, figsize=(1 + 2.5 * fig_dim * len_cols, 1 + len_rows * fig_dim), constrained_layout=True)
    fig, axes = plt.subplots(len_rows, len_cols, figsize=(16, 1 + len_rows * fig_dim), constrained_layout=True)
    # fig, axes = plt.subplots(len_rows, len_cols, figsize=(19, 6), constrained_layout=True)

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

    if legend_items is None:
        legend_items = []

    idx = -1

    key = 'HadGEM2-ES_CCLM4-8-17_ADAMONT_MORDOR-TS_by-horizon'
    for row_idx, row in enumerate(rows_plot['values_var']):
        for col_idx, col in enumerate(cols_plot['values_var']):
            # idx = len_cols * row_idx + col_idx
            idx += 1
            ax = axes_flatten[idx]

            # Select data
            if cols_plot['names_coord'] == 'indicator':
                current_indicator = col
            elif rows_plot['names_coord'] == 'indicator':
                current_indicator = row
            elif isinstance(indicator_plot, list):
                current_indicator = indicator_plot[idx]
            else:
                current_indicator = indicator_plot

            # Sublpot title
            subplot_title = None
            if subplot_titles:
                subplot_title = subplot_titles[idx]

            temp_dict = {}
            if cols_plot['names_coord'] is not None and col is not None and cols_plot['names_coord'] != 'indicator':
                temp_dict |= {cols_plot['names_coord']: col}
            if rows_plot['names_coord'] is not None and row is not None and rows_plot['names_coord'] != 'indicator':
                temp_dict |= {rows_plot['names_coord']: row}

            ds_selection = ds_plot.sel(temp_dict)
            if 'names_sorted' in x_axis.keys():
                ds_selection.sel(**{f"{x_axis['names_coord']}": x_axis['names_sorted']})
            else:
                ds_selection = ds_selection.sortby(x_axis['names_coord'])

            for key, value in current_indicator.items():
                if np.issubdtype(ds_selection[key].values.dtype, np.floating):
                    valid_1 = (np.logical_not(np.isnan(ds_selection[key].values))) 
                else:
                    valid_1 = np.full(ds_selection[key].values.shape, True, dtype=bool)
                if np.issubdtype(ds_selection[x_axis['names_coord']].values.dtype, np.floating):
                    valid_2 = (np.logical_not(np.isnan(ds_selection[x_axis['names_coord']].values)))
                else:
                    valid_2 = np.full(ds_selection[x_axis['names_coord']].values.shape, True, dtype=bool)
                
                valid = valid_1 & valid_2                
                
                # if 'names_plot' in x_axis.keys():
                #     ax.plot([v for v, m in zip(x_axis['names_plot'], valid) if m], ds_selection[key].values[valid], **value)
                # else:
                ax.plot(ds_selection[x_axis['names_coord']].values[valid], ds_selection[key].values[valid], **value)

                if value not in legend_items:
                    legend_items.append(value)

            # Vertical lines
            if vlines is not None:
                valid = np.logical_not(np.isnan(vlines[x_axis['names_coord']]))
                vlines = vlines[valid]

                if 'color' not in vlines.columns:
                    vlines['color'] = 'k'
                if 'ymin' not in vlines.columns:
                    vlines['ymin'] = ymin
                if 'ymax' not in vlines.columns:
                    vlines['ymax'] = ymax - 0.1*(ymax-ymin)
                if 'annotate' not in vlines.columns:
                    vlines['annotate'] = 0.02
                if 'alpha' not in vlines.columns:
                    vlines['alpha'] = 1
                if 'fontsize' not in vlines.columns:
                    vlines['fontsize'] = fontsize

                for i in range(len(vlines)):
                    ax.vlines(x=vlines.iloc[i][x_axis['names_coord']],
                              ymin=vlines.iloc[i]['ymin'], ymax=vlines.iloc[i]['ymax'],
                              color=vlines.iloc[i]['color'], linewidth=1)

                    ax.text(vlines.iloc[i][x_axis['names_coord']], ymax - vlines.iloc[i]['annotate']*(ymax-ymin),
                            vlines.iloc[i]['label'], rotation=90, verticalalignment='top', horizontalalignment='center',
                            alpha=vlines.iloc[i]['alpha'], fontsize=vlines.iloc[i]['fontsize'])

            # Plot data as line
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, dashes=(10,10),
                    zorder=1000)
            
            if references is not None:
                ref_args = {'color': 'k', 'alpha': 1, 'zorder': 20, 'label': 'Référence', 'linestyle': ':', 'linewidth': 2}
                for var in references.data_vars:
                    ax.plot(references[x_axis['names_coord']].values, references[var].sel(temp_dict).values,
                    **ref_args)
                
                if ref_args not in legend_items:
                    legend_items.append(value)

            if subplot_title:
                ax.set_title(subplot_title)

            ax.spines[['right', 'top']].set_visible(False)

            if percent:
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())

            # Headers and axes label
            add_header(ax, rows_plot, cols_plot, ylabel=y_title, xlabel=x_title, del_axes=del_axes)

    if common_axes:
        abs_max = max([ymax, -ymin])
        # n = 2*abs_max / 4
        n = (ymax - ymin) / 4
        exponent = round(math.log10(n))
        step = np.round(n, -exponent+1)
        if step == 0:
            step = n
        ticks = mirrored(abs_max, inc=step, val_center=0)
        
        for ax in axes_flatten:
            ax.set_yticks(ticks)
            ax.set_ylim(ymin, ymax)

            sbs = ax.get_subplotspec()
            if not sbs.is_first_col():
                ax.set_yticklabels([])
    else:
        for ax in axes_flatten:
            ax_ymin, ax_ymax = ax.get_ylim()
            abs_max = max([ymax, -ymin])
            n = (ax_ymax - ax_ymin) / 4
            exponent = round(math.log10(n))
            step = np.round(n, -exponent+1)
            if step == 0:
                step = n
            ticks = mirrored(abs_max, inc=step, val_center=0)
            ax.set_yticks(ticks)
            ax.set_ylim(ax_ymin, ax_ymax)
    
    for ax in axes_flatten:
        ax.grid(True, linestyle="--", color='lightgray', linewidth=0.1, alpha=0.4)
        sbs = ax.get_subplotspec()
        if not sbs.is_last_row():
            remove_x = True
        else: 
            remove_x = False
        
        if del_axes:
            n_rows, n_cols = sbs.get_geometry()[:2]
            c_row, c_col = sbs.rowspan.stop, sbs.colspan.stop
            if c_row == n_rows - 1 and c_col > n_cols-del_axes:
                remove_x = False

        if remove_x and remove_xticks:
            ax.set_xticklabels([])
        elif 'names_plot' in x_axis:
            if isinstance(x_axis['names_plot'][0], str):
                ax.set_xticks([int(i) for i in ds_selection[x_axis['names_coord']].values])
                ax.set_xticklabels(x_axis['names_plot'])
            elif np.issubdtype(ds[x_axis['names_coord']].values.dtype, np.datetime64):
                n_labels = 8
                step = len(ds[x_axis['names_coord']].values) // (n_labels - 1)
                indices = list(range(0, len(ds[x_axis['names_coord']].values), step))
                selected_dates = [ds[x_axis['names_coord']].values[i] for i in indices]
                selected_labels = [x_axis['names_plot'][i] for i in indices]
                ax.set_xticks(selected_dates)
                ax.set_xticklabels(selected_labels)
        
        ax.set_xlim(xmin, xmax)

    # Legend
    legend_handles = []
    for item in legend_items:
        # label_length = 28
        # wrapper = textwrap.TextWrapper(width=label_length, break_long_words=True, break_on_hyphens=True)
        # wrapped_label = wrapper.wrap(item['label'])

        handle = Line2D(
            [0], [0],  # Ligne fictive
            color=item.get('color', 'k'),
            linestyle=item.get('linestyle', '-'),
            linewidth=5,
            alpha=item.get('alpha', 1),
            label=item['label']
        )
        legend_handles.append(handle)

    # Estimate necessary width for each legend's column
    fig_width = fig.get_size_inches()[0]
    avg_label_length = np.median([len(handle.get_label()) for handle in legend_handles])

    # Déterminer dynamiquement le nombre de colonnes
    ncol = max(1, int(fig_width * 5 / avg_label_length))

    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0), fancybox=True, shadow=False,
               ncol=ncol)

    plt.savefig(path_result, bbox_inches='tight')

