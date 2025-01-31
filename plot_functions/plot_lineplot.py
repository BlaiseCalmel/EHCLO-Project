import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
import textwrap
from plot_functions.plot_common import *

def lineplot(ds, indicator_plot, x_axis, y_axis, path_result, cols, rows, vlines=None, legend_items= None,
             xmin=None, xmax=None, ymin=None, ymax=None,
             title=None, percent=True, fontsize=14, font='sans-serif'):

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

    # Find extrema
    xmin, xmax, ymin, ymax = find_extrema(ds_plot, x_axis, y_axis, indicator_plot, xmin, xmax, ymin, ymax)

    # Font parameters
    plt.rcParams['font.family'] = font
    plt.rcParams['font.size'] = fontsize

    fig, axes = plt.subplots(len_rows, len_cols, figsize=(1 + 8 * len_cols, len_rows * 4), constrained_layout=True)
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
        fig.suptitle(title, fontsize=plt.rcParams['font.size'] + 2)

    if legend_items is None:
        legend_items = []

    idx = -1
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
            ds_selection = ds_selection.sortby(x_axis['names_coord'])

            for key, value in current_indicator.items():
                valid = ((np.logical_not(np.isnan(ds_selection[key].values))) &
                         (np.logical_not(np.isnan(ds_selection[x_axis['names_coord']].values))))

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
            ax.plot([xmin, xmax], [0, 0], color='k', linestyle='--', linewidth=0.5, dashes=(10,10),
                    zorder=1000)

            if subplot_title:
                ax.set_title(subplot_title)

            ax.spines[['right', 'top']].set_visible(False)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            if percent:
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())

            plt.rc('grid', linestyle="dashed", color='lightgray', linewidth=0.1, alpha=0.4)
            ax.grid(True)

            # Headers and axes label
            add_header(ax, rows_plot, cols_plot, ylabel=y_title, xlabel=x_title)

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
    avg_label_length = np.median([len(handle.get_label()) for handle in legend_handles ])  # Longueur moyenne des labels

    # Déterminer dynamiquement le nombre de colonnes
    ncol = max(1, int(fig_width * 4 / avg_label_length))

    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0), fancybox=True, shadow=False,
                   ncol=ncol)

    plt.savefig(path_result, bbox_inches='tight')

