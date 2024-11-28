import matplotlib.ticker as mtick
from matplotlib.lines import Line2D

from plot_functions.plot_common import *

def lineplot(ds, x_axis, y_axis, path_result, cols, rows, vlines=None, xmin=None, xmax=None, ymin=None, ymax=None,
             title=None, percent=True, fontsize=14, font='sans-serif', plot_type='line'):

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

    x_flatten = flatten_to_strings(x_axis.keys())
    y_flatten = flatten_to_strings(y_axis.keys())
    ds_plot = ds_plot[x_flatten + y_flatten]

    # Find extrema
    xmin, xmax, ymin, ymax = find_extrema(ds_plot, x_axis, y_axis, xmin, xmax, ymin, ymax)

    # Font parameters
    plt.rcParams['font.family'] = font
    plt.rcParams['font.size'] = fontsize

    fig, axes = plt.subplots(len_rows, len_cols, figsize=(1 + 6 * len_cols, len_rows * 4), constrained_layout=True)
    if hasattr(axes, 'flatten'):
        axes_flatten = axes.flatten()
    else:
        axes_flatten = [axes]

    # Main title
    if title is not None:
        fig.suptitle(title, fontsize=plt.rcParams['font.size'] + 2)

    legend_items = []
    for col_idx, col in enumerate(cols_plot['values_var']):
        for row_idx, row in enumerate(rows_plot['values_var']):
            idx = len_cols * row_idx + col_idx
            ax = axes_flatten[idx]

            temp_dict = {}
            if cols_plot['names_coord'] is not None and col is not None:
                temp_dict |= {cols_plot['names_coord']: col}
            if rows_plot['names_coord'] is not None and row is not None:
                temp_dict |= {rows_plot['names_coord']: row}

            ds_selection = ds_plot.sel(temp_dict)

            for x_var, x_values in x_axis.items():
                for y_var, y_values in y_axis.items():
                    print(y_var)
                    ds_selection = ds_selection.sortby(x_var)
                    valid = np.logical_not(np.isnan(ds_selection[x_var].values) |
                                           np.isnan(ds_selection[y_var].values))
                    if plot_type == 'line':
                        ax.plot(ds_selection[x_var].values[valid], ds_selection[y_var].values[valid], **x_values, **y_values)
                    else:
                        ax.scatter(ds_selection[x_var].values[valid], ds_selection[y_var].values[valid], **x_values, **y_values)

                    if y_values not in legend_items:
                        legend_items.append(y_values)

                if vlines is not None:
                    ax.vlines(x=vlines[x_var], ymin=ymin, ymax=ymax - 0.1*(ymax-ymin), color="k", linewidth=1)

                    for i in range(len(vlines)):
                        ax.text(vlines.iloc[i][x_var], ymax - 0.02*(ymax-ymin), vlines.iloc[i]['tag'], rotation=90,
                                verticalalignment='top')

            ax.spines[['right', 'top']].set_visible(False)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            if percent:
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())

            plt.rc('grid', linestyle="dashed", color='lightgray')
            ax.grid(True)

            # Headers and axes label
            add_header(ax, rows_plot, cols_plot, ylabel=y_title, xlabel=x_title)

    # Legend
    legend_handles = []
    for item in legend_items:
        handle = Line2D(
            [0], [0],  # Ligne fictive
            color=item.get('color', 'k'),
            linestyle=item.get('linestyle', '-'),
            alpha=item.get('alpha', 1),
            label=item['label']
        )
        legend_handles.append(handle)

    plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=False, shadow=False,
               ncol=len(legend_handles))

    plt.savefig(path_result, bbox_inches='tight')

