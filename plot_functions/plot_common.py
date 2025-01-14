import copy
import math
import matplotlib.pyplot as plt
import geopandas.geodataframe as gpd
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def init_grid(grid_dict, ds_plot):
    grid_dict_temp = copy.deepcopy(grid_dict)
    if grid_dict_temp is not None:
        if isinstance(grid_dict_temp, int):
            length_dict = grid_dict_temp
            grid_dict_temp = {'values_var': [None], 'names_plot': [None], 'names_coord': [None]}
        else:
            length_dict = len(grid_dict_temp['values_var'])
            if 'names_coord' in grid_dict_temp.keys():
                if grid_dict_temp['names_coord'] != 'indicator':
                    if grid_dict_temp['names_coord'] is not None:
                        ds_plot = ds_plot.sel({grid_dict_temp['names_coord']: grid_dict_temp['values_var']})
            else:
                grid_dict_temp |= {'names_plot': [None], 'names_coord': [None]}
    else:
        length_dict = 1
        grid_dict_temp = {'values_var': [None], 'names_plot': [None], 'names_coord': [None]}

    return length_dict, grid_dict_temp, ds_plot

def define_cbar(fig, axes_flatten, len_rows, len_cols, cmap, bounds_cmap,
                cbar_title=None, percent=False, **text_kwargs):
    # Scalar mappable
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.BoundaryNorm(bounds_cmap, cmap.N))

    if len_rows == 1:
        bottom = axes_flatten[-1].get_position().y0 - 0.05
        height = axes_flatten[-1].get_position().y1 - axes_flatten[-1].get_position().y0 + 0.025
    elif len_rows < 4:
        bottom = 0.15
        height = 0.7
    else:
        middle = int(len_rows * len_cols / 2)
        if len_rows % 2 == 0:
            center = (axes_flatten[middle-1].get_position().y0 + axes_flatten[middle].get_position().y1)/2
        else:
            center = (axes_flatten[middle].get_position().y0 + axes_flatten[middle].get_position().y1)/2
        size = axes_flatten[middle-1].get_position().y1 - axes_flatten[middle-1].get_position().y0 + 0.1
        bottom = center - size
        height = 2 * size

        # bottom = axes_flatten[middle].get_position().y0 - 0.2
        # height = axes_flatten[middle].get_position().y1 + 0.2

    cbar_ax = fig.add_axes([axes_flatten[len_cols - 1].get_position().x1 + 0.05, bottom, 0.02, height])

    sm._A = []
    if percent:
        cbar = fig.colorbar(sm, cax=cbar_ax, drawedges=True, ticks=bounds_cmap, format='%.0f')
    else:
        cbar = fig.colorbar(sm, cax=cbar_ax, drawedges=True, ticks=bounds_cmap)
    if cbar_title:
        label_ax = fig.add_axes([cbar_ax.get_position().x1 + 0.075, cbar_ax.get_position().y0, 0.02, height])
        label_ax.annotate(cbar_title, xy=(0, 0.45), wrap=True, **text_kwargs)
        label_ax.axis('off')
        # cbar.set_label(cbar_title, rotation=0, wrap=True, labelpad=25, **text_kwargs)

    return cbar

def add_header(ax, rows_plot, cols_plot, ylabel='', xlabel=''):
    sbs = ax.get_subplotspec()
    if sbs.is_first_col():
        name_row = None
        if sbs.rowspan.start < len(rows_plot['names_plot']):
            name_row = rows_plot['names_plot'][sbs.rowspan.start]
            if name_row is not None:
                name_row = name_row.replace(' ', '~')


        if name_row is None:
            ax.set_ylabel(f"{ylabel}")
        elif ylabel is None or len(ylabel) == 0:
            ax.set_ylabel(f"$\\bf{{{name_row}}}$")
        else:
            ax.set_ylabel(f"$\\bf{{{name_row}}}$ \n\n{ylabel}")

    if sbs.is_first_row():
        if sbs.colspan.start < len(cols_plot['names_plot']):
            name_col = cols_plot['names_plot'][sbs.colspan.start]
            if name_col is not None:
                ax.annotate(
                            name_col,
                            xy=(0.5, 1),
                            xytext=(0, 5),
                            xycoords="axes fraction",
                            textcoords="offset points",
                            ha="center",
                            va="baseline",
                            **{'weight': 'bold'},
                        )

    if sbs.is_last_row():
        if xlabel and len(xlabel) > 0:
            ax.set_xlabel(f"{xlabel}")


def find_extrema(ds_plot, x_axis, y_axis, indicator_plot, xmin, xmax, ymin, ymax):

    if xmin is None:
        try:
            x_min_temp = 0
            if x_axis['names_coord'] != 'indicator':
                x_min_temp = ds_plot.variables[x_axis['names_coord']].min().values
                xmin = np.nanmin(x_min_temp)
            else:
                var_names = [i for subdict in indicator_plot for i in subdict]
                xmin = math.ceil((ds_plot[var_names].to_array()).min())
        except ValueError:
            xmin = min(x_min_temp)
        except KeyError:
            xmin = None
    if xmax is None:
        try:
            x_max_temp = 0
            if x_axis['names_coord'] != 'indicator':
                x_max_temp = ds_plot.variables[x_axis['names_coord']].max().values
                xmax = np.nanmin(x_max_temp)
            else:
                var_names = [i for subdict in indicator_plot for i in subdict]
                xmax = math.ceil((ds_plot[var_names].to_array()).max())
        except ValueError:
            xmax = max(x_max_temp)
        except KeyError:
            xmax = None

    if ymin is None:
        try:
            y_min_temp = 0
            if y_axis['names_coord'] != 'indicator':
                y_min_temp = ds_plot.variables[y_axis['names_coord']].min().values
                ymin = np.nanmin(y_min_temp)
            else:
                var_names = [i for subdict in indicator_plot for i in subdict]
                ymin = math.ceil((ds_plot[var_names].to_array()).min())
        except ValueError:
            ymin = min(y_min_temp)
        except KeyError:
            ymin = None

    if ymax is None:
        try:
            y_max_temp = 0
            if y_axis['names_coord'] != 'indicator':
                y_max_temp = ds_plot.variables[y_axis['names_coord']].max().values
                ymax = np.nanmax(y_max_temp)
            else:
                var_names = [i for subdict in indicator_plot for i in subdict]
                ymax = math.ceil((ds_plot[var_names].to_array()).max())
        except ValueError:
            ymax = max(y_max_temp)
        except KeyError:
            ymax = None

    return xmin, xmax, ymin, ymax

def flatten_to_strings(input_list):
    result = []
    for item in input_list:
        if isinstance(item, list):
            result.extend(flatten_to_strings(item))
        else:
            result.append(item)
    return result

def plot_selection(ds_selection, names_var, value):
    if names_var == 'month':
        ds_selection = ds_selection.sel(time = ds_selection.time.dt.month == value)
    elif names_var == 'indicator':
        ds_selection = ds_selection[value]
    else:
        temp_dict = {names_var: value}
        ds_selection = ds_selection.sel(temp_dict)

    return ds_selection

def save_shp_figure(back_shp:gpd.GeoDataFrame, path_result:str, study_shp:gpd.GeoDataFrame=None,
                    rivers_shp:gpd.GeoDataFrame=None,
                    figsize:tuple=None, **kwargs):
    """

    :param current_shp:
    :param path_result:
    :param figsize:
    :param kwargs:
    :return:
    """
    if figsize is not None:
        figsize = figsize
    else:
        figsize = (18, 18)

    fig, ax = plt.subplots(figsize=figsize)
    back_shp.plot(ax=ax, figsize=figsize, color='gainsboro', edgecolor='black')

    if study_shp is not None:
        study_shp.plot(ax=ax, color='gainsboro', linewidth=2, edgecolor='firebrick', linestyle="--")
        bounds = study_shp.geometry.total_bounds
    else:
        bounds = back_shp.geometry.total_bounds

    if rivers_shp is not None:
        rivers_tresh = 0.5 * ((bounds[2] - bounds[0])**2 + (bounds[3] - bounds[1])**2)**0.5
        long_rivers_idx = rivers_shp.geometry.length > rivers_tresh

        long_rivers_shp = rivers_shp[long_rivers_idx]

        long_rivers_shp.plot(ax=ax, linewidth=2, color='royalblue')

    ax.set_xlim(bounds[0] - 5000, bounds[2] + 5000)
    ax.set_ylim(bounds[1] - 5000, bounds[3] + 5000)

    plt.savefig(path_result, **kwargs)
    plt.close()


def plot_timeline(df_station, station_name, path_result=None, figsize=(18, 18), selected_sim=None, name='red'):
    # Init plot
    fig, ax = plt.subplots(figsize=figsize, layout='compressed')

    for key, grp in df_station[df_station['sim'] != selected_sim].groupby(['sim']):
        ax.plot(grp['year'], grp[station_name], c='grey', linewidth=1, )

    selected_df = df_station[df_station['sim'] == selected_sim]
    ax.plot(grp['year'], grp[station_name], c='r', linewidth=1, label=name)

    ax.set_title(station_name)
    ax.legend()

    # Save
    if path_result is not None:
        plt.savefig(path_result, bbox_inches='tight')
