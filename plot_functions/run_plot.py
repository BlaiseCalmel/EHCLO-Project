from plot_functions.plot_map import *
from plot_functions.plot_lineplot import *
from plot_functions.plot_boxplot import *

def plot_linear_pk(ds, name, simulations, path_result, name_y_axis='', percent=False, vlines=None):
    x_axis = {'PK': {},
              'name_axis': 'PK (km)'
              }

    y_axis = {i: {'color': 'lightgray', 'alpha': 0.8, 'zorder': 1, 'label': 'Simulation'}
              for i in simulations}
    y_axis |= {
        f'{name}_quantile5': {'color': '#fdb863', 'linestyle': '--', 'label': 'q05', 'zorder': 2},
        f'{name}_median': {'color': '#5e3c99', 'linestyle': '--', 'label': 'q50', 'zorder': 2},
        f'{name}_quantile95': {'color': '#e66101', 'linestyle': '--', 'label': 'q95', 'zorder': 2},
        'name_axis': f'{name_y_axis}'
    }

    cols = None
    rows = {
        'names_coord': 'horizon',
        'values_var': ['horizon1', 'horizon2', 'horizon3'],
        'names_plot': ['Horizon 1 (2021-2050)', 'Horizon 2 (2041-2070)', 'Horizon 3 (2070-2099)']
    }
    if vlines is not None:
        if 'Suggesti_2' in vlines.columns:
            cities = [i.split(' A ')[-1].split(' [')[0] for i in vlines['Suggesti_2']]
            vlines.loc[:, 'tag'] = cities
        else:
            vlines = None

    lineplot(ds, x_axis, y_axis, path_result=path_result, cols=cols, rows=rows, vlines=vlines,
             title=None, percent=percent, fontsize=14, font='sans-serif', ymax=None, plot_type='line')

def plot_linear_time(ds, name, simulations, path_result, name_y_axis='', percent=False, references=None):
    x_axis = {'time': {},
              'name_axis': 'Date'
              }

    y_axis = {i: {'color': 'lightgray', 'alpha': 0.8, 'zorder': 1, 'label': 'Simulation'}
              for i in simulations}
    y_axis |= {
        f'{name}_quantile5': {'color': '#fdb863', 'linestyle': '--', 'label': 'q05', 'zorder': 2},
        f'{name}_median': {'color': '#5e3c99', 'linestyle': '--', 'label': 'q50', 'zorder': 2},
        f'{name}_quantile95': {'color': '#e66101', 'linestyle': '--', 'label': 'q95', 'zorder': 2},
        'name_axis': f'{name_y_axis}'
    }

    cols = None

    # TODO chose station/UG mean
    rows = {
        'names_coord': 'gid',
        'values_var': ['K091001011','M624001000'],
        'names_plot': ['K091001011 amont','M624001000 aval']
    }

    lineplot(ds, x_axis, y_axis, path_result=path_result, cols=cols, rows=rows, vlines=references,
             title=None, percent=percent, fontsize=14, font='sans-serif', ymax=None, plot_type='line')

def plot_boxplot_station(ds, simulations, path_result, name_y_axis='', percent=False):

    y_axis = {i: {} for i in simulations}
    y_axis |= {'name_axis': f'{name_y_axis}'}

    x_axis = {
        'names_coord': 'gid',
        'values_var': ['K091001011','M624001000'],
        'names_plot': ['K091001011','M624001000']
    }

    x2_axis = {
        'names_coord': 'horizon',
        'values_var': ['horizon1', 'horizon2', 'horizon3'],
        'names_plot': ['Horizon 1 (2021-2050)', 'Horizon 2 (2041-2070)', 'Horizon 3 (2070-2099)']
    }

    cols = None
    rows = None

    boxplot(ds, x_axis, x2_axis, y_axis, path_result=path_result, cols=cols, rows=rows,
             title=None, percent=percent, fontsize=14, font='sans-serif', ymax=None)

def plot_map_indicator(gdf, ds, indicator_plot, path_result, cbar_title, dict_shapefiles, percent, bounds,
                       rows=None, cbar_ticks=None, discretize=None, palette='BrBG', fontsize=14, font='sans-serif', title=None,
                       vmin=None, vmax=None, edgecolor='k', cmap_zero=False):
    cols = {
        'names_coord': 'horizon',
        'values_var': ['horizon1', 'horizon2', 'horizon3'],
        'names_plot': ['Horizon 1 (2021-2050)', 'Horizon 2 (2041-2070)', 'Horizon 3 (2070-2099)']
    }


    mapplot(gdf=gdf, ds=ds, indicator_plot=indicator_plot,
            path_result=path_result,
            cols=cols, rows=rows, cbar_ticks=cbar_ticks,
            cbar_title=cbar_title, title=title, dict_shapefiles=dict_shapefiles, cmap_zero=cmap_zero,
            percent=percent, bounds=bounds,
            discretize=discretize, palette=palette, fontsize=fontsize, edgecolor=edgecolor,
            font=font, vmax=vmax, vmin=vmin)