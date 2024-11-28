from plot_functions.plot_map import *
from plot_functions.plot_lineplot import *
from plot_functions.plot_boxplot import *

def plot_linear_pk(ds, name, simulations, path_result, name_y_axis='', percent=False, references=None):
    x_axis = {'PK': {},
              'name_axis': 'PK (km)'
              }
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
    rows = {
        'names_coord': 'horizon',
        'values_var': ['horizon1', 'horizon2', 'horizon3'],
        'names_plot': ['Horizon 1 (2021-2050)', 'Horizon 2 (2041-2070)', 'Horizon 3 (2070-2099)']
    }
    if references is not None:
        if 'Suggesti_2' in references.columns:
            cities = [i.split(' A ')[-1].split(' [')[0] for i in references['Suggesti_2']]
            references.loc[:, 'tag'] = cities
        else:
            references = None

    lineplot(ds, x_axis, y_axis, path_result=path_result, cols=cols, rows=rows, vlines=references,
             title=None, percent=percent, fontsize=14, font='sans-serif', ymax=None, plot_type='line')

def plot_map_indicator(gdf, ds, indicator_plot, path_result, cbar_title, dict_shapefiles, percent, bounds,
                       cbar_ticks=None, discretize=None, palette='BrBG', fontsize=14, font='sans-serif', title=None,
                       vmin=None, vmax=None, edgecolor='k', cmap_zero=False):
    cols = {
        'names_coord': 'horizon',
        'values_var': ['horizon1', 'horizon2', 'horizon3'],
        'names_plot': ['Horizon 1 (2021-2050)', 'Horizon 2 (2041-2070)', 'Horizon 3 (2070-2099)']
    }

    rows = None

    mapplot(gdf=gdf, ds=ds, indicator_plot=indicator_plot,
            path_result=path_result,
            cols=cols, rows=rows, cbar_ticks=cbar_ticks,
            cbar_title=cbar_title, title=title, dict_shapefiles=dict_shapefiles, cmap_zero=cmap_zero,
            percent=percent, bounds=bounds,
            discretize=discretize, palette=palette, fontsize=fontsize, edgecolor=edgecolor,
            font=font, vmax=vmax, vmin=vmin)