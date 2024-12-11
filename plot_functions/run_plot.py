from plot_functions.plot_map import *
from plot_functions.plot_lineplot import *
from plot_functions.plot_boxplot import *

def plot_linear_pk(ds, simulations, path_result, narratives=None,
                   name_x_axis='', name_y_axis='', percent=False, vlines=None):
    x_axis = {'names_coord': 'PK',
              'name_axis': name_x_axis
              }

    y_axis = {'names_coord': 'indicator',
              'name_axis': name_y_axis
              }

    value_simulations = [i for i in simulations.values()]
    indicator_plot = [{val: {'color': 'lightgray', 'alpha': 0.8, 'zorder': 1, 'label': 'Simulation', 'linewidth': 0.5}
      for val in sublist} for sublist in value_simulations]

    if narratives is not None:
        for subdict in indicator_plot:
            for key in subdict.keys():
                for narr_name, kwargs in narratives.items():
                    if narr_name in key:
                        subdict[key] = kwargs

    cols = {
        'names_coord': 'horizon',
        'values_var': ['horizon1', 'horizon2', 'horizon3'],
        'names_plot': ['Horizon 1 (2021-2050)', 'Horizon 2 (2041-2070)', 'Horizon 3 (2070-2099)']
    }

    rows = {
        'names_coord': 'indicator',
        'values_var': indicator_plot,
        'names_plot': list(simulations.keys())
    }

    if vlines is not None:
        if 'Suggesti_2' in vlines.columns:
            cities = [i.split(' A ')[-1].split(' [')[0] for i in vlines['Suggesti_2']]
            vlines.insert(loc=0, column='label', value=cities)
            vlines['annotate'] = 0.02
        else:
            vlines = None

    lineplot(ds, indicator_plot, x_axis, y_axis, path_result=path_result, cols=cols, rows=rows, vlines=vlines,
             title=None, percent=percent, fontsize=14, font='sans-serif', ymax=None)

def plot_linear_time(ds, simulations, path_result, name_x_axis='', name_y_axis='', percent=False,
                     references=None):

    x_axis = {'names_coord': 'time',
              'name_axis': 'Date'
              }

    y_axis = {'names_coord': 'indicator',
              'name_axis': name_y_axis
              }

    # y_axis = {i: {'color': 'lightgray', 'alpha': 0.8, 'zorder': 1, 'label': 'Simulation'}
    #           for i in simulations}
    # y_axis |= {
    #     f'{name}_quantile5': {'color': '#fdb863', 'linestyle': '--', 'label': 'q05', 'zorder': 2},
    #     f'{name}_median': {'color': '#5e3c99', 'linestyle': '--', 'label': 'q50', 'zorder': 2},
    #     f'{name}_quantile95': {'color': '#e66101', 'linestyle': '--', 'label': 'q95', 'zorder': 2},
    #     'name_axis': f'{name_y_axis}'
    # }

    value_simulations = [i for i in simulations.values()]
    indicator_plot = [{val: {'color': 'lightgray', 'alpha': 0.8, 'zorder': 1, 'label': 'Simulation', 'linewidth': 0.5}
                  for val in sublist} for sublist in value_simulations]

    cols = None

    # TODO chose station/UG mean
    rows = {
        'names_coord': 'gid',
        'values_var': ['K091001011','M624001000'],
        'names_plot': ['K091001011 amont','M624001000 aval']
    }

    lineplot(ds, indicator_plot, x_axis, y_axis, path_result=path_result, cols=cols, rows=rows, vlines=references,
             title=None, percent=percent, fontsize=14, font='sans-serif', ymax=None)

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

def plot_map_indicator_hm(gdf, ds, path_result, cbar_title, dict_shapefiles, percent, bounds,
                          variables, cbar_ticks=None, discretize=None, palette='BrBG', fontsize=14, font='sans-serif', title=None,
                          vmin=None, vmax=None, edgecolor='k', cbar_midpoint=None, markersize=50):

    mean_by_hm = [s for sublist in variables['hydro_model_deviation'].values() for s in sublist if "mean" in s]

    # Dictionnary sim by HM
    hm_names = [name.split('_')[-1] for name in variables['simulation_cols']]
    hm_dict_deviation = {i: [] for i in np.unique(hm_names)}
    for idx, name_sim in enumerate(variables['simulation_deviation']):
        hm_dict_deviation[hm_names[idx]].append(name_sim)

    rows = {
        'names_coord': 'indicator',
        'values_var': mean_by_hm,
        'names_plot': list(variables['hydro_model_deviation'].keys())
    }
    cols = 3

    mapplot(gdf=gdf, ds=ds, indicator_plot=mean_by_hm,
            path_result=path_result,
            cols=cols, rows=rows, cbar_ticks=cbar_ticks,
            cbar_title=cbar_title, title=title, dict_shapefiles=dict_shapefiles, cbar_midpoint=cbar_midpoint,
            percent=percent, bounds=bounds,
            discretize=discretize, palette=palette, fontsize=fontsize, edgecolor=edgecolor,
            font=font, vmax=vmax, vmin=vmin, markersize=markersize)




    # if cols == 'horizon':
    #     cols = {
    #             'names_coord': 'horizon',
    #             'values_var': ['horizon1', 'horizon2', 'horizon3'],
    #             'names_plot': ['Horizon 1 (2021-2050)', 'Horizon 2 (2041-2070)', 'Horizon 3 (2070-2099)']
    #         }
    #
    # if rows is None:
    #     rows = 1

