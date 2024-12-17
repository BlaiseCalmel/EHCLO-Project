import copy

from plot_functions.plot_map import *
from plot_functions.plot_lineplot import *
from plot_functions.plot_boxplot import *

def plot_linear_pk_hm(ds, simulations, path_result, narratives=None,
                   name_x_axis='', name_y_axis='', percent=False, vlines=None,
                      fontsize=14, font='sans-serif'):
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

    lineplot(ds, indicator_plot, x_axis, y_axis, path_result=path_result, cols=cols, rows=rows, vlines=vlines,
             title=None, percent=percent, fontsize=fontsize, font=font, ymax=None)

def plot_linear_pk_narrative(ds, simulations, path_result, narratives=None,
                      name_x_axis='', name_y_axis='', percent=False, vlines=None,
                             fontsize=14, font='sans-serif'):
    x_axis = {'names_coord': 'PK',
              'name_axis': name_x_axis
              }
    y_axis = {'names_coord': 'indicator',
              'name_axis': name_y_axis
              }

    dict_sim = {sim_name: {'color': 'lightgray', 'alpha': 0.8, 'zorder': 1, 'label': 'Simulation', 'linewidth': 0.5}
                for sim_name in simulations}

    if narratives is not None:
        indicator_plot = [copy.deepcopy(dict_sim) for i in range(len(narratives))]
        idx = -1
        for narr_name, kwargs in narratives.items():
            idx += 1
            for sim_name, values in indicator_plot[idx].items():
                if narr_name in sim_name:
                    indicator_plot[idx][sim_name] = kwargs
    else:
        indicator_plot = [copy.deepcopy(dict_sim)]

    cols = {
        'names_coord': 'horizon',
        'values_var': ['horizon1', 'horizon2', 'horizon3'],
        'names_plot': ['Horizon 1 (2021-2050)', 'Horizon 2 (2041-2070)', 'Horizon 3 (2070-2099)']
    }

    rows = {
        'names_coord': 'indicator',
        'values_var': indicator_plot,
        'names_plot': list(narratives.keys())
    }

    lineplot(ds, indicator_plot, x_axis, y_axis, path_result=path_result, cols=cols, rows=rows, vlines=vlines,
             title=None, percent=percent, fontsize=fontsize, font=font, ymax=None)

def plot_linear_pk(ds, simulations, path_result, narratives=None,
                   name_x_axis='', name_y_axis='', percent=False, vlines=None,
                   fontsize=14, font='sans-serif'):
    x_axis = {'names_coord': 'PK',
              'name_axis': name_x_axis
              }
    y_axis = {'names_coord': 'indicator',
              'name_axis': name_y_axis
              }

    indicator_plot = {sim_name: {'color': 'lightgray', 'alpha': 0.8, 'zorder': 1, 'label': 'Simulation', 'linewidth': 0.5}
                for sim_name in simulations}

    for sim_name, subdict in indicator_plot.items():
        for narr_name, kwargs in narratives.items():
            if narr_name in sim_name:
                indicator_plot[sim_name] = kwargs

    if isinstance(indicator_plot, dict):
        indicator_plot = [indicator_plot]
    cols = {
        'names_coord': 'horizon',
        'values_var': ['horizon1', 'horizon2', 'horizon3'],
        'names_plot': ['Horizon 1 (2021-2050)', 'Horizon 2 (2041-2070)', 'Horizon 3 (2070-2099)']
    }

    rows = {
        'names_coord': 'indicator',
        'values_var': indicator_plot,
        'names_plot': ''
    }

    lineplot(ds, indicator_plot, x_axis, y_axis, path_result=path_result, cols=cols, rows=rows, vlines=vlines,
             title=None, percent=percent, fontsize=fontsize, font=font, ymax=None)


def plot_linear_time(ds, simulations, path_result, narratives=None,
                     name_x_axis='', name_y_axis='', percent=False, vlines=None, fontsize=14, font='sans-serif'):

    x_axis = {'names_coord': 'time',
              'name_axis': name_x_axis
              }

    y_axis = {'names_coord': 'indicator',
              'name_axis': name_y_axis
              }

    station_references = {'M842001000': 'La Loire à St Nazaire',
                          'M530001010': 'La Loire à Mont Jean',
                          'K683002001': 'La Loire à Langeais',
                          'K480001001': 'La Loire à Onzain',
                          'K418001201': 'La Loire à Gien',
                          'K193001010': 'La Loire à Nevers',
                          'K091001011': 'La Loire à Villerest',
                          'K365081001': "L'Allier à Cuffy"}

    dict_sim = {sim_name: {'color': 'lightgray', 'alpha': 0.8, 'zorder': 1, 'label': 'Simulation', 'linewidth': 0.5}
       for sim_name in simulations}

    if narratives is not None:
        for key in dict_sim.keys():
            for narr_name, kwargs in narratives.items():
                if narr_name in key:
                    dict_sim[key] = kwargs

    indicator_plot = [dict_sim for i in range(len(station_references))]

    cols = {
        'names_coord': 'gid',
        'values_var': list(station_references.keys()),
        'names_plot': list(station_references.values())
    }

    rows = 4

    lineplot(ds, indicator_plot, x_axis, y_axis, path_result=path_result, cols=cols, rows=rows, vlines=vlines,
             title=None, percent=percent, fontsize=fontsize, font=font, ymax=None)


def plot_boxplot_station_narrative(ds, simulations, narratives, references, path_result, name_y_axis='', percent=False):


    narratives = {
        "HadGEM2-ES_ALADIN63_ADAMONT": {'boxprops':dict(facecolor='#569A71', alpha=0.9),
                                        'medianprops': dict(color="black"), 'widths':0.5, 'patch_artist':True},
        "CNRM-CM5_ALADIN63_ADAMONT": {'boxprops':dict(facecolor='#EECC66', alpha=0.9),
                                      'medianprops': dict(color="black"), 'widths':0.5, 'patch_artist':True},
        "EC-EARTH_HadREM3-GA7_ADAMONT": {'boxprops':dict(facecolor='#E09B2F', alpha=0.9),
                                         'medianprops': dict(color="black"), 'widths':0.5, 'patch_artist':True},
        "HadGEM2-ES_CCLM4-8-17_ADAMONT": {'boxprops':dict(facecolor='#791F5D', alpha=0.9),
                                          'medianprops': dict(color="black"), 'widths':0.5, 'patch_artist':True},
    }

    station_references = {'M842001000': 'La Loire à St Nazaire',
                          'M530001010': 'La Loire à Mont Jean',
                          'K683002001': 'La Loire à Langeais',
                          'K480001001': 'La Loire à Onzain',
                          'K418001201': 'La Loire à Gien',
                          'K193001010': 'La Loire à Nevers',
                          'K091001011': 'La Loire à Villerest',
                          'K365081001': "L'Allier à Cuffy"}

    dict_sims = {}
    for narr_name, kwargs in narratives.items():
        dict_sims[narr_name] = {'values': [], 'kwargs': kwargs}
        for sim_name in simulations:
            if narr_name in sim_name:
                dict_sims[narr_name]['values'].append(sim_name)

    indicator_plot = [dict_sims]
    y_axis = {'names_coord': 'indicator',
              'values_var': indicator_plot,
              'names_plot': list(narratives.keys()),
              'name_axis': name_y_axis
              }

    x_axis = {
        'names_coord': 'horizon',
        'values_var': ['horizon1', 'horizon2', 'horizon3'],
        'names_plot': ['H1', 'H2', 'H3']
    }

    cols = None
    rows = {
        'names_coord': 'gid',
        'values_var': list(station_references.keys()),
        'names_plot': list(station_references.values())
    }

    boxplot(ds, x_axis, y_axis, path_result=path_result, cols=cols, rows=rows,
             title=None, percent=percent, fontsize=14, font='sans-serif', ymax=None)

def plot_map_indicator_hm(gdf, ds, path_result, cbar_title, dict_shapefiles, percent, bounds,
                          variables, discretize=None, palette='BrBG', fontsize=14, font='sans-serif', title=None,
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
            cols=cols, rows=rows,
            cbar_title=cbar_title, title=title, dict_shapefiles=dict_shapefiles, cbar_midpoint=cbar_midpoint,
            percent=percent, bounds=bounds,
            discretize=discretize, palette=palette, fontsize=fontsize, edgecolor=edgecolor,
            font=font, vmax=vmax, vmin=vmin, markersize=markersize)

def plot_map_indicator_climate(gdf, ds, path_result, cbar_title, dict_shapefiles, percent, bounds,
                               indicator_plot, cbar_ticks=None, discretize=None, palette='BrBG', fontsize=14, font='sans-serif', title=None,
                               vmin=None, vmax=None, edgecolor='k', cbar_midpoint=None, markersize=50,cbar_values=None):

    cols = {
            'names_coord': 'horizon',
            'values_var': ['horizon1', 'horizon2', 'horizon3'],
            'names_plot': ['Horizon 1 (2021-2050)', 'Horizon 2 (2041-2070)', 'Horizon 3 (2070-2099)']
             }

    rows = 1

    mapplot(gdf=gdf, ds=ds, indicator_plot=indicator_plot,
            path_result=path_result,
            cols=cols, rows=rows, cbar_ticks=cbar_ticks,
            cbar_title=cbar_title, title=title, dict_shapefiles=dict_shapefiles, cbar_midpoint=cbar_midpoint,
            percent=percent, bounds=bounds, cbar_values=cbar_values,
            discretize=discretize, palette=palette, fontsize=fontsize, edgecolor=edgecolor,
            font=font, vmax=vmax, vmin=vmin, markersize=markersize)


def plot_map_HM_by_station(hydro_sim_points_gdf_simplified, dict_shapefiles, bounds, path_global_figures,
                           fontsize=14):
    shape_hp = {
        'CTRIP': 'D',
        'EROS': 'H',
        'GRSD': '*',
        'J2000': 's',
        'MORDOR-SD': 'v',
        'MORDOR-TS': '^',
        'SIM2': '>',
        'SMASH': '<',
        'ORCHIDEE': 'o',
    }
    cols_map = {
        'values_var': list(shape_hp.keys()),
        'names_plot': list(shape_hp.keys())
    }

    rows = 3
    mapplot(gdf=hydro_sim_points_gdf_simplified, ds=None, indicator_plot=list(shape_hp.keys()),
            path_result=f"{path_global_figures}HM_by_sim.pdf",
            cols=cols_map, rows=rows,
            cbar_title=f"Simulation", title=None, dict_shapefiles=dict_shapefiles, percent=False, bounds=bounds,
            discretize=2, cbar_ticks='mid', palette='RdBu_r', cbar_midpoint='min', fontsize=fontsize, font='sans-serif', edgecolor='k',
            vmin=-0.5, vmax=1.5, markersize=75,
            cbar_values=['Absente', 'Présente'])

def plot_map_N_HM_ref_station(hydro_sim_points_gdf_simplified, dict_shapefiles,
                              path_global_figures, bounds, fontsize=14):
    station_references = {
        'M842001000': {'s':90, 'edgecolors':'k', 'zorder':10, 'linewidth': 1.5,
                       'facecolors':'none',
                       'text': {'text': 'La Loire à\nSt Nazaire', 'arrowprops':dict(arrowstyle='-'),
                                'xytext':(0.1, 0.8), 'textcoords':'axes fraction', 'ha':'center'}
                       },
        'M530001010': {'s':90, 'edgecolors':'k', 'zorder':10, 'linewidth': 1.5,
                       'facecolors':'none',
                       'text': {'text': 'La Loire à \nMont Jean', 'ha':'center', 'arrowprops':dict(arrowstyle='-'),
                                'xytext':(0.2, 0.55), 'textcoords':'axes fraction'}
                       },
        'K683002001': {'s':90, 'edgecolors':'k', 'zorder':10, 'linewidth': 1.5,
                       'facecolors':'none',
                       'text': {'text': ' La Loire à \nLangeais', 'ha':'center', 'arrowprops':dict(arrowstyle='-'),
                                'xytext':(0.35, 0.75), 'textcoords':'axes fraction'}
                       },
        'K480001001': {'s':90, 'edgecolors':'k', 'zorder':10, 'linewidth': 1.5,
                       'facecolors':'none',
                       'text': {'text': ' La Loire à \nOnzain', 'ha':'center', 'arrowprops':dict(arrowstyle='-'),
                                'xytext':(0.5, 0.55), 'textcoords':'axes fraction'}
                       },
        'K418001201': {'s':90, 'edgecolors':'k', 'zorder':10, 'linewidth': 1.5,
                       'facecolors':'none',
                       'text': {'text': ' La Loire à \nGien', 'ha':'center', 'arrowprops':dict(arrowstyle='-'),
                                'xytext':(0.75, 0.8), 'textcoords':'axes fraction'}
                       },
        'K193001010': {'s':90, 'edgecolors':'k', 'zorder':10, 'linewidth': 1.5,
                       'facecolors':'none',
                       'text': {'text': ' La Loire à \nNevers', 'ha':'center', 'arrowprops':dict(arrowstyle='-'),
                                'xytext':(0.9, 0.65), 'textcoords':'axes fraction'}
                       },
        'K365081001': {'s':90, 'edgecolors':'k','zorder':10, 'linewidth': 1.5,
                       'facecolors':'none',
                       'text': {'text': 'L\'Allier à \nCuffy ', 'ha':'center', 'arrowprops':dict(arrowstyle='-'),
                                'xytext':(0.72, 0.42), 'textcoords':'axes fraction'}
                       },
        'K091001011': {'s':90, 'edgecolors':'k', 'zorder':10, 'linewidth': 1.5,
                       'facecolors':'none',
                       'text': {'text': ' La Loire à \nVillerest', 'ha':'center', 'arrowprops':dict(arrowstyle='-'),
                                'xytext':(0.9, 0.2), 'textcoords':'axes fraction'}
                       }
    }

    for key in station_references.keys():
        station_references[key] |= {'x': hydro_sim_points_gdf_simplified.loc[key].geometry.x,
                                    'y': hydro_sim_points_gdf_simplified.loc[key].geometry.y}
        station_references[key]['text'] |= {'xy': (hydro_sim_points_gdf_simplified.loc[key].geometry.x,
                                                   hydro_sim_points_gdf_simplified.loc[key].geometry.y)}

    print(f"> Plot Number of HM by station...")
    j = -1
    for key in dict_shapefiles.keys():
        j += 1
        dict_shapefiles[key]['alpha'] = 0.2
        dict_shapefiles[key]['zorder'] = -j

    mapplot(gdf=hydro_sim_points_gdf_simplified, indicator_plot='n', path_result=path_global_figures+'count_HM.pdf', ds=None,
            cols=None, rows=None, references=station_references, cbar_ticks='mid',  cbar_values=1,
            cbar_title=f"Nombre de HM", title=None, dict_shapefiles=dict_shapefiles, percent=False, bounds=bounds,
            discretize=6, palette='RdBu_r', fontsize=fontsize-5, font='sans-serif', edgecolor='k',
            cbar_midpoint='min', vmin=3.5, vmax=9.5)
