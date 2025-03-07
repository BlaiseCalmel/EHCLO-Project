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
from plot_functions.plot_map import *
from plot_functions.plot_lineplot import *
from plot_functions.plot_boxplot import *

def plot_linear_pk_hm(ds, simulations, path_result, narratives=None,
                   name_x_axis='', name_y_axis='', percent=False, vlines=None, title=None,
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
                for narr_name in narratives.values():
                    for sim_name, kwargs in narr_name.items():
                        if sim_name in key:
                            subdict[key] = kwargs

    legend_items = [{'color': 'lightgray', 'alpha': 0.8, 'zorder': 1, 'label': 'Simulation', 'linewidth': 0.5}]
    legend_items += [value for narr_name in narratives.values() for value in narr_name.values()]

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
             legend_items=legend_items, title=title, percent=percent, fontsize=fontsize, font=font, ymax=None)

def plot_linear_pk_narrative(ds, simulations, path_result, narratives=None,
                             name_x_axis='', name_y_axis='', percent=False, vlines=None, title=None, ymax=None, xmax=None,
                             fontsize=14, font='sans-serif', by_narrative=False):
    x_axis = {'names_coord': 'PK',
              'name_axis': name_x_axis
              }
    y_axis = {'names_coord': 'indicator',
              'name_axis': name_y_axis
              }

    dict_sim = {sim_name: {'color': 'lightgray', 'alpha': 0.5, 'zorder': 1, 'label': 'Ensemble des projections', 'linewidth': 0.5}
                for sim_name in simulations}

    if narratives is not None:
        if not by_narrative:
            indicator_plot = [copy.deepcopy(dict_sim) for narr_name in narratives.values() for i in narr_name]
            idx = -1
            for narr in narratives.values():
                for narr_name, kwargs in narr.items():
                    idx += 1
                    for sim_name, values in indicator_plot[idx].items():
                        if narr_name in sim_name:
                            indicator_plot[idx][sim_name] = kwargs
            row_names_plot = [f"Narratif {i['label'].split(' ')[0]}" for value in narratives.values() for i in
                              value.values()]
        else:
            indicator_plot = [copy.deepcopy(dict_sim) for narr_name in narratives.values()]
            idx = -1
            for narr in narratives.values():
                idx += 1
                for narr_name, kwargs in narr.items():
                    for sim_name, values in indicator_plot[idx].items():
                        if narr_name in sim_name:
                            indicator_plot[idx][sim_name] = kwargs
            row_names_plot = [f"{key.title()}" for key in narratives.keys()]
    # indicator_plot = [copy.deepcopy(dict_sim) for narr_name in narratives.values() for i in narr_name]
    # idx = -1
    # for narr in narratives.values():
    #     for narr_name, kwargs in narr.items():
    #         idx += 1
    #         for sim_name, values in indicator_plot[idx].items():
    #             if narr_name in sim_name:
    #                 indicator_plot[idx][sim_name] = kwargs

    legend_items = [{'color': 'lightgray', 'alpha': 0.5, 'zorder': 1, 'label': 'Ensemble des projections', 'linewidth': 0.5}]
    legend_items += [value for narr_value in narratives.values() for value in narr_value.values()]
    # Suppression des doublons en convertissant les dictionnaires en tuples immuables
    unique_legend_items = []
    for i in legend_items:
        if i not in unique_legend_items:
            unique_legend_items.append(i)
    # legend_items = [value for value in narratives.values()]

    cols = {
        'names_coord': 'horizon',
        'values_var': ['horizon1', 'horizon2', 'horizon3'],
        'names_plot': ['Horizon 1 (2021-2050)', 'Horizon 2 (2041-2070)', 'Horizon 3 (2070-2099)']
    }

    rows = {
        'names_coord': 'indicator',
        'values_var': indicator_plot,
        'names_plot': row_names_plot #list(narratives.keys())
    }
    # rows = {
    #     'names_coord': 'indicator',
    #     'values_var': indicator_plot,
    #     'names_plot': [f"Centrés", "Éloignés", "Mixtes"] #list(narratives.keys())
    # }

    lineplot(ds, indicator_plot, x_axis, y_axis, path_result=path_result, cols=cols, rows=rows, vlines=vlines,
             legend_items=unique_legend_items, title=title, percent=percent, ymax=ymax, xmax=xmax,
             fontsize=fontsize, font=font)

def plot_linear_pk(ds, simulations, path_result, narratives=None,
                   name_x_axis='', name_y_axis='', percent=False, vlines=None, title=None,
                   fontsize=14, font='sans-serif', by_narrative=False, xmax=None, ymax=None):
    x_axis = {'names_coord': 'PK',
              'name_axis': name_x_axis
              }
    y_axis = {'names_coord': 'indicator',
              'name_axis': name_y_axis
              }

    dict_sim = {sim_name: {'color': 'lightgray', 'alpha': 0.5, 'zorder': 1, 'label': 'Ensemble des projections', 'linewidth': 0.5}
                for sim_name in simulations}

    row_names_plot = None
    if narratives is not None:
        indicator_plot = [copy.deepcopy(dict_sim) for narr_name in narratives.values()]
        idx = -1
        for narr in narratives.values():
            idx += 1
            for narr_name, kwargs in narr.items():
                for sim_name, values in indicator_plot[idx].items():
                    if narr_name in sim_name:
                        indicator_plot[idx][sim_name] = kwargs
        if not by_narrative:
            row_names_plot = [None for value in narratives.values() for i in
                              value.values()]
        else:
            row_names_plot = [f"{key.title()}" for key in narratives.keys()]

    legend_items = [{'color': 'lightgray', 'alpha': 0.5, 'zorder': 1, 'label': 'Ensemble des projections', 'linewidth': 0.5}]
    legend_items += [value for narr_value in narratives.values() for value in narr_value.values()]
    # Suppression des doublons en convertissant les dictionnaires en tuples immuables
    unique_legend_items = []
    for i in legend_items:
        if i not in unique_legend_items:
            unique_legend_items.append(i)
    # legend_items = [value for value in narratives.values()]

    cols = {
        'names_coord': 'horizon',
        'values_var': ['horizon1', 'horizon2', 'horizon3'],
        'names_plot': ['Horizon 1 (2021-2050)', 'Horizon 2 (2041-2070)', 'Horizon 3 (2070-2099)']
    }

    rows = {
        'names_coord': 'indicator',
        'values_var': indicator_plot,
        'names_plot': row_names_plot
    }

    lineplot(ds, indicator_plot, x_axis, y_axis, path_result=path_result, cols=cols, rows=rows, vlines=vlines,
             legend_items=unique_legend_items, title=title, percent=percent, ymax=ymax, xmax=xmax,
             fontsize=fontsize, font=font)


def plot_linear_time(ds, simulations, path_result, station_references, narratives=None,
                     name_x_axis='', name_y_axis='', percent=False, vlines=None, title=None,
                     fontsize=14, font='sans-serif'):

    x_axis = {'names_coord': 'time',
              'name_axis': name_x_axis
              }

    y_axis = {'names_coord': 'indicator',
              'name_axis': name_y_axis
              }

    dict_sim = {sim_name: {'color': 'lightgray', 'alpha': 0.5, 'zorder': 1, 'label': 'Ensemble des projections', 'linewidth': 0.5}
       for sim_name in simulations}

    if narratives is not None:
        for key in dict_sim.keys():
            for narr_type, narr in narratives.items():
                for sim_name, kwargs in narr.items():
                    if sim_name in key:
                        dict_sim[key] = kwargs

    legend_items = [{'color': 'lightgray', 'alpha': 0.5, 'zorder': 1, 'label': 'Ensemble des projections', 'linewidth': 0.5}]
    legend_items += [i for narr_value in narratives.values() for i in narr_value.values()]

    indicator_plot = [dict_sim for i in range(len(station_references))]

    rows = {
        'names_coord': 'gid',
        'values_var': list(station_references.keys()),
        'names_plot': list(station_references.values())
    }

    cols = 2

    lineplot(ds, indicator_plot, x_axis, y_axis, path_result=path_result, cols=cols, rows=rows, vlines=vlines,
             title=title, percent=percent, legend_items=legend_items, fontsize=fontsize, font=font, ymax=None)


def plot_boxplot_station_narrative(ds, station_references, narratives, references, path_result, name_y_axis='', percent=False,
                                   title=None, fontsize=14, font='sans-serif'):

    simulations = list(ds.data_vars)

    narratives_bp = {key: {'boxprops':dict(facecolor=value['color'], alpha=0.9),
    'medianprops': dict(color="black"), 'widths':0.9, 'patch_artist':True,
    'label': value['label']} for narr_value in narratives.values() for key, value in narr_value.items()}

    dict_sims = {}
    dict_sims['simulations'] = {'values': simulations, 'kwargs': {'boxprops':dict(facecolor='lightgray', alpha=0.8),
                                                                  'medianprops': dict(color="black"), 'widths': 0.9,
                                                                  'patch_artist':True, 'label': 'Ensemble des projections'}}
    for narr_name, kwargs in narratives_bp.items():
        dict_sims[narr_name] = {'values': [], 'kwargs': kwargs}
        for sim_name in simulations:
            if narr_name in sim_name:
                dict_sims[narr_name]['values'].append(sim_name)

    # indicator_plot = [dict_sims]
    y_axis = {'names_coord': 'indicator',
              'values_var': dict_sims, #indicator_plot
              'names_plot': list(dict_sims.keys()),
              'name_axis': name_y_axis
              }

    x_axis = {
        'names_coord': 'horizon',
        'values_var': ['horizon1', 'horizon2', 'horizon3'],
        'names_plot': ['H1', 'H2', 'H3']
    }

    cols = 2
    rows = {
        'names_coord': 'gid',
        'values_var': list(station_references.keys()),
        'names_plot': list(station_references.values())
    }

    boxplot(ds, x_axis, y_axis, path_result=path_result, cols=cols, rows=rows, vlines=True,
             title=title, percent=percent, fontsize=fontsize, font=font, ymax=None, blank_space=0.25)

def plot_boxplot_station_month_horizon(ds, station_references, narratives, path_result, name_y_axis='', percent=False,
                                       title=None, fontsize=14, font='sans-serif', common_yaxes=False, normalized=False,
                                       ymin=None, ymax=None):

    narratives_bp = {}
    narratives_bp |= {key: {'type': 'histo', 'kwargs': {
        'label': value['label'], 'zorder': 1, 'color':'lightgray', 'edgecolor':'k', 'alpha':0.5}} for key, value in
                      narratives.items() if key == "historical"}
    narratives_bp |= {key: {'kwargs': {'boxprops': dict(facecolor=value['color'], alpha=0.9),
                            'medianprops': dict(color="black"), 'widths': 0.8, 'patch_artist': True,
                            'label': value['label'], 'zorder': 2}} for key, value in narratives.items() if key != "historical"}

    if normalized:
        mean_historical = ds.sel(horizon='historical').mean(dim=['month'])
        ds = ds / mean_historical

    y_axis = {'names_coord': 'horizon',
              'values_var': narratives_bp,
              'names_plot': list(narratives_bp.keys()),
              'name_axis': name_y_axis
              }

    x_axis = {
        'names_coord': 'month',
        'values_var': list(ds.month.values),
        'names_plot': [i[0] for i in list(ds.month.values)]
    }

    cols = 2
    rows = {
        'names_coord': 'gid',
        'values_var': list(station_references.keys()),
        'names_plot': list(station_references.values())
    }

    boxplot(ds, x_axis, y_axis, path_result=path_result, cols=cols, rows=rows, vlines=True, common_yaxes=common_yaxes,
            title=title, percent=percent, fontsize=fontsize, font=font, ymin=ymin, ymax=ymax, blank_space=0.25)

def plot_map_indicator_hm(gdf, ds, path_result, cbar_title, dict_shapefiles, bounds,
                          variables, plot_type, discretize=None, palette='BrBG', fontsize=14, font='sans-serif', title=None,
                          vmin=None, vmax=None, edgecolor='k', cbar_midpoint=None, markersize=50, alpha=1, selected_stats='median'):

    # mean_by_hm = [s for sublist in variables['hydro-model_deviation'].values() for s in sublist if "median" in s]

    stats_by_hm = [s for sublist in variables[f'hydro-model_{plot_type}'].values() for s in sublist if selected_stats in s]

    # Dictionnary sim by HM
    hm_names = [name.split('_')[-1] for name in variables['simulation_cols']]
    hm_dict_deviation = {i: [] for i in np.unique(hm_names)}
    for idx, name_sim in enumerate(variables['simulation_deviation']):
        hm_dict_deviation[hm_names[idx]].append(name_sim)

    rows = {
        'names_coord': 'indicator',
        'values_var': stats_by_hm,
        'names_plot': list(variables['hydro-model_deviation'].keys())
    }
    cols = 3

    mapplot(gdf=gdf, ds=ds, indicator_plot=stats_by_hm,
            path_result=path_result,
            cols=cols, rows=rows,
            cbar_title=cbar_title, title=title, dict_shapefiles=dict_shapefiles, cbar_midpoint=cbar_midpoint,
            bounds=bounds,
            discretize=discretize, palette=palette, fontsize=fontsize, edgecolor=edgecolor,
            font=font, vmax=vmax, vmin=vmin, markersize=markersize, alpha=alpha)

def plot_map_indicator_narratives(gdf, ds, narratives, path_result, cbar_title, dict_shapefiles, bounds,
                                  variables, plot_type, discretize=None, palette='BrBG', fontsize=14, font='sans-serif', title=None,
                                  vmin=None, vmax=None, edgecolor='k', cbar_midpoint=None, markersize=50, alpha=1):

    narr = [key for narr_values in narratives.values() for key in narr_values.keys()]
    stats_by_narr = [sim for sim in variables[f'simulation-horizon_by-sims_{plot_type}'] if any(n in sim for n in narr)]

    # rows = {
    #     'names_coord': 'indicator',
    #     'values_var': stats_by_narr,
    #     'names_plot': [v['label'] for narr in narratives.values() for v in narr.values()]
    # }
    rows = {
        'names_coord': 'indicator',
        'values_var': stats_by_narr,
        'names_plot': [{'label': v['label'], 'color': v['color']} for narr in narratives.values() for v in narr.values()]
    }
    cols = 2

    mapplot(gdf=gdf, ds=ds, indicator_plot=stats_by_narr,
            path_result=path_result,
            cols=cols, rows=rows,
            cbar_title=cbar_title, title=title, dict_shapefiles=dict_shapefiles, cbar_midpoint=cbar_midpoint,
            bounds=bounds,
            discretize=discretize, palette=palette, fontsize=fontsize, edgecolor=edgecolor,
            font=font, vmax=vmax, vmin=vmin, markersize=markersize, alpha=alpha)

def plot_map_indicator(gdf, ds, path_result, cbar_title, dict_shapefiles, bounds,
                       indicator_plot, cbar_ticks=None, discretize=None, palette='BrBG', fontsize=14,
                       font='sans-serif', title=None, vmin=None, vmax=None, edgecolor='k',
                       cbar_midpoint=None, markersize=50, alpha=1, cbar_values=None,
                       start_cbar_ticks='', end_cbar_ticks=''):

    cols = {
            'names_coord': 'horizon',
            'values_var': ['horizon1', 'horizon2', 'horizon3'],
            'names_plot': ['Horizon 1\n(2021-2050)', 'Horizon 2\n(2041-2070)', 'Horizon 3\n(2070-2099)']
             }

    used_coords = [dim for dim in ds[indicator_plot].dims if dim in ds.coords]
    if 'season' in used_coords:
        rows = {
            'names_coord': 'season',
            'values_var': ['Hiver', 'Printemps', 'Été', 'Automne'],
            'names_plot': ['Hiver', 'Printemps', 'Été', 'Automne']
        }
    # elif 'month' in used_coords:
    #
    #     rows = {
    #         'names_coord': 'month',
    #         'values_var': list(ds.month.values),
    #         'names_plot': ['Janv.', 'Fev.', 'Mars',
    #                        'Avril.', 'Mai', 'Juin.',
    #                        'Juill.', 'Août', 'Sept.',
    #                        'Oct.', 'Nov.', 'Déc.']
    #     }
    else:
        rows = 1

    mapplot(gdf=gdf, ds=ds, indicator_plot=indicator_plot,
            path_result=path_result,
            cols=cols, rows=rows, cbar_ticks=cbar_ticks,
            cbar_title=cbar_title, title=title, dict_shapefiles=dict_shapefiles, cbar_midpoint=cbar_midpoint,
            bounds=bounds, cbar_values=cbar_values,
            discretize=discretize, palette=palette, fontsize=fontsize, edgecolor=edgecolor,
            font=font, vmax=vmax, vmin=vmin, markersize=markersize, alpha=alpha,
            start_cbar_ticks=start_cbar_ticks, end_cbar_ticks=end_cbar_ticks)

def plot_map_matching_sim(gdf, ds, path_result, cbar_title, dict_shapefiles, bounds,
                               indicator_plot, cbar_ticks=None, discretize=None, palette='BrBG', fontsize=14,
                               font='sans-serif', title=None, vmin=None, vmax=None, edgecolor='k',
                               cbar_midpoint=None, markersize=50, alpha=1, cbar_values=None,
                               start_cbar_ticks='', end_cbar_ticks=''):

    cols = {
        'names_coord': 'horizon',
        'values_var': ['horizon1', 'horizon2', 'horizon3'],
        'names_plot': ['Horizon 1 (2021-2050)', 'Horizon 2 (2041-2070)', 'Horizon 3 (2070-2099)']
    }

    used_coords = [dim for dim in ds[indicator_plot].dims if dim in ds.coords]
    if 'season' in used_coords:
        rows = {
            'names_coord': 'season',
            'values_var': ['Hiver', 'Printemps', 'Été', 'Automne'],
            'names_plot': ['Hiver', 'Printemps', 'Été', 'Automne']
        }
    else:
        rows = 1

    mapplot(gdf=gdf, ds=ds, indicator_plot=indicator_plot,
            path_result=path_result,
            cols=cols, rows=rows, cbar_ticks=cbar_ticks,
            cbar_title=cbar_title, title=title, dict_shapefiles=dict_shapefiles, cbar_midpoint=cbar_midpoint,
            bounds=bounds, cbar_values=cbar_values,
            discretize=discretize, palette=palette, fontsize=fontsize, edgecolor=edgecolor,
            font=font, vmax=vmax, vmin=vmin, markersize=markersize, alpha=alpha,
            start_cbar_ticks=start_cbar_ticks, end_cbar_ticks=end_cbar_ticks)

def plot_map_HM_by_station(hydro_sim_points_gdf_simplified, dict_shapefiles, bounds, path_global_figures,
                           fontsize=14):
    shape_hp = {
        'CTRIP': 'D',
        'EROS': 'H',
        'GRSD': '*',
        'J2000': 's',
        'MORDOR-SD': 'v',
        'MORDOR-TS': '^',
        'ORCHIDEE': 'o',
        'SIM2': '>',
        'SMASH': '<',
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
                              path_global_figures, bounds, station_references=None,fontsize=14):
    if station_references is None:
        station_references_plot = {
            'M842001000': {'s':90, 'edgecolors':'k', 'zorder':10, 'linewidth': 1.5,
                           'facecolors':'none',
                           'text': {'text': 'La Loire à\nSt Nazaire', 'arrowprops':dict(arrowstyle='-'),
                                    'xytext':(0.1, 0.8), 'textcoords':'axes fraction', 'ha':'center'}
                           },
            'M530001010': {'s':90, 'edgecolors':'k', 'zorder':10, 'linewidth': 1.5,
                           'facecolors':'none',
                           'text': {'text': 'La Loire à \nMontjean', 'ha':'center', 'arrowprops':dict(arrowstyle='-'),
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
    else:
        station_references_plot = {}
        i = 25
        idx = -1
        coord_couples = [(-i, -i), (-i, i), (i, i), (i, -i)]
        alphabet = [chr(i) for i in range(65, 65 + len(station_references))]
        count_idx = -1
        for key, value in station_references.items():
            idx += 1
            count_idx += 1
            if idx == len(coord_couples):
                idx = 0
            # station_references_plot |= {key: {'s':90, 'edgecolors':'k', 'zorder':10, 'linewidth': 1.5,
            #                                   'facecolors':'none',
            #                                   'text': {'text': ' '.join(value.split(' ')[:3]) + '\n' + ' '.join(value.split(' ')[3:]),
            #                                            'xytext': coord_couples[idx],
            #                                            'arrowprops':dict(arrowstyle='-', connectionstyle="arc3,rad=.2"),
            #                                            'textcoords': 'offset points'
            #                                            }
            #                                   }
            #                             }
            station_references_plot |= {key: {'s':90, 'edgecolors':'k', 'zorder':10, 'linewidth': 1.5,
                                              'facecolors':'none', 'label': value,
                                              'text': {'text': alphabet[count_idx],
                                                       'xytext': coord_couples[idx],
                                                       'arrowprops':dict(arrowstyle='-'),
                                                       'textcoords': 'offset points'
                                                       }
                                              }
                                        }


    for key in station_references_plot.keys():
        station_references_plot[key] |= {'x': hydro_sim_points_gdf_simplified.loc[key].geometry.x,
                                    'y': hydro_sim_points_gdf_simplified.loc[key].geometry.y}
        station_references_plot[key]['text'] |= {'xy': (hydro_sim_points_gdf_simplified.loc[key].geometry.x,
                                                   hydro_sim_points_gdf_simplified.loc[key].geometry.y)}

    print(f"> Plot Number of HM by station...")
    j = -1
    for key in dict_shapefiles.keys():
        j += 1
        dict_shapefiles[key]['alpha'] = 0.2
        dict_shapefiles[key]['zorder'] = -j

    mapplot(gdf=hydro_sim_points_gdf_simplified, indicator_plot='n', path_result=path_global_figures+'count_HM.pdf', ds=None,
            cols=None, rows=None, references=station_references_plot, cbar_ticks='mid',  cbar_values=1,
            cbar_title=f"Nombre de HM", title=None, dict_shapefiles=dict_shapefiles, percent=False, bounds=bounds,
            discretize=6, palette='RdBu_r', fontsize=fontsize-10, font='sans-serif', edgecolor='k',
            cbar_midpoint='min', vmin=3.5, vmax=9.5)
