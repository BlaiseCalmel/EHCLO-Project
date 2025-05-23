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
from global_functions.format_data import *

def plot_climate(ds, sim_points_gdf_simplified, horizons, narratives, settings, coordinate_value, path_indicator_figures):
    print(f"{settings['name_indicator']} >> {settings['plot_type_name'].title()} map plot")
    plot_map_indicator(gdf=sim_points_gdf_simplified, ds=ds, indicator_plot=f'horizon_{settings["plot_type"]}-median',
                        path_result=path_indicator_figures+f"{settings['title_join']}_map_{settings['plot_type']}.pdf", horizons=horizons,
                        cbar_title=f"{settings['plot_type_name'].title()} {settings['function_name']} {settings['title']}{settings['units']}", 
                        cbar_ticks=settings['cbar_ticks'],
                        title=coordinate_value, dict_shapefiles=settings['dict_shapefiles'],
                        bounds=settings['bounds'], palette=settings['palette'], cbar_midpoint='zero', cbar_values=settings['cbar_values'],
                        start_cbar_ticks=settings['start_cbar_ticks'], end_cbar_ticks=settings['end_cbar_ticks'],
                        fontsize=settings['fontsize']-2, alpha=1,
                        font=settings['font'], discretize=settings['discretize'], edgecolor=settings['edgecolor'], markersize=75,
                        vmin=settings['vmin'], vmax=settings['vmax'], uncertainty='horizon_matching')
    if len(narratives) == 1:
        # Climate Narratives
        if len(list(list(narratives.values())[0].keys())[0].split("_")) == 3:
            if 'season' in ds.coords:
                for key, value in horizons.items():
                    print(f"{settings['name_indicator']} >>>  {settings['plot_type_name'].title()} narratives map plot {value}")
                    plot_map_narratives(gdf=sim_points_gdf_simplified, ds=ds.sel(horizon=key), narratives=narratives, 
                        variables=variables[f"simulation-horizon_by-sims_{settings['plot_type']}"],
                        path_result=path_indicator_figures+f"{settings['title_join']}_narrative_map_{key}.pdf",
                        cbar_title=f"{settings['plot_type_name'].title()} {settings['function_name']} {settings['title']}{settings['units']}", 
                        cbar_ticks=settings['cbar_ticks'],
                        title=value, dict_shapefiles=settings['dict_shapefiles'],
                        bounds=settings['bounds'], palette=settings['palette'], cbar_midpoint='zero', cbar_values=settings['cbar_values'],
                        start_cbar_ticks=settings['start_cbar_ticks'], end_cbar_ticks=settings['end_cbar_ticks'],
                        fontsize=settings['fontsize'], alpha=1,
                        font=settings['font'], discretize=settings['discretize'], edgecolor=settings['edgecolor'], markersize=75,
                        vmin=settings['vmin'], vmax=settings['vmax'])
            else:
                print(f"{settings['name_indicator']} >>>  {settings['plot_type_name'].title()} narratives map plot")
                plot_map_narratives(gdf=sim_points_gdf_simplified, ds=ds, narratives=narratives, 
                    variables=variables[f"simulation-horizon_by-sims_{settings['plot_type']}"],
                    path_result=path_indicator_figures+f"{settings['title_join']}_narrative_map.pdf",
                    cbar_title=f"{settings['plot_type_name'].title()} {settings['function_name']} {settings['title']}{settings['units']}", 
                    cbar_ticks=settings['cbar_ticks'],
                    title=coordinate_value, dict_shapefiles=settings['dict_shapefiles'],
                    bounds=settings['bounds'], palette=settings['palette'], cbar_midpoint='zero', cbar_values=settings['cbar_values'],
                    start_cbar_ticks=settings['start_cbar_ticks'], end_cbar_ticks=settings['end_cbar_ticks'],
                    fontsize=settings['fontsize'], alpha=1,
                    font=settings['font'], discretize=settings['discretize'], edgecolor=settings['edgecolor'], markersize=75,
                    vmin=settings['vmin'], vmax=settings['vmax'])           

def plot_hydro_narraclimate(ds, variables, sim_points_gdf_simplified, horizons, narratives, 
                            settings, coordinate_value, path_indicator_figures, reference_stations):
    median_by_hm = [s for sublist in variables[f"hydro-model_{settings['plot_type']}"].values() for s in sublist if "median" in s]
    sim_points_gdf_simplified['S_HYDRO'][np.isnan(sim_points_gdf_simplified['S_HYDRO'])] = 0
    label_df = sim_points_gdf_simplified['S_HYDRO'].astype(int).astype(str) + 'km² [' + sim_points_gdf_simplified['n'].astype(str) + 'HM]'
    # Map per horizon
    for key, value in horizons.items():
        print(f"{settings['name_indicator']} >>> Narratives map {value}")
        if coordinate_value is not None:
            map_title = f"{value}: {coordinate_value} "
        else:
            map_title = f"{value}"
            plot_map_indicator_narratives(gdf=sim_points_gdf_simplified, ds=ds.sel(horizon=key),
                                            narratives=narratives, variables=variables, plot_type=settings['plot_type'],
                                            path_result=path_indicator_figures+f"{settings['title_join']}_map_{settings['plot_type']}_narratives_{key}.pdf",
                                            cbar_title=f"{settings['plot_label'].title()} {settings['function_name']} {settings['title']}{settings['units']}", 
                                            title=map_title,
                                            cbar_midpoint='zero',
                                            dict_shapefiles=settings['dict_shapefiles'], bounds=settings['bounds'], edgecolor=settings['edgecolor'],
                                            markersize=170, discretize=settings['discretize'], palette=settings['palette'],
                                            fontsize=settings['fontsize'],
                                            font=settings['font'], alpha=settings['alpha'],
                                            vmin=settings['vmin'], vmax=settings['vmax'])

    print(f"> Linear plot...")
    if 'PK' in sim_points_gdf_simplified.columns:
        ds = ds.assign(PK=("gid", sim_points_gdf_simplified.loc[ds.gid.values, "PK"]))

        villes = ['Villerest', 'Nevers', 'Orleans', 'Tours', 'Saumur', 'Nantes'] #'Blois',
        regex = "|".join(villes)
        vlines = sim_points_gdf_simplified[sim_points_gdf_simplified['Suggesti_2'].str.contains(regex, case=False, na=False)]
        vlines.loc[: ,'color'] = 'none'
        cities = [i.split(' A ')[-1].split(' [')[0] for i in vlines['Suggesti_2']]
        vlines.insert(loc=0, column='label', value=cities)
        vlines['annotate'] = 0.02
        vlines['fontsize'] = settings['fontsize'] - 2

        # Limit size of y axis label
        name_y_axis = optimize_label_length(f"{settings['plot_label'].title()} {settings['title']}{settings['units']}", settings)

        print(f"{name_indicator} >> Linear {plot_type} PK for Narrative & Horizon")
        plot_linear_pk(ds,
                        simulations=variables[f"simulation-horizon_by-sims_{settings['plot_type']}"],
                        narratives=narratives,
                        horizons=horizons,
                        title=coordinate_value,
                        name_x_axis=f'PK (km)',
                        name_y_axis=name_y_axis,
                        percent=settings['percent'],
                        vlines=vlines,
                        fontsize=settings['fontsize'],
                        font=settings['font'],
                        path_result=path_indicator_figures+f"{settings['title_join']}_lineplot_{settings['plot_type']}_PK_narratives_horizon.pdf")
    
    for river, river_stations in reference_stations.items():
        extended_station_name = {key : f"{value}: {label_df.loc[key]}" for key, value in river_stations.items()}
        for key, value in extended_station_name.items():
            extended_station_name[key] = optimize_label_length(value, settings, length=30)
        
        print(f"{name_indicator} >> Strip plot {settings['plot_type']} for {river} selected stations with narratives")

        plot_boxplot_station_narrative(ds=ds[variables[f"simulation-horizon_by-sims_{settings['plot_type']}"]],
                                        station_references=extended_station_name,
                                        narratives=narratives,
                                        title=coordinate_value,
                                        references=None,
                                        name_y_axis=name_y_axis,
                                        percent=settings['percent'],
                                        fontsize=settings['fontsize'],
                                        font=settings['font'],
                                        path_result=path_indicator_figures+f"{settings['title_join']}_boxplot_{settings['plot_type']}_stations-{river}_horizons_narratives.pdf",)

        # plot_boxplot_station_narrative_tracc(   ds=ds[variables[f"simulation-horizon_by-sims_{settings['plot_type']}"]],
        #                                         horizons=horizons,
        #                                         station_references=extended_station_name,
        #                                         narratives=narratives,
        #                                         title=coordinate_value,
        #                                         name_y_axis=name_y_axis,
        #                                         percent=SETTINGS['percent'],
        #                                         fontsize=settings['fontsize'],
        #                                         font=settings['font'],
        #                                         path_result=path_indicator_figures+f"{settings['title_join']}_boxplot_{settings['plot_type']}_stations-{river}_horizons_narratives.pdf",)
                                    
def plot_hydro_narrahydro(ds, variables, sim_points_gdf_simplified, horizons, narratives, 
                          settings, coordinate_value, path_indicator_figures, reference_stations):
    median_by_hm = [s for sublist in variables[f"hydro-model_{settings['plot_type']}"].values() for s in sublist if "median" in s]
    sim_points_gdf_simplified.loc[np.isnan(sim_points_gdf_simplified['S_HYDRO'])]['S_HYDRO'] = 0
    label_df = sim_points_gdf_simplified['S_HYDRO'].astype(int).astype(str) + 'km² [' + sim_points_gdf_simplified['n'].astype(str) + 'HM]'
    # Map per horizon
    for key, value in horizons.items():
        print(f"{settings['name_indicator']} >>> Map {value}")
        if coordinate_value is not None:
            map_title = f"{value}: {coordinate_value} "
        else:
            map_title = f"{value}"
        plot_map_indicator_narratives(gdf=sim_points_gdf_simplified, 
                                      ds=ds.sel(horizon=key),
                                      narratives=narratives, 
                                      variables=variables, 
                                      plot_type=settings['plot_type'],
                                      path_result=path_indicator_figures+f"{settings['title_join']}_map_{settings['plot_type']}_narratives_{key}.pdf",
                                      cbar_title=f"{settings['plot_label'].title()} {settings['function_name']} {settings['title']}{settings['units']}", 
                                      title=map_title,
                                      cbar_midpoint='zero',
                                      dict_shapefiles=settings['dict_shapefiles'], 
                                      bounds=settings['bounds'], 
                                      edgecolor=settings['edgecolor'],
                                      markersize=170, 
                                      discretize=settings['discretize'], 
                                      palette=settings['palette'],
                                      fontsize=settings['fontsize'],
                                      font=settings['font'], 
                                      alpha=settings['alpha'],
                                      vmin=settings['vmin'], 
                                      vmax=settings['vmax'])

    horizon_name = list(horizons.keys())[0]
    print(f"> Linear plot...")
    if 'PK' in sim_points_gdf_simplified.columns:
        ds = ds.assign(PK=("gid", sim_points_gdf_simplified.loc[ds.gid.values, "PK"]))

        villes = ['Villerest', 'Nevers', 'Orleans', 'Tours', 'Saumur', 'Nantes'] #'Blois',
        regex = "|".join(villes)
        vlines = sim_points_gdf_simplified[sim_points_gdf_simplified['Suggesti_2'].str.contains(regex, case=False, na=False)]
        vlines.loc[: ,'color'] = 'none'
        cities = [i.split(' A ')[-1].split(' [')[0] for i in vlines['Suggesti_2']]
        vlines.insert(loc=0, column='label', value=cities)
        vlines['annotate'] = 0.02
        vlines['fontsize'] = settings['fontsize'] - 2

        # Limit size of y axis label
        name_y_axis = optimize_label_length(f"{settings['plot_label'].title()} {settings['function_name']} {settings['title']}{settings['units']}", 
                                            settings)

        plot_linear_pk(ds,
                       simulations=variables[f"simulation-horizon_by-sims_{settings['plot_type']}"],
                       narratives=narratives,
                       horizons=horizons,
                       title=coordinate_value,
                       name_x_axis=f'PK (km)',
                       name_y_axis=name_y_axis,
                       percent=settings['percent'],
                       vlines=vlines,
                       fontsize=settings['fontsize'],
                       font=settings['font'],
                       path_result=path_indicator_figures+f"{settings['title_join']}_lineplot_{settings['plot_type']}_PK_narratives_{horizon_name}.pdf")
                                            
        # plot_linear_pk_narrative(ds,
        #                             simulations=variables[f"simulation-horizon_by-sims_{settings['plot_type']}"],
        #                             narratives=narratives,
        #                             title=coordinate_value,
        #                             name_x_axis=f'PK (km)',
        #                             name_y_axis=name_y_axis,
        #                             percent=settings['percent'],
        #                             vlines=vlines,
        #                             fontsize=settings['fontsize'],
        #                             font=settings['font'],
        #                             path_result=path_indicator_figures+f"{settings['title_join']}_lineplot_{settings['plot_type']}_PK_horizon.pdf")

    # Plot for selected hydro stations per river
    name_y_axis = optimize_label_length(f"{settings['plot_label'].title()} {settings['title']}{settings['units']}", settings)
    for river, river_stations in reference_stations.items():
        extended_station_name = {key : f"{value}: {label_df.loc[key]}" for key, value in river_stations.items()}
        for key, value in extended_station_name.items():
            extended_station_name[key] = optimize_label_length(value, settings, length=30)

        print(f"{settings['name_indicator']} >> Boxplot {settings['plot_type']} [{river}] by horizon and selected stations")
        plot_boxplot_station_narrative_tracc(ds,
                                            variables=variables,
                                            settings=settings,
                                            #  ds=ds[variables[f"simulation_{settings['plot_type']}"]].sel(time=getattr(ds, list(horizons.keys())[0])), 
                                            #  references = {'simulations': {f"{sim}_{settings['plot_type']}": copy.deepcopy(args) 
                                            #                 for narr in narratives.values() for sim, args in narr.items()}},
                                            references = {'simulations': {f"{sim}_{settings['plot_type']}": copy.deepcopy(args) 
                                                            for narr in narratives.values() for sim, args in narr.items()}},
                                             station_references=extended_station_name, 
                                             narratives=narratives, 
                                             horizons=horizons, 
                                             name_y_axis=name_y_axis, 
                                             percent=settings['percent'],
                                             title=list(horizons.values())[0], 
                                             fontsize=settings['fontsize'], 
                                             font=settings['font'],
                                             path_result=path_indicator_figures+f"{settings['title_join']}_boxplot_{settings['plot_type']}_stations-{river}_narratives_{horizon_name}.pdf",
                                             )

        print(f"{settings['name_indicator']} >> Linear timeline {settings['plot_type']} [{river}] for selected stations")
        plot_linear_time(ds=ds.sel(time=getattr(ds, list(horizons.keys())[0])),
                         station_references=extended_station_name, 
                         simulations=variables['simulation_deviation'], 
                         path_result=path_indicator_figures+f"{settings['title_join']}_timeline_{settings['plot_type']}_stations-{river}_narratives_{horizon_name}.pdf", 
                         narratives=narratives,
                         name_x_axis='Années TRACC', 
                         name_y_axis=name_y_axis, 
                         percent=settings['percent'], 
                         vlines=None, 
                         title=list(horizons.values())[0],
                         fontsize=settings['fontsize'], 
                         font=settings['font'],
                         )

def plot_monthly(ds_stats, variables, sim_points_gdf_simplified, horizons, narratives, 
                 settings, path_indicator, reference_stations):
    label_df = sim_points_gdf_simplified['S_HYDRO'].astype(int).astype(str) + 'km² [' + sim_points_gdf_simplified['n'].astype(str) + 'HM]'

    horizon_boxes = {
        "historical": {'color': '#f5f5f5', 'zorder': 10, 'label': 'Historique (1991-2020)',
                        'linewidth': 1}
    }
    color_boxes = {'horizon1': '#80cdc1',  'horizon2': '#dfc27d', 'horizon3': '#a6611a'}
    horizon_boxes |= {key: {'zorder': 10, 'label': val,
                        'linewidth': 1, 'color': color_boxes[key]} for key, val in horizons.items()}

    for river, river_stations in reference_stations.items():
        extended_station_name = {key : f"{value}: {label_df.loc[key]}" for key, value in river_stations.items()}
        for key, value in extended_station_name.items():
            extended_station_name[key] = optimize_label_length(value, settings, length=28)
        # print(f"> Box plot...")
        # print(f">> Boxplot normalized {settings['title_join']} by month and horizon")
        # name_y_axis = optimize_label_length(f"{settings['title_join']} normalisé", settings)
        # plot_boxplot_station_month_horizon(ds=ds_stats[variables['simulation_horizon']],
        #                                     station_references=extended_station_name,
        #                                     narratives=horizon_boxes,
        #                                     title=None,
        #                                     name_y_axis=name_y_axis,
        #                                     normalized=True,
        #                                     percent=False,
        #                                     common_yaxes=True,
        #                                     fontsize=settings['fontsize'],
        #                                     font=settings['font'],
        #                                     path_result=path_indicator+f"{settings['title_join']}_boxplot_normalized_{river}_month.pdf")
        
        # print(f">> Boxplot {settings['plot_type']} by month and horizon")
        # name_y_axis = optimize_label_length(f"{settings['plot_type_name'].title()} {settings['title']}{settings['units']}", settings,
        #                                     length=18)

        # plot_boxplot_station_month_horizon(ds=ds_stats[variables[f"simulation-horizon_by-sims_{settings['plot_type']}"]],
        #                                     station_references=extended_station_name,
        #                                     narratives={key: value for key, value in horizon_boxes.items() if key!='historical'},
        #                                     title=None,
        #                                     name_y_axis=name_y_axis,
        #                                     percent=settings['percent'],
        #                                     common_yaxes=True,
        #                                     ymin=settings['vmin'],
        #                                     ymax=settings['vmax'],
        #                                     fontsize=settings['fontsize'],
        #                                     font=settings['font'],
        #                                     path_result=path_indicator+f"{settings['title_join']}_boxplot_{settings['plot_type']}_{river}_month.pdf")

        for key, value in horizons.items():
            print(f"{settings['name_indicator']} >> Linear {settings['plot_type']} month per station {value}")
            name_y_axis = optimize_label_length(f"{settings['plot_label'].title()} {settings['title']}{settings['units']}", settings,
                                            length=18)
            plot_linear_month(ds=ds_stats,
                            station_references=extended_station_name,
                            simulations=variables[f"simulation-horizon_by-sims_{settings['plot_type']}"],
                            horizon=key,
                            narratives=narratives,
                            title=value,
                            name_x_axis=f'Mois',
                            name_y_axis=name_y_axis,
                            percent=settings['percent'],
                            vlines=None,
                            fontsize=settings['fontsize'],
                            font=settings['font'],
                            path_result=path_indicator+f"{settings['title_join']}_lineplot_{settings['plot_type']}_{river}_month_narratives_{key}.pdf")
            
            print(f"{settings['name_indicator']} >> Linear {settings['title_join']} month per station {value}")
            name_y_axis = optimize_label_length(f"{settings['title_join']} (m3/s)", settings,
                                            length=18)
            plot_linear_month(ds=ds_stats,
                              station_references=extended_station_name,
                              simulations=variables[f"simulation_horizon"],
                              references=ds_stats[[f"horizon_value-median"]].sel(horizon='historical'),
                              horizon=key,
                              narratives=narratives,
                              title=value,
                              name_x_axis=f'Mois',
                              name_y_axis=name_y_axis,
                              percent=False,
                              vlines=None,
                              fontsize=settings['fontsize'],
                              font=settings['font'],
                              common_axes=False,
                              path_result=path_indicator+f"{settings['title_join']}_lineplot_value_{river}_month_narratives_{key}.pdf")


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

def plot_linear_pk(ds, simulations, path_result, horizons, narratives=None,
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
    legend_items += [value for narr_value in narratives.values() for key, value in narr_value.items()]
    for leg in legend_items:
        if leg['zorder'] > 1:
            leg['label'] = leg['label'][0]
    # Suppression des doublons en convertissant les dictionnaires en tuples immuables
    unique_legend_items = []
    for i in legend_items:
        if i not in unique_legend_items:
            unique_legend_items.append(i)
    # legend_items = [value for value in narratives.values()]

    cols = {
        'names_coord': 'horizon',
        'values_var': [key for key in horizons.keys()],
        'names_plot': [val for val in horizons.values()]
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
              'name_axis': name_x_axis,
              'names_plot': np.arange(1, len(ds.time)+1)
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
             title=title, percent=percent, legend_items=legend_items, fontsize=fontsize, font=font, ymax=None,
             remove_xticks=False, common_axes=True)

def plot_linear_month(ds, simulations, path_result, station_references, horizon, references=None, narratives=None,
                     name_x_axis='', name_y_axis='', percent=False, vlines=None, title=None,
                     fontsize=14, font='sans-serif', common_axes=True):

    ds_plot = ds.sel(horizon=horizon)
    x_axis = {'names_coord': 'month',
              'name_axis': name_x_axis,
              'names_plot': ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'],
              'names_sorted': ds.month.values.tolist()
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
    legend_items += [{'color': 'k', 'alpha': 1, 'zorder': 20, 'label': 'Référence', 'linestyle': ':', 'linewidth': 2}]
    legend_items += [i for narr_value in narratives.values() for i in narr_value.values()]

    # if references is not None:
    #     references = {i: {'color': 'k', 'alpha': 1, 'zorder': 20, 'label': 'Référence', 'linestyle': ':', 'linewidth': 2} 
    #                 for i in references}

    indicator_plot = [dict_sim for i in range(len(station_references))]

    rows = {
        'names_coord': 'gid',
        'values_var': list(station_references.keys()),
        'names_plot': list(station_references.values())
    }

    cols = 2

    lineplot(ds_plot, indicator_plot, x_axis, y_axis, path_result=path_result, cols=cols, rows=rows, vlines=vlines,
             title=title, percent=percent, legend_items=legend_items, fontsize=fontsize, font=font, ymax=None,
             common_axes=common_axes, remove_xticks=False, references=references)


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

def plot_boxplot_station_narrative_tracc(ds, references, station_references, variables, settings, narratives, path_result, horizons, 
                                         name_y_axis='', percent=False, title=None, fontsize=14, font='sans-serif'):

    # simulations = list(ds.data_vars)
    simulations = variables[f"simulation-horizon_by-sims_{settings['plot_type']}"]

    narratives_bp = {key: {'color': value['color'], 'alpha': 0.9, 'zorder': 1, 'label': value['label'][0], 'linewidth': 2} 
                  for narr_value in narratives.values() for key, value in narr_value.items()}
    
    dict_sims = {}
    dict_sims['simulations'] = {'values': simulations, 'kwargs': {'color': 'lightgray', 'alpha': 0.35, 'zorder': 0,
                                                                  'linewidth': 1, 'label': 'Ensemble des projections'}}
    
    narr_simulations_timeline = variables[f"simulation_{settings['plot_type']}"]
    references = {'simulations': {}}
    for narr_name, kwargs in narratives_bp.items():
        dict_sims[narr_name] = {'values': [], 'kwargs': kwargs}
        references['simulations'][f"{narr_name}_by-horizon_{settings['plot_type']}"] = kwargs
        for sim_name in narr_simulations_timeline:
            if narr_name in sim_name:
                dict_sims[narr_name]['values'].append(sim_name)
    
    # for key, value in references.items():
    #     for sim_name, args in value.items():
    #         args['function'] = 'median'

    # references = {}
    # for n in narratives.values():
    #     for narr_name, kwargs in n.items():
    #         # dict_sims[narr_name] = {'values': [], 'kwargs': kwargs}
    #         for sim_name in simulations:
    #             if narr_name in sim_name:
    #                 references[sim_name] = {'color': kwargs['color'], 'alpha': 0.9, 'zorder': 10, 'label': kwargs['label'], 'linewidth': 3} 
                                                                  
    horizons_tracc = {'horizon1': '+2.0°C', 'horizon2': '+2.7°C', 'horizon3': '+4.0°C'}    
    selected_horizons = {key: value for key, value in horizons_tracc.items() if key in horizons.keys()}                                                       

    # indicator_plot = [dict_sims]
    y_axis = {'names_coord': 'indicator',
              'values_var': dict_sims, #indicator_plot
              'values_flatten': [i for val in dict_sims.values() for i in val['values'] ],
              'names_plot': list(dict_sims.keys()),
              'name_axis': name_y_axis
              }

    x_axis = {
        'names_coord': 'horizon',
        'values_var': list(selected_horizons.keys()),
        'names_plot': [None]
    }
    # x_axis = {
    #     'names_coord': None,
    #     'values_var': [None],
    #     'names_plot': [None]
    # }

    cols = 2
    rows = {
        'names_coord': 'gid',
        'values_var': list(station_references.keys()),
        'names_plot': list(station_references.values())
    }

    boxplot(ds, x_axis, y_axis, path_result=path_result, references=references, cols=cols, rows=rows, vlines=False, 
             title=title, percent=percent, fontsize=fontsize, font=font, blank_space=0.1, strip=True, common_yaxes=True)

def plot_boxplot_station_month_horizon(ds, station_references, narratives, path_result, name_y_axis='', percent=False,
                                       title=None, fontsize=14, font='sans-serif', common_yaxes=False, normalized=False,
                                       ymin=None, ymax=None):

    narratives_bp = {}
    narratives_bp |= {key: {'type': 'histo', 'kwargs': {
        'label': value['label'], 'zorder': 1, 'color':'lightgray', 'edgecolor':'k', 'alpha':1}} for key, value in
                      narratives.items() if key == "historical"}
    narratives_bp |= {key: {'kwargs': {'boxprops': dict(facecolor=value['color'], alpha=0.8),
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
            font=font, vmax_user=vmax, vmin_user=vmin, markersize=markersize, alpha=alpha)

def plot_map_indicator_narratives(gdf, ds, narratives, path_result, cbar_title, dict_shapefiles, bounds,
                                  variables, plot_type, discretize=None, palette='BrBG', fontsize=14, font='sans-serif', title=None,
                                  vmin=None, vmax=None, edgecolor='k', cbar_midpoint=None, markersize=50, alpha=1):

    narr = [key for narr_values in narratives.values() for key in narr_values.keys()]
    # stats_by_narr = [sim for sim in variables[f'simulation-horizon_by-sims_{plot_type}'] if any(n in sim for n in narr)]
    stats_by_narr = [n+f'_by-horizon_{plot_type}' for n in narr]

    # rows = {
    #     'names_coord': 'indicator',
    #     'values_var': stats_by_narr,
    #     'names_plot': [v['label'] for narr in narratives.values() for v in narr.values()]
    # }
    rows = {
        'names_coord': 'indicator',
        'values_var': stats_by_narr,
        'names_plot': [{'label': v['label'][0], 'color': v['color']} for narr in narratives.values() for v in narr.values()]
    }
    cols = 2

    mapplot(gdf=gdf, ds=ds, indicator_plot=stats_by_narr,
            path_result=path_result,
            cols=cols, rows=rows,
            cbar_title=cbar_title, title=title, dict_shapefiles=dict_shapefiles, cbar_midpoint=cbar_midpoint,
            bounds=bounds,
            discretize=discretize, palette=palette, fontsize=fontsize, edgecolor=edgecolor,
            font=font, vmax_user=vmax, vmin_user=vmin, markersize=markersize, alpha=alpha)

def plot_map_indicator(gdf, ds, path_result, horizons, cbar_title, dict_shapefiles, bounds,
                       indicator_plot, cbar_ticks=None, discretize=None, palette='BrBG', fontsize=14,
                       font='sans-serif', title=None, vmin=None, vmax=None, edgecolor='k',
                       cbar_midpoint=None, markersize=50, alpha=1, cbar_values=None,
                       start_cbar_ticks='', end_cbar_ticks='', uncertainty=None):

    cols = {
            'names_coord': 'horizon',
            'values_var': [key for key in horizons.keys()],
            'names_plot': [val for val in horizons.values()]
             }
    # cols = {
    #         'names_coord': 'horizon',
    #         'values_var': ['horizon1', 'horizon2', 'horizon3'],
    #         'names_plot': ['Horizon 1\n(2021-2050)', 'Horizon 2\n(2041-2070)', 'Horizon 3\n(2070-2099)']
    #          }

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
            font=font, vmax_user=vmax, vmin_user=vmin, markersize=markersize, alpha=alpha,
            start_cbar_ticks=start_cbar_ticks, end_cbar_ticks=end_cbar_ticks, uncertainty=uncertainty)

def plot_map_narratives(gdf, ds, narratives, variables, path_result, cbar_title, dict_shapefiles, bounds,
                               cbar_ticks=None, discretize=None, palette='BrBG', fontsize=14,
                               font='sans-serif', title=None, vmin=None, vmax=None, edgecolor='k',
                               cbar_midpoint=None, markersize=50, alpha=1, cbar_values=None,
                               start_cbar_ticks='', end_cbar_ticks=''):

    narr = [key for narr_values in narratives.values() for key in narr_values.keys()]
    stats_by_narr = [sim for sim in variables if any(n in sim for n in narr)]
    
    # rows = {
    #     'names_coord': 'indicator',
    #     'values_var': stats_by_narr,
    #     'names_plot': [v['label'].split(' ')[0] for narr in narratives.values() for v in narr.values()]
    #     }

    rows = {
        'names_coord': 'indicator',
        'values_var': stats_by_narr,
        'names_plot': [{'label': v['label'].split(' ')[0], 'color': v['color']} for narr in narratives.values() for v in narr.values()]
    }

    used_coords = [dim for dim in ds[stats_by_narr].dims if dim in ds.coords]
    if 'season' in used_coords:
        cols = {
            'names_coord': 'season',
            'values_var': ['Hiver', 'Printemps', 'Été', 'Automne'],
            'names_plot': ['Hiver', 'Printemps', 'Été', 'Automne']
        }
    else:
        # cols = 2
        cols = {
            'names_coord': 'horizon',
            'values_var': ['horizon1', 'horizon2', 'horizon3'],
            'names_plot': ['Horizon 1\n(2021-2050)', 'Horizon 2\n(2041-2070)', 'Horizon 3\n(2070-2099)']
             }

    mapplot(gdf=gdf, ds=ds, indicator_plot=stats_by_narr,
            path_result=path_result,
            cols=cols, rows=rows, cbar_ticks=cbar_ticks,
            cbar_title=cbar_title, title=title, dict_shapefiles=dict_shapefiles, cbar_midpoint=cbar_midpoint,
            bounds=bounds, cbar_values=cbar_values,
            discretize=discretize, palette=palette, fontsize=fontsize, edgecolor=edgecolor,
            font=font, vmax_user=vmax, vmin_user=vmin, markersize=markersize, alpha=alpha,
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
            vmin_user=-0.5, vmax_user=1.5, markersize=75,
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
        loire_value = 0
        allier_value = 0
        chapeauroux_value = 0
        for key, value in station_references.items():
            idx += 1
            count_idx += 1
            if idx == len(coord_couples):
                idx = 0
            # station_references_plot |= {key: {'s':90, 'edgecolors':'k', 'zorder':10, 'linewidth': 1.5,
            #                                   'facecolors':'none', 'label': value,
            #                                   'text': {'text': alphabet[count_idx],
            #                                            'xytext': coord_couples[idx],
            #                                            'arrowprops':dict(arrowstyle='-'),
            #                                            'textcoords': 'offset points'
            #                                            }
            #                                   }
            #                             }
            if "Loire" in value:
                loire_value += 1
                my_text = f"L{loire_value}"
                edgecolors = 'gray'
                zorder = 9
            elif "Allier" in value:
                allier_value += 1
                my_text = f"A{allier_value}"
                edgecolors = 'green'
                zorder = 8
            else:
                chapeauroux_value += 1
                my_text = f"C{chapeauroux_value}"
                edgecolors = 'cornflowerblue'
                zorder = 7

            station_references_plot |= {key: {'s':90, 'edgecolors':edgecolors, 'zorder':zorder, 'linewidth': 5,
                                              'facecolors':'none', 'label': value,
                                              # 'text': {'text': my_text,
                                              #          'xytext': coord_couples[idx],
                                              #          'textcoords': 'offset points',
                                              #          'arrowprops':dict(arrowstyle='-'),
                                              #          'zorder': 20
                                              #          }
                                              }
                                        }

    for key in station_references_plot.keys():
        station_references_plot[key] |= {'x': hydro_sim_points_gdf_simplified.loc[key].geometry.x,
                                    'y': hydro_sim_points_gdf_simplified.loc[key].geometry.y}
        # station_references_plot[key]['text'] |= {'xy': (hydro_sim_points_gdf_simplified.loc[key].geometry.x,
        #                                            hydro_sim_points_gdf_simplified.loc[key].geometry.y)}

    print(f"> Plot Number of HM by station...")
    j = -1
    for key in dict_shapefiles.keys():
        j += 1
        dict_shapefiles[key]['alpha'] = 0.2
        dict_shapefiles[key]['zorder'] = -j

    mapplot(gdf=hydro_sim_points_gdf_simplified, indicator_plot='n', path_result=path_global_figures+'count_HM.pdf', ds=None,
            cols=None, rows=None, references=station_references_plot, cbar_ticks='mid', cbar_values=1,
            cbar_title=f"Nombre\nde HM", title=None, dict_shapefiles=dict_shapefiles, bounds=bounds,
            discretize=9, palette='RdBu_r', fontsize=fontsize-4, font='sans-serif', edgecolor='k',
            cbar_midpoint='min', vmin_user=2, vmax_user=9, markersize=90)