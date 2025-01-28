import pyfiglet
ascii_banner = pyfiglet.figlet_format("FORMAT NCDF")
print(f'##########################################################################################', end='\n')
print(ascii_banner, end='\n')

print(f'################################ IMPORT & INITIALIZATION ################################', end='\n')

print(f'> General imports...', end='\n')
import sys
import os
import copy
# sys.path.insert(0, os.getcwd())
import time
import json

print(f'> Local imports...', end='\n')
from global_functions.load_data import *
from plot_functions.run_plot import *
from global_functions.format_data import *
from global_functions.shp_geometry import *
from global_functions.path_functions import  *

# Avoid crash with console when launched manually
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.switch_backend('agg')

# Load environments variables
print(f'> Load json inputs...', end='\n')
with open('config.json') as config_file:
    config = json.load(config_file)

with open('files_setup.json') as files_setup:
    files_setup = json.load(files_setup)

print(f'> Define paths...', end='\n')
dict_paths = define_paths(config)

#%% Files names
# Study folder
print(f'> Create output directories...', end='\n')
if not os.path.isdir(dict_paths['folder_study_results']):
    os.makedirs(dict_paths['folder_study_results'])

# Study figures folder
if not os.path.isdir(dict_paths['folder_study_figures']):
    os.makedirs(dict_paths['folder_study_figures'])

# Study data folder
if not os.path.isdir(dict_paths['folder_study_data']):
    os.makedirs(dict_paths['folder_study_data'])

if not os.path.isdir(dict_paths['folder_study_data'] + 'shapefiles'):
    os.makedirs(dict_paths['folder_study_data'] + 'shapefiles')

#%% LOAD STUDY REGION SHAPEFILE
print(f'################################ DEFINE STUDY AREA ################################', end='\n')
print(f'> Load shapefiles...', end='\n')
regions_shp, study_hydro_shp, study_climate_shp, rivers_shp = load_shp(dict_paths, files_setup)

# Check if study area is already matched with sim points
print(f'> Searching sim points in study area...', end='\n')
with open('reference_stations.json') as ref_stations:
    reference_stations = json.load(ref_stations)

flatten_reference_stations = {key: value for subdict in reference_stations.values() for key, value in subdict.items()}

for data_type, path in dict_paths['dict_study_points_sim'].items():
    if not os.path.isfile(path):
        print(f'>> Find {data_type} data points in study area')
        sim_all_points_info = open_shp(path_shp=dict_paths['dict_global_points_sim'][data_type])
        if data_type == 'hydro':
            overlay_shapefile(shapefile=study_hydro_shp, data=sim_all_points_info,
                              path_result=path, force_contains={'Suggesti_2': ['LA LOIRE', 'L\'ALLIER'],
                                                                'Suggestion': flatten_reference_stations.keys()})
        else:
            overlay_shapefile(shapefile=study_climate_shp, data=sim_all_points_info,
                              path_result=path)
    else:
        print(f'>> {data_type.capitalize()} data points already in the study area')

print(f'> Simplify shapefiles...', end='\n')
study_hydro_shp_simplified, study_climate_shp_simplified, study_rivers_shp_simplified, regions_shp_simplified, bounds = (
    simplify_shapefiles(study_hydro_shp, study_climate_shp, rivers_shp, regions_shp, tolerance=1000, zoom=1000))

print(f'################################ RUN OVER NCDF ################################', end='\n')
# Get paths for selected sim
print(f'> Load ncdf data paths...', end='\n')
path_files = get_files_path(dict_paths=dict_paths, setup=files_setup)

# Run among data type climate/hydro
data_type='climate'
subdict=path_files[data_type]
rcp='rcp85'
subdict2=subdict[rcp]
indicator = "tasAdjust"
paths = subdict2[indicator]

hydro_sim_points_gdf = open_shp(path_shp=dict_paths['dict_study_points_sim']['hydro'])
hydro_sim_points_gdf_simplified = hydro_sim_points_gdf[hydro_sim_points_gdf['n'] >= 4]
hydro_sim_points_gdf_simplified = hydro_sim_points_gdf_simplified.reset_index(drop=True).set_index('Suggestion')
hydro_sim_points_gdf_simplified.index.names = ['name']

climate_sim_points_gdf = open_shp(path_shp=dict_paths['dict_study_points_sim']['climate'])
climate_sim_points_gdf_simplified = climate_sim_points_gdf.loc[
    climate_sim_points_gdf.groupby('name')['gid'].idxmin()].reset_index(drop=True)

for data_type, subdict in path_files.items():
    # Load simulation points for current data type
    # sim_points_gdf = open_shp(path_shp=dict_paths['dict_study_points_sim'][data_type])

    if data_type == "hydro":
        sim_points_gdf_simplified = hydro_sim_points_gdf_simplified
    else:
        sim_points_gdf_simplified = climate_sim_points_gdf_simplified
        # sim_points_gdf['weight'] = sim_points_gdf['surface'] / sim_points_gdf['total_surf']

    for rcp, subdict2 in subdict.items():
        for indicator, paths in subdict2.items():
            print(f'################################ RUN {data_type} {rcp} {indicator} ################################', end='\n')
            for name_indicator, settings_dict in files_setup[f'{data_type}_indicator'][indicator].items():
                timestep = 'YE'
                function = None
                if 'timestep' in settings_dict:
                    timestep = settings_dict['timestep']

                if 'extract_function' in settings_dict:
                    function = settings_dict['extract_function']

                name_join = name_indicator.replace(" ", "-").replace(".", "")
                path_ncdf = f"{dict_paths['folder_study_data']}{name_join}_{rcp}_{timestep}.nc"

                if not os.path.isfile(path_ncdf):
                    print(f'> Create {indicator} export...', end='\n')
                    if len(paths) > 0 :
                        extract_ncdf_indicator(
                            paths_data=paths, param_type=data_type, sim_points_gdf=sim_points_gdf_simplified,
                            indicator=indicator, function=function, timestep=timestep,
                            start=files_setup['historical'][0], path_result=path_ncdf,
                        )
                    else:
                        print(f'> Invalid {indicator} name', end='\n')
                else:
                    print(f'> {path_ncdf} already exists', end='\n')

name = 'tasAdjust'
data_to_plot = {name: files_setup['climate_indicator'][name]}
# data_to_plot = (files_setup['climate_indicator'] | files_setup['hydro_indicator'])
overwrite = True
for indicator, subdicts in data_to_plot.items():
    for name_indicator, indicator_setup in subdicts.items():
        print(f'################################ STATS {name_indicator.upper()} ################################', end='\n')
        # if name_indicator.upper() == "NJ<0":
        #     break
        if overwrite:
            write_fig = True
        else:
            write_fig = False

        # Get plot settings
        settings = load_settings(indicator_setup)

        # Define variable name for axis
        if settings['label'] is None:
            title = name_indicator
        else:
            title = settings['label']

        if settings['units'] != '':
            units = f" ({settings['units']})"
        else:
            units = ''

        plot_type = settings['plot_type']
        start_cbar_ticks = ''
        end_cbar_ticks = ''
        if plot_type == 'difference':
            plot_type_name = 'difference'
        else:
            plot_type_name = 'variation'
            units = " (%)"

        if settings['start_cbar_ticks'] != '':
            start_cbar_ticks = f" {settings['start_cbar_ticks']}"
        if settings['end_cbar_ticks'] != '':
            end_cbar_ticks = f" {settings['end_cbar_ticks']}"

        # Create folder
        title_join = name_indicator.replace(" ", "-").replace(".", "")
        path_indicator = dict_paths['folder_study_figures'] + title_join + os.sep
        if not os.path.isdir(path_indicator):
            os.makedirs(path_indicator)
            write_fig = True

        if write_fig:
            path_ncdf = f"{dict_paths['folder_study_data']}{title_join}_{rcp}_{settings['timestep']}.nc"

            # Compute PK
            if indicator in files_setup['hydro_indicator']:
                data_type = 'hydro'
                sim_points_gdf_simplified = hydro_sim_points_gdf_simplified
                # loire = sim_points_gdf_simplified.loc[sim_points_gdf_simplified['gid'] < 7]
                loire = sim_points_gdf_simplified[(sim_points_gdf_simplified['Suggesti_2'].str.contains('LA LOIRE ', case=False, na=False))]
                value = compute_river_distance(rivers_shp, loire, river_name='loire',
                                               start_from='last')
                hydro_sim_points_gdf_simplified["PK"] = value
                # sim_points_gdf_simplified.loc[sim_points_gdf_simplified['gid'] < 7, 'PK'] = value
                edgecolor = 'k'
            else:
                data_type = 'climate'
                sim_points_gdf_simplified = climate_sim_points_gdf_simplified
                sim_points_gdf_simplified = sim_points_gdf_simplified.set_index('name')
                edgecolor = None

            # Open ncdf dataset
            ds_stats = xr.open_dataset(path_ncdf)

            # Compute stats
            ds_stats, variables = format_dataset(ds=ds_stats, data_type=data_type, files_setup=files_setup,
                                                 plot_function=settings['plot_function'],
                                                 return_period=settings['return_period'])

            # Geodataframe
            sim_points_gdf_simplified = sim_points_gdf_simplified.loc[ds_stats.gid]
            dict_shapefiles = define_plot_shapefiles(regions_shp_simplified, study_climate_shp_simplified, study_rivers_shp_simplified,
                                   indicator, files_setup)

            # Check for additional coordinates
            used_coords = set()
            for var in ds_stats.data_vars:
                used_coords.update(ds_stats[var].dims)

            # Use another dimension to create different plots (ex: one plot per month)
            additional_coordinates = {'': [None]}
            if settings['additional_coordinates'] is not None:
                additional_coordinates = {settings['additional_coordinates']: ds_stats[settings['additional_coordinates']].values}

            for coordinate, unique_value in additional_coordinates.items():
                for coordinate_value in unique_value:
                    print(f'################################ PLOT {name_indicator.upper()}{coordinate_value if coordinate_value is not None else ""} '
                          f'################################', end='\n')
                    # Selection from the current coordinate value
                    if coordinate_value is not None:
                        ds = ds_stats.sel({coordinate: coordinate_value})

                        path_indicator_figures = path_indicator + coordinate_value + os.sep
                        if not os.path.isdir(path_indicator_figures):
                            os.makedirs(path_indicator_figures)
                    else:
                        ds = copy.deepcopy(ds_stats)
                        path_indicator_figures = path_indicator

                    print(f"> Map plot...")
                    print(f">> {plot_type_name.title()} matching map plot {indicator}")
                    plot_map_indicator_climate(gdf=sim_points_gdf_simplified, ds=ds, indicator_plot='horizon_matching',
                                               path_result=path_indicator_figures+f'{title_join}_map_matching_sims.pdf',
                                               cbar_title=f"{title_join} Accord des modèles sur le sens d'évolution (%)", cbar_ticks=None, title=coordinate_value, dict_shapefiles=dict_shapefiles,
                                               bounds=bounds, palette='PuOr', cbar_midpoint='zero', cbar_values=settings['cbar_values'],
                                               start_cbar_ticks=settings['start_cbar_ticks'], end_cbar_ticks=settings['end_cbar_ticks'],
                                               fontsize=settings['fontsize'],
                                               font=settings['font'], discretize=settings['discretize'], edgecolor=edgecolor, markersize=75,
                                               vmin=-100, vmax=100)
                    # Climate difference map
                    if indicator in files_setup['climate_indicator']:
                        print(f">> {plot_type_name.title()} map plot {indicator}")
                        plot_map_indicator_climate(gdf=sim_points_gdf_simplified, ds=ds, indicator_plot=f'horizon_{plot_type}-median',
                                      path_result=path_indicator_figures+f'{title_join}_map_{plot_type}.pdf',
                                      cbar_title=f"{plot_type_name.title()} médiane {title}{units}", cbar_ticks=settings['cbar_ticks'],
                                                   title=coordinate_value, dict_shapefiles=dict_shapefiles,
                                      bounds=bounds, palette=settings['palette'], cbar_midpoint='zero', cbar_values=settings['cbar_values'],
                                      start_cbar_ticks=settings['start_cbar_ticks'], end_cbar_ticks=settings['end_cbar_ticks'],
                                      fontsize=settings['fontsize'],
                                      font=settings['font'], discretize=settings['discretize'], edgecolor=edgecolor, markersize=75,
                                      vmin=settings['vmin'], vmax=settings['vmax'])

                        # Histogramme Différence par moyenne multi-modèle annuelle par rapport à la période de référence
                        # timeline_difference_mean mais pour l'ensemble du territoire
                    elif indicator in files_setup['hydro_indicator']:
                        print(f">> {plot_type_name.title()} map plot by HM")
                        horizons = {'horizon1': 'Horizon 1 (2021-2050)',
                                    'horizon2': 'Horizon 2 (2041-2070)',
                                    'horizon3': 'Horizon 3 (2070-2099)',
                                    }
                        mean_by_hm = [s for sublist in variables['hydro-model_deviation'].values() for s in sublist if "mean" in s]
                        if settings['vmax'] is None:
                            vmax = math.ceil(abs(ds[mean_by_hm].to_array()).max() / 5) * 5
                        else:
                            vmax = settings['vmax']

                        for key, value in horizons.items():
                            print(f">>> Map {value}")
                            if coordinate_value is not None:
                                map_title = f"{value}: {coordinate_value} "
                            else:
                                map_title = f"{value}"
                            plot_map_indicator_hm(gdf=sim_points_gdf_simplified, ds=ds.sel(horizon=key), variables=variables,
                                                  path_result=path_indicator_figures+f'{title_join}_map_{plot_type}_median_{key}.pdf',
                                                  cbar_title=f"{plot_type_name.title()} médiane {title}{units}", title=map_title, cbar_midpoint='zero',
                                                  dict_shapefiles=dict_shapefiles, bounds=bounds, edgecolor=edgecolor,
                                                  markersize=75, discretize=settings['discretize'], palette=settings['palette'], fontsize=settings['fontsize'],
                                                  font=settings['font'],
                                                  vmin=settings['vmin'], vmax=vmax)

                        # Sim by PK + quantile
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

                            narratives = {
                                "HadGEM2-ES_ALADIN63_ADAMONT": {'color': '#569A71', 'zorder': 10, 'label': 'Vert [HadGEM2-ES_ALADIN63_ADAMONT]',
                                                                'linewidth': 1},
                                "CNRM-CM5_ALADIN63_ADAMONT": {'color': '#EECC66', 'zorder': 10, 'label': 'Jaune [CNRM-CM5_ALADIN63_ADAMONT]',
                                                              'linewidth': 1},
                                "EC-EARTH_HadREM3-GA7_ADAMONT": {'color': '#E09B2F', 'zorder': 10, 'label': 'Orange [EC-EARTH_HadREM3-GA7_ADAMONT]',
                                                                 'linewidth': 1},
                                "HadGEM2-ES_CCLM4-8-17_ADAMONT": {'color': '#791F5D', 'zorder': 10, 'label': 'Violet [HadGEM2-ES_CCLM4-8-17_ADAMONT]',
                                                                  'linewidth': 1},
                            }

                            print(f">> Linear deviation - x: PK, y: {indicator}, row: HM, col: Horizon")
                            plot_linear_pk_hm(ds,
                                              simulations=variables['hydro-model_sim-horizon_deviation'],
                                              narratives=narratives,
                                              title=coordinate_value,
                                              name_x_axis=f'PK (km)',
                                              name_y_axis=f'{plot_type_name.title()} {title}{units}',
                                              percent=True,
                                              vlines=vlines,
                                              fontsize=settings['fontsize'],
                                              font=settings['font'],
                                              path_result=path_indicator_figures+f'lineplot_{plot_type}_x-PK_y-{title_join}_row-HM_col-horizon.pdf')


                            print(f">> Linear deviation - x: PK, y: {indicator}, row: Narratif, col: Horizon")
                            plot_linear_pk_narrative(ds,
                                                     simulations=variables['simulation-horizon_by-sims_deviation'],
                                                     narratives=narratives,
                                                     title=coordinate_value,
                                                     name_x_axis=f'PK (km)',
                                                     name_y_axis=f'{plot_type_name.title()} {title}{units}',
                                                     percent=True,
                                                     vlines=vlines,
                                                     fontsize=settings['fontsize'],
                                                     font=settings['font'],
                                                     path_result=path_indicator_figures+f'lineplot_{plot_type}_x-PK_y-{title_join}_row-narrative_col-horizon.pdf')


                            print(f">> Linear deviation - x: PK, y: {indicator}, col: Horizon")
                            plot_linear_pk(ds,
                                           simulations=variables['simulation-horizon_by-sims_deviation'],
                                           narratives=narratives,
                                           title=coordinate_value,
                                           name_x_axis=f'PK (km)',
                                           name_y_axis=f'{plot_type_name.title()} {title}{units}',
                                           percent=True,
                                           vlines=vlines,
                                           fontsize=settings['fontsize'],
                                           font=settings['font'],
                                           path_result=path_indicator_figures+f'lineplot_{plot_type}_x-PK_y-{title_join}_col-horizon.pdf')

                            for river, river_stations in reference_stations.items():
                                print(f">> Linear timeline deviation - x: time, y: {indicator}, row/col: Stations ref")
                                plot_linear_time(ds,
                                                 simulations=variables['simulation_deviation'],
                                                 station_references=river_stations,
                                                 narratives=narratives,
                                                 title=coordinate_value,
                                                 name_x_axis='Date',
                                                 name_y_axis=f'{plot_type_name.title()} {title}{units}',
                                                 percent=True,
                                                 vlines=None,
                                                 fontsize=settings['fontsize'],
                                                 font=settings['font'],
                                                 path_result=path_indicator_figures+f'lineplot_{plot_type}_{river}_x-time_y-{title_join}_row-col-stations-ref.pdf',)


                                print(f"> Box plot...")
                                print(f">> Boxplot deviation by horizon and selected stations")
                                plot_boxplot_station_narrative(ds,
                                                               simulations=variables['simulation-horizon_by-sims_deviation'],
                                                               station_references=river_stations,
                                                               narratives=narratives,
                                                               title=coordinate_value,
                                                               references=None,
                                                               name_y_axis=f'{plot_type_name.title()} {title}{units}',
                                                               percent=True,
                                                               fontsize=settings['fontsize'],
                                                               font=settings['font'],
                                                               path_result=path_indicator_figures+f'{title_join}_boxplot_{plot_type}_{river}_narratives.pdf',)


# print(f'################################ PLOT GLOBAL ################################', end='\n')
# path_global_figures = dict_paths['folder_study_figures'] + 'global' + os.sep
# if not os.path.isdir(path_global_figures):
#     os.makedirs(path_global_figures)
#
# print(f"> Plot HM by station...")
# plot_map_HM_by_station(hydro_sim_points_gdf_simplified, dict_shapefiles, bounds, path_global_figures,
#                        fontsize=settings['fontsize']+2)
#
# print(f"> Plot #HM by station and Ref station...")
# plot_map_N_HM_ref_station(hydro_sim_points_gdf_simplified, dict_shapefiles, path_global_figures, bounds,
#                           station_references=flatten_reference_stations, fontsize=settings['fontsize'])


########################################################################################################################
#
# # Cols  and rows of subplots
# cols = {'names_var': 'id_geometry', 'values_var': ['K001872200', 'M850301010'], 'names_plot': ['Station 1', 'Station 2']}
# rows = {
#     'names_var': row_name,
#     'values_var': list(value.keys()),
#     'names_plot': list(value.values())
# }
#
# y_axis = {
#     'values_var': ['QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_CTRIP',
#                    'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_EROS',
#                    'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_GRSD'],
#     'names_plot': ['QA CTRIP', 'QA EROS', 'QA GRSD']
# }
# x_axis = {
#     'names_var': 'horizon',
#     'values_var': ['horizon1', 'horizon2', 'horizon3'],
#     'names_plot': ['H1', 'H2', 'H3']
# }
#
# # TEST 1
# cols = {
#     'names_var': 'id_geometry',
#     'values_var': ['K001872200', 'M850301010'],
#     'names_plot': ['K001872200', 'M850301010'],
# }
# rows = {
#     'names_var': row_name,
#     'values_var': list(value.keys()),
#     'names_plot': list(value.values())
# }
# x_axis = {
#     'names_var': 'horizon',
#     'values_var': ['horizon1', 'horizon2', 'horizon3'],
#     'names_plot': ['H1', 'H2', 'H3']
# }
# y_axis = {
#     'names_var': 'indicator',
#     'values_var': ['QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_CTRIP',
#                    'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_EROS',
#                    'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_GRSD'],
#     'names_plot': ['QA CTRIP', 'QA EROS', 'QA GRSD']
# }
#
# # TEST 2
# cols = {'names_var': 'horizon',
#         'values_var': ['horizon1', 'horizon2', 'horizon3'],
#         'names_plot': ['H1', 'H2', 'H3']}
# rows = {
#     'names_var': 'indicator',
#     'values_var': ['QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_CTRIP',
#                    'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_EROS',
#                    'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_GRSD'],
#     'names_plot': ['QA CTRIP', 'QA EROS', 'QA GRSD']
# }
# x_axis = {
#     'names_var': row_name,
#     'values_var': list(value.keys()),
#     'names_plot': list(value.values())
# }
# y_axis = {
#     'names_var': 'id_geometry',
#     'values_var': ['K001872200', 'M850301010'],
#     'names_plot': ['K001872200', 'M850301010'],
# }
#
# # TEST 3
# cols = {'names_var': 'indicator',
#         'values_var': ['QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_CTRIP',
#                        'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_EROS',
#                        'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_GRSD'],
#         'names_plot': ['QA CTRIP', 'QA EROS', 'QA GRSD']}
# rows = {
#     'names_var': 'horizon',
#     'values_var': ['horizon1', 'horizon2', 'horizon3'],
#     'names_plot': ['H1', 'H2', 'H3']
# }
# x_axis = {
#     'names_var': 'id_geometry',
#     'values_var': ['K001872200', 'M850301010'],
#     'names_plot': ['K001872200', 'M850301010'],
#     'name_axis': 'Stations'
#
# }
# y_axis = {
#     'names_var': row_name,
#     'values_var': list(value.keys()),
#     'names_plot': list(value.values()),
#     'name_axis': indicator + ' (m3/s)'
# }
#
# boxplot(ds, x_axis, y_axis, path_result=path_indicator_figures+'boxplot.pdf', cols=cols, rows=rows,
#         title=None, percent=False, palette='BrBG', fontsize=14, font='sans-serif', ymax=None)



print(f'################################ END ################################', end='\n')
input("Press Enter to close")