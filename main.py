print(f'################################ IMPORT & INITIALIZATION ################################', end='\n')

print(f'> General imports...', end='\n')
import sys
import os
# sys.path.insert(0, os.getcwd())
import time
import json
# import pyfiglet
# ascii_banner = pyfiglet.figlet_format("Hello")
# print(ascii_banner)

print(f'> Local imports...', end='\n')
from global_functions.load_data import *
from global_functions.format_data import *
from plot_functions.plot_map import *
from plot_functions.plot_lineplot import *
from plot_functions.plot_boxplot import *
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
regions_shp, study_ug_shp, study_ug_bv_shp, rivers_shp = load_shp(dict_paths, files_setup)

# Check if study area is already matched with sim points
print(f'> Searching sim points in study area...', end='\n')
for data_type, path in dict_paths['dict_study_points_sim'].items():
    if not os.path.isfile(path):
        print(f'>> Find {data_type} data points in study area')
        sim_all_points_info = open_shp(path_shp=dict_paths['dict_global_points_sim'][data_type])
        if data_type == 'hydro':
            overlay_shapefile(shapefile=study_ug_shp, data=sim_all_points_info,
                              path_result=path)
        else:
            overlay_shapefile(shapefile=study_ug_bv_shp, data=sim_all_points_info,
                              path_result=path)
    else:
        print(f'>> {data_type.capitalize()} data points already in the study area')

print(f'> Simplify shapefiles...', end='\n')
study_ug_shp_simplified, study_ug_bv_shp_simplified, study_rivers_shp_simplified, regions_shp_simplified, bounds = (
    simplify_shapefiles(study_ug_shp, study_ug_bv_shp, rivers_shp, regions_shp, tolerance=1000, zoom=50000))

print(f'################################ RUN OVER NCDF ################################', end='\n')
# Get paths for selected sim
print(f'> Load ncdf data paths...', end='\n')
path_files = get_files_path(dict_paths=dict_paths, setup=files_setup)

# Run among data type climate/hydro
data_type='hydro'
subdict=path_files[data_type]
rcp='rcp85'
subdict2=subdict[rcp]
indicator = 'QA_mon'
paths = subdict2[indicator]
for data_type, subdict in path_files.items():
    # Load simulation points for current data type
    sim_points_gdf = open_shp(path_shp=dict_paths['dict_study_points_sim'][data_type])
    if data_type == "hydro":
        # sim_points_gdf = sim_points_gdf[sim_points_gdf['REFERENCE'] == 1]
        sim_points_gdf = sim_points_gdf[sim_points_gdf['n'] >= 4]
        valid_stations = pd.isna(sim_points_gdf['PointsSupp'])
        sim_points_gdf = sim_points_gdf[valid_stations].reset_index(drop=True).set_index('Suggestion')
        sim_points_gdf.index.names = ['name']
    else:
        sim_points_gdf['weight'] = sim_points_gdf['surface'] / sim_points_gdf['total_surf']

    for rcp, subdict2 in subdict.items():
        for indicator, paths in subdict2.items():
            # split_indicator = indicator_raw.split('-')
            # indicator = split_indicator[0]
            timestep = 'ME'
            # if len(split_indicator) > 1:
            #     timestep = split_indicator[1]
            #
            # if timestep == 'mon':
            #     timestep = 'M'

            path_ncdf = f"{dict_paths['folder_study_data']}{indicator.split('$')[0]}_{timestep}_{rcp}.nc"

            if not os.path.isfile(path_ncdf):
                print(f'> Create {indicator} export...', end='\n')
                if len(paths) > 0 :
                    extract_ncdf_indicator(
                        paths_data=paths, param_type=data_type, sim_points_gdf=sim_points_gdf, indicator=indicator,
                        timestep=timestep, start=files_setup['historical'][0], path_result=path_ncdf,
                    )

            print(f'################################ FORMAT DATA ################################', end='\n')
            print(f'> Load from {indicator} export...', end='\n')
            # path_ncdf = f"{dict_paths['folder_study_data']}QA_mon_M_rcp85.nc"
            # indicator='QA'
            ds = xr.open_dataset(path_ncdf)
            indicator_cols = [i for i in list(ds.variables) if indicator in i]

            # Define geometry for each data (Points hydro, Polygon climate)
            print(f'> Match geometry and data...', end='\n')
            other_dimension = None
            if data_type == 'climate':
                sim_all_points_info = open_shp(path_shp=dict_paths['dict_global_points_sim'][data_type])
                ds = ds.assign_coords(geometry=(
                    'name', sim_all_points_info.set_index('name').loc[ds['name'].values, 'geometry']))
                ds = ds.rename({'name': 'id_geometry'})

                # Find matching area
                geometry_dict = {row['gid']: row['geometry'] for _, row in study_ug_shp.iterrows()}
                region_da = xr.DataArray(sim_points_gdf['gid'].values, dims=['name'],
                                         coords={'name': sim_points_gdf['name']})
                ds = ds.assign_coords(region=region_da)
                # ds = ds.assign_coords(geometry=('region', [geometry_dict[code] for code in ds['region'].values]))
                # ds = ds.rename({'region': 'id_geometry'})
            else:
                sim_points_gdf_simplified = sim_points_gdf.copy()
                sim_points_gdf_simplified = sim_points_gdf_simplified.simplify(tolerance=1000, preserve_topology=True)
                geometry_dict = sim_points_gdf['geometry'].to_dict()
                ds['geometry'] = ('code', [
                    geometry_dict[code] if code in geometry_dict.keys() else None for code in ds['code'].values
                ])
                # ds = ds.assign_coords(geometry=('code', [geometry_dict[code] for code in ds['code'].values]))
                ds = ds.rename({'code': 'id_geometry'})

                if indicator == 'QA':
                    # other_dimension = {'time': 'time.month'}
                    ds = ds.assign_coords(month=ds['time.month'])
                    other_dimension = 'month'

            print(f'> Define horizons...', end='\n')
            # Define horizons
            ds = define_horizon(ds, files_setup)

            # Compute mean value for each horizon for each sim
            ds = compute_mean_by_horizon(ds=ds, indicator_cols=indicator_cols,
                                         files_setup=files_setup, other_dimension=other_dimension)

            indicator_horizon = [i for i in list(ds.variables) if '_by_horizon' in i]

            # Compute statistic among all sims
            ds_mean_spatial_horizon = apply_statistic(ds=ds[indicator_horizon].to_array(dim='new'),
                                                      function=files_setup['function'],
                                                      q=files_setup['quantile']
                                                      )
            indicator_statistics = [f"{indicator}_{i}" for i in list(ds_mean_spatial_horizon.data_vars)]
            ds[indicator_statistics] = ds_mean_spatial_horizon

            # ds[indicator+'_by_horizon_among_sims'] = ds_mean_spatial_horizon

            # Compute deviation to historical
            ds = compute_deviation_to_ref(ds, cols=indicator_horizon + indicator_statistics)

            indicator_horizon_deviation = [i for i in list(ds.variables) if
                                           indicator+'_by_horizon_among_sims_deviation' in i]
            indicator_horizon_difference = [i for i in list(ds.variables) if
                                            indicator+'_by_horizon_among_sims_difference' in i]
            indicator_horizon_deviation_sims = [i for i in list(ds.variables) if '_by_horizon_deviation' in i]
            indicator_horizon_difference_sims = [i for i in list(ds.variables) if '_by_horizon_difference' in i]

            print(f'################################ PLOT DATA ################################', end='\n')
            print(f"> Initialize plot...")
            # col_name='horizon'
            # col_headers = {'horizon1': 'Horizon 1 (2021-2050)',
            #                'horizon2': 'Horizon 2 (2041-2070)',
            #                'horizon3': 'Horizon 3 (2070-2100)'}
            #
            row_name = None

            subplots = {'': None}
            discretize = 7
            vmax = None
            percent = True
            if indicator == 'QA':
                vmax = 100
                row_name = 'month'
                discretize = 11
                subplots = {
                    'DJF' : {12: 'Décembre',
                             1: 'Janvier',
                             2: 'Février'},
                    'MAM': {3: 'Mars',
                            4: 'Avril',
                            5: 'Mai'},
                    'JJA': {6: 'Juin',
                            7: 'Juillet',
                            8: 'Août'},
                    'SON': {9: 'Septembre',
                            10: 'Octobre',
                            11: 'Novembre'},
                }

            gdf = gpd.GeoDataFrame({
                'geometry': ds['geometry'].values,
                'id_geometry': ds['id_geometry'].values
            })

            dict_shapefiles = {'rivers_shp': {'shp': study_rivers_shp_simplified, 'color': 'paleturquoise',
                                              'linewidth': 1, 'zorder': 20, 'alpha': 1},
                               'background_shp': {'shp': regions_shp_simplified, 'color': 'gainsboro',
                                                  'edgecolor': 'black', 'zorder': 0},
                               }
            if data_type == 'hydro':
                dict_shapefiles |= {'study_shp': {'shp': study_ug_shp_simplified, 'color': 'white',
                                                 'edgecolor': 'k', 'zorder': 1, 'linewidth': 1.2},}
            else:
                dict_shapefiles |= {'study_shp': {'shp': study_ug_shp_simplified, 'color': 'none',
                                                 'edgecolor': 'k', 'zorder': 1, 'linewidth': 1.2},}

            print(f"> Map plot...")
            path_indicator_figures = dict_paths['folder_study_figures'] + indicator + os.sep
            if not os.path.isdir(path_indicator_figures):
                os.makedirs(path_indicator_figures)

            for key, value in subplots.items():
                if key is None:
                    path_results = f"{path_indicator_figures}{timestep}_{rcp}_"
                else:
                    path_results = f"{path_indicator_figures}{timestep}_{rcp}_{key}_"
                cols_map = {
                    'names_var': 'horizon',
                    'values_var': ['horizon1', 'horizon2', 'horizon3'],
                    'names_plot': ['Horizon 1 (2021-2050)', 'Horizon 2 (2041-2070)', 'Horizon 3 (2070-2100)']
                }

                rows = {
                    'names_var': row_name,
                    'values_var': list(value.keys()),
                    'names_plot': list(value.values())
                }

                rows = None

                # Plot map
                # Relative
                print(f"> Relative map plot {indicator}_{timestep}_{rcp}_{key}...")
                mapplot(gdf, ds, indicator_plot=indicator_horizon_deviation[0], path_result=path_indicator_figures+'map_deviation.pdf',
                        cols=cols_map, rows=rows,
                        cbar_title=f"{indicator} relatif (%)", title=None, dict_shapefiles=dict_shapefiles, percent=True, bounds=bounds,
                        discretize=discretize, palette='BrBG', fontsize=14, font='sans-serif', vmax=100)

                # Abs diff
                print(f"> Difference map plot {indicator}_{timestep}_{rcp}_{key}...")
                mapplot(gdf, ds, indicator_plot=indicator_horizon_difference[0], path_result=path_indicator_figures+'map_difference.pdf',
                        cols=cols_map, rows=rows,
                        cbar_title=f"{indicator} difference", title=None, dict_shapefiles=dict_shapefiles, percent=False, bounds=bounds,
                        discretize=4, palette='RdBu_r', cmap_zero=True, fontsize=14, font='sans-serif', edgecolor=None, vmin=0, vmax=4)


                print(f"> Relative line plot {indicator}_{timestep}_{rcp}_{key}...")


                x_axis = {
                    'names_var': 'time',
                    'values_var': 'time',
                    'name_axis': 'Date'
                }
                y_axis = {
                    'names_var': 'indicator',
                    'values_var': indicator_cols,
                    'name_axis': 'QA'
                }

                rows = {'names_var': 'id_geometry',
                        'values_var': ds['id_geometry'].where(ds['geometry'].notnull(), drop=True).values,
                        'names_plot': ds['id_geometry'].where(ds['geometry'].notnull(), drop=True).values}
                cols = {
                    'names_var': row_name,
                    'values_var': list(value.keys()),
                    'names_plot': list(value.values())
                }

                lineplot(ds, x_axis, y_axis, path_result=path_indicator_figures+'lineplot.pdf', cols=cols, rows=rows,
                         title=None, percent=False, fontsize=14, font='sans-serif', ymax=None, plot_type='line')


                # Cols  and rows of subplots
                cols = {'names_var': 'id_geometry', 'values_var': ['K001872200', 'M850301010'], 'names_plot': ['Station 1', 'Station 2']}
                rows = {
                    'names_var': row_name,
                    'values_var': list(value.keys()),
                    'names_plot': list(value.values())
                }

                y_axis = {
                    'values_var': ['QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_CTRIP',
                                  'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_EROS',
                                  'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_GRSD'],
                    'names_plot': ['QA CTRIP', 'QA EROS', 'QA GRSD']
                }
                x_axis = {
                    'names_var': 'horizon',
                    'values_var': ['horizon1', 'horizon2', 'horizon3'],
                    'names_plot': ['H1', 'H2', 'H3']
                }

                # TEST 1
                cols = {
                    'names_var': 'id_geometry',
                        'values_var': ['K001872200', 'M850301010'],
                        'names_plot': ['K001872200', 'M850301010'],
                }
                rows = {
                    'names_var': row_name,
                    'values_var': list(value.keys()),
                    'names_plot': list(value.values())
                }
                x_axis = {
                    'names_var': 'horizon',
                    'values_var': ['horizon1', 'horizon2', 'horizon3'],
                    'names_plot': ['H1', 'H2', 'H3']
                }
                y_axis = {
                    'names_var': 'indicator',
                    'values_var': ['QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_CTRIP',
                                   'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_EROS',
                                   'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_GRSD'],
                    'names_plot': ['QA CTRIP', 'QA EROS', 'QA GRSD']
                }

                # TEST 2
                cols = {'names_var': 'horizon',
                        'values_var': ['horizon1', 'horizon2', 'horizon3'],
                        'names_plot': ['H1', 'H2', 'H3']}
                rows = {
                    'names_var': 'indicator',
                    'values_var': ['QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_CTRIP',
                                   'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_EROS',
                                   'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_GRSD'],
                    'names_plot': ['QA CTRIP', 'QA EROS', 'QA GRSD']
                }
                x_axis = {
                    'names_var': row_name,
                    'values_var': list(value.keys()),
                    'names_plot': list(value.values())
                }
                y_axis = {
                    'names_var': 'id_geometry',
                    'values_var': ['K001872200', 'M850301010'],
                    'names_plot': ['K001872200', 'M850301010'],
                }

                # TEST 3
                cols = {'names_var': 'indicator',
                        'values_var': ['QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_CTRIP',
                                       'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_EROS',
                                       'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_GRSD'],
                        'names_plot': ['QA CTRIP', 'QA EROS', 'QA GRSD']}
                rows = {
                    'names_var': 'horizon',
                    'values_var': ['horizon1', 'horizon2', 'horizon3'],
                    'names_plot': ['H1', 'H2', 'H3']
                }
                x_axis = {
                    'names_var': 'id_geometry',
                    'values_var': ['K001872200', 'M850301010'],
                    'names_plot': ['K001872200', 'M850301010'],
                    'name_axis': 'Stations'

                }
                y_axis = {
                    'names_var': row_name,
                    'values_var': list(value.keys()),
                    'names_plot': list(value.values()),
                    'name_axis': indicator + ' (m3/s)'
                }

                boxplot(ds, x_axis, y_axis, path_result=path_indicator_figures+'boxplot.pdf', cols=cols, rows=rows,
                         title=None, percent=False, palette='BrBG', fontsize=14, font='sans-serif', ymax=None)

