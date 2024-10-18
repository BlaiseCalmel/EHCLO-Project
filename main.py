print(f'################################ IMPORT & INITIALIZATION ################################', end='\n')

print(f'General imports...', end='\n')
import time
import json
# import pyfiglet
# ascii_banner = pyfiglet.figlet_format("Hello")
# print(ascii_banner)

print(f'Local imports...', end='\n')
from global_functions.load_data import *
from global_functions.format_data import *
from plot_functions.plot_data import *
from global_functions.shp_geometry import *
from global_functions.path_functions import  *

# Avoid crash with console when launched manually
import matplotlib
matplotlib.use('TkAgg')
plt.switch_backend('agg')

# Load environments variables
print(f'Load json inputs...', end='\n')
with open('config.json') as config_file:
    config = json.load(config_file)

with open('files_setup.json') as files_setup:
    files_setup = json.load(files_setup)

print(f'Define paths...', end='\n')
dict_paths = define_paths(config)

#%% Files names
# Study folder
print(f'Create output directories...', end='\n')
if not os.path.isdir(dict_paths['folder_study_results']):
    os.makedirs(dict_paths['folder_study_results'])

# Study figures folder
if not os.path.isdir(dict_paths['folder_study_figures']):
    os.makedirs(dict_paths['folder_study_figures'])

# Study data folder
if not os.path.isdir(dict_paths['folder_study_data']):
    os.makedirs(dict_paths['folder_study_data'])

#%% LOAD STUDY REGION SHAPEFILE
print(f'################################ STUDY AREA ################################', end='\n')
print(f'Load shapefiles...', end='\n')
regions_shp = open_shp(path_shp=dict_paths['file_regions_shp'])
study_regions_shp = regions_shp[regions_shp['gid'].isin(files_setup['regions'])]
rivers_shp = open_shp(path_shp=dict_paths['file_rivers_shp'])

# Check if study area is already matched with sim points
print(f'Find sim points in study area...', end='\n')
for data_type, path in dict_paths['dict_study_points_sim'].items():
    if not os.path.isfile(path):
        print(f'Find {data_type} data points in study area')
        sim_all_points_info = open_shp(path_shp=dict_paths['dict_global_points_sim'][data_type])
        overlay_shapefile(shapefile=study_regions_shp, data=sim_all_points_info,
                          path_result=path)
    else:
        print(f'{data_type.capitalize()} data points already in the study area')


print(f'################################ RUN OVER NCDF ################################', end='\n')
# Get paths for selected sim
print(f'Load ncdf data paths...', end='\n')
path_files = get_files_path(dict_paths=dict_paths, setup=files_setup)

# Run among data type climate/hydro
start_run = time.time()
total_iterations = len(path_files.keys())

for data_type, subdict in path_files.items():
    # Load simulation points for current data type
    sim_points_gdf = open_shp(path_shp=dict_paths['dict_study_points_sim'][data_type])
    if data_type == "hydro":
        sim_points_gdf = sim_points_gdf[sim_points_gdf['REFERENCE'] == 1]
        valid_stations = pd.isna(sim_points_gdf['PointsSupp'])
        sim_points_gdf = sim_points_gdf[valid_stations].reset_index(drop=True).set_index('Suggestion')
        sim_points_gdf.index.names = ['name']
    else:
        sim_points_gdf['weight'] = sim_points_gdf['surface'] / sim_points_gdf['total_surf']

    for rcp, subdict2 in subdict.items():
        for indicator_raw, paths in subdict2.items():
            split_indicator = indicator_raw.split('-')
            indicator = split_indicator[0]
            timestep = 'YE'
            if len(split_indicator) > 1:
                timestep = split_indicator[1]

            path_ncdf = f"{dict_paths['folder_study_data']}{indicator}_{timestep}_{rcp}.nc"

            if not os.path.isfile(path_ncdf):
                print(f'Create {indicator} export...', end='\n')
                extract_ncdf_indicator(
                    paths_data=paths, param_type=data_type, sim_points_gdf=sim_points_gdf,
                    indicator=indicator, timestep=timestep, start=files_setup['historical'][0], path_result=path_ncdf,
                )

            print(f'################################ FORMAT DATA ################################', end='\n')
            print(f'Load from {indicator} export...', end='\n')
            path_ncdf = f"{dict_paths['folder_study_data']}tasminAdjust_JJA_rcp85.nc"
            ds = xr.open_dataset(path_ncdf)
            indicator_cols = [i for i in list(ds.variables) if indicator in i]

            # Simplify shapefiles
            tolerance = 1000
            regions_shp_simplified = regions_shp.copy()
            regions_shp_simplified.simplify(tolerance, preserve_topology=True)

            rivers_shp_simplified = rivers_shp.copy()
            rivers_shp_simplified.simplify(tolerance, preserve_topology=True)

            # Define geometry for each data (Points hydro, Polygon climate)
            if data_type == 'climate':
                geometry_dict = {row['gid']: row['geometry'] for _, row in regions_shp.iterrows()}
                ds = ds.assign_coords(geometry=('region', [geometry_dict[code] for code in ds['region'].values]))
            else:
                geometry_dict = sim_points_gdf['geometry'].to_dict()
                ds = ds.assign_coords(geometry=('code', [geometry_dict[code] for code in ds['code'].values]))

            print(f'Define horizons...', end='\r')
            # Define horizons
            ds_horizon = define_horizon(ds, files_setup)
            # Compute mean value for each horizon
            ds_mean_horizon = compute_mean_by_horizon(ds=ds_horizon, indicator_cols=indicator_cols,
                                                      files_setup=files_setup)

            da_results = apply_statistic(ds_mean_horizon.to_array(dim='new'), function=files_setup['function'],
                                         q=files_setup['quantile'])

            da_plot = compute_deviation_to_ref(da_results)

            from plot_functions.plot_map import *

            #TODO NEED TO SIMPLIFY SHAPEFILES


            plot_map(path_result=f"{dict_paths['folder_study_figures']}map_{indicator}_{timestep}_{rcp}.png",
                     # background_shp=background_shp,
                     df_plot=da_plot,
                     study_shp=regions_shp,
                     rivers_shp=rivers_shp,
                     cols=['REFERENCE'],
                     # data=da_results,
                     nrow=1,
                     ncol=3,
                     palette='viridis'
                     )





            plot_map(path_result=path_results+'count_HM_by_stations.pdf', back_shp=regions_shapefile,
                                study_shp=selected_regions_shp, rivers_shp=rivers_shp, df_plot=df_plot, cols=cols,
                                indicator='Count HM by station', figsize=(10, 10), nrow=2, ncol=2, palette='BrBG', discretize=1)



# Temporal


# N points dans la région
# M dates
# P simulations
# 1) Aggréger les N points à l'échelle de chaque région (id region)
# 2) Aggréger les dates par horizons (start-end)
# 3) Grouper les simulations (func)

# Compute value by horizon (mean, quantile, media etc)
# idem region -> groupby horizon for each point
# Supression time, remplacé par historical, horizon 1 etc
# Point 1 : Histo, H1, H2, H3
# ...
# Point n : Histo, H1, H2, H3
# Variable(region, period)
# var1, point1, h1


# Group by horizon
# df_histo, df_H1, df_H2, df_H3 = group_by_horizon(df=df_data, stations_name=selected_stations_name,
#                                                  col_by=['param_hm'], function='count', relative=False)
#
#
#
# #TODO Iterate properly
# df_H1 = df_H1.mean(axis=0).to_frame().set_axis(['H1'], axis=1)
# df_H2 = df_H2.mean(axis=0).to_frame().set_axis(['H2'], axis=1)
# df_H3 = df_H3.mean(axis=0).to_frame().set_axis(['H3'], axis=1)
#
# df_plot = pd.concat([matched_stations, df_H1, df_H2, df_H3], axis=1)

df_plot = group_by_function(df=df_data, stations_name=selected_stations_name,
                            col_by=['sim'], function='mean', relative=True, bool_cols=cols,
                            matched_stations=matched_stations)

df_plot = group_by_function(df=df_data, stations_name=selected_stations_name,
                            col_by=['param_hm'], function='any', relative=False,
                            matched_stations=None)

hm = ['CTRIP', 'EROS', 'GRSD', 'J2000', 'MORDOR-SD', 'MORDOR-TS', 'ORCHIDEE', 'SIM2', 'SMASH']

print(f'################################ PLOT ################################')

# Plot number of HM on each station
plot_scatter_on_map(path_result=path_results+'count_HM_by_stations.pdf', back_shp=regions_shapefile,
                    study_shp=selected_regions_shp, rivers_shp=rivers_shp, df_plot=df_plot, cols=cols,
                    indicator='Count HM by station', figsize=(10, 10), nrow=2, ncol=2, palette='BrBG', discretize=1)


temp = 100 * df_plot[hm].sum(axis=0)/len(df_plot)
title = [str(idx) + ' (' + str(round(val))+ '%)' for idx, val in temp.items()]
# Plot bool if HM exists on each stations
plot_scatter_on_map(path_result=path_results+'HM_by_stations_ref.pdf', back_shp=regions_shapefile,
                    study_shp=selected_regions_shp, rivers_shp=rivers_shp, df_plot=df_plot, cols=hm,
                    indicator='HM by station', figsize=(10, 10), nrow=3, ncol=3, palette='Dark2_r', discretize=1, s=20,
                    title=title)

# Scatter indicator plot on a map
plot_scatter_on_map(path_result=path_results+parameters['param_indicator']+parameters['param_timestep']+'.pdf',
                    back_shp=regions_shapefile, study_shp=selected_regions_shp, rivers_shp=rivers_shp, nrow=1, ncol=3,
                    cols=['H1', 'H2', 'H3'], df_plot=df_plot, figsize=(12, 6), palette='BrBG', vmin=-100, vmax=100,
                    indicator=f"Relvative {parameters['param_indicator']} {parameters['param_timestep']} variation (%)")


# Graphique 2 : Dispersion des résultats sur une stations : trajectoires
my_station = 'K002000101'
# X = time
# Y = param_indicator

df_station = df_data[['year', 'sim', my_station]]
my_sim = 'QA_seas-JJA_1975-2100_TIMEseries_GEOstation_FR-METRO_EXPLORE2-2024_MF-ADAMONT_historical-rcp85_EC-EARTH_RACMO22E_SMASH'