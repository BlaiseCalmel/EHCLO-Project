print(f'################################ IMPORT & INITIALIZATION ################################')
# General import
print(f'General imports...')
import os
import glob
import json
# import pyfiglet
# ascii_banner = pyfiglet.figlet_format("Hello")
# print(ascii_banner)

# Local import
print(f'Local imports...')
from global_functions.load_data import *
from global_functions.plot_data import *
from global_functions.shp_geometry import *
from global_functions.path_functions import  *

# Avoid crash with console when launched manually
import matplotlib
matplotlib.use('TkAgg')
plt.switch_backend('agg')

# Load environments variables
print(f'Load json inputs...')
with open('config.json') as config_file:
    config = json.load(config_file)

with open('files_setup.json') as files_setup:
    files_setup = json.load(files_setup)

# Define current main paths environment
print(f'Define paths...')
dict_paths = define_paths(config)

#%% Files names
# Study folder
print(f'Create output directories...')
if not os.path.isdir(dict_paths['folder_study_results']):
    os.makedirs(dict_paths['folder_study_results'])

# Study figures folder
if not os.path.isdir(dict_paths['folder_study_figures']):
    os.makedirs(dict_paths['folder_study_figures'])

# Study data folder
if not os.path.isdir(dict_paths['folder_study_data']):
    os.makedirs(dict_paths['folder_study_data'])

#%% LOAD STUDY REGION SHAPEFILE
print(f'################################ STUDY AREA ################################')
print(f'Load shapefiles...')
regions_shp = open_shp(path_shp=dict_paths['file_regions_shp'])
study_regions_shp = regions_shp[regions_shp['gid'].isin(config['regions'])]
rivers_shp = open_shp(path_shp=dict_paths['file_rivers_shp'])

# Check if study area is already matched with sim points
print(f'Find sim points in study area...')
for i in range(len(dict_paths['list_global_points_sim'])):
    if not os.path.isfile(dict_paths['list_study_points_sim'][i]):
        print(f'Find {config["param_type"][i]} data points in study area')
        sim_all_points_info = load_csv(path_file=dict_paths['list_global_points_sim'][i])
        is_data_in_shape(shapefile=study_regions_shp, data=sim_all_points_info, cols=['XL93', 'YL93'],
                         path_result=dict_paths['list_study_points_sim'][i])


print(f'################################ RUN OVER NCDF ################################')
# Get paths for selected sim
print(f'Load ncdf data paths...')
path_files = get_files_path(dict_paths=dict_paths, setup=files_setup)

# Run among data type climate/hydro
dict_data = {}
start_run = time.time()
for data_type in config['param_type']:
    print(f'###### Loading sim point for {data_type} data')
    idx = config['param_type'].index(data_type)
    # Load selected sim points from study area
    if data_type == "hydro":
        sim_points_df = pd.read_csv(dict_paths['list_study_points_sim'][idx], index_col=None)
        # Stations de references pour hydro uniquement
        sim_points_df = sim_points_df[sim_points_df['Référence'] == 1]
        valid_stations = pd.isna(sim_points_df['PointsSupprimes'])
        sim_points_df = sim_points_df[valid_stations].reset_index(drop=True).set_index('SuggestionCode')
        sim_points_df.index.names = ['name']
    else:
        sim_points_df = pd.read_csv(dict_paths['list_study_points_sim'][idx], index_col=0)

    # data_path = dict_paths[f'folder_{data_type}_data']
    # Run among indicator for the current data type
    for indicator in files_setup[data_type + '_indicator']:
        print(f'### Loading {indicator}')
        paths_indicator = path_files[indicator]
        if not os.path.isfile(f"{dict_paths['folder_study_data']}{os.sep}{indicator}'.nc'"):
            print(f'Create {indicator} export...')
            dict_data[indicator] = extract_ncdf_indicator(
                path_ncdf=paths_indicator, param_type=data_type, sim_points_df=sim_points_df,
                indicator=indicator, path_result=dict_paths['folder_study_data']
            )
        else:
            print(f'Load from export...')
            dict_data[indicator] = xr.open_dataset(f"{dict_paths['folder_study_data']}{os.sep}{indicator}'.nc'")




print(f'################################ FORMAT DATA ################################')
# Convert to df and define horizon
df_data = from_dict_to_df(dict_data)

cols = ['Histo', 'H1', 'H2', 'H3']

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