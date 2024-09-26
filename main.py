print(f'################################ IMPORT & INITIALIZATION ################################')
# General import
import os
import glob
import json
# import pyfiglet
# ascii_banner = pyfiglet.figlet_format("Hello")
# print(ascii_banner)

# Local import
from global_functions.load_data import *
from global_functions.plot_data import *
from global_functions.shp_geometry import *
from global_functions.path_functions import  *

# Avoid crash with console when launched manually
import matplotlib
matplotlib.use('TkAgg')
plt.switch_backend('agg')

# Load environments variables
with open('config.json') as config_file:
    config = json.load(config_file)

# Define current main paths environment
cwd = os.sep.join(os.getcwd().split(os.sep)[:-2]) + os.sep
dict_paths = define_paths(cwd, config)


#%% Files names
# Study folder
if not os.path.isdir(dict_paths['folder_study_results']):
    os.makedirs(dict_paths['folder_study_results'])

# Study figures folder
if not os.path.isdir(dict_paths['folder_study_figures']):
    os.makedirs(dict_paths['folder_study_figures'])

# Study contour folder
if not os.path.isdir(dict_paths['folder_study_contour']):
    os.makedirs(dict_paths['folder_study_contour'])

#%% LOAD STUDY REGION SHAPEFILE
regions_shp = open_shp(path_shp=dict_paths['file_regions_shp'])
study_regions_shp = regions_shp[regions_shp['gid'].isin(config['regions'])]
rivers_shp = open_shp(path_shp=dict_paths['file_rivers_shp'])

# Check if study area is already matched with sim points
if not os.path.isfile(dict_paths['file_study_points_sim']):
    sim_all_points_info = load_csv(path_files=dict_paths['file_data_points_sim'])
    is_data_in_shape(shapefile=study_regions_shp, data=sim_all_points_info, cols=['XL93', 'YL93'],
                     path_result=dict_paths['file_study_points_sim'])

# Load selected sim points from study area
sim_points_df = load_csv(dict_paths['file_study_points_sim'])
# Stations de references pour hydro uniquement
# stations_data = stations_data[stations_data['Référence'] == 1]


# is_data_in_shape(shapefile, data, cols=None, path_result=None)
stations_data = load_csv(path_stations)
valid_stations = pd.isna(stations_data['PointsSupprimes'])
stations_data = stations_data[valid_stations].reset_index(drop=True).set_index('SuggestionCode')



# Create file matching shape and stations
matched_stations = is_data_in_shape(selected_regions_shp, stations_data, cols=['XL93', 'YL93'],
                                    path_result=None)




#%% IF NO CSV = LOAD NETCDF AND FIND POINTS INSIDE STUDY AREA

# Data to load
# Background region & rivers shapefiles
id_regions = config['regions']

regions_shp = 'map' + os.sep + 'regionHydro' + os.sep + 'regionHydro.shp'
rivers_shp = 'france_rivers' + os.sep + 'france_rivers.shp'

# Data in study area
if not os.path.isdir(path_study_contour):
    os.makedirs(path_study_contour)

file_hydro = 'hydro_points_sim.csv'
file_climpoints = 'clim_points_sim.csv'



# Paths
path_stations = path_contour + file_stations
# path_climpoints = path_contour + file_climpoints
path_climpoints = path_contour+'meteo_sim_point_in.csv'
path_regions_shp = path_contour + regions_shp
path_rivers_shp = path_contour + rivers_shp

# Define data type to analyse
# TODO define it for climate data
# param_type = 'climat'
# parameters = {'param_indicator': 'QA',
#               'param_timestep': 'seas-JJA',
#               'param_timeperiod': '',
#               'param_time': '',
#               'param_geo': '',
#               'param_area': '',
#               'param_project': '',
#               'param_bc': 'ADAMONT',
#               'param_rcp': 'historical-rcp85',
#               'param_gcm': '',
#               'param_rcm': '',
#               'param_hm': ''}
# if param_type == 'hydro':
#     extension = 'nc'
# else:
#     extension = 'csv'

print(f'################################ DEFINE STUDY AREA ################################')

# Load Regions shapefile

regions_shapefile = open_shp(path_shp=path_regions_shp)
selected_regions_shp = regions_shapefile[regions_shapefile['gid'].isin(id_regions)]

# Load Rivers shapefile
rivers_shp = open_shp(path_shp=path_rivers_shp)

# Test current shapefile
path_result = path_results+'test15.png'
save_shp_figure(back_shp=regions_shapefile, path_result=path_result, study_shp=selected_regions_shp,
                rivers_shp=rivers_shp)

# Load stations info
stations_data = load_csv(path_stations)
valid_stations = pd.isna(stations_data['PointsSupprimes'])
stations_data = stations_data[valid_stations].reset_index(drop=True).set_index('SuggestionCode')

stations_data = stations_data[stations_data['Référence'] == 1]

# Create file matching shape and stations
matched_stations = is_data_in_shape(selected_regions_shp, stations_data, cols=['XL93', 'YL93'],
                                 path_result=None)


# Load climat simpoints info
climpoints_data = load_csv(path_climpoints)
climpoints_data = climpoints_data.reset_index(drop=True).set_index('name')
matched_climpoints = is_data_in_shape(shapefile=selected_regions_shp, data=climpoints_data, cols=['XL93', 'YL93'],
                                      path_result=None)


# matched_stations[matched_stations['n'] < 4] #23
# matched_rivers = is_data_in_shape(selected_regions_shapefile, rivers_shapefile,
#                                  path_results=path_contour+os.sep+'shapefiles'+os.sep+'rivers.shp')

# Get stations in selected area
# selected_stations = matched_stations[matched_stations['code'].isin(selected_id)]
selected_stations_name = matched_stations.index.to_list()
selected_climpoints_name = matched_climpoints.index.to_list()


########################################################################
file_test = 'prtotAdjust_France_MPI-M-MPI-ESM-LR_historical_r1i1p1_CLMcom-CCLM4-8-17_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19500101-20051231.nc'
path_test = path_data + file_test
# Read netCDF
open_netcdf = netCDF4.Dataset(path_test,'r', encoding='utf-8')

test = is_data_in_shape(shapefile=selected_regions_shp, data=open_netcdf, cols=['lon', 'lat'],
                        path_result=path_contour+'meteo_sim_point_in.csv')


start_date = pd.to_datetime('1850-01-01')
dates = start_date + pd.to_timedelta(open_netcdf.variables['time'][:].data, unit='D')
reference_date = pd.to_datetime('1976-01-01')

convert_timedelta64(dates, reference_date='1850-01-01')

valid_dates = dates[dates > reference_date].to_list()
positions_idx = np.where(dates > reference_date)[0]
dict_test = {'time': valid_dates}
start_day = 46021


i = 0
duration = len(matched_climpoints)
open_netcdf = netCDF4.Dataset(path_test,'r', encoding='utf-8')
dict_netcdf = {}
start_time = time.time()
for index, row in matched_climpoints[:10].iterrows():
    i += 1
    # dict_test[index] = open_netcdf.variables['prtotAdjust'][positions_idx, row['coordx'], row['coordy']].data
    dict_netcdf[index] = open_netcdf.variables['prtotAdjust'][:, row['coordx'], row['coordy']].data
    # timedelta = (time.time() - start_time)
    # print(f'Loaded {i} files/{duration} for {dt.timedelta(seconds=round(timedelta))}')
timedelta = (time.time() - start_time)
print(f'10 files loaded by netcdf4 in {dt.timedelta(seconds=round(timedelta))}')

import xarray as xr
ds = xr.open_dataset(path_test)
df = load_csv(path_climpoints)
df = df.reset_index(drop=True).set_index('name')
start_time = time.time()
ds_after_1976 = ds.sel(time=slice('1976-01-01', None))
value = ds_after_1976['prtotAdjust'].sel(x=xr.DataArray(df.iloc[:10]['coordx'], dims="z"),
                      y=xr.DataArray(df.iloc[:10]['coordy'], dims="z"),
                      method="nearest")
value_df = value.to_dataframe()
timedelta = (time.time() - start_time)
print(f'10 files loaded by xarray in {dt.timedelta(seconds=round(timedelta))}')







open_netcdf['time'][:].data.shape
dict_test[index].shape
df = pd.concat({k: pd.DataFrame(v) for k, v in dict_test2.items()})


print(f'################################ EXTRACT DATA ################################')
# Get indicator info for each station and group them
path_indicator_files = glob.glob(path_data + f"{param_type}/{parameters['param_indicator']}*"
                                             f"{parameters['param_timestep']}*{parameters['param_timeperiod']}*"
                                             f"{parameters['param_time']}*{parameters['param_geo']}*"
                                             f"{parameters['param_area']}*{parameters['param_project']}*"
                                             f"{parameters['param_project']}*{parameters['param_bc']}*"
                                             f"{parameters['param_rcp']}*{parameters['param_gcm']}*"
                                             f"{parameters['param_rcm']}*{parameters['param_hm']}.{extension}")

# Load data from each files
if param_type == 'hydro':
    dict_data = iterate_over_path(path_indicator_files, param_type, parameters, selected_stations_name)
else:
    dict_data = load_csv(path_files=path_indicator_files, data_type='sqr', sep=';')
    # df.to_csv(path_contour+'maillage_climat.csv', sep=',')

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