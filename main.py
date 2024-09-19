print(f'################################ IMPORT & INITIALIZATION ################################')
# General import
import os
import glob
import time
import pandas as pd
# import pyfiglet
# ascii_banner = pyfiglet.figlet_format("Hello")
# print(ascii_banner)

# Local import
from global_functions.load_data import *
from global_functions.plot_data import *
from global_functions.shp_geometry import *

# Avoid crash with console when launched manually
import matplotlib
matplotlib.use('TkAgg')

# Define current main paths environment
path_parent = os.sep.join(os.getcwd().split(os.sep)[:-2]) + os.sep
path_data = path_parent + '20_data' + os.sep
path_contour = path_data + 'contours' + os.sep
path_results = path_parent + '21_results' + os.sep

# Files names
regions_shp = 'map' + os.sep + 'regionHydro' + os.sep + 'regionHydro.shp'
# rivers_shp = 'map' + os.sep + 'coursEau' + os.sep + 'CoursEau_FXX.shp'
# file_ncdf = 'hydro/VCN10_seas-MJJASON_1976-2100_TIMEseries_GEOstation_FR-Rhone-Loire_EXPLORE2-2024_MF-ADAMONT_historical-rcp85_NorESM1-M_WRF381P_J2000.nc'
# file_csv = 'climat' + os.sep + 'ETP' + os.sep + 'ETP_Hargreaves_coefficient_0.175_1970-1979.csv'
file_stations = 'Selection_points_simulation.csv'
# file_stations = 'shapefiles' + os.sep + 'stations.shp'

# Paths
path_stations = path_contour + file_stations
path_regions_shp = path_contour + regions_shp
# path_rivers_shp = path_contour + rivers_shp

print(f'################################ DEFINE STUDY AREA ################################')

# Load Regions shapefile
id_regions = [23, 27, 30]
regions_shapefile = open_shp(path_shp=path_regions_shp)
selected_regions_shapefile = regions_shapefile[regions_shapefile['gid'].isin(id_regions)]

# Load Rivers shapefile
# rivers_shapefile = open_shp(path_shp=path_rivers_shp)
# rivers_shapefile['TopoOH'] = rivers_shapefile['TopoOH'].astype(str)
# selected_rivers_shapefile = rivers_shapefile[rivers_shapefile['TopoOH'].str.contains("Fleuve la Loire")]

# Test current shapefile
# TEMP
path_result = path_results+'test11.png'
save_shp_figure(regions_shapefile, path_result, study_shapefile=selected_regions_shapefile,
                rivers_shp=None)

# Load stations info
stations_data = load_csv(path_stations)
valid_stations = pd.isna(stations_data['PointsSupprimes'])
stations_data = stations_data[valid_stations].reset_index(drop=True).set_index('SuggestionCode')
# stations_data = open_shp(path_shp=path_stations)

# Create file matching shape and stations
matched_stations = is_data_in_shape(selected_regions_shapefile, stations_data, cols=['XL93', 'YL93'],
                                 path_results=None)

# matched_rivers = is_data_in_shape(selected_regions_shapefile, rivers_shapefile,
#                                  path_results=path_contour+os.sep+'shapefiles'+os.sep+'rivers.shp')

# Get stations in selected area
# selected_stations = matched_stations[matched_stations['code'].isin(selected_id)]
selected_stations_name = matched_stations.index.to_list()


print(f'################################ EXTRACT DATA ################################')
# Get indicator info for each station and group them
# Define data type to analyse
# TODO define it for climate data
param_type = 'hydro'
parameters = {'param_indicator': 'VCN10', 'param_timestep': '', 'param_timeperiod': '',
              'param_time': '', 'param_geo': '', 'param_area': '', 'param_project': '',
              'param_bc': 'ADAMONT', 'param_rcp': '', 'param_gcm': '', 'param_rcm': '', 'param_hm': ''}
extension = 'nc'

path_indicator_files = glob.glob(path_data + f"{param_type}/{parameters['param_indicator']}*"
                                             f"{parameters['param_timestep']}*.{extension}")
data_dict = {}
time_start = time.time()
estimate_timestep = 0.004
i = -1
for path_ncdf in path_indicator_files:
    i += 1
    timedelta = (time.time() - time_start)
    files_to_open = (len(path_indicator_files) - i)
    if i > 5:
        estimate_timestep = timedelta / i
    # Get current file info
    file_name = os.path.basename(path_ncdf)[:-3]
    split_name = file_name.split('_')
    split_name += [''] * (len(parameters) - len(split_name))

    # Save them in dict
    file_dict = dict(zip(parameters, split_name))

    # Load ncdf [HYDRO]
    if param_type == 'hydro':
        data_dict[file_name] = load_ncdf(path_ncdf=path_ncdf, file_dict=file_dict,
                                         indicator=parameters['param_indicator'], station_codes=selected_stations_name)

    print(f'============= {file_name} =============\n'
          f'Running for {dt.timedelta(seconds=round(timedelta))}\n'
          f'Ends in {dt.timedelta(seconds=round(files_to_open*estimate_timestep))} '
          f'[{i+1} files/{len(path_indicator_files)}]')
    # hydro_df = pd.DataFrame(hydro_ncdf)
    # hydro_df['time'] = pd.to_datetime(hydro_df['time'], unit='D', origin=pd.Timestamp('1950-01-01')
    #                                  ).apply(lambda x: x.date())


print(f'################################ FORMAT DATA ################################')
# Transform dict to DataFrame
df = pd.concat({k: pd.DataFrame(v) for k, v in data_dict.items()})
df = df.reset_index().rename(columns={'level_0': 'sim', 'level_1': 'iteration'})

# Convert number of days since 01-01-1950 to year
df = convert_timedelta64(df)

# Define Horizon
df = define_horizon(df)

# Group by horizon
df_histo, df_H1, df_H2, df_H3 = group_by_horizon(df=df, stations_name=selected_stations_name, function='median')


#TODO Iterate properly
cols = ['H1', 'H2', 'H3']

df_H1_relative = df_H1 / df_histo
df_H2_relative = df_H2 / df_histo
df_H3_relative = df_H3 / df_histo
df_H1_relative = df_H1_relative.mean(axis=0).to_frame().set_axis(['H1'], axis=1)
df_H2_relative = df_H2_relative.mean(axis=0).to_frame().set_axis(['H2'], axis=1)
df_H3_relative = df_H3_relative.mean(axis=0).to_frame().set_axis(['H3'], axis=1)

plot_df = pd.concat([matched_stations, df_H1_relative, df_H2_relative, df_H3_relative], axis=1)


print(f'################################ PLOT ################################')

# Scatter plot on a map
plot_scatter_on_map(path_result=path_result, region_shp=selected_regions_shapefile, df=plot_df, cols=cols,
                    indicator=parameters['param_indicator'], figsize=None, palette='BrBG')
