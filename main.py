# General import
import os
import glob
import time
import pandas as pd

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

###################################### SHAPEFILES ######################################

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
matched_stations = data_in_shape(selected_regions_shapefile, stations_data, cols=['XL93', 'YL93'],
                                 path_results=None)

# matched_rivers = data_in_shape(selected_regions_shapefile, rivers_shapefile,
#                                  path_results=path_contour+os.sep+'shapefiles'+os.sep+'rivers.shp')

# Get stations in selected area
# selected_stations = matched_stations[matched_stations['code'].isin(selected_id)]
selected_stations_name = matched_stations.index.to_list()


###################################### NETCDF DATA ######################################

# Get indicator info for each station and group them
# Define data type to analyse
# TODO define it for climate data
param_type = 'hydro'
parameters = {'param_indicator': 'VCN10', 'param_timestep': '', 'param_timeperiod': '',
              'param_time': '', 'param_geo': '', 'param_area': '', 'param_project': '',
              'param_bc': 'ADAMONT', 'param_rcp': '', 'param_gcm': '', 'param_rcm': '', 'param_hm': ''}

path_indicator_files = glob.glob(path_data + f"{param_type}/{parameters['param_indicator']}*"
                                             f"{parameters['param_timestep']}*.nc")
my_dict = {}
time_start = time.time()
estimate_timestep = 0.004
i = -1
for path_ncdf in path_indicator_files[:4]:
    i += 1
    time_min = (time.time() - time_start) / 60
    files_to_open = (len(path_indicator_files) - i)
    if i > 5:
        estimate_timestep = time_min / i
    # Get current file info
    file_name = os.path.basename(path_ncdf)[:-3]
    split_name = file_name.split('_')
    split_name += [''] * (len(parameters) - len(split_name))

    # Save them in dict
    file_dict = dict(zip(parameters, split_name))

    # Load ncdf [HYDRO]
    if param_type == 'hydro':
        my_dict[file_name] = load_ncdf(path_ncdf=path_ncdf, file_dict=file_dict, indicator=parameters['param_indicator'],
                                       station_codes=selected_stations_name)

    print(f'============= {file_dict} =============\n'
          f'Running for {time_min.__round__(1)}min\n'
          f'End estimation in {(files_to_open * estimate_timestep).__round__(1)}min '
          f'[{i} files/{len(path_indicator_files)}]')
    # hydro_df = pd.DataFrame(hydro_ncdf)
    # hydro_df['time'] = pd.to_datetime(hydro_df['time'], unit='D', origin=pd.Timestamp('1950-01-01')
    #                                  ).apply(lambda x: x.date())

# Transform dict to DataFrame
df = pd.concat({k: pd.DataFrame(v) for k, v in my_dict.items()})
df = df.reset_index().rename(columns={'level_0': 'sim', 'level_1': 'iteration'})

import datetime as dt
# Convert number of days since 01-01-1950 to year
start = dt.datetime(1950,1,1,0,0)
datetime_series = df['time'].astype('timedelta64[D]') + start
df['year'] = datetime_series.dt.year

# Define Horizon
df['Histo'] = df['year'] < 2006
df['H1'] = (df['year'] >= 2021) & (df['year'] <= 2050)
df['H2'] = (df['year'] >= 2041) & (df['year'] <= 2070)
df['H3'] = df['year'] >= 2070

df_stations = [i for i in selected_stations_name if i in df.columns]

import numpy as np
groupby_dict = {k: 'median' for k in df_stations}
df_histo = df[df['Histo'] == True].groupby(['sim']).agg(groupby_dict)
df_H1 = df[df['H1'] == True].groupby(['sim']).agg(groupby_dict)
df_H2 = df[df['H2'] == True].groupby(['sim']).agg(groupby_dict)
df_H3 = df[df['H3'] == True].groupby(['sim']).agg(groupby_dict)








# Load csv [CLIM]
clim_data = load_csv(path_clim_csv)


# PLOT
dict_plot = {'dpi': 300}


plot_shp_figure(path_result=path_result, shapefile=shapefile, shp_column=None, df=hydro_data, indicator='value',
                figsize=None, palette='BrBG')
