# General import
import os
import glob

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

# Define indicator
indicator = 'VCN10'

# Files names
file_shp = 'sage_metrople-shp' + os.sep + 'sage_shape.shp'
file_ncdf = 'hydro/VCN10_seas-MJJASON_1976-2100_TIMEseries_GEOstation_FR-Rhone-Loire_EXPLORE2-2024_MF-ADAMONT_historical-rcp85_NorESM1-M_WRF381P_J2000.nc'
file_csv = 'climat' + os.sep + 'ETP' + os.sep + 'ETP_Hargreaves_coefficient_0.175_1970-1979.csv'
file_stations = 'Selection_points_simulation.csv'

# Files paths
path_clim_csv = path_data + file_csv
path_stations = path_data + file_stations
path_ncdf = path_data + file_ncdf
path_shp = path_contour + file_shp

# Get indicator files
path_indicator_files = glob.glob(path_data + f'hydro/{indicator}*.nc')

# Select study area
# TODO match station with multiple hydrographic areas
selected_id = ['SAGE06020', 'SAGE01015', 'SAGE03027', 'SAGE01013']

# Load stations info
stations_data = load_csv(path_stations)
valid_stations = pd.isna(stations_data['PointsSupprimes'])
stations_data = stations_data[valid_stations].reset_index(drop=True).set_index('SuggestionCode')

# Load shapefile
shapefile = open_shp(path_shp=path_shp)

# Create file matching shape and stations
matched_stations = stations_in_shape(shapefile, stations_data)

# Get stations in selected area
selected_stations = matched_stations[matched_stations['code'].isin(selected_id)]


# Get indicator info for each station and group them
for path_ncdf in path_indicator_files:
    # Load ncdf [HYDRO]
    hydro_ncdf = load_ncdf(path_ncdf=path_ncdf, indicator=indicator, station_codes=selected_stations_name)
    hydro_df = pd.DataFrame(hydro_ncdf)
    hydro_df['time'] = pd.to_datetime(hydro_df['time'], unit='D', origin=pd.Timestamp('1950-01-01')
                                     ).apply(lambda x: x.date())




    # hydro_data = format_data(dict_ncdf=hydro_ncdf, stations_data=stations_data)


# Load csv [CLIM]
clim_data = load_csv(path_clim_csv)


# PLOT
dict_plot = {'dpi': 300}
path_result = path_results+'test7.png'

plot_shp_figure(path_result=path_result, shapefile=shapefile, shp_column=None, df=hydro_data, indicator='value',
                figsize=None, palette='BrBG')





# Faire des beaux graphiques des stations
# Choix palette
# Choix date/réchauffement (à voir)
