# General import
import os

# Local import
from global_functions.load_data import *
from global_functions.plot_data import *

# Define current paths environment
path_parent = os.sep.join(os.getcwd().split(os.sep)[:-2]) + os.sep
path_data = path_parent + '20_data' + os.sep
path_contour = path_data + 'contours' + os.sep

path_results = path_parent + '21_results' + os.sep

## Files names
file_shp = 'france' + os.sep + 'régions_2016.shp'
file_ncdf = 'hydro/VCN10_seas-MJJASON_1976-2100_TIMEseries_GEOstation_FR-Rhone-Loire_EXPLORE2-2024_MF-ADAMONT_historical-rcp85_NorESM1-M_WRF381P_J2000.nc'
file_csv = 'climat' + os.sep + 'ETP' + os.sep + 'ETP_Hargreaves_coefficient_0.175_1970-1979.csv'
file_stations = 'Selection_points_simulation.csv'
path_climate_csv = path_data + file_csv
path_stations = path_data + file_stations


# Load ncdf
dict_ncdf, dict_coord = load_ncdf(path_ncdf= path_data + file_ncdf, indicator='VCN10')
df_data = format_data(dict_ncdf, dict_coord)

# Load & plot shapefile
dict_plot = {'dpi': 300}
shapefile = open_shp(path_shp=path_contour + file_shp)
save_shp_figure(shapefile, path_result=path_results+'test1.png')

## Variable
# Name
# Time step

## Time

## RCM

## GCM

## BC

## HM

## Territory
