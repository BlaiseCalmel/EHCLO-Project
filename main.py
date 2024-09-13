# General import
import os

# Local import
from global_functions.load_data import *
from global_functions.plot_data import *

# Avoid crash with console when launched manually
import matplotlib
matplotlib.use('TkAgg')

# Define current main paths environment
path_parent = os.sep.join(os.getcwd().split(os.sep)[:-2]) + os.sep
path_data = path_parent + '20_data' + os.sep
path_contour = path_data + 'contours' + os.sep
path_results = path_parent + '21_results' + os.sep

# Files names
file_shp = 'france' + os.sep + 'régions_2016.shp'
file_ncdf = 'hydro/VCN10_seas-MJJASON_1976-2100_TIMEseries_GEOstation_FR-Rhone-Loire_EXPLORE2-2024_MF-ADAMONT_historical-rcp85_NorESM1-M_WRF381P_J2000.nc'
file_csv = 'climat' + os.sep + 'ETP' + os.sep + 'ETP_Hargreaves_coefficient_0.175_1970-1979.csv'
file_stations = 'Selection_points_simulation.csv'

# Files paths
path_clim_csv = path_data + file_csv
path_stations = path_data + file_stations
path_ncdf = path_data + file_ncdf
path_shp = path_contour + file_shp

# Load Stations
stations_data = load_csv(path_stations)

# Load ncdf [HYDRO]
hydro_ncdf = load_ncdf(path_ncdf=path_ncdf, indicator='VCN10')
hydro_data = format_data(hydro_ncdf, stations_data)

# Load csv [CLIM]
clim_data = load_csv(path_clim_csv)

# Load shapefile
dict_plot = {'dpi': 300}
shapefile = open_shp(path_shp=path_shp)
path_result = path_results+'test7.png'

plot_shp_figure(path_result=path_result, shapefile=shapefile, shp_column=None, df=hydro_data, indicator='value',
                figsize=None, palette='BrBG')





# Faire des beaux graphiques des stations
# Choix palette
# Choix date/réchauffement (à voir)
