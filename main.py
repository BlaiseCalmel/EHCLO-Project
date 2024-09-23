print(f'################################ IMPORT & INITIALIZATION ################################')
# General import

import glob
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
plt.switch_backend('agg')

# Define current main paths environment
path_parent = os.sep.join(os.getcwd().split(os.sep)[:-2]) + os.sep
path_data = path_parent + '2_data' + os.sep
path_contour = path_data + 'contours' + os.sep
path_results = path_parent + '3_results' + os.sep

# Files names
regions_shp = 'map' + os.sep + 'regionHydro' + os.sep + 'regionHydro.shp'
rivers_shp = 'france_rivers' + os.sep + 'france_rivers.shp'
file_stations = 'Selection_points_simulation.csv'

# Paths
path_stations = path_contour + file_stations
path_regions_shp = path_contour + regions_shp
path_rivers_shp = path_contour + rivers_shp

# Define data type to analyse
# TODO define it for climate data
param_type = 'hydro'
parameters = {'param_indicator': 'VCN10',
              'param_timestep': 'seas-MJJASON',
              'param_timeperiod': '',
              'param_time': '',
              'param_geo': '',
              'param_area': '',
              'param_project': '',
              'param_bc': 'ADAMONT',
              'param_rcp': 'historical-rcp85',
              'param_gcm': '',
              'param_rcm': '',
              'param_hm': ''}
extension = 'nc'

print(f'################################ DEFINE STUDY AREA ################################')

# Load Regions shapefile
id_regions = [23, 27, 30]
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

stations_ref = stations_data[stations_data['Référence'] == 1]
# stations_four = stations_data[stations_data['n'] >= 4]


# Create file matching shape and stations
matched_stations = is_data_in_shape(selected_regions_shp, stations_data, cols=['XL93', 'YL93'],
                                 path_results=None)
# matched_stations[matched_stations['n'] < 4] #23
# matched_rivers = is_data_in_shape(selected_regions_shapefile, rivers_shapefile,
#                                  path_results=path_contour+os.sep+'shapefiles'+os.sep+'rivers.shp')

# Get stations in selected area
# selected_stations = matched_stations[matched_stations['code'].isin(selected_id)]
selected_stations_name = matched_stations.index.to_list()


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
dict_data = iterate_over_path(path_indicator_files, param_type, parameters, selected_stations_name)

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

# Scatter plot on a map
plot_scatter_on_map(path_result=path_result, back_shp=selected_regions_shp, df_plot=df_plot, cols=cols,
                    indicator=parameters['param_indicator'], figsize=None, palette='BrBG')
