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
path_data = path_parent + '20_data' + os.sep
path_contour = path_data + 'contours' + os.sep
path_results = path_parent + '21_results' + os.sep

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
              'param_timestep': '',
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
# TEMP
path_result = path_results+'test15.png'
save_shp_figure(back_shp=regions_shapefile, path_result=path_result, study_shp=selected_regions_shp,
                rivers_shp=rivers_shp)

# Load stations info
stations_data = load_csv(path_stations)
valid_stations = pd.isna(stations_data['PointsSupprimes'])
stations_data = stations_data[valid_stations].reset_index(drop=True).set_index('SuggestionCode')

stations_ref = stations_data[stations_data['Référence'] == 1]
stations_four = stations_data[stations_data['n'] >= 4]

stations_data = stations_ref


# Create file matching shape and stations
matched_stations = is_data_in_shape(selected_regions_shp, stations_data, cols=['XL93', 'YL93'],
                                 path_results=None)

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

dict_data = iterate_over_path(path_indicator_files, param_type, parameters, selected_stations_name)

print(f'################################ FORMAT DATA ################################')
df_data = from_dict_to_df(dict_data)

# Group by horizon
df_histo, df_H1, df_H2, df_H3 = group_by_horizon(df=dict_data, stations_name=selected_stations_name,
                                                 col_by=['param_hm'], function='count', relative=False)

#TODO Iterate properly
df_H1 = df_H1.mean(axis=0).to_frame().set_axis(['H1'], axis=1)
df_H2 = df_H2.mean(axis=0).to_frame().set_axis(['H2'], axis=1)
df_H3 = df_H3.mean(axis=0).to_frame().set_axis(['H3'], axis=1)

df_plot = pd.concat([matched_stations, df_H1, df_H2, df_H3], axis=1)


print(f'################################ PLOT ################################')
# Count HM by station
cols = ['Histo', 'H1', 'H2', 'H3']
list_df = [df_histo, df_H1, df_H2, df_H3]
i = -1
for dataframe in list_df:
    i += 1
    dataframe = dataframe.loc[dataframe.index != '']
    dataframe = (dataframe > 0).sum(axis=0).to_frame().set_axis([cols[i]], axis=1)
    list_df[i] = dataframe

list_df.append(matched_stations)
df_plot = pd.concat(list_df, axis=1)

row = df_plot.loc[df_plot['H1'] == 0]

sum(df_plot['H1'] != df_plot['n'])


# Scatter plot on a map
plot_scatter_on_map(path_result=path_result, back_shp=regions_shapefile, study_shp=selected_regions_shp,
                    rivers_shp=rivers_shp, df_plot=df_plot, cols=cols, indicator='Count HM by station',
                    figsize=(10, 10), nrow=1, ncol=2, palette='BrBG', discretize=1)


# Scatter plot on a map
plot_scatter_on_map(path_result=path_result, back_shp=selected_regions_shp, df_plot=df_plot, cols=cols,
                    indicator=parameters['param_indicator'], figsize=None, palette='BrBG')
