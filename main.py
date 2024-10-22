print(f'################################ IMPORT & INITIALIZATION ################################', end='\n')

print(f'> General imports...', end='\n')
import time
import json
# import pyfiglet
# ascii_banner = pyfiglet.figlet_format("Hello")
# print(ascii_banner)

print(f'> Local imports...', end='\n')
from global_functions.load_data import *
from global_functions.format_data import *
from plot_functions.plot_map import *
from global_functions.shp_geometry import *
from global_functions.path_functions import  *

# Avoid crash with console when launched manually
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.switch_backend('agg')

# Load environments variables
print(f'> Load json inputs...', end='\n')
with open('config.json') as config_file:
    config = json.load(config_file)

with open('files_setup.json') as files_setup:
    files_setup = json.load(files_setup)

print(f'> Define paths...', end='\n')
dict_paths = define_paths(config)

#%% Files names
# Study folder
print(f'> Create output directories...', end='\n')
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
print(f'> Load shapefiles...', end='\n')
regions_shp = open_shp(path_shp=dict_paths['file_regions_shp'])
study_regions_shp = regions_shp[regions_shp['gid'].isin(files_setup['regions'])]
rivers_shp = open_shp(path_shp=dict_paths['file_rivers_shp'])

# Check if study area is already matched with sim points
print(f'> Searching sim points in study area...', end='\n')
for data_type, path in dict_paths['dict_study_points_sim'].items():
    if not os.path.isfile(path):
        print(f'>> Find {data_type} data points in study area')
        sim_all_points_info = open_shp(path_shp=dict_paths['dict_global_points_sim'][data_type])
        overlay_shapefile(shapefile=study_regions_shp, data=sim_all_points_info,
                          path_result=path)
    else:
        print(f'>> {data_type.capitalize()} data points already in the study area')

# Study geographical limits
bounds = define_bounds(study_regions_shp, zoom=1000)

# Select long rivers
print(f'> Simplify shapefiles...', end='\n')
tolerance = 1000
rivers_thresh = 0.4 * ((bounds[2] - bounds[0])**2 + (bounds[3] - bounds[1])**2)**0.5
long_rivers_idx = rivers_shp.geometry.length > rivers_thresh
rivers_shp = rivers_shp[long_rivers_idx]

# Select rivers in study area
study_rivers_shp = overlay_shapefile(shapefile=bounds, data=rivers_shp)
study_rivers_shp = overlay_shapefile(shapefile=study_regions_shp, data=study_rivers_shp)
study_rivers_shp_simplified = simplify_shapefiles(study_rivers_shp, tolerance=tolerance)
# Simplify regions shapefile (background)
regions_shp = overlay_shapefile(shapefile=bounds, data=regions_shp)
regions_shp_simplified = simplify_shapefiles(regions_shp, tolerance=tolerance)

# Simplify study areas shapefile
study_regions_shp_simplified = simplify_shapefiles(study_regions_shp, tolerance=tolerance)


print(f'################################ RUN OVER NCDF ################################', end='\n')
# Get paths for selected sim
print(f'> Load ncdf data paths...', end='\n')
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
                print(f'> Create {indicator} export...', end='\n')
                extract_ncdf_indicator(
                    paths_data=paths, param_type=data_type, sim_points_gdf=sim_points_gdf,
                    indicator=indicator, timestep=timestep, start=files_setup['historical'][0], path_result=path_ncdf,
                )

            print(f'################################ FORMAT DATA ################################', end='\n')
            print(f'> Load from {indicator} export...', end='\n')
            path_ncdf = f"{dict_paths['folder_study_data']}tasminAdjust_JJA_rcp85.nc"
            ds = xr.open_dataset(path_ncdf)
            indicator_cols = [i for i in list(ds.variables) if indicator in i]

            # Define geometry for each data (Points hydro, Polygon climate)
            print(f'> Match geometry and data...', end='\n')
            if data_type == 'climate':
                geometry_dict = {row['gid']: row['geometry'] for _, row in regions_shp.iterrows()}
                ds = ds.assign_coords(geometry=('region', [geometry_dict[code] for code in ds['region'].values]))
            else:
                sim_points_gdf_simplified = sim_points_gdf.copy()
                sim_points_gdf_simplified.simplify(tolerance, preserve_topology=True)
                geometry_dict = sim_points_gdf['geometry'].to_dict()
                ds = ds.assign_coords(geometry=('code', [geometry_dict[code] for code in ds['code'].values]))

            print(f'> Define horizons...', end='\n')
            # Define horizons
            ds_horizon = define_horizon(ds, files_setup)
            # Compute mean value for each horizon
            ds_mean_horizon = compute_mean_by_horizon(ds=ds_horizon, indicator_cols=indicator_cols,
                                                      files_setup=files_setup)

            ds_results = apply_statistic(ds_mean_horizon.to_array(dim='new'), function=files_setup['function'],
                                         q=files_setup['quantile']).to_dataset(name=indicator)

            ds_resume = compute_deviation_to_ref(ds_results)

            print(f'################################ PLOT DATA ################################', end='\n')
            col_headers = {'horizon1': 'Horizon 1 (2021-2050)',
                           'horizon2': 'Horizon 2 (2041-2070)',
                           'horizon3': 'Horizon 3 (2070-2100)'}

            gdf = gpd.GeoDataFrame({
                'geometry': ds_resume['geometry'].values,
                'code': ds_resume['code'].values
            })

            cbar_title = indicator + ' deviation to ref (%)'
            path_results = f"{dict_paths['folder_study_figures']}{indicator}_{timestep}_{rcp}_"

            dict_shapefiles = {'rivers_shp': {'shp': study_rivers_shp_simplified, 'color': 'royalblue', 'linewidth': 2, 'zorder': 2},
                               'background_shp': {'shp': regions_shp_simplified, 'color': 'gainsboro', 'edgecolor': 'black', 'zorder': 0},
                               'study_shp': {'shp': study_regions_shp_simplified, 'color': 'white', 'edgecolor': 'firebrick', 'zorder': 1, 'linewidth': 1.2},}

            print(f"> MAP")
            # Plot map
            plot_map(gdf, ds_resume, indicator, path_result=path_results+'map.pdf',
                     row_name=None, row_headers=None, col_name='horizon', col_headers=col_headers,
                     cbar_title=cbar_title, title=None, dict_shapefiles=dict_shapefiles, percent=True, bounds=bounds,
                     discretize=7, palette='BrBG', fontsize=14, font='sans-serif')

