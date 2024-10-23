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

if not os.path.isdir(dict_paths['folder_study_data'] + 'shapefiles'):
    os.makedirs(dict_paths['folder_study_data'] + 'shapefiles')

#%% LOAD STUDY REGION SHAPEFILE
print(f'################################ STUDY AREA ################################', end='\n')
print(f'> Load shapefiles...', end='\n')
regions_shp = open_shp(path_shp=dict_paths['file_regions_shp'])
study_ug_shp = open_shp(path_shp=dict_paths['file_ug_shp'])
study_ug_shp = study_ug_shp[study_ug_shp['gid'].isin(files_setup['regions'])]
rivers_shp = open_shp(path_shp=dict_paths['file_rivers_shp'])

# Check if study area is already matched with sim points
print(f'> Searching sim points in study area...', end='\n')
for data_type, path in dict_paths['dict_study_points_sim'].items():
    if not os.path.isfile(path):
        print(f'>> Find {data_type} data points in study area')
        sim_all_points_info = open_shp(path_shp=dict_paths['dict_global_points_sim'][data_type])
        overlay_shapefile(shapefile=study_ug_shp, data=sim_all_points_info,
                          path_result=path)

        # test = overlay_shapefile(shapefile=study_ug_shp, data=sim_all_points_info)
        # valid_stations = pd.isna(test['PointsSupp'])
        # test = test[valid_stations].reset_index(drop=True).set_index('Suggestion')
        #
        #
        # bassinHydro = open_shp(path_shp='/home/bcalmel/Documents/2_data/contours_all/map/entiteHydro/BV_4207_stations.shp')
        # from shapely.ops import unary_union
        # study_ug_shp['']
        #
        # for ug in np.unique(test['toponyme1']):
        #     selected_ug = test[test['toponyme1'] == ug]
        #     selectedBassin = bassinHydro[bassinHydro['Code'].isin(selected_ug['CODE'])]
        #     merged_polygon = unary_union(selectedBassin['geometry'])
        #     study_ug_shp.loc[study_ug_shp['toponyme1'] == ug, 'geometry'] = merged_polygon
        #
        # save = '/home/bcalmel/Documents/2_data/contours/bassin_loire_ug.shp'
        # study_ug_shp.to_file(save, index=False)
    else:
        print(f'>> {data_type.capitalize()} data points already in the study area')

# Study geographical limits
bounds = define_bounds(study_ug_shp, zoom=2500)

# Select long rivers
print(f'> Simplify shapefiles...', end='\n')
tolerance = 1000
rivers_thresh = 0.4 * ((bounds[2] - bounds[0])**2 + (bounds[3] - bounds[1])**2)**0.5
long_rivers_idx = rivers_shp.geometry.length > rivers_thresh
rivers_shp = rivers_shp[long_rivers_idx]

# Select rivers in study area
study_rivers_shp = overlay_shapefile(shapefile=bounds, data=rivers_shp)
study_rivers_shp = overlay_shapefile(shapefile=study_ug_shp, data=study_rivers_shp)
study_rivers_shp_simplified = simplify_shapefiles(study_rivers_shp, tolerance=tolerance)
# Simplify regions shapefile (background)
regions_shp = overlay_shapefile(shapefile=bounds, data=regions_shp)
regions_shp_simplified = simplify_shapefiles(regions_shp, tolerance=tolerance)

# Simplify study areas shapefile
study_ug_shp_simplified = simplify_shapefiles(study_ug_shp, tolerance=tolerance)

print(f'################################ RUN OVER NCDF ################################', end='\n')
# Get paths for selected sim
print(f'> Load ncdf data paths...', end='\n')
path_files = get_files_path(dict_paths=dict_paths, setup=files_setup)

# Run among data type climate/hydro
start_run = time.time()
total_iterations = len(path_files.keys())

#
# data_type='hydro'
# subdict=path_files[data_type]
# rcp='rcp85'
# subdict2=subdict[rcp]
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
            split_indicator = indicator_raw.split('_')
            indicator = split_indicator[0]
            timestep = 'YE'
            if len(split_indicator) > 1:
                timestep = split_indicator[1]

            if timestep == 'mon':
                timestep = 'M'

            path_ncdf = f"{dict_paths['folder_study_data']}{indicator}_{timestep}_{rcp}.nc"

            if not os.path.isfile(path_ncdf):
                print(f'> Create {indicator} export...', end='\n')
                extract_ncdf_indicator(
                    paths_data=paths, param_type=data_type, sim_points_gdf=sim_points_gdf,
                    indicator=indicator, timestep=timestep, start=files_setup['historical'][0], path_result=path_ncdf,
                )

            print(f'################################ FORMAT DATA ################################', end='\n')
            print(f'> Load from {indicator} export...', end='\n')
            path_ncdf = f"{dict_paths['folder_study_data']}QA_mon_YE_rcp85.nc"
            indicator='QA'
            ds = xr.open_dataset(path_ncdf)
            indicator_cols = [i for i in list(ds.variables) if indicator in i]

            # Define geometry for each data (Points hydro, Polygon climate)
            print(f'> Match geometry and data...', end='\n')
            other_dimension = None
            if data_type == 'climate':
                geometry_dict = {row['gid']: row['geometry'] for _, row in regions_shp.iterrows()}
                ds = ds.assign_coords(geometry=('region', [geometry_dict[code] for code in ds['region'].values]))
                ds = ds.rename({'region': 'id_geometry'})
            else:
                sim_points_gdf_simplified = sim_points_gdf.copy()
                sim_points_gdf_simplified.simplify(tolerance, preserve_topology=True)
                geometry_dict = sim_points_gdf['geometry'].to_dict()
                ds['geometry'] = ('code', [geometry_dict[code] for code in ds['code'].values])
                # ds = ds.assign_coords(geometry=('code', [geometry_dict[code] for code in ds['code'].values]))
                ds = ds.rename({'code': 'id_geometry'})

                if indicator == 'QA':
                    # other_dimension = {'time': 'time.month'}
                    ds = ds.assign_coords(month=ds['time.month'])
                    other_dimension = 'month'

            print(f'> Define horizons...', end='\n')
            # Define horizons
            ds = define_horizon(ds, files_setup)
            # Compute mean value for each horizon
            ds = compute_mean_by_horizon(ds=ds, indicator_cols=indicator_cols,
                                         files_setup=files_setup, other_dimension=other_dimension)

            indicator_horizon = [i for i in list(ds.variables) if indicator+'_by_horizon' in i]

            # ds_mean_spatial_horizon = apply_statistic(ds=ds.to_array(dim='new'),
            #                                           function=files_setup['function'],
            #                                           q=files_setup['quantile']).to_dataset(name=indicator)

            # Compute deviation to historical
            ds = compute_deviation_to_ref(ds, cols=indicator_horizon)
            indicator_horizon_deviation = [i for i in list(ds.variables) if indicator+'_by_horizon_deviation' in i]

            print(f'################################ PLOT DATA ################################', end='\n')
            print(f"> Initialize plot...")
            col_name='horizon'
            col_headers = {'horizon1': 'Horizon 1 (2021-2050)',
                           'horizon2': 'Horizon 2 (2041-2070)',
                           'horizon3': 'Horizon 3 (2070-2100)'}

            row_name = None
            iterates = {'': None}
            discretize = 7
            vmax = None
            if indicator == 'QA':
                vmax = 100
                row_name = 'month'
                discretize = 11
                iterates = {
                    'DJF' : {12: 'Décembre',
                             1: 'Janvier',
                             2: 'Février'},
                    'MAM': {3: 'Mars',
                            4: 'Avril',
                            5: 'Mai'},
                    'JJA': {6: 'Juin',
                            7: 'Juillet',
                            8: 'Août'},
                    'SON': {9: 'Septembre',
                            10: 'Octobre',
                            11: 'Novembre'},
                }

            gdf = gpd.GeoDataFrame({
                'geometry': ds['geometry'].values,
                'id_geometry': ds['id_geometry'].values
            })

            cbar_title = indicator + ' relatif (%)'

            dict_shapefiles = {'rivers_shp': {'shp': study_rivers_shp_simplified, 'color': 'royalblue', 'linewidth': 2, 'zorder': 20, 'alpha': 0.5},
                               'background_shp': {'shp': regions_shp_simplified, 'color': 'gainsboro', 'edgecolor': 'black', 'zorder': 0},
                               'study_shp': {'shp': study_ug_shp_simplified, 'color': 'white', 'edgecolor': 'firebrick', 'zorder': 1, 'linewidth': 1.2},}

            print(f"> Map plot...")
            for key, value in iterates.items():
                print(f"> Map plot {indicator}_{timestep}_{rcp}_{key}...")
                path_results = f"{dict_paths['folder_study_figures']}{indicator}_{timestep}_{rcp}_{key}_"
                # Plot map
                plot_map(gdf, ds, indicator=indicator_horizon_deviation[0], path_result=path_results+'map.pdf',
                         row_name=row_name, row_headers=value, col_name=col_name, col_headers=col_headers,
                         cbar_title=cbar_title, title=None, dict_shapefiles=dict_shapefiles, percent=True, bounds=bounds,
                         discretize=discretize, palette='BrBG', fontsize=14, font='sans-serif', vmax=vmax)

