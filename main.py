import pyfiglet
ascii_banner = pyfiglet.figlet_format("FORMAT NCDF")
print(f'##########################################################################################', end='\n')
print(ascii_banner, end='\n')

print(f'################################ IMPORT & INITIALIZATION ################################', end='\n')

print(f'> General imports...', end='\n')
import sys
import os
# sys.path.insert(0, os.getcwd())
import time
import json


print(f'> Local imports...', end='\n')
from global_functions.load_data import *
from global_functions.format_data import *
from plot_functions.plot_map import *
from plot_functions.plot_lineplot import *
from plot_functions.plot_boxplot import *
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
print(f'################################ DEFINE STUDY AREA ################################', end='\n')
print(f'> Load shapefiles...', end='\n')
regions_shp, study_hydro_shp, study_climate_shp, rivers_shp = load_shp(dict_paths, files_setup)

# Check if study area is already matched with sim points
print(f'> Searching sim points in study area...', end='\n')
for data_type, path in dict_paths['dict_study_points_sim'].items():
    if not os.path.isfile(path):
        print(f'>> Find {data_type} data points in study area')
        sim_all_points_info = open_shp(path_shp=dict_paths['dict_global_points_sim'][data_type])
        if data_type == 'hydro':
            overlay_shapefile(shapefile=study_hydro_shp, data=sim_all_points_info,
                              path_result=path)
        else:
            overlay_shapefile(shapefile=study_climate_shp, data=sim_all_points_info,
                              path_result=path)
    else:
        print(f'>> {data_type.capitalize()} data points already in the study area')

print(f'> Simplify shapefiles...', end='\n')
study_hydro_shp_simplified, study_climate_shp_simplified, study_rivers_shp_simplified, regions_shp_simplified, bounds = (
    simplify_shapefiles(study_hydro_shp, study_climate_shp, rivers_shp, regions_shp, tolerance=1000, zoom=50000))

# path_sh = f"/home/bcalmel/Documents/2_data/climat/SH/Liste_SH_TX_metro.csv"
#
# data_sh = pd.read_csv(path_sh, sep=";", header=None, engine="python", names=[str(i) for i in range(9)])
# new_header = data_sh.iloc[2] #grab the first row for the header
# data_sh = data_sh[3:] #take the data less the header row
# data_sh.columns = new_header
#
# search_path = f"/home/bcalmel/Documents/2_data/climat/SH/SH_TN_metropole"
# result_path = f"/home/bcalmel/Documents/2_data/climat/SH/Loire/SH_TN_Loire"
# if not os.path.isdir(result_path):
#     os.makedirs(result_path)
# import glob
# import shutil
# df_station = pd.read_csv(f"/home/bcalmel/Documents/2_data/climat/SH/Loire/Liste_SH_IN_Loire.csv", sep=';')
# stations = df_station['num_poste'].to_list()
# stations = [str(i) for i in stations]
# stations = [i[6:] for i in stations]
# files_in_dir = glob.glob(f"{search_path}/*")
# for file in files_in_dir:
#     if any(word in os.path.basename(file) for word in stations):
#         data = pd.read_csv(file, sep=";", header=None, engine="python", names=[str(i) for i in range(3)])
#         new_header = data.iloc[12]
#         data = data[13:] #take the data less the header row
#         data.columns = new_header
#         data.to_csv(result_path +os.sep+ os.path.basename(file), index=False)
#         # shutil.copy2(file, result_path +os.sep+ os.path.basename(file))
#
# data = df_station
# shapefile = sim_points_gdf

print(f'################################ RUN OVER NCDF ################################', end='\n')
# Get paths for selected sim
print(f'> Load ncdf data paths...', end='\n')
path_files = get_files_path(dict_paths=dict_paths, setup=files_setup)

# Run among data type climate/hydro
data_type='climate'
subdict=path_files[data_type]
rcp='rcp85'
subdict2=subdict[rcp]
indicator = "tasminAdjust"
paths = subdict2[indicator]
for data_type, subdict in path_files.items():
    # Load simulation points for current data type
    sim_points_gdf = open_shp(path_shp=dict_paths['dict_study_points_sim'][data_type])

    if data_type == "hydro":
        # sim_points_gdf = sim_points_gdf[sim_points_gdf['REFERENCE'] == 1]
        sim_points_gdf = sim_points_gdf[sim_points_gdf['n'] >= 4]
        valid_stations = pd.isna(sim_points_gdf['PointsSupp'])
        sim_points_gdf = sim_points_gdf[valid_stations].reset_index(drop=True).set_index('Suggestion')
        sim_points_gdf.index.names = ['name']
    else:
        sim_points_gdf['weight'] = sim_points_gdf['surface'] / sim_points_gdf['total_surf']

    for rcp, subdict2 in subdict.items():
        for indicator, paths in subdict2.items():
            # split_indicator = indicator_raw.split('-')
            # indicator = split_indicator[0]
            timestep = 'ME'
            # if len(split_indicator) > 1:
            #     timestep = split_indicator[1]
            #
            # if timestep == 'mon':
            #     timestep = 'M'

            path_ncdf = f"{dict_paths['folder_study_data']}{indicator.split('$')[0]}_{timestep}_{rcp}.nc"

            if not os.path.isfile(path_ncdf):
                print(f'> Create {indicator} export...', end='\n')
                if len(paths) > 0 :
                    extract_ncdf_indicator(
                        paths_data=paths, param_type=data_type, sim_points_gdf=sim_points_gdf, indicator=indicator,
                        timestep=timestep, start=files_setup['historical'][0], path_result=path_ncdf,
                    )
                else:
                    print(f'> Invalid {indicator} name', end='\n')
            else:
                print(f'> {path_ncdf} already exists', end='\n')

            continue

            print(f'################################ FORMAT DATA ################################', end='\n')
            print(f'> Load from {indicator} export...', end='\n')
            # path_ncdf = f"{dict_paths['folder_study_data']}QA_mon_ME_rcp85.nc"
            # indicator='QA'
            ds = xr.open_dataset(path_ncdf)
            # TODO ITERATE OVER QA INDICATORS
            indicator_cols = [i for i in list(ds.variables) if indicator in i]

            # Define geometry for each data (Points hydro, Polygon climate)
            print(f'> Match geometry and data...', end='\n')
            other_dimension = None
            if data_type == 'climate':
                sim_points_gdf_simplified = open_shp(path_shp=dict_paths['dict_global_points_sim'][data_type])
                # TODO Rename 'name' with id_geometry
                ds = ds.assign_coords(geometry=(
                    'name', sim_all_points_info.set_index('name').loc[ds['name'].values, 'geometry']))

                # Find matching area
                geometry_dict = {row['gid']: row['geometry'] for _, row in study_hydro_shp.iterrows()}
                region_da = xr.DataArray(sim_points_gdf['gid'].values, dims=['name'],
                                         coords={'name': sim_points_gdf['name']})
                ds = ds.assign_coords(region=region_da)
                # ds = ds.assign_coords(geometry=('region', [geometry_dict[code] for code in ds['region'].values]))
                # ds = ds.rename({'region': 'id_geometry'})
            else:
                sim_points_gdf_simplified = sim_points_gdf.copy()
                sim_points_gdf_simplified = sim_points_gdf_simplified.simplify(tolerance=1000, preserve_topology=True)
                geometry_dict = sim_points_gdf['geometry'].to_dict()
                # TODO Rename 'code' with id_geometry
                ds['geometry'] = ('code', [
                    geometry_dict[code] if code in geometry_dict.keys() else None for code in ds['code'].values
                ])
                ds = ds.rename({'code': 'id_geometry'})

                # Compute PK
                value = compute_river_distance(rivers_shp, sim_points_gdf_simplified, river_name='loire',
                                               start_from='last')
                sim_points_gdf['PK'] = value

                if indicator == 'QA':
                    # other_dimension = {'time': 'time.month'}
                    ds = ds.assign_coords(month=ds['time.month'])
                    other_dimension = 'month'
                elif indicator == 'seas':
                    def get_season(month):
                        if month in [12, 1, 2]:
                            return 'DJF'
                        elif month in [3, 4, 5]:
                            return 'MAM'
                        elif month in [6, 7, 8]:
                            return 'JJA'
                        else:
                            return 'SON'

                    seasons = xr.DataArray(
                        [get_season(i) for i in ds['time.month'].values],
                        coords={'time': ds['time']},
                        dims='time'
                    )
                    ds = ds.assign_coords(season=seasons)
                    other_dimension = 'season'

            print(f'> Define horizons...', end='\n')
            # Define horizons
            ds = define_horizon(ds, files_setup)

            # Return period
            # ds = compute_return_period(ds, indicator_cols, files_setup, return_period=5, other_dimension=other_dimension)

            # Compute mean value for each horizon for each sim
            ds = compute_mean_by_horizon(ds=ds, indicator_cols=indicator_cols,
                                         files_setup=files_setup, other_dimension=other_dimension)

            indicator_horizon = [i for i in list(ds.variables) if '_by_horizon' in i]

            # Compute deviation/difference to reference
            ds = compute_deviation_to_ref(ds, cols=indicator_horizon)
            indicator_horizon_deviation_sims = [i for i in list(ds.variables) if '_by_horizon_deviation' in i]
            indicator_horizon_difference_sims = [i for i in list(ds.variables) if '_by_horizon_difference' in i]

            # Compute statistic among all sims
            ds_deviation_stats = apply_statistic(ds=ds[indicator_horizon_deviation_sims].to_array(dim='new'),
                                                 function=files_setup['function'],
                                                 q=files_setup['quantile']
                                                 )
            ds_difference_stats = apply_statistic(ds=ds[indicator_horizon_difference_sims].to_array(dim='new'),
                                                  function=files_setup['function'],
                                                  q=files_setup['quantile']
                                                  )
            indicator_horizon_deviation = [f"{indicator}_deviation_{i}" for i in
                                           list(ds_deviation_stats.data_vars)]
            indicator_horizon_difference = [f"{indicator}_difference_{i}" for i in
                                            list(ds_difference_stats.data_vars)]
            ds[indicator_horizon_deviation] = ds_deviation_stats
            ds[indicator_horizon_difference] = ds_difference_stats

            # Merge sim points information to dataset
            for col in sim_points_gdf:
                ds[col] = ("code", sim_points_gdf[col])

            print(f'################################ PLOT DATA ################################', end='\n')
            print(f"> Initialize plot...")
            # col_name='horizon'
            # col_headers = {'horizon1': 'Horizon 1 (2021-2050)',
            #                'horizon2': 'Horizon 2 (2041-2070)',
            #                'horizon3': 'Horizon 3 (2070-2100)'}
            #
            row_name = None

            subplots = {'': None}
            discretize = 7
            vmax = None
            percent = True
            if indicator == 'QA':
                vmax = 100
                row_name = 'month'
                discretize = 11
                subplots = {
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

            shape_hp = {
                'CTRIP': 'o',
                'EROS': 'H',
                'GRSD': '*',
                'J2000': 's',
                'MORDOR-TS': '^',
                'MORDOR-SD': 'v',
                'SIM2': '>',
                'SMASH': '<',
                'ORCHIDEE': 'D',
            }

            # gdf = gpd.GeoDataFrame({
            #     'geometry': ds['geometry'].values,
            #     'id_geometry': ds['id_geometry'].values
            # })

            dict_shapefiles = {'rivers_shp': {'shp': study_rivers_shp_simplified, 'color': 'paleturquoise',
                                              'linewidth': 1, 'zorder': 2, 'alpha': 0.8},
                               'background_shp': {'shp': regions_shp_simplified, 'color': 'gainsboro',
                                                  'edgecolor': 'black', 'zorder': 0},
                               }
            if data_type == 'hydro':
                dict_shapefiles |= {'study_shp': {'shp': study_climate_shp_simplified, 'color': 'white',
                                                  'edgecolor': 'k', 'zorder': 1, 'linewidth': 1.2},}
            else:
                dict_shapefiles |= {'study_shp': {'shp': study_climate_shp_simplified, 'color': 'white',
                                                  'edgecolor': 'k', 'zorder': 1, 'linewidth': 1.2},}

            print(f"> Map plot...")
            path_indicator_figures = dict_paths['folder_study_figures'] + indicator + os.sep
            if not os.path.isdir(path_indicator_figures):
                os.makedirs(path_indicator_figures)

            for key, value in subplots.items():
                if key is None:
                    path_results = f"{path_indicator_figures}{timestep}_{rcp}_"
                else:
                    path_results = f"{path_indicator_figures}{timestep}_{rcp}_{key}_"
                cols_map = {
                    'names_coord': 'horizon',
                    'values_var': ['horizon1', 'horizon2', 'horizon3'],
                    'names_plot': ['Horizon 1 (2021-2050)', 'Horizon 2 (2041-2070)', 'Horizon 3 (2070-2100)']
                }

                rows = {
                    'names_coord': row_name,
                    'values_var': list(value.keys()),
                    'names_plot': list(value.values())
                }
                rows = None

                # Plot map
                # Relative
                print(f"> Relative map plot {indicator}_{timestep}_{rcp}_{key}...")
                mapplot(gdf=sim_points_gdf_simplified, ds=ds, indicator_plot=indicator_horizon_deviation[0], path_result=path_indicator_figures+'map_deviation.pdf',
                        cols=cols_map, rows=rows,
                        cbar_title=f"{indicator} relatif (%)", title=None, dict_shapefiles=dict_shapefiles, percent=True, bounds=bounds,
                        discretize=8, palette='BrBG', fontsize=14, font='sans-serif', vmax=100)

                # Abs diff
                print(f"> Difference map plot {indicator}_{timestep}_{rcp}_{key}...")
                mapplot(gdf, ds, indicator_plot=indicator_horizon_difference[0], path_result=path_indicator_figures+'map_difference.pdf',
                        cols=cols_map, rows=rows,
                        cbar_title=f"{indicator} difference", title=None, dict_shapefiles=dict_shapefiles, percent=False, bounds=bounds,
                        discretize=4, palette='RdBu_r', cmap_zero=True, fontsize=14, font='sans-serif', edgecolor='k', vmin=0, vmax=4)

                # Plot number of HM by station
                indicator = 'n'
                ds = sim_points_gdf
                gdf = sim_points_gdf
                cols_map = None
                rows = None

                mapplot(gdf, ds, indicator_plot='J2000', path_result=path_indicator_figures+'HM.pdf',
                        cols=cols_map, rows=rows,
                        cbar_title=f"Nombre de HM", title=None, dict_shapefiles=dict_shapefiles, percent=False, bounds=bounds,
                        discretize=6, cbar_ticks='mid', palette='RdBu_r', cmap_zero=True, fontsize=14, font='sans-serif', edgecolor='k', vmin=3.5, vmax=9.5)


                # Plot sim by station
                # 3 cols * 3 rows
                hm = list(shape_hp.keys())


                gdf = gpd.GeoDataFrame({i: ds[i] for i in hm + ['id_geometry', 'geometry']})
                gdf = gpd.GeoDataFrame({i: ds[i] for i in ['id_geometry', 'geometry']})
                cols_map = {
                    'values_var': hm,
                }
                rows = 3

                mapplot(gdf=sim_points_gdf, ds=ds, indicator_plot=hm, path_result=f"{path_indicator_figures}HM_by_sim.pdf",
                        cols=cols_map, rows=3,
                        cbar_title=f"Simulations présentes", title=None, dict_shapefiles=dict_shapefiles, percent=False, bounds=bounds,
                        discretize=2, cbar_ticks='mid', palette='RdBu_r', cmap_zero=True, fontsize=14, font='sans-serif', edgecolor='k', vmin=-0.5, vmax=1.5,
                        cbar_values=['Absente', 'Présente'])



                print(f"> Relative line plot {indicator}_{timestep}_{rcp}_{key}...")


                x_axis = {
                    'names_var': 'time',
                    'values_var': 'time',
                    'name_axis': 'Date'
                }
                y_axis = {
                    'names_var': 'indicator',
                    'values_var': indicator_cols,
                    'name_axis': 'QA'
                }

                rows = {'names_var': 'id_geometry',
                        'values_var': ds['id_geometry'].where(ds['geometry'].notnull(), drop=True).values,
                        'names_plot': ds['id_geometry'].where(ds['geometry'].notnull(), drop=True).values}
                cols = {
                    'names_var': row_name,
                    'values_var': list(value.keys()),
                    'names_plot': list(value.values())
                }

                lineplot(ds, x_axis, y_axis, path_result=path_indicator_figures+'lineplot.pdf', cols=cols, rows=rows,
                         title=None, percent=False, fontsize=14, font='sans-serif', ymax=None, plot_type='line')

                # Cols  and rows of subplots
                cols = {'names_var': 'id_geometry', 'values_var': ['K001872200', 'M850301010'], 'names_plot': ['Station 1', 'Station 2']}
                rows = {
                    'names_var': row_name,
                    'values_var': list(value.keys()),
                    'names_plot': list(value.values())
                }

                y_axis = {
                    'values_var': ['QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_CTRIP',
                                   'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_EROS',
                                   'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_GRSD'],
                    'names_plot': ['QA CTRIP', 'QA EROS', 'QA GRSD']
                }
                x_axis = {
                    'names_var': 'horizon',
                    'values_var': ['horizon1', 'horizon2', 'horizon3'],
                    'names_plot': ['H1', 'H2', 'H3']
                }

                # TEST 1
                cols = {
                    'names_var': 'id_geometry',
                    'values_var': ['K001872200', 'M850301010'],
                    'names_plot': ['K001872200', 'M850301010'],
                }
                rows = {
                    'names_var': row_name,
                    'values_var': list(value.keys()),
                    'names_plot': list(value.values())
                }
                x_axis = {
                    'names_var': 'horizon',
                    'values_var': ['horizon1', 'horizon2', 'horizon3'],
                    'names_plot': ['H1', 'H2', 'H3']
                }
                y_axis = {
                    'names_var': 'indicator',
                    'values_var': ['QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_CTRIP',
                                   'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_EROS',
                                   'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_GRSD'],
                    'names_plot': ['QA CTRIP', 'QA EROS', 'QA GRSD']
                }

                # TEST 2
                cols = {'names_var': 'horizon',
                        'values_var': ['horizon1', 'horizon2', 'horizon3'],
                        'names_plot': ['H1', 'H2', 'H3']}
                rows = {
                    'names_var': 'indicator',
                    'values_var': ['QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_CTRIP',
                                   'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_EROS',
                                   'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_GRSD'],
                    'names_plot': ['QA CTRIP', 'QA EROS', 'QA GRSD']
                }
                x_axis = {
                    'names_var': row_name,
                    'values_var': list(value.keys()),
                    'names_plot': list(value.values())
                }
                y_axis = {
                    'names_var': 'id_geometry',
                    'values_var': ['K001872200', 'M850301010'],
                    'names_plot': ['K001872200', 'M850301010'],
                }

                # TEST 3
                cols = {'names_var': 'indicator',
                        'values_var': ['QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_CTRIP',
                                       'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_EROS',
                                       'QA_historical-rcp85_CNRM-CM5_ALADIN63_ADAMONT_GRSD'],
                        'names_plot': ['QA CTRIP', 'QA EROS', 'QA GRSD']}
                rows = {
                    'names_var': 'horizon',
                    'values_var': ['horizon1', 'horizon2', 'horizon3'],
                    'names_plot': ['H1', 'H2', 'H3']
                }
                x_axis = {
                    'names_var': 'id_geometry',
                    'values_var': ['K001872200', 'M850301010'],
                    'names_plot': ['K001872200', 'M850301010'],
                    'name_axis': 'Stations'

                }
                y_axis = {
                    'names_var': row_name,
                    'values_var': list(value.keys()),
                    'names_plot': list(value.values()),
                    'name_axis': indicator + ' (m3/s)'
                }

                boxplot(ds, x_axis, y_axis, path_result=path_indicator_figures+'boxplot.pdf', cols=cols, rows=rows,
                        title=None, percent=False, palette='BrBG', fontsize=14, font='sans-serif', ymax=None)

                import matplotlib.lines as mlines
                from sklearn.cluster import KMeans
                from scipy.spatial import Voronoi, voronoi_plot_2d

                stations = list(sim_points_gdf[sim_points_gdf['INDEX'].isin([1774,1895,2294,1633,1786,2337])].index)
                stations = ['M842001000']

                x = ds[indicator_horizon_deviation_sims].sel(season='DJF', horizon='horizon3', id_geometry=stations)
                y = ds[indicator_horizon_deviation_sims].sel(season = 'JJA', horizon='horizon3', id_geometry=stations)
                x_list = []
                y_list = []
                for var in x.data_vars:
                    if any(np.isnan(x[var].values)) and any(~np.isnan(y[var].values)):
                        x_list.append(np.nanmedian(x[var].values))
                        y_list.append(np.nanmedian(y[var].values))

                # Narratifs by clustering (K-means)
                kmeans = KMeans(n_clusters=4, random_state=0)
                df = pd.DataFrame({'x': x_list, 'y': y_list})
                df['cluster'] = kmeans.fit_predict(df[['x','y']])
                # Sélectionner un point représentatif par cluster (par exemple, le plus proche du centroïde)
                representative_points = df.loc[
                    df.groupby('cluster').apply(
                        lambda group: group[['x', 'y']].sub(kmeans.cluster_centers_[group.name]).pow(2).sum(axis=1).idxmin()
                    )
                ]
                couples = list(zip(representative_points['x'], representative_points['y']))
                # Obtenir les centroïdes
                centroids = kmeans.cluster_centers_
                # Créer un Voronoi pour délimiter les aires
                vor = Voronoi(centroids)

                fig, ax = plt.subplots(1, 1, figsize=(6,4), constrained_layout=True)
                ax.grid()
                dict_hm = {key: [] for key in shape_hp.keys()}

                for var in x.data_vars:
                    print(var)
                    # Identifier la clé du dictionnaire présente dans le nom de la variable
                    hm = next((key for key in shape_hp if key in var), 'NONE')
                    marker = shape_hp[hm]
                    dict_hm[hm].append(var)

                    # Tracer la variable
                    x_value = np.nanmedian(x[var].values)
                    y_value = np.nanmedian(y[var].values)

                    if (x_value, y_value) in couples:
                        plt.scatter(x_value, y_value, marker=marker, alpha=1,
                                    color='green', zorder=2)
                    else:
                        plt.scatter(x_value, y_value, marker=marker, alpha=0.8,
                                    color='k', zorder=0)


                for key, shape in shape_hp.items():
                    plt.scatter(np.nanmedian(x[dict_hm[key]].to_array()), np.nanmedian(y[dict_hm[key]].to_array()),
                                marker=shape, alpha=0.8,
                                color='firebrick', zorder=1)

                # Tracer les aires des clusters avec Voronoi
                voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False, line_colors='green',
                                line_width=0.4, line_alpha=0.6, point_size=0, linestyle='--')


                ax.spines[['right', 'top']].set_visible(False)
                ax.set_xlim(-70, 70)
                ax.set_ylim(-70, 70)
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                ax.xaxis.set_major_formatter(mtick.PercentFormatter())
                ax.set_ylabel('Qm estival')
                ax.set_xlabel('Qm hivernal')
                legend_handles = [
                    mlines.Line2D([], [], color='black', marker=shape, linestyle='None', markersize=8,
                                  label=f'{key}')
                    for key, shape in shape_hp.items()
                ]
                plt.legend(
                    handles=legend_handles,
                    loc="center left",  # Position relative
                    bbox_to_anchor=(1, 0.5)  # Placer la légende à droite du graphique
                )

                plt.savefig(f"/home/bcalmel/Documents/3_results/HMUC_Loire_Bretagne/figures/global/narratifs.pdf",
                                            bbox_inches='tight')


print(f'################################ END ################################', end='\n')
input("Press Enter to close")