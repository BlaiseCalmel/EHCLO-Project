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
from plot_functions.run_plot import *
from global_functions.format_data import *
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
                              path_result=path, force_contains={'Suggesti_2': ['LA LOIRE', 'L\'ALLIER']})
        else:
            overlay_shapefile(shapefile=study_climate_shp, data=sim_all_points_info,
                              path_result=path)
    else:
        print(f'>> {data_type.capitalize()} data points already in the study area')

print(f'> Simplify shapefiles...', end='\n')
study_hydro_shp_simplified, study_climate_shp_simplified, study_rivers_shp_simplified, regions_shp_simplified, bounds = (
    simplify_shapefiles(study_hydro_shp, study_climate_shp, rivers_shp, regions_shp, tolerance=1000, zoom=1000))

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
data_type='hydro'
subdict=path_files[data_type]
rcp='rcp85'
subdict2=subdict[rcp]
indicator = "QA_yr"
paths = subdict2[indicator]

hydro_sim_points_gdf = open_shp(path_shp=dict_paths['dict_study_points_sim']['hydro'])
hydro_sim_points_gdf_simplified = hydro_sim_points_gdf[hydro_sim_points_gdf['n'] >= 4]
hydro_sim_points_gdf_simplified = hydro_sim_points_gdf_simplified.reset_index(drop=True).set_index('Suggestion')
hydro_sim_points_gdf_simplified.index.names = ['name']

climate_sim_points_gdf = open_shp(path_shp=dict_paths['dict_study_points_sim']['climate'])
climate_sim_points_gdf_simplified = climate_sim_points_gdf.loc[
    climate_sim_points_gdf.groupby('name')['gid'].idxmin()].reset_index(drop=True)

for data_type, subdict in path_files.items():
    # Load simulation points for current data type
    # sim_points_gdf = open_shp(path_shp=dict_paths['dict_study_points_sim'][data_type])

    if data_type == "hydro":
        sim_points_gdf_simplified = hydro_sim_points_gdf_simplified
    else:
        sim_points_gdf_simplified = climate_sim_points_gdf_simplified
        # sim_points_gdf['weight'] = sim_points_gdf['surface'] / sim_points_gdf['total_surf']

    for rcp, subdict2 in subdict.items():
        for indicator, paths in subdict2.items():
            print(f'################################ RUN {data_type} {rcp} {indicator} ################################', end='\n')
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
                        paths_data=paths, param_type=data_type, sim_points_gdf=sim_points_gdf_simplified, indicator=indicator,
                        timestep=timestep, start=files_setup['historical'][0], path_result=path_ncdf,
                    )
                else:
                    print(f'> Invalid {indicator} name', end='\n')
            else:
                print(f'> {path_ncdf} already exists', end='\n')

            print(f'################################ FORMAT DATA ################################', end='\n')
            print(f'> Load from {indicator} export...', end='\n')
            # path_ncdf = f"{dict_paths['folder_study_data']}QA_seas-JJA_ME_rcp85.nc"
            # indicator='QA_seas-JJA'

print(f'################################ PLOT INDICATOR ################################', end='\n')
for indicator in files_setup['hydro_indicator'] + files_setup['climate_indicator']:
    print(indicator)
    path_ncdf = f"{dict_paths['folder_study_data']}{indicator.split('$')[0]}_ME_rcp85.nc"
    path_indicator_figures = dict_paths['folder_study_figures'] + indicator + os.sep

    if not os.path.isdir(path_indicator_figures):
        os.makedirs(path_indicator_figures)

    # Compute PK
    if indicator in files_setup['hydro_indicator']:
        data_type = 'hydro'
        sim_points_gdf_simplified = hydro_sim_points_gdf_simplified
        loire = sim_points_gdf_simplified.loc[sim_points_gdf_simplified['gid'] < 7]
        value = compute_river_distance(rivers_shp, loire, river_name='loire',
                                       start_from='last')
        sim_points_gdf_simplified['PK'] = np.nan
        sim_points_gdf_simplified.loc[sim_points_gdf_simplified['gid'] < 7, 'PK'] = value
        edgecolor = 'k'
    else:
        data_type = 'climate'
        sim_points_gdf_simplified = climate_sim_points_gdf_simplified
        sim_points_gdf_simplified = sim_points_gdf_simplified.set_index('name')
        edgecolor = None

    ds = xr.open_dataset(path_ncdf)
    ds, variables = format_dataset(ds, data_type, files_setup)

    sim_points_gdf_simplified = sim_points_gdf_simplified.loc[ds.gid]

    dict_shapefiles = define_plot_shapefiles(regions_shp_simplified, study_climate_shp_simplified, study_rivers_shp_simplified,
                           indicator, files_setup)

    # Plot map
    # Relative
    print(f"> Map plot...")
    if indicator in files_setup['climate_indicator']:
        print(f">> Difference map plot {indicator}")
        plot_map_indicator_climate(gdf=sim_points_gdf_simplified, ds=ds, indicator_plot='horizon_deviation_mean',
                      path_result=path_indicator_figures+'map_difference.pdf',
                      cbar_title=f"{indicator} mean difference", cbar_ticks=None, title=None, dict_shapefiles=dict_shapefiles,
                      percent=False, bounds=bounds, palette='RdBu_r', cbar_midpoint='zero', fontsize=14,
                      font='sans-serif', discretize=7, edgecolor=edgecolor, markersize=75, vmin=0, cbar_values=1)

    if indicator in files_setup['hydro_indicator']:
        print(f">> Deviation map plot by HM")
        horizons = {'horizon1': 'Horizon 1 (2021-2050)',
                    'horizon2': 'Horizon 2 (2041-2070)',
                    'horizon3': 'Horizon 3 (2070-2099)',
                    }
        mean_by_hm = [s for sublist in variables['hydro_model_deviation'].values() for s in sublist if "mean" in s]
        vmax = math.ceil(abs(ds[mean_by_hm].to_array()).max() / 5) * 5
        for key, value in horizons.items():
            print(f">>> {value}")
            plot_map_indicator_hm(gdf=sim_points_gdf_simplified, ds=ds.sel(horizon=key), variables=variables,
                                  path_result=path_indicator_figures+f'maps_variation_mean_{key}.pdf',
                                  cbar_title=f"Variation moyenne {indicator} (%)", title=value, cbar_midpoint='zero',
                                  dict_shapefiles=dict_shapefiles, percent=True, bounds=bounds, edgecolor=edgecolor,
                                  markersize=75, discretize=10, palette='BrBG', fontsize=14, font='sans-serif',
                                  vmin=None, vmax=vmax)

        # Sim by PK + quantile
        print(f"> Linear plot...")

        if 'PK' in sim_points_gdf_simplified.columns:
            ds = ds.assign(PK=("gid", sim_points_gdf_simplified.loc[ds.gid.values, "PK"]))

            villes = ['Villerest', 'Nevers', 'Orleans', 'Tours', 'Saumur', 'Nantes'] #'Blois',
            regex = "|".join(villes)
            vlines = sim_points_gdf_simplified[sim_points_gdf_simplified['Suggesti_2'].str.contains(regex, case=False, na=False)]
            vlines.loc[: ,'color'] = 'none'
            cities = [i.split(' A ')[-1].split(' [')[0] for i in vlines['Suggesti_2']]
            vlines.insert(loc=0, column='label', value=cities)
            vlines['annotate'] = 0.02

            narratives = {
                "HadGEM2-ES_ALADIN63_ADAMONT": {'color': '#569A71', 'zorder': 10, 'label': 'HadGEM2-ES_ALADIN63_ADAMONT',
                                                'linewidth': 1},
                "CNRM-CM5_ALADIN63_ADAMONT": {'color': '#EECC66', 'zorder': 10, 'label': 'CNRM-CM5_ALADIN63_ADAMONT',
                                              'linewidth': 1},
                "EC-EARTH_HadREM3-GA7_ADAMONT": {'color': '#E09B2F', 'zorder': 10, 'label': 'EC-EARTH_HadREM3-GA7_ADAMONT',
                                                 'linewidth': 1},
                "HadGEM2-ES_CCLM4-8-17_ADAMONT": {'color': '#791F5D', 'zorder': 10, 'label': 'HadGEM2-ES_CCLM4-8-17_ADAMONT',
                                                  'linewidth': 1},
            }

            print(f">> Linear deviation - x: PK, y: {indicator}, row: HM, col: Horizon")
            plot_linear_pk_hm(ds,
                              simulations=variables['hydro_model_deviation_sim_horizon'],
                              narratives=narratives,
                              name_x_axis=f'PK (km)',
                              name_y_axis=f'{indicator} variation (%)',
                              percent=True,
                              vlines=vlines,
                              path_result=path_indicator_figures+f'lineplot_variation_x-PK_y-{indicator}_row-HM_col-horizon.pdf')


            print(f">> Linear deviation - x: PK, y: {indicator}, row: Narratif, col: Horizon")
            plot_linear_pk_narrative(ds,
                           simulations=variables['simulation_horizon_deviation_by_sims'],
                           narratives=narratives,
                           name_x_axis=f'PK (km)',
                           name_y_axis=f'{indicator} variation (%)',
                           percent=True,
                           vlines=vlines,
                           path_result=path_indicator_figures+f'lineplot_variation_x-PK_y-{indicator}_row-narrative_col-horizon.pdf')

            print(f">> Linear deviation - x: PK, y: {indicator}, col: Horizon")
            plot_linear_pk(ds,
                           simulations=variables['simulation_horizon_deviation_by_sims'],
                           narratives=narratives,
                           name_x_axis=f'PK (km)',
                           name_y_axis=f'{indicator} variation (%)',
                           percent=True,
                           vlines=vlines,
                           path_result=path_indicator_figures+f'lineplot_variation_x-PK_y-{indicator}_col-horizon.pdf')

            print(f">> Linear timeline deviation - x: time, y: {indicator}, row/col: Stations ref")
            plot_linear_time(ds,
                             simulations=variables['simulation_deviation'],
                             path_result=path_indicator_figures+f'lineplot_variation_x-time_y-{indicator}_row-col-stations-ref.pdf',
                             narratives=narratives,
                             name_x_axis='Date',
                             name_y_axis=f'{indicator} variation (%)',
                             percent=True,
                             vlines=None)


        print(f"> Box plot...")
        print(f">> Boxplot deviation by horizon and selected stations")
        plot_boxplot_station(ds=ds, simulations=variables['simulation_horizon_deviation_by_sims'],
                             name_y_axis=f'{indicator} variation (%)', percent=True,
                             path_result=path_indicator_figures+'boxplot_deviation.pdf')


path = (f'/media/bcalmel/Ultra Touch/Tonio/Données SWI/Fichiers netcdf/'
        f'SWI_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_CNRM-ALADIN63_v2_MF-ADAMONT-SAFRAN-1980-2011_MF-SIM2_day_20050801-21000731.nc')
ds = xr.open_dataset(path)


ds['GroundwaterBody']
dir(ds)
print(f'################################ PLOT GLOBAL ################################', end='\n')
path_global_figures = dict_paths['folder_study_figures'] + 'global' + os.sep
if not os.path.isdir(path_global_figures):
    os.makedirs(path_global_figures)

print(f"> Plot HM by station...")
plot_map_HM_by_station(hydro_sim_points_gdf_simplified, dict_shapefiles, bounds, path_global_figures)

print(f"> Plot #HM by station and Ref station...")
plot_map_N_HM_ref_station(hydro_sim_points_gdf_simplified, dict_shapefiles, path_global_figures, bounds)


########################################################################################################################

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