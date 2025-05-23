"""
    EHCLO Project
    Copyright (C) 2025  Blaise CALMEL (INRAE)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import pyfiglet
ascii_banner = pyfiglet.figlet_format("EHCLO PROJECT")
print(f'##########################################################################################', end='\n')
print(ascii_banner, end='\n')

print(f'################################ IMPORT & INITIALIZATION ################################', end='\n')

print(f'> General imports...', end='\n')
import sys
import os
import copy
# sys.path.insert(0, os.getcwd())
import time
import json

print(f'> Local imports...', end='\n')
from global_functions.load_data import *
from plot_functions.run_plot import *
from global_functions.format_data import *
from global_functions.shp_geometry import *
from global_functions.path_functions import  *
from global_functions.compute_narratives import compute_narratives

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

settings_flatten = {}
for main_key in ['hydro_indicator', 'climate_indicator']:
    for key, value in files_setup[main_key].items():
        for subkey, subvalue in value.items():
            subvalue |= {'parent': key, 'type': main_key}
            settings_flatten |= {subkey: subvalue}

print(f'> Define paths...', end='\n')
path_data = r"/media/bcalmel/One Touch/2_Travail/3_INRAE_EHCLO/20_data"
folder_path_results = r"/media/bcalmel/One Touch/2_Travail/3_INRAE_EHCLO"
# path_data = r"D:\2_Travail\3_INRAE_EHCLO\20_data"
# folder_path_results = r"D:\2_Travail\3_INRAE_EHCLO"
study_name = f"HMUC_Loire_Bretagne"
dict_paths = define_paths(config, path_data, folder_path_results, study_name)

# Get paths for selected sim
print(f'> Load data paths...', end='\n')
path_files = get_files_path(dict_paths=dict_paths, setup=files_setup)

### Files names
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
if not os.path.isdir(dict_paths['folder_study_data_narratives']):
    os.makedirs(dict_paths['folder_study_data_narratives'])
if not os.path.isdir(dict_paths['folder_study_data'] + 'shapefiles'):
    os.makedirs(dict_paths['folder_study_data'] + 'shapefiles')
if not os.path.isdir(dict_paths['folder_study_data_ncdf']):
    os.makedirs(dict_paths['folder_study_data_ncdf'])
if not os.path.isdir(dict_paths['folder_study_data_formated-ncdf']):
    os.makedirs(dict_paths['folder_study_data_formated-ncdf'])

#%% LOAD STUDY REGION SHAPEFILE
print(f'################################ DEFINE STUDY AREA ################################', end='\n')
print(f'> Load shapefiles...', end='\n')
regions_shp, study_hydro_shp, study_climate_shp, rivers_shp = load_shp(dict_paths, files_setup)

# Load stations of interest
print(f'> Searching sim points in study area...', end='\n')
with open('reference_stations.json') as ref_stations:
    reference_stations = json.load(ref_stations)

flatten_reference_stations = {key: value for subdict in reference_stations.values() for key, value in subdict.items()}

# Select region area
path_files = get_files_path(dict_paths=dict_paths, setup=files_setup)

# Automatic run among each region
auto_run = False
if auto_run:
    regions = regions_shp['CdRegionHy'].values
else:
    regions = [None]

for region_id in regions:
    load_shapefiles = True
    while load_shapefiles:
        # Ask region
        if not auto_run:
            region_id = input("Select code region [A-Y]:")
        region_input_list = re.split(r"[ ]", region_id)
        force_contains= None
        # If empty, use Loire UGs
        if len(region_id) == 0:
            region_input_list = 'UG-Loire'
            
            force_contains = {'Suggesti_2': ['LA LOIRE', 'L\'ALLIER'],
                              'Suggestion': flatten_reference_stations.keys()}
            shapefile_hydro = study_hydro_shp
            shapefile_climate = study_climate_shp

            region_name = region_input_list
            region_narrative = 'K'

        else:
            selected_codes = []
            for code in region_input_list:
                if code in regions_shp['CdRegionHy'].values:
                    selected_codes.append(code)
            print(f">> Load {' '.join(selected_codes)} region(s)")
            shapefile_hydro = regions_shp[regions_shp['CdRegionHy'].isin(selected_codes)]
            shapefile_climate = shapefile_hydro

            region_name = '-'.join(selected_codes)
            region_narrative = region_name

        region_name_shp = region_name + '.shp'

        dict_paths['dict_study_points_sim']['hydro_narrative'] = dict_paths['dict_study_points_sim_base']['hydro'] + '_' + region_narrative + '.shp'

        if len(shapefile_hydro) > 0:
            load_shapefiles = False
        else:
            print(f">>> Invalid region selected ({region_id}), please select again")

    for data_type, path in dict_paths['dict_study_points_sim_base'].items():
        path += '_' + region_name_shp
        dict_paths['dict_study_points_sim'][data_type] = path
        if not os.path.isfile(path):
            print(f'>> Find {data_type} data points in study area')
            # sim_all_points_info = open_shp(f"/home/bcalmel/Documents/2_data/contours_all/points_sim_hydro/points_debit_simulation_Explore2.shp")
            sim_all_points_info = open_shp(path_shp=dict_paths['dict_global_points_sim'][data_type])
            if data_type == 'hydro':
                overlay_shapefile(shapefile=shapefile_hydro, data=sim_all_points_info, path_result=path,
                                  force_contains=force_contains)
            else:
                overlay_shapefile(shapefile=shapefile_climate, data=sim_all_points_info, path_result=path)
        else:
            print(f'>> {data_type.capitalize()} data points already in the study area')

    print(f'> Simplify shapefiles...', end='\n')
    study_hydro_shp_simplified, study_climate_shp_simplified, study_rivers_shp_simplified, regions_shp_simplified, bounds = (
        simplify_shapefiles(study_hydro_shp, study_climate_shp, rivers_shp, regions_shp, tolerance=1000, zoom=1000))

    hydro_sim_points_gdf_simplified = open_shp(path_shp=dict_paths['dict_study_points_sim']['hydro'])
    # hydro_sim_points_gdf_simplified = open_shp(path_shp='/home/bcalmel/Documents/3_results/HMUC_Loire_Bretagne/data/shapefiles/hydro_points_sim_all-BV.shp')
    hydro_sim_points_gdf_simplified = hydro_sim_points_gdf_simplified[hydro_sim_points_gdf_simplified['n'] >= 4]
    hydro_sim_points_gdf_simplified = hydro_sim_points_gdf_simplified.reset_index(drop=True).set_index('Suggestion')
    hydro_sim_points_gdf_simplified.index.names = ['name']
    hydro_sim_points_gdf_simplified = hydro_sim_points_gdf_simplified[~hydro_sim_points_gdf_simplified.index.duplicated(keep='first')]

    climate_sim_points_gdf = open_shp(path_shp=dict_paths['dict_study_points_sim']['climate'])
    climate_sim_points_gdf_simplified = climate_sim_points_gdf.loc[
        climate_sim_points_gdf.groupby('name')['gid'].idxmin()].reset_index(drop=True)

    all_years = files_setup["historical"] + [year for key, values in files_setup["horizons"].items() for
                                                year in values if key != "tracc" ]
    start_year = min(all_years)
    tracc_year = None
    if 'tracc' in files_setup.keys() and files_setup['tracc']:
        if not 'tracc' in files_setup['horizons']:
            files_setup['horizons']['tracc'] = [1.4, 2.1, 3.4]
        print(f'> Load TRACC...', end='\n')
        tracc_year = pd.read_csv(dict_paths['file_tracc'], sep=";")
        tracc_year = tracc_year[tracc_year['Niveau_de_rechauffement'].isin(files_setup["horizons"]['tracc'])]
        tracc_year = tracc_year.set_index('Niveau_de_rechauffement')
        end_year = 2100
        extended_name = f"TRACC"
    else:
        end_year = max(all_years)
        extended_name = f"{start_year}-{end_year}"

    #%% NCDF Loading
    if not auto_run:
        load_ncdf = input("Load new NCDF ? (y/[n])")
    else:
        load_ncdf = 'y'

    # selected_points_narratives = open_shp(dict_paths['folder_data_contour']+ os.sep + 'hydro_points_sim_noeud_gestion.shp')
    # selected_points_narratives = selected_points_narratives.reset_index(drop=True).set_index('Suggestion')
    rcp = 'rcp85'
    if load_ncdf.lower().replace(" ", "") in ['y', 'yes']:
        print(f'################################ RUN OVER NCDF ################################', end='\n')
        # Define indicator to load
        if not auto_run:
            data_input = input('What should I run ?')
        else:
            data_input = 'QA QJXA VCN10'

        data_input_list = re.split(r"[ ]", data_input)
        if len(data_input) == 0:
            path_dict = path_files
        else:
            path_dict = {'hydro': {rcp: {}}, 'climate': {rcp: {}}}
            if data_input == 'climate':
                indicator_type =  'climate_indicator'
            elif data_inpu == 'hydro':
                indicator_type =  'hydro_indicator'
            else: 
                indicator_type = None
            for key, value in settings_flatten.items():
                if value['parent'] in data_input_list or key in data_input_list or indicator_type == value['type']:
                    data_type = value['type'].split('_')[0]
                    path_dict[data_type][rcp][value['parent']] = path_files[data_type][rcp][value['parent']]

        for data_type, subdict in path_dict.items():
            # Load simulation points for current data type
            # sim_points_gdf = open_shp(path_shp=dict_paths['dict_study_points_sim'][data_type])

            if data_type == "hydro":
                sim_points_gdf_simplified = hydro_sim_points_gdf_simplified
                # sim_points_gdf_simplified = selected_points_narratives
            else:
                sim_points_gdf_simplified = climate_sim_points_gdf_simplified
                # sim_points_gdf['weight'] = sim_points_gdf['surface'] / sim_points_gdf['total_surf']

            for rcp, subdict2 in subdict.items():
                for indicator, paths in subdict2.items():
                    print(f'################################ RUN {data_type} {rcp} {indicator} ################################', end='\n')
                    for name_indicator, settings_dict in files_setup[f'{data_type}_indicator'][indicator].items():
                        timestep = 'YE'
                        function = None
                        if 'timestep' in settings_dict:
                            timestep = settings_dict['timestep']

                        if 'extract_function' in settings_dict:
                            function = settings_dict['extract_function']

                        name_join = name_indicator.replace(" ", "-").replace(".", "")
                        name_join = name_join.replace('>', 'sup').replace('<', 'inf')

                        path_ncdf = f"{dict_paths['folder_study_data_ncdf']}{name_join}_{rcp}_{timestep}_{extended_name}_{region_name}.nc"
                        # path_ncdf = f"{dict_paths['folder_study_data']}{name_join}_{rcp}_{timestep}_narratest.nc"

                        print(f'> Create {indicator} export...', end='\n')
                        if len(paths) > 0 and not os.path.isfile(path_ncdf):
                            # paths = [
                            #     '/home/bcalmel/Documents/2_data/historical-rcp85/HadGEM2-ES/ALADIN63/ADAMONT/SMASH/debit_France_MOHC-HadGEM2-ES_historical-rcp85_r1i1p1_CNRM-ALADIN63_v3_MF-ADAMONT-SAFRAN-1980-2011_INRAE-SMASH_day_20050801-20990731.nc'
                            # ]
                            # paths_data=paths
                            # param_type=data_type
                            # sim_points_gdf=sim_points_gdf_simplified
                            #
                            # path_result=path_ncdf
                            # path_ncdf = f"{dict_paths['folder_study_data']}{name_join}_{rcp}_{timestep}_{start}-{end}.csv"
                            extract_ncdf_indicator(
                                paths_data=paths, param_type=data_type, sim_points_gdf=sim_points_gdf_simplified,
                                indicator=indicator, function=function, files_setup=files_setup, timestep=timestep,
                                start=start_year,
                                end=end_year,
                                tracc_year=tracc_year,
                                path_result=path_ncdf,
                            )
                        else:
                            print(f'> Invalid {indicator} name', end='\n')


    #%% Visualize results
    narratives = None
    quantiles = [0.5]
    str_quantiles = 'quant'+('-').join([f"{int(i*100)}" for i in  quantiles])
    horizon_ref='horizon2'
    if region_input_list == 'UG-Loire':
        path_narratives = f"{dict_paths['folder_study_data_narratives']}narratives_{extended_name}_{horizon_ref}_{str_quantiles}_K.json"
    else:
        path_narratives = f"{dict_paths['folder_study_data_narratives']}narratives_{extended_name}_{horizon_ref}_{str_quantiles}_{region_name}.json"

    if not auto_run:
        load_narratives = input("Compute new narrative ? (y/[n])")
    else:
        load_narratives = 'y'
    if load_narratives.lower().replace(" ", "") in ['y', 'yes']:

        # horizons_narrative = ['horizon1','horizon2', 'horizon3']
        print('> Define Narratives')
        hydro_narrative_gdf_simplified = open_shp(path_shp=dict_paths['dict_study_points_sim']['hydro_narrative'])
        hydro_narrative_gdf_simplified = hydro_narrative_gdf_simplified[hydro_narrative_gdf_simplified['n'] >= 4]
        hydro_narrative_gdf_simplified = hydro_narrative_gdf_simplified.reset_index(drop=True).set_index('Suggestion')
        hydro_narrative_gdf_simplified.index.names = ['name']

        indicator_values = ["QJXA", "QA", "VCN10"]
        paths_ds_narratives = [f"{dict_paths['folder_study_data_ncdf']}{indicator}_{rcp}_YE_{extended_name}_{region_narrative}.nc"
                               for indicator in indicator_values]

        if not os.path.isdir(dict_paths['folder_study_figures_narratives']):
            os.makedirs(dict_paths['folder_study_figures_narratives'])

        compute_narratives( paths_ds_narratives,
                            files_setup=files_setup,
                            indicator_values=indicator_values,
                            path_narratives=path_narratives,
                            path_figures=dict_paths['folder_study_figures_narratives']+region_narrative+'_',
                            path_performances=dict_paths['folder_hydro_performances'],
                            path_formated_ncdf=f"{dict_paths['folder_study_data_formated-ncdf']}",
                            horizon_ref=horizon_ref,
                            quantiles=quantiles
                            )

    print('> Load Narratives')
    with open(path_narratives, "r", encoding="utf-8") as f:
        narratives = json.load(f)

# narratives = {'horizon2' : narratives['horizon2']}

# narratives =  compute_narratives(dict_paths,
#                                  stations=list(reference_stations['La Loire'].keys()),
#                                  files_setup=files_setup,
#                                  data_shp=hydro_shp_bv,
#                                  indictor_values=["QJXA", "QA", "VCN10"],
#                                  threshold=0.7,
#                                  narrative_method='combine')

run_plot = True
while run_plot:
    run_all = input("Run all ? (y/hydro/climate/[n])")
    data_to_plot = {}
    if run_all.lower().replace(" ", "") in ['y', 'yes']:
        for key, value in settings_flatten.items():
            data_to_plot |= {key: value}
    elif run_all.lower().replace(" ", "") == 'hydro':
        for key, value in settings_flatten.items():
            if value['type'] == 'hydro_indicator':
                data_to_plot |= {key: value}
    elif run_all.lower().replace(" ", "") == 'climate':
        for key, value in settings_flatten.items():
            if value['type'] == 'climate_indicator':
                data_to_plot |= {key: value}
    else:
        data_input = input('What should I run ?')
        data_input_list = re.split(r"[ ]", data_input)
        for name in data_input_list:
            name = name.replace(",", "").replace(" ", "")
            if len(name) > 0:
                if name in files_setup['climate_indicator'].keys():
                    for key, value in files_setup['climate_indicator'][name].items():
                        data_to_plot |= {key: value}
                elif name in files_setup['hydro_indicator'].keys():
                    for key, value in files_setup['hydro_indicator'][name].items():
                        data_to_plot |= {key: value}
                elif name in settings_flatten.keys():
                    if settings_flatten[name]['parent'] not in data_input_list:
                        data_to_plot |= {name: settings_flatten[name]}
                else:
                    keep_going = input(f"Invalid data name '{name}', ignore and keep running ? (y/n)")
                    if keep_going.lower().replace(" ", "") in ['n', 'no', 'non']:
                        data_to_plot = {}
                        break

    # name = 'QA_yr'
    # data_to_plot = {name: files_setup['hydro_indicator'][name]}
    # data_to_plot = (files_setup['climate_indicator'] | files_setup['hydro_indicator'])
    if len(data_to_plot) > 0:
        overwrite = True
        rcp = 'rcp85'
        runned_data = []
        if narratives is None:
            narratives = {"Explore2": {
                "HadGEM2-ES_ALADIN63_ADAMONT": {'color': '#569A71', 'zorder': 10,
                                                'label': 'Vert [HadGEM2-ES_ALADIN63_ADAMONT]',
                                                'linewidth': 2},
                "CNRM-CM5_ALADIN63_ADAMONT": {'color': '#EECC66', 'zorder': 10,
                                              'label': 'Jaune [CNRM-CM5_ALADIN63_ADAMONT]',
                                              'linewidth': 2},
                "EC-EARTH_HadREM3-GA7_ADAMONT": {'color': '#E09B2F', 'zorder': 10,
                                                 'label': 'Orange [EC-EARTH_HadREM3-GA7_ADAMONT]',
                                                 'linewidth': 2},
                "HadGEM2-ES_CCLM4-8-17_ADAMONT": {'color': '#791F5D', 'zorder': 10,
                                                  'label': 'Violet [HadGEM2-ES_CCLM4-8-17_ADAMONT]',
                                                  'linewidth': 2},
            }}

        for name_indicator, indicator_setup in data_to_plot.items():
            # if name_indicator != 'T moy.':
            #     continue
            if name_indicator in runned_data:
                continue
            print(f'################################ STATS {name_indicator.upper()} ################################', end='\n')
            # if name_indicator.upper() == "T MOY.":
            #     break
            if overwrite:
                write_fig = True
            else:
                write_fig = False

            # Get plot settings
            settings = load_settings(indicator_setup, name_indicator)

            # Define horizons
            if tracc_year is not None:
                horizons_tracc = {'horizon1': 'Horizon +2.0°C | 2030',
                            'horizon2': 'Horizon +2.7°C | 2050',
                            'horizon3': 'Horizon +4.0°C | 2100',
                }
                # Get selected tracc horizon
                if indicator_setup['type']  == 'hydro_indicator':
                    horizons = {key: value for key, value in horizons_tracc.items() if key == horizon_ref}
                else:
                    horizons = horizons_tracc
            else:
                horizons = {'horizon1': 'Horizon 1 (2021-2050)',
                            'horizon2': 'Horizon 2 (2041-2070)',
                            'horizon3': 'Horizon 3 (2070-2099)',
                }

            # Create folder
            title_join = name_indicator.replace(" ", "-").replace(".", "").replace('>', 'sup').replace('<', 'inf')
            path_indicator = (dict_paths['folder_study_figures'] + title_join + os.sep)
            if not os.path.isdir(path_indicator):
                os.makedirs(path_indicator)
                write_fig = True
            
            # Add to settings info
            settings["name_indicator"] = name_indicator
            settings["title_join"] = title_join

            if write_fig:
                # Compute PK
                if indicator_setup['type']  == 'hydro_indicator':
                    data_type = 'hydro'
                    sim_points_gdf_simplified = hydro_sim_points_gdf_simplified = hydro_sim_points_gdf_simplified[~hydro_sim_points_gdf_simplified.index.duplicated(keep='first')]
                    
                    # loire = sim_points_gdf_simplified.loc[sim_points_gdf_simplified['gid'] < 7]
                    loire = sim_points_gdf_simplified[(sim_points_gdf_simplified['Suggesti_2'].str.contains('LA LOIRE ', case=False, na=False))]
                    value = compute_river_distance(rivers_shp, loire, river_name='loire',
                                                   start_from='last')

                    hydro_sim_points_gdf_simplified["PK"] = value                    
                    # sim_points_gdf_simplified.loc[sim_points_gdf_simplified['gid'] < 7, 'PK'] = value
                    settings["edgecolor"] = 'k'
                else:
                    data_type = 'climate'
                    sim_points_gdf_simplified = climate_sim_points_gdf_simplified
                    sim_points_gdf_simplified = sim_points_gdf_simplified.set_index('name')
                    settings["edgecolor"] = None

                # TODO load formated-ncdf                
                # Open ncdf dataset
                path_ncdf = f"{dict_paths['folder_study_data_ncdf']}{settings['title_join']}_{rcp}_{settings['timestep']}_{extended_name}_{region_name}.nc".replace('>', 'sup').replace('<', 'inf')
                path_formated_ncdf = f"{dict_paths['folder_study_data_formated-ncdf']}formated-{path_ncdf.split(os.sep)[-1]}".replace('>', 'sup').replace('<', 'inf')
                path_variables = f"{dict_paths['folder_study_data_formated-ncdf']}{indicator_setup['type']}_variables.json"
                if os.path.isfile(path_formated_ncdf) and os.path.isfile(path_variables):
                    # Get saved dataset and var names
                    ds_stats = xr.open_dataset(path_formated_ncdf)
                    with open(path_variables, "r", encoding="utf-8") as f:
                        variables = json.load(f)
                else:
                    # Compute variables
                    ds_stats = xr.open_dataset(path_ncdf)
                    # Format code hydro
                    if data_type == 'hydro':
                        ds_stats['gid'] = ds_stats['gid'].astype(str)
                    gid_values = np.unique([code for code in sim_points_gdf_simplified.index.values])
                    codes_to_select = [code for code in gid_values if code in ds_stats['gid'].values]
                    if len(codes_to_select) > 0:
                        ds_stats = ds_stats.sel(gid=codes_to_select)
                    # Compute stats
                    ds_stats, variables = format_dataset(ds=ds_stats, data_type=data_type, files_setup=files_setup,
                                                        path_result=path_formated_ncdf, path_variables=path_variables,
                                                        plot_function=settings['additional_coordinates'],
                                                        return_period=settings['return_period'])
                # ds_stats.sel(gid=ds_stats["gid"] == b'----------')
                # ds_stats = ds_stats.sel(gid=ds_stats["gid"] != b'----------')

                sim_points_gdf_simplified = sim_points_gdf_simplified.loc[ds_stats.gid]
                dict_shapefiles = define_plot_shapefiles(regions_shp_simplified, study_climate_shp_simplified, study_rivers_shp_simplified,
                                       indicator_setup['type'], files_setup)
                settings["dict_shapefiles"] = dict_shapefiles
                settings["bounds"] = bounds

                # Check for additional coordinates
                used_coords = set()
                for var in ds_stats.data_vars:
                    used_coords.update(ds_stats[var].dims)

                # Use another dimension to create different plots (ex: one plot per month)
                additional_plot_folders = {'': [None]}
                if settings['additional_plot_folders'] is not None:
                    additional_plot_folders = {settings['additional_plot_folders']: ds_stats[settings['additional_plot_folders']].values}

                if settings['additional_coordinates'] != 'month':
                    for coordinate, unique_value in additional_plot_folders.items():
                        for coordinate_value in unique_value:
                            print(f'################################ PLOT {name_indicator.upper()} {coordinate_value if coordinate_value is not None else ""} '
                                  f'################################', end='\n')
                            # Selection from the current coordinate value
                            if coordinate_value is not None:
                                ds = ds_stats.sel({coordinate: coordinate_value})
                                path_indicator_figures = path_indicator + coordinate_value + os.sep
                                if not os.path.isdir(path_indicator_figures):
                                    os.makedirs(path_indicator_figures)
                            else:
                                ds = copy.deepcopy(ds_stats)
                                path_indicator_figures = path_indicator
                            
                            print(f">> Map plot...")
                            print(f"{name_indicator} >>> {settings['plot_type_name'].title()} matching map plot")
                            plot_map_indicator(gdf=sim_points_gdf_simplified, ds=ds, indicator_plot='horizon_matching',
                                               path_result=path_indicator_figures+f"{settings['title_join']}_map_matching_sims.pdf", horizons=horizons,
                                               cbar_title=f"{settings['title_join']} Accord des modèles sur le sens d'évolution (%)", cbar_ticks=None,
                                               title=coordinate_value, dict_shapefiles=settings['dict_shapefiles'],
                                               bounds=settings['bounds'], palette='PuOr', cbar_midpoint='zero', cbar_values=0,
                                               start_cbar_ticks=settings['start_cbar_ticks'], end_cbar_ticks=settings['end_cbar_ticks'],
                                               fontsize=settings['fontsize']-2, alpha=1,
                                               font=settings['font'], discretize=settings['discretize'], edgecolor=settings['edgecolor'], markersize=75,
                                               vmin=-100, vmax=100)
                            
                            if indicator_setup['type'] == 'climate_indicator':                                  
                                plot_climate(ds, sim_points_gdf_simplified, horizons, narratives, settings, coordinate_value, 
                                             path_indicator_figures)
                                
                            elif indicator_setup['type'] == 'hydro_indicator':
                                if len(list(list(narratives.values())[0].keys())[0].split("_")) == 3:
                                    # Explore2 climate narratives
                                    plot_hydro_narraclimate(ds, variables, sim_points_gdf_simplified, horizons, narratives, 
                                                            settings, coordinate_value, path_indicator_figures, reference_stations)
                                else:
                                    # Hydro narratives
                                    plot_hydro_narrahydro(ds, variables, sim_points_gdf_simplified, horizons, narratives, 
                                                          settings, coordinate_value, path_indicator_figures, reference_stations)
                                
                elif settings['additional_coordinates'] == 'month':
                    print(f'################################ PLOT {name_indicator.upper()} Monthly variation ################################', end='\n')
                    plot_monthly(ds_stats, variables, sim_points_gdf_simplified, horizons, narratives, 
                                 settings, path_indicator, reference_stations)
                    
            # Save name indicator
            runned_data.append(name_indicator)

    keep_plotting = input("Plot again ? ([y]/n)")
    if keep_plotting.lower().replace(" ", "") in ['n', 'no']:
        run_plot = False

# Plot map of station for narratives computation
dict_shapefiles = define_plot_shapefiles(regions_shp_simplified, study_climate_shp_simplified, study_rivers_shp_simplified,
                                         "hydro_indicator", files_setup)
mapplot(gdf=selected_points_narratives, indicator_plot='n', path_result='/home/bcalmel/Documents/3_results/stations_narrative_all-BV.pdf', ds=None,
            cols=None, rows=None,  cbar_ticks='mid', dict_shapefiles=dict_shapefiles, 
            cbar_title=f"Nombre\nde HM", title=None, palette='RdBu_r', font='sans-serif', edgecolor='k',
            vmin_user=1, vmax_user=9,  discretize=18, markersize=90)

# print(f'################################ PLOT GLOBAL ################################', end='\n')
# path_global_figures = dict_paths['folder_study_figures'] + 'global' + os.sep
# if not os.path.isdir(path_global_figures):
#     os.makedirs(path_global_figures)
#
# print(f"> Plot HM by station...")
# plot_map_HM_by_station(hydro_sim_points_gdf_simplified, dict_shapefiles, bounds, path_global_figures,
#                        fontsize=settings['fontsize']+2)
#
# print(f"> Plot #HM by station and Ref station...")
# plot_map_N_HM_ref_station(hydro_sim_points_gdf_simplified, dict_shapefiles, path_global_figures, bounds,
#                           station_references=flatten_reference_stations, fontsize=settings['fontsize'])

print(f'################################ END ################################', end='\n')
input("Press Enter to close")

# %%
