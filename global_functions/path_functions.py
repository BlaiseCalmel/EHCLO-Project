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
import os

def define_paths(config):

    param_type = ['hydro', 'climate']
    folder_data_contour = os.path.abspath(config['folder_path_contour_data']) + os.sep

    # Main folders path
    # folder_results = config['folder_path_results'] + os.sep
    folder_results = os.path.abspath(config['folder_path_results'])  + os.sep
    folder_study = folder_results + config['study_name'] + os.sep
    folder_study_figures = folder_study + 'figures' + os.sep
    folder_study_data = folder_study + 'data' + os.sep

    # File paths
    # points_sim = config['param_type'] + '_points_sim.csv'
    dict_global_points_sim = {}
    dict_study_points_sim = {}
    for data_type in param_type:
        dict_global_points_sim[data_type] = folder_data_contour + data_type + '_points_sim.shp'
        dict_study_points_sim[data_type] = folder_study_data + 'shapefiles' + os.sep + data_type + '_points_sim.shp'


    dict_paths = {
        'folder_hydro_data': os.path.abspath(config['folder_path_hydro_data']),
        'folder_climate_data': os.path.abspath(config['folder_path_climate_data']),
        'folder_raw_hydro_data': os.path.abspath(config['folder_path_raw_hydro_data']),
        'folder_data_contour': folder_data_contour,
        'folder_study_results': folder_study,
        'folder_study_figures': folder_study_figures,
        'folder_study_data': folder_study_data,
        'file_regions_shp': folder_data_contour + config['regions_shp'],
        'file_rivers_shp': folder_data_contour + config['rivers_shp'],
        'file_hydro_shp': folder_data_contour + config['hydro_shp'],
        'file_climate_shp': folder_data_contour + config['climate_shp'],
        'dict_global_points_sim': dict_global_points_sim,
        'dict_study_points_sim': dict_study_points_sim,
    }

    return dict_paths

def get_files_path(dict_paths, setup, extension='.nc'):

    ext_files = []
    for p in [dict_paths[f'folder_hydro_data']]+[dict_paths[f'folder_climate_data']]+[dict_paths[f'folder_raw_hydro_data']]:
        for root, dirs, files in os.walk(p):
            for file in files:
                if file.endswith(extension):
                    ext_files.append(os.path.join(root, file))

    dict_path = {}
    for data_type in ['hydro', 'climate']:
        dict_path[data_type] = {}
        item_indicator = [i.split('$')[0] for i in setup[f'{data_type}_indicator'].keys()]

        # Filter by indicator name
        data_files = [s for s in ext_files if any(word in s for word in item_indicator)]

        # Filter by sim chain
        keys = ['select_gcm', 'select_rcm', 'select_bc']
        for key in keys:
            if len(setup[key]) > 0:
               data_files = [s for s in data_files if any(word in s for word in setup[key])]

        # Run on selected rcp
        if len(setup['select_rcp']) == 0:
            setup['select_rcp'] = ["rcp26", "rcp45", "rcp85"]
            # setup['select_rcp'] = ["SAFRAN"]

        for rcp in setup['select_rcp']:
            dict_path[data_type][rcp] = {}

            if data_type == 'climate':
                rcp_files = [s for s in data_files if any(word in s for word in [rcp] + ['historical'])]
            else:
                rcp_files = [s for s in data_files if any(word in s for word in [rcp])]

            for (item, indic) in zip(item_indicator, setup[f'{data_type}_indicator'].keys()):
                indic_values = [s for s in rcp_files if item in s.split(os.sep)[-1]]
                indic_values = [i for i in indic_values
                                if '_'.join(i.split(os.sep)[-1].split('_')[:item.count('_')+1]) == item]                # Filter by HM
                if data_type == 'hydro':
                    if len(setup['select_hm']) > 0:
                        indic_values = [s for s in indic_values if any(word in s for word in setup['select_hm'])]

                # Save in dict
                dict_path[data_type][rcp][indic] = indic_values

    return dict_path