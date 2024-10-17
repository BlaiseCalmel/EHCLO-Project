import copy
import os

def define_paths(config):

    param_type = ['hydro', 'climate']
    folder_data_contour = config['folder_contour_data'] + os.sep

    # Main folders path
    folder_results = config['folder_results'] + os.sep
    folder_study = folder_results + config['study_name'] + os.sep
    folder_study_figures = folder_study + config['folder_figures'] + os.sep
    folder_study_data = folder_study + config['folder_data'] + os.sep

    # File paths
    # points_sim = config['param_type'] + '_points_sim.csv'
    dict_global_points_sim = {}
    dict_study_points_sim = {}
    for data_type in param_type:
        dict_global_points_sim[data_type] = folder_data_contour + data_type + '_points_sim.shp'
        dict_study_points_sim[data_type] = folder_study_data + data_type + '_points_sim.shp'


    dict_paths = {
        'folder_hydro_data': config['folder_hydro_data'],
        'folder_climate_data': config['folder_climate_data'],
        'folder_data_contour': folder_data_contour,
        'folder_study_results': folder_study,
        'folder_study_figures': folder_study_figures,
        'folder_study_data': folder_study_data,
        'file_regions_shp': folder_data_contour + config['regions_shp'],
        'file_rivers_shp': folder_data_contour + config['rivers_shp'],
        'dict_global_points_sim': dict_global_points_sim,
        'dict_study_points_sim': dict_study_points_sim,
    }

    return dict_paths

def get_files_path(dict_paths, setup, extension='.nc'):

    ext_files = []
    for p in [dict_paths[f'folder_hydro_data']]+[dict_paths[f'folder_climate_data']]:
        for root, dirs, files in os.walk(p):
            for file in files:
                if file.endswith(extension):
                    ext_files.append(os.path.join(root, file))

    dict_path = {}
    for data_type in ['hydro', 'climate']:
        dict_path[data_type] = {}
        setup_indicator = setup[f'{data_type}_indicator']
        indicators = [i.split('-')[0] for i in setup_indicator]

        # Filter by indicator name
        data_files = [s for s in ext_files if any(word in s for word in indicators)]

        # Filter by sim chain
        keys = ['select_gcm', 'select_rcm', 'select_bc']
        for key in keys:
            if len(setup[key]) > 0:
               data_files = [s for s in data_files if any(word in s for word in setup[key])]

        # Run on selected rcp
        for rcp in setup['select_rcp']:
            dict_path[data_type][rcp] = {}
            if data_type == 'climate':
                rcp_files = [s for s in data_files if any(word in s for word in [rcp] + ['historical'])]
            else:
                rcp_files = [s for s in data_files if any(word in s for word in [rcp])]

            for indic in setup_indicator:
                indicator = indic.split('-')[0]
                indic_values = [s for s in rcp_files if indicator in s]
                # Filter by HM
                if indic in setup['hydro_indicator']:
                    indic_values = [s for s in indic_values if any(word in s for word in setup['select_hm'])]

                # Save in dict
                dict_path[data_type][rcp][indic] = indic_values

    return dict_path