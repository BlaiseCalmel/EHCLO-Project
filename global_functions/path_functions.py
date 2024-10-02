import os

def define_paths(config):

    parent_folder = config['parent_folder'] + os.sep
    # folder_data = cwd + config['folder_data'] + os.sep
    # list_folder_data_data = []
    # for data_type in config['param_type']:
    #     list_folder_data_data.append(folder_data + data_type + os.sep)
    # folder_data_data = folder_data + config['param_type'] + os.sep
    folder_data_contour = parent_folder + config['folder_data'] + os.sep

    # Main folders path
    folder_results = parent_folder + config['folder_results'] + os.sep
    folder_study = folder_results + config['study_name'] + os.sep
    folder_study_figures = folder_study + config['folder_figures'] + os.sep
    folder_study_data = folder_study + config['folder_data'] + os.sep

    # File paths
    # points_sim = config['param_type'] + '_points_sim.csv'
    list_global_points_sim = []
    list_study_points_sim = []
    for data_type in config['param_type']:
        list_global_points_sim.append(folder_data_contour + data_type + '_points_sim.csv')
        list_study_points_sim.append(folder_study_data + data_type + '_points_sim.csv')

    dict_paths = {
        'folder_hydro_data': config['folder_hydro_data'],
        'folder_climate_data': config['folder_climate_data'],
        'folder_data_contour': folder_data_contour,
        'folder_study_results': folder_study,
        'folder_study_figures': folder_study_figures,
        'folder_study_data': folder_study_data,
        'file_regions_shp': folder_data_contour + config['regions_shp'],
        'file_rivers_shp': folder_data_contour + config['rivers_shp'],
        'list_global_points_sim': list_global_points_sim,
        'list_study_points_sim': list_study_points_sim,
    }

    return dict_paths

def get_files_path(dict_paths, setup, extension='.nc'):
    ext_files = []
    for p in [dict_paths[f'folder_hydro_data']]+[dict_paths[f'folder_climate_data']]:
        for root, dirs, files in os.walk(p):
            for file in files:
                if file.endswith(extension):
                    ext_files.append(os.path.join(root, file))

    setup_indicator = setup['climate_indicator'] + setup['hydro_indicator']

    for key, value in setup.items():
        if 'indicator' not in key:
            if len(value) > 0:
                ext_files = [s for s in ext_files if any(word in s for word in value)]

    dict_path = {}
    for indic in setup_indicator:
        dict_path[indic] = [s for s in ext_files if indic in s]

    return dict_path