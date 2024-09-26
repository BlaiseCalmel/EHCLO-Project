import os

def define_paths(cwd, config):
    folder_data = cwd + config['folder_data'] + os.sep
    folder_data_data = folder_data + config['param_type'] + os.sep
    folder_data_contour = folder_data + config['folder_contours'] + os.sep

    # Main folders path
    folder_results = cwd + config['folder_results'] + os.sep
    folder_study = folder_results + config['study_name'] + os.sep
    folder_study_figures = folder_study + config['folder_figures'] + os.sep
    folder_study_contour = folder_study + config['folder_contours'] + os.sep

    # File paths
    points_sim = config['param_type'] + '_points_sim.csv'

    dict_paths = {
        'folder_data_data': folder_data,
        'folder_data_contour': folder_data_contour,
        'folder_study_results': folder_study,
        'folder_study_figures': folder_study_figures,
        'folder_study_contour': folder_study_contour,

        'file_regions_shp': folder_data_contour + config['regions_shp'],
        'file_rivers_shp': folder_data_contour + config['rivers_shp'],
        'file_data_points_sim': folder_data_contour + points_sim,
        'file_study_points_sim': folder_study_contour + points_sim,

    }

    return dict_paths