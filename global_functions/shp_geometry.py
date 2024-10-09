import geopandas
import pandas as pd
from shapely.geometry import Point, Polygon, LineString
import netCDF4
import numpy as np
import xarray as xr

def is_data_in_shape(shapefile, data, cols, path_result=None):

    x = data[cols[0]]
    y = data[cols[1]]
    geo_df = geopandas.GeoDataFrame(
        data, geometry=geopandas.points_from_xy(x, y),
        crs={'init': 'epsg:2154'}

    )
    # # Polygons to L93
    # shapefile_l93 = shapefile.to_crs(crs={'init': 'epsg:2154'})

    # Join both
    matched_points = geo_df.sjoin(shapefile, how='inner', predicate='intersects')

    matched_points = matched_points.drop('index_right', axis=1)

    # Save the matched points shapefile
    if path_result is not None:
        matched_points.to_csv(path_result, index=False)
    else:
        return matched_points

def get_coordinates(data, path_result):
    # Station points LII
    cols = ['lon', 'lat']
    col1 = data[cols[0]][:].data
    col2 = data[cols[1]][:].data
    col1_flat = col1.flatten()
    col2_flat = col2.flatten()
    # Columns and rows index
    x_idx = np.tile(np.arange(col1.shape[1]), (col1.shape[0], 1)).flatten()
    y_idx = np.tile(np.arange(col1.shape[0]), (col1.shape[1], 1)).T.flatten()
    # Coordinates
    x = np.tile(data['x'][:].data.flatten(), (col1.shape[0], 1)).flatten()
    y = np.tile(data['y'][:].data.flatten(), (col1.shape[1], 1)).T.flatten()

    names = [str(i)+'_'+str(j) for i, j in zip(x_idx, y_idx)]
    data = pd.DataFrame({'name': names, 'x_idx': x_idx, 'y_idx': y_idx, 'lat': col2_flat, 'lon': col1_flat,
                         'x': x, 'y': y})

    cell_size = 8000
    # Liste pour stocker les polylignes
    polylines = []
    # Créer des polylignes pour chaque maille
    for index, row in data.iterrows():
        x_center = row['lon']
        y_center = row['lat']

        # Calculer les coins de la maille
        x_min = x_center - cell_size / 2
        x_max = x_center + cell_size / 2
        y_min = y_center - cell_size / 2
        y_max = y_center + cell_size / 2

        # Créer un polyligne (ligne fermée)
        line = LineString([
            (x_min, y_min), (x_max, y_min),
            (x_max, y_max), (x_min, y_max),
            (x_min, y_min)  # Fermer la ligne
        ])

        # Ajouter la polyligne à la liste
        polylines.append(line)

    geo_df = geopandas.GeoDataFrame(
        data, geometry=polylines,
        crs={'init': 'epsg:27572'})
    geo_df = geo_df.to_crs(crs={'init': 'epsg:2154'})

    geo_df.to_csv(path_result, index=False)

    # geo_df.to_file(path_result+'hydro_points_sim.shp')
