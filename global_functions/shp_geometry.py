import geopandas
import pandas as pd
from shapely.geometry import Point, Polygon
import netCDF4
import numpy as np
import xarray as xr

def is_data_in_shape(shapefile, data, cols, path_result=None):
    # Station points L93

    # col1 = data[cols[0]][:].data
    # col2 = data[cols[1]][:].data
    # col1_flat = col1.flatten()
    # col2_flat = col2.flatten()
    # # Columns and rows index
    # x_idx = np.tile(np.arange(col1.shape[1]), (col1.shape[0], 1)).flatten()
    # y_idx = np.tile(np.arange(col1.shape[0]), (col1.shape[1], 1)).T.flatten()
    # # Coordinates
    # x = np.tile(data['x'][:].data.flatten(), (col1.shape[0], 1)).flatten()
    # y = np.tile(data['y'][:].data.flatten(), (col1.shape[1], 1)).T.flatten()
    #
    # names = [str(i)+'_'+str(j) for i, j in zip(x_idx, y_idx)]
    #
    # data = pd.DataFrame({'name': names, 'x_idx': x_idx, 'coordy': y_idx, 'lat': col2_flat, 'lon': col1_flat,
    #                      'x': x, 'y': y})
    #
    # data = geopandas.GeoDataFrame(data, geometry=geopandas.points_from_xy(col1_flat, col2_flat),
    #                               crs={'init': 'epsg:4326'})
    # data = data.to_crs(crs={'init': 'epsg:2154'})
    #
    # data['XL93'] = data.geometry.x
    # data['YL93'] = data.geometry.y
    # if path_result is not None:
    #     data.to_csv(path_result, index=False)

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