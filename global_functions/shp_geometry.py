import geopandas
import pandas as pd
from shapely.geometry import Point, Polygon
import netCDF4
import numpy as np

def is_data_in_shape(shapefile, data, cols=None, path_result=None):
    # Station points L93
    if cols is not None:
        if isinstance(data, netCDF4.Dataset):
            x = data[cols[0]][:].data
            y = data[cols[1]][:].data
            x_flat = x.flatten()
            y_flat = y.flatten()

            coordx = np.tile(np.arange(x.shape[1]), (x.shape[0], 1)).flatten()
            coordy = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T.flatten()
            names = [str(i)+'_'+str(j) for i, j in zip(coordx, coordy)]

            data = pd.DataFrame({'name': names, 'coordx': coordx, 'coordy': coordy})

            data = geopandas.GeoDataFrame(data, geometry=geopandas.points_from_xy(x_flat, y_flat),
                                          crs={'init': 'epsg:4326'})
            data = data.to_crs(crs={'init': 'epsg:2154'})

            data['XL93'] = data.geometry.x
            data['YL93'] = data.geometry.y
            data.to_csv(path_result, index=False)

        else:
            x = data[cols[0]]
            y = data[cols[1]]
            data = geopandas.GeoDataFrame(
                data, geometry=geopandas.points_from_xy(x, y),
                crs={'init': 'epsg:2154'}

            )

    # # Polygons to L93
    # shapefile_l93 = shapefile.to_crs(crs={'init': 'epsg:2154'})

    # Join both
    matched_points = data.sjoin(shapefile, how='inner', predicate='intersects')

    matched_points = matched_points.drop('index_right', axis=1)

    # Save the matched points shapefile
    if path_result is not None:
        matched_points.to_file(path_result, index=False)

    return matched_points