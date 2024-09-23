import geopandas
from shapely.geometry import Point, Polygon

def is_data_in_shape(shapefile, data, cols=None, path_results=None):
    # Station points L93
    if cols is not None:
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

    if path_results is not None:
        matched_points.to_file(path_results)

    return matched_points