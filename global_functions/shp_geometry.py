import geopandas
import pandas as pd
from shapely.geometry import Point, Polygon, LineString
import netCDF4
import numpy as np
import geopandas as gpd

def overlay_shapefile(shapefile, data, path_result=None, col='gid'):

    # x = data[cols[0]]
    # y = data[cols[1]]
    # geo_df = geopandas.GeoDataFrame(
    #     data, geometry=geopandas.points_from_xy(x, y),
    #     crs={'init': 'epsg:2154'}
    #
    # )
    # # Polygons to L93
    # shapefile_l93 = shapefile.to_crs(crs={'init': 'epsg:2154'})
    # Join both
    geometry_type = data['geometry'].apply(check_geometry_type)

    if isinstance(shapefile, list):
        coords = [
            (shapefile[0], shapefile[1]),  # Coin supérieur gauche
            (shapefile[2], shapefile[1]),    # Coin supérieur droit
            (shapefile[2], shapefile[3]),     # Coin inférieur droit
            (shapefile[0], shapefile[3])    # Coin inférieur gauche
        ]
        polygon = Polygon(coords)
        shapefile = gpd.GeoDataFrame({'geometry': [polygon]})
        shapefile.crs = data.crs

    matched_points = None
    if geometry_type.value_counts().idxmax() == "Polygon":
        data['geometry'] = data.intersection(shapefile.union_all())

        matched_points = gpd.overlay(data, shapefile, how='intersection')
        matched_points['surface'] = matched_points.area

        total_surface = matched_points.groupby(col).agg({'surface':'sum'})
        total_surface = total_surface.rename(columns={'surface': 'total_surface'})
        if 'total_surface' in matched_points.columns:
            matched_points = matched_points[[i for i in matched_points.columns if i != 'total_surface']]

        matched_points = matched_points.merge(total_surface, left_on=col, right_index=True)

    elif geometry_type.value_counts().idxmax() in ["Point", "LineString"]:
        matched_points = data.sjoin(shapefile, how='inner', predicate='intersects')
        matched_points = matched_points.drop('index_right', axis=1)

    # Save the matched points shapefile
    if matched_points is not None and path_result is not None:
        matched_points.to_file(path_result, index=False)
    else:
        return matched_points

def check_geometry_type(geometry):
    if isinstance(geometry, Polygon):
        return "Polygon"
    elif isinstance(geometry, Point):
        return "Point"
    elif isinstance(geometry, LineString):
        return "LineString"
    else:
        return "Other"

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

    geo_points = geopandas.GeoDataFrame(
        data, geometry=geopandas.points_from_xy(col1_flat, col2_flat),
        crs={'init': 'epsg:4326'})

    geo_df2 = geo_points.to_crs(crs={'init': 'epsg:27572'})
    geo_df2['geometry'] = geo_df2['geometry'].apply(create_polygon_from_point)

    geo_df93 = geo_df2.to_crs(crs={'init': 'epsg:2154'})

    geo_df93.to_file(path_result+'climate_points_sim.shp')

def create_polygon_from_point(point, cell_size=8000):
    half_size = cell_size / 2
    x, y = point.x, point.y

    # Compute corner
    x_min = x - half_size
    x_max = x + half_size
    y_min = y - half_size
    y_max = y + half_size

    # Generate a polygon
    polygon = Polygon([
        (x_min, y_min), (x_max, y_min),
        (x_max, y_max), (x_min, y_max),
        (x_min, y_min)
    ])

    return polygon

def simplify_shapefiles(shapefile, tolerance=1000):
    shapefile_simplified = shapefile.copy()
    shapefile_simplified['geometry'] = shapefile_simplified['geometry'].simplify(
        tolerance, preserve_topology=True)

    return shapefile_simplified