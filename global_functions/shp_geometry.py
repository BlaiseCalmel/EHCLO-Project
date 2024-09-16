import geopandas
from shapely.geometry import Point, Polygon

def stations_in_shape(shapefile, stations_data):
    # Station points L93
    points = geopandas.GeoDataFrame(
        stations_data, geometry=geopandas.points_from_xy(stations_data.XL93, stations_data.YL93),
        crs={'init': 'epsg:2154'}

    )
    # Polygons to L93
    shapefile_l93 = shapefile.to_crs(crs={'init': 'epsg:2154'})

    # Join both
    matched_points = points.sjoin(shapefile_l93, how='inner', predicate='intersects')

    return matched_points