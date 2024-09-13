import geopandas
from shapely.geometry import Point, Polygon

def stations_in_shape(shapefile, selected_id, stations_data):
    # TODO function is really slow (use numba ?)
    # Format polygons
    selected_polygons = geopandas.GeoSeries({
        str(i): shapefile.loc[shapefile['id_newregi'] == i, 'geometry'].values[0] for i in selected_id})

    # Convert station coordinates to geometry Points
    _station_points = [Point(x, y) for x, y in stations_data[['XL93', 'YL93']].itertuples(index=False)]
    station_points = geopandas.GeoDataFrame(geometry=_station_points, index=stations_data['SuggestionCode'])

    # Find which stations is in selected polygons
    selected_stations = station_points.assign(**{
        key: station_points.within(geom) for key, geom in selected_polygons.items()})

    return selected_stations