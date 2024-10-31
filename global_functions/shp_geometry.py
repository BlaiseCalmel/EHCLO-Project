import geopandas
import pandas as pd
from shapely.geometry import Point, Polygon, LineString
import netCDF4
import numpy as np
import geopandas as gpd

def define_bounds(shapefile, zoom=5000):
    raw_bounds = shapefile.geometry.total_bounds
    return [raw_bounds[0] - zoom, raw_bounds[1] - zoom, raw_bounds[2] + zoom, raw_bounds[3] + zoom]

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

    else:
        matched_points = data.sjoin(shapefile, how='inner', predicate='intersects')
        matched_points = matched_points.drop('index_right', axis=1)

        # if geometry_type.value_counts().idxmax() == "LineString":
        #     # Keep only lines with more than 75% of their length inside polygons
        #     selected_lines = []
        #     for line in matched_points.itertuples():
        #         total_length = line.geometry.length
        #         intersection = shapefile.geometry.unary_union.intersection(line.geometry)
        #         intersection_length = intersection.length
        #         if intersection_length / total_length > 0.75:
        #             selected_lines.append(line.Index)
        #     matched_points = matched_points.loc[selected_lines]

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

def simplify(shapefile, tolerance=1000):
    shapefile_simplified = shapefile.copy()
    shapefile_simplified['geometry'] = shapefile_simplified['geometry'].simplify(
        tolerance, preserve_topology=True)

    return shapefile_simplified

def open_shp(path_shp: str):
    current_shp = geopandas.read_file(path_shp)

    # Correct if current shapefile is not from Lambert93 projection
    if 'Lambert-93' not in current_shp.crs.name:
        current_shp = current_shp.to_crs(crs={'init': 'epsg:2154'})

    return current_shp

def load_shp(dict_paths, files_setup):
    regions_shp = open_shp(path_shp=dict_paths['file_regions_shp'])
    study_ug_shp = open_shp(path_shp=dict_paths['file_ug_shp'])
    study_ug_shp = study_ug_shp[study_ug_shp['gid'].isin(files_setup['gid'])]
    study_ug_bv_shp = open_shp(path_shp=dict_paths['file_ug_bv_shp'])
    study_ug_bv_shp = study_ug_bv_shp[study_ug_bv_shp['gid'].isin(files_setup['gid'])]
    rivers_shp = open_shp(path_shp=dict_paths['file_rivers_shp'])

    return regions_shp, study_ug_shp, study_ug_bv_shp, rivers_shp

def simplify_shapefiles(study_ug_shp, study_ug_bv_shp, rivers_shp, regions_shp, tolerance=1000, zoom=50000):
    # Study geographical limits
    bounds = define_bounds(study_ug_shp, zoom=zoom)

    print(f'>> Simplify rivers...', end='\n')
    # Select rivers in study area
    study_rivers_shp_simplified = overlay_shapefile(shapefile=study_ug_bv_shp, data=rivers_shp)
    study_rivers_shp_simplified = simplify(study_rivers_shp_simplified, tolerance=tolerance)

    print(f'>> Simplify regions background...', end='\n')
    # Simplify regions shapefile (background)
    regions_shp_simplified = overlay_shapefile(shapefile=bounds, data=regions_shp)
    regions_shp_simplified = simplify(regions_shp_simplified, tolerance=tolerance)

    print(f'>> Simplify study area...', end='\n')
    # Simplify study areas shapefile
    study_ug_shp_simplified = simplify(study_ug_shp, tolerance=tolerance)
    study_ug_bv_shp_simplified = simplify(study_ug_bv_shp, tolerance=tolerance)

    return (study_ug_shp_simplified, study_ug_bv_shp_simplified, study_rivers_shp_simplified,
            regions_shp_simplified, bounds)

def test_merge_rivers(study_rivers_shp, study_rivers_shp_simplified):
    from shapely.ops import linemerge, unary_union
    from shapely.geometry import LineString, MultiLineString
    import matplotlib.pyplot as plt
    # 2. Combiner toutes les géométries (LineString et MultiLineString) en une seule avec unary_union
    # combined = unary_union(study_rivers_shp.geometry)
    # # 3. Fusionner les lignes continues en utilisant linemerge
    # if isinstance(combined, (MultiLineString, LineString)):
    #     merged_lines = linemerge(combined)
    # else:
    #     merged_lines = combined
    # merged_gdf = gpd.GeoDataFrame(geometry=[merged_lines])

    # 2. Fusionner toutes les géométries avec unary_union
    combined = unary_union(study_rivers_shp.geometry)

    # 3. Si c'est un MultiLineString, on le traite
    if isinstance(combined, MultiLineString):
        # Convertir en une liste de segments de lignes
        lines = list(combined.geoms)
    else:
        lines = [combined]

    # Fonction pour calculer la distance entre deux lignes
    def distance_between_lines(line1, line2):
        return line1.distance(line2)

    def distance_between_points(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    # Fonction pour relier deux lignes avec une interpolation
    def connect_lines(line1, line2):
        end_point = np.array(line1.coords[-1])
        start_point = np.array(line2.coords[0])
        connection = LineString([end_point, start_point])
        return linemerge([line1, connection, line2])

    # 4. Trier les lignes par leur longueur (commencer par la plus grande)
    lines.sort(key=lambda line: line.length, reverse=True)

    # 5. Initialiser avec la première (la plus grande) ligne
    merged_continuous = lines[0]
    lines_remaining = lines[1:]

    # 6. Boucler et connecter les lignes restantes, en commençant par les plus proches
    i=-1
    while lines_remaining:
        i+=1
        print(i)
        # Trouver la ligne la plus proche de la dernière ligne connectée
        distances = [distance_between_lines(merged_continuous, line) for line in lines_remaining]

        closest_index = np.argmin(distances)

        line2 = lines_remaining[closest_index]
        distance_between_lines(merged_continuous, line2)

        # Connecter la ligne la plus proche
        line2 = lines_remaining.pop(closest_index)

        global_min_distance = np.nan
        selected_point_from_merged = None
        selected_point = None
        # Choisir une extrémité de line2 (par exemple line2_end ici)
        for point_from_line in [line2.coords[0], line2.coords[-1]]:
            print(point_from_line)
            # Trouver le point le plus proche sur line1
            if isinstance(merged_continuous, MultiLineString):
                distances = [(point_j, LineString([point_from_line, point_j]).length) for line_i in merged_continuous.geoms for point_j in line_i.coords ]

            else:
                distances = [(point, LineString([point_from_line, point]).length) for point in merged_continuous.coords]

            # Trouver le point sur line1 avec la plus petite distance
            closest_point, min_distance = min(distances, key=lambda x: x[1])

            if np.isnan(global_min_distance) or min_distance < global_min_distance:
                selected_point_from_merged = closest_point
                selected_point = point_from_line

        connection = LineString([selected_point_from_merged, selected_point])
        if isinstance(merged_continuous, MultiLineString):
            merged_continuous = MultiLineString(list(merged_continuous.geoms) + [connection, line2])
        else:
            merged_continuous = linemerge([merged_continuous, connection, line2])

    def connect_lines(line1, line2):
        end_point = np.array(line1.coords[-1])
        start_point = np.array(line2.coords[0])
        connection = LineString([end_point, start_point])
        return linemerge([line1, connection, line2])


    # 6. Créer un nouveau GeoDataFrame avec la ligne continue
    merged_gdf = gpd.GeoDataFrame(geometry=[merged_continuous])


    # merged_gdf = gpd.GeoDataFrame(geometry=[merged_line])
    #
    lines_gdf = gpd.GeoDataFrame(geometry=[line2])


    # study_rivers_shp_simplified2 = overlay_shapefile(shapefile=study_ug_shp, data=study_rivers_shp_simplified)
    # test = study_rivers_shp_simplified['geometry'].apply(convert_multilinestring_to_linestring)

    fig, ax = plt.subplots(figsize=(10, 10))
    # for idx, row in merged_gdf.iterrows():
    #     # Vérifier si la géométrie est un MultiLineString
    #     if row['geometry'].geom_type == 'MultiLineString':
    #         for line in row['geometry'].geoms:
    #             ax.plot(*line.xy, color=row['color'], linewidth=0.01)  # Tracer chaque LineString dans le MultiLineString
    #     else:
    #         ax.plot(*row['geometry'].xy, color=row['color'], linewidth=0.01)  # Tracer le LineString
    study_rivers_shp_simplified[study_rivers_shp_simplified.length > 10000].plot(ax=ax, edgecolor='red', facecolor='none', linewidth=0.1, alpha=1)
    merged_gdf.plot(ax=ax, edgecolor='blue', facecolor='none', linewidth=1, alpha=0.4)
    lines_gdf.plot(ax=ax, edgecolor='k', facecolor='none', linewidth=3, alpha=0.4)
    plt.savefig('/home/bcalmel/Documents/3_results/HMUC_Loire_Bretagne/figures/test.pdf')