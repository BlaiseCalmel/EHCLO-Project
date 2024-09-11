import pandas as pd
import numpy as np
import geopandas
import netCDF4

def open_shp(path_shp: str):
    current_shp = geopandas.read_file(path_shp)
    return current_shp

def load_ncdf(path_ncdf: str, indicator: str, station_codes: list[str]=None) -> tuple[
    dict[str, float], dict[str, tuple[float, float]]]:

    # Read netCDF
    file2read = netCDF4.Dataset(path_ncdf,'r')

    # Get matching idx
    all_codes = file2read['code'][:].data

    if station_codes is None:
        station_codes = all_codes

    dict_data = {'time': file2read['time'][:].data}
    dict_coord = {}
    for code in station_codes:
        # Get matching code index
        code_idx = np.where((all_codes==code).all(axis=1))[0]

        # Get data
        data_indicator = file2read[indicator][:, code_idx].data
        code_str = b''.join(code).decode('utf-8')
        dict_data[code_str] = data_indicator.flatten()
        dict_coord[code_str] = (file2read['L93_X'][code_idx].data[0], file2read['L93_Y'][code_idx].data[0])

    return dict_data, dict_coord

def format_data(dict_ncdf, dict_coord, ):

    df_data = pd.DataFrame(dict_ncdf)

    df_data = df_data.drop('time', axis=1)
    df_mean = pd.DataFrame(df_data.mean(axis=0)).set_axis(['value'], axis=1)

    df_coord = pd.DataFrame(dict_coord).T.set_axis(['lat', 'lon'], axis=1)

    df = pd.merge(df_mean, df_coord, left_index=True, right_index=True)

    return df










