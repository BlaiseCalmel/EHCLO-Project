import pandas as pd
import numpy as np
import geopandas
import netCDF4
import re

def open_shp(path_shp: str):
    current_shp = geopandas.read_file(path_shp)
    return current_shp

def load_csv(path_csv, data_type='csv', sep=','):
    """

    :param path_csv:
    :param data_type:
    :param sep:
    :return:
    """
    # Climatic csv has a specific format
    if data_type == 'sqr':
        current_csv = pd.read_csv(path_csv, sep=sep, header=None, engine="python",
                                  names=[str(i) for i in range(3)])

        data_csv = current_csv.loc[9:]
        data_csv = data_csv.rename(columns={'0': 'date', '1': 'value', '2': 'indicator'}).reset_index(drop=True)

        # Format info
        resume_df = current_csv.iloc[:6, 0]
        resume_name = ['titre', 'num_poste', 'nom_usuel', 'lat', 'lon', 'alt']
        resume_dict = {}
        for idx, row in resume_df.items():
            value = re.split('= |#', row)[-1]
            try:
                value = float(value)
            except ValueError:
                pass
            resume_dict[resume_name[idx]] = value

        resume_dict['timeline'] = data_csv

        return resume_dict

    else:
        current_csv = pd.read_csv(path_csv, sep=sep)
        return current_csv


def load_ncdf(path_ncdf: str, indicator: str, station_codes: list[str]=None) -> dict:
    """

    :param path_ncdf:
    :param indicator:
    :param station_codes:
    :return:
    """

    # Read netCDF
    file2read = netCDF4.Dataset(path_ncdf,'r')

    # Get matching idx
    all_codes = file2read['code'][:].data

    if station_codes is None:
        station_codes = all_codes

    dict_data = {'time': file2read['time'][:].data}
    # dict_coord = {}
    for code in station_codes:
        # Get matching code index
        code_idx = np.where((all_codes==code).all(axis=1))[0]

        # Get data
        data_indicator = file2read[indicator][:, code_idx].data
        code_str = b''.join(code).decode('utf-8')
        dict_data[code_str] = data_indicator.flatten()
        # dict_coord[code_str] = (file2read['L93_X'][code_idx].data[0], file2read['L93_Y'][code_idx].data[0])

    return dict_data#, dict_coord

def format_data(dict_ncdf, stations_data):

    df_data = pd.DataFrame(dict_ncdf)

    df_data = df_data.drop('time', axis=1)
    df_mean = pd.DataFrame(df_data.mean(axis=0)).set_axis(['value'], axis=1)

    stations_data = stations_data.set_index('SuggestionCode')
    # df_coord = pd.DataFrame(dict_coord).T.set_axis(['lat', 'lon'], axis=1)

    df = pd.merge(df_mean, stations_data[['XL93', 'YL93']], left_index=True, right_index=True)

    return df










