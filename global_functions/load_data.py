import datetime as dt
import pandas as pd
import numpy as np
import geopandas
import netCDF4
import re
import time
import os


def open_shp(path_shp: str):
    current_shp = geopandas.read_file(path_shp)

    # Correct if current shapefile is not from Lambert93 projection
    if 'Lambert-93' not in current_shp.crs.name:
        current_shp = current_shp.to_crs(crs={'init': 'epsg:2154'})

    return current_shp

def load_csv(path_files, data_type='csv', sep=','):
    """

    :param path_csv:
    :param data_type:
    :param sep:
    :return:
    """
    data_dict = {}
    time_start = time.time()
    estimate_timestep = 0
    i = -1
    if data_type == 'sqr':
        for path_csv in path_files :
            i += 1
            timedelta = (time.time() - time_start)
            files_to_open = (len(path_files) - i)
            if i > 1:
                estimate_timestep = timedelta / i
            # Climatic csv has a specific format
            name = os.path.splitext(os.path.basename(path_csv))[0]
            current_csv = pd.read_csv(path_csv, sep=sep, header=None, engine="python",
                                      names=[str(i) for i in range(3)])

            data_csv = current_csv.loc[9:]
            data_csv = data_csv.rename(columns={'0': 'date_'+name, '1': 'value_'+name, '2': 'indicator'+name
                                                }).reset_index(drop=True)
            data_dict[name] = data_csv
            # Format info
            # resume_df = current_csv.iloc[:6, 0]
            # resume_name = ['titre', 'num_poste', 'nom_usuel', 'lat', 'lon', 'alt']
            # resume_dict = {}
            # for idx, row in resume_df.items():
            #     value = re.split('= |#', row)[-1]
            #     try:
            #         value = float(value)
            #     except ValueError:
            #         pass
            #     resume_dict[resume_name[idx]] = value
            #
            # data_csv['date'] = pd.to_datetime(data_csv['date'])
            # resume_dict['timeline'] = data_csv
            print(f'============= {name} =============\n'
                  f'Running for {dt.timedelta(seconds=round(timedelta))}\n'
                  f'Ends in {dt.timedelta(seconds=round(files_to_open*estimate_timestep))} '
                  f'[{i+1} files/{len(path_files)}]')
    else:
        resume_dict = pd.read_csv(path_files, sep=sep)
        # name = os.path.splitext(os.path.basename(path_files))[0]
        #
        # data_dict[name] = resume_dict
        # df = pd.DataFrame.from_dict(data_dict).T
        # df.index = [i.split('_')[1] for i in df.index]

    return resume_dict


def split_ncdf(path):
    info = ['indicator', 'timestep', 'dates', 'timetype', 'geotype', 'localisation', 'project',
            'bc', 'rcp', 'gcm', 'rcm', 'hm']
    path_split = path.split('_')

    dict_info = {info[i]: path_split[i] for i in range(len(path_split))}

    return dict_info


def load_ncdf(path_ncdf: str, file_dict: dict, indicator: str, station_codes: list[str]=None) -> dict:
    """

    :param path_ncdf:
    :param indicator:
    :param station_codes:
    :return:
    """

    # Read netCDF
    open_netcdf = netCDF4.Dataset(path_ncdf,'r', encoding='utf-8')

    # Get matching idx
    netcdf_codes = open_netcdf['code'][:].data

    if station_codes is None:
        station_codes = netcdf_codes

    file_dict['time'] = open_netcdf['time'][:].data

    # dict_data = {'time': open_netcdf['time'][:].data, 'info': file_dict}

    for code in station_codes:
        code_to_bytes = [i.encode('utf-8') for i in code]

        # Get matching code index
        code_idx = np.where((netcdf_codes==code_to_bytes).all(axis=1))[0]

        if len(code_idx) > 0:
            # Get data
            data_indicator = open_netcdf[indicator][:, code_idx].data
            file_dict[code]= data_indicator.flatten()

    return file_dict

def iterate_over_path(path_indicator_files, param_type, parameters, selected_stations_name):
    dict_data = {}
    time_start = time.time()
    estimate_timestep = 0
    i = -1
    for path_ncdf in path_indicator_files:
        i += 1
        timedelta = (time.time() - time_start)
        files_to_open = (len(path_indicator_files) - i)
        if i > 1:
            estimate_timestep = timedelta / i
        # Get current file info
        file_name = os.path.basename(path_ncdf)[:-3]
        split_name = file_name.split('_')
        split_name += [''] * (len(parameters) - len(split_name))

        # Save them in dict
        file_dict = dict(zip(parameters, split_name))

        # Load ncdf [HYDRO]
        if param_type == 'hydro':
            dict_data[file_name] = load_ncdf(path_ncdf=path_ncdf, file_dict=file_dict,
                                             indicator=parameters['param_indicator'],
                                             station_codes=selected_stations_name)

        elif param_type == 'climat':
            dict_data[file_name] = load_ncdf(path_ncdf=path_ncdf, file_dict=file_dict,
                                             indicator=parameters['param_indicator'],
                                             station_codes=selected_stations_name)

        print(f'============= {file_name} =============\n'
              f'Running for {dt.timedelta(seconds=round(timedelta))}\n'
              f'Ends in {dt.timedelta(seconds=round(files_to_open*estimate_timestep))} '
              f'[{i+1} files/{len(path_indicator_files)}]')

    return dict_data

def from_dict_to_df(data_dict):
    # Transform dict to DataFrame
    df = pd.concat({k: pd.DataFrame(v) for k, v in data_dict.items()})
    df = df.reset_index().rename(columns={'level_0': 'sim', 'level_1': 'iteration'})

    # Convert number of days since 01-01-1950 to year
    df = convert_timedelta64(df)

    # Define Horizon
    df = define_horizon(df)
    return df

def convert_timedelta64(df, reference_date='1950-01-01'):
    reference_date = pd.to_datetime(reference_date)
    # start = dt.datetime(1950,1,1,0,0)
    datetime_series = df['time'].astype('timedelta64[D]') + reference_date
    df['year'] = datetime_series.dt.year
    return df

def define_horizon(df):
    df['Histo'] = df['year'] < 2006
    df['H1'] = (df['year'] >= 2021) & (df['year'] <= 2050)
    df['H2'] = (df['year'] >= 2041) & (df['year'] <= 2070)
    df['H3'] = df['year'] >= 2070
    return df

def group_by_function(df, stations_name, col_by=['sim'], function='mean', function2='median', bool_cols=None,
                     relative=False, matched_stations=None):
    df_stations = [i for i in stations_name if i in df.columns]

    groupby_dict = {k: function for k in df_stations}
    dict_temp = {}

    if bool_cols is not None:
        for col in bool_cols:
            # Apply function on selected columns
            df_temp = df[df[col] == True].groupby(col_by).agg(groupby_dict)
            # Apply function2
            df_temp = df_temp.agg(function2).to_frame().set_axis([col], axis=1)
            dict_temp[col] = df_temp
        df_plot = pd.concat([val for val in dict_temp.values()], axis=1)
    else:
        df_plot = df.groupby(col_by).agg(groupby_dict).T

    if relative:
        print('Warning: first column is used as reference')
        for col in bool_cols[1:]:
            df_plot[col]  = 100 * (df_plot[col] - df_plot[bool_cols[0]]) / df_plot[bool_cols[0]]
            # dict_temp[col] = dict_temp[col][col] / dict_temp[bool_cols[0]][bool_cols[0]]

    if matched_stations is not None:
        df_plot = pd.concat([matched_stations[['XL93', 'YL93']], df_plot], axis=1)

    # df_histo = df[df['Histo'] == True].groupby(col_by).agg(groupby_dict)
    # df_H1 = df[df['H1'] == True].groupby(col_by).agg(groupby_dict)
    # df_H2 = df[df['H2'] == True].groupby(col_by).agg(groupby_dict)
    # df_H3 = df[df['H3'] == True].groupby(col_by).agg(groupby_dict)
    #
    # if relative:
    #     df_H1 = df_H1 / df_histo
    #     df_H2 = df_H2 / df_histo
    #     df_H3 = df_H3 / df_histo

    return df_plot












