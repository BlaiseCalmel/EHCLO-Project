import datetime as dt
import pandas as pd
import numpy as np
import geopandas
from tqdm import tqdm
import xarray as xr
import time
import os


def open_shp(path_shp: str):
    current_shp = geopandas.read_file(path_shp)

    # Correct if current shapefile is not from Lambert93 projection
    if 'Lambert-93' not in current_shp.crs.name:
        current_shp = current_shp.to_crs(crs={'init': 'epsg:2154'})

    return current_shp

def load_csv(path_file, sep=',', index_col=None):
    """
    :param path_csv:
    :param sep:
    :return:
    """
    df = pd.read_csv(path_file, sep=sep, index_col=index_col)

    return df

def resample_df(df, timestep, operation):
    if operation == 'mean':
        return df.resample(time=timestep).mean()
    elif operation == 'sum':
        return df.resample(time=timestep).sum()
    elif operation == 'max':
        return df.resample(time=timestep).max()
    elif operation == 'min':
        return df.resample(time=timestep).min()
    else:
        raise ValueError(f"Operation '{operation}' is not supported.")

def rename_variables(dataset, suffix, indicator):
    return dataset.rename({var: var + '_' + suffix for var in dataset.data_vars if var == indicator})

def extract_ncdf_indicator(path_ncdf, param_type, sim_points_df, indicator, timestep=None, operation=None,
                           path_result=None, files_setup=None):
    datasets = []
    code_bytes = None
    total_iterations = len(path_ncdf)
    if param_type == 'hydro':
        code_bytes = [i.encode('utf-8') for i in sim_points_df.index]

    # Progress bar
    with tqdm(total=total_iterations, desc=f"Load {indicator} ncdf") as pbar:

        for i, file in enumerate(path_ncdf):
            if code_bytes is None:
                split_name = file.split(os.sep)[-5:-1]
            else:
                split_name = file.split(os.sep)[-6:-1]
            file_name = '_'.join(split_name)

            ds = xr.open_dataset(file)

            # Add sim suffix
            ds_renamed = rename_variables(ds, file_name, indicator)
            if files_setup is not None:
                ds_formated = ds_renamed.sel(time=slice(dt.datetime(files_setup['historical'][0], 1, 1), None))

            # LII generates bug
            if 'LII' in ds_formated.variables:
                del ds_formated['LII']


            if timestep is not None and operation is not None:
                ds_formated = resample_df(ds_formated, timestep, operation)

            if code_bytes is None:
                ds_selection = ds_formated.sel(
                    x=xr.DataArray(sim_points_df.iloc[:]['x'], dims="z"),
                    y=xr.DataArray(sim_points_df.iloc[:]['y'], dims="z"),
                    method="nearest")
            else:
                idx_stations = ds_formated['code'].isin(code_bytes)
                val_station = ds_formated['station'].where(idx_stations, drop=True)
                ds_selection = ds_formated.sel(
                    station=val_station,
                    method="nearest")
            datasets.append(ds_selection)

            # Update progress bar
            pbar.update(1)

    # Merge datasets
    combined_dataset = xr.merge(datasets, compat='override')
    # if climate data merge historical and sim data
    if param_type == 'climate':
        column_groups = {}
        for col in list(combined_dataset.variables.keys()):
            if 'rcp' in col or 'historical' in col:
                group_id = '_'.join(col.split('_')[2:])

                if group_id not in column_groups:
                    column_groups[group_id] = []
                column_groups[group_id].append(col)

        for group_id, columns in column_groups.items():
            columns_sorted = sorted(columns, key=lambda x: ('rcp' in x, x))

            for col in columns_sorted[1:]:
                combined_dataset[col] = combined_dataset[col].fillna(combined_dataset[columns_sorted[0]])
            combined_dataset = combined_dataset.drop_vars(columns_sorted[0])

    if path_result is not None:
        if timestep is not None and operation is not None:
            combined_dataset.to_netcdf(path=f"{path_result}{indicator}_{timestep}_{operation}.nc")
        else:
            combined_dataset.to_netcdf(path=f"{path_result}{indicator}.nc")

    return combined_dataset

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












