import copy
import datetime as dt
import pandas as pd
import numpy as np
import geopandas
from tqdm import tqdm
import xarray as xr
import os
from global_functions.format_data import weighted_mean_per_region

def open_shp(path_shp: str):
    current_shp = geopandas.read_file(path_shp)

    # Correct if current shapefile is not from Lambert93 projection
    if 'Lambert-93' not in current_shp.crs.name:
        current_shp = current_shp.to_crs(crs={'init': 'epsg:2154'})

    return current_shp

def load_csv(path_file, sep=',', index_col=None):
    """
    :param path_file:
    :param sep:
    :return:
    """
    df = pd.read_csv(path_file, sep=sep, index_col=index_col)

    return df

def resample_ds(ds, var, timestep, operation='mean'):
    # Seasonal indicator
    if timestep.lower() == 'jja':
        ds = ds.sel(time=ds['time'].dt.month.isin([6, 7, 8]))
        timestep = 'YE'
    elif timestep.lower() == 'djf':
        ds = ds.sel(time=ds['time'].dt.month.isin([1, 2, 12]))
        timestep = 'YE'

    if operation == 'mean':
        return ds[var].resample(time=timestep).mean()
    elif operation == 'sum':
        return ds[var].resample(time=timestep).sum()
    elif operation == 'max':
        return ds[var].resample(time=timestep).max()
    elif operation == 'min':
        return ds[var].resample(time=timestep).min()
    else:
        raise ValueError(f"Operation '{operation}' is not supported.")

def rename_variables(dataset, suffix, indicator):
    return dataset.rename({var: var + '_' + suffix for var in dataset.data_vars if var == indicator})

def extract_ncdf_indicator(paths_data, param_type, sim_points_gdf, indicator, timestep=None,
                           start=None, path_result=None):
    datasets = []

    if param_type == 'climate':
        # Only load historical paths for available sim
        historical_paths = [path for path in paths_data if 'historical' in path]
        rcp_paths =  [path for path in paths_data if 'rcp' in path]
        historical_dir = [path.split(os.sep)[-4:-1] for path in historical_paths]
        rcp_dir =  [path.split(os.sep)[-4:-1] for path in rcp_paths]
        indexes_sim = [i for i, val in enumerate(historical_dir) if val in rcp_dir]

        paths_data = [historical_paths[idx] for idx in indexes_sim] + rcp_paths

    # Progress bar setup
    if path_result is None:
        title = indicator
    else:
        title = os.path.basename(path_result)
    total_iterations = len(paths_data)

    with tqdm(total=total_iterations, desc=f"Create {title} file") as pbar:

        for i, file in enumerate(paths_data):
            if param_type == "climate":
                split_name = file.split(os.sep)[-5:-1]
            else:
                split_name = file.split(os.sep)[-6:-1]
            indicator = indicator.split('_')[0]
            file_name = '_'.join(split_name)
            var = indicator+'_'+file_name

            ds = xr.open_dataset(file)
            # Add sim suffix
            ds_renamed = rename_variables(ds, file_name, indicator)
            if start is not None:
                ds_renamed = ds_renamed.sel(time=slice(dt.datetime(
                    start, 1, 1), None))

            # LII generates bug
            if 'LII' in ds_renamed.variables:
                del ds_renamed['LII']

            if param_type == "climate":
                # TODO Look for seasonal indicator (climate) DJF/JJA
                if timestep is not None:
                    resampled_var = resample_ds(ds_renamed, var, timestep)
                    coordinates = {i: ds_renamed[i] for i in ds_renamed._coord_names if i != 'time'}
                    coordinates['time'] = resampled_var['time']

                    ds_renamed = xr.Dataset({
                        var: (('time', 'y', 'x'), resampled_var.values)
                    }, coords=coordinates
                    )

                # Temporal selection, it's not a grid anymore
                ds_selection = ds_renamed.sel(
                    x=sim_points_gdf.iloc[:]['x'].values,
                    y=sim_points_gdf.iloc[:]['y'].values,
                    method="nearest")

                # Removed spatial aggregation
                # ds_selection = weighted_mean_per_region(ds=ds_selection, var=var, sim_points_gdf=sim_points_gdf,
                #                                         region_col='gid')

            else:
                ds_renamed = ds_renamed.set_coords('code')
                ds_renamed = ds_renamed.swap_dims({'station': 'code'})
                del ds_renamed['station']

                ds_renamed['code'] = ds_renamed['code'].astype(str)
                code_values = np.unique(sim_points_gdf.index.values)
                codes_to_select = [code for code in code_values if code in ds_renamed['code'].values]
                # TODO Rename dims to name
                ds_selection = ds_renamed.sel(code=codes_to_select)

            datasets.append(ds_selection)

            # Update progress bar
            pbar.update(1)

    # Merge datasets
    combined_dataset = xr.merge(datasets, compat='override')
    # if climate data merge historical and sim data
    if param_type == 'climate':
        column_groups = {}
        # Find historical and recent variable name
        for col in list(combined_dataset.variables.keys()):
            if 'rcp' in col or 'historical' in col:
                group_id = '_'.join(col.split('_')[2:])

                if group_id not in column_groups:
                    column_groups[group_id] = []
                column_groups[group_id].append(col)

        # Join historical and recent data
        for group_id, columns in column_groups.items():
            columns_sorted = sorted(columns, key=lambda x: ('rcp' in x, x))
            if len(columns_sorted) > 1:
                for col in columns_sorted[1:]:
                    combined_dataset[col] = combined_dataset[col].fillna(combined_dataset[columns_sorted[0]])
                combined_dataset = combined_dataset.drop_vars(columns_sorted[0])

    # Save as ncdf
    if path_result is not None:
        combined_dataset.to_netcdf(path=f"{path_result}")
    else:
         return combined_dataset

# def from_dict_to_df(data_dict):
#     # Transform dict to DataFrame
#     df = pd.concat({k: pd.DataFrame(v) for k, v in data_dict.items()})
#     df = df.reset_index().rename(columns={'level_0': 'sim', 'level_1': 'iteration'})
#
#     # Convert number of days since 01-01-1950 to year
#     df = convert_timedelta64(df)
#
#     # Define Horizon
#     df = define_horizon(df)
#     return df
#
# def convert_timedelta64(df, reference_date='1950-01-01'):
#     reference_date = pd.to_datetime(reference_date)
#     # start = dt.datetime(1950,1,1,0,0)
#     datetime_series = df['time'].astype('timedelta64[D]') + reference_date
#     df['year'] = datetime_series.dt.year
#     return df
#
# def group_by_function(df, stations_name, col_by=['sim'], function='mean', function2='median', bool_cols=None,
#                      relative=False, matched_stations=None):
#     df_stations = [i for i in stations_name if i in df.columns]
#
#     groupby_dict = {k: function for k in df_stations}
#     dict_temp = {}
#
#     if bool_cols is not None:
#         for col in bool_cols:
#             # Apply function on selected columns
#             df_temp = df[df[col] == True].groupby(col_by).agg(groupby_dict)
#             # Apply function2
#             df_temp = df_temp.agg(function2).to_frame().set_axis([col], axis=1)
#             dict_temp[col] = df_temp
#         df_plot = pd.concat([val for val in dict_temp.values()], axis=1)
#     else:
#         df_plot = df.groupby(col_by).agg(groupby_dict).T
#
#     if relative:
#         print('Warning: first column is used as reference')
#         for col in bool_cols[1:]:
#             df_plot[col]  = 100 * (df_plot[col] - df_plot[bool_cols[0]]) / df_plot[bool_cols[0]]
#             # dict_temp[col] = dict_temp[col][col] / dict_temp[bool_cols[0]][bool_cols[0]]
#
#     if matched_stations is not None:
#         df_plot = pd.concat([matched_stations[['XL93', 'YL93']], df_plot], axis=1)
#
#     # df_histo = df[df['Histo'] == True].groupby(col_by).agg(groupby_dict)
#     # df_H1 = df[df['H1'] == True].groupby(col_by).agg(groupby_dict)
#     # df_H2 = df[df['H2'] == True].groupby(col_by).agg(groupby_dict)
#     # df_H3 = df[df['H3'] == True].groupby(col_by).agg(groupby_dict)
#     #
#     # if relative:
#     #     df_H1 = df_H1 / df_histo
#     #     df_H2 = df_H2 / df_histo
#     #     df_H3 = df_H3 / df_histo
#
#     return df_plot
#











