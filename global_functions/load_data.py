import copy
import datetime as dt
import pandas as pd
import numpy as np
import geopandas
from tqdm import tqdm
import xarray as xr
import os


def resample_ds(ds, var, timestep, operation='mean', q=None):
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
    elif operation == 'quantile':
        return ds[var].resample(time=timestep).quantile(q)
    else:
        raise ValueError(f"Operation '{operation}' is not supported.")

def rename_variables(dataset, suffix, var_name):
    return dataset.rename({var: suffix for var in dataset.data_vars if var == var_name})

def extract_ncdf_indicator(paths_data, param_type, sim_points_gdf, indicator, timestep=None,
                           start=None, path_result=None):

    # Create temporary directory
    temp_dir = os.path.dirname(path_result) + os.sep + '_temp'
    if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir)

    if param_type == 'climate':
        # Only load historical paths for available sim
        historical_paths = [path for path in paths_data if 'historical' in path]
        rcp_paths =  [path for path in paths_data if 'rcp' in path]
        historical_dir = [path.split(os.sep)[-4:-1] for path in historical_paths]
        rcp_dir =  [path.split(os.sep)[-4:-1] for path in rcp_paths]

        indexes_sim = [historical_dir.index(i) for i in rcp_dir]
        paths_data = [[historical_paths[val], rcp_paths[idx]] for idx, val in enumerate(indexes_sim)]
    else:
        paths_data = [[i] for i in paths_data]

    # Progress bar setup
    if path_result is None:
        title = indicator
    else:
        title = os.path.basename(path_result)
    total_iterations = len(paths_data)

    temp_paths = []
    with tqdm(total=total_iterations, desc=f"Create {title} file") as pbar:
        for i, files in enumerate(paths_data):
            if param_type == "climate":
                split_name = files[0].split(os.sep)[-4:-1]
            else:
                split_name = files[0].split(os.sep)[-5:-1]

            indicator = indicator.split('_')[0]
            var_name = indicator.split('$')[-1]
            indicator = indicator.split('$')[0]

            file_name = '_'.join(split_name)
            var = indicator+'_'+file_name

            datasets = []
            for file in files:
                ds = xr.open_dataset(file)
                # Add sim suffix
                ds = rename_variables(ds, file_name, var_name)
                # Load only selected period
                if start is not None:
                    ds = ds.sel(time=slice(dt.datetime(
                        start, 1, 1), None))

                # LII generates bug
                if 'LII' in ds.variables:
                    del ds['LII']

                if param_type == "climate":
                    # TODO Look for seasonal indicator (climate) DJF/JJA
                    if timestep is not None:
                        resampled_var = resample_ds(ds, file_name, timestep)
                        coordinates = {i: ds[i] for i in ds._coord_names if i != 'time'}
                        coordinates['time'] = resampled_var['time']
                        ds = xr.Dataset({
                            file_name: (('time', 'y', 'x'), resampled_var.values)
                        }, coords=coordinates
                        )

                    ds = ds.sel(x=xr.DataArray(sim_points_gdf['x']), y=xr.DataArray(sim_points_gdf['y']))
                    ds = ds.assign_coords(dim_0=sim_points_gdf['name']).rename(dim_0='name')
                    ds = ds.rename({'name': 'gid'})

                else:
                    if indicator == 'Q2':
                        resampled_var = resample_ds(ds, var, timestep, operation='quantile', q=0.98)
                        coordinates = {i: ds[i] for i in ds._coord_names if i != 'time'}
                        coordinates['time'] = resampled_var['time']
                        ds = xr.Dataset({
                            var: (('time', 'station'), resampled_var.values)
                        }, coords=coordinates
                        )

                    ds = ds.set_coords('code')
                    ds = ds.swap_dims({'station': 'code'})
                    del ds['station']

                    ds['code'] = ds['code'].astype(str)
                    code_values = np.unique(sim_points_gdf.index.values)
                    codes_to_select = [code for code in code_values if code in ds['code'].values]
                    missing = [code for code in code_values if code not in ds['code'].values]

                    ds = ds.rename({'code': 'gid'})
                    if len(codes_to_select) > 0:
                        # TODO Rename dims to name
                        ds = ds.sel(gid=codes_to_select)

                        # Clean dataset
                        ds = xr.Dataset({
                            file_name: (('time', 'gid'), ds[file_name].values),
                            'x': (('gid'), ds['L93_X'].values),
                            'y': (('gid'), ds['L93_Y'].values),
                        }, coords={i: ds[i] for i in ds._coord_names}
                        )
                        ds = ds.set_coords('x')
                        ds = ds.set_coords('y')
                    else:
                        continue

                datasets.append(ds)

            if len(datasets) > 1:
                ds = xr.concat(datasets, dim="time").sortby("time")

            # datasets.append(ds[var])
            ds.to_netcdf(path=f"{temp_dir}{os.sep}{var}.nc")
            temp_paths.append(f"{temp_dir}{os.sep}{var}.nc")

            # Update progress bar
            pbar.update(1)

    # Open temporary files and merge datasets
    combined_dataset = xr.open_mfdataset(temp_paths, combine='nested', compat='override')

    # Save as ncdf
    if path_result is not None:
        combined_dataset.to_netcdf(path=f"{path_result}")
        del combined_dataset
        del ds
        # Delete temporary directory
        for path in temp_paths:
            if os.path.isfile(path):
                os.unlink(path)
        os.removedirs(temp_dir)
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











