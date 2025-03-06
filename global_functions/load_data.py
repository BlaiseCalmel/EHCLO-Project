"""
    EHCLO Project
    Copyright (C) 2025  Blaise CALMEL (INRAE)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import datetime as dt
import pandas as pd
import numpy as np
import geopandas
from tqdm import tqdm
import xarray as xr
import os
import re


def resample_ds(ds, var, timestep, operation='mean', q=None):
    # Seasonal indicator
    if timestep.lower() == 'jja':
        ds = ds.sel(time=ds['time'].dt.month.isin([6, 7, 8]))
        timestep = 'YE'
    elif timestep.lower() == 'djf':
        ds = ds.sel(time=ds['time'].dt.month.isin([1, 2, 12]))
        timestep = 'YE'

    if timestep == 'all':
        return ds[var].quantile(q, dim="time")
    else:
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
    return dataset.rename({var: suffix for var in dataset.data_vars if var.lower().find(var_name.lower()) != -1})

def apply_function_to_ds(ds, function, file_name, timestep):
    string_function = 'mean'
    int_value = None
    if function is not None:
        # function = 'sup0cons'
        match = re.match(r"([a-zA-Z]+)(\d+)?", function)
        if match:
            string_function = match.group(1).lower()
            int_value = int(match.group(2)) if match.group(2) else None

            if string_function == 'quantile':
                int_value = 1-float(int_value)/100
            elif string_function in ['inf', 'sup']:
                if ds[file_name].attrs.get("units") == 'K':
                    int_value = int_value + 273.15

                if string_function == 'sup':
                    ds[file_name] = xr.where(ds[file_name] > int_value, 1, 0)
                elif string_function == 'inf':
                    ds[file_name] = xr.where(ds[file_name] < int_value, 1, 0)
                string_function = 'sum'

    if int_value is not None:
        resampled_var = resample_ds(ds, var=file_name, timestep=timestep, operation=string_function,
                                    q=int_value)
    else:
        resampled_var = resample_ds(ds, file_name, timestep, operation=string_function)

    return resampled_var

def extract_ncdf_indicator(paths_data, param_type, sim_points_gdf, indicator, timestep=None, function=None,
                           start=None, end=None, path_result=None):

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
        if not 'debit' in indicator:
            paths_data = [[i] for i in paths_data]
        else:
            sim_chains = [i.split(os.sep)[-5:-1] for i in paths_data]
            unique_chains = [list(x) for x in set(tuple(x) for x in sim_chains)]
            temp_paths = []
            for chain in unique_chains:
                positions = [i for i, sub in enumerate(sim_chains) if sub == chain]
                temp_paths.append([paths_data[i] for i in positions])
            paths_data = temp_paths

    # Progress bar setup
    if path_result is None:
        title = indicator
    else:
        title = os.path.basename(path_result)
    total_iterations = len(paths_data)

    temp_paths = []
    with (tqdm(total=total_iterations, desc=f"Create {title} file") as pbar):
        # i=2
        # files=paths_data[i]
        # file1 = [
        #         '/home/bcalmel/Documents/2_data/historical-rcp85/HadGEM2-ES/ALADIN63/ADAMONT/SMASH/debit_France_MOHC-HadGEM2-ES_historical-rcp85_r1i1p1_CNRM-ALADIN63_v3_MF-ADAMONT-SAFRAN-1980-2011_INRAE-SMASH_day_20050801-20990731.nc'
        #     ]
        # file2=['/home/bcalmel/Documents/2_data/historical-rcp85/IPSL-CM5A-MR/RCA4/ADAMONT/J2000/debit_Rhone-Loire_IPSL-IPSL-CM5A-MR_historical-rcp85_r1i1p1_SMHI-RCA4_v3_MF-ADAMONT-SAFRAN-France-1980-2011_INRAE-J2000_day_20050801-21000731.nc']
        # paths_data = [file1, file2]
        #
        # ds1 = test[0]
        # file_name1 = 'HadGEM2-ES_ALADIN63_ADAMONT_SMASH'
        # ds2 = test[1]
        # file_name2 = 'IPSL-CM5A-MR_RCA4_ADAMONT_J2000'
        #
        # ds1[file_name1].quantile(0.9, dim='time')
        # ds2[file_name2].quantile(0.9, dim='time')
        # ds[file_name].quantile(0.9, dim='time')
        #
        # ds2 = xr.open_dataset(file2[0])
        # ds2.debit.quantile(0.9)
        #
        # test = []
        for i, files in enumerate(paths_data):
            if param_type == "climate":
                split_name = files[0].split(os.sep)[-4:-1]
            else:
                split_name = files[0].split(os.sep)[-5:-1]
                if "SAFRAN" in split_name:
                    split_name = files[0].split(os.sep)[-3:-1]

            file_name = '_'.join(split_name)
            var = indicator+'_'+file_name

            # if os.path.isfile(f"{temp_dir}{os.sep}{var}.nc"):
            #     if f"{temp_dir}{os.sep}{var}.nc" not in temp_paths:
            #         temp_paths.append(f"{temp_dir}{os.sep}{var}.nc")
            #     continue

            datasets = []
            for file in files:
                # print(file)
                # mypath= "/media/bcalmel/Explore2/hydrological-projection_daily-time-series_by-chain_raw-netcdf/rcp85/CNRM-CM5/ALADIN63/ADAMONT/CTRIP/debit_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_CNRM-ALADIN63_v2_MF-ADAMONT-SAFRAN-1980-2011_MF-ISBA-CTRIP_day_20050801-21000731.nc"
                ds = xr.open_dataset(file)

                # Check for coordinates without dimension
                dims_without_coords = [dim for dim in ds.dims if dim not in ds.coords]
                for dims in dims_without_coords:
                    if dims != 'station':
                        if len(ds[dims]) == len(ds['station']):
                            ds = ds.assign_coords({dims: ds.station})
                            ds = ds.swap_dims({dims: "station"}).drop_vars(dims)
                        else:
                            ds = ds.assign_coords({dims: ds[dims]})
                            vars_to_remove = [var for var in ds.data_vars if dims in ds[var].dims]
                            ds = ds.drop_vars(vars_to_remove + [dims])

                if len(dims_without_coords) > 0:
                    ds = ds.assign_coords({"station": ds.station})

                # Add sim suffix
                ds = rename_variables(ds, file_name, indicator.split('_')[0])
                # Load only selected period
                if start is not None:
                    ds = ds.sel(time=slice(dt.datetime(
                        start, 1, 1), None))
                if end is not None:
                    ds = ds.sel(time=slice(None, dt.datetime(
                        end, 12, 31)))

                # LII generates bug
                if 'LII' in ds.variables:
                    del ds['LII']

                # Remove attribute and define code as coordinates
                ds.attrs = {}
                ds[file_name].attrs = {}
                if 'code' in ds.data_vars:
                    ds = ds.set_coords("code")
                    ds = ds.swap_dims({'station': 'code'})
                if len(files) > 1:
                    datasets.append(ds)

            if len(datasets) > 1:
                # ds = xr.concat(datasets, dim="time").sortby("time")
                merge_dim = [file_name, 'L93_X', 'L93_Y']
                ds = xr.combine_by_coords([temp_ds[merge_dim] for temp_ds in datasets])

            # Compute quantile
            # if function is not None:
            # print('resample')
            resampled_var = apply_function_to_ds(ds, function, file_name, timestep)
            # Create a new dataset
            coordinates = {}
            for dim in resampled_var.dims:
                coordinates |= {dim: resampled_var[dim].values}

            # print(f"Create")
            if param_type == "climate":
                if ds[file_name].attrs.get("units") == 'kg.m-2.s-1':
                    ds[file_name] = ds[file_name] * 86400
                resampled_var = apply_function_to_ds(ds, function, file_name, timestep)
                # if timestep is not None:
                # resampled_var = resample_ds(ds, file_name, timestep)
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
                # Use station code as coordinate
                # ds = ds.set_coords('code')
                # ds = ds.swap_dims({'station': 'code'})
                # if 'station' in ds.data_vars:
                #     del ds['station']
                #
                # ds = ds.sel(code=ds["code"] != b'----------')
                # test.append(ds)

                ds = xr.Dataset({
                    file_name: (resampled_var.dims, resampled_var.values),
                    'x': (ds.L93_X.dims, ds.L93_X.values),
                    'y': (ds.L93_Y.dims, ds.L93_Y.values)
                }, coords=coordinates
                )

                if 'SAFRAN' in split_name:
                    # Get hydro model
                    hm = split_name[-1]
                    # Get remove station list
                    if os.path.isfile(f"/home/bcalmel/Documents/2_data/code_correction/{hm}_rm.csv"):
                        rm_hm = pd.read_csv(f"/home/bcalmel/Documents/2_data/code_correction/{hm}_rm.csv", sep=";")
                        rm_hm["AncienNom"] = rm_hm["AncienNom"].apply(lambda x: x.encode())
                        # ds = ds.sel(station=[int(j) for j in ds["station"].where(~ds["code"].isin(rm_hm["AncienNom"]),
                        #                                                          drop=True)])
                        ds = ds.where(~ds["code"].isin(rm_hm["AncienNom"]), drop=True)

                    # Get modify station list
                    if os.path.isfile(f"/home/bcalmel/Documents/2_data/code_correction/{hm}_mv.csv"):
                        mv_hm = pd.read_csv(f"/home/bcalmel/Documents/2_data/code_correction/{hm}_mv.csv", sep=";")
                        mv_hm["AncienNom"] = mv_hm["AncienNom"].apply(lambda x: x.encode())
                        mv_hm["NouveauNom"] = mv_hm["NouveauNom"].apply(lambda x: x.encode())
                        codes_updated = ds["code"].values.copy()
                        remove_list = []
                        for _, row in mv_hm.iterrows():
                            mask = (
                                    (ds["code"] == row["AncienNom"]) &
                                    (np.round(ds["x"]) == np.round(row["AncienX"])) &
                                    (np.round(ds["y"]) == np.round(row["AncienY"]))
                            )

                            if np.sum(mask) == 0:
                                mask = (ds["code"] == row["AncienNom"])

                            if np.sum(mask) == 1:
                                codes_updated = xr.where(mask, row["NouveauNom"], codes_updated)
                            else:
                                if row["AncienNom"] in ds.code:
                                    remove_list.append(row["AncienNom"])

                        ds = ds.assign_coords(code=("code", codes_updated.values))
                        if len(remove_list) > 0:
                            ds = ds.where(~ds["code"].isin(remove_list), drop=True)

                ds = ds.sel(code=ds["code"] != b'----------')
                gid_values = np.unique([code.encode() for code in sim_points_gdf.index.values])
                codes_to_select = [code for code in gid_values if code in ds['code'].values]
                if len(codes_to_select) > 0:
                    ds = ds.sel(code=codes_to_select)
                # station_to_select = ds.station.isel(station=ds.code.isin(codes_to_select))
                # if len(station_to_select) > 0:
                #     ds = ds.sel(station=station_to_select)

                # ds = ds.set_coords('code')
                # ds = ds.swap_dims({'station': 'code'})
                # if 'station' in ds.data_vars:
                # del ds['station']

                # ds = ds.rename({'code': 'gid'})
                # ds['gid'] = ds['gid'].astype(str)
                # gid_values = np.unique(sim_points_gdf.index.values)
                # codes_to_select = [code for code in gid_values if code in ds['gid'].values]
                # if len(codes_to_select) > 0:
                #     ds = ds.sel(gid=codes_to_select)

            # coordinates = {'gid': ds.gid.values, 'time': resampled_var['time'].values}
            # ds = xr.Dataset({
            #     file_name: (('time', 'gid'), resampled_var.values),
            #     'L93_X': (('gid'), ds.L93_X.values),
            #     'L93_Y': (('gid'), ds.L93_Y.values)
            # }, coords=coordinates
            # )

            # ds['gid'] = ds['gid'].astype(str)

            # Clean dataset
            # ds = xr.Dataset({
            #     file_name: (ds[file_name].dims, ds[file_name].values),
            #     'x': (('gid'), ds['L93_X'].values),
            #     'y': (('gid'), ds['L93_Y'].values),
            # }, coords={i: ds[i] for i in ds._coord_names}
            # )
            # ds = ds.set_coords('x')
            # ds = ds.set_coords('y')

            # datasets.append(ds[var])
            # print(f"Save")
            ds.to_netcdf(path=f"{temp_dir}{os.sep}{var}.nc")
            # print('Saved')
            temp_paths.append(f"{temp_dir}{os.sep}{var}.nc")

            # Update progress bar
            pbar.update(1)

    # Open temporary files and merge datasets
    combined_dataset = xr.open_mfdataset(temp_paths, combine='nested', compat='override')

    if 'code' in combined_dataset.coords:
        combined_dataset = combined_dataset.rename({'code': 'gid'})
        combined_dataset = combined_dataset.sel(gid=combined_dataset["gid"] != b'----------')

    combined_dataset = combined_dataset.set_coords('x')
    combined_dataset = combined_dataset.set_coords('y')

    # Save as ncdf
    if path_result is not None:
        if path_result[-2:] == 'nc':
            combined_dataset.to_netcdf(path=f"{path_result}")
            del combined_dataset
            del ds
            # Delete temporary directory
            for path in temp_paths:
                if os.path.isfile(path):
                    os.unlink(path)
            os.removedirs(temp_dir)
        else:
            df = combined_dataset.to_dataframe().reset_index()
            df.to_csv(f"{path_result}", index=False, sep=";")
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











