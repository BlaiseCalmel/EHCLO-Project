import xarray as xr
import numpy as np
from scipy.stats import norm


def weighted_mean_per_region(ds, var, sim_points_gdf, region_col='gid'):
    # Spatial selection
    region_da = xr.DataArray(sim_points_gdf[region_col].values, dims=['x'], coords={'x': sim_points_gdf['x']})
    weight_da = xr.DataArray(sim_points_gdf['weight'].values, dims=['x'], coords={'x': sim_points_gdf['x']})

    # Add column to dataset
    ds = ds.assign_coords(region=region_da)
    ds['weight'] = weight_da

    # Spatial aggregation
    ds_mean_region = ds.groupby('region').mean(dim=('x', 'y'))
    ds_selection = ds_mean_region[[var]]

    return ds_selection

def define_horizon(ds, files_setup):
    # Define horizon
    years = ds['time'].dt.year
    period_mask = (years >= files_setup['historical'][0]) & (years <= files_setup['historical'][1])
    ds = ds.assign_coords({'historical': period_mask})
    for horizon, dates in files_setup['horizons'].items():
        period_mask = (years >= dates[0]) & (years <= dates[1])
        ds = ds.assign_coords({horizon: period_mask})

    return ds

def compute_mean_by_horizon(ds, indicator_cols, files_setup, other_dimension=None):
    # Select period
    mean_historical = ds.sel(time=ds['historical'])
    # Compute mean
    if other_dimension:
        mean_historical = mean_historical[indicator_cols].groupby(other_dimension).mean(dim='time')
    else:
        mean_historical = mean_historical.mean(dim='time')
    # Add horizon
    mean_historical = mean_historical.assign_coords(horizon='historical')
    horizon_list = [mean_historical[indicator_cols]]

    for horizon, dates in files_setup['horizons'].items():
        # Select period
        mean_horizon = ds.sel(time=ds[horizon])
        # Compute mean
        if other_dimension:
            mean_horizon = mean_horizon[indicator_cols].groupby(other_dimension).mean(dim='time')
        else:
            mean_horizon = mean_horizon.mean(dim='time')
        # Add horizon
        mean_horizon = mean_horizon.assign_coords(horizon=horizon)
        horizon_list.append(mean_horizon[indicator_cols])

    combined_means = xr.concat(objs=horizon_list, dim='horizon')
    combined_means = combined_means.rename({i: i+'_by_horizon' for i in indicator_cols})

    combined_means = xr.merge([ds, combined_means])

    return combined_means

def apply_statistic(ds, function='mean', q=None):
    if isinstance(function, list):
        agg_vars = {}
        for func_name in function:
            if func_name.lower() == 'mean':
                agg_vars[func_name] = ds.mean(dim='new')
            elif func_name.lower() == 'median':
                agg_vars[func_name] = ds.median(dim='new')
            elif func_name.lower() == 'quantile':
                for q_value in q:
                    if q_value > 1:
                        q_value = q_value / 100
                    da_quantile = ds.quantile(q_value, dim='new')
                    del da_quantile['quantile']
                    agg_vars[func_name+str(int(q_value*100))] = da_quantile
        ds_agg = xr.Dataset(agg_vars)
        return ds_agg
    else:
        # Apply selected function
        if function.lower() == 'mean':
            return ds.mean(dim='new')
        elif function.lower() == 'median':
            return ds.median(dim='new')
        elif function.lower() == 'quantile':
            if q is None:
                raise ValueError("You need to specify a quantile value")
            try:
                q = float(q)
            except ValueError:
                raise ValueError("Quantile should be a number")
            return ds.quantile(q, dim='new')
        elif  function.lower() == 'max':
            return ds.max(dim='time')
        elif  function.lower() == 'min':
            return ds.min(dim='time')
        else:
            raise ValueError("Unknown function, chose 'mean', 'median', 'max', 'min' or 'quantile'")

def compute_deviation_to_ref(ds, cols, ref='historical'):
    horizons = [i for i in ds.horizon.data if i != ref]
    for col in cols:
        ds[col+'_difference'] = (ds[col].sel(horizon=horizons) - ds[col].sel(horizon=ref))
        ds[col+'_deviation'] = (ds[col+'_difference']) * 100 / ds[col].sel(horizon=ref)

    return ds


def compute_return_period(ds, indicator_cols, files_setup, return_period=5, other_dimension=None):
    """
    Compute the LogNormal return period value for each station in an xarray Dataset.

    Parameters:
    ds (xarray.Dataset): The dataset containing flow data with dimensions ('time', 'station').
    indicator_cols (str): The name of the variable representing flow rates in the dataset.
    return_period (float): The return period for which to calculate the threshold value.

    Returns:
    xarray.DataArray: An array with the computed LogNormal return period values for each station.
    """

    # Check that return_period is a valid numeric value and greater than 1
    if not isinstance(return_period, (int, float)):
        raise ValueError("return_period must be a numeric value")
    if return_period <= 1:
        raise ValueError("return_period must be greater than 1")

    # Initialize the output array to store results for each station
    horizons = ['historical'] + list(files_setup['horizons'].keys())

    # if other_dimension:
    #     mean_historical = mean_historical[indicator_cols].groupby(other_dimension).mean(dim='time')
    # else:
    #     mean_historical = mean_historical.mean(dim='time')

    if other_dimension:
        data_dim = np.unique(ds[other_dimension])
        dict_by_horizon = {
            f"{i}_by_horizon_PdR{return_period}": (["id_geometry", "horizon", other_dimension],
                                                   np.full((len(ds['id_geometry']),
                                                            len(horizons),
                                                            len(data_dim)), np.nan))
            for i in indicator_cols
        }
        coords = {"id_geometry": ds['id_geometry'].data, "horizon": horizons, other_dimension: data_dim.data}
    else:
        dict_by_horizon = {
            f"{i}_PdR{return_period}_by_horizon": (["id_geometry", "horizon"],
                                                   np.full((len(ds['id_geometry']), len(horizons)), np.nan))
            for i in indicator_cols
        }
        coords = {"id_geometry": ds['id_geometry'].data, "horizon": horizons}

    result = xr.Dataset(dict_by_horizon, coords=coords)

    for var_name in indicator_cols:
        print(var_name)
        ds_var = ds[var_name]
        # Iterate over horizon
        for horizon in horizons:
            print(f"> {horizon}")
            # Select period
            ds_horizon = ds_var.sel(time=ds_var[horizon])
            if other_dimension:
                for dim in data_dim:
                    ds_dim = ds_horizon.sel(time=ds_horizon.time.where(ds_horizon.month == dim, drop=True))
                    Xn = xr.apply_ufunc(compute_LogNormal, ds_dim, input_core_dims=[["time"]], vectorize=True)
                    result[f"{var_name}_PdR{return_period}_by_horizon"].loc[:, horizon, dim] = Xn
            else:
                ds_dim = ds_horizon.groupby('time.year').min()
                Xn = xr.apply_ufunc(compute_LogNormal, ds_dim, input_core_dims=[["year"]], vectorize=True)
                result[f"{var_name}_by_horizon_PdR{return_period}"].loc[:, horizon] = Xn

        combined_means = xr.merge([ds, result])

    return combined_means


def compute_LogNormal(X, return_period=5):
    """
    Compute the quantile value (e.g., QMNA5) based on a log-normal distribution
    by calculating the 1/returnPeriod quantile of the data.

    Parameters:
    - X : xarray.DataArray
        The data to analyze (should be a 1D array of discharge values).
    - return_period : float
        The return period, typically 5 for QMNA5 (i.e., 1/5 probability of not exceeding the value).

    Returns:
    - Xn : float
        The computed quantile value corresponding to the return period.
    """
    # Remove NaN values from X
    X = X[~np.isnan(X)]

    # Check if there are valid values to compute
    if len(X) == 0:
        return np.nan  # Return NaN if no valid values exist

    # Frequency corresponding to the return period
    Freq = 1 / return_period

    # Handle the case where there are no zeros or very few zeros
    nbXnul = np.sum(X == 0)
    nbY = len(X)

    if (nbXnul / nbY) <= Freq:
        # If the proportion of zeros is small enough, calculate the quantile
        Xn = np.exp(np.percentile(np.log(X[X > 0]), Freq * 100))
    else:
        # If too many zeros, return zero
        Xn = 0

    return Xn
