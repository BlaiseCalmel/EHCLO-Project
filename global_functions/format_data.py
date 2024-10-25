import xarray as xr

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

