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

def compute_mean_by_horizon(ds, indicator_cols, files_setup):
    mean_historical = ds.sel(time=ds['historical']).mean(dim='time')
    mean_historical = mean_historical.assign_coords(horizon='historical')
    horizon_list = [mean_historical[indicator_cols]]

    for horizon, dates in files_setup['horizons'].items():
        mean_horizon = ds.sel(time=ds[horizon]).mean(dim='time')
        mean_horizon = mean_horizon.assign_coords(horizon=horizon)
        horizon_list.append(mean_horizon[indicator_cols])

    combined_means = xr.concat(objs=horizon_list, dim='horizon')

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
    elif  function.lower() == 'new':
        return ds.min(dim='time')
    else:
        raise ValueError("Unknown function, chose 'mean', 'median', 'max', 'min' or 'quantile'")

