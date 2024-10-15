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

def compute_horizon(ds, files_setup, function='mean'):
    # Define horizon
    years = ds['time'].dt.year
    period_mask = (years >= files_setup['historical'][0]) & (years <= files_setup['historical'][1])
    ds = ds.assign_coords({'historical': period_mask})
    for horizon, dates in files_setup['horizons'].items():
        period_mask = (years >= dates[0]) & (years <= dates[1])
        ds = ds.assign_coords({horizon: period_mask})

    # Compute
    dict_horizon = {}
    dict_horizon['historical'] = ds.sel(time=ds['historical']).mean(dim='time')
    for horizon, _ in files_setup['horizons'].items():
        dict_horizon[horizon] = ds.sel(time=ds[horizon]).mean(dim='time')

    return dict_horizon


def apply_statistic(ds, stat='mean', q=None):
    # Apply selected function
    if stat == 'mean':
        return ds.mean(dim='time')
    elif stat == 'median':
        return ds.median(dim='time')
    elif stat == 'quantile':
        if q is None:
            raise ValueError("You need to specify a quantile value")
        return ds.quantile(q, dim='time')
    else:
        raise ValueError("Unknown function, chose 'mean', 'median' or 'quantile'")

