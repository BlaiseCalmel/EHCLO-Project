import xarray as xr
import numpy as np
import pandas as pd

def define_horizon(ds, files_setup):
    years = ds['time'].dt.year
    period_mask = (years >= files_setup['historical'][0]) & (years <= files_setup['historical'][1])
    ds = ds.assign_coords({'historical': period_mask})
    for horizon, dates in files_setup['horizons'].items():
        period_mask = (years >= dates[0]) & (years <= dates[1])
        ds = ds.assign_coords({horizon: period_mask})

    return ds

def weighted_mean_per_region(ds, variables, weight_var='weight', dim=('x', 'y')):
    regions = ds['region'].values
    unique_regions = np.unique(regions)

    # Dictionary to save results
    weighted_means = {var: [] for var in variables}

    # Compute weighted mean for each region
    for region in unique_regions:
        mask_region = ds['region'] == region
        ds_region = ds.where(mask_region, drop=True)

        for var in variables:
            # Compute weighted mean
            weighted_mean = ds_region[var].weighted(ds_region[weight_var]).mean(dim=dim)
            weighted_means[var].append(weighted_mean)

    # Save as dataset
    results = xr.Dataset(
        {var: xr.concat(weighted_means[var], dim=pd.Index(unique_regions, name='region')) for var in variables},
        coords={'time': ds['time'], 'region': unique_regions}
    )

    return results
