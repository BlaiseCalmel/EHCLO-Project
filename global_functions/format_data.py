
def define_horizon(ds, files_setup):
    years = ds['time'].dt.year
    period_mask = (years >= files_setup['historical'][0]) & (years <= files_setup['historical'][1])
    ds = ds.assign_coords({'historical': period_mask})
    for horizon, dates in files_setup['horizons'].items():
        period_mask = (years >= dates[0]) & (years <= dates[1])
        ds = ds.assign_coords({horizon: period_mask})

    return ds
