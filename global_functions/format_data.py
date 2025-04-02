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
import copy
import xarray as xr
import numpy as np
import pandas as pd
import textwrap
import re
import json

def optimize_label_length(label, settings, length=20):
    # label_length = max([length*20/settings['fontsize'], len(max(re.split(r"[ ]", label), key=len))])
    label_length = max([length + 2*(20 - settings['fontsize']), len(max(re.split(r"[ ]", label), key=len))])
    wrapper = textwrap.TextWrapper(width=label_length, break_long_words=False, break_on_hyphens=True)
    wrapped_label = wrapper.wrap(label)
    return "\n".join(wrapped_label)

def load_settings(indicator_setup, name_indicator):
    with open('init_setup.json') as init_setup_file:
        init_setup = json.load(init_setup_file)

    settings = copy.deepcopy(init_setup)
    for key, value in indicator_setup.items():
        settings.update({key: value})

    settings = {key: value if value != 'None' else None for key, value in settings.items()}


    # Define variable name for axis
    if settings['label'] is None:
        title = name_indicator
    else:
        title = settings['label']

    if settings['units'] != '':
        units = f" ({settings['units']})"
    else:
        units = ''

    plot_type = settings['plot_type']
    start_cbar_ticks = ''
    end_cbar_ticks = ''
    if plot_type == 'difference':
        plot_type_name = 'difference'
        percent = False
    else:
        plot_type_name = 'delta'
        percent = True
        units = " (%)"

    if settings['start_cbar_ticks'] != '':
        start_cbar_ticks = f" {settings['start_cbar_ticks']}"
    if settings['end_cbar_ticks'] != '':
        end_cbar_ticks = f" {settings['end_cbar_ticks']}"

    return settings, title, units, plot_type, plot_type_name, percent, start_cbar_ticks, end_cbar_ticks

def get_season(month):
    if month in [12, 1, 2]:
        return 1
    elif month in [3, 4, 5]:
        return 2
    elif month in [6, 7, 8]:
        return 3
    elif month in [9, 10, 11]:
        return 4

def format_dataset(ds, data_type, files_setup, plot_function=None, return_period=None, tracc_year=None):
    other_dimension = None
    dimension_names = None
    if plot_function is not None:
        # TODO Define HM as secondary dimension
        if plot_function == 'min':
            ds = ds.groupby("time.year").min(dim="time")
            ds = ds.rename({"year": "time"})
            ds["time"] = pd.to_datetime(ds["time"].values, format="%Y")

        elif plot_function == 'max':
            ds = ds.groupby("time.year").max(dim="time")
            ds = ds.rename({"year": "time"})
            ds["time"] = pd.to_datetime(ds["time"].values, format="%Y")

        elif plot_function == 'month':
            ds = ds.assign_coords(month=ds['time.month'])
            other_dimension = 'month'
            dimension_names = {
                1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
                5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
                9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre",
            }
        elif plot_function == 'season':
            seasons = [get_season(date) for date in ds["time.month"]]
            ds = ds.assign_coords(season=("time", seasons))
            other_dimension = 'season'
            dimension_names = {1: "Hiver", 2: "Printemps", 3: "Été", 4: "Automne"}

    if 'tracc' in files_setup.keys() and files_setup['tracc']:
        horizons = {}
        adapted_date = -300
        i = 0
        if 1.4 in files_setup['horizons']['tracc']:
            i +=1
            horizons |= {f"horizon{i}" : [2020+adapted_date, 2039+adapted_date]}
        
        if 2.1 in files_setup['horizons']['tracc']:
            i +=1
            horizons |= {f"horizon{i}" : [2040+adapted_date, 2059+adapted_date]}
        
        if 3.4 in files_setup['horizons']['tracc']:
            i +=1
            horizons |= {f"horizon{i}" : [2090+adapted_date, 2109+adapted_date]}
        
        files_setup['horizons'] |= horizons
    columns = {}
    print(f'>> Define horizons...', end='\n')

    # Define horizons
    ds = define_horizon(ds, files_setup)
    simulation_cols = [i for i in list(ds.data_vars)]

    # Return period
    if return_period is not None:
        print(f'>> Compute value per return period {return_period} by horizon...', end='\n')
        ds = compute_return_period(ds, indicator_cols=list(ds.data_vars), files_setup=files_setup, return_period=return_period,
        other_dimension=other_dimension)
        simulation_horizon = [i for i in list(ds.data_vars) if '_by-horizon' in i]
    else:
        # Compute mean value for each horizon for each sim
        print(f'>> Compute mean by horizon...', end='\n')
        ds, simulation_horizon = compute_mean_by_horizon(ds=ds, indicator_cols=simulation_cols,
                                     files_setup=files_setup, other_dimension=other_dimension)
    # Compute deviation/difference to reference
    print(f'>> Compute deviation & difference by horizon for each simulation...', end='\n')
    ds = compute_deviation_to_ref(ds, cols=simulation_horizon)

    # Deviation/difference by timestep
    simulation_deviation = [i for i in list(ds.variables) if 'deviation' in i and '_by-horizon' not in i]
    simulation_difference = [i for i in list(ds.variables) if 'difference' in i and '_by-horizon' not in i]

    # Deviation/difference by horizon
    simulation_horizon_deviation_by_sims = [i for i in list(ds.variables) if '_by-horizon_deviation' in i]
    simulation_horizon_difference_by_sims = [i for i in list(ds.variables) if '_by-horizon_difference' in i]
    simulation_horizon_matching_by_sims = [i for i in list(ds.variables) if '_by-horizon_matching' in i]

    # Compute statistic among all sims
    print(f'>> Compute stats by horizon among simulations...', end='\n')
    ds, horizon_deviation = run_stats(ds, cols=simulation_horizon_deviation_by_sims, function=files_setup['function'],
                                      quantile=files_setup['quantile'], name="horizon_deviation")
    ds, horizon_difference = run_stats(ds, cols=simulation_horizon_difference_by_sims, function=files_setup['function'],
                                       quantile=files_setup['quantile'], name="horizon_difference")
    ds, horizon_matching = run_stats(ds, cols=simulation_horizon_matching_by_sims, function='stats',
                                     quantile=files_setup['quantile'], name="horizon_matching")

    # Find every HM
    if data_type == 'hydro':
        for plot_type in ["difference", "deviation"]:
            print(f'>> Compute stats by HM by horizon among simulations [{plot_type}]...', end='\n')
            if plot_type == "deviation":
                simulation_by_sims = simulation_horizon_deviation_by_sims
                simulation_stats = simulation_deviation
            else:
                simulation_by_sims = simulation_horizon_difference_by_sims
                simulation_stats = simulation_difference
            hm_names = [name.split('_')[-3] for name in simulation_by_sims]
            hm_dict_stats_horizon = {i: [] for i in np.unique(hm_names)}
            for idx, name_sim in enumerate(simulation_by_sims):
                hm_dict_stats_horizon[hm_names[idx]].append(name_sim)
            columns |= {f'hydro-model_sim-horizon_{plot_type}': hm_dict_stats_horizon}

            # Load timeline by HM
            hm_dict_stats_timeline = {i: [] for i in np.unique(hm_names)}
            for idx, name_sim in enumerate(simulation_stats):
                hm_dict_stats_timeline[hm_names[idx]].append(name_sim)
            columns |= {f'hydro-model_sim-timeline_{plot_type}': hm_dict_stats_timeline}

            # Compute stats for Horizons
            hydro_model_stats = {i: [] for i in np.unique(hm_names)}
            for hm, var_list in hm_dict_stats_horizon.items():
                ds, stats_name = run_stats(ds, hm_dict_stats_horizon[hm], function=files_setup['function'],
                                               quantile=files_setup['quantile'], name=f"horizon_{plot_type}_{hm}")
                hydro_model_stats[hm] = stats_name

            columns |= {f'hydro-model_{plot_type}': hydro_model_stats}

    ds, timeline_deviation = run_stats(ds, simulation_deviation, function=files_setup['function'],
                                       quantile=files_setup['quantile'], name="timeline-deviation")
    ds, timeline_difference = run_stats(ds, simulation_difference, function=files_setup['function'],
                                        quantile=files_setup['quantile'], name="timeline-difference")

    columns |= {'simulation_cols': simulation_cols, # raw value
                'simulation_deviation': simulation_deviation, # deviation from averaged historical reference
                'simulation_difference': simulation_difference, # difference from AHR
                'simulation_horizon': simulation_horizon, # mean value per horizon
                'simulation-horizon_by-sims_deviation': simulation_horizon_deviation_by_sims, # Horz deviation from AHR
                'simulation-horizon_by-sims_difference': simulation_horizon_difference_by_sims, # Horz difference from AHR
                'horizon_deviation': horizon_deviation, # mean horizon deviation among sims
                'horizon_difference': horizon_difference,
                'horizon_matching': horizon_matching,
                'timeline_deviation': timeline_deviation, # mean timeline deviation among sims
                'timeline_difference': timeline_difference
    }

    if dimension_names is not None:
        ds = ds.assign_coords({plot_function: [dimension_names[m] for m in ds[plot_function].values]})

    return ds, columns

def run_stats(ds, cols, function=['mean'], quantile=[], name="deviation"):
    ds_stats = apply_statistic(ds=ds[cols].to_array(dim='new'), function=function, q=quantile)
    if hasattr(ds_stats, "data_vars"):
        simulation_horizon = [f"{name}-{i}" for i in
                              list(ds_stats.data_vars)]
        ds[simulation_horizon] = ds_stats
    else:
        simulation_horizon = [name]
        ds[name] = ds_stats
    return ds, simulation_horizon


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
    # Define temporal horizon
    years = ds['time'].dt.year
    period_mask = (years >= files_setup['historical'][0]) & (years <= files_setup['historical'][1])
    ds = ds.assign_coords({'historical': period_mask})
    for horizon, dates in files_setup['horizons'].items():
        if horizon != 'tracc':
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
        if horizon != 'tracc':
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
    combined_means = combined_means.rename({i: i+'_by-horizon' for i in indicator_cols})

    combined_means = xr.merge([ds, combined_means])

    simulation_horizon = [i for i in list(combined_means.variables) if '_by-horizon' in i]

    return combined_means, simulation_horizon

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
            elif  func_name.lower() == 'max':
                agg_vars[func_name] =  ds.max(dim='new')
            elif  func_name.lower() == 'min':
                agg_vars[func_name] =  ds.min(dim='new')
            elif func_name.lower() == 'stats':
                agg_vars[func_name] = 100 * ds.sum(dim='new') / ds.count(dim='new')

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
            return ds.max(dim='new')
        elif  function.lower() == 'min':
            return ds.min(dim='new')
        elif function.lower() == 'stats':
            return 100 * ds.sum(dim='new') / ds.count(dim='new')
        else:
            raise ValueError("Unknown function, chose 'mean', 'median', 'max', 'min' or 'quantile'")

def compute_deviation_to_ref(ds, cols, ref='historical'):
    horizons = [i for i in ds.horizon.data if i != ref]

    for col in cols:
        data_col = col.split('_by-horizon')[0]
        ds[data_col+'_difference'] = ds[data_col] - ds[col].sel(horizon=ref)
        ds[data_col+'_deviation'] = (ds[data_col+'_difference']) * 100 / ds[col].sel(horizon=ref)

        ds[col+'_difference'] = (ds[col].sel(horizon=horizons) - ds[col].sel(horizon=ref))
        ds[col+'_deviation'] = (ds[col+'_difference']) * 100 / ds[col].sel(horizon=ref)

        ds[col+'_matching'] = ds[col+'_deviation'] > 0

        ds[col+'_matching'] = xr.where(ds[col+'_deviation'] > 0, 1,
                                       xr.where(ds[col+'_deviation'] < 0, -1, np.nan))

    return ds


def compute_return_period(ds, indicator_cols, files_setup, return_period=5, other_dimension=None):
    """
    Compute the LogNormal return period value for each station in xarray Dataset.

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
    horizons = ['historical'] + [k for k in files_setup['horizons'].keys() if 'horizon' in k]

    if other_dimension:
        data_dim = np.unique(ds[other_dimension].values)
        dict_by_horizon = {
            f"{i}_by-horizon": (["gid", "horizon", other_dimension],
                                                   np.full((len(ds['gid']),
                                                            len(horizons),
                                                            len(data_dim)), np.nan))
            for i in indicator_cols
        }
        coords = {"gid": ds['gid'].data, "horizon": horizons, other_dimension: data_dim}
    else:
        dict_by_horizon = {
            f"{i}_by-horizon": (["gid", "horizon"],
                                                   np.full((len(ds['gid']), len(horizons)), np.nan))
            for i in indicator_cols
        }
        coords = {"gid": ds['gid'].data, "horizon": horizons}

    result = xr.Dataset(dict_by_horizon, coords=coords)

    for var_name in indicator_cols:
        ds_var = ds[var_name]
        # Iterate over horizon
        for horizon in horizons:
            # Select period
            ds_horizon = ds_var.sel(time=ds_var[horizon])
            if other_dimension:
                for dim in data_dim:
                    ds_dim = ds_horizon.sel(time=ds_horizon.time.where(ds_horizon.month == dim, drop=True))
                    Xn = xr.apply_ufunc(compute_LogNormal, ds_dim, input_core_dims=[["time"]], vectorize=True)
                    result[f"{var_name}_by-horizon"].loc[:, horizon, dim] = Xn
            else:
                ds_dim = ds_horizon.groupby('time.year').min()
                Xn = xr.apply_ufunc(compute_LogNormal, ds_dim, input_core_dims=[["year"]], vectorize=True)
                result[f"{var_name}_by-horizon"].loc[:, horizon] = Xn

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
