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
from sklearn.cluster import KMeans
import os
import xarray as xr
import numpy as np
from sklearn.decomposition import PCA
from global_functions.format_data import format_dataset
from plot_functions.plot_narratives import plot_narratives
from global_functions.load_data import open_fst
import pandas as pd
import json

def representative_item(X_cluster, centroids, cluster, cluster_id, indices_cluster, 
                        method='closest', weight=None, distance_max=None):
    idx = None
    if weight is None:
        weight = 1
    
    if method == 'closest':
        # Compute distance from cluster centroid
        distances = np.linalg.norm(X_cluster - centroids[cluster], axis=1)

        # Get index of the closest sim
        idx = indices_cluster[np.argmin(distances)]
    elif method == 'furthest':
        # Compute distance from other cluster centroids
        distances_list = []
        for c in cluster_id:
            if c != cluster:
                distances_list.append(np.linalg.norm(X_cluster - centroids[c], axis=1))

            distances = np.mean(distances_list, axis=0)
            idx = indices_cluster[np.argmax(distances)]
    elif method == 'combine':
        # Compute distance from cluster centroid
        distances_cluster = np.linalg.norm(X_cluster - centroids[cluster], axis=1) 
        if distance_max is not None:
            mask_cluster = distances_cluster <= distance_max
        else:
            mask_cluster = np.logical_not(np.isnan(distances_cluster))
        # mask_cluster = np.tile(True, distances_cluster.shape)

        # Compute distance from other cluster centroids
        distances_list = []
        for c in cluster_id:
            if c != cluster:
                distances_list.append(np.linalg.norm(X_cluster - centroids[c], axis=1) * weight) 
        distances_other = np.min(distances_list, axis=0)

        # min_distances = np.minimum.reduce([distances_other[0], distances_other[1], distances_other[2]])
        if np.any(distances_other) and np.any(mask_cluster):
            idx = indices_cluster[mask_cluster][np.argmax(distances_other[mask_cluster])]

    return idx


def compute_narratives(paths_ds_narratives, files_setup, path_narratives, stations,
                       path_performances, path_figures, path_formated_ncdf,
                       indicator_values=["QJXA", "QA", "VCN10"],
                       horizon_ref='horizon3', quantiles=[0.5]):

    # Load selected indicators
    print(f">> Load & Format {indicator_values} datasets...", end='\n')
    datasets_list = []
    for path_ncdf in paths_ds_narratives:
        # Open ncdf dataset
        # path_ncdf = f"{dict_paths['folder_study_data']}{indicator}_rcp85_YE_TRACC_all-BV.nc"
        if not os.path.isfile(f"{path_formated_ncdf}formated-{path_ncdf.split(os.sep)[-1]}"):
            ds_stats = xr.open_dataset(path_ncdf)
            ds_stats, var_names = format_dataset(ds=ds_stats, data_type='hydro', files_setup=files_setup,
                                                 path_result=f"{path_formated_ncdf}formated-{path_ncdf.split(os.sep)[-1]}")
        else:
            ds_stats = xr.open_dataset(f"{path_formated_ncdf}formated-{path_ncdf.split(os.sep)[-1]}")
            var_names = {}
            var_names[f'simulation-horizon_by-sims_deviation'] = [i for i in ds_stats.data_vars if '_by-horizon_deviation' in i]
        ds_stats['gid'] = ds_stats['gid'].astype(str)
        datasets_list.append(ds_stats)

    print(f">> Merge {indicator_values} datasets...", end='\n')
    narratives = {}
    data_arrays = []
    # for h in horizons:
    datasets = [ds_i[var_names[f'simulation-horizon_by-sims_deviation']].sel(
        horizon=horizon_ref, gid=stations) for ds_i in datasets_list] #, gid=stations
    for i in range(len(datasets)):
        ds = datasets[i]
        for var_name, da in ds.data_vars.items():
            # Extract names part
            parts = var_name.split("_")
            nom_gcmrcm = "_".join(parts[:2])
            nom_bc, nom_hm = parts[2:4]

            # Generate new DataArray with sim as dimension
            da_expanded = da.expand_dims({
                "gcm-rcm": [nom_gcmrcm],
                "bc":  [nom_bc],
                "hm":  [nom_hm],
                "horizon":  [horizon_ref],
            })

            # Get name of the current indicator as var name
            da_expanded.name = indicator_values[i]

            data_arrays.append(da_expanded)

    print(f">> Compute KMeans clusters...", end='\n')
    available_stations = data_arrays[0].gid.values
    # Combine DataArrays
    combined_da = xr.combine_by_coords(data_arrays)
    str_quantiles = 'quant'+('-').join([f"{int(i*100)}" for i in  quantiles])
    combined_da = combined_da.quantile(dim='gid', q=quantiles)
    ds_stacked = combined_da.stack(sample=("gcm-rcm", "bc", "hm"))
    list_ds = []

    for var in indicator_values:
        list_ds.append(ds_stacked.sel(horizon=horizon_ref, quantile=0.5)[var].values)
    X_imputed = np.column_stack(list_ds)

    # KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X_imputed)

    # Find centroids
    centroids = kmeans.cluster_centers_  

    # Identify clusters
    cluster_id = np.unique(labels)

    # # Load hm performances .fst files
    # # Run once to install the related R packages
    # utils = importr('utils')
    # utils.install_packages('fst')
    # utils.install_packages('data.table')

    print(f">> Get hydrological models performance...", end='\n')
    performances = ['Biais', 'Q10', 'Q90']
   
    # Baseflow performance
    df = open_fst(f"{path_performances}dataEX_Explore2_criteria_diagnostic_performance.fst")
    baseflow_criteria = df[['HM', 'code','Biais']]

    # Highflow performance
    df = open_fst(f"{path_performances}dataEX_Explore2_criteria_diagnostic_HF.fst")
    highflow_criteria = df[['HM', 'code','Q10']]

    # Lowflow performance
    df = open_fst(f"{path_performances}dataEX_Explore2_criteria_diagnostic_LF.fst")
    lowflow_criteria = df[['HM', 'code','Q90']]

    # Merge in a single dataframe
    merged_performances = pd.merge(pd.merge(baseflow_criteria, lowflow_criteria, on=['HM', 'code']),
                                highflow_criteria, on=['HM', 'code'])

    # Define threshold for each criteria
    performance_thresholds = {'Biais': 0.2, 'Q10': 0.2, 'Q90': 0.8}

    # Get selected stations
    merged_performances = merged_performances[merged_performances['code'].isin(available_stations)]
    count_stations = len(np.unique(merged_performances['code']))

    # Compare to thresholds
    merged_performances[performances] = abs(merged_performances[performances])
    valid_performance = []
    for key, value in performance_thresholds.items():
        merged_performances[key+'_valid'] = merged_performances[key] < value
        valid_performance.append(key+'_valid')

    # Compute percentage of station above threshold for each HM
    hm_performances = merged_performances.groupby(['HM']).agg({v: 'sum' for v in valid_performance})

    # Normalize by total station among area
    hm_performances[[f'{i}_ratio' for i in valid_performance]] = hm_performances[valid_performance].div(
        count_stations, axis=0)

    performance_threshold = np.quantile(hm_performances[[f'{i}_ratio' for i in valid_performance]], 0.5, axis=0)
    hm_performances[[f'{i}_bool' for i in valid_performance]] = hm_performances[[f'{i}_ratio' for i in valid_performance]] >= performance_threshold
    hm_performances['sum'] = hm_performances[[f'{i}_bool' for i in valid_performance]].all(axis=1)

    hydrological_models = ds_stacked["hm"].values
    above_threshold = np.array([hm_performances.loc[hm]['sum'] for hm in hydrological_models])

    # Bleunavy Orange Brun Turquoise https://www.canva.com/colors/color-palettes/freshly-sliced-fruit/
    cluster_names = ['Argousier', 'Cèdre', 'Séquoia', 'Genévrier']
    hex_colors = ["#E66912", "#016367", "#870000", "#0f063b"]
    # ["#E66912", "#016367", "#9E3A14", "#0B1C48"]

    rows = None

    above_threshold_array = np.tile(False, hydrological_models.shape)
    print(f">> Define narrative simulation per cluster...", end='\n')
    representative_groups = {}
    for cluster in cluster_id:
        # Index of cluster values
        indices_cluster = np.where(labels == cluster)[0]

        # # Get performances for current cluster
        # cluster_hm = np.unique(hydrological_models[indices_cluster])
        # cluster_hm_performances = hm_performances.loc[cluster_hm]
        # cluster_hm_performances[[f'{i}_ratio' for i in valid_performance]] = cluster_hm_performances[valid_performance].div(count_stations, axis=0)
        # # Define threshold
        # # performance_threshold = np.quantile(cluster_hm_performances[[f'{i}_ratio' for i in valid_performance]], 0.25)
        # # TODO add max count threshold to prevent crash
        # performance_threshold = np.quantile(cluster_hm_performances[[f'{i}_ratio' for i in valid_performance]], 0.5, axis=0)
        # cluster_hm_performances[[f'{i}_bool' for i in valid_performance]] = cluster_hm_performances[[f'{i}_ratio' for i in valid_performance]] >= performance_threshold
        # cluster_hm_performances['sum'] = cluster_hm_performances[[f'{i}_bool' for i in valid_performance]].all(axis=1)
        # # print(f"Cluster {cluster} thld: {np.round(performance_threshold,2)} {cluster_hm_performances[cluster_hm_performances['sum']].index.values}")
        # above_threshold = np.array([cluster_hm_performances.loc[hm]['sum'] if hm in cluster_hm_performances.index else False for hm in hydrological_models])

        # Filter indices for sim above threshold
        indices_mask = above_threshold[indices_cluster]

        above_threshold_array[indices_cluster] = indices_mask

        if len(indices_mask) > 0:
            indices_cluster = indices_cluster[indices_mask]

        # Get vector of these sims
        X_cluster = X_imputed[indices_cluster, :]

        # Compute distance from cluster centroid
        distances_cluster = np.linalg.norm(X_cluster[indices_cluster, :] - centroids[cluster], axis=1)

        # Distance max from current centroid
        distance_max = np.median(distances_cluster) + 2 * np.std(distances_cluster)
        sum_cluster_distance_treshold = np.sum(distances_cluster <= distance_max, axis=0)
        mask_cluster = (sum_cluster_distance_treshold == max(sum_cluster_distance_treshold))
        # ds_stacked['hm'].isel(sample=indices_cluster).values

        # Compute distance from other cluster centroids
        distances_list = []
        for c in cluster_id:
            if c != cluster:
                distances_list.append(np.linalg.norm(X_cluster - centroids[c], axis=1))
        distances_other = np.min(distances_list, axis=0)

        if np.any(distances_other) and np.any(mask_cluster):
            idx = indices_cluster[mask_cluster][np.argmax(distances_other[mask_cluster])]

        if idx is not None:

            # Extract coordinate (gcm-rcm, bc, hm) of selected sim
            coords_gcm_rcm = ds_stacked["gcm-rcm"].isel(sample=idx).values
            coords_bc      = ds_stacked["bc"].isel(sample=idx).values
            coords_hm      = ds_stacked["hm"].isel(sample=idx).values

            # Save result in dict
            representative_groups[cluster] = {
                "gcm-rcm": coords_gcm_rcm,
                "bc": coords_bc,
                "hm": coords_hm,
                "idx": idx,
                "color": hex_colors[cluster],
                "name": cluster_names[cluster]
            }

    narratives |= {horizon_ref : {f"{value['gcm-rcm']}_{value['bc']}_{value['hm']}": {'color': value['color'], 'zorder': 10,
    'label': f"{value['name'].title()}",
    'linewidth': 2} for key, value in rp.items()} for i, rp in enumerate(meth_list)}

    # PCA for 2D visualisation
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_imputed)

    # Transform centroids to pca space
    centroids_pca = pca.transform(kmeans.cluster_centers_)

    # Get pca ratio
    ratio1, ratio2 = pca.explained_variance_ratio_

    # Compute pca var contribution
    loadings = pca.components_
    pc1_contributions = np.abs(loadings[0]) / np.sum(np.abs(loadings[0]))
    pc2_contributions = np.abs(loadings[1]) / np.sum(np.abs(loadings[1]))

    # Plot for PCA
    print(f">> Narrative PCA Plot [{horizon_ref}]", end='\n')
    # x1 = (', ').join([f"{indicator_values[var_idx]}q{q}: {pc1_contributions[var_idx]:.1%}" for q in quantiles for var_idx, var in enumerate(indicator_values)])
    # xlabel = f"Dim 1 {ratio1:.1%}: [{x1}]"
    # y1 = (', ').join([f"{indicator_values[var_idx]}q{q}: {pc2_contributions[var_idx]:.1%}" for q in quantiles for var_idx, var in enumerate(indicator_values)])
    # ylabel = f"Dim 2 {ratio2:.1%}: [{y1}]"
    xlabel = f"Dim 1 {ratio1:.1%} ({indicator_values[0]}: {pc1_contributions[0]:.1%}, {indicator_values[1]}: {pc1_contributions[1]:.1%}, {indicator_values[2]}: {pc1_contributions[2]:.1%})"
    ylabel = f"Dim 2 {ratio2:.1%} ({indicator_values[0]}: {pc2_contributions[0]:.1%}, \n{indicator_values[1]}: {pc2_contributions[1]:.1%}, {indicator_values[2]}: {pc2_contributions[2]:.1%})"
    title = "Clusters et points représentatifs (après PCA)"
    path_result = f"{path_figures}narratest_pca-comparatives_{horizon_ref}_{str_quantiles}.pdf"
    plot_narratives(X_pca, ds_stacked, meth_list, labels, cluster_names,
                    path_result, xlabel, ylabel, title=None, centroids=centroids_pca, count_stations=None,
                    above_threshold=above_threshold_array, palette=hex_colors, n=4, rows=rows,
                    cols=None)

    # Plot comparison with every indicator

    print(f">> Narrative every indicators Plot {indicator_values} [{horizon_ref}]", end='\n')
    path_result = f"{path_figures}narratest_indicator-comparatives_{horizon_ref}_{str_quantiles}.pdf"
    plot_narratives(X_imputed, ds_stacked, meth_list, labels, cluster_names,
                    path_result, xlabel, ylabel, title, centroids=None, count_stations=None,
                    above_threshold=above_threshold_array, palette=hex_colors, n=4, rows=rows,
                    cols=indicator_values)

    # # PLOT BY INDICATOR
    # for idx1 in range(len(indicator_values)):
    #     if idx1 != len(indicator_values)-1:
    #         idx2 =idx1+1
    #     else:
    #         idx2 = 0

    #     # Construire les noms des axes
    #     if indicator_values[idx2] == 'QA':
    #         idx1, idx2 = idx2, idx1
    #     print(f"Narrative Plot {indicator_values[idx1]} & {indicator_values[idx2]}")
    #     xlabel = f"Variation {indicator_values[idx1]} (%)"
    #     ylabel = f"Variation {indicator_values[idx2]} (%)"
    #     title = "Clusters et points représentatifs"
    #     path_result=f"/home/bcalmel/Documents/3_results/narratest_{indicator_values[idx1]}-{indicator_values[idx2]}.pdf"
    #     # path_result=f"/home/bcalmel/Documents/3_results/narratest_spatial_mean_comparison.pdf"

    #     plot_narratives(X_imputed[:, [idx1, idx2]], ds_stacked, meth_list, labels, cluster_names,
    #                     path_result, xlabel, ylabel, title=None, centroids=None, count_stations=None,
    #                     above_threshold=above_threshold, palette=hex_colors, n=4, rows=rows)
    print(f">> Save narratives as .json", end='\n')
    with open(path_narratives, "w", encoding="utf-8") as f:
        json.dump(narratives, f, ensure_ascii=False, indent=4)


