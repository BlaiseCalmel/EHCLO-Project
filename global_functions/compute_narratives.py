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
import xarray as xr
import numpy as np
from sklearn.decomposition import PCA
from global_functions.format_data import format_dataset
from plot_functions.plot_narratives import plot_narratives
from global_functions.load_data import open_fst

def representative_item(X_cluster, centroids, cluster, cluster_id, indices_cluster, method='closest',
                        weight=None, distance_max=None):
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
        mask_cluster = distances_cluster < distance_max
        # mask_cluster = np.tile(True, distances_cluster.shape)

        # Compute distance from other cluster centroids
        distances_list = []
        for c in cluster_id:
            if c != cluster:
                distances_list.append(np.linalg.norm(X_cluster - centroids[c], axis=1) * weight) 
        distances_other = np.mean(distances_list, axis=0)
        if np.any(distances_other):
            idx = indices_cluster[mask_cluster][np.argmax(distances_other[mask_cluster])]

    return idx


def compute_narratives(dict_paths, stations, files_setup, data_shp,
                       indictor_values=["QJXA", "QA", "VCN10"], threshold=0, narrative_method='closest'):

    # Load selected indicators
    datasets_list = []
    for indicator in indictor_values:
        # Open ncdf dataset
        path_ncdf = f"{dict_paths['folder_study_data']}{indicator}_rcp85_YE_TRACC.nc"
        ds_stats  = xr.open_dataset(path_ncdf)

        # Compute stats
        ds_stats, var_names = format_dataset(ds=ds_stats, data_type='hydro', files_setup=files_setup)
        ds_stats['gid'] = ds_stats['gid'].astype(str)
        datasets_list.append(ds_stats)

    data_arrays = []
    datasets = [ds_i[var_names[f'simulation-horizon_by-sims_deviation']].sel(
        horizon='horizon3', gid=stations) for ds_i in datasets_list]
    for i in range(len(datasets)):
        ds = datasets[i]
        for var_name, da in ds.data_vars.items():
            # Extract names part
            parts = var_name.split("_")
            nom_gcmrcm = "_".join(parts[:2])
            nom_bc, nom_hm = parts[2:4]

            # Generate new DataArray with sim as dimension
            da_expanded = da.expand_dims({
                # "indicator": [indicator_names[i]],
                "gcm-rcm": [nom_gcmrcm],
                # "rcm": [nom_rcm],
                "bc":  [nom_bc],
                "hm":  [nom_hm]
            })

            # Get name of the current indicator as var name
            da_expanded.name = indictor_values[i]

            data_arrays.append(da_expanded)

    # Combine DataArrays
    combined_da = xr.combine_by_coords(data_arrays)

    # Count stations per sim
    count_stations = combined_da[["QA"]].count(dim="gid")['QA'].values.flatten()

    # Compute mean on selected stations
    combined_da = combined_da.mean(dim='gid')

    # # Weighted mean by cumulative distance between station
    # gdf = data_shp.loc[stations]
    # gdf["sum_distance"] = gdf.geometry.apply(lambda p: gdf.distance(p).sum())
    # gdf["sum_distance"] = gdf["sum_distance"] / gdf["sum_distance"].mean()
    #
    # combined_da = combined_da.assign_coords(weights=("gid", gdf.reindex(ds["gid"].values)["sum_distance"].values))
    # combined_da = combined_da.weighted(combined_da["weights"]).mean(dim="gid")

    # Flatten dataset and generate new coordinate named "sample"
    ds_stacked = combined_da.stack(sample=("gcm-rcm", "bc", "hm"))

    # Generate matrix
    X_imputed = np.column_stack([ds_stacked[var].values for var in indictor_values])

    # KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X_imputed)

    # # Add labels to DataArray with sample dimension
    # labels_da = xr.DataArray(labels, dims="sample", coords={"sample": ds_stacked.sample})

    # # Unstack to same dimension as origin DataArray
    # labels_unstacked = labels_da.unstack("sample")

    # # Add labels as a new variable
    # ds_clustered = combined_da.assign(cluster=labels_unstacked)

    # Find centroids
    centroids = kmeans.cluster_centers_  # de forme (n_clusters, n_features)

    # Create mask for sim above threshold
    # above_threshold = count_stations > threshold
    # Run on each cluster
    cluster_id = np.unique(labels)
    # Cluster info
    # colors = plt.get_cmap("Dark2", 4).colors
    # hex_colors = [mcolors.to_hex(c) for c in colors]

    # # Load hm performances .fst files
    # # Run once to install the related R packages
    # utils = importr('utils')
    # utils.install_packages('fst')
    # utils.install_packages('data.table')

    performances = ['Biais', 'Q10', 'Q90']
    # Baseflow performance
    df = open_fst(f"/home/bcalmel/Documents/2_data/hydrologie/dataEX_Explore2_criteria_diagnostic_performance.fst")
    baseflow_criteria = df[['HM', 'code','Biais']]

    # Highflow performance
    df = open_fst(f"/home/bcalmel/Documents/2_data/hydrologie/dataEX_Explore2_criteria_diagnostic_HF.fst")
    highflow_criteria = df[['HM', 'code','Q10']]

    # Lowflow performance
    df = open_fst(f"/home/bcalmel/Documents/2_data/hydrologie/dataEX_Explore2_criteria_diagnostic_LF.fst")
    lowflow_criteria = df[['HM', 'code','Q90']]

    merged_performances = pd.merge(pd.merge(baseflow_criteria, lowflow_criteria, on=['HM', 'code']), 
                                   highflow_criteria, on=['HM', 'code'])
    
    # Define threshold for each criteria
    performance_thresholds = {'Biais': 0.2, 'Q10': 0.2, 'Q90': 0.8}

    # Get selected stations
    merged_performances = merged_performances[merged_performances['code'].isin(data_shp.index)]
    count_stations = len(np.unique(merged_performances['code']))

    # Compare to thresholds
    merged_performances[performances] = abs(merged_performances[performances])
    valid_performance = []
    for key, value in performance_thresholds.items():
        merged_performances[key+'_valid'] = merged_performances[key] < value
        valid_performance.append(key+'_valid')

    # Compute percentage of station above threshold for each HM
    hm_performances = merged_performances.groupby(['HM']).agg({v: 'sum' for v in valid_performance})
    # hm_performances['sum'] = hm_performances[valid_performance].sum(axis=1)
    hm_count_stations = merged_performances.groupby(['HM']).size()
    # hm_performances[[f'{i}_ratio' for i in valid_performance]] = hm_performances[valid_performance].div(
    #     hm_performances.index.map(hm_count_stations), axis=0) >= threshold
    
    hm_performances[[f'{i}_ratio' for i in valid_performance]] = hm_performances[valid_performance].div(
        count_stations, axis=0) >= threshold
    hm_performances['sum'] = hm_performances[[f'{i}_ratio' for i in valid_performance]].all(axis=1)
    
    hydrological_models = ds_stacked["hm"].values
    above_threshold = np.array([hm_performances.loc[hm]['sum'] for hm in hydrological_models])

    cluster_names = ['A', 'B', 'C', 'D']

    # Rank clusters
    # ranks = np.argsort(np.argsort(centroids, axis=0), axis=0) + 1
    # cumulative_ranks = ranks[:, 0] + ranks[:, 2]
    # mask = np.ones(ranks.shape, dtype=bool)
    #
    # extreme = np.argmin(cumulative_ranks)
    # mask[extreme, :] = False
    # dry = np.argmin(np.where(mask, ranks, np.inf)[:, 2])
    # mask[dry, :] = False
    # flood = np.argmax(np.where(mask, ranks, -np.inf)[:, 0])
    # mask[flood, :] = False
    # last = np.argmin(np.where(mask, ranks, np.inf)[:, 2])
    #
    # narra_idx = [flood, dry, last, extreme]
    # narra_idx = [1, 2, 0, 3]
    hex_colors = ["#016367", "#9E3A14", "#E66912", "#0B1C48"]
    # Bleunavy Orange Brun Turquoise https://www.canva.com/colors/color-palettes/freshly-sliced-fruit/
    # hex_colors = [hex_colors[i] for i in narra_idx]

    rows = None
    if narrative_method is None:
        methods = ['closest', 'furthest', 'combine']
        rows = ['Proche', 'Lointain', 'Mixte']
    else:
        if isinstance(narrative_method, str):
            methods = [narrative_method]
        else:
            methods = narrative_method
    meth_list = []
    for narrative_method in methods:
        representative_groups = {}
        for cluster in cluster_id:
            # Index of cluster values
            indices_cluster = np.where(labels == cluster)[0]

            distance_max = 1.5 * np.mean(np.linalg.norm(X_imputed[indices_cluster, :] - centroids[cluster], axis=1))

            # Filter indices for sim above threshold
            indices_mask = above_threshold[indices_cluster]
            if len(indices_mask) > 0:
                indices_cluster = indices_cluster[indices_mask]

            # Get vector of these sims
            X_cluster = X_imputed[indices_cluster, :]

            idx = representative_item(
                X_cluster, centroids, cluster, cluster_id, indices_cluster, method=narrative_method, distance_max=distance_max)
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
                    # "distance": distances[np.argmin(distances)],
                    "idx": idx,
                    "color": hex_colors[cluster],
                    "name": cluster_names[cluster],
                    "method": narrative_method
                }
        meth_list.append(representative_groups)

    narratives = {methods[i] : {f"{value['gcm-rcm']}_{value['bc']}_{value['hm']}": {'color': value['color'], 'zorder': 10,
    'label': f"{value['name'].title()}", # [{value['gcm-rcm']}_{value['bc']}_{value['hm']}]",
    'linewidth': 1} for key, value in rp.items()} for i, rp in enumerate(meth_list)}
                                      
    # # Generate dataframe to export
    # df = ds_stacked.to_dataframe()
    # df = df.reset_index(drop=True)
    # df["cluster"] = [cluster_names[i] for i in labels]
    # for key in narratives.keys():
    #     df[key] = None
    #     for id_row, row in df.iterrows():
    #         # print(row)
    #         name = f"{row.loc['gcm-rcm']}_{row['bc']}_{row['hm']}"
    #         if name in narratives[key].keys():
    #             print()
    #             df.loc[id_row, key] = narratives[key][name]['label']
    # df.to_csv(f"/home/bcalmel/Documents/3_results/test.csv", sep=";")

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
    print(f"Narrative PCA Plot")
    xlabel = f"Dim 1 {ratio1:.1%} ({indictor_values[0]}: {pc1_contributions[0]:.1%}, {indictor_values[1]}: {pc1_contributions[1]:.1%}, {indictor_values[2]}: {pc1_contributions[2]:.1%})"
    ylabel = f"Dim 2 {ratio2:.1%} ({indictor_values[0]}: {pc2_contributions[0]:.1%}, \n{indictor_values[1]}: {pc2_contributions[1]:.1%}, {indictor_values[2]}: {pc2_contributions[2]:.1%})"
    title = "Clusters et points représentatifs (après PCA)"
    path_result = f"/home/bcalmel/Documents/3_results/narratest_pca_comparatives.pdf"
    plot_narratives(X_pca, ds_stacked, meth_list, labels, cluster_names,
                    path_result, xlabel, ylabel, title=None, centroids=centroids_pca, count_stations=None,
                    above_threshold=above_threshold, palette=hex_colors, n=4, rows=rows,
                    cols=None)

    # Plot comparison with every indicator
    print(f"Narrative every indicators Plot {indictor_values}")
    path_result = f"/home/bcalmel/Documents/3_results/narratest_spatial_mean_comparatives.pdf"
    plot_narratives(X_imputed, ds_stacked, meth_list, labels, cluster_names,
                    path_result, xlabel, ylabel, title, centroids=None, count_stations=None,
                    above_threshold=above_threshold, palette=hex_colors, n=4, rows=rows,
                    cols=indictor_values)

    # PLOT BY INDICATOR
    for idx1 in range(len(indictor_values)):
        if idx1 != len(indictor_values)-1:
            idx2 =idx1+1
        else:
            idx2 = 0

        # Construire les noms des axes
        if indictor_values[idx2] == 'QA':
            idx1, idx2 = idx2, idx1
        print(f"Narrative Plot {indictor_values[idx1]} & {indictor_values[idx2]}")
        xlabel = f"Variation {indictor_values[idx1]} (%)"
        ylabel = f"Variation {indictor_values[idx2]} (%)"
        title = "Clusters et points représentatifs"
        path_result=f"/home/bcalmel/Documents/3_results/narratest_{indictor_values[idx1]}-{indictor_values[idx2]}.pdf"
        # path_result=f"/home/bcalmel/Documents/3_results/narratest_spatial_mean_comparison.pdf"

        plot_narratives(X_imputed[:, [idx1, idx2]], ds_stacked, meth_list, labels, cluster_names,
                        path_result, xlabel, ylabel, title=None, centroids=None, count_stations=None,
                        above_threshold=above_threshold, palette=hex_colors, n=4, rows=rows)

    return narratives

