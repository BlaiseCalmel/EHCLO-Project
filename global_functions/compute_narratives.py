from sndhdr import whathdr

from sklearn.cluster import KMeans
import xarray as xr
import numpy as np
from sklearn.decomposition import PCA
from global_functions.format_data import format_dataset
from plot_functions.plot_narratives import plot_narratives

def compute_narratives(dict_paths, stations, files_setup,
                       indictor_values=["QJXA", "QA", "VCN10"], threshold=0):

    # Load selected indicators
    datasets_list = []
    for indicator in indictor_values:
        # Open ncdf dataset
        path_ncdf = f"{dict_paths['folder_study_data']}{indicator}_rcp85_YE.nc"
        ds_stats  = xr.open_dataset(path_ncdf)

        # Compute stats
        ds_stats, var_names = format_dataset(ds=ds_stats, data_type='hydro', files_setup=files_setup)
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

    # Dict to find the closest sim from the centroid (gcm-rcm, bc, hm)
    representative_groups = {}

    # Create mask for sim above threshold
    above_threshold = count_stations > threshold
    # Run on each cluster
    for cluster in np.unique(labels):
        # Index of cluster values
        indices_cluster = np.where(labels == cluster)[0]

        # Filter indices for sim above threshold
        indices_mask = above_threshold[indices_cluster]
        if len(indices_mask) > 0:
            indices_cluster = indices_cluster[indices_mask]

        # Get vector of these sims
        X_cluster = X_imputed[indices_cluster, :]  # de forme (nombre_d'observations_dans_le_cluster, n_features)

        # Compute distance from cluster centroid
        distances = np.linalg.norm(X_cluster - centroids[cluster], axis=1)

        # Get index of the closest sim
        idx_min = indices_cluster[np.argmin(distances)]

        # Extract coordinate (gcm-rcm, bc, hm) of selected sim
        coords_gcm_rcm = ds_stacked["gcm-rcm"].isel(sample=idx_min).values
        coords_bc      = ds_stacked["bc"].isel(sample=idx_min).values
        coords_hm      = ds_stacked["hm"].isel(sample=idx_min).values

        # Save result in dict
        representative_groups[cluster] = {
            "gcm-rcm": coords_gcm_rcm,
            "bc": coords_bc,
            "hm": coords_hm,
            "distance": distances[np.argmin(distances)],
            "idx": idx_min
        }

    cluster_names = ['A', 'B', 'C', 'D']

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
    ylabel = f"Dim 2 {ratio2:.1%} ({indictor_values[0]}: {pc2_contributions[0]:.1%}, {indictor_values[1]}: {pc2_contributions[1]:.1%}, {indictor_values[2]}: {pc2_contributions[2]:.1%})"
    title = "Clusters et points représentatifs (après PCA)"
    path_result = f"/home/bcalmel/Documents/3_results/narratest_closest_pca_spatial_mean_centroides.pdf"
    plot_narratives(X_pca, ds_stacked, representative_groups, labels, cluster_names,
                    path_result, xlabel, ylabel, title, centroids=centroids_pca, count_stations=None,
                    above_threshold=above_threshold, palette='Dark2', n=4)

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
        path_result=f"/home/bcalmel/Documents/3_results/narratest_closest_{indictor_values[idx1]}_{indictor_values[idx2]}_spatial_mean_centroides.pdf"

        plot_narratives(X_imputed[:, [idx1, idx2]], ds_stacked, representative_groups, labels, cluster_names,
                        path_result, xlabel, ylabel, title, centroids=None, count_stations=None,
                        above_threshold=above_threshold, palette='Dark2', n=4)

        return representative_groups

