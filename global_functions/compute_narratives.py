import matplotlib.lines as mlines
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d
import pandas as pd


def compute_narratives(datasets_list, sim_points_gdf, data_type, variables, plot_type, title_join,plot_type_name):
    # ds = ds_stats
    sim_points_gdf = sim_points_gdf_simplified

    # stations = list(sim_points_gdf.index)
    stations = list(reference_stations['La Loire'].keys())
    # Sélectionner les stations avec le plus de simulations #### ATTENTION SPATIALISATION
    # n_values = sim_points_gdf_simplified['n']
    # stations = list(n_values[n_values == max(n_values)].index)
    ind_values = ["QJXA", "QA", "VCN10"]

    data_arrays = []
    datasets = [ds_i[variables[f'simulation-horizon_by-sims_{plot_type}']].sel(
        horizon='horizon3', gid=stations) for ds_i in datasets_list]
    indicator_names = list(data_to_plot.keys())
    for i in range(len(datasets)):
        ds = datasets[i]
        for var_name, da in ds.data_vars.items():
            # 1. Extraire les 4 parties du nom en se basant sur le séparateur "_" (adapté si besoin)
            parts = var_name.split("_")
            nom_gcmrcm = "_".join(parts[:2])
            nom_bc, nom_hm = parts[2:4]

            # 2. Ajouter 4 dimensions à la DataArray, chacune de taille 1 avec la valeur extraite
            da_expanded = da.expand_dims({
                # "indicator": [indicator_names[i]],
                "gcm-rcm": [nom_gcmrcm],
                # "rcm": [nom_rcm],
                "bc":  [nom_bc],
                "hm":  [nom_hm]
            })

            # 3. Donner le même nom à toutes les DataArrays pour qu'elles soient fusionnées en une seule variable.
            da_expanded.name = indicator_names[i]

            data_arrays.append(da_expanded)

    # 4. Fusionner toutes les DataArrays en une seule DataArray
    # Le résultat aura les dimensions : ('gid', 'nom-gcm', 'nom-rcm', 'nom-bc', 'nom-hm')
    combined_da = xr.combine_by_coords(data_arrays)
    count_stations = combined_da[["QA"]].count(dim="gid")['QA'].values.flatten()

    # Calculer la moyenne (sur le territoire) par chaine de simulation
    combined_da = combined_da.mean(dim='gid')

    # combined_da = combined_da.dropna(dim="gid", how="any")
    # print("Stations conservées :", combined_da.gid.values)

    # 1. Aplatir le dataset : on combine les dimensions pour obtenir un indice unique "sample"
    ds_stacked = combined_da.stack(sample=("gcm-rcm", "bc", "hm"))
    hm_list = list(ds_stacked.hm.values)

    # 2. Construire la matrice X des features
    # Chaque colonne correspond à une des variables et chaque ligne à un échantillon
    X_imputed = np.column_stack([ds_stacked[var].values for var in ind_values])

    # 3 colonnes QJXA QA VCN10, 1683 lignes 11 stations * 153 sim (9 HM * 17 GCM-RCM)

    # 3. Imputation des valeurs manquantes (NaN) par la moyenne de la colonne
    # from sklearn.impute import SimpleImputer
    # imputer = SimpleImputer(strategy="mean")
    # X_imputed = imputer.fit_transform(X)

    # 4. Appliquer le clustering (ici KMeans avec 4 clusters)
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X_imputed)
    print("Labels obtenus:", np.unique(labels))

    # 5. Remettre les labels dans une DataArray avec la dimension "sample"
    labels_da = xr.DataArray(labels, dims="sample", coords={"sample": ds_stacked.sample})

    # 6. Optionnel : déplier (unstack) pour retrouver les dimensions d'origine
    # labels_unstacked = labels_da.unstack("sample")

    # Vous pouvez ajouter les labels comme nouvelle variable dans votre dataset original
    # ds_clustered = combined_da.assign(cluster=labels_unstacked)

    # Récupération des centroïdes du clustering (lorsque KMeans a été appliqué sur X_imputed)
    centroïdes = kmeans.cluster_centers_  # de forme (n_clusters, n_features)

    # Dictionnaire pour stocker, pour chaque cluster, le groupe (gcm-rcm, bc, hm) le plus proche du centroïde
    groupes_representatifs = {}

    # Pour chaque cluster, on parcourt les observations qui lui appartiennent
    # TODO Filter les sim par nombre de stations parmi la sélection
    threshold = 0.8*len(stations)
    above_threshold = count_stations > threshold
    for cluster in np.unique(labels):
        # Indices des observations appartenant au cluster 'cluster'
        indices_cluster = np.where(labels == cluster)[0]

        # Filter indices for sim which see at least 80% of stations
        indices_mask = above_threshold[indices_cluster]
        if len(indices_mask) > 0:
            indices_cluster = indices_cluster[indices_mask]

        # Récupérer les vecteurs de caractéristiques de ces observations
        X_cluster = X_imputed[indices_cluster, :]  # de forme (nombre_d'observations_dans_le_cluster, n_features)

        # Calculer la distance euclidienne entre chaque vecteur et le centroïde du cluster
        distances = np.linalg.norm(X_cluster - centroïdes[cluster], axis=1)

        # Identifier l'indice (dans X_cluster) de l'observation la plus proche du centroïde
        idx_min = indices_cluster[np.argmin(distances)]

        # Extraire les coordonnées (gcm-rcm, bc, hm) de l'observation sélectionnée.
        # On suppose ici que vous avez un DataArray "ds_stacked" obtenu via un stacking sur les dimensions ("gcm-rcm", "bc", "hm", "gid").
        # Vous pouvez adapter en fonction de vos noms de dimensions.
        coords_gcm_rcm = ds_stacked["gcm-rcm"].isel(sample=idx_min).values
        coords_bc      = ds_stacked["bc"].isel(sample=idx_min).values
        coords_hm      = ds_stacked["hm"].isel(sample=idx_min).values

        # On peut par exemple stocker ces informations sous forme de dictionnaire
        groupes_representatifs[cluster] = {
            "gcm-rcm": coords_gcm_rcm,
            "bc": coords_bc,
            "hm": coords_hm,
            "distance": distances[np.argmin(distances)],
            "idx": idx_min
        }

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # ---------------------------
    # Supposons que vous avez déjà :
    # - X_imputed : votre matrice de features (shape: n_samples x 3)
    # - labels      : le résultat du clustering (un label par observation)
    # - kmeans      : l'objet KMeans entraîné, avec kmeans.cluster_centers_
    # - ds_stacked  : votre DataArray "aplatit" avec la dimension "sample"
    #                issue d'un stacking des dimensions ("gcm-rcm", "bc", "hm", "gid")
    # - Un dictionnaire groupes_representatifs avec pour chaque cluster son indice représentatif dans ds_stacked
    #   Par exemple, vous l'avez construit ainsi :
    #
    # groupes_representatifs = {}
    # for cluster in np.unique(labels):
    #     indices_cluster = np.where(labels == cluster)[0]
    #     distances = np.linalg.norm(X_imputed[indices_cluster] - kmeans.cluster_centers_[cluster], axis=1)
    #     idx_min = indices_cluster[np.argmin(distances)]
    #     groupes_representatifs[cluster] = idx_min
    # ---------------------------
    cluster_names = ['A', 'B', 'C', 'D']
    shape_hp = {
        'CTRIP': 'D',
        'EROS': '+',
        'GRSD': '*',
        'J2000': 's',
        'MORDOR-SD': 'v',
        'MORDOR-TS': '^',
        'ORCHIDEE': 'o',
        'SIM2': '>',
        'SMASH': '<',
    }
    shape_list = [shape_hp[hm_value] for hm_value in hm_list]

    # Réduction de dimension avec PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_imputed)

    # Transformation des centroïdes dans l'espace PCA
    centroids_pca = pca.transform(kmeans.cluster_centers_)

    # Exprimer la contribution de chaque variable sous forme de pourcentage (valeur absolue normalisée)
    loadings = pca.components_
    variables = ["QJXA", "QA", "VCN10"]
    pc1_contributions = np.abs(loadings[0]) / np.sum(np.abs(loadings[0]))
    pc2_contributions = np.abs(loadings[1]) / np.sum(np.abs(loadings[1]))

    # Construire les noms des axes
    ratio1, ratio2 = pca.explained_variance_ratio_


    def plot_narratives(x_data, ds_stacked, path_result, xlabel, ylabel, title, centroids=None, count_stations=None,
                        above_threshold=None, palette='viridis', n=4):
        # Affichage des points représentatifs de chaque cluster
        cmap = plt.get_cmap(palette, n)
        norm = plt.Normalize(vmin=-0.5, vmax=n-0.5)  # Normalisation entre 0 et 3

        # Préparez la figure
        # plt.figure(figsize=(10, 8))
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.grid(alpha=0.3)

        # Affichage des observations, colorées par cluster
        for j in range(len(x_data)):
            alpha=0.7
            if above_threshold is not None and not above_threshold[j]:
                alpha=0.2
            scatter = plt.scatter(x_data[j, 0], x_data[j, 1], c=labels[j], cmap=cmap, norm=norm, alpha=alpha,
                                  marker=shape_list[j], zorder=1)
            if count_stations is not None:
                plt.annotate(count_stations[j], (x_data[j, 0], x_data[j, 1]),
                             fontsize=9, color='black', zorder=13)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        # Affichage des centroïdes
        representative_legends = []
        if centroids is not None:
            representative_legends.append(plt.scatter(centroids[:, 0], centroids[:, 1],
                        marker='X', c='red', s=100, label='Centroïdes', zorder=9))

        # Affichage des points représentatifs de chaque cluster

        for cluster, values in groupes_representatifs.items():
            idx = values['idx']

            # Récupérer les coordonnées d'origine (gcm-rcm, bc, hm)
            coord_gcm_rcm = ds_stacked["gcm-rcm"].isel(sample=idx).values
            coord_bc      = ds_stacked["bc"].isel(sample=idx).values
            coord_hm      = ds_stacked["hm"].isel(sample=idx).values

            # Annoter le graphique avec ces coordonnées
            annotation = f"{coord_gcm_rcm}\n{coord_bc}\n{coord_hm}"
            # Marquer le point représentatif dans l'espace PCA
            representative_legends.append(plt.scatter(x_data[idx, 0], x_data[idx, 1], c=cluster,
                                                      edgecolors='k', s=150, marker=shape_list[idx],
                                                      cmap=cmap, norm=norm, zorder=10, label=annotation))
            # plt.annotate(annotation, (x_data[idx, 0], x_data[idx, 1]), textcoords="offset points", xytext=(5,5), fontsize=9, color='black',
            #              weight='bold', zorder=11)

        # Add line to zero
        xmin, ymin = np.min(x_data, axis=0)
        xmax, ymax = np.max(x_data, axis=0)
        min_val = np.min([xmin, ymin])*0.9
        max_val = np.max([xmax, ymax])*1.1
        plt.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3, zorder=0)
        plt.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.3, zorder=0)
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        ax.set_aspect('equal')

        # Ajout d'une légende et d'une barre de couleur
        hm_legends = []
        for hm, shape in shape_hp.items():
            hm_legends.append(plt.scatter(np.nan, np.nan, alpha=0.6, marker=shape, label=hm, color='k'))

        legend1 = plt.legend(handles=hm_legends, loc='center left', bbox_to_anchor=(1.2, 0.82),
                   title="Modèles Hydro", ncol=2)
        ax.add_artist(legend1)
        plt.legend(handles=representative_legends, loc='center left', bbox_to_anchor=(1.2, 0.3),
                   title="Narratifs", ncol=1)
        cbar = plt.colorbar(scatter, fraction=0.03, pad=0.02)
        cbar.set_label("Cluster")
        cbar.set_ticks([0,1,2,3])
        cbar.set_ticklabels(cluster_names)
        cbar.ax.invert_yaxis()
        # cbar.set_ticks(cbar.get_ticks()[::-1])  # Inverser les ticks de la colorbar
        # cbar.set_ticklabels(cluster_names[::-1])

        plt.savefig(path_result, bbox_inches='tight')

    # Plot for PCA
    xlabel = f"Dim 1 {ratio1:.1%} ({variables[0]}: {pc1_contributions[0]:.1%}, {variables[1]}: {pc1_contributions[1]:.1%}, {variables[2]}: {pc1_contributions[2]:.1%})"
    ylabel = f"Dim 2 {ratio2:.1%} ({variables[0]}: {pc2_contributions[0]:.1%}, {variables[1]}: {pc2_contributions[1]:.1%}, {variables[2]}: {pc2_contributions[2]:.1%})"
    title = "Clusters et points représentatifs (après PCA)"
    path_result = f"/home/bcalmel/Documents/3_results/narratest_closest_pca_spatial_mean_centroides.pdf"
    plot_narratives(X_pca, ds_stacked, path_result, xlabel, ylabel, title, centroids=centroids_pca,
                    count_stations=None, above_threshold=above_threshold, palette='Dark2')

    # PLOT BY INDICATOR
    for idx1 in range(len(ind_values)):
        if idx1 != len(ind_values)-1:
            idx2 =idx1+1
        else:
            idx2 = 0

        # Construire les noms des axes
        if ind_values[idx2] == 'QA':
            idx1, idx2 = idx2, idx1
        xlabel = f"Variation {ind_values[idx1]} (%)"
        ylabel = f"Variation {ind_values[idx2]} (%)"
        title = "Clusters et points représentatifs"
        path_result=f"/home/bcalmel/Documents/3_results/narratest_closest_{ind_values[idx1]}_{ind_values[idx2]}_spatial_mean_centroides.pdf"

        plot_narratives(X_imputed[:, [idx1, idx2]], ds_stacked, path_result, xlabel, ylabel, title, centroids=None,
                        count_stations=None, above_threshold=above_threshold, palette='Dark2')
