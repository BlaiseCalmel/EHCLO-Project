import matplotlib.lines as mlines
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from format_data import compute_mean_by_horizon
from main import data_to_plot


def compute_narratives(datasets, sim_points_gdf, data_type, variables, plot_type, title_join,plot_type_name):
    # ds = ds_stats
    # sim_points_gdf = sim_points_gdf_simplified

    stations = list(sim_points_gdf.index)


    data_arrays = []
    datasets = [ds_i[variables[f'simulation-horizon_by-sims_{plot_type}']].sel(
        horizon='horizon3', gid=list(stations.keys())) for ds_i in datasets]
    indicator_names = list(data_to_plot.keys())
    for i in range(len(datasets)):
        ds = datasets[i]
        for var_name, da in ds.data_vars.items():
            # 1. Extraire les 4 parties du nom en se basant sur le séparateur "_" (adapté si besoin)
            parts = var_name.split("_")
            # if len(parts) != 4:
            #     raise ValueError(f"Le nom de variable '{var_name}' ne respecte pas le format attendu")
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

    # 1. Aplatir le dataset : on combine les dimensions pour obtenir un indice unique "sample"
    ds_stacked = combined_da.stack(sample=("gcm-rcm", "bc", "hm", "gid"))

    # 2. Construire la matrice X des features
    # Chaque colonne correspond à une des variables et chaque ligne à un échantillon
    X = np.column_stack([ds_stacked[var].values for var in ["QJXA", "QA", "VCN10"]])
    print("Shape de X:", X.shape)

    # 3. Imputation des valeurs manquantes (NaN) par la moyenne de la colonne
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # 4. Appliquer le clustering (ici KMeans avec 4 clusters)
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X_imputed)
    print("Labels obtenus:", np.unique(labels))

    # 5. Remettre les labels dans une DataArray avec la dimension "sample"
    labels_da = xr.DataArray(labels, dims="sample", coords={"sample": ds_stacked.sample})

    # 6. Optionnel : déplier (unstack) pour retrouver les dimensions d'origine
    labels_unstacked = labels_da.unstack("sample")

    # Vous pouvez ajouter les labels comme nouvelle variable dans votre dataset original
    ds_clustered = combined_da.assign(cluster=labels_unstacked)

    # Récupération des centroïdes du clustering (lorsque KMeans a été appliqué sur X_imputed)
    centroïdes = kmeans.cluster_centers_  # de forme (n_clusters, n_features)

    # Dictionnaire pour stocker, pour chaque cluster, le groupe (gcm-rcm, bc, hm) le plus proche du centroïde
    groupes_representatifs = {}

    # Pour chaque cluster, on parcourt les observations qui lui appartiennent
    for cluster in np.unique(labels):
        # Indices des observations appartenant au cluster 'cluster'
        indices_cluster = np.where(labels == cluster)[0]

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
            "idx": idx_min  # optionnel : la distance minimale
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
    xlabel = f"Dim 1 ({variables[0]}: {pc1_contributions[0]:.1%}, {variables[1]}: {pc1_contributions[1]:.1%}, {variables[2]}: {pc1_contributions[2]:.1%})"
    ylabel = f"Dim 2 ({variables[0]}: {pc2_contributions[0]:.1%}, {variables[1]}: {pc2_contributions[1]:.1%}, {variables[2]}: {pc2_contributions[2]:.1%})"

    # Préparez la figure
    plt.figure(figsize=(10, 8))

    # Affichage des observations, colorées par cluster
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Clusters et points représentatifs (après PCA)")

    # Affichage des centroïdes
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                marker='X', c='red', s=200, label='Centroïdes')

    # Affichage des points représentatifs de chaque cluster
    for cluster, values in groupes_representatifs.items():
        idx = values['idx']
        # Marquer le point représentatif dans l'espace PCA
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], c='black', edgecolors='white', s=150, marker='o', label=f'Rep. cluster {cluster}' if cluster==list(groupes_representatifs.keys())[0] else "")

        # Récupérer les coordonnées d'origine (gcm-rcm, bc, hm)
        coord_gcm_rcm = ds_stacked["gcm-rcm"].isel(sample=idx).values
        coord_bc      = ds_stacked["bc"].isel(sample=idx).values
        coord_hm      = ds_stacked["hm"].isel(sample=idx).values

        # Annoter le graphique avec ces coordonnées
        annotation = f"C{cluster}\n{coord_gcm_rcm}, {coord_bc}, {coord_hm}"
        plt.annotate(annotation, (X_pca[idx, 0], X_pca[idx, 1]), textcoords="offset points", xytext=(5,5), fontsize=9, color='black')

    # Ajout d'une légende et d'une barre de couleur
    plt.legend()
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster")

    plt.savefig(f"/home/bcalmel/Documents/3_results/test42.png")












    # NARRATIVES BY SEASON
    x = ds[variables[f'simulation-horizon_by-sims_{plot_type}']].sel(season='Hiver', horizon='horizon3')
    y = ds[variables[f'simulation-horizon_by-sims_{plot_type}']].sel(season='Été', horizon='horizon3')

    x_list = []
    y_list = []
    name_list = []
    for var in x.var():
        if any(~np.isnan(x[var].values)) and any(~np.isnan(y[var].values)):
            x_list.append(np.nanmedian(x[var].values))
            y_list.append(np.nanmedian(y[var].values))
            if data_type == 'climate':
                name_list.append(var.split('_')[1])

    # Liste des marqueurs disponibles
    available_markers = ['o', 'D', '^', 'v', 'p', '*', 'h', 'x', '+', 's', 'H', '<', '>', '|', '_', ]

    # Vérification pour gérer une liste plus longue que les marqueurs disponibles
    if len(np.unique(name_list)) > len(available_markers):
        raise ValueError("La liste contient plus de chaînes uniques que de marqueurs disponibles.")

    # Association des chaînes aux marqueurs
    marker_mapping = {string: marker for string, marker in zip(np.unique(name_list), available_markers)}
    marker_list = [marker_mapping[val] for val in name_list]

    # Narratifs by clustering (K-means)
    kmeans = KMeans(n_clusters=4, random_state=0)
    df = pd.DataFrame({'x': x_list, 'y': y_list})
    df['cluster'] = kmeans.fit_predict(df[['x','y']])
    # Sélectionner un point représentatif par cluster (par exemple, le plus proche du centroïde)
    representative_points = df.loc[
        df.groupby('cluster').apply(
            lambda group: group[['x', 'y']].sub(kmeans.cluster_centers_[group.name]).pow(2).sum(axis=1).idxmin()
        )
    ]
    couples = list(zip(representative_points['x'], representative_points['y']))
    # Obtenir les centroïdes
    centroids = kmeans.cluster_centers_
    # Créer un Voronoi pour délimiter les aires
    vor = Voronoi(centroids)

    fig, ax = plt.subplots(1, 1, figsize=(6,4), constrained_layout=True)
    ax.grid()

    # dict_hm = {key: [] for key in marker_mapping.keys()}
    # for var in x.data_vars:
    #     print(var)
    #     # Identifier la clé du dictionnaire présente dans le nom de la variable
    #     hm = next((key for key in shape_hp if key in var), 'NONE')
    #     marker = shape_hp[hm]
    #     dict_hm[hm].append(var)
    #
    #     # Tracer la variable
    #     x_value = np.nanmedian(x[var].values)
    #     y_value = np.nanmedian(y[var].values)
    #
    #     if (x_value, y_value) in couples:
    #         plt.scatter(x_value, y_value, marker=marker, alpha=1,
    #                     color='green', zorder=2)
    #     else:
    #         plt.scatter(x_value, y_value, marker=marker, alpha=0.8,
    #                     color='k', zorder=0)
    #
    #
    # for key, shape in shape_hp.items():
    #     plt.scatter(np.nanmedian(x[dict_hm[key]].to_array()), np.nanmedian(y[dict_hm[key]].to_array()),
    #                 marker=shape, alpha=0.8,
    #                 color='firebrick', zorder=1)

    for i in range(len(x_list)):
        if (x_list[i], y_list[i]) in couples:
                plt.scatter(x_list[i], y_list[i], marker=marker_list[i], alpha=1,
                            color='green', zorder=2)
        else:
            plt.scatter(x_list[i], y_list[i], marker=marker_list[i], alpha=0.8, color='firebrick', zorder=1)


    # Tracer les aires des clusters avec Voronoi
    voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False, line_colors='green',
                    line_width=0.4, line_alpha=0.6, point_size=0, linestyle='--')

    val_min = np.min([x_list, y_list])
    val_max = np.max([x_list, y_list])
    lag = (val_max - val_min) * 0.05
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlim(val_min - lag, val_max + lag)
    ax.set_ylim(val_min - lag, val_max + lag)
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    # ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel(f'{title_join} estival {plot_type_name}')
    ax.set_xlabel(f'{title_join} hivernal {plot_type_name}')
    legend_handles = [
        mlines.Line2D([], [], color='black', marker=shape, linestyle='None', markersize=8,
                      label=f'{key}')
        for key, shape in marker_mapping.items()
    ]
    plt.legend(
        handles=legend_handles,
        loc="center left",  # Position relative
        bbox_to_anchor=(1, 0.5)  # Placer la légende à droite du graphique
    )

    plt.savefig(f"/home/bcalmel/Documents/3_results/HMUC_Loire_Bretagne/figures/global/narratifs_test.pdf",
                bbox_inches='tight')
