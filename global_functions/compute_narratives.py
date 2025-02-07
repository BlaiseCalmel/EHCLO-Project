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

    # Selected stations
    stations = reference_stations['La Loire']
    data = [ds[variables[f'simulation-horizon_by-sims_{plot_type}']].sel(gid=list(stations.keys()),
                                                                         horizon='horizon3') for ds in datasets]

    any(~np.isnan(data[0][var].values))

    # NARRATIVES BY SEASON
    x = ds[variables[f'simulation_horizon_{plot_type}_by_sims']].sel(season='Hiver', horizon='horizon3')
    y = ds[variables[f'simulation_horizon_{plot_type}_by_sims']].sel(season='Été', horizon='horizon3')
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
