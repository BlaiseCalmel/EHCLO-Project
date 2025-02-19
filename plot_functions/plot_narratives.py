import matplotlib.pyplot as plt
import numpy as np

def plot_narratives(x_data, ds_stacked, groupes_representatifs, labels, cluster_names,
                    path_result, xlabel, ylabel, title, centroids=None, count_stations=None,
                    above_threshold=None, palette='viridis', n=4):
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
    hm_list = list(ds_stacked.hm.values)
    shape_list = [shape_hp[hm_value] for hm_value in hm_list]
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

