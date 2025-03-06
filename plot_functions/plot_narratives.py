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
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib as mpl

def plot_narratives(x_data, ds_stacked, representative_groups, labels, cluster_names,
                    path_result, xlabel, ylabel, title, centroids=None, count_stations=None,
                    above_threshold=None, palette='viridis', n=4, rows=None, cols=None):

    shape_hp = {
        'CTRIP': 'D',
        'EROS': 'P',
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
    hm_legends = []
    for hm, shape in shape_hp.items():
        hm_legends.append(plt.scatter(np.nan, np.nan, alpha=0.6, marker=shape, label=hm, color='k'))

    # Affichage des points représentatifs de chaque cluster
    if isinstance(palette, str):
        cmap = plt.get_cmap(palette, n)
    else:
        # Création d'une colormap personnalisée
        cmap = mcolors.LinearSegmentedColormap.from_list("hydroclim", palette, N=n)

    norm = plt.Normalize(vmin=-0.5, vmax=n-0.5)  # Normalisation entre 0 et 3

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 18
    # plt.rcParams['font.size'] = 10

    # plt.figure(figsize=(10, 8))
    if rows is None:
        rows = [None]
    if cols is None:
        cols = [None]
    # if len(rows) > 1:
    #     base_width = 1
    # else:
    #     base_width = 5
    fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), figsize=(1 + len(cols) * 5, 2 + len(rows) * 4))
    # fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), figsize=(6 + len(cols) * 5, 2 + len(rows) * 4))
    plt.subplots_adjust(hspace=0.12, wspace=0.32)
    # Affichage des points représentatifs de chaque cluster
    edgecolors = ['k', 'red', 'blue', 'green', 'orange']
    if not isinstance(representative_groups, list):
        representative_groups = [representative_groups]

    if hasattr(axes, 'flatten'):
        axes_flatten = axes.flatten()
    elif isinstance(axes, list):
        axes_flatten = axes
    else:
        axes_flatten = [axes]

    idx_ax = -1
    for row_idx, row in enumerate(rows):
        for col_idx in range(len(cols)):
            if cols[col_idx] is not None:
                if col_idx != len(cols)-1:
                    col_idx2 =col_idx+1
                else:
                    col_idx2 = 0
                # Construire les noms des axes
                if cols[col_idx2] == 'QA':
                    col_idx, col_idx2 = col_idx2, col_idx

                xlabel = f"Variation {cols[col_idx]} (%)"
                ylabel = f"Variation {cols[col_idx2]} (%)"
                title = f"{cols[col_idx]} - {cols[col_idx2]} "
            else:
                col_idx2 = 1

            x_data_plot = x_data[:, [col_idx, col_idx2]]
            # idx = len_cols * row_idx + col_idx
            idx_ax += 1
            ax = axes_flatten[idx_ax]
            sbs = ax.get_subplotspec()

            ax.grid(alpha=0.3)
            # Affichage des observations, colorées par cluster
            for j in range(len(x_data_plot)):
                alpha=0.7
                if above_threshold is not None and not above_threshold[j]:
                    alpha=0.2
                ax.scatter(x_data_plot[j, 0], x_data_plot[j, 1], c=labels[j], cmap=cmap, norm=norm, alpha=alpha,
                           marker=shape_list[j], zorder=1)
                if count_stations is not None:
                    ax.annotate(count_stations[j], (x_data_plot[j, 0], x_data_plot[j, 1]),
                                 fontsize=9, color='black', zorder=13)

            # scatter = ax.scatter([np.nan]*len(representative_groups), [np.nan]*len(representative_groups),
            #                      c=np.arange(len(representative_groups)), cmap=cmap, norm=norm)

                # Affichage des centroïdes
            representative_legends = []
            if centroids is not None:
                representative_legends.append(ax.scatter(centroids[:, 0], centroids[:, 1],
                                                          marker='X', c='red', s=200, label='Centroïdes', zorder=9))

            rg = representative_groups[row_idx]
            for cluster, values in rg.items():

                idx = values['idx']

                # Récupérer les coordonnées d'origine (gcm-rcm, bc, hm)
                coord_gcm_rcm = ds_stacked["gcm-rcm"].isel(sample=idx).values
                # coord_bc      = ds_stacked["bc"].isel(sample=idx).values
                coord_hm      = ds_stacked["hm"].isel(sample=idx).values

                # Annoter le graphique avec ces coordonnées
                annotation = f"{coord_gcm_rcm}\n{coord_hm}"
                # Marquer le point représentatif dans l'espace PCA
                representative_legends.append(ax.scatter(x_data_plot[idx, 0], x_data_plot[idx, 1], c=cluster,
                                                          edgecolors='k', s=400, marker=shape_list[idx],
                                                          cmap=cmap, norm=norm, zorder=10, label=annotation))


            # Add line to zero
            xmin, ymin = np.min(x_data_plot, axis=0)
            xmax, ymax = np.max(x_data_plot, axis=0)
            min_val = np.min([xmin, ymin])
            max_val = np.max([xmax, ymax])
            ax.hlines(y=0, xmin=min_val, xmax=max_val, color='k', linestyle='--', linewidth=1, alpha=0.3, zorder=0)
            ax.vlines(x=0, ymin=min_val, ymax=max_val,color='k', linestyle='--', linewidth=1, alpha=0.3, zorder=0)
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.set_aspect('equal')

            # ax.set_ylabel(ylabel)
            # ax.set_title(title)

            if sbs.is_last_row():
                ax.set_xlabel(xlabel)
            if sbs.is_first_row():
                if title is not None:
                    ax.set_title(title)

            if not sbs.is_last_row():
                ax.set_xticklabels([])

            if sbs.is_first_col():
                if rows[row_idx] is not None:
                    ax.set_ylabel(f"$\\bf{{{rows[row_idx].title()}}}$ \n\n{ylabel}")
                else:
                    ax.set_ylabel(ylabel)
            else:
                if cols[col_idx] is not None:
                    ax.set_ylabel(ylabel)

            # ax.set_aspect(10)
            # ax.set_aspect('equal')
            if sbs.is_last_col():
                cbar_ax = fig.add_axes([ax.get_position().x1 + 0.025, ax.get_position().y0,
                                        0.04 / len(cols),
                                        ax.get_position().y1 - ax.get_position().y0])
                # Add colorbar
                sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
                # cax = axes_flatten[idx_ax-1]
                cbar = plt.colorbar(sm, cax=cbar_ax, drawedges=True)
                cbar.set_label("Cluster")
                cbar.set_ticks([0,1,2,3])
                cbar.set_ticklabels(cluster_names)
                cbar.ax.invert_yaxis()

                if len(axes_flatten) > 1:
                    ax.legend(handles=representative_legends, loc='center left', bbox_to_anchor=(1.34, 0.5),
                              title="Narratifs", ncol=1)
                else:
                    ax.legend(handles=representative_legends, loc='center left', bbox_to_anchor=(1.28, 0.28),
                              title="Narratifs", ncol=1)

    if len(axes_flatten) > 1:
        if len(cols) == 1:
            fig.legend(handles=hm_legends, loc='upper left', bbox_to_anchor=(0.0, 0.07),
                       title="Modèles Hydro", ncol=2+len(cols))
        elif len(rows) == 1:
            fig.legend(handles=hm_legends, loc='upper center', bbox_to_anchor=(0.5, 0.1),
                       title="Modèles Hydro", ncol=2+len(cols))
        else:
            fig.legend(handles=hm_legends, loc='upper center', bbox_to_anchor=(0.5, 0.06),
                       title="Modèles Hydro", ncol=2+len(cols))
    else:
        fig.legend(handles=hm_legends, loc='upper left', bbox_to_anchor=(1.11, 0.9),
                   title="Modèles Hydro", ncol=3)

    # plt.subplots_adjust(hspace=0.12, wspace=0.32)

    plt.savefig(path_result, bbox_inches='tight')

