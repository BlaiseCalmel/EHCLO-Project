import matplotlib.pyplot as plt
import matplotlib as mpl
import geopandas.geodataframe as gpd
import numpy as np
import math


def plot_map(path_result, background_shp, data, col, indicator, study_shp=None, rivers_shp=None,
             figsize=(18, 18), nrow=1, ncol=1, palette='BrBG', discretize=None, zoom=5000, s=30,
             title=None):
    """
    """


def plot_map(gdf, column, row_headers, col_headers, cbar_title, path_result, title=None, dict_shapefiles=None,
             percent=True, bounds=None, discretize=7, palette='BrBG', zoom=5000, fontsize=14, font='sans-serif'):

    # gdf = gpd.GeoDataFrame({
    #     'geometry': ds['geometry'].values,
    #     'code': ds[column].values
    # })


    col_keys = col_headers.keys()
    col_values = col_headers.values()
    row_keys = row_headers.keys()
    row_values = row_headers.values()

    len_rows = len(row_keys)
    len_cols = len(col_keys)
    vmax = math.ceil(abs(ds.variables[indicator]).max() / discretize) * discretize
    vmin = -vmax

    bounds_cmap = np.linspace(vmin, vmax, discretize+1)
    cmap = mpl.cm.get_cmap(palette, discretize)
    norm = mpl.colors.BoundaryNorm(bounds_cmap, cmap.N)

    plt.rcParams['font.family'] = font      # Type de police
    plt.rcParams['font.size'] = fontsize
    text_kwargs ={'weight': 'bold'}

    fig, axes = plt.subplots(len_rows, len_cols, figsize=(1 + 4 * len_cols, len_rows * 4), constrained_layout=True)
    # Titre principal de la figure
    if title is not None:
        fig.suptitle(title, fontsize=plt.rcParams['font.size'] + 2)

    axes_flatten = axes.flatten()

    # Tracer chaque horizon en sélectionnant les données correspondantes
    # Horizon 1
    for col_idx, col in enumerate(col_headers):
        for row_idx, row in enumerate(row_headers):
            idx = len_cols * row_idx + col_idx
            ax = axes_flatten[idx]
            row_data = ds.sel(horizon=row)[indicator].sel(code=col).values
            gdf[indicator] = row_data
            #####################################################################
            # TODO SPECIFY FOR MAP PLOT
            # Background shapefiles
            if dict_shapefiles is not None:
                for key, subdict in dict_shapefiles.items():
                    shp_kwargs = {k: subdict[k] for k in subdict.keys() if k != 'shp'}
                    subdict['shp'].plot(ax=ax, figsize=(18, 18), **shp_kwargs)


            gdf.plot(column=indicator, cmap=cmap, norm=norm, ax=ax, legend=False, markersize=100, edgecolor='k')

            if bounds is not None:
                ax.set_xlim(bounds[0] - zoom, bounds[2] + zoom)
                ax.set_ylim(bounds[1] - zoom, bounds[3] + zoom)
            ax.set_axis_off()
            #####################################################################

    add_headers(fig, col_headers=col_values, row_headers=row_values, row_pad=0, col_pad=5, **text_kwargs)

    # Colorbar
    #####################################################################
    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([1, 0.15, 0.04, 0.7])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.BoundaryNorm(bounds_cmap, cmap.N))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(cbar_title, **text_kwargs)
    if percent:
        cbar.ax.set_yticklabels([f'{int(b * 100)}%' for b in bounds_cmap])
    #####################################################################

    plt.savefig(path_result, bbox_inches='tight')


def add_headers(fig, *, row_headers=None, col_headers=None, row_pad=1, col_pad=5, rotate_row_headers=True,
                **text_kwargs):
    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                # xycoords=ax.yaxis.label,
                xycoords="axes fraction",
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )