import matplotlib.pyplot as plt
import geopandas.geodataframe as gpd
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

def define_bounds(shapefile, zoom=5000):
    raw_bounds = shapefile.geometry.total_bounds
    return [raw_bounds[0] - zoom, raw_bounds[1] - zoom, raw_bounds[2] + zoom, raw_bounds[3] + zoom]

def define_cbar(fig, axes_flatten, cmap, bounds_cmap, cbar_title=None, percent=False, **text_kwargs):
    # fig.subplots_adjust(right=0.95)
    # cbar_ax = fig.add_axes([1, 0.15, 0.04, 0.7])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.BoundaryNorm(bounds_cmap, cmap.N))
    # divider = make_axes_locatable(axes)
    # cbar_ax = divider.append_axes("right", size="5%", pad=0.05)

    cbar_ax = fig.add_axes([axes_flatten[-1].get_position().x1*1.01,
                            axes_flatten[-1].get_position().y0+0.01,
                            0.02,
                            axes_flatten[0].get_position().y1-axes_flatten[-1].get_position().y0-0.02])

    sm._A = []
    if percent:
        cbar = fig.colorbar(sm, cax=cbar_ax, drawedges=True, ticks=bounds_cmap, format='%.0f')
    else:
        cbar = fig.colorbar(sm, cax=cbar_ax, drawedges=True, ticks=bounds_cmap)
    if cbar_title:
        cbar.set_label(cbar_title, **text_kwargs)


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

def save_shp_figure(back_shp:gpd.GeoDataFrame, path_result:str, study_shp:gpd.GeoDataFrame=None,
                    rivers_shp:gpd.GeoDataFrame=None,
                    figsize:tuple=None, **kwargs):
    """

    :param current_shp:
    :param path_result:
    :param figsize:
    :param kwargs:
    :return:
    """
    if figsize is not None:
        figsize = figsize
    else:
        figsize = (18, 18)

    fig, ax = plt.subplots(figsize=figsize)
    back_shp.plot(ax=ax, figsize=figsize, color='gainsboro', edgecolor='black')

    if study_shp is not None:
        study_shp.plot(ax=ax, color='gainsboro', linewidth=2, edgecolor='firebrick', linestyle="--")
        bounds = study_shp.geometry.total_bounds
    else:
        bounds = back_shp.geometry.total_bounds

    if rivers_shp is not None:
        rivers_tresh = 0.5 * ((bounds[2] - bounds[0])**2 + (bounds[3] - bounds[1])**2)**0.5
        long_rivers_idx = rivers_shp.geometry.length > rivers_tresh

        long_rivers_shp = rivers_shp[long_rivers_idx]

        long_rivers_shp.plot(ax=ax, linewidth=2, color='royalblue')

    ax.set_xlim(bounds[0] - 5000, bounds[2] + 5000)
    ax.set_ylim(bounds[1] - 5000, bounds[3] + 5000)

    plt.savefig(path_result, **kwargs)
    plt.close()


def plot_timeline(df_station, station_name, path_result=None, figsize=(18, 18), selected_sim=None, name='red'):
    # Init plot
    fig, ax = plt.subplots(figsize=figsize, layout='compressed')

    for key, grp in df_station[df_station['sim'] != selected_sim].groupby(['sim']):
        ax.plot(grp['year'], grp[station_name], c='grey', linewidth=1, )

    selected_df = df_station[df_station['sim'] == selected_sim]
    ax.plot(grp['year'], grp[station_name], c='r', linewidth=1, label=name)

    ax.set_title(station_name)
    ax.legend()

    # Save
    if path_result is not None:
        plt.savefig(path_result, bbox_inches='tight')
