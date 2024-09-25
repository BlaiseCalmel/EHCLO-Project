import matplotlib.pyplot as plt
import geopandas.geodataframe as gpd
import numpy as np



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


def plot_scatter_on_map(path_result, back_shp, df_plot, cols, indicator, study_shp=None, rivers_shp=None,
                        figsize=(18, 18), nrow=1, ncol=1, palette='BrBG', discretize=None, zoom=5000, s=30,
                        title=None, vmin=None, vmax=None):
    """
    Function to plot scatter data with a colorbar (continuous/discrete) on a shapefile background
    :param path_result:
    :param back_shp:
    :param df_plot:
    :param cols:
    :param indicator:
    :param study_shp:
    :param rivers_shp:
    :param figsize:
    :param nrow:
    :param ncol:
    :param palette:
    :param discretize:
    :param zoom:
    :return:
    """
    # Compute min and max values
    if vmin is None:
        vmin = min(df_plot[cols].min())
    if vmax is None:
        vmax = max(df_plot[cols].max())
    # norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    if discretize is not None:
        ticks = np.arange(vmin, vmax+discretize, discretize)
        cm = plt.get_cmap(palette, len(ticks))
        vmin = vmin - discretize * 0.5
        vmax = vmax + discretize * 0.5
    else:
        cm = plt.get_cmap(palette)

    if study_shp is not None:
        bounds = study_shp.geometry.total_bounds
    else:
        bounds = back_shp.geometry.total_bounds

    if rivers_shp is not None:
        rivers_thresh = 0.5 * ((bounds[2] - bounds[0])**2 + (bounds[3] - bounds[1])**2)**0.5
        long_rivers_idx = rivers_shp.geometry.length > rivers_thresh
        long_rivers_shp = rivers_shp[long_rivers_idx]

    # Init plot
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize, layout='compressed')

    count_idx = -1
    for row_idx in range(nrow):
        for col_idx in range(ncol):
            count_idx += 1

            if isinstance(axs, (np.ndarray, np.generic)):
                if len(axs.shape) > 1:
                    ax = axs[row_idx, col_idx]
                else:
                    ax =axs[col_idx]
            else:
                ax = axs

            back_shp.plot(ax=ax, figsize=figsize, color='gainsboro', edgecolor='black', zorder=0)

            if study_shp is not None:
                study_shp.plot(ax=ax, figsize=figsize, color='white', edgecolor='firebrick', zorder=1)

            if rivers_shp is not None:
                long_rivers_shp.plot(ax=ax, linewidth=1, color='royalblue', zorder=2)

            p = ax.scatter(x=df_plot['XL93'], y=df_plot['YL93'], s=s, c=df_plot[cols[count_idx]], cmap=cm, zorder=3,
                           vmin=vmin, vmax=vmax)
            ax.set_xlim(bounds[0] - zoom, bounds[2] + zoom)
            ax.set_ylim(bounds[1] - zoom, bounds[3] + zoom)
            if title is None:
                ax.set_title(cols[count_idx])
            else:
                ax.set_title(title[count_idx])
            ax.set_axis_off()

    if discretize is not None:
        cbar = fig.colorbar(p, ax=axs, orientation='vertical', ticks=ticks)
    else:
        cbar = fig.colorbar(p, ax=axs, orientation='vertical')
    cbar.set_label(indicator)
    # fig.tight_layout()
    # Save
    plt.savefig(path_result, bbox_inches='tight')

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