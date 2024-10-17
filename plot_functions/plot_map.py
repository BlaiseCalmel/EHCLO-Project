import matplotlib.pyplot as plt
import geopandas.geodataframe as gpd
import numpy as np


def plot_map(path_result, background_shp, df_plot, cols, indicator, study_shp=None, rivers_shp=None,
             figsize=(18, 18), nrow=1, ncol=1, palette='BrBG', discretize=None, zoom=5000, s=30,
             title=None, vmin=None, vmax=None):
    """
    """
    # Compute min and max values
    # if vmin is None:
    #     vmin = df_plot.variable.data.min()
    # if vmax is None:
    #     vmax = df_plot.variable.data.max()

    if vmin is None:
        vmin = df_plot[cols].min().values
    if vmax is None:
        vmax = df_plot[cols].max().values
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
        bounds = background_shp.geometry.total_bounds

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

            # Background
            # background_shp.plot(ax=ax, figsize=figsize, color='gainsboro', edgecolor='black', zorder=0)

            # Administrative shapes
            if study_shp is not None:
                study_shp.plot(ax=ax, figsize=figsize, color='white', edgecolor='firebrick', zorder=1)

            # River shapes
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