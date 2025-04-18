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
from matplotlib.colors import from_levels_and_colors
from matplotlib.patches import Circle, Rectangle 
from plot_functions.plot_common import *
import matplotlib.cm as cm
import matplotlib.patches as mpatches
# from adjustText import adjust_text

def decimal_places(x):
    x_str = str(x).rstrip("0")
    if "." in x_str:
        return len(x_str.split(".")[1])
    return 0


def mapplot(gdf, indicator_plot, path_result, cols=None, rows=None, ds=None,
            cbar_title=None, title=None, dict_shapefiles=None, bounds=None, discretize=7,
            vmin_user=None, vmax_user=None, palette='BrBG', cbar_midpoint=None,
            fontsize=14, edgecolor='k', cbar_values=None, cbar_ticks='border',
            font='sans-serif', references=None, markersize=50, start_cbar_ticks='', end_cbar_ticks='',
            alpha=1, uncertainty=None):

    ds_plot = copy.deepcopy(ds)
    len_cols, cols_plot, ds_plot = init_grid(cols, ds_plot)
    len_rows, rows_plot, ds_plot = init_grid(rows, ds_plot)
    subplot_titles = None
    if isinstance(rows, int):
        len_cols = int(len_cols / len_rows)
        subplot_titles = cols['names_plot']
        cols_plot['names_plot'] = [None]
    if isinstance(cols, int):
        len_rows = int(len_rows / len_cols)
        subplot_titles = rows['names_plot']
        rows_plot['names_plot'] = [None]

    if np.logical_not(isinstance(indicator_plot, list)) and indicator_plot in gdf.columns:
        plot_vmax = abs(gdf[indicator_plot]).max()
    else:
        if np.logical_not(isinstance(indicator_plot, list)):
            plot_vmax = abs(ds[indicator_plot]).max().values
        else:
            plot_vmax = abs(ds[indicator_plot].to_array()).max().values
    if vmax_user is None:
        vmax = plot_vmax
    else:
        vmax =vmax_user

    if np.logical_not(isinstance(indicator_plot, list)) and indicator_plot in gdf.columns:
        plot_vmin = gdf[indicator_plot].min()
    else:
        if np.logical_not(isinstance(indicator_plot, list)):
            plot_vmin = ds[indicator_plot].min().values
        else:
            plot_vmin = ds[indicator_plot].to_array().min().values

    if vmin_user is None:
        if abs(plot_vmin) >= vmax:
            vmin = plot_vmin
        else:
            vmin = -vmax
    else:
        vmin = vmin_user
    
    if cbar_midpoint == 'min':
        midpoint = vmin
    elif isinstance(cbar_midpoint, (float, int)):
        midpoint = cbar_midpoint
    else:
        midpoint = 0

    abs_max = max([vmax, -vmin])
    n = 2*(abs_max-midpoint) / discretize
    # exponent = round(math.log10(n))
    exponent = round(np.floor(math.log10(n)))
    step = np.round(n, -exponent)
    if step == 0:
        step = n

    # if cbar_values is None:
    #     cbar_values = decimal_places(step)
    # else:
    #     step = cbar_values
    levels = mirrored(maxval=abs_max, inc=step, val_center=midpoint)

    # levels = np.arange(vmin, abs_max+1, 1)
    # levels = np.round(levels, -exponent+1)

    if levels[0] > levels[-1]:
        levels = levels[::-1]

    if cbar_ticks == 'mid':
        levels = np.array([levels[0] - step/2] + [i + step/2 for i in levels])
        
    # levels = np.linspace(vmin, vmax, discretize+1)
    extended_levels = copy.deepcopy(levels)
    if midpoint is not None:
        midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
        # vals = np.interp(midp, [midpoint-max(abs(vmin), abs(vmax)), midpoint, midpoint+max(abs(vmin), abs(vmax))],
        #                  [0, 0.5, 1])
        vals = np.interp(midp, [vmin, midpoint, vmax],
                         [0, 0.5, 1])
        colormap = getattr(plt.cm, palette)
        colors = colormap(vals)

        # Limit values to extrema        
        extended_indices = ((levels >= min((plot_vmin, midpoint))) & (levels <= max((plot_vmax, midpoint))))
        # extended_indices = np.where((levels >= vmin) & (levels <= vmax))[0]
        import bisect
        extended_levels = extended_levels[extended_indices]
        if extended_levels[-1] < vmax:
            if extended_indices[-1] < len(levels):
                next_idx = bisect.bisect_right(levels.tolist(), extended_levels[-1])
                extended_levels = np.append(extended_levels, levels[next_idx])
        
        if extended_levels[0] > vmin:
            if extended_indices[0] < len(levels):
                prev_idx = bisect.bisect_left(levels.tolist(), extended_levels[0])
                extended_levels = np.append(levels[prev_idx], extended_levels)


        midp = np.mean(np.c_[extended_levels[:-1], extended_levels[1:]], axis=1)
        vals = np.interp(midp, [min(extended_levels), midpoint, max(extended_levels)],
                         [0, 0.5, 1])
        colormap = getattr(plt.cm, palette)
        colors = colormap(vals)
        extended_colors = colors

        # # extended_indices = np.unique(np.concatenate([indices - 1, indices, indices + 1]))
        # # extended_indices = extended_indices[(extended_indices >= 0) & (extended_indices < len(levels))]
        # extended_colors = colors[extended_indices[:-1]]
        # extended_levels = extended_levels[extended_indices]
        # if extended_levels[-1] < vmax:
        #     if extended_indices[-1] < len(levels):
        #         next_idx = bisect.bisect_right(levels.tolist(), extended_levels[-1])

        #         extended_levels = np.append(extended_levels, levels[next_idx])
        #         extended_colors = np.vstack([extended_colors, colors[extended_indices[:-1][-1] + 1]])

        # if extended_levels[0] > vmin:
        #     if extended_indices[0] > 0:
        #         extended_levels = np.insert(extended_levels, 0, levels[extended_indices[0] - 1])
        #         extended_colors = np.vstack([colors[extended_indices[0] - 1], extended_colors])

        if extended_levels[-1] < vmax and extended_levels[0] > vmin:
            cmap, norm = from_levels_and_colors(extended_levels, np.vstack([
                colors[max([extended_indices[0]-1, 0])],
                extended_colors,
                colors[min([extended_indices[-1]+1, len(colors)-1])]
            ]), extend='both')
        elif extended_levels[-1] < vmax:
            cmap, norm = from_levels_and_colors(extended_levels, np.vstack([
                extended_colors,
                colors[min([extended_indices[-1]+1, len(colors)-1])]
            ]), extend='max')
        elif extended_levels[0] > vmin:
            cmap, norm = from_levels_and_colors(extended_levels, np.vstack([
                colors[max([extended_indices[0]-1, 0])],
                extended_colors
            ]), extend='min')
        else:
            temp_levels = copy.deepcopy(extended_levels)
            temp_levels[0] -= 0.01*exponent*step
            temp_levels[-1] += 0.01*exponent*step
            cmap, norm = from_levels_and_colors(temp_levels, extended_colors)

    else:
        # levels = np.linspace(vmin, vmax, discretize+1)
        cmap = plt.get_cmap(palette, len(extended_levels)-1)
        norm = mpl.colors.BoundaryNorm(boundaries=extended_levels, ncolors=len(extended_levels)-1, clip=True)

    plt.rcParams['font.family'] = font
    plt.rcParams['font.size'] = fontsize
    text_kwargs ={'weight': 'bold', 'fontsize': fontsize}
    hatching = False

    fig_dim = 4
    # ratio = fontsize / 18
    figsize = (fig_dim * len_cols, len_rows * fig_dim)
    # if bounds is not None:
    #     x_y_ratio = abs((bounds[2] - bounds[0]) / (bounds[3] - bounds[1]))
    #     if x_y_ratio > 1:
    #         figsize = (fig_dim * len_cols , len_rows * fig_dim / x_y_ratio)
    #     else:
    #         figsize = (fig_dim * len_cols * x_y_ratio , len_rows * fig_dim)
    # else:
    #     figsize = (fig_dim * len_cols, len_rows * fig_dim)

    if title:
        figsize = (figsize[0], figsize[1] * 1.02)
    if subplot_titles:
        subtitles_lines = 0
        if isinstance(cols, dict) and 'names_plot' in cols.keys():
            subtitles_lines = max(s.count('\n') for s in cols['names_plot'])
        figsize= (figsize[0], figsize[1] * (1 + 0.02 * len_rows) + subtitles_lines)
    
    # if len_rows == 1:
    #     figsize= (figsize[0] * 1.05, figsize[1])

    fig, axes = plt.subplots(len_rows, len_cols, figsize=figsize, constrained_layout=True)
    # hspace = 0.03
    # wspace = 0.01
    # # wspace = -0.3
    # top = 1
    # if subplot_titles:
    #     hspace += 0.05 * (fontsize - 18)
    #     top -= 0.03 * (fontsize - 18)
    #     wspace += 0.18 * (fontsize - 18)

    # Main title
    if title is not None:
        # top -= 0.04 * (fontsize - 18)
        fig.suptitle(title, fontsize=plt.rcParams['font.size'], weight='bold')

    # plt.subplots_adjust(left=0, bottom=0.01, right=1, top=top, wspace=wspace, hspace=hspace)

    if len_rows > 1 or len_cols > 1:
        axes_flatten = axes.flatten()
    else:
        axes_flatten = [axes]

    # Save axes position
    # for ax in axes_flatten:
    #     initial_position = ax.get_position()
    #     ax.set_position(initial_position)

    # Iterate over subplots
    idx = -1
    for row_idx, row in enumerate(rows_plot['values_var']):
        for col_idx, col in enumerate(cols_plot['values_var']):
            idx += 1
            # idx = len_cols * row_idx + col_idx
            subplot_title = None
            ax = axes_flatten[idx]
            if cols_plot['names_coord'] == 'indicator':
                current_indicator = col
            elif rows_plot['names_coord'] == 'indicator':
                current_indicator = row
            elif isinstance(indicator_plot, list):
                current_indicator = indicator_plot[idx]
            else:
                current_indicator = indicator_plot

            if subplot_titles:
                subplot_title = subplot_titles[idx]

            gdf_plot = copy.deepcopy(gdf)
            # If not in gdf_plot, add a col from selection
            if current_indicator not in gdf.columns:
                temp_dict = {}
                if cols_plot['names_coord'] is not None and col is not None and cols_plot['names_coord'] != 'indicator':
                    temp_dict |= {cols_plot['names_coord']: col}
                if rows_plot['names_coord'] is not None and row is not None and rows_plot['names_coord'] != 'indicator':
                    temp_dict |= {rows_plot['names_coord']: row}

                row_data = ds_plot.sel(temp_dict)[current_indicator].values

                if uncertainty is not None:
                    uncertainty_data = abs(ds_plot.sel(temp_dict)[uncertainty].values) < 80
                    gdf_plot[uncertainty] = uncertainty_data

                gdf_plot[current_indicator] = row_data

            # Background shapefiles
            if dict_shapefiles is not None:
                for key, subdict in dict_shapefiles.items():
                    shp_kwargs = {k: subdict[k] for k in subdict.keys() if k != 'shp'}
                    subdict['shp'].plot(ax=ax, figsize=(18, 18), **shp_kwargs)

            # Plot
            gdf_plot.plot(column=current_indicator, cmap=cmap, norm=norm, ax=ax, legend=False, markersize=markersize,
                          edgecolor=edgecolor, alpha=alpha, zorder=18, )
            

            if uncertainty is not None:
                if any(gdf_plot[uncertainty]):
                    gdf_plot[gdf_plot[uncertainty]].plot(ax=ax, legend=False, facecolor='none',   
                                edgecolor='k', hatch='////', linewidth=0, alpha=0.7, zorder=19)
                    hatching = True


            if references is not None:
                for key, values in references.items():
                    scatter_kwarg = {i: j for i, j in values.items() if i != 'text'}          
                    
                    ax.scatter(**scatter_kwarg)
                    if 'text' in values.keys():
                        text_kwarg = values['text']
                        ax.annotate(xycoords='data',
                                    weight='bold',
                                    fontsize=plt.rcParams['font.size'], **text_kwarg)

            if subplot_title:
                if title:
                    if isinstance(subplot_title, dict):
                        # ax.text(bounds[0], bounds[3], subplot_title['label'], fontsize=18, color='w',
                        #         ha='center', va='center', fontweight='bold',
                        #         bbox=dict(facecolor=subplot_title['color'], edgecolor=subplot_title['color'],
                        #                   boxstyle='circle'))
                        
                        rectangle = Rectangle((0, 0.9), 0.1, 0.3, facecolor=subplot_title['color'], edgecolor=subplot_title['color'], 
                                              transform=ax.transAxes, zorder=20)
                        ax.add_patch(rectangle) 
                        ax.text(0.05, 0.945, subplot_title['label'], ha='center', va='center', fontweight='bold',
                                transform=ax.transAxes, fontsize=18, color='w', zorder=21)
                                # ha='center', va='center', fontweight='bold',
                                # bbox=dict(facecolor=subplot_title['color'], edgecolor=subplot_title['color'],
                                #           boxstyle='circle'))
                    else:
                        ax.set_title(subplot_title, fontsize=plt.rcParams['font.size'] - 2)

                else:
                    if isinstance(subplot_title, dict):
                        # cercle = Circle((0.05, 0.95), 0.06, facecolor=subplot_title['color'], edgecolor=subplot_title['color'], 
                        #                 transform=ax.transAxes, zorder=20)
                        rectangle = Rectangle((0, 0.9), 0.1, 0.3, facecolor=subplot_title['color'], edgecolor=subplot_title['color'], 
                                              transform=ax.transAxes, zorder=20)
                        ax.add_patch(rectangle) 
                        ax.text(0.05, 0.945, subplot_title['label'], ha='center', va='center', fontweight='bold',
                                transform=ax.transAxes, fontsize=18, color='w', zorder=21)
                    else:
                        ax.set_title(subplot_title, weight='bold', fontsize=plt.rcParams['font.size'])

            # Headers and axes label
            add_header(ax, rows_plot, cols_plot, ylabel='', xlabel='')

            if bounds is not None:
                ax.set_xlim(bounds[0], bounds[2])
                ax.set_ylim(bounds[1], bounds[3])
            # ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])
            plt.setp(ax.spines.values(), color=None)

    if cbar_values is None:
        nombre_str = str(step).rstrip('0')
        if '.' in nombre_str:
            cbar_values = len(nombre_str.split('.')[1]) 
        else:
            cbar_values = 0
        
    hatching_patch = None
    if hatching:
        hatching_patch = mpatches.Patch(facecolor='white', hatch='///', edgecolor='black', label=f"Signe de\nchangement\nincertain")

    cbar = define_cbar(fig, axes_flatten, len_rows, len_cols, cmap, bounds_cmap=extended_levels,
                       cbar_title=cbar_title, cbar_values=cbar_values, cbar_ticks=cbar_ticks,
                       start_cbar_ticks=start_cbar_ticks, end_cbar_ticks=end_cbar_ticks, hatching_patch=hatching_patch,
                       **text_kwargs)

    plt.savefig(path_result, bbox_inches='tight')

