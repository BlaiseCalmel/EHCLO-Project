import matplotlib.pyplot as plt
import geopandas.geodataframe as gpd
import pandas as pd

def save_shp_figure(current_shp:gpd.GeoDataFrame, path_result:str, figsize:tuple=None, **kwargs):
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
        figsize = (30, 18)

    fig, ax = plt.subplots(figsize=figsize)
    current_shp.plot(ax=ax, figsize=figsize, edgecolor='black')
    plt.savefig(path_result, **kwargs)

    plt.clf()


def plot_shp_figure(path_result:str, shapefile:gpd.GeoDataFrame, shp_column:str=None, df:pd.DataFrame=None,
                    indicator:str=None, figsize:tuple=None, shp_palette:str=None, scatter_palette:str='BrBG',
                    **kwargs):
    """

    :param path_result:
    :param shapefile:
    :param df:
    :param shp_column:
    :param indicator:
    :param figsize:
    :param palette:
    :param kwargs:
    :return:
    """

    if figsize is not None:
        figsize = figsize
    else:
        figsize = (30, 18)

    # Init plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot Shapefile
    if shp_column is not None:
        ax = shapefile.plot(figsize=figsize, edgecolor='black', column=shp_column, cmap=shp_palette,
                            legend=True, legend_kwds={"label": shp_column})

    else:
        ax = shapefile.plot(figsize=figsize, edgecolor='black', facecolor="none", cmap=shp_palette)

    # Plot Scatter
    if df is not None and indicator is not None:
        # Color map for scatter
        # scatter_cmap = colormaps[scatter_palette]
        p1 = ax.scatter(x=df['XL93'], y=df['YL93'], s=50, c=df[indicator], cmap=scatter_palette)
        fig.colorbar(p1, label=indicator)

    # Legend
    ax.legend()
    # ax.get_legend().set_bbox_to_anchor((1.5, 1))
    plt.axis('off')
    # Save and delete
    plt.savefig(path_result)


# SAVE AS SVG
# ## Import libraries
# import geopandas as gpd
# from bs4 import BeautifulSoup
# import random
# import textwrap
#
# ## Read data obtained from European Council as GeoPandas Dataframe
# fp = path
# data = gpd.read_file(fp)
#
# ## Define function that converts shapefile row to path
# def process_to_svg_group(row):
#     rd = row.to_dict()
#     del rd["geometry"]
#
#     # Keep dataframe columns as attributes of the path
#     to_add = []
#     for key, val in rd.items():
#         to_add.append('data-{}="{}"'.format(key, val))
#
#     # Convert geometry column to SVG paths
#     ps = BeautifulSoup(row.geometry._repr_svg_(), "xml").find_all("path")
#     paths = []
#     for p in ps:
#         new_path = f"""<g {' '.join(to_add)}>{str(p)}</g>"""
#         paths.append(new_path)
#
#     return "\n\n".join(paths)
#
# ## Convert each row of the dataframe to an SVG path
# processed_rows = []
# for i, row in data.sample(frac=1).iterrows():
#     p = process_to_svg_group(row)
#     processed_rows.append(p)
#
# ## Setup svg attributes
# props = {
#     "viewBox": f"{data.total_bounds[0]} {data.total_bounds[1]} {data.total_bounds[2] - data.total_bounds[0]} {data.total_bounds[3] -data.total_bounds[1] }",
#     "xmlns": "http://www.w3.org/2000/svg",
#     "xmlns:xlink": "http://www.w3.org/1999/xlink",
#     "transform": "scale(1, -1)",
#     "preserveAspectRatio":"XMinYMin meet"
# }
#
# ## Add attributes
# template = '{key:s}="{val:s}"'
# attrs = " ".join([template.format(key=key, val=props[key]) for key in props])
#
# ## Create SVG
# raw_svg_str = textwrap.dedent(r"""
# 	<?xml version="1.0" encoding="utf-8" ?>
# 	<svg {attrs:s}>
# 	{data:s}
# 	</svg>""").format(attrs=attrs, data="".join(processed_rows)).strip()
#
# ## Save SVG
# with open("map.svg", "w") as f:
#     f.write(raw_svg_str)