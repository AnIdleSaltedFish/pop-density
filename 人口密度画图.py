import numpy as np
# np.set_printoptions(threshold=np.inf)
import pandas as pd
import netCDF4 as nc
import xarray as xr
import rioxarray
import openpyxl
import math

import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms


import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import NaturalEarthFeature
import cartopy.io.shapereader as shpreader
import shapefile
import geopandas as gpd
from shapely import geometry
import os
import rioxarray

def adjust_sub_axes(ax_main, ax_sub, shrink):
    '''
    将ax_sub调整到ax_main的右下角. shrink指定缩小倍数.
    当ax_sub是GeoAxes时, 需要在其设定好范围后再使用此函数.
    '''
    bbox_main = ax_main.get_position()
    bbox_sub = ax_sub.get_position()
    # 使shrink=1时ax_main与ax_sub等宽或等高.
    if bbox_sub.width > bbox_sub.height:
        ratio = bbox_main.width / bbox_sub.width * shrink
    else:
        ratio = bbox_main.height / bbox_sub.height * shrink
    wnew = bbox_sub.width * ratio
    hnew = bbox_sub.height * ratio
    bbox_new = mtransforms.Bbox.from_extents(bbox_main.x1 - wnew, bbox_main.y0, bbox_main.x1, bbox_main.y0 + hnew)
    ax_sub.set_position(bbox_new)

path = r'C:\Users\28166\Desktop\SecondOutput_China_Grid_Population_Density_Y2000toY2021.nc'
data = nc.Dataset(path)
lat = np.array(data.variables['latitude'][:])  # from 15.05 to 59.95 by 0.1  (450)
lon = np.array(data.variables['longitude'][:])  # from 70.05 to 139.95 by 0.1  (700)
pop = np.array(data.variables['pop_density4each_age'][13:22, :, :, :])  # (year, age, lat, lon),(9, 18, 450, 700)
# pop中为0
# pop_nan = np.where(pop == 0, np.nan, pop)  # (9, 18, 450, 700)
# # 这步换成了nan
# print(pop_nan)
# print(pop_nan.shape)
# pop_new = np.nansum(pop_nan, axis=1)  # (9, 450, 700)
# 使用nansum之后又变成了0，所以一开始最后画图的时候还是用的有0的数据

pop_new = np.nansum(pop, axis=1)
pop_new = np.where(pop_new == 0, np.nan, pop_new)  # np数组替换为nan的语句

# 设置标题和字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
#  画图
proj = ccrs.PlateCarree()  # 创建地图投影
fig = plt.figure(figsize=(33.4, 20))
fig.suptitle('Space Distribution of Grid Population ', fontsize=20, weight='bold', y=0.94)  # 设置大图标题,y用来控制大标题的相对位置
axes_main = fig.subplots(3, 3, subplot_kw=dict(projection=proj))
axes_sub = fig.subplots(3, 3, subplot_kw=dict(projection=proj))
extent_main = [70, 140, 15, 55]
extents_sub = [105, 125, 0, 25]
fig.subplots_adjust(right=0.85)  # 设置色条
china = gpd.read_file("D:/各类文件/china_map/分省各级别shp（超全）/1. Country/country.shp")
nanhai = gpd.read_file("D:/各类文件/china_map/9duanxian/9duanxian.shp")
shengji = gpd.read_file("D:/各类文件/china_map/分省各级别shp（超全）/2. Province/province.shp")
nineduanxian = gpd.read_file("D:\各类文件\china_map\9duanxian\9duanxian.shp")

# colorlevel = np.linspace(0, 100000, 11)
# colordict = ['#FFFFFF', '#C2E8FA', '#86C5EB', '#5196CF', '#49A383', '#6ABF4A',  '#D9DE58', '#F8B246', '#F26429', '#DD3528', '#BC1B23', '#921519']
# color_map = mcolors.ListedColormap(colordict)
# norm = mcolors.BoundaryNorm(colorlevel, 12)
# norm = mcolors.Normalize(vmin=0, vmax=100000)
# 'inferno' ,


# color_list = ['#FFFFFF', '#FCBA1D', '#E35634', '#7A1C6C', '#110931']
# new_cmap = mcolors.LinearSegmentedColormap.from_list('new_cmap', color_list, 20)
norm = mpl.colors.Normalize(vmin=0, vmax=100000)


n = 2013
k = 0
for i in range(3):
    for j in range(3):
        axes_main[i, j].set_extent(extent_main, crs=proj)
        china.plot(ax=axes_main[i, j], color='None', edgecolor='gray', linewidths=1, zorder=3)
        shengji.plot(ax=axes_main[i, j], color='None', edgecolor='gray', linewidths=0.4, zorder=3)
        nineduanxian.plot(ax=axes_main[i, j], color='None', edgecolor='gray', linewidths=2, zorder=4)
        # 绘制填色图
        sc = axes_main[i, j].contourf(lon, lat, pop_new[k], cmap='cividis',  levels=np.linspace(0, 100000, 11),
                                      transform=proj, zorder=2)
        # 设置经纬度
        gl = axes_main[i, j].gridlines(crs=proj, draw_labels=True, linestyle=":", linewidth=0.1, x_inline=False,
                                       y_inline=False, color='k', alpha=0.5, xlines=False, ylines=False)
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
        gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度的格式
        gl.xlocator = mticker.FixedLocator([80, 90, 100, 110, 120, 130, 140])  # extent[0], extent[1]+0.5, 10
        gl.ylocator = mticker.FixedLocator([20, 30, 40, 50, 60])  # extent[2], extent[3]+0.5, 10
        gl.xlines = False
        gl.ylines = False
        font2 = {'size': 10, 'family': 'Times New Roman', 'weight': 'normal'}
        gl.xlabel_style = font2
        gl.ylabel_style = font2
        # 画南海及九段线
        axes_sub[i, j].set_extent(extents_sub, crs=proj)
        sc2 = axes_sub[i, j].contourf(lon, lat, pop_new[k], cmap='cividis', levels=np.linspace(0, 100000, 11),
                                      transform=proj, zorder=2)
        china.plot(ax=axes_sub[i, j], color='white', edgecolor='gray', zorder=0, linewidths=0.35)
        nanhai.plot(ax=axes_sub[i, j], color='gray', edgecolor='gray', zorder=1)
        nineduanxian.plot(ax=axes_main[i, j], color='None', edgecolor='gray', linewidths=2, zorder=3)
        adjust_sub_axes(axes_main[i, j], axes_sub[i, j], shrink=0.3)
        # 添加年份文字
        axes_main[i, j].text(0.03, 0.91, f'{n}', bbox={'facecolor': 'white', 'alpha': 1}, fontsize=8,
                             transform=axes_main[i, j].transAxes, color='k', weight='bold')

        cbar_ax = fig.add_axes([0.89, 0.15, 0.01, 0.7])  # 位置
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='cividis'),
                            cax=cbar_ax, format='%.2f', shrink=0.88, ticks=np.linspace(0, 100000, 11), drawedges=False, extend='neither')
        # drawedges=True,
        ax0 = cbar.ax  # 将colorbar变成一个新的ax对象，可通过ax对象的各种命令来调整colorbar
        ax0.set_title('  Population  (person)', fontproperties='Times New Roman', weight='normal', size=12, pad=20)
        ax0.tick_params(which='major', direction='in', labelsize=12, length=4)
        n = n + 1
        k = k + 1
plt.show()






















