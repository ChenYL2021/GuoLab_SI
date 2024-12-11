# -*- coding: utf-8 -*-
# @Time    : 2024/4/29 22:19
# @Author  : ChenYuling
# @FileName: Climate.py
# @Software: PyCharm
# @Describe：TERRACLIMATE data
# step1 根据.nc文件提取各月份数据
import arcpy
import os
# 设置年份
year = 2021
# 设置变量
var = "soil"
# 设置工作环境
arcpy.env.workspace = f"F:\DB\TH\Xdata\TerraClimate2021_2023\{var}"

#%%
arcpy.md.MakeNetCDFRasterLayer(f"TerraClimate_{var}_{year}.nc", f"{var}", "lon", "lat", f"{var}_Layer", "time", None, "BY_VALUE", "CENTER")#pet_Layer#TODO:修改处1TerraClimate_pet_2023

TIF_directory = f"F:\DB\TH\Xdata\TerraClimate2021_2023\{var}\{year}" #每个植被区下用于保存预测的tif#TODO:修改处2***2023
# 创建输出目录（如果不存在）
if not os.path.exists(TIF_directory):
    os.makedirs(TIF_directory)

for i in range(1, 13):
    print(i)
    output_filename=f"{var}_{year}_"+ str(i) + ".tif"#TODO:修改处3********2023
    arcpy.management.MakeRasterLayer(f"{var}_Layer", output_filename, '', '-180 -90 180 90 GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]', i)
    # 使用CopyRaster工具复制数据到新的TIFF文件中
    output_tiff = os.path.join(TIF_directory, output_filename)
    arcpy.management.CopyRaster(output_filename, output_tiff)
#%%step2,计算多个tif的空间均值、方差、最大值，生成对应统计值的tif
# coding=utf-8
import arcpy
from arcpy import env
from arcpy.sa import *

# Set environment settings
# 设置年份
year = 2021
# 设置变量
var = "soil"
# 输入工作空间文件夹（即存放需批处理tif影像的文件夹）
env.workspace = f"F:\DB\TH\Xdata\TerraClimate2021_2023\{var}\{year}"
# env.workspace = "D:/DATA/2000" # 注意此处‘/’的方向
# Set local variables
# 遍历工作空间中的tif格式数据
rasters = arcpy.ListRasters("*", "tif")

# Check out the ArcGIS Spatial Analyst extension license
arcpy.CheckOutExtension("Spatial")

# MEAN均值；SUM总和；STD标准差；MINIMUM最小值；MAXIMUM最大值；
outCellStatistics = CellStatistics(rasters, "MEAN", "DATA")
# 输出结果影像的路径和名称
outCellStatistics.save(f"{var}_{year}_mean.tif")
print("All project is OK！")
#%%


