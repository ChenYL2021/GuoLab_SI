# -*- coding: utf-8 -*-
# @Time    : 2024/5/11 15:17
# @Author  : ChenYuling
# @FileName: latFIG.py
# @Software: PyCharm
# @Describe：LAT FIG


#%%
import arcpy
import os
from arcpy.sa import *
import pandas as pd
import numpy as np
from osgeo import gdal
import shutil
import joblib
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from catboost import CatBoostRegressor
#忽略一些版本不兼容等警告
import warnings
warnings.filterwarnings("ignore")
#%%
# 输入的 Shapefile,在此确定是哪个植被区的范围，植被区
input_shapefile = r"E:\20240511\lat\lat100.shp"#TODO:修改处1*******************************************************
# 输出目录
shp_directory = r"E:\20240511\lat\shp" #每个植被区下用于保存shp要素 #TODO:修改处2************************************************************
tif_directory = r"E:\20240511\lat\tif" #每个植被区下用于保存每个要素下掩膜的tif#TODO:修改处3************************************************************

# 创建输出目录（如果不存在）
if not os.path.exists(shp_directory):
    os.makedirs(shp_directory)

# 创建保存目录（如果不存在）
if not os.path.exists(tif_directory):
    os.makedirs(tif_directory)


#%%
# #最开始需要根据渔网格对其分大块，每个植被区只需运行一次就行
# #TODO: 使用 SplitByAttributes工具分割 Shapefile
split_field = "Id" #分割字段
arcpy.analysis.SplitByAttributes(input_shapefile, shp_directory, split_field)
