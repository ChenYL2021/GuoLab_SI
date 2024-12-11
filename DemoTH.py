# -*- coding: utf-8 -*-
# @Time    : 2024/5/21 11:38
# @Author  : ChenYuling
# @FileName: DemoTH.py
# @Software: PyCharm
# @Describe：以林分平均高为例，演示模型的尺度上推

#%%
import arcpy
import os
from arcpy.sa import *
import pandas as pd
import numpy as np
from osgeo import gdal
import shutil
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from math import sqrt
#%忽略一些版本不兼容等警告
import warnings
warnings.filterwarnings("ignore")
### TODO: FIT MODEL
######################step1 train#####################################
#以随机森林为例进行建模
#%%read data
df1 = pd.read_csv('./DATA/demoTH.csv', sep=',')
trainX = df1[['AGE', 'bio1', 'bio4', 'bio12', 'bio15', 'slope',
       'elevation', 'aspect']]

trainY = df1[['TH']]

#%%
from sklearn.model_selection import train_test_split, KFold
X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, train_size=2/3, random_state=2024)  # 数据集划分

#%% HistGradientBoostingRegressor####################
HistGB_model = HistGradientBoostingRegressor(random_state=2024)
HistGB_model.fit(X_train, y_train)
HistGB_predict = HistGB_model.predict(X_test)
HistGB_predict1 = HistGB_model.predict(X_train)
result = pd.DataFrame([['HistGradientBoostingRegressor',r2_score(y_train, HistGB_predict1), sqrt(mean_squared_error(y_train, HistGB_predict1)),r2_score(y_test, HistGB_predict),sqrt(mean_squared_error(y_test, HistGB_predict))]],columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
print(result)
#%%
#将训练的模型保存到磁盘(value=模型名)   默认当前文件夹下
joblib.dump(filename = r"./model/demoTH.model",value=HistGB_model)

# TODO: MAPPING
###############################################################################
#%%
sub_shapefile = r"G:\TH\quhua\zones8\CT.shp"
# 所有输入数据的tif
all_tiff_directory = r"E:\demo_CK\DATA\xTIF"#所有tiff位置,预测数据集
# 输出目录
preTIF_directory = r"E:\demo_CK\DATA\PreTIF1" #用于保存预测的tif
tif_directory = r"E:\demo_CK\DATA\outTIF"

# 创建保存目录（如果不存在）
if not os.path.exists(preTIF_directory):
    os.makedirs(preTIF_directory)
#
# # 创建保存目录（如果不存在）
# if not os.path.exists(tif_directory):
#     os.makedirs(tif_directory)

#%%根据矢量shp先
# 循环遍历tif文件,获取某一小块的tif数据
tiff_files = [file for file in os.listdir(all_tiff_directory) if file.endswith('.tif')]
# 循环遍历tif文件,获取某一小块的tif数据
for tif_file in tiff_files:
    tif_path = os.path.join(all_tiff_directory, tif_file)
    tif_name = os.path.basename(tif_path).split('.')[0]  # 当前tif名称os.path.basename(file_path).split('.')[0]
    # 创建保存的tif名及位置
    output_filename = str(tif_name) + ".tif"
    output_tiff = os.path.join(tif_directory, output_filename)
    # TODO:执行按掩膜提
    print("Extract:", output_filename)
    arcpy.gp.ExtractByMask_sa(tif_path, sub_shapefile, output_tiff)#对应小块shp掩膜的所有tif保存临时目录在tif_directory文件里
#

#%%
#子函数，读取整个tiff像元值
def read_tiff_pixels(tiff_file):
    dataset = gdal.Open(tiff_file)
    band = dataset.GetRasterBand(1)  # 获取第一个波段（索引从1开始）
    # 读取整个波段的像元值
    pixel_data = band.ReadAsArray()
    # 输出像元值
    dataset = None  # 关闭数据集
    return pixel_data
#%%
x_df = pd.DataFrame()#读取所有tif输入数据，保存数据框
# 获取文件夹中的tif文件路径列表
sub_tiff_files = [file for file in os.listdir(tif_directory) if file.endswith('.tif')]
for subtif_file in sub_tiff_files:#该循环结束后，sub_shapefile_df数据框保存该小shp块下对应自变量列数据
    subtif_path = os.path.join(tif_directory, subtif_file)
    subtif_name = os.path.basename(subtif_path).split('.')[0]  # 当前tif名称os.path.basename(file_path).split('.')[0]
    print("Read:",subtif_name)
    tif_pixel_data = read_tiff_pixels(subtif_path)  # 读取整个tiff像元值,保存二维数组行列号
    ### TODO: 读取tif转为一维数组
    block_data_row = tif_pixel_data.flatten()  # 转一维数组
    x_df[subtif_name] = block_data_row  # 新读取tif数据作为数据框一列
#%%
# 遍历所有列进行判断，取出值不等于NoData_value的子集
for column in x_df.columns:
    x_df[column] = x_df[column].replace(-340282346638528859811704183484516925440.00000, np.NaN)
#其他变量空值替换
x_df['slope'] = x_df['slope'].replace(-128, np.NaN)  #将NoData代表的最大值替换为空值
x_df['elevation'] = x_df['elevation'].replace(65535, np.NaN)  #将NoData代表的最大值替换为空值
x_df['aspect'] = x_df['aspect'].replace(65535, np.NaN)  # 将NoData代表的最大值替换为空值
x_df['age'] = x_df['age'].replace(65535, np.NaN)  # 将NoData代表的最大值替换为空值
x_df['pnf'] = x_df['pnf'].replace(127, np.NaN)  # 将NoData代表的最大值替换为空值
#%%
x_df['ID'] = range(len(x_df)) #新增自增列
#根据天然林和人工林pnf数据，选出森林需要预测的数据
subset_pre = x_df[(x_df['pnf'] >0 ) & (x_df['pnf'] < 3)]
#非森林区域，不需要预测
subset_null = x_df[(x_df['pnf'] <=0 ) | (x_df['pnf'] >= 3)]

#调整数据与训练数据一致
pre_x = subset_pre[['age', 'bio_1', 'bio_4', 'bio_12', 'bio_15', 'slope', 'elevation', 'aspect']]
pre_xvalues = pre_x.values
# 加载模型
loaded_model = joblib.load(filename=r"./model/demoTH.model")
# 使用模型对测试数据进行预测
y_pred = loaded_model.predict(pre_xvalues)
y_pred_list = y_pred.tolist()
#%%

# %将预测和空值数据合并在一起
if len(subset_null) == len(x_df):  # 预测全为空
    pre_2 = subset_null[['ID']]
    pre_2['preTH'] = np.NaN
    pre = pre_2
else:
    pre_1 = subset_pre[['ID']]
    pre_1.loc[:, 'preTH'] = y_pred_list
    pre_2 = subset_null[['ID']]
    pre_2['preTH'] = np.NaN
    pre = pd.concat([pre_1, pre_2], ignore_index=True)

sorted_pre = pre.sort_values('ID')
#%%
# ###TODO:将数组转换为预测小tif
basepathtif = r"E:\demo_CK\DATA\outTIF\pnf.tif"
# 读取tif文件
origin_dataset = gdal.Open(basepathtif)
# 获取数据集的行数和列数(维度和numpy数组相反)
rows = origin_dataset.RasterYSize
cols = origin_dataset.RasterXSize
###TODO:处理像元值数据
# 提取某一列的值作为一维数组
preAGE_values = sorted_pre['preTH'].values
# 将一维数组转换为二维数组
preAGE_2d = np.reshape(preAGE_values, (rows, cols))
# %
# 创建保存的tif名及位置
output_filename = 'preTH.tif'
output_tiff = os.path.join(preTIF_directory, output_filename)
'''
driver.Create(filename, xsize, ysize, [bands], [data_type], [options])
filename是要创建的数据集的路径。
xsize是新数据集中的列数。
ysize是新数据集中的行数。
bands是新数据集中的波段数。默认值为1。
data_type是将存储在新数据集中的数据类型。默认值为GDT_Byte。
options是创建选项字符串的列表。可能的值取决于正在创建的数据集的类型。
'''
# 创建tif文件
driver = gdal.GetDriverByName('GTiff')
# 维度和numpy数组相反
new_dataset = driver.Create(output_tiff, preAGE_2d.shape[1], preAGE_2d.shape[0], 1, gdal.GDT_Float32)

# 读取之前的tif信息，为新生成的tif添加地理信息
# 如果不增加这一步，则生成的图片没有经纬度坐标、投影的信息
# 获取投影信息
origin_proj = origin_dataset.GetProjection()
new_dataset.SetProjection(origin_proj)
# 仿射矩阵
origin_geotrans = origin_dataset.GetGeoTransform()
new_dataset.SetGeoTransform(origin_geotrans)

band = new_dataset.GetRasterBand(1)
band.WriteArray(preAGE_2d)(new_dataset)
new_data = new_dataset.ReadAsArray()
print(new_data)

# 关闭数据集
origin_dataset = None
new_dataset = None
print("保存成功！")
print