# -*- coding: utf-8 -*-
# @Time    : 2024/7/4 13:00
# @Author  : ChenYuling
# @FileName: SI_Mapping.py
# @Software: PyCharm
# @Describe：决策树绘图

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

#忽略一些版本不兼容等警告
import warnings
warnings.filterwarnings("ignore")
#%%
# 输入的 Shapefile,在此确定是哪个植被区的范围，植被区
input_shapefile = r"G:\TH\Predict\China_tile\alltile.shp"#TODO:修改处1*******************************************************
# 所有输入数据的tif
all_tiff_directory = r"G:\TH\Xdata30m\1km_forPredict\mask"#所有tiff位置,预测数据集
# 输出目录
shp_directory = r"E:\TH\WN\20240627\HT\shp150" #每个植被区下用于保存shp要素 #TODO:修改处2************************************************************
tif_directory = r"E:\TH\WN\20240627\HT\tif150" #每个植被区下用于保存每个要素下掩膜的tif#TODO:修改处3************************************************************
preTIF_directory = r"E:\TH\WN\20240627\HT\preTIF150" #每个植被区下用于保存预测的tif#TODO:修改处4************************************************************

# 创建输出目录（如果不存在）
if not os.path.exists(shp_directory):
    os.makedirs(shp_directory)

# 创建保存目录（如果不存在）
if not os.path.exists(tif_directory):
    os.makedirs(tif_directory)

# 创建保存目录（如果不存在）
if not os.path.exists(preTIF_directory):
    os.makedirs(preTIF_directory)
#%%
# # #最开始需要根据渔网格对其分大块，每个植被区只需运行一次就行
# # #TODO: 使用 SplitByAttributes工具分割 Shapefile
split_field = "TileName" #分割字段
arcpy.analysis.SplitByAttributes(input_shapefile, shp_directory, split_field)

#%%
#子函数，读取整个tiff像元值
def read_tiff_pixels(tiff_file):
    dataset = gdal.Open(tiff_file)
    band = dataset.GetRasterBand(1)  # 获取第一个波段（索引从1开始）
    # 读取整个波段的像元值
    pixel_data = band.ReadAsArray()
    # 输出像元值
    # print(pixel_data)
    dataset = None  # 关闭数据集
    return pixel_data


#子函数，清空文件中所有文件
def clear_folder(folder_path):
    # 确保文件夹存在
    if not os.path.exists(folder_path):
        return
    # 遍历文件夹中的文件并删除
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)




#%%地形因子定性处理
# 定义坡向分类的函数
def classify_aspect(aspect):
    if 0 <= aspect < 45:
        return 'North'
    elif 45 <= aspect < 90:
        return 'Northeast'
    elif 90 <= aspect < 135:
        return 'East'
    elif 135 <= aspect < 180:
        return 'Southeast'
    elif 180 <= aspect < 225:
        return 'South'
    elif 225 <= aspect < 270:
        return 'Southwest'
    elif 270 <= aspect < 315:
        return 'West'
    elif 315 <= aspect <= 360:
        return 'Northwest'
    else:
        return ''  # 用于处理无效的aspect值

# 定义坡度分类的函数
def classify_slope(slope):
    if 0 <= slope <= 5:
        return 'FlatSlope'
    elif 6 <= slope <= 15:
        return 'GentleSlope'
    elif 16 <= slope <= 25:
        return 'Incline'
    elif 26 <= slope <= 35:
        return 'AbruptSlope'
    elif 36 <= slope <= 45:
        return 'SteepSlope'
    elif slope > 45:
        return 'DangerousSlope'
    else:
        return ''  # 用于处理无效的slope值

# 定义海拔分类的函数
def classify_elevation(elevation):
    if elevation < 1000:
        return 'Low'
    elif 1000 <= elevation <= 3500:
        return 'Medium'
    elif elevation > 3500:
        return 'Medium' #'High'
    else:
        return ''  # 用于处理无效的elevation值


#%% 定义trzd分类的函数
def classify_trzd(trzd):
    if trzd == '1':
        return 'CoarseTextured'
    elif trzd == '2':
        return 'MediumTextured'
    elif trzd == '3':
        return 'FineTextured'
    else:
        return ''  # 用于处理无效的trzd值


# 定义tchd分类的函数
def classify_tchd(tchd):
    if tchd == '10':
        return 'Middle' #ma训练数据有缺失'Thin'
    elif tchd == '30':
        return 'Middle'
    elif tchd == '100':
        return 'Thick'
    else:
        return ''


#%%
#######step2 循环不同shapefile下不同tif（按切割块读信息）
# 获取文件夹中的tif文件路径列表
tiff_files = [file for file in os.listdir(all_tiff_directory) if file.endswith('.tif')]

for filename in os.listdir(shp_directory):
    if filename.endswith(".shp"):
        sub_shapefile = os.path.join(shp_directory, filename) #子shp：每一块
        sub_shapefile_name = filename.split('.')[0]  #获取该一块shp名称
        print('shp:',sub_shapefile_name)
        # 创建数据框，保存以子shp命名
        sub_shapefile_df = pd.DataFrame()#用该shp块进行掩膜所有tif输入数据
        # 清空文件夹
        clear_folder(tif_directory)
        # 循环遍历tif文件,获取某一小块的tif数据
        for tif_file in tiff_files:
            tif_path = os.path.join(all_tiff_directory, tif_file)
            tif_name = os.path.basename(tif_path).split('.')[0]  # 当前tif名称os.path.basename(file_path).split('.')[0]
            # 创建保存的tif名及位置
            output_filename = str(tif_name) + ".tif"
            output_tiff = os.path.join(tif_directory, output_filename)
            # TODO:执行按掩膜提
            # print("Extract:", output_filename)
            arcpy.gp.ExtractByMask_sa(tif_path, sub_shapefile, output_tiff)#对应小块shp掩膜的所有tif保存临时目录在tif_directory文件里
        #
        #接下来读取掩膜获取的tif文件将其作为数据变量
        # 获取文件夹中的tif文件路径列表
        sub_tiff_files = [file for file in os.listdir(tif_directory) if file.endswith('.tif')]
        for subtif_file in sub_tiff_files:#该循环结束后，sub_shapefile_df数据框保存该小shp块下对应自变量列数据
            subtif_path = os.path.join(tif_directory, subtif_file)
            subtif_name = os.path.basename(subtif_path).split('.')[0]  # 当前tif名称os.path.basename(file_path).split('.')[0]
            # print("Read:",subtif_name)
            tif_pixel_data = read_tiff_pixels(subtif_path)  # 读取整个tiff像元值,保存二维数组行列号
            ### TODO: 读取tif转为一维数组
            block_data_row = tif_pixel_data.flatten()  # 转一维数组
            sub_shapefile_df[subtif_name] = block_data_row  # 新读取tif数据作为数据框一列

        ####TODO: 对子自变量数据框进行处理，作为预测数据。主要3方面：NoData、变量顺序、土壤名称和土壤质地的哑变量化
        sub_shapefile_df['ID'] = range(len(sub_shapefile_df)) #新增自增列
        # 遍历所有列进行判断，取出值不等于NoData_value的子集
        for column in sub_shapefile_df.columns:
            sub_shapefile_df[column] = sub_shapefile_df[column].replace(-340282346638528859811704183484516925440.00000, np.nan)  #将所有列中NoData代表的最大值替换为空值
        #其他变量空值替换
        sub_shapefile_df['slope'] = sub_shapefile_df['slope'].replace(-128, np.NaN)  #将NoData代表的最大值替换为空值
        sub_shapefile_df['elevation'] = sub_shapefile_df['elevation'].replace(65535, np.NaN)  #将NoData代表的最大值替换为空值
        sub_shapefile_df['aspect'] = sub_shapefile_df['aspect'].replace(65535, np.NaN)  # 将NoData代表的最大值替换为空值
        # sub_shapefile_df['age'] = sub_shapefile_df['age'].replace(65535, np.NaN)  # 将NoData代表的最大值替换为空值
        sub_shapefile_df['TRZD'] = sub_shapefile_df['TRZD'].replace(0, np.NaN)  #将NoData代表的最大值替换为空值
        sub_shapefile_df['TCHD'] = sub_shapefile_df['TCHD'].replace(-128, np.NaN)  # 将NoData代表的最大值替换为空值
        sub_shapefile_df['TRMC'] = sub_shapefile_df['TRMC'].replace(-128, np.NaN)  #将NoData代表的最大值替换为空值,trmc
        sub_shapefile_df['TRMC'] = sub_shapefile_df['TRMC'].replace(255, np.NaN)  #将NoData代表的最大值替换为空值,trmc
        sub_shapefile_df['pnf'] = sub_shapefile_df['pnf'].replace(127, np.NaN)  # 将NoData代表的最大值替换为空值
        #根据天然林和人工林pnf数据，选出森林需要预测的数据
        subset_pre = sub_shapefile_df[(sub_shapefile_df['pnf'] >0 ) & (sub_shapefile_df['pnf'] < 3)]
        #非森林区域，不需要预测
        subset_null = sub_shapefile_df[(sub_shapefile_df['pnf'] <=0 ) | (sub_shapefile_df['pnf'] >= 3)]

        # 重命名列名
        #调整数据与训练数据一致
        subset_pre0 = subset_pre[['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7', 'bio_8', 'bio_9',
                                  'bio_10','bio_11', 'bio_12', 'bio_13', 'bio_14', 'bio_15', 'bio_16', 'bio_17',
                                  'bio_18', 'bio_19','TCHD', 'TRMC', 'TRZD', 'aspect', 'elevation', 'slope', 'pnf']]

        # %读取参数值#气候因子进行降维
        df_p = pd.read_csv(r'E:\TH\WN\20240627\DATA\df_loadings.csv', encoding='gb2312')
        bio19 = subset_pre0[['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7', 'bio_8', 'bio_9', 'bio_10',
                             'bio_11','bio_12','bio_13', 'bio_14', 'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19']]
        bio19_m = bio19.values  # 获取气候数据矩阵
        prams = df_p.values  # 获取PCA载荷矩阵
        TPvalues = np.dot(bio19_m, prams)
        TPdf = pd.DataFrame(TPvalues)
        TPdf.columns = ['TP', 'T', 'P']
        # 重新设置索引并按列连接
        subset_pre0 = subset_pre0.reset_index(drop=True)
        TPdf = TPdf.reset_index(drop=True)
        subset_pre0A = pd.concat([subset_pre0, TPdf], axis=1)  # axis=1按照列
        # 选择需要的列
        subset_pre0B = subset_pre0A[['TP', 'T', 'P', 'TCHD', 'TRMC', 'TRZD', 'aspect', 'elevation', 'slope', 'pnf']]

        # %处理离散数据-哑变量处理
        # 将列转换为字符类型
        subset_pre0B['TRMC'] = subset_pre0B['TRMC'].round().astype('Int64')
        subset_pre0B['TRMC'] = subset_pre0B['TRMC'].astype(str)
        subset_pre0B['pnf'] = subset_pre0B['pnf'].round().astype('Int64')
        subset_pre0B['pnf'] = subset_pre0B['pnf'].astype(str)

        # %读取TRMC
        df_TRMC = pd.read_csv(r'E:\TH\WN\20240627\DATA\luo_TRMC.csv')  # ,encoding = 'gb2312'
        # 确保 TRMC 列的数据类型一致
        df_TRMC['TRMC'] = df_TRMC['TRMC'].round().astype('Int64').astype(str)
        # 使用 'TRMC' 字段进行左连接
        merged_df = pd.merge(subset_pre0B, df_TRMC, on='TRMC', how='left')

        # 应用分类函数并创建新列
        merged_df['aspect_str'] = merged_df['aspect'].apply(classify_aspect)
        merged_df['slope_str'] = merged_df['slope'].apply(classify_slope)
        merged_df['elevation_str'] = merged_df['elevation'].apply(classify_elevation)

        # % 将 'trzd' 列和 'tchd' 列转换为整数然后转换为字符串
        merged_df.loc[:, 'TRZD'] = merged_df['TRZD'].round().astype('Int64').astype(str)
        merged_df.loc[:, 'TCHD'] = merged_df['TCHD'].round().astype('Int64').astype(str)
        merged_df['tchd_str'] = merged_df['TCHD'].apply(classify_tchd)
        merged_df['trzd_str'] = merged_df['TRZD'].apply(classify_trzd)
        subset_pre5 = merged_df[['TP', 'T', 'P', 'aspect_str', 'slope_str', 'elevation_str',
                                 'tchd_str', 'trzd_str', 'trmc_str']]


        #%
        import csv
        def save_array_to_csv(array, file_path):
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(array)

        #%将数据分成10份
        import math
        from tqdm import tqdm
        import joblib
        def cut_df(df, n):
            list_pre = []#创建空列表用于存储整个block预测值
            df_num = len(df)
            every_epoch_num = math.floor((df_num/n))
            for index in tqdm(range(n)):
                print(index)
                df_name = f'subset_pre6{index}'
                if index < n-1:
                    df_tem = df[every_epoch_num * index: every_epoch_num * (index + 1)]
                else:
                    df_tem = df[every_epoch_num * index:]
                df_name = df_tem
                df_name = df_name[['TP', 'T', 'P','aspect_str', 'slope_str', 'elevation_str',
               'tchd_str','trzd_str', 'trmc_str']]
                df_name.fillna(method='ffill', inplace=True)
                df_name.to_csv("E:/TH/WN/20240627/DATA/X_TRAIN150.csv")
                print(len(df_name))
                #看分割的子数据框是否为空？
                if len(df_name) == 0:
                    continue
                else:

                    # pre_sub= df_name.values
                    ###TODO:模型预测
                    # %#####调用R语言的决策树模型###########
                    import rpy2.robjects as ro
                    from rpy2.robjects import pandas2ri
                    from rpy2.robjects.packages import importr
                    import pandas as pd
                    # 激活 pandas2ri
                    pandas2ri.activate()
                    print("激活成功！")

                    # % 导入 R 包
                    base = importr('base')
                    utils = importr('utils')
                    rpart = importr('rpart')
                    rpart_plot = importr('rpart.plot')
                    rattle = importr('rattle')
                    caret = importr('caret')
                    pROC = importr('pROC')
                    # 设置工作目录
                    ro.r('setwd("E:/TH/WN/20240627/DATA")')
                    # 读取数据并移除 NA 值
                    ro.r('''
                    data <- read.csv("X_TRAIN150.csv")
                    subdata <- data[, c('TP', 'T', 'P', 'trmc_str', 'trzd_str', 'tchd_str',
                                                'aspect_str', 'elevation_str', 'slope_str')]
                    ''')
                    # % 读取模型文件
                    fit2 = ro.r('fit2 <- readRDS("fit2_150model.rds")')

                    # % 预测并将结果添加到 subdata 数据框
                    ro.r('''
                    predictions <- predict(fit2, subdata, type = "class")
                    subdata$predictions <- predictions
                    write.csv(subdata, file = "subdata_with_predictions150.csv", row.names = FALSE)
                    ''')

                    # %读取已经预测好的数据
                    subdata_with_predictions = pd.read_csv(r'E:\TH\WN\20240627\DATA\subdata_with_predictions150.csv',encoding = 'gb2312')

                    y_pred = subdata_with_predictions["predictions"].values
                    # # 保存数组到CSV文件
                    # save_array_to_csv(y_pred, 'y_pred.csv')
                    y_pred_list = y_pred.tolist()
                    list_pre= list_pre + y_pred_list
                    print(y_pred)
            return list_pre
        #查看读取块的数据框是否为空？
        if len(subset_pre5)==0:
            continue
        else:
            subset_pre5.replace('', np.nan, inplace=True)
            # 计算每列的众数
            mode_values = subset_pre5.mode().iloc[0]
            # 使用每列的众数填充空值
            subset_pre6 = subset_pre5.fillna(mode_values)
            # 检查是否有空值
            if subset_pre6.isnull().values.any():
                pre_y_block = [0] * len(subset_pre6)
            else:
                pre_y_block = cut_df(subset_pre6, 1)
            # pre_y_block = cut_df(subset_pre6, 1)#非空数据集预测结果
        # pre_y_block = cut_df(subset_pre6, 1)#非空数据集预测结果
        #%将预测和空值数据合并在一起
        if len(subset_null) ==len(sub_shapefile_df):#预测全为空
            pre_2 = subset_null[['ID']]
            pre_2['preHT'] = np.NaN
            pre = pre_2
        else:
            pre_1 = subset_pre[['ID']]
            pre_1.loc[:,'preHT'] = pre_y_block
            pre_2 = subset_null[['ID']]
            pre_2['preHT'] = np.NaN
            pre = pd.concat([pre_1, pre_2], ignore_index=True)

        sorted_pre = pre.sort_values('ID')
        #%
        # ###TODO:将数组转换为预测小tif
        basepathtif = r"E:\TH\WN\20240627\HT\tif150\pnf.tif" #TODO:修改处5************************************************************
        # 读取tif文件
        origin_dataset = gdal.Open(basepathtif)
        # 获取数据集的行数和列数(维度和numpy数组相反)
        rows = origin_dataset.RasterYSize
        cols = origin_dataset.RasterXSize
        ###TODO:处理像元值数据
        # 提取某一列的值作为一维数组
        pre_values = sorted_pre['preHT'].values
        # 将一维数组转换为二维数组
        pre_2d = np.reshape(pre_values, (rows, cols))

        # 将字符串数据转换为整数数据
        # 这里假设 'SI1' 对应的整数是 1，'SI2' 对应的整数是 2，'SI0' 对应的整数是 0
        def convert_to_integer(data):
            conversion_dict = {'SI1': 1, 'SI2': 2, 'SI3': 3}
            return np.vectorize(lambda x: conversion_dict.get(x, np.nan))(data)
        pre_2d_int = convert_to_integer(pre_2d)
        # 创建保存的tif名及位置
        output_filename = str(sub_shapefile_name) + '_preHT.tif'
        # output_filename = 't1_preAGE.tif'
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
        new_dataset = driver.Create(output_tiff, pre_2d_int.shape[1], pre_2d_int.shape[0], 1, gdal.GDT_Int32)

        # 读取之前的tif信息，为新生成的tif添加地理信息
        # 如果不增加这一步，则生成的图片没有经纬度坐标、投影的信息
        # 获取投影信息
        origin_proj = origin_dataset.GetProjection()
        new_dataset.SetProjection(origin_proj)
        # 仿射矩阵
        origin_geotrans = origin_dataset.GetGeoTransform()
        new_dataset.SetGeoTransform(origin_geotrans)

        band = new_dataset.GetRasterBand(1)
        band.WriteArray(pre_2d_int)
        print(new_dataset)
        new_data = new_dataset.ReadAsArray()
        print(new_data)

        # 关闭数据集
        origin_dataset = None
        new_dataset = None
        print("保存成功！")

#%%###TODO:循环结束后，一个植被区所有子块都生成tif，将文件夹预测小tif进行合并形成一个植被区预测tif

