# -*- coding: utf-8 -*-
# @Time        : 2024/8/8 13:12
# @Author      : ChenYuling
# @File        : SI_MappingB
# @Desc        : 基于SI_MappingA，实现不分块的预测
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
import shutil
#忽略一些版本不兼容等警告
import warnings
warnings.filterwarnings("ignore")
#%%
TREE = "110"
Timeplist = ['2021_2040','2041_2060','2061_2080','2081_2100']
ssplist = ['ssp585','ssp245','ssp126']
GCMlist = ['MPI_ESM1_2_HR','MRI_ESM2_0','GISS_E2_1_G']#

def copy_tif_files(src_folder, dst_folder):
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(src_folder):
        # 只处理以 .tif 结尾的文件
        if filename.endswith('.tif'):
            # 构造完整的源文件路径和目标文件路径
            src_file = os.path.join(src_folder, filename)
            dst_file = os.path.join(dst_folder, filename)

            # 复制文件到目标文件夹
            shutil.copy2(src_file, dst_file)
            print(f'复制文件: {src_file} 到 {dst_file}')

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


# 嵌套循环遍历 GCMlist, Timeplist 和 ssplist
for GCM in GCMlist:
    for Timep in Timeplist:
        for ssp in ssplist:
            print(f'GCM: {GCM}, Timep: {Timep}, ssp: {ssp}')

            # Timep = '2081_2100'
            # ssp = 'ssp126'
            # GCM = 'MPI_ESM1_2_HR'

            # ssp585   ssp245  ssp126  MPI_ESM1_2_HR  MRI_ESM2_0 GISS_E2_1_G  2021_2040 2041_2060  2061_2080 2081_2100
            # 所有输入数据的tif
            all_tiff_directory = fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\China\mask"#所有tiff位置,预测数据集
            # all_tiff_directory = r"E:\TH\Xdata30m\1km_forPredict\mask"#所有tiff位置,预测数据集
            # all_tiff_directory = r"E:\CIMP6\mask_boundary\tif_xz"#所有tiff位置,预测数据集

            # 合并所有tif
            src_folder = 'E:\CIMP6\mask_other'
            copy_tif_files(src_folder, all_tiff_directory)

            # 输出目录
            preTIF_directory = fr"E:\CIMP6\FIGS\{GCM}\{Timep}" #每个植被区下用于保存预测的tif#TODO:修改处4************************************************************
            # 创建保存目录（如果不存在）
            if not os.path.exists(preTIF_directory):
                os.makedirs(preTIF_directory)


            #接下来读取掩膜获取的tif文件将其作为数据变量
            # 创建数据框，保存以子shp命名
            sub_shapefile_df = pd.DataFrame()#用该shp块进行掩膜所有tif输入数据
            # 获取文件夹中的tif文件路径列表
            sub_tiff_files = [file for file in os.listdir(all_tiff_directory) if file.endswith('.tif')]
            for subtif_file in sub_tiff_files:#该循环结束后，sub_shapefile_df数据框保存该小shp块下对应自变量列数据
                subtif_path = os.path.join(all_tiff_directory, subtif_file)
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
            # 调整数据与训练数据一致
            subset_pre0 = subset_pre[['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7', 'bio_8', 'bio_9', 'bio_10',
                                      'bio_11', 'bio_12', 'bio_13', 'bio_14', 'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19',
                                      'VH', 'VV', 'TCHD', 'TRMC', 'TRZD', 'aspect', 'elevation', 'slope', 'pnf', 'NDVI_MAX']]
            # 看subset_pre0数据框是否为空？
            if len(subset_pre0) == 0:
                subset_pre01 = subset_pre0
            else:
                # %空值处理‘’视为空值
                # subset_pre0.loc[:, 'pnf'] = subset_pre0['pnf'].apply(lambda x: np.nan if x == 0 else x)
                subset_pre0.replace('', np.nan, inplace=True)
                # 计算每列的众数
                mode_values = subset_pre0.mode().iloc[0]
                # 使用每列的众数填充空值
                subset_pre01 = subset_pre0.fillna(mode_values)
            # %对于土壤名称和土壤质地取值情况问题
            # 读取 trmc属性表CSV 文件，获取其土壤名称
            trmc_csv_df = pd.read_csv('./DATA/sym90.csv', sep=',')
            # 左连接两个数据框,添加土壤名称TRMC
            subset_pre1 = pd.merge(subset_pre01, trmc_csv_df, on='TRMC', how='left')
            # 删除trmc字段
            subset_pre2 = subset_pre1.drop('trmc', axis=1)  # 保留soil
            # %处理离散数据-哑变量处理
            # 将列转换为字符类型
            subset_pre2['TRZD'] = subset_pre2['TRZD'].round().astype('Int64')
            subset_pre2['TRZD'] = subset_pre2['TRZD'].astype(str)
            subset_pre2['pnf'] = subset_pre2['pnf'].round().astype('Int64')
            subset_pre2['pnf'] = subset_pre2['pnf'].astype(str)

            subset_pre5 = pd.get_dummies(
                subset_pre2,
                columns=['TRZD', 'SU_SYM90', 'pnf'],
                prefix=['TRZD', 'TRMC', 'PNF'],
                prefix_sep="_",
                dtype=int,
                dummy_na=False,
                drop_first=False
            )
            subset_pre5["S_age"] = 30  # TODO:计算SI的标准年龄
            # %训练数据框列明
            train_columns = ['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7',
       'bio_8', 'bio_9', 'bio_10', 'bio_11', 'bio_12', 'bio_13', 'bio_14',
       'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19', 'T2', 'VH', 'VV',
       'tchd', 'aspect', 'elevation', 'slope', 'NDVI_MAX', 'TRZD_1', 'TRZD_2',
       'TRZD_3', 'TRMC_ACu', 'TRMC_ALh', 'TRMC_CHh', 'TRMC_CHk', 'TRMC_CHl',
       'TRMC_CMc', 'TRMC_CMd', 'TRMC_CMe', 'TRMC_CMi', 'TRMC_GLm', 'TRMC_GRh',
       'TRMC_LP', 'TRMC_LPe', 'TRMC_LPi', 'TRMC_LPm', 'TRMC_LVa', 'TRMC_LVh',
       'TRMC_LVk', 'TRMC_PDd', 'TRMC_PDj', 'TRMC_PHc', 'TRMC_PHg', 'TRMC_PLe',
       'TRMC_RGe', 'PNF_1', 'PNF_2']

            for column in train_columns:  # 主要解决一些预测字段（土壤）不在训练字段中
                # 判断字符串是否在列表中
                if column in subset_pre5.columns:
                    print(f"{column} 存在于训练特征表中")
                else:
                    print(f"{column} 不存在于训练特征表中")
                    subset_pre5[column] = 0


            subset_pre6 = subset_pre5[train_columns]
            # %将数据分成10份
            import math
            from tqdm import tqdm
            import joblib
            def cut_df(df, n):
                list_pre = []  # 创建空列表用于存储整个block预测值
                df_num = len(df)
                every_epoch_num = math.floor((df_num / n))
                for index in tqdm(range(n)):
                    print(index)
                    df_name = f'subset_pre6{index}'
                    if index < n - 1:
                        df_tem = df[every_epoch_num * index: every_epoch_num * (index + 1)]
                    else:
                        df_tem = df[every_epoch_num * index:]
                    df_name = df_tem
                    df_name = df_name[train_columns]
                    # df_name.to_csv("X_TRAIN.csv")
                    print(len(df_name))
                    # 看分割的子数据框是否为空？
                    if len(df_name) == 0:
                        continue
                    else:
                        pre_sub = df_name.values
                        ###TODO:模型预测
                        # 加载模型
                        loaded_model1 = joblib.load(r"F:\WorkingNotes\TH\WN\20240729\THdata\MODEL3\Cat{}.model".format(TREE))
                        # 使用模型对测试数据进行预测
                        y_pred = loaded_model1.predict(pre_sub)
                        # # 保存数组到CSV文件
                        # save_array_to_csv(y_pred, 'y_pred.csv')
                        y_pred_list = y_pred.tolist()
                        list_pre = list_pre + y_pred_list
                        print(y_pred)
                return list_pre

            # %将预测和空值数据合并在一起
            if len(subset_null) == len(sub_shapefile_df):  # 预测全为空
                pre_2 = subset_null[['ID']]
                pre_2['preSI'] = np.NaN
                pre = pre_2
            else:
                pre_y_block = cut_df(subset_pre6, 1)  # 非空数据集预测结果
                pre_1 = subset_pre[['ID']]
                pre_1.loc[:, 'preSI'] = pre_y_block
                pre_2 = subset_null[['ID']]
                pre_2['preSI'] = np.NaN
                pre = pd.concat([pre_1, pre_2], ignore_index=True)

            sorted_pre = pre.sort_values('ID')
            # ###TODO:将数组转换为预测小tif
            basetifname = 'pnf.tif'
            basepathtif = os.path.join(all_tiff_directory, basetifname)
            # 读取tif文件
            origin_dataset = gdal.Open(basepathtif)
            # 获取数据集的行数和列数(维度和numpy数组相反)
            rows = origin_dataset.RasterYSize
            cols = origin_dataset.RasterXSize
            ###TODO:处理像元值数据
            # 提取某一列的值作为一维数组
            preAGE_values = sorted_pre['preSI'].values
            # 将一维数组转换为二维数组
            preAGE_2d = np.reshape(preAGE_values, (rows, cols))
            # %
            # 创建保存的tif名及位置
            output_filename = fr'preSI{TREE}_{ssp}.tif'
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
            band.WriteArray(preAGE_2d)
            print(new_dataset)
            new_data = new_dataset.ReadAsArray()
            print(new_data)

            # 关闭数据集
            origin_dataset = None
            new_dataset = None
            print("保存成功！")
