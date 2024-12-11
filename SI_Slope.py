# -*- coding: utf-8 -*-
# @Time        : 2024/9/18 9:11
# @Author      : ChenYuling
# @File        : SI_Slope
# @Desc        : 计算5期数据对应的斜率,及其斜率显著性

from scipy.stats import linregress
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
aTIF_directory = r"E:\CIMP6\FIGS\DataSlope"
# 创建保存目录（如果不存在）
if not os.path.exists(aTIF_directory):
    os.makedirs(aTIF_directory)

pTIF_directory = r"E:\CIMP6\FIGS\Datap_value"
# 创建保存目录（如果不存在）
if not os.path.exists(pTIF_directory):
    os.makedirs(pTIF_directory)

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
"""
#%%
TREE = "230"
ssplist = ['ssp585','ssp245','ssp126']
# ssp = 'ssp126'
for ssp in ssplist:
    print(f'TREE: {TREE}, ssp: {ssp}')
    #%5期数据路径
    Time2020 = fr"E:\CIMP6\FIGS\DataCurrent2020\preSI{TREE}.tif"
    Time2030 = fr"E:\CIMP6\FIGS\Ensemble_3GCMs\2021_2040\preSI{TREE}_{ssp}.tif"
    Time2050 = fr"E:\CIMP6\FIGS\Ensemble_3GCMs\2041_2060\preSI{TREE}_{ssp}.tif"
    Time2070 = fr"E:\CIMP6\FIGS\Ensemble_3GCMs\2061_2080\preSI{TREE}_{ssp}.tif"
    Time2090 = fr"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\preSI{TREE}_{ssp}.tif"


    # 接下来读取掩膜获取的tif文件将其作为数据变量# TODO: 读取tif转为一维数组,创建数据框
    Timesdf = pd.DataFrame()  # 用该shp块进行掩膜所有tif输入数据

    tif_pixel2020 = read_tiff_pixels(Time2020)  # 读取整个tiff像元值,保存二维数组行列号
    block2020 = tif_pixel2020.flatten()  # 转一维数组
    Timesdf['Time2020'] = block2020  # 新读取tif数据作为数据框一列

    tif_pixel2030 = read_tiff_pixels(Time2030)  # 读取整个tiff像元值,保存二维数组行列号
    block2030 = tif_pixel2030.flatten()  # 转一维数组
    Timesdf['Time2030'] = block2030  # 新读取tif数据作为数据框一列

    tif_pixel2050 = read_tiff_pixels(Time2050)  # 读取整个tiff像元值,保存二维数组行列号
    block2050 = tif_pixel2050.flatten()  # 转一维数组
    Timesdf['Time2050'] = block2050  # 新读取tif数据作为数据框一列

    tif_pixel2070 = read_tiff_pixels(Time2070)  # 读取整个tiff像元值,保存二维数组行列号
    block2070 = tif_pixel2070.flatten()  # 转一维数组
    Timesdf['Time2070'] = block2070  # 新读取tif数据作为数据框一列

    tif_pixel2090 = read_tiff_pixels(Time2090)  # 读取整个tiff像元值,保存二维数组行列号
    block2090 = tif_pixel2090.flatten()  # 转一维数组
    Timesdf['Time2090'] = block2090  # 新读取tif数据作为数据框一列

    #% 新增自增列
    Timesdf['ID'] = range(len(Timesdf))
    # 遍历所有列进行判断，取出值不等于NoData_value的子集
    for column in Timesdf.columns:
        Timesdf[column] = Timesdf[column].replace(-340282346638528859811704183484516925440.00000, np.nan)
    # 根据 2020 年数据，选出 slope 需要计算的数据 (Time2020 > 0 且不为空)
    subset_pre = Timesdf[(Timesdf['Time2020'] > 0) & (Timesdf['Time2020'].notnull())]
    # 非计算区域 (Time2020 <= 0 或者 Time2020 为空)
    subset_null = Timesdf[(Timesdf['Time2020'] <= 0) | (Timesdf['Time2020'].isnull())]

    subset_cal = subset_pre[['Time2020', 'Time2030', 'Time2050', 'Time2070', 'Time2090']]
    #% 定义时间序列
    time_points = [2020, 2030, 2050, 2070, 2090]

    # 初始化存储a, b和p值的列表
    a_values = []
    b_values = []
    p_values = []

    # 对每一行数据进行拟合
    for index, row in subset_cal.iterrows():
        # 获取这一行对应的y值
        y_values = row.values
        # 进行线性回归拟合
        slope, intercept, r_value, p_value, std_err = linregress(time_points, y_values)
        # 保存结果
        a_values.append(slope)  # 斜率
        # b_values.append(intercept)  # 截距
        p_values.append(p_value)  # p值

    # # 将结果添加到数据框中
    # subset_cal['a'] = a_values
    # # subset_cal['b'] = b_values
    # subset_cal['p_value'] = p_values

    #%将计算和空值数据合并在一起
    pre_1 = subset_pre[['ID']]
    pre_1.loc[:, 'a'] = a_values
    pre_2 = subset_null[['ID']]
    pre_2['a'] = np.NaN
    pre = pd.concat([pre_1, pre_2], ignore_index=True)

    sorted_pre = pre.sort_values('ID')

    #%##TODO:将数组转换为预测小tif
    # 读取tif文件
    origin_dataset = gdal.Open(Time2020)
    # 获取数据集的行数和列数(维度和numpy数组相反)
    rows = origin_dataset.RasterYSize
    cols = origin_dataset.RasterXSize
    ###TODO:处理像元值数据
    # 提取某一列的值作为一维数组
    preA_values = sorted_pre['a'].values
    # 将一维数组转换为二维数组
    preA_2d = np.reshape(preA_values, (rows, cols))
    # %
    # 创建保存的tif名及位置
    output_filename = fr'preSI{TREE}_{ssp}.tif'
    output_tiff = os.path.join(aTIF_directory, output_filename)
    '''
    # driver.Create(filename, xsize, ysize, [bands], [data_type], [options])
    # filename是要创建的数据集的路径。
    # xsize是新数据集中的列数。
    # ysize是新数据集中的行数。
    # bands是新数据集中的波段数。默认值为1。
    # data_type是将存储在新数据集中的数据类型。默认值为GDT_Byte。
    # options是创建选项字符串的列表。可能的值取决于正在创建的数据集的类型。
    '''
    # 创建tif文件
    driver = gdal.GetDriverByName('GTiff')
    # 维度和numpy数组相反
    new_dataset = driver.Create(output_tiff, preA_2d.shape[1], preA_2d.shape[0], 1, gdal.GDT_Float32)

    # 读取之前的tif信息，为新生成的tif添加地理信息
    # 如果不增加这一步，则生成的图片没有经纬度坐标、投影的信息
    # 获取投影信息
    origin_proj = origin_dataset.GetProjection()
    new_dataset.SetProjection(origin_proj)
    # 仿射矩阵
    origin_geotrans = origin_dataset.GetGeoTransform()
    new_dataset.SetGeoTransform(origin_geotrans)
    band = new_dataset.GetRasterBand(1)
    band.WriteArray(preA_2d)
    # print(new_dataset)
    new_data = new_dataset.ReadAsArray()
    # print(new_data)
    # 关闭数据集
    origin_dataset = None
    new_dataset = None
    print(fr"{TREE}_{ssp}保存a成功！")

    # %将计算和空值数据合并在一起
    pre_1 = subset_pre[['ID']]
    pre_1.loc[:, 'p'] = p_values
    pre_2 = subset_null[['ID']]
    pre_2['p'] = np.NaN
    pre = pd.concat([pre_1, pre_2], ignore_index=True)

    sorted_pre = pre.sort_values('ID')

    # %##TODO:将数组转换为预测小tif
    # 读取tif文件
    origin_dataset = gdal.Open(Time2020)
    # 获取数据集的行数和列数(维度和numpy数组相反)
    rows = origin_dataset.RasterYSize
    cols = origin_dataset.RasterXSize
    ###TODO:处理像元值数据
    # 提取某一列的值作为一维数组
    preA_values = sorted_pre['p'].values
    # 将一维数组转换为二维数组
    preA_2d = np.reshape(preA_values, (rows, cols))
    # %
    # 创建保存的tif名及位置
    output_filename = fr'preSI{TREE}_{ssp}.tif'
    output_tiff = os.path.join(pTIF_directory, output_filename)
    '''
    # driver.Create(filename, xsize, ysize, [bands], [data_type], [options])
    # filename是要创建的数据集的路径。
    # xsize是新数据集中的列数。
    # ysize是新数据集中的行数。
    # bands是新数据集中的波段数。默认值为1。
    # data_type是将存储在新数据集中的数据类型。默认值为GDT_Byte。
    # options是创建选项字符串的列表。可能的值取决于正在创建的数据集的类型。
    '''
    # 创建tif文件
    driver = gdal.GetDriverByName('GTiff')
    # 维度和numpy数组相反
    new_dataset = driver.Create(output_tiff, preA_2d.shape[1], preA_2d.shape[0], 1, gdal.GDT_Float32)

    # 读取之前的tif信息，为新生成的tif添加地理信息
    # 如果不增加这一步，则生成的图片没有经纬度坐标、投影的信息
    # 获取投影信息
    origin_proj = origin_dataset.GetProjection()
    new_dataset.SetProjection(origin_proj)
    # 仿射矩阵
    origin_geotrans = origin_dataset.GetGeoTransform()
    new_dataset.SetGeoTransform(origin_geotrans)
    band = new_dataset.GetRasterBand(1)
    band.WriteArray(preA_2d)
    # print(new_dataset)
    new_data = new_dataset.ReadAsArray()
    # print(new_data)
    # 关闭数据集
    origin_dataset = None
    new_dataset = None
    print(fr"{TREE}_{ssp}保存p_values成功！")

"""
#%% 统计斜率图相关变量
import pandas as pd
import numpy as np
from osgeo import gdal
import warnings
warnings.filterwarnings("ignore")

# 定义函数读取tiff文件像元值
def read_tiff_pixels(tiff_file):
    dataset = gdal.Open(tiff_file)
    band = dataset.GetRasterBand(1)  # 获取第一个波段（索引从1开始）
    pixel_data = band.ReadAsArray()  # 读取整个波段的像元值
    dataset = None  # 关闭数据集
    return pixel_data
TreeList = ['110','120','150','200','220','230','250','310','350','420','530']
# TreeList = ['200','220','230','250']
ssplist = ['ssp126']
# ssplist = ['ssp585', 'ssp245', 'ssp126']
for TREE in TreeList:
    for ssp in ssplist:
        print(f'TREE: {TREE}, ssp: {ssp}')

        # 读取三个 TIFF 文件路径
        SI2020 = fr"E:\CIMP6\FIGS\DataCurrent2020\preSI{TREE}.tif"
        Slope = fr"E:\CIMP6\FIGS\DataSlope\preSI{TREE}_{ssp}.tif"
        pValue = fr"E:\CIMP6\FIGS\Datap_value\preSI{TREE}_{ssp}.tif"

        # 创建空数据框
        ALLdf = pd.DataFrame()

        # 读取所有 TIFF 文件并转换为一维数组，存储在数据框中
        ALLdf['SI2020'] = read_tiff_pixels(SI2020).flatten()
        ALLdf['Slope'] = read_tiff_pixels(Slope).flatten()
        ALLdf['pValue'] = read_tiff_pixels(pValue).flatten()
        # 查看前2行
        print(ALLdf['SI2020'][1])

        # 新增自增列
        ALLdf['ID'] = range(len(ALLdf))

        # 遍历所有列进行判断，取出值不等于NoData_value的子集
        for column in ALLdf.columns:
            ALLdf[column] = ALLdf[column].replace(-340282346638528859811704183484516925440.00000, np.nan)

        # 根据 2020 年数据选取非空数据
        subset_pre = ALLdf[(ALLdf['SI2020'] > 0) & (ALLdf['SI2020'] < 40) & (ALLdf['SI2020'].notnull())]

        # 假设 subset_pre 是你的 DataFrame
        min_value = np.min(subset_pre['SI2020'])
        max_value = np.max(subset_pre['SI2020'])

        # 计算等间距区间
        interval_size = (max_value - min_value) / 3
        intervals = [
            (min_value, min_value + interval_size),
            (min_value + interval_size, min_value + 2 * interval_size),
            (min_value + 2 * interval_size, max_value)
        ]

        # 定义一个函数来分类
        def classify_si(value):
            if intervals[0][0] <= value < intervals[0][1]:
                return 'poor'
            elif intervals[1][0] <= value < intervals[1][1]:
                return 'medium'
            else:
                return 'good'


        # 应用分类函数并创建新字段
        subset_pre['SIClass'] = subset_pre['SI2020'].apply(classify_si)

        subset_pre.loc[:, 'SlopeClass'] = np.where(subset_pre['Slope'] > 0, 'increase', 'decrease')

        subset_pre.loc[:, 'pSignificance'] = np.where(subset_pre['pValue'] < 0.05, 'significant', 'non-significant')
        #给F4标注统计信息
        # 统计不同 SIClass 组各 SlopeClass 的数量
        class_counts = subset_pre.groupby('SIClass')['SlopeClass'].value_counts().unstack(fill_value=0)
        # 计算百分比
        percentage = class_counts.div(class_counts.sum(axis=1), axis=0) * 100
        # 输出结果
        print(percentage)
        # 保存 CSV 文件
        output_csv = fr"E:\CIMP6\FIGS\DataF4\preSI{TREE}_{ssp}.csv"
        percentage.to_csv(output_csv, encoding='utf-8')
        # output_csv1 = fr"E:\CIMP6\FIGS\DataF4\data\preSI{TREE}_{ssp}.csv"
        # subset_pre.to_csv(output_csv1, encoding='utf-8')
        print(f'Saved: {output_csv}')

        # #绘制右下角条形图
        # # 计算'pSignificance'为'significant'的百分比
        # total_count = len(subset_pre)
        # significant_count = len(subset_pre[subset_pre['pSignificance'] == 'significant'])
        # significant_percentage = (significant_count / total_count) * 100
        #
        # # 计算'pSignificance'为'significant'下各'SlopeClass'类别的百分比
        # significant_subset = subset_pre[subset_pre['pSignificance'] == 'significant']
        # slope_class_counts = significant_subset['SlopeClass'].value_counts(normalize=True) * 100
        #
        # # 直接添加新数据
        # slope_class_counts['significant'] = significant_percentage
        #
        # # 将 Series 转换为 DataFrame
        # df3 = slope_class_counts.reset_index()
        # output_csv1 = fr"E:\CIMP6\FIGS\DataF4\Rdata\preSI{TREE}_{ssp}.csv"
        # df3.to_csv(output_csv1, encoding='utf-8')



#%%读取表中数据，对poor面积变化趋势进行统计

#%%忽略一些版本不兼容等警告
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import linregress
#%read data
df = pd.read_csv(r'E:\CIMP6\Figures\Version4\Fig4\statistics_areas1017.csv')

# 假设你的数据存储在一个名为 df 的 DataFrame 中
# 示例：df = pd.DataFrame({'tree': [...], 'rp': [...], 'time': [...], 'good_area': [...]})

# 创建一个空的结果列表来保存每个组合的结果
results = []

# 使用 groupby 方法按 'tree' 和 'rcp' 字段分组
grouped = df.groupby(['Tree', 'RCP'])

# 遍历每一个分组，进行线性回归
for (tree, rp), group in grouped:
    # 获取 time 和 good_area 数据
    x = group['Time']
    y = group['medium_area']

    # 执行线性回归
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    # 根据条件判断趋势
    if slope > 0:
        trend = 'increase'
    elif slope < 0:
        trend = 'decrease'
    else:
        trend = 'equal'

    # 将结果存入字典并添加到结果列表
    results.append({'Tree': tree, 'RCP': rp, 'slope': slope, 'p_value': p_value, 'trend': trend})

# 将结果转换为 DataFrame 便于查看
result_df = pd.DataFrame(results)
print(result_df)


#%%SI  slope像元增减的量判断总体是增加还是减少趋势
import os
import rasterio
import numpy as np
import pandas as pd

# 设定存放 .tif 文件的目录路径
tif_directory = r"E:\CIMP6\FIGS\DataSlope"

# 初始化一个列表来存储每个tif文件的统计结果
results = []

# 遍历指定目录下的所有文件
for filename in os.listdir(tif_directory):
    if filename.endswith(".tif"):
        tif_path = os.path.join(tif_directory, filename)

        # 打开 tif 文件
        with rasterio.open(tif_path) as src:
            # 读取所有波段的数据到数组
            data = src.read(1)  # 读取第一波段，如果是多波段数据可以调整波段

            # 统计值小于等于0的像元数
            less_equal_zero_count = np.sum(data <= 0)

            # 统计值大于0的像元数
            greater_than_zero_count = np.sum(data > 0)

            # 根据条件判断趋势
            if less_equal_zero_count > greater_than_zero_count:
                trend = 'decrease'
            elif less_equal_zero_count < greater_than_zero_count:
                trend = 'increase'
            else:
                trend = 'equal'

        # 保存每个tif文件的统计结果
        results.append({
            'filename': filename,
            'less_equal_zero_count': less_equal_zero_count,
            'greater_than_zero_count': greater_than_zero_count,
            'trend': trend  # 修正此处的逗号
        })
        print(f"File: {filename}, <= 0 count: {less_equal_zero_count}, > 0 count: {greater_than_zero_count}")

# 将结果转换为 pandas DataFrame
dfresults = pd.DataFrame(results)