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
from pypinyin import pinyin, Style
#忽略一些版本不兼容等警告
import warnings
warnings.filterwarnings("ignore")



Timep = '2081_2100'
ssp = 'ssp585'
GCM = 'GISS_E2_1_G'
# ssp585   ssp245  ssp126  MPI_ESM1_2_HR  MRI_ESM2_0 GISS_E2_1_G
#%按省份提取（为甚么不直接提取中国？太大了）
# 输入的 Shapefile,在此确定是哪个植被区的范围，植被区
input_shapefile = r"E:\TOPH\中国_省\中国_省2.shp"#掩膜数据图层范围
# 所有输入数据的tif
all_tiff_directory = fr"E:\CIMP6\{Timep}\World\{GCM}\temp"#所有tiff位置,处理数据集
# 输出目录
shp_directory = fr"E:\CIMP6\{Timep}\shpProvinces1" #每个省份保存shp要素
tif_directory = fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}" #输出掩膜数据位置


# 创建保存目录（如果不存在）
if not os.path.exists(tif_directory):
    os.makedirs(tif_directory)

# 创建输出目录（如果不存在）
if not os.path.exists(shp_directory):
    os.makedirs(shp_directory)

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
#子函数，汉字转拼音
def chinese_to_pinyin(text):
    return ''.join([y[0] for y in pinyin(text, style=Style.NORMAL)])#TODO:#Style.TONE  Style.NORMAL

#%
# # #最开始需要根据渔网格对其分大块，每个植被区只需运行一次就行
# # #TODO: 使用 SplitByAttributes工具分割 Shapefile
# split_field = "name" #分割字段
# arcpy.analysis.SplitByAttributes(input_shapefile, shp_directory, split_field)

#%
#######step2 循环不同shapefile下不同tif（按切割块读信息）
# 获取文件夹中的tif文件路径列表
tiff_files = [file for file in os.listdir(all_tiff_directory) if file.endswith('.tif')]

for filename in os.listdir(shp_directory):
    if filename.endswith(".shp"):
        sub_shapefile = os.path.join(shp_directory, filename) #子shp：每一块
        sub_shapefile_name = filename.split('.')[0]  #获取该一块shp名称
        pinyin_shp = chinese_to_pinyin(sub_shapefile_name)
        print('shp:',sub_shapefile_name,pinyin_shp)
        # 创建数据框，保存以子shp命名
        sub_shapefile_df = pd.DataFrame()#用该shp块进行掩膜所有tif输入数据
        # # 清空文件夹
        # clear_folder(tif_directory)
        # 循环遍历tif文件,获取某一小块的tif数据
        for tif_file in tiff_files:
            # 动态获取名称
            tif_path = os.path.join(all_tiff_directory, tif_file)
            tif_pathname = tif_path.replace("-", "_")
            tif_name = os.path.basename(tif_pathname).split('.')[1]  # 当前tif名称os.path.basename(file_path).split('.')[0]
            tif_name1 = tif_name.split('_')
            tif_name2 = tif_name1[3:]
            tif_name3 = '_'.join(tif_name2)
            print("当前处理TIF:", tif_name3)
            # 创建保存的tif名及位置
            output_filename = str(pinyin_shp)+'_'+str(tif_name3) + ".tif"
            output_tiff = os.path.join(tif_directory, output_filename)
            # TODO:执行按掩膜提
            # print("Extract:", output_filename)
            arcpy.gp.ExtractByMask_sa(tif_path, sub_shapefile, output_tiff)#对应小块shp掩膜的所有tif保存临时目录在tif_directory文件里



#%%提取bio1-19个信息
import arcpy
import os
# ssp585   ssp245  ssp126  MPI_ESM1_2_HR  MRI_ESM2_0 GISS_E2_1_G
all_tiff_directory = fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}"
# 获取文件夹中的tif文件路径列表
tiff_files = [file for file in os.listdir(all_tiff_directory) if file.endswith('.tif')]
# 循环遍历tif文件,获取某一小块的tif数据
for tif_file in tiff_files:
    tif_path = os.path.join(all_tiff_directory, tif_file)
    tif_name = os.path.basename(tif_path).split('_')[0]  # 当前tif名称os.path.basename(file_path).split('.')[0]
    output_folder = os.path.join(all_tiff_directory, tif_name)
    # 创建保存目录（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 检查 Image Analyst 扩展
    if arcpy.CheckExtension("ImageAnalyst") == "Available":
        arcpy.CheckOutExtension("ImageAnalyst")
    else:
        raise RuntimeError("Image Analyst 扩展不可用。")

    # 检查输入文件是否存在
    if not os.path.exists(tif_path):
        raise RuntimeError(f"输入栅格文件 {tif_path} 不存在。")

    # 确保输出目录存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 循环提取波段并保存
    for band_number in range(1, 20):  # 从1到19提取波段
        try:
            # 提取波段
            out_bands_raster = arcpy.ia.ExtractBand(tif_path, [band_number])

            # 生成输出文件路径
            output_raster = os.path.join(output_folder, f"bio_{band_number}.tif")

            # 保存提取的波段
            out_bands_raster.save(output_raster)

            print(f"波段 {band_number} 已提取并保存为: {output_raster}")

        except Exception as e:
            print(f"提取波段 {band_number} 时出错: {e}")

    # 归还 Image Analyst 扩展
    arcpy.CheckInExtension("ImageAnalyst")

#%% 批量合成合并全国的波段 2
import arcpy
import os
# Timep = '2081_2100'
# ssp = 'ssp585'
# GCM = 'MPI_ESM1_2_HR'
# ssp585   ssp245  ssp126  MPI_ESM1_2_HR  MRI_ESM2_0 GISS_E2_1_G
# 设置输入和输出文件夹路径
China_folder = fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\China"

# 创建输出文件夹（如果不存在）
if not os.path.exists(China_folder):
    os.makedirs(China_folder)

for i in range(1, 20):
    bio = f'bio_{i}'
    print(f"合并全国的波段 {bio}")
    arcpy.management.MosaicToNewRaster(
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\anhuisheng\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\aomentebiexingzhengqu\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\beijingshi\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\chongqingshi\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\fujiansheng\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\gansusheng\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\guangxizhuangzuzizhiqu\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\guangdongsheng\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\guizhousheng\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\hainansheng\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\hebeisheng\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\heilongjiangsheng\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\neimengguzizhiqu\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\henansheng\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\hubeisheng\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\hunansheng\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\jiangsusheng\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\jiangxisheng\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\jilinsheng\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\liaoningsheng\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\ningxiahuizuzizhiqu\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\qinghaisheng\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\shandongsheng\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\shanghaishi\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\shanxisheng\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\shanxi\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\sichuansheng\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\taiwansheng\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\tianjinshi\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\xianggangtebiexingzhengqu\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\xinjiangweiwuerzizhiqu\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\xizangzizhiqu\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\yunnansheng\{bio}.tif;"
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\zhejiangsheng\{bio}.tif",
        fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\China", f"{bio}.tif",
        'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],'
        'UNIT["Degree",0.0174532925199433]]',
        "32_BIT_FLOAT", None, 1, "LAST", "FIRST"
    )

#%% 批量重采样resample 3
import arcpy
import os

# 设置时间段
# Timep = '2081_2100'
# ssp = 'ssp585'
# GCM = 'MPI_ESM1_2_HR'
# 设置输入和输出文件夹路径
input_folder = fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\China"
output_folder = fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\China\resample"

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义重采样参数
cell_size = "0.0083333 0.0083333"
resample_method = "NEAREST"

# 循环处理 bio_1 到 bio_19 的文件
for i in range(1, 20):  # 1到19的循环
    input_file = os.path.join(input_folder, f"bio_{i}.tif")
    output_file = os.path.join(output_folder, f"bio_{i}.tif")

    # 检查输入文件是否存在
    if os.path.exists(input_file):
        # 执行重采样
        arcpy.management.Resample(input_file, output_file, cell_size, resample_method)
        print(f"{input_file} 已成功重采样为 {output_file}")
    else:
        print(f"输入文件 {input_file} 不存在，跳过该文件。")

#%% 批量掩膜mask 4
import arcpy
from arcpy.sa import *
import os
# Timep = '2081_2100'
# ssp = 'ssp585'
# GCM = 'MPI_ESM1_2_HR'
arcpy.env.workspace = fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\China\resample"
arcpy.env.snapRaster = r"E:\TH\Xdata30m\1km_forPredict\mask\bio_1.tif"

mask_folder = fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\China\mask"
# 创建输出文件夹（如果不存在）
if not os.path.exists(mask_folder):
    os.makedirs(mask_folder)

rasters = [
    f"bio_{i}.tif" for i in range(1, 20)
]

for raster in rasters:
    print(f"{raster} 已成功")
    inRaster = fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\China\resample\{raster}"
    outRaster = fr"E:\CIMP6\{Timep}\China_Extract\{ssp}\{GCM}\China\mask\{raster}"
    mask = r"E:\TH\Xdata30m\1km_forPredict\mask\bio_1.tif"
    extractedRaster = ExtractByMask(inRaster, mask)
    extractedRaster.save(outRaster)



#
#
# #%%用于处理南北分界线
# import arcpy
# from arcpy.sa import *
#
# inRaster = r"F:\WorkingNotes\TH\WN\20240729\SIpre0\SI200\preSI200_S.tif"
# outRaster = r"F:\WorkingNotes\TH\WN\20240729\SIpre0\SI200\mask\preSI200_S.tif"
# mask = r"E:\CIMP6\shp_boundary\shp2\sourth.shp"
#
# arcpy.gp.ExtractByMask_sa(inRaster, mask, outRaster)#对应小块shp掩膜的所有tif保存临时目录在tif_directory文件里
# #%%
# inRaster = r"F:\WorkingNotes\TH\WN\20240729\SIpre0\SI200\preSI200_N.tif"
# outRaster = r"F:\WorkingNotes\TH\WN\20240729\SIpre0\SI200\mask\preSI200_N.tif"
# mask = r"E:\CIMP6\shp_boundary\shp1\north.shp"
#
# arcpy.gp.ExtractByMask_sa(inRaster, mask, outRaster)#对应小块shp掩膜的所有tif保存临时目录在tif_directory文件里
# #%%
# inRaster = r"F:\WorkingNotes\TH\WN\20240729\SIpre0\SI200\preSI200_b.tif"
# outRaster = r"F:\WorkingNotes\TH\WN\20240729\SIpre0\SI200\mask\preSI200_b.tif"
# mask = r"E:\CIMP6\shp_boundary\shp\boundary.shp"
#
# arcpy.gp.ExtractByMask_sa(inRaster, mask, outRaster)#对应小块shp掩膜的所有tif保存临时目录在tif_directory文件里
# #%%
# inRaster = r"F:\WorkingNotes\TH\WN\20240729\SIpre0\SI200\preSI200_xz.tif"
# outRaster = r"F:\WorkingNotes\TH\WN\20240729\SIpre0\SI200\mask\preSI200_xz.tif"
# mask = r"E:\CIMP6\shp_boundary\shp0\xizang.shp"
#
# arcpy.gp.ExtractByMask_sa(inRaster, mask, outRaster)#对应小块shp掩膜的所有tif保存临时目录在tif_directory文件里

#%% 计算3个tif的平均值
import os
import arcpy
from arcpy.sa import *

# 确保 Spatial Analyst 扩展可用
arcpy.CheckOutExtension("Spatial")

# 设置工作空间 (可选)
arcpy.env.workspace = r"E:\CIMP6\FIGS\Ensemble_3GCMs"

TREE = "350"
Timeplist = ['2021_2040','2041_2060','2061_2080','2081_2100']
ssplist = ['ssp585','ssp245','ssp126']
# GCMlist = ['MPI_ESM1_2_HR','MRI_ESM2_0','GISS_E2_1_G']#
# 嵌套循环遍历 GCMlist, Timeplist 和 ssplist
# for GCM in GCMlist:
for Timep in Timeplist:
    for ssp in ssplist:
        print(f'Timep: {Timep}, ssp: {ssp}')
        # 定义三个 TIFF 文件的路径
        raster1 = Raster(fr"E:\CIMP6\FIGS\GISS_E2_1_G\{Timep}\preSI{TREE}_{ssp}.tif")
        raster2 = Raster(fr"E:\CIMP6\FIGS\MPI_ESM1_2_HR\{Timep}\preSI{TREE}_{ssp}.tif")
        raster3 = Raster(fr"E:\CIMP6\FIGS\MRI_ESM2_0\{Timep}\preSI{TREE}_{ssp}.tif")

        # 计算每个像素的平均值
        average_raster = (raster1 + raster2 + raster3) / 3

        output_folder = fr"E:\CIMP6\FIGS\Ensemble_3GCMs\{Timep}"
        # 创建输出文件夹（如果不存在）
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # 输出结果到新的 TIFF 文件
        output_path = fr"E:\CIMP6\FIGS\Ensemble_3GCMs\{Timep}\preSI{TREE}_{ssp}.tif"
        average_raster.save(output_path)

        # 确保释放 Spatial Analyst 扩展
        arcpy.CheckInExtension("Spatial")

        print(f"平均栅格已成功保存到: {output_path}")


#%%循环读取文件夹中tif数据，按照equal interval划分3个区间并计算3个区间像素数量个数，对应字段列名为poor，medium，good，保存成csv，包含tif文件名，poor，medium，good
import arcpy
from arcpy.sa import *
import os
import csv
import numpy as np
from osgeo import gdal

# 设置工作环境和文件夹路径
arcpy.env.workspace = r"E:\CIMP6\FIGS\DataEnsemble_3GCMs\2081_2100"  # 设置工作空间路径
arcpy.CheckOutExtension("Spatial")

# 指定存储 TIFF 文件的文件夹路径
input_folder = arcpy.env.workspace

# 输出 CSV 文件路径
Time = '2081_2100'
output_csv = fr"E:\CIMP6\FIGS\Rdata\statistics{Time}.csv"

# 获取文件夹中所有的 TIFF 文件
tif_files = arcpy.ListRasters("*.tif")

# 创建一个空列表来存储所有统计数据
all_stats = []


# 子函数，读取整个 TIFF 像元值
def read_tiff_pixels(tiff_file):
    dataset = gdal.Open(tiff_file)
    band = dataset.GetRasterBand(1)  # 获取第一个波段（索引从1开始）
    # 读取整个波段的像元值
    pixel_data = band.ReadAsArray()
    dataset = None  # 关闭数据集
    return pixel_data


# 循环读取每个 TIFF 文件
for tif_file in tif_files:
    try:
        # 拼接完整路径
        tif_file_path = os.path.join(input_folder, tif_file)

        print(f'Start processing {tif_file_path}')

        # 读取整个 TIFF 像元值
        raster_array = read_tiff_pixels(tif_file_path)
        raster_array = raster_array[raster_array > 0]  # 过滤掉无效值

        # # 确保数组中没有 NaN 或负值（根据实际情况）
        # raster_array = np.where(np.isnan(raster_array), 0, raster_array)

        # 获取栅格数据的最小值和最大值
        min_value = np.min(raster_array)
        max_value = np.max(raster_array)

        # 计算等间距的区间范围
        interval_size = (max_value - min_value) / 3
        intervals = [(min_value, min_value + interval_size),
                     (min_value + interval_size, min_value + 2 * interval_size),
                     (min_value + 2 * interval_size, max_value)]

        # 创建一个空字典来存储统计数据
        stats_dict = {"filename": os.path.basename(tif_file), "poor": 0, "medium": 0, "good": 0}

        # 使用 NumPy 进行区间统计
        for i, (low, high) in enumerate(intervals):
            # 计算每个区间的像素数量
            count = np.sum((raster_array >= low) & (raster_array <= high))

            # 将统计结果添加到字典中
            if i == 0:
                stats_dict["poor"] = count
            elif i == 1:
                stats_dict["medium"] = count
            elif i == 2:
                stats_dict["good"] = count

        # 将每个文件的统计数据添加到列表中
        all_stats.append(stats_dict)

    except Exception as e:
        print(f"Error processing {tif_file}: {e}")

# 将结果写入 CSV 文件
with open(output_csv, "w", newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["filename", "poor", "medium", "good"])
    writer.writeheader()
    writer.writerows(all_stats)

# 释放扩展模块
arcpy.CheckInExtension("Spatial")

print(f"所有 TIFF 文件的统计已完成，结果保存在: {output_csv}")


#%%读取地图的tif数据，进行立地质量评价分区，并统计各分区面积
import numpy as np
import os
from osgeo import gdal
import csv
import math

Times = '2020_250'
#2021_2040  2041_2060  2061_2080   2081_2100
def lat_lon_to_meters(cell_size_deg, latitude):
    """
    Convert cell size in degrees to meters given a latitude.
    :param cell_size_deg: Cell size in degrees (width and height are assumed to be the same).
    :param latitude: Latitude of the cell center in degrees.
    :return: Cell size in meters (width and height).
    """
    # Earth's radius in meters
    earth_radius = 6378137.0

    # Convert latitude to radians
    lat_rad = math.radians(latitude)

    # Approximate meters per degree at the given latitude
    meters_per_degree_lat = (math.pi / 180) * earth_radius
    meters_per_degree_lon = meters_per_degree_lat * math.cos(lat_rad)

    # Cell size in meters
    cell_size_meters_lat = cell_size_deg * meters_per_degree_lat
    cell_size_meters_lon = cell_size_deg * meters_per_degree_lon

    return cell_size_meters_lat, cell_size_meters_lon


def calculate_area_in_square_meters(cell_size_deg, latitude):
    cell_size_meters_lat, cell_size_meters_lon = lat_lon_to_meters(cell_size_deg, latitude)
    return cell_size_meters_lat * cell_size_meters_lon


def get_raster_latitude(raster):
    """
    Retrieve the latitude of the center of the raster.
    :param raster: GDAL raster dataset object.
    :return: Latitude of the center of the raster.
    """
    geo_transform = raster.GetGeoTransform()
    # Calculate center latitude
    pixel_size_y = geo_transform[5]  # This is the pixel size in the y direction (positive for north-up)
    nrows = raster.RasterYSize
    center_y = (nrows / 2) * pixel_size_y + geo_transform[3]
    return center_y


def process_tif_file(tif_file_path, cell_size_deg):
    # Open the raster file
    dataset = gdal.Open(tif_file_path)
    if dataset is None:
        raise Exception(f"Failed to open {tif_file_path}")

    # Retrieve the latitude
    latitude = get_raster_latitude(dataset)

    # Read raster data
    band = dataset.GetRasterBand(1)
    pixel_data = band.ReadAsArray()

    # Get valid raster values
    raster_array = pixel_data[pixel_data > 0]

    # Calculate area of each cell
    cell_area = calculate_area_in_square_meters(cell_size_deg, latitude)

    # Calculate min and max values
    min_value = np.min(raster_array)
    max_value = np.max(raster_array)

    # Calculate equal intervals
    interval_size = (max_value - min_value) / 3
    intervals = [(min_value, min_value + interval_size),
                 (min_value + interval_size, min_value + 2 * interval_size),
                 (min_value + 2 * interval_size, max_value)]

    # Initialize areas for each category
    areas = {"poor": 0, "medium": 0, "good": 0}

    # Calculate areas for each interval
    for i, (low, high) in enumerate(intervals):
        count = np.sum((raster_array >= low) & (raster_array <= high))
        area = count * cell_area
        if i == 0:
            areas["poor"] = area
        elif i == 1:
            areas["medium"] = area
        elif i == 2:
            areas["good"] = area

    return areas


# Parameters
# input_folder = fr"E:\CIMP6\FIGS\DataEnsemble_3GCMs\{Times}"
input_folder = fr"E:\CIMP6\FIGS\DataCurrent2020\SI250"
output_csv = fr"E:\CIMP6\FIGS\Rdata\statistics_areas{Times}.csv"
cell_size_deg = 0.0083333

# Get all TIFF files
tif_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tif')]

# Process each TIFF file
all_stats = []
for tif_file in tif_files:
    try:
        print(f"Processing {tif_file}")
        areas = process_tif_file(tif_file, cell_size_deg)
        stats = {"filename": os.path.basename(tif_file), "poor_area": areas["poor"], "medium_area": areas["medium"],
                 "good_area": areas["good"]}
        all_stats.append(stats)
    except Exception as e:
        print(f"Error processing {tif_file}: {e}")

# Write results to CSV
with open(output_csv, "w", newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["filename", "poor_area", "medium_area", "good_area"])
    writer.writeheader()
    writer.writerows(all_stats)

print(f"Area calculation completed. Results saved to: {output_csv}")


#%%计算最优树种
import rasterio
import numpy as np

# 读取TIF文件
with rasterio.open(r"E:\CIMP6\FIGS\DataCurrent2020\maxmin\SI110.tif") as src_110, \
        rasterio.open(r"E:\CIMP6\FIGS\DataCurrent2020\maxmin\SI120.tif") as src_120, \
        rasterio.open(r"E:\CIMP6\FIGS\DataCurrent2020\maxmin\SI150.tif") as src_150, \
        rasterio.open(r"E:\CIMP6\FIGS\DataCurrent2020\maxmin\SI200.tif") as src_200, \
        rasterio.open(r"E:\CIMP6\FIGS\DataCurrent2020\maxmin\SI220.tif") as src_220, \
        rasterio.open(r"E:\CIMP6\FIGS\DataCurrent2020\maxmin\SI230.tif") as src_230, \
        rasterio.open(r"E:\CIMP6\FIGS\DataCurrent2020\maxmin\SI250.tif") as src_250, \
        rasterio.open(r"E:\CIMP6\FIGS\DataCurrent2020\maxmin\SI310.tif") as src_310, \
        rasterio.open(r"E:\CIMP6\FIGS\DataCurrent2020\maxmin\SI350.tif") as src_350, \
        rasterio.open(r"E:\CIMP6\FIGS\DataCurrent2020\maxmin\SI420.tif") as src_420, \
        rasterio.open(r"E:\CIMP6\FIGS\DataCurrent2020\maxmin\SI530.tif") as src_530:
    # 读取每个TIF的第一个波段数据
    data_110 = src_110.read(1)
    data_120 = src_120.read(1)
    data_150 = src_150.read(1)
    data_200 = src_200.read(1)
    data_220 = src_220.read(1)
    data_230 = src_230.read(1)
    data_250 = src_250.read(1)
    data_310 = src_310.read(1)
    data_350 = src_350.read(1)
    data_420 = src_420.read(1)
    data_530 = src_530.read(1)

    # 获取栅格元数据（假设所有TIF的栅格形状和坐标系相同）
    profile = src_110.profile

# 创建新的空数组存储结果
result = np.zeros_like(data_110, dtype=np.int16)

# 对比每个栅格点的值，选取最大值对应的TIF名称值
result[(data_110 > data_120) & (data_110 >= data_150)& (data_110 >= data_200)& (data_110 >= data_220) & (data_110 >= data_230) & (data_110 >= data_250) & (data_110 >= data_310)  & (data_110 >= data_350) & (data_110 >= data_420) & (data_110 >= data_530)] = 110
result[(data_120 > data_110) & (data_120 >= data_150)& (data_120 >= data_200)& (data_120 >= data_220) & (data_120 >= data_230) & (data_120 >= data_250) & (data_120 >= data_310)  & (data_120 >= data_350) & (data_120 >= data_420) & (data_120 >= data_530)] = 120
result[(data_150 > data_110) & (data_150 >= data_120)& (data_150 >= data_200)& (data_150 >= data_220) & (data_150 >= data_230) & (data_150 >= data_250) & (data_150 >= data_310)  & (data_150 >= data_350) & (data_150 >= data_420) & (data_150 >= data_530)] = 150
result[(data_200 > data_110) & (data_200 >= data_120)& (data_200 >= data_150)& (data_200 >= data_220) & (data_200 >= data_230) & (data_200 >= data_250) & (data_200 >= data_310)  & (data_200 >= data_350) & (data_200 >= data_420) & (data_200 >= data_530)] = 200
result[(data_220 > data_120) & (data_220 >= data_150)& (data_220 >= data_200)& (data_220 >= data_110) & (data_220 >= data_230) & (data_220 >= data_250) & (data_220 >= data_310)  & (data_220 >= data_350) & (data_220 >= data_420) & (data_220 >= data_530)] = 220
result[(data_230 > data_120) & (data_230 >= data_150)& (data_230 >= data_200)& (data_230 >= data_220) & (data_230 >= data_110) & (data_230 >= data_250) & (data_230 >= data_310)  & (data_230 >= data_350) & (data_230 >= data_420) & (data_230 >= data_530)] = 230
result[(data_250 > data_120) & (data_250 >= data_150)& (data_250 >= data_200)& (data_250 >= data_220) & (data_250 >= data_230) & (data_250 >= data_110) & (data_250 >= data_310)  & (data_250 >= data_350) & (data_250 >= data_420) & (data_250 >= data_530)] = 250
result[(data_310 > data_120) & (data_310 >= data_150)& (data_310 >= data_200)& (data_310 >= data_220) & (data_310 >= data_230) & (data_310 >= data_250) & (data_310 >= data_110)  & (data_310 >= data_350) & (data_310 >= data_420) & (data_310 >= data_530)] = 310
result[(data_350 > data_120) & (data_350 >= data_150)& (data_350 >= data_200)& (data_350 >= data_220) & (data_350 >= data_230) & (data_350 >= data_250) & (data_350 >= data_310)  & (data_350 >= data_110) & (data_350 >= data_420) & (data_350 >= data_530)] = 350
result[(data_420 > data_120) & (data_420 >= data_150)& (data_420 >= data_200)& (data_420 >= data_220) & (data_420 >= data_230) & (data_420 >= data_250) & (data_420 >= data_310)  & (data_420 >= data_350) & (data_420 >= data_110) & (data_420 >= data_530)] = 420
result[(data_530 > data_120) & (data_530 >= data_150)& (data_530 >= data_200)& (data_530 >= data_220) & (data_530 >= data_230) & (data_530 >= data_250) & (data_530 >= data_310)  & (data_530 >= data_350) & (data_530 >= data_420) & (data_530 >= data_110)] = 530


# 保存新的TIF文件
profile.update(dtype=rasterio.int16)

with rasterio.open("E:\CIMP6\FIGS\DataCurrent2020\maxmin\OptimalTreeSpecies.tif", 'w', **profile) as dst:
    dst.write(result, 1)

#%%
import rasterio
import numpy as np

# 所有TIF文件的路径和对应的TIF值
tif_files = {
    110: r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\SI110.tif",
    120: r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\SI120.tif",
    150: r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\SI150.tif",
    200: r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\SI200.tif",
    220: r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\SI220.tif",
    230: r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\SI230.tif",
    250: r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\SI250.tif",
    310: r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\SI310.tif",
    350: r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\SI350.tif",
    420: r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\SI420.tif",
    530: r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\SI530.tif"
}

# 初始化变量
data_list = []
keys = list(tif_files.keys())
nodata_value = None

# 读取TIF文件并将数据存储到data_list中
for key, file_path in tif_files.items():
    with rasterio.open(file_path) as src:
        data = src.read(1)  # 读取栅格数据
        if nodata_value is None:
            nodata_value = src.nodata  # 获取nodata值
        profile = src.profile if 'profile' not in locals() else profile  # 获取栅格元数据
        data[data == nodata_value] = np.nan  # 将nodata值替换为NaN
        data_list.append(data)

# 使用NumPy的stack函数将所有栅格数据堆叠为一个3D数组（第一维为各个TIF的栅格值）
data_stack = np.stack(data_list)

# 创建一个空数组存储结果
result = np.zeros_like(data_stack[0], dtype=np.int16)

# 查找每个栅格位置的最大值的索引，忽略NaN
for row in range(data_stack.shape[1]):
    for col in range(data_stack.shape[2]):
        # 获取每个栅格位置的值
        values = data_stack[:, row, col]
        # 过滤掉NaN
        valid_values = values[~np.isnan(values)]

        if valid_values.size > 0:  # 如果有有效值
            max_index = np.argmax(valid_values)
            result[row, col] = keys[max_index]
        else:  # 如果全为NaN，设置为默认值
            result[row, col] = 0  # 或者其他默认值

# 更新nodata值和数据类型
profile.update(dtype=rasterio.int16, nodata=-32768)

# 保存结果TIF文件
with rasterio.open(r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\OptimalTreeSpecies585.tif", 'w', **profile) as dst:
    dst.write(result, 1)

#%%
import numpy as np
import pandas as pd
df = pd.read_csv(r'F:\WorkingNotes\TH\WN\20240729\THdata\YUNSHAN120HT.csv')#,encoding="gb2312"
y = df['preHT']
ydes = y.describe()


#%%已经两点经纬度计算距离
import math
# Haversine公式计算两点之间的球面距离
def haversine(lon1, lat1, lon2, lat2):
    # 地球半径，单位为公里
    R = 6371.0
    # 将经纬度从度转换为弧度
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # 计算经纬度的差值
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    # Haversine公式
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    # 计算距离
    distance = R * c
    return distance

# 点1和点2的经纬度
lon1, lat1 = 117.3513469	,	38.99042371


lon2, lat2 =111.2387059	,	34.82045524

# 计算两点之间的距离
distance = haversine(lon1, lat1, lon2, lat2)
distance

'''
最优树种统计面积
已知tif数据是int类型，将每个int值代表一个树种。统计各树种面积。从读取TIF文件、根据像素值计算每个分区的面积，到将结果保存为CSV文件
'''

#%%import numpy as np
import numpy as np
import os
from osgeo import gdal
import csv
import math

Times = '585'  # 时间标识

# 函数：将纬度和像素的角度单位大小转换为米
def lat_lon_to_meters(cell_size_deg, latitude):
    earth_radius = 6378137.0  # 地球半径，单位：米
    lat_rad = math.radians(latitude)  # 将纬度转换为弧度
    meters_per_degree_lat = (math.pi / 180) * earth_radius
    meters_per_degree_lon = meters_per_degree_lat * math.cos(lat_rad)
    cell_size_meters_lat = cell_size_deg * meters_per_degree_lat
    cell_size_meters_lon = cell_size_deg * meters_per_degree_lon
    return cell_size_meters_lat, cell_size_meters_lon

# 函数：根据纬度计算每个像素的面积（平方米）
def calculate_area_in_square_meters(cell_size_deg, latitude):
    cell_size_meters_lat, cell_size_meters_lon = lat_lon_to_meters(cell_size_deg, latitude)
    return cell_size_meters_lat * cell_size_meters_lon

# 函数：获取栅格文件的中心纬度
def get_raster_latitude(raster):
    geo_transform = raster.GetGeoTransform()
    pixel_size_y = geo_transform[5]  # 像素大小（y方向）
    nrows = raster.RasterYSize
    center_y = (nrows / 2) * pixel_size_y + geo_transform[3]
    return center_y

# 函数：处理TIF文件，统计各树种的面积
def process_tif_file(tif_file_path, cell_size_deg):
    dataset = gdal.Open(tif_file_path)
    if dataset is None:
        raise Exception(f"Failed to open {tif_file_path}")

    latitude = get_raster_latitude(dataset)  # 获取中心纬度
    band = dataset.GetRasterBand(1)  # 读取第一个波段
    pixel_data = band.ReadAsArray()  # 读取像素数据

    # 计算每个像素的面积
    cell_area = calculate_area_in_square_meters(cell_size_deg, latitude)

    # 获取所有唯一的树种（int值）
    unique_values, counts = np.unique(pixel_data, return_counts=True)

    # 计算每个树种的面积
    species_areas = {value: count * cell_area for value, count in zip(unique_values, counts)}

    return species_areas

#% 参数设置
tif_file = fr"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\good\OptimalTreeSpecies585.tif"  # 输入TIF文件路径
output_csv = fr"E:\CIMP6\Figures\Version4\Fig5\goodOptimalTreeSpecies{Times}.csv"  # 输出CSV文件路径
cell_size_deg = 0.0083333  # 每个像素的角度大小（大约1公里）

# 初始化结果列表
all_stats = []

print(f"Processing {tif_file}")
species_areas = process_tif_file(tif_file, cell_size_deg)

#% 保存每个树种的面积
for species, area in species_areas.items():
    stats = {"filename": os.path.basename(tif_file), "species": species, "area": area}
    all_stats.append(stats)

#% 将结果写入CSV文件
with open(output_csv, "w", newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["filename", "species", "area"])
    writer.writeheader()
    writer.writerows(all_stats)

print(f"Species area calculation completed. Results saved to: {output_csv}")


#%%2-1统计最优树种对应good medium poor样地对应点数：提取点到图层属性表
import arcpy
from arcpy.sa import *
import os


TreeList = ['110','120','150','200','220','230','250','310','350','420','530']
# 嵌套循环遍历 Timeplist 和 ssplist
for TREE in TreeList:
    print(f'TREE: {TREE}')
    point_shp = fr"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\PointSHP\Opl_SI{TREE}.shp"
    raster_tif = fr"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\preSI{TREE}_ssp585.tif"
    outpoint_shp = fr"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\PointSI\Opl_SIClass585{TREE}.shp"
    #提取值到点
    arcpy.sa.ExtractValuesToPoints(point_shp,raster_tif,outpoint_shp, "NONE", "VALUE_ONLY")
#%%2-2统计最优树种对应good medium poor样地对应点数：保存属性表到excel
# pip install geopandas openpyxl
import geopandas as gpd
import os

TreeList = ['110', '120', '150', '200', '220', '230', '250', '310', '350', '420', '530']

# 嵌套循环遍历 TreeList
for TREE in TreeList:
    # 定义 Shapefile 文件路径
    # outpoint_shp = fr"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\126\PointSI\Opl_SIClass126{TREE}.shp"
    outpoint_shp = fr"E:\CIMP6\FIGS\DataCurrent2020\PointSI\Opl_SIClass{TREE}.shp"

    # 检查文件是否存在
    if os.path.exists(outpoint_shp):
        # 读取 Shapefile 文件
        gdf = gpd.read_file(outpoint_shp)

        # 定义输出 Excel 文件路径
        excel_output = fr'E:\CIMP6\Figures\Version4\Fig5\data\{TREE}.xlsx'

        # 将属性表导出为 Excel 文件
        gdf.to_excel(excel_output, index=False)
        print(f"Exported {excel_output}")
    else:
        print(f"Shapefile {outpoint_shp} does not exist.")



# %% Import necessary libraries
import pandas as pd
import numpy as np
import os
from osgeo import gdal
import csv
import math

# Define TreeList for iteration
TreeList = ['110', '120', '150', '200', '220', '230', '250', '310', '350', '420', '530']

# Loop over each tree type in TreeList
for TREE in TreeList:
    # Define raster and Excel output paths
    raster_tif = fr"E:\CIMP6\FIGS\DataCurrent2020\preSI{TREE}.tif"
    excel_output = fr'E:\CIMP6\Figures\Version4\Fig5\data\{TREE}.xlsx'

    # Step1: Get interval boundary values from raster data

    # Open the raster file
    dataset = gdal.Open(raster_tif)
    if dataset is None:
        raise Exception(f"Failed to open {raster_tif}")

    # Read raster data from the first band
    band = dataset.GetRasterBand(1)
    pixel_data = band.ReadAsArray()

    # Get valid raster values (non-zero positive values)
    raster_array = pixel_data[(pixel_data > 0) & (pixel_data < 100)]

    # Calculate min and max values, and the interval sizes
    min_value = np.min(raster_array)
    max_value = np.max(raster_array)
    interval_size = (max_value - min_value) / 3

    # Define key values for classification
    keyvalue1 = min_value + interval_size
    keyvalue2 = min_value + 2 * interval_size

    # Step2: Classification based on raster values

    # Read the Excel data
    df = pd.read_excel(excel_output)

    # Ensure that 'RASTERVALU' column exists
    if 'RASTERVALU' not in df.columns:
        raise Exception(f"'RASTERVALU' column missing in {excel_output}")


    # Apply classification rules
    def classify(value):
        if value < keyvalue1:
            return "poor"
        elif value > keyvalue2:
            return "good"
        else:
            return "medium"


    # Apply the classification function to each row in 'RASTERVALU' column
    df['SIClass'] = df['RASTERVALU'].apply(classify)

    # Save the updated DataFrame to a CSV file
    df.to_csv(fr'E:\CIMP6\Figures\Version4\Fig5\dataSIClass\{TREE}.csv', index=False)

    # Output success message for each tree type
    print(f"Exported {TREE} classification results.")

#%%
import pandas as pd

# Define TreeList for iteration
TreeList = ['110', '120', '150', '200', '220', '230', '250', '310', '350', '420', '530']

# 存储所有分类统计的列表
all_class_distributions = []

# Loop over each tree type in TreeList
for TREE in TreeList:
    # 读取 CSV 文件
    df = pd.read_csv(fr'E:\CIMP6\Figures\Version4\Fig5\dataSIClass\{TREE}.csv')

    # 统计 'SIClass' 列中每个类别的计数和百分比
    class_counts = df['SIClass'].value_counts()
    class_percentages = df['SIClass'].value_counts(normalize=True) * 100

    # 创建一个 DataFrame 来合并计数和百分比
    class_distribution = pd.DataFrame({
        'Class': class_counts.index,
        'Count': class_counts.values,
        'Percentage (%)': class_percentages.values,
        'Tree': TREE
    })

    # 将每个 DataFrame 添加到列表中
    all_class_distributions.append(class_distribution)

# 合并所有 DataFrame
final_distribution = pd.concat(all_class_distributions, ignore_index=True)

# 保存结果到 CSV 文件
output_csv_path = r'E:\CIMP6\Figures\Version4\Fig5\dataSIClass\class_distribution_2020.csv'
final_distribution.to_csv(output_csv_path, index=False)

# 输出成功信息
print(f"Class distribution summary saved to {output_csv_path}")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%
from osgeo import gdal
import numpy as np

# 定义树木列表以进行迭代
TreeList = ['110', '120', '150', '200', '220', '230', '250', '310', '350', '420', '530']

# 遍历 TreeList 中的每个树木类型
for TREE in TreeList:
    # 定义栅格和输出路径
    raster_tif = fr"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\SI{TREE}.tif"
    out_tif = fr"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\good\SI{TREE}.tif"

    # 打开栅格文件
    origin_dataset = gdal.Open(raster_tif)
    if origin_dataset is None:
        print(f"Failed to open {raster_tif}")
        continue  # 如果文件无法打开，则跳过该文件

    # 读取第一个波段的数据
    band = origin_dataset.GetRasterBand(1)
    pixel_data = band.ReadAsArray()  # 读取栅格数据为 NumPy 数组

    # 计算 2/3 的值
    threshold_value = 2 / 3

    # 将小于 2/3 的值设置为 0
    pixel_data[pixel_data < threshold_value] = 0

    # 创建新的 TIFF 文件
    driver = gdal.GetDriverByName('GTiff')
    new_dataset = driver.Create(out_tif, pixel_data.shape[1], pixel_data.shape[0], 1, gdal.GDT_Float32)

    # 复制原始栅格文件的投影和地理信息
    origin_proj = origin_dataset.GetProjection()  # 获取投影信息
    origin_geotrans = origin_dataset.GetGeoTransform()  # 获取仿射矩阵

    new_dataset.SetProjection(origin_proj)  # 设置投影
    new_dataset.SetGeoTransform(origin_geotrans)  # 设置仿射矩阵

    # 写入修改后的数据
    new_band = new_dataset.GetRasterBand(1)
    new_band.WriteArray(pixel_data)

    # 关闭数据集以确保数据保存
    origin_dataset = None
    new_dataset = None

    print(f"保存成功: {out_tif}")



#%%good下计算最优树种
import rasterio
import numpy as np

# 读取TIF文件
with rasterio.open(r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\good\SI110.tif") as src_110, \
        rasterio.open(r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\good\SI120.tif") as src_120, \
        rasterio.open(r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\good\SI150.tif") as src_150, \
        rasterio.open(r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\good\SI200.tif") as src_200, \
        rasterio.open(r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\good\SI220.tif") as src_220, \
        rasterio.open(r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\good\SI230.tif") as src_230, \
        rasterio.open(r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\good\SI250.tif") as src_250, \
        rasterio.open(r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\good\SI310.tif") as src_310, \
        rasterio.open(r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\good\SI350.tif") as src_350, \
        rasterio.open(r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\good\SI420.tif") as src_420, \
        rasterio.open(r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\good\SI530.tif") as src_530:
    # 读取每个TIF的第一个波段数据
    data_110 = src_110.read(1)
    data_120 = src_120.read(1)
    data_150 = src_150.read(1)
    data_200 = src_200.read(1)
    data_220 = src_220.read(1)
    data_230 = src_230.read(1)
    data_250 = src_250.read(1)
    data_310 = src_310.read(1)
    data_350 = src_350.read(1)
    data_420 = src_420.read(1)
    data_530 = src_530.read(1)

    # 获取栅格元数据（假设所有TIF的栅格形状和坐标系相同）
    profile = src_110.profile

# 创建新的空数组存储结果
result = np.zeros_like(data_110, dtype=np.int16)

# 对比每个栅格点的值，选取最大值对应的TIF名称值
result[(data_110 > data_120) & (data_110 >= data_150)& (data_110 >= data_200)& (data_110 >= data_220) & (data_110 >= data_230) & (data_110 >= data_250) & (data_110 >= data_310)  & (data_110 >= data_350) & (data_110 >= data_420) & (data_110 >= data_530)] = 110
result[(data_120 > data_110) & (data_120 >= data_150)& (data_120 >= data_200)& (data_120 >= data_220) & (data_120 >= data_230) & (data_120 >= data_250) & (data_120 >= data_310)  & (data_120 >= data_350) & (data_120 >= data_420) & (data_120 >= data_530)] = 120
result[(data_150 > data_110) & (data_150 >= data_120)& (data_150 >= data_200)& (data_150 >= data_220) & (data_150 >= data_230) & (data_150 >= data_250) & (data_150 >= data_310)  & (data_150 >= data_350) & (data_150 >= data_420) & (data_150 >= data_530)] = 150
result[(data_200 > data_110) & (data_200 >= data_120)& (data_200 >= data_150)& (data_200 >= data_220) & (data_200 >= data_230) & (data_200 >= data_250) & (data_200 >= data_310)  & (data_200 >= data_350) & (data_200 >= data_420) & (data_200 >= data_530)] = 200
result[(data_220 > data_120) & (data_220 >= data_150)& (data_220 >= data_200)& (data_220 >= data_110) & (data_220 >= data_230) & (data_220 >= data_250) & (data_220 >= data_310)  & (data_220 >= data_350) & (data_220 >= data_420) & (data_220 >= data_530)] = 220
result[(data_230 > data_120) & (data_230 >= data_150)& (data_230 >= data_200)& (data_230 >= data_220) & (data_230 >= data_110) & (data_230 >= data_250) & (data_230 >= data_310)  & (data_230 >= data_350) & (data_230 >= data_420) & (data_230 >= data_530)] = 230
result[(data_250 > data_120) & (data_250 >= data_150)& (data_250 >= data_200)& (data_250 >= data_220) & (data_250 >= data_230) & (data_250 >= data_110) & (data_250 >= data_310)  & (data_250 >= data_350) & (data_250 >= data_420) & (data_250 >= data_530)] = 250
result[(data_310 > data_120) & (data_310 >= data_150)& (data_310 >= data_200)& (data_310 >= data_220) & (data_310 >= data_230) & (data_310 >= data_250) & (data_310 >= data_110)  & (data_310 >= data_350) & (data_310 >= data_420) & (data_310 >= data_530)] = 310
result[(data_350 > data_120) & (data_350 >= data_150)& (data_350 >= data_200)& (data_350 >= data_220) & (data_350 >= data_230) & (data_350 >= data_250) & (data_350 >= data_310)  & (data_350 >= data_110) & (data_350 >= data_420) & (data_350 >= data_530)] = 350
result[(data_420 > data_120) & (data_420 >= data_150)& (data_420 >= data_200)& (data_420 >= data_220) & (data_420 >= data_230) & (data_420 >= data_250) & (data_420 >= data_310)  & (data_420 >= data_350) & (data_420 >= data_110) & (data_420 >= data_530)] = 420
result[(data_530 > data_120) & (data_530 >= data_150)& (data_530 >= data_200)& (data_530 >= data_220) & (data_530 >= data_230) & (data_530 >= data_250) & (data_530 >= data_310)  & (data_530 >= data_350) & (data_530 >= data_420) & (data_530 >= data_110)] = 530


# 保存新的TIF文件
profile.update(dtype=rasterio.int16)

with rasterio.open(fr"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\good\OptimalTreeSpecies.tif", 'w', **profile) as dst:
    dst.write(result, 1)

#%%
out_raster = arcpy.sa.ExtractByMask("preSI110.tif", "pnf.tif", "INSIDE", '73.600612577997 18.1363289514827 134.933700577997 53.5611872514827 GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]');
out_raster.save(r"E:\CIMP6\FIGS\CodeCurrent2020\MyProject.gdb\Extract_preS1")