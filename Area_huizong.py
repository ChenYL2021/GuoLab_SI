# -*- coding: utf-8 -*-
# @Time    : 2024/2/25 16:23
# @Author  : ChenYuling
# @FileName: Area_huizong.py
# @Software: PyCharm
# @Describe：folder_path = r"G:\mianji"
#方法1：计算栅格非空面积求和
#%%
import os
from osgeo import gdal, ogr, osr
import numpy as np

# 定义栅格文件夹路径
# folder_path = r"F:\DB\TH\原始数据\CHM\dongbeidayangdi52"
folder_path = r"G:\mianji"
# 获取文件夹中所有栅格文件的路径
input_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.tif')]
#%%记录总面积
alltotal_area = 0
i=0
# 遍历输入文件
for input_file in input_files:
    raster_ds = gdal.Open(input_file)

    # 读取栅格数据的地理参考信息
    projection = raster_ds.GetProjection()
    geotransform = raster_ds.GetGeoTransform()

    # 读取栅格数据的数组
    band = raster_ds.GetRasterBand(1)
    array = band.ReadAsArray()

    # 计算有效面积
    valid_area = np.sum(array != band.GetNoDataValue())

    # 计算栅格单元的面积
    cell_size = abs(geotransform[1] * geotransform[5])

    # 计算栅格的有效面积
    total_area = valid_area * cell_size

    # 打印结果
    print("tifName:", input_file)
    print("Total area without nodata value:", total_area)
    # 打印总面积

    alltotal_area += total_area
    i = i+1

    # # 关闭数据集
    # raster_ds = None
    # vector_ds = None

print("all Total area:", alltotal_area)


#方法2：栅格转int，转面，计算面积
#%%
import os
import arcpy

# 设置工作空间和输出面图层路径
workspace = "G:\mianji"
# 打开工作空间
arcpy.env.workspace = workspace

# 初始化总面积
total_area = 0

# 遍历文件夹中的所有 TIFF 文件
for root, dirs, files in os.walk(workspace):
    for file in files:
        if file.endswith(".tif"):
            # 构建输入栅格数据路径
            in_raster = os.path.join(root, file)

            # 输出整数型栅格数据
            int_raster = arcpy.sa.Int(in_raster)

            # 栅格转换为面图层
            out_polygon = r"polygon.shp"
            arcpy.conversion.RasterToPolygon(int_raster, out_polygon, "NO_SIMPLIFY", "Value", "SINGLE_OUTER_PART")

            # 统计面图层的面积
            with arcpy.da.SearchCursor(out_polygon, ["SHAPE@AREA"]) as cursor:
                for row in cursor:
                    total_area += row[0]

            # 打印结果
            print("tifName:", file)
            print("Total area:", total_area)

            # 删除临时整数型栅格数据和面图层
            arcpy.Delete_management(int_raster)
            arcpy.Delete_management(out_polygon)

# 打印总面积
print("Total area of all raster layers:", total_area)

#%%
from osgeo import gdal, ogr, osr

# 打开栅格数据
raster_file = r"G:\mianji\201801_海南吊罗山大样地_001_数字高程模型.tif"
raster_ds = gdal.Open(raster_file)

# 将栅格数据转换为整数类型
band = raster_ds.GetRasterBand(1)
array = band.ReadAsArray()
array = array.astype(int)  # 转换为整数类型

# 创建矢量化驱动
driver = ogr.GetDriverByName('ESRI Shapefile')
vector_file = "output_vector.shp"
vector_ds = driver.CreateDataSource(vector_file)
srs = osr.SpatialReference(wkt=raster_ds.GetProjection())

# 创建图层
layer = vector_ds.CreateLayer("polygons", srs, ogr.wkbPolygon)

# 将栅格数据转换为面图层
gdal.Polygonize(band, None, layer, -1, [], callback=None)

# # 关闭数据源
# raster_ds = None
# vector_ds = None

print("Vector layer saved to:", vector_file)



