# -*- coding: utf-8 -*-
# @Time    : 2023/10/19 16:02
# @Author  : ChenYuling
# @FileName: DataProcess.py
# @Software: PyCharm
# @Describe：基于tif范围批量生成渔网面
#%%
import arcpy
import os
import warnings
warnings.filterwarnings("ignore")#忽略一些版本不兼容等警告

#初始化变量
tif_directory = r"F:\DB\CHM\CHM\jinguangjituanhainan" #保存的原始CHM.tif文件夹位置
shp_directory = r"F:\DB\CHM\jinguangjituanhainan\fishnet" #输出面shp渔网文件夹位置
cell_width = 30  # 渔网格宽度
cell_height = 30  # 渔网格高度

#子函数实现已知模板范围生成指定大小的渔网面shp
def create_fishnet(template_extent,output_fc, cell_width, cell_height):
    # 设置工作环境的范围
    arcpy.env.outputCoordinateSystem = arcpy.Describe(template_extent).spatialReference
    arcpy.env.extent = arcpy.Describe(template_extent).extent
    # 创建渔网
    arcpy.CreateFishnet_management(output_fc, str(arcpy.env.extent.lowerLeft), str(arcpy.env.extent.XMin) + " " + str(arcpy.env.extent.YMin + cell_height), "0", "0", int(arcpy.env.extent.width / cell_width), int(arcpy.env.extent.height / cell_height), str(arcpy.env.extent.upperRight), "NO_LABELS", template_extent, "POLYGON")

#%%循环批量生成渔网面shp
# 获取文件夹中的tif文件路径列表
sub_tiff_files = [file for file in os.listdir(tif_directory) if file.endswith('.tif')]
for subtif_file in sub_tiff_files:#该循环结束后，sub_shapefile_df数据框保存该小shp块下对应自变量列数据
    subtif_path = os.path.join(tif_directory, subtif_file)
    subtif_name = os.path.basename(subtif_path).split('.')[0]  # 当前tif名称os.path.basename(file_path).split('.')[0]
    # 创建保存的tif名及位置
    subshp_name = str(subtif_name) + ".shp"
    subshp_path = os.path.join(shp_directory, subshp_name)

    create_fishnet(subtif_path, subshp_path, cell_width, cell_height)
    print("CreateFishnet:",subtif_name,"over!")


# #%%单个渔网生成
# template_extent = r"F:\DB\CHM\CHM\dongbeidayangdi\dongbeidayangdi3_CHM.tif"  # 替换为你的模板范围要素类路径
# output_fc = r"F:\DB\CHM\CHMseg02\fishnet\dongbeidayangdi3_CHM.shp"  # 替换为你希望保存渔网的要素类路径
# cell_width = 30  # 替换为你希望的网格宽度
# cell_height = 30  # 替换为你希望的网格高度
# create_fishnet(template_extent,output_fc, cell_width, cell_height)
