#%% -*- coding: utf-8 -*-
# @Time        : 2024/9/18 16:16
# @Author      : ChenYuling
# @File        : SI_DirectionalDistribution
# @Desc        : 标准椭圆分析，2倍标准差
import arcpy
from arcpy.sa import *
import os

TreeList = ['110','120','150','200','220','230','250','310','350','420','530']
Timeplist = ['2021_2040','2041_2060','2061_2080','2081_2100']
ssplist = ['ssp585','ssp245','ssp126']
# 设置环境变量和工作空间
arcpy.env.workspace = r"E:\CIMP6\FIGS\Ensemble_3GCMs"
arcpy.env.overwriteOutput = True  # 允许覆盖输出文件
# 嵌套循环遍历 Timeplist 和 ssplist
for TREE in TreeList:
    for Timep in Timeplist:
        for ssp in ssplist:
            print(f'TREE: {TREE}, Timep: {Timep}, ssp: {ssp}')
            # 输出目录
            # point_directory = fr"E:\CIMP6\FIGS\Ensemble_3GCMs\{Timep}\PointSHP"
            # ellipse_directory = fr"E:\CIMP6\FIGS\Ensemble_3GCMs\{Timep}\ellipse1"
            center_point_directory = fr"E:\CIMP6\FIGS\Ensemble_3GCMs\{Timep}\EllipseCenterPoints"

            # # 创建输出目录（如果不存在）
            # if not os.path.exists(point_directory):
            #     os.makedirs(point_directory)
            #
            # # 创建保存目录（如果不存在）
            # if not os.path.exists(ellipse_directory):
            #     os.makedirs(ellipse_directory)

            # 创建保存目录（如果不存在）
            if not os.path.exists(center_point_directory):
                os.makedirs(center_point_directory)

            #%1. 读取栅格数据并计算区间
            # 读取栅格文件
            # raster = arcpy.Raster(fr"E:\CIMP6\FIGS\Ensemble_3GCMs\{Timep}\preSI{TREE}_{ssp}.tif")
            #
            # # 获取最小值和最大值
            # min_value = raster.minimum
            # max_value = raster.maximum
            #
            # # 计算等值3等分的间隔大小
            # interval_size = (max_value - min_value) / 3
            #
            # # 定义区间
            # lower_bound = min_value + 2 * interval_size
            # upper_bound = max_value
            # #% 2. 提取指定区间的数据并转为点 shapefile
            # # 使用Con函数提取栅格中指定区间的数据
            # raster_filtered = Con((raster >= lower_bound) & (raster <= upper_bound), raster)
            #
            # # # 将提取的栅格转换为点
            # output_point_shp = fr"E:\CIMP6\FIGS\Ensemble_3GCMs\{Timep}\PointSHP\preSI{TREE}_{ssp}.shp"
            # #
            # arcpy.RasterToPoint_conversion(raster_filtered, output_point_shp)
            #% 3. 使用 DirectionalDistribution 进行方向分析
            # 设置输出椭圆的路径
            output_ellipse_shp = fr"E:\CIMP6\FIGS\Ensemble_3GCMs\{Timep}\ellipse1\preSI{TREE}_{ssp}.shp"

            # 使用"VALUE"字段作为加权字段进行方向分析
            # arcpy.stats.DirectionalDistribution(output_point_shp, output_ellipse_shp,"1_STANDARD_DEVIATION", "grid_code", None)

            # 4 输出椭圆点图层
            # 输出的点 shapefile 路径
            output_point_shp = fr"E:\CIMP6\FIGS\Ensemble_3GCMs\{Timep}\EllipseCenterPoints\preSI{TREE}_{ssp}.shp"

            # 临时表路径
            temp_table = "in_memory\\temp_table"

            # 将面图层的属性表复制到临时表格
            arcpy.TableToTable_conversion(output_ellipse_shp, "in_memory", "temp_table")

            # 使用 XYTableToPoint_management 将临时表格转换为点图层
            # "CenterX" 和 "CenterY" 字段作为 X 和 Y 坐标
            arcpy.management.XYTableToPoint(temp_table, output_point_shp, "CenterX", "CenterY")

            # 为生成的点图层指定坐标系（根据具体数据选择合适的坐标系）
            spatial_ref = arcpy.SpatialReference(4326)  # WGS 1984
            arcpy.DefineProjection_management(output_point_shp, spatial_ref)

            print(f"点图层已生成，保存于: {output_point_shp}")

#%%
import arcpy
from arcpy.sa import *
import os
TreeList = ['110','120','150','200','220','230','250','310','350','420','530']
# 设置环境变量和工作空间
arcpy.env.workspace = r"E:\CIMP6\FIGS\DataCurrent2020"
arcpy.env.overwriteOutput = True  # 允许覆盖输出文件

# 嵌套循环遍历 Timeplist 和 ssplist
for TREE in TreeList:
    print(f'TREE: {TREE}')
    # 输出目录
    point_directory = fr"E:\CIMP6\FIGS\DataCurrent2020\PointSHP"
    ellipse_directory = fr"E:\CIMP6\FIGS\DataCurrent2020\ellipse1"
    center_point_directory = r"E:\CIMP6\FIGS\DataCurrent2020\EllipseCenterPoints"


    # 创建输出目录（如果不存在）
    if not os.path.exists(point_directory):
        os.makedirs(point_directory)

    # 创建保存目录（如果不存在）
    if not os.path.exists(ellipse_directory):
        os.makedirs(ellipse_directory)

    # 创建保存目录（如果不存在）
    if not os.path.exists(center_point_directory):
        os.makedirs(center_point_directory)

    #%1. 读取栅格数据并计算区间
    # 读取栅格文件
    raster = arcpy.Raster(fr"E:\CIMP6\FIGS\DataCurrent2020\preSI{TREE}.tif")

    # 获取最小值和最大值
    min_value = raster.minimum
    max_value = raster.maximum

    # 计算等值3等分的间隔大小
    interval_size = (max_value - min_value) / 3

    # 定义区间
    lower_bound = min_value + 2 * interval_size
    upper_bound = max_value
    # #% 2. 提取指定区间的数据并转为点 shapefile
    # # 使用Con函数提取栅格中指定区间的数据
    raster_filtered = Con((raster >= lower_bound) & (raster <= upper_bound), raster)
    #
    # # # 将提取的栅格转换为点
    output_point_shp = fr"E:\CIMP6\FIGS\DataCurrent2020\PointSHP\preSI{TREE}.shp"
    # #
    arcpy.RasterToPoint_conversion(raster_filtered, output_point_shp)
    # #% 3. 使用 DirectionalDistribution 进行方向分析
    # # 设置输出椭圆的路径
    output_ellipse_shp = fr"E:\CIMP6\FIGS\DataCurrent2020\ellipse1\preSI{TREE}.shp"

    # # 使用"VALUE"字段作为加权字段进行方向分析
    arcpy.stats.DirectionalDistribution(output_point_shp, output_ellipse_shp,"1_STANDARD_DEVIATION", "grid_code", None)

    # 4 输出点图层
    # 输出的点 shapefile 路径
    output_point_shp = fr"E:\CIMP6\FIGS\DataCurrent2020\EllipseCenterPoints\preSI{TREE}.shp"

    # 临时表路径
    temp_table = "in_memory\\temp_table"

    # 将面图层的属性表复制到临时表格
    arcpy.TableToTable_conversion(output_ellipse_shp, "in_memory", "temp_table")

    # 使用 XYTableToPoint_management 将临时表格转换为点图层
    # "CenterX" 和 "CenterY" 字段作为 X 和 Y 坐标
    arcpy.management.XYTableToPoint(temp_table, output_point_shp, "CenterX", "CenterY")

    # 为生成的点图层指定坐标系（根据具体数据选择合适的坐标系）
    spatial_ref = arcpy.SpatialReference(4326)  # WGS 1984
    arcpy.DefineProjection_management(output_point_shp, spatial_ref)

    print(f"点图层已生成，保存于: {output_point_shp}")



#%% 统计椭圆属性表（未来气候情境下）
import arcpy
import pandas as pd
import os

# TreeList、Timeplist、ssplist
TreeList = ['110', '120', '150', '200', '220', '230', '250', '310', '350', '420', '530']
Timeplist = ['2021_2040', '2041_2060', '2061_2080', '2081_2100']
ssplist = ['ssp126', 'ssp245','ssp585']

# 设置工作空间和环境变量
arcpy.env.workspace = r"E:\CIMP6\FIGS\Ensemble_3GCMs"
arcpy.env.overwriteOutput = True  # 允许覆盖输出文件

# 创建一个空列表，用于存储每次循环中的DataFrame
dataframes = []

# 嵌套循环遍历 Timeplist 和 ssplist
for TREE in TreeList:
    for Timep in Timeplist:
        for ssp in ssplist:
            print(f'TREE: {TREE}, Timep: {Timep}, ssp: {ssp}')
            # 构建shp文件路径
            shp_path = fr"E:\CIMP6\FIGS\Ensemble_3GCMs\{Timep}\ellipse1\preSI{TREE}_{ssp}.shp"
            # 检查文件是否存在
            if os.path.exists(shp_path):
                # 获取字段名称
                fields = [field.name for field in arcpy.ListFields(shp_path) if field.type != 'Geometry']

                # 使用arcpy的SearchCursor来读取shp文件的属性表
                data = []
                with arcpy.da.SearchCursor(shp_path, fields) as cursor:
                    for row in cursor:
                        data.append(row)

                # 将属性表转换为pandas DataFrame
                df = pd.DataFrame(data, columns=fields)
                df['TREE'] = TREE
                df['Timep'] = Timep
                df['ssp'] = ssp

                # 将DataFrame添加到列表中
                dataframes.append(df)
            else:
                print(f"文件未找到: {shp_path}")

# 合并所有DataFrame
dfALL = pd.concat(dataframes, ignore_index=True)

# 输出结果
print(dfALL)

# 可选：将结果保存为CSV文件
dfALL.to_csv("E:\CIMP6\FIGS\Ensemble_3GCMs\merged_shp_data.csv", index=False)


#%% 统计椭圆属性表（2020年）
import arcpy
import pandas as pd
import os
TreeList = ['110','120','150','200','220','230','250','310','350','420','530']
# TreeList、Timeplist、ssplist
TreeList = [110, 120, 150, 200, 220, 230, 250, 310, 350, 420, 530]
# Timeplist = ['2021_2040', '2041_2060', '2061_2080', '2081_2100']
# ssplist = ['ssp126', 'ssp245','ssp585']

# 设置工作空间和环境变量
arcpy.env.workspace = r"E:\CIMP6\FIGS\DataCurrent2020"
arcpy.env.overwriteOutput = True  # 允许覆盖输出文件

# 创建一个空列表，用于存储每次循环中的DataFrame
dataframes = []

# 嵌套循环遍历 Timeplist 和 ssplist
for TREE in TreeList:
    print(f'TREE: {TREE}')
    # 构建shp文件路径
    shp_path = fr"E:\CIMP6\FIGS\DataCurrent2020\ellipse1\preSI{TREE}.shp"
    # 检查文件是否存在
    if os.path.exists(shp_path):
        # 获取字段名称
        fields = [field.name for field in arcpy.ListFields(shp_path) if field.type != 'Geometry']

        # 使用arcpy的SearchCursor来读取shp文件的属性表
        data = []
        with arcpy.da.SearchCursor(shp_path, fields) as cursor:
            for row in cursor:
                data.append(row)

        # 将属性表转换为pandas DataFrame
        df = pd.DataFrame(data, columns=fields)
        df['TREE'] = TREE
        # df['Timep'] = Timep
        # df['ssp'] = ssp

        # 将DataFrame添加到列表中
        dataframes.append(df)
    else:
        print(f"文件未找到: {shp_path}")

# 合并所有DataFrame
dfALL = pd.concat(dataframes, ignore_index=True)

# 输出结果
print(dfALL)

# 可选：将结果保存为CSV文件
dfALL.to_csv("E:\CIMP6\FIGS\Ensemble_3GCMs\merged_shp_data2020.csv", index=False)


#%%3-1从栅格数据中提取特定树种的分布点，然后使用 DirectionalDistribution 进行椭圆方向分析---报错的话就直接去arcgispro中调python脚本界面用
import arcpy
from arcpy.sa import *
import os

# 设置环境变量和工作空间
arcpy.env.workspace = r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585"
arcpy.env.overwriteOutput = True  # 允许覆盖输出文件

TreeList = [110, 120, 150, 200, 220, 230, 250, 310, 350, 420, 530]
point_directory = r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\PointSHP"
ellipse_directory = r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\ellipse"
center_point_directory = r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\EllipseCenterPoints"

# 创建输出目录（如果不存在）
os.makedirs(point_directory, exist_ok=True)
os.makedirs(ellipse_directory, exist_ok=True)
os.makedirs(center_point_directory, exist_ok=True)

# 读取栅格文件
raster = arcpy.Raster(r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\OptimalTreeSpecies585.tif")

# 嵌套循环遍历 TreeList
for TREE in TreeList:
    print(f'Processing TREE: {TREE}')

    # 提取指定区间的数据并转为点 shapefile
    raster_filtered = Con((raster == TREE), raster)
    output_point_shp = os.path.join(point_directory, f"Opl_SI{TREE}.shp")

    # 将提取的栅格转换为点
    arcpy.RasterToPoint_conversion(raster_filtered, output_point_shp)

    # 定义投影（如果需要）
    spatial_ref = arcpy.SpatialReference(4326)  # 4326使用 WGS 1984 或(4547) # CGCS 2000
    arcpy.DefineProjection_management(output_point_shp, spatial_ref)

    # 使用 DirectionalDistribution 进行方向分析
    output_ellipse_shp = os.path.join(ellipse_directory, f"Opl_SI{TREE}.shp")
    try:
        arcpy.stats.DirectionalDistribution(output_point_shp, output_ellipse_shp, "1_STANDARD_DEVIATION", "grid_code",
                                            None)
        print(f'Ellipse generated for TREE: {TREE}')
    except arcpy.ExecuteError:
        print(arcpy.GetMessages())
        continue
#%%3-2保存椭圆属性表中点数据
import arcpy
from arcpy.sa import *
import os

# 设置环境变量和工作空间
arcpy.env.workspace = r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585"
arcpy.env.overwriteOutput = True  # 允许覆盖输出文件

TreeList = [110, 120, 150, 200, 220, 230, 250, 310, 350, 420, 530]
point_directory = r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\PointSHP"
ellipse_directory = r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\ellipse"
center_point_directory = r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\EllipseCenterPoints"

# 创建输出目录（如果不存在）
os.makedirs(point_directory, exist_ok=True)
os.makedirs(ellipse_directory, exist_ok=True)
os.makedirs(center_point_directory, exist_ok=True)
for TREE in TreeList:
    print(f'Processing TREE: {TREE}')
    # # 将面图层的属性表复制到临时表格
    output_ellipse_shp = os.path.join(ellipse_directory, f"Opl_SI{TREE}.shp")
    temp_table = "in_memory\\temp_table"
    arcpy.TableToTable_conversion(output_ellipse_shp, "in_memory", "temp_table")

    # 使用 XYTableToPoint_management 将临时表格转换为点图层
    output_center_point_shp = os.path.join(center_point_directory, f"Opl_SI{TREE}_center.shp")
    arcpy.management.XYTableToPoint(temp_table, output_center_point_shp, "CenterX", "CenterY")

    # 定义输出点图层的投影
    spatial_ref = arcpy.SpatialReference(4326)  # 4326使用 WGS 1984
    arcpy.DefineProjection_management(output_center_point_shp, spatial_ref)
    print(f"Center point layer saved at: {output_center_point_shp}")

#%%3-3保存属性表数据-统计椭圆属性表
import arcpy
import pandas as pd
import os
TreeList = ['110','120','150','200','220','230','250','310','350','420','530']
# 设置工作空间和环境变量
arcpy.env.workspace = r"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\ellipse"
arcpy.env.overwriteOutput = True  # 允许覆盖输出文件

# 创建一个空列表，用于存储每次循环中的DataFrame
dataframes = []

# 嵌套循环遍历 Timeplist 和 ssplist
for TREE in TreeList:
    print(f'TREE: {TREE}')
    # 构建shp文件路径
    shp_path = fr"E:\CIMP6\FIGS\Ensemble_3GCMs\2081_2100\maxmin\585\ellipse\Opl_SI{TREE}.shp"
    # 检查文件是否存在
    if os.path.exists(shp_path):
        # 获取字段名称
        fields = [field.name for field in arcpy.ListFields(shp_path) if field.type != 'Geometry']

        # 使用arcpy的SearchCursor来读取shp文件的属性表
        data = []
        with arcpy.da.SearchCursor(shp_path, fields) as cursor:
            for row in cursor:
                data.append(row)

        # 将属性表转换为pandas DataFrame
        df = pd.DataFrame(data, columns=fields)
        df['TREE'] = TREE
        # df['Timep'] = Timep
        # df['ssp'] = ssp

        # 将DataFrame添加到列表中
        dataframes.append(df)
    else:
        print(f"文件未找到: {shp_path}")

# 合并所有DataFrame
dfALL = pd.concat(dataframes, ignore_index=True)

# 输出结果
print(dfALL)

# 可选：将结果保存为CSV文件
dfALL.to_csv("E:\CIMP6\Figures\Version4\Fig5\Opl_SI2090585.csv", index=False)


#%%计算good和opl两个椭圆中心点之间的椭球面距离
import pandas as pd
import math

# Haversine公式计算两点之间的球面距离
def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0  # 地球半径，单位为公里
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])  # 转换为弧度
    dlon = lon2 - lon1  # 经度差
    dlat = lat2 - lat1  # 纬度差
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2  # Haversine公式
    c = 2 * math.asin(math.sqrt(a))  # 角度
    distance = R * c  # 计算距离
    return distance


# 读取数据集（假设CSV文件中包含您提供的样本数据）
file_path = "E:/CIMP6/Figures/Version4/Fig5/Opl.xlsx"  # Adjust path as necessary
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 创建一个空列表用于保存结果
results = []

# 遍历每个唯一的TREE和Time组合
for (tree, time), group in df.groupby(['TREE', 'Time']):
    # 如果该组合中有多于一行，并且class不相同
    if len(group) > 1 and len(group['class'].unique()) > 1:
        # 获取两个点的经纬度
        point1 = group.iloc[0]  # 假设第一个点
        point2 = group.iloc[1]  # 假设第二个点
        lon1, lat1 = point1['CenterX'], point1['CenterY']
        lon2, lat2 = point2['CenterX'], point2['CenterY']

        # 计算距离
        distance = haversine(lon1, lat1, lon2, lat2)

        # 保存结果
        results.append({
            'TREE': tree,
            'Time': time,
            'Distance': distance
        })

# 将结果转换为DataFrame
distance_df = pd.DataFrame(results)

# 输出结果
print(distance_df)

# 可选：将结果保存为CSV文件
distance_df.to_csv("E:\\CIMP6\\Figures\\Version4\\Fig5\\DistanceResults.csv", index=False)
