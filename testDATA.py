# -*- coding: utf-8 -*-
# @Time    : 2024/6/3 9:03
# @Author  : ChenYuling
# @FileName: testDATA.py
# @Software: PyCharm
#%% @Describe：tree data process  杨秋丽数据处理


import pandas as pd
from pyproj import Proj, Transformer, datadir

# 设置 PROJ 数据目录
datadir.set_data_dir("D:/ProgramData/anaconda3/envs/envs/deepforest/Library/share/proj")

# 定义UTM 50N条带的投影
utm_proj = Proj(proj="utm", zone=50, ellps="WGS84")
# 定义WGS84的投影
wgs84_proj = Proj(proj="latlong", datum="WGS84")

# 创建 Transformer 对象
transformer = Transformer.from_proj(utm_proj, wgs84_proj)

# 读取Excel文件
file_path = r'G:\样方调查数据\SHB_BH.xlsx'
df = pd.read_excel(file_path)

# 假设UTM坐标列名为 'UTM_X' 和 'UTM_Y'
utm_x_coords = df['UTM_X']
utm_y_coords = df['UTM_Y']

# 函数将UTM坐标转换为WGS84坐标
def utm_to_wgs84(utm_x, utm_y):
    lon, lat = transformer.transform(utm_x, utm_y)
    return lon, lat

# 创建新的列保存转换后的经纬度坐标
df['Longitude'], df['Latitude'] = zip(*df.apply(lambda row: utm_to_wgs84(row['UTM_X'], row['UTM_Y']), axis=1))

# 保存结果到新的Excel文件
output_file_path = r'G:\样方调查数据\SHB_BHnew.xlsx'
df.to_excel(output_file_path, index=False)

print(f"转换后的数据已保存到 {output_file_path}")


#%%

import pandas as pd

# 读取Excel文件
excel_file = 'G:\样方调查数据\清源样地.xlsx'
xls = pd.ExcelFile(excel_file)

# 获取Excel文件中所有sheet的名字
sheet_names = xls.sheet_names

# 遍历每个sheet，将其保存为CSV文件
for sheet_name in sheet_names:
    # 读取当前sheet的数据
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    # 构造CSV文件名
    csv_file = f'G:\样方调查数据\qy\{sheet_name}.csv'
    # 保存为CSV文件
    df.to_csv(csv_file, index=False)

print("所有sheet已保存为CSV文件")

#%%
import os
import pandas as pd
csv_directory = r"G:\样方调查数据\qy"
# 存储样地属性的列表
plot_THs = []
sub_tiff_files = [file for file in os.listdir(csv_directory) if file.endswith('.csv')]
for subcsv_file in sub_tiff_files:
    subcsv_path = os.path.join(csv_directory, subcsv_file)
    subcsv_name = os.path.basename(subcsv_path).split('.')[0]
    data = pd.read_csv(subcsv_path, sep=',')#, encoding = 'gb2312'
    data.columns = ['TreeID', 'DBH','TH']#
    if len(data)>0:
        Ha = data['TH'].mean()
        Hw = (data['TH'] * data['TH'] * data['TH']).sum()/(data['TH'] * data['TH']).sum()
        Hl = (data['TH'] * data['DBH'] * data['DBH']).sum()/(data['DBH'] * data['DBH']).sum()

        #将平均高几个指标进行合并
        plot_THs.append((subcsv_name, Ha, Hw,Hl))
    else:
        continue
#%% 创建数据框
dfH = pd.DataFrame(plot_THs, columns=['plot', 'Ha', 'Hw', 'Hl'])
#%%
import arcpy
import os
# 输出目录
shp_directory = r"E:\植被图\20240722\shp_province" #每个植被区下用于保存shp要素

# 创建输出目录（如果不存在）
if not os.path.exists(shp_directory):
    os.makedirs(shp_directory)


input_shapefile = r"G:\TH\FIGS\FIG4\Administrative boundaries of China\Chian_bound.shp"

# # #最开始需要根据渔网格对其分大块，每个植被区只需运行一次就行
# # #TODO: 使用 SplitByAttributes工具分割 Shapefile
split_field = "name" #分割字段
arcpy.analysis.SplitByAttributes(input_shapefile, shp_directory, split_field)
