#%% -*- coding: utf-8 -*-
# @Time    : 2023/10/24 14:00
# @Author  : ChenYuling
# @FileName: DBFcsv.py
# @Software: PyCharm
# @Describe：将dbf转为csv

import os
from simpledbf import Dbf5
dbf_directory = r"F:\DB\TH\Xdata"
csv_directory = r"F:\DB\TH\Xdata"

# 获取文件夹中的tif文件路径列表
sub_tiff_files = [file for file in os.listdir(dbf_directory) if file.endswith('.dbf')]
for subdbf_file in sub_tiff_files:
    subdbf_path = os.path.join(dbf_directory, subdbf_file)
    subdbf_name = os.path.basename(subdbf_path).split('.')[0]
    data = Dbf5(subdbf_path)
    # 创建保存的csv名及位置
    subcsv_name = str(subdbf_name) + ".csv"
    subcsv_path = os.path.join(csv_directory, subcsv_name)
    data.to_csv(subcsv_path)


#%%
from simpledbf import Dbf5
import os
import pandas as pd
datdbf = Dbf5("G:\TH\lidar30m.dbf")
datdbf.to_csv("G:\TH\lidar30m.csv")
