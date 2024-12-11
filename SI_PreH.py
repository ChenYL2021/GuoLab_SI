# -*- coding: utf-8 -*-
# @Time        : 2024/8/4 1:36
# @Author      : ChenYuling
# @File        : SI_PreH.py
# @Desc        : 计算一清数据中主要树种的优势高

#%%忽略一些版本不兼容等警告
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
#%%#####################################################################################################################
###########################################读取数据进行哑变量化处理#################################################
########################################################################################################################
TREE = 'LENGSHAN110' #TODO:file name
#%read data
df1 = pd.read_csv(r'F:\WorkingNotes\TH\WN\20240729\THdata\{}HT.csv'.format(TREE), sep=',')
df1.rename(columns={'SU_SYM90': 'trmc'}, inplace=True)
df1["S_age"] = 30  #TODO:计算SI的标准年龄
sym90 = pd.read_csv('./DATA/sym90.csv', sep=',')#trmc代码转名称, encoding = 'gb2312'
data1 = pd.merge(df1, sym90, on='trmc', how='left')
#%%空值处理
#0和‘’视为空值
data1.loc[:, 'pnf'] = data1['pnf'].apply(lambda x: np.nan if x == 0 else x)
data1.replace('', np.nan, inplace=True)
# 计算每列的众数
mode_values = data1.mode().iloc[0]
# 使用每列的众数填充空值
data2 = data1.fillna(mode_values)
#%%
data3 = data2[[
        'bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7', 'bio_8', 'bio_9', 'bio_10',
        'bio_11', 'bio_12', 'bio_13', 'bio_14', 'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19', 'S_age',
        'VH', 'VV', 'tchd', 'trzd', 'aspect', 'elevation', 'slope', 'pnf', 'NDVI_MAX', 'SU_SYM90']]

data4 = data3.dropna(axis=0,how='any')

#%%处理离散数据-哑变量处理
# 将列转换为字符类型
data4['trzd'] = data4['trzd'].round().astype('Int64')
data4['trzd'] = data4['trzd'].astype(str)
data4['pnf'] = data4['pnf'].round().astype('Int64')
data4['pnf'] = data4['pnf'].astype(str)
# sub_df['pnf'] = round(sub_df['pnf'])
data5 = pd.get_dummies(
    data4,
    columns=['trzd','SU_SYM90','pnf'],
    prefix=['TRZD','TRMC','PNF'],
    prefix_sep="_",
    dtype=int,
    dummy_na=False,
    drop_first=False)
#%%
columns_to_update = ['TRMC_ACf', 'TRMC_ACh', 'TRMC_ACp', 'TRMC_ALf', 'TRMC_ANh', 'TRMC_ANu', 'TRMC_ARb', 'TRMC_ARc', 'TRMC_ARh', 'TRMC_ATc', 'TRMC_CMg', 'TRMC_CMo', 'TRMC_CMx', 'TRMC_FLc', 'TRMC_FLe', 'TRMC_FRh', 'TRMC_FRx', 'TRMC_GLt', 'TRMC_GYp', 'TRMC_KSh', 'TRMC_KSk', 'TRMC_LVg', 'TRMC_LVj', 'TRMC_LVx', 'TRMC_LXa', 'TRMC_LXf', 'TRMC_NTu', 'TRMC_PHh', 'TRMC_RGc', 'TRMC_RGd', 'TRMC_SCk', 'TRMC_SNk', 'TRMC_VRd', 'TRMC_WR']

# 将指定列的值设置为 0
data5[columns_to_update] = 0
#%%
X_th = data5[['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7',
       'bio_8', 'bio_9', 'bio_10', 'bio_11', 'bio_12', 'bio_13', 'bio_14',
       'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19', 'S_age', 'VH', 'VV',
       'tchd', 'aspect', 'elevation', 'slope', 'NDVI_MAX', 'TRZD_1', 'TRZD_2',
       'TRZD_3', 'TRMC_ACf', 'TRMC_ACh', 'TRMC_ACp', 'TRMC_ACu', 'TRMC_ALf',
       'TRMC_ALh', 'TRMC_ANh', 'TRMC_ANu', 'TRMC_ARb', 'TRMC_ARc', 'TRMC_ARh',
       'TRMC_ATc', 'TRMC_CHh', 'TRMC_CHl', 'TRMC_CMc', 'TRMC_CMd', 'TRMC_CMe',
       'TRMC_CMg', 'TRMC_CMo', 'TRMC_CMx', 'TRMC_FLc', 'TRMC_FLe', 'TRMC_FRh',
       'TRMC_FRx', 'TRMC_GLm', 'TRMC_GLt', 'TRMC_GRh', 'TRMC_GYp', 'TRMC_KSh',
       'TRMC_KSk', 'TRMC_LPe', 'TRMC_LPm', 'TRMC_LVa', 'TRMC_LVg', 'TRMC_LVh',
       'TRMC_LVj', 'TRMC_LVk', 'TRMC_LVx', 'TRMC_LXa', 'TRMC_LXf', 'TRMC_NTu',
       'TRMC_PHh', 'TRMC_PLe', 'TRMC_RGc', 'TRMC_RGd', 'TRMC_RGe', 'TRMC_SCk',
       'TRMC_SNk', 'TRMC_VRd', 'TRMC_WR', 'PNF_1', 'PNF_2']].values

#%% X_train
loaded_model = joblib.load(filename="model/lightgbm_hT_0620.model")#加载模型
preHT = loaded_model.predict(X_th)# 使用模型对测试数据进行预测
df1["SI"] = preHT

#%%
df1.to_csv(r"F:\WorkingNotes\TH\WN\20240729\SIdata\{}_SI.csv".format(TREE), index=False,encoding='utf-8-sig')


######################################################################################
##########################立地质量评价################################################
#%%#####################################################################################
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
def classSI(SI,one,two):
    if SI <= one:
        return "poor"
    elif SI >= two:
        return "good"
    else:
        return "medium"

# 自定义颜色列表
# Define a color palette as a dictionary
custom_palette = {
    'good': '#CC79A7',    # Blue
    'medium': '#f8c471',  # Orange
    'poor': '#85c1e9'      # Green
}
# custom_palette = ['#f8c471', '#CC79A7', '#85c1e9']  # 例如蓝色、橙色和绿色
# 循环读取CSV数据
# 设置要读取的文件夹路径
folder_path = r"F:\WorkingNotes\TH\WN\20240729\SIdata"
# 获取文件夹中的所有文件
files = os.listdir(folder_path)
# 遍历文件夹中的所有文件
for file in files:
    # 检查文件是否为CSV文件
    if file.endswith('.csv'):
        file_path = os.path.join(folder_path, file)
        # read data
        SIdata = pd.read_csv(file_path)
        print(f"Reading file: {file}")
        # # 计算三分之一和三分之二分位数
        # one_third_quantile = SIdata['SI'].quantile(1/3)
        # two_third_quantile = SIdata['SI'].quantile(2/3)
        # max min 3等分
        # 计算最小值和最大值
        min_value = SIdata['SI'].min()
        max_value = SIdata['SI'].max()
        interval_size = (max_value - min_value) / 3 # 计算每个区间的边界
        left_value = min_value+interval_size
        right_value = min_value+interval_size*2

        # SIdata['SIclass'] = SIdata['SI'].apply(classSI,one=one_third_quantile,two=two_third_quantile)
        SIdata['SIclass'] = SIdata['SI'].apply(classSI, one=left_value, two=right_value)
        SIclass_name = file.split('_')[0]+'_SIclass.csv'
        SIclassfile_path = os.path.join(folder_path, SIclass_name)
        #save
        SIdata.to_csv(SIclassfile_path, index=False, encoding='utf-8-sig')
        #figure
        order = ['good', 'medium', 'poor']
        # 创建 countplot
        plt.figure(figsize=(4, 4))  # 可选：设置图形大小
        sns.countplot(data=SIdata, x='SIclass', palette=custom_palette, legend=False, order=order)
        # 添加标题和标签
        plt.title(file.split('_')[0])
        plt.xlabel('SIclass')
        plt.ylabel('Count')
        # 保存图像到文件
        SIclass_figure = file.split('_')[0] + '.png'
        figurefile_path = os.path.join(folder_path, SIclass_figure)
        plt.savefig(figurefile_path)  # 你可以选择其他格式，如 .pdf, .svg 等
        # 显示图形（可选）
        # plt.show()



#%%三维立地图
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# 设置要读取的文件夹路径
folder_path = r"F:\WorkingNotes\TH\WN\20240729\SIdata"
# 获取文件夹中的所有文件
files = os.listdir(folder_path)
# 遍历文件夹中的所有文件
for file in files:
    # 检查文件是否为CSV文件
    if file.endswith('.csv'):
        file_path = os.path.join(folder_path, file)
        # read data
        SIdata = pd.read_csv(file_path)
        print(f"Reading file: {file}")
        #figure
        # 创建3D 图形对象
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        # 绘制 3D 散点图







        sc = ax.scatter(SIdata['T2'], SIdata['preHT'], SIdata['SI'], c=SIdata['SI'], cmap='viridis')
        # 设置坐标轴标签
        ax.set_xlabel('Age')
        ax.set_ylabel('TH')
        ax.set_zlabel('SI')
        # 添加颜色条
        plt.colorbar(sc, ax=ax, label='Age')
        # 保存图像到文件
        SIclass_figure = file.split('_')[0] + '_3D.png'
        figurefile_path = os.path.join(folder_path, SIclass_figure)
        plt.savefig(figurefile_path)  # 你可以选择其他格式，如 .pdf, .svg 等
        # 显示图形（可选）
        # plt.show()


