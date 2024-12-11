# -*- coding: utf-8 -*-
# @Time    : 2024/5/10 15:56
# @Author  : ChenYuling
# @FileName: shap.py
# @Software: PyCharm
# @Describe：看因子重要性排序gini+SHAP

#%%忽略一些版本不兼容等警告
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
#%read data
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
# from deepforest import CascadeForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import optuna
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from math import sqrt
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import joblib
#%%#####################################################################################################################
###########################################读取数据进行哑变量化处理#################################################
########################################################################################################################
#%read data
df1 = pd.read_csv('./DATA/TrainDATA30m.csv', sep=',')
sym90 = pd.read_csv('./DATA/sym90.csv', sep=',')#trmc代码转名称, encoding = 'gb2312'
data1 = pd.merge(df1, sym90, on='trmc', how='left')

#%%
data22 = data1[['H3','REGION',
        'bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7', 'bio_8', 'bio_9', 'bio_10',
        'bio_11', 'bio_12', 'bio_13', 'bio_14', 'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19', 'age',
        'VH', 'VV', 'tchd', 'trzd', 'aspect', 'elevation', 'slope', 'pnf', 'NDVI_MAX', 'SU_SYM90']]
data2 = data1[['H3',
        'bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7', 'bio_8', 'bio_9', 'bio_10',
        'bio_11', 'bio_12', 'bio_13', 'bio_14', 'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19', 'age',
        'VH', 'VV', 'tchd', 'trzd', 'aspect', 'elevation', 'slope', 'pnf', 'NDVI_MAX', 'SU_SYM90']]
#%
data2 = data2.dropna(axis=0,how='any')
data22 = data22.dropna(axis=0,how='any')
#%%处理离散数据-哑变量处理
# 将列转换为字符类型
data2['trzd'] = data2['trzd'].round().astype('Int64')
data2['trzd'] = data2['trzd'].astype(str)
data2['pnf'] = data2['pnf'].round().astype('Int64')
data2['pnf'] = data2['pnf'].astype(str)
# sub_df['pnf'] = round(sub_df['pnf'])
data3 = pd.get_dummies(
    data2,
    columns=['trzd','SU_SYM90','pnf'],
    prefix=['TRZD','TRMC','PNF'],
    prefix_sep="_",
    dummy_na=False,
    drop_first=False)

#%%#####################################################################################################################
###########################################独立检验数据集汇总结果（未调参数）#################################################
########################################################################################################################
trainX = data3[['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7',
       'bio_8', 'bio_9', 'bio_10', 'bio_11', 'bio_12', 'bio_13', 'bio_14',
       'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19', 'age', 'VH', 'VV',
       'tchd', 'aspect', 'elevation', 'slope', 'NDVI_MAX', 'TRZD_1', 'TRZD_2',
       'TRZD_3', 'TRMC_ACf', 'TRMC_ACh', 'TRMC_ACp', 'TRMC_ACu', 'TRMC_ALf',
       'TRMC_ALh', 'TRMC_ANh', 'TRMC_ANu', 'TRMC_ARb', 'TRMC_ARc', 'TRMC_ARh',
       'TRMC_ATc', 'TRMC_CHh', 'TRMC_CHl', 'TRMC_CMc', 'TRMC_CMd', 'TRMC_CMe',
       'TRMC_CMg', 'TRMC_CMo', 'TRMC_CMx', 'TRMC_FLc', 'TRMC_FLe', 'TRMC_FRh',
       'TRMC_FRx', 'TRMC_GLm', 'TRMC_GLt', 'TRMC_GRh', 'TRMC_GYp', 'TRMC_KSh',
       'TRMC_KSk', 'TRMC_LPe', 'TRMC_LPm', 'TRMC_LVa', 'TRMC_LVg', 'TRMC_LVh',
       'TRMC_LVj', 'TRMC_LVk', 'TRMC_LVx', 'TRMC_LXa', 'TRMC_LXf', 'TRMC_NTu',
       'TRMC_PHh', 'TRMC_PLe', 'TRMC_RGc', 'TRMC_RGd', 'TRMC_RGe', 'TRMC_SCk',
       'TRMC_SNk', 'TRMC_VRd', 'TRMC_WR', 'PNF_1', 'PNF_2']]

trainY = data3[['H3']]
#%%
from sklearn.model_selection import train_test_split, KFold
X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, train_size=2/3, random_state=2024)  # 数据集划分
#%%
# loaded_modelh3 = joblib.load(filename="./model/lightgbm_h3_0509.model")#加载模型, encoding = 'gb2312'
loaded_modelh1 = joblib.load(filename="./model/lightgbm_h1_0509.model")#加载模型, encoding = 'gb2312'
#%% 输出特征重要性
# 特征重要度
print('Feature importances:', list(loaded_modelh1.feature_importances_))
#%%绘制特征重要性图
feature_importances = loaded_modelh1.feature_importances_

#%%保存重要性
columns = ['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7',
       'bio_8', 'bio_9', 'bio_10', 'bio_11', 'bio_12', 'bio_13', 'bio_14',
       'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19', 'age', 'VH', 'VV',
       'tchd', 'aspect', 'elevation', 'slope', 'NDVI_MAX', 'TRZD_1', 'TRZD_2',
       'TRZD_3', 'TRMC_ACf', 'TRMC_ACh', 'TRMC_ACp', 'TRMC_ACu', 'TRMC_ALf',
       'TRMC_ALh', 'TRMC_ANh', 'TRMC_ANu', 'TRMC_ARb', 'TRMC_ARc', 'TRMC_ARh',
       'TRMC_ATc', 'TRMC_CHh', 'TRMC_CHl', 'TRMC_CMc', 'TRMC_CMd', 'TRMC_CMe',
       'TRMC_CMg', 'TRMC_CMo', 'TRMC_CMx', 'TRMC_FLc', 'TRMC_FLe', 'TRMC_FRh',
       'TRMC_FRx', 'TRMC_GLm', 'TRMC_GLt', 'TRMC_GRh', 'TRMC_GYp', 'TRMC_KSh',
       'TRMC_KSk', 'TRMC_LPe', 'TRMC_LPm', 'TRMC_LVa', 'TRMC_LVg', 'TRMC_LVh',
       'TRMC_LVj', 'TRMC_LVk', 'TRMC_LVx', 'TRMC_LXa', 'TRMC_LXf', 'TRMC_NTu',
       'TRMC_PHh', 'TRMC_PLe', 'TRMC_RGc', 'TRMC_RGd', 'TRMC_RGe', 'TRMC_SCk',
       'TRMC_SNk', 'TRMC_VRd', 'TRMC_WR', 'PNF_1', 'PNF_2']
df = pd.DataFrame()
df['feature name'] = columns
#%%
df['importance'] = list(loaded_modelh1.feature_importances_)

# df = df.sort_values('importance')

#%% 确保 feature_importances 是整数标量数组
feature_importances = np.array(feature_importances).astype(int)
# 获取特征名称
feature_names = ['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7',
       'bio_8', 'bio_9', 'bio_10', 'bio_11', 'bio_12', 'bio_13', 'bio_14',
       'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19', 'age', 'VH', 'VV',
       'tchd', 'aspect', 'elevation', 'slope', 'NDVI_MAX', 'TRZD_1', 'TRZD_2',
       'TRZD_3', 'TRMC_ACf', 'TRMC_ACh', 'TRMC_ACp', 'TRMC_ACu', 'TRMC_ALf',
       'TRMC_ALh', 'TRMC_ANh', 'TRMC_ANu', 'TRMC_ARb', 'TRMC_ARc', 'TRMC_ARh',
       'TRMC_ATc', 'TRMC_CHh', 'TRMC_CHl', 'TRMC_CMc', 'TRMC_CMd', 'TRMC_CMe',
       'TRMC_CMg', 'TRMC_CMo', 'TRMC_CMx', 'TRMC_FLc', 'TRMC_FLe', 'TRMC_FRh',
       'TRMC_FRx', 'TRMC_GLm', 'TRMC_GLt', 'TRMC_GRh', 'TRMC_GYp', 'TRMC_KSh',
       'TRMC_KSk', 'TRMC_LPe', 'TRMC_LPm', 'TRMC_LVa', 'TRMC_LVg', 'TRMC_LVh',
       'TRMC_LVj', 'TRMC_LVk', 'TRMC_LVx', 'TRMC_LXa', 'TRMC_LXf', 'TRMC_NTu',
       'TRMC_PHh', 'TRMC_PLe', 'TRMC_RGc', 'TRMC_RGd', 'TRMC_RGe', 'TRMC_SCk',
       'TRMC_SNk', 'TRMC_VRd', 'TRMC_WR', 'PNF_1', 'PNF_2']
# 根据特征重要性排序
sorted_idx = np.argsort(feature_importances)[::-1]
sorted_feature_importances = feature_importances[sorted_idx]
# 确保 sorted_idx 是整数数组
sorted_idx = np.array(sorted_idx).astype(int)
# 根据特征重要性排序特征名称
sorted_feature_names = [feature_names[idx] for idx in sorted_idx]


sorted_feature_importances = sorted_feature_importances[::-1]
sorted_feature_names = sorted_feature_names[::-1]

plt.figure(figsize=(10, 12))  # 调整图形大小
plt.barh(range(len(sorted_feature_importances)), sorted_feature_importances, align='center')
plt.yticks(range(len(sorted_feature_importances)), sorted_feature_names)  # 使用 yticks
plt.xlabel('Importance')  # 修改 x 轴标签
plt.ylabel('Feature')  # 修改 y 轴标签
plt.title('Feature Importance')  # 修改标题
plt.show()

##############################################################################################
#%%########################**************SHAP***************########################################
import shap
import warnings
warnings.filterwarnings("ignore")
explainer = shap.TreeExplainer(loaded_modelh1)
shap_values = explainer.shap_values(X_test)
# shap_values_abs = np.abs(shap_values)  # 绝对值
# shap_imp_abs = shap_values_abs.mean(axis=0)  # 绝对值求均值
# shap_imp = shap_values.mean(axis=0)  # 求均值

#%% combin 左闭右开
shap_left = shap_values[:,0:27]
# ['SF_AGE', 'TreeDensit', 'TreeHeight', 'Annual_T', 'Annual_P', 'Aspect','Elevation', 'Slope', 'Footprint']
shap_TRMC = shap_values[:, 30:80]  # TRMC_一些列
shap_TRZD = shap_values[:, [27,28,29]]  #
shap_PNF = shap_values[:, [80,81]]  #

shapTRMC = shap_TRMC.sum(axis=1).reshape(-1, 1)  # 求和
shapTRZD = shap_TRZD.sum(axis=1).reshape(-1, 1)  # 求和
shapPNF = shap_PNF.sum(axis=1).reshape(-1, 1)  # 求和

merged_shap = np.concatenate((shap_left, shapTRZD, shapTRMC, shapPNF), axis=1)  # h 合并

#%%
merged_shap = np.load(r'G:\TH\FIGS\FIG2\SHAP\merged_shap_h3.npy')

#%% combin 左闭右开 对测试集也进行相同合并处理
X_left = X_test.values[:,0:27]
# ['SF_AGE', 'TreeDensit', 'TreeHeight', 'Annual_T', 'Annual_P', 'Aspect','Elevation', 'Slope', 'Footprint']
X_TRMC = X_test.values[:, 30:80]  # TRMC_一些列
X_TRZD = X_test.values[:, [27,28,29]]  #
X_PNF = X_test.values[:, [80,81]]  #

xTRMC = X_TRMC.sum(axis=1).reshape(-1, 1)  # 求和
xTRZD = X_TRZD.sum(axis=1).reshape(-1, 1)  # 求和
xPNF = X_PNF.sum(axis=1).reshape(-1, 1)  # 求和

merged_x = np.concatenate((X_left, xTRZD, xTRMC, xPNF), axis=1)  # h 合并

#%% 保存为.npy文件
np.save('figs/fig_SHAP/merged_shap_h1.npy', merged_shap)


#%%
shap_imp = merged_shap.mean(axis=0)  # 求均值

merged_shap_abs = np.abs(merged_shap)  # 绝对值
shap_imp_abs = merged_shap_abs.mean(axis=0)  # 绝对值求均值


Xname = ['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7',
       'bio_8', 'bio_9', 'bio_10', 'bio_11', 'bio_12', 'bio_13', 'bio_14',
       'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19', 'age', 'VH', 'VV',
       'tchd', 'aspect', 'elevation', 'slope', 'NDVI_MAX', 'TRZD', 'TRMC',
       'PNF']

#%%
shap_impDF = pd.DataFrame({'Xname': Xname, 'shap_mean_abs': shap_imp_abs, 'shap_mean': shap_imp})

#%%
feature_importance = shap_imp
# 排序特征重要性分数
sorted_idx = np.argsort(feature_importance)

# 创建一个颜色数组，前四个特征使用蓝色，后四个特征使用绿色
colors = ['#4477be' if i < 15 else '#fa8e8e' for i in range(len(sorted_idx))]
#设置误差棒颜色和大小
error_kw = {'ecolor': '#381da6', 'capsize': 4}

# 设置全局的字体大小和样式
plt.rcParams["font.size"] = 14
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(10,10), dpi=500)

# 去除外边框
plt.box(False)
# 添加竖线网格
plt.grid(axis='x', linestyle='--', alpha=0.7)

# 绘制条形图(带误差棒)
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx],  align="center", height=0.5, color=colors, error_kw=error_kw)# 设置颜色
plt.yticks(range(len(sorted_idx)), [Xname[i] for i in sorted_idx])
plt.xlabel("Importance")
plt.ylabel("Feature")
# plt.title("Features importance with Error Bars")

# # 保存图片到本地
# plt.savefig("./figs/fig3/H1_abs.jpg", dpi=400, format='jpeg',pad_inches=0.02,bbox_inches='tight')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中


plt.show()

#%%
import matplotlib.pyplot as plt

# print the JS visualization code to the notebook
shap.initjs()

# 设置画布
fig = plt.figure(facecolor="white", figsize=(10, 10))  #
# shap.summary_plot(merged_shap, merged_x, feature_names=Xname,show = False,alpha=0.2, color='#abcdef')# , plot_type="bar",,cmap='summer'
shap.summary_plot(merged_shap, merged_x, feature_names=Xname,max_display=35,show = False,alpha=0.2)
plt.xticks(fontsize=10)  # 默认字体大小为10
plt.yticks(fontsize=10)
# plt.xlabel('SHAP value', fontdict={'weight': 'normal', 'size': 10})  # 改变坐标轴标题字体
# 设置全局的字体大小和样式
plt.rcParams["font.size"] = 14
plt.rcParams["font.family"] = "Times New Roman"
# 设置x轴范围为1到5
# plt.xlim(-1.0, 1.0)
# plt.title('Wet MO', fontdict={'weight': 'normal', 'size': 10});
# plt.savefig("./figs/fig3/shao_h1.svg", dpi=1000, format='svg', pad_inches=0.02,
#             bbox_inches='tight')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
# # 保存图片到本地
# plt.savefig("./figs/fig3/shap_ha.jpg", dpi=400, format='jpeg',pad_inches=0.02,bbox_inches='tight')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中

plt.show()



