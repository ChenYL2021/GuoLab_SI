# -*- coding: utf-8 -*-
# @Time    : 2024/4/29 10:46
# @Author  : ChenYuling
# @FileName: figures.py
# @Software: PyCharm
# @Describe：密度散点图
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import rcParams
from statistics import mean
from sklearn.metrics import explained_variance_score,r2_score,median_absolute_error,mean_squared_error,mean_absolute_error
from scipy.stats import pearsonr
config = {"font.family":'Times New Roman',"font.size": 16,"mathtext.fontset":'stix'}
import pandas as pd
#%%
df = pd.read_csv('./DATA/H3dat.csv')#,encoding="gb2312"

x = df['predict_H3']; y = df['H3']
#%%
# 计算散点密度
xy = np.vstack([x, y])
z = stats.gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x.iloc[idx], y.iloc[idx], z[idx]
#%% 保存为.npy文件
np.save('./figs/fig2/z_H3.npy', z)
np.save('./figs/fig2/y_H3.npy', y)
np.save('./figs/fig2/x_H3.npy', x)

#%%LOAD
z = np.load(r'F:\FIGS\FIG3\H3\z_H3.npy')
y = np.load(r'F:\FIGS\FIG3\H3\y_H3.npy')
x = np.load(r'F:\FIGS\FIG3\H3\x_H3.npy')
#%%
# 拟合（若换MK，自行操作）最小二乘
def slope(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs) * mean(xs)) - mean(xs * xs)))
    b = mean(ys) - m * mean(xs)
    return m, b
k, b = slope(x, y)
regression_line = []
for a in x:
    regression_line.append((k * a) + b)
#%%
BIAS = mean(x - y)
MSE = mean_squared_error(x, y)
RMSE = np.power(MSE, 0.5)
r = pearsonr(x, y)[0]
adjR2 = 1-((1-r2_score(x,y))*(len(x)-1))/(len(x)-1-1)
MAE = mean_absolute_error(x, y)
EV = explained_variance_score(x, y)
NSE = 1 - (RMSE ** 2 / np.var(x))
# BIAS: 偏差的均值
# MSE: 均方误差
# RMSE: 均方根误差
# R2: Pearson相关系数
# adjR2: 调整后的R²
# MAE: 平均绝对误差
# EV: 解释的方差分数
# NSE: Nash-Sutcliffe效率
# print(RMSE,R2)
#%%
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
scatter = ax.scatter(x, y, marker='o', c=z * 100, edgecolors=None, s=15,   alpha=0.8)
cbar = plt.colorbar(scatter, shrink=1, orientation='vertical', extend='both', pad=0.015, aspect=30, label='frequency')
plt.plot(x, regression_line, 'black', lw=1.5, label='Regression Line')
# plt.axis([0, 22, 0, 22])  # 设置线的范围
plt.plot([0, 60], [0, 60], 'red', lw=1.5, linestyle='--', label='1:1 line')
plt.show()
#%%


#%%
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
scatter = ax.scatter(x, y, marker='o', c=z * 100, edgecolors=None, s=15,   alpha=0.8)
cbar = plt.colorbar(scatter, shrink=1, orientation='vertical', extend='both', pad=0.015, aspect=30, label='frequency')
plt.plot([0, 60], [0, 60], 'red', lw=1.5, linestyle='--', label='1:1 line')
plt.plot(x, regression_line, 'black', lw=1.5, label='Regression Line')
plt.text(10,-12.5, '$R^2=%.3f$' % R2, family = 'Times New Roman', horizontalalignment='right')
plt.text(10,-17.5, '$NSE=%.3f$' % NSE, family = 'Times New Roman', horizontalalignment='right')
plt.text(10,-22.5, '$MAE=%.3f$' % MAE, family = 'Times New Roman', horizontalalignment='right')
plt.text(10,-27.5, '$RMSE=%.3f$' % RMSE, family = 'Times New Roman', horizontalalignment='right')

plt.axis([0, 60, 0, 60])  # 设置线的范围
ax.legend(loc='upper left', frameon = False)
plt.show()
plt.savefig("E:\\River\\WN\\20231111\\sheng\\zhejiang10\\figures_ADD\\" + str(strNAME) + ".svg", dpi=1000, format='svg',
            pad_inches=0.02,
            bbox_inches='tight')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
#%%
# n = len(y)
# t_value = 1.96  # 95% 置信区间对应的 t 值
# std_err = np.std(y - (k * x + b))
# margin_of_error = t_value * (std_err / np.sqrt(n))
# lower_confidence_bound = k * x + b - margin_of_error
# upper_confidence_bound = k * x + b + margin_of_error

fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
# plt.plot(x, lower_confidence_bound, linestyle='--', color='black', dashes=(1, 40), label='95% Prediction Band')
# plt.plot(x, upper_confidence_bound, linestyle='--', color='black', dashes=(1, 40)), cmap='RdBu_r'
scatter = ax.scatter(x, y, marker='o', c=z * 100, edgecolors=None,  s=12,   alpha=0.8)
cbar = plt.colorbar(scatter, shrink=1, orientation='vertical', extend='both', pad=0.015, aspect=30, label='Frequency')
plt.plot([0, 70], [0, 70], 'red', lw=1.5, linestyle='--', label='1:1 line')#, label='1:1 line'
plt.plot(x, regression_line, 'black', lw=1.5, label='Regression Line')
ax.grid(True, linestyle='--', alpha=0.2)

plt.text(3.5,65, '$r=%.3f$' % r, family = 'Arial', horizontalalignment='left',fontsize=12)
plt.text(3.5,61, '$RMSE=%.1f$' % RMSE, family = 'Arial', horizontalalignment='left',fontsize=12)
plt.text(3.5,57, '$MAE =%.1f$' % MAE, family = 'Arial', horizontalalignment='left',fontsize=12)

plt.axis([0, 70, 0, 70])  # 设置线的范围
plt.ylabel('Hw (m)', family = 'Arial',fontsize=12)  # 改变坐标轴标题字体
plt.xlabel('Predicted Hw (m)', family = 'Arial',fontsize=12)  # 改变坐标轴标题字体
ax.legend(loc='lower right', frameon = False)
# plt.savefig("./figs/fig2/H3.svg", dpi=400, format='svg',pad_inches=0.02,bbox_inches='tight')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
# plt.show()

plt.savefig("F:/FIGS/FIG3/H3.jpg", dpi=600, format='jpeg',pad_inches=0.02,bbox_inches='tight')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
plt.savefig("F:/FIGS/FIG3/H3.svg", dpi=600, format='svg',pad_inches=0.02,bbox_inches='tight')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中


