# -*- coding: utf-8 -*-
# @Time        : 2024/9/3 9:09
# @Author      : ChenYuling
# @File        : FIGmeanH
# @Desc        : 验证加权公式
#%%
import matplotlib.pyplot as plt
import numpy as np

# 定义x的范围和y的函数
h = np.arange(5, 101)  # x从5到100
d = np.arange(5, 101)  # x从5到100
w1 = d*d
w2 = h*h
Hl = h * w1
Hw = h * w2
# 绘制曲线图
plt.plot(h, Hl, label='Hl ~ h', color='b', linestyle='-', marker='o')
plt.plot(h, Hw, label='Hw ~ h', color='r', linestyle='-', marker='o')
plt.xlabel('h')
plt.ylabel('Hl')
plt.title('Plot of Hl ~ h')
plt.legend()
plt.grid(True)
plt.show()

#%%
import os
import pandas as pd

# 文件夹路径
folder_path = r'E:\杨秋丽样方调查数据\qy'  # 替换为你的文件夹路径

# 初始化一个空的DataFrame来存储所有数据
all_data = pd.DataFrame()

# 遍历文件夹中的所有CSV文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        # 获取PlotID（去掉文件扩展名）
        plot_id = os.path.splitext(file_name)[0]

        # 构建文件的完整路径
        file_path = os.path.join(folder_path, file_name)

        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 确保CSV文件至少有3列
        if df.shape[1] >= 3:
            # 提取第一列为treeid，第二列为dbh，第三列为th
            df_subset = df.iloc[:, [0, 1, 2]].copy()
            df_subset.columns = ['treeid', 'dbh', 'th']  # 重命名列

            # 添加PlotID列
            df_subset['PlotID'] = plot_id

            # 将当前文件的数据追加到总数据中
            all_data = pd.concat([all_data, df_subset], ignore_index=True)

#%%输出合并后的数据
print(all_data)

# 如果需要，可以将合并后的数据保存为一个新的CSV文件
all_data.to_csv('combined_data.csv', index=False)

#%%
import os
import pandas as pd
import matplotlib.pyplot as plt

# 文件夹路径
file_path = r'E:\杨秋丽样方调查数据\TREE_1.csv'  # 替换为你的文件夹路径
# 读取CSV文件
all_data = pd.read_csv(file_path)
# 绘制曲线散点图
plt.figure(figsize=(10, 6))


# 添加对角线（45度线）
min_value = min(all_data['DBH'].min(), all_data['TH'].min())
max_value = max(all_data['DBH'].max(), all_data['TH'].max())
plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--', label='45 Degree Line')

# 遍历不同的PlotID，分别绘制散点图
plt.scatter(all_data['DBH'], all_data['TH'])
plt.xlabel('DBH \ cm')
plt.ylabel('TH \ m')
plt.legend()
plt.grid(True)
plt.show()

#%%
# 文件夹路径
plotdf = r'E:\杨秋丽样方调查数据\PLOTALLH.csv'
# 读取CSV文件
plot_data = pd.read_csv(plotdf)

#%% 使用 isin 进行筛选
all_data_1 = all_data[all_data['PLOTID'].isin(plot_data['PLOTID'])]

#%% 使用 isin 进行筛选
all_data_1 = all_data[all_data['PLOTID'].isin(plot_data['PLOTID'])]


#%%筛选hl-hw散点图样地数据对应的样木数据
import pandas as pd
# 文件夹路径
tree = r'E:\TH\FIGS\FIG11\FIG3\allTREE.csv'  # 替换为你的文件夹路径
plot = r'E:\TH\FIGS\FIG11\FIG3\BJ.csv'
# 读取CSV文件
tree_data = pd.read_csv(tree)
plot_data = pd.read_csv(plot)

# 在 plot_data 中创建一个布尔掩码，用于标识每个记录的存在
mask = (tree_data['Time'].isin(plot_data['Time'])) & (tree_data['PLOT'].isin(plot_data['PLOT']))
# 选取满足条件的 tree_data 中的记录
filtered_data = tree_data[mask]
filtered_data['E'] = filtered_data['DBH'] - filtered_data['H']
#%%
import seaborn as sns
import matplotlib.pyplot as plt

# 绘制频率分布图
sns.histplot(filtered_data['E'], bins=30, kde=True)  # 使用直方图和核密度估计曲线
plt.xlabel('DBH-Height')  # 设置x轴标签
plt.ylabel('Frequency')  # 设置y轴标签
# plt.title('Frequency Distribution of (DBH-H)')  # 设置图标题
# 在 E = 0 处绘制红色垂直虚线
plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5)  # x=0表示E的值为0的位置，color='red'表示红色，linestyle='--'表示虚线



plt.savefig(r"E:\TH\FIGS\FIG11\FIG3\dbhH.jpg", dpi=600, format='jpeg',pad_inches=0.02,bbox_inches='tight')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中

plt.show()