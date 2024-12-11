# -*- coding: utf-8 -*-
# @Time    : 2024/6/27 9:23
# @Author  : ChenYuling
# @FileName: SI_Modelling.py
# @Software: PyCharm
# @Describe：决策树进行马尾松数据和落叶松数据立地质量评价建模

################################优势高探索立地质量评价###################################
# 决策树
# 原理:决策树是一种递归地将数据划分成小的子集的算法，直到每个子集中的数据点都属于同一个类。
# 它使用特征的阈值来进行分裂，选择使得每次分裂后的子集尽可能纯的特征和阈值。
#%%
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn import tree
import graphviz
import warnings
warnings.filterwarnings("ignore")
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#%%read data
# 读取数据
# madf0 = pd.read_csv(r'E:\TH\WN\20240627\DATA\MA220C.csv', sep=',', encoding='gb2312')
madf0 = pd.read_csv(r'E:\TH\WN\20240627\DATA\luo150d_X.csv', sep=',', encoding='gb2312')
# 删除缺失值
madf1 = madf0.dropna()

#%%读取参数值#气候因子进行降维
df_p = pd.read_csv(r'E:\TH\WN\20240627\DATA\df_loadings.csv',encoding = 'gb2312')
bio19 = madf1[['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7', 'bio_8','bio_9', 'bio_10', 'bio_11', 'bio_12',
               'bio_13', 'bio_14', 'bio_15', 'bio_16','bio_17', 'bio_18', 'bio_19']]
bio19_m = bio19.values  #获取气候数据矩阵
prams = df_p.values #获取PCA载荷矩阵
TPvalues = np.dot(bio19_m,prams)
TPdf = pd.DataFrame(TPvalues)
TPdf.columns = ['TP','T','P']
madf1_col = pd.concat([madf1, TPdf], axis=1) #axis=1按照列

# 新增字段 class
def classify(row):
    if row['y'] < row['h_25']:
        return 'SI1'
    elif row['y'] > row['h_75']:
        return 'SI3'
    else:
        return 'SI2'

madf1_col['class'] = madf1_col.apply(classify, axis=1)
# 选择需要的列
madf2 = madf1_col[['TP','T','P', 'trmc', 'trzd', 'tchd',
               'aspect', 'elevation', 'slope','class']]

#%%地形因子定性处理
# 定义坡向分类的函数
def classify_aspect(aspect):
    if 0 <= aspect < 45:
        return 'North'
    elif 45 <= aspect < 90:
        return 'Northeast'
    elif 90 <= aspect < 135:
        return 'East'
    elif 135 <= aspect < 180:
        return 'Southeast'
    elif 180 <= aspect < 225:
        return 'South'
    elif 225 <= aspect < 270:
        return 'Southwest'
    elif 270 <= aspect < 315:
        return 'West'
    elif 315 <= aspect <= 360:
        return 'Northwest'
    else:
        return ''  # 用于处理无效的aspect值

# 定义坡度分类的函数
def classify_slope(slope):
    if 0 <= slope <= 5:
        return 'FlatSlope'
    elif 6 <= slope <= 15:
        return 'GentleSlope'
    elif 16 <= slope <= 25:
        return 'Incline'
    elif 26 <= slope <= 35:
        return 'AbruptSlope'
    elif 36 <= slope <= 45:
        return 'SteepSlope'
    elif slope > 45:
        return 'DangerousSlope'
    else:
        return ''  # 用于处理无效的slope值

# 定义海拔分类的函数
def classify_elevation(elevation):
    if elevation < 1000:
        return 'Low'
    elif 1000 <= elevation <= 3500:
        return 'Medium'
    elif elevation > 3500:
        return 'High'
    else:
        return ''  # 用于处理无效的elevation值
# 应用分类函数并创建新列
madf2['aspect_str'] = madf2['aspect'].apply(classify_aspect)
madf2['slope_str'] = madf2['slope'].apply(classify_slope)
madf2['elevation_str'] = madf2['elevation'].apply(classify_elevation)


#%% 将 'trzd' 列和 'tchd' 列转换为整数然后转换为字符串
madf2.loc[:, 'trzd']  = madf2['trzd'].round().astype('Int64').astype(str)
madf2.loc[:, 'tchd']  = madf2['tchd'].round().astype('Int64').astype(str)

#%% 定义trzd分类的函数
def classify_trzd(trzd):
    if trzd == '1':
        return 'CoarseTextured'
    elif trzd == '2':
        return 'MediumTextured'
    elif trzd == '3':
        return 'FineTextured'
    else:
        return ''  # 用于处理无效的trzd值


# 定义tchd分类的函数
def classify_tchd(tchd):
    if tchd == '10':
        return 'Thin'
    elif tchd == '30':
        return 'Middle'
    elif tchd == '100':
        return 'Thick'
    else:
        return ''

madf2['tchd_str'] = madf2['tchd'].apply(classify_tchd)
madf2['trzd_str'] = madf2['trzd'].apply(classify_trzd)

#%%读取TRMC
df_TRMC = pd.read_csv(r'E:\TH\WN\20240627\DATA\TRMC.csv')#,encoding = 'gb2312'

# 使用 'trmc' 字段进行左连接
merged_df = pd.merge(madf2, df_TRMC, on='trmc', how='left')
#%%保存马尾松训练集中对应土壤名称
ma_TRMC = merged_df[['TRMC', 'trmc_str']]
ma_TRMC['TRMC'] = ma_TRMC['TRMC'].round().astype('Int64')
# ma_TRMC['trmc'] = ma_TRMC['trmc'].astype(str)
# 删除所有列的重复行
ma_TRMC1 = ma_TRMC.drop_duplicates()
ma_TRMC1.to_csv(r"E:\TH\WN\20240627\DATA\luo_TRMC.csv")
#%% 删除缺失值
madf3 = merged_df.dropna()
madf4 = madf3[['TP', 'T', 'P', 'class', 'aspect_str', 'slope_str', 'elevation_str',
               'tchd_str','trzd_str', 'trmc_str']]

#
# #%% 定义特征和目标变量
# X = madf4.drop('class', axis=1)
# y = madf4['class']
# # 将目标变量编码为数值
# target_mapping = {'SI2': 2,'SI1': 1, 'SI0': 0}
# y_encoded = y.map(target_mapping)
#
# #%% 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=2024)
# from category_encoders import TargetEncoder
# # 使用目标编码对分类特征进行编码
# enc = TargetEncoder(cols=['aspect_str', 'slope_str', 'elevation_str', 'tchd_str', 'trzd_str', 'trmc_str']) #大于
#
# # 转换数据集
# training_numeric_dataset = enc.fit_transform(X_train, y_train)
# testing_numeric_dataset = enc.transform(X_test)
#
#
# #%% 定义目标函数
# def objective(trial):
#     # 定义需要优化的参数
#     max_depth = trial.suggest_int('max_depth', 1, 10)
#     min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
#     min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
#     # 创建决策树模型
#     clf = DecisionTreeClassifier(
#         max_depth=max_depth,
#         min_samples_split=min_samples_split,
#         min_samples_leaf=min_samples_leaf,
#         random_state=2024
#     )
#     # 训练模型
#     clf.fit(training_numeric_dataset, y_train)
#     # 预测
#     y_pred = clf.predict(testing_numeric_dataset)
#     # 计算F1分数
#     f1 = f1_score(y_test, y_pred, average='weighted')
#     return f1
#
# # 创建Optuna的研究对象
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=200)
#
# # 输出最佳参数和最佳F1分数
# print("Best parameters: ", study.best_params)
# print("Best F1: ", study.best_value)
#
# #%%使用决策树算法进行分类
# paras = {'max_depth': 6, 'min_samples_split': 7, 'min_samples_leaf': 7,'random_state':2024}
# clf_best = DecisionTreeClassifier(**paras)
# clf_best.fit(training_numeric_dataset, y_train)
#
# #%% 打印生成的规则
# tree_rules = export_text(clf_best, feature_names=list(training_numeric_dataset.columns))
# print(tree_rules)
# #%% 可视化决策树
# dot_data = tree.export_graphviz(clf_best, out_file=None,
#                                 feature_names=X.columns,
#                                 class_names=y.unique(),
#                                 filled=True, rounded=True,
#                                 special_characters=True)
#
# graph = graphviz.Source(dot_data)
# graph.render("decision_tree", format='png')  # 保存为图像文件
# graph.view()  # 打开默认图像查看器显示决策树
# #%% 对测试数据进行预测
# y_pred = clf_best.predict(testing_numeric_dataset)
# # 计算准确度
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")
# # 计算精确度
# precision = precision_score(y_test, y_pred, average='weighted')
# print(f"Precision: {precision}")
# # 计算召回率
# recall = recall_score(y_test, y_pred, average='weighted')
# print(f"Recall: {recall}")
# # 计算F1分数
# f1 = f1_score(y_test, y_pred, average='weighted')
# print(f"F1-Score: {f1}")
# # 计算混淆矩阵
# conf_matrix = confusion_matrix(y_test, y_pred)
# print("计算混淆矩阵Confusion Matrix:")
# print(conf_matrix)
# # 输出分类报告
# print("分类报告:")
# print(classification_report(y_test, y_pred))


#%%#####调用R语言的决策树模型###########
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import pandas as pd

# 激活 pandas2ri
pandas2ri.activate()

#%% 导入 R 包
base = importr('base')
utils = importr('utils')
rpart = importr('rpart')
rpart_plot = importr('rpart.plot')
rattle = importr('rattle')
caret = importr('caret')
pROC = importr('pROC')

# 设置工作目录
ro.r('setwd("E:/TH/WN/20240627/DATA")')

# 读取数据并移除 NA 值
ro.r('''
data <- read.csv("madf4_150.csv")
cleaned_data <- na.omit(data)
subdata <- cleaned_data[, c('TP', 'T', 'P', 'trmc_str', 'trzd_str', 'tchd_str',
                            'aspect_str', 'elevation_str', 'slope_str', 'class')]
subdata$class <- as.factor(subdata$class)
''')
#%% 读取模型文件
fit2 = ro.r('fit2 <- readRDS("fit2_150model.rds")')


#%% 预测并将结果添加到 subdata 数据框
ro.r('''
predictions <- predict(fit2, subdata, type = "class")
subdata$predictions <- predictions
write.csv(subdata, file = "subdata_with_predictions150.csv", row.names = FALSE)
''')

#%% 读取R保存的模型文件
# 读取数据文件
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,classification_report

subdata_with_predictions = pd.read_csv("E:/TH/WN/20240627/DATA/subdata_with_predictions150.csv")
y_test = subdata_with_predictions["class"]
y_pred = subdata_with_predictions["predictions"]

# classindex = pd.DataFrame(None,columns=['Accuracy','Precision','Recall','F1-Score'])
# 计算准确度
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
# 计算精确度
precision = precision_score(y_test, y_pred, average='weighted')
print(f"Precision: {precision}")
# 计算召回率
recall = recall_score(y_test, y_pred, average='weighted')
print(f"Recall: {recall}")
# 计算F1分数
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1-Score: {f1}")

#%%
classindex = pd.DataFrame([[accuracy,precision,recall,f1]],columns=['Accuracy','Precision','Recall','F1-Score'])
print("模型分类指标:")
print(classindex)
# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("计算混淆矩阵Confusion Matrix:")
print(conf_matrix)
# 输出分类报告
print("分类报告:")
print(classification_report(y_test, y_pred))

#%%
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

subdata_with_predictions = pd.read_csv("E:/TH/WN/20240627/DATA/subdata_with_predictions150.csv")
y_true = subdata_with_predictions["class"]
y_pred = subdata_with_predictions["predictions"]

#%% 类别标签
classes = ['SI0', 'SI1', 'SI2']
n_classes = len(classes)

#%% 将类别标签二值化
y_true_bin = label_binarize(y_true, classes=classes)
y_pred_bin = label_binarize(y_pred, classes=classes)

#%% 计算 ROC 曲线和 AUC 值
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 计算微平均 ROC 曲线和 AUC 值
fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# 绘制 ROC 曲线
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:0.2f})')

plt.plot(fpr["micro"], tpr["micro"], label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})', linestyle='--')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


#%%接下来进行尺度上推

