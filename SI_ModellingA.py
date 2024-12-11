# -*- coding: utf-8 -*-
# @Time        : 2024/8/5 17:14
# @Author      : ChenYuling
# @File        : SI_ModellingA
# @Desc        : 分树种构建地位级指数模型
#%%忽略一些版本不兼容等警告
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
#%read data
import seaborn as sns
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from math import sqrt
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import joblib
#%%#####################################################################################################################
###########################################读取数据进行哑变量化处理#################################################
########################################################################################################################
TREE='200'
#%read data
df1 = pd.read_csv(r'F:\WorkingNotes\TH\WN\20240729\THdata\YOUSONG{}HT.csv'.format(TREE), sep=',')
sym90 = pd.read_csv('./DATA/sym90.csv', sep=',')#trmc代码转名称, encoding = 'gb2312'
data1 = pd.merge(df1, sym90, on='trmc', how='left')

#%空值处理
#0和‘’视为空值
data1.loc[:, 'pnf'] = data1['pnf'].apply(lambda x: np.nan if x == 0 else x)
data1.replace('', np.nan, inplace=True)
# 计算每列的众数
mode_values = data1.mode().iloc[0]
# 使用每列的众数填充空值
data2 = data1.fillna(mode_values)
#%
data3 = data2[['preHT',
        'bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7', 'bio_8', 'bio_9', 'bio_10',
        'bio_11', 'bio_12', 'bio_13', 'bio_14', 'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19', 'T2',
        'VH', 'VV', 'tchd', 'trzd', 'aspect', 'elevation', 'slope', 'pnf', 'NDVI_MAX', 'SU_SYM90']]

data4 = data3.dropna(axis=0,how='any')

#%处理离散数据-哑变量处理
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

#%%#####################################################################################################################
###########################################初始化模型筛选汇总结果（未调参数）#################################################
########################################################################################################################
# check version
from pycaret.utils import version
version()
################### Setup ➡️ Compare Models ➡️ Analyze Model ➡️ Prediction ➡️ Save Model ##############################

# This function initializes the training environment and creates the transformation pipeline. Setup function must be called before executing any other function. It takes two required parameters: data and target. All the other parameters are optional.
from pycaret.regression import *

s = setup(data3, target = 'preHT', session_id = 2024,use_gpu = True,train_size = 2/3)
# check available models
all_models = models()

#%Compare Models
best = compare_models()
print(best)

allmodels = pull()

allmodels.to_csv(r"F:\WorkingNotes\TH\WN\20240729\THdata\MODEL1\allmodels{}.csv".format(TREE), index=False,encoding='utf-8-sig')
# #%%
# compare_tree_models = compare_models(include = ['et', 'rf', 'dt', 'gbr', 'xgboost', 'lightgbm', 'catboost','ada','ridge','br'])  #H1
# # compare_tree_models = compare_models(include = ['et', 'rf', 'dt', 'gbr', 'xgboost', 'lightgbm', 'catboost','ada','ridge','br'])    #H2
# # compare_tree_models = compare_models(include = ['et', 'rf', 'dt', 'gbr', 'xgboost', 'lightgbm', 'catboost','ada','ridge'])         #H3
#
# #%
# compare_tree_models_results = pull()
# compare_tree_models_results
#%%#####################################################################################################################
###########################################独立检验数据集汇总结果（未调参数）#################################################
########################################################################################################################
trainX = data5[['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7',
       'bio_8', 'bio_9', 'bio_10', 'bio_11', 'bio_12', 'bio_13', 'bio_14',
       'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19', 'T2', 'VH', 'VV',
       'tchd', 'aspect', 'elevation', 'slope', 'NDVI_MAX', 'TRZD_1', 'TRZD_2',
       'TRZD_3', 'TRMC_ACp', 'TRMC_ACu', 'TRMC_ALh', 'TRMC_ARb', 'TRMC_ARc',
       'TRMC_ARh', 'TRMC_ATa', 'TRMC_ATc', 'TRMC_CHg', 'TRMC_CHh', 'TRMC_CHk',
       'TRMC_CHl', 'TRMC_CLh', 'TRMC_CMc', 'TRMC_CMd', 'TRMC_CMe', 'TRMC_CMg',
       'TRMC_CMo', 'TRMC_CMu', 'TRMC_DS', 'TRMC_FLc', 'TRMC_FLe', 'TRMC_FLs',
       'TRMC_GLe', 'TRMC_GLk', 'TRMC_GLm', 'TRMC_GRh', 'TRMC_GYh', 'TRMC_GYk',
       'TRMC_GYp', 'TRMC_KSh', 'TRMC_KSk', 'TRMC_KSl', 'TRMC_LPe', 'TRMC_LPi',
       'TRMC_LPk', 'TRMC_LPm', 'TRMC_LVa', 'TRMC_LVg', 'TRMC_LVh', 'TRMC_LVk',
       'TRMC_LVx', 'TRMC_PDd', 'TRMC_PHc', 'TRMC_PHg', 'TRMC_PHh', 'TRMC_PHj',
       'TRMC_PLd', 'TRMC_PLe', 'TRMC_RGc', 'TRMC_RGe', 'TRMC_SCg', 'TRMC_SCh',
       'TRMC_SCk', 'TRMC_SCm', 'TRMC_SCy', 'TRMC_SNg', 'TRMC_UR', 'TRMC_VRd',
       'TRMC_VRe', 'TRMC_WR', 'PNF_1', 'PNF_2']]

trainY = data5[['preHT']]
#%
from sklearn.model_selection import train_test_split, KFold
X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, train_size=2/3, random_state=2024)  # 数据集划分

#%#####################################################################################################################
###########################################独立检验数据集进行调参过程#################################################
########################################################################################################################
#%% LGBM + Optuna#########################################################################################################
def objective(trial):
    params = {
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 11, 333),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.02, 0.05, 0.005, 0.1]),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.5),
        'n_estimators': trial.suggest_int('n_estimators', 7000, 8000),
        'cat_smooth': trial.suggest_int('cat_smooth', 10, 100),
        'cat_l2': trial.suggest_int('cat_l2', 1, 20),
        'min_data_per_group': trial.suggest_int('min_data_per_group', 50, 200),
        'cat_feature': trial.suggest_int('cat_feature', 10, 60),
        'n_jobs': -1,
        'force_col_wise': 'true',
        'random_state': 2024,
    }
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print('Best value: ', study.best_value)
study.best_params



#%% #######################CatBoost + Optuna######################MODEL2
import time
start = time.process_time()
def objective(trial):
    params = {
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.005, 0.01, 0.05]),
        'iterations': trial.suggest_int('iterations', 5000, 8000),
        'max_bin': trial.suggest_int('max_bin', 200, 400),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.0001, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.8),
        'random_seed': 2024
    }

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=2022, verbose=False)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print('Best value: ', study.best_value)
# study.best_params
end = time.process_time()
print('CPU Times ', end-start)
study.best_params



#%%############################################Xgboost + Optuna#####################################################
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.005, 0.01]),
        'n_estimators': trial.suggest_int('n_estimators', 2000, 8000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'gamma': trial.suggest_float('gamma', 0.0001, 1.0, log=True),
        'reg_alpha': trial.suggest_float('alpha', 0.0001, 10.0, log=True),
        'reg_lambda': trial.suggest_float('lambda', 0.0001, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.8),
        'subsample': trial.suggest_float('subsample', 0.6, 0.8),
        'random_state': 2024
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)],  verbose=False)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print('Best value: ', study.best_value)
study.best_params


########################################################################################################################
####################################################对已经调好的参数模型，统计其变量值########################################
########################################################################################################################
#%%统计调参后结果
import pandas as pd
INDEX_TRAIN = pd.DataFrame(None,columns=['Model','R2','RMSE','MSE','MAE','ME'])
INDEX_TEST = pd.DataFrame(None,columns=['Model','R2','RMSE','MSE','MAE','ME'])

#%%1 CatBoostRegressor
paras = {'depth': 6,
 'learning_rate': 0.05,
 'iterations': 6923,
 'max_bin': 391,
 'min_data_in_leaf': 28,
 'l2_leaf_reg': 0.6862744571275803,
 'subsample': 0.7064193711942983,
         'random_state': 2024}


model1 = CatBoostRegressor(**paras)
model1.fit(X_train, y_train,eval_set=[(X_test, y_test)],plot=True)

pred_train = model1.predict(X_train)
R2_train = r2_score(y_train, pred_train)
MSE_train = mean_squared_error(y_train, pred_train, squared=False)
RMSE_train = MSE_train**0.5
MAE_train = mean_absolute_error(y_train, pred_train)
# ME_train = np.mean(y_train- pred_train)
#%
pred_test = model1.predict(X_test)
R2_test = r2_score(y_test, pred_test)
RMSE_test = mean_squared_error(y_test, pred_test, squared=False)
MSE_test = mean_squared_error(y_test, pred_test)
MAE_test = mean_absolute_error(y_test, pred_test)
# ME_test = np.mean(y_test-pred_test)

INDEX_TRAIN1 = pd.DataFrame([['CatBoostRegressor',R2_train,RMSE_train,MSE_train,MAE_train]],columns=['Model','R2','RMSE','MSE','MAE'])
INDEX_TEST1 = pd.DataFrame([['CatBoostRegressor',R2_test,RMSE_test,MSE_test,MAE_test]],columns=['Model','R2','RMSE','MSE','MAE'])
#%
INDEX_TRAIN = pd.concat([INDEX_TRAIN, INDEX_TRAIN1], ignore_index=True)
INDEX_TEST = pd.concat([INDEX_TEST, INDEX_TEST1], ignore_index=True)

print(INDEX_TRAIN)
print(INDEX_TEST)

#%%
#将训练的模型保存到磁盘(value=模型名)   默认当前文件夹下
joblib.dump(filename = r"F:\WorkingNotes\TH\WN\20240729\THdata\MODEL3/Cat{}.model".format(TREE),value=model1)


#%%5 LGBMRegressor
paras = {'reg_alpha': 0.794635547674086,
 'reg_lambda': 7.650900219601761,
 'num_leaves': 213,
 'min_child_samples': 27,
 'max_depth': 9,
 'learning_rate': 0.02,
 'colsample_bytree': 0.47623426283242953,
 'n_estimators': 7605,
 'cat_smooth': 66,
 'cat_l2': 18,
 'min_data_per_group': 172,
 'cat_feature': 18,
 'random_state': 2024}

#%
# model5 = joblib.load(filename="./model/newh1.model")#加载模型
model5 = LGBMRegressor(**paras)
model5.fit(X_train, y_train)
#%
#%
pred_train = model5.predict(X_train)
R2_train = r2_score(y_train, pred_train)
MSE_train = mean_squared_error(y_train, pred_train, squared=False)
RMSE_train = MSE_train**0.5
MAE_train = mean_absolute_error(y_train, pred_train)
# ME_train = np.mean(y_train- pred_train)

pred_test = model5.predict(X_test)
R2_test = r2_score(y_test, pred_test)
RMSE_test = mean_squared_error(y_test, pred_test, squared=False)
MSE_test = mean_squared_error(y_test, pred_test)
MAE_test = mean_absolute_error(y_test, pred_test)
# ME_test = np.mean(y_test-pred_test)
#%
INDEX_TRAIN5 = pd.DataFrame([['LGBMRegressor',R2_train,RMSE_train,MSE_train,MAE_train]],columns=['Model','R2','RMSE','MSE','MAE'])
INDEX_TEST5 = pd.DataFrame([['LGBMRegressor',R2_test,RMSE_test,MSE_test,MAE_test]],columns=['Model','R2','RMSE','MSE','MAE'])
#%
INDEX_TRAIN = pd.concat([INDEX_TRAIN, INDEX_TRAIN5], ignore_index=True)
INDEX_TEST = pd.concat([INDEX_TEST, INDEX_TEST5], ignore_index=True)

print(INDEX_TRAIN)
print(INDEX_TEST)
# #%%
# preht_test = y_test
# preht_test['preHT'] =  pred_test

#%%

#将训练的模型保存到磁盘(value=模型名)   默认当前文件夹下
joblib.dump(filename = r"F:\WorkingNotes\TH\WN\20240729\THdata\MODEL3/LGBM{}.model".format(TREE),value=model5)


#%%6 XGBRegressor
paras = {'max_depth': 7,
 'learning_rate': 0.01,
 'n_estimators': 5366,
 'min_child_weight': 18,
 'gamma': 0.0001284684894370148,
 'alpha': 0.011113888363673304,
 'lambda': 0.013157721227618551,
 'colsample_bytree': 0.5973320710525675,
 'subsample': 0.6312002108369503,
 'random_state': 2024}


model6 = XGBRegressor(**paras)
model6.fit(X_train, y_train)

pred_train = model6.predict(X_train)
R2_train = r2_score(y_train, pred_train)
MSE_train = mean_squared_error(y_train, pred_train, squared=False)
RMSE_train = MSE_train**0.5
MAE_train = mean_absolute_error(y_train, pred_train)
# ME_train = np.mean(y_train- pred_train)
#%
pred_test = model6.predict(X_test)
R2_test = r2_score(y_test, pred_test)
RMSE_test = mean_squared_error(y_test, pred_test, squared=False)
MSE_test = mean_squared_error(y_test, pred_test)
MAE_test = mean_absolute_error(y_test, pred_test)
# ME_test = np.mean(y_test-pred_test)

INDEX_TRAIN6 = pd.DataFrame([['XGBRegressor',R2_train,RMSE_train,MSE_train,MAE_train]],columns=['Model','R2','RMSE','MSE','MAE'])
INDEX_TEST6 = pd.DataFrame([['XGBRegressor',R2_test,RMSE_test,MSE_test,MAE_test]],columns=['Model','R2','RMSE','MSE','MAE'])
#
INDEX_TRAIN = pd.concat([INDEX_TRAIN, INDEX_TRAIN6], ignore_index=True)
INDEX_TEST = pd.concat([INDEX_TEST, INDEX_TEST6], ignore_index=True)

print(INDEX_TRAIN)
print(INDEX_TEST)

#%%
#将训练的模型保存到磁盘(value=模型名)   默认当前文件夹下
joblib.dump(filename = r"F:\WorkingNotes\TH\WN\20240729\THdata\MODEL3/XGB{}.model".format(TREE),value=model6)

#%%
modeldf = pd.concat([INDEX_TRAIN, INDEX_TEST], ignore_index=True)
modeldf.to_csv(r"F:\WorkingNotes\TH\WN\20240729\THdata\MODEL2\M_{}.csv".format(TREE), index=False,encoding='utf-8-sig')