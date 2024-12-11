# -*- coding: utf-8 -*-
# @Time    : 2024/1/5 14:42
# @Author  : ChenYuling
# @FileName: TRAIN_HT.py
# @Software: PyCharm
# @Describe：30M训练优势高


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

#%%#####################################################################################################################
###########################################读取数据进行哑变量化处理#################################################
########################################################################################################################
#%read data
df1 = pd.read_csv('./DATA/TrainDATA30m.csv', sep=',')
sym90 = pd.read_csv('./DATA/sym90.csv', sep=',')#trmc代码转名称, encoding = 'gb2312'
data1 = pd.merge(df1, sym90, on='trmc', how='left')
data2 = data1[['H3',
        'bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7', 'bio_8', 'bio_9', 'bio_10',
        'bio_11', 'bio_12', 'bio_13', 'bio_14', 'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19', 'age',
        'VH', 'VV', 'tchd', 'trzd', 'aspect', 'elevation', 'slope', 'pnf', 'NDVI_MAX', 'SU_SYM90']]

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
       'TRMC_SNk', 'TRMC_VRd', 'TRMC_WR', 'PNF_1', 'PNF_2']].values

trainY = data3[['H3']].values
#%
from sklearn.model_selection import train_test_split, KFold
X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, train_size=2/3, random_state=2024)  # 数据集划分


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


#%% RF + Optuna############################################MODEL4
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3,18),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_features':trial.suggest_categorical('max_features', ['auto']),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'random_state': 2024
    }

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
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
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=2024, verbose=False)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print('Best value: ', study.best_value)
study.best_params

