# -*- coding: utf-8 -*-
# @Time    : 2023/12/6 8:42
# @Author  : ChenYuling
# @FileName: Train_data979.py
# @Software: PyCharm
# @Describe：将chm和lidar数据进行汇总，然后进行模型训练初探

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
df1 = pd.read_csv(r'G:\TH\Ydata\20240222/data2.csv', sep=',', encoding = 'gb2312')

df1.dropna(inplace=True)
#%%
data2 = df1[['H1',
        'bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7', 'bio_8', 'bio_9', 'bio_10',
        'bio_11', 'bio_12', 'bio_13', 'bio_14', 'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19', 'age',
        'VH', 'VV', 'tchd', 'trzd', 'aspect', 'elevation', 'slope', 'pnf', 'NDVI_MAX', 'SU_SYM90']]

data2 = data2.dropna(axis=0,how='any')#去除空值

#%%处理离散数据-量处理
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
###########################################初始化模型筛选汇总结果（未调参数）#################################################
########################################################################################################################
# check version
from pycaret.utils import version
version()
################### Setup ➡️ Compare Models ➡️ Analyze Model ➡️ Prediction ➡️ Save Model ##############################

# This function initializes the training environment and creates the transformation pipeline. Setup function must be called before executing any other function. It takes two required parameters: data and target. All the other parameters are optional.
from pycaret.regression import *
s = setup(data3, target = 'H1', session_id = 2024,use_gpu = True,train_size = 2/3)
# check available models
# all_models = models()

#%%Compare Models
best = compare_models()
print(best)

allmodels = pull()
#%%
compare_tree_models = compare_models(include = ['et', 'rf', 'dt', 'gbr', 'xgboost', 'lightgbm', 'catboost','ada','ridge','br'])  #H1
# compare_tree_models = compare_models(include = ['et', 'rf', 'dt', 'gbr', 'xgboost', 'lightgbm', 'catboost','ada','ridge','br'])    #H2
# compare_tree_models = compare_models(include = ['et', 'rf', 'dt', 'gbr', 'xgboost', 'lightgbm', 'catboost','ada','ridge'])         #H3

#%
compare_tree_models_results = pull()
compare_tree_models_results
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
       'TRMC_KSk', 'TRMC_KSl', 'TRMC_LPe', 'TRMC_LPm', 'TRMC_LVa', 'TRMC_LVg',
       'TRMC_LVh', 'TRMC_LVj', 'TRMC_LVk', 'TRMC_LVx', 'TRMC_LXa', 'TRMC_LXf',
       'TRMC_NTu', 'TRMC_PHh', 'TRMC_PLe', 'TRMC_RGc', 'TRMC_RGd', 'TRMC_RGe',
       'TRMC_SCk', 'TRMC_SNk', 'TRMC_VRd', 'TRMC_WR', 'PNF_0', 'PNF_1',
       'PNF_2']]

trainY = data3[['H1']]
#%%
from sklearn.model_selection import train_test_split, KFold
X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, train_size=2/3, random_state=2024)  # 数据集划分

#%%
import pandas as pd
AGEdata = pd.DataFrame(None,columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
#%%# 1 DecisionTreeRegressor
dt_model = DecisionTreeRegressor(random_state=2024)
dt_model.fit( X_train, y_train)
dt_predict = dt_model.predict(X_test)
dt_predict1 = dt_model.predict(X_train)
# #测试集验证结果
AGE1 = pd.DataFrame([['DecisionTreeRegressor',r2_score(y_train, dt_predict1), sqrt(mean_squared_error(y_train, dt_predict1)),r2_score(y_test, dt_predict),sqrt(mean_squared_error(y_test, dt_predict))]],columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
AGEdata= AGEdata.append(AGE1)
print(AGEdata)

#%%# 2ExtraTreeRegressor
et_model = ExtraTreeRegressor(random_state=2024)
et_model.fit( X_train, y_train)
et_predict = et_model.predict(X_test)
et_predict1 = et_model.predict(X_train)
AGE2 = pd.DataFrame([['ExtraTreeRegressor',r2_score(y_train, et_predict1), sqrt(mean_squared_error(y_train, et_predict1)),r2_score(y_test, et_predict),sqrt(mean_squared_error(y_test, et_predict))]],columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
AGEdata= AGEdata.append(AGE2)
print(AGEdata)

#%% 3XGBRegressor####################model1
xgbr_model = XGBRegressor(random_state=2024)
xgbr_model.fit(X_train, y_train)
xgbr_predict = xgbr_model.predict(X_test)
xgbr_predict1 = xgbr_model.predict(X_train)

AGE4 = pd.DataFrame([['XGBRegressor',r2_score(y_train, xgbr_predict1), sqrt(mean_squared_error(y_train, xgbr_predict1)),r2_score(y_test, xgbr_predict),sqrt(mean_squared_error(y_test, xgbr_predict))]],columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
AGEdata= AGEdata.append(AGE4)
print(AGEdata)
#%% 4 HistGradientBoostingRegressor#############
HistGB_model = HistGradientBoostingRegressor(random_state=2024)
HistGB_model.fit(X_train, y_train)
HistGB_predict = HistGB_model.predict(X_test)
HistGB_predict1 = HistGB_model.predict(X_train)
AGE5 = pd.DataFrame([['HistGradientBoostingRegressor',r2_score(y_train, HistGB_predict1), sqrt(mean_squared_error(y_train, HistGB_predict1)),r2_score(y_test, HistGB_predict),sqrt(mean_squared_error(y_test, HistGB_predict))]],columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
AGEdata= AGEdata.append(AGE5)
print(AGEdata)
#%%# 5 LGBMRegressor###################model3
ligthgbmc_model = LGBMRegressor(random_state=2024)
ligthgbmc_model.fit(X_train, y_train)
ligthgbmc_predict = ligthgbmc_model.predict(X_test)
ligthgbmc_predict1 = ligthgbmc_model.predict(X_train)
AGE6 = pd.DataFrame([['LGBMRegressor',r2_score(y_train, ligthgbmc_predict1), sqrt(mean_squared_error(y_train, ligthgbmc_predict1)),r2_score(y_test, ligthgbmc_predict),sqrt(mean_squared_error(y_test, ligthgbmc_predict))]],columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
AGEdata= AGEdata.append(AGE6)
print(AGEdata)

#%%6 RandomForestRegressor####################model5
rf_model = RandomForestRegressor(random_state=2024)
rf_model.fit(X_train, y_train)
rf_predict = rf_model.predict(X_test)
rf_predict1 = rf_model.predict(X_train)
AGE7 = pd.DataFrame([['RandomForestRegressor',r2_score(y_train, rf_predict1), sqrt(mean_squared_error(y_train, rf_predict1)),r2_score(y_test, rf_predict),sqrt(mean_squared_error(y_test, rf_predict))]],columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
AGEdata= AGEdata.append(AGE7)
print(AGEdata)

#%%7 GradientBoostingRegressor################model4
gb_model = GradientBoostingRegressor(random_state=2024)
gb_model.fit(X_train, y_train)
gb_predict = gb_model.predict(X_test)
gb_predict1 = gb_model.predict(X_train)

AGE9 = pd.DataFrame([['GradientBoostingRegressor',r2_score(y_train, gb_predict1), sqrt(mean_squared_error(y_train, gb_predict1)),r2_score(y_test, gb_predict),sqrt(mean_squared_error(y_test, gb_predict))]],columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
AGEdata= AGEdata.append(AGE9)
print(AGEdata)
#%%8 AdaBoostRegressor###############
AdaB_model = AdaBoostRegressor(random_state=2024)
AdaB_model.fit(X_train, y_train)
AdaB_predict = AdaB_model.predict(X_test)
AdaB_predict1 = AdaB_model.predict(X_train)

AGE11 = pd.DataFrame([['AdaBoostRegressor',r2_score(y_train, AdaB_predict1), sqrt(mean_squared_error(y_train, AdaB_predict1)),r2_score(y_test, AdaB_predict),sqrt(mean_squared_error(y_test, AdaB_predict))]],columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
AGEdata= AGEdata.append(AGE11)
print(AGEdata)
#%%12 KNeighborsRegressor###############
KN_model = KNeighborsRegressor(n_jobs=-1)
KN_model.fit(X_train, y_train)
KN_predict = KN_model.predict(X_test)
KN_predict1 = KN_model.predict(X_train)

AGE12 = pd.DataFrame([['KNeighborsRegressor',r2_score(y_train, KN_predict1), sqrt(mean_squared_error(y_train, KN_predict1)),r2_score(y_test, KN_predict),sqrt(mean_squared_error(y_test, KN_predict))]],columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
AGEdata= AGEdata.append(AGE12)
print(AGEdata)

#%%# 9CatBoostRegressor#############model2
Cat_model = CatBoostRegressor(random_state=2024, verbose=False)
Cat_model.fit(X_train, y_train)
Cat_predict = Cat_model.predict(X_test)
Cat_predict1 = Cat_model.predict(X_train)
AGE13 = pd.DataFrame([['CatBoostRegressor',r2_score(y_train, Cat_predict1), sqrt(mean_squared_error(y_train, Cat_predict1)),r2_score(y_test, Cat_predict),sqrt(mean_squared_error(y_test, Cat_predict))]],columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
AGEdata= AGEdata.append(AGE13)
print(AGEdata)

#%%10 deepforest-CascadeForestRegressor
from deepforest import CascadeForestRegressor

deeprf_model = CascadeForestRegressor(random_state=2024)
deeprf_model.fit(X_train, y_train)
deeprf_predict = deeprf_model.predict(X_test)
deeprf_predict1 = deeprf_model.predict(X_train)
AGE10 = pd.DataFrame([['CascadeForestRegressor',r2_score(y_train, deeprf_predict1), sqrt(mean_squared_error(y_train, deeprf_predict1)),r2_score(y_test, deeprf_predict),sqrt(mean_squared_error(y_test, deeprf_predict))]],columns=['Model','TrainR2','TrainRMSE','TestR2','TestRMSE'])
AGEdata= AGEdata.append(AGE10)
print(AGEdata)
#%%#####################################################################################################################
###########################################独立检验数据集进行调参过程#################################################
########################################################################################################################
# GBDT + Optuna#########################################################################################################
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 10),#大容易过拟合
        'learning_rate': trial.suggest_categorical('learning_rate', [0.001,0.005,0.01,0.05,0.1]),#大容易过拟合
        'n_estimators': trial.suggest_int('n_estimators', 4000, 5000),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),#小容易过拟合
        'max_features':trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),#大容易过拟合
        'random_state': 2024
    }

    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return rmse
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print('Best value: ', study.best_value)
study.best_params

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
study.optimize(objective, n_trials=10)
print('Best value: ', study.best_value)
study.best_params

#%% #######################CatBoost + Optuna######################MODEL2
import time
start = time.process_time()
def objective(trial):
    params = {
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.005, 0.01,0.03,0.05]),
        'iterations': trial.suggest_int('iterations', 5000, 8000),
        'max_bin': trial.suggest_int('max_bin', 200, 400),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.0001, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.8),
        'random_seed': 2024
    }

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=2024, verbose=False)
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

#%% #######################CascadeForestRegressor + Optuna######################MODEL2
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_layers': trial.suggest_int('max_layers', 3, 10),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'random_state': 2024
    }
    model = CascadeForestRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print('Best value: ', study.best_value)
study.best_params

#%%####################### HistGradientBoostingRegressor + Optuna ############################################
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),#大容易过拟合
        'learning_rate': trial.suggest_categorical('learning_rate', [0.001,0.005,0.01,0.05,0.1]),#大容易过拟合
        'max_leaf_nodes':trial.suggest_int('max_leaf_nodes', 30, 40),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 15, 25),#大容易过拟合
        'random_state': 2024
    }

    model = HistGradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print('Best value: ', study.best_value)
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


#%%############################################AdaBoost + Optuna#####################################################
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'loss': trial.suggest_categorical('loss', ['linear', 'exponential']),
        'base_estimator': DecisionTreeRegressor(max_depth=trial.suggest_int('max_depth', 1, 5)),
        'random_state': 2024
    }
    model = AdaBoostRegressor(**params)
    model.fit(X_train, y_train)
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
paras = {'depth': 10,
 'learning_rate': 0.05,
 'iterations': 7994,
 'max_bin': 309,
 'min_data_in_leaf': 18,
 'l2_leaf_reg': 0.00852652656040184,
 'subsample': 0.7060358070781709,
         'random_state': 2024}


model1 = CatBoostRegressor(**paras)
model1.fit(X_train, y_train,eval_set=[(X_test, y_test)],plot=True)
#%%
pred_train = model1.predict(X_train)
R2_train = r2_score(y_train, pred_train)
MSE_train = mean_squared_error(y_train, pred_train, squared=False)
RMSE_train = MSE_train**0.5
MAE_train = mean_absolute_error(y_train, pred_train)
# ME_train = np.mean(y_train- pred_train)
#%
pred_test = model1.predict(X_test)
R2_test = r2_score(y_test, pred_test)
MSE_test = mean_squared_error(y_test, pred_test, squared=False)
RMSE_test = MSE_test**0.5
MAE_test = mean_absolute_error(y_test, pred_test)
# ME_test = np.mean(y_test-pred_test)

INDEX_TRAIN1 = pd.DataFrame([['CatBoostRegressor',R2_train,RMSE_train,MSE_train,MAE_train]],columns=['Model','R2','RMSE','MSE','MAE'])
INDEX_TEST1 = pd.DataFrame([['CatBoostRegressor',R2_test,RMSE_test,MSE_test,MAE_test]],columns=['Model','R2','RMSE','MSE','MAE'])
#
INDEX_TRAIN= INDEX_TRAIN.append(INDEX_TRAIN1)
INDEX_TEST= INDEX_TEST.append(INDEX_TEST1)
print(INDEX_TRAIN)
print(INDEX_TEST)

#%%
#将训练的模型保存到磁盘(value=模型名)   默认当前文件夹下
joblib.dump(filename =r"model/h1_catboost.model", value=model1)

#%%2 HistGradientBoostingRegressor
paras = {'max_depth': 6,
         'learning_rate': 0.1,
         'max_leaf_nodes': 36,
         'min_samples_leaf': 15,
        'random_state': 2024}

model2 = HistGradientBoostingRegressor(**paras)
model2.fit(X_train, y_train)


pred_train = model2.predict(X_train)
R2_train = r2_score(y_train, pred_train)
MSE_train = mean_squared_error(y_train, pred_train, squared=False)
RMSE_train = MSE_train**0.5
MAE_train = mean_absolute_error(y_train, pred_train)
ME_train = np.mean(y_train- pred_train)

pred_test = model2.predict(X_test)
R2_test = r2_score(y_test, pred_test)
MSE_test = mean_squared_error(y_test, pred_test, squared=False)
RMSE_test = MSE_test**0.5
MAE_test = mean_absolute_error(y_test, pred_test)
ME_test = np.mean(y_test-pred_test)

INDEX_TRAIN2 = pd.DataFrame([['HistGradientBoostingRegressor',R2_train,RMSE_train,MSE_train,MAE_train,ME_train]],columns=['Model','R2','RMSE','MSE','MAE','ME'])
INDEX_TEST2 = pd.DataFrame([['HistGradientBoostingRegressor',R2_test,RMSE_test,MSE_test,MAE_test,ME_test]],columns=['Model','R2','RMSE','MSE','MAE','ME'])
#
INDEX_TRAIN= INDEX_TRAIN.append(INDEX_TRAIN2)
INDEX_TEST= INDEX_TEST.append(INDEX_TEST2)
print(INDEX_TRAIN)
print(INDEX_TEST)


#%%3 GradientBoostingRegressor
paras = {'max_depth': 9,
         'learning_rate': 0.001,
         'n_estimators': 4124,
         'subsample': 0.8938061287060033,
         'max_features': 'log2',
         'min_samples_split': 8,
         'random_state': 2024}

model3 = GradientBoostingRegressor(**paras)
model3.fit(X_train, y_train)

pred_train = model3.predict(X_train)
R2_train = r2_score(y_train, pred_train)
MSE_train = mean_squared_error(y_train, pred_train, squared=False)
RMSE_train = MSE_train**0.5
MAE_train = mean_absolute_error(y_train, pred_train)
ME_train = np.mean(y_train- pred_train)

pred_test = model3.predict(X_test)
R2_test = r2_score(y_test, pred_test)
MSE_test = mean_squared_error(y_test, pred_test, squared=False)
RMSE_test = MSE_test**0.5
MAE_test = mean_absolute_error(y_test, pred_test)
ME_test = np.mean(y_test-pred_test)

INDEX_TRAIN3 = pd.DataFrame([['GradientBoostingRegressor',R2_train,RMSE_train,MSE_train,MAE_train,ME_train]],columns=['Model','R2','RMSE','MSE','MAE','ME'])
INDEX_TEST3 = pd.DataFrame([['GradientBoostingRegressor',R2_test,RMSE_test,MSE_test,MAE_test,ME_test]],columns=['Model','R2','RMSE','MSE','MAE','ME'])
#
INDEX_TRAIN= INDEX_TRAIN.append(INDEX_TRAIN3)
INDEX_TEST= INDEX_TEST.append(INDEX_TEST3)
print(INDEX_TRAIN)
print(INDEX_TEST)

#%%4 RandomForestRegressor
paras = {
         'n_estimators': 100,
         'max_features': 1,
         'min_samples_split': 2,
         'min_samples_leaf': 1,
         'random_state': 2024}

model4 = RandomForestRegressor(**paras)
model4.fit(X_train, y_train)


pred_train = model4.predict(X_train)
R2_train = r2_score(y_train, pred_train)
MSE_train = mean_squared_error(y_train, pred_train, squared=False)
RMSE_train = MSE_train**0.5
MAE_train = mean_absolute_error(y_train, pred_train)
# ME_train = np.mean(y_train- pred_train)

pred_test = model4.predict(X_test)
R2_test = r2_score(y_test, pred_test)
MSE_test = mean_squared_error(y_test, pred_test, squared=False)
RMSE_test = MSE_test**0.5
MAE_test = mean_absolute_error(y_test, pred_test)
# ME_test = np.mean(y_test-pred_test)

INDEX_TRAIN4 = pd.DataFrame([['RandomForestRegressor',R2_train,RMSE_train,MSE_train,MAE_train]],columns=['Model','R2','RMSE','MSE','MAE'])
INDEX_TEST4 = pd.DataFrame([['RandomForestRegressor',R2_test,RMSE_test,MSE_test,MAE_test]],columns=['Model','R2','RMSE','MSE','MAE'])
#
INDEX_TRAIN= INDEX_TRAIN.append(INDEX_TRAIN4)
INDEX_TEST= INDEX_TEST.append(INDEX_TEST4)
print(INDEX_TRAIN)
print(INDEX_TEST)
#%%
#将训练的模型保存到磁盘(value=模型名)   默认当前文件夹下
joblib.dump(filename =r"model/ht_rf.model", value=model4)
#%%5 LGBMRegressor
paras = {'reg_alpha': 7.05795093552413,
 'reg_lambda': 0.2613104874833126,
 'num_leaves': 246,
 'min_child_samples': 5,
 'max_depth': 13,
 'learning_rate': 0.1,
 'colsample_bytree': 0.4766864350277183,
 'n_estimators': 7213,
 'cat_smooth': 76,
 'cat_l2': 4,
 'min_data_per_group': 79,
 'cat_feature': 47,
         'random_state': 2024}


model5 = LGBMRegressor(**paras)
model5.fit(X_train, y_train)
#%%
pred_train = model5.predict(X_train)
R2_train = r2_score(y_train, pred_train)
MSE_train = mean_squared_error(y_train, pred_train, squared=False)
RMSE_train = MSE_train**0.5
MAE_train = mean_absolute_error(y_train, pred_train)
# ME_train = np.mean(y_train- pred_train)

pred_test = model5.predict(X_test)
R2_test = r2_score(y_test, pred_test)
MSE_test = mean_squared_error(y_test, pred_test, squared=False)
RMSE_test = MSE_test**0.5
MAE_test = mean_absolute_error(y_test, pred_test)
# ME_test = np.mean(y_test-pred_test)
#%%
INDEX_TRAIN5 = pd.DataFrame([['LGBMRegressor',R2_train,RMSE_train,MSE_train,MAE_train]],columns=['Model','R2','RMSE','MSE','MAE'])
INDEX_TEST5 = pd.DataFrame([['LGBMRegressor',R2_test,RMSE_test,MSE_test,MAE_test]],columns=['Model','R2','RMSE','MSE','MAE'])
#%%
INDEX_TRAIN= INDEX_TRAIN.append(INDEX_TRAIN5)
INDEX_TEST= INDEX_TEST.append(INDEX_TEST5)
print(INDEX_TRAIN)
print(INDEX_TEST)
#%%
#将训练的模型保存到磁盘(value=模型名)   默认当前文件夹下
joblib.dump(filename = r"./model/ht_lightgbm.model",value=model5)

#%%6 XGBRegressor
paras = {'max_depth': 10,
 'learning_rate': 0.01,
 'n_estimators': 5415,
 'min_child_weight': 1,
 'gamma': 0.3746778226969783,
 'alpha': 0.5333712995663684,
 'lambda': 0.0008231094823690853,
 'colsample_bytree': 0.5556409947371114,
 'subsample': 0.770256729572532,
         'random_state': 2024}

model6 = XGBRegressor(**paras)
model6.fit(X_train, y_train)
#%%
#将训练的模型保存到磁盘(value=模型名)   默认当前文件夹下
joblib.dump(filename =r"model/h1_xgboost.model", value=model6)
#%%
pred_train = model6.predict(X_train)
R2_train = r2_score(y_train, pred_train)
MSE_train = mean_squared_error(y_train, pred_train, squared=False)
RMSE_train = MSE_train**0.5
MAE_train = mean_absolute_error(y_train, pred_train)
# ME_train = np.mean(y_train- pred_train)
#%
pred_test = model6.predict(X_test)
R2_test = r2_score(y_test, pred_test)
MSE_test = mean_squared_error(y_test, pred_test, squared=False)
RMSE_test = MSE_test**0.5
MAE_test = mean_absolute_error(y_test, pred_test)
# ME_test = np.mean(y_test-pred_test)

INDEX_TRAIN6 = pd.DataFrame([['XGBRegressor',R2_train,RMSE_train,MSE_train,MAE_train]],columns=['Model','R2','RMSE','MSE','MAE'])
INDEX_TEST6 = pd.DataFrame([['XGBRegressor',R2_test,RMSE_test,MSE_test,MAE_test]],columns=['Model','R2','RMSE','MSE','MAE'])
#
INDEX_TRAIN= INDEX_TRAIN.append(INDEX_TRAIN6)
INDEX_TEST= INDEX_TEST.append(INDEX_TEST6)
print(INDEX_TRAIN)
print(INDEX_TEST)



#%%zone7 AdaBoostRegressor
paras = {'n_estimators': 239,
         'learning_rate': 0.06425924663287144,
         'loss': 'linear',
         'base_estimator': DecisionTreeRegressor(max_depth=5),
         'random_state': 2024}

model7 = AdaBoostRegressor(**paras)
model7.fit(X_train, y_train)


pred_train = model7.predict(X_train)
R2_train = r2_score(y_train, pred_train)
MSE_train = mean_squared_error(y_train, pred_train, squared=False)
RMSE_train = MSE_train**0.5
MAE_train = mean_absolute_error(y_train, pred_train)
ME_train = np.mean(y_train- pred_train)

pred_test = model7.predict(X_test)
R2_test = r2_score(y_test, pred_test)
MSE_test = mean_squared_error(y_test, pred_test, squared=False)
RMSE_test = MSE_test**0.5
MAE_test = mean_absolute_error(y_test, pred_test)
ME_test = np.mean(y_test-pred_test)

INDEX_TRAIN7 = pd.DataFrame([['AdaBoostRegressor',R2_train,RMSE_train,MSE_train,MAE_train,ME_train]],columns=['Model','R2','RMSE','MSE','MAE','ME'])
INDEX_TEST7 = pd.DataFrame([['AdaBoostRegressor',R2_test,RMSE_test,MSE_test,MAE_test,ME_test]],columns=['Model','R2','RMSE','MSE','MAE','ME'])

INDEX_TRAIN= INDEX_TRAIN.append(INDEX_TRAIN7)
INDEX_TEST= INDEX_TEST.append(INDEX_TEST7)
print(INDEX_TRAIN)
print(INDEX_TEST)

#%%zone8 CascadeForestRegressor
paras = {'n_estimators': 239,
         'learning_rate': 0.06425924663287144,
         'loss': 'linear',
         'base_estimator': DecisionTreeRegressor(max_depth=5),
         'random_state': 2024}
model8 = CascadeForestRegressor(**paras)
model8.fit(X_train, y_train)


pred_train = model8.predict(X_train)
R2_train = r2_score(y_train, pred_train)
MSE_train = mean_squared_error(y_train, pred_train, squared=False)
RMSE_train = MSE_train**0.5
MAE_train = mean_absolute_error(y_train, pred_train)
ME_train = np.mean(y_train- pred_train)

pred_test = model8.predict(X_test)
R2_test = r2_score(y_test, pred_test)
MSE_test = mean_squared_error(y_test, pred_test, squared=False)
RMSE_test = MSE_test**0.5
MAE_test = mean_absolute_error(y_test, pred_test)
ME_test = np.mean(y_test-pred_test)

INDEX_TRAIN8 = pd.DataFrame([['CascadeForestRegressor',R2_train,RMSE_train,MSE_train,MAE_train,ME_train]],columns=['Model','R2','RMSE','MSE','MAE','ME'])
INDEX_TEST8 = pd.DataFrame([['CascadeForestRegressor',R2_test,RMSE_test,MSE_test,MAE_test,ME_test]],columns=['Model','R2','RMSE','MSE','MAE','ME'])

INDEX_TRAIN= INDEX_TRAIN.append(INDEX_TRAIN8)
INDEX_TEST= INDEX_TEST.append(INDEX_TEST8)
print(INDEX_TRAIN)
print(INDEX_TEST)

#%%##############################################基于最优结果计算其固定效应预测值##############################################
data2A = df1[['HT','REGION']]
trainX1 = data2A[['REGION']].values
trainY1 = data2A[['HT']].values

from sklearn.model_selection import train_test_split, KFold
X_train1, X_test1, y_train1, y_test1 = train_test_split(trainX1, trainY1, train_size=2/3, random_state=2024)  # 数据集划分
#%% X_train
loaded_model = joblib.load(filename="model/ht_xgboost.model")#加载模型
preH3 = loaded_model.predict(X_train)# 使用模型对测试数据进行预测

# 将ndarray转换为DataFrame
data2A = pd.DataFrame({'REGION': X_train1.flatten(), 'HT': y_train1.flatten(), 'fixedHT': preH3})

data2A.to_csv("./DATA/R_Train30km_HT.csv", index=False,encoding='utf-8-sig')

########################################################################################################################
####################################################基于lem4py包，实现混合效应模型构建########################################
#%%#####################################################################################################################
'''
此部分实现用R语言进行运行，得出固定效应的参数值以及随机效应参数（按植被区对应），进行下一下预测。各参数值如下所示：
固定效应截距项：fixed_Intercept         (1个值）
固定效应preH1对应斜率参数：fixed_preH1   (1个值）
随机效应截距项：random_Intercept        (8个值，每个植被区对应一个值）
'''

########################################################################################################################
###################################################测试集进行检验########################################
#%%#####################################################################################################################
loaded_model = joblib.load(filename="model/ht_xgboost.model")#加载模型
preHT = loaded_model.predict(X_test)# 使用模型对测试数据进行预测
#%%取test数据集X_test2
RandomHT = pd.read_csv('./DATA/RandomHT.csv', sep=',')
HTdat = pd.merge(data1, RandomHT, on='REGION', how='left')
#%%
trainX2 = HTdat[['REGION','InterceptHT','fixedHT']].values
trainY2 = HTdat[['HT']].values
X_train2, X_test2, y_train2, y_test2 = train_test_split(trainX2, trainY2, train_size=2/3, random_state=2024)  # 数据集划分
#%%
data2B = pd.DataFrame({'REGION': X_test2[:,0],'HT': y_test2.flatten(), 'InterceptHT': X_test2[:,1], 'fixedHT': X_test2[:,2],'preHT': preH3})
data2B['predict_HT'] = data2B['fixedHT']*data2B['preHT'] + data2B['InterceptHT']
#%%
pred_test = data2B[['predict_HT']].values
R2_test = r2_score(y_test2, pred_test)
MSE_test = mean_squared_error(y_test2, pred_test, squared=False)
RMSE_test = MSE_test**0.5
MAE_test = mean_absolute_error(y_test2, pred_test)
#%%
print(R2_test,MSE_test,RMSE_test,MAE_test)