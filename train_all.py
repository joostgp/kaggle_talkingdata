# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 16:19:02 2016

@author: joostbloom
"""
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split

from ml_toolbox.runner import CaseRunner


import xgboost as xgb 
   
# --------------------
# - Support functions 
# --------------------
    
def map_column(table, f):
    labels = sorted(table[f].unique())
    mappings = dict()
    for i in range(len(labels)):
        mappings[labels[i]] = i
    table = table.replace({f: mappings})
    return table

# --------------------
# - Cases
# --------------------

def compare_effect_feat_group():
    params = {'seed':rs, 
        'n_estimators':400,
        'eta':0.05,
        'max-depth':6,
        'subsample':0.8, 
        'colsample_bytree':0.7,
        'num_class': 12}  
    
    cr = CaseRunner('compare_features_groups', outputdir)
    
    cr.submit_to_kaggle = True
    
    #feat = ['phone_brand']
    #cr.add_case('phone_brand',train[feat],y,"xgb",params, testsize=0.1, random_state=rs, X_test=test[feat], ids_test=test['device_id'])
    feat = ['phone_brand','device_model','close_to_city'] + dist_cols + close_cols + event_col + app_cat_col
    cr.add_case('all',train[feat],y,"xgb",params, testsize=0.1, random_state=rs, X_test=test[feat], ids_test=test['device_id']) 
    feat = ['phone_brand','device_model','close_to_city'] + dist_cols + close_cols + app_cat_col 
    cr.add_case('-a',train[feat],y,"xgb",params, testsize=0.1, random_state=rs, X_test=test[feat], ids_test=test['device_id']) 
    feat = ['phone_brand','device_model','close_to_city'] + app_cat_col + close_cols 
    cr.add_case('-ae',train[feat],y,"xgb",params, testsize=0.1, random_state=rs, X_test=test[feat], ids_test=test['device_id']) 
    feat = ['phone_brand','device_model','close_to_city'] + app_cat_col 
    cr.add_case('-aec',train[feat],y,"xgb",params, testsize=0.1, random_state=rs, X_test=test[feat], ids_test=test['device_id']) 
    feat = ['phone_brand','device_model','close_to_city']
    cr.add_case('-aecd',train[feat],y,"xgb",params, testsize=0.1, random_state=rs, X_test=test[feat], ids_test=test['device_id']) 
    
    cr.run_cases()
    
def compare_gbtree_gblinear():
    params = {'seed':rs, 
        'n_estimators':400,
        'eta':0.05,
        'max-depth':6,
        'subsample':0.8, 
        'colsample_bytree':0.7,
        'num_class': 12}  
    
    cr = CaseRunner('compare_gblinear_gbtree', outputdir)
    
    cr.submit_to_kaggle = False
    
    #feat = ['phone_brand']
    #cr.add_case('phone_brand',train[feat],y,"xgb",params, testsize=0.1, random_state=rs, X_test=test[feat], ids_test=test['device_id'])
    feat = ['phone_brand','device_model','close_to_city'] + dist_cols + close_cols + event_col + app_cat_col
    
    cr.add_case('gbtree',train[feat],y,"xgb",params, testsize=0.1, random_state=rs, X_test=test[feat], ids_test=test['device_id']) 
    params2 = {'seed':rs, 
        'n_estimators':400,
        'eta':0.05,
        'max-depth':6,
        'subsample':0.8, 
        'colsample_bytree':0.7,
        'booster': 'gblinear',
        'num_class': 12}  
    cr.add_case('gblinear',train[feat],y,"xgb",params2, testsize=0.1, random_state=rs, X_test=test[feat], ids_test=test['device_id']) 
    
    cr.run_cases()



if __name__ == "__main__":
    
    inputdir = './data/'
    outputdir = './report_all/'
    rs = 123
    
    ylim_logloss = [2.2,2.6]
    
    hdlist = ['device_id', 'F23-', 'F24-26', 'F27-28', 'F29-32', 'F33-42','F43+', 
                           'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+']        
    
    data_tr = pd.read_csv(inputdir + 'gender_age_train.csv', dtype={'device_id': np.str})

    print('Read brand features...')
    pbd = pd.read_csv(inputdir + 'phone_brand_device_model.csv', dtype={'device_id': np.str})
    pbd.drop_duplicates('device_id', keep='first', inplace=True)
    pbd = map_column(pbd, 'phone_brand')
    pbd = map_column(pbd, 'device_model')
    
    print('Read geo features..')
    feat_geo = pd.read_csv(inputdir + 'features_geo.csv', dtype={'device_id': np.str})
    dist_cols = list(feat_geo.filter(regex=('dist_.*')).columns.values)
    close_cols = list(feat_geo.filter(regex=('close_to.*')).columns.values)
   
    print('Read app cat features...')
    features_app_cat = pd.read_csv(inputdir + 'device_app_cats.csv', dtype={'device_id': np.str})
    app_cat_col = list(features_app_cat.columns.drop(['device_id','app counts']).values)
    
    print('Read event features...')
    features_event = pd.read_csv(inputdir + 'features_event.csv', dtype={'device_id': np.str})
    event_col = list(features_event.columns.drop(['device_id']).values)
   
    print('Read train...')
    train = pd.read_csv(inputdir + 'gender_age_train.csv', dtype={'device_id': np.str})
    train = map_column(train, 'gender')
    train = map_column(train, 'group')
    train = pd.merge(train, pbd, how='left', on='device_id', left_index=True)
    train = pd.merge(train, feat_geo, how='left', on='device_id')
    train = pd.merge(train, features_app_cat, how='left', on='device_id')
    train = pd.merge(train, features_event, how='left', on='device_id')
    train.fillna(-1, inplace=True)
    
    print('Read test...')
    test = pd.read_csv(inputdir + 'gender_age_test.csv', dtype={'device_id': np.str})
    test = pd.merge(test, pbd, how='left', on='device_id', left_index=True)
    test = pd.merge(test, feat_geo, how='left', on='device_id', left_index=True)
    test = pd.merge(test, features_app_cat, how='left', on='device_id', left_index=True)
    test = pd.merge(test, features_event, how='left', on='device_id', left_index=True)
    test.fillna(-1, inplace=True)
    
    y = train['group']
    
    #X_test = test[['phone_brand','device_model']]
    
    
    
    train['close_to_city'] = train[close_cols].sum(axis=1)
    test['close_to_city'] = test[close_cols].sum(axis=1)
    
    # Test for columns without data
    (train==-1).sum().sort_values()/train.shape[0]
    (test==-1).sum().sort_values()/test.shape[0]
    compare_gbtree_gblinear()    
    #compare_effect_feat_group()
    #compare_effect_min_child_weight()
    #compare_effect_geo()
    # Hardly any difference (freq encoding slightly better)
    #compare_phone_brand_encoding()
    
    #compare_gbtree_linear()
    
    

