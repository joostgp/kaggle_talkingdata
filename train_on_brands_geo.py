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

def compare_effect_geo():
    params = {'seed':rs, 
        'n_estimators':500,
        'eta':0.01,
        'max-depth':6,
        'subsample':0.8, 
        'colsample_bytree':0.7,
        'num_class': 12}  
    
    cr = CaseRunner('compare_features', outputdir)
    
    cr.submit_to_kaggle = True
    
    #feat = ['phone_brand']
    #cr.add_case('phone_brand',train[feat],y,"xgb",params, testsize=0.1, random_state=rs, X_test=test[feat], ids_test=test['device_id'])
    feat = ['phone_brand','device_model'] 
    cr.add_case('brand',train[feat],y,"xgb",params, testsize=0.1, random_state=rs) 
    feat = ['phone_brand','device_model','travel_dist']     
    cr.add_case('brand + travel',train[feat],y,"xgb",params, testsize=0.1, random_state=rs, X_test=test[feat], ids_test=test['device_id']) 
    feat = ['phone_brand','device_model','close_to_city']     
    cr.add_case('brand + city',train[feat],y,"xgb",params, testsize=0.1, random_state=rs, X_test=test[feat], ids_test=test['device_id']) 
    feat = ['phone_brand','device_model']  + close_cols
    cr.add_case('brand + cities',train[feat],y,"xgb",params, testsize=0.1, random_state=rs, X_test=test[feat], ids_test=test['device_id']) 
    feat = ['phone_brand','device_model']  + dist_cols
    cr.add_case('brand + distance',train[feat],y,"xgb",params, testsize=0.1, random_state=rs, X_test=test[feat], ids_test=test['device_id']) 
    
    cr.run_cases()



if __name__ == "__main__":
    
    inputdir = './data/'
    outputdir = './report_geo/'
    rs = 123
    
    upload_to_kaggle = False
    
    ylim_logloss = [2.2,2.6]
    
    hdlist = ['device_id', 'F23-', 'F24-26', 'F27-28', 'F29-32', 'F33-42','F43+', 
                           'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+']        
    
    data_tr = pd.read_csv(inputdir + 'gender_age_train.csv', dtype={'device_id': np.str})

    print('Read brands...')
    pbd = pd.read_csv(inputdir + 'phone_brand_device_model.csv', dtype={'device_id': np.str})
    pbd.drop_duplicates('device_id', keep='first', inplace=True)
    pbd = map_column(pbd, 'phone_brand')
    pbd = map_column(pbd, 'device_model')
    
    print('Rad geo..')
    feat_geo = pd.read_csv(inputdir + 'features_geo.csv', dtype={'device_id': np.str})
   
    print('Read train...')
    train = pd.read_csv(inputdir + 'gender_age_train.csv', dtype={'device_id': np.str})
    train = map_column(train, 'gender')
    train = map_column(train, 'group')
    train = pd.merge(train, pbd, how='left', on='device_id', left_index=True)
    train = pd.merge(train, feat_geo, how='left', on='device_id')
    train.fillna(-1, inplace=True)
    
    print('Read test...')
    test = pd.read_csv(inputdir + 'gender_age_test.csv', dtype={'device_id': np.str})
    test = pd.merge(test, pbd, how='left', on='device_id', left_index=True)
    test = pd.merge(test, feat_geo, how='left', on='device_id', left_index=True)
    test.fillna(-1, inplace=True)
    
    X = train[['phone_brand','device_model']]
    y = train['group']
    
    #X_test = test[['phone_brand','device_model']]
    
    dist_cols = list(train.filter(regex=('dist_.*')).columns.values)
    close_cols = list(train.filter(regex=('close_to.*')).columns.values)
    
    
    train['close_to_city'] = train[close_cols].sum(axis=1)
    test['close_to_city'] = test[close_cols].sum(axis=1)
    
    
    compare_effect_geo()
    # Hardly any difference (freq encoding slightly better)
    #compare_phone_brand_encoding()
    
    #compare_gbtree_linear()
    
    

