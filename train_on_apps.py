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

def compare_effect_apps():
    params = {'seed':rs, 
        'n_estimators':100,
        'eta':0.1,
        'max-depth':6,
        'subsample':0.8, 
        'colsample_bytree':0.7,
        'num_class': 12}  
    
    cr = CaseRunner('compare_app_cat_feature_types', dir_out)
    
    #
    fea_files = ['feat_apps_sum_installed.csv','feat_apps_rel_installed.csv','feat_apps_any_installed.csv', \
                 'feat_apps_sum_active.csv','feat_apps_rel_active.csv','feat_apps_any_active.csv']
    
    for fea_file in fea_files:
        feat = pd.read_csv('./data/' + fea_file, dtype={'device_id': np.str})
        df = pd.merge(train, feat, how='left', on='device_id')
        df.fillna(-1, inplace=True)
        
#        print fea_file        
#        print df.shape
#        print df.columns
#        print df.drop(['device_id','age','group','gender'],axis=1).head()
#        print ''
        cr.add_case(fea_file,df.drop(['device_id','age','group','gender'],axis=1),y,"xgb",params, testsize=0.1, random_state=rs)
    
    cr.run_cases()
    
def compare_effect_installed_and_active():
    params = {'seed':rs, 
        'n_estimators':1000,
        'eta':0.01,
        'max-depth':6,
        'subsample':0.8, 
        'colsample_bytree':0.7,
        'num_class': 12}  
    
    cr = CaseRunner('compare_app_cat_feature_types', dir_out)
    
    #
    fea_files = ['feat_apps_any_installed.csv', 'feat_apps_any_active.csv']
    df = train
    for fea_file in fea_files:
        feat = pd.read_csv('./data/' + fea_file, dtype={'device_id': np.str})
        df = pd.merge(df, feat, how='left', on='device_id')
        df.fillna(-1, inplace=True)
        
        cr.add_case(fea_file,df.drop(['device_id','age','group','gender'],axis=1),y,"xgb",params, testsize=0.1, random_state=rs)
    
    cr.run_cases()



if __name__ == "__main__":
    
    dir_in = './data/'
    dir_out = './report_apps/'
    rs = 123
    
    ylim_logloss = [2.2,2.6]
    
   #hdlist = ['device_id', 'F23-', 'F24-26', 'F27-28', 'F29-32', 'F33-42','F43+', 
    #                       'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+']        
    
    data_tr = pd.read_csv(dir_in + 'gender_age_train.csv', dtype={'device_id': np.str})

    print('Read brands...')
    pbd = pd.read_csv(dir_in + 'phone_brand_device_model.csv', dtype={'device_id': np.str})
    pbd.drop_duplicates('device_id', keep='first', inplace=True)
    pbd = map_column(pbd, 'phone_brand')
    pbd = map_column(pbd, 'device_model')
   
    print('Read train...')
    train = pd.read_csv(dir_in + 'gender_age_train.csv', dtype={'device_id': np.str})
    train = map_column(train, 'gender')
    train = map_column(train, 'group')
    train = pd.merge(train, pbd, how='left', on='device_id', left_index=True)
    train.fillna(-1, inplace=True)
    
    print('Read test...')
    test = pd.read_csv(dir_in + 'gender_age_test.csv', dtype={'device_id': np.str})
    test = pd.merge(test, pbd, how='left', on='device_id', left_index=True)
    test.fillna(-1, inplace=True)
    
    X = train[['phone_brand','device_model']]
    y = train['group']
    
    #X_test = test[['phone_brand','device_model']]
    
    compare_effect_installed_and_active()
    
    
    

