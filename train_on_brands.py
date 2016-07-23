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

from ml_toolbox.kaggle import KaggleResult
from ml_toolbox.plot import newfigure
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

def compare_phone_brand_encoding(): 
    
    params = {'seed':rs, 'n_estimators':60,'eta':0.1,'max-depth':3,'subsample':0.7, 'colsample_bytree':0.7, "eval_metric": "mlogloss","objective": "multi:softprob", "num_class": 12}  
    
    cases=[]

    # Base case    
    X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.3, random_state=rs, stratify=y)
    cases.append( ("base",  X_train, y_train, X_val, y_val, params, "xgb") )
    
    # One hot
    X_one_hot = pd.get_dummies(X['phone_brand'])
    X_one_hot['device_model'] = X['device_model']
    
    X_test_one_hot = pd.get_dummies(X_test['phone_brand'])
    X_test_one_hot['device_model'] = X_test['device_model']
    
    X_train, X_val, y_train, y_val = train_test_split( X_one_hot, y, test_size=0.3, random_state=rs, stratify=y)
    cases.append( ("onehot", X_train, y_train, X_val, y_val, params, "xgb") )
    
    # Freq encoding
    pb_freqs = X['phone_brand'].value_counts()
    
    # Phone brands not in train dataset
    pb_freqs[0]=0
    pb_freqs[17]=0
    pb_freqs[37]=0
    pb_freqs[47]=0
    pb_freqs[53]=0
    pb_freqs[56]=0
    pb_freqs[70]=0
    pb_freqs[79]=0
    pb_freqs[86]=0
    pb_freqs[90]=0
    pb_freqs[113]=0
    X_freq = X[['device_model']]
    X_freq['pb_freq'] = [pb_freqs[x] for x in X['phone_brand'].values]
    
    X_freq_test = X_test[['device_model']]
    X_freq_test['pb_freq'] = [pb_freqs[x] for x in X_test['phone_brand'].values]
    
    X_train, X_val, y_train, y_val = train_test_split( X_freq, y, test_size=0.3, random_state=rs, stratify=y)
    cases.append( ("freq-enco", X_train, y_train, X_val, y_val, params, "xgb") )
    
    # Freq one-hot encoded
    X_freq_one_hot = pd.get_dummies(X_freq['pb_freq'])
    X_freq_one_hot['device_model'] = X_test['device_model']
    
    X_test_freq_one_hot = pd.get_dummies(X_freq_test['pb_freq'])
    X_test_freq_one_hot['device_model'] = X['device_model']
    X_train, X_val, y_train, y_val = train_test_split( X_freq, y, test_size=0.3, random_state=rs, stratify=y)
    cases.append( ("onehot freq enco", X_train, y_train, X_val, y_val, params, "xgb") )
    
    #run_cases(cases,"Comparison phone brand model encoding") 

def compare_gbtree_linear():
    params = {'seed':rs, 
        'n_estimators':60,
        'eta':0.1,
        'max-depth':3,
        'subsample':0.7, 
        'colsample_bytree':0.7,
        'num_class': 12}  
    
    params = {'seed':rs, 'n_estimators':5,'eta':0.1,'max-depth':3,'subsample':0.7, 'colsample_bytree':0.7, "eval_metric": "mlogloss","objective": "multi:softprob", "num_class": 12}  
    
    cr = CaseRunner('tester', outputdir)
    
    #cr.add_case()
    cr.add_case('test1',X,y,"xgb",params, testsize=0.1, random_state=rs, X_test=X_test, ids_test=x_test)  
    cr.add_case('test2',X,y,"xgb",params, testsize=0.2, random_state=rs, X_test=X_test, ids_test=x_test)  
    cr.add_case('test3',X,y,"xgb",params, testsize=0.3, random_state=rs, X_test=X_test, ids_test=x_test)
    
    cr.run_cases()


if __name__ == "__main__":
    
    inputdir = './data_ori/'
    outputdir = './report_brands/'
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
   
    print('Read train...')
    train = pd.read_csv(inputdir + 'gender_age_train.csv', dtype={'device_id': np.str})
    train = map_column(train, 'gender')
    train = map_column(train, 'group')
    train = pd.merge(train, pbd, how='left', on='device_id', left_index=True)
    train.fillna(-1, inplace=True)
    
    print('Read test...')
    test = pd.read_csv(inputdir + 'gender_age_test.csv', dtype={'device_id': np.str})
    test = pd.merge(test, pbd, how='left', on='device_id', left_index=True)
    test.fillna(-1, inplace=True)
    
    X = train[['phone_brand','device_model']]
    y = train['group']
    
    X_test = test[['phone_brand','device_model']]
    x_test = test['device_id']
    
    
    # Hardly any difference (freq encoding slightly better)
    #compare_phone_brand_encoding()
    
    compare_gbtree_linear()
    
    

