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


import xgboost as xgb 

# --------------------
# - Basic methods required to run cases 
# --------------------

def run_case(case):
    name = case[0]
    X_train = case[1]
    y_train = case[2]
    X_val = case[3]
    y_val = case[4]
    clfparams = case[5]
    classifier = case[6]
    
    print("Running case: %s" % name)
    
    if classifier=="xgb":
        return get_xgboost_classifier(X_train, y_train, X_val, y_val, clfparams, output_eval=True)
    
def run_cases(cases, groupName):
    
    print("Running %s" % groupName)    
    print("")
    
    names = [x[0] for x in cases]
    
    scores = []
    clfs = []
    times_t = []
    times_p = []
    
    for c in cases:
        
        classifier = c[6]
        casename = c[0]
        
        X_val = c[3]
        y_val = c[4]
        
        s = time.time()      
        
        # Train model
        (clf,df_eval) = run_case(c)
        clfs.append( clf )
        times_t.append( time.time()-s )
        
        plot_cv_curves(df_eval, groupName + " - " + casename, outputdir)
        
        # Calculate score
        s = time.time()  
        if classifier=="xgb":
            score = report_result(clf, X_val, y_val)
            scores.append( score )
        else:
            scores.append( report_result(clf, X_val, y_val) )
        times_p.append( time.time()-s )
        
        # Create submissions file if X_test is provided as 7th element
        if len(c)>7:
            X_test = c[8]
            test_device_ids = c[7]
            create_submission_file(clf, test_device_ids, X_test, score, groupName + " - " + casename)
        
    # Approximate memory usage by pickle dump
        
    width = 0.5    
    
    newfigure(groupName)
    plt.bar(np.arange(len(cases)),scores, width)
    plt.xticks(np.arange(len(cases))+width/2, names, rotation='vertical')
    plt.ylabel("mllogloss Score")
    plt.ylim(ylim_logloss)
    plt.xlim([-width,len(cases)])
    plt.tight_layout()
    plt.grid()
    
    ax = plt.gca()
    rects = ax.patches
    for rect, label in zip(rects, scores):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height, "{:.4f}".format(label), ha='center', va='bottom')
        
    plt.savefig(outputdir + '%s_logloss.png' % groupName)

def report_result(clf, X_test, y_true):
    if str(type(clf)) == "<class 'xgboost.core.Booster'>":
        y_pred = clf.predict(xgb.DMatrix(X_test))  
    
    return log_loss(y_true, y_pred)
    
def plot_cv_curves(df, identifier, path):
    newfigure(str(identifier))
    plt.plot(df['Eval set'],'g',label='Validation set')
    plt.plot(df['Train set'],'r',label='Train set')
    plt.grid()
    plt.xlabel('Boosting round')
    plt.ylabel('Logloss Score')
    plt.legend()  
    plt.savefig(path + 'eval_curves_' + str(identifier) + '.png')

def get_xgboost_classifier(X_train, y_train, X_val, y_val, params, rs=123, output_eval=False):    
    
    if "objective" not in params: params["objective"] = "multi:softprob"
    if "booster" not in params: params["booster"] = "gbtree"
    if "eval_metric" not in params: params["eval_metric"] =  "mlogloss"
    
    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_val, y_val)
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    evals_result = {}
    gbm = xgb.train(params, dtrain, params['n_estimators'], evals=watchlist, early_stopping_rounds=20, verbose_eval=True, evals_result=evals_result)
    
    
    if output_eval:
        test_scores = evals_result['eval'][params["eval_metric"]]
        train_scores = evals_result['train'][params["eval_metric"]]
        
        df = pd.DataFrame()
        
        df['Eval set'] = test_scores
        df['Train set'] = train_scores
        
        return gbm, df
    else:
        return gbm

def create_submission_file(clf,test_device_ids, X_test, score, description):
    
    y_test = clf.predict(xgb.DMatrix(X_test))
    
    kag = KaggleResult(test_device_ids, prediction=y_test, score=score, description=description, sub_path=outputdir)
    
    if kag.validate() and upload_to_kaggle:
        try:
            kag.upload(description)
        except:
            pass
   
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


def compare_gbtree_linear():
    params = {'seed':rs, 
        'n_estimators':60,
        'eta':0.1,
        'max-depth':3,
        'subsample':0.7, 
        'colsample_bytree':0.7,
        'num_class': 12}  
    
    params = {'seed':rs, 'n_estimators':10,'eta':0.1,'max-depth':3,'subsample':0.7, 'colsample_bytree':0.7, "eval_metric": "mlogloss","objective": "multi:softprob", "num_class": 12}  
       
    cases=[]

    # Base case    
    X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.2, random_state=rs, stratify=y)
    cases.append( ("base",  X_train, y_train, X_val, y_val, params, "xgb") )
    
    params2 = params.copy()
    params2['booster'] = 'gblinear'
    cases.append( ('base',  X_train, y_train, X_val, y_val, params2, 'xgb') )
    
    run_cases(cases,'Comparison gbtree gblinear') 
    
def compare_test_sizes():
    params = {'seed':rs, 
        'n_estimators':150,
        'eta':0.1,
        'max_depth':5,
        'subsample':0.7, 
        'colsample_bytree':0.7,
        'num_class': 12}  
           
    cases=[]

    # Base case    
    X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.1, random_state=rs, stratify=y)
    cases.append( ('0.1',  X_train, y_train, X_val, y_val, params, "xgb") )
    
    X_train2, X_val2, y_train2, y_val2 = train_test_split( X, y, test_size=0.2, random_state=rs, stratify=y)
    cases.append( ('0.2',  X_train2, y_train2, X_val2, y_val2, params, 'xgb') )
    
    X_train3, X_val3, y_train3, y_val3 = train_test_split( X, y, test_size=0.3, random_state=rs, stratify=y)
    cases.append( ('0.3',  X_train3, y_train3, X_val3, y_val3, params, 'xgb') )
    
    X_train4, X_val4, y_train4, y_val4 = train_test_split( X, y, test_size=0.4, random_state=rs, stratify=y)
    cases.append( ('0.4',  X_train4, y_train4, X_val4, y_val4, params, 'xgb') )
    
    X_train5, X_val5, y_train5, y_val5 = train_test_split( X, y, test_size=0.5, random_state=rs, stratify=y)
    cases.append( ('0.5',  X_train5, y_train5, X_val5, y_val5, params, 'xgb') )
    
    run_cases(cases,'Comparison test sizes') 
    
def compare_max_depth():
           
    cases=[]

    # Base case    
    X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.3, random_state=rs, stratify=y)
    cases.append( ('2',  X_train, y_train, X_val, y_val, {'seed':rs, 'n_estimators':150,'eta':0.1,'max_depth':2,'subsample':0.7, 'colsample_bytree':0.7, "eval_metric": "mlogloss","objective": "multi:softprob", "num_class": 12}, "xgb", test_device_ids, X_test) )
    cases.append( ('3',  X_train, y_train, X_val, y_val, {'seed':rs, 'n_estimators':150,'eta':0.1,'max_depth':3,'subsample':0.7, 'colsample_bytree':0.7, "eval_metric": "mlogloss","objective": "multi:softprob", "num_class": 12}, 'xgb', test_device_ids, X_test) )
    cases.append( ('4',  X_train, y_train, X_val, y_val, {'seed':rs, 'n_estimators':150,'eta':0.1,'max_depth':4,'subsample':0.7, 'colsample_bytree':0.7, "eval_metric": "mlogloss","objective": "multi:softprob", "num_class": 12}, 'xgb', test_device_ids, X_test) )
    cases.append( ('5',  X_train, y_train, X_val, y_val, {'seed':rs, 'n_estimators':150,'eta':0.1,'max_depth':5,'subsample':0.7, 'colsample_bytree':0.7, "eval_metric": "mlogloss","objective": "multi:softprob", "num_class": 12}, 'xgb', test_device_ids, X_test) )
    cases.append( ('6',  X_train, y_train, X_val, y_val, {'seed':rs, 'n_estimators':150,'eta':0.1,'max_depth':6,'subsample':0.7, 'colsample_bytree':0.7, "eval_metric": "mlogloss","objective": "multi:softprob", "num_class": 12}, 'xgb', test_device_ids, X_test) )
    
    run_cases(cases,'Comparison test sizes') 
    

def compare_max_depth_large():
           
    cases=[]

    # Base case    
    X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.3, random_state=rs, stratify=y)
    cases.append( ('max-depth: 6',  X_train, y_train, X_val, y_val, {'seed':rs, 'n_estimators':150,'eta':0.1,'max_depth':6,'subsample':0.7, 'colsample_bytree':0.7, "eval_metric": "mlogloss","objective": "multi:softprob", "num_class": 12}, "xgb", test_device_ids, X_test) )
    cases.append( ('max-depth: 8',  X_train, y_train, X_val, y_val, {'seed':rs, 'n_estimators':150,'eta':0.1,'max_depth':8,'subsample':0.7, 'colsample_bytree':0.7, "eval_metric": "mlogloss","objective": "multi:softprob", "num_class": 12}, 'xgb', test_device_ids, X_test) )
    cases.append( ('max-depth: 10',  X_train, y_train, X_val, y_val, {'seed':rs, 'n_estimators':150,'eta':0.1,'max_depth':10,'subsample':0.7, 'colsample_bytree':0.7, "eval_metric": "mlogloss","objective": "multi:softprob", "num_class": 12}, 'xgb', test_device_ids, X_test) )
    cases.append( ('max-depth: 12',  X_train, y_train, X_val, y_val, {'seed':rs, 'n_estimators':150,'eta':0.1,'max_depth':12,'subsample':0.7, 'colsample_bytree':0.7, "eval_metric": "mlogloss","objective": "multi:softprob", "num_class": 12}, 'xgb', test_device_ids, X_test) )
    cases.append( ('max-depth: 14',  X_train, y_train, X_val, y_val, {'seed':rs, 'n_estimators':150,'eta':0.1,'max_depth':14,'subsample':0.7, 'colsample_bytree':0.7, "eval_metric": "mlogloss","objective": "multi:softprob", "num_class": 12}, 'xgb', test_device_ids, X_test) )
    
    run_cases(cases,'Comparison max-depth') 



if __name__ == "__main__":
    
    inputdir = './data/'
    outputdir = './report_timing/'
    rs = 123
    
    upload_to_kaggle = False
    
    ylim_logloss = [2.2,2.6]
    
    hdlist = ['device_id', 'F23-', 'F24-26', 'F27-28', 'F29-32', 'F33-42','F43+', 
                           'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+']        

    df = pd.read_csv(inputdir + 'gender_age_train.csv', dtype={'device_id': np.str})
    df = map_column(df, 'group')

    print('Read brands...')
    pbd = pd.read_csv(inputdir + 'phone_brand_device_model.csv', dtype={'device_id': np.str})
    pbd.drop_duplicates('device_id', keep='first', inplace=True)
    pbd = map_column(pbd, 'phone_brand')
    pbd = map_column(pbd, 'device_model')
   
    print('Read train...')
    app_cat_per_device = pd.read_csv(inputdir + 'device_app_cats.csv', dtype={'device_id': np.str})
    
    train = pd.merge(df, pbd, how='left', on='device_id', left_index=True)
    train = pd.merge(train, app_cat_per_device, how='left', on='device_id')
    train.fillna(-1, inplace=True)
    
    X = train.drop(['device_id','gender','age','group'],axis=1)
    y = train['group']
    
    test = pd.read_csv(inputdir + 'gender_age_test.csv', dtype={'device_id': np.str})
    
    test = pd.merge(test, pbd, how='left', on='device_id', left_index=True)
    test = pd.merge(test, app_cat_per_device, how='left', on='device_id')
    test.fillna(-1, inplace=True)
    
    X_test = test.drop('device_id', axis=1)
    test_device_ids = test['device_id']

    #compare_max_depth_large()
    compare_test_sizes()
    
    # Normalize a
    
    

