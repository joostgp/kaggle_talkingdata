# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 16:19:02 2016

@author: joostbloom
"""
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split

from ml_toolbox.classifiers import get_xgboost_classifier
from ml_toolbox.kaggle import KaggleResult


outputdir = 'reporting'
rs = 123
import xgboost as xgb 
def get_xgboost_classifier(X_train, y_train, X_val, y_val, params=None, rs=123, output_eval=False):    
    
    if params is None:
        param_grid = {
            "gamma": [0], # Default 0
            "min_child_weight": [4], # Default 1
            'reg_alpha': [0], # Default 0
            "learning_rate": [0.3], # Default 0.3
            "max_depth": [5,15,25], # Default 6
            "subsample": [1], # Default 1
            "colsample_bytree": [1], # Default 1
            "n_estimators": [50],
            "silent": [1],
            "seed": [rs]
        }
        
        gbm = XGBClassifier()
        cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=5,test_size=0.2, random_state=rs)
        clf = grid_search.GridSearchCV(gbm, param_grid, cv=cv, n_jobs=2, scoring='roc_auc', verbose=5)    
        #clf = grid_search.RandomizedSearchCV(gbm, param_grid, n_iter=5, scoring='roc_auc', n_jobs=4, cv=cv, verbose=0, random_state=rs)
        clf = clf.fit(X_train,y_train)
            
        print("Best score:{} with scorer {}".format(clf.best_score_, clf.scorer_))
        print "With parameters:"
        
        best_parameters = clf.best_estimator_.get_params()
        for param_name in sorted(param_grid.keys()):
            print '\t%s: %r' % (param_name, best_parameters[param_name])
        
        
        params = {
            "gamma": best_parameters['gamma'], # Default 0
            "min_child_weight": best_parameters['min_child_weight'], # Default 1
            'alpha': best_parameters['reg_alpha'], # Default 0
            "eta": best_parameters['learning_rate'], # Default 0.3
            "max_depth": best_parameters['max_depth'], # Default 6
            "subsample": best_parameters['subsample'], # Default 1
            "colsample_bytree": best_parameters['colsample_bytree'], # Default 1
            "n_estimators": 50,
            "silent": 1,
            "seed": rs
        }
    
    params["objective"] = "multi:softprob"
    params["booster"] = "gbtree"
    params["eval_metric"] =  "mlogloss"
    
    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_val, y_val)
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    evals_result = {}
    gbm = xgb.train(params, dtrain, params['n_estimators'], evals=watchlist, early_stopping_rounds=50, verbose_eval=True, evals_result=evals_result)
    
    
    if output_eval:
        test_scores = evals_result['eval'][params["eval_metric"]]
        train_scores = evals_result['train'][params["eval_metric"]]
        
        df = pd.DataFrame()
        
        df['Eval set'] = test_scores
        df['Train set'] = train_scores
        
        plt.figure()
        plt.title("Progress AUC score during boosting")
        plt.plot(test_scores,'g',label='Validation set')
        plt.plot(train_scores,'r',label='Train set')
        plt.grid()
        plt.xlabel('Boosting round')
        plt.ylabel('AUC Score')
        plt.legend(loc=4)  
        plt.savefig('reporting/eval_curves_' + str(rs) + '.png')
        
        return gbm, df
    else:
        return gbm
def map_column(table, f):
    labels = sorted(table[f].unique())
    mappings = dict()
    for i in range(len(labels)):
        mappings[labels[i]] = i
    table = table.replace({f: mappings})
    return table

def age_group(sex, age):
    # Based on:
    #ageGroupsF = ['23-','24-26','27-28','29-32','33-42,''43+']
    #ageGroupsM = ['22-','23-26','27-28','29-31','32-38,''39+']

    if sex not in ['M','F']:
        ValueError('%s is not a valid gender' % sex)
        
    if age not in range(100):
        ValueError('%s is not a valid age' % age)
    
    print sex, age
    
    
    if sex=="M":
        if age<=22:
            g = 0
        elif age<=26:
            g = 1
        elif age<=28:
            g = 2
        elif age<=31:
            g = 3
        elif age<=38:
            g = 4
        else:
            g = 5
    elif sex=="F":
        if age<=23:
            g = 0
        elif age<=26:
            g = 1
        elif age<=28:
            g = 2
        elif age<=32:
            g = 3
        elif age<=42:
            g = 4
        else:
            g = 5
    
    return g

def combine_prob(y_age, y_gender):
    df=pd.DataFrame()
    
    hdlist = ['F23-', 'F24-26', 'F27-28', 'F29-32', 'F33-42','F43+', 
              'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+']
    
    # Add all female age groups
    for i in range(6):
        y = pd.Series(np.multiply(y_age[:,i],y_gender[:,0]))
        
        df[hdlist[i]] = y
    
    # Add all male age groups
    for i in range(6):
        y = pd.Series(np.multiply(y_age[:,i],y_gender[:,1]))
        
        df[hdlist[i+6]] = y
    
    return df


if __name__ == "__main__":
    
    random.seed(2016)
    
    hdlist = ['device_id', 'F23-', 'F24-26', 'F27-28', 'F29-32', 'F33-42','F43+', 
                           'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+']        
    
    print('Read events...')
    events = pd.read_csv("./data_ori/events.csv", dtype={'device_id': np.str})
    events['counts'] = events.groupby(['device_id'])['event_id'].transform('count')
    events_small = events[['device_id', 'counts']].drop_duplicates('device_id', keep='first')

    # Phone brand
    print('Read brands...')
    pbd = pd.read_csv("./data_ori/phone_brand_device_model.csv", dtype={'device_id': np.str})
    pbd.drop_duplicates('device_id', keep='first', inplace=True)
    pbd = map_column(pbd, 'phone_brand')
    pbd = map_column(pbd, 'device_model')
    
    # Read appst
    appt = pd.read_csv("./data/device_apps_per_timeunit.csv", dtype={'Unnamed: 0': np.str})
    appt['device_id']=appt['Unnamed: 0']
    appt.drop('Unnamed: 0', axis=1, inplace=True)
    

    # Train
    print('Read train...')
    train = pd.read_csv("./data_ori/gender_age_train.csv", dtype={'device_id': np.str})
    train = map_column(train, 'group')
    train = train.drop(['age'], axis=1)
    train = train.drop(['gender'], axis=1)
    train = pd.merge(train, pbd, how='left', on='device_id', left_index=True)
    train = pd.merge(train, events_small, how='left', on='device_id', left_index=True)
    train = pd.merge(train, appt, how='left', on='device_id', left_index=True)
    train.fillna(-1, inplace=True)
    
    print('Read test...')
    test = pd.read_csv("./data_ori/gender_age_test.csv", dtype={'device_id': np.str})
    test = pd.merge(test, pbd, how='left', on='device_id', left_index=True)
    test = pd.merge(test, events_small, how='left', on='device_id', left_index=True)
    test = pd.merge(test, appt, how='left', on='device_id', left_index=True)
    test.fillna(-1, inplace=True)
    
    #one_hot = pd.get_dummies(X_train['phone_brand'])
    X = train.drop(['group','device_id'], axis=1)
    y = train['group']
    X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.3, random_state=rs, stratify=y)
    #X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    f = ['counts','phone_brand','device_model']
    
    # Test on group
    params = {'n_estimators':500,'eta':0.1,'max-depth':5,'subsample':0.7, 'colsample_bytree':0.7, "eval_metric": "mlogloss","objective": "multi:softprob", "num_class": 12}  
    (gbm, eval_result) = get_xgboost_classifier(X_train, y_train, X_val, y_val, params, rs=2016, output_eval=True)
    
    y_val_check = gbm.predict(xgb.DMatrix(X_val), ntree_limit=gbm.best_iteration)
    score = log_loss(y_val, y_val_check)
    
    print "Total score: {:.4f}".format(score)
    
    
    X_test = test.drop(['device_id'], axis=1)
    
    y_test = gbm.predict(xgb.DMatrix(X_test))
    
    kag = KaggleResult(test['device_id'], prediction=y_test, score=score, description="second script", sub_path='second_test') 
    print kag.validate()
    #lb_score = kag.upload("Second actual test")
    
    
    

