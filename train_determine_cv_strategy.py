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
import pickle
import os

from scipy.sparse import csr_matrix, hstack

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


from ml_toolbox.kaggle import KaggleResult
from ml_toolbox.runner import CaseRunner


import xgboost as xgb 

# --------------------
# - Main
# --------------------

def run():
    
    # Load features for brands & device models
    # TO-DO: Find out if it matters whether train on full data or just data 
    # without events    
    # For now train on all devices
    gatrain = pd.read_csv('./data_ori/gender_age_train.csv')
    gatest = pd.read_csv('./data_ori/gender_age_test.csv')
    brand = pd.read_csv('./data/old/feat_brands_labelencoded.csv')
    X_train = pd.merge(gatrain, brand, on='device_id', how='inner')
    X_test = pd.merge(gatest, brand, on='device_id', how='inner')
    
    y = X_train['group']
    
    letarget = LabelEncoder().fit(y)
    y = letarget.transform(y)
    n_classes = len(letarget.classes_)   
    
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y, stratify=y, 
                                                  test_size=0.1, random_state=42)
     
    
    # Train Bayesian model on train data
    (clfs_bay_brand, clfs_bay_device) = train_bayesian_classifiers(X_train, y_train, n_classes, rs)
    
    pred_tr = []
    pred_te = []
    
    w1 = 1
    w2 = 1.3
    
    for clf_brand, clf_device in zip(clfs_bay_brand,clfs_bay_device):
        pred_tr.append( (w1 * clf_brand.predict_proba(X_val, 'phone_brand') + w2 * clf_device.predict_proba(X_val, 'device_model')) / (w1 + w2) )
        pred_te.append( (w1 * clf_brand.predict_proba(X_test, 'phone_brand') + w2 * clf_device.predict_proba(X_test, 'device_model')) / (w1 + w2) )
    
    score_bay = log_loss(y_val, sum(pred_tr)/len(pred_tr) )
    print('Score on train Bayesian model: %f' % score_bay)
    
#    kag = KaggleResult(test['device_id'], sum(pred_te)/len(pred_te), score, 'test bayesian models', outputdir)
#    if kag.validate():
#        kag.upload('test bayesian models')
    
    feature_files = ['features_brand_bag',
                 'features_brand_model_bag',
                 'features_brand_model.csv']
    
    # Load features for brands & device models
    # TO-DO: Find out if it matters whether train on full data or just data 
    # without events    
    # For now train on all devices
    
    Xtrain = hstack([open_feature_file(f,'train', gatrain) for f in feature_files], format='csr')
    X_test = hstack([open_feature_file(f,'test', gatest) for f in feature_files], format='csr')
    
    
    X_train, X_val, y_train, y_val = train_test_split(Xtrain, y, stratify=y, 
                                                  test_size=0.1, random_state=42)
    
    # Train XGBoost model on train
    clfs_xgb = train_xgboost_classifiers(X_train, y_train, n_classes, rs)
    
    pred_xgb_tr = []
    pred_xgb_te = []
    for clf_xgb in clfs_xgb:
        pred_xgb_tr.append(clf_xgb.predict(xgb.DMatrix(X_val, label=y)))
        pred_xgb_te.append(clf_xgb.predict(xgb.DMatrix(X_test)) )
    
    score_xgb = log_loss(y_val, sum(pred_xgb_tr)/len(pred_xgb_tr) )
    print('Score on train XGBoost model: %f' % score_xgb)
    
    clfs_log = train_linear_classifiers(X_train, y_train, n_classes, rs)
    
    # Train Logistic model on train (with hot encoded data)
    pred_log_tr = []
    pred_log_te = []
    for clf_log in clfs_log:
        pred_log_tr.append(clf_log.predict_proba(X_val))
        pred_log_te.append(clf_log.predict_proba(X_test)) 
    
    score_log = log_loss(y_val, sum(pred_log_tr)/len(pred_log_tr) )
    print('Score on train logistic model: %f' % score_log)

    
    # 4 1 1 resulted in: 2.38688
    # 2 1 1 resulted in: 2.38708   
    # 8 0 1 resulted in: 2.38664  
    # 2 0 1 resulted in: 2.38655
    # 3 0 2 resulted in: 2.38657
    # 3 0 1 resulted in: 2.38655
    # 3 1 2 resulted in: 
    # No correlation between CV score and LB score
#    (3, 0, 2) : 2.35228380172
#    (3, 0, 1) : 2.35047487322
#    (3, 1, 2) : 2.35381211921
#    (6, 1, 4) : 2.35310657137
#    (12, 1, 8) : 2.35271151441
#    (12, 2, 6) : 2.35245220214
#    (12, 3, 6) : 2.35288554969
#    (12, 4, 8) : 2.35381211921

    ws = [(3, 0, 1),
          (12, 1, 6)]    
    for w in ws:
        score_final = log_loss(y_val, (w[0] * sum(pred_tr)/len(pred_tr) +
                      w[1] * sum(pred_xgb_tr)/len(pred_xgb_tr) +
                      w[2] * sum(pred_log_tr)/len(pred_log_tr) ) / (w[0]+w[1]+w[2]))
        
        print w,' cv score:',score_final    
        pred_final =  (w[0] * sum(pred_te)/len(pred_te) +
                      w[1] * sum(pred_xgb_te)/len(pred_xgb_te) +
                      w[2] * sum(pred_log_te)/len(pred_log_te) ) / (w[0]+w[1]+w[2])
        
        kag = KaggleResult(gatest['device_id'], pred_final, score_final, 'test V2 new CV strategy %s' % (w, ), outputdir)
        if kag.validate():
            kag.upload()
        print w,' lb score:', kag.lb_score
                 
                    
    
    # Export train predictions to use as features
    
    # Predict using three models
    
    # Stack & Ensemble
    

# --------------------
# - Support functions / classes
# --------------------

def open_feature_file(fname, samples, device_data):
    if fname[-3:] == 'csv':
        if samples=='train':
            X = device_data[['device_id']].merge( pd.read_csv(os.path.join(feat_dir, fname)), on='device_id', how='left')
        else:
            X = device_data[['device_id']].merge( pd.read_csv(os.path.join(feat_dir, fname)), on='device_id', how='left')
            
        X.drop('device_id', axis=1, inplace=True)
        X.fillna(0, inplace=True)
        
        if use_scaler:
            for c in X.columns:
                if X[c].max()>1:
                    X[c] = StandardScaler().fit_transform(X)
            
        #print X.shape
        return csr_matrix(X.values)
    else:
        # Assume it is a pickle file
        with open(os.path.join(feat_dir, '{}_{}.pickle'.format(fname,samples)), 'rb') as f:
            return pickle.load(f)


# Based on https://www.kaggle.com/dvasyukova/talkingdata-mobile-user-demographics/brand-and-model-based-benchmarks/comments

def train_bayesian_classifiers(X, y, n_classes, rs ):
    n_fold = 10 
#    w1 = 1
#    w2 = 1.3   
    
    clfs_brand = []
    clfs_device = []

    
    
    kf = StratifiedKFold(y, n_folds=n_fold, shuffle=True, random_state=rs)
#    predb = np.zeros((X.shape[0],n_classes))
#    predm = np.zeros((X.shape[0],n_classes))

    for itrain, itest in kf:
        train = X.iloc[itrain,:]
#        test = X.iloc[itest,:]
        clf = GenderAgeGroupProb(prior_weight=30.).fit(train,'phone_brand')
        clfs_brand.append(clf)
#        pred1 = clf.predict_proba(X,'phone_brand')
#        predb[itest,:] = clf.predict_proba(test,'phone_brand')
        clf = GenderAgeGroupProb(prior_weight=20.).fit(train,'device_model')
        clfs_device.append(clf)
#        pred2 = clf.predict_proba(X,'device_model')
#        predm[itest,:] = clf.predict_proba(test,'device_model')
    
#        print log_loss(y, (w1 * pred1 + w2 * pred2) / (w1 + w2) )     
        

    return clfs_brand, clfs_device

class GenderAgeGroupProb(object):
    def __init__(self, prior_weight=10.):
        self.prior_weight = prior_weight
    
    def fit(self, df, by):
        self.by = by
        #self.label = 'pF_' + by
        self.prior = df['group'].value_counts().sort_index()/df.shape[0]
        # fit gender probs by grouping column
        c = df.groupby([by, 'group']).size().unstack().fillna(0)
        self.prob = (c.add(self.prior_weight*self.prior)).div(c.sum(axis=1)+self.prior_weight, axis=0)
        return self
    
    def predict_proba(self, df, by):
        self.by = by
        pred = df[[self.by]].merge(self.prob, how='left', 
                                left_on=self.by, right_index=True).fillna(self.prior)[self.prob.columns]
        pred.loc[pred.iloc[:,0].isnull(),:] = self.prior
        return pred.values    

def train_linear_classifiers(X, y, n_classes, rs):
    n_fold = 10 

    kf = StratifiedKFold(y, n_folds=n_fold, shuffle=True, random_state=rs)
    pred_l = np.zeros((X.shape[0],n_classes))
    c=1
    
    clfs = []
    
    for itrain, itest in kf:
        ytrain, ytest = y[itrain], y[itest]
        xg_train = X[itrain, :]
        xg_test = X[itest, :]
        
        clf = LogisticRegression(C=0.13, penalty='l2')
        clf.fit(xg_train, ytrain)
        pred_l[itest,:] = clf.predict_proba(xg_test)
        c+=1
        clfs.append(clf)
    print log_loss(y, pred_l)
    
    return clfs   

def train_xgboost_classifiers(X, y, n_classes, rs):
    n_fold = 10 
    
    params = {
        "objective": "multi:softprob",
        'booster': 'gblinear',
        'num_class': 12,
        "eta": 0.01,
        "silent": 1,
        'alpha': 1,
        'lambda': 3,
        'n_estimators': 150,
        'eval_metric': 'mlogloss',
        'seed': rs
    }

    kf = StratifiedKFold(y, n_folds=n_fold, shuffle=True, random_state=rs)
#    pred = np.zeros((X.shape[0],n_classes))
    
    clfs = []
    
    for itrain, itest in kf:
        ytrain, ytest = y[itrain], y[itest]
        xg_train = xgb.DMatrix( X[itrain, :], label=ytrain)
        xg_test = xgb.DMatrix(X[itest, :], label=ytest)
        watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
        bst = xgb.train(params, xg_train, params['n_estimators'], watchlist, verbose_eval=50 )
#        pred[itest,:] = bst.predict(xg_test)
        clfs.append(bst)
#    print log_loss(y, pred)
    
    return clfs 
# --------------------
# - Cases
# --------------------



if __name__ == "__main__":
    
    outputdir = './report_no_events/'
    feat_dir = './data/'
    rs = 123
    use_scaler= True
    
    run()
    
    

