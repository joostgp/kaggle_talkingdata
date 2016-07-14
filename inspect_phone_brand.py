# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 19:56:21 2016

@author: guusjeschouten
"""
import pandas as pd
import numpy as np


def get_phone_brand_translation():
    
    df=pd.read_csv('./data_ori/phone_brands_trans.csv') #,encoding='utf-8'
    
    # Check whether there are no duplicates before setting index
    print("Duplicates?: %d vs. %d" % (len(df),len(df.ix[:,1].unique())))
    df = df.set_index(df.columns[1])
    return df[df.columns[1]].to_dict()
    
def get_devide_model_translation():
    
    df=pd.read_csv('./data_ori/device_model_trans.csv') #,encoding='utf-8'
    
    # Check whether there are no duplicates before setting index
    print("Duplicates?: %d vs. %d" % (len(df),len(df.ix[:,1].unique())))
    df = df.set_index(df.columns[1])
    return df[df.columns[1]].to_dict()

if __name__=='__main__':
    a=pd.read_csv('./data_ori/phone_brand_device_model.csv')
    
    #print(len(a))
    #print(len(a['label_id'].unique()))
    #print(len(a['category'].unique()))
    
    # Output:
    # 930
    # 930
    # 836
    # So some label_ids have the same category label
    # TO-DO: What to do with this?
    
    pd.Series(a['phone_brand'].unique()).to_csv('./data_ori/phone_brands_raw.csv')
    pd.Series(a['device_model'].unique()).to_csv('./data_ori/device_model_raw.csv')
    
    #print a['phone_brand']
    
    #print a['device_model']
    
    phone_brand_transl = get_phone_brand_translation()
    device_model_transl = get_devide_model_translation()
    
