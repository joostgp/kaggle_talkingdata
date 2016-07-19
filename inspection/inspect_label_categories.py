# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 19:56:21 2016

@author: guusjeschouten
"""
import pandas as pd


if __name__=='__main__':
    a=pd.read_csv('./data_ori/label_categories.csv')
    
    print(len(a))
    print(len(a['label_id'].unique()))
    print(len(a['category'].unique()))
    
    # Output:
    # 930
    # 930
    # 836
    # So some label_ids have the same category label
    # TO-DO: What to do with this?