# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 16:19:02 2016

@author: joostbloom
"""
import math
import random
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def age_group(sex, age):
    # Convert age column to age group
    #ageGroupsF = ['23-','24-26','27-28','29-32','33-42,''43+']
    #ageGroupsM = ['22-','23-26','27-28','29-31','32-38,''39+']

    if sex not in ['M','F']:
        ValueError('%s is not a valid gender' % sex)
        
    if age not in range(100):
        ValueError('%s is not a valid age' % age)
    
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

if __name__ == "__main__":
    
    dir_in = './data_ori/'
    dir_out = './data/'
    feature_file = 'features_brand.csv'
    rs = 123
    
    print('Reading data...')
    brands = pd.read_csv(dir_in + 'phone_brand_device_model.csv', index_col='device_id')
    brands = brands.reset_index().drop_duplicates(subset='device_id',inplace=False).set_index('device_id')
    
    # Validation
#    print brands.index.nunique()
#    print brands.shape
#    # Both return 186716 rows
    
    print('Lbel encoding phone brand and device model...')
    # LabelEncoding
    brands_le = brands.copy()
    
    brands_le['phone_brand'] = preprocessing.LabelEncoder().fit_transform(brands.phone_brand)
    brands_le['device_model'] = preprocessing.LabelEncoder().fit_transform(brands.device_model)
    
    # Validation
#    print brands_le.phone_brand.min()
#    # Returns 0
#    print brands_le.phone_brand.max()
#    # Returns 130
#    print brands.phone_brand.nunique()
#    # Returns 131
#    print brands_le.device_model.min()
#    # Returns 0
#    print brands_le.device_model.max()
#    # Returns 1598
#    print brands.device_model.nunique()
#    # Return 1599
    
    brands_le.to_csv(dir_out+'feat_brands_labelencoded.csv', index_label='device_id')
    
    # OneHot encoding of both brand and device
    print('One hot encoding phone brand and device model...')
    brands_ohe = brands_le.copy()
    brands_ohe_both = pd.DataFrame(preprocessing.OneHotEncoder(sparse=False).fit_transform(brands_le[['phone_brand','device_model']]), index=brands.index)
    brands_ohe_both.columns = 'both_ohe_' + brands_ohe_both.columns.astype(str)
    brands_ohe_both.to_csv(dir_out+'feat_brands_both_combined_onehotencoded.csv', index_label='device_id')
    
    # Validation
    # Compare number of columns to unique
    
    # onehot encode brand only   
    print('One hot encoding phone brand...') 
    brands_ohe_brand_only = pd.DataFrame(preprocessing.OneHotEncoder(sparse=False).fit_transform(brands_le.phone_brand.reshape([-1,1])), index=brands.index)
    brands_ohe_brand_only.columns = 'brand_ohe_' + brands_ohe_brand_only.columns.astype(str)
    brands_ohe_brand_only['device_model'] = brands_le['device_model']
    brands_ohe_brand_only.to_csv(dir_out+'feat_brands_brand_onehotencoded.csv', index_label='device_id')
    
    # Validation
    # Compare number of columns to unique
   
   # onehot encode device model only   
    print('One hot encoding device model...') 
    brands_ohe_model_only = pd.DataFrame(preprocessing.OneHotEncoder(sparse=False).fit_transform(brands_le.device_model.reshape([-1,1])), index=brands.index)
    brands_ohe_model_only.columns = 'model_ohe_' + brands_ohe_model_only.columns.astype(str)
    brands_ohe_both_separate = brands_ohe_brand_only.drop('device_model', axis=1).join(brands_ohe_model_only)
    brands_ohe_both_separate.to_csv(dir_out+'feat_brands_both_separate_onehotencoded.csv', index_label='device_id')
    
    # Validation    
    # Compare number of columns to unique
    
    
    
    