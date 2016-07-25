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
    
    rs = 123
    
    print('Reading data...')
    # Load app_events
    print('Reading app categories...')
    app_cat = pd.read_csv(dir_in + 'label_categories.csv')
    
    # Event ID is unique integer
    print('Reading app labels...')
    app_labels=pd.read_csv(dir_in + 'app_labels.csv')
    
    app_labels = app_labels.merge(app_cat, how='left', on='label_id')
    app_labels['ones'] = 1
    
    appcats = app_labels.pivot_table(values='ones',columns='category',index=['app_id'])
    appcats.fillna(0,inplace=True)
    appcat_cols = appcats.columns
    
    #print('Reading train...')
    #train = pd.read_csv(dir_in + 'gender_age_train.csv', dtype={'device_id':np.str})
    
    
    #data = train
    
    # This merge works!! (takes a long time though)
    if 0:
    
        print('Reading app events...')
        app_events = pd.read_csv(dir_in + 'app_events.csv')
       
        print('Reading events...')
        events = pd.read_csv(dir_in + 'events.csv', dtype={'device_id':np.str})
        
        app_events = pd.merge(app_events,events,how='inner',on='event_id')
        app_events = app_events[['app_id','device_id','is_installed','is_active']]
        
        # This dataframe contains per device how many times each app_id was installed or active
        app_per_device = app_events.groupby(['device_id','app_id']).sum().reset_index(level=0)
        
        # Do installed
        a=time.time()
        app_cat_per_device = pd.merge(app_per_device, appcats, how='left', left_index=True, right_index=True).groupby('device_id').sum()
        print time.time()-a
        app_cat_per_device.to_csv(dir_out + 'app_cat_per_device_installed.csv', index_label='device_id')
        
        
        # Do only active
        # This reduces the number of rows from 2,369,025 to 910,458.
        # Corresponds to (app_per_device.is_active>0).sum()
        # Takes approx 30s
        app_per_device_active = app_per_device[app_per_device.is_active>0]
        a=time.time()
        app_cat_per_device_active = pd.merge(app_per_device_active, appcats, how='left', left_index=True, right_index=True).groupby('device_id').sum()
        print time.time()-a
        # Correct is_installed column
        app_cat_per_device_active['is_installed']=app_cat_per_device['is_installed']
        app_cat_per_device_active.to_csv(dir_out + 'app_cat_per_device_active.csv', index_label='device_id')
    else:
        print('Reading pre-calculated apps per device...')
        app_cat_per_device = pd.read_csv(dir_out + 'app_cat_per_device.csv', index_col='device_id')
        app_cat_per_device_active = pd.read_csv(dir_out + 'app_cat_per_device_active.csv', index_col='device_id')
    
        # is_installed column in app_cat_per_device is incorrect
        app_cat_per_device_active['is_installed']=app_cat_per_device['is_installed']
    
    # 
    """
    Make six different transformations
    1) For each category sum of installed for all events
    2) For each category relative amount installed
    3) For each category any installed during any events
    4) For each category sum of active for all events
    5) For each category relative amount installed
    6) For each category yes/no active during any event
    """
    # 1) For each category sum of installed for all events
    # No further processing required for 
    print('Saving feat apps sum installed...')
    df1 = app_cat_per_device.copy()
    cols = ('app_cat_si_' + df1.columns[2:].astype(str)) 
    cols = cols.insert(0,'is_active').insert(1,'is_installed')
    df1.columns = cols
    df1.to_csv(dir_out + 'feat_apps_sum_installed.csv', index_label='device_id')
    del df1
    
    # 2) For each category sum of active for all events
    print('Saving feat apps rel installed...')
    df2 = app_cat_per_device.copy()  
    df2[appcat_cols] = app_cat_per_device[appcat_cols].divide(app_cat_per_device['is_installed'], axis=0)
    cols = ('app_cat_ri_' + df2.columns[2:].astype(str)) 
    cols = cols.insert(0,'is_active').insert(1,'is_installed')
    df2.columns = cols
    df2.to_csv(dir_out + 'feat_apps_rel_installed.csv', index_label='device_id')
    del df2
    
    # 3) For each category relative amount installed
    print('Saving feat apps any installed...')
    df3 = app_cat_per_device.copy()  
    df3[appcat_cols] = (app_cat_per_device[appcat_cols]>0).astype(int)
    cols = ('app_cat_ai_' + df3.columns[2:].astype(str)) 
    cols = cols.insert(0,'is_active').insert(1,'is_installed')
    df3.columns = cols
    df3.to_csv(dir_out + 'feat_apps_any_installed.csv', index_label='device_id')
    
    # 4) For each category sum of active for all events
    # No further processing required 
    print('Saving feat apps sum active...')
    df4 = app_cat_per_device_active.copy()
    df4[appcat_cols] = (app_cat_per_device[appcat_cols]>0).astype(int)
    cols = ('app_cat_sa_' + df4.columns[2:].astype(str)) 
    cols = cols.insert(0,'is_active').insert(1,'is_installed')
    df4.columns = cols
    df4.to_csv(dir_out + 'feat_apps_sum_active.csv', index_label='device_id')
    
    # 5) For each category relative amount installed
    print('Saving feat apps rel active...')
    df5 = app_cat_per_device_active.copy()  
    df5[appcat_cols] = app_cat_per_device_active[appcat_cols].divide(app_cat_per_device_active['is_active'], axis=0)
    cols = ('app_cat_ra_' + df5.columns[2:].astype(str)) 
    cols = cols.insert(0,'is_active').insert(1,'is_installed')
    df5.columns = cols
    df5.to_csv(dir_out + 'feat_apps_rel_active.csv', index_label='device_id')
    del df4
    
    # 6) For each category yes/no active during any event
    print('Saving feat apps any active...')
    df6 = app_cat_per_device_active.copy()  
    df6[appcat_cols] = (app_cat_per_device_active[appcat_cols]>0).astype(int)
    cols = ('app_cat_aa_' + df6.columns[2:].astype(str)) 
    cols = cols.insert(0,'is_active').insert(1,'is_installed')
    df6.columns = cols
    df6.to_csv(dir_out + 'feat_apps_any_active.csv', index_label='device_id')
