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

def unpack_date(df, date_col):
    # Create columns for elements of date_col
    
    df[date_col+'_h'] = df[date_col].dt.hour
    df[date_col+'_d'] = df[date_col].dt.day
    
    return df

def split_events_per_timeunit():
    
    print('Reading data...')
    #train = pd.read_csv(dir_in + 'gender_age_train.csv')
    #test  =pd.read_csv(dir_in + 'gender_age_test.csv')    
    events = pd.read_csv(dir_in + 'events.csv')
    
    #print('Merging data...')
    #train = pd.merge(train, events, how='inner', on='device_id')
    #test = pd.merge(test, events, how='inner', on='device_id')
    
    print('Preprocessing data...')
    # Much faster than dateparser!
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    events = unpack_date(events, 'timestamp')
    
    print('Splitting and saving events per day...')
    days = events.timestamp_d.unique()
    
    count = 0 
    for day in days:
        df = events[events.timestamp_d==day]
        count += df.shape[0]
        df.to_csv(dir_out + 'events_per_timeunit/' + 'events_d_' + str(day) + '.csv', index=False)
        print day, count
    print events.shape[0], count
    
    print('Splitting and saving events per hour...')
    hours = events.timestamp_h.unique()
    
    count = 0 
    for hour in hours:
        df = events[events.timestamp_h==hour]
        count += df.shape[0]
        df.to_csv(dir_out + 'events_per_timeunit/' + 'events_h_' + str(hour) + '.csv', index=False)
        print hour, count
    print events.shape[0], count

def generate_app_cat_files():
    
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
    appcats.columns = 'app_cat_sa_' + appcats.columns.astype(str)
    
    print('Reading app events...')
    app_events = pd.read_csv(dir_in + 'app_events.csv')
    app_events_active = app_events[app_events.is_active>0]
    app_events_active['is_installed']=app_events['is_installed']
    
    days = range(1,9) +[30]
    hours = range(0, 24)
    
    for hour in hours:
        print('Merging with hour ' + str(hour)), 
        a=time.time()
        
        events = pd.read_csv(dir_out + 'events_h_' + str(hour) + '.csv')
        
        df = pd.merge(events,app_events_active,how='inner',on='event_id')
        df = df[['app_id','device_id','is_installed','is_active']]
        
        df = df.groupby(['device_id','app_id']).sum().reset_index(level=0)
        
        app_cat_per_device = pd.merge(df, appcats, how='left', left_index=True, right_index=True).groupby('device_id').sum()
        app_cat_per_device.to_csv(dir_out + 'app_cat_per_device_active_h_' + str(hour) + '.csv', index_label='device_id')
        
        print time.time()-a
    
    for day in days:
        print('Merging with day ' + str(day)), 
        a=time.time()
        
        events = pd.read_csv(dir_out + 'events_d_' + str(day) + '.csv')
        
        df = pd.merge(events,app_events_active,how='inner',on='event_id')
        df = df[['app_id','device_id','is_installed','is_active']]
        
        df = df.groupby(['device_id','app_id']).sum().reset_index(level=0)
        
        app_cat_per_device = pd.merge(df, appcats, how='left', left_index=True, right_index=True).groupby('device_id').sum()
        app_cat_per_device.to_csv(dir_out + 'app_cat_per_device_active_d_' + str(day) + '.csv', index_label='device_id')
        
        
        print time.time()-a
        
def generate_app_group_files():
    
    print('Reading data...')
    # Load app_events
    
    print('Reading app groups...')
    app_labels = pd.read_csv('./data/app_cats_grouped.csv')
    app_labels['ones'] = 1
    
    appcats = app_labels.pivot_table(values='ones',columns='general_groups',index=['app_id'])
    appcats.fillna(0,inplace=True)
    appcats.columns = 'app_cat_sa_' + appcats.columns.astype(str)
    
    print('Reading app events...')
    app_events = pd.read_csv(dir_in + 'app_events.csv')
    app_events_active = app_events[app_events.is_active>0]
    app_events_active['is_installed']=app_events['is_installed']
    
    days = range(1,9) +[30]
    hours = range(0, 24)
    
    for hour in hours:
        print('Merging with hour ' + str(hour)), 
        a=time.time()
        
        events = pd.read_csv(dir_out + 'events_h_' + str(hour) + '.csv')
        
        df = pd.merge(events,app_events_active,how='inner',on='event_id')
        df = df[['app_id','device_id','is_installed','is_active']]
        
        df = df.groupby(['device_id','app_id']).sum().reset_index(level=0)
        
        app_cat_per_device = pd.merge(df, appcats, how='left', left_index=True, right_index=True).groupby('device_id').sum()
        app_cat_per_device.to_csv(dir_out + 'app_group_per_device_active_h_' + str(hour) + '.csv', index_label='device_id')
        
        print time.time()-a
    
    for day in days:
        print('Merging with day ' + str(day)), 
        a=time.time()
        
        events = pd.read_csv(dir_out + 'events_d_' + str(day) + '.csv')
        
        df = pd.merge(events,app_events_active,how='inner',on='event_id')
        df = df[['app_id','device_id','is_installed','is_active']]
        
        df = df.groupby(['device_id','app_id']).sum().reset_index(level=0)
        
        app_cat_per_device = pd.merge(df, appcats, how='left', left_index=True, right_index=True).groupby('device_id').sum()
        app_cat_per_device.to_csv(dir_out + 'app_group_per_device_active_d_' + str(day) + '.csv', index_label='device_id')
        
        
        print time.time()-a

def calculate_app_usage_per_timeunit():
    
    night = [0,1,2,3,4,5]
    morning = [6,7,8,9,10,11]
    afternoon = [12,13,14,15,16,17]
    evening = [18,19,20,21,22,23]
    
    hour = 0
    
    if 0:
        filestr = 'app_cat_per_device'
    else:
        filestr = 'app_group_per_device'
        
    
    df0 = pd.read_csv(dir_out + filestr + '_active_h_0.csv', index_col='device_id')
    df1 = pd.read_csv(dir_out + filestr + '_active_h_1.csv', index_col='device_id')
    
    print('df0 shape %s' % df0.shape[0])
    print('df0 unique %d' % df0.index.nunique())
    print('df1 shape %s' % df1.shape[0])
    print('df1 unique %d' % df1.index.nunique())
    print('Difference %d' % len(set(df0.index) - set(df1.index)))
    print('Difference2:  %d' % (df0.shape[0] - df1.shape[0]))
    
    df = df0.add(df1)
    print('Difference after adding: %d' % len(set(df.index)-set(df0.index) - set(df1.index)))
    
    # Do night
    df = pd.read_csv(dir_out + filestr + '_active_h_' + str(night[0]) + '.csv', index_col='device_id')
    for hour in night[1:]:
        print('Adding %d...' % hour)
        df_l = pd.read_csv(dir_out + filestr + '_active_h_' + str(hour) + '.csv', index_col='device_id')
        df = df.add(df_l, fill_value=0)
    
    df_night = df
    
    # Do morning
    df = pd.read_csv(dir_out + filestr + '_active_h_' + str(morning[0]) + '.csv', index_col='device_id')
    for hour in morning[1:]:
        print('Adding %d...' % hour)
        df_l = pd.read_csv(dir_out + filestr + '_active_h_' + str(hour) + '.csv', index_col='device_id')
        df = df.add(df_l, fill_value=0)
    
    df_morning = df
    
    # Do afternoon
    df = pd.read_csv(dir_out + filestr + '_active_h_' + str(afternoon[0]) + '.csv', index_col='device_id')
    for hour in afternoon[1:]:
        print('Adding %d...' % hour)
        df_l = pd.read_csv(dir_out + filestr + '_active_h_' + str(hour) + '.csv', index_col='device_id')
        df = df.add(df_l, fill_value=0)
    
    df_afternoon = df
    
    # Do evening
    df = pd.read_csv(dir_out + filestr + '_active_h_' + str(evening[0]) + '.csv', index_col='device_id')
    for hour in evening[1:]:
        print('Adding %d...' % hour)
        df_l = pd.read_csv(dir_out + filestr + '_active_h_' + str(hour) + '.csv', index_col='device_id')
        df = df.add(df_l, fill_value=0)
    
    df_evening = df
    
    df_night.to_csv(dir_out + filestr + '_active_night.csv', index_label='device_id')
    df_morning.to_csv(dir_out + filestr + '_active_morning.csv', index_label='device_id')
    df_afternoon.to_csv(dir_out + filestr + '_active_afternoon.csv', index_label='device_id')
    df_evening.to_csv(dir_out + filestr + '_active_evening.csv', index_label='device_id')
    
    weekdays = [2,3,4,5,6]
    weekends = [30,1,7,8]
    
    # Do weekdays
    df = pd.read_csv(dir_out + filestr + '_active_d_' + str(weekdays[0]) + '.csv', index_col='device_id')
    for day in weekdays[1:]:
        print('Adding day %d...' % day)
        df_l = pd.read_csv(dir_out + filestr + '_active_d_' + str(day) + '.csv', index_col='device_id')
        df = df.add(df_l, fill_value=0)
    
    df_weekdays = df
    
    # Do weekends
    df = pd.read_csv(dir_out + filestr + '_active_d_' + str(weekends[0]) + '.csv', index_col='device_id')
    for day in weekends[1:]:
        print('Adding day %d...' % day)
        df_l = pd.read_csv(dir_out + filestr + '_active_d_' + str(day) + '.csv', index_col='device_id')
        df = df.add(df_l, fill_value=0)
    
    df_weekends = df
    
    df_weekends.to_csv(dir_out + filestr + '_active_weekends.csv', index_label='device_id')
    df_weekdays.to_csv(dir_out + filestr + '_active_weekdays.csv', index_label='device_id')


if __name__ == "__main__":
    
    dir_in = './data_ori/'
    dir_out = './data/events_per_timeunit/'
    
    rs = 123
    
    #print('Calculating app events per timeunit...')  
    #generate_app_group_files()
    
    print('Calculating app usage per time period...')  
    calculate_app_usage_per_timeunit()
    

    
    
        
    
    
    
   
