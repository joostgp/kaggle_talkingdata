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


def unpack_date(df, date_col):
    # Create columns for elements of date_col
    
    df[date_col+'_h'] = df[date_col].dt.hour
    df[date_col+'_d'] = df[date_col].dt.day
    
    return df
    
    
def calculate_event_features(df_in):
    
    df_out = pd.DataFrame(index=df_in['device_id'].unique())
    
    # Number of days with events
    count = df_in.groupby(['device_id','timestamp_d']).timestamp_d.nunique().unstack(['timestamp_d'])
    count = count.sum(axis=1).to_frame('days with events')
    
    df_out['n day events'] = count
    
    # Number of hours with events
    count = df_in.groupby(['device_id','timestamp_h']).timestamp_h.nunique().unstack(['timestamp_h'])
    count = count.sum(axis=1).to_frame('hours with events')

    df_out['n hour events'] = count

    #Number of events per day (sum, max, mean, std)
    count = df_in.groupby(['device_id','timestamp_d']).size().unstack('timestamp_d')
    
    df_out['n event sum'] = count.sum(axis=1)
    
    df_out['n event day max'] = count.max(axis=1)
    df_out['n event day mean'] = count.mean(axis=1)
    df_out['n event day std'] = count.std(axis=1)

    #Number of events per hour (sum, max, mean, std)
    count = df_in.groupby(['device_id','timestamp_h']).size().unstack('timestamp_h')
    
    df_out['n event hour max'] = count.max(axis=1)
    df_out['n event hour mean'] = count.mean(axis=1)
    df_out['n event hour std'] = count.std(axis=1)
    
    return df_out

if __name__ == "__main__":
    
    dir_in = './data_ori/'
    dir_out = './data/'
    feature_file = 'features_event.csv'
    rs = 123
    
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
    
    #split_events_per_timeunit()
    
    print('Calculating features...')
    event_features = calculate_event_features(events)
    
    # Validation:
    event_features.loc[-9221026417907250887]
    
    event_features.to_csv(dir_out + feature_file, index_label='device_id' )
    print('Saved features to: ' + dir_out + feature_file)