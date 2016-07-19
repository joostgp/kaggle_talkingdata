# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 23:15:46 2016

@author: joostbloom
"""
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import time
import random

from ml_toolbox.plot import newfigure
from ml_toolbox.files import adjustfile_path

    
def split_date(df, date_values):
    # Create columns for each unique value in date_col
    
    di=pd.DatetimeIndex(date_values)
    
    for h in set(di.hour):
        df['h%d' % h] = 0
    
    for d in set(di.day):
        df['d%d' % d] = 0
        
    return df

def unpack_date(df, date_col):
    # Create columns for elements of date_col
    di=pd.DatetimeIndex(df[date_col])
    
    df[date_col+'_h'] = di.hour
    df[date_col+'_d'] = di.day
    
    return df

def plot_bar(df1, df2=None, df3=None, labels=None):
    # Plot two series in same dataframe with columns next to eachother
    
    if df3 is not None:
        width = 0.3
    elif df2 is not None:
        width = 0.45
    else:
        width = 0.9
    
    ind = np.arange(len(df1))   
    
    plt.bar(ind, df1, width, color='b')
    
    
    if df2 is not None:
        plt.bar(ind+width, df2, width, color='g') 
    if df3 is not None:
        plt.bar(ind+2*width, df3, width, color='r') 
    
    if labels is not None: 
        plt.legend(labels)
        
    plt.xticks(ind+0.5,df1.index, rotation='vertical')
    plt.grid()
    plt.tight_layout()

def norm(series):
    return (series - series.mean()) / (series.max()-series.min())
    
def show_app_usage(device_id, events, app_events):
    # Show for a specific device_id
    # How many events
    # How many apps per event
    # Bar chart with app usage during days
    # Overview with categories of this device_id
    # Something with location
    
    print('Loading info for device id %s' % device_id)
    
    hrange = range(24)
    drange = [30,1,2,3,4,5,6,7,8]
    
    # Check events
    events_id = events[events['device_id']==device_id]
    app_events_id = pd.merge(app_events,events_id,how='inner',left_index=True, right_index=True)
    
    print('%d events found' % events_id.shape[0])
    print('%d apps events found' % app_events_id.shape[0])
    print('%d unique apps found' % len(app_events_id.app_id.unique()))
    
    # Events per day
    events_h = events_id.groupby(['timestamp_h']).timestamp.count()
    events_d = events_id.groupby(['timestamp_d']).timestamp.count()    
    
    print('Events per hour min: %d, max: %d, mean: %d, median: %d' % (events_h.min(), events_h.max(), events_h.mean(), events_h.median()))
    print('Events per day min: %d, max: %d, mean: %d, median: %d' % (events_d.min(), events_d.max(), events_d.mean(), events_d.median()))
   
    events_h_n = norm(events_h) 
    events_d_n = norm(events_d)
    
    # Make equal axis
    events_h = events_h.reindex(hrange).fillna(0)
    events_d = events_d.reindex(drange).fillna(0)
    events_h_n = events_h_n.reindex(hrange).fillna(0)
    events_d_n = events_d_n.reindex(drange).fillna(0)
    
    # Check installed apps: 
    #13s
#    a = time.time()
#    app_events_id = app_events[app_events.isin({'event_id':events_id.index}).event_id==True]
#    print time.time()-a
    
    # 0.06s
    apps_i_h = app_events_id.groupby(['timestamp_h']).is_installed.sum()
    apps_i_d = app_events_id.groupby(['timestamp_d']).is_installed.sum()
    apps_a_h = app_events_id.groupby(['timestamp_h']).is_active.sum()
    apps_a_d = app_events_id.groupby(['timestamp_d']).is_active.sum()
    
    print('Installed apps per hour min: %d, max: %d, mean: %f, median: %f' % (apps_i_h.min(), apps_i_h.max(), apps_i_h.mean(), apps_i_h.median()))
    print('Installed apps per day min: %d, max: %d, mean: %f, median: %f' % (apps_i_d.min(), apps_i_d.max(), apps_i_d.mean(), apps_i_d.median()))
    print('Active apps per hour min: %d, max: %d, mean: %f, median: %f' % (apps_a_h.min(), apps_a_h.max(), apps_a_h.mean(), apps_a_h.median()))
    print('Active apps per day min: %d, max: %d, mean: %f, median: %f' % (apps_a_d.min(), apps_a_d.max(), apps_a_d.mean(), apps_a_d.median()))
    
    # normalized
    apps_i_h_n = norm(apps_i_h)
    apps_i_d_n = norm(apps_i_d)
    apps_a_h_n = norm(apps_a_h)
    apps_a_d_n = norm(apps_a_d)
    
    # reindex to make equal axis for all
    apps_i_h = apps_i_h.reindex(hrange).fillna(0)
    apps_i_d = apps_i_d.reindex(drange).fillna(0)
    apps_a_h = apps_a_h.reindex(hrange).fillna(0)
    apps_a_d = apps_a_d.reindex(drange).fillna(0)
    apps_i_h_n = apps_i_h_n.reindex(hrange).fillna(0)
    apps_i_d_n = apps_i_d_n.reindex(drange).fillna(0)
    apps_a_h_n = apps_a_h_n.reindex(hrange).fillna(0)
    apps_a_d_n = apps_a_d_n.reindex(drange).fillna(0)
    
    # apps per events
    apps_per_event = app_events_id.groupby(app_events_id.index).is_installed.count().sort_values(ascending=False)
    
    newfigure('Apps per event for %s' % device_id)
    plot_bar(apps_per_event)
    plt.ylabel('Apps')
    plt.xlabel('Event id')
    plt.show()
    
    
    
    newfigure('App usage per hour for %s' % device_id)
    plot_bar(events_h,apps_i_h,apps_a_h, labels=['Events','Installed apps','Active apps'])
    plt.ylabel('Events / apps count')
    plt.show()
    
    newfigure('App usage per day for %s' % device_id, loc='bottomleft')
    plot_bar(events_d,apps_i_d,apps_a_d, labels=['Events','Installed apps','Active apps'])
    plt.ylabel('Events / apps count')
    plt.show()
    
    newfigure('App usage per day for %s' % device_id, loc='topright')
    plot_bar(events_h_n,apps_i_h_n,apps_a_h_n, labels=['Events','Installed apps','Active apps'])
    plt.ylabel('Events / apps count')
    plt.show()
    
    newfigure('App usage per day for %s' % device_id, loc='bottomright')
    plot_bar(events_d_n,apps_i_d_n,apps_a_d_n, labels=['Events','Installed apps','Active apps'])
    plt.ylabel('Events / apps count')
    plt.show()
    
    #for event_id in events_id:

def show_random_items(event_data, app_event_data):
    
    while True:
        device_id = random.choice(event_data['device_id'])
        
        show_app_usage(device_id, event_data, app_event_data)
        plt.show()
        plt.pause(1)

        i = raw_input("Press enter to show next (or Enter 'q' to quit): ")
        
        if i=='q':
            break
        
    print("Done")
    
    
if __name__ == "__main__":
    
    dir_source = './data_ori/'
    dir_train = './data_train/'
    dir_test = './data_test/'
    dir_val = './data_val/'
    
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    c = {'event_id': np.int64,'device_id': np.str,'timestamp': np.object, 'longitude': np.float, 'latitude':np.float}
    
    # Event ID is unique integer
    print('Reading events...')
    a=pd.read_csv(dir_source + 'events.csv', dtype=c, parse_dates=['timestamp'], date_parser=dateparse, index_col='event_id')
    
    dfe = unpack_date(a, 'timestamp')
    
    # Load app_events
    print('Reading app events...')
    dfa = pd.read_csv(dir_source + 'app_events.csv', index_col='event_id')
    
    show_random_items(dfe,dfa)
    '''
    # I want to end with
    # device id
    # day1 : number of apps installed
    # day2 : number of apps installed
    # ...
    
    # Step 1:
    # Get number of apps installed per event_id from app_events
    app_count = dfa.groupby('event_id').app_id.count().to_frame('app_count')
    # 1488096 different counts (same number as unique event_id)
    # sum(app_count) = 32473067 = dfa.shape = 32473067
    
    
    # Step 2:
    # Merge with events table
    dfe = dfe.merge(app_count,how='left',left_index=True, right_index=True)
    dfe.fillna(0, inplace=True)
    app_count_h = dfe.groupby(['device_id','timestamp_h']).app_count.mean().to_frame('app_count')
    app_count_d = dfe.groupby(['device_id','timestamp_d']).app_count.mean().to_frame('app_count')
    # sum(dfe['app_count'])==dfa.shape[0] # True
    # len(app_count_h): 510645 = unique device & hour combinations
    # len(dfe['device_id'].unique()) = 60685 unique devices
    
    # Step 3:
    # transform into columns
    df = pd.DataFrame(index=dfe['device_id'].unique())
    df = split_date(df, dfe['timestamp'])
    
    for r in app_count_h.iterrows():
        df.loc[r[0][0]]['h%d' %r[0][1]] = r[1][0]
    for r in app_count_d.iterrows():
        df.loc[r[0][0]]['d%d' %r[0][1]] = r[1][0]
        
    # Save
    df.to_csv('device_apps_per_timeunit.csv')
    '''
    
    