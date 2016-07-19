# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:39:56 2016

@author: joostbloom
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ml_toolbox.plot import newfigure
    
def inspect_events_per_device():
    
    b = a.device_id.value_counts()
    
    newfigure('Rows per event')
    plt.hist(b[0:],bins=100)
    plt.xlabel('Events per device')
    plt.ylabel('Count')
    plt.grid()
    
    print "Events per device (n, max, min, median): %d, %d, %d, %d" % (len(b),min(b),max(b),np.median(b))

def inspect_time_stamp():
    d=pd.DatetimeIndex(a.timestamp)
     
    # Per year
    print a.groupby([d.year]).timestamp.count()
    
    # Per month
    print a.groupby([d.month]).timestamp.count()
    
    # Per day
    print a.groupby([d.day]).timestamp.count()
    
    # Per hour
    print a.groupby([d.hour]).timestamp.count()

def simple_inspection():
    a.head()
    a.describe()
    
    print "Rows: %d" % len(a)
    
    # event_id
    print "event_id unique: %d" % len(a.event_id.unique())
    
    # Timestamp
    
    # Timestamp
    ts = a['timestamp']
    print "Timestamp (n, max, min, mean): %d, %s, %s," % (len(ts.unique()), max(ts), min(ts))
    
    # Lat
    lon = a['longitude']
    print "Lon (n, max, min, mean): %d, %s, %s, %s" % (len(lon.unique()), max(lon), min(lon), np.mean(lon))
    
    # Lon
    lat = a['latitude']
    print "Lat (n, max, min, mean): %d, %s, %s, %s" % (len(lat.unique()), max(lat), min(lat), np.mean(lat))

if __name__=="__main__":
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    c = {'event_id': np.int64,'devide_id': np.int64,'timestamp': np.object, 'longitude': np.float, 'latitude':np.float}
    a=pd.read_csv('./data_ori/events.csv', dtype=c, parse_dates=['timestamp'], date_parser=dateparse)
    
    
    # Check missing values
    simple_inspection()
    
    #inspect_time_stamp()
    
    #inspect_group()