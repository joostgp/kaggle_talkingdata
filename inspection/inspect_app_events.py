# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:39:56 2016

@author: joostbloom
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ml_toolbox.plot import newfigure
    
def inspect_events_per_app():
    
    b = a.app_id.value_counts()
    
    newfigure('Events per app')
    plt.hist(b[0:],bins=100)
    plt.xlabel('Apps per event')
    plt.ylabel('Count')
    plt.grid()
    
    print "Events per app (n, min, max, median): %d, %d, %d, %d" % (len(b),min(b),max(b),np.median(b))
def inspect_apps_per_event():
    
    b = a.event_id.value_counts()
    
    newfigure('Events per app')
    plt.hist(b[0:],bins=100)
    plt.xlabel('Apps per event')
    plt.ylabel('Count')
    plt.grid()
    
    print "Events per app (n, min, max, median): %d, %d, %d, %d" % (len(b),min(b),max(b),np.median(b))

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
    print len(a.event_id.unique())
    print len(a.app_id.unique())
    print np.mean(a.is_installed)
    print np.mean(a.is_active)
    print a.is_active.value_counts()
    
#    Output:
#    1488096
#    19237
#    1.0
#    0.392109436414
if __name__=="__main__":
    a=pd.read_csv('./data_ori/app_events.csv')
    
    a.head()
    
    
    # Check missing values
    #simple_inspection()
    
    inspect_apps_per_event()
    
    #inspect_group()