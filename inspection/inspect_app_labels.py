# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 19:56:21 2016

@author: guusjeschouten
"""
import pandas as pd
import matplotlib.pyplot as plt

from ml_toolbox.plot import newfigure

def labels_per_app():
    
    b = a.label_id.value_counts()
    print b
    newfigure('Rows per event')
    plt.hist(b[0:],bins=100)
    plt.xlabel('Events per device')
    plt.ylabel('Count')
    plt.grid()
    
    print "Labels per app (n, max, min, median): %d, %d, %d, %d" % (len(b),min(b),max(b),np.median(b))
    
if __name__=='__main__':
    a=pd.read_csv('./data_ori/app_labels.csv')
    
    print(len(a))
    print(len(a['app_id'].unique()))
    print(len(a['label_id'].unique()))
    
    
    # Output:
    # 930
    # 930
    # 836
    # So some label_ids have the same category label
    # TO-DO: What to do with this?
    
    labels_per_app()