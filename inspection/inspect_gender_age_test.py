# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:39:56 2016

@author: joostbloom
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ml_toolbox.plot import newfigure


    
def inspect_device_id():
    b=a.sample(1e4)
    
    newfigure("device_id per age")
    plt.scatter(x=b.index, y=b.device_id,c='b', marker='.')
    plt.xlabel('Index')
    plt.ylabel('Device ID')
    
    newfigure('device id')
    plt.hist(a.device_id,bins=30)
    plt.xlabel('Device id')
    plt.ylabel('Count')
    plt.grid()


if __name__=="__main__":
    a=pd.read_csv('./data_ori/gender_age_test.csv')
    
    print len(a)
    print len(a.device_id.unique())
    
    inspect_device_id()