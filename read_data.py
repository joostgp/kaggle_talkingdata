# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:39:56 2016

@author: joostbloom
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ml_toolbox.plot import newfigure


def pre_process():
    a.gender[a.gender=='M'] == 1
    a.gender[a.gender=='F'] == 0
    
def inspect_device_id():
    newfigure("device_id")
    plt.scatter()
    

def simple_inspection():
    print a.head()
    
    print "Rows: %d" % len(a)
    
    # Gender
    nm = sum(a['gender']=='M')
    nf = sum(a['gender']=='F')
    print "Male: %d" % nm
    print "Female: %d" % nf
    print "Check: %s" % (len(a)==nm+nf)
    
    # Age
    age = a['age']
    print "Ages (n, max, min, mean): %d, %d, %d, %d" % (len(age.unique()), max(age), min(age), np.mean(age))
    
    # Group
    group = a['age'].unique()
    print "Group: %d uniques" % len(group)

if __name__=="__main__":
    a=pd.read_csv('./data_ori/gender_age_train.csv')
    
    # Check missing values
    simple_inspection()