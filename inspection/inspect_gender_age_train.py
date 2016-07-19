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
    b=a.sample(1e4)
    newfigure("device_id per gender")
    plt.scatter(x=b[b.gender=='M'].index, y=b[b.gender=='M'].device_id,c='b', marker='.')
    plt.scatter(x=b[b.gender=='F'].index, y=b[b.gender=='F'].device_id,c='r', marker='.')
    plt.legend(['Male','Female'])
    plt.xlabel('Index')
    plt.ylabel('Device ID')
    
    newfigure("device_id per age")
    plt.scatter(x=b[b.age<=31].index, y=b[b.age<=31].device_id,c='b', marker='.')
    plt.scatter(x=b[b.age>31].index, y=b[b.age>31].device_id,c='r', marker='.')
    plt.legend(['<=31','>31'])
    plt.xlabel('Index')
    plt.ylabel('Device ID')
    
    newfigure('device id')
    plt.hist(a.device_id,bins=30)
    plt.xlabel('Device id')
    plt.ylabel('Count')
    plt.grid()
    
def inspect_group():
    m = a[a.gender=='M']
    f = a[a.gender=='F']
    
    cm = sorted(m.group.unique())
    cf = sorted(f.group.unique())
    
    m.group.value_counts()
    
    
    width = 0.4
    indm = np.arange(len(cm))
    indf = np.arange(len(cf))
    newfigure('Count per age group')
    plt.bar(indm+width/2, m.group.value_counts(), width, color='b')
    plt.bar(indf+1.5*width, f.group.value_counts(), width, color='r')
    plt.xlim([0,6+width/2])
    plt.xticks(np.concatenate([indm+width,indf+2*width]),np.concatenate([cm,cf]),rotation='vertical')
    plt.tight_layout()
    plt.grid()
    #plt.xticks(indf,cf)
    

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
    group = a['group'].unique()
    print "Group: %d uniques" % len(group)

if __name__=="__main__":
    a=pd.read_csv('./data_ori/gender_age_train.csv')
    
    # Check missing values
    simple_inspection()
    
    inspect_device_id()
    
    inspect_group()