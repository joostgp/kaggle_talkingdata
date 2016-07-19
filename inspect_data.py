# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:39:56 2016

@author: joostbloom
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------
# - Cases
# ---------------------

def inspect_gender_age():
    
    # To-do: inspect device-id
    
    print("Inspect table GENDER_AGE_TRAIN...")
    print(gender_age.head(5))
    print("")
    print(describe(gender_age))
    print("")
    print("Value counts age:")
    print(unique_values(gender_age, col='age', outputtype = 'str_abs'))
    print("")
    print("Value counts gender:")
    print(unique_values(gender_age, col='gender', outputtype = 'str_rel'))
    print("")
    print("Value counts group:")
    print(unique_values(gender_age, col='group', outputtype = 'str_rel'))
    print("")
    plot_distribution(gender_age,col='group',split_col='gender')
 
def inspect_test():
    print("Inspect table GENDER_AGE_TEST...")
    print(test.head(5))
    print("Rows: %d" % test.shape[0])
    print("Unique device_id values: %d" % len(test.device_id.unique()))
   
def inspect_events():
    print("Inspect table EVENTS...")
    print(events.head(5))
    print("")
    print(describe(events))
    
    d=pd.DatetimeIndex(events.timestamp)
    print("Value counts timestamp:")
    print("Per year:")
    print(unique_values(events,'timestamp',col_group=d.year,outputtype='str_rel'))
    print("Per month:")
    print(unique_values(events,'timestamp',col_group=d.month,outputtype='str_rel'))
    print("Per day:")
    print(unique_values(events,'timestamp',col_group=d.day,outputtype='str_rel'))
    print("Per hour:")
    print(unique_values(events,'timestamp',col_group=d.hour,outputtype='str_rel'))
    print("")
    
    print("Rows: %d" % events.shape[0])
    print("Unique event_id values: %d" % len(events.event_id.unique()))
    print("Unique device_id values: %d" % len(events.device_id.unique()))
    print("Device_id: %s " % unique_values(events, 'device_id', outputtype="analysis"))
    print("Device_id Top10: %s" % unique_values(events, 'device_id', outputtype="str_abs", top=10))
    
def inspect_brands():
    print("Inspect PHONE_BRAND_DEVICE_MODEL...")
    print(brands.head())
    print("")
    print(describe(brands))
    print("")
    
    print("Rows: %d" % brands.shape[0])
    print("Unique device_id values: %d" % len(brands.device_id.unique()))
    print("Device_id Top10: %s" % unique_values(brands, 'device_id', outputtype="str_abs", top=10))
    print("")
    print("Phone brands:")
    print(unique_values(brands, col='phone_brand', outputtype='analysis'))
    print(unique_values(brands, col='phone_brand'))
    print("Device models:")
    print(unique_values(brands, col='device_model', outputtype='analysis'))
    print(unique_values(brands, col='device_model', top=25))
    
def inspect_app_events():
    print("Inspect APP_EVENTS...")
    print(app_events.head())
    print("")
    print(describe(app_events))
    print("")
    
    print("Rows: %d" % app_events.shape[0])
    print("event_id: %s" % unique_values(app_events, col='event_id', outputtype='analysis'))
    print("app_id: %s" % unique_values(app_events, col='app_id', outputtype='analysis'))
    print("app_id top 25: %s" % unique_values(app_events, col='app_id', top=25))
    
def inspect_app_labels():
    print("Inspect APP_LABELS...")
    print(app_labels.head())
    print("")
    print(describe(app_labels))
    print("")
    
    print("Rows: %d" % app_labels.shape[0])
    print("app_id: %s" % unique_values(app_labels, col='app_id', outputtype='analysis'))
    print("label_id: %s" % unique_values(app_labels, col='label_id', outputtype='analysis'))
    
def inspect_label_categories():
    print("Inspect LABEL_CATEGORIES...")
    print(app_cats.head())
    print("")
    print(describe(app_cats))
    print("")
    
    print("Rows: %d" % app_cats.shape[0])
    print("label_id: %s" % unique_values(app_cats, col='label_id', outputtype='analysis'))
    print("category: %s" % unique_values(app_cats, col='category', outputtype='analysis'))
    print("category top-25: %s" % unique_values(app_cats, col='category', top=25, outputtype='str_abs'))
    

# ---------------------
# - Support functions
# ---------------------

def describe(df):
    des = df.describe(include='all')
    des.drop('25%',inplace=True)
    des.drop('50%',inplace=True)
    des.drop('75%',inplace=True)
    des.loc['nans'] = np.sum(df.isnull())
    
    return des
   
def unique_values(df, col, outputtype='str_rel', col_group=None, top=None):
    
    if col_group is None: col_group = col
    
    
    counts = df.groupby([col_group])[col].count().sort_values(ascending=False)
    
    if top > 0:
        freq_other = sum(counts[top:])
        counts = counts[:top]
        counts['Other'] = freq_other

    if outputtype == 'series':
        return counts
    elif outputtype == "str_abs":
        return ', '.join([str(i)+' (' + str(v) + ')' for (i,v) in counts.iteritems()])
    elif outputtype == 'str_rel':
        return ', '.join([str(i)+' (' + '{:.0f}%'.format(float(v)/len(df)*100) + ')' for (i,v) in counts.iteritems()])
    elif outputtype == "analysis":
        return '%d unique values, freq. min: %d (%s), freq. max: %d (%s), freq. median: %d' % (len(counts), counts.min(),counts.idxmin(), counts.max(), counts.idxmax(), np.median(counts) )

def check_relation(df1, df2, col_key1, col_key2=None):
    
    if col_key2 is None: col_key2 = col_key1
    
    print("Number of rows in 1: %d" % df1.shape[0])
    print("Number of rows in 2: %d" % df2.shape[0])
    print("Unique keys in 1: %d" % len(set(df1[col_key1])))
    print("Unique keys in 2: %d" % len(set(df2[col_key2])))
    print("Key coverage 1 by 2: %f" % ( float(len(set(df1[col_key1]) & set(df2[col_key2]))) / len(set(df1[col_key1])) ))
    print("Key coverage 2 by 1: %f" % ( float(len(set(df1[col_key1]) & set(df2[col_key2]))) / len(set(df2[col_key2])) ))
    
    # To calculate row coverage, do a count per key first
    count1 = df1.groupby([col_key1])[col_key1].count()
    count2 = df2.groupby([col_key2])[col_key2].count()
    sum_count1 = float( count1.loc[set(df2[col_key2])].sum() )
    sum_count2 = float( count2.loc[set(df1[col_key1])].sum() )
    print("Row coverage 1 by 2: %f" % ( sum_count1 / df1.shape[0] ))
    print("Row coverage 2 by 1: %f" % ( sum_count2 / df2.shape[0] ))
        
        
def plot_distribution(df, col, split_col):
    # REMARK: ONLY WORKS FOR SPLIT_COL HAVING TWO UNIQUE VALUES
    
    u_s = df[split_col].unique()    
    
    df0 = df[df[split_col]==u_s[0]]
    df1 = df[df[split_col]==u_s[1]]
    
    col_0 = sorted(df0[col].unique())
    col_1 = sorted(df1[col].unique())
    
    cnt_df0 = df0[col].value_counts()
    cnt_cf1 = df1[col].value_counts()
    
    
    width = 0.4
    indm = np.arange(len(col_0))
    indf = np.arange(len(col_1))
    plt.figure()
    plt.title('Distribution of ' + str(col) + ' by ' + str(split_col))
    plt.bar(indm+width/2, cnt_df0, width, color='b')
    plt.bar(indf+1.5*width, cnt_cf1, width, color='r')
    plt.xlim([0,6+width/2])
    plt.xticks(np.concatenate([indm+width,indf+2*width]),np.concatenate([col_0,col_1]),rotation='vertical')
    plt.tight_layout()
    plt.legend([str(x) for x in u_s])
    plt.grid()
    
    ax = plt.gca()
    rects = ax.patches
    scores = np.concatenate([cnt_df0,cnt_cf1]).astype(float)/len(df)*100
    for rect, label in zip(rects, scores):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height, "{:.0f}%".format(label), ha='center', va='bottom')

        
if __name__=="__main__":
    pd.options.display.float_format = '{:.2f}'.format
    datadir = '../input/'
        
    print('Read train...')
    c = {'decice_id': np.str, 'gender': np.str, 'age': np.int, 'group': np.str}
    gender_age = pd.read_csv(datadir + 'gender_age_train.csv', dtype=c)
    
    inspect_gender_age()
    
    print('Read test...')
    c = {'decice_id': np.str}
    test = pd.read_csv(datadir + 'gender_age_test.csv', dtype=c)
    
    inspect_test()
    
    print("Reading events...")
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    c = {'event_id': np.int64,'devide_id': np.str,'timestamp': np.object, 'longitude': np.float, 'latitude':np.float}
    events=pd.read_csv(datadir + 'events.csv', dtype=c, parse_dates=['timestamp'], date_parser=dateparse)
    
    inspect_events()
        
    print('Read brands...')
    c = {'decice_id': np.str, 'phone_brand': np.str, 'device_model': np.str}
    brands = pd.read_csv(datadir + 'phone_brand_device_model.csv', dtype=c)
    
    inspect_brands()
    
    print('Read app events...')
    c = {'event_id': np.int64, 'app_id': np.int64, 'is_installed': np.int, 'is_active': np.int}
    app_events = pd.read_csv(datadir + 'app_events.csv', dtype=c)
    
    inspect_app_events()
    
    print('Read app labels...')
    c = {'label_id': np.int64, 'app_id': np.int64}
    app_labels = pd.read_csv(datadir + 'app_labels.csv', dtype=c)
    
    inspect_app_labels()
    
    print('Read label categories...')
    c = {'label_id': np.int64, 'category': np.str}
    app_cats = pd.read_csv(datadir + 'label_categories.csv', dtype=c)
    app_cats.category.fillna('unknown',inplace=True)
    
    inspect_label_categories()   
    
    # Check relations
    print('Checking relation GENDER_AGE and BRANDS')
    check_relation(gender_age, brands, 'device_id')
    
    print('Checking relation TEST and BRANDS')
    check_relation(test, brands, 'device_id')
    
    print('Checking relation GENDER_AGE and EVENTS')
    check_relation(gender_age, events, 'device_id')
    
    print('Checking relation TEST and EVENTS')
    check_relation(test, events, 'device_id')
    
    print('Checking relation EVENTS and APP_EVENTS')
    check_relation(events, app_events, 'event_id')
    
    print('Checking relation APP_EVENTS and APP_LABELS')
    check_relation(app_events, app_labels, 'app_id')
    
    print('Checking relation APP_LABELS and LABEL_CATEGORIES')
    check_relation(app_labels, app_cats, 'label_id')
