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


# --------------------
# - Cases
# --------------------

def basic_analysis():
    
    # Analyze labels
    print('Unique label_ids: %d' % app_cat.label_id.nunique())
    print('Unique category: %d' % app_cat.category.nunique())
    # 930 unique labels
    # 835 unique categories
    # 95 categories are not unique
    print('Categories more than one:')
    print(app_cat.groupby('category').size().sort_values(ascending=False).head(61))
    # There are 60 categories that occur more than once
    # Can we group them together? -> yes they do not have an ambiguous meaning 
    # (all unknowns are really unknown)
    
    # Analyse app_ids
    print('Unique app_id: %d' % app_labels['app_id'].nunique()) 
    
    # Show top 10
    print('Top 10 categories:')
    topcats = app_labels.groupby('category').size().sort_values(ascending=False)
    print topcats.head(10)
    # Top 10 has between 10k and 57k count of apps
    
    print('Top 10 apps with most categories')    
    topapps = app_labels.groupby('app_id').size().sort_values(ascending=False)
    print topapps.head(10)
    # Top 10 has between 18 and 26 categories
    
    # Number of categories without apps    
    print('Number of unique categories with apps: %d' % app_labels['category'].nunique())
    print('Number of categories with apps: %d' % (app_cat['category'].nunique()-app_labels['category'].nunique()))
    # So 473 different categories, so 362 categories have no associated apps
    
    # Check number of columns (should be equal to number of unique categories + one for app_id)
    print('Correct number of columns: %s' % (appcats_sparse.shape[1] == app_labels['category'].nunique() + 1))
    
    # Check number of rows (should be equal to number of unique apps)
    print('Correct number of rows: %s' % (appcats_sparse.shape[0] == app_labels['app_id'].nunique()))

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
    
if __name__ == "__main__":
    
    dir_source = './data_ori/'
    dir_out = './data/'
    
    
    # Load app_events
    print('Reading app categories...')
    app_cat = pd.read_csv(dir_source + 'label_categories.csv')
    
    # Event ID is unique integer
    print('Reading app labels...')
    app_labels=pd.read_csv(dir_source + 'app_labels.csv')
    
    app_labels = app_labels.merge(app_cat, how='left', on='label_id')
    app_labels['ones'] = 1
    
    appcats_sparse = app_labels.pivot_table(values='ones',columns='category',index=['app_id']).to_sparse()
    
    appcats_sparse.fillna(0,inplace=True)
    #appcats_sparse.reset_index(level=0, inplace=True)
    
    #basic_analysis()
    
    # Merge with app_events
    print('Reading app events...')
    #app_events = pd.read_csv(dir_source + 'app_events.csv')
    app_events = pd.read_csv(dir_source + 'app_events.csv')
    
    print('Reading events...')
    events = pd.read_csv(dir_source + 'events.csv')
    
    print('Shape app_events before merge: ' + str(app_events.shape))
    print('Shape events before merge: ' + str(events.shape))
    app_events = pd.merge(app_events,events,how='inner',on='event_id')
    print('Shape app_events after merge: ' + str(app_events.shape))
    
    # Grouped
    app_per_device = app_events.groupby(['device_id','app_id']).size().to_frame('app counts').reset_index(level=0)
    
    # This merge works!!
    app_cat_per_device = pd.merge(app_per_device, appcats_sparse, how='left', left_index=True, right_index=True).groupby('device_id').sum()
    app_cat_per_device.to_csv(dir_out + 'device_app_cats.csv')
    
    print('Reading device info...')
    train = pd.read_csv(dir_source + 'gender_age_train.csv')
    
    print('Shape app_cat_per_device before merge: ' + str(app_cat_per_device.shape))
    print('Shape train before merge: ' + str(train.shape))
    train = pd.merge(train,app_cat_per_device,how='inner',left_on='device_id', right_index=True)
    print('Shape train after merge: ' + str(train.shape))
    
    # Now we are finally ready to do analysis
    print train.groupby('gender').size()
    # F: 8039
    # M: 15251
    print train.groupby('gender')['app counts'].mean()
    # Men have higher average app count
    
    gr = train.groupby('group').mean()
    
    newfigure('Average number of apps per age group')
    plot_bar(gr[0:6],gr[6:12],labels=['Female','Male'])
    plt.grid()
    plt.ylabel('Average total app count')
    plt.tight_layout()
    
    col = random.choice(gr.columns)
    newfigure('Category ' + col)
    plot_bar(gr[0:6][col],gr[6:12][col],labels=['Female','Male'])
    plt.ylabel('Average total app count')
    plt.tight_layout()
    plt.grid()
    
    
    
