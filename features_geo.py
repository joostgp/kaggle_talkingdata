# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 16:19:02 2016

@author: joostbloom
"""
import math
import random
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def age_group(sex, age):
    # Convert age column to age group
    #ageGroupsF = ['23-','24-26','27-28','29-32','33-42,''43+']
    #ageGroupsM = ['22-','23-26','27-28','29-31','32-38,''39+']

    if sex not in ['M','F']:
        ValueError('%s is not a valid gender' % sex)
        
    if age not in range(100):
        ValueError('%s is not a valid age' % age)
    
    if sex=="M":
        if age<=22:
            g = 0
        elif age<=26:
            g = 1
        elif age<=28:
            g = 2
        elif age<=31:
            g = 3
        elif age<=38:
            g = 4
        else:
            g = 5
    elif sex=="F":
        if age<=23:
            g = 0
        elif age<=26:
            g = 1
        elif age<=28:
            g = 2
        elif age<=32:
            g = 3
        elif age<=42:
            g = 4
        else:
            g = 5
    
    return g
def map_attempt():
    
    idx_beijing = (train["longitude"]>=lon_min) &\
                  (train["longitude"]<=lon_max) &\
                  (train["latitude"]>=lat_min) &\
                  (train["latitude"]<=lat_max)

    events_beijing = train[idx_beijing]
    
    # Smooth coordinates
    events_beijing.loc[:,"lon_round"] = events_beijing["longitude"].round(decimals=2)
    events_beijing.loc[:,"lat_round"] = events_beijing["latitude"].round(decimals=2)
    
    # Get gridmap for percentage of males  
    events_beijing.loc[events_beijing.gender=='M','gender']=1.0
    events_beijing.loc[events_beijing.gender=='F','gender']=0.0
    events_beijing['gender'] = events_beijing['gender'].astype(float)
    
    beijing_male = pd.pivot_table(events_beijing,\
                        values="gender",\
                        index="lon_round",\
                        columns="lat_round",\
                        aggfunc=np.sum).astype(float)
    beijing_cnt = pd.pivot_table(events_beijing,\
                        values="gender",\
                        index="lon_round",\
                        columns="lat_round",\
                        aggfunc="count")
    beijing_std = pd.pivot_table(events_beijing,\
                        values="gender",\
                        index="lon_round",\
                        columns="lat_round",\
                        aggfunc=np.std).astype(float)
    beijing_male = beijing_male / beijing_cnt
    beijing_male.fillna(-1,inplace=True)
    
    # Create map
    lons = beijing_male.index.values
    lats = beijing_male.columns.values
    x, y = np.meshgrid(lons, lats) 
    data_values = beijing_male.values.T
    
    xnew = np.arange(100*lon_min, 100*lon_max, 1).astype(float)/100
    ynew = np.arange(100*lat_min, 100*lat_max, 1).astype(float)/100
    from matplotlib.mlab import griddata
    z_gender = griddata(x.ravel(), y.ravel(), data_values.ravel(), xnew, ynew, interp='linear')
    z_gender_cnt = griddata(x.ravel(), y.ravel(), beijing_cnt.values.T.ravel(), xnew, ynew, interp='linear')
    z_gender_std = griddata(x.ravel(), y.ravel(), beijing_std.values.T.ravel(), xnew, ynew, interp='linear')
    
    beijing_gender = pd.DataFrame(z_gender, columns=xnew, index=ynew).fillna(-1).round(decimals=3)
    
    beijing_age = pd.pivot_table(events_beijing,\
                        values="age_group",\
                        index="lon_round",\
                        columns="lat_round",\
                        aggfunc=np.mean).astype(float)
    z_age = griddata(x.ravel(), y.ravel(), beijing_age.values.T.ravel(), xnew, ynew, interp='linear')
    
    beijing_age_std = pd.pivot_table(events_beijing,\
                        values="age_group",\
                        index="lon_round",\
                        columns="lat_round",\
                        aggfunc=np.std).astype(float)
    z_age_std = griddata(x.ravel(), y.ravel(), beijing_age_std.values.T.ravel(), xnew, ynew, interp='linear')
    
    # Plot age and gender
    
    #plt.colorbar()
    
    print('Validating data...')
    
    train_gt0 = train[(train.longitude>0) & (train.latitude>0)]
    train_beijing = train[ (train.longitude>=lon_min) & (train.longitude<=lon_max) & (train.latitude>=lat_min) & (train.latitude<=lat_max)]
    print('Train devices with events: %d' % train.groupby('device_id').size().shape[0])
    print('Train device with events and long>0: %d' % train[train.longitude>0].groupby('device_id').size().shape[0])
    print('Train device with events and long>0 & lat >0: %d' % train_gt0.groupby('device_id').size().shape[0])
    print('Train device with events in Beijing: %d' % train_beijing.groupby('device_id').size().shape[0])
    
    #feat_geo = train_gt0.groupby(['device_id'])[['latitude','longitude']].std()
    train_beijing['age_from_map'] = train_beijing[['longitude','latitude']].apply(lambda x: age_from_map(x[0],x[1], xnew, ynew, z_age), axis=1)
    
    n=7
    lat = round(train_beijing.iloc[n].latitude,2)
    lon = round(train_beijing.iloc[n].longitude,2)
    ix = np.where(xnew==lon)[0][0]
    iy = np.where(ynew==lat)[0][0]
    z_gender[iy, ix]
    
def age_from_map(lon,lat, x,y,z,decimals=2):
    lat = round(lat, decimals)
    lon = round(lon, decimals)
    
    #print lon,lat
    
    ix = np.where(x==lon)[0]
    iy = np.where(y==lat)[0]
    
    if len(ix)==0 or len(iy)==0:
        return None
    
    ix = ix[0]
    iy = iy[0]
    
    
    return z[iy,ix]  
if __name__ == "__main__":
    
    dir_in = './data_ori/'
    dir_out = './data/'
    feature_file = 'features_geo.csv'
    rs = 123
    
    print('Reading data...')
    train = pd.read_csv(dir_in + 'gender_age_train.csv')
    #test  =pd.read_csv(dir_in + 'gender_age_test.csv')    
    events = pd.read_csv(dir_in + 'events.csv')
    
    
    print('Merging data...')
    train.loc[:,'age_group'] = train[['gender','age']].apply(lambda x: age_group(x[0],x[1]), axis=1)
    train = pd.merge(train, events, how='inner', on='device_id')
    #test = pd.merge(test, events, how='inner', on='device_id')

    # Get events in Beijing area
    # From http://www.tageo.com/index-e-ch-cities-CN.htm
    print('Calculating features...')
    gps_cities = {}
    gps_cities['beijing'] = (39.93, 116.4) #
    gps_cities['wuhan'] = (30.58, 114.27) #
    gps_cities['chengdu'] = (30.670, 104.07)#
    gps_cities['tianjin'] = (39.130, 117.20)#
    gps_cities['shenyang'] = (41.800, 123.450)#
    gps_cities['xian'] = (34.270, 108.900) #
    gps_cities['guangzhou'] = (23.120, 113.250) # Canton
    gps_cities['harbin'] = (45.750, 126.650)
    gps_cities['chongqing'] = (29.57, 106.58) #
    gps_cities['hongkong'] = (22.20, 114.10) #
    gps_cities['nanjing'] = (32.050, 118.78)
    gps_cities['shanghai'] = (31.230, 121.470) #
    
    events_0 = events[(events.longitude>0) & (events.latitude>0)].groupby('device_id', as_index=False).first()    
    
    feat_geo = pd.DataFrame()    
    feat_geo['device_id'] = events_0['device_id']
    distance_for_close = 1.1

    for k in gps_cities.keys():
        v = gps_cities[k]
        feat_geo['dist_' + k] = events_0[['longitude','latitude']].apply(lambda x: math.sqrt( (x[0]-v[1])**2 + (x[1]-v[0])**2 ), axis=1)
        feat_geo['close_to_' + k] = (feat_geo['dist_' + k] < distance_for_close).astype(int)
    
    # Longitude makes more sense than latitude
    feat_geo['travel_dist'] = events[(events.longitude>0) & (events.latitude>0)] \
                                  .groupby('device_id')\
                                  ['longitude','latitude'] \
                                  .std().reset_index(drop=True).max(axis=1)
    
    # Couple of elements of travel_dist have NaN: find out why
    feat_geo.fillna(0, inplace=True)                           
    feat_geo.to_csv(dir_out + feature_file, index=False)
    print('Saved features to: ' + dir_out + feature_file)
    
    