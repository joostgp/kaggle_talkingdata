# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 22:12:32 2016

@author: joostbloom
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ml_toolbox.plot import newfigure

# --------------------
# - Support functions 
# --------------------

def get_brand_trans_dict():
    original_names = ["三星","天语","海信","联想","欧比","爱派尔","努比亚","优米","朵唯","黑米","锤子","酷比魔方","美图","尼比鲁","一加","优购","诺基亚","糖葫芦","中国移动","语信","基伍","青橙","华硕","夏新","维图","艾优尼","摩托罗拉","乡米","米奇","大可乐","沃普丰","神舟","摩乐","飞秒","米歌","富可视","德赛","梦米","乐视","小杨树","纽曼","邦华","E派","易派","普耐尔","欧新","西米","海尔","波导","糯米","唯米","酷珀","谷歌","昂达","聆韵","小米","华为","魅族","中兴","酷派","金立","SUGAR","OPPO","vivo","HTC","LG","ZUK","TCL","LOGO","Lovme","PPTV","ZOYE","MIL","索尼" ,"欧博信" ,"奇酷" ,"酷比" ,"康佳" ,"亿通" ,"金星数码" ,"至尊宝" ,"百立丰" ,"贝尔丰" ,"百加" ,"诺亚信" ,"广信" ,"世纪天元" ,"青葱" ,"果米" ,"斐讯" ,"长虹" ,"欧奇" ,"先锋" ,"台电" ,"大Q" ,"蓝魔" ,"奥克斯"]
    english_names = ["samsung", "Ktouch", "hisense", "lenovo", "obi", "ipair", "nubia", "youmi", "dowe", "heymi", "hammer", "koobee", "meitu", "nibilu", "oneplus", "yougo", "nokia", "candy", "ccmc", "yuxin", "kiwu", "greeno", "asus", "panosonic", "weitu", "aiyouni", "moto", "xiangmi", "micky", "bigcola", "wpf", "hasse", "mole", "fs", "mige", "fks", "desci", "mengmi", "lshi", "smallt", "newman", "banghua", "epai", "epai", "pner", "ouxin", "ximi", "haier", "bodao", "nuomi", "weimi", "kupo", "google", "ada", "lingyun", "Xiaomi", "Huawei", "Meizu", "ZTE", "Coolpad", "Gionee", "SUGAR", "OPPO", "vivo", "HTC", "LG", "ZUK", "TCL", "LOGO", "Lovme", "PPTV", "ZOYE", "MIL", "Sony", "Opssom", "Qiku", "CUBE", "Konka", "Yitong", "JXD", "Monkey King", "Hundred Li Feng", "Bifer", "Bacardi", "Noain", "Kingsun", "Ctyon", "Cong", "Taobao", "Phicomm", "Changhong", "Oukimobile", "XFPLAY", "Teclast", "Daq", "Ramos", "AUX"]

    return {v: english_names[i] for (i,v) in enumerate(original_names)}

# --------------------
# - Cases 
# --------------------

def plot_phone_brands_test_train():
    
    train_counts = train.groupby('phone_brand').phone_brand.value_counts().sort_values(ascending=False)[0:15]
    test_counts = test.groupby('phone_brand').phone_brand.value_counts().sort_values(ascending=False)[0:15]
    
    train_counts = train_counts.astype(float) / train.shape[0]
    test_counts = test_counts.astype(float) / test.shape[0]
    
    width = 0.4
    ind = np.arange(len(train_counts))
    
    newfigure('Phone brands train and test')
    
    plt.bar(ind+width/2, train_counts, width, color='b')
    plt.bar(ind+1.5*width, test_counts, width, color='r')
    plt.xlim([0,6+width/2])
    plt.xticks(np.concatenate([ind+width,ind+2*width]),np.concatenate([train_counts.index,test_counts.index]),rotation='vertical')
    #plt.xticks(,,rotation='vertical')
    plt.tight_layout()
    plt.legend(['Train','Test'])
    plt.ylabel('Relative (-)')
    plt.grid()
    
def plot_phone_brands_male_female():
    
    count1 = train[train.gender=='M'].groupby('phone_brand').phone_brand.value_counts().sort_values(ascending=False)[0:15]
    count2 = train[train.gender=='F'].groupby('phone_brand').phone_brand.value_counts().sort_values(ascending=False)[0:15]
    
    count1 = count1.astype(float) / train[train.gender=='M'].shape[0]
    count2 = count2.astype(float) / train[train.gender=='F'].shape[0]
    
    width = 0.4
    ind = np.arange(len(count1))
    
    newfigure('Phone brands per gender')
    
    plt.bar(ind+width/2, count1, width, color='b')
    plt.bar(ind+1.5*width, count2, width, color='r')
    plt.xlim([0,6+width/2])
    plt.xticks(np.concatenate([ind+width,ind+2*width]),np.concatenate([count1.index,count2.index]),rotation='vertical')
    #plt.xticks(,,rotation='vertical')
    plt.tight_layout()
    plt.legend(['Male','Female'])
    plt.ylabel('Relative (-)')
    plt.grid()
    
def plot_phone_brands_age():
    
    count1 = train[train.age>=31].groupby('phone_brand').phone_brand.value_counts().sort_values(ascending=False)[0:15]
    count2 = train[train.age<31].groupby('phone_brand').phone_brand.value_counts().sort_values(ascending=False)[0:15]
    
    count1 = count1.astype(float) / train[train.age>=31].shape[0]
    count2 = count2.astype(float) / train[train.age<31].shape[0]
    
    width = 0.4
    ind = np.arange(len(count1))
    
    newfigure('Phone brands per age')
    
    plt.bar(ind+width/2, count1, width, color='b')
    plt.bar(ind+1.5*width, count2, width, color='r')
    plt.xlim([0,6+width/2])
    plt.xticks(np.concatenate([ind+width,ind+2*width]),np.concatenate([count1.index,count2.index]),rotation='vertical')
    #plt.xticks(,,rotation='vertical')
    plt.tight_layout()
    plt.legend(['>=31','<31'])
    plt.ylabel('Relative (-)')
    plt.grid()
    
def plot_device_models_per_brand():
    
    top_brands = ['Xiaomi','samsung','Huawei','OPPO','vivo','Meizu','Coolpad','lenovo']     
    
    count1 = train[train.gender=='M'].groupby(['phone_brand','device_model']).size()
    count2 = train[train.gender=='F'].groupby(['phone_brand','device_model']).size()
    
    
    
    newfigure('Device models per gender')
    
    for (i,v) in enumerate(top_brands):
        width = 0.4
        
        count1_l = count1[v].sort_values(ascending=False)[0:12]
        count2_l = count2[v].sort_values(ascending=False)[0:12]
        
        count1_l = count1_l.astype(float) / count1[v].sum()
        count2_l = count2_l.astype(float) / count2[v].sum()
        
        ind = np.arange(len(count1_l))
        
        plt.subplot(len(top_brands),1,i+1)
    
        plt.bar(ind+width/2, count1_l, width, color='b')
        plt.bar(ind+1.5*width, count2_l, width, color='r')
        plt.legend(['Male','Female'])
        plt.title('Device models ' + v)
        plt.grid()
        plt.xticks([])
    
def plot_device_models_per_age():
    
    top_brands = ['Xiaomi','samsung','Huawei','OPPO','vivo','Meizu','Coolpad','lenovo']     
    
    count1 = train[train.age>=31].groupby(['phone_brand','device_model']).size()
    count2 = train[train.age<31].groupby(['phone_brand','device_model']).size()
    
    
    
    newfigure('Device models per age')
    
    for (i,v) in enumerate(top_brands):
        width = 0.4
        
        count1_l = count1[v].sort_values(ascending=False)[0:12]
        count2_l = count2[v].sort_values(ascending=False)[0:12]
        
        count1_l = count1_l.astype(float) / count1[v].sum()
        count2_l = count2_l.astype(float) / count2[v].sum()
        
        ind = np.arange(len(count1_l))
        
        plt.subplot(len(top_brands),1,i+1)
    
        plt.bar(ind+width/2, count1_l, width, color='b')
        plt.bar(ind+1.5*width, count2_l, width, color='r')
        plt.legend(['>=31','<31'])
        plt.title('Device models ' + v)
        plt.grid()
        plt.xticks([])
    
def plot_device_models_test_train():
    
    top_brands = ['Xiaomi','samsung','Huawei','OPPO','vivo','Meizu','Coolpad','lenovo']     
    
    count1 = train.groupby(['phone_brand','device_model']).size()
    count2 = test.groupby(['phone_brand','device_model']).size()
    
    
    
    newfigure('Device models per age')
    
    for (i,v) in enumerate(top_brands):
        width = 0.4
        
        count1_l = count1[v].sort_values(ascending=False)[0:12]
        count2_l = count2[v].sort_values(ascending=False)[0:12]
        
        count1_l = count1_l.astype(float) / count1[v].sum()
        count2_l = count2_l.astype(float) / count2[v].sum()
        
        ind = np.arange(len(count1_l))
        
        plt.subplot(len(top_brands),1,i+1)
    
        plt.bar(ind+width/2, count1_l, width, color='b')
        plt.bar(ind+1.5*width, count2_l, width, color='r')
        plt.legend(['Train','Test'])
        plt.title('Device models ' + v)
        plt.grid()
        plt.xticks([])
    

if __name__ == '__main__':
    
    plt.close('all')
    
    inputdir = './data_ori/'
    outputdir = './analysis_brands/'
    rs = 123
    
    hdlist = ['device_id', 'F23-', 'F24-26', 'F27-28', 'F29-32', 'F33-42','F43+', 
                           'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+']        
                         
    tr_brands = get_brand_trans_dict()
            
    print('Read brands...')
    pbd = pd.read_csv(inputdir + 'phone_brand_device_model.csv', dtype={'device_id': np.str})
    pbd.drop_duplicates('device_id', keep='first', inplace=True)
    pbd['phone_brand'].replace(tr_brands, inplace=True)
   
    print('Read train...')
    train = pd.read_csv(inputdir + 'gender_age_train.csv', dtype={'device_id': np.str})
    train = pd.merge(train, pbd, how='left', on='device_id', left_index=True)
    train.fillna(-1, inplace=True)
    
    print('Read test...')
    test = pd.read_csv(inputdir + 'gender_age_test.csv', dtype={'device_id': np.str})
    test = pd.merge(test, pbd, how='left', on='device_id', left_index=True)
    test.fillna(-1, inplace=True)
    
    plot_phone_brands_test_train()
    
    plot_phone_brands_male_female()
    
    plot_phone_brands_age()
    
    plot_device_models_test_train()
    
    plot_device_models_per_age()
    
    plot_device_models_per_brand()
    
