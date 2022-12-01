# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 2022

@author: celestec
"""
rando = 1
import pandas as pd
import utils
from sklearn.model_selection import train_test_split

#%% Ethnic Splitting

def ethnic_split():
    
    X,y = utils.get_XY()
    ethnic_index = utils.ethnicities()
    X_w = X.loc[ethnic_index == 'White']
    X_b = X.loc[ethnic_index == 'Black']
    X_a = X.loc[ethnic_index == 'Asian']
    X_h = X.loc[ethnic_index == 'Hispanic']
    
    y_w = y.loc[ethnic_index == 'White']
    y_b = y.loc[ethnic_index == 'Black']
    y_a = y.loc[ethnic_index == 'Asian']
    y_h = y.loc[ethnic_index == 'Hispanic']
    
    Xt_train, Xt_test, yt_train, yt_test = train_test_split(
        X, y, test_size=0.2, random_state=rando, stratify = y)
    
    m = 0
    for i in [X_w, X_b, X_a, X_h]:
        train_index = list(set(i.index).intersection(set(Xt_train.index)))
        test_index = list(set(i.index).intersection(set(Xt_test.index)))
        
        if m == 0:
            Xw_train = X_w.loc[train_index]
            yw_train = y_w.loc[train_index]
            Xw_test = X_w.loc[test_index]
            yw_test = y_w.loc[test_index]
        if m == 1:
            Xb_train = X_b.loc[train_index]
            yb_train = y_b.loc[train_index]
            Xb_test = X_b.loc[test_index]
            yb_test = y_b.loc[test_index]
        if m == 2:
            Xa_train = X_a.loc[train_index]
            ya_train = y_a.loc[train_index]
            Xa_test = X_a.loc[test_index]
            ya_test = y_a.loc[test_index]
        if m == 3:
            Xh_train = X_h.loc[train_index]
            yh_train = y_h.loc[train_index]
            Xh_test = X_h.loc[test_index]
            yh_test = y_h.loc[test_index]
            
        m += 1
        
    return Xt_train, Xt_test, yt_train, yt_test, Xw_train, Xw_test, yw_train, yw_test,Xb_train, Xb_test, yb_train, yb_test,Xa_train, Xa_test, ya_train, ya_test,Xh_train, Xh_test, yh_train, yh_test


def ethnic_test_train_spread():
    
    Xt_train, Xt_test, yt_train, yt_test, Xw_train, Xw_test, yw_train, yw_test,Xb_train, Xb_test, yb_train, yb_test,Xa_train, Xa_test, ya_train, ya_test,Xh_train, Xh_test, yh_train, yh_test = ethnic_split()
    
    Xt_test_count,_ = Xt_test.shape
    Xw_test_count,_ = Xw_test.shape
    Xb_test_count,_ = Xb_test.shape
    Xa_test_count,_ = Xa_test.shape
    Xh_test_count,_ = Xh_test.shape

    Xt_train_count,_ = Xt_train.shape
    Xw_train_count,_ = Xw_train.shape
    Xb_train_count,_ = Xb_train.shape
    Xa_train_count,_ = Xa_train.shape
    Xh_train_count,_ = Xh_train.shape

    train_spread = pd.DataFrame(index = ['Percent Representation'])
    train_spread['White'] = Xw_train_count/Xt_train_count*100
    train_spread['Black'] = Xb_train_count/Xt_train_count*100
    train_spread['Asian'] = Xa_train_count/Xt_train_count*100
    train_spread['Hispanic'] = Xh_train_count/Xt_train_count*100
    
    test_spread = pd.DataFrame(index = ['Percent Representation'])
    test_spread['White'] = Xw_test_count/Xt_test_count*100
    test_spread['Black'] = Xb_test_count/Xt_test_count*100
    test_spread['Asian'] = Xa_test_count/Xt_test_count*100
    test_spread['Hispanic'] = Xh_test_count/Xt_test_count*100
    
    
    test_label_dist = pd.DataFrame(data = {'yt': yt_test.value_counts(),
              'yw':yw_test.value_counts(),
              'yb':yb_test.value_counts(),
              'ya':ya_test.value_counts(),
              'yh':yh_test.value_counts()})
    
    train_label_dist = pd.DataFrame(data = {'yt': yt_train.value_counts(),
              'yw':yw_train.value_counts(),
              'yb':yb_train.value_counts(),
              'ya':ya_train.value_counts(),
              'yh':yh_train.value_counts()})
    
    return train_spread, test_spread, train_label_dist, test_label_dist
        
# train_spread.to_csv('Ethnic Train Data Distribution.csv')    
# test_spread.to_csv('Ethnic Test Data Distribution.csv')
# train_label_dist.to_csv('Ethnic Train Data Label Counts')
# test_label_dist.to_csv('Ethnic Test Data Label Counts')

#%% Community Group Split

def comm_group_split():
    
    X,y = utils.get_XY()
    comm_index = utils.groups()
    X_i = X.loc[comm_index == 'I']
    X_ii = X.loc[comm_index == 'II']
    X_iii = X.loc[comm_index == 'III']
    X_iv = X.loc[comm_index == 'IV']
    X_v = X.loc[comm_index == 'V']
    
    y_i = y.loc[comm_index == 'I']
    y_ii = y.loc[comm_index == 'II']
    y_iii = y.loc[comm_index == 'III']
    y_iv = y.loc[comm_index == 'IV']
    y_v = y.loc[comm_index == 'V']
    
    Xt_train, Xt_test, yt_train, yt_test = train_test_split(
        X, y, test_size=0.2, random_state=rando, stratify = y)
    
    m = 0
    for i in [X_i, X_ii, X_iii, X_iv, X_v]:
        train_index = list(set(i.index).intersection(set(Xt_train.index)))
        test_index = list(set(i.index).intersection(set(Xt_test.index)))
        
        if m == 0:
            Xi_train = X_i.loc[train_index]
            yi_train = y_i.loc[train_index]
            Xi_test = X_i.loc[test_index]
            yi_test = y_i.loc[test_index]
        if m == 1:
            Xii_train = X_ii.loc[train_index]
            yii_train = y_ii.loc[train_index]
            Xii_test = X_ii.loc[test_index]
            yii_test = y_ii.loc[test_index]
        if m == 2:
            Xiii_train = X_iii.loc[train_index]
            yiii_train = y_iii.loc[train_index]
            Xiii_test = X_iii.loc[test_index]
            yiii_test = y_iii.loc[test_index]
        if m == 3:
            Xiv_train = X_iv.loc[train_index]
            yiv_train = y_iv.loc[train_index]
            Xiv_test = X_iv.loc[test_index]
            yiv_test = y_iv.loc[test_index]
        if m == 4:
            Xv_train = X_v.loc[train_index]
            yv_train = y_v.loc[train_index]
            Xv_test = X_v.loc[test_index]
            yv_test = y_v.loc[test_index]
            
        m += 1
        
    return Xt_train, Xt_test, yt_train, yt_test, Xi_train, Xi_test, yi_train, yi_test,Xii_train, Xii_test, yii_train, yii_test,Xiii_train, Xiii_test, yiii_train, yiii_test,Xiv_train, Xiv_test, yiv_train, yiv_test, Xv_train, Xv_test, yv_train, yv_test 


def comm_group_test_train_spread():
    
    Xt_train, Xt_test, yt_train, yt_test, Xi_train, Xi_test, yi_train, yi_test,Xii_train, Xii_test, yii_train, yii_test,Xiii_train, Xiii_test, yiii_train, yiii_test,Xiv_train, Xiv_test, yiv_train, yiv_test, Xv_train, Xv_test, yv_train, yv_test=comm_group_split()
    
    Xt_test_count,_ = Xt_test.shape
    Xi_test_count,_ = Xi_test.shape
    Xii_test_count,_ = Xii_test.shape
    Xiii_test_count,_ = Xiii_test.shape
    Xiv_test_count,_ = Xiv_test.shape
    Xv_test_count,_ = Xv_test.shape

    Xt_train_count,_ = Xt_train.shape
    Xi_train_count,_ = Xi_train.shape
    Xii_train_count,_ = Xii_train.shape
    Xiii_train_count,_ = Xiii_train.shape
    Xiv_train_count,_ = Xiv_train.shape
    Xv_train_count,_ = Xv_train.shape

    train_spread = pd.DataFrame(index = ['Percent Representation'])
    train_spread['I'] = Xi_train_count/Xt_train_count*100
    train_spread['II'] = Xii_train_count/Xt_train_count*100
    train_spread['III'] = Xiii_train_count/Xt_train_count*100
    train_spread['IV'] = Xiv_train_count/Xt_train_count*100
    train_spread['V'] = Xv_train_count/Xt_train_count*100
    
    test_spread = pd.DataFrame(index = ['Percent Representation'])
    test_spread['I'] = Xi_test_count/Xt_test_count*100
    test_spread['II'] = Xii_test_count/Xt_test_count*100
    test_spread['III'] = Xiii_test_count/Xt_test_count*100
    test_spread['IV'] = Xiv_test_count/Xt_test_count*100
    test_spread['V'] = Xv_test_count/Xt_test_count*100

    
    test_label_dist = pd.DataFrame(data = {'yt': yt_test.value_counts(),
              'yi':yi_test.value_counts(),
              'yii':yii_test.value_counts(),
              'yiii':yiii_test.value_counts(),
              'yiv':yiv_test.value_counts(),
              'yv':yv_test.value_counts()})
    
    train_label_dist = pd.DataFrame(data = {'yt': yt_train.value_counts(),
              'yi':yi_train.value_counts(),
              'yii':yii_train.value_counts(),
              'yiii':yiii_train.value_counts(),
              'yiv':yiv_train.value_counts(),
              'yv':yv_train.value_counts()})
    
    return train_spread, test_spread, train_label_dist, test_label_dist