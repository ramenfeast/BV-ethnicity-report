# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 2022

@author: celestec
"""
rando = 1
import pandas as pd
import utils
from sklearn.model_selection import train_test_split

X,y = utils.get_XY()

def ethnic_split():
    
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
            
        m = m+1
    return(Xt_train, Xt_test, yt_train, yt_test,
           Xw_train, Xw_test, yw_train, yw_test,
           Xb_train, Xb_test, yb_train, yb_test,
           Xa_train, Xa_test, ya_train, ya_test,
           Xh_train, Xh_test, yh_train, yh_test)



Xt_train, Xt_test, yt_train, yt_test,Xw_train, Xw_test, yw_train, yw_test,Xb_train, Xb_test, yb_train, yb_test,Xa_train, Xa_test, ya_train, ya_test,Xh_train, Xh_test, yh_train, yh_test = ethnic_split()


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


train_label_dist = pd.DataFrame(data = {'yt': yt_test.value_counts(),
          'yw':yw_test.value_counts(),
          'yb':yb_test.value_counts(),
          'ya':ya_test.value_counts(),
          'yh':yh_test.value_counts()})

test_label_dist = pd.DataFrame(data = {'yt': yt_train.value_counts(),
          'yw':yw_train.value_counts(),
          'yb':yb_train.value_counts(),
          'ya':ya_train.value_counts(),
          'yh':yh_train.value_counts()})
    
# train_spread.to_csv('Ethnic Train Data Distribution.csv')    
# test_spread.to_csv('Ethnic Test Data Distribution.csv')
# train_label_dist.to_csv('Ethnic Train Data Label Counts')
# test_label_dist.to_csv('Ethnic Test Data Label Counts')