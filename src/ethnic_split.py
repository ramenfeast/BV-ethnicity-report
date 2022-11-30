# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 2022

@author: celestec
"""
rando = 1

import utils
from sklearn.model_selection import train_test_split

X,y = utils.get_XY()

ethnic_index = utils.ethnicities()
X_w = X.loc[ethnic_index == 'White']

Xt_train, Xt_test, yt_train, yt_test = train_test_split(
    X, y, test_size=0.2, random_state=rando, stratify = y)


Xw_train = X_w[Xt_train.index]
