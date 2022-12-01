# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:05:21 2022

@author: celestec
"""

from utils import get_XY, ethnicities
from group_split import ethnic_split
from model_training import class_train, ethnic_pred, ethnic_stack_train
random = 1
X,y = get_XY()
ethnic_index = ethnicities()

(Xt_train, Xt_test, yt_train, yt_test, 
        Xw_train, Xw_test, yw_train, yw_test,
        Xb_train, Xb_test, yb_train, yb_test,
        Xa_train, Xa_test, ya_train, ya_test,
        Xh_train, Xh_test, yh_train, yh_test) = ethnic_split(X,y,ethnic_index, random_state = random)

clfw = class_train("Random Forest", Xw_train, yw_train, random_state = random)
clfb = class_train("Random Forest", Xb_train, yb_train, random_state = random)
clfa = class_train("Random Forest", Xa_train, ya_train, random_state = random)
clfh = class_train("Random Forest", Xh_train, yh_train, random_state = random)

clfs = ethnic_stack_train(clfw, clfb, clfa, clfh, Xt_train, yt_train)

y_pred_clfw, y_pred_clfb,y_pred_clfa, y_pred_clfh, y_pred_clft = ethnic_pred(clfs, Xw_test, Xb_test, Xa_test, Xh_test, Xt_test)

