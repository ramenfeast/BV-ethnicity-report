# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:05:21 2022

@author: celestec
"""
import time
start_time = time.time()

from utils import get_XY, ethnicities
from group_split import ethnic_split
from model_training import class_train, ethnic_stack_train
from group_metrics import standard_metrics, ethnic_metrics, ethnic_accuracy_grid, ethnic_pred_metric_pipe, ethnic_acc_breakdown

random = 2022
#%%


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
clft = class_train("Random Forest", Xt_train, yt_train, random_state = random)
clfs = ethnic_stack_train(clfw, clfb, clfa, clfh, Xt_train, yt_train)

#%%

classifiers = (clfw,clfb,clfa,clfh,clft,clfs)
Xtest = (Xw_test, Xb_test, Xa_test, Xh_test, Xt_test)
ytest = (yw_test, yb_test, ya_test, yh_test, yt_test)

grid = ethnic_pred_metric_pipe(classifiers, Xtest, ytest)
#%%
acc_breakdown = ethnic_acc_breakdown(classifiers, Xt_test,yt_test,ethnic_index)

print("--- %s seconds ---" % (time.time() - start_time))