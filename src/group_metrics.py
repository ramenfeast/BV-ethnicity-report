# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:20:46 2022

@author: camer
"""

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd
import numpy as np

def ethnic_pred(clf, Xw_test, Xb_test, Xa_test, Xh_test, Xt_test):
    
    y_pred_clfw = clf.predict(Xw_test)
    y_pred_clfb = clf.predict(Xb_test)
    y_pred_clfa = clf.predict(Xa_test)
    y_pred_clfh = clf.predict(Xh_test)
    y_pred_clft = clf.predict(Xt_test)
    
    return y_pred_clfw, y_pred_clfb, y_pred_clfa, y_pred_clfh, y_pred_clft
    
def comm_pred(clf, Xi_test, Xii_test, Xiii_test, Xiv_test, Xv_test, Xt_test):
    
    y_pred_clfi = clf.predict(Xi_test)
    y_pred_clfii = clf.predict(Xii_test)
    y_pred_clfiii = clf.predict(Xiii_test)
    y_pred_clfiv = clf.predict(Xiv_test)
    y_pred_clfv = clf.predict(Xv_test)
    y_pred_clft = clf.predict(Xt_test)
    
    return y_pred_clfi, y_pred_clfii, y_pred_clfiii, y_pred_clfiv, y_pred_clfv, y_pred_clft

def standard_metrics(ytrue, ypred):
    accuracy = accuracy_score(ytrue, ypred)
    f1 = f1_score(ytrue, ypred, zero_division = 1)
    precision = precision_score(ytrue, ypred,zero_division=1)
    recall = recall_score(ytrue, ypred, zero_division = 1)
    
    return(accuracy, f1, precision, recall)

def ethnic_metrics(predw, truew, predb, trueb, preda, truea, predh, trueh, predt, truet ):
    
    accw,f1w,precw,recw = standard_metrics(truew, predw)
    accb,f1b,precb,recb = standard_metrics(trueb, predb)
    acca,f1a,preca,reca = standard_metrics(truea, preda)
    acch,f1h,prech,rech = standard_metrics(trueh, predh)
    acct,f1t,prect,rect = standard_metrics(truet, predt)
    
    ethnic_metrics = pd.DataFrame(index = ['White','Black','Asian','Hispanic', 'Total'],
                                  data = {'Accuracy':[accw,accb,acca,acch,acct],
                                          'F1 Score':[f1w,f1b,f1a,f1h,f1t],
                                          'Precision':[precw,precb,preca,prech,prect],
                                          'Recall':[recw,recb,reca,rech,rect]},
                                  )
    
    return ethnic_metrics

def ethnic_accuracy_grid(clfmw, clfmb, clfma, clfmh, clfmt, clfmst):
    data = {'White Trained': np.zeros(5),
        'Black Trained': np.zeros(5),
        'Asian Trained': np.zeros(5),
        'Hispanic Trained': np.zeros(5),
        'Total Trained': np.zeros(5),
        'Stack Classifier': np.zeros(5)}
    grid = pd.DataFrame(index = ['White','Black','Asian','Hispanic', 'Total'],
                    data = data
                    )

    grid['White Trained'] = clfmw['Accuracy']
    grid['Black Trained'] = clfmb['Accuracy']
    grid['Asian Trained'] = clfma['Accuracy']
    grid['Hispanic Trained'] = clfmh['Accuracy']
    grid['Total Trained'] = clfmt['Accuracy']
    grid['Stack Classifier'] = clfmst['Accuracy']
    
    return grid

def ethnic_pred_metric_pipe(classifiers, Xtest, ytest):
    data = {'White Trained': np.zeros(5),
        'Black Trained': np.zeros(5),
        'Asian Trained': np.zeros(5),
        'Hispanic Trained': np.zeros(5),
        'Total Trained': np.zeros(5),
        'Stack Classifier': np.zeros(5)}
    grid = pd.DataFrame(index = ['White','Black','Asian','Hispanic', 'Total'],
                    data = data
                    )
    n = 0
    for clf in classifiers:
        x = 0
        accuracy = []
        for test in Xtest:
            pred = clf.predict(Xtest[x])
            acc = [accuracy_score(ytest[x],pred)]
            accuracy = accuracy + acc
            x +=1
        for y in [0,1,2,3,4]:
            grid.iloc[y,n] = accuracy[y]
        n+=1
    return grid

        
def ethnic_acc_breakdown(classifiers, Xtest, ytest, ethnic_index):
    data = {'White Trained': np.zeros(5),
        'Black Trained': np.zeros(5),
        'Asian Trained': np.zeros(5),
        'Hispanic Trained': np.zeros(5),
        'Total Trained': np.zeros(5),
        'Stack Classifier': np.zeros(5)}
    grid = pd.DataFrame(index = ['White','Black','Asian','Hispanic', 'Total'],
                    data = data
                    )
    n = 0
    for clf in classifiers:
        pred = clf.predict(Xtest)
        total_acc = accuracy_score(ytest,pred)
        tracker = np.zeros(4)
        total = np.zeros(4)
        x=0
        col = ytest.shape
        for i in range(col[0]):
            guess = pred[i]
            check = ytest.iloc[i]
            
            if ethnic_index[ytest.index[i]] == 'White':
                total[0] = total[0]+1
                if guess == check:
                    tracker[0]=tracker[0]+1

            elif ethnic_index[ytest.index[i]] == 'Black':
                total[1] = total[1]+1
                if guess == check:
                    tracker[1]=tracker[1]+1

            elif ethnic_index[ytest.index[i]] == 'Asian':
                total[2] = total[2]+1
                if guess == check:
                    tracker[2]=tracker[2]+1

            elif ethnic_index[ytest.index[i]] == 'Hispanic':
                total[3] = total[3]+1
                if guess == check:
                    tracker[3]=tracker[3]+1
            x+=1
        ethnic_acc = np.divide(tracker,total)
        for y in [0,1,2,3]:
            grid.iloc[y,n] = ethnic_acc[y]
        grid.iloc[4,n]=total_acc
        n+=1
    return grid
    