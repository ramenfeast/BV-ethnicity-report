# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 10:21:45 2022

@author: celestec
"""
# %%Imports and Functions
#%%%Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None, 'display.max_rows', None)
#%%% Random State
rng = np.random.default_rng()
#rando= rng.integers(0,100)
rando = 1
# %%%learning Curve Function

from sklearn.model_selection import learning_curve

def plot_learning_curve(clf, Xtrain, ytrain, title):
    train_sizes, train_scores, validation_scores = learning_curve(estimator=clf,
                               X=Xtrain,
                               y=ytrain,
                               cv=5,
                               scoring='neg_mean_squared_error')

    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)
    print('Mean training scores\n\n', pd.Series(
        train_scores_mean, index=train_sizes))
    print('\n', '-' * 20)  # separator
    print('\nMean validation scores\n\n', pd.Series(
        validation_scores_mean, index=train_sizes))

    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, validation_scores_mean, label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title(title, fontsize=18, y=1.03)
    plt.legend()
    


# %%% Ethnicity checked accuracy function

def ethnic_based_acc(pred, test, save):
    rng=[j for j in range(0, test.size)]
    x=0
    tracker=np.zeros(4)
    for k in rng:
        guess=pred[x]
        check=test.iloc[x]
        if guess == check:
            if save.iloc[x, 0] == 'White':
                tracker[0]=tracker[0]+1

            elif save.iloc[x, 0] == 'Black':
                tracker[1]=tracker[1]+1

            elif save.iloc[x, 0] == 'Asian':
                tracker[2]=tracker[2]+1

            elif save.iloc[x, 0] == 'Hispanic':
                tracker[3]=tracker[3]+1

        x=x+1
    ethnic_accuracy=pd.DataFrame(index=['Accuracy'])
    total_ethnic=es_xttest.value_counts()
    ethnic_accuracy['White']=tracker[0]/total_ethnic['White']
    ethnic_accuracy['Black']=tracker[1]/total_ethnic['Black']
    ethnic_accuracy['Asian']=tracker[2]/total_ethnic['Asian']
    ethnic_accuracy['Hispanic']=tracker[3]/total_ethnic['Hispanic']
    return(ethnic_accuracy)

# %%% Community Group checked accuracy function

def comm_group_acc(pred, test, save):
    rng=[j for j in range(0, test.size)]
    x=0
    tracker=np.zeros(4)
    for k in rng:
        guess=pred[x]
        check=test.iloc[x]
        if guess == check:
            if save.iloc[x, 0] == 'I':
                tracker[0]=tracker[0]+1

            elif save.iloc[x, 0] == 'II':
                tracker[1]=tracker[1]+1

            elif save.iloc[x, 0] == 'III':
                tracker[2]=tracker[2]+1

            elif save.iloc[x, 0] == 'IV':
                tracker[3]=tracker[3]+1

        x=x+1
    community_accuracy=pd.DataFrame(index=['Accuracy'])
    total_comm=cs_xttest.value_counts()
    community_accuracy['I']=tracker[0]/total_comm['I']
    community_accuracy['II']=tracker[1]/total_comm['II']
    community_accuracy['III']=tracker[2]/total_comm['III']
    community_accuracy['IV']=tracker[3]/total_comm['IV']
    return(community_accuracy)

#%%% Ethnic Breakdown of Missclassified Results

def missclass_ethnic(pred, test, save):
    rng=[j for j in range(0, test.size)]
    x=0
    y=0
    tracker=np.zeros(4)
    for k in rng:
        guess=pred[x]
        check=test.iloc[x]
        if guess != check:
            y=y+1
            if save.iloc[x, 0] == 'White':
                tracker[0]=tracker[0]+1

            elif save.iloc[x, 0] == 'Black':
                tracker[1]=tracker[1]+1

            elif save.iloc[x, 0] == 'Asian':
                tracker[2]=tracker[2]+1

            elif save.iloc[x, 0] == 'Hispanic':
                tracker[3]=tracker[3]+1
        x=x+1
    missclass_breakdown=pd.DataFrame(index=['Percentage'])
    missclass_breakdown['White']=100*tracker[0]/y
    missclass_breakdown['Black']=100*tracker[1]/y
    missclass_breakdown['Asian']=100*tracker[2]/y
    missclass_breakdown['Hispanic']=100*tracker[3]/y
    return missclass_breakdown

#%%% Community Group Breakdown of Missclassified Results

def missclass_comm(pred, test, save):
    rng=[j for j in range(0, test.size)]
    x=0
    y=0
    tracker=np.zeros(4)
    for k in rng:
        guess=pred[x]
        check=test.iloc[x]
        if guess != check:
            y=y+1
            if save.iloc[x, 0] == 'I':
                tracker[0]=tracker[0]+1

            elif save.iloc[x, 0] == 'II':
                tracker[1]=tracker[1]+1

            elif save.iloc[x, 0] == 'III':
                tracker[2]=tracker[2]+1

            elif save.iloc[x, 0] == 'IV':
                tracker[3]=tracker[3]+1
        x=x+1
    missclass_breakdown=pd.DataFrame(index=['Percentage'])
    missclass_breakdown['I']=100*tracker[0]/y
    missclass_breakdown['II']=100*tracker[1]/y
    missclass_breakdown['III']=100*tracker[2]/y
    missclass_breakdown['IV']=100*tracker[3]/y
    return missclass_breakdown


#%%% Confusion Matrix Function
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(test, pred, color, title, fignum):
    cm=confusion_matrix(test,pred)

    plt.figure(fignum)
    ax=plt.subplot()
# annot=True to annotate cells, ftm='g' to disable scientific notation
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap=sns.color_palette(color));
 # labels, title and ticks
    ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels');
    ax.set_title(title);
    ax.xaxis.set_ticklabels(['BV Negative', 'BV Positive']); ax.yaxis.set_ticklabels(
    ['BV Negative', 'BV Positive']);

#%%% Standard Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def standard_metrics(ytrue, ypred):
    accuracy = accuracy_score(ytrue, ypred)
    f1 = f1_score(ytrue, ypred)
    precision = precision_score(ytrue, ypred,zero_division=1)
    recall = recall_score(ytrue, ypred, zero_division = 1)
    
    return(accuracy, f1, precision, recall)
# %%% Ethnic Trained Displays
from sklearn.metrics import RocCurveDisplay

def ethnic_spec_displays(classifier, xtrain, ytrain, predw, predb, preda, predh, predt, color,  ):
    plt.figure(0)
    plot_learning_curve(classifier, xtrain, ytrain, "Classifier Learning Curve")

    plt.figure(1)
    RocCurveDisplay.from_estimator(classifier, Xw_test, yw_test)
    plt.figure(2)
    RocCurveDisplay.from_estimator(classifier, Xb_test, yb_test)
    plt.figure(3)
    RocCurveDisplay.from_estimator(classifier, Xa_test, ya_test)
    plt.figure(4)
    RocCurveDisplay.from_estimator(classifier, Xh_test, yh_test)
    plt.figure(5)
    RocCurveDisplay.from_estimator(classifier, Xt_test, yt_test)

    plt.figure(6)
    plot_confusion_matrix(yw_test, predw, color, "Confusion Matrix",7)
    plot_confusion_matrix(yb_test, predb, color, "Confusion Matrix",8)
    plot_confusion_matrix(ya_test, preda, color, "Confusion Matrix",9)
    plot_confusion_matrix(yh_test, predh, color, "Confusion Matrix",10)
    plot_confusion_matrix(yt_test, predt, color, "Confusion Matrix",11)
#%%% Ethnic Trained Metrics
def ethnic_spec_metrics(predw, predb, preda, predh, predt):
    
    #Standard metrics
    accw,f1w,precw,recw = standard_metrics(yw_test, predw)
    accb,f1b,precb,recb = standard_metrics(yb_test, predb)
    acca,f1a,preca,reca = standard_metrics(ya_test, preda)
    acch,f1h,prech,rech = standard_metrics(yh_test, predh)
    acct,f1t,prect,rect = standard_metrics(yt_test, predt)
    
    ethnic_metrics = pd.DataFrame(index = ['White','Black','Asian','Hispanic', 'Total'],
                                  data = {'Accuracy':[accw,accb,acca,acch,acct],
                                          'F1 Score':[f1w,f1b,f1a,f1h,f1t],
                                          'Precision':[precw,precb,preca,prech,prect],
                                          'Recall':[recw,recb,reca,rech,rect]},
                                  )
    print(ethnic_metrics.to_string())
    
    #Ethnic and Community group Accuracy
    ethnic_acc = ethnic_based_acc(predt, yt_test, es_xttest)
    print(ethnic_acc)
    comm_acc = comm_group_acc(predt, yt_test, cs_xttest)
    print(comm_acc)

    #Ethnic and Community Group Breakdown of Missclassified
    miss_ethnic = missclass_ethnic(predt, yt_test, es_xttest)
    print(miss_ethnic)

    miss_comm = missclass_comm(predt, yt_test, cs_xttest)
    print(miss_comm)
    
    return(ethnic_metrics)

#%%% Ethnic Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def ethnic_train_metric_pipe(classifier, xtrain, ytrain, color):

    if classifier == "Logistic Regression": 
        classify = LogisticRegression()
    elif classifier == "Random Forest":
        classify = RandomForestClassifier()
    elif classifier == "SVM":
        classify = SVC()
        
    
    classify.fit(xtrain, ytrain)

    y_pred_clfw = classify.predict(Xw_test)
    y_pred_clfb = classify.predict(Xb_test)
    y_pred_clfa = classify.predict(Xa_test)
    y_pred_clfh = classify.predict(Xh_test)
    y_pred_clft = classify.predict(Xt_test)

    ethnic_spec_displays(
                    classify, xtrain, ytrain, 
                    y_pred_clfw,
                    y_pred_clfb,
                    y_pred_clfa,
                    y_pred_clfh,
                    y_pred_clft,
                    color
                    )
    ethnic_metrics = ethnic_spec_metrics( y_pred_clfw,
     y_pred_clfb,
     y_pred_clfa,
     y_pred_clfh,
     y_pred_clft,)
    return(classify, ethnic_metrics)

#%%% Ethnic Based Stacking Classifier Pipeline
from sklearn.ensemble import StackingClassifier

def ethnic_stack_pipe(clfw, clfb, clfa, clfh, color):
    estimators = [('clfw',clfw), ('clfb', clfb), ('clfa',clfa), ('clfh', clfh)]
    clf = StackingClassifier(
        estimators = estimators, 
        final_estimator = LogisticRegression(), cv= 'prefit')
    clf.fit(Xt_train,yt_train)
    
    y_pred_clfw = clf.predict(Xw_test)
    y_pred_clfb = clf.predict(Xb_test)
    y_pred_clfa = clf.predict(Xa_test)
    y_pred_clfh = clf.predict(Xh_test)
    y_pred_clft = clf.predict(Xt_test)

    ethnic_spec_displays(
                    clf, Xt_train, yt_train, 
                    y_pred_clfw,
                    y_pred_clfb,
                    y_pred_clfa,
                    y_pred_clfh,
                    y_pred_clft,
                    color
                    )
    ethnic_metrics = ethnic_spec_metrics( y_pred_clfw,
     y_pred_clfb,
     y_pred_clfa,
     y_pred_clfh,
     y_pred_clft,)
    return(clf, ethnic_metrics)

#%%% Metrics Grid
def accuracy_grid(clfmw, clfmb, clfma, clfmh, clfmt, clfmst):
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
    
    print(grid)
    return(grid)

# %%% Community Group Trained Displays
def comm_spec_displays(classifier, xtrain, ytrain, predi, predii, prediii, prediv, predt, color,  ):
    plt.figure(0)
    plot_learning_curve(classifier, xtrain, ytrain, "Classifier Learning Curve")

    plt.figure(1)
    RocCurveDisplay.from_estimator(classifier, Xi_test, yi_test)
    plt.figure(2)
    RocCurveDisplay.from_estimator(classifier, Xii_test, yii_test)
    plt.figure(3)
    RocCurveDisplay.from_estimator(classifier, Xiii_test, yiii_test)
    plt.figure(4)
    RocCurveDisplay.from_estimator(classifier, Xiv_test, yiv_test)
    plt.figure(5)
    RocCurveDisplay.from_estimator(classifier, Xt_test, yt_test)

    plt.figure(6)
    #plot_confusion_matrix(yi_test, predi, color, "Confusion Matrix",7)
   # plot_confusion_matrix(yii_test, predii, color, "Confusion Matrix",8)
    plot_confusion_matrix(yiii_test, prediii, color, "Confusion Matrix",9)
    plot_confusion_matrix(yiii_test, prediv, color, "Confusion Matrix",10)
    plot_confusion_matrix(yiv_test, predt, color, "Confusion Matrix",11)

#%%% Community Metrics
def comm_spec_metrics(predi, predii, prediii, prediv, predt):
    
    #Standard metrics
    accw,f1w,precw,recw = standard_metrics(yi_test, predi)
    accb,f1b,precb,recb = standard_metrics(yii_test, predii)
    acca,f1a,preca,reca = standard_metrics(yiii_test, prediii)
    acch,f1h,prech,rech = standard_metrics(yiv_test, prediv)
    acct,f1t,prect,rect = standard_metrics(yt_test, predt)
    
    comm_metrics = pd.DataFrame(index = ['I','II','III','IV', 'Total'],
                                  data = {'Accuracy':[accw,accb,acca,acch,acct],
                                          'F1 Score':[f1w,f1b,f1a,f1h,f1t],
                                          'Precision':[precw,precb,preca,prech,prect],
                                          'Recall':[recw,recb,reca,rech,rect]},
                                  )
    print(comm_metrics.to_string())
    

    #Ethnic and Community group Accuracy
    ethnic_acc = ethnic_based_acc(predt, yt_test, es_xttest)
    print(ethnic_acc)
    comm_acc = comm_group_acc(predt, yt_test, cs_xttest)
    print(comm_acc)

    #Ethnic and Community Group Breakdown of Missclassified
    miss_ethnic = missclass_ethnic(predt, yt_test, es_xttest)
    print(miss_ethnic)

    miss_comm = missclass_comm(predt, yt_test, cs_xttest)
    print(miss_comm)

#%%% Comm Pipeline
def comm_train_metric_pipe(classifier, xtrain, ytrain, color):

    if classifier == "Logistic Regression": 
        classify = LogisticRegression()
    elif classifier == "Random Forest":
        classify = RandomForestClassifier()
    elif classifier == "SVM":
        classify = SVC()
        
    
    classify.fit(xtrain, ytrain)

    y_pred_clfi = classify.predict(Xi_test)
    y_pred_clfii = classify.predict(Xii_test)
    y_pred_clfiii = classify.predict(Xiii_test)
    y_pred_clfiv = classify.predict(Xiv_test)
    y_pred_clft = classify.predict(Xt_test)

    comm_spec_displays(
                    classify, xtrain, ytrain, 
                    y_pred_clfi,
                    y_pred_clfii,
                    y_pred_clfiii,
                    y_pred_clfiv,
                    y_pred_clft,
                    color
                    )
    comm_spec_metrics(
        y_pred_clfi,
        y_pred_clfii,
        y_pred_clfiii,
        y_pred_clfiv,
        y_pred_clft,)
    return(classify)

#%%% Comm Based Stacking Classifier Pipeline
def comm_stack_pipe(clfi, clfii, clfiii, clfiv, color):
    estimators = [('clfi',clfi), ('clfii', clfii), ('clfiii',clfiii), ('clfiv', clfiv)]
    clf = StackingClassifier(
        estimators = estimators, 
        final_estimator = LogisticRegression(), cv= 'prefit')
    clf.fit(Xt_train,yt_train)
    
    y_pred_clfi = clf.predict(Xi_test)
    y_pred_clfii = clf.predict(Xii_test)
    y_pred_clfiii = clf.predict(Xiii_test)
    y_pred_clfiv = clf.predict(Xiv_test)
    y_pred_clft = clf.predict(Xt_test)

    ethnic_spec_displays(
                    clf, Xt_train, yt_train, 
                    y_pred_clfi,
                    y_pred_clfii,
                    y_pred_clfiii,
                    y_pred_clfiv,
                    y_pred_clft,
                    color
                    )
    comm_spec_metrics(
        y_pred_clfi,
        y_pred_clfii,
        y_pred_clfiii,
        y_pred_clfiv,
        y_pred_clft,)
    return(clf)

#%%% Overall Metrics Grid

# %% Import and Clean Data
# %%% Import Data

import requests
import io

url = "https://raw.githubusercontent.com/ramenfeast/BV-ethnicity-report/main/BV%20Dataset%20copy.csv"
download = requests.get(url).content
df = pd.read_csv(io.StringIO(download.decode('utf-8')))

print(df)

# %%%Clean data
df = df.drop([394, 395, 396], axis=0)

# %% Train Test Split and Normalization (Ethnic Isolated)
#%%% Initial X y split
X_total = df.iloc[:, :-1]
y_total = df.iloc[:, -1]

# %%% Initial Test Train Split
from sklearn.model_selection import train_test_split

Xt_train, Xt_test, yt_train, yt_test = train_test_split(
    X_total, y_total, test_size=0.2, random_state=1)
# %%% Sort and Split Ethnicities
X_w = Xt_train.loc[Xt_train['Ethnic Groupa'] == 'White']
X_b = Xt_train.loc[Xt_train['Ethnic Groupa'] == 'Black']
X_a = Xt_train.loc[Xt_train['Ethnic Groupa'] == 'Asian']
X_h = Xt_train.loc[Xt_train['Ethnic Groupa'] == 'Hispanic']

y_w = yt_train.loc[Xt_train['Ethnic Groupa'] == 'White']
y_b = yt_train.loc[Xt_train['Ethnic Groupa'] == 'Black']
y_a = yt_train.loc[Xt_train['Ethnic Groupa'] == 'Asian']
y_h = yt_train.loc[Xt_train['Ethnic Groupa'] == 'Hispanic']


# %%% Test Train Split
Xw_train, Xw_test, yw_train, yw_test = train_test_split(
    X_w, y_w, test_size=0.2, random_state=rando)

Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_b, y_b, test_size=0.2, random_state=rando)

Xa_train, Xa_test, ya_train, ya_test = train_test_split(
    X_a, y_a, test_size=0.2, random_state=rando)

Xh_train, Xh_test, yh_train, yh_test = train_test_split(
    X_h, y_h, test_size=0.2, random_state=rando)

#%%% Save Demographic Info on overall data
es_xttest = Xt_test[['Ethnic Groupa']].copy()
cs_xttest = Xt_test[['Community groupc ']].copy()

es_xttrain = Xt_train[['Ethnic Groupa']].copy()
cs_xttrain = Xt_train[['Community groupc ']].copy()

# %%% Drop Ethinic group and commmunity group data

Xt_test = Xt_test.drop(labels=['Ethnic Groupa', 'Community groupc '], axis=1)
Xt_train = Xt_train.drop(labels=['Ethnic Groupa', 'Community groupc '], axis=1)

Xw_test = Xw_test.drop(labels=['Ethnic Groupa', 'Community groupc '], axis=1)
Xw_train = Xw_train.drop(labels=['Ethnic Groupa', 'Community groupc '], axis=1)

Xb_test = Xb_test.drop(labels=['Ethnic Groupa', 'Community groupc '], axis=1)
Xb_train = Xb_train.drop(labels=['Ethnic Groupa', 'Community groupc '], axis=1)

Xa_test = Xa_test.drop(labels=['Ethnic Groupa', 'Community groupc '], axis=1)
Xa_train = Xa_train.drop(labels=['Ethnic Groupa', 'Community groupc '], axis=1)

Xh_test = Xh_test.drop(labels=['Ethnic Groupa', 'Community groupc '], axis=1)
Xh_train = Xh_train.drop(labels=['Ethnic Groupa', 'Community groupc '], axis=1)

# %%% Normalization

# Normalize pH
Xt_train['pH'] = Xt_train['pH']/14
Xt_test['pH'] = Xt_test['pH']/14

Xw_train['pH'] = Xw_train['pH']/14
Xw_test['pH'] = Xw_test['pH']/14

Xb_train['pH'] = Xb_train['pH']/14
Xb_test['pH'] = Xb_test['pH']/14

Xa_train['pH'] = Xa_train['pH']/14
Xa_test['pH'] = Xa_test['pH']/14

Xh_train['pH'] = Xh_train['pH']/14
Xh_test['pH'] = Xh_test['pH']/14

# Normalize 16s RNA data
Xt_train.iloc[:, 1::] = Xt_train.iloc[:, 1::]/100
Xt_test.iloc[:, 1::] = Xt_test.iloc[:, 1::]/100

Xw_train.iloc[:, 1::] = Xw_train.iloc[:, 1::]/100
Xw_test.iloc[:, 1::] = Xw_test.iloc[:, 1::]/100

Xb_train.iloc[:, 1::] = Xb_train.iloc[:, 1::]/100
Xb_test.iloc[:, 1::] = Xb_test.iloc[:, 1::]/100

Xa_train.iloc[:, 1::] = Xa_train.iloc[:, 1::]/100
Xa_test.iloc[:, 1::] = Xa_test.iloc[:, 1::]/100

Xh_train.iloc[:, 1::] = Xh_train.iloc[:, 1::]/100
Xh_test.iloc[:, 1::] = Xh_test.iloc[:, 1::]/100

# %%% Binary y
yt_train[yt_train < 7] = 0
yt_train[yt_train >= 7] = 1

yt_test[yt_test < 7] = 0
yt_test[yt_test >= 7] = 1

yw_train[yw_train < 7] = 0
yw_train[yw_train >= 7] = 1

yw_test[yw_test < 7] = 0
yw_test[yw_test >= 7] = 1

yb_train[yb_train < 7] = 0
yb_train[yb_train >= 7] = 1

yb_test[yb_test < 7] = 0
yb_test[yb_test >= 7] = 1

ya_train[ya_train < 7] = 0
ya_train[ya_train >= 7] = 1

ya_test[ya_test < 7] = 0
ya_test[ya_test >= 7] = 1

yh_train[yh_train < 7] = 0
yh_train[yh_train >= 7] = 1

yh_test[yh_test < 7] = 0
yh_test[yh_test >= 7] = 1

# %% Logistic Regression (Ethnic Isolated)
#%%% LR Normal Training
clflrt, clflrmt = ethnic_train_metric_pipe("Logistic Regression", Xt_train, yt_train, "Greys")
#%%% LR Just White
clflrw, clflrmw = ethnic_train_metric_pipe("Logistic Regression", Xw_train, yw_train, "Blues")

#%%% LR Just Black
clflrb, clflrmb = ethnic_train_metric_pipe("Logistic Regression", Xb_train, yb_train, "Reds")

#%%% LR Just Asian
clflra, clflrma = ethnic_train_metric_pipe("Logistic Regression", Xa_train, ya_train, "Greens")

#%%% LR Just Hispanic
clflrh, clflrmh = ethnic_train_metric_pipe("Logistic Regression", Xh_train, yh_train, "Purples")

#%%% LR Stack
clflrst, clflrmst = ethnic_stack_pipe(clflrh,clflrb,clflra,clflrh, "Oranges")

#%%% LR Metrics Grid
lrgrid = accuracy_grid(clflrmw, clflrmb, clflrma,clflrmh,clflrmt,clflrmst)
#%% Random Forest (Ethnic Isolated)

#%%% RF Normal Training
clfrft, clfrfmt = ethnic_train_metric_pipe("Random Forest", Xt_train, yt_train, "Greys")
#%%% RF Just White
clfrfw, clfrfmw = ethnic_train_metric_pipe("Random Forest", Xw_train, yw_train, "Blues")

#%%% RF Just Black
clfrfb, clfrfmb = ethnic_train_metric_pipe("Random Forest", Xb_train, yb_train, "Reds")

#%%% RF Just Asian
clfrfa, clfrfma = ethnic_train_metric_pipe("Random Forest", Xa_train, ya_train, "Greens")

#%%% RF Just Hispanic
clfrfh, clfrfmh = ethnic_train_metric_pipe("Random Forest", Xh_train, yh_train, "Purples")

#%%% RF Stack
clfrft, clfrfmst = ethnic_stack_pipe(clfrfw,clfrfb,clfrfa,clfrfh, "Oranges")

#%%% RF Metrics Grid
lrgrid = accuracy_grid(clfrfmw, clfrfmb, clfrfma,clfrfmh,clfrfmt,clfrfmst)
#%% SVM (Ethnic Isolated)
#%%% SVM Normal Training
clfsvmt, clfsvmmt = ethnic_train_metric_pipe("SVM", Xt_train, yt_train, "Greys")
#%%% SVM Just White
clfsvmw, clfsvmmw = ethnic_train_metric_pipe("SVM", Xw_train, yw_train, "Blues")

#%%% SVM Just Black
clfsvmb, clfsvmmb = ethnic_train_metric_pipe("SVM", Xb_train, yb_train, "Reds")

#%%% SVM Just Asian
clfsvma, clfsvmma = ethnic_train_metric_pipe("SVM", Xa_train, ya_train, "Greens")

#%%% SVM Just Hispanic
clfsvmh, clfsvmmh = ethnic_train_metric_pipe("SVM", Xh_train, yh_train, "Purples")
#%%% SVM Stack
clfsvmt, clfsvmmst = ethnic_stack_pipe(clfsvmw,clfsvmb,clfsvma,clfsvmh, "Oranges")

#%%% SVM Metrics Grid
svmgrid = accuracy_grid(clfsvmmw, clfsvmmb, clfsvmma,clfsvmmh,clfsvmmt,clfsvmmst)
# %% Train Test Split and Normalization (Community Group Isolated)
#%%% Initial X y split
X_total = df.iloc[:, :-1]
y_total = df.iloc[:, -1]

# %%% Initial Test Train Split
Xt_train, Xt_test, yt_train, yt_test = train_test_split(
    X_total, y_total, test_size=0.2, random_state=1)
# %%% Sort and Split Ethnicities
X_I = Xt_train.loc[Xt_train['Community groupc '] == 'I']
X_II = Xt_train.loc[Xt_train['Community groupc '] == 'II']
X_III = Xt_train.loc[Xt_train['Community groupc '] == 'III']
X_IV = Xt_train.loc[Xt_train['Community groupc '] == 'IV']

y_I = yt_train.loc[Xt_train['Community groupc '] == 'I']
y_II = yt_train.loc[Xt_train['Community groupc '] == 'II']
y_III = yt_train.loc[Xt_train['Community groupc '] == 'III']
y_IV = yt_train.loc[Xt_train['Community groupc '] == 'IV']


# %%% Test Train Split
Xi_train, Xi_test, yi_train, yi_test = train_test_split(
    X_I, y_I, test_size=0.2, random_state=rando)

Xii_train, Xii_test, yii_train, yii_test = train_test_split(
    X_II, y_II, test_size=0.2, random_state=rando)

Xiii_train, Xiii_test, yiii_train, yiii_test = train_test_split(
    X_III, y_III, test_size=0.2, random_state=rando)

Xiv_train, Xiv_test, yiv_train, yiv_test = train_test_split(
    X_IV, y_IV, test_size=0.2, random_state=rando)

#%%% Save Demographic Info on overall data
es_xttest = Xt_test[['Ethnic Groupa']].copy()
cs_xttest = Xt_test[['Community groupc ']].copy()

es_xttrain = Xt_train[['Ethnic Groupa']].copy()
cs_xttrain = Xt_train[['Community groupc ']].copy()

# %%% Drop Ethinic group and commmunity group data

Xt_test = Xt_test.drop(labels=['Ethnic Groupa', 'Community groupc '], axis=1)
Xt_train = Xt_train.drop(labels=['Ethnic Groupa', 'Community groupc '], axis=1)

Xi_test = Xi_test.drop(labels=['Ethnic Groupa', 'Community groupc '], axis=1)
Xi_train = Xi_train.drop(labels=['Ethnic Groupa', 'Community groupc '], axis=1)

Xii_test = Xii_test.drop(labels=['Ethnic Groupa', 'Community groupc '], axis=1)
Xii_train = Xii_train.drop(labels=['Ethnic Groupa', 'Community groupc '], axis=1)

Xiii_test = Xiii_test.drop(labels=['Ethnic Groupa', 'Community groupc '], axis=1)
Xiii_train = Xiii_train.drop(labels=['Ethnic Groupa', 'Community groupc '], axis=1)

Xiv_test = Xiv_test.drop(labels=['Ethnic Groupa', 'Community groupc '], axis=1)
Xiv_train = Xiv_train.drop(labels=['Ethnic Groupa', 'Community groupc '], axis=1)

# %%% Normalization

# Normalize pH
Xt_train['pH'] = Xt_train['pH']/14
Xt_test['pH'] = Xt_test['pH']/14

Xi_train['pH'] = Xi_train['pH']/14
Xi_test['pH'] = Xi_test['pH']/14

Xii_train['pH'] = Xii_train['pH']/14
Xii_test['pH'] = Xii_test['pH']/14

Xiii_train['pH'] = Xiii_train['pH']/14
Xiii_test['pH'] = Xiii_test['pH']/14

Xiv_train['pH'] = Xiv_train['pH']/14
Xiv_test['pH'] = Xiv_test['pH']/14

# Normalize 16s RNA data
Xt_train.iloc[:, 1::] = Xt_train.iloc[:, 1::]/100
Xt_test.iloc[:, 1::] = Xt_test.iloc[:, 1::]/100

Xi_train.iloc[:, 1::] = Xi_train.iloc[:, 1::]/100
Xi_test.iloc[:, 1::] = Xi_test.iloc[:, 1::]/100

Xii_train.iloc[:, 1::] = Xii_train.iloc[:, 1::]/100
Xii_test.iloc[:, 1::] = Xii_test.iloc[:, 1::]/100

Xiii_train.iloc[:, 1::] = Xiii_train.iloc[:, 1::]/100
Xiii_test.iloc[:, 1::] = Xiii_test.iloc[:, 1::]/100

Xiv_train.iloc[:, 1::] = Xiv_train.iloc[:, 1::]/100
Xiv_test.iloc[:, 1::] = Xiv_test.iloc[:, 1::]/100

# %%% Binary y
yt_train[yt_train < 7] = 0
yt_train[yt_train >= 7] = 1

yt_test[yt_test < 7] = 0
yt_test[yt_test >= 7] = 1

yi_train[yi_train < 7] = 0
yi_train[yi_train >= 7] = 1

yi_test[yi_test < 7] = 0
yi_test[yi_test >= 7] = 1

yii_train[yii_train < 7] = 0
yii_train[yii_train >= 7] = 1

yii_test[yii_test < 7] = 0
yii_test[yii_test >= 7] = 1

yiii_train[yiii_train < 7] = 0
yiii_train[yiii_train >= 7] = 1

yiii_test[yiii_test < 7] = 0
yiii_test[yiii_test >= 7] = 1

yiv_train[yiv_train < 7] = 0
yiv_train[yiv_train >= 7] = 1

yiv_test[yiv_test < 7] = 0
yiv_test[yiv_test >= 7] = 1
# %% Logistic Regression (Comm Isolated)
#%%% I
#clflri = ethnic_train_metric_pipe("Logistic Regression", Xi_train, yi_train, "Blues")

#%%% II
#clflrii = ethnic_train_metric_pipe("Logistic Regression", Xii_train, yii_train, "Reds")

#%%% III
#clflriii = comm_train_metric_pipe("Logistic Regression", Xiii_train, yiii_train, "Greens")

#%%% IV
#clflriv = comm_train_metric_pipe("Logistic Regression", Xiv_train, yiv_train, "Purples")

#%%% Stack
#clflrt = comm_stack_pipe(clflrh,clflrb,clflra,clflrh, "Oranges")

#%% Random Forest (Comm Isolated)
#%%% I
#clfrfw = ethnic_train_metric_pipe("Random Forest", Xi_train, yi_train, "Blues")

#%%% II
#clfrfb = ethnic_train_metric_pipe("Random Forest", Xii_train, yii_train, "Reds")

#%%% III
#clfrfa = comm_train_metric_pipe("Random Forest", Xiii_train, yiii_train, "Greens")

#%%% IV
#clfrfh = comm_train_metric_pipe("Random Forest", Xiv_train, yiv_train, "Purples")
#%%% Stack
#clfrft = comm_stack_pipe(clfrfw,clfrfb,clfrfa,clfrfh, "Oranges")

#%% SVM (Comm Isolated)
#%%% I
#clfsvmw = ethnic_train_metric_pipe("SVM", Xi_train, yi_train, "Blues")

#%%% II
#clfsvmb = ethnic_train_metric_pipe("SVM", Xii_train, yii_train, "Reds")

#%%% III
#clfsvma = comm_train_metric_pipe("SVM", Xiii_train, yiii_train, "Greens")

#%%% IV
#clfsvmh = comm_train_metric_pipe("SVM", Xiv_train, yiv_train, "Purples")
#%%% Stack
#clfsvmt = comm_stack_pipe(clfsvmw,clfsvmb,clfsvma,clfsvmh, "Oranges")