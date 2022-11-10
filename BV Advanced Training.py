# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 12:08:14 2022

@author: celestec
"""
#%%Imports and Functions
#%%%Imports
import numpy as np
import pandas as pd
import requests
import io
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
import seaborn as sns

# %%%learning Curve Function


def plot_learning_curve(clf, title):
    train_sizes, train_scores, validation_scores = learning_curve(estimator=clf,
                               X=X_train,
                               y=y_train,
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
    
# %%% Confusion Matrix Function

def plot_confusion_matrix(test, pred, color, title):
    cm=confusion_matrix(test,pred)

    ax=plt.subplot()
# annot=True to annotate cells, ftm='g' to disable scientific notation
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap=sns.color_palette(color));

# labels, title and ticks
    ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels');
    ax.set_title(title);
    ax.xaxis.set_ticklabels(['BV Positive', 'BV Negative']); ax.yaxis.set_ticklabels(
    ['BV Positive', 'BV Negative']);

# %%% Ethnicity checked accuracy function

def ethnic_based_acc(pred):
    rng=[j for j in range(0, y_test.size)]
    x=0
    tracker=np.zeros(4)
    for k in rng:
        guess=pred[x]
        check=y_test.iloc[x]
        if guess == check:
            if es_xtest.iloc[x, 0] == 'White':
                tracker[0]=tracker[0]+1

            elif es_xtest.iloc[x, 0] == 'Asian':
                tracker[1]=tracker[1]+1

            elif es_xtest.iloc[x, 0] == 'Black':
                tracker[2]=tracker[2]+1

            elif es_xtest.iloc[x, 0] == 'Hispanic':
                tracker[3]=tracker[3]+1

        x=x+1
    ethnic_accuracy=pd.DataFrame(index=['Accuracy'])
    total_ethnic=es_xtest.value_counts()
    ethnic_accuracy['White']=tracker[0]/total_ethnic['White']
    ethnic_accuracy['Asian']=tracker[1]/total_ethnic['Asian']
    ethnic_accuracy['Black']=tracker[2]/total_ethnic['Black']
    ethnic_accuracy['Hispanic']=tracker[3]/total_ethnic['Hispanic']
    return(ethnic_accuracy)

# %%% Community Group checked accuracy function

def comm_group_acc(pred):
    rng=[j for j in range(0, y_test.size)]
    x=0
    tracker=np.zeros(4)
    for k in rng:
        guess=pred[x]
        check=y_test.iloc[x]
        if guess == check:
            if cs_xtest.iloc[x, 0] == 'I':
                tracker[0]=tracker[0]+1

            elif cs_xtest.iloc[x, 0] == 'II':
                tracker[1]=tracker[1]+1

            elif cs_xtest.iloc[x, 0] == 'III':
                tracker[2]=tracker[2]+1

            elif cs_xtest.iloc[x, 0] == 'IV':
                tracker[3]=tracker[3]+1

        x=x+1
    community_accuracy=pd.DataFrame(index=['Accuracy'])
    total_comm=cs_xtest.value_counts()
    community_accuracy['I']=tracker[0]/total_comm['I']
    community_accuracy['II']=tracker[1]/total_comm['II']
    community_accuracy['III']=tracker[2]/total_comm['III']
    community_accuracy['IV']=tracker[3]/total_comm['IV']
    return(community_accuracy)

#%%% Ethnic Breakdown of Missclassified Results

def missclass_ethnic(pred):
    rng=[j for j in range(0, y_test.size)]
    x=0
    y=0
    tracker=np.zeros(4)
    for k in rng:
        guess=pred[x]
        check=y_test.iloc[x]
        if guess != check:
            y=y+1
            if es_xtest.iloc[x, 0] == 'White':
                tracker[0]=tracker[0]+1

            elif es_xtest.iloc[x, 0] == 'Asian':
                tracker[1]=tracker[1]+1

            elif es_xtest.iloc[x, 0] == 'Black':
                tracker[2]=tracker[2]+1

            elif es_xtest.iloc[x, 0] == 'Hispanic':
                tracker[3]=tracker[3]+1
        x=x+1
    missclass_breakdown=pd.DataFrame(index=['Percentage'])
    missclass_breakdown['White']=100*tracker[0]/y
    missclass_breakdown['Asian']=100*tracker[1]/y
    missclass_breakdown['Black']=100*tracker[2]/y
    missclass_breakdown['Hispanic']=100*tracker[3]/y
    return missclass_breakdown

#%%% Community Group Breakdown of Missclassified Results

def missclass_comm(pred):
    rng=[j for j in range(0, y_test.size)]
    x=0
    y=0
    tracker=np.zeros(4)
    for k in rng:
        guess=pred[x]
        check=y_test.iloc[x]
        if guess != check:
            y=y+1
            if cs_xtest.iloc[x, 0] == 'I':
                tracker[0]=tracker[0]+1

            elif cs_xtest.iloc[x, 0] == 'II':
                tracker[1]=tracker[1]+1

            elif cs_xtest.iloc[x, 0] == 'III':
                tracker[2]=tracker[2]+1

            elif cs_xtest.iloc[x, 0] == 'IV':
                tracker[3]=tracker[3]+1
        x=x+1
    missclass_breakdown=pd.DataFrame(index=['Percentage'])
    missclass_breakdown['I']=100*tracker[0]/y
    missclass_breakdown['II']=100*tracker[1]/y
    missclass_breakdown['III']=100*tracker[2]/y
    missclass_breakdown['IV']=100*tracker[3]/y
    return missclass_breakdown
# %% Import and Clean Data
# %%% Import Data



url = "https://raw.githubusercontent.com/ramenfeast/BV-ethnicity-report/main/BV%20Dataset%20copy.csv"
download = requests.get(url).content
df = pd.read_csv(io.StringIO(download.decode('utf-8')))

print(df)

# %%%Clean data
df = df.drop([394, 395, 396], axis=0)

# %% Train Test Split and Normalization (Standard)
# %%% Separate the Data and Labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# %%% Test Train Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)
# %%% Extract Ethinic group and commmunity group data
es_xtest = X_test[['Ethnic Groupa']].copy()
cs_xtest = X_test[['Community groupc ']].copy()
X_test = X_test.drop(labels=['Ethnic Groupa', 'Community groupc '], axis=1)


es_xtrain = X_train[['Ethnic Groupa']].copy()
cs_xtrain = X_train[['Community groupc ']].copy()
X_train = X_train.drop(labels=['Ethnic Groupa', 'Community groupc '], axis=1)

# %%% Normalization

# Normalize pH
X_train['pH'] = X_train['pH']/14
X_test['pH'] = X_test['pH']/14

# Normalize 16s RNA data
X_train.iloc[:, 1::] = X_train.iloc[:, 1::]/100
X_test.iloc[:, 1::] = X_test.iloc[:, 1::]/100

# %%% Binary y
y_train[y_train < 7] = 0
y_train[y_train >= 7] = 1

y_test[y_test < 7] = 0
y_test[y_test >= 7] = 1

# %% Train Test Split and Normalization (Ethnic Isolated)
#%%% Initial X y split
X_total = df.iloc[:, :-1]
y_total = df.iloc[:, -1]

# %%% Initial Test Train Split
Xt_train, Xt_test, yt_train, yt_test = train_test_split(
    X_total, y_total, test_size=0.2, random_state=1)
# %%% Sort and Split Ethnicities
X_w = Xt_train.loc[Xt_train.loc['Ethnic Groupa'] == 'White']
X_b = Xt_train.loc[Xt_train.loc['Ethnic Groupa'] == 'Black']
X_a = Xt_train.loc[Xt_train.loc['Ethnic Groupa'] == 'Asian']
X_h = Xt_train.loc[Xt_train.loc['Ethnic Groupa'] == 'Hispanic']

#y_w = yt_train.loc[Xt_train.loc['Ethnic Groupa'] == 'White']

#%%% Split x and y
X_w = white_samples.iloc[:, :-1]
y_w = white_samples.iloc[:, -1]

X_b = black_samples.iloc[:, :-1]
y_b = black_samples.iloc[:, -1]

X_a = asian_samples.iloc[:, :-1]
y_a = asian_samples.iloc[:, -1]

X_h = hispanic_samples.iloc[:, :-1]
y_h = hispanic_samples.iloc[:, -1]

# %%% Test Train Split
Xw_train, Xw_test, yw_train, yw_test = train_test_split(
    X_w, y_w, test_size=0.2, random_state=1)

Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_b, y_b, test_size=0.2, random_state=1)

Xa_train, Xa_test, ya_train, ya_test = train_test_split(
    X_a, y_a, test_size=0.2, random_state=1)

Xh_train, Xh_test, yh_train, yh_test = train_test_split(
    X_h, y_h, test_size=0.2, random_state=1)
# %%% Drop Ethinic group and commmunity group data

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
Xw_train['pH'] = Xw_train['pH']/14
Xw_test['pH'] = Xw_test['pH']/14

Xb_train['pH'] = Xb_train['pH']/14
Xb_test['pH'] = Xb_test['pH']/14

Xa_train['pH'] = Xa_train['pH']/14
Xa_test['pH'] = Xa_test['pH']/14

Xh_train['pH'] = Xh_train['pH']/14
Xh_test['pH'] = Xh_test['pH']/14

# Normalize 16s RNA data
Xw_train.iloc[:, 1::] = Xw_train.iloc[:, 1::]/100
Xw_test.iloc[:, 1::] = Xw_test.iloc[:, 1::]/100

Xb_train.iloc[:, 1::] = Xb_train.iloc[:, 1::]/100
Xb_test.iloc[:, 1::] = Xb_test.iloc[:, 1::]/100

Xa_train.iloc[:, 1::] = Xa_train.iloc[:, 1::]/100
Xa_test.iloc[:, 1::] = Xa_test.iloc[:, 1::]/100

Xh_train.iloc[:, 1::] = Xh_train.iloc[:, 1::]/100
Xh_test.iloc[:, 1::] = Xh_test.iloc[:, 1::]/100

# %%% Binary y
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
# %% Logistic Regression (Simple Training)
# %%% Model Training
clflr = LogisticRegression().fit(X_train, y_train)
y_pred_clflr = clflr.predict(X_test)

# %%%Logistic Regression Learning Curve
plot_learning_curve(clflr, "Learning Curve for Logistic Regression Classifier")

# %%% LR ROC
RocCurveDisplay.from_estimator(clflr, X_test, y_test)

#%%%LR Confusion Matrix
plot_confusion_matrix(y_test, y_pred_clflr, "Blues", "Logistic Regression Confusion Matrix")

#%%% LR Ethnic and Community group Accuracy
ethnic_acc_clflr = ethnic_based_acc(y_pred_clflr)
print(ethnic_acc_clflr)
comm_acc_clflr = comm_group_acc(y_pred_clflr)
print(comm_acc_clflr)

#%%% LR Ethnic and Community Group Breakdown of Missclassified
miss_ethnic_clflr = missclass_ethnic(y_pred_clflr)
print(miss_ethnic_clflr)

miss_comm_clflr = missclass_comm(y_pred_clflr)
print(miss_comm_clflr)

# %%Random Forest (Simple Training)
#%%% Model Training
clfrf = RandomForestClassifier().fit(X_train, y_train)
y_pred_clfrf = clfrf.predict(X_test)

# %%% RF learning curve
plot_learning_curve(clfrf, "Learning Curve for Random Forest Classifier")

# %%% RF ROC
RocCurveDisplay.from_estimator(clfrf, X_test, y_test)

# %%% RF Confusion Matrix
plot_confusion_matrix(y_test, y_pred_clfrf, "Greens", "Random Forest Confusion Matrix")

#%%% RF Ethnic and Community group Accuracy
ethnic_acc_clfrf = ethnic_based_acc(y_pred_clfrf)
print(ethnic_acc_clfrf)
comm_acc_clfrf = comm_group_acc(y_pred_clfrf)
print(comm_acc_clfrf)

#%%% RF Ethnic and Community Group Breakdown of Missclassified
miss_ethnic_clfrf = missclass_ethnic(y_pred_clfrf)
print(miss_ethnic_clfrf)

miss_comm_clfrf = missclass_comm(y_pred_clfrf)
print(miss_comm_clfrf)

# %% Multinomial Naive Bayes (Simple Training)
#%%% Model Training
clfmnb = MultinomialNB().fit(X_train, y_train)
y_pred_clfmnb = clfmnb.predict(X_test)

# %%% MNB Learning Curve
plot_learning_curve(clfmnb, "Learning Curve for Multinomial Naive Bayes Classifier")

# %%% MNB ROC
RocCurveDisplay.from_estimator(clfmnb, X_test, y_test)

# %%% MNB Confusion Matrix
plot_confusion_matrix(y_test, y_pred_clfmnb, "Reds", "Multinomial Naive Bayes Confusion Matrix")

#%%% LR Ethnic and Community group Accuracy
ethnic_acc_clfmnb = ethnic_based_acc(y_pred_clfmnb)
print(ethnic_acc_clfmnb)
comm_acc_clfmnb = comm_group_acc(y_pred_clfmnb)
print(comm_acc_clfmnb)

#%%% LR Ethnic and Community Group Breakdown of Missclassified
miss_ethnic_clfmnb = missclass_ethnic(y_pred_clfmnb)
print(miss_ethnic_clfmnb)

miss_comm_clfmnb = missclass_comm(y_pred_clfmnb)
print(miss_comm_clfmnb)
#%% Logistic Regression (Ethnic Isolated)
#%%% Just White
# %%%% Model Training
clflrw = LogisticRegression().fit(Xw_train, yw_train)
#Note: notation is y_pred_clflrw_b means white trained, black tested
y_pred_clflrw_w = clflrw.predict(Xw_test)
y_pred_clflrw_b = clflrw.predict(Xb_test)
y_pred_clflrw_a = clflrw.predict(Xa_test)
y_pred_clflrw_h = clflrw.predict(Xh_test)

# %%%%Logistic Regression Learning Curve
plot_learning_curve(clflr, "Learning Curve for Logistic Regression Classifier")

# %%% LR ROC
RocCurveDisplay.from_estimator(clflr, X_test, y_test)

#%%%LR Confusion Matrix
plot_confusion_matrix(y_test, y_pred_clflr, "Blues", "Logistic Regression Confusion Matrix")

#%%% LR Ethnic and Community group Accuracy
ethnic_acc_clflr = ethnic_based_acc(y_pred_clflr)
print(ethnic_acc_clflr)
comm_acc_clflr = comm_group_acc(y_pred_clflr)
print(comm_acc_clflr)

#%%% LR Ethnic and Community Group Breakdown of Missclassified
miss_ethnic_clflr = missclass_ethnic(y_pred_clflr)
print(miss_ethnic_clflr)

miss_comm_clflr = missclass_comm(y_pred_clflr)
print(miss_comm_clflr)
