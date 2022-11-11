# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 12:08:14 2022

@author: celestec
"""
# %%Imports and Functions
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

#%%% Random State
rando=1
# %%%learning Curve Function


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
    
# %%% Confusion Matrix Function

def plot_confusion_matrix(test, pred, color, title, fignum):
    cm=confusion_matrix(test,pred)
    plt.figure(fignum)
    ax=plt.subplot()
# annot=True to annotate cells, ftm='g' to disable scientific notation
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap=sns.color_palette(color));

# labels, title and ticks
    ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels');
    ax.set_title(title);
    ax.xaxis.set_ticklabels(['BV Positive', 'BV Negative']); ax.yaxis.set_ticklabels(
    ['BV Positive', 'BV Negative']);

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

            elif save.iloc[x, 0] == 'Asian':
                tracker[1]=tracker[1]+1

            elif save.iloc[x, 0] == 'Black':
                tracker[2]=tracker[2]+1

            elif save.iloc[x, 0] == 'Hispanic':
                tracker[3]=tracker[3]+1

        x=x+1
    ethnic_accuracy=pd.DataFrame(index=['Accuracy'])
    total_ethnic=es_xttest.value_counts()
    ethnic_accuracy['White']=tracker[0]/total_ethnic['White']
    ethnic_accuracy['Asian']=tracker[1]/total_ethnic['Asian']
    ethnic_accuracy['Black']=tracker[2]/total_ethnic['Black']
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

            elif save.iloc[x, 0] == 'Asian':
                tracker[1]=tracker[1]+1

            elif save.iloc[x, 0] == 'Black':
                tracker[2]=tracker[2]+1

            elif save.iloc[x, 0] == 'Hispanic':
                tracker[3]=tracker[3]+1
        x=x+1
    missclass_breakdown=pd.DataFrame(index=['Percentage'])
    missclass_breakdown['White']=100*tracker[0]/y
    missclass_breakdown['Asian']=100*tracker[1]/y
    missclass_breakdown['Black']=100*tracker[2]/y
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

# %%% Ethnic Trained Metrics
def ethnic_spec_metrics(classifier, xtrain, ytrain, predw, predb, preda, predh, predt, color,  ):
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
    
    return(ethnic_acc, comm_acc, miss_ethnic, miss_comm)

# %% Import and Clean Data
# %%% Import Data



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
#%%% Just White
# %%%% Model Training
clflrw = LogisticRegression().fit(Xw_train, yw_train)
#Note: notation is y_pred_clflrw_b means white trained, black tested
y_pred_clflrw_w = clflrw.predict(Xw_test)
y_pred_clflrw_b = clflrw.predict(Xb_test)
y_pred_clflrw_a = clflrw.predict(Xa_test)
y_pred_clflrw_h = clflrw.predict(Xh_test)
y_pred_clflrw_t = clflrw.predict(Xt_test)


#%%%% Model Metrics
clflrw_ethnic_acc, clflrw_comm_acc, clflrw_miss_ethnic, clflrw_miss_comm = ethnic_spec_metrics(
                    clflrw, Xw_train, yw_train, 
                    y_pred_clflrw_w,
                    y_pred_clflrw_b,
                    y_pred_clflrw_a,
                    y_pred_clflrw_h,
                    y_pred_clflrw_t,
                    "Blues"
                    )
#%%% Just Black
# %%%% Model Training
clflrb = LogisticRegression().fit(Xb_train, yb_train)
#Note: notation is y_pred_clflrw_b means white trained, black tested
y_pred_clflrb_w = clflrb.predict(Xw_test)
y_pred_clflrb_b = clflrb.predict(Xb_test)
y_pred_clflrb_a = clflrb.predict(Xa_test)
y_pred_clflrb_h = clflrb.predict(Xh_test)
y_pred_clflrb_t = clflrb.predict(Xt_test)


#%%%% Model Metrics
clflrb_ethnic_acc, clflrb_comm_acc, clflrb_miss_ethnic, clflrb_miss_comm = ethnic_spec_metrics(
                    clflrb, Xb_train, yb_train, 
                    y_pred_clflrb_w,
                    y_pred_clflrb_b,
                    y_pred_clflrb_a,
                    y_pred_clflrb_h,
                    y_pred_clflrb_t,
                    "Reds"
                    )

#%%% Just Asian
# %%%% Model Training
clflra = LogisticRegression().fit(Xa_train, ya_train)
#Note: notation is y_pred_clflrw_b means white trained, black tested
y_pred_clflra_w = clflra.predict(Xw_test)
y_pred_clflra_b = clflra.predict(Xb_test)
y_pred_clflra_a = clflra.predict(Xa_test)
y_pred_clflra_h = clflra.predict(Xh_test)
y_pred_clflra_t = clflra.predict(Xt_test)


#%%%% Model Metrics
clflra_ethnic_acc, clflra_comm_acc, clflra_miss_ethnic, clflra_miss_comm = ethnic_spec_metrics(
                    clflra, Xa_train, ya_train, 
                    y_pred_clflra_w,
                    y_pred_clflra_b,
                    y_pred_clflra_a,
                    y_pred_clflra_h,
                    y_pred_clflra_t,
                    "Greens"
                    )

#%%% Just Hispanic
# %%%% Model Training
clflrh = LogisticRegression().fit(Xh_train, yh_train)
#Note: notation is y_pred_clflrw_b means white trained, black tested
y_pred_clflrh_w = clflrh.predict(Xw_test)
y_pred_clflrh_b = clflrh.predict(Xb_test)
y_pred_clflrh_a = clflrh.predict(Xa_test)
y_pred_clflrh_h = clflrh.predict(Xh_test)
y_pred_clflrh_t = clflrh.predict(Xt_test)


#%%%% Model Metrics
clflrh_ethnic_acc, clflrh_comm_acc, clflrh_miss_ethnic, clflrh_miss_comm = ethnic_spec_metrics(
                    clflrh, Xh_train, yh_train, 
                    y_pred_clflrh_w,
                    y_pred_clflrh_b,
                    y_pred_clflrh_a,
                    y_pred_clflrh_h,
                    y_pred_clflrh_t,
                    "Purples"
                    )