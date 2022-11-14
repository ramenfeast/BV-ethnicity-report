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
from sklearn.ensemble import StackingClassifier

#%%% Random State
rng = np.random.default_rng()
rando= rng.integers(0,10000000)
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


#%%% Confusion Matrix Function

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


#%%% Ethnic Pipeline
def ethnic_train_metric_pipe(classifier, xtrain, ytrain, color):

    if classifier == "Logistic Regression": 
        classify = LogisticRegression()
    elif classifier == "Random Forest":
        classify = RandomForestClassifier()
    elif classifier == "MNB":
        classify = MultinomialNB()
        
    
    classify.fit(xtrain, ytrain)

    y_pred_clfw = classify.predict(Xw_test)
    y_pred_clfb = classify.predict(Xb_test)
    y_pred_clfa = classify.predict(Xa_test)
    y_pred_clfh = classify.predict(Xh_test)
    y_pred_clft = classify.predict(Xt_test)

    clf_ethnic_acc, clf_comm_acc, clf_miss_ethnic, clf_miss_comm = ethnic_spec_metrics(
                    classify, xtrain, ytrain, 
                    y_pred_clfw,
                    y_pred_clfb,
                    y_pred_clfa,
                    y_pred_clfh,
                    y_pred_clft,
                    color
                    )
    return(classify)

#%%% Ethnic Based Stacking Classifier Pipeline
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

    clf_ethnic_acc, clf_comm_acc, clf_miss_ethnic, clf_miss_comm = ethnic_spec_metrics(
                    clf, Xt_train, yt_train, 
                    y_pred_clfw,
                    y_pred_clfb,
                    y_pred_clfa,
                    y_pred_clfh,
                    y_pred_clft,
                    color
                    )
    return(clf)

# %%% Community Group Trained Metrics
def comm_spec_metrics(classifier, xtrain, ytrain, predi, predii, prediii, prediv, predt, color,  ):
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
    plot_confusion_matrix(yi_test, predi, color, "Confusion Matrix",7)
    plot_confusion_matrix(yii_test, predii, color, "Confusion Matrix",8)
    plot_confusion_matrix(yiii_test, prediii, color, "Confusion Matrix",9)
    plot_confusion_matrix(yiii_test, prediv, color, "Confusion Matrix",10)
    plot_confusion_matrix(yiv_test, predt, color, "Confusion Matrix",11)

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


#%%% Comm Pipeline
def comm_train_metric_pipe(classifier, xtrain, ytrain, color):

    if classifier == "Logistic Regression": 
        classify = LogisticRegression()
    elif classifier == "Random Forest":
        classify = RandomForestClassifier()
    elif classifier == "MNB":
        classify = MultinomialNB()
        
    
    classify.fit(xtrain, ytrain)

    y_pred_clfi = classify.predict(Xi_test)
    y_pred_clfii = classify.predict(Xii_test)
    y_pred_clfiii = classify.predict(Xiii_test)
    y_pred_clfiv = classify.predict(Xiv_test)
    y_pred_clft = classify.predict(Xt_test)

    clf_ethnic_acc, clf_comm_acc, clf_miss_ethnic, clf_miss_comm = comm_spec_metrics(
                    classify, xtrain, ytrain, 
                    y_pred_clfi,
                    y_pred_clfii,
                    y_pred_clfiii,
                    y_pred_clfiv,
                    y_pred_clft,
                    color
                    )
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

    clf_ethnic_acc, clf_comm_acc, clf_miss_ethnic, clf_miss_comm = ethnic_spec_metrics(
                    clf, Xt_train, yt_train, 
                    y_pred_clfi,
                    y_pred_clfii,
                    y_pred_clfiii,
                    y_pred_clfiv,
                    y_pred_clft,
                    color
                    )
    return(clf)

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
clflrw = ethnic_train_metric_pipe("Logistic Regression", Xw_train, yw_train, "Blues")

#%%% Just Black
clflrb = ethnic_train_metric_pipe("Logistic Regression", Xb_train, yb_train, "Reds")

#%%% Just Asian
clflra = ethnic_train_metric_pipe("Logistic Regression", Xa_train, ya_train, "Greens")

#%%% Just Hispanic
clflrh = ethnic_train_metric_pipe("Logistic Regression", Xh_train, yh_train, "Purples")

#%%% Stack
clflrt = ethnic_stack_pipe(clflrh,clflrb,clflra,clflrh, "Oranges")

#%% Random Forest (Ethnic Isolated)
#%%% Just White
clfrfw = ethnic_train_metric_pipe("Random Forest", Xw_train, yw_train, "Blues")

#%%% Just Black
clfrfb = ethnic_train_metric_pipe("Random Forest", Xb_train, yb_train, "Reds")

#%%% Just Asian
clfrfa = ethnic_train_metric_pipe("Random Forest", Xa_train, ya_train, "Greens")

#%%% Just Hispanic
clfrfh = ethnic_train_metric_pipe("Random Forest", Xh_train, yh_train, "Purples")
#%%% Stack
clfrft = ethnic_stack_pipe(clfrfw,clfrfb,clfrfa,clfrfh, "Oranges")

#%% Multinomial Naive Bayes (Ethnic Isolated)
#%%% Just White
clfmnbw = ethnic_train_metric_pipe("MNB", Xw_train, yw_train, "Blues")

#%%% Just Black
clfmnbb = ethnic_train_metric_pipe("MNB", Xb_train, yb_train, "Reds")

#%%% Just Asian
clfmnba = ethnic_train_metric_pipe("MNB", Xa_train, ya_train, "Greens")

#%%% Just Hispanic
clfmnbh = ethnic_train_metric_pipe("MNB", Xh_train, yh_train, "Purples")
#%%% Stack
clfmnbt = ethnic_stack_pipe(clfmnbw,clfmnbb,clfmnba,clfmnbh, "Oranges")
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
clflri = ethnic_train_metric_pipe("Logistic Regression", Xi_train, yi_train, "Blues")

#%%% II
clflrii = ethnic_train_metric_pipe("Logistic Regression", Xii_train, yii_train, "Reds")

#%%% III
clflriii = ethnic_train_metric_pipe("Logistic Regression", Xiii_train, yiii_train, "Greens")

#%%% IV
clflriv = ethnic_train_metric_pipe("Logistic Regression", Xiv_train, yiv_train, "Purples")

#%%% Stack
clflrt = ethnic_stack_pipe(clflrh,clflrb,clflra,clflrh, "Oranges")

#%% Random Forest (Comm Isolated)
#%%% I
clfrfw = ethnic_train_metric_pipe("Random Forest", Xi_train, yi_train, "Blues")

#%%% II
clfrfb = ethnic_train_metric_pipe("Random Forest", Xii_train, yii_train, "Reds")

#%%% III
clfrfa = ethnic_train_metric_pipe("Random Forest", Xiii_train, yiii_train, "Greens")

#%%% IV
clfrfh = ethnic_train_metric_pipe("Random Forest", Xiv_train, yiv_train, "Purples")
#%%% Stack
clfrft = ethnic_stack_pipe(clfrfw,clfrfb,clfrfa,clfrfh, "Oranges")

#%% Multinomial Naive Bayes (Comm Isolated)
#%%% I
clfmnbw = ethnic_train_metric_pipe("MNB", Xi_train, yi_train, "Blues")

#%%% II
clfmnbb = ethnic_train_metric_pipe("MNB", Xii_train, yii_train, "Reds")

#%%% III
clfmnba = ethnic_train_metric_pipe("MNB", Xiii_train, yiii_train, "Greens")

#%%% IV
clfmnbh = ethnic_train_metric_pipe("MNB", Xiv_train, yiv_train, "Purples")
#%%% Stack
clfmnbt = ethnic_stack_pipe(clfmnbw,clfmnbb,clfmnba,clfmnbh, "Oranges")