# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:36:17 2022

@author: camer
"""
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
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve  
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
import seaborn as sns     

#%% Import Data
url = "https://raw.githubusercontent.com/ramenfeast/BV-ethnicity-report/main/BV%20Dataset%20copy.csv?token=GHSAT0AAAAAAB2MPZZ4KP7GV5IIS6TFUI64Y22WZUA"
download = requests.get(url).content

df = pd.read_csv(io.StringIO(download.decode('utf-8')))

print("Hi there")

#%%Clean data
df = df.drop([394,395,396], axis = 0)

#%% Separate the Data and Labels
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#%% Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=1)
#%% Extract Ethinic group and commmunity group data
es_xtest = X_test[['Ethnic Groupa', 'Community groupc ']].copy()
X_test=X_test.drop(labels= ['Ethnic Groupa', 'Community groupc '], axis=1)

es_xtrain = X_train[['Ethnic Groupa', 'Community groupc ']].copy()
X_train=X_train.drop(labels= ['Ethnic Groupa', 'Community groupc '], axis=1)

#%%Normalization

#Normalize pH
X_train['pH']=X_train['pH']/14
X_test['pH']=X_test['pH']/14

#Normalize 16s RNA data
X_train.iloc[:,1::]=X_train.iloc[:,1::]/100
X_test.iloc[:,1::]=X_test.iloc[:,1::]/100

#%%Binary y
y_train[y_train<7]=0
y_train[y_train>=7]=1

y_test[y_test<7]=0
y_test[y_test>=7]=1
#%% Logistic Regression
clflr = LogisticRegression().fit(X_train, y_train)
y_pred_clflr = clflr.predict(X_test)

#%%learning Curve
train_sizes, train_scores, validation_scores =learning_curve(estimator=clflr,
                               X=X_train,
                               y=y_train,
                               cv=5,
                               scoring = 'neg_mean_squared_error')

train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)
print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
print('\n', '-' * 20) # separator
print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))

plt.style.use('seaborn')
plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a linear regression model', fontsize = 18, y = 1.03)
plt.legend()

#%%Random Forest
clfrf = RandomForestClassifier().fit(X_train, y_train)
y_pred_clfrf = clfrf.predict(X_test)

#%% RF learning curve
train_sizes, train_scores, validation_scores =learning_curve(estimator=clfrf,
                               X=X_train,
                               y=y_train,
                               cv=5,
                               scoring = 'neg_mean_squared_error')

train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)
print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
print('\n', '-' * 20) # separator
print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))

plt.style.use('seaborn')
plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a random forest model', fontsize = 18, y = 1.03)
plt.legend()

#%% Naive Bayes
clfmnb = MultinomialNB().fit(X_train, y_train)
y_pred_clfmnb=clfmnb.predict(X_test)
#%% MNB Learning Curve
train_sizes, train_scores, validation_scores =learning_curve(estimator=clfmnb,
                               X=X_train,
                               y=y_train,
                               cv=5,
                               scoring = 'neg_mean_squared_error')

train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)
print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
print('\n', '-' * 20) # separator
print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))

plt.style.use('seaborn')
plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a Multinomial Naive-Bayes Model', fontsize = 18, y = 1.03)
plt.legend()

#%% LR ROC

fpr, tpr, thresholds = roc_curve(y_test, y_pred_clflr)
roc_auc = auc(fpr, tpr)

plot_roc_curve(clflr,X_test,y_test, color='blue')
#%% RF ROC

fpr, tpr, thresholds = roc_curve(y_test, y_pred_clfrf)
roc_auc = auc(fpr, tpr)

plot_roc_curve(clfrf,X_test,y_test, color='green')
#%% MNB ROC

fpr, tpr, thresholds = roc_curve(y_test, y_pred_clfmnb)
roc_auc = auc(fpr, tpr)

plot_roc_curve(clfmnb,X_test,y_test, color='red')

#%% LR Confusion Matrix

cm = confusion_matrix(y_test, y_pred_clflr)

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap=sns.color_palette("Blues"));  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Logistic Regression Confusion Matrix'); 
ax.xaxis.set_ticklabels(['BV Positive', 'BV Negative']); ax.yaxis.set_ticklabels(['BV Positive', 'BV Negative']);

#%% RF Confusion Matrix

cm = confusion_matrix(y_test, y_pred_clfrf)

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap=sns.color_palette("Greens"));  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Random Forest Confusion Matrix'); 
ax.xaxis.set_ticklabels(['BV Positive', 'BV Negative']); ax.yaxis.set_ticklabels(['BV Positive', 'BV Negative']);

#%% GNB Confusion Matrix

cm = confusion_matrix(y_test, y_pred_clfmnb)

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap=sns.color_palette("Reds"));  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Multinomial Naive-Bayes Confusion Matrix'); 
ax.xaxis.set_ticklabels(['BV Positive', 'BV Negative']); ax.yaxis.set_ticklabels(['BV Positive', 'BV Negative']);




#%% Ethnicity checked accuracy

