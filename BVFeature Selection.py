#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import requests
import io
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve  
from sklearn.metrics import auc
import seaborn as sns   
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score


# In[11]:



url = "https://cdn.jsdelivr.net/gh/ramenfeast/BV-ethnicity-report/BV%20Dataset%20copy.csv"
download = requests.get(url).content
df = pd.read_csv(io.StringIO(download.decode('utf-8')))

#%%Clean data
df = df.drop([394,395,396], axis = 0)

#%% Separate the Data and Labels
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#%% Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=1)
#%% Extract Ethinic group and commmunity group data
es_xtest = X_test[['Ethnic Groupa']].copy()
cs_xtest = X_test[['Community groupc ']].copy()
X_test=X_test.drop(labels= ['Ethnic Groupa', 'Community groupc '], axis=1)


es_xtrain = X_train[['Ethnic Groupa']].copy()
cs_xtrain = X_train[['Community groupc ']].copy()
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


# In[12]:


plt.figure(figsize=(14,8))
sns.set_theme(style="white")
corr = df.corr()
heatmap = sns.heatmap(corr, cmap="coolwarm")


# In[13]:


#Feature Selection: Outputs features above mean importance from train set
clfrf = RandomForestClassifier(n_estimators = 100,random_state=1)
clfrf.fit(X_train, y_train)
featnames = list(df.columns)

#print out gini scores for each feature
#for feature in zip(featnames, clfrf.feature_importances_):
    #print(feature)

#Use training set to avoid overfitting
#SelectFromModel select features importance is greater than the mean importance,number of trees 
selfeat = SelectFromModel(clfrf) #,threshold=0.0015)
selfeat.fit(X_train, y_train)

#threshold if values above or below, is mean of dataset
print(selfeat.threshold_)
print(selfeat.estimator_.feature_importances_.mean())

#array of boolean, True is features greater than mean, False features less than mean of gini impurity
selfeat.get_support()

#list and count selected features (greater than mean)
for feature_list_index in selfeat.get_support(indices=True):
    print(featnames[feature_list_index])


# In[14]:


# Transform the data to create a new dataset containing only the most important features
# apply the transform to both the training X and test X data.
X_imp_train = selfeat.transform(X_train)
X_imp_test = selfeat.transform(X_test)

#Original RF
y_pred = clfrf.predict(X_test)
print(f'Accuracy Score RF = {accuracy_score(y_test, y_pred)}')

# Create a new RF for the most important features
clfrf_imp = RandomForestClassifier(n_estimators=100, random_state=1)
clfrf_imp.fit(X_imp_train, y_train)
y_pred_clfrf_imp = clfrf_imp.predict(X_imp_test)
print(f'Important Features Accuracy Score RF = {accuracy_score(y_test, y_pred_clfrf_imp)}')


# In[15]:


#print(X_train)
#print(X_imp_train)

