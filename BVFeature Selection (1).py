#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:



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


# In[4]:


plt.figure(figsize=(14,8))
sns.set_theme(style="white")
corr = df.corr()
heatmap = sns.heatmap(corr, cmap="coolwarm")


# In[5]:


# Loop over bottom diagonal of correlation matrix
for i in range(len(corr.columns)):
    for j in range(i):
 
        # Print variables with high correlation to each other (meaning you can eliminate one basically saying same thing)
        if abs(corr.iloc[i, j]) > 0.7:
            print(corr.columns[i], corr.columns[j], corr.iloc[i, j])


# In[6]:


#Add target column back in to run correlation to target
X_y = X_train.copy()
X_y['Nugent score'] = y_train
 
print(X_y)


# In[7]:


#Feature to target 
corr_matrix = X_y.corr()

# Isolate the column corresponding to `exam_score`
corr_target = corr_matrix[['Nugent score']].drop(labels=['Nugent score'])
 
#sns.heatmap(corr_target, annot=True, fmt='.3', cmap='RdBu_r')
#plt.show()

#Data is categorical for target of nugent not work well
print(corr_target)


# In[8]:


#Feature Selection: Outputs features above mean importance from train set
clfrf = RandomForestClassifier(n_estimators = 100,random_state=1)
clfrf.fit(X_train, y_train)
featnames = list(df.columns)

#print out gini scores for each feature
#for feature in zip(featnames, clfrf.feature_importances_):
    #print(feature)

#Use training set to avoid overfitting
#SelectFromModel select features importance is greater than the mean importance,number of trees 
selfeat = SelectFromModel(clfrf,threshold=0.0015)
selfeat.fit(X_train, y_train)

#threshold if values above or below, is mean of dataset
print(selfeat.threshold_)
print(selfeat.estimator_.feature_importances_.mean())

#array of boolean, True is features importance greater than mean, False features less importance than mean of gini impurity
selfeat.get_support()

#list and count selected features (greater than mean)
for feature_list_index in selfeat.get_support(indices=True):
    print(featnames[feature_list_index])


# In[9]:


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


# In[10]:


print(X_train)
print(X_imp_train)


# In[23]:


from sklearn.tree import DecisionTreeClassifier
 
#calculates gini gain (higher gain more important feature)
clf = DecisionTreeClassifier(criterion='gini')
 
# Fit the decision tree classifier
clf = clf.fit(X_train, y_train)

# Print the feature importances
feature_importances = clf.feature_importances_
#print(feature_importances)
 
# Sort the feature importances from greatest to least using the sorted indices
sorted_indices = feature_importances.argsort()[::-1]

#array of columns # in feature importance
print(sorted_indices)
#sorted_feature_names = dataset.feature_names[sorted_indices]

sorted_feature_names = X_train.columns[sorted_indices]
#array of names sorted accoridng to index of feature importance
print(sorted_feature_names)

#print(X_train.columns)

sorted_importances = feature_importances[sorted_indices]
#print(sorted_importances)

 
# Create a bar plot of the feature importances
sns.set(rc={'figure.figsize':(100,50)})
sns.barplot(x = sorted_importances, y = sorted_feature_names)
#sns.barplot(sorted_importances)

#Choose these top 10-15 features and only use these ones on the forward and backward models


# In[ ]:




