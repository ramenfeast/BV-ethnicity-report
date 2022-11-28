#import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from metrics import fairness
from xgboost import XGBClassifier
from sklearn import svm
import utils
import numpy as np
import pandas as pd
from tensorflow.keras import datasets, layers, models, metrics
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
def mlp_model():
  model = models.Sequential()
  
  model.add(layers.Dense(512, activation='relu'))
  model.add(layers.Dense(256, activation='relu'))

  model.add(layers.Dense(1, activation='sigmoid'))
  return model
  
X,y = utils.get_XY()
ethnicities = utils.get_data()["Ethnic Groupa"].to_numpy()
X = X.to_numpy()
y = y.to_numpy()
stats = np.zeros((5,3,4))
i = 0
for kfold, (train, test) in enumerate(StratifiedKFold(n_splits=3, 
                                shuffle=False).split(X,np.array(list(map(str,y))) + ethnicities)):
   #random forest
    RFC = RandomForestClassifier(n_estimators = 100, random_state=0)
    RFC.fit(X[train], y[train])
    #SVM
    svm_model = svm.LinearSVC()
    svm_model.fit(X[train], y[train])
    #logistic regression
    log_model = LogisticRegression()
    log_model.fit(X[train], y[train])
    seq_model = mlp_model()
    seq_model.compile(loss='binary_crossentropy', 
                      optimizer='adam', 
                      metrics=['Accuracy'])
    #Fully connected Neural net
    seq_model.fit(X[train], y[train],
             epochs=30, 
             validation_data=(X[test], 
                              y[test]),
                  verbose = False)
    ethnic_fold = ethnicities[test]
    #xgboost
    bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
    bst.fit(X[train], y[train])

    y_true = y[test]
    #metrics for fully connected"
    
    stats[0][i][0:3] =fairness(seq_model, X[test], y[test],ethnicities[test])
    stats[0][i][3] = f1_score(y_true,seq_model.predict(X[test]) >.5)
    #metrics for logistic regression
   
    stats[1][i][0:3] =fairness(log_model, X[test], y[test],ethnicities[test])
    stats[1][i][3] = f1_score(y_true,log_model.predict(X[test]) >.5)
    #metrics for random forest
    stats[2][i][0:3]= fairness(RFC, X[test], y[test],ethnicities[test])
    stats[2][i][3] = f1_score(y_true,RFC.predict(X[test]) >.5)
    #metrics for svm
    stats[3][i][0:3]=fairness(svm_model, X[test], y[test],ethnicities[test])
    stats[3][i][3] = f1_score(y_true,svm_model.predict(X[test]) >.5)

    #metrics for xgboost
    stats[4][i][0:3]= fairness(bst, X[test], y[test],ethnicities[test])
    stats[4][i][3] = f1_score(y_true,bst.predict(X[test]) >.5)
    i+=1
avg = np.sum(stats,axis=1)/3
dataset = pd.DataFrame({'EOpp0': avg[:, 0],
                        'EOpp1': avg[:, 1],
                        'EOdd':avg[:,2],
                        'f1': avg[:, 3],
             
                        })
dataset.index = ['Fully connected NN', 'Logistic Regression', 'Random Forest', 'svm','bst']
dataset.to_csv("strafied_kfold_metrics.csv")
