#import tensorflow as tf
import utils
import numpy as np
import pandas as pd
from metrics import fairness_metrics, get_f1_p_r
from tensorflow.keras import datasets, layers, models, metrics
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
def mlp_model():
  model = models.Sequential()
  model.add(layers.Dense(100, activation='relu'))
  model.add(layers.Dense(100, activation='relu'))
  model.add(layers.Dense(1, activation='sigmoid'))
  return model
  
X,y = utils.get_XY()
ethnicities = utils.get_data()["Ethnic Groupa"].to_numpy()
X = X.to_numpy()
y = y.to_numpy()
stats = np.zeros((3,3,9))
i = 0
for kfold, (train, test) in enumerate(KFold(n_splits=3, 
                                shuffle=False).split(X, y)):
   #random forest
    RFC = RandomForestClassifier(n_estimators = 100, random_state=0)
    RFC.fit(X[train], y[train])
    
    #logistic regression
    log_model = LogisticRegression()
    log_model.fit(X[train], y[train])
    seq_model = mlp_model()
    seq_model.compile(loss='binary_crossentropy', 
                      optimizer='adam', 
                      metrics=['Accuracy'])
    #Fully connected Neural net
    seq_model.fit(X[train], y[train],
             epochs=20, 
             validation_data=(X[test], 
                              y[test]),
                  verbose = False)
    ethnic_fold = ethnicities[test]
    
    Whites = test[np.where(ethnic_fold=="White")]
    Blacks = test[np.where(ethnic_fold=="Black")]
    B_pred = seq_model.predict(X[Blacks]) > .5
    W_pred = seq_model.predict(X[Whites]) >.5
    B_pred_log = log_model.predict(X[Blacks]) > .5
    W_pred_log = log_model.predict(X[Whites]) >.5
    B_pred_RFC = RFC.predict(X[Blacks]) > .5
    W_pred_RFC = RFC.predict(X[Whites]) > .5


    B_true = y[Blacks]
    W_true = y[Whites]
    #metrics for fully connected"
    stats[0][i][0:3] = fairness_metrics(B_pred, W_pred , B_true,W_true)
    stats[0][i][3:9]= get_f1_p_r(B_pred, W_pred , B_true,W_true)
    #metrics for logistic regression
    stats[1][i][0:3] = fairness_metrics(B_pred_log, W_pred_log , B_true,W_true)
    stats[1][i][3:9]=  get_f1_p_r(B_pred_log, W_pred_log , B_true,W_true)
    
    stats[2][i][0:3]= fairness_metrics(B_pred_RFC, W_pred_RFC , B_true,W_true)
    stats[2][i][3:9]= get_f1_p_r(B_pred_RFC, W_pred_RFC , B_true,W_true)
    i+=1
avg = np.sum(stats,axis=1)/3
dataset = pd.DataFrame({'EOpp0': avg[:, 0],
                        'EOpp1': avg[:, 1],
                        'EOdd':avg[:,2],
                        'f1_score_B': avg[:, 3],
                        'f1_score_W': avg[:, 4],
                        'recall_B':avg[:,5],
                        'recall_W': avg[:, 6],
                        'precision_B': avg[:, 7],
                        'precision_W':avg[:,8],
                        })
dataset.index = ['Fully connected NN', 'Logistic Regression', 'Random Forest']
dataset.to_csv("example_stats.csv")
