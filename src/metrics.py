from sklearn.metrics import recall_score, precision_score,f1_score
import numpy as np
import itertools
"""
metric implemented from
 Y. Wu, D. Zeng, X. Xu, Y. Shi, and J. Hu,
 “Fairprune: Achieving fairness through pruning for&nbsp;dermatological disease diagnosis,”
  Lecture Notes in Computer Science, pp. 743–753, 2022.
"""

def false_positives(y_true,y_pred):
  return np.sum(y_pred * (y_true==False))
def true_positives(y_true,y_pred):
  return np.sum(y_pred*y_true)
def true_negatives(y_true,y_pred):
  return np.sum((False==y_pred)* (False==y_true))
def false_negatives(y_true,y_pred):
  return np.sum((False==y_pred)*y_true)


def TPR(y_true,y_pred):
  TP = true_positives(y_true,y_pred)
  FN = false_negatives(y_true,y_pred)
  if (TP+FN) == 0:
    return 0
  return TP/(TP+FN)

def FPR(y_true,y_pred):
  FP = false_positives(y_true,y_pred)
  TN = true_negatives(y_true,y_pred)
  if (FP+TN) == 0:
    return 0
  return FP/(TN+FP)

def TNR(y_true,y_pred):
  TN = true_positives( y_true,y_pred)
  FP = false_positives(y_true,y_pred)
  if (TN+FP) == 0:
    return 0
  return TN/(TN+FP)
def EOpp0(y_true0, y_true1, y_pred0, y_pred1,k):
  ret = 0
  for i in range(0,k):
    ret+= np.abs(TNR(y_true1==i, y_pred1==i)- TNR(y_true0==i, y_pred0==i))
  return ret

def EOpp1(y_true0, y_true1, y_pred0, y_pred1,k):
  ret = 0
  for i in range(0,k):
    ret+= np.abs(TPR(y_true1==i, y_pred1==i)- TPR(y_true0==i, y_pred0==i))
  return ret

def EOdd(y_true0, y_true1, y_pred0, y_pred1,k):
  ret = 0
  for i in range(0,k):
    ret+= np.abs(TPR(y_true1==i, y_pred1==i)-
                 TPR(y_true0==i, y_pred0==i) +
                 FPR(y_true1==i, y_pred1==i)-
                 FPR(y_true0==i, y_pred0==i))
  return ret

def fairness_metrics(y_true0, y_true1, y_pred0, y_pred1):
  ret = np.zeros(3)
  ret[0] = EOpp0(y_true0, y_true1, y_pred0, y_pred1,2)
  ret[1] = EOpp1(y_true0, y_true1, y_pred0, y_pred1,2)
  ret[2] = EOdd(y_true0, y_true1, y_pred0, y_pred1,2)
  return ret
def get_f1_p_r(y_pred0, y_pred1, y_true0, y_true1):
  ret = np.zeros(6)
  ret[0] = f1_score(y_true0,y_pred0)
  ret[1] = f1_score(y_true1,y_pred1)
  ret[2] = recall_score(y_true0,y_pred0)
  ret[3] = recall_score(y_true1,y_pred1)
  ret[4]= precision_score(y_true0,y_pred0)
  ret[5] = precision_score(y_true1,y_pred1)
  return ret

def fairness(model, x_test, y_test, groups_test):
  predictions = model.predict(x_test)>.5
  ret = np.zeros(3)
  i = 0
  for each in itertools.combinations(np.unique(groups_test),2):
    group0 = np.where(groups_test==each[0])
    group1 = np.where(groups_test==each[1])
    ret += fairness_metrics(y_test[group0],
                            y_test[group1],
                            predictions[group0],
                            predictions[group1])
  return ret



