from sklearn.metrics import recall_score, precision_score,f1_score
import numpy as np
"""
metric implemented from
 Y. Wu, D. Zeng, X. Xu, Y. Shi, and J. Hu, 
 “Fairprune: Achieving fairness through pruning for&nbsp;dermatological disease diagnosis,”
  Lecture Notes in Computer Science, pp. 743–753, 2022. 
"""

def false_positives(y_pred, y_true):
  return np.sum(y_pred * (y_true==False))
def true_positives(y_pred, y_true):
  return np.sum(y_pred*y_true)
def true_negatives(y_pred,y_true):
  return np.sum((False==y_pred)* (False==y_true))
def false_negatives(y_pred,y_true):
  return np.sum((False==y_pred)*y_true)
#df["Ethnic Groupa"]

def TPR(y_pred, y_true):
  TP = true_positives(y_pred, y_true)
  FN = false_negatives(y_pred, y_true)
  if (TP+FN) == 0:
    return 0
  return TP/(TP+FN)

def FPR(y_pred, y_true):
  FP = false_positives(y_pred, y_true)
  TN = true_negatives(y_pred, y_true)
  if (FP+TN) == 0:
    return 0
  return FP/(TN+FP)

def TNR(y_pred, y_true):
  TN = true_positives(y_pred, y_true)
  FP = false_positives(y_pred, y_true)
  if (TN+FP) == 0:
    return 0
  return TN/(TN+FP)
def EOpp0(TNR_1,TNR_0):
  return np.abs(TNR_1-TNR_0)

def EOpp1(TPR_1,TPR_0):
  return np.abs(TPR_1-TPR_0)

def EOdd(TPR_1, TPR_0, FPR_1, FPR_0):
  return np.abs(TPR_1- TPR_0+ FPR_1 - FPR_0)

def fairness_metrics(y_pred0, y_pred1, y_true0, y_true1):
  ret = np.zeros(3)
  TNR_1 = TNR(y_pred1, y_true1)
  TNR_0 = TNR(y_pred0, y_true0)
  TPR_1 = TPR(y_pred1, y_true1)
  TPR_0 = TPR(y_pred0, y_true0)
  FPR_1 = FPR(y_pred1, y_true1)
  FPR_0 = FPR(y_pred0, y_true0)
  ret[0] = EOpp0(TNR_1,TNR_0)
  ret[1] = EOpp1(TPR_1,TPR_0)
  ret[2] = EOdd(TPR_1, TPR_0, FPR_1, FPR_0)
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



