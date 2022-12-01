"""
Created on Wed Nov 30 22:55:26 2022

@author: camer
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
def class_train(classifier, xtrain, ytrain, random_state):

    if classifier == "Logistic Regression": 
        classify = LogisticRegression()
        solvers = ['newton-cg', 'lbfgs', 'liblinear']
        penalty = ['l2']
        c_values = [100, 10, 1.0, 0.1, 0.01]
        # define grid search
        grid = dict(solver=solvers,penalty=penalty,C=c_values)
        
        
    elif classifier == "Random Forest":
        classify = RandomForestClassifier()
        n_estimators = [10, 100, 1000]
        max_features = ['sqrt', 'log2']
        # define grid search
        grid = dict(n_estimators=n_estimators,max_features=max_features)
        
    elif classifier == "SVM":
        classify = SVC()
        kernel = ['poly', 'rbf', 'sigmoid']
        C = [50, 10, 1.0, 0.1, 0.01]
        gamma = ['scale']
        # define grid search
        grid = dict(kernel=kernel,C=C,gamma=gamma)
        
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=10, random_state=random_state)
    grid_search = GridSearchCV(estimator=classify, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    clf = grid_search.fit(xtrain, ytrain)
    print(clf.best_params_)
    
    return(clf)

from sklearn.ensemble import StackingClassifier

def ethnic_stack_train(clfw, clfb, clfa, clfh, Xt_train, yt_train):
    estimators = [('clfw',clfw), ('clfb', clfb), ('clfa',clfa), ('clfh', clfh)]
    clf = StackingClassifier(
        estimators = estimators, 
        final_estimator = LogisticRegression(), 
        cv= 'prefit')
    clf.fit(Xt_train,yt_train)
    return (clf)

def ethnic_pred(clf, Xw_test, Xb_test, Xa_test, Xh_test, Xt_test):
    
    y_pred_clfw = clf.predict(Xw_test),
    y_pred_clfb = clf.predict(Xb_test)
    y_pred_clfa = clf.predict(Xa_test)
    y_pred_clfh = clf.predict(Xh_test)
    y_pred_clft = clf.predict(Xt_test)
    
    return y_pred_clfw, y_pred_clfb, y_pred_clfa, y_pred_clfh, y_pred_clft
    

def comm_pred(clf, Xi_test, Xii_test, Xiii_test, Xiv_test, Xv_test, Xt_test):
    
    y_pred_clfi = clf.predict(Xi_test)
    y_pred_clfii = clf.predict(Xii_test)
    y_pred_clfiii = clf.predict(Xiii_test)
    y_pred_clfiv = clf.predict(Xiv_test)
    y_pred_clfv = clf.predict(Xv_test)
    y_pred_clft = clf.predict(Xt_test)
    
    return y_pred_clfi, y_pred_clfii, y_pred_clfiii, y_pred_clfiv, y_pred_clfv, y_pred_clft

def comm_group_stack_train(clfi, clfii, clfiii, clfiv, clfv, Xt_train, yt_train):
    estimators = [('clfi',clfi), ('clfii', clfii), ('clfiii',clfiii), ('clfiv', clfiv), ('clfv', clfv)]
    clf = StackingClassifier(
        estimators = estimators, 
        final_estimator = LogisticRegression(), 
        cv= 'prefit')
    clf.fit(Xt_train,yt_train)
    return (clf)
