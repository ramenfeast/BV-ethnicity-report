"""
Created on Wed Nov 30 22:55:26 2022

@author: camer
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def class_train(classifier, xtrain, ytrain):

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
        
    #cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=3, random_state=rando)
    grid_search = GridSearchCV(estimator=classify, param_grid=grid, n_jobs=-1, cv=10, scoring='accuracy',error_score=0)
    clf = grid_search.fit(xtrain, ytrain)
    print(clf.best_params_)
    
    return(clf)

def ethnic_pred(clf):
    y_pred_clfw = clf.predict(Xw_test)
    y_pred_clfb = clf.predict(Xb_test)
    y_pred_clfa = clf.predict(Xa_test)
    y_pred_clfh = clf.predict(Xh_test)
    y_pred_clft = clf.predict(Xt_test)

def comm_pred(clf):
    y_pred_clfi = clf.predict(Xi_test)
    y_pred_clfii = clf.predict(Xii_test)
    y_pred_clfiii = clf.predict(Xiii_test)
    y_pred_clfiv = clf.predict(Xiv_test)
    y_pred_clft = clf.predict(Xt_test)