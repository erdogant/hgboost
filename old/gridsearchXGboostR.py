# The process of performing random search with cross validation is:
# 1. Set up a grid of hyperparameters to evaluate
# 2. Randomly sample a combination of hyperparameters
# 3. Create a model with the selected combination
# 4. Evaluate the model using cross validation
# 5. Decide which hyperparameters worked the best

#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
#https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
#https://xgboost.readthedocs.io/en/latest/parameter.html
#--------------------------------------------------------------------------
# Name        : gridsearchGradientBoosting.py
# Version     : 1.0
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : Dec. 2018
#--------------------------------------------------------------------------
#
'''
NOTE:
IF you see something like this:    training data did not have the following fields: f73, f40, f66, f147, f62, f39, f2, f83, f127, f84, f54, f97, f114, f102, f49, f7, f8, f56, f23, f107, f138, f28, f71, f152, f80, f57, f46, f58, f139, f121, f140, f20, f45, f113, f5, f60, f135, f101, f68, f76, f65, f41, f99, f131, f109, f117, f13, f100, f128, f52, f15, f50, f95, f124, f19, f12, f43, f137, f33, f22, f32, f72, f142, f151, f74, f90, f48, f122, f133, f26, f79, f94, f18, f10, f51, f0, f53, f92, f29, f115, f143, f14, f116, f47, f69, f82, f34, f89, f35, f6, f132, f16, f118, f31, f96, f59, f75, f1, f110, f61, f108, f25, f21, f11, f17, f85, f150, f3, f98, f24, f77, f103, f112, f91, f144, f70, f86, f119, f55, f130, f106, f44, f36, f64, f67, f4, f145, f37, f126, f88, f93, f104, f81, f149, f27, f136, f146, f30, f38, f42, f141, f134, f120, f105, f129, f9, f148, f87, f125, f123, f111, f78, f63
Then, it may be caused by the incompatibility of sklearn's CalibratedClassifierCV and pandas.DataFrame

Or your data has 0 in it!
Just replace the last element with a very small number, like so:
X=X.replace(0,0.0000001)

https://github.com/dmlc/xgboost/issues/2334

'''

#%% Libraries
import xgboost
#from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split

#%% Gridsearch for GradientBoostingRegressor
def gridsearchXGboostR(X, y, cv=10, n_iter=20, n_jobs=1, verbose=True):
    if verbose==True: verbose=2
    
    n_jobs=np.maximum(n_jobs,1)

#    print "Checkinf for NaN and Inf" 
#    print "np.inf=", np.where(np.isnan(X))
#    print "is.inf=", np.where(np.isinf(X)) 
#    print "np.max=", np.max(abs(X))

#    [X_train, X_test, y_train, y_test] = train_test_split(X.iloc[:-1,:].values, y.iloc[:-1].values, train_size=0.8, test_size=0.2)

    min_child_weight = [0.5, 1.0, 3.0, 5.0, 7.0, 10.0]

    n_estimators = [100, 250, 300, 500]

    gamma =  [0, 0.25, 0.5, 1.0]

    subsample = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Maximum depth of each tree
    max_depth = [2, 3, 4, 5, 10, 15]

    silent = [False]
    
    learning_rate = [0.001, 0.01, 0.1, 0.2, 0,3]
    
    colsample_bylevel = [0.4, 0.6, 0.8, 1.0]

    colsample_bytree = [0.4, 0.6, 0.8, 1.0]

    reg_lambda = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
    
    num_round=[10,50,100]

    # Control the balance of positive and negative weights, useful for unbalanced classes.
    scale_pos_weight = [1]

    hyperparameter_grid = {
#            'min_child_weight': min_child_weight,
            'n_estimators': n_estimators,
            'gamma': gamma,
            'subsample': subsample,
            'max_depth': max_depth,
            'silent': silent,
            'learning_rate': learning_rate,
            'colsample_bylevel': colsample_bylevel,
            'colsample_bytree': colsample_bytree,
            'reg_lambda': reg_lambda,
            'scale_pos_weight': scale_pos_weight,
#                           'num_round':num_round,
                           }

    
    # Create the model to use for hyperparameter tuning
    model = xgboost.XGBRegressor()
	
    
    # Set up the random search with 5-fold cross validation
    random_cv = RandomizedSearchCV(model, 
                                   hyperparameter_grid, 
                                   cv=cv,
                                   n_iter=n_iter,
                                   n_jobs=n_jobs, 
                                   verbose=verbose,
                                   scoring='neg_mean_absolute_error', #neg_mean_squared_error
                                   return_train_score = False,
                                   refit=True, #Refit an estimator using the best found parameters on the whole dataset.
                                   )


    # Fit on the training data
#    random_cv = xgboost.XGBRegressor()
#    X.dropna(inplace=True)
#    y.dropna(inplace=True)    
#    X = X.fillna(X.mean())
#    np.where(X.values >= np.finfo(np.float64).max)    
#    np.isnan(X.values.any())
#    col_mask=X.isnull().any(axis=0).sum()
#    row_mask=X.isnull().any(axis=1).sum()
#    X[X==np.inf]=np.nan
#    X.fillna(X.mean(), inplace=True)
 
#    IND=X.asmatrix(columns=['ColumnA', 'ColumnB'])
#    np.isnan(IND).any()
    if 'pandas' in str(type(X)):
        X = X.as_matrix().astype(np.float)
    if 'pandas' in str(type(y)):
        y = y.as_matrix().astype(np.float)


    search_time_start = time.time()
    
    random_cv.fit(X, y)
    
    # Show some results:
    if verbose:
        print("Randomized search time:", time.time() - search_time_start)
        report(random_cv.cv_results_)

    # Find the best combination of settings
    model=random_cv.best_estimator_
#    random_cv.best_score_
#    random_cv.best_params_
#    random_cv.best_index_
#    random_cv.cv_results_['params'][search.best_index_]
#    random_results = pd.DataFrame(random_cv.cv_results_).sort_values('mean_test_score', ascending = False)
#    bestparams=random_cv.cv_results_['params'][random_cv.best_index_]

    return(model,random_cv)

#%% Report best scores
def report(results, n_top=5):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(results['mean_test_score'][candidate], results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

#%% END