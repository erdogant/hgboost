# The process of performing random search with cross validation is:
# 1. Set up a grid of hyperparameters to evaluate
# 2. Randomly sample a combination of hyperparameters
# 3. Create a model with the selected combination
# 4. Evaluate the model using cross validation
# 5. Decide which hyperparameters worked the best

#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
#https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
#https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/
#--------------------------------------------------------------------------
# Name        : gridsearchGradientBoosting.py
# Version     : 1.0
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : Dec. 2018
#--------------------------------------------------------------------------
#
#    MODEL THAT CAN ALSO BE TESTED
#    from sklearn.ensemble import GradientBoostingRegressor#, AdaBoostRegressor
#    model = GradientBoostingRegressor(n_estimators=100)
#    from sklearn.tree import DecisionTreeRegressor
#    from sklearn.linear_model import LinearRegression
#    model = LinearRegression()
#    model  = AdaBoostRegressor(n_estimators=100)
#    from sklearn.tree import DecisionTreeRegressor #, ExtraTreeRegressor
#    model = DecisionTreeRegressor()
#    model = ExtraTreeRegressor()
#    
#%% Libraries
#import xgboost
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

#%% Gridsearch for GradientBoostingRegressor
def gridsearchGradientBoostingR(X, y, n_jobs=1, verbose=True):
    if verbose==True: verbose=2
    cv=10
    n_iter=100
#    cv=2
#    n_iter=10
    n_jobs=np.maximum(n_jobs,1)


    if 'pandas' in str(type(X)):
        X = X.as_matrix().astype(np.float)
    if 'pandas' in str(type(y)):
        y = y.as_matrix().astype(np.float)

    
    # Loss function to be optimized (minimize)
    loss = ['ls', 'lad', 'huber', 'quantile']
    
    # Number of weak learnes (trees) used in the boosting process
    n_estimators = [100, 250, 300, 500, 600, 750]
    
    # Maximum depth of each tree
    max_depth = [2, 3, 5, 10, 15]
    
    # Minimum number of samples per leaf
    min_samples_leaf = [1, 2, 4, 6, 8, 10]
    
    # Minimum number of samples to split a node
    min_samples_split = [2, 4, 6, 10, 12]
    
    # Maximum number of features to consider for making splits
    max_features = ['auto', 'sqrt', 'log2', None]

    # Maximum number of features to consider for making splits
    criterion  = ['friedman_mse', 'mse']
    
    #%% Make the grid.
    hyperparameter_grid = {'loss': loss,
                           'n_estimators': n_estimators,
                           'max_depth': max_depth,
                           'min_samples_leaf': min_samples_leaf,
                           'min_samples_split': min_samples_split,
                           'max_features': max_features,
                           'criterion': criterion}
    
    # Create the model to use for hyperparameter tuning
    model = GradientBoostingRegressor()
#    model = xgboost.XGBRegressor()
	
    # Set up the random search with 5-fold cross validation
    random_cv = RandomizedSearchCV(estimator=model,
                                   param_distributions=hyperparameter_grid,
                                   cv=cv, 
                                   n_iter=n_iter, 
                                   scoring = 'neg_mean_absolute_error',
                                   n_jobs = n_jobs, 
                                   verbose = verbose, 
                                   return_train_score = True,
                                   refit=True, #Refit using the best found parameters on the whole dataset.
                                   )
        
    # Fit on the training data
    random_cv.fit(X, y)
    
    # Show some results:
    if verbose:
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

#%% Setup a gridsearch for only the trees
    # Create a range of trees to evaluate
#    from sklearn.model_selection import GridSearchCV
#    trees_grid = {'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 1000]}
#    model = GradientBoostingRegressor(loss = bestparams['loss'], 
#                                      max_depth = bestparams['max_depth'],
#                                      min_samples_leaf = bestparams['min_samples_leaf'],
#                                      min_samples_split = bestparams['min_samples_split'],
#                                      max_features = bestparams['max_features'])
#    
#    # Grid Search Object using the trees range and the random forest model
#    grid_search = GridSearchCV(estimator = model, param_grid=trees_grid, 
#                               cv = cv, 
#                               scoring = 'neg_mean_absolute_error', 
#                               verbose = verbose,
#                               n_jobs = n_jobs, 
#                               return_train_score = True)
#    # Fit the grid search
#    grid_search.fit(X, y)
#    # Get the results into a dataframe
#    results = pd.DataFrame(grid_search.cv_results_)
#    
#    # Plot the training and testing error vs number of trees
#    plt.figure()
#    plt.style.use('fivethirtyeight')
#    plt.plot(results['param_n_estimators'], -1 * results['mean_test_score'], label = 'Testing Error')
#    plt.plot(results['param_n_estimators'], -1 * results['mean_train_score'], label = 'Training Error')
#    plt.xlabel('Number of Trees'); plt.ylabel('Mean Abosolute Error'); plt.legend();
#    plt.title('Performance vs Number of Trees');
#
#    results.sort_values('mean_test_score', ascending = False).head(5)
#
#    # Select the best model
#    final_model = grid_search.best_estimator_

