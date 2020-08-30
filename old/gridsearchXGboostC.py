# The process of performing random search with cross validation is:
# 1. Set up a grid of hyperparameters to evaluate
# 2. Randomly sample a combination of hyperparameters
# 3. Create a model with the selected combination
# 4. Evaluate the model using cross validation
# 5. Decide which hyperparameters worked the best

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/
#--------------------------------------------------------------------------
# Name        : gridsearchXGboostC.py
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
'''
   from TRANSFORMERS.df2onehot import df2onehot
   import SUPERVISED.twoClassSummary as twoClassSummary

   df=pd.read_csv('../DATA/OTHER/titanic/titanic_train.csv')
   dfc=df2onehot(df)[0]
   dfc.dropna(inplace=True)
   
   y=dfc['Survived'].astype(float).values
   del dfc['Survived']
   X=dfc

   from SUPERVISED.gridsearchXGboostC import gridsearchXGboostC
   model = gridsearchXGboostC(X, y)
   twoClassSummary.allresults(model['y_test'], model['y_pred_proba'][:,1])

   from SUPERVISED.gridsearchGradientBoostingC import gridsearchGradientBoostingC
   model = gridsearchGradientBoostingC(X, y)
   twoClassSummary.allresults(model['y_test'], model['y_pred_proba'][:,1])
   
#   import STATS.genetic as genetic
#   [_,dfc,_,_]=df2onehot(df)
#   A = genetic.thompson(dfc.astype(int))
#   A = genetic.UCB(dfc)
#   A = genetic.randomUCB(dfc)
   
   
'''
#%% Libraries
#import xgboost
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.model_selection import RandomizedSearchCV
#import numpy as np
#from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import make_scorer
##from sklearn.metrics import balanced_accuracy_score

from xgboost import XGBClassifier

#from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
#import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
#import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

#%% Gridsearch for classification
def gridsearchXGboostC(X, y, cv=5, n_iter=50, scoring='f1',n_jobs=6, showfig=False, verbose=3):
    if verbose==True: verbose=2
#    cv=2
#    n_iter=10
    n_jobs=np.maximum(n_jobs,1)

    if verbose>=3: print('[GRIDSEARCH XGBOOST] Scoring type: %s' %scoring)
    if 'pandas' in str(type(X)):
        X = X.values.astype(np.float)
#    if 'pandas' in str(type(y)):
#        y = y.values.astype(np.float)

    if scoring=='kappa':
        scoring = make_scorer(cohen_kappa_score, greater_is_better=True)
           
    #%% Make train and validation dataset
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.2)

    #%% Parameter to be tweaked
    # Though there are 2 types of boosters, I’ll consider only tree booster here because it always outperforms the linear booster and thus the later is rarely used.
    booster = 'gbtree'
    
    # Analogous to learning rate in GBM
    learning_rate = [0.01, 0.05, 0.1, 0.15, 0.2]
    
    # Defines the minimum sum of weights of all observations required in a child.
    # Used to control over-fitting.
    # Too high values can lead to under-fitting
    # Default:1
    min_child_weight = [1,3,5,7]
    
    # Maximum depth of each tree (default:6)
    max_depth = [2, 3, 6, 10]
    
    # Same as the subsample of GBM. Denotes the fraction of observations to be randomly samples for each tree.
    # values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting.
    subsample  =[0.5, 0.65, 0.8, 0.9, 0.95, 1.0]

    gamma=[0.5, 1, 1.5, 2, 5]
    
    # For classification
    eval_metric = ['error', 'auc']
    
    colsample_bytree=[ 0.3, 0.4, 0.5 , 0.7 ]
    
    scale_pos_weight = [0, 0.5, 1]

    # Number of weak learnes (trees) used in the boosting process
    n_estimators = [10, 25, 100, 250]

    if len(np.unique(y))<=2:
        objective='binary:logistic'
    else:
        objective='multi:softprob'
    
    #%% A parameter grid for XGBoost
    hyperparameter_grid = {
            'learning_rate':learning_rate,
            'min_child_weight': min_child_weight,
            'max_depth': max_depth,
            'subsample': subsample,
            'gamma': gamma,
            'eval_metric':eval_metric,
            'colsample_bytree': colsample_bytree,
            'scale_pos_weight':scale_pos_weight,
            'n_estimators':n_estimators,
            }

    #%% Parameter to be tweaked

    # Stratified folds
#    skf = StratifiedKFold(n_splits=3, shuffle = True, random_state=1001)

    # Setup classifier defaults
    xgb = XGBClassifier(objective=objective, booster=booster, early_stopping_rounds=10, silent=True)
#    xgb = XGBClassifier(objective=objective, booster=booster, early_stopping_rounds=10)

    # Set up the random search with 5-fold cross validation
    random_cv = RandomizedSearchCV(estimator=xgb, 
                                   param_distributions=hyperparameter_grid, 
                                   n_iter=n_iter, 
                                   scoring=scoring, 
                                   n_jobs=n_jobs, 
                                   cv=cv,
#                                   cv=skf.split(X,y), 
                                   verbose=verbose, 
                                   return_train_score=True,
                                   refit=True,
                                   random_state=1001)
    
    #%% Run
    # Fit on the training data
#    random_cv.fit(X_train, y_train)
    random_cv.fit(X_train, y_train)
    # Get the best parameter combination
    model=random_cv.best_estimator_

    #%% Show some results:
    if verbose>=3:
        report(random_cv.cv_results_)

#    random_cv.best_score_ 
#    random_cv.best_params_ 
#    random_cv.best_index_ 
#    random_cv.cv_results_['params'][search.best_index_]
#    random_results = pd.DataFrame(random_cv.cv_results_).sort_values('mean_test_score', ascending = False)
#    bestparams=random_cv.cv_results_['params'][random_cv.best_index_]
    
    #%%
#    xgb_model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=False)
#    xgb.plot_importance(xgb_model)
    
    # plot the output tree via matplotlib, specifying the ordinal number of the target tree
    # xgb.plot_tree(xgb_model, num_trees=xgb_model.best_iteration)
    
    # converts the target tree to a graphviz instance
#    xgb.to_graphviz(xgb_model, num_trees=xgb_model.best_iteration)

    #%% Test on indepdendent validation set
    y_pred  = model.predict(X_test)
    y_pred_proba    = model.predict_proba(X_test)
    kappscore = cohen_kappa_score(y_test, y_pred)
    
#    outConf=None
#    outROC=None
#    if showfig:
#        from VIZ.ROCplot import ROCplot
#        import SUPERVISED.confmatrix as confmatrix
#        print('Score corrected for in-balanced classes: %.2f' %kappscore)
#        outROC=ROCplot(y_test,y_pred[:,1])
#        outConf=confmatrix.twoclass(y_test,y_pred[:,1])
#        makefig(random_cv)
        
    #%% Return
    out              = dict()
    out['model']     = model
    out['random_cv'] = random_cv
#    out['X_test']    = X_test
    out['y_test']    = y_test
    out['y_pred_proba']    = y_pred_proba
    out['y_pred'] = y_pred
    out['kappa']     = kappscore
#    out['confusion'] = outConf
#    out['stats']     = outROC

    return(out)

#%% Report best scores
def report(results, n_top=5):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(results['mean_test_score'][candidate], results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def makefig(random_cv):
    random_results = pd.DataFrame(random_cv.cv_results_).sort_values('mean_test_score', ascending = False)
    fig, ax = plt.subplots(figsize=(15,8))
    plt.errorbar(np.arange(random_results.shape[0]), random_results.mean_test_score.values, random_results.std_test_score.values, marker='s', mfc='red',mec='green', ms=20, mew=4)
    plt.xlabel('Number of iterations')
    plt.ylabel('Score')
    plt.title('Scoring %s (higher is better)')
    plt.show()

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

