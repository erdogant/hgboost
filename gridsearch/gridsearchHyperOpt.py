""" This function optimizes XGBoost, LightGBM and CatBoost with Hyperopt

	model = gridsearchHyperOpt.fit(X, y)
	out   = gridsearchHyperOpt.predict(data)
	        gridsearchHyperOpt.plot(out)
    
    model = gridsearchHyperOpt.refit(X,y,model)

 INPUT:
   X:             [pd.DataFrame] Pandas DataFrame for which the columns should match response variable y

                      f1  ,f2  ,f3
                   s1 0   ,0   ,1
                   s2 0   ,1   ,0
                   s3 1   ,1   ,0
  
    y              [numpy array] Vector of labels
                   [0,1,0,1,1,2,1,2,2,2,2,0,0,1,0,1,..]
                   ['aap','aap','boom','mies','boom','aap',..]


 OPTIONAL

   method:         [string]: Classifier to be used
                   'xgb_reg' : (default) XGboost regressor
                   'lgb_reg' : LGBM Regressor
                   'ctb_reg' : CatBoost Regressor
                   'xgb_clf' : XGboost Classifier (two-class or multi-class classifier is automatically choosen based on the number of classes)

   max_evals=      [integer]: Search space is created on the number of evaluations
                   100: (default)
   
   verbose:        Integer [0..5] if verbose >= DEBUG: print('debug message')
                   0: (default)
                   1: ERROR
                   2: WARN
                   3: INFO
                   4: DEBUG

 OUTPUT
	output

 DESCRIPTION
   1. define the parameter spaces for all three libraries:
       XGBRegressor()
       LGBMRegressor()
       CatBoostRegressor()
   2. Create a class HPOpt that is instantiated with training and testing data and provides the training functions.

 INFO
   https://towardsdatascience.com/an-introductory-example-of-bayesian-optimization-in-python-with-hyperopt-aae40fff4ff0

 REQUIREMENTS
   hyperopt
   lightgbm
   catboost
   xgboost
 

 EXAMPLE
   %reset -f
   %matplotlib auto
   import pandas as pd
   from TRANSFORMERS.df2onehot import df2onehot
   import SUPERVISED.gridsearchHyperOpt as gridsearchHyperOpt

   ##### REGRESSION #####
   X=pd.read_csv('../DATA/OTHER/titanic/titanic_train.csv')
   X=df2onehot(X)[0]
   X.dropna(inplace=True)
   y=X['Age'].astype(float).values
   X.drop(labels='Age', axis=1, inplace=True)

   # Learn model
   model1 = gridsearchHyperOpt.fit(X, y, max_evals=10, method='xgb_reg')
   model2 = gridsearchHyperOpt.fit(X, y, max_evals=10, method='lgb_reg')
   model3 = gridsearchHyperOpt.fit(X, y, max_evals=10, method='ctb_reg')

   # Predict
   out1   = gridsearchHyperOpt.predict(model1, X.values)
   out2   = gridsearchHyperOpt.predict(model2, X.values)
   out3   = gridsearchHyperOpt.predict(model3, X.values)

   # Plot
   gridsearchHyperOpt.plot(model1)
   plt.scatter(y, out['y_pred'])


   ##### CLASSIFICATION TWO-CLASS #####
   from sklearn import datasets
   iris = datasets.load_iris()
   X = pd.DataFrame(iris.data)
   y = iris.target

   model = gridsearchHyperOpt.fit(X, y==1, max_evals=100, method='xgb_clf')
   gridsearchHyperOpt.plot(model)

   ##### CLASSIFICATION MULTI-CLASS #####
   model = gridsearchHyperOpt.fit(X, y, max_evals=100, method='xgb_clf')
   gridsearchHyperOpt.plot(model)


 SEE ALSO
   gridsearchXGboostR, gridsearchGradientBoostingR
"""

#--------------------------------------------------------------------------
# Name        : gridsearchHyperOpt.py
# Version     : 1.0
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : Sep. 2019
#--------------------------------------------------------------------------

#%% Libraries
import matplotlib.pyplot as plt
from hyperopt import hp

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials

#from EXPLAINERS.explainmodel import explainmodel
import VIZ.treeplot as treeplot
import SUPERVISED.twoClassSummary as twoClassSummary

#%% Fit model after parameter hyper-optimization
def fit(X, y, method=None, max_evals=100, verbose=3):
    assert type(X)==pd.DataFrame, 'dataset X should be of type pd.DataFrame'
    assert type(y)==np.ndarray, 'Response variable y should be of type np.ndarray'
    assert method!=None, 'Choose method to use!'
    if (len(np.unique(y))>2) and (method=='xgb_clf'): method='xgb_clf_multi'
    
    if verbose>=3: print('[HYPEROPT.FIT] Optimizing parameters using hyper-optimization for %s..' %(method))

    # Split train-test set
    [X_train, X_test, y_train, y_test]=train_test_split(X.values, y, test_size=0.2)
    # Setup the hyperOptimizer parameters
    clf = HPOpt(X_train, X_test, y_train, y_test)
    # Find best parameters
    [best_params, results, _] = clf.optimize(method=method, trials=Trials(), algo=tpe.suggest, max_evals=max_evals)
    # Fit on all data using best parameters
    model = refit_model(X, y, method, best_params)

    # Store
    out=dict()
    out['model']=model
    out['method']=method
    out['best_params']=best_params
    out['results']=results

    # Return
    return(out)

#%% Predict
def predict(model, X):
    assert np.isin('model', list(model.keys())), 'Input dictionary should contain key [model] that contains the trained model.'
    assert type(X)==np.ndarray, 'dataset X should be of type np.ndarray'
    
    # Reshape if vector
    if len(X.shape)==1:
        X=X.reshape(1,-1)
    
    # Make prediction
    y_pred=model['model'].predict(X)
    # Return
    return(y_pred)

#%%
def refit(X, y, out, verbose=3):
    model = retrain_model(X, y, out['method'], out['best_params'], verbose=verbose)
    out['model']=model
    return(out)
    
#%%
def refit_model(X, y, method, best_params, verbose=3):
    model=None
    try:
        if verbose>=3: print('[HYPEROPT.FIT] Fitting model using best parameters..')
        if method=='xgb_reg':
            model = xgb.XGBRegressor()
        if method=='lgb_reg':
            model = lgb.LGBMRegressor()
        if method=='ctb_reg':
            model = ctb.CatBoostRegressor()
        if method=='xgb_clf':
            model = xgb.XGBClassifier(objective='binary:logistic', booster='gbtree', early_stopping_rounds=10, silent=True)
        if method=='xgb_clf_multi':
            model = xgb.XGBClassifier(objective='multi:softprob', booster='gbtree', early_stopping_rounds=10, silent=True)
    
        # Fit
        model.set_params(**best_params).fit(X.values, y)
    except:
        print('[HYPEROPT.FIT] Warning Oh noo could not fit model on all data using best parameters..')
        print('[HYPEROPT.FIT] Warning Try to retrain again using: gridsearchHyperOpt.retrain(X,y,model)')
        

    return(model)
    
#%% Make simple plot
def plot(model):
    # Extract and plot the trials
    df  = model['results']
    # Index of the best performing parameters (ie those with the lowest loss, which is in this case the RMSE)
    idx = df['loss'].idxmin()
    
    # Setupt plot
    plt.figure(figsize=(15,8))
    plt.scatter(df['tid'], df['loss'], c='b', s=25)
    plt.scatter(df['tid'][idx], df['loss'][idx], c='r', s=100)
    plt.xlabel('iteration')
    plt.ylabel('loss function score')
    plt.grid(True)
    plt.show()

    # Histogram
    plt.figure(figsize=(15,8))
    plt.hist(df['loss'].values, bins=25)
    plt.xlabel('Loss function score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Best performing parameters
    print('[HYPEROPT.PLOT] Best performing parameters:')
    print(df.iloc[idx,:])

    # Explain model
    try:
    #    explainmodel(out['model'], X, y, url='model_explainmodel.html', showfig=True, verbose=3)
        treeplot.xgboost(model['model'], plottype='horizontal')
    except:
        pass

#%% Class
class HPOpt(object):
    # Initialization
    def __init__(self, x_train, x_test, y_train, y_test):
        assert type(x_train)==np.ndarray, 'X_train should be of type np.ndarray'
        assert type(x_test)==np.ndarray, 'X_test should be of type np.ndarray'
        assert type(y_train)==np.ndarray, 'y_train should be of type np.ndarray'
        assert type(y_test)==np.ndarray, 'y_test should be of type np.ndarray'
        
        # Set data sets        
        self.x_train = x_train
        self.x_test  = x_test
        self.y_train = y_train
        self.y_test  = y_test

    def optimize(self, method, trials, algo, max_evals):
        # Import the function that is gonig to be used
        fn = getattr(self, method)
        # Import search space for the specific function
        space = get_params(method)
        # Run
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
            out = to_df(trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return(result, out, trials)

    def xgb_reg(self, para):
        reg = xgb.XGBRegressor(**para['reg_params'])
        return self.train_reg(reg, para)

    def lgb_reg(self, para):
        reg = lgb.LGBMRegressor(**para['reg_params'])
        return self.train_reg(reg, para)

    def ctb_reg(self, para):
        reg = ctb.CatBoostRegressor(**para['reg_params'])
        return self.train_reg(reg, para)

    def xgb_clf(self, para):
        clf = xgb.XGBClassifier(**para['clf_params'])
        return self.train_clf(clf, para)
    
    def xgb_clf_multi(self, para):
        clf = xgb.XGBClassifier(**para['clf_params'])
        return self.train_clf(clf, para)

    def train_reg(self, reg, para):
        # Fit model
        reg.fit(self.x_train, self.y_train, eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)], **para['fit_params'])
        # Make prediction
        pred = reg.predict(self.x_test)
        loss = para['loss_func'](self.y_test, pred)
        return {'loss': loss, 'status': STATUS_OK}

    def train_clf(self, clf, para):
        # Fit model
        clf.fit(self.x_train, self.y_train, eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)], **para['fit_params'])
        # Make prediction
        y_pred = clf.predict(self.x_test)
        y_pred_proba = clf.predict_proba(self.x_test)
        
        # Scoring two class classification
        if len(np.unique(self.y_test))==2:
            # If we have a value that we want to maximize, such as f1, then we just have our function return the negative of that metric.
            results=twoClassSummary.twoClassStats(self.y_test, y_pred_proba[:,1], threshold=0.5, verbose=0)
            return {'loss': -results[para['scoring']], 'auc': results['auc'], 'kappa': results['kappa'], 'status': STATUS_OK}
        else:
            # If we have a value that we want to maximize, such as f1, then we just have our function return the negative of that metric.
            #from sklearn.metrics import multilabel_confusion_matrix
            #multilabel_confusion_matrix(self.y_test, y_pred)
            kappscore = cohen_kappa_score(self.y_test, y_pred)
            return {'loss': -kappscore, 'status': STATUS_OK}

#%% Transform results into dataframe
def to_df(trials):
    loss_score = list(map(lambda x: x['loss'], trials.results))
    df=pd.DataFrame(trials.vals)
    df['loss']=loss_score
    df['tid']=trials.tids
#    df['iteration']=trials.idxs_vals[0]['x']
#    df['x']=trials.idxs_vals[1]['x']

    return(df)

#%% Set the search spaces
def get_params(fn_name, verbose=3):
    # XGB parameters
    if fn_name=='xgb_reg':
        if verbose>=3: print('[HYPEROPT] Collecting XGB parameters for regression..')
        xgb_reg_params = {
            'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 0.31, 0.05)),
            'max_depth':        hp.choice('max_depth',        np.arange(5, 16, 1, dtype=int)),
            'min_child_weight': hp.choice('min_child_weight', np.arange(1, 10, 1, dtype=int)),
            'gamma':            hp.choice('gamma', [0, 0.25, 0.5, 1.0]),
            'reg_lambda':            hp.choice('reg_lambda', [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]),
            #'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
            'subsample':        hp.uniform('subsample', 0.5, 1),
            'n_estimators':     100,
        }
        xgb_fit_params = {
            'eval_metric': 'rmse',
            'early_stopping_rounds': 10,
            'verbose': False
        }
        xgb_para = dict()
        xgb_para['reg_params'] = xgb_reg_params
        xgb_para['fit_params'] = xgb_fit_params
        xgb_para['loss_func' ] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))

        #return
        return(xgb_para)
    
    # LightGBM parameters
    if fn_name=='lgb_reg':
        if verbose>=3: print('[HYPEROPT] Collecting LightGBM parameters for regression..')
        lgb_reg_params = {
            'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 0.31, 0.05)),
            'max_depth':        hp.choice('max_depth',        np.arange(5, 16, 1, dtype=int)),
            'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
            #'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
            'subsample':        hp.uniform('subsample', 0.8, 1),
            'n_estimators':     100,
        }
        lgb_fit_params = {
            'eval_metric': 'l2',
            'early_stopping_rounds': 10,
            'verbose': False
        }
        lgb_para = dict()
        lgb_para['reg_params'] = lgb_reg_params
        lgb_para['fit_params'] = lgb_fit_params
        lgb_para['loss_func' ] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))
    
        #return
        return(lgb_para)
    
    # LightGBM parameters
    if fn_name=='ctb_reg':
        # CatBoost parameters
        if verbose>=3: print('[HYPEROPT] Collecting CatBoost parameters for regression..')
        ctb_reg_params = {
            'learning_rate':     hp.choice('learning_rate',     np.arange(0.05, 0.31, 0.05)),
            'max_depth':         hp.choice('max_depth',         np.arange(5, 16, 1, dtype=int)),
            'colsample_bylevel': hp.choice('colsample_bylevel', np.arange(0.3, 0.8, 0.1)),
            'n_estimators':      100,
            'eval_metric':       'RMSE',
        }
        ctb_fit_params = {
            'early_stopping_rounds': 10,
            'verbose': False
        }
        ctb_para = dict()
        ctb_para['reg_params'] = ctb_reg_params
        ctb_para['fit_params'] = ctb_fit_params
        ctb_para['loss_func' ] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))

        #return
        return(ctb_para)

    if 'xgb_clf' in fn_name:
        if verbose>=3: print('[HYPEROPT] Collecting XGB parameters for classification..')
        xgb_clf_params = {
            'learning_rate':    hp.choice('learning_rate',    np.arange(0.01, 0.3, 0.05)),
            'max_depth':        hp.choice('max_depth',        np.arange(5, 16, 1, dtype=int)),
            'min_child_weight': hp.choice('min_child_weight', np.arange(1, 10, 1, dtype=int)),
            'gamma':            hp.choice('gamma', [0.5, 1, 1.5, 2, 5]),
            'subsample':        hp.uniform('subsample', 0.5, 1),
            'scale_pos_weight': hp.choice('scale_pos_weight', [0, 0.5, 1]),
            'n_estimators':     100,
            'booster': 'gbtree',
            #'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
        }
        
        if fn_name=='xgb_clf':
            xgb_clf_params['eval_metric']=hp.choice('eval_metric', ['error', 'auc'])
            xgb_clf_params['objective']='binary:logistic'
            scoring='f1'
        if fn_name=='xgb_clf_multi':
            xgb_clf_params['objective']='multi:softprob'
            #from sklearn.metrics import make_scorer
            #scoring = make_scorer(cohen_kappa_score, greater_is_better=True)
            scoring='kappa' # Note that this variable is not used. 


        xgb_fit_params = {
            'early_stopping_rounds': 10,
            'verbose': False
        }
        xgb_para = dict()
        xgb_para['clf_params'] = xgb_clf_params
        xgb_para['fit_params'] = xgb_fit_params
        xgb_para['scoring' ] = scoring

        #return
        return(xgb_para)



        
