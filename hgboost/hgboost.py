"""hgboost: Hyperoptimized Gradient Boosting library.

Contributors: https://github.com/erdogant/hgboost
"""

import warnings
warnings.filterwarnings("ignore")

import classeval as cle
from df2onehot import df2onehot
import treeplot as tree
import colourmap
import pypickle

import os
import numpy as np
import pandas as pd
import wget

from sklearn.metrics import mean_squared_error, cohen_kappa_score, mean_absolute_error, log_loss, roc_auc_score, f1_score
from sklearn.ensemble import VotingClassifier, VotingRegressor
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import xgboost as xgb
import catboost as ctb

try:
    import lightgbm as lgb
except:
    pass

from hyperopt import fmin, tpe, STATUS_OK, Trials, hp

from tqdm import tqdm
import time
import copy


# %%
class hgboost:
    """Create a class hgboost that is instantiated with the desired method."""

    def __init__(self, max_eval=250, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, is_unbalance=True, random_state=None, n_jobs=-1, gpu=False, verbose=3):
        """Initialize hgboost with user-defined parameters.

        Parameters
        ----------
        max_eval : int, (default : 250)
            Search space is created on the number of evaluations.
        threshold : float, (default : 0.5)
            Classification threshold. In case of two-class model this is 0.5
        cv : int, optional (default : 5)
            Cross-validation. Specifying the test size by test_size.
        top_cv_evals : int, (default : 10)
            Number of top best performing models that is evaluated.
            If set to None, each iteration (max_eval) is tested.
            If set to 0, cross validation is not performed.
        test_size : float, (default : 0.2)
            Percentage split for the testset based on the total dataset.
        val_size : float, (default : 0.2)
            Percentage split for the validationset based on the total dataset. This part is kept untouched, and used only once to determine the model performance.
        is_unbalance : Bool, (default: True)
            Control the balance of positive and negative weights, useful for unbalanced classes.
            xgboost clf : sum(negative instances) / sum(positive instances)
            catboost clf : sum(negative instances) / sum(positive instances)
            lightgbm clf : balanced
            False: grid search
        random_state : int, (default : None)
            Fix the random state for validation set and test set. Note that is not used for the crossvalidation.
        n_jobs : int, (default : -1)
            The number of jobs to run in parallel for fit. None means 1 unless in a joblib.parallel_backend context.
            -1 means using all processors.
        gpu : bool, (default : False)
            Computing using either GPU or CPU. Note that GPU usage is not very well supported because various optimizations are performed during training/testing/crossvalidation.
            True: Use GPU.
            False: Use CPU.
        verbose : int, (default : 3)
            Print progress to screen.
            0: None, 1: ERROR, 2: WARN, 3: INFO, 4: DEBUG, 5: TRACE

        Returns
        -------
        None.

        References
        ----------
        * https://github.com/hyperopt/hyperopt
        * https://www.districtdatalabs.com/parameter-tuning-with-hyperopt
        * https://scikit-learn.org/stable/modules/model_evaluation.html

        """
        if (threshold is None) or (threshold <= 0): raise ValueError('[hgboost] >Error: [threshold] must be >0 and not [None]')
        if (max_eval is None) or (max_eval <= 0): max_eval=1
        if top_cv_evals is None: max_eval=0
        if (test_size is None) or (test_size <= 0): raise ValueError('[hgboost] >Error: test_size must be >0 and not [None] Note: the final model is learned on the entire dataset. [test_size] may help you getting a more robust model.')
        if (val_size is not None) and (val_size<=0): val_size=None

        self.max_eval=max_eval
        self.top_cv_evals=top_cv_evals
        self.threshold=threshold
        self.test_size=test_size
        self.val_size=val_size
        self.algo=tpe.suggest
        self.cv=cv
        self.random_state=random_state
        self.n_jobs=n_jobs
        self.verbose=verbose
        self.is_unbalance = is_unbalance
        self.gpu = gpu

    def _fit(self, X, y, pos_label=None):
        """Fit the best performing model.

        Description
        -----------
        Minimize a function over a hyperparameter space.
        More realistically: *explore* a function over a hyperparameter space
        according to a given algorithm, allowing up to a certain number of
        function evaluations.  As points are explored, they are accumulated in
        "trials".

        Parameters
        ----------
        X : pd.DataFrame
            Input dataset.
        y : array-like.
            Response variable.
        pos_label : string/int.
            In case of classification (_clf), the model will be fitted on the pos_label that is in y.

        Returns
        -------
        results: dict
            * best_params (dict): containing the  optimized model hyperparameters.
            * summary (DataFrame): containing the parameters and performance for all evaluations.
            * trials: Hyperopt object with the trials.
            * model (object): Final optimized model based on the k-fold crossvalidation, with the hyperparameters as described in "params".
            * val_results (dict): Results of the final model on independent validation dataset.
            * dict: comparison_results: Comparison between HyperOptimized parameters vs. default parameters.

        """
        # Check input data
        X, y, self.pos_label=_check_input(X, y, pos_label, self.method, verbose=self.verbose)

        # Recaculate test size. This should be the percentage of the total dataset after removing the validation set.
        if (self.val_size is not None) and (self.val_size > 0):
            self.test_size = np.round((self.test_size * X.shape[0]) / (X.shape[0] - (self.val_size * X.shape[0])), 2)

        # Print to screen
        if self.verbose>=3:
            print('[hgboost] >method: %s' %(self.method))
            print('[hgboost] >eval_metric: %s' %(self.eval_metric))
            print('[hgboost] >greater_is_better: %s' %(self.greater_is_better))

        # Set validation set
        self._set_validation_set(X, y)
        # Find best parameters
        self.model, self.results = self._HPOpt()
        # Fit on all data using best parameters
        if self.verbose>=3: print('[hgboost] >*********************************************************************************')
        if self.verbose>=3: print('[hgboost] >Retrain [%s] on the entire dataset with the optimal hyperparameters.' %(self.method))
        self.model.fit(X, y)
        # Return
        return self.results

    def _classification(self, X, y, eval_metric, greater_is_better, params):
        # Gather for method, the default metric and greater is better.
        self.eval_metric, self.greater_is_better =_check_eval_metric(self.method, eval_metric, greater_is_better)
        # Import search space for the specific function
        if params == 'default': params = _get_params(self.method, eval_metric=self.eval_metric, y=y, pos_label=self.pos_label, is_unbalance=self.is_unbalance, gpu=self.gpu, verbose=self.verbose)
        self.space = params
        # Fit model
        self.results = self._fit(X, y, pos_label=self.pos_label)
        # Fin
        if self.verbose>=3: print('[hgboost] >Fin!')

    def _regression(self, X, y, eval_metric, greater_is_better, params):
        # Gather for method, the default metric and greater is better.
        self.eval_metric, self.greater_is_better = _check_eval_metric(self.method, eval_metric, greater_is_better)
        # Import search space for the specific function
        if params == 'default': params = _get_params(self.method, eval_metric=self.eval_metric, gpu=self.gpu, verbose=self.verbose)
        self.space = params
        # Fit model
        self.results = self._fit(X, y)
        # Fin
        if self.verbose>=3: print('[hgboost] >Fin!')

    def xgboost_reg(self, X, y, eval_metric='rmse', greater_is_better=False, params='default'):
        """Xgboost Regression with hyperparameter optimization.

        Parameters
        ----------
        X : pd.DataFrame.
            Input dataset.
        y : array-like
            Response variable.
        eval_metric : str, (default : 'rmse').
            Evaluation metric for the regressor model.
                * 'rmse': root mean squared error.
                * 'mse': mean squared error.
                * 'mae': mean absolute error.
        greater_is_better : bool (default : False).
            If a loss, the output of the python function is negated by the scorer object, conforming to the cross validation convention that scorers return higher values for better models.
        params : dict, (default : 'default').
            Hyper parameters.

        Returns
        -------
        results: dict
            * best_params (dict): containing the  optimized model hyperparameters.
            * summary (DataFrame): containing the parameters and performance for all evaluations.
            * trials: Hyperopt object with the trials.
            * model (object): Final optimized model based on the k-fold crossvalidation, with the hyperparameters as described in "params".
            * val_results (dict): Results of the final model on independent validation dataset.
            * comparison_results (dict): Comparison between HyperOptimized parameters vs. default parameters.

        """
        if self.verbose>=3: print('[hgboost] >Start hgboost regression.')
        # Method
        self.method='xgb_reg'
        # Run method
        self._regression(X, y, eval_metric, greater_is_better, params)
        # Return
        return self.results

    def lightboost_reg(self, X, y, eval_metric='rmse', greater_is_better=False, params='default'):
        """Light Regression with hyperparameter optimization.

        Parameters
        ----------
        X : pd.DataFrame.
            Input dataset.
        y : array-like.
            Response variable.
        eval_metric : str, (default : 'rmse').
            Evaluation metric for the regressor model.
                * 'rmse': root mean squared error.
                * 'mse': mean squared error.
                * 'mae': mean absolute error.
        greater_is_better : bool (default : False).
            If a loss, the output of the python function is negated by the scorer object, conforming to the cross validation convention that scorers return higher values for better models.
        params : dict, (default : 'default').
            Hyper parameters.

        Returns
        -------
        results: dict
            * best_params (dict): containing the  optimized model hyperparameters.
            * summary (DataFrame): containing the parameters and performance for all evaluations.
            * trials: Hyperopt object with the trials.
            * model (object): Final optimized model based on the k-fold crossvalidation, with the hyperparameters as described in "params".
            * val_results (dict): Results of the final model on independent validation dataset.
            * comparison_results (dict): Comparison between HyperOptimized parameters vs. default parameters.

        """
        if self.verbose>=3: print('[hgboost] >Start hgboost regression.')
        # Method
        self.method='lgb_reg'
        # Run method
        self._regression(X, y, eval_metric, greater_is_better, params)
        # Return
        return self.results

    def catboost_reg(self, X, y, eval_metric='rmse', greater_is_better=False, params='default'):
        """Catboost Regression with hyperparameter optimization.

        Parameters
        ----------
        X : pd.DataFrame.
            Input dataset.
        y : array-like.
            Response variable.
        eval_metric : str, (default : 'rmse').
            Evaluation metric for the regressor model.
                * 'rmse': root mean squared error.
                * 'mse': mean squared error.
                * 'mae': mean absolute error.
        greater_is_better : bool (default : False).
            If a loss, the output of the python function is negated by the scorer object, conforming to the cross validation convention that scorers return higher values for better models.
        params : dict, (default : 'default').
            Hyper parameters.

        Returns
        -------
        results: dict
            * best_params (dict): containing the  optimized model hyperparameters.
            * summary (DataFrame): containing the parameters and performance for all evaluations.
            * trials: Hyperopt object with the trials.
            * model (object): Final optimized model based on the k-fold crossvalidation, with the hyperparameters as described in "params".
            * val_results (dict): Results of the final model on independent validation dataset.
            * comparison_results (dict): Comparison between HyperOptimized parameters vs. default parameters.

        """
        if self.verbose>=3: print('[hgboost] >Start hgboost regression.')
        if self.gpu:
            print('[hgboost] >GPU for catboost is not supported. It throws an error because multiple evaluation sets are readily optimized.')
            self.gpu=False
        # Method
        self.method='ctb_reg'
        # Run method
        self._regression(X, y, eval_metric, greater_is_better, params)
        # Return
        return self.results

    def xgboost(self, X, y, pos_label=None, method='xgb_clf', eval_metric=None, greater_is_better=None, params='default'):
        """Xgboost Classification with hyperparameter optimization.

        Parameters
        ----------
        X : pd.DataFrame.
            Input dataset.
        y : array-like.
            Response variable.
        pos_label : string/int.
            Fit the model on the pos_label that that is in [y].
        method : String, (default : 'auto').
            * 'xgb_clf': XGboost two-class classifier
            * 'xgb_clf_multi': XGboost multi-class classifier
        eval_metric : str, (default : None).
            Evaluation metric for the regressor of classification model.
                * 'auc': area under ROC curve (default for two-class)
                * 'kappa': (default for multi-class)
                * 'f1': F1-score
                * 'logloss'
                * 'auc_cv': Compute average auc per iteration in each cross. This approach is computational expensive.
        greater_is_better : bool.
            If a loss, the output of the python function is negated by the scorer object, conforming to the cross validation convention that scorers return higher values for better models.
                * auc :  True -> two-class
                * kappa : True -> multi-class

        Returns
        -------
        results: dict
            * best_params (dict): containing the  optimized model hyperparameters.
            * summary (DataFrame): containing the parameters and performance for all evaluations.
            * trials: Hyperopt object with the trials.
            * model (object): Final optimized model based on the k-fold crossvalidation, with the hyperparameters as described in "params".
            * val_results (dict): Results of the final model on independent validation dataset.
            * comparison_results (dict): Comparison between HyperOptimized parameters vs. default parameters.

        """
        if self.verbose>=3: print('[hgboost] >Start hgboost classification.')
        self.method = method
        self.pos_label = pos_label
        # Run method
        self._classification(X, y, eval_metric, greater_is_better, params)
        # Return
        return self.results

    def catboost(self, X, y, pos_label=None, eval_metric='auc', greater_is_better=True, params='default'):
        """Catboost Classification with hyperparameter optimization.

        Parameters
        ----------
        X : pd.DataFrame.
            Input dataset.
        y : array-like.
            Response variable.
        pos_label : string/int.
            Fit the model on the pos_label that that is in [y].
        eval_metric : str, (default : 'auc').
            Evaluation metric for the regressor of classification model.
                * 'auc': area under ROC curve (default for two-class)
                * 'kappa': (default for multi-class)
                * 'f1': F1-score
                * 'logloss'
                * 'auc_cv': Compute average auc per iteration in each cross. This approach is computational expensive.
        greater_is_better : bool (default : True).
            If a loss, the output of the python function is negated by the scorer object, conforming to the cross validation convention that scorers return higher values for better models.

        Returns
        -------
        results: dict
            * best_params (dict): containing the  optimized model hyperparameters.
            * summary (DataFrame): containing the parameters and performance for all evaluations.
            * trials: Hyperopt object with the trials.
            * model (object): Final optimized model based on the k-fold crossvalidation, with the hyperparameters as described in "params".
            * val_results (dict): Results of the final model on independent validation dataset.
            * comparison_results (dict): Comparison between HyperOptimized parameters vs. default parameters.

        """
        if self.verbose>=3: print('[hgboost] >Start hgboost classification.')
        if self.gpu:
            print('[hgboost] >GPU for catboost is not supported. It throws an error because I am readily optimizing across multiple evaluation sets.')

        self.method = 'ctb_clf'
        self.pos_label = pos_label
        # Run method
        self._classification(X, y, eval_metric, greater_is_better, params)
        # Return
        return self.results

    def lightboost(self, X, y, pos_label=None, eval_metric='auc', greater_is_better=True, params='default'):
        """Lightboost Classification with hyperparameter optimization.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataset.
        y : array-like
            Response variable.
        pos_label : string/int.
            Fit the model on the pos_label that that is in [y].
        eval_metric : str, (default : 'auc')
            Evaluation metric for the regressor of classification model.
                * 'auc': area under ROC curve (default for two-class)
                * 'kappa': (default for multi-class)
                * 'f1': F1-score
                * 'logloss'
                * 'auc_cv': Compute average auc per iteration in each cross. This approach is computational expensive.
        greater_is_better : bool (default : True)
            If a loss, the output of the python function is negated by the scorer object, conforming to the cross validation convention that scorers return higher values for better models.

        Returns
        -------
        results: dict
            * best_params (dict): containing the  optimized model hyperparameters.
            * summary (DataFrame): containing the parameters and performance for all evaluations.
            * trials: Hyperopt object with the trials.
            * model (object): Final optimized model based on the k-fold crossvalidation, with the hyperparameters as described in "params".
            * val_results (dict): Results of the final model on independent validation dataset.
            * comparison_results (dict): Comparison between HyperOptimized parameters vs. default parameters.

        """
        if self.verbose>=3: print('[hgboost] >Start hgboost classification.')
        self.method = 'lgb_clf'
        self.pos_label = pos_label
        # Run method
        self._classification(X, y, eval_metric, greater_is_better, params)
        # Return
        return self.results

    def ensemble(self, X, y, pos_label=None, methods=['xgb_clf', 'ctb_clf', 'lgb_clf'], eval_metric=None, greater_is_better=None, voting='soft'):
        """Ensemble Classification with hyperparameter optimization.

        Description
        -----------
        Fit best model for xgboost, catboost and lightboost, and then combine the individual models to a new one.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataset.
        y : array-like
            Response variable.
        pos_label : string/int.
            Fit the model on the pos_label that that is in [y].
        methods : list of strings, (default : ['xgb_clf','ctb_clf','lgb_clf']).
            The models included for the ensemble classifier or regressor. The clf and reg models can not be combined.
                * ['xgb_clf','ctb_clf','lgb_clf']
                * ['xgb_reg','ctb_reg','lgb_reg']
        eval_metric : str, (default : 'auc')
            Evaluation metric for the regressor of classification model.
                * 'auc': area under ROC curve (two-class classification : default)
        greater_is_better : bool (default : True)
            If a loss, the output of the python function is negated by the scorer object, conforming to the cross validation convention that scorers return higher values for better models.
                * auc :  True -> two-class
        voting : str, (default : 'soft')
            Combining classifier using a voting scheme.
                * 'hard': using predicted classes.
                * 'soft': using the Probabilities.

        Returns
        -------
        results: dict
            * best_params (dict): containing the  optimized model hyperparameters.
            * summary (DataFrame): containing the parameters and performance for all evaluations.
            * trials: Hyperopt object with the trials.
            * model (object): Final optimized model based on the k-fold crossvalidation, with the hyperparameters as described in "params".
            * val_results (dict): Results of the final model on independent validation dataset.
            * comparison_results (dict): Comparison between HyperOptimized parameters vs. default parameters.

        """
        # Store parameters in object
        self.results = {}
        self.voting = voting
        self.methods = methods

        if np.all(list(map(lambda x: 'clf' in x, methods))):
            if self.verbose>=3: print('[hgboost] >Create ensemble classification model..')
            self.method = 'ensemble_clf'
        elif np.all(list(map(lambda x: 'reg' in x, methods))):
            if self.verbose>=3: print('[hgboost] >Create ensemble regression model..')
            self.method = 'ensemble_reg'
        else:
            raise ValueError('[hgboost] >Error: The input [methods] must be of type "_clf" or "_reg" but can not be combined.')

        # Check input data
        X, y, self.pos_label = _check_input(X, y, pos_label, self.method, verbose=self.verbose)
        # Gather for method, the default metric and greater is better.
        self.eval_metric, self.greater_is_better = _check_eval_metric(self.method, eval_metric, greater_is_better)
        # Store the clean initialization in hgb
        hgb = copy.copy(self)

        # Create independent validation set.
        if self.method == 'ensemble_clf':
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size, random_state=self.random_state, shuffle=True, stratify=y)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size, random_state=self.random_state, shuffle=True)

        # Hyperparameter optimization for boosting models
        models = []
        for method in methods:
            # Make copy of clean init
            hgbM = copy.copy(hgb)
            hgbM.method = method
            hgbM._classification(X_train, y_train, eval_metric, greater_is_better, 'default')
            # Store
            models.append((method, copy.copy(hgbM.model)))
            self.results[method] = {}
            self.results[method]['model'] = copy.copy(hgbM)

        # Create the ensemble model
        if self.verbose>=3: print('[hgboost] >Fit ensemble model with [%s] voting..' %(self.voting))
        if self.method == 'ensemble_clf':
            model = VotingClassifier(models, voting=voting, n_jobs=self.n_jobs)
            model.fit(X, y==pos_label)
        else:
            model = VotingRegressor(models, n_jobs=self.n_jobs)
            model.fit(X, y)
        # Store ensemble model
        self.model = model

        # Validation error for the ensemble model
        if self.verbose>=3: print('[hgboost] >Evalute [ensemble] model on independent validation dataset (%.0f samples, %.2g%%)' %(len(y_val), self.val_size * 100))
        # Evaluate results on the same validation set
        val_score, val_results = self._eval(X_val, y_val, model, verbose=2)
        if self.verbose>=3: print('[hgboost] >[Ensemble] [%s]: %.4g on independent validation dataset' %(self.eval_metric, val_score['loss']))

        # Validate each of the independent methods to show differences in loss-scoring
        if self.val_size is not None:
            self.X_val = X_val
            self.y_val = y_val
            for method in methods:
                # Evaluation
                val_score_M, val_results_M = self._eval(X_val, y_val, self.results[method]['model'].model, verbose=2)
                # Store
                self.results[method]['loss'] = val_score_M['loss']
                self.results[method]['val_results'] = val_results_M
                if self.verbose>=3: print('[hgboost] >[%s]  [%s]: %.4g on independent validation dataset' %(method, self.eval_metric, val_score_M['loss']))

        # Store
        self.results['val_results'] = val_results
        self.results['model'] = model
        # self.results['summary'] = pd.concat([hgbX.results['summary'], hgbC.results['summary'], hgbL.results['summary']])

        # Return
        return self.results

    def _set_validation_set(self, X, y):
        """Set the validation set.

        Description
        -----------
        Here we separate the data as the validation set.
        * The new data is stored in self.X and self.y
        * The validation X and y are stored in self.X_val and self.y_val
        """
        if self.verbose>=3: print('[hgboost] >*********************************************************************************')
        if self.verbose>=3: print('[hgboost] >Total dataset: %s ' %(str(X.shape)))

        if (self.val_size is not None):
            if '_clf' in self.method:
                self.X, self.X_val, self.y, self.y_val = train_test_split(X, y, test_size=self.val_size, random_state=self.random_state, shuffle=True, stratify=y)
            elif '_reg' in self.method:
                self.X, self.X_val, self.y, self.y_val = train_test_split(X, y, test_size=self.val_size, random_state=self.random_state, shuffle=True)
            if self.verbose>=3: print('[hgboost] >Validation set: %s ' %(str(self.X_val.shape)))
        else:
            self.X = X
            self.y = y
            self.X_val = None
            self.y_val = None

    def _HPOpt(self):
        """Hyperoptimization of the search space.

        Description
        -----------
        Minimize a function over the hyperparameter search space.
        More realistically: *explore* a function over a hyperparameter space
        according to a given algorithm, allowing up to a certain number of function evaluations.
        As points are explored, they are accumulated in "trials".

        Returns
        -------
        model : object
            Fitted model.
        results: dict
            * best_params (dict): containing the  optimized model hyperparameters.
            * summary (DataFrame): containing the parameters and performance for all evaluations.
            * trials: Hyperopt object with the trials.
            * model (object): Final optimized model based on the k-fold crossvalidation, with the hyperparameters as described in "params".
            * val_results (dict): Results of the final model on independent validation dataset.
            * comparison_results (dict): Comparison between HyperOptimized parameters vs. default parameters.

        """
        # Import the desired model-function for the classification/regression
        disable = (False if (self.verbose<3) else True)
        fn = getattr(self, self.method)

        # Split train-test set. This set is used for parameter optimization. Note that parameters are shuffled and the train-test set is retained constant.
        # This will make the comparison across parameters and not differences in train-test variances.
        if '_clf' in self.method:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=True, stratify=self.y)
        elif '_reg' in self.method:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=True)

        if self.verbose>=3: print('[hgboost] >Test-set: %s ' %(str(self.X_test.shape)))
        if self.verbose>=3: print('[hgboost] >Train-set: %s ' %(str(self.X_train.shape)))
        if self.verbose>=3: print('[hgboost] >*********************************************************************************')
        if self.verbose>=3: print('[hgboost] >Searching across hyperparameter space for best performing parameters using maximum nr. evaluations: %.0d' %(self.max_eval))

        # Hyperoptimization to find best performing model. Set the trials which is the object where all the HPopt results are stored.
        trials=Trials()
        best_params = fmin(fn=fn, space=self.space, algo=self.algo, max_evals=self.max_eval, trials=trials, show_progressbar=disable)
        # Summary results
        results_summary, model, best_params = self._to_df(trials, best_params)

        # Cross-validation over the top n models. To speed up we can decide to test only the best performing ones. The best performing model is returned.
        if self.cv is not None:
            model, results_summary, best_params = self._cv(results_summary, self.space, best_params)

        # Create a basic model by using default parameters.
        space_basic = {}
        space_basic['fit_params'] = {'verbose': 0}
        space_basic['model_params'] = {}
        model_basic = getattr(self, self.method)
        model_basic = fn(space_basic)['model']
        comparison_results = {}

        # Validation error
        val_results = None
        if (self.val_size is not None):
            # Evaluate results
            if self.verbose>=3: print('[hgboost] >*********************************************************************************')
            if self.verbose>=3: print('[hgboost] >Evaluate best [%s] model on validation dataset (%.0f samples, %.2g%%)' %(self.method, len(self.y_val), self.val_size * 100))
            # With hyperparameter optimization.
            val_score, val_results = self._eval(self.X_val, self.y_val, model, verbose=2)
            # With defaults parameters.
            val_score_basic, val_results_basic = self._eval(self.X_val, self.y_val, model_basic, verbose=2)
            # Store
            comparison_results['Model with optimized hyperparameters (validation set)'] = val_results
            comparison_results['Model with default parameters (validation set)'] = val_results_basic
            if self.verbose>=3: print('[hgboost] >[%s]: %.4g using optimized hyperparameters on validation set.' %(self.eval_metric, val_score['loss']))
            if self.verbose>=3: print('[hgboost] >[%s]: %.4g using default (not optimized) parameters on validation set.' %(self.eval_metric, val_score_basic['loss']))
            # Store Validation results
            results_summary = _store_validation_scores(results_summary, best_params, model_basic, val_score_basic, val_score, self.greater_is_better)

        # Remove the model column
        del results_summary['model']
        # Store
        results = {}
        results['params'] = best_params
        results['summary'] = results_summary
        results['trials'] = trials
        results['model'] = model
        results['val_results'] = val_results
        results['comparison_results'] = comparison_results
        # Return
        return model, results

    def _cv(self, results_summary, space, best_params):
        ascending = False if self.greater_is_better else True
        results_summary['loss_mean'] = np.nan
        results_summary['loss_std'] = np.nan

        # Determine maximum folds
        top_cv_evals = np.minimum(results_summary.shape[0], self.top_cv_evals)
        idx_top_models = results_summary['loss'].sort_values(ascending=ascending).index[0:top_cv_evals]
        if self.verbose>=3: print('[hgboost] >*********************************************************************************')
        if self.verbose>=3: print('[hgboost] >%.0d-fold cross validation for the top %.0d scoring models, Total nr. tests: %.0f' %(self.cv, len(idx_top_models), self.cv * len(idx_top_models)))
        disable = (False if (self.verbose<3) else True)

        # For each model, compute the performance.
        for idx in tqdm(idx_top_models, disable=disable):
            scores = []
            # Do the k-fold cross-validation
            for k in np.arange(0, self.cv):
                # Split train-test set
                if '_clf' in self.method:
                    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=None, shuffle=True, stratify=self.y)
                elif '_reg' in self.method:
                    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=None, shuffle=True)

                # Evaluate the top10 best performing models
                score, _ = self._train_model(results_summary['model'].iloc[idx], space)
                score.pop('model')
                scores.append(score)

            # Compute the mean and std of the p-best-performing models across the k-fold crossvalidation.
            results_summary['loss_mean'].iloc[idx] = pd.DataFrame(scores)['loss'].mean()
            results_summary['loss_std'].iloc[idx] = pd.DataFrame(scores)['loss'].std()

        # Negate scoring if required. The hpopt is optimized for loss functions (lower is better). Therefore we need to set eg the auc to negative and here we need to return.
        if self.greater_is_better:
            results_summary['loss_mean'] = results_summary['loss_mean'] * -1
            idx_best = results_summary['loss_mean'].argmax()
        else:
            idx_best = results_summary['loss_mean'].argmin()

        # Get best k-fold CV performing model based on the mean scores.
        if self.verbose>=3: print('[hgboost] >[%s] (average): %.4g Best %.0d-fold CV model using optimized hyperparameters.' %(self.eval_metric, results_summary['loss_mean'].iloc[idx_best], self.cv))
        model = results_summary['model'].iloc[idx_best]
        results_summary['best_cv'] = False
        results_summary['best_cv'].iloc[idx_best] = True
        # Collect best parameters for this model
        best_params = dict(results_summary.iloc[idx_best, np.isin(results_summary.columns, [*best_params.keys()])])
        # Return
        return model, results_summary, best_params

    def _train_model(self, model, space):
        verbose = 2 if self.verbose<=3 else 3
        # Evaluation is determine for both training and testing set. These results can plotted after finishing.
        eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        # Make fit with stopping-rule to avoid overfitting. Directly perform evaluation with the eval_set.
        model.fit(self.X_train, self.y_train, eval_set=eval_set, **space['fit_params'])
        # Evaluate results
        out, eval_results = self._eval(self.X_test, self.y_test, model, verbose=verbose)
        # Return
        return out, eval_results

    def xgb_reg(self, space):
        """Train Xgboost regression model."""
        reg = xgb.XGBRegressor(**space['model_params'], n_jobs=self.n_jobs, verbosity=0)
        out, _ = self._train_model(reg, space)
        return out

    def lgb_reg(self, space):
        """Train lightboost regression model."""
        reg = lgb.LGBMRegressor(**space['model_params'], n_jobs=self.n_jobs, verbosity=-1)
        out, _ = self._train_model(reg, space)
        return out

    def ctb_reg(self, space):
        """Train catboost regression model."""
        reg = ctb.CatBoostRegressor(**space['model_params'])
        out, _ = self._train_model(reg, space)
        return out

    def xgb_clf(self, space):
        """Train xgboost classification model."""
        clf = xgb.XGBClassifier(**space['model_params'], n_jobs=self.n_jobs, verbosity=0)
        out, _ = self._train_model(clf, space)
        return out

    def ctb_clf(self, space):
        """Train catboost classification model."""
        clf = ctb.CatBoostClassifier(**space['model_params'])
        out, _ = self._train_model(clf, space)
        return out

    def lgb_clf(self, space):
        """Train lightboost classification model."""
        clf = lgb.LGBMClassifier(**space['model_params'], n_jobs=self.n_jobs, verbosity=-1)
        out, _ = self._train_model(clf, space)
        return out

    def xgb_clf_multi(self, space):
        """Train xgboost multi-class classification model."""
        clf = xgb.XGBClassifier(**space['model_params'], n_jobs=self.n_jobs, verbosity=0)
        out, _ = self._train_model(clf, space)
        return out

    # Transform results into dataframe
    def _to_df(self, trials, best_params=None):
        if self.verbose>=3: print('[hgboost]> Collecting the hyperparameters from the [%.0d] trials.' %(len(trials.trials)))

        # Combine params with scoring results
        model_params = [*self.space['model_params'].keys()]
        if best_params is not None:
            model_params = np.array(model_params + [*best_params.keys()])
            model_params = list(np.unique(model_params))

        # model_params = [*trials.vals.keys()]
        df_params = pd.DataFrame(index=np.arange(0, len(trials.trials)), columns=model_params)

        # Gather all hyperparameter settings.
        # The trials.vals stores the index for some parameters instead of the real values.
        gather_params_legacy = False
        for i, trial in enumerate(trials.trials):
            for param in model_params:
                try:
                    if 'ctb' in self.method:
                        df_params[param].iloc[i] = trial['result']['model'].get_all_params().get(param)
                    else:
                        df_params[param].iloc[i] = getattr(trial['result']['model'], param)
                except:
                    if self.verbose>=3: print('[hgboost]> Skip [%s]' %(param))
                    gather_params_legacy = True

        # The trials.vals stores the index for some parameters instead of the real values.
        if gather_params_legacy:
            df_params = pd.DataFrame(trials.vals)
        df_scoring = pd.DataFrame(trials.results)
        df = pd.concat([df_params, df_scoring], axis=1)
        df['tid'] = trials.tids

        # Retrieve only the models with OK status
        Iloc = df['status']=='ok'
        df = df.loc[Iloc, :]

        # Retrieve idx for best model.
        idx = np.where(trials.best_trial['tid']==df['tid'])[0][0]
        # Als retrieve best model based on loss-score.
        if self.greater_is_better:
            df['loss'] = df['loss'] * -1
            idx_best_loss = df['loss'].argmax()
        else:
            idx_best_loss = df['loss'].argmin()

        if idx!=idx_best_loss:
            if self.verbose>=4: print('[hgboost] >[Warning]> Best model of hyperOpt does not have best loss score(?)')

        model = df['model'].iloc[idx_best_loss]
        score = df['loss'].iloc[idx_best_loss]
        df['best'] = False
        df['best'].iloc[idx_best_loss] = True

        # Get best_params
        try:
            best_params = df[best_params].iloc[idx_best_loss].to_dict()
            # Should be the same as:
            # trials.best_trial['result']['model']
        except:
            pass

        # Return
        if self.verbose>=3: print('[hgboost] >[%s]: %.4g Best performing model across %.0d iterations using Bayesian Optimization with Hyperopt.' %(self.eval_metric, score, df.shape[0]))
        return(df, model, best_params)

    # Predict
    def predict(self, X, model=None):
        """Prediction using fitted model.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        y_pred : array-like
            predictions results.
        y_proba : array-like
            Probability of the predictions.

        """
        if not hasattr(self, 'model'):
            if self.verbose>2: print('[hgboost] >Warning: No model found. Hint: fit a model first using xgboost, catboost or lightboost <return>')
            return None, None
        if model is None:
            model = self.model

        # Reshape if vector
        if len(X.shape)==1: X=X.reshape(1, -1)
        # Make prediction
        y_pred = model.predict(X)
        if '_clf' in self.method:
            y_proba = model.predict_proba(X)
        else:
            y_proba = None

        # Return
        return y_pred, y_proba

    def _eval(self, X_test, y_test, model, verbose=3):
        """Model Evaluation.

        Description
        -----------
        Note that the loss function is by default maximized towards small/negative values by the hptop method.
        When you want to optimize auc or f1, you simply need to negate the score.
        The negation is fixed with the parameter: greater_is_better=False
        """
        results = None
        # Make prediction
        y_pred = model.predict(X_test)

        # Evaluate results
        if '_clf' in self.method:
            # Compute probability
            y_proba = model.predict_proba(X_test)
            # y_score = model.decision_function(self.X_test)

            # multi-class classification
            if ('_clf_multi' in self.method):
                if self.eval_metric=='kappa':
                    loss = cohen_kappa_score(y_test, y_pred)
                elif self.eval_metric=='logloss':
                    loss = log_loss(y_test, y_pred)
                elif self.eval_metric=='auc':
                    loss = roc_auc_score(y_test, y_pred, multi_class='ovr')
                elif self.eval_metric=='f1':
                    loss = f1_score(y_test, y_pred)
                else:
                    raise ValueError('[hgboost] >Error: [%s] is not a valid [eval_metric] for [%s].' %(self.eval_metric, self.method))
                # Negative loss score if required
                if self.greater_is_better: loss = loss * -1
                # Store
                out = {'loss': loss, 'status': STATUS_OK, 'eval_time': time.time(), 'model': model}
            else:
                # Two-class classification
                results = cle.eval(y_test, y_proba[:, 1], y_pred=y_pred, threshold=self.threshold, pos_label=self.pos_label, verbose=0)
                # results = cle.ROC.eval(y_test, y_proba[:, 1], threshold=self.threshold, pos_label=self.pos_label, verbose=verbose)

                if self.eval_metric=='kappa':
                    loss = results[self.eval_metric]
                elif self.eval_metric=='logloss':
                    loss = log_loss(y_test, y_pred)
                elif self.eval_metric=='auc':
                    loss = results[self.eval_metric]
                elif self.eval_metric=='f1':
                    loss = results[self.eval_metric]
                elif self.eval_metric=='auc_cv':
                    loss = np.mean(cross_val_score(model, self.X_train, self.y_train, cv=self.cv, n_jobs=self.n_jobs))
                else:
                    raise ValueError('[hgboost] >Error: [%s] is not a valid [eval_metric] for [%s].' %(self.eval_metric, self.method))

                # Negative loss score if required
                if self.greater_is_better: loss = loss * -1
                # Store
                out = {'loss': loss, 'eval_time': time.time(), 'status': STATUS_OK, 'model': model}
                # out = {'loss': loss, 'eval_time': time.time(), 'auc': results['auc'], 'kappa': results['kappa'], 'f1': results['f1'], 'status': STATUS_OK, 'model': model}
        elif '_reg' in self.method:
            # Regression
            # loss = space['loss_func'](y_test, y_pred)
            if self.eval_metric=='mse':
                loss = mean_squared_error(y_test, y_pred, squared=True)
            elif self.eval_metric=='rmse':
                loss = mean_squared_error(y_test, y_pred, squared=False)
            elif self.eval_metric=='mae':
                loss = mean_absolute_error(y_test, y_pred)
            else:
                raise ValueError('[hgboost] >Error: [%s] is not a valid [eval_metric] for [%s].' %(self.eval_metric, self.method))

            # Negative loss score if required
            if self.greater_is_better: loss = loss * -1
            # Store results
            out = {'loss': loss, 'eval_time': time.time(), 'status': STATUS_OK, 'model': model}
        else:
            raise ValueError('[hgboost] >Error: Method %s does not exists.' %(self.method))

        if self.verbose>=5: print('[hgboost] >[%s] - [%s] - loss: %s' %(self.method, self.eval_metric, loss))
        return out, results

    def preprocessing(self, df, y_min=2, perc_min_num=0.8, excl_background='0.0', hot_only=False, verbose=None):
        """Pre-processing of the input data.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        y_min : int [0..len(y)], optional
            Minimal number of samples that must be present in a group. All groups with less then y_min samples are labeled as _other_ and are not used in the enriching model. The default is None.
        perc_min_num : float [None, 0..1], optional
            Force column (int or float) to be numerical if unique non-zero values are above percentage. The default is None. Alternative can be 0.8
        verbose : int, (default: 3)
            Print progress to screen.
            0: NONE, 1: ERROR, 2: WARNING, 3: INFO, 4: DEBUG, 5: TRACE

        Returns
        -------
        data : pd.Datarame
            Processed data.

        """
        if verbose is None: verbose = self.verbose
        X = df2onehot(df, y_min=y_min, hot_only=hot_only, perc_min_num=perc_min_num, excl_background=excl_background, verbose=verbose)
        return X['onehot']

    def import_example(self, data='titanic', url=None, sep=',', verbose=3):
        """Import example dataset from github source.

        Description
        -----------
        Import one of the few datasets from github source or specify your own download url link.

        Parameters
        ----------
        data : str
            Name of datasets: 'sprinkler', 'titanic', 'student', 'fifa', 'cancer', 'waterpump', 'retail'
        url : str
            url link to to dataset.
        verbose : int, (default : 3)
            Print progress to screen.
            0: None, 1: ERROR, 2: WARN, 3: INFO, 4: DEBUG, 5: TRACE

        Returns
        -------
        pd.DataFrame()
            Dataset containing mixed features.

        """
        return import_example(data=data, url=url, sep=sep, verbose=verbose)

    def treeplot(self, num_trees=None, plottype='horizontal', figsize=(20, 25), return_ax=False, verbose=3):
        """Tree plot.

        Parameters
        ----------
        num_trees : int, default None
            Best tree is shown when None. Specify the ordinal number of any other target tree.
        plottype : str, (default : 'horizontal')
            Works only in case of xgb model.
                * 'horizontal'
                * 'vertical'
        figsize: tuple, default (25,25)
            Figure size, (height, width)
        verbose : int, (default : 3)
            Print progress to screen.
            0: None, 1: ERROR, 2: WARN, 3: INFO, 4: DEBUG, 5: TRACE

        Returns
        -------
        ax : object

        """
        if ('ensemble' in self.method):
            if self.verbose>=2: print('[hgboost] >Warning: No plot for ensemble is possible yet. <return>')
            return None
        if not hasattr(self, 'model'):
            print('[hgboost] >No model found. Hint: fit a model first using xgboost, catboost or lightboost <return>')
            return None
        ax = None
        # Plot the tree
        ax = tree.plot(self.model, num_trees=num_trees, plottype=plottype, figsize=figsize, verbose=verbose)
        # Return
        if return_ax: return ax

    def plot_cv(self, figsize=(15, 8), cmap='Set2', return_ax=False):
        """Plot the results on the crossvalidation set.

        Parameters
        ----------
        figsize: tuple, default (25,25)
            Figure size, (height, width)

        Returns
        -------
        ax : object
            Figure axis.

        """
        if ('ensemble' in self.method):
            if self.verbose>=2: print('[hgboost] >Warning: No plot for ensemble is possible yet. <return>')
            return None

        print('[hgboost] >%.0d-fold crossvalidation is performed with [%s]' %(self.cv, self.method))
        disable = (False if (self.verbose<3) else True)

        ax = None

        # Make model by using all default parameters
        # space_dumb = {}
        # space_dumb['fit_params'] = {}
        # space_dumb['model_params'] = {}

        # Run the cross-validations
        cv_results = {}
        for i in tqdm(np.arange(0, self.cv), disable=disable):
            name = 'cross ' + str(i)
            if ('_clf' in self.method) and not ('_multi' in self.method):
                # Make train-test
                _, X_test, _, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=None, shuffle=True, stratify=self.y)
                # Evaluate model using hyperoptimized model
                _, cl_results = self._eval(X_test, y_test, self.model, verbose=0)
                cv_results[name] = cl_results
                # Evaluate using default settings
                # _, cl_results_dumb = self._eval(X_test, y_test, model_dumb['model'], verbose=0)
                # cv_results[name + ' (default)'] = cl_results_dumb

            elif '_reg' in self.method:
                # Make train-test
                _, X_test, _, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=None, shuffle=True)
                # Evaluate model using hyperoptimized model
                y_pred = self.predict(X_test, model=self.model)[0]
                cv_results[name] = pd.DataFrame(np.c_[y_test, y_pred], columns=['y', 'y_pred'])

        # Make plots
        if ('_clf' in self.method) and not ('_multi' in self.method):
            ax = cle.plot_cross(cv_results, title=('%.0d-fold crossvalidation results on best-performing %s' %(self.cv, self.method)), figsize=figsize)
        elif '_reg' in self.method:
            fig, ax = plt.subplots(figsize=figsize)
            colors = colourmap.generate(len(cv_results), cmap=cmap)
            for i, key in enumerate(cv_results.keys()):
                sns.regplot('y', 'y_pred', data=cv_results.get(key), ax=ax, color=colors[i, :], label=key)
            ax.legend()
            ax.grid(True)
            ax.set_xlabel('True value')
            ax.set_ylabel('Predicted value')

        return ax

    def plot_validation(self, figsize=(15, 8), cmap='Set2', normalized=None, return_ax=False):
        """Plot the results on the validation set.

        Parameters
        ----------
        normalized: Bool, (default : None)
            Normalize the confusion matrix when True.
        figsize: tuple, default (25,25)
            Figure size, (height, width)

        Returns
        -------
        ax : object
            Figure axis.

        """
        ax = None
        if not hasattr(self, 'model'):
            print('[hgboost] >No model found. Hint: fit a model first using xgboost, catboost or lightboost <return>')
            return None
        if self.val_size is None:
            print('[hgboost] >No validation set found. Hint: use the parameter [val_size=0.2] first <return>')
            return None

        title = 'Results on independent validation set'
        if ('_clf' in self.method) and not ('_multi' in self.method):
            if (self.results.get('val_results', None)) is not None:
                print('[hgboost] >Results are plot from key: "results["val_results"]"')
                if normalized is not None: self.results['val_results']['confmat']['normalized']=normalized
                ax = cle.plot(self.results['val_results'], title=title)
                if return_ax: return ax
        elif ('_reg' in self.method):
            # fig, ax = plt.subplots(figsize=figsize)
            y_pred = self.predict(self.X_val, model=self.model)[0]
            df = pd.DataFrame(np.c_[self.y_val, y_pred], columns=['y', 'y_pred'])

            fig, ax = plt.subplots(figsize=figsize)
            sns.regplot('y', 'y_pred', data=df, ax=ax, color='k', label='Validation set')
            ax.legend()
            ax.grid(True)
            ax.set_title(title)
            ax.set_xlabel('True value')
            ax.set_ylabel('Predicted value')

        return ax

    def plot_params(self, top_n=10, shade=True, cmap='Set2', figsize=(18, 18), return_ax=False):
        """Distribution of parameters.

        Description
        -----------
        This plot demonstrate the density distribution of the used parameters.
        Green will depict the best detected parameter and red demonstrates the top n paramters with best loss.

        Parameters
        ----------
        top_n : int, (default : 10)
            Top n parameters that scored highest are plotted with a black dashed vertical line.
        shade : bool, (default : True)
            Fill the density plot.
        figsize: tuple, default (15,15)
            Figure size, (height, width)

        Returns
        -------
        ax : object
            Figure axis.

        """
        if ('ensemble' in self.method):
            if self.verbose>=2: print('[hgboost] >Warning: No plot for ensemble is possible yet. <return>')
            return None, None

        top_n = np.minimum(top_n, self.results['summary'].shape[0])
        # getcolors = colourmap.generate(top_n, cmap='Reds_r')
        ascending = False if self.greater_is_better else True
        summary_results = self.results['summary'].copy()
        summary_results = summary_results.loc[~summary_results['default_params'], :]

        # Only numerical columns
        # summary_results = summary_results._get_numeric_data()
        # summary_results = summary_results.select_dtypes(include= np.number)
        # params1 = summary_results.columns

        # Sort data based on loss
        colname = 'loss'
        colbest = 'best' # best without cv
        if self.cv is not None:
            colname_cv = 'loss_mean'
            colbest_cv = 'best_cv'
            colnames = [colname_cv, colname]
        else:
            colnames = colname

        # Sort on best loss
        df_summary = summary_results.sort_values(by=colnames, ascending=ascending)
        df_summary.reset_index(inplace=True, drop=True)

        # Get parameters for best scoring model
        idx_best = np.where(df_summary[colbest])[0]
        if self.cv is not None:
            idx_best_cv = np.where(df_summary[colbest_cv])[0]
        # Collect parameters
        params = np.array([*self.results['params'].keys()])
        color_params = colourmap.generate(len(params), cmap=cmap)
        # Setup figure size
        nrCols = 3
        nrRows = int(np.ceil(len(params) / 3))

        # Density plot for each parameter
        fig, ax = plt.subplots(nrRows, nrCols, figsize=figsize)
        i_row = -1

        i=0
        for param in params:
            try:
                # Get row number
                i_col = np.mod(i, nrCols)
                # Make new column
                if i_col == 0: i_row = i_row + 1
                if self.verbose>=5: print('>Plot row: %.0d, col: %.0d' %(i_row, i_col))
                # Retrieve the data from the seperate plots
                y_data = sns.distplot(summary_results[param], hist=False, kde=True, ax=ax[i_row][i_col]).get_lines()[0].get_data()[1]
                # y_data = linefit.get_lines()[0].get_data()[1]

                # Make density
                sns.distplot(summary_results[param],
                             hist=False,
                             kde=True,
                             rug=True,
                             color='darkblue',
                             kde_kws={'shade': shade, 'linewidth': 1, 'color': color_params[i, :]},
                             rug_kws={'color': 'black'},
                             ax=ax[i_row][i_col])

                getvals = df_summary[param].values
                if len(y_data)>0:
                    # Plot the top n (not the first because that one is plotted in green)
                    ax[i_row][i_col].vlines(getvals[1:top_n], np.min(y_data), np.max(y_data), linewidth=1, colors='k', linestyles='dashed', label='Top ' + str(top_n) + ' models')

                    # Plot the best (without cv)
                    ax[i_row][i_col].vlines(getvals[idx_best], np.min(y_data), np.max(y_data), linewidth=2, colors='g', linestyles='dashed', label='Best (without cv)')

                    # Plot the best (with cv)
                    if self.cv is not None:
                        ax[i_row][i_col].vlines(getvals[idx_best_cv], np.min(y_data), np.max(y_data), linewidth=2, colors='r', linestyles='dotted', label='Best ' + str(self.cv) + '-fold cv')

                if self.cv is not None:
                    ax[i_row][i_col].set_title(('%s: %.3g (%.0d-fold cv)' %(param, getvals[idx_best_cv], self.cv)))
                else:
                    ax[i_row][i_col].set_title(('%s: %.3g' %(param, getvals[idx_best])))
                ax[i_row][i_col].set_ylabel('Density')
                ax[i_row][i_col].grid(True)
                ax[i_row][i_col].legend(loc='upper right')
                # print(param)
                # print(i_col)
                # print(i_row)
                i = i + 1
            except:
                pass

        # Scatter plot
        df_sum = self.results['summary'].copy()
        df_sum = df_sum.loc[~df_sum['default_params'], :]
        df_sum = df_sum.sort_values(by='tid', ascending=True)
        df_sum.reset_index(inplace=True, drop=True)
        idx_best_without_cv = np.where(df_sum[colbest])[0]

        if self.cv is not None:
            idx_best_cv = np.where(df_sum[colbest_cv])[0]
        df_sum = df_sum.fillna(0)

        fig2, ax2 = plt.subplots(nrRows, nrCols, figsize=figsize)
        i_row = -1
        i=0
        for param in params:
            try:
                # Get row number
                i_col = np.mod(i, nrCols)
                # Make new column
                if i_col == 0: i_row = i_row + 1
                # Make the plot
                sns.regplot('tid', param, data=df_sum, ax=ax2[i_row][i_col], color=color_params[i, :])

                # Scatter top n values, start with 1 because the 0 is, based on the ranking, with CV.
                ax2[i_row][i_col].scatter(df_summary['tid'].values[1:top_n], df_summary[param].values[1:top_n], s=50, color='k', marker='.', label='Top ' + str(top_n) + ' models')

                # Scatter best value
                ax2[i_row][i_col].scatter(df_sum['tid'].values[idx_best_without_cv], df_sum[param].values[idx_best_without_cv], s=100, color='g', marker='*', label='Best (without cv)')

                # Scatter best cv
                if self.cv is not None:
                    ax2[i_row][i_col].scatter(df_sum['tid'].values[idx_best_cv], df_sum[param].values[idx_best_cv], s=100, color='r', marker='x', label='Best ' + str(self.cv) + '-fold cv')

                # Set labels
                ax2[i_row][i_col].set(xlabel='iteration', ylabel='{}'.format(param), title='{} over Search'.format(param))
                if self.cv is not None:
                    ax2[i_row][i_col].set_title(('%s: %.3g (%.0d-fold cv)' %(param, df_sum[param].values[idx_best_cv], self.cv)))
                else:
                    ax2[i_row][i_col].set_title(('%s: %.3g' %(param, df_sum[param].values[idx_best_without_cv])))

                ax2[i_row][i_col].grid(True)
                ax2[i_row][i_col].legend(loc='upper right')
                i = i + 1
            except:
                pass

        if return_ax:
            return ax, ax2

    def plot(self, ylim=None, figsize=(15, 10), plot2=True, return_ax=False):
        """Plot the summary results.

        Parameters
        ----------
        ylim : tuple
            Set the y-limit. In case of auc it can be: (0.5, 1)
        figsize: tuple, default (25,25)
            Figure size, (height, width)

        Returns
        -------
        ax : object
            Figure axis.

        """
        ax1, ax2 = None, None
        if ('ensemble' in self.method):
            if self.verbose>=2: print('[hgboost] >Warning: No plot for ensemble is possible yet. <return>')
            self.plot_ensemble(ylim, figsize, ax1, ax2)
            return ax1, ax2
        if (not hasattr(self, 'model')):
            print('[hgboost] >No model found. Hint: fit a model first using xgboost, catboost or lightboost <return>')
            return ax1, ax2

        # Plot comparison between hyperoptimized vs basic model
        if (self.results.get('comparison_results', None) is not None) and ([*self.results['comparison_results'].values()][0] is not None):
            cle.plot_cross(self.results['comparison_results'], title='Comparison between model with optimized hyperparamters vs. default parameters on validation set.')

        if hasattr(self.model, 'evals_result') and plot2:
            _, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        else:
            _, ax1 = plt.subplots(1, 1, figsize=figsize)

        tmpdf = self.results['summary'].sort_values(by='tid', ascending=True)
        tmpdf = tmpdf.loc[~tmpdf['loss'].isna(), :]

        # Plot results with testsize
        idx = np.where(tmpdf['best'].values)[0]
        ax1.hlines(tmpdf['loss'].iloc[idx], 0, tmpdf['loss'].shape[0], colors='g', linestyles='dashed', label='Best model (without cv)')
        ax1.vlines(idx, tmpdf['loss'].min(), tmpdf['loss'].iloc[idx], colors='g', linestyles='dashed')
        best_loss = tmpdf['loss'].iloc[idx]
        title = ('%s (%s: %.3g)' %(self.method, self.eval_metric, best_loss))

        # Plot results with cv
        if self.cv is not None:
            ax1.errorbar(tmpdf['tid'], tmpdf['loss_mean'], tmpdf['loss_std'], marker='s', mfc='red', label=str(self.cv) + '-fold cv for top ' + str(self.top_cv_evals) + ' scoring models')
            idx = np.where(tmpdf['best_cv'].values)[0]
            ax1.hlines(tmpdf['loss_mean'].iloc[idx], 0, tmpdf['loss_mean'].shape[0], colors='r', linestyles='dotted', label='Best model (' + str(self.cv) + '-fold cv)')
            ax1.vlines(idx, tmpdf['loss'].min(), tmpdf['loss_mean'].iloc[idx], colors='r', linestyles='dashed')
            best_loss = tmpdf['loss_mean'].iloc[idx]
            title = ('%s (%.0d-fold cv mean %s: %.3g)' %(self.method, self.cv, self.eval_metric, best_loss))
            ax1.set_xlabel('Model number')

        # Plot all other evalution results
        ax1.scatter(tmpdf['tid'].values, tmpdf['loss'].values, s=10, label='All models')

        # Set labels
        ax1.set_title(title)
        ax1.set_ylabel(self.eval_metric)
        ax1.grid(True)
        ax1.legend()
        if ylim is not None: ax1.set_ylim(ylim)

        if hasattr(self.model, 'evals_result') and plot2:
            eval_metric = [*self.model.evals_result()['validation_0'].keys()][0]
            ax2.plot([*self.model.evals_result()['validation_0'].values()][0], label='Train error')
            ax2.plot([*self.model.evals_result()['validation_1'].values()][0], label='Test error')
            ax2.set_ylabel(eval_metric)
            ax2.set_title(self.method)
            ax2.grid(True)
            ax2.legend()

        if return_ax:
            return ax1, ax2

    def plot_ensemble(self, ylim, figsize, ax1, ax2):
        """Plot ensemble results.

        Parameters
        ----------
        ylim : tuple
            Set the y-limit. In case of auc it can be: (0.5, 1)
        figsize: tuple, default (25,25)
            Figure size, (height, width)
        ax1 : Object
            Axis of figure 1
        ax2 : Object
            Axis of figure 2

        Returns
        -------
        ax : object
            Figure axis.

        """
        # Get models
        keys = np.array([*self.results.keys()])
        if self.method=='ensemble_reg':
            Iloc = list(map(lambda x: '_reg' in x, keys))
        else:
            Iloc = list(map(lambda x: '_clf' in x, keys))

        methods = keys[Iloc]

        for method in methods:
            model = self.results[method]
            # self.results['summary'] = pd.concat([hgbX.results['summary'], hgbC.results['summary'], hgbL.results['summary']])
            # ax1, ax2 = plot_summary(model, ylim=ylim, figsize=figsize, return_ax=True, method=method, ax1=ax1, ax2=ax2)

    def save(self, filepath='hgboost_model.pkl', overwrite=False, verbose=3):
        """Save learned model in pickle file.

        Parameters
        ----------
        filepath : str, (default: 'hgboost_model.pkl')
            Pathname to store pickle files.
        overwrite : bool, (default=False)
            Overwite file if exists.
        verbose : int, optional
            Show message. A higher number gives more informatie. The default is 3.

        Returns
        -------
        bool : [True, False]
            Status whether the file is saved.

        """
        if (filepath is None) or (filepath==''):
            filepath = 'hgboost_model.pkl'
        if filepath[-4:] != '.pkl':
            filepath = filepath + '.pkl'
        # Store data
        storedata = {}
        storedata['results'] = self.results
        storedata['method'] = self.method
        storedata['eval_metric'] = self.eval_metric
        storedata['greater_is_better'] = self.greater_is_better
        storedata['space'] = self.space
        storedata['model'] = self.model
        storedata['algo'] = self.algo
        storedata['max_eval'] = self.max_eval
        storedata['top_cv_evals'] = self.top_cv_evals
        storedata['threshold'] = self.threshold
        storedata['test_size'] = self.test_size
        storedata['val_size'] = self.val_size
        storedata['cv'] = self.cv
        storedata['random_state'] = self.random_state
        storedata['n_jobs'] = self.n_jobs
        storedata['verbose'] = self.verbose
        storedata['is_unbalance'] = self.is_unbalance
        # Save
        status = pypickle.save(filepath, storedata, overwrite=overwrite, verbose=verbose)
        if verbose>=3: print('[hgboost] >Saving.. %s' %(status))
        # return
        return status

    def load(self, filepath='hgboost_model.pkl', verbose=3):
        """Load learned model.

        Parameters
        ----------
        filepath : str
            Pathname to stored pickle files.
        verbose : int, optional
            Show message. A higher number gives more information. The default is 3.

        Returns
        -------
        Object.

        """
        if (filepath is None) or (filepath==''):
            filepath = 'hgboost_model.pkl'
        if filepath[-4:]!='.pkl':
            filepath = filepath + '.pkl'
        # Load
        storedata = pypickle.load(filepath, verbose=verbose)
        # Store in self
        if storedata is not None:
            self.results = storedata['results']
            self.method = storedata['method']
            self.eval_metric = storedata['eval_metric']
            self.greater_is_better = storedata['greater_is_better']
            self.model = storedata['model']
            self.space = storedata['space']
            self.algo = storedata['algo']
            self.max_eval = storedata['max_eval']
            self.top_cv_evals = storedata['top_cv_evals']
            self.threshold = storedata['threshold']
            self.test_size = storedata['test_size']
            self.val_size = storedata['val_size']
            self.cv = storedata['cv']
            self.is_unbalance = storedata['is_unbalance']
            self.random_state = storedata['random_state']
            self.n_jobs = storedata['n_jobs']
            self.verbose = storedata['verbose']

            if verbose>=3: print('[hgboost] >Loading succesful!')
            # Return results
            return self.results
        else:
            if verbose>=2: print('[hgboost] >WARNING: Could not load data.')


# %%
def _store_validation_scores(results_summary, best_params, model_basic, val_score_basic, val_score, greater_is_better):
    # Store default parameters
    params = [*best_params.keys()]
    idx = results_summary.index.max() + 1
    results_summary.loc[idx] = np.nan
    # Store the params of the basic model
    for param in params:
        results_summary[param].iloc[idx] = model_basic.get_params().get(param, None)

    results_summary['loss_validation'] = np.nan
    results_summary['loss_validation'].iloc[idx] = val_score_basic['loss'] * (-1 if greater_is_better else 1)
    results_summary['default_params'] = False
    results_summary['default_params'].iloc[idx] = True

    # Store the best loss validation score for the hyperoptimized model
    if np.any(results_summary.columns.str.contains('best_cv')):
        idx = np.where(results_summary['best_cv']==1)[0]
    else:
        idx = np.where(results_summary['best']==1)[0]
    results_summary['loss_validation'].iloc[idx] = val_score['loss'] * (-1 if greater_is_better else 1)

    return results_summary


# %% Import example dataset from github.
def import_example(data='titanic', url=None, sep=',', verbose=3):
    """Import example dataset from github source.

    Description
    -----------
    Import one of the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str, (default : "titanic")
        Name of datasets: 'sprinkler', 'titanic', 'student', 'fifa', 'cancer', 'waterpump', 'retail'
    url : str
        url link to to dataset.
    verbose : int, (default : 3)
        Print progress to screen.
        0: None, 1: ERROR, 2: WARN, 3: INFO, 4: DEBUG, 5: TRACE

    Returns
    -------
    pd.DataFrame()
        Dataset containing mixed features.

    """
    if url is None:
        if data=='sprinkler':
            url='https://erdogant.github.io/datasets/sprinkler.zip'
        elif data=='titanic':
            url='https://erdogant.github.io/datasets/titanic_train.zip'
        elif data=='student':
            url='https://erdogant.github.io/datasets/student_train.zip'
        elif data=='cancer':
            url='https://erdogant.github.io/datasets/cancer_dataset.zip'
        elif data=='fifa':
            url='https://erdogant.github.io/datasets/FIFA_2018.zip'
        elif data=='waterpump':
            url='https://erdogant.github.io/datasets/waterpump/waterpump_test.zip'
        elif data=='retail':
            url='https://erdogant.github.io/datasets/marketing_data_online_retail_small.zip'
    else:
        data = wget.filename_from_url(url)

    if url is None:
        if verbose>=3: print('[hgboost] >Nothing to download.')
        return None

    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    PATH_TO_DATA = os.path.join(curpath, wget.filename_from_url(url))
    if not os.path.isdir(curpath):
        os.makedirs(curpath, exist_ok=True)

    # Check file exists.
    if not os.path.isfile(PATH_TO_DATA):
        if verbose>=3: print('[hgboost] >Downloading [%s] dataset from github source..' %(data))
        wget.download(url, curpath)

    # Import local dataset
    if verbose>=3: print('[hgboost] >Import dataset [%s]' %(data))
    df = pd.read_csv(PATH_TO_DATA, sep=sep)
    # Return
    return df


# %% Set the search spaces
def _get_params(fn_name, eval_metric=None, y=None, pos_label=None, is_unbalance=False, gpu=False, verbose=3):
    # choice : categorical variables
    # quniform : discrete uniform (integers spaced evenly)
    # uniform: continuous uniform (floats spaced evenly)
    # loguniform: continuous log uniform (floats spaced evenly on a log scale)
    early_stopping_rounds = 25
    if eval_metric is None: raise ValueError('[hgboost] >eval_metric must be provided.')
    if verbose>=3: print('[hgboost] >Collecting %s parameters.' %(fn_name))

    ############### XGB parameters ###############
    if fn_name=='xgb_reg':
        # Enable/Disable GPU
        if gpu:
            # 'gpu_hist' Equivalent to the XGBoost fast histogram algorithm. Much faster and uses considerably less memory. NOTE: May run very slowly on GPUs older than Pascal architecture.
            tree_method = 'auto'  # 'gpu_hist' gives throws random errors
            predictor = 'gpu_predictor'
        else:
            tree_method = 'hist'
            predictor = 'cpu_predictor'

        xgb_reg_params = {
            'learning_rate': hp.quniform('learning_rate', 0.05, 0.31, 0.05),
            'max_depth': hp.choice('max_depth', np.arange(5, 30, 1, dtype=int)),
            'min_child_weight': hp.choice('min_child_weight', np.arange(1, 10, 1, dtype=int)),
            'gamma': hp.choice('gamma', [0, 0.25, 0.5, 1.0]),
            'reg_lambda': hp.choice('reg_lambda', [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'n_estimators': hp.choice('n_estimators', range(20, 205, 5)),
            'tree_method': tree_method,
            'gpu_id': 0,
            'predictor': predictor,
        }
        space = {}
        space['model_params'] = xgb_reg_params
        space['fit_params'] = {'early_stopping_rounds': early_stopping_rounds, 'verbose': 0}
        return(space)

    ############### LightGBM regression parameters ###############
    if fn_name=='lgb_reg':
        # Enable/Disable GPU
        device = 'gpu' if gpu else 'cpu'

        lgb_reg_params = {
            'learning_rate': hp.quniform('learning_rate', 0.05, 0.31, 0.05),
            'max_depth': hp.choice('max_depth', np.arange(5, 30, 1, dtype=int)),
            'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
            'subsample': hp.uniform('subsample', 0.8, 1),
            'n_estimators': hp.choice('n_estimators', range(20, 205, 5)),
            'device': device,
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
        }
        space = {}
        space['model_params'] = lgb_reg_params
        space['fit_params'] = {'eval_metric': 'l2', 'early_stopping_rounds': early_stopping_rounds, 'verbose': 0}

        return(space)

    ############### CatBoost regression parameters ###############
    if fn_name=='ctb_reg':
        # Enable/Disable GPU
        task_type = 'CPU' if gpu else 'CPU'

        # Catboost parameters
        ctb_reg_params = {
            'learning_rate': hp.quniform('learning_rate', 0.05, 0.31, 0.05),
            'max_depth': hp.choice('max_depth', np.arange(2, 16, 1, dtype=int)),
            'colsample_bylevel': hp.choice('colsample_bylevel', np.arange(0.3, 0.8, 0.1)),
            'n_estimators': hp.choice('n_estimators', range(20, 205, 5)),
            'task_type': task_type,
            'devices': '0',
        }
        space = {}
        space['model_params'] = ctb_reg_params
        space['fit_params'] = {'early_stopping_rounds': early_stopping_rounds, 'verbose': 0}
        return(space)

    ############### CatBoost classification parameters ###############
    if fn_name=='ctb_clf':
        # Enable/Disable GPU
        task_type = 'CPU' if gpu else 'CPU'

        # Class sizes
        if is_unbalance:
            # https://catboost.ai/docs/concepts/python-reference_parameters-list.html#python-reference_parameters-list
            if verbose>=3: print('[hgboost] >Correct for unbalanced classes using [auto_class_weights].')
            scale_pos_weight = np.sum(y!=pos_label) / np.sum(y==pos_label)
        else:
            scale_pos_weight = hp.choice('scale_pos_weight', np.arange(1, 101, 9))

        ctb_clf_params={
            'learning_rate': hp.choice('learning_rate', np.logspace(np.log10(0.005), np.log10(0.31), base=10, num=1000)),
            'depth': hp.choice('max_depth', np.arange(2, 16, 1, dtype=int)),
            'iterations': hp.choice('iterations', np.arange(100, 1000, 100)),
            'l2_leaf_reg': hp.choice('l2_leaf_reg', np.arange(1, 100, 2)),
            'border_count': hp.choice('border_count', np.arange(5, 200, 1)),
            'thread_count': 4,
            'scale_pos_weight': scale_pos_weight,
            'task_type': task_type,
            'devices': '0',
        }
        space={}
        space['model_params']=ctb_clf_params
        space['fit_params']={'early_stopping_rounds': early_stopping_rounds, 'verbose': 0}
        return(space)

    ############### LightBoost classification parameters ###############
    if fn_name=='lgb_clf':
        # Enable/Disable GPU
        device = 'gpu' if gpu else 'cpu'

        # Class sizes
        if is_unbalance:
            if verbose>=3: print('[hgboost] >Correct for unbalanced classes using [is_unbalance]..')
            is_unbalance = [True]
        else:
            is_unbalance = [True, False]

        lgb_clf_params={
            'learning_rate': hp.choice('learning_rate', np.logspace(np.log10(0.005), np.log10(0.5), base=10, num=1000)),
            'max_depth': hp.choice('max_depth', np.arange(5, 75, 1)),
            'boosting_type': hp.choice('boosting_type', ['gbdt', 'goss', 'dart']),
            'num_leaves': hp.choice('num_leaves', np.arange(100, 1000, 100)),
            'n_estimators': hp.choice('n_estimators', np.arange(20, 205, 5)),
            'subsample_for_bin': hp.choice('subsample_for_bin', np.arange(20000, 300000, 20000)),
            'min_child_samples': hp.choice('min_child_weight', np.arange(20, 500, 5)),
            'reg_alpha': hp.quniform('reg_alpha', 0, 1, 0.01),
            'reg_lambda': hp.quniform('reg_lambda', 0, 1, 0.01),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.6, 1, 0.01),
            'subsample': hp.quniform('subsample', 0.5, 1, 100),
            'bagging_fraction': hp.choice('bagging_fraction', np.arange(0.2, 1, 0.2)),
            'is_unbalance': hp.choice('is_unbalance', is_unbalance),
            'device': device,
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
        }
        space={}
        space['model_params']=lgb_clf_params
        space['fit_params']={'early_stopping_rounds': early_stopping_rounds, 'verbose': 0}
        return(space)

    # ############## XGboost classification parameters ###############
    if 'xgb_clf' in fn_name:
        # Enable/Disable GPU
        if gpu:
            # 'gpu_hist' Equivalent to the XGBoost fast histogram algorithm. Much faster and uses considerably less memory. NOTE: May run very slowly on GPUs older than Pascal architecture.
            tree_method = 'auto'  # 'gpu_hist' gives throws random errors
            predictor = 'gpu_predictor'
        else:
            tree_method = 'hist'
            predictor = 'cpu_predictor'

        xgb_clf_params={
            'learning_rate': hp.choice('learning_rate', np.logspace(np.log10(0.005), np.log10(0.5), base=10, num=1000)),
            'max_depth': hp.choice('max_depth', range(5, 32, 1)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            'gamma': hp.choice('gamma', [0.5, 1, 1.5, 2, 5]),
            'subsample': hp.quniform('subsample', 0.1, 1, 0.01),
            'n_estimators': hp.choice('n_estimators', range(20, 205, 5)),
            'booster': 'gbtree',
            'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1.0, 0.01),
            'tree_method': tree_method,
            'gpu_id': 0,
            'predictor': predictor,
        }

        if fn_name=='xgb_clf':
            # xgb_clf_params['eval_metric']=hp.choice('eval_metric', ['error', eval_metric])
            xgb_clf_params['objective']='binary:logistic'
            # Class sizes
            if is_unbalance:
                if verbose>=3: print('[hgboost] >Correct for unbalanced classes using [scale_pos_weight]..')
                scale_pos_weight = np.sum(y!=pos_label) / np.sum(y==pos_label)
            else:
                scale_pos_weight = hp.choice('scale_pos_weight', np.arange(1, 101, 9))
            xgb_clf_params['scale_pos_weight']=scale_pos_weight

        if fn_name=='xgb_clf_multi':
            xgb_clf_params['objective']='multi:softprob'
            # scoring='kappa'

        space={}
        space['model_params']=xgb_clf_params
        space['fit_params']={'early_stopping_rounds': early_stopping_rounds, 'verbose': 0}

        if verbose>=3: print('[hgboost] >[%.0d] hyperparameters in gridsearch space. Used loss function: [%s].' %(len([*space['model_params']]), eval_metric))
        return(space)


def _check_input(X, y, pos_label, method, verbose=4):
    # X should be of type dataframe
    if (type(X) is not pd.DataFrame):
        raise ValueError('[hgboost] >Error: dataset X should be of type pd.DataFrame')

    # y should be of type array-like
    if (type(y) is not np.ndarray):
        raise ValueError('[hgboost] >Error: Response variable y should be of type numpy array')

    # Check None and np.nan values in string/numeric type
    if 'str' in str(type(y[0])):
        if any(elem is None for elem in y): raise ValueError('[hgboost] >Error: Response variable y can not have None values.')
    else:
        if np.any(np.isnan(y)): raise ValueError('[hgboost] >Error: Response variable y can not have nan values.')

    if ('_reg' in method):
        pos_label=None

    # Set pos_label and y
    if (pos_label is not None) and ('_clf' in method):
        if verbose>=4: print('[hgboost] >pos_label is used to set [%s].' %(pos_label))
        y=y==pos_label
        pos_label=True

    # Checks pos_label status in case of method is classification
    if ('_clf' in method) and (pos_label is None) and (str(y.dtype)=='bool'):
        pos_label=True
        if verbose>=4: print('[hgboost] >[pos_label] is set to [%s] because [y] is of type [bool].' %(pos_label))

    # Raise ValueError in case of pos_label is not set and not bool.
    if ('_clf' in method) and (pos_label is None) and (len(np.unique(y))==2) and not (str(y.dtype)=='bool'):
        raise ValueError('[hgboost] >Error: In a two-class approach [%s], pos_label needs to be set or of type bool.' %(pos_label))

    # Check label for classificaiton and two-class model
    if ('_clf' in method) and (pos_label is not None) and (len(np.unique(y))==2) and not (np.any(np.isin(y.astype(str), str(pos_label)))):
        raise ValueError('[hgboost] >Error: y contains values %s but none matches pos_label=%s <return>' %(str(np.unique(y)), pos_label))

    # two-class classifier should have 2 classes
    if ('_clf' in method) and not ('_multi' in method) and (len(np.unique(y))>2) and (pos_label is None):
        raise ValueError('[hgboost] >Error: [y] should contain exactly 2 unique classes. Hint: use method="xgb_clf_multi"')

    # multi-class method should have >2 classes
    if ('_clf_multi' in method) and (len(np.unique(y))<=2):
        raise ValueError('[hgboost] >Error: [xgb_clf_multi] requires >2 classes. Hint: use method="xgb_clf"')

    if ('_clf_multi' in method):
        pos_label=None
        if verbose>=4: print('[hgboost] >[pos_label] is set to [None] because [method] is of type [%s].' %(method))

    # Check counts y
    y_counts=np.unique(y, return_counts=True)[1]
    if np.any(y_counts<5) and ('_clf' in method):
        raise ValueError('[hgboost] >Error: [y] contains [%.0d] classes with < 5 samples. Each class should have >=5 samples.' %(sum(y_counts<5)))
    # Check number of classes, should be >1
    if (len(np.unique(y))<=1) and ('_clf' in method):
        raise ValueError('[hgboost] >Error: [y] should have >= 2 classes.')

    # Set X
    X.reset_index(drop=True, inplace=True)
    X.columns=X.columns.values.astype(str)
    if verbose>=4: print('[hgboost] >Reset index for X.')

    # Return
    return X, y, pos_label


def _check_eval_metric(method, eval_metric, greater_is_better, verbose=3):
    # Check the eval_metric
    if (eval_metric is None) and ('_reg' in method):
        eval_metric='rmse'
    elif (eval_metric is None) and ('_clf_multi' in method):
        eval_metric='kappa'
    elif (eval_metric is None) and ('_clf' in method):
        eval_metric='auc'

    # Check the greater_is_better for evaluation metric
    if greater_is_better is None:
        if (eval_metric == 'f1'):
            greater_is_better=True
        elif 'auc' in eval_metric:
            greater_is_better=True
        elif (eval_metric == 'kappa'):
            greater_is_better=True
        elif (eval_metric == 'rmse'):
            greater_is_better=False
        elif (eval_metric == 'mse'):
            greater_is_better=False
        elif (eval_metric == 'mae'):
            greater_is_better=False
        else:
            if verbose>=2: print('[hgboost] >[%s] is not a implemented option. [greater_is_better] is set to %s' %(eval_metric, str(greater_is_better)))

    # Return
    return eval_metric, greater_is_better
