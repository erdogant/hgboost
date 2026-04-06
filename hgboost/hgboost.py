"""hgboost: Hyperoptimized Gradient Boosting library.

Contributors: https://github.com/erdogant/hgboost
"""

import warnings
warnings.filterwarnings("ignore")
import logging

import classeval as cle
from df2onehot import df2onehot
import treeplot as tree
import colourmap
import pypickle
import datazets as dz

import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import copy

from sklearn.metrics import mean_squared_error, cohen_kappa_score, mean_absolute_error, log_loss, roc_auc_score, f1_score
from sklearn.ensemble import VotingClassifier, VotingRegressor
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from hyperopt import fmin, tpe, STATUS_OK, Trials, hp
import xgboost as xgb

try:
    import catboost as ctb
    catboost_available = True
except ImportError:
    catboost_available = False


try:
    import lightgbm as lgb
    lightgbm_available = True
except ImportError:
    lightgbm_available = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verbose → logging-level mapping
#   verbose 0  → logging.CRITICAL  (silent)
#   verbose 1  → logging.ERROR
#   verbose 2  → logging.WARNING
#   verbose 3  → logging.INFO
#   verbose 4  → logging.DEBUG
# ---------------------------------------------------------------------------
_VERBOSE_TO_LEVEL = {
    'silent': logging.CRITICAL,
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG,
}


def _set_logger_level(verbose: int) -> None:
    """Adjust the module logger threshold to match *verbose*."""
    level = _VERBOSE_TO_LEVEL.get(verbose.lower(), logging.DEBUG)
    logger.setLevel(level)
    # Set logger hyperopt to warning
    logging.getLogger("hyperopt").setLevel(level)


# %%
class hgboost:
    """hgboost: Hyperoptimized Gradient Boosting.

    Description
    -----------
    HGBoost stands for Hyperoptimized Gradient Boosting and is a Python package for hyperparameter optimization
    for XGBoost, LightBoost, and CatBoost. It will carefully split the dataset into a train, test, and independent
    validation set. Within the train-test set, there is the inner loop for optimizing the hyperparameters using
    Bayesian optimization (with hyperopt) and, the outer loop to score how well the top performing models can
    generalize based on k-fold cross validation. As such, it will make the best attempt to select the most robust
    model with the best performance.

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
    is_unbalanced : Bool, (default: True)
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
    verbose : str, (default : 'info')
        Print progress to screen.
        0: None, 1: ERROR, 2: WARN, 3: INFO, 4: DEBUG, 5: TRACE

    Returns
    -------
    None.

    References
    ----------
    * Medium : https://erdogant.medium.com
    * Github : https://github.com/erdogant/hgboost
    * Documentation pages: https://erdogant.github.io/hgboost/
    * Notebook Classification: https://colab.research.google.com/github/erdogant/hgboost/blob/master/notebooks/hgboost_classification_examples.ipynb
    * Notebook Regression: https://colab.research.google.com/github/erdogant/hgboost/blob/master/notebooks/hgboost_regression_examples.ipynb

    """
    def __init__(self, max_eval=250, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, is_unbalance=True, random_state=None, n_jobs=-1, gpu=False, early_stopping_rounds=25, verbose='info'):
        """Initialize hgboost with user-defined parameters."""
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
        self.is_unbalanced = is_unbalanced
        self.gpu = gpu
        self.early_stopping_rounds=early_stopping_rounds
        _set_logger_level(verbose)


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
        logger.info('method: %s' % self.method)
        logger.info('eval_metric: %s' % self.eval_metric)
        logger.info('larger_is_better: %s' % self.larger_is_better)

        # Set validation set
        self._set_validation_set(X, y)
        # Find best parameters
        self.model, self.results = self._HPOpt()
        # Fit on all data using best parameters
        logger.info('*' * 89)
        logger.info('Retrain [%s] on the entire dataset with the optimal hyperparameters.' % self.method)
        self.model.fit(X, y)
        # Return
        return self.results

    def _classification(self, X, y, eval_metric, larger_is_better, params):
        # Gather for method, the default metric and greater is better.
        self.eval_metric, self.larger_is_better =_check_eval_metric(self.method, eval_metric, larger_is_better)
        # Import search space for the specific function
        if params == 'default': params = _get_params(self.method, eval_metric=self.eval_metric, y=y, pos_label=self.pos_label, is_unbalanced=self.is_unbalanced, gpu=self.gpu, early_stopping_rounds=self.early_stopping_rounds, verbose=self.verbose)
        self.space = params
        # Fit model
        self.results = self._fit(X, y, pos_label=self.pos_label)
        # Fin
        logger.info('Fin!')

    def _regression(self, X, y, eval_metric, larger_is_better, params):
        # Gather for method, the default metric and greater is better.
        self.eval_metric, self.larger_is_better = _check_eval_metric(self.method, eval_metric, larger_is_better)
        # Import search space for the specific function
        if params == 'default': params = _get_params(self.method, eval_metric=self.eval_metric, gpu=self.gpu, verbose=self.verbose)
        self.space = params
        # Fit model
        self.results = self._fit(X, y)
        # Fin
        logger.info('Fin!')

    def xgboost_reg(self, X, y, eval_metric='rmse', larger_is_better=False, params='default'):
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
        larger_is_better : bool (default : False).
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
        logger.info('Start hgboost regression.')
        # Method
        self.method='xgb_reg'
        # Run method
        self._regression(X, y, eval_metric, larger_is_better, params)
        # Return
        return self.results

    def lightboost_reg(self, X, y, eval_metric='rmse', larger_is_better=False, params='default'):
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
        larger_is_better : bool (default : False).
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
        logger.info('Start hgboost regression.')
        if not lightgbm_available:
            logger.warning('Lightboost not installed. First pip install lightgbm>=4.1.0')
            return

        # Method
        self.method='lgb_reg'
        # Run method
        self._regression(X, y, eval_metric, larger_is_better, params)
        # Return
        return self.results

    def catboost_reg(self, X, y, eval_metric='rmse', larger_is_better=False, params='default'):
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
        larger_is_better : bool (default : False).
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
        logger.info('Start hgboost regression.')
        if not catboost_available:
            logger.warning('Catboost not installed. First pip install catboost')
            return

        if self.gpu:
            logger.warning('GPU for catboost is not supported. It throws an error because multiple evaluation sets are readily optimized.')
            self.gpu=False
        # Method
        self.method='ctb_reg'
        # Run method
        self._regression(X, y, eval_metric, larger_is_better, params)
        # Return
        return self.results

    def xgboost(self, X, y, pos_label=None, method='xgb_clf', eval_metric=None, larger_is_better=None, params='default'):
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
        larger_is_better : bool.
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
        logger.info('Start hgboost classification.')
        self.method = method
        self.pos_label = pos_label
        # Run method
        self._classification(X, y, eval_metric, larger_is_better, params)
        # Return
        return self.results

    def catboost(self, X, y, pos_label=None, eval_metric='auc', larger_is_better=True, params='default'):
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
        larger_is_better : bool (default : True).
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
        logger.info('Start hgboost classification.')
        if not catboost_available:
            logger.warning('Catboost not installed. First pip install catboost')
            return
        if self.gpu:
            logger.warning('GPU for catboost is not supported. It throws an error because I am readily optimizing across multiple evaluation sets.')

        self.method = 'ctb_clf'
        self.pos_label = pos_label
        # Run method
        self._classification(X, y, eval_metric, larger_is_better, params)
        # Return
        return self.results

    def lightboost(self, X, y, pos_label=None, eval_metric='auc', larger_is_better=True, params='default'):
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
        larger_is_better : bool (default : True)
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
        logger.info('Start hgboost classification.')
        if not lightgbm_available:
            logger.warning('Lightboost not installed. First pip install lightgbm>=4.1.0')
            return
        
        self.method = 'lgb_clf'
        self.pos_label = pos_label
        # Run method
        self._classification(X, y, eval_metric, larger_is_better, params)
        # Return
        return self.results

    def ensemble(self, X, y, pos_label=None, methods=['xgb_clf', 'ctb_clf', 'lgb_clf'], eval_metric=None, larger_is_better=None, voting='soft'):
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
        larger_is_better : bool (default : True)
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
        
        self.methods = []
        if np.isin('xgb_clf', methods):
            self.methods.append('xgb_clf')
        if catboost_available:
            self.methods.append('ctb_clf')
        if lightgbm_available:
            self.methods.append('lgb_clf')
        if np.isin('xgb_reg', methods):
            self.methods.append('xgb_reg')
        if catboost_available:
            self.methods.append('ctb_reg')
        if lightgbm_available:
            self.methods.append('lgb_reg')
        
        if np.all(list(map(lambda x: 'clf' in x, methods))):
            logger.info('Create ensemble classification model..')
            self.method = 'ensemble_clf'
        elif np.all(list(map(lambda x: 'reg' in x, methods))):
            logger.info('Create ensemble regression model..')
            self.method = 'ensemble_reg'
        else:
            raise ValueError('[hgboost] >Error: The input [methods] must be of type "_clf" or "_reg" but can not be combined.')

        # Check input data
        X, y, self.pos_label = _check_input(X, y, pos_label, self.method, verbose=self.verbose)
        # Gather for method, the default metric and greater is better.
        self.eval_metric, self.larger_is_better = _check_eval_metric(self.method, eval_metric, larger_is_better)
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
            hgbM._classification(X_train, y_train, eval_metric, larger_is_better, 'default')
            # Store
            models.append((method, copy.copy(hgbM.model)))
            self.results[method] = {}
            self.results[method]['model'] = copy.copy(hgbM)

        # Create the ensemble model
        logger.info('Fit ensemble model with [%s] voting..' % self.voting)
        if self.method == 'ensemble_clf':
            model = VotingClassifier(models, voting=voting, n_jobs=self.n_jobs)
            model.fit(X, y==pos_label)
        else:
            model = VotingRegressor(models, n_jobs=self.n_jobs)
            model.fit(X, y)
        # Store ensemble model
        self.model = model

        # Validation error for the ensemble model
        logger.info('Evalute [ensemble] model on independent validation dataset (%.0f samples, %.2g%%)' % (len(y_val), self.val_size * 100))
        # Evaluate results on the same validation set
        val_score, val_results = self._eval(X_val, y_val, model)
        logger.info('[Ensemble] [%s]: %.4g on independent validation dataset' % (self.eval_metric, val_score['loss']))

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
                logger.info('[%s]  [%s]: %.4g on independent validation dataset' % (method, self.eval_metric, val_score_M['loss']))

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
        logger.info('*' * 89)
        logger.info('Total dataset: %s ' % str(X.shape))

        if (self.val_size is not None):
            if '_clf' in self.method:
                self.X, self.X_val, self.y, self.y_val = train_test_split(X, y, test_size=self.val_size, random_state=self.random_state, shuffle=True, stratify=y)
            elif '_reg' in self.method:
                self.X, self.X_val, self.y, self.y_val = train_test_split(X, y, test_size=self.val_size, random_state=self.random_state, shuffle=True)
            logger.info('Validation set: %s ' % str(self.X_val.shape))
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
        disable = (False if (self.verbose=='silent') else True)
        fn = getattr(self, self.method)

        # Split train-test set. This set is used for parameter optimization. Note that parameters are shuffled and the train-test set is retained constant.
        # This will make the comparison across parameters and not differences in train-test variances.
        if '_clf' in self.method:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=True, stratify=self.y)
        elif '_reg' in self.method:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=True)

        logger.info('Test-set: %s ' % str(self.X_test.shape))
        logger.info('Train-set: %s ' % str(self.X_train.shape))
        logger.info('*' * 89)
        logger.info('Searching across hyperparameter space for best performing parameters using maximum nr. evaluations: %.0d' % self.max_eval)


        # Hyperoptimization to find best performing model. Set the trials which is the object where all the HPopt results are stored.
        trials=Trials()
        best_params = fmin(fn=fn, space=self.space, algo=self.algo, max_evals=self.max_eval, trials=trials, verbose=False, show_progressbar=False)
        # Summary results
        results_summary, model, best_params = self._to_df(trials, best_params)

        # Cross-validation over the top n models. To speed up we can decide to test only the best performing ones. The best performing model is returned.
        if self.cv is not None:
            model, results_summary, best_params = self._cv(results_summary, self.space, best_params)
            # early_stopping_rounds needs to be set on None
            model.early_stopping_rounds=None

        # Create a basic model by using default parameters.
        space_basic = {}
        space_basic['fit_params'] = {}
        space_basic['model_params'] = {}
        model_basic = getattr(self, self.method)
        model_basic = fn(space_basic)['model']
        comparison_results = {}

        # Validation error
        val_results = None
        if (self.val_size is not None):
            # Evaluate results
            logger.info('*' * 89)
            logger.info('Evaluate best [%s] model on validation dataset (%.0f samples, %.2g%%)' % (self.method, len(self.y_val), self.val_size * 100))
            # With hyperparameter optimization.
            val_score, val_results = self._eval(self.X_val, self.y_val, model)
            # With defaults parameters.
            val_score_basic, val_results_basic = self._eval(self.X_val, self.y_val, model_basic)
            # Store
            comparison_results['Model with optimized hyperparameters (validation set)'] = val_results
            comparison_results['Model with default parameters (validation set)'] = val_results_basic
            logger.info('[%s]: %.4g using optimized hyperparameters on validation set.' % (self.eval_metric, val_score['loss']))
            logger.info('[%s]: %.4g using default (not optimized) parameters on validation set.' % (self.eval_metric, val_score_basic['loss']))
            # Store Validation results
            results_summary = _store_validation_scores(results_summary, best_params, model_basic, val_score_basic, val_score, self.larger_is_better)

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
        ascending = False if self.larger_is_better else True
        results_summary['loss_mean'] = np.nan
        results_summary['loss_std'] = np.nan

        # Determine maximum folds
        top_cv_evals = np.minimum(results_summary.shape[0], self.top_cv_evals)
        idx_top_models = results_summary['loss'].sort_values(ascending=ascending).index[0:top_cv_evals]
        logger.info('*' * 89)
        logger.info('%.0d-fold cross validation for the top %.0d scoring models, Total nr. tests: %.0f' % (self.cv, len(idx_top_models), self.cv * len(idx_top_models)))
        disable = (False if (self.verbose=='silent') else True)

        # For each model, compute the performance.
        for idx in tqdm(idx_top_models, disable=disable):
            scores = []
            # Retrieve the template model using label-based indexing (.loc).
            template_model = results_summary['model'].loc[idx]
            # Do the k-fold cross-validation
            for k in np.arange(0, self.cv):
                # Split train-test set
                if '_clf' in self.method:
                    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=None, shuffle=True, stratify=self.y)
                elif '_reg' in self.method:
                    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=None, shuffle=True)

                # Clone a fresh unfitted copy so stateful attributes (early_stopping bookkeeping, fitted weights) do not bleed between folds.
                try:
                    from sklearn.base import clone as _sklearn_clone
                    fold_model = _sklearn_clone(template_model)
                except Exception:
                    fold_model = copy.deepcopy(template_model)

                # Disable early_stopping_rounds on the clone: the eval_set passed
                # to fit() already handles stopping; leaving it set on the object
                # causes XGBoost >=1.6 to raise when n_estimators and
                # early_stopping_rounds interact unexpectedly across re-fits.
                if hasattr(fold_model, 'early_stopping_rounds'):
                    fold_model.early_stopping_rounds = None

                try:
                    score, _ = self._train_model(fold_model, space)
                    score.pop('model')
                    scores.append(score)
                except Exception as e:
                    logger.error('CV fold %d model idx %d FAILED: %s', k, idx, str(e), exc_info=True)

            if len(scores) > 0:
                # Compute the mean and std across successful folds.
                results_summary.loc[idx, 'loss_mean'] = pd.DataFrame(scores)['loss'].mean()
                results_summary.loc[idx, 'loss_std'] = pd.DataFrame(scores)['loss'].std()

        # Negate scoring if required. The hpopt is optimized for loss functions (lower is better). Therefore we need to set eg the auc to negative and here we need to return.
        if self.larger_is_better:
            results_summary['loss_mean'] = results_summary['loss_mean'] * -1
            valid_mask = results_summary['loss_mean'].notna()
            if not valid_mask.any():
                raise RuntimeError('[hgboost] >All cross-validation folds failed. Check model params and library compatibility.')
            idx_best = results_summary.loc[valid_mask, 'loss_mean'].idxmax()
        else:
            valid_mask = results_summary['loss_mean'].notna()
            if not valid_mask.any():
                raise RuntimeError('[hgboost] >All cross-validation folds failed. Check model params and library compatibility.')
            idx_best = results_summary.loc[valid_mask, 'loss_mean'].idxmin()

        # Get best k-fold CV performing model based on the mean scores.
        logger.info('[%s] (average): %.4g Best %.0d-fold CV model using optimized hyperparameters.' % (self.eval_metric, results_summary['loss_mean'].loc[idx_best], self.cv))
        model = results_summary['model'].loc[idx_best]
        results_summary['best_cv'] = False
        results_summary.loc[idx_best, 'best_cv'] = True
        # Collect best parameters for this model
        best_params = dict(results_summary.loc[idx_best, np.isin(results_summary.columns, [*best_params.keys()])])
        # Return
        return model, results_summary, best_params

    def _train_model(self, model, space):
        # Evaluation is determined for both training and testing set.
        eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        # Build fit kwargs from space, stripping internal sentinel keys.
        fit_kwargs = {k: v for k, v in space['fit_params'].items() if not k.startswith('_')}
        
        # LightGBM early-stopping callbacks are stateful and must be recreated
        # fresh for every fit() call (a reused callback object raises errors on
        # subsequent folds).  We stored the round count under a sentinel key.
        if '_lgb_early_stopping_rounds' in space['fit_params']:
            from lightgbm import early_stopping as _lgb_es, log_evaluation as _lgb_log
            n = space['fit_params']['_lgb_early_stopping_rounds']
            fit_kwargs['callbacks'] = [_lgb_es(stopping_rounds=n, verbose=False), _lgb_log(period=-1)]
        elif 'callbacks' in fit_kwargs:
            # Fallback: ensure callbacks is always a plain list, never a tuple.
            fit_kwargs['callbacks'] = list(fit_kwargs['callbacks'])
            # Force LightGBM to be silent
            fit_kwargs['verbose'] = False
        if 'XGBClassifier' in str(model):
            fit_kwargs['verbose'] = False

        # Make fit with stopping-rule to avoid overfitting.
        model.fit(self.X_train, self.y_train, eval_set=eval_set, **fit_kwargs)
        # Evaluate results
        out, eval_results = self._eval(self.X_test, self.y_test, model)
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
        logger.info('Collecting the hyperparameters from the [%.0d] trials.' % len(trials.trials))

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
                        df_params.loc[i, param] = trial['result']['model'].get_all_params().get(param)
                    else:
                        df_params.loc[i, param] = getattr(trial['result']['model'], param)
                except:
                    logger.debug('Skip [%s]' % param)
                    gather_params_legacy = True

        # The trials.vals stores the index for some parameters instead of the real values.
        # Only fall back column-by-column, never replace the whole df_params.
        if gather_params_legacy:
            legacy_df = pd.DataFrame(trials.vals)
            for col in legacy_df.columns:
                if col in df_params.columns and df_params[col].isna().all():
                    df_params[col] = legacy_df[col].values
        df_scoring = pd.DataFrame(trials.results)
        df = pd.concat([df_params, df_scoring], axis=1)
        df['tid'] = trials.tids

        # Retrieve only the models with OK status
        Iloc = df['status']=='ok'
        if not Iloc.any():
            raise RuntimeError(
                '[hgboost] >All %.0d trials failed (status != "ok"). '
                'Check that your model params and fit_params are compatible with the installed library versions.' % len(df)
            )
        df = df.loc[Iloc, :]

        # Retrieve idx for best model.
        idx = np.where(trials.best_trial['tid']==df['tid'])[0][0]
        # Als retrieve best model based on loss-score.
        if self.larger_is_better:
            df['loss'] = df['loss'] * -1
            idx_best_loss = df['loss'].argmax()
        else:
            idx_best_loss = df['loss'].argmin()

        if idx!=idx_best_loss:
            logger.debug('[Warning] Best model of hyperOpt does not have best loss score(?)')

        # model = df['model'].iloc[idx_best_loss]
        # score = df['loss'].iloc[idx_best_loss]
        model = df.loc[idx_best_loss, 'model']
        score = df.loc[idx_best_loss, 'loss']
        df['best'] = False
        # df.iloc[idx_best_loss, df.columns.get_loc('best')] = True
        df.loc[idx_best_loss, 'best'] = True

        # Get best_params
        try:
            # best_params = df.loc[idx_best_loss, best_params].to_dict()
            best_params = df.loc[idx_best_loss].to_dict()
            # Should be the same as:
            # trials.best_trial['result']['model']
        except:
            pass

        # Return
        logger.info('[%s]: %.4g Best performing model across %.0d iterations using Bayesian Optimization with Hyperopt.' % (self.eval_metric, score, df.shape[0]))
        return df, model, best_params

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
            logger.info('Warning: No model found. Hint: fit a model first using xgboost, catboost or lightboost <return>')
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

    def _eval(self, X_test, y_test, model):
        """Model Evaluation.

        Description
        -----------
        Note that the loss function is by default maximized towards small/negative values by the hptop method.
        When you want to optimize auc or f1, you simply need to negate the score.
        The negation is fixed with the parameter: larger_is_better=False
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
                if self.larger_is_better: loss = loss * -1
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
                if self.larger_is_better: loss = loss * -1
                # Store
                out = {'loss': loss, 'eval_time': time.time(), 'status': STATUS_OK, 'model': model}
                # out = {'loss': loss, 'eval_time': time.time(), 'auc': results['auc'], 'kappa': results['kappa'], 'f1': results['f1'], 'status': STATUS_OK, 'model': model}
        elif '_reg' in self.method:
            # Regression
            # loss = space['loss_func'](y_test, y_pred)
            if self.eval_metric=='mse':
                loss = np.sqrt(mean_squared_error(y_test, y_pred))
            elif self.eval_metric=='rmse':
                loss = mean_squared_error(y_test, y_pred)
            elif self.eval_metric=='mae':
                loss = mean_absolute_error(y_test, y_pred)
            else:
                raise ValueError('[hgboost] >Error: [%s] is not a valid [eval_metric] for [%s].' %(self.eval_metric, self.method))

            # Negative loss score if required
            if self.larger_is_better: loss = loss * -1
            # Store results
            out = {'loss': loss, 'eval_time': time.time(), 'status': STATUS_OK, 'model': model}
        else:
            raise ValueError('[hgboost] >Error: Method %s does not exists.' %(self.method))

        logger.debug('[%s] - [%s] - loss: %s' % (self.method, self.eval_metric, loss))
        return out, results

    def preprocessing(self, df, y_min=2, perc_min_num=0.8, excl_background='0.0', hot_only=False):
        """Pre-processing of the input data.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        y_min : int [0..len(y)], optional
            Minimal number of samples that must be present in a group. All groups with less then y_min samples are labeled as _other_ and are not used in the enriching model. The default is None.
        perc_min_num : float [None, 0..1], optional
            Force column (int or float) to be numerical if unique non-zero values are above percentage. The default is None. Alternative can be 0.8

        Returns
        -------
        data : pd.Datarame
            Processed data.

        """
        X = df2onehot(df, y_min=y_min, hot_only=hot_only, perc_min_num=perc_min_num, excl_background=excl_background, verbose=self.verbose)
        return X['onehot']

    def import_example(self, data='titanic', url=None, sep=','):
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

        Returns
        -------
        pd.DataFrame()
            Dataset containing mixed features.

        References
        ----------
            * https://github.com/erdogant/datazets

        """
        return dz.get(data=data, url=url, sep=sep)

    def treeplot(self, num_trees=None, plottype='horizontal', figsize=(20, 25), return_ax=False):
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

        Returns
        -------
        ax : object

        """
        if not hasattr(self, 'method') or (not hasattr(self, 'model')):
            logger.warning('No model found. Hint: fit a model first using xgboost, catboost or lightboost <return>')
            return None
        if ('ensemble' in self.method):
            logger.warning('Warning: No plot for ensemble is possible yet. <return>')
            return None
        verbose = 0 if self.verbose == 'silent' else 3

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
        if not hasattr(self, 'method') or ('ensemble' in self.method):
            logger.warning('Warning: No plot for ensemble is possible yet. <return>')
            return None

        logger.info('%.0d-fold crossvalidation is performed with [%s]' % (self.cv, self.method))
        disable = (False if (self.verbose=='silent') else True)

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
                _, cl_results = self._eval(X_test, y_test, self.model)
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
                sns.regplot(x='y', y='y_pred', data=cv_results.get(key), ax=ax, color=colors[i, :], label=key)
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
        if not hasattr(self, 'method') or (not hasattr(self, 'model')):
            logger.warning('No model found. Hint: fit a model first using xgboost, catboost or lightboost <return>')
            return None
        if self.val_size is None:
            logger.warning('No validation set found. Hint: use the parameter [val_size=0.2] first <return>')
            return None

        title = 'Results on independent validation set'
        if ('_clf' in self.method) and not ('_multi' in self.method):
            if (self.results.get('val_results', None)) is not None:
                logger.info('Results are plotted from key: "results[\'val_results\']"')
                if normalized is not None: self.results['val_results']['confmat']['normalized']=normalized
                ax = cle.plot(self.results['val_results'], title=title)
                if return_ax: return ax
        elif ('_reg' in self.method):
            # fig, ax = plt.subplots(figsize=figsize)
            y_pred = self.predict(self.X_val, model=self.model)[0]
            df = pd.DataFrame(np.c_[self.y_val, y_pred], columns=['y', 'y_pred'])
            eval_score = None

            if self.eval_metric=='rmse':
                eval_score = np.sqrt(mean_squared_error(self.y_val, y_pred))
            if self.eval_metric=='mse':
                eval_score = mean_squared_error(self.y_val, y_pred)

            fig, ax = plt.subplots(figsize=figsize)
            sns.regplot(x='y', y='y_pred', data=df, ax=ax, color='k', label='Validation set')
            ax.legend()
            ax.grid(True)
            if eval_score is not None:
                ax.set_title(f'{title}\nRMSE: {eval_score:.4f}')
            else:
                ax.set_title(f'{title}')
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
        if not hasattr(self, 'method') or ('ensemble' in self.method):
            logger.warning( 'Warning: No plot for ensemble is possible yet. <return>')
            return None, None

        top_n = np.minimum(top_n, self.results['summary'].shape[0])
        # getcolors = colourmap.generate(top_n, cmap='Reds_r')
        ascending = False if self.larger_is_better else True
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
        # Ensure ax is always 2D
        if nrRows == 1:
            ax = np.array([ax])
        i_row = -1

        i=0
        for param in params:
            try:
                # Get row number
                i_col = np.mod(i, nrCols)
                # Make new column
                if i_col == 0: i_row = i_row + 1
                logger.debug('Plot row: %.0d, col: %.0d' % (i_row, i_col))

                col_data = pd.to_numeric(summary_results[param], errors='coerce').dropna()
                if col_data.nunique() < 2:
                    i = i + 1
                    continue

                # KDE plot (replaces deprecated sns.distplot)
                sns.kdeplot(col_data, fill=shade, linewidth=1,
                            color=color_params[i, :],
                            ax=ax[i_row][i_col])
                # Rug plot
                sns.rugplot(col_data, color='black', ax=ax[i_row][i_col])

                # Get y range from the plotted kde line
                lines = ax[i_row][i_col].get_lines()
                if lines:
                    y_data = lines[0].get_ydata()
                    ymin, ymax = np.min(y_data), np.max(y_data)
                else:
                    ymin, ymax = 0, 1

                getvals = df_summary[param].values
                # Plot the top n (not the first because that one is plotted in green)
                ax[i_row][i_col].vlines(getvals[1:top_n], ymin, ymax, linewidth=1, colors='k', linestyles='dashed', label='Top ' + str(top_n) + ' models')
                # Plot the best (without cv)
                ax[i_row][i_col].vlines(getvals[idx_best], ymin, ymax, linewidth=2, colors='g', linestyles='dashed', label='Best (without cv)')
                # Plot the best (with cv)
                if self.cv is not None:
                    ax[i_row][i_col].vlines(getvals[idx_best_cv], ymin, ymax, linewidth=2, colors='r', linestyles='dotted', label='Best ' + str(self.cv) + '-fold cv')

                if self.cv is not None:
                    ax[i_row][i_col].set_title(('%s: %.3g (%.0d-fold cv)' %(param, getvals[idx_best_cv[0]], self.cv)))
                else:
                    ax[i_row][i_col].set_title(('%s: %.3g' %(param, getvals[idx_best[0]])))
                ax[i_row][i_col].set_ylabel('Density')
                ax[i_row][i_col].grid(True)
                ax[i_row][i_col].legend(loc='upper right')
                i = i + 1
            except Exception as e:
                logger.warning( 'Warning: Could not plot param [%s]: %s' % (param, str(e)))
                i = i + 1

        # Hide unused subplots in density figure
        for empty in range(i, nrRows * nrCols):
            ax[empty // nrCols][empty % nrCols].set_visible(False)
        fig.tight_layout()

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
        # Ensure ax2 is always 2D
        if nrRows == 1:
            ax2 = np.array([ax2])
        i_row = -1
        i=0
        for param in params:
            try:
                # Get row number
                i_col = np.mod(i, nrCols)
                # Make new column
                if i_col == 0: i_row = i_row + 1

                col_data = pd.to_numeric(df_sum[param], errors='coerce')
                if col_data.nunique() < 2:
                    i = i + 1
                    continue

                # Make the plot
                sns.regplot(x='tid', y=param, data=df_sum, ax=ax2[i_row][i_col], color=color_params[i, :])

                # Scatter top n values, start with 1 because the 0 is, based on the ranking, with CV.
                ax2[i_row][i_col].scatter(df_summary['tid'].values[1:top_n], df_summary[param].values[1:top_n], s=50, color='k', marker='.', label='Top ' + str(top_n) + ' models')

                # Scatter best value
                ax2[i_row][i_col].scatter(df_sum['tid'].values[idx_best_without_cv], df_sum[param].values[idx_best_without_cv], s=100, color='g', marker='*', label='Best (without cv)')

                # Scatter best cv
                if self.cv is not None:
                    ax2[i_row][i_col].scatter(df_sum['tid'].values[idx_best_cv], df_sum[param].values[idx_best_cv], s=100, color='r', marker='x', label='Best ' + str(self.cv) + '-fold cv')

                # Set labels
                if self.cv is not None:
                    ax2[i_row][i_col].set_title(('%s: %.3g (%.0d-fold cv)' %(param, df_sum[param].values[idx_best_cv[0]], self.cv)))
                else:
                    ax2[i_row][i_col].set_title(('%s: %.3g' %(param, df_sum[param].values[idx_best_without_cv[0]])))
                ax2[i_row][i_col].set_xlabel('iteration')
                ax2[i_row][i_col].set_ylabel(param)
                ax2[i_row][i_col].grid(True)
                ax2[i_row][i_col].legend(loc='upper right')
                i = i + 1
            except Exception as e:
                logger.warning('Warning: Could not plot scatter param [%s]: %s' % (param, str(e)))
                i = i + 1

        # Hide unused subplots in scatter figure
        for empty in range(i, nrRows * nrCols):
            ax2[empty // nrCols][empty % nrCols].set_visible(False)
        fig2.tight_layout()

        if return_ax:
            return ax, ax2

    def plot(self, ylim=None, figsize=(20, 15), plot2=True, return_ax=False):
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
        if not hasattr(self, 'method') or (not hasattr(self, 'model')):
            logger.warning('No model found. Hint: fit a model first using xgboost, catboost or lightboost <return>')
            return ax1, ax2
        if ('ensemble' in self.method):
            logger.warning('Warning: No plot for ensemble is possible yet. <return>')
            self.plot_ensemble(ylim, figsize, ax1, ax2)
            return ax1, ax2

        # Plot comparison between hyperoptimized vs basic model
        if (self.results.get('comparison_results', None) is not None) and ([*self.results['comparison_results'].values()][0] is not None):
            cle.plot_cross(self.results['comparison_results'], title='Comparison between model with optimized hyperparamters vs. default parameters on validation set.')

        if hasattr(self.model, 'evals_result') and plot2:
            _, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        else:
            _, ax1 = plt.subplots(1, 1, figsize=figsize)

        tmpdf = self.results['summary'].sort_values(by='tid', ascending=True).copy()
        tmpdf = tmpdf.loc[~tmpdf['loss'].isna(), :]
        tmpdf.reset_index(drop=True, inplace=True)

        # Make the plot
        sns.regplot(x='tid', y='loss', data=tmpdf, ax=ax1, color='#23395d', scatter=True, fit_reg=True, label=str(tmpdf.shape[0])+' models iterations')
        # Plot all other evalution results
        # ax1.scatter(tmpdf['tid'].values, tmpdf['loss'].values, s=10, label=str(tmpdf.shape[0])+'models iterations')

        # Plot results with testsize
        idx = np.where(tmpdf['best'].values)[0]
        best_loss = float(tmpdf['loss'].iloc[idx[0]])
        ax1.hlines(best_loss, 0, tmpdf['loss'].shape[0], colors='g', linestyles='dashed', label='Best model (without cv)')
        loss_mean_at_idx = tmpdf['loss_mean'].iloc[idx[0]] if not pd.isna(tmpdf['loss_mean'].iloc[idx[0]]) else best_loss
        ax1.vlines(idx[0], tmpdf['loss'].min(), loss_mean_at_idx, colors='g', linestyles='dashed')

        title = ('%s (%s: %.3g)' %(self.method, self.eval_metric, best_loss))

        # Plot results with cv
        if self.cv is not None:
            # Plot the top CVs for the top k models.
            ax1.errorbar(tmpdf['tid'], tmpdf['loss_mean'], tmpdf['loss_std'], marker='s', mfc='red', label=str(self.cv) + '-fold cv for top ' + str(self.top_cv_evals) + ' scoring models')
            topk = np.where(~tmpdf['loss_mean'].isna())[0]

            # Mark the best CV in Red
            idx_cv = np.where(tmpdf['best_cv'].values)[0]
            best_loss_cv = float(tmpdf['loss_mean'].iloc[idx_cv[0]])
            ax1.hlines(best_loss_cv, 0, tmpdf['loss_mean'].shape[0], colors='r', linestyles='dotted', label='Best model (' + str(self.cv) + '-fold cv)')
            ax1.vlines(idx_cv[0], tmpdf['loss'].min(), best_loss_cv, colors='r', linestyles='dashed')

            # Highlight the top k best scoring models.
            ax1.scatter(tmpdf['tid'].values[topk], tmpdf['loss'].values[topk], s=30, marker='s', label='Top ' + str(self.top_cv_evals) + ' scoring models', color='#960000')

            # Set title
            title = ('%s (%.0d-fold cv mean %s: %.3g)' %(self.method, self.cv, self.eval_metric, best_loss_cv))
            ax1.set_xlabel('Model iterations')

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

        Examples
        --------
        >>> # Initialize libraries
        >>> from hgboost import hgboost
        >>> import pandas as pd
        >>> from sklearn import datasets
        >>>
        >>> # Load example dataset
        >>> iris = datasets.load_iris()
        >>> X = pd.DataFrame(iris.data, columns=iris['feature_names'])
        >>> y = iris.target
        >>>
        >>> # Train model using user-defined parameters
        >>> hgb = hgboost(max_eval=10, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=42)
        >>> results = hgb.xgboost(X, y, method="xgb_clf_multi")
        >>>
        >>> # Save
        >>> hgb.save(filepath='hgboost_model.pkl', overwrite=True)
        >>>

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
        storedata['larger_is_better'] = self.larger_is_better
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
        storedata['is_unbalanced'] = self.is_unbalanced
        # Save
        status = pypickle.save(filepath, storedata, overwrite=overwrite, verbose=verbose)
        if status:
            logger.info('Save model results.')
            logger.info('Save user-defined parameters.')
            logger.info('Save trained model.')
            logger.info('Save successful!')
        else:
            logger.warning('Could not save. Tip: "hgb.save(overwrite=True)"')
        # return
        return status

    def load(self, filepath='hgboost_model.pkl', verbose=3):
        """Load learned model.

        Description
        -----------
        The load function will restore the trained model and results.
        In a fresh (new) start, you need to re-initialize the hgboost model first.
        By loading the model, the user defined parameters are also restored.

        Parameters
        ----------
        filepath : str
            Pathname to stored pickle files.
        verbose : int, optional
            Show message. A higher number gives more information. The default is 3.

        Examples
        --------
        >>> # Initialize libraries
        >>> from hgboost import hgboost
        >>> import pandas as pd
        >>> from sklearn import datasets
        >>>
        >>> # Load example dataset
        >>> iris = datasets.load_iris()
        >>> X = pd.DataFrame(iris.data, columns=iris['feature_names'])
        >>> y = iris.target
        >>>
        >>> # Train model using user-defined parameters
        >>> hgb = hgboost(max_eval=10, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=42)
        >>> results = hgb.xgboost(X, y, method="xgb_clf_multi")
        >>>
        >>> # Save
        >>> hgb.save(filepath='hgboost_model.pkl', overwrite=True)
        >>>
        >>> # Load
        >>> from hgboost import hgboost
        >>> hgb = hgboost()
        >>> results = hgb.load(filepath='hgboost_model.pkl')
        >>>
        >>> # Make predictions again with:
        >>> y_pred, y_proba = hgb.predict(X)

        Returns
        -------
        * dictionary containing model results.
        * Object with trained model.

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
            self.larger_is_better = storedata['larger_is_better']
            self.model = storedata['model']
            self.space = storedata['space']
            self.algo = storedata['algo']
            self.max_eval = storedata['max_eval']
            self.top_cv_evals = storedata['top_cv_evals']
            self.threshold = storedata['threshold']
            self.test_size = storedata['test_size']
            self.val_size = storedata['val_size']
            self.cv = storedata['cv']
            self.is_unbalanced = storedata['is_unbalanced']
            self.random_state = storedata['random_state']
            self.n_jobs = storedata['n_jobs']
            self.verbose = storedata['verbose']
            logger.info('Restore model results.')
            logger.info('Restore user-defined parameters.')
            logger.info('Restore trained model.')
            logger.info('Loading successful!')
            # Return results
            return self.results
        else:
            logger.warning('Could not load data.')


# %%
def _store_validation_scores(results_summary, best_params, model_basic, val_score_basic, val_score, larger_is_better):
    # Store default parameters
    params = [*best_params.keys()]
    idx = results_summary.index.max() + 1
    results_summary.loc[idx] = np.nan
    # Store the params of the basic model
    for param in params:
        results_summary.loc[idx, param] = model_basic.get_params().get(param, None)

    results_summary['loss_validation'] = np.nan
    results_summary.loc[idx, 'loss_validation'] = val_score_basic['loss'] * (-1 if larger_is_better else 1)
    results_summary['default_params'] = False
    results_summary.loc[idx, 'default_params'] = True

    # Store the best loss validation score for the hyperoptimized model
    if np.any(results_summary.columns.str.contains('best_cv')):
        idx_best = results_summary.index[results_summary['best_cv']==1]
    else:
        idx_best = results_summary.index[results_summary['best']==1]
    results_summary.loc[idx_best, 'loss_validation'] = val_score['loss'] * (-1 if larger_is_better else 1)

    return results_summary


# %% Import example dataset from github.
# def import_example(data='titanic', url=None, sep=',', verbose=3):
    # """Import example dataset from github source.

    # Description
    # -----------
    # Import one of the few datasets from github source or specify your own download url link.

    # Parameters
    # ----------
    # data : str, (default : "titanic")
    #     Name of datasets: 'sprinkler', 'titanic', 'student', 'fifa', 'cancer', 'waterpump', 'retail'
    # url : str
    #     url link to to dataset.
    # verbose : int, (default : 3)
    #     Print progress to screen.
    #     0: None, 1: ERROR, 2: WARN, 3: INFO, 4: DEBUG, 5: TRACE

    # Returns
    # -------
    # pd.DataFrame()
    #     Dataset containing mixed features.

    # """
    # if url is None:
    #     if data=='sprinkler':
    #         url='https://erdogant.github.io/datasets/sprinkler.zip'
    #     elif data=='titanic':
    #         url='https://erdogant.github.io/datasets/titanic_train.zip'
    #     elif data=='student':
    #         url='https://erdogant.github.io/datasets/student_train.zip'
    #     elif data=='cancer':
    #         url='https://erdogant.github.io/datasets/cancer_dataset.zip'
    #     elif data=='fifa':
    #         url='https://erdogant.github.io/datasets/FIFA_2018.zip'
    #     elif data=='waterpump':
    #         url='https://erdogant.github.io/datasets/waterpump/waterpump_test.zip'
    #     elif data=='retail':
    #         url='https://erdogant.github.io/datasets/marketing_data_online_retail_small.zip'
    # else:
    #     data = wget.filename_from_url(url)

    # if url is None:
    #     print('[hgboost] >Nothing to download.')
    #     return None

    # curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    # PATH_TO_DATA = os.path.join(curpath, wget.filename_from_url(url))
    # if not os.path.isdir(curpath):
    #     os.makedirs(curpath, exist_ok=True)

    # # Check file exists.
    # if not os.path.isfile(PATH_TO_DATA):
    #     print('[hgboost] >Downloading [%s] dataset from github source..' %(data))
    #     wget.download(url, curpath)

    # # Import local dataset
    # print('[hgboost] >Import dataset [%s]' %(data))
    # df = pd.read_csv(PATH_TO_DATA, sep=sep)
    # # Return
    # return df


# %% Set the search spaces
def _get_params(fn_name, eval_metric=None, y=None, pos_label=None, is_unbalanced=False, gpu=False, early_stopping_rounds=25, verbose=3):
    # choice : categorical variables
    # quniform : discrete uniform (integers spaced evenly)
    # uniform: continuous uniform (floats spaced evenly)
    # loguniform: continuous log uniform (floats spaced evenly on a log scale)
    if eval_metric is None: raise ValueError('[hgboost] >eval_metric must be provided.')
    logger.info('Collecting %s parameters.', fn_name)

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
            # 'gpu_id': 0,
            'predictor': predictor,
            'early_stopping_rounds': early_stopping_rounds,
        }
        space = {}
        space['model_params'] = xgb_reg_params
        space['fit_params'] = {'verbose': 0}
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
        space['fit_params'] = {
            'eval_metric': 'l2',
            '_lgb_early_stopping_rounds': early_stopping_rounds,  # rebuilt fresh each fit
        }

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
            'early_stopping_rounds': early_stopping_rounds,
        }
        space = {}
        space['model_params'] = ctb_reg_params
        space['fit_params'] = {'verbose': 0}
        return(space)

    ############### CatBoost classification parameters ###############
    if fn_name=='ctb_clf':
        # Enable/Disable GPU
        task_type = 'CPU' if gpu else 'CPU'

        # Class sizes
        if is_unbalanced:
            # https://catboost.ai/docs/concepts/python-reference_parameters-list.html#python-reference_parameters-list
            logger.info('Correct for unbalanced classes using [auto_class_weights].')
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
        if is_unbalanced:
            logger.info('Correct for unbalanced classes using [is_unbalanced]..')
            is_unbalanced = [True]
        else:
            is_unbalanced = [True, False]

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
            'is_unbalanced': hp.choice('is_unbalanced', is_unbalanced),
            'device': device,
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
        }
        space={}
        space['model_params'] = lgb_clf_params
        space['fit_params'] = {
            '_lgb_early_stopping_rounds': early_stopping_rounds,  # rebuilt fresh each fit
        }
        return space

    # ############## XGboost classification parameters ###############
    if 'xgb_clf' in fn_name:
        # Enable/Disable GPU
        if gpu:
            # 'gpu_hist' Equivalent to the XGBoost fast histogram algorithm. Much faster and uses considerably less memory. NOTE: May run very slowly on GPUs older than Pascal architecture.
            tree_method = 'auto'  # 'gpu_hist' throws random errors
            predictor = 'gpu_predictor'
        else:
            tree_method = 'hist'
            predictor = 'cpu_predictor'

        xgb_clf_params = {
            'learning_rate': hp.choice('learning_rate', np.logspace(np.log10(0.005), np.log10(0.5), base=10, num=1000)),
            'max_depth': hp.choice('max_depth', range(5, 32, 1)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            'gamma': hp.choice('gamma', [0.5, 1, 1.5, 2, 3, 4, 5]),
            'subsample': hp.quniform('subsample', 0.1, 1, 0.01),
            'n_estimators': hp.choice('n_estimators', range(20, 205, 5)),
            'booster': 'gbtree',
            'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1.0, 0.01),
            'tree_method': tree_method,
            # 'gpu_id': 0,
            'predictor': predictor,
            'early_stopping_rounds': early_stopping_rounds,
        }

        if fn_name=='xgb_clf':
            # xgb_clf_params['eval_metric']=hp.choice('eval_metric', ['error', eval_metric])
            xgb_clf_params['objective']='binary:logistic'
            # Class sizes
            if is_unbalanced:
                logger.info('Correct for unbalanced classes using [scale_pos_weight]..')
                scale_pos_weight = np.sum(y!=pos_label) / np.sum(y==pos_label)
            else:
                scale_pos_weight = hp.choice('scale_pos_weight', np.arange(1, 101, 9))
            xgb_clf_params['scale_pos_weight']=scale_pos_weight

        if fn_name=='xgb_clf_multi':
            xgb_clf_params['objective']='multi:softprob'
            # scoring='kappa'

        space = {}
        space['model_params']=xgb_clf_params
        space['fit_params']={'verbose': 0}

        logger.info('[%.0d] hyperparameters in gridsearch space. Used loss function: [%s].', len([*space['model_params']]), eval_metric)
        return space


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
        logger.debug('pos_label is used to set [%s].', pos_label)
        y=y==pos_label
        pos_label=True

    # Checks pos_label status in case of method is classification
    if ('_clf' in method) and (pos_label is None) and (str(y.dtype)=='bool'):
        pos_label=True
        logger.debug('[pos_label] is set to [%s] because [y] is of type [bool].', pos_label)

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
        logger.debug('[pos_label] is set to [None] because [method] is of type [%s].', method)

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
    logger.debug('Reset index for X.')

    # Return
    return X, y, pos_label


def _check_eval_metric(method, eval_metric, larger_is_better, verbose=3):
    # Check the eval_metric
    if (eval_metric is None) and ('_reg' in method):
        eval_metric='rmse'
    elif (eval_metric is None) and ('_clf_multi' in method):
        eval_metric='kappa'
    elif (eval_metric is None) and ('_clf' in method):
        eval_metric='auc'

    # Check the larger_is_better for evaluation metric
    if larger_is_better is None:
        if (eval_metric == 'f1'):
            larger_is_better=True
        elif 'auc' in eval_metric:
            larger_is_better=True
        elif (eval_metric == 'kappa'):
            larger_is_better=True
        elif (eval_metric == 'rmse'):
            larger_is_better=False
        elif (eval_metric == 'mse'):
            larger_is_better=False
        elif (eval_metric == 'mae'):
            larger_is_better=False
        else:
            logger.warning('[%s] is not an implemented option. [larger_is_better] is set to %s', eval_metric, str(larger_is_better))

    # Return
    return eval_metric, larger_is_better