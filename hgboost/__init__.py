import logging
from datazets import get as import_example
from hgboost.hgboost import hgboost

# Setup package-level logger
_logger = logging.getLogger('hgboost')
_log_handler = logging.StreamHandler()
_formatter = logging.Formatter(fmt='[{asctime}] [{name:<12.12}] [{levelname:<8}] {message}', style='{', datefmt='%d-%m-%Y %H:%M:%S')
_log_handler.setFormatter(_formatter)
_log_handler.setLevel(logging.DEBUG)
if not _logger.hasHandlers():  # avoid duplicate handlers if re-imported
    _logger.addHandler(_log_handler)
_logger.setLevel(logging.DEBUG)
_logger.propagate = True  # allow submodules to inherit this handler

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '1.2.0'

# module level doc-string
__doc__ = """
hgboost - Hyperoptimized Gradient Boosting
=====================================================================

HGBoost stands for Hyperoptimized Gradient Boosting and is a Python package for hyperparameter optimization
for XGBoost, LightBoost, and CatBoost. It will carefully split the dataset into a train, test, and independent
validation set. Within the train-test set, there is the inner loop for optimizing the hyperparameters using
Bayesian optimization (with hyperopt) and, the outer loop to score how well the top performing models can
generalize based on k-fold cross validation. As such, it will make the best attempt to select the most robust
model with the best performance.
The hgboost approach contains the following parts:
    * Regression
    * Binary classification
    * multiclass classification
    * K-fold cross-validation
    * Hyperparameter searching
    * Independent validation dataset
    * Feature importance
    * Early stopping
    * Plotting: parameter space, validation set, summary, tree

Example
-------
>>> from hgboost import hgboost
>>>
>>> ######## CLASSIFICATION ########
>>>
>>> # Initialize
>>> hgb = hgboost(max_eval=100, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, verbose='info')
>>>
>>> # Import data
>>> df = hgb.import_example()
>>> y = df['Survived'].values
>>> del df['Survived']
>>> X = hgb.preprocessing(df)
>>>
>>> # Fit a classification model
>>> results = hgb.xgboost(X, y, pos_label=1)
>>> results = hgb.catboost(X, y, pos_label=1)
>>> results = hgb.lightboost(X, y, pos_label=1)
>>>
>>> # Make explainable plots
>>> hgb.plot_params()
>>> hgb.plot()
>>> hgb.treeplot()
>>> hgb.plot_validation()
>>> hgb.plot_cv()
>>>
# use the predictor on new data
>>> y_pred, y_proba = hgb.predict(X)
>>>
>>> ######## REGRESSION ########
>>>
>>> # Initialize
>>> hgb = hgboost(max_eval=100, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, verbose='info')
>>>
>>> # Import data
>>> df = hgb.import_example()
>>> y = df['Age'].values
>>> del df['Age']
>>> I = ~np.isnan(y)
>>> X = hgb.preprocessing(df)
>>> X = X.loc[I,:]
>>> y = y[I]
>>>
>>> # Fit
>>> results = hgb.xgboost_reg(X, y, eval_metric='mae')
>>> results = hgb.catboost_reg(X, y, eval_metric='mae')
>>> results = hgb.lightboost_reg(X, y, eval_metric='mae')
>>>
>>> # Make some plots
>>> hgb.plot_params()
>>> hgb.plot()
>>> hgb.treeplot()
>>> hgb.plot_validation()
>>> hgb.plot_cv()
>>>
>>> # use the predictor
>>> y_pred, y_proba = hgb.predict(X)
>>>

References
----------
* Medium: https://erdogant.medium.com
* Github: https://github.com/erdogant/hgboost
* Documentation pages: https://erdogant.github.io/hgboost/
* Notebook Classification: https://colab.research.google.com/github/erdogant/hgboost/blob/master/notebooks/hgboost_classification_examples.ipynb
* Notebook Regression: https://colab.research.google.com/github/erdogant/hgboost/blob/master/notebooks/hgboost_regression_examples.ipynb

"""
