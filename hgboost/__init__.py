from datazets import get as import_example
from hgboost.hgboost import hgboost

# from hgboost.hgboost import (
#     import_example,
#     )

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '1.1.5'

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
>>> hgb = hgboost(max_eval=500, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, verbose=3)
>>>
>>> # Import data
>>> df = hgb.import_example()
>>> y = df['Survived'].values
>>> del df['Survived']
>>> X = hgb.preprocessing(df, verbose=0)
>>>
>>> # Fit a classification model
>>> results = hgb.xgboost(X, y, pos_label=1)
>>> results = hgb.catboost(X, y, pos_label=1)
>>> results = hgb.lightboost(X, y, pos_label=1)
>>>
>>> # Make some plots
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
>>> hgb = hgboost(max_eval=500, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, verbose=3)
>>>
>>> # Import data
>>> df = hgb.import_example()
>>> y = df['Age'].values
>>> del df['Age']
>>> I = ~np.isnan(y)
>>> X = hgb.preprocessing(df, verbose=0)
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
* Blog: https://towardsdatascience.com/a-guide-to-find-the-best-boosting-model-using-bayesian-hyperparameter-tuning-but-without-c98b6a1ecac8
* Blog: Classifiction: https://erdogant.medium.com/hands-on-guide-for-hyperparameter-tuning-with-bayesian-optimization-for-classification-models-2002224bfa3d
* Github: https://github.com/erdogant/hgboost
* Documentation pages: https://erdogant.github.io/hgboost/
* Notebook Classification: https://colab.research.google.com/github/erdogant/hgboost/blob/master/notebooks/hgboost_classification_examples.ipynb
* Notebook Regression: https://colab.research.google.com/github/erdogant/hgboost/blob/master/notebooks/hgboost_regression_examples.ipynb

"""
