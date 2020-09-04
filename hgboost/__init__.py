from hgboost.hgboost import hgboost

from hgboost.hgboost import (
    import_example,
    )

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.1.1'

# module level doc-string
__doc__ = """
hgboost - Hyperoptimized Gradient Boosting
=====================================================================

Description
-----------
hgboost is a python package for hyperparameter optimization for xgboost, 
catboost and lightboost using cross-validation, and evaluating the results
on an independent validation set. hgboost can be applied for classification and regression tasks.
The hgboost approach contains the following parts:
    * Regression
    * Binary classification
    * multiclass classification
    * K-fold cross-validation
    * Hyperparameter searching
    * Independent validation dataset
    * Feature importance
    * Early stopping
    * Plotting
        - parameter space
        - validation set
        - summary
        - tree

Example
-------
>>> from hgboost import hgboost

>>> ######## CLASSIFICATION ########

>>> # Initialize
>>> hgb = hgboost(max_eval=500, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, verbose=3)

>>> # Import data
>>> df = hgb.import_example()
>>> y = df['Survived'].values
>>> del df['Survived']
>>> X = hgb.preprocessing(df, verbose=0)
>>> y = y.astype(str)
>>> y[y=='1']='survived'
>>> y[y=='0']='dead'

>>> # Fit a classification model
>>> results = hgb.xgboost(X, y, pos_label='survived')
>>> results = hgb.catboost(X, y, pos_label='survived')
>>> results = hgb.lightboost(X, y, pos_label='survived')

>>> # Make some plots
>>> hgb.plot_params()
>>> hgb.plot()
>>> hgb.treeplot()
>>> hgb.plot_validation()
>>> hgb.plot_cv()

# use the predictor on new data
>>> y_pred, y_proba = hgb.predict(X)

>>> ######## REGRESSION ########

>>> # Initialize
>>> hgb = hgboost(max_eval=500, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, verbose=3)

>>> # Import data
>>> df = hgb.import_example()
>>> y = df['Age'].values
>>> del df['Age']
>>> I = ~np.isnan(y)
>>> X = hgb.preprocessing(df, verbose=0)
>>> X = X.loc[I,:]
>>> y = y[I]

>>> # Fit
>>> results = hgb.xgboost_reg(X, y, eval_metric='mae')
>>> results = hgb.catboost_reg(X, y, eval_metric='mae')
>>> results = hgb.lightboost_reg(X, y, eval_metric='mae')

>>> # Make some plots
>>> hgb.plot_params()
>>> hgb.plot()
>>> hgb.treeplot()
>>> hgb.plot_validation()
>>> hgb.plot_cv()

>>> # use the predictor
>>> y_pred, y_proba = hgb.predict(X)

References
----------
* https://github.com/erdogant/hgboost

"""
