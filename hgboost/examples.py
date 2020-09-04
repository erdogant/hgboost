# https://scikit-learn.org/stable/modules/cross_validation.html
# https://towardsdatascience.com/fine-tuning-xgboost-in-python-like-a-boss-b4543ed8b1e
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py
# https://github.com/WillKoehrsen/hyperparameter-optimization/blob/master/Bayesian%20Hyperparameter%20Optimization%20of%20Gradient%20Boosting%20Machine.ipynb
# https://www.kaggle.com/henrylidgley/xgboost-with-hyperopt-tuning
# https://towardsdatascience.com/tree-boosted-mixed-effects-models-4df610b624cb
# https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/
# http://hyperopt.github.io/hyperopt/getting-started/minimizing_functions/

# Objective is to demonstrate:

# regression
# binary classification
# multiclass classification
# cross-validation
# hyperparameter searching
# feature importance
# early stopping
# plotting

# %%
from hgboost import hgboost
print(dir(hgboost))
# print(hgboost.__version__)
import numpy as np

# %% HYPEROPTIMIZED XGBOOST
hgb_xgb = hgboost(max_eval=100, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None, verbose=3)
hgb_cat = hgboost(max_eval=100, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None, verbose=3)
hgb_light = hgboost(max_eval=100, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None, verbose=3)

# Import data
df = hgb_xgb.import_example()
y = df['Survived'].values
del df['Survived']
X = hgb_xgb.preprocessing(df, verbose=0)

# Fit
results = hgb_xgb.xgboost(X, y, pos_label=1)
results = hgb_cat.catboost(X, y, pos_label=1)
results = hgb_light.lightboost(X, y, pos_label=1)

# Make some plots
hgb_xgb.plot_params()
hgb_xgb.plot()
hgb_xgb.treeplot()
hgb_xgb.plot_validation()
hgb_xgb.plot_cv()

# use the predictor
y_pred, y_proba = hgb_xgb.predict(X)
y_pred, y_proba = hgb_cat.predict(X)
y_pred, y_proba = hgb_light.predict(X)


# %% HYPEROPTIMIZED MULTI-XGBOOST
hgb = hgboost(max_eval=10, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=42)

# Import data
df = hgb.import_example()
y = df['Parch'].values
y[y>=3]=3
del df['Parch']
X = hgb.preprocessing(df, verbose=0)

# FIT MULTI-CLASS CLASSIFIER
results = hgb.xgboost(X, y, method='xgb_clf_multi')

# Make some plots
hgb.plot_params()
hgb.plot()
hgb.treeplot()
hgb.plot_validation()
hgb.plot_cv()

# use the predictor
y_pred, y_proba = hgb.predict(X)


# %% HYPEROPTIMIZED REGRESSION-XGBOOST
hgb = hgboost(max_eval=10, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=42)

# Import data
df = hgb.import_example()
y = df['Age'].values
del df['Age']
I = ~np.isnan(y)
X = hgb.preprocessing(df, verbose=0)
X = X.loc[I,:]
y = y[I]

# Fit
results = hgb.xgboost_reg(X, y, eval_metric='mae')
results = hgb.catboost_reg(X, y, eval_metric='mae')
results = hgb.lightboost_reg(X, y, eval_metric='mae')

# Make some plots
hgb.plot_params()
hgb.plot()
hgb.treeplot()
hgb.plot_validation()
hgb.plot_cv()

# use the predictor
y_pred, y_proba = hgb.predict(X)


# %% ENSEMBLE CLASSIFIER
hgb = hgboost(max_eval=10, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None, verbose=3)

# Import data
df = hgb.import_example()
y = df['Survived'].values
del df['Survived']
X = hgb.preprocessing(df, verbose=0)

results = hgb.ensemble(X, y, pos_label=1)

# use the predictor
y_pred, y_proba = hgb.predict(X)

# Plot
hgb.plot_validation()

# %% ENSEMBLE REGRESSION
hgb = hgboost(max_eval=10, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None, verbose=3)

# Import data
df = hgb.import_example()
y = df['Age'].values
del df['Age']
I = ~np.isnan(y)
X = hgb.preprocessing(df, verbose=0)
X = X.loc[I,:]
y = y[I]

results = hgb.ensemble(X, y, methods=['xgb_reg','ctb_reg','lgb_reg'])

# use the predictor
y_pred, y_proba = hgb.predict(X)

# Plot
hgb.plot_validation()

# %% CLASSIFICATION TWO-CLASS #####
from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris['feature_names'])
y = iris.target

hgb = hgboost(max_eval=10, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=42)
results = hgb.xgboost(X, y, pos_label=0)

# Plot
hgb.plot_params()
hgb.plot()
hgb.treeplot()
hgb.plot_validation()
hgb.plot_cv()

# %% CLASSIFICATION MULTI-CLASS #####
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris['feature_names'])
y = iris.target

hgb = hgboost(max_eval=10, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=42)
results = hgb.xgboost(X, y, method="xgb_clf_multi")

hgb.plot_params()
hgb.plot()
hgb.treeplot()
hgb.plot_validation()
hgb.plot_cv()
