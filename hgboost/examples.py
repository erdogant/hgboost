# https://scikit-learn.org/stable/modules/cross_validation.html
# https://towardsdatascience.com/fine-tuning-xgboost-in-python-like-a-boss-b4543ed8b1e
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py
# https://github.com/WillKoehrsen/hyperparameter-optimization/blob/master/Bayesian%20Hyperparameter%20Optimization%20of%20Gradient%20Boosting%20Machine.ipynb
# https://www.kaggle.com/henrylidgley/xgboost-with-hyperopt-tuning

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
hgb = hgboost(max_evals=10, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=42)

# Import data
df = hgb.import_example()
y = df['Survived'].values
del df['Survived']
X = hgb.preprocessing(df, verbose=0)
y = y.astype(str)
y[y=='1']='survived'
y[y=='0']='dead'

# Fit
results = hgb.xgboost(X, y, pos_label='survived')
results = hgb.catboost(X, y, pos_label='survived')
results = hgb.lightboost(X, y, pos_label='survived')

# Make some plots
hgb.plot_params()
hgb.plot()
hgb.treeplot()
hgb.plot_validation()
hgb.plot_cv()

# use the predictor
y_pred, y_proba = hgb.predict(X)


# %% HYPEROPTIMIZED MULTI-XGBOOST
hgb = hgboost(max_evals=10, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=42)

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
hgb = hgboost(max_evals=10, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=42)

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


# %% CLASSIFICATION TWO-CLASS #####
from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris['feature_names'])
y = iris.target

hgb = hgboost(method='xgb_clf', max_evals=100, eval_metric='auc')
results = hgb.fit(X, y==1)

# Plot
hgb.plot_params()
hgb.plot()
hgb.treeplot()
hgb.plot_validation()

# %% CLASSIFICATION MULTI-CLASS #####
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris['feature_names'])
y = iris.target

hgb = hgboost(method='xgb_clf_multi', max_evals=100, eval_metric='mlogloss')
results = hgb.fit(X, y)
hgb.treeplot()
hgb.plot()
hgb.plot_validation()
