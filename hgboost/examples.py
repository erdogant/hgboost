# %%




# %% CLASSIFICATION TWO-CLASS #####

# Import
from hgboost import hgboost

# Initialize
hgb = hgboost(max_eval=50)

# Initialize library.
# hgb = hgboost(
#     max_eval=250,      # Search space is based  on the number of evaluations.
#     threshold=0.5,     # Classification threshold. In case of two-class model this is 0.5.
#     cv=5,              # k-folds cross-validation.
#     test_size=0.2,     # Percentage split for the testset.
#     val_size=0.2,      # Percentage split for the validationset.
#     top_cv_evals=10,   # Number of top best performing models that is evaluated.
#     is_unbalanced=True, # Control the balance of positive and negative weights, useful for unbalanced classes.
#     random_state=None, # Fix the random state to create reproducible results.
#     n_jobs=-1,         # The number of CPU jobs to run in parallel. -1 means using all processors.
#     gpu=False,         # Compute using GPU in case of True.
#     verbose='info',    # Print progress to screen.
# )


# Example data
df = hgb.import_example(data='titanic')
# Cleaning
y = df['Survived'].values
df = df.drop(['Survived', 'PassengerId', 'Name'], axis=1)

# Preprocessing to one-hot
X = hgb.preprocessing(df)

# Train a model with XGBoost
results = hgb.xgboost(X, y, pos_label=1, eval_metric='auc')

# Train a model with CatBoost
# results = hgb.catboost(X, y, pos_label=1, eval_metric='auc')
# Train a model with LightBoost
# results = hgb.lightboost(X, y, pos_label=1, eval_metric='auc')
# Results are stored in the object itself.

# Plot the hyperparameter space
hgb.plot_params()

# Plot the summary of all evaluated models
hgb.plot()

# Plot results on the k-fold cross validation
hgb.plot_cv()

# Plot results on the validation set
hgb.plot_validation()

# Plot best performing tree
hgb.treeplot()

# %%

# Import library
from hgboost import hgboost

# Initialize
hgb = hgboost(max_eval=50, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None, verbose='info')

# Load example data set
df = hgb.import_example()
# Prepare data for classification
y = df['Survived'].values
del df['Survived']
X = hgb.preprocessing(df)

# Fit best model with desired evaluation metric:
results = hgb.xgboost(X, y, pos_label=1, eval_metric='f1')
# [hgboost] >Start hgboost classification..
# [hgboost] >Collecting xgb_clf parameters.
# [hgboost] >Number of variables in search space is [10], loss function: [f1].
# [hgboost] >method: xgb_clf
# [hgboost] >eval_metric: f1
# [hgboost] >larger_is_better: True
# [hgboost] >Total dataset: (891, 204)
# [hgboost] >Hyperparameter optimization..

# Plot the parameter space
hgb.plot_params()
# Plot the summary results
hgb.plot()
# Plot the best performing tree
hgb.treeplot()
# Plot results on the validation set
hgb.plot_validation()
# Plot results on the cross-validation
hgb.plot_cv()

# Make new prdiction using the model (suppose that X is new and unseen data which is similarly prepared as for the learning process)
y_pred, y_proba = hgb.predict(X)

# %%
# Import library
from hgboost import hgboost

# Initialize
hgb = hgboost(max_eval=50, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None, verbose='info')

# Load example data set
df = hgb.import_example()
# Prepare data for classification
y = df['Survived'].values
del df['Survived']
X = hgb.preprocessing(df)

# Fit best model with desired evaluation metric:
results = hgb.catboost(X, y, pos_label=1, eval_metric='auc')
# [hgboost] >Start hgboost classification..
# [hgboost] >Collecting ctb_clf parameters.
# [hgboost] >Number of variables in search space is [10], loss function: [auc].
# [hgboost] >method: ctb_clf
# [hgboost] >eval_metric: auc
# [hgboost] >larger_is_better: True
# [hgboost] >Total dataset: (891, 204)
# [hgboost] >Hyperparameter optimization..

# Plot the parameter space
hgb.plot_params()
# Plot the summary results
hgb.plot()
# Plot the best performing tree
hgb.treeplot()
# Plot results on the validation set
hgb.plot_validation()
# Plot results on the cross-validation
hgb.plot_cv()

# Make new prdiction using the model (suppose that X is new and unseen data which is similarly prepared as for the learning process)
y_pred, y_proba = hgb.predict(X)


# %%

# Import library
from hgboost import hgboost

# Initialize
hgb = hgboost(max_eval=50, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None, verbose='info')

# Load example data set
df = hgb.import_example()
# Prepare data for classification
y = df['Survived'].values
del df['Survived']
X = hgb.preprocessing(df)

# Fit best model with desired evaluation metric:
results = hgb.lightboost(X, y, pos_label=1, eval_metric='auc')
# [hgboost] >Start hgboost classification..
# [hgboost] >Collecting lgb_clf parameters.
# [hgboost] >Number of variables in search space is [10], loss function: [auc].
# [hgboost] >method: lgb_clf
# [hgboost] >eval_metric: auc
# [hgboost] >larger_is_better: True
# [hgboost] >Total dataset: (891, 204)
# [hgboost] >Hyperparameter optimization..

# Plot the parameter space
hgb.plot_params()
# Plot the summary results
hgb.plot()
# Plot the best performing tree
hgb.treeplot()
# Plot results on the validation set
hgb.plot_validation()
# Plot results on the cross-validation
hgb.plot_cv()

# Make new prdiction using the model (suppose that X is new and unseen data which is similarly prepared as for the learning process)
y_pred, y_proba = hgb.predict(X)

# %%
# Import library
from hgboost import hgboost

# Initialize
hgb = hgboost(max_eval=50, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None, verbose='info')

# Load example data set
df = hgb.import_example()
# Prepare data for classification
y = df['Parch'].values.copy()
y[y>=3]=3
del df['Parch']
X = hgb.preprocessing(df)

# Fit best model with desired evaluation metric:
results = hgb.xgboost(X, y, method='xgb_clf_multi', eval_metric='kappa')
# [hgboost] >Start hgboost classification..
# [hgboost] >Collecting xgb_clf parameters
# [hgboost] >Number of variables in search space is [10], loss function: [kappa]
# [hgboost] >method: xgb_clf_multi
# [hgboost] >eval_metric: kappa
# [hgboost] >larger_is_better: True
# [hgboost] >Total dataset: (891, 204)
# [hgboost] >Hyperparameter optimization..

# Plot the parameter space
hgb.plot_params()
# Plot the summary results
hgb.plot()
# Plot the best performing tree
hgb.treeplot()
# Plot results on the validation set
hgb.plot_validation()
# Plot results on the cross-validation
hgb.plot_cv()

# Make new prdiction using the model (suppose that X is new and unseen data which is similarly prepared as for the learning process)
y_pred, y_proba = hgb.predict(X)


# %%
# Import library
from hgboost import hgboost
import numpy as np

# Initialize
hgb = hgboost(max_eval=250, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None)

# Load example data set
df = hgb.import_example()
y = df['Age'].values
del df['Age']
I = ~np.isnan(y)
X = hgb.preprocessing(df)
X = X.loc[I,:]
y = y[I]

# Fit best model with desired evaluation metric:
results = hgb.xgboost_reg(X, y, eval_metric='rmse')
# [hgboost] >Start hgboost regression..
# [hgboost] >Collecting xgb_reg parameters.
# [hgboost] >Number of variables in search space is [10], loss function: [rmse].
# [hgboost] >method: xgb_reg
# [hgboost] >eval_metric: rmse
# [hgboost] >larger_is_better: True
# [hgboost] >Total dataset: (891, 204)
# [hgboost] >Hyperparameter optimization..

# Plot the parameter space
hgb.plot_params()
# Plot the summary results
hgb.plot()
# Plot the best performing tree
hgb.treeplot()
# Plot results on the validation set
hgb.plot_validation()
# Plot results on the cross-validation
hgb.plot_cv()

# Make new prdiction using the model (suppose that X is new and unseen data which is similarly prepared as for the learning process)
y_pred, y_proba = hgb.predict(X)


# %%

# Import library
from hgboost import hgboost

# Initialize
hgb = hgboost(max_eval=250, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None)

# Load example data set
df = hgb.import_example()
y = df['Age'].values
del df['Age']
I = ~np.isnan(y)
X = hgb.preprocessing(df)
X = X.loc[I,:]
y = y[I]

# Fit best model with desired evaluation metric:
results = hgb.lightboost_reg(X, y, eval_metric='rmse')
# [hgboost] >Start hgboost regression..
# [hgboost] >Collecting lgb_reg parameters.
# [hgboost] >Number of variables in search space is [10], loss function: [rmse].
# [hgboost] >method: lgb_reg
# [hgboost] >eval_metric: rmse
# [hgboost] >larger_is_better: True
# [hgboost] >Total dataset: (891, 204)
# [hgboost] >Hyperparameter optimization..

# Plot the parameter space
hgb.plot_params()
# Plot the summary results
hgb.plot()
# Plot the best performing tree
hgb.treeplot()
# Plot results on the validation set
hgb.plot_validation()
# Plot results on the cross-validation
hgb.plot_cv()

# Make new prdiction using the model (suppose that X is new and unseen data which is similarly prepared as for the learning process)
y_pred, y_proba = hgb.predict(X)


# %%

# Import library
from hgboost import hgboost

# Initialize
hgb = hgboost(max_eval=250, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None)

# Load example data set
df = hgb.import_example()
y = df['Age'].values
del df['Age']
I = ~np.isnan(y)
X = hgb.preprocessing(df)
X = X.loc[I,:]
y = y[I]

# Fit best model with desired evaluation metric:
results = hgb.catboost_reg(X, y, eval_metric='rmse')
# [hgboost] >Start hgboost regression..
# [hgboost] >Collecting ctb_reg parameters.
# [hgboost] >Number of variables in search space is [10], loss function: [rmse].
# [hgboost] >method: ctb_reg
# [hgboost] >eval_metric: rmse
# [hgboost] >larger_is_better: True
# [hgboost] >Total dataset: (891, 204)
# [hgboost] >Hyperparameter optimization..

# Plot the parameter space
hgb.plot_params()
# Plot the summary results
hgb.plot()
# Plot the best performing tree
hgb.treeplot()
# Plot results on the validation set
hgb.plot_validation()
# Plot results on the cross-validation
hgb.plot_cv()

# Make new prdiction using the model (suppose that X is new and unseen data which is similarly prepared as for the learning process)
y_pred, y_proba = hgb.predict(X)


# %%
# Import library
from hgboost import hgboost

# Initialize
hgb = hgboost(max_eval=50, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None, verbose='info')

# Import data
df = hgb.import_example()
y = df['Survived'].values
del df['Survived']
X = hgb.preprocessing(df)

# Fit ensemble model using the three boosting methods. By default these are readily set.
results = hgb.ensemble(X, y, pos_label=1)
# [hgboost] >Create ensemble regression model..
# [hgboost] >...
# [hgboost] >Fit ensemble model with [soft] voting..
# [hgboost] >Evalute [ensemble] model on independent validation dataset (179 samples, 20%)
# [hgboost] >[Ensemble] [auc]: -0.9788 on independent validation dataset
# [hgboost] >[xgb_clf]  [auc]: -0.8434 on independent validation dataset
# [hgboost] >[ctb_clf]  [auc]: -0.8875 on independent validation dataset
# [hgboost] >[lgb_clf]  [auc]: -0.8816 on independent validation dataset

# use the predictor
y_pred, y_proba = hgb.predict(X)

# Plot
hgb.plot_validation()


# %%
# Import library
from hgboost import hgboost

# Initialize
hgb = hgboost(max_eval=50, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None)

# Load example data set
df = hgb.import_example()
y = df['Age'].values
del df['Age']
I = ~np.isnan(y)
X = hgb.preprocessing(df)
X = X.loc[I,:]
y = y[I]

# Fit ensemble model using the three boosting methods:
results = hgb.ensemble(X, y, methods=['xgb_reg','ctb_reg','lgb_reg'])
# [hgboost] >Create ensemble regression model..
# [hgboost] >...
# [hgboost] >Evalute [ensemble] model on independent validation dataset (143 samples, 20%).
# [hgboost] >[Ensemble] [rmse]: 64.62 on independent validation dataset
# [hgboost] >[xgb_reg]  [rmse]: 172.2 on independent validation dataset
# [hgboost] >[ctb_reg]  [rmse]: 183 on independent validation dataset
# [hgboost] >[lgb_reg]  [rmse]: 205.9 on independent validation dataset

# Make new prdiction using the model (suppose that X is new and unseen data which is similarly prepared as for the learning process)
y_pred, y_proba = hgb.predict(X)

# Plot
hgb.plot_validation()

# =============================================================================
# 
# =============================================================================

# %% CLASSIFICATION TWO-CLASS #####
from hgboost import hgboost
from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris['feature_names'])
y = iris.target


hgb = hgboost(max_eval=10, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=42, verbose='silent')
results = hgb.xgboost(X, y, pos_label=0)

# Plot
hgb.plot_params()
hgb.plot()
hgb.treeplot()
hgb.plot_validation()
hgb.plot_cv()



# =============================================================================
# OLD EXAMPLES BELOW!!
# =============================================================================
# %% HYPEROPTIMIZED REGRESSION-XGBOOST
import numpy as np
from hgboost import hgboost

hgb_xgb = hgboost(max_eval=25, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=42, verbose='info')
# hgb_cat = hgboost(max_eval=25, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=42, verbose='info')
hgb_light = hgboost(max_eval=25, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=42, verbose='info')

# Import data
df = hgb_xgb.import_example()
y = df['Age'].values
df.drop(['Age', 'PassengerId', 'Name'], axis=1, inplace=True)

# Preprocessing
X = hgb_xgb.preprocessing(df)
I = ~np.isnan(y)
X = X.loc[I, :]
y = y[I]

# Fit
results = hgb_xgb.xgboost_reg(X, y, eval_metric='mae')
# results2 = hgb_cat.catboost_reg(X, y, eval_metric='mae')
results3 = hgb_light.lightboost_reg(X, y, eval_metric='mae')

# hgb_xgb.save('c:\\temp\\hgb_xgb.pkl')
# hgb_cat.save('c:\\temp\\hgb_cat.pkl')
# hgb_light.save('c:\\temp\\hgb_light.pkl')

# Make some plots
hgb_xgb.plot(ylim=[8.5, 13], plot2=False)
hgb_xgb.plot_params()
hgb_xgb.plot_validation()
hgb_xgb.plot_cv()
hgb_xgb.treeplot(plottype='vertical')

# hgb_cat.plot_params()
# hgb_cat.plot(ylim=[8, 12])
# hgb_cat.treeplot()
# hgb_cat.plot_validation()
# hgb_cat.plot_cv()

hgb_light.plot_params()
hgb_light.plot(ylim=[8, 12])
hgb_light.treeplot()
hgb_light.plot_validation()
hgb_light.plot_cv()

# use the predictor
# y_pred, y_proba = hgb.predict(X)


#%%
# https://scikit-learn.org/stable/modules/cross_validation.html
# https://towardsdatascience.com/fine-tuning-xgboost-in-python-like-a-boss-b4543ed8b1e
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py
# https://github.com/WillKoehrsen/hyperparameter-optimization/blob/master/Bayesian%20Hyperparameter%20Optimization%20of%20Gradient%20Boosting%20Machine.ipynb
# https://www.kaggle.com/henrylidgley/xgboost-with-hyperopt-tuning
# https://towardsdatascience.com/tree-boosted-mixed-effects-models-4df610b624cb
# https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/
# http://hyperopt.github.io/hyperopt/getting-started/minimizing_functions/

import numpy as np
from hgboost import hgboost

hgb = hgboost(max_eval=25, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=42)

# Import data
df = hgb.import_example()
y = df['Age'].values
df.drop(['Age', 'PassengerId', 'Name'], axis=1, inplace=True)

# Preprocessing
X = hgb.preprocessing(df)
I = ~np.isnan(y)
X = X.loc[I, :]
y = y[I]

# Fit
hgb.lightboost_reg(X, y, eval_metric='mae');

# Make some plots
hgb.plot(ylim=[8.5, 13], plot2=False)
hgb.plot_params()
hgb.plot_validation()
hgb.plot_cv()
hgb.treeplot(plottype='vertical')

# %%
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
# import numpy as np
# from hgboost import hgboost
# print(dir(hgboost))
# print(hgboost.__version__)

# %%
from hgboost import hgboost
hgb = hgboost()
df = hgb.import_example(data='ds_salaries')

y = df['salary_in_usd'].values
df.drop(['salary_in_usd', 'salary_currency', 'salary'], axis=1, inplace=True)
X = hgb.preprocessing(df)

import numpy as np
I = ~np.isnan(y)
X = X.loc[I, :]
y = y[I]

results  = hgb.xgboost_reg(X, y, eval_metric='mae')      # XGBoost
# results  = hgb.xgboost_reg(X, y, eval_metric='rmse')      # XGBoost

hgb.plot(ylim=[8.5, 13], plot2=False)
hgb.plot_params()
hgb.plot_validation()
hgb.plot_cv()
hgb.treeplot(plottype='vertical')




# %% HYPEROPTIMIZED regression XGBOOST
from hgboost import hgboost
import pandas as pd
file = "C://Users//playground//Downloads//Dataset//Dataset//Training//Features_Variant_1.csv"
df = pd.read_csv(file, header=None)
y = df.loc[:,53].values
X  = df.loc[:,:52] 

hgb_xgb = hgboost(max_eval=10, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=42, verbose='debug')
results = hgb_xgb.xgboost_reg(X, y, eval_metric='mae')
# Baseline MAE is 11.31

hgb_xgb.plot_params()
hgb_xgb.plot()
hgb_xgb.treeplot()
hgb_xgb.plot_validation()
hgb_xgb.plot_cv()



from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.1, random_state=42)

model = LinearRegression().fit(hgb_xgb.X_train, hgb_xgb.y_train)
model.score(hgb_xgb.X_train, hgb_xgb.y_train)
model.score(hgb_xgb.X_val, hgb_xgb.y_val)
 
# "Learn" the mean from the training data
mean_train = np.mean(y_train)
# Get predictions on the test set
baseline_predictions = np.ones(y_test.shape) * mean_train
# Compute MAE
mae_baseline = mean_absolute_error(y_test, baseline_predictions)
print("Baseline MAE is {:.2f}".format(mae_baseline))
# Baseline MAE is 11.31


# %% HYPEROPTIMIZED regression XGBOOST
from hgboost import hgboost

from sklearn.datasets import fetch_california_housing, load_diabetes
df = fetch_california_housing(as_frame=True)
df = df['frame']
y = df['MedHouseVal'].values
del df['MedHouseVal']
del df['Latitude']
del df['Longitude']

df = load_diabetes(as_frame=True)
df = df['frame']
y = df['target'].values
del df['target']


hgb_xgb = hgboost(max_eval=50, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=42, verbose='debug')
results = hgb_xgb.xgboost_reg(df, y, eval_metric='mae')

# Make some plots
hgb_xgb.plot_params()
hgb_xgb.plot()
hgb_xgb.treeplot()
hgb_xgb.plot_validation()
hgb_xgb.plot_cv()

# BAseline simple regression
model = LinearRegression().fit(hgb_xgb.X_train, hgb_xgb.y_train)
model.score(hgb_xgb.X_train, hgb_xgb.y_train)
model.score(hgb_xgb.X_test, hgb_xgb.y_test)
model.score(hgb_xgb.X_val, hgb_xgb.y_val)
 
# "Learn" the mean from the training data
mean_train = np.mean(hgb_xgb.y_train)
# Get predictions on the test set
baseline_predictions = np.ones(hgb_xgb.y_test.shape) * mean_train
# Compute MAE
mae_baseline = mean_absolute_error(hgb_xgb.y_test, baseline_predictions)
print("Baseline MAE is {:.2f}".format(mae_baseline))
# Baseline MAE is 11.31


# %% HYPEROPTIMIZED CLASSIFICATION XGBOOST
from hgboost import hgboost
hgb_xgb = hgboost(max_eval=25, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=0, gpu=False, verbose='info')
hgb_cat = hgboost(max_eval=25, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=0, gpu=False, verbose='info')
hgb_light = hgboost(max_eval=25, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=0, gpu=False, verbose='info')

# Import data
df = hgb_xgb.import_example()
del df['PassengerId']
del df['Name']

y = df['Survived'].values
del df['Survived']
X = hgb_xgb.preprocessing(df)

# Fit
results = hgb_xgb.xgboost(df, y, pos_label=1)
results = hgb_xgb.xgboost(X, y, pos_label=1)
results = hgb_cat.catboost(X, y, pos_label=1)
results = hgb_light.lightboost(X, y, pos_label=1)

# Make some plots
hgb_xgb.plot_params()
hgb_xgb.plot()
hgb_xgb.treeplot()
hgb_xgb.plot_validation()
hgb_xgb.plot_cv()

hgb_cat.plot_params()
hgb_cat.plot()
hgb_cat.treeplot()
hgb_cat.plot_validation()
hgb_cat.plot_cv()

hgb_light.plot_params()
hgb_light.plot()
hgb_light.treeplot()
hgb_light.plot_validation()
hgb_light.plot_cv()

# use the predictor
y_pred, y_proba = hgb_xgb.predict(X)
y_pred, y_proba = hgb_cat.predict(X)
y_pred, y_proba = hgb_light.predict(X)

import matplotlib.pyplot as plt
plt.figure();plt.plot(results['summary']['loss'])

#     booster colsample_bytree gamma  ... best_cv loss_validation default_params
# 0    gbtree             0.55     2  ...     0.0             NaN          False
# 1    gbtree             0.56     1  ...     0.0             NaN          False
# 2    gbtree             0.64     5  ...     0.0             NaN          False
# 3    gbtree             0.59     5  ...     0.0             NaN          False
# 4    gbtree             0.22   1.5  ...     0.0             NaN          False
# ..      ...              ...   ...  ...     ...             ...            ...
# 246  gbtree              0.8     2  ...     0.0             NaN          False
# 247  gbtree             0.45     1  ...     0.0             NaN          False
# 248  gbtree             0.88   0.5  ...     0.0             NaN          False
# 249  gbtree             0.57     1  ...     0.0             NaN          False
# 250     NaN                1     0  ...     NaN         0.85415           True

# [251 rows x 23 columns]

# %%
from hgboost import hgboost
hg = hgboost(max_eval=10, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=0, gpu=True, verbose='info')
df = hg.import_example()
y = df['Survived'].values
del df['Survived']
X = hg.preprocessing(df)

# # Fit
results = hg.xgboost(X, y, pos_label=1)

# %%
hg.save('test')
hg.load('test')
# results['model']
# results['params']




# %% HYPEROPTIMIZED MULTI-CLASS XGBOOST
from hgboost import hgboost
hgb = hgboost(max_eval=10, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=42)

# Import data
df = hgb.import_example()
y = df['Parch'].values.copy()
y[y>=3]=3
del df['Parch']
X = hgb.preprocessing(df)

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


# %% ENSEMBLE CLASSIFIER
hgb = hgboost(max_eval=10, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None, gpu=True, verbose='info')

# Import data
df = hgb.import_example()
y = df['Survived'].values
del df['Survived']
X = hgb.preprocessing(df)

results = hgb.ensemble(X, y, pos_label=1, methods=['xgb_clf', 'ctb_clf', 'lgb_clf'])

# use the predictor
y_pred, y_proba = hgb.predict(X)

# Plot
hgb.plot_validation()
hgb.plot()

# %% ENSEMBLE REGRESSION
hgb = hgboost(max_eval=10, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None, verbose='info')

# Import data
df = hgb.import_example()
y = df['Age'].values
del df['Age']
I = ~np.isnan(y)
X = hgb.preprocessing(df)
X = X.loc[I, :]
y = y[I]

results = hgb.ensemble(X, y, methods=['xgb_reg', 'ctb_reg', 'lgb_reg'])

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
from sklearn import datasets
import pandas as pd

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

# %%