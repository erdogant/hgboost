# https://scikit-learn.org/stable/modules/cross_validation.html
# https://towardsdatascience.com/fine-tuning-xgboost-in-python-like-a-boss-b4543ed8b1e
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py
# https://github.com/WillKoehrsen/hyperparameter-optimization/blob/master/Bayesian%20Hyperparameter%20Optimization%20of%20Gradient%20Boosting%20Machine.ipynb

# %%
from gridsearch import gridsearch
print(dir(gridsearch))
# print(gridsearch.__version__)
import numpy as np

# %% classifier
gs = gridsearch(method='xgb_clf', max_evals=25, cv=5)
df = gs.import_example()
y = df['Survived'].values
del df['Survived']
X = gs.preprocessing(df, verbose=0)

results = gs.fit(X, y==1)
results = gs.fit(X, y, pos_label=1)

# use the predictor
y_pred, y_proba = gs.predict(X)
gs.plot_summary()
gs.plot()

import matplotlib.pyplot as plt
plt.scatter(y, y_pred)

# %%
best_bayes_params['method'] = 'Bayesian optimization'
best_params = pd.DataFrame(gs.results['trials'], index = [0])
append(pd.DataFrame(best_random_params, index = [0]), ignore_index = True, sort = True)

# Create a new dataframe for storing parameters
bayes_params = pd.DataFrame(columns = list(ast.literal_eval(gs.results.loc[0, 'params']).keys()), index = list(range(len(results))))

# %% Regression

gs = gridsearch(method='xgb_reg')
# gs = gridsearch(method='lgb_reg')
# gs = gridsearch(method='ctb_reg')
df = gs.import_example()
y = df['Age'].values
del df['Age']
I = ~np.isnan(y)
X = gs.preprocessing(df, verbose=0)
y = y[I]
X = X.loc[I,:]

# Fit
results = gs.fit(X, y)
# Prdict
y_pred, y_proba = gs.predict(X)
# Plot
gs.plot_summary()
gs.plot()

import matplotlib.pyplot as plt
plt.scatter(y, y_pred)




# %% CLASSIFICATION TWO-CLASS #####
from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris['feature_names'])
y = iris.target

gs = gridsearch(method='xgb_clf', max_evals=100)
results = gs.fit(X, y==1)
gs.plot()
gs.plot_summary()

##### CLASSIFICATION MULTI-CLASS #####
gs = gridsearch(method='xgb_clf_multi', max_evals=100)
results = gs.fit(X, y)
gs.plot()
gs.plot_summary()
