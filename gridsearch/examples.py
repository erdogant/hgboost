# %%
from gridsearch import gridsearch
print(dir(gridsearch))
# print(gridsearch.__version__)
import numpy as np

# %% Regression

gs = gridsearch(method='xgb_reg')
# gs = gridsearch(method='lgb_reg')
# gs = gridsearch(method='ctb_reg')
df = gs.import_example()
y = df['Age'].values
del df['Age']
X = gs.preprocessing(df, verbose=0)

I = ~np.isnan(y)
results = gs.fit(X.loc[I,:], y[I])
gs.plot()
# use the predictor
y_pred, y_proba = gs.predict(X)
gs.plot_summary()
gs.plot()

import matplotlib.pyplot as plt
plt.scatter(y, y_pred)

# %% classifier
gs = gridsearch(method='xgb_clf', max_evals=25)
df = gs.import_example()
y = df['Survived'].values
del df['Survived']
X = gs.preprocessing(df, verbose=2)

results = gs.fit(X, y==1)
results = gs.fit(X, y, pos_label=1)

# use the predictor
y_pred, y_proba = gs.predict(X)
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
