""" This function provides explaintion of ensemble models such as XGboost, random forests etc.

	A= shapplot(model, <optional>)

 INPUT:
   model:          Trained model

 OPTIONAL

   verbose:        Integer or String
                   NOTHING=0
                   ERROR=1
                   WARN=2
                   INFO=3
                   DEBUG=4

 OUTPUT
	output

 DESCRIPTION
   Short description what your function does and how it is processed

 EXAMPLE
   import shap
   from sklearn.model_selection import train_test_split
   import xgboost
   import numpy as np
   from VIZ.shapplot import shapplot
   
   X,y = shap.datasets.adult()
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
   d_train = xgboost.DMatrix(X_train, label=y_train)
   d_test  = xgboost.DMatrix(X_test, label=y_test)

   params = {
    "eta": 0.01,
    "objective": "binary:logistic",
    "subsample": 0.5,
    "base_score": np.mean(y_train),
    "eval_metric": "logloss"
    }

   model = xgboost.train(params, d_train, 5000, evals = [(d_test, "test")], verbose_eval=100, early_stopping_rounds=20)

   A = shapplot(model,X,verbose=1)

 SEE ALSO
   treeplot, explainmodel
"""

#--------------------------------------------------------------------------
# Name        : shapplot.py
# Version     : 1.0
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : July. 2019
#--------------------------------------------------------------------------

#%% Libraries
import matplotlib.pyplot as plt
from types import SimpleNamespace
import GENERAL.log as log
import shap
import os

#%%
def shapplot(model, X, verbose=False):
	# DECLARATIONS
    out =dict()
    # Make dictionary to store Parameters
    Param = SimpleNamespace()
    Param.verbose = verbose
    Param.bundle='<script src="file:///D:/stack/TOOLBOX_PY/PY/resources/SHAP/bundle.js"></script>'

    # Make tree explainer
    # this takes a minute or two since we are explaining over 30 thousand samples in a model with over a thousand trees

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.values)
    
    #%% summarize the effects of all the features
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar")

    plt.figure()
    shap.summary_plot(shap_values, X)
    
    #%%
    
    #The dependence plot for the top feature shows that XGBoost captured most the linear relationship
    shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(X.values)
    plt.figure()
    A=shap.dependence_plot(X.columns[0], shap_values, X)
    plt.figure()
    shap.dependence_plot((1,2), shap_interaction_values, X.values)
    
    
    ## visualize the first prediction's explanation
    #A=shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
    #with open('./data_cached/shap_explainer.html','w') as f:
    #    f.write(bundle)
    #    f.write(A.data)
    #
    ## visualize the training set predictions
    ##A=shap.force_plot(explainer.expected_value, shap_values, X.iloc[:,0:10])
    ##To keep the browser happy we only visualize 1000 individuals.
    #
    #A=shap.force_plot(explainer.expected_value, shap_values[:10000,:], X.iloc[:10000,:])
    ##outfile = open('./data_cached/','wb')
    #with open('./data_cached/shap_expected_value.html','w') as f:
    #    f.write(bundle)
    #    f.write(A.data)
    #
    #
    ## create a SHAP dependence plot to show the effect of a single feature across the whole dataset
    #A=shap.dependence_plot("RM",shap_values[0,:], X.iloc[0,:])
    ##outfile = open('./data_cached/','wb')
    #with open('./data_cached/shap_dependence.html','w') as f:
    #    f.write(bundle)
    #    f.write(A.data)
    #
    #for name in X.columns:
    #    shap.dependence_plot(name, shap_values, X, display_features=X)

        
    #%% END
    return(out,Param)
