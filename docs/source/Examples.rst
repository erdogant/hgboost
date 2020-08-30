.. _code_directive:

-------------------------------------

Classification Examples
''''''''''''''''''''''''

xgboost two-class
-------------------

Function documentation can be found here :func:`hgboost.hgboost.hgboost.xgboost`

.. code:: python

    # Import library
    from hgboost import hgboost
    
    # Initialize
    hgb = hgboost(max_eval=250, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None, verbose=3)

    # Load example data set    
    df = hgb.import_example()
    # Prepare data for classification
    y = df['Survived'].values
    del df['Survived']
    X = hgb.preprocessing(df, verbose=0)

    # Fit best model with desired evaluation metric:
    results = hgb.xgboost(X, y, pos_label=1, eval_metric='f1')
    # [hgboost] >Start hgboost classification..
    # [hgboost] >Collecting xgb_clf parameters.
    # [hgboost] >Number of variables in search space is [10], loss function: [f1].
    # [hgboost] >method: xgb_clf
    # [hgboost] >eval_metric: f1
    # [hgboost] >greater_is_better: True
    # [hgboost] >Total datset: (891, 204) 
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



xgboost multi-class
---------------------

Function documentation can be found here :func:`hgboost.hgboost.hgboost.xgboost`

.. code:: python

    # Import library
    from hgboost import hgboost
    
    # Initialize
    hgb = hgboost(max_eval=250, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None, verbose=3)

    # Load example data set    
    df = hgb.import_example()
    # Prepare data for classification
    y = df['Parch'].values
    y[y>=3]=3
    del df['Parch']
    X = hgb.preprocessing(df, verbose=0)

    # Fit best model with desired evaluation metric:
    results = hgb.xgboost(X, y, method='xgb_clf_multi', eval_metric='kappa')
    # [hgboost] >Start hgboost classification..
    # [hgboost] >Collecting xgb_clf parameters.
    # [hgboost] >Number of variables in search space is [10], loss function: [kappa].
    # [hgboost] >method: xgb_clf_multi
    # [hgboost] >eval_metric: kappa
    # [hgboost] >greater_is_better: True
    # [hgboost] >Total datset: (891, 204) 
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


catboost
-------------

Function documentation can be found here :func:`hgboost.hgboost.hgboost.catboost`

.. code:: python

    # Import library
    from hgboost import hgboost
    
    # Initialize
    hgb = hgboost(max_eval=250, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None, verbose=3)

    # Load example data set    
    df = hgb.import_example()
    # Prepare data for classification
    y = df['Survived'].values
    del df['Survived']
    X = hgb.preprocessing(df, verbose=0)

    # Fit best model with desired evaluation metric:
    results = hgb.catboost(X, y, pos_label=1, eval_metric='auc')
    # [hgboost] >Start hgboost classification..
    # [hgboost] >Collecting ctb_clf parameters.
    # [hgboost] >Number of variables in search space is [10], loss function: [auc].
    # [hgboost] >method: ctb_clf
    # [hgboost] >eval_metric: auc
    # [hgboost] >greater_is_better: True
    # [hgboost] >Total datset: (891, 204) 
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


lightboost
-------------

Function documentation can be found here :func:`hgboost.hgboost.hgboost.lightboost`

.. code:: python

    # Import library
    from hgboost import hgboost
    
    # Initialize
    hgb = hgboost(max_eval=250, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None, verbose=3)

    # Load example data set    
    df = hgb.import_example()
    # Prepare data for classification
    y = df['Survived'].values
    del df['Survived']
    X = hgb.preprocessing(df, verbose=0)

    # Fit best model with desired evaluation metric:
    results = hgb.lightboost(X, y, pos_label=1, eval_metric='auc')
    # [hgboost] >Start hgboost classification..
    # [hgboost] >Collecting lgb_clf parameters.
    # [hgboost] >Number of variables in search space is [10], loss function: [auc].
    # [hgboost] >method: lgb_clf
    # [hgboost] >eval_metric: auc
    # [hgboost] >greater_is_better: True
    # [hgboost] >Total datset: (891, 204) 
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


Regression Examples
''''''''''''''''''''''''

xgboost_reg
-------------------

Function documentation can be found here :func:`hgboost.hgboost.hgboost.xgboost_reg`

.. code:: python

    # Import library
    from hgboost import hgboost
    
    # Initialize
    hgb = hgboost(max_eval=250, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None)

    # Load example data set
    df = hgb.import_example()
    y = df['Age'].values
    del df['Age']
    I = ~np.isnan(y)
    X = hgb.preprocessing(df, verbose=0)
    X = X.loc[I,:]
    y = y[I]

    # Fit best model with desired evaluation metric:
    results = hgb.xgboost_reg(X, y, eval_metric='rmse')
    # [hgboost] >Start hgboost regression..
    # [hgboost] >Collecting xgb_reg parameters.
    # [hgboost] >Number of variables in search space is [10], loss function: [rmse].
    # [hgboost] >method: xgb_reg
    # [hgboost] >eval_metric: rmse
    # [hgboost] >greater_is_better: True
    # [hgboost] >Total datset: (891, 204) 
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


lightboost_reg
-------------------

Function documentation can be found here :func:`hgboost.hgboost.hgboost.lightboost_reg`

.. code:: python

    # Import library
    from hgboost import hgboost
    
    # Initialize
    hgb = hgboost(max_eval=250, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None)

    # Load example data set
    df = hgb.import_example()
    y = df['Age'].values
    del df['Age']
    I = ~np.isnan(y)
    X = hgb.preprocessing(df, verbose=0)
    X = X.loc[I,:]
    y = y[I]

    # Fit best model with desired evaluation metric:
    results = hgb.lightboost_reg(X, y, eval_metric='rmse')
    # [hgboost] >Start hgboost regression..
    # [hgboost] >Collecting lgb_reg parameters.
    # [hgboost] >Number of variables in search space is [10], loss function: [rmse].
    # [hgboost] >method: lgb_reg
    # [hgboost] >eval_metric: rmse
    # [hgboost] >greater_is_better: True
    # [hgboost] >Total datset: (891, 204) 
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


catboost_reg
-------------------

Function documentation can be found here :func:`hgboost.hgboost.hgboost.catboost_reg`

.. code:: python

    # Import library
    from hgboost import hgboost
    
    # Initialize
    hgb = hgboost(max_eval=250, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None)

    # Load example data set
    df = hgb.import_example()
    y = df['Age'].values
    del df['Age']
    I = ~np.isnan(y)
    X = hgb.preprocessing(df, verbose=0)
    X = X.loc[I,:]
    y = y[I]

    # Fit best model with desired evaluation metric:
    results = hgb.catboost_reg(X, y, eval_metric='rmse')
    # [hgboost] >Start hgboost regression..
    # [hgboost] >Collecting ctb_reg parameters.
    # [hgboost] >Number of variables in search space is [10], loss function: [rmse].
    # [hgboost] >method: ctb_reg
    # [hgboost] >eval_metric: rmse
    # [hgboost] >greater_is_better: True
    # [hgboost] >Total datset: (891, 204) 
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


Plots
''''''''''''''''''''''''

For each model, the following 5 plots can be created:


plot_params
-------------------

Figure 1 depicts the density of the specific parameter values. As an example, the **gamma** parameter shows that most iterations converges towards value **0**.
This may indicate that this parameter with this value has an important role in the in computing the optimal loss.
Figure 2 depicts the iterations performed for hyper-optimization per parameter. In case of **colsample_bytree** we see a convergence towards the range [0.5-0.7].

In both figures, the parameters for all fitted models are plotted together with the **best** performing models with and without the **k-fold crossvalidation**.
In addition, we also plot the top n performing models. The top performing models can be usefull to deeper examine the used parameter.

Function documentation can be found here :func:`hgboost.hgboost.hgboost.plot_params`

.. code:: python

    # Plot the parameter space
    hgb.plot_params()


.. |figS1| image:: ../figs/plot_params_clf_1.png
.. |figS2| image:: ../figs/plot_params_clf_2.png

.. table:: Parameter plot
   :align: center

   +----------+
   | |figS1|  |
   +----------+
   | |figS2|  |
   +----------+



plot summary
-------------------

This figure exists out of two subfigures. The top figure depicts all evaluated models with the loss score.
The **best** performing models with and without the **k-fold crossvalidation** are depicted together with the top n performing models.
The bottom figure depicts the train and test-error.
Function documentation can be found here :func:`hgboost.hgboost.hgboost.plot`

.. code:: python

    # Plot the summary results
    hgb.plot()


.. |figS3| image:: ../figs/plot_clf.png

.. table:: Summary plot of the results.
   :align: center

   +----------+
   | |figS3|  |
   +----------+


treeplot
-------------------

Function documentation can be found here :func:`hgboost.hgboost.hgboost.treeplot`

.. code:: python

    # Plot the best performing tree
    hgb.treeplot()

.. |figS4| image:: ../figs/treeplot_clf_1.png
.. |figS5| image:: ../figs/treeplot_clf_2.png

.. table:: Best performing tree.
   :align: center

   +----------+
   | |figS4|  |
   +----------+
   | |figS5|  |
   +----------+


plot_validation
-------------------

Function documentation can be found here :func:`hgboost.hgboost.hgboost.plot_validation`

.. code:: python

    # Plot results on the validation set
    hgb.plot_validation()


.. |figS6| image:: ../figs/plot_validation_clf_1.png

.. table:: Results on the validation set.
   :align: center

   +----------+
   | |figS6|  |
   +----------+


.. |figS7| image:: ../figs/plot_validation_clf_2.png
.. |figS8| image:: ../figs/plot_validation_clf_3.png

.. table:: Results on the validation set.
   :align: center

   +----------+----------+
   | |figS7|  | |figS8|  |
   +----------+----------+


plot_cv
-------------------

Function documentation can be found here :func:`hgboost.hgboost.hgboost.plot_cv`

.. code:: python

    # Plot results on the cross-validation
    hgb.plot_cv()

.. |figS9| image:: ../figs/plot_cv_clf.png

.. table:: results on the cross-validation.
   :align: center

   +----------+
   | |figS9|  |
   +----------+
