.. _code_directive:

-------------------------------------

Classification Examples
''''''''''''''''''''''''

xgboost two-class
-------------------

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