.. include:: add_top.add


Regression
''''''''''''''''''''''''''

The ``hgboost`` method consists 3 **regression** methods: ``xgboost_reg``, ``catboost_reg``, ``lightboost_reg``.
Each algorithm provides hyperparameters that must very likely be tuned for a specific dataset.
Although there are many hyperparameters to tune, some are more important the others. The parameters used in ``hgboost`` are lised below:

Parameters
    * The number of trees or estimators.
    * The learning rate.
    * The row and column sampling rate for stochastic models.
    * The maximum tree depth.
    * The minimum tree weight.
    * The regularization terms alpha and lambda.


xgboost
---------

The specific list of parameters used for xgboost: :func:`hgboost.hgboost.hgboost.xgboost_reg`

.. code:: python

    # Parameters:
    'learning_rate'     : hp.quniform('learning_rate', 0.05, 0.31, 0.05)
    'max_depth'         : hp.choice('max_depth', np.arange(5, 30, 1, dtype=int))
    'min_child_weight'  : hp.choice('min_child_weight', np.arange(1, 10, 1, dtype=int))
    'gamma'             : hp.choice('gamma', [0, 0.25, 0.5, 1.0])
    'reg_lambda'        : hp.choice('reg_lambda', [0.1, 1.0, 5.0, 10.0, 50.0, 100.0])
    'subsample'         : hp.uniform('subsample', 0.5, 1)
    'n_estimators'      : hp.choice('n_estimators', range(20, 205, 5))
    'early_stopping_rounds' : 25


catboost
-------------

The specific list of parameters used for catboost: :func:`hgboost.hgboost.hgboost.catboost_reg`

.. code:: python

    'learning_rate'     : hp.quniform('learning_rate', 0.05, 0.31, 0.05),
    'max_depth'         : hp.choice('max_depth', np.arange(2, 16, 1, dtype=int)),
    'colsample_bylevel' : hp.choice('colsample_bylevel', np.arange(0.3, 0.8, 0.1)),
    'n_estimators'      : hp.choice('n_estimators', range(20, 205, 5)),
    'early_stopping_rounds' : 10


lightboost
--------------------------

The specific list of parameters used for lightboost: :func:`hgboost.hgboost.hgboost.lightboost_reg`

.. code:: python

    # Parameters:
    'learning_rate'     : hp.quniform('learning_rate', 0.05, 0.31, 0.05),
    'max_depth'         : hp.choice('max_depth', np.arange(5, 30, 1, dtype=int)),
    'min_child_weight'  : hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
    'subsample'         : hp.uniform('subsample', 0.8, 1),
    'n_estimators'      : hp.choice('n_estimators', range(20, 205, 5)),
    'eval_metric'       : 'l2'
    'early_stopping_rounds' : 25




.. include:: add_bottom.add