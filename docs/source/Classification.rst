.. include:: add_top.add


Classification
''''''''''''''''''''''''''

The ``hgboost`` method consists 3 **classification** methods: ``xgboost``, ``catboost``, ``lightboost``.
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

The specific list of parameters used for xgboost: :func:`hgboost.hgboost.hgboost.xgboost`

.. code:: python

    # Parameters:
    'learning_rate'     : hp.choice('learning_rate', np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000))
    'max_depth'         : hp.choice('max_depth', range(5, 32, 1))
    'min_child_weight'  : hp.quniform('min_child_weight', 1, 10, 1)
    'gamma'             : hp.choice('gamma', [0.5, 1, 1.5, 2, 3, 4, 5])
    'subsample'         : hp.quniform('subsample', 0.1, 1, 0.01)
    'n_estimators'      : hp.choice('n_estimators', range(20, 205, 5))
    'colsample_bytree'  : hp.quniform('colsample_bytree', 0.1, 1.0, 0.01)
    'scale_pos_weight'  : np.arange(0, 0.5, 1)
    'booster'           : 'gbtree'
    'early_stopping_rounds' : 25

    # In case of two-class classification
    objective = 'binary:logistic'
    # In case of multi-class classification
    objective = 'multi:softprob'


catboost
-------------

The specific list of parameters used for catboost: :func:`hgboost.hgboost.hgboost.catboost`

.. code:: python

    'learning_rate'     : hp.choice('learning_rate', np.logspace(np.log10(0.005), np.log10(0.31), base = 10, num = 1000))
    'depth'             : hp.choice('max_depth', np.arange(2, 16, 1, dtype=int))
    'iterations'        : hp.choice('iterations', np.arange(100, 1000, 100))
    'l2_leaf_reg'       : hp.choice('l2_leaf_reg', np.arange(1, 100, 2))
    'border_count'      : hp.choice('border_count', np.arange(5, 200, 1))
    'thread_count'      : 4
    'early_stopping_rounds' : 25


lightboost
--------------------------

The specific list of parameters used for lightboost: :func:`hgboost.hgboost.hgboost.lightboost`

.. code:: python

    # Parameters:
    'learning_rate'     : hp.choice('learning_rate', np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000))
    'max_depth'         : hp.choice('max_depth', np.arange(5, 75, 1))
    'boosting_type'     : hp.choice('boosting_type', ['gbdt','goss','dart'])
    'num_leaves'        : hp.choice('num_leaves', np.arange(100, 1000, 100))
    'n_estimators'      : hp.choice('n_estimators', np.arange(20, 205, 5))
    'subsample_for_bin' : hp.choice('subsample_for_bin', np.arange(20000, 300000, 20000))
    'min_child_samples' : hp.choice('min_child_weight', np.arange(20, 500, 5))
    'reg_alpha'         : hp.quniform('reg_alpha', 0, 1, 0.01)
    'reg_lambda'        : hp.quniform('reg_lambda', 0, 1, 0.01)
    'colsample_bytree'  : hp.quniform('colsample_bytree', 0.6, 1, 0.01)
    'subsample'         : hp.quniform('subsample', 0.5, 1, 100)
    'bagging_fraction'  : hp.choice('bagging_fraction', np.arange(0.2, 1, 0.2))
    'is_unbalance'      : hp.choice('is_unbalance', [True, False])
    'early_stopping_rounds' : 25




.. include:: add_bottom.add