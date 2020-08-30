.. _code_directive:

-------------------------------------

Abstract
''''''''

Background
    Gradient boosting is a powerful ensemble machine learning algorithm for predictive modeling that can be applied on tabular data.
    There are many implementations of gradient boosting, some efficently uses the GPU, whereas others have specific interfaces.
    For this library ``hgboost``, we incorporated the *eXtreme Gradient Boosting* ``xgboost``[1], *Light Gradient Boosting Machine* ``LightGBM``[2],
    and *Category Gradient Boosting* ``catboost``[3].

Aim
    The aim of the gradient boosting algorithm is to fit a boosted decision trees by minimizing an error gradient. However, there are many
    parameters that can be tuned, for which a combination of parameter can result in more accurate predictions. Searching across
    the combination of parameters is often performed with gridsearches. A gridsearch comes with high computational costs, and can easily result
    in overtrained models. The aim of ``hgboost`` is too determine the most robust model by efficiently searching across the parameter space using
    **hyperoptimization** for which the loss is evaluated using by means of a train/test-set with k-fold cross-validation.
    In addition, the final optimized model is evaluated on an independent validation set.
    
Results
    ``hgboost`` is Python package that minimizes the function from the model xgboost, catboost and lightboost over a hyperparameter space
    by using k-fold cross-validation and evaluting the results on an indepdendent validation set.

    
Schematic overview
'''''''''''''''''''

The schematic overview of our approach is as following:

.. _schematic_overview:

.. figure:: ../figs/schematic_overview.png


References
-----------
    * [1] https://github.com/dmlc/xgboost
    * [2] https://github.com/microsoft/LightGBM
    * [3] https://github.com/catboost/catboost
