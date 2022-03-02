.. _code_directive:

-------------------------------------

Abstract
''''''''

Background
    Gradient boosting is a powerful ensemble machine learning algorithm for predictive modeling that can be applied on tabular data.
    Creating predictions with models such as xgboost are often used in data science projects.
    But without having good knowledge of the data in combination with the model parameters, this can quickly result in a poor/overtrained model.
    By controlling parameters such as the "early stopping rounds" can certainly be helpful.

    Parameters can be tuned, and a combination of parameters can result in more accurate predictions. Searching across
    combinations of parameters is often performed with gridsearches. A gridsearch comes with high computational costs, and can easily result
    in overtrained models as the search space can easily consist tens of thousands combinations to evaluate.

    Luckily we have optimizations models, such as ``hyperopt`` [1], that can do the heavy lifting using bayesian optimization. 
    But there is more to it because an optimized gridsearch approach may still result in overtrained models.
    It is wise to carefully split your data into an independent evaluation set, a train, and test set, and then examine, by means of k-fold cross validation, the hyper-parameter space. 
    
Aim
    The aim of this library is to determine the most robust gradient boosting model model by evaluating on an independent validation set.
    The optimal set of parameters are determined by bayesian hyperoptimization using k-fold cross-validation approach on independent train/testsets.
    ``hgboost`` can be applied for classification tasks, such as two-class or multi-class, and regression tasks using xgboost, catboost or lightboost.

    The aim of ``hgboost`` is to determine the most robust model by efficiently searching across the parameter space using
    **hyperoptimization** for which the loss is evaluated using by means of a train/test-set with k-fold cross-validation.
    In addition, the final optimized model is evaluated on an independent validation set.
    
Results
    ``hgboost`` is a python package for hyperparameter optimization for xgboost, catboost and lightboost using cross-validation, and evaluating the results on an independent validation set.
    There are many implementations of gradient boosting, some efficiently uses the GPU, whereas others have specific interfaces.
    For this library ``hgboost``, we incorporated the *eXtreme Gradient Boosting* ``xgboost`` [2], *Light Gradient Boosting Machine* ``LightGBM`` [3],
    and *Category Gradient Boosting* ``catboost`` [4]. We also created the option to learn an ``ensemble`` model.

    
Schematic overview
'''''''''''''''''''

The schematic overview of our approach is as following:

.. _schematic_overview:

.. figure:: ../figs/schematic_overview.png


References
-----------
    * [1] http://hyperopt.github.io/hyperopt/
    * [2] https://github.com/dmlc/xgboost
    * [3] https://github.com/microsoft/LightGBM
    * [4] https://github.com/catboost/catboost


.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>
