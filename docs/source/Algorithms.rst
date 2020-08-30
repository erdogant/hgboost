.. _code_directive:

-------------------------------------

Algorithms
''''''''''''''''''''''''''

The ``hgboost`` method consists three methods, ``xgboost``, ``catboost``, ``lightboost``.


xgboost
---------

Gradient boosting is also known as gradient tree boosting, stochastic gradient boosting (an extension), and gradient boosting machines, or GBM for short.
Ensembles are constructed from decision tree models. Trees are added one at a time to the ensemble and fit to correct the prediction errors made by prior models.
This is a type of ensemble machine learning model referred to as boosting. Models are fit using any arbitrary differentiable loss function and gradient descent optimization algorithm.
This gives the technique its name, “gradient boosting,” as the loss gradient is minimized as the model is fit, much like a neural network [1].


catboost
-------------

CatBoost is a high-performance open source library for gradient boosting on decision trees.
It is developed by Yandex researchers and engineers, and is used for search,
recommendation systems, personal assistant, self-driving cars, weather prediction and many other tasks. It is in open-source and can be used by anyone.
It is desribed that *CatBoost* provides great results with default parameter and will therefore have reduce time spent on parameter tuning [2].
Another advantage that is described is that CatBoost allows you to use non-numeric factors, instead of having to pre-process your data or spend time and effort turning it to numbers [2].

Although it is described that default settings would be OK-ish, I'm not so sure about that. In addition, finding the optimzal parameters is no issues using ``hgboost``.


lightboost
--------------------------

LightGBM is a gradient boosting framework that uses tree based learning algorithms.
Many boosting tools use pre-sort-based algorithms for decision tree learning. It is a simple solution, but not easy to optimize.
LightGBM uses histogram-based algorithms, which bucket continuous feature (attribute) values into discrete bins.
This speeds up training and reduces memory usage. It is designed to be distributed and efficient, and is also described with many advantages, such as 
Fast training, high efficiency, Low memory usage, better accuracy, support of parallel and GPU learning, and capable of handling large-scale data [3].



**References**
    * [1] https://xgboost.ai/
    * [2] https://catboost.ai/
    * [3] https://lightgbm.readthedocs.io/
