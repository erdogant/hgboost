from gridsearch.gridsearch import gridsearch

from gridsearch.gridsearch import (
    import_example,
    )

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.1.0'

# module level doc-string
__doc__ = """
gridsearch - Determine best model by minimizing xgboost function over a hyperparameter space.
=====================================================================

Description
-----------
Determine best model by minimizing xgboost function over a hyperparameter space.
Explore a function over a hyperparameter space according to a given algorithm,
allowing up to a certain number of function evaluations.
The library consists the underneath parts that can be used set:
    * regression
    * binary classification
    * multiclass classification
    * cross-validation
    * Independent validation dataset
    * hyperparameter searching
    * feature importance
    * early stopping
    * plotting
        - parameter space
        - validation set
        - summary
        - tree

Example
-------
>>> import gridsearch as gridsearch
>>> model = gridsearch.fit(X)
>>> fig,ax = gridsearch.plot(model)

References
----------
* https://github.com/erdogant/gridsearch

"""
