hgboost's documentation!
========================

``hgboost`` is short for Hyperoptimized Gradient Boosting and is a python package for hyperparameter optimization for xgboost, catboost and lightboost using cross-validation, and evaluating the results on an independent validation set.
``hgboost`` can be applied for classification and regression tasks.

``hgboost`` is fun because:

    * 1. Hyperoptimization of the Parameter-space using bayesian approach.
    * 2. Determines the best scoring model(s) using k-fold cross validation.
    * 3. Evaluates best model on independent evaluation set.
    * 4. Fit model on entire input-data using the best model.
    * 5. Works for classification and regression
    * 6. Creating a super-hyperoptimized model by an ensemble of all individual optimized models.
    * 7. Return model, space and test/evaluation results.
    * 8. Makes insightful plots.


Content
=======

.. toctree::
   :maxdepth: 1
   :caption: Background
   
   Abstract


.. toctree::
   :maxdepth: 1
   :caption: Installation
   
   Installation


.. toctree::
  :maxdepth: 1
  :caption: Methods

  Algorithms
  Cross validation and hyperparameter tuning
  Classification
  Regression
  Performance
  Save and Load


.. toctree::
  :maxdepth: 1
  :caption: Examples

  Examples


.. toctree::
  :maxdepth: 1
  :caption: Code Documentation
  
  Coding quality
  hgboost.hgboost



Quick install
-------------

.. code-block:: console

   pip install hgboost




Source code and issue tracker
------------------------------

Available on Github, `erdogant/hgboost <https://github.com/erdogant/hgboost/>`_.
Please report bugs, issues and feature extensions there.

Colab notebooks
------------------------------

Some of the described examples can also be found in the notebooks:
    * See `classification Colab notebook`_. 
    * See `regression Colab notebook`_. 

.. _classification Colab notebook: https://colab.research.google.com/github/erdogant/hgboost/blob/master/notebooks/hgboost_classification_examples.ipynb

.. _regression Colab notebook: https://colab.research.google.com/github/erdogant/hgboost/blob/master/notebooks/hgboost_regression_examples.ipynb


Citing *hgboost*
----------------
Here is an example BibTeX entry:

@misc{erdogant2020hgboost,
  title={hgboost},
  author={Erdogan Taskesen},
  year={2020},
  howpublished={\url{https://github.com/erdogant/hgboost}}}



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
