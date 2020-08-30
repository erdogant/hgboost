hgboost's documentation!
========================

``hgboost`` is python package to minimize the function for xgboost, catboost or lightboost over a hyper-parameter space by using cross-validation, and evaluating the results on an independent validation set. hgboost can be applied for classification and regression tasks. hgboost can be applied for classification and regression tasks.
``hgboost`` can be applied for classification and regression tasks.

``hgboost`` is fun because:

    * 1. Hyperoptimization of the Parameter-space using bayesian approach.
    * 2. Determines the best scoring model(s) using k-fold cross validation.
    * 3. Evaluates best model on independent evaluation set.
    * 4. Fit model on entire input-data using the best model.
    * 5. Return model, space and test/evaluation results.
    * 6. Makes insightful plots.


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
