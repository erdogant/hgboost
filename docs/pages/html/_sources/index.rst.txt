hgboost's documentation!
========================

|python| |pypi| |docs| |stars| |LOC| |downloads_month| |downloads_total| |license| |forks| |open issues| |project status| |medium| |colab| |DOI| |repo-size| |donate|

.. include:: add_top.add

.. _schematic_overview:

.. figure:: ../figs/schematic_overview.png


.. tip::
	`Medium Blog 1: A Guide to Find the Best Boosting Model using Bayesian Hyperparameter Tuning but without Overfitting. <https://towardsdatascience.com/a-guide-to-find-the-best-boosting-model-using-bayesian-hyperparameter-tuning-but-without-c98b6a1ecac8>`_

.. tip::
	`Medium Blog 2: A Hands-on Guide To Create Explainable Gradient Boosting Classification models using Bayesian Hyperparameter Optimization. <https://erdogant.medium.com/hands-on-guide-for-hyperparameter-tuning-with-bayesian-optimization-for-classification-models-2002224bfa3d>`_

The Hyperoptimized Gradient Boosting library (``hgboost``), is a Python package for hyperparameter optimization for **XGBoost**, **LightBoost**, and **CatBoost**. *HGBoost* will carefully split the dataset into a train, test, and an independent validation set. Within the train-test set there is the inner loop for optimizing the hyperparameters using Bayesian optimization (based on *Hyperopt*) and, the outer loop is to test how well the best-performing models can generalize using an external k-fold cross validation. This approach will select the most robust model with the highest performance.

``hgboost`` is fun because:

	* 1. It contains the most popular decision trees; XGBoost, LightBoost and Catboost.
	* 2. It consists Bayesian hyperparameter optimization.
	* 3. It automates splitting the data set into a train-test and independent validation.
	* 4. It contains a nested scheme with an inner loop for hyperparameter optimization and an outer loop with crossvalidation to determine the best model.
	* 5. It handles both classification and regression tasks.
	* 6. It allows multi-class and ensemble of boosted decision tree models.
	* 7. It takes care of unbalanced datasets.
	* 8. It creates explainable results for the hyperparameter search-space, and model performance.
	* 9. It is open-source.
	* 10. It is documented with many examples.



-----------------------------------

.. note::
	**Your ❤️ is important to keep maintaining this package.** You can `support <https://erdogant.github.io/hgboost/pages/html/Documentation.html>`_ in various ways, have a look at the `sponser page <https://erdogant.github.io/hgboost/pages/html/Documentation.html>`_.
	Report bugs, issues and feature extensions at `github <https://github.com/erdogant/hgboost/>`_ page.

	.. code-block:: console

	   pip install hgboost

-----------------------------------



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
  :caption: Background

  Algorithms
  Cross validation and hyperparameter tuning


.. toctree::
  :maxdepth: 1
  :caption: Methods

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
  :caption: Documentation

  Documentation
  Coding quality
  hgboost.hgboost


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



.. |python| image:: https://img.shields.io/pypi/pyversions/hgboost.svg
    :alt: |Python
    :target: https://erdogant.github.io/hgboost/

.. |pypi| image:: https://img.shields.io/pypi/v/hgboost.svg
    :alt: |Python Version
    :target: https://pypi.org/project/hgboost/

.. |docs| image:: https://img.shields.io/badge/Sphinx-Docs-blue.svg
    :alt: Sphinx documentation
    :target: https://erdogant.github.io/hgboost/

.. |stars| image:: https://img.shields.io/github/stars/erdogant/hgboost
    :alt: Stars
    :target: https://img.shields.io/github/stars/erdogant/hgboost

.. |LOC| image:: https://sloc.xyz/github/erdogant/hgboost/?category=code
    :alt: lines of code
    :target: https://github.com/erdogant/hgboost

.. |downloads_month| image:: https://static.pepy.tech/personalized-badge/hgboost?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads/month
    :alt: Downloads per month
    :target: https://pepy.tech/project/hgboost

.. |downloads_total| image:: https://static.pepy.tech/personalized-badge/hgboost?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads
    :alt: Downloads in total
    :target: https://pepy.tech/project/hgboost

.. |license| image:: https://img.shields.io/badge/license-MIT-green.svg
    :alt: License
    :target: https://github.com/erdogant/hgboost/blob/master/LICENSE

.. |forks| image:: https://img.shields.io/github/forks/erdogant/hgboost.svg
    :alt: Github Forks
    :target: https://github.com/erdogant/hgboost/network

.. |open issues| image:: https://img.shields.io/github/issues/erdogant/hgboost.svg
    :alt: Open Issues
    :target: https://github.com/erdogant/hgboost/issues

.. |project status| image:: http://www.repostatus.org/badges/latest/active.svg
    :alt: Project Status
    :target: http://www.repostatus.org/#active

.. |medium| image:: https://img.shields.io/badge/Medium-Blog-green.svg
    :alt: Medium Blog
    :target: https://erdogant.github.io/hgboost/pages/html/Documentation.html#medium-blog

.. |donate| image:: https://img.shields.io/badge/Support%20this%20project-grey.svg?logo=github%20sponsors
    :alt: donate
    :target: https://erdogant.github.io/hgboost/pages/html/Documentation.html#

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Colab example
    :target: https://erdogant.github.io/hgboost/pages/html/Documentation.html#colab-notebook

.. |DOI| image:: https://zenodo.org/badge/257025146.svg
    :alt: Cite
    :target: https://zenodo.org/badge/latestdoi/257025146

.. |repo-size| image:: https://img.shields.io/github/repo-size/erdogant/hgboost
    :alt: repo-size
    :target: https://img.shields.io/github/repo-size/erdogant/hgboost

.. include:: add_bottom.add
