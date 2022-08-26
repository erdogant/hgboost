hgboost's documentation!
========================

The Hyperoptimized Gradient Boosting library (``hgboost``), is a Python package for hyperparameter optimization for **XGBoost**, **LightBoost**, and **CatBoost**. *HGBoost* will carefully split the dataset into a train, test, and an independent validation set. Within the train-test set there is the inner loop for optimizing the hyperparameters using Bayesian optimization (based on *Hyperopt*) and, the outer loop is to test how well the best-performing models can generalize using an external k-fold cross validation. This approach will select the most robust model with the highest performance.

``hgboost`` is fun because:

	* 1. It consists three of the most popular decision tree algorithms; XGBoost, LightBoost and Catboost.
	* 2. It consists the most popular hyperparameter optimization library for Bayesian Optimization; Hyperopt.
	* 3. An automated manner to split the data set into a train-test and independent validation to reliably determine the model performance.
	* 4. The pipeline has a nested scheme with an inner loop for hyperparameter optimization and an outer loop with k-fold crossvalidation to determine the most robust and best-performing model.
	* 5. It can handle both classification and regression tasks.
	* 6. It is easy to go wild and create a multi-class model or an ensemble of boosted decision tree models.
	* 7. It takes care of unbalanced datasets.
	* 8. It aims to create explainable results for the hyperparameter search-space, and model performance results by creating insightful plots.
	* 9. It is open-source.
	* 10. It is documented with many examples.

.. _schematic_overview:

.. figure:: ../figs/schematic_overview.png


Sponsor
=======
**This library is created and maintained in my free time**. I like to work on my open-source libraries, and you can help by becoming a sponsor! The easiest way is by simply following me on medium, and it will cost you nothing! Simply go to my `medium profile <https://erdogant.medium.com/>`_ and press "follow". Read more on my `sponsor github page <https://github.com/sponsors/erdogant/>`_ why this is important. This also gives you various other ways to sponsor me!


Star is important too!
======================
If you like this project, **star** this repo at the github page! This is important because only then I know how much you like it :)


Quick install
=============
.. code-block:: console

   pip install hgboost


Github
======
`Github hgboost <https://github.com/erdogant/hgboost/>`_.
Please report bugs, issues and feature extensions there.


Citing hgboost
==============
The bibtex can be found in the right side menu at the `github page <https://github.com/erdogant/hgboost/>`_.


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

  Blog
  Coding quality
  hgboost.hgboost



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>

