.. _code_directive:

-------------------------------------

Quickstart
''''''''''

A quick example how to learn a model on a given dataset.


.. code:: python

    # Import library
    from hgboost import hgboost
    
    # Initialize with default settings
    hgb = hgboost()

    # Find best model on the data
    results = hgb.xgboost(X, y, pos_label)

    # Plot
    ax = hgb.plot()


Installation
''''''''''''

Create environment
------------------


It is recommended to install ``hgboost`` from an isolated Python environment. Using Conda this can be done as following:

.. code-block:: python

    conda create -n env_hgboost python=3.6
    conda activate env_hgboost



Install directly from github source:

.. code-block:: console

    pip install git+https://github.com/erdogant/hgboost


Install via ``pip``:

.. code-block:: console

    # The installation from pypi is disabled:
    pip install hgboost
