.. _code_directive:

-------------------------------------

Quickstart
''''''''''

A quick example how to learn a model on a given dataset.


.. code:: python

    # Import library
    import hgboost

    # Retrieve URLs of malicous and normal urls:
    X, y = hgboost.load_example()

    # Learn model on the data
    model = hgboost.fit_transform(X, y, pos_label='bad')

    # Plot the model performance
    results = hgboost.plot(model)


Installation
''''''''''''

Create environment
------------------


If desired, install ``hgboost`` from an isolated Python environment using conda:

.. code-block:: python

    conda create -n env_hgboost python=3.6
    conda activate env_hgboost


Install via ``pip``:

.. code-block:: console

    # The installation from pypi is disabled:
    pip install hgboost

    # Install directly from github
    pip install git+https://github.com/erdogant/hgboost


