.. _code_directive:

-------------------------------------

Quickstart
''''''''''

A quick example how to learn a model on a given dataset.


.. code:: python

    # Import library
    import gridsearch

    # Retrieve URLs of malicous and normal urls:
    X, y = gridsearch.load_example()

    # Learn model on the data
    model = gridsearch.fit_transform(X, y, pos_label='bad')

    # Plot the model performance
    results = gridsearch.plot(model)


Installation
''''''''''''

Create environment
------------------


If desired, install ``gridsearch`` from an isolated Python environment using conda:

.. code-block:: python

    conda create -n env_gridsearch python=3.6
    conda activate env_gridsearch


Install via ``pip``:

.. code-block:: console

    # The installation from pypi is disabled:
    pip install gridsearch

    # Install directly from github
    pip install git+https://github.com/erdogant/gridsearch


