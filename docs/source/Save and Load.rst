.. _code_directive:

-------------------------------------

Save and Load
''''''''''''''

Saving and loading models is desired as the learning proces of a model for ``gridsearch`` can take up to hours.
In order to accomplish this, we created two functions: function :func:`gridsearch.save` and function :func:`gridsearch.load`
Below we illustrate how to save and load models.


Saving
----------------

Saving a learned model can be done using the function :func:`gridsearch.save`:

.. code:: python

    import gridsearch

    # Load example data
    X,y_true = gridsearch.load_example()

    # Learn model
    model = gridsearch.fit_transform(X, y_true, pos_label='bad')

    Save model
    status = gridsearch.save(model, 'learned_model_v1')



Loading
----------------------

Loading a learned model can be done using the function :func:`gridsearch.load`:

.. code:: python

    import gridsearch

    # Load model
    model = gridsearch.load(model, 'learned_model_v1')
