.. _code_directive:

-------------------------------------

Examples
''''''''''

Learn new model with hgboost and train-test set
--------------------------------------------------

AAA

.. code:: python

    # Import library
    import hgboost

    # Load example data set    
    X,y_true = hgboost.load_example()

    # Retrieve URLs of malicous and normal urls:
    model = hgboost.fit_transform(X, y_true, pos_label='bad', train_test=True, hgboost=True)

    # The test error will be shown
    results = hgboost.plot(model)


Learn new model on the entire data set
--------------------------------------------------

BBBB


.. code:: python

    # Import library
    import hgboost

    # Load example data set    
    X,y_true = hgboost.load_example()

    # Retrieve URLs of malicous and normal urls:
    model = hgboost.fit_transform(X, y_true, pos_label='bad', train_test=False, hgboost=True)

    # The train error will be shown. Such results are heavily biased as the model also learned on this set of data
    results = hgboost.plot(model)

