.. _code_directive:

-------------------------------------

Save and Load
''''''''''''''

Saving and loading models is desired as the learning proces of a model for ``hgboost`` can take up to hours.
In order to accomplish this, we created two functions: function :func:`hgboost.save` and function :func:`hgboost.load`
Below we illustrate how to save and load models.


Saving
----------------

Saving a learned model can be done using the function :func:`hgboost.save`:

.. code:: python

    import hgboost

    # Load example data
    X,y_true = hgboost.load_example()

    # Learn model
    model = hgboost.fit_transform(X, y_true, pos_label='bad')

    Save model
    status = hgboost.save(model, 'learned_model_v1')



Loading
----------------------

Loading a learned model can be done using the function :func:`hgboost.load`:

.. code:: python

    import hgboost

    # Load model
    model = hgboost.load(model, 'learned_model_v1')

.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>
