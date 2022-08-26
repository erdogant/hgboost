.. _code_directive:

-------------------------------------

Save and Load
''''''''''''''

Saving and loading models is desired as the learning proces of a model for ``hgboost`` can take up to hours.
In order to accomplish this, we created two functions: function :func:`hgboost.hgboost.hgboost.save` and function :func:`hgboost.hgboost.hgboost.load`
Below we illustrate how to save and load models.


Saving
----------------

Saving a learned model can be done using the function :func:`hgboost.hgboost.hgboost.save`:

.. code:: python

    from hgboost import hgboost

    Save model
    status = hgb.save(filepath='hgboost_model.pkl', overwrite=True)
    # [pypickle] Pickle file saved: [hgboost_model.pkl]
    # [hgboost] >Saving.. True


Loading
----------------------

Loading a learned model can be done using the function :func:`hgboost.hgboost.hgboost.load`:

.. code:: python

    from hgboost import hgboost

    # Load model
    model = hgb.load(filepath='hgboost_model.pkl')
    # [pypickle] Pickle file loaded: [hgboost_model.pkl]
    # [hgboost] >Loading succesful!

.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>
