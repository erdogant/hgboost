Installation
################


Create conda environment
*************************

It is recommended to install ``hgboost`` from an isolated Python environment. Using Conda this can be done as following:

.. code-block:: python

    conda create -n env_hgboost python=3.10
    conda activate env_hgboost

Pypi
*********************

Install via ``pip`` (recommended):

.. code-block:: console

    pip install hgboost


Github source
************************************

Install directly from github source (beta versions):

.. code-block:: console

    pip install git+https://github.com/erdogant/hgboost


Graphviz
************************************

Tree plots are created using the ``treeplot`` package which contains the required graphviz libraries.
In general, it should work out of the box for both Windows and Unix machines. However, in some cases it does require a manual installation of the graphviz package.
Binaries for graphviz can be downloaded from the graphviz project homepage, and the Python wrapper installed from pypi with pip install graphviz.

If you use the conda package manager, the graphviz binaries and the python package can be installed with conda install python-graphviz.

.. code-block:: console

    conda install python-graphviz

If you use the pip package manager, try the Python wrapper installed from Pypi.

.. code-block:: console

    pip install graphviz

An alternative example is to download and install this for Unix machines:

.. code-block:: console

    sudo apt install python-pydot python-pydot-ng graphviz
    # or 
    sudo apt install graphviz

For Mac OS install it as follows:

.. code-block:: console

    brew install graphviz

Quickstart
############

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
    

Uninstalling
################


Remove installation
**********************

Note that the removal of the environment will also remove the ``hgboost`` installation.

.. code-block:: console

    # Install from Pypi:
    pip uninstall hgboost


.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>
