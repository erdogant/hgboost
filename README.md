# hgboost - Hyperoptimized Gradient Boosting

[![Python](https://img.shields.io/pypi/pyversions/hgboost)](https://img.shields.io/pypi/pyversions/hgboost)
[![PyPI Version](https://img.shields.io/pypi/v/hgboost)](https://pypi.org/project/hgboost/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/hgboost/blob/master/LICENSE)
[![Github Forks](https://img.shields.io/github/forks/erdogant/hgboost.svg)](https://github.com/erdogant/hgboost/network)
[![GitHub Open Issues](https://img.shields.io/github/issues/erdogant/hgboost.svg)](https://github.com/erdogant/hgboost/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Downloads](https://pepy.tech/badge/hgboost/month)](https://pepy.tech/project/hgboost/month)
[![Downloads](https://pepy.tech/badge/hgboost)](https://pepy.tech/project/hgboost)
[![DOI](https://zenodo.org/badge/257025146.svg)](https://zenodo.org/badge/latestdoi/257025146)
[![Sphinx](https://img.shields.io/badge/Sphinx-Docs-Green)](https://erdogant.github.io/hgboost/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erdogant/hgboost/blob/master/notebooks/hgboost_classification_examples.ipynb)
<!---[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)-->
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->


``hgboost`` is short for **Hyperoptimized Gradient Boosting** and is a python package for hyperparameter optimization for *xgboost*, *catboost* and *lightboost* using cross-validation, and evaluating the results on an independent validation set.
``hgboost`` can be applied for classification and regression tasks.

``hgboost`` is fun because:

    * 1. Hyperoptimization of the Parameter-space using bayesian approach.
    * 2. Determines the best scoring model(s) using k-fold cross validation.
    * 3. Evaluates best model on independent evaluation set.
    * 4. Fit model on entire input-data using the best model.
    * 5. Works for classification and regression
    * 6. Creating a super-hyperoptimized model by an ensemble of all individual optimized models.
    * 7. Return model, space and test/evaluation results.
    * 8. Makes insightful plots.

# 
**Star this repo if you like it! ⭐️**
#


## Documentation/Notebooks

* [**hgboost documentation pages (Sphinx)**](https://erdogant.github.io/hgboost/)
* Regression example <a href="https://colab.research.google.com/github/erdogant/hgboost/blob/master/notebooks/hgboost_regression_examples.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open regression example In Colab"/> </a>
* Classification example <a href="https://colab.research.google.com/github/erdogant/hgboost/blob/master/notebooks/hgboost_classification_examples.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open classification example In Colab"/> </a>


### Schematic overview of hgboost

<p align="center">
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/schematic_overview.png" width="600" />
</p>


### Installation Environment
* Install hgboost from PyPI (recommended). hgboost is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
* A new environment is recommended and created as following: 

```python
conda create -n env_hgboost python=3.6
conda activate env_hgboost
```

### Install newest version hgboost from pypi

```bash
pip install hgboost

```

**Force to install latest version**

```bash
pip install -U hgboost
```

### Install from github-source

```bash
pip install git+https://github.com/erdogant/hgboost#egg=master
```  

#### Import hgboost package
```python
import hgboost as hgboost
```

#### Classification example for xgboost, catboost and lightboost:

```python

# Load library
from hgboost import hgboost

# Initialization
hgb = hgboost(max_eval=10, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=42)

```

```python

# Import data
df = hgb.import_example()
y = df['Survived'].values
y = y.astype(str)
y[y=='1']='survived'
y[y=='0']='dead'

# Preprocessing by encoding variables
del df['Survived']
X = hgb.preprocessing(df)

```

```python
# Fit catboost by hyperoptimization and cross-validation
results = hgb.catboost(X, y, pos_label='survived')

# Fit lightboost by hyperoptimization and cross-validation
results = hgb.lightboost(X, y, pos_label='survived')

# Fit xgboost by hyperoptimization and cross-validation
results = hgb.xgboost(X, y, pos_label='survived')

# [hgboost] >Start hgboost classification..
# [hgboost] >Collecting xgb_clf parameters.
# [hgboost] >Number of variables in search space is [11], loss function: [auc].
# [hgboost] >method: xgb_clf
# [hgboost] >eval_metric: auc
# [hgboost] >greater_is_better: True
# [hgboost] >pos_label: True
# [hgboost] >Total dataset: (891, 204) 
# [hgboost] >Hyperparameter optimization..
#  100% |----| 500/500 [04:39<05:21,  1.33s/trial, best loss: -0.8800619834710744]
# [hgboost] >Best performing [xgb_clf] model: auc=0.881198
# [hgboost] >5-fold cross validation for the top 10 scoring models, Total nr. tests: 50
# 100%|██████████| 10/10 [00:42<00:00,  4.27s/it]
# [hgboost] >Evalute best [xgb_clf] model on independent validation dataset (179 samples, 20.00%).
# [hgboost] >[auc] on independent validation dataset: -0.832
# [hgboost] >Retrain [xgb_clf] on the entire dataset with the optimal parameters settings.

```


```python

# Plot searched parameter space 
hgb.plot_params()

```

<p align="center">
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/plot_params_clf_1.png" width="600" />
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/plot_params_clf_2.png" width="600" />
</p>


```python

# Plot summary results
hgb.plot()

```

<p align="center">
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/plot_clf.png" width="600" />
</p>


```python

# Plot the best tree
hgb.treeplot()

```

<p align="center">
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/treeplot_clf_1.png" width="600" />
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/treeplot_clf_2.png" width="600" />
</p>


```python

# Plot the validation results
hgb.plot_validation()

```

<p align="center">
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/plot_validation_clf_1.png" width="600" />
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/plot_validation_clf_2.png" width="400" />
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/plot_validation_clf_3.png" width="600" />
</p>


```python

# Plot the cross-validation results
hgb.plot_cv()

```

<p align="center">
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/plot_cv_clf.png" width="600" />
</p>


```python

# use the learned model to make new predictions.
y_pred, y_proba = hgb.predict(X)

```

### Create ensemble model for Classification

```python

from hgboost import hgboost

hgb = hgboost(max_eval=100, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None, verbose=3)

# Import data
df = hgb.import_example()
y = df['Survived'].values
del df['Survived']
X = hgb.preprocessing(df, verbose=0)

results = hgb.ensemble(X, y, pos_label=1)

# use the predictor
y_pred, y_proba = hgb.predict(X)

```

### Create ensemble model for Regression

```python

from hgboost import hgboost

hgb = hgboost(max_eval=100, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None, verbose=3)

# Import data
df = hgb.import_example()
y = df['Age'].values
del df['Age']
I = ~np.isnan(y)
X = hgb.preprocessing(df, verbose=0)
X = X.loc[I,:]
y = y[I]

results = hgb.ensemble(X, y, methods=['xgb_reg','ctb_reg','lgb_reg'])

# use the predictor
y_pred, y_proba = hgb.predict(X)

```

```python

# Plot the ensemble classification validation results
hgb.plot_validation()

```

<p align="center">
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/plot_ensemble_clf_1.png" width="600" />
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/plot_ensemble_clf_2.png" width="400" />
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/plot_ensemble_clf_3.png" width="600" />
</p>



**References**

    * http://hyperopt.github.io/hyperopt/
    * https://github.com/dmlc/xgboost
    * https://github.com/microsoft/LightGBM
    * https://github.com/catboost/catboost
    
**Maintainers**
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)

**Contribute**
* Contributions are welcome.

**Licence**
See [LICENSE](LICENSE) for details.

**Coffee**
* If you wish to buy me a <a href="https://www.buymeacoffee.com/erdogant">Coffee</a> for this work, it is very appreciated :)
