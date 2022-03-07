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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://erdogant.github.io/hgboost/pages/html/Blog.html#colab-classification-notebook)
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
**⭐️ Star this repo if you like it ⭐️**
# 

### [Documentation pages](https://erdogant.github.io/hgboost/)

On the [documentation pages](https://erdogant.github.io/hgboost/) you can find detailed information about the working of the ``hgboost`` with many examples. 


## Colab Notebooks

* <a href="https://erdogant.github.io/hgboost/pages/html/Blog.html#colab-regression-notebook"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open regression example In Colab"/> </a> Regression example 

* <a href="https://erdogant.github.io/hgboost/pages/html/Blog.html#colab-classification-notebook"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open classification example In Colab"/> </a> Classification example 


### Schematic overview of hgboost

<p align="center">
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/schematic_overview.png" width="600" />
</p>


### Installation Environment

```python
conda create -n env_hgboost python=3.8
conda activate env_hgboost
```

### Install from pypi

```bash
pip install hgboost
pip install -U hgboost # Force update

```

#### Import hgboost package
```python
import hgboost as hgboost
```

#### Examples

* [Example: Fit catboost by hyperoptimization and cross-validation](https://erdogant.github.io/hgboost/pages/html/Examples.html#catboost)

#

* [Example: Fit lightboost by hyperoptimization and cross-validation](https://erdogant.github.io/hgboost/pages/html/Examples.html#lightboost)

#

* [Example: Fit xgboost by hyperoptimization and cross-validation](https://erdogant.github.io/hgboost/pages/html/Examples.html#xgboost-two-class)

#

* [Example: Plot searched parameter space](https://erdogant.github.io/hgboost/pages/html/Examples.html#plot-params)

<p align="left">
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/plot_params_clf_1.png" width="400" />
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/plot_params_clf_2.png" width="400" />
  </a>
</p>

#

* [Example: plot summary](https://erdogant.github.io/hgboost/pages/html/Examples.html#plot-summary)

<p align="left">
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/plot_clf.png" width="600" />
  </a>
</p>


#

* [Example: Tree plot](https://erdogant.github.io/hgboost/pages/html/Examples.html#treeplot)

<p align="left">
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/treeplot_clf_1.png" width="400" />
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/treeplot_clf_2.png" width="400" />
  </a>
</p>


#

* [Example: Plot the validation results](https://erdogant.github.io/hgboost/pages/html/Examples.html#plot-validation)

<p align="left">
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/plot_validation_clf_1.png" width="600" />
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/plot_validation_clf_2.png" width="400" />
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/plot_validation_clf_3.png" width="600" />
</p>

#

* [Example: Plot the cross-validation results](https://erdogant.github.io/hgboost/pages/html/Examples.html#plot-cv)

<p align="left">
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/plot_cv_clf.png" width="600" />
</p>


#

* [Example: use the learned model to make new predictions](https://erdogant.github.io/hgboost/pages/html/hgboost.hgboost.html?highlight=predict#hgboost.hgboost.hgboost.predict)

#

* [Example: Create ensemble model for Classification](https://erdogant.github.io/hgboost/pages/html/Examples.html#ensemble-classification)

#

* [Example: Create ensemble model for Regression](https://erdogant.github.io/hgboost/pages/html/Examples.html#ensemble-regression)

#

#### Classification example for xgboost, catboost and lightboost:
```python

# Load library
from hgboost import hgboost

# Initialization
hgb = hgboost(max_eval=10, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=42)

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

# Plot the ensemble classification validation results
hgb.plot_validation()

```

<p align="center">
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/plot_ensemble_clf_1.png" width="600" />
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/plot_ensemble_clf_2.png" width="400" />
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/plot_ensemble_clf_3.png" width="600" />
</p>


<hr>

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
