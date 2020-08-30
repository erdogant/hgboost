# hgboost

[![Python](https://img.shields.io/pypi/pyversions/hgboost)](https://img.shields.io/pypi/pyversions/hgboost)
[![PyPI Version](https://img.shields.io/pypi/v/hgboost)](https://pypi.org/project/hgboost/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/hgboost/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/hgboost/month)](https://pepy.tech/project/hgboost/month)

``hgboost`` is Python package to minimize a function from the model xgboost, catboost or lightboost over a hyperparameter space by using cross-validation and evaluting the results on an indepdendent validation set.
``hgboost`` can be applied for classification, such as two-class or multi-class, and regression tasks.

**Documentation**

* API Documentation: https://erdogant.github.io/hgboost/

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

# Load libray
from hgboost import hgboost

# Initizalization
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
# [hgboost] >Total datset: (891, 204) 
# [hgboost] >Hyperparameter optimization..
#  100% |----| 500/500 [04:39<05:21,  1.33s/trial, best loss: -0.8800619834710744]
# [hgboost] >Best peforming [xgb_clf] model: auc=0.881198
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
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/plot_validation_clf_2.png" width="600" />
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


#### Citation
Please cite hgboost in your publications if this is useful for your research. Here is an example BibTeX entry:
```BibTeX
@misc{erdogant2020hgboost,
  title={hgboost},
  author={Erdogan Taskesen},
  year={2019},
  howpublished={\url{https://github.com/erdogant/hgboost}},
}
```

#### References
* 
   
#### Maintainers
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)

#### Contribute
* Contributions are welcome.

#### Licence
See [LICENSE](LICENSE) for details.

#### Coffee
* This work is created and maintained in my free time. If you wish to buy me a <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Coffee</a> for this work, it is very appreciated.
