# hgboost

[![Python](https://img.shields.io/pypi/pyversions/hgboost)](https://img.shields.io/pypi/pyversions/hgboost)
[![PyPI Version](https://img.shields.io/pypi/v/hgboost)](https://pypi.org/project/hgboost/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/hgboost/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/hgboost/week)](https://pepy.tech/project/hgboost/week)
[![Donate](https://img.shields.io/badge/donate-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)

* hgboost is Python package

### Contents
- [Installation](#-installation)
- [Requirements](#-Requirements)
- [Contribute](#-contribute)
- [Citation](#-citation)
- [Maintainers](#-maintainers)
- [License](#-copyright)

### Installation
* Install hgboost from PyPI (recommended). hgboost is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
* A new environment is created as following: 

```python
conda create -n env_hgboost python=3.6
conda activate env_hgboost
pip install -r requirements
```

```bash
pip install hgboost
```

* Alternatively, install hgboost from the GitHub source:
```bash
# Directly install from github source
pip install -e git://github.com/erdogant/hgboost.git@0.1.0#egg=master
pip install git+https://github.com/erdogant/hgboost#egg=master

# By cloning
pip install git+https://github.com/erdogant/hgboost
git clone https://github.com/erdogant/hgboost.git
cd hgboost
python setup.py install
```  

#### Import hgboost package
```python
import hgboost as hgboost
```

#### Example:
```python
df = pd.read_csv('https://github.com/erdogant/hnet/blob/master/hgboost/data/example_data.csv')
model = hgboost.fit(df)
G = hgboost.plot(model)
```
<p align="center">
  <img src="https://github.com/erdogant/hgboost/blob/master/docs/figs/fig1.png" width="600" />
  
</p>


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
