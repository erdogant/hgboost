# gridsearch

[![Python](https://img.shields.io/pypi/pyversions/gridsearch)](https://img.shields.io/pypi/pyversions/gridsearch)
[![PyPI Version](https://img.shields.io/pypi/v/gridsearch)](https://pypi.org/project/gridsearch/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/gridsearch/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/gridsearch/week)](https://pepy.tech/project/gridsearch/week)
[![Donate](https://img.shields.io/badge/donate-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)

* gridsearch is Python package

### Contents
- [Installation](#-installation)
- [Requirements](#-Requirements)
- [Contribute](#-contribute)
- [Citation](#-citation)
- [Maintainers](#-maintainers)
- [License](#-copyright)

### Installation
* Install gridsearch from PyPI (recommended). gridsearch is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
* A new environment is created as following: 

```python
conda create -n env_gridsearch python=3.6
conda activate env_gridsearch
pip install -r requirements
```

```bash
pip install gridsearch
```

* Alternatively, install gridsearch from the GitHub source:
```bash
# Directly install from github source
pip install -e git://github.com/erdogant/gridsearch.git@0.1.0#egg=master
pip install git+https://github.com/erdogant/gridsearch#egg=master

# By cloning
pip install git+https://github.com/erdogant/gridsearch
git clone https://github.com/erdogant/gridsearch.git
cd gridsearch
python setup.py install
```  

#### Import gridsearch package
```python
import gridsearch as gridsearch
```

#### Example:
```python
df = pd.read_csv('https://github.com/erdogant/hnet/blob/master/gridsearch/data/example_data.csv')
model = gridsearch.fit(df)
G = gridsearch.plot(model)
```
<p align="center">
  <img src="https://github.com/erdogant/gridsearch/blob/master/docs/figs/fig1.png" width="600" />
  
</p>


#### Citation
Please cite gridsearch in your publications if this is useful for your research. Here is an example BibTeX entry:
```BibTeX
@misc{erdogant2020gridsearch,
  title={gridsearch},
  author={Erdogan Taskesen},
  year={2019},
  howpublished={\url{https://github.com/erdogant/gridsearch}},
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
