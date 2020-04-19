# comming up
# --------------------------------------------------
# Name        : gridsearch.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/gridsearch
# Licence     : MIT
# --------------------------------------------------

import os
import pandas as pd
import wget

# %% Import example dataset from github.
def download_example(url='https://erdogant.github.io/datasets/titanic_train.zip', verbose=3):
    """Import example dataset from github.

    Parameters
    ----------
    url : str, optional
        url-Link to dataset. The default is 'https://erdogant.github.io/datasets/titanic_train.zip'.
    verbose : int, optional
        Print message to screen. The default is 3.

    Returns
    -------
    tuple containing dataset and response variable (X,y).

    """
    import wget
    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    PATH_TO_DATA = os.path.join(curpath, wget.filename_from_url(url))

    # Check file exists.
    if not os.path.isfile(PATH_TO_DATA):
        if verbose>=3: print('[classeval] Downloading example dataset..')
        wget.download(url, curpath)

    # Import local dataset
    if verbose>=3: print('[classeval] Import dataset..')
    df = pd.read_csv(PATH_TO_DATA)

    # Get data
    # y = df['Survived'].values
    # X = df.drop(labels=['Survived'], axis=1)
    # Return
    return df


# %% Import example dataset from github.
def load_example(data='breast'):
    """Import example dataset from sklearn.

    Parameters
    ----------
    'breast' : str, two-class
    'titanic': str, two-class
    'iris' : str, multi-class

    Returns
    -------
    tuple containing dataset and response variable (X,y).

    """
    try:
        from sklearn import datasets
    except:
        print('This requires: <pip install sklearn>')
        return None, None
    
    if data=='iris':
        X, y = datasets.load_iris(return_X_y=True)
    elif data=='breast':
        X, y = datasets.load_breast_cancer(return_X_y=True)
    elif data=='titanic':
        X, y = datasets.fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

    return X, y


# %% Main
if __name__ == "__main__":
    import gridsearch as gridsearch
    df = gridsearch.import_example()
    out = gridsearch.fit(df)
    fig,ax = gridsearch.plot(out)