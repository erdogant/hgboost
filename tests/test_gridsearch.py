from gridsearch import gridsearch
import pandas as pd
import numpy as np

def test_fit():
    # CHECK WHETHER CLF AND CLF_MULTI RUNS LIKE A CHARM
    from gridsearch import gridsearch
    gs = gridsearch(method='xgb_clf')
    # df = gs.import_example()
    # y = df['Survived'].values
    # del df['Survived']
    # X = gs.preprocessing(df, verbose=0)
    # y = y.astype(str)
    # y[y=='1']='survived'
    # y[y=='0']='dead'
    df = gs.import_example()
    y = df['Parch'].values
    y[y>=3]=3
    del df['Parch']
    X = gs.preprocessing(df, verbose=0)

    # from sklearn.datasets import make_classification
    # # define dataset
    # X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
    # # summarize the dataset
    # print(X.shape, y.shape)
    # X = pd.DataFrame(X)
    

    max_evals = [None, 10, 25]
    cvs = [None, 1, 5]
    val_sizes = [None, 0.2]
    methods = ['xgb_clf','xgb_clf_multi']
    pos_labels = [None, 0, 2, 'value not in y']

    for max_eval in max_evals:
        for cv in cvs:
            for val_size in val_sizes:
                for method in methods:
                    for pos_label in pos_labels:
                        # Setup model
                        gs = gridsearch(method=method, max_evals=max_eval, cv=cv, eval_metric=None, val_size=val_size, verbose=2)
                        # Fit model
                        try:
                            results = gs.fit(X, y, pos_label=pos_label)
                            # use the predictor
                            y_pred, y_proba = gs.predict(X)
                            # Make some plots
                            assert gs.plot_params(return_ax=True)
                            assert gs.plot_summary(return_ax=True)
                            assert gs.treeplot(return_ax=True)
                            if (val_size is not None) and (method is "xgb_clf"):
                                ax = gs.plot_validation(return_ax=True)
                                assert len(ax)>=2
                        except ValueError as err:
                            assert not 'gridsearch' in err.args
                            print(err.args)

    # REGRESSION
    df = gs.import_example()
    y = df['Age'].values
    del df['Age']
    I = ~np.isnan(y)
    X = gs.preprocessing(df, verbose=0)
    y = y[I]
    X = X.loc[I,:]

    max_evals = [None, 10, 25]
    cvs = [None, 1, 5]
    val_sizes = [None, 0.2]
    methods = ['xgb_reg','lgb_reg']
    # methods = ['xgb_reg','lgb_reg','ctb_reg']
    pos_labels = [None, 0, 2, 'value not in y']

    for max_eval in max_evals:
        for cv in cvs:
            for val_size in val_sizes:
                for method in methods:
                    for pos_label in pos_labels:
                        # Setup model
                        gs = gridsearch(method=method, max_evals=max_eval, cv=cv, eval_metric=None, val_size=val_size, verbose=2)
                        # Fit model
                        try:
                            results = gs.fit(X, y, pos_label=pos_label)
                            # use the predictor
                            y_pred, y_proba = gs.predict(X)
                            # Make some plots
                            assert gs.plot_params(return_ax=True)
                            assert gs.plot(return_ax=True)
                            assert gs.treeplot(return_ax=True)
                            assert gs.plot_validation(return_ax=True)
                        except ValueError as err:
                            assert not 'gridsearch' in err.args
                            print(err.args)
