from hgboost import hgboost
import pandas as pd
import numpy as np

def xgboost():
    ############################## CLASSIFICATION########################
    # Check whether all combinations of parameters runs like a charm
    #####################################################################
    from hgboost import hgboost
    X, y = get_data()

    # Set all parameters to be evaluated
    max_evals = [None, 10]
    cvs = [None, 5, 11]
    val_sizes = [None, 0.2]
    test_sizes = [None, 0.2]
    methods = ['xgb_clf', 'xgb_clf_multi']
    pos_labels = [None, 0, 2, 'value not in y']
    top_cv_evals = [None, 1, 20]
    thresholds = [None, 0.5]
    eval_metrics = [None]
    
    # Evaluate across all paramters
    out = run_over_all_input_parameters(X, y, max_evals, cvs, val_sizes, methods, pos_labels, test_sizes, top_cv_evals, thresholds, eval_metrics)

    
def catboost():
    ############################## CLASSIFICATION########################
    # Check whether all combinations of parameters runs like a charm
    #####################################################################
    from hgboost import hgboost
    X, y = get_data()

    # Set all parameters to be evaluated
    max_evals = [None, 10]
    cvs = [None, 5, 11]
    val_sizes = [None, 0.2]
    test_sizes = [None, 0.2]
    pos_labels = [None, 0, 2, 'value not in y']
    top_cv_evals = [None, 1, 20]
    thresholds = [None, 0.5]
    eval_metrics = [None]
    methods = ['ctb_clf']

    # Evaluate across all paramters
    out = run_over_all_input_parameters(X, y, max_evals, cvs, val_sizes, methods, pos_labels, test_sizes, top_cv_evals, thresholds, eval_metrics)

    
def lightboost():
    ############################## CLASSIFICATION########################
    # Check whether all combinations of parameters runs like a charm
    #####################################################################
    from hgboost import hgboost
    X, y = get_data()

    # Set all parameters to be evaluated
    max_evals = [None, 10]
    cvs = [None, 5, 11]
    val_sizes = [None, 0.2]
    test_sizes = [None, 0.2]
    pos_labels = [None, 0, 2, 'value not in y']
    top_cv_evals = [None, 1, 20]
    thresholds = [None, 0.5]
    eval_metrics = [None]
    methods = ['lgb_clf']

    # Evaluate across all paramters
    out = run_over_all_input_parameters(X, y, max_evals, cvs, val_sizes, methods, pos_labels, test_sizes, top_cv_evals, thresholds, eval_metrics)


def xgboost_reg():
    pass

def lightboost_reg():
    pass

def catboost_reg():
    pass

def run_over_all_input_parameters(X, y, max_evals, cvs, val_sizes, methods, pos_labels, test_sizes, top_cv_evals, thresholds, eval_metrics):
    nr_classes = len(np.unique(y))
    random_state = 42
    out = []
    count = 0
    for max_eval in max_evals:
        for cv in cvs:
            for val_size in val_sizes:
                for method in methods:
                    for pos_label in pos_labels:
                        for test_size in test_sizes:
                            for top_cv_eval in top_cv_evals:
                                for threshold in thresholds:
                                    for eval_metric in eval_metrics:
                                        try:
                                            status = 'OK'
                                            loss = np.nan
                                            # Setup model
                                            hgb = hgboost(max_eval=max_eval, threshold=threshold, cv=cv, test_size=test_size, val_size=val_size, top_cv_evals=top_cv_eval, random_state=random_state, verbose=2)
                                            # Fit model
                                            if np.any(np.isin(method,['xgb_clf', 'xgb_clf_multi'])):
                                                hgb.xgboost(X, y, method=method, pos_label=pos_label, eval_metric=eval_metric);
                                            elif method=='ctb_clf':
                                                hgb.catboost(X, y, pos_label=pos_label, eval_metric=eval_metric);
                                            elif method=='lgb_clf':
                                                hgb.lightboost(X, y, pos_label=pos_label, eval_metric=eval_metric);

                                            # use the predictor
                                            y_pred, y_proba = hgb.predict(X)
                                            # Loss score
                                            loss = hgb.results['summary']['loss'].iloc[np.where(hgb.results['summary']['best'])[0]].values
                                            # Make some plots
                                            # assert gs.plot_params(return_ax=True)
                                            # assert gs.plot(return_ax=True)
                                            # assert gs.treeplot(return_ax=True)
                                            # if (val_size is not None):
                                            #     ax = gs.plot_validation(return_ax=True)
                                            #     assert len(ax)>=2
                                        except ValueError as err:
                                            assert not 'hgboost' in err.args
                                            status = err.args
                                            print(err.args)

                                        tmpout = {'max_eval':max_eval,
                                                       'threshold':threshold,
                                                       'cv':cv,
                                                       'test_size':test_size,
                                                       'val_size':val_size,
                                                       'top_cv_evals':top_cv_eval,
                                                       'random_state':random_state,
                                                       'pos_label':pos_label,
                                                       'method':method,
                                                       'eval_metric':eval_metric,
                                                       'nr_classes':nr_classes,
                                                       'loss':loss,
                                                       'status':status,
                                                       }
                                        out.append(tmpout)
                                        count=count+1

    print('Fin! Total number of models evaluated with different paramters: %.0d' %(count))
    return(pd.DataFrame(out))

def get_data():
    from hgboost import hgboost
    gs = hgboost()
    df = gs.import_example()
    y = df['Parch'].values
    y[y>=3]=3
    del df['Parch']
    X = gs.preprocessing(df, verbose=0)
    return X, y

# %%
def test_fit():
    ############################## CLASSIFICATION########################
    # Check whether all combinations of parameters runs like a charm
    #####################################################################
    from hgboost import hgboost
    gs = hgboost(method='xgb_clf')
    df = gs.import_example()
    y = df['Parch'].values
    y[y>=3]=3
    del df['Parch']
    X = gs.preprocessing(df, verbose=0)

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
                        gs = hgboost(method=method, max_evals=max_eval, cv=cv, eval_metric=None, val_size=val_size, verbose=3)
                        # gs = hgboost(method=method, max_evals=10, cv=5, eval_metric='auc', val_size=0.2, verbose=3)
                        # Fit model
                        try:
                            results = gs.fit(X, y, pos_label=pos_label)
                            # use the predictor
                            y_pred, y_proba = gs.predict(X)
                            # Make some plots
                            # assert gs.plot_params(return_ax=True)
                            # assert gs.plot(return_ax=True)
                            # assert gs.treeplot(return_ax=True)
                            # if (val_size is not None):
                            #     ax = gs.plot_validation(return_ax=True)
                            #     assert len(ax)>=2
                        except ValueError as err:
                            assert not 'hgboost' in err.args
                            print(err.args)


    ############################## REGRESSION ###########################
    # Check whether all combinations of parameters runs like a charm
    #####################################################################
    from hgboost import hgboost
    gs = hgboost(method='xgb_reg')
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
    methods = ['xgb_reg','lgb_reg','ctb_reg']
    eval_metrics = ['rmse','mae']

    for max_eval in max_evals:
        for cv in cvs:
            for val_size in val_sizes:
                for method in methods:
                    for eval_metric in eval_metrics:
                        # Setup model
                        gs = hgboost(method=method, max_evals=max_eval, cv=cv, eval_metric=eval_metric, val_size=val_size, verbose=2)
                        # Fit model
                        try:
                            results = gs.fit(X, y, pos_label=pos_label)
                            # use the predictor
                            y_pred, y_proba = gs.predict(X)
                            # Make some plots
                            assert gs.plot_params(return_ax=True)
                            assert gs.plot(return_ax=True)
                            assert gs.treeplot(return_ax=True)
                            if val_size is not None:
                                assert gs.plot_validation(return_ax=True)
                        except ValueError as err:
                            assert not 'hgboost' in err.args
                            print(err.args)

    # from sklearn.datasets import make_classification
    # # define dataset
    # X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
    # # summarize the dataset
    # print(X.shape, y.shape)
    # X = pd.DataFrame(X)
