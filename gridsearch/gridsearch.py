# --------------------------------------------------
# Name        : gridsearch.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/gridsearch
# Licence     : See licences
# --------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import wget

from sklearn.metrics import mean_squared_error, cohen_kappa_score
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials, hp

import classeval as cle
from df2onehot import df2onehot
import treeplot as tree
from tqdm import tqdm


# %%
class gridsearch():
    """Create a class gridsearch that is instantiated with the desired method."""

    def __init__(self, method, max_evals=25, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_max_evals=None, eval_metric=None, greater_is_better=None, random_state=None, verbose=3):
        """Initialize gridsearch with user-defined parameters.

        Parameters
        ----------
        method : String,
            Classifier.
            * 'xgb_clf': XGboost Classifier (two-class or multi-class classifier is automatically choosen based on the number of classes)
            * 'xgb_reg': XGboost regressor
            * 'lgb_reg': LGBM Regressor
            * 'ctb_reg': CatBoost Regressor
        max_evals : int, (default : 100)
            Search space is created on the number of evaluations.
        threshold : float, (default : 0.5)
            Classification threshold. In case of two-class model this is 0.5
        eval_metric : str, (default : None)
            Evaluation metric for the regressor of classification model.
            * 'auc' : classification (default)
            * 'rmse' : regression  (default)
        greater_is_better : bool, (default : depending on method: _clf or _reg)
            If a loss, the output of the python function is negated by the scorer object, conforming to the cross validation convention that scorers return higher values for better models.
            * clf (default: True)
            * reg (default: False)
        test_size : float, (default : 0.2)
            Splitting train/test set with test_size=0.2 and train = 1-test_size.

        Returns
        -------
        None.
        
        References
        ----------
        * https://github.com/hyperopt/hyperopt
        * https://www.districtdatalabs.com/parameter-tuning-with-hyperopt
        * https://scikit-learn.org/stable/modules/model_evaluation.html

        """
        if (method is None): raise Exception('[gridsearch] >Set the method type.')
        # Check the eval_metric
        if (eval_metric is None) and ('_reg' in method):
            eval_metric = 'rmse'
        elif (eval_metric is None) and ('_clf' in method):
            eval_metric = 'auc'
        # Check the greater_is_better for evaluation metric
        if (greater_is_better is None) and ('_reg' in method):
            greater_is_better = False
        elif (greater_is_better is None) and ('_clf' in method):
            greater_is_better = True
        if top_max_evals is None: top_max_evals=max_evals

        self.method=method
        self.eval_metric=eval_metric
        self.greater_is_better=greater_is_better
        self.max_evals=max_evals
        self.top_max_evals=top_max_evals
        self.threshold = threshold
        self.test_size=test_size
        self.val_size=val_size
        self.algo=tpe.suggest
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, pos_label=None, verbose=None):
        """Learn best performing model.

        Description
        -----------
        Minimize a function over a hyperparameter space.
        More realistically: *explore* a function over a hyperparameter space
        according to a given algorithm, allowing up to a certain number of
        function evaluations.  As points are explored, they are accumulated in
        "trials".

        Parameters
        ----------
        X : pd.DataFrame
            Input dataset.
        y : array-like
            Response variable.
        pos_label : string/int.
            In case of classification, the model will be fitted on pos_label in y.
        verbose : int, (default : 3)
            Print progress to screen.
            0: None, 1: ERROR, 2: WARN, 3: INFO, 4: DEBUG, 5: TRACE

        Returns
        -------
        results : dict
            * best_params: Best performing parameters.
            * summary: Summary of the models with the loss and other variables.
            * trials: All models.
            * model: Best performing model.
            * status: ok if model was done correctly.
            * exception: In case of error.

        """
        if verbose is None: verbose = self.verbose
        # Check input parameters
        self.pos_label, self.method = _check_input(X, y, pos_label, self.method, verbose=self.verbose)
        # Set validation set
        self.set_validation_set(X, y)

        # Fit model
        # if self.cv is not None:
        #     scores = []
        #     models = []
        #     for p in np.arange(0, self.cv):
        #         # Split train-test set
        #         self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=True, stratify=self.y)
        #         # Find best parameters
        #         model, results = self.HPOpt(verbose=self.verbose)
        #         scores.append(results['summary'])
        #         models.append(model)
        # else:
        # Split train-test set
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=True, stratify=self.y)
        # Find best parameters
        self.model, self.results = self.HPOpt(verbose=self.verbose)
        

        # Fit on all data using best parameters
        if self.verbose>=3: print('[gridsearch] >Refit %s on the entire dataset with the optimal parameters settings.' %(self.method))
        self.model.fit(X, y)
        # Return
        return self.results

    def set_validation_set(self, X, y):
        # from sklearn.model_selection import cross_val_score, KFold
        if self.verbose>=3: print('[gridsearch] >Total datset: %s ' %(str(X.shape)))

        # Make split for validation set
        if self.val_size is not None:
            skf = StratifiedKFold(n_splits=int(1 / self.val_size), random_state=self.random_state, shuffle=True)
            for train_index, val_index in skf.split(X, y): pass
            if len(np.unique(np.append(train_index, val_index)))!=X.shape[0]: raise Exception('[gridsearch] >Error: Split for validation set not correct.')
            if self.verbose>=3: print('[gridsearch] >Validation datset: %s samples.' %(str(len(val_index))))
            self.X_val = X.iloc[val_index, :]
            self.y_val = y[val_index]
            self.X = X.iloc[train_index, :]
            self.y = y[train_index]
        else:
            self.X = X
            self.y = y
            self.X_val = None
            self.y_val = None

        # # Split validation-set
        # self.X, self.x_val, self.y, self.y_val = train_test_split(X.values, y, test_size=self.val_size)
        # if self.verbose>=3: print('[gridsearch] >Validation set: %s ' %(str(self.x_val.shape)))
        # if self.verbose>=3: print('[gridsearch] >Remining dataset: %s ' %(str(self.X.shape)))
        # # Split train-test set
        # # self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X.values, y, test_size=self.test_size)
        
    def HPOpt(self, verbose=3):
        """Hyperoptimization of the search space.

        Description
        -----------
        Minimize a function over a hyperparameter space.
        More realistically: *explore* a function over a hyperparameter space
        according to a given algorithm, allowing up to a certain number of
        function evaluations.  As points are explored, they are accumulated in
        "trials".

        Parameters
        ----------
        verbose : int, (default: 3)
            Print progress to screen.
            0: NONE, 1: ERROR, 2: WARNING, 3: INFO, 4: DEBUG, 5: TRACE

        Returns
        -------
        model : object
            Fitted model.
        results : dict
            * best_params: Best performing parameters.
            * summary: Summary of the models with the loss and other variables.
            * trials: All models.
            * model: Best performing model.
            * status: ok if model was done correctly.
            * exception: In case of error.

        """
        exception = None
        # Import the function that is to be used
        fn = getattr(self, self.method)
        # Import search space for the specific function
        space = _get_params(self.method, eval_metric=self.eval_metric)

        # Split train-test set
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=True, stratify=self.y)
        # Hyperoptimization to find best performing model
        trials=Trials()
        best_params = fmin(fn=fn, space=space, algo=self.algo, max_evals=self.max_evals, trials=trials, show_progressbar=True)
        # Summary results
        results_summary, model = self.to_df(trials, verbose=self.verbose)
        status = 'ok'
        
        # Cross-validation
        if self.cv is not None:
            ascending = False if self.greater_is_better else True
            results_summary['loss_mean'] = np.nan
            results_summary['loss_std'] = np.nan
            # Gather top n best results
            # if self.greater_is_better:
            #     ascending = False
            # else:
            #     ascending = True

            top_max_evals = np.minimum(results_summary.shape[0], self.top_max_evals)
            idx = results_summary['loss'].sort_values(ascending=ascending).index[0:top_max_evals]
            if verbose>=3: print('[gridsearch] >%.0d-fold cross validation for the top %.0d models.' %(self.cv, len(idx)))

            # Run over the top models that are sorted from best first.
            for i in tqdm(idx):
                scores = []
                # Run over the cross-validations
                for k in np.arange(0, self.cv):
                    # Split train-test set
                    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=True, stratify=self.y)
                    # Evaluate model
                    score = self._train_clf(results_summary['model'].iloc[i], space)
                    score.pop('model')
                    scores.append(score)

                # Store mean and std summary
                results_summary['loss_mean'].iloc[i] = pd.DataFrame(scores)['loss'].mean()
                results_summary['loss_std'].iloc[i] = pd.DataFrame(scores)['loss'].std()

            # Retrieve best model
            if self.greater_is_better:
                results_summary['loss_mean'] = results_summary['loss_mean'] * -1
                idx_best = results_summary['loss_mean'].argmax()
            else:
                idx_best= results_summary['loss_mean'].argmin()

            model = results_summary['model'].iloc[idx_best]

                # mean_scores = dict(pd.DataFrame(scores).mean(axis=0))
                # colnames = list(map(lambda x: x + '_std', [*scores[0]]))
                # std_scores = dict(zip(colnames, pd.DataFrame(scores).std(axis=0).values))
                # out = {}
                # out.update(mean_scores)
                # out.update(std_scores)
                # out['model'] = model

        # except Exception as e:
        #     if verbose>=1: print('[gridsearch] >Error %s' %(e))
        #     status = STATUS_FAIL
        #     exception = str(e)
        #     best_params = None
        #     results_summary = None
        #     model = None
        #     status = None

        # Remove the model column
        del results_summary['model']
        # Store
        results = {}
        results['params'] = best_params
        results['summary'] = results_summary
        results['trials'] = trials
        results['model'] = model
        # results['status'] = status
        # results['exception'] = exception
        # Return
        return model, results

    def cv_clf(self, model, para):
        verbose = 2 if self.verbose<=3 else 3
        scores = []
        for p in np.arange(0, self.cv):
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=True, stratify=self.y)
            # Fit model
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], **para['fit_params'])
            # model.fit(X_train, y_train)
            # Make prediction
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            # Evaluation
            if len(np.unique(y_test))==2:
                results = cle.eval(y_test, y_proba[:, 1], y_pred=y_pred, threshold=self.threshold, pos_label=self.pos_label, verbose=verbose)
                loss = results[para['scoring']]
    
                # Negation of the loss function if required
                if self.greater_is_better: loss = loss * -1
                # Store
                score = {'loss': loss, 'auc': results['auc'], 'kappa': results['kappa'], 'f1': results['f1'], 'status': STATUS_OK}
            else:
                # Compute the loss
                kappscore = cohen_kappa_score(y_test, y_pred)
                loss = kappscore
                # Negation of the loss function if required
                if self.greater_is_better: loss = loss * -1
                # Store
                score = {'loss': loss, 'status': STATUS_OK, 'model': model}

            scores.append(score)

        # Return
        mean_scores = dict(pd.DataFrame(scores).mean(axis=0))
        colnames = list(map(lambda x: x + '_std', [*scores[0]]))
        std_scores = dict(zip(colnames, pd.DataFrame(scores).std(axis=0).values))

        out = {}
        out.update(mean_scores)
        # out.update(std_scores)
        out['model'] = model
        return out


    def _train_reg(self, reg, para):
        # Fit model
        reg.fit(self.x_train, self.y_train, eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)], **para['fit_params'])
        # Make prediction
        y_pred = reg.predict(self.x_test)
        loss = para['loss_func'](self.y_test, y_pred)
        # Negation of the loss function if required
        if self.greater_is_better: loss = loss * -1
        # Store results
        out = {'loss': loss, 'status': STATUS_OK, 'model' : reg}
        # Return
        return out

    def _train_clf(self, clf, para):
        verbose = 2 if self.verbose<=3 else 3

        # Fit model
        # if self.cv is not None:
            # out = self.cv_clf(clf, para)
            # scores = []
            # # getmodels = []
            # # for train_index, test_index in skf.split(self.X, self.y):
            # for p in np.arange(0,self.cv):
            #     X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=True, stratify=self.y)
            #     # Fit model
            #     clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], **para['fit_params'])
            #     # model.fit(X_train, y_train)
            #     # Make prediction
            #     y_pred = clf.predict(X_test)
            #     y_proba = clf.predict_proba(X_test)
            #     # Evaluate
            #     score = _eval(clf, para, y_test, y_proba, y_pred=y_pred, threshold=self.threshold, pos_label=self.pos_label, greater_is_better=self.greater_is_better, verbose=verbose)
            #     score.pop('model')
            #     scores.append(score)

            # mean_scores = dict(pd.DataFrame(scores).mean(axis=0))
            # colnames = list(map(lambda x: x + '_std', [*scores[0]]))
            # std_scores = dict(zip(colnames, pd.DataFrame(scores).std(axis=0).values))
            # out = {}
            # out.update(mean_scores)
            # out.update(std_scores)
            # out['model'] = clf

        clf.fit(self.x_train, self.y_train, eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)], **para['fit_params'])
        # clf.fit(self.x_train, self.y_train)
        # Make prediction
        y_pred = clf.predict(self.x_test)
        y_proba = clf.predict_proba(self.x_test)
        # y_score = clf.decision_function(self.x_test)

        # Note that the loss function is by default maximized towards small/negative values by the hptop method.
        # When you want to optimize auc or f1, you simply need to negate the score.
        # The negation is fixed with the parameter: greater_is_better=False
        out = _eval(clf, para, self.y_test, y_proba, y_pred=y_pred, threshold=self.threshold, pos_label=self.pos_label, greater_is_better=self.greater_is_better, verbose=verbose)

        # # Scoring classification
        # if len(np.unique(y_test))==2:
        #     results = cle.eval(y_test, y_proba[:, 1], y_pred=y_pred, threshold=self.threshold, pos_label=self.pos_label, verbose=verbose)
        #     loss = results[para['scoring']]

        #     # Negation of the loss function if required
        #     if self.greater_is_better: loss = loss * -1
        #     # Store
        #     out = {'loss': loss, 'auc': results['auc'], 'kappa': results['kappa'], 'f1': results['f1'], 'status': STATUS_OK, 'model' : clf}
        # else:
        #     # Compute the loss
        #     kappscore = cohen_kappa_score(self.y_test, y_pred)
        #     loss = kappscore
        #     # Negation of the loss function if required
        #     if self.greater_is_better: loss = loss * -1
        #     # Store
        #     out = {'loss': loss, 'status': STATUS_OK, 'model': clf}
        # Return
        return out

    def xgb_reg(self, para):
        reg = xgb.XGBRegressor(**para['model_params'])
        return self._train_reg(reg, para)

    def lgb_reg(self, para):
        reg = lgb.LGBMRegressor(**para['model_params'])
        return self._train_reg(reg, para)

    def ctb_reg(self, para):
        reg = ctb.CatBoostRegressor(**para['model_params'])
        return self._train_reg(reg, para)

    def xgb_clf(self, para):
        clf = xgb.XGBClassifier(**para['model_params'])
        return self._train_clf(clf, para)

    def xgb_clf_multi(self, para):
        clf = xgb.XGBClassifier(**para['model_params'])
        return self._train_clf(clf, para)

    # Transform results into dataframe
    def to_df(self, trials, verbose=3):
        # Combine params with scoring results
        df_params = pd.DataFrame(trials.vals)
        df_scoring = pd.DataFrame(trials.results)
        df = pd.concat([df_params, df_scoring], axis=1)
        df['tid'] = trials.tids

        # Retrieve only the models with OK status
        Iloc = df['status']=='ok'
        df = df.loc[Iloc, :]

        # Retrieve best model
        if self.greater_is_better:
            df['loss'] = df['loss'] * -1
            idx = df['loss'].argmax()
        else:
            idx = df['loss'].argmin()

        model = df['model'].iloc[idx]
        score = df['loss'].iloc[idx]

        # Remove the model column
        # del df['model']

        # Return
        if verbose>=3: print('[gridsearch] >Best peforming [%s] model: %s=%g' %(self.method, self.eval_metric, score))
        return(df, model)

    # Predict
    def predict(self, X):
        """Prediction using fitted model.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        y_pred : array-like
            predictions results.
        y_proba : array-like
            Probability of the predictions.

        """
        if not hasattr(self, 'model'):
            print('[gridsearch] >No model found. Hint: use the .fit() function first <return>')
            return None

        # Reshape if vector
        if len(X.shape)==1: X=X.reshape(1, -1)
        # Make prediction
        y_pred = self.model.predict(X)
        if '_clf' in self.method:
            y_proba = self.model.predict_proba(X)
        else:
            y_proba = None
        # Return
        return y_pred, y_proba

    def preprocessing(self, df, y_min=2, perc_min_num=0.8, verbose=None):
        """Pre-processing of the input data.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        y_min : int [0..len(y)], optional
            Minimal number of sampels that must be present in a group. All groups with less then y_min samples are labeled as _other_ and are not used in the enriching model. The default is None.
        perc_min_num : float [None, 0..1], optional
            Force column (int or float) to be numerical if unique non-zero values are above percentage. The default is None. Alternative can be 0.8
        verbose : int, (default: 3)
            Print progress to screen.
            0: NONE, 1: ERROR, 2: WARNING, 3: INFO, 4: DEBUG, 5: TRACE

        Returns
        -------
        data : pd.Datarame
            Processed data.

        """
        if verbose is None: verbose = self.verbose
        X = df2onehot(df, y_min=y_min, hot_only=False, perc_min_num=perc_min_num, excl_background='0.0', verbose=verbose)
        return X['onehot']

    def import_example(self, data='titanic', url=None, sep=',', verbose=3):
        """Import example dataset from github source.

        Description
        -----------
        Import one of the few datasets from github source or specify your own download url link.

        Parameters
        ----------
        data : str
            Name of datasets: 'sprinkler', 'titanic', 'student', 'fifa', 'cancer', 'waterpump', 'retail'
        url : str
            url link to to dataset.
        verbose : int, (default : 3)
            Print progress to screen.
            0: None, 1: ERROR, 2: WARN, 3: INFO, 4: DEBUG, 5: TRACE

        Returns
        -------
        pd.DataFrame()
            Dataset containing mixed features.

        """
        return import_example(data=data, url=url, sep=sep, verbose=verbose)

    def plot(self, num_trees=0, plottype='horizontal', figsize=(15, 25), verbose=3):
        """Tree plot.

        Parameters
        ----------
        num_trees : int, default 0
            Specify the ordinal number of target tree
        plottype : str, (default : 'horizontal')
            Works only in case of xgb model.
            * 'horizontal'
            * 'vertical'
        figsize: tuple, default (25,25)
            Figure size, (height, width)
        verbose : int, (default : 3)
            Print progress to screen.
            0: None, 1: ERROR, 2: WARN, 3: INFO, 4: DEBUG, 5: TRACE

        Returns
        -------
        ax : object

        """
        if not hasattr(self, 'model'):
            print('[gridsearch] >No model found. Hint: use the .fit() function first <return>')
            return None

        ax = tree.plot(self.model, num_trees=num_trees, plottype=plottype, figsize=figsize, verbose=verbose)
        return ax

    def plot_summary(self, ylim=None, figsize=(15, 8)):
        """Plot the summary results.

        Parameters
        ----------
        ylim : tuple
            Set the y-limit. In case of auc it can be: (0.5, 1)
        figsize: tuple, default (25,25)
            Figure size, (height, width)

        Returns
        -------
        ax : object

        """
        # figsize: tuple, default (25,25)
        #     Figure size, (height, width)

        if not hasattr(self, 'model'):
            print('[gridsearch] >No model found. Hint: use the .fit() function first <return>')
            return None

        # if self.greater_is_better:
        #     ascending = False
        # else:
        #     ascending = True

        fig, ax = plt.subplots(figsize=figsize)
        tmpdf = self.results['summary'].sort_values(by='tid', ascending=True)

        if np.any(tmpdf.columns=='loss_mean'):
            # ax.errorbar(tmpdf['tid'], tmpdf['loss_mean'], tmpdf['loss_std'], marker='s', mfc='red', mec='green', ms=20, mew=4)
            ax.errorbar(tmpdf['tid'], tmpdf['loss_mean'], tmpdf['loss_std'], marker='s', mfc='red')
        else:
            ax.plot(tmpdf['loss'].values)

        ax.set_title(self.method)
        ax.set_xlabel('Space id')
        ax.set_ylabel(self.eval_metric)
        ax.grid(True)
        if ylim is not None: ax.set_ylim(ylim)

        return ax


# %% Import example dataset from github.
def import_example(data='titanic', url=None, sep=',', verbose=3):
    """Import example dataset from github source.

    Description
    -----------
    Import one of the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str, (default : "titanic")
        Name of datasets: 'sprinkler', 'titanic', 'student', 'fifa', 'cancer', 'waterpump', 'retail'
    url : str
        url link to to dataset.
    verbose : int, (default : 3)
        Print progress to screen.
        0: None, 1: ERROR, 2: WARN, 3: INFO, 4: DEBUG, 5: TRACE

    Returns
    -------
    pd.DataFrame()
        Dataset containing mixed features.

    """
    if url is None:
        if data=='sprinkler':
            url='https://erdogant.github.io/datasets/sprinkler.zip'
        elif data=='titanic':
            url='https://erdogant.github.io/datasets/titanic_train.zip'
        elif data=='student':
            url='https://erdogant.github.io/datasets/student_train.zip'
        elif data=='cancer':
            url='https://erdogant.github.io/datasets/cancer_dataset.zip'
        elif data=='fifa':
            url='https://erdogant.github.io/datasets/FIFA_2018.zip'
        elif data=='waterpump':
            url='https://erdogant.github.io/datasets/waterpump/waterpump_test.zip'
        elif data=='retail':
            url='https://erdogant.github.io/datasets/marketing_data_online_retail_small.zip'
    else:
        data = wget.filename_from_url(url)

    if url is None:
        if verbose>=3: print('[gridsearch] >Nothing to download.')
        return None

    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    PATH_TO_DATA = os.path.join(curpath, wget.filename_from_url(url))
    if not os.path.isdir(curpath):
        os.makedirs(curpath, exist_ok=True)

    # Check file exists.
    if not os.path.isfile(PATH_TO_DATA):
        if verbose>=3: print('[gridsearch] >Downloading [%s] dataset from github source..' %(data))
        wget.download(url, curpath)

    # Import local dataset
    if verbose>=3: print('[gridsearch] >Import dataset [%s]' %(data))
    df = pd.read_csv(PATH_TO_DATA, sep=sep)
    # Return
    return df


# %% Set the search spaces
def _get_params(fn_name, eval_metric=None, verbose=3):
    if eval_metric is None: raise Exception('[gridsearch] >eval_metric must be provided.')
    if verbose>=3: print('[gridsearch] >Collecting %s parameters.' %(fn_name))

    # XGB parameters
    if fn_name=='xgb_reg':
        xgb_reg_params = {
            'learning_rate': hp.choice('learning_rate', np.arange(0.05, 0.31, 0.05)),
            'max_depth': hp.choice('max_depth', np.arange(5, 16, 1, dtype=int)),
            'min_child_weight': hp.choice('min_child_weight', np.arange(1, 10, 1, dtype=int)),
            'gamma': hp.choice('gamma', [0, 0.25, 0.5, 1.0]),
            'reg_lambda': hp.choice('reg_lambda', [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]),
            # 'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'n_estimators': 100,
        }
        xgb_fit_params = {
            'eval_metric': eval_metric,
            'early_stopping_rounds': 10,
            'verbose': False
        }
        xgb_para = {}
        xgb_para['model_params'] = xgb_reg_params
        xgb_para['fit_params'] = xgb_fit_params
        xgb_para['loss_func'] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))
        return(xgb_para)

    # LightGBM parameters
    if fn_name=='lgb_reg':
        lgb_reg_params = {
            'learning_rate': hp.choice('learning_rate', np.arange(0.05, 0.31, 0.05)),
            'max_depth': hp.choice('max_depth', np.arange(5, 16, 1, dtype=int)),
            'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
            # 'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
            'subsample': hp.uniform('subsample', 0.8, 1),
            'n_estimators': 100,
        }
        lgb_fit_params = {
            'eval_metric': 'l2',
            'early_stopping_rounds': 10,
            'verbose': False
        }
        lgb_para = {}
        lgb_para['model_params'] = lgb_reg_params
        lgb_para['fit_params'] = lgb_fit_params
        lgb_para['loss_func'] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))

        return(lgb_para)

    # LightGBM parameters
    if fn_name=='ctb_reg':
        # CatBoost parameters
        ctb_reg_params = {
            'learning_rate' : hp.choice('learning_rate', np.arange(0.05, 0.31, 0.05)),
            'max_depth' : hp.choice('max_depth', np.arange(5, 16, 1, dtype=int)),
            'colsample_bylevel' : hp.choice('colsample_bylevel', np.arange(0.3, 0.8, 0.1)),
            'n_estimators' : 100,
            'eval_metric' : eval_metric,
        }
        ctb_fit_params = {
            'early_stopping_rounds': 10,
            'verbose': False
        }
        ctb_para = {}
        ctb_para['model_params'] = ctb_reg_params
        ctb_para['fit_params'] = ctb_fit_params
        ctb_para['loss_func'] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))

        return(ctb_para)

    if 'xgb_clf' in fn_name:
        xgb_clf_params = {
            'learning_rate' : hp.choice('learning_rate', np.arange(0.01, 0.3, 0.05)),
            'max_depth' : hp.choice('max_depth', np.arange(5, 16, 1, dtype=int)),
            'min_child_weight' : hp.choice('min_child_weight', np.arange(1, 10, 1, dtype=int)),
            'gamma' : hp.choice('gamma', [0.5, 1, 1.5, 2, 5]),
            'subsample' : hp.uniform('subsample', 0.5, 1),
            'n_estimators' : hp.choice('n_estimators', [10, 25, 100, 250]),
            'booster' : 'gbtree',
            # 'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
        }

        if fn_name=='xgb_clf':
            xgb_clf_params['eval_metric'] = hp.choice('eval_metric', ['error', eval_metric])
            xgb_clf_params['objective'] = 'binary:logistic'
            xgb_clf_params['scale_pos_weight'] = hp.choice('scale_pos_weight', [0, 0.5, 1])
            scoring = eval_metric

        if fn_name=='xgb_clf_multi':
            xgb_clf_params['objective']='multi:softprob'
            # from sklearn.metrics import make_scorer
            # scoring = make_scorer(cohen_kappa_score, greater_is_better=True)
            scoring='kappa'  # Note that this variable is not used.

        xgb_fit_params = {'early_stopping_rounds': 10, 'verbose': False}

        xgb_para = {}
        xgb_para['model_params'] = xgb_clf_params
        xgb_para['fit_params'] = xgb_fit_params
        xgb_para['scoring'] = scoring
        if verbose>=3: print('[gridsearch] >Total search space is across [%.0d] variables, loss function: [%s].' %(len([*xgb_para['model_params']]), eval_metric))

        return(xgb_para)


def _check_input(X, y, pos_label, method, verbose=3):
    # if (type(X) is not np.ndarray): raise Exception('[gridsearch] >Error: dataset X should be of type numpy array')
    if (type(X) is not pd.DataFrame): raise Exception('[gridsearch] >Error: dataset X should be of type pd.DataFrame')
    if (type(y) is not np.ndarray): raise Exception('[gridsearch] >Error: Response variable y should be of type numpy array')
    if np.any(np.isnan(y)): raise Exception('[gridsearch] >Error: Response variable y can not have nan values.')

    # Checks pos_label status in case of method is classification
    if ('_clf' in method):
        # Set pos_label to True when boolean
        if (pos_label is None) and (str(y.dtype)=='bool'):
            pos_label = True
        # Raise exception in case of pos_label is not set and not bool.
        if (pos_label is None) and (len(np.unique(y))==2) and not (str(y.dtype)=='bool'):
            if verbose>=1: raise Exception('[gridsearch] >Error: In a two-class approach [%s], pos_label needs to be set or of type bool.' %(pos_label))

        # Check label for classificaiton and two-class model
        if (pos_label is not None) and (len(np.unique(y))==2) and not (np.any(np.isin(y.astype(str), str(pos_label)))):
            if verbose>=2: raise Exception('[gridsearch] >Error: y contains values %s but none matches pos_label=%s <return>' %(str(np.unique(y)), pos_label))

    # Method checks
    if (len(np.unique(y))>2) and not ('_clf_multi' in method):
        method='xgb_clf_multi'
        if verbose>=2: print('[gridsearch] >Warning: y contains more then 2 classes; method is set to: %s' %(method))
    if (len(np.unique(y))==2) and ('_clf_multi' in method):
        method='xgb_clf'
        if verbose>=2: print('[gridsearch] >Warning: y contains 2 classes; method is set to: %s' %(method))

    return pos_label, method


def _eval(clf, para, y_test, y_proba, y_pred, threshold, pos_label, greater_is_better, verbose):
    # Scoring classification
    if len(np.unique(y_test))==2:
        results = cle.eval(y_test, y_proba[:, 1], y_pred=y_pred, threshold=threshold, pos_label=pos_label, verbose=verbose)
        loss = results[para['scoring']]
        # Negation of the loss function if required
        if greater_is_better: loss = loss * -1
        # Store
        out = {'loss': loss, 'auc': results['auc'], 'kappa': results['kappa'], 'f1': results['f1'], 'status': STATUS_OK, 'model' : clf}
    else:
        # Compute the loss
        kappscore = cohen_kappa_score(y_test, y_pred)
        loss = kappscore
        # Negation of the loss function if required
        if greater_is_better: loss = loss * -1
        # Store
        out = {'loss': loss, 'status': STATUS_OK, 'model': clf}
    return out
