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
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials, hp

import classeval as cle
from df2onehot import df2onehot
import treeplot as tree
import colourmap
from tqdm import tqdm


# %%
class gridsearch():
    """Create a class gridsearch that is instantiated with the desired method."""

    def __init__(self, method, max_evals=25, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, eval_metric=None, greater_is_better=None, random_state=None, verbose=3):
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
            * 'auc' : area under ROC curve (classification : default)
            * 'rmse' : root mean squared error (regression: default)
            * 'kappa' : (multi-classification : default)
            * 'mae' : mean absolute error. (regression)
            * 'logloss' : for binary logarithmic loss.
            * 'mlogloss' : for binary logarithmic multi-class log loss (cross entropy).
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
        elif (eval_metric is None) and ('_clf_multi' in method):
            eval_metric = 'kappa'
        elif (eval_metric is None) and ('_clf' in method):
            eval_metric = 'auc'
        # Check the greater_is_better for evaluation metric
        if (greater_is_better is None) and ('_reg' in method):
            greater_is_better = False
        elif (greater_is_better is None) and ('_clf' in method):
            greater_is_better = True
        if top_cv_evals is None: top_cv_evals=0
        if (test_size<=0) or (test_size is None): raise Exception('[gridsearch] >Error: test_size must be >0 and not None. Note that the final model is learned on the entire dataset.')

        self.method=method
        self.eval_metric=eval_metric
        self.greater_is_better=greater_is_better
        self.max_evals=max_evals
        self.top_cv_evals=top_cv_evals
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
            In case of classification (_clf), the model will be fitted on the pos_label that is in y.
        verbose : int, (default : 3)
            Print progress to screen.
            0: None, 1: ERROR, 2: WARN, 3: INFO, 4: DEBUG, 5: TRACE

        Returns
        -------
        results : dict
            * best_params: Best performing parameters.
            * summary: Summary of the models with the loss and other variables.
            * trials: All model results.
            * model: Best performing model.
            * val_results: Results on indepedent validation dataset.

        """
        if self.verbose>=3: print('[gridsearch] >Start hyperparameter optimization.')
        if verbose is None: verbose = self.verbose
        # Check input parameters
        self.pos_label, self.method = _check_input(X, y, pos_label, self.method, verbose=self.verbose)
        # Set validation set
        self._set_validation_set(X, y)
        # Find best parameters
        self.model, self.results = self.HPOpt(verbose=self.verbose)
        # Fit on all data using best parameters
        if self.verbose>=3: print('[gridsearch] >Refit %s on the entire dataset with the optimal parameters settings.' %(self.method))
        self.model.fit(X, y)
        # Return
        return self.results

    def _set_validation_set(self, X, y):
        """Set the validation set.

        Description
        -----------
        Here we seperate a small part of the data as the validation set.
        * The new data is stored in self.X and self.y
        * The validation X and y are stored in self.X_val and self.y_val
        """
        # from sklearn.model_selection import cross_val_score, KFold
        if self.verbose>=3: print('[gridsearch] >Total datset: %s ' %(str(X.shape)))

        if (self.val_size is not None):
            if '_clf' in self.method:
                self.X, self.X_val, self.y, self.y_val = train_test_split(X, y, test_size=self.val_size, random_state=self.random_state, shuffle=True, stratify=y)
            elif '_reg' in self.method:
                self.X, self.X_val, self.y, self.y_val = train_test_split(X, y, test_size=self.val_size, random_state=self.random_state, shuffle=True)
        else:
            self.X = X
            self.y = y
            self.X_val = None
            self.y_val = None

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
            * trials: All model results.
            * model: Best performing model.
            * val_results: Results on indepedent validation dataset.

        """
        # Import the desired model-function for the classification/regression
        fn = getattr(self, self.method)
        # Import search space for the specific function
        space = _get_params(self.method, eval_metric=self.eval_metric)

        # Split train-test set
        if '_clf' in self.method:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=True, stratify=self.y)
        elif '_reg' in self.method:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=True)

        # Hyperoptimization to find best performing model. Set the trials which is the object where all the HPopt results are stored.
        trials=Trials()
        best_params = fmin(fn=fn, space=space, algo=self.algo, max_evals=self.max_evals, trials=trials, show_progressbar=True)
        # Summary results
        results_summary, model = self.to_df(trials, verbose=self.verbose)

        # Cross-validation over the optimized models.
        if self.cv is not None:
            model, results_summary, best_params = self._cv(results_summary, space, best_params)

        # Validation error
        val_results = None
        if self.val_size is not None:
            if self.verbose>=3: print('[gridsearch] >Evalute best %s model on independent validation dataset (%.0f samples).' %(self.method, len(self.y_val)))
            # Evaluate results
            val_score, val_results = self._eval(self.X_val, self.y_val, model, space, verbose=2)
            if self.verbose>=3: print('[gridsearch] >%s on independent validation dataset: %.4g' %(self.eval_metric, val_score['loss']))

        # Remove the model column
        del results_summary['model']
        # Store
        results = {}
        results['params'] = best_params
        results['summary'] = results_summary
        results['trials'] = trials
        results['model'] = model
        results['val_results'] = val_results
        # Return
        return model, results

    def _cv(self, results_summary, space, best_params):
        ascending = False if self.greater_is_better else True
        results_summary['loss_mean'] = np.nan
        results_summary['loss_std'] = np.nan

        # Determine maximum folds
        top_cv_evals = np.minimum(results_summary.shape[0], self.top_cv_evals)
        idx = results_summary['loss'].sort_values(ascending=ascending).index[0:top_cv_evals]
        if self.verbose>=3: print('[gridsearch] >%.0d-fold cross validation for the top %.0d scoring models, Total: %.0f iterations.\n' %(self.cv, len(idx), self.cv * len(idx)))

        # Run over the top-scoring models.
        for i in tqdm(idx):
            scores = []
            # Run over the cross-validations
            for k in np.arange(0, self.cv):
                # Split train-test set
                if '_clf' in self.method:
                    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=True, stratify=self.y)
                elif '_reg' in self.method:
                    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=True)

                # Evaluate model
                score, _ = self._train_model(results_summary['model'].iloc[i], space)
                score.pop('model')
                scores.append(score)

            # Store mean and std summary
            results_summary['loss_mean'].iloc[i] = pd.DataFrame(scores)['loss'].mean()
            results_summary['loss_std'].iloc[i] = pd.DataFrame(scores)['loss'].std()

        # Negate scoring if required. The hpopt is optimized for loss functions (lower is better). Therefore we need to set eg the auc to negative and here we need to return.
        if self.greater_is_better:
            results_summary['loss_mean'] = results_summary['loss_mean'] * -1
            idx_best = results_summary['loss_mean'].argmax()
        else:
            idx_best = results_summary['loss_mean'].argmin()

        # Get best performing model based on the mean scores.
        model = results_summary['model'].iloc[idx_best]
        results_summary['best_cv'] = False
        results_summary['best_cv'].iloc[idx_best] = True
        # Collect best parameters for this model
        best_params = dict(results_summary.iloc[idx_best, np.isin(results_summary.columns, [*best_params.keys()])])
        # Return
        return model, results_summary, best_params

    def _train_model(self, model, para):
        verbose = 2 if self.verbose<=3 else 3
        # Evaluation is determine for both training and testing set. These results can plotted after finishing.
        eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        # Make fit with stopping-rule to avoid overfitting.
        model.fit(self.X_train, self.y_train, eval_set=eval_set, **para['fit_params'])
        # Evaluate results
        out, eval_results = self._eval(self.X_test, self.y_test, model, para, verbose=verbose)
        # Return
        if self.verbose>=4: print("[gridsearch] >best score: {0}, best iteration: {1}".format(model.best_score, model.best_iteration))
        return out, eval_results

    def xgb_reg(self, para):
        reg = xgb.XGBRegressor(**para['model_params'])
        out, _ = self._train_model(reg, para)
        return out

    def lgb_reg(self, para):
        reg = lgb.LGBMRegressor(**para['model_params'])
        out, _ = self._train_model(reg, para)
        return out

    def ctb_reg(self, para):
        reg = ctb.CatBoostRegressor(**para['model_params'])
        out, _ = self._train_model(reg, para)
        return out

    def xgb_clf(self, para):
        clf = xgb.XGBClassifier(**para['model_params'])
        out, _ = self._train_model(clf, para)
        return out

    def xgb_clf_multi(self, para):
        clf = xgb.XGBClassifier(**para['model_params'])
        out, _ = self._train_model(clf, para)
        return out

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
        df['best'] = False
        df['best'].iloc[idx] = True

        # Return
        if verbose>=3: print('[gridsearch] >Best peforming [%s] model: %s=%g' %(self.method, self.eval_metric, score))
        return(df, model)

    # Predict
    def predict(self, X, model=None):
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
        if model is None:
            model = self.model

        # Reshape if vector
        if len(X.shape)==1: X=X.reshape(1, -1)
        # Make prediction
        y_pred = model.predict(X)
        if '_clf' in self.method:
            y_proba = model.predict_proba(X)
        else:
            y_proba = None

        # Return
        return y_pred, y_proba

    def _eval(self, X_test, y_test, model, para, verbose=3):
        """Classifier Evaluation.

        Description
        -----------
        Note that the loss function is by default maximized towards small/negative values by the hptop method.
        When you want to optimize auc or f1, you simply need to negate the score.
        The negation is fixed with the parameter: greater_is_better=False
        """
        results = None
        # Make prediction
        y_pred = model.predict(X_test)
        # Evaluate results
        if '_clf' in self.method:
            y_proba = model.predict_proba(X_test)
            # y_score = model.decision_function(self.X_test)

            # 2-class classification
            if len(np.unique(y_test))==2:
                results = cle.eval(y_test, y_proba[:, 1], y_pred=y_pred, threshold=self.threshold, pos_label=self.pos_label, verbose=verbose)
                loss = results[para['scoring']]
                # Negation of the loss function if required
                if self.greater_is_better: loss = loss * -1
                # Store
                out = {'loss': loss, 'auc': results['auc'], 'kappa': results['kappa'], 'f1': results['f1'], 'status': STATUS_OK, 'model' : model}
            else:
                # Multi-class classification
                # Compute the loss
                kappscore = cohen_kappa_score(y_test, y_pred)
                loss = kappscore
                # Negation of the loss function if required
                if self.greater_is_better: loss = loss * -1
                # Store
                out = {'loss': loss, 'status': STATUS_OK, 'model': model}
        elif '_reg' in self.method:
            # Regression
            loss = para['loss_func'](y_test, y_pred)
            # Negation of the loss function if required
            if self.greater_is_better: loss = loss * -1
            # Store results
            out = {'loss': loss, 'status': STATUS_OK, 'model' : model}
        else:
            raise Exception('[gridsearch] >Error: Method %s does not exists.' %(self.method))

        return out, results

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

    def treeplot(self, num_trees=None, plottype='horizontal', figsize=(15, 25), verbose=3):
        """Tree plot.

        Parameters
        ----------
        num_trees : int, default None
            Best tree is shown when None. Specify the ordinal number of any other target tree.
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

        if num_trees is None: num_trees = self.model.best_iteration
        ax = tree.plot(self.model, num_trees=num_trees, plottype=plottype, figsize=figsize, verbose=verbose)
        return ax

    def plot_validation(self):
        """Plot the results on the validation set.

        Returns
        -------
        ax : object
            Figure axis.

        """
        if not hasattr(self, 'model'):
            print('[gridsearch] >No model found. Hint: use the .fit() function first <return>')
            return None

        if '_clf' in self.method:
            if (self.results.get('val_results', None)) is not None:
                ax = cle.plot(self.results['val_results'])
                return ax
        else:
            # fig, ax = plt.subplots(figsize=figsize)
            # plt.scatter(y, y_pred)
            print('[gridsearch] >This plot only works for classifcation. <return>')

    def plot_params(self, top_n=10, shade=True, figsize=(15, 15)):
        """Distribution of parameters.

        Description
        -----------
        This plot demonstrate the density distribution of the used parameters.
        Green will depic the best detected parameter and red demonstrates the top n paramters with best loss.

        Parameters
        ----------
        top_n : int, (default : 10)
            Top n paramters that scored highest are plotted in red.
        shade : bool, (default : True)
            Fill the density plot.
        figsize: tuple, default (15,15)
            Figure size, (height, width)

        Returns
        -------
        ax : object
            Figure axis.

        """
        top_n = np.minimum(top_n, self.results['summary'].shape[0])
        getcolors = colourmap.generate(top_n, cmap='Reds_r')
        ascending = False if self.greater_is_better else True

        # Sort data based on loss
        if np.any(self.results['summary'].columns=='loss_mean'):
            colname = 'loss_mean'
            colbest = 'best_cv'
        else:
            colname = 'loss'
            colbest = 'best'

        # Retrieve data
        df_summary = self.results['summary'].sort_values(by=colname, ascending=ascending)
        idx_best = np.where(df_summary[colbest])[0]
        # Collect parameters
        params = [*self.results['params'].keys()]
        # Setup figure size
        nrRows = np.mod(3, len(params)) + len(params) - (np.mod(3, len(params)) * 3)
        nrCols = 3
        fig, ax = plt.subplots(nrCols, nrRows, figsize=figsize)

        # Plot density for each parameter.
        c = 0
        for i in range(0, nrRows):
            # Density Plot with Rug Plot
            for k in range(0, nrCols):
                if c<=len(params):
                    param_name = params[c]
                    c = c + 1
                    linefit = sns.distplot(self.results['summary'][param_name],
                                           hist=False,
                                           kde=True,
                                           rug=True,
                                           color='darkblue',
                                           kde_kws={'shade': shade, 'linewidth': 1},
                                           rug_kws={'color': 'black'},
                                           ax=ax[i][k])

                    y_data = linefit.get_lines()[0].get_data()[1]
                    getvals = df_summary[param_name].values
                    if len(y_data)>0:
                        # Plot the top n (not the first because that one is plotted in green)
                        ax[i][k].vlines(getvals[1:top_n], np.min(y_data), np.max(y_data), linewidth=1, colors=getcolors, linestyles='dashed', label='Top ' + str(top_n))
                        # Plot the best one
                        ax[i][k].vlines(getvals[idx_best], np.min(y_data), np.max(y_data), linewidth=2, colors='g', linestyles='solid', label='Best')

                    ax[i][k].set_title(('%s: %.3g' %(param_name, getvals[idx_best])))
                    ax[i][k].set_ylabel('Density')
                    ax[i][k].grid(True)
                    ax[i][k].legend()

            # Scatter plot
            df_sum = self.results['summary'].sort_values(by='tid', ascending=True)
            idx_best = np.where(df_sum[colbest])[0]
            _, ax2 = plt.subplots(nrCols, nrRows, figsize=figsize)
            c = 0
            for i in range(0, nrRows):
                for k in np.arange(0, nrCols):
                    if c<=len(params):
                        param_name = params[c]
                        sns.regplot('tid', param_name, data=df_sum, ax=ax2[i][k], label='all results')
                        # Scatter best value
                        ax2[i][k].scatter(df_sum['tid'].values[idx_best], df_sum[param_name].values[idx_best], s=100, color='r', marker='x', label='best')
                        # Scatter top n values
                        ax2[i][k].scatter(df_summary['tid'].values[1:top_n], df_summary[param_name].values[1:top_n], s=100, color=getcolors[0:top_n-1, :], marker='.', label='Top ' + str(top_n))
                        # Set labels
                        ax2[i][k].set(xlabel = 'iteration', ylabel = '{}'.format(param_name), title = '{} over Search'.format(param_name));
                        ax2[i][k].set_title(('%s: %.3g' %(param_name, df_sum[param_name].values[idx_best])))
                        ax2[i][k].grid(True)
                        ax2[i][k].legend()
                        c = c + 1

        return ax, ax2

    def plot_summary(self, ylim=None, figsize=(15, 10)):
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
            Figure axis.

        """
        if not hasattr(self, 'model'):
            print('[gridsearch] >No model found. Hint: use the .fit() function first <return>')
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        tmpdf = self.results['summary'].sort_values(by='tid', ascending=True)

        if np.any(tmpdf.columns=='loss_mean'):
            ax1.errorbar(tmpdf['tid'], tmpdf['loss_mean'], tmpdf['loss_std'], marker='s', mfc='red', label=str(self.cv) + '-fold cv')
            idx = np.where(tmpdf['best_cv'].values)[0]
            ax1.hlines(tmpdf['loss_mean'].iloc[idx], 0, tmpdf['loss_mean'].shape[0], colors='r', linestyles='dashed', label='best model with cv')
            ax1.vlines(idx, tmpdf['loss'].min(), tmpdf['loss_mean'].iloc[idx], colors='r', linestyles='dashed')
        else:
            idx = np.where(tmpdf['best'].values)[0]
            ax1.hlines(tmpdf['loss'].iloc[idx], 0, tmpdf['loss'].shape[0], colors='r', linestyles='dashed', label='best model')
            ax1.vlines(idx, tmpdf['loss'].min(), tmpdf['loss'].iloc[idx], colors='r', linestyles='dashed')

        # Plot all other evalution results on the single test-set
        ax1.scatter(tmpdf['tid'].values, tmpdf['loss'].values, s=10, label='Test size: '+ str(self.test_size))
        # Set labels
        ax1.set_title(self.method)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel(self.eval_metric)
        ax1.grid(True)
        ax1.legend()
        if ylim is not None: ax1.set_ylim(ylim)

        # scores = tmpdf[['loss','tid']]
        # sns.lmplot('tid', 'loss', data = scores, size = 8)
        # plt.title(self.method)
        # plt.xlabel('Iteration')
        # plt.ylabel(self.eval_metric)
        # plt.grid(True)
        # plt.legend()
        # if ylim is not None: ax1.set_ylim(ylim)
        
        
        eval_metric = [*self.model.evals_result()['validation_0'].keys()][0]
        ax2.plot([*self.model.evals_result()['validation_0'].values()][0], label='Train error')
        ax2.plot([*self.model.evals_result()['validation_1'].values()][0], label='Test error')
        ax2.set_ylabel(eval_metric)
        ax2.set_title(self.method)
        ax2.grid(True)
        ax2.legend()

        return ax1, ax2


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
    # choice : categorical variables
    # quniform : discrete uniform (integers spaced evenly)
    # uniform: continuous uniform (floats spaced evenly)
    # loguniform: continuous log uniform (floats spaced evenly on a log scale)

    if eval_metric is None: raise Exception('[gridsearch] >eval_metric must be provided.')
    if verbose>=3: print('[gridsearch] >Collecting %s parameters.' %(fn_name))

    # XGB parameters
    if fn_name=='xgb_reg':
        xgb_reg_params = {
            'learning_rate' : hp.quniform('learning_rate', 0.05, 0.31, 0.05),
            'max_depth' : hp.choice('max_depth', np.arange(5, 30, 1, dtype=int)),
            'min_child_weight': hp.choice('min_child_weight', np.arange(1, 10, 1, dtype=int)),
            'gamma': hp.choice('gamma', [0, 0.25, 0.5, 1.0]),
            'reg_lambda': hp.choice('reg_lambda', [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'n_estimators' : hp.choice('n_estimators', range(20, 205, 5)),
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
            'learning_rate' : hp.quniform('learning_rate', 0.05, 0.31, 0.05),
            'max_depth' : hp.choice('max_depth', np.arange(5, 30, 1, dtype=int)),
            'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
            'subsample': hp.uniform('subsample', 0.8, 1),
            'n_estimators' : hp.choice('n_estimators', range(20, 205, 5)),
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
            'learning_rate' : hp.quniform('learning_rate', 0.05, 0.31, 0.05),
            'max_depth' : hp.choice('max_depth', np.arange(5, 30, 1, dtype=int)),
            'colsample_bylevel' : hp.choice('colsample_bylevel', np.arange(0.3, 0.8, 0.1)),
            'n_estimators' : hp.choice('n_estimators', range(20, 205, 5)),
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
            'learning_rate' : hp.quniform('learning_rate', 0.01, 0.5, 0.01),
            'max_depth' : hp.choice('max_depth', range(5, 30, 1)),
            'min_child_weight' : hp.quniform('min_child_weight', 1, 10, 1),
            'gamma' : hp.choice('gamma', [0.5, 1, 1.5, 2, 5]),
            'subsample' : hp.quniform('subsample', 0.1, 1, 0.01),
            'n_estimators' : hp.choice('n_estimators', range(20, 205, 5)),
            'booster' : 'gbtree',
            'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1.0, 0.01),
        }

        if fn_name=='xgb_clf':
            xgb_clf_params['eval_metric'] = hp.choice('eval_metric', ['error', eval_metric])
            xgb_clf_params['objective'] = 'binary:logistic'
            xgb_clf_params['scale_pos_weight'] = hp.choice('scale_pos_weight', [0, 0.5, 1])
            scoring = eval_metric

        if fn_name=='xgb_clf_multi':
            xgb_clf_params['objective']='multi:softprob'
            scoring='kappa'  # Note that this variable is not used.

        xgb_fit_params = {'early_stopping_rounds': 10, 'verbose': False}

        xgb_para = {}
        xgb_para['model_params'] = xgb_clf_params
        xgb_para['fit_params'] = xgb_fit_params
        xgb_para['scoring'] = scoring
        if verbose>=3: print('[gridsearch] >Number of variables in search space is [%.0d], loss function: [%s].' %(len([*xgb_para['model_params']]), eval_metric))

        return(xgb_para)


def _check_input(X, y, pos_label, method, verbose=3):
    # if (type(X) is not np.ndarray): raise Exception('[gridsearch] >Error: dataset X should be of type numpy array')
    if (type(X) is not pd.DataFrame): raise Exception('[gridsearch] >Error: dataset X should be of type pd.DataFrame')
    if (type(y) is not np.ndarray): raise Exception('[gridsearch] >Error: Response variable y should be of type numpy array')
    if 'str' in str(type(y[0])):
        if any(elem is None for elem in y): raise Exception('[gridsearch] >Error: Response variable y can not have None values.')
    else:
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
    if (len(np.unique(y))>2) and ('_clf' in method) and not ('_clf_multi' in method):
        method='xgb_clf_multi'
        if verbose>=2: print('[gridsearch] >Warning: y contains more then 2 classes; method is set to: %s' %(method))
    if (len(np.unique(y))==2) and ('_clf_multi' in method):
        method='xgb_clf'
        if verbose>=2: print('[gridsearch] >Warning: y contains 2 classes; method is set to: %s' %(method))

    return pos_label, method
