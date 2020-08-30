"""hgboost: Hyperoptimization Gradient Boosting library.

Contributors: https://github.com/erdogant/hgboost
"""

import warnings
warnings.filterwarnings("ignore")

import classeval as cle
from df2onehot import df2onehot
import treeplot as tree
import colourmap

import os
import numpy as np
import pandas as pd
import wget

from sklearn.metrics import mean_squared_error, cohen_kappa_score, mean_absolute_error, log_loss, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials, hp

from tqdm import tqdm
import time

# %%
class hgboost:
    """Create a class hgboost that is instantiated with the desired method."""

    def __init__(self, max_eval=250, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10, random_state=None, verbose=3):
        """Initialize hgboost with user-defined parameters.

        Parameters
        ----------
        max_eval : int, (default : 250W)
            Search space is created on the number of evaluations.
        threshold : float, (default : 0.5)
            Classification threshold. In case of two-class model this is 0.5
        cv : int, optional (default : 5)
            Cross-validation. Specifying the test size by test_size.
        top_cv_evals : int, (default : 10)
            Number of top best performing models that is evaluated.
        test_size : float, (default : 0.2)
            Splitting train/test set with test_size=0.2 and train = 1-test_size.
        val_size : float, (default : 0.2)
            Setup the validation set. This part is kept entirely seperate from the test-size.
        random_state : int, (default : None)
            Fix the random state for validation set and test set. Note that is not used for the crossvalidation.
        verbose : int, (default : 3)
            Print progress to screen.
            0: None, 1: ERROR, 2: WARN, 3: INFO, 4: DEBUG, 5: TRACE

        Returns
        -------
        None.

        References
        ----------
        * https://github.com/hyperopt/hyperopt
        * https://www.districtdatalabs.com/parameter-tuning-with-hyperopt
        * https://scikit-learn.org/stable/modules/model_evaluation.html

        """
        if (threshold is None) or (threshold <= 0): raise ValueError('[hgboost] >Error: [threshold] must be >0 and not [None]')
        if (max_eval is None) or (max_eval <= 0): max_eval = 1
        if top_cv_evals is None: top_cv_evals=0
        if (test_size is None) or (test_size <= 0): raise ValueError('[hgboost] >Error: test_size must be >0 and not [None] Note: the final model is learned on the entire dataset. [test_size] may help you getting a more robust model.')

        self.max_eval=max_eval
        self.top_cv_evals=top_cv_evals
        self.threshold = threshold
        self.test_size=test_size
        self.val_size=val_size
        self.algo=tpe.suggest
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose

    def _fit(self, X, y, pos_label=None):
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
        y : array-like.
            Response variable.
        pos_label : string/int.
            In case of classification (_clf), the model will be fitted on the pos_label that is in y.

        Returns
        -------
        results : dict
            * best_params: Best performing parameters.
            * summary: Summary of the models with the loss and other variables.
            * trials: All model results.
            * model: Best performing model.
            * val_results: Results on indepedent validation dataset.

        """
        # Check input data
        X, y, self.pos_label = _check_input(X, y, pos_label, self.method, verbose=self.verbose)

        # Print to screen
        if self.verbose>=3:
            print('[hgboost] >method: %s' %(self.method))
            print('[hgboost] >eval_metric: %s' %(self.eval_metric))
            print('[hgboost] >greater_is_better: %s' %(self.greater_is_better))

        # Set validation set
        self._set_validation_set(X, y)
        # Find best parameters
        self.model, self.results = self._HPOpt()
        # Fit on all data using best parameters
        if self.verbose>=3: print('[hgboost] >Retrain [%s] on the entire dataset with the optimal parameters settings.' %(self.method))
        self.model.fit(X, y)
        # Return
        return self.results

    def _classification(self, X, y, eval_metric, greater_is_better, params):
        # Gather for method, the default metric and greater is better.
        self.eval_metric, self.greater_is_better = _check_eval_metric(self.method, eval_metric, greater_is_better)
        # Import search space for the specific function
        if params == 'default': params = _get_params(self.method, eval_metric=self.eval_metric, verbose=self.verbose)
        self.space = params
        # Fit model
        self.results = self._fit(X, y, pos_label=self.pos_label)
        # Return
        return self.results

    def _regression(self, X, y, eval_metric, greater_is_better, params):
        # Gather for method, the default metric and greater is better.
        self.eval_metric, self.greater_is_better = _check_eval_metric(self.method, eval_metric, greater_is_better)
        # Import search space for the specific function
        if params == 'default': params = _get_params(self.method, eval_metric=self.eval_metric, verbose=self.verbose)
        self.space = params
        # Fit model
        self.results = self._fit(X, y)

    def xgboost_reg(self, X, y, eval_metric='rmse', greater_is_better=False, params='default'):
        """Xgboost Regression with parameter hyperoptimization.

        Parameters
        ----------
        X : pd.DataFrame.
            Input dataset.
        y : array-like
            Response variable.
        eval_metric : str, (default : 'rmse').
            Evaluation metric for the regressor model.
                * 'rmse' : root mean squared error.
                * 'mae' : mean absolute error.
        greater_is_better : bool (default : False).
            If a loss, the output of the python function is negated by the scorer object, conforming to the cross validation convention that scorers return higher values for better models.
        params : dict, (default : 'default').
            Hyper parameters.

        Returns
        -------
        results : dict
            * best_params: Best performing parameters.
            * summary: Summary of the models with the loss and other variables.
            * trials: All model results.
            * model: Best performing model.
            * val_results: Results on indepedent validation dataset.

        """
        if self.verbose>=3: print('[hgboost] >Start hgboost regression..')
        # Method
        self.method='xgb_reg'
        # Run method
        self._regression(X, y, eval_metric, greater_is_better, params)
        # Return
        return self.results

    def lightboost_reg(self, X, y, eval_metric='rmse', greater_is_better=False, params='default'):
        """Light Regression with parameter hyperoptimization.

        Parameters
        ----------
        X : pd.DataFrame.
            Input dataset.
        y : array-like.
            Response variable.
        eval_metric : str, (default : 'rmse').
            Evaluation metric for the regressor model.
            * 'rmse' : root mean squared error.
            * 'mae' : mean absolute error.
        greater_is_better : bool (default : False).
            If a loss, the output of the python function is negated by the scorer object, conforming to the cross validation convention that scorers return higher values for better models.
        params : dict, (default : 'default').
            Hyper parameters.

        Returns
        -------
        results : dict
            * best_params: Best performing parameters.
            * summary: Summary of the models with the loss and other variables.
            * trials: All model results.
            * model: Best performing model.
            * val_results: Results on indepedent validation dataset.

        """
        if self.verbose>=3: print('[hgboost] >Start hgboost regression..')
        # Method
        self.method='lgb_reg'
        # Run method
        self._regression(X, y, eval_metric, greater_is_better, params)
        # Return
        return self.results

    def catboost_reg(self, X, y, eval_metric='rmse', greater_is_better=False, params='default'):
        """Catboost Regression with parameter hyperoptimization.

        Parameters
        ----------
        X : pd.DataFrame.
            Input dataset.
        y : array-like.
            Response variable.
        eval_metric : str, (default : 'rmse').
            Evaluation metric for the regressor model.
                * 'rmse' : root mean squared error.
                * 'mae' : mean absolute error.
        greater_is_better : bool (default : False).
            If a loss, the output of the python function is negated by the scorer object, conforming to the cross validation convention that scorers return higher values for better models.
        params : dict, (default : 'default').
            Hyper parameters.

        Returns
        -------
        results : dict.
            * best_params: Best performing parameters.
            * summary: Summary of the models with the loss and other variables.
            * trials: All model results.
            * model: Best performing model.
            * val_results: Results on indepedent validation dataset.

        """
        if self.verbose>=3: print('[hgboost] >Start hgboost regression..')
        # Method
        self.method='ctb_reg'
        # Run method
        self._regression(X, y, eval_metric, greater_is_better, params)
        # Return
        return self.results

    def xgboost(self, X, y, pos_label=None, method='xgb_clf', eval_metric=None, greater_is_better=None, params='default'):
        """Catboost Classification with parameter hyperoptimization.

        Parameters
        ----------
        X : pd.DataFrame.
            Input dataset.
        y : array-like.
            Response variable.
        pos_label : string/int.
            Fit the model on the pos_label that that is in [y].
        method : String, (default : 'auto').
            * 'xgb_clf': XGboost two-class classifier
            * 'xgb_clf_multi': XGboost multi-class classifier
        eval_metric : str, (default : None).
            Evaluation metric for the regressor of classification model.
            * 'auc' : area under ROC curve (default for two-class)
            * 'kappa' : (default for multi-class)
            * 'f1' : F1-score
            * 'logloss'
        greater_is_better : bool.
            If a loss, the output of the python function is negated by the scorer object, conforming to the cross validation convention that scorers return higher values for better models.
            * auc :  True -> two-class
            * kappa : True -> multi-class

        Returns
        -------
        results : dict.
            * best_params: Best performing parameters.
            * summary: Summary of the models with the loss and other variables.
            * trials: All model results.
            * model: Best performing model.
            * val_results: Results on indepedent validation dataset.

        """
        if self.verbose>=3: print('[hgboost] >Start hgboost classification..')
        self.method = method
        self.pos_label = pos_label
        # Run method
        self._classification(X, y, eval_metric, greater_is_better, params)
        # Return
        return self.results

    def catboost(self, X, y, pos_label=None, eval_metric='auc', greater_is_better=True, params='default'):
        """Catboost Classification with parameter hyperoptimization.

        Parameters
        ----------
        X : pd.DataFrame.
            Input dataset.
        y : array-like.
            Response variable.
        pos_label : string/int.
            Fit the model on the pos_label that that is in [y].
        eval_metric : str, (default : 'auc').
            Evaluation metric for the regressor of classification model.
            * 'auc' : area under ROC curve (two-class classification : default)
        greater_is_better : bool (default : True).
            If a loss, the output of the python function is negated by the scorer object, conforming to the cross validation convention that scorers return higher values for better models.
            * auc :  two-class

        Returns
        -------
        results : dict.
            * best_params: Best performing parameters.
            * summary: Summary of the models with the loss and other variables.
            * trials: All model results.
            * model: Best performing model.
            * val_results: Results on indepedent validation dataset.

        """
        if self.verbose>=3: print('[hgboost] >Start hgboost classification..')
        self.method = 'ctb_clf'
        self.pos_label = pos_label
        # Run method
        self._classification(X, y, eval_metric, greater_is_better, params)
        # Return
        return self.results

    def lightboost(self, X, y, pos_label=None, eval_metric='auc', greater_is_better=True, params='default'):
        """Lightboost Classification with parameter hyperoptimization.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataset.
        y : array-like
            Response variable.
        pos_label : string/int.
            Fit the model on the pos_label that that is in [y].
        eval_metric : str, (default : 'auc')
            Evaluation metric for the regressor of classification model.
            * 'auc' : area under ROC curve (two-class classification : default)
        greater_is_better : bool (default : True)
            If a loss, the output of the python function is negated by the scorer object, conforming to the cross validation convention that scorers return higher values for better models.
            * auc :  True -> two-class

        Returns
        -------
        results : dict
            * best_params: Best performing parameters.
            * summary: Summary of the models with the loss and other variables.
            * trials: All model results.
            * model: Best performing model.
            * val_results: Results on indepedent validation dataset.

        """
        if self.verbose>=3: print('[hgboost] >Start hgboost classification..')
        self.method = 'lgb_clf'
        self.pos_label = pos_label
        # Run method
        self._classification(X, y, eval_metric, greater_is_better, params)
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
        if self.verbose>=3: print('[hgboost] >Total datset: %s ' %(str(X.shape)))

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

    def _HPOpt(self):
        """Hyperoptimization of the search space.

        Description
        -----------
        Minimize a function over a hyperparameter space.
        More realistically: *explore* a function over a hyperparameter space
        according to a given algorithm, allowing up to a certain number of
        function evaluations.  As points are explored, they are accumulated in
        "trials".

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
        if self.verbose>=3: print('[hgboost] >Hyperparameter optimization..')
        # Import the desired model-function for the classification/regression
        disable = (False if (self.verbose<3) else True)
        fn = getattr(self, self.method)

        # Split train-test set
        if '_clf' in self.method:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=True, stratify=self.y)
        elif '_reg' in self.method:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=True)

        # Hyperoptimization to find best performing model. Set the trials which is the object where all the HPopt results are stored.
        trials=Trials()
        best_params = fmin(fn=fn, space=self.space, algo=self.algo, max_evals=self.max_eval, trials=trials, show_progressbar=disable)
        # Summary results
        results_summary, model = self._to_df(trials, verbose=self.verbose)

        # Cross-validation over the optimized models.
        if self.cv is not None:
            model, results_summary, best_params = self._cv(results_summary, self.space, best_params)

        # Validation error
        val_results = None
        if self.val_size is not None:
            if self.verbose>=3: print('[hgboost] >Evalute best [%s] model on independent validation dataset (%.0f samples, %.2f%%).' %(self.method, len(self.y_val), self.val_size * 100))
            # Evaluate results
            val_score, val_results = self._eval(self.X_val, self.y_val, model, self.space, verbose=2)
            if self.verbose>=3: print('[hgboost] >[%s] on independent validation dataset: %.4g' %(self.eval_metric, val_score['loss']))

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
        if self.verbose>=3: print('[hgboost] >%.0d-fold cross validation for the top %.0d scoring models, Total nr. tests: %.0f' %(self.cv, len(idx), self.cv * len(idx)))
        disable = (True if (self.verbose==0 or self.verbose>3) else False)

        # Run over the top-scoring models.
        for i in tqdm(idx, disable=disable):
            scores = []
            # Run over the cross-validations
            for k in np.arange(0, self.cv):
                # Split train-test set
                if '_clf' in self.method:
                    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=None, shuffle=True, stratify=self.y)
                elif '_reg' in self.method:
                    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=None, shuffle=True)

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

    def _train_model(self, model, space):
        verbose = 2 if self.verbose<=3 else 3
        # Evaluation is determine for both training and testing set. These results can plotted after finishing.
        eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        # Make fit with stopping-rule to avoid overfitting. Directly perform evaluation with the eval_set.
        model.fit(self.X_train, self.y_train, eval_set=eval_set, **space['fit_params'])
        # auc_mean = cross_val_score(model, self.X, self.y, cv=self.cv)
        # Evaluate results
        out, eval_results = self._eval(self.X_test, self.y_test, model, space, verbose=verbose)
        # Return
        # The model.best_score is the default eval_metric from eval_set in the fit function. The default depends on the selected method.
        # if self.verbose>=4: print("[hgboost] >best score: {0}, best iteration: {1}".format(model.best_score, model.best_iteration))
        return out, eval_results

    def xgb_reg(self, space):
        reg = xgb.XGBRegressor(**space['model_params'])
        out, _ = self._train_model(reg, space)
        return out

    def lgb_reg(self, space):
        reg = lgb.LGBMRegressor(**space['model_params'])
        out, _ = self._train_model(reg, space)
        return out

    def ctb_reg(self, space):
        reg = ctb.CatBoostRegressor(**space['model_params'])
        out, _ = self._train_model(reg, space)
        return out

    def xgb_clf(self, space):
        clf = xgb.XGBClassifier(**space['model_params'])
        out, _ = self._train_model(clf, space)
        return out

    def ctb_clf(self, space):
        clf = ctb.CatBoostClassifier(**space['model_params'])
        out, _ = self._train_model(clf, space)
        return out

    def lgb_clf(self, space):
        clf = lgb.LGBMClassifier(**space['model_params'])
        out, _ = self._train_model(clf, space)
        return out

    def xgb_clf_multi(self, space):
        clf = xgb.XGBClassifier(**space['model_params'])
        out, _ = self._train_model(clf, space)
        return out

    # Transform results into dataframe
    def _to_df(self, trials, verbose=3):
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
        if verbose>=3: print('[hgboost] >Best peforming [%s] model: %s=%g' %(self.method, self.eval_metric, score))
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
            if self.verbose>2: print('[hgboost] >Warning: No model found. Hint: fit a model first using xgboost, catboost or lightboost <return>')
            return None, None
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

    def _eval(self, X_test, y_test, model, space, verbose=3):
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
            # Compute probability
            y_proba = model.predict_proba(X_test)
            # y_score = model.decision_function(self.X_test)

            # multi-class classification
            if ('_clf_multi' in self.method):
                if self.eval_metric=='kappa':
                    loss = cohen_kappa_score(y_test, y_pred)
                elif self.eval_metric=='logloss':
                    loss = log_loss(y_test, y_pred)
                elif self.eval_metric=='auc':
                    loss = roc_auc_score(y_test, y_pred, multi_class='ovr')
                elif self.eval_metric=='f1':
                    loss = f1_score(y_test, y_pred)
                else:
                    raise ValueError('[hgboost] >Error: [%s] is not a valid [eval_metric] for [%s].' %(self.eval_metric, self.method))
                # Negative loss score if required
                if self.greater_is_better: loss = loss * -1
                # Store
                out = {'loss': loss, 'status': STATUS_OK, 'eval_time': time.time(), 'model': model}
            else:
                # Two-class classification
                results = cle.eval(y_test, y_proba[:, 1], y_pred=y_pred, threshold=self.threshold, pos_label=self.pos_label, verbose=0)
                # results = cle.ROC.eval(y_test, y_proba[:, 1], threshold=self.threshold, pos_label=self.pos_label, verbose=verbose)

                if self.eval_metric=='kappa':
                    loss = results[self.eval_metric]
                elif self.eval_metric=='logloss':
                    loss = log_loss(y_test, y_pred)
                elif self.eval_metric=='auc':
                    loss = results[self.eval_metric]
                elif self.eval_metric=='f1':
                    loss = results[self.eval_metric]
                elif self.eval_metric=='auc_cv':
                    loss = np.mean(cross_val_score(model, self.X_train, self.y_train, cv=self.cv))
                else:
                    raise ValueError('[hgboost] >Error: [%s] is not a valid [eval_metric] for [%s].' %(self.eval_metric, self.method))

                # Negative loss score if required
                if self.greater_is_better: loss = loss * -1
                # Store
                out = {'loss': loss, 'eval_time': time.time(), 'status': STATUS_OK, 'model' : model}
                # out = {'loss': loss, 'eval_time': time.time(), 'auc': results['auc'], 'kappa': results['kappa'], 'f1': results['f1'], 'status': STATUS_OK, 'model' : model}
        elif '_reg' in self.method:
            # Regression
            # loss = space['loss_func'](y_test, y_pred)
            if self.eval_metric=='rmse':
                loss = mean_squared_error(y_test, y_pred)
            elif self.eval_metric=='mae':
                loss = mean_absolute_error(y_test, y_pred)
            else:
                raise ValueError('[hgboost] >Error: [%s] is not a valid [eval_metric] for [%s].' %(self.eval_metric, self.method))

            # Negative loss score if required
            if self.greater_is_better: loss = loss * -1
            # Store results
            out = {'loss': loss, 'eval_time': time.time(), 'status': STATUS_OK, 'model' : model}
        else:
            raise ValueError('[hgboost] >Error: Method %s does not exists.' %(self.method))

        if self.verbose>=5: print('[hgboost] >[%s] - [%s] - loss: %s' %(self.method, self.eval_metric, loss))
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

    def treeplot(self, num_trees=None, plottype='horizontal', figsize=(20, 25), return_ax=False, verbose=3):
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
            print('[hgboost] >No model found. Hint: fit a model first using xgboost, catboost or lightboost <return>')
            return None
        ax = None
        # Plot the tree
        ax = tree.plot(self.model, num_trees=num_trees, plottype=plottype, figsize=figsize, verbose=verbose)
        # Return
        if return_ax: return ax

    def plot_cv(self, figsize=(15, 8), cmap='Set2', return_ax=False):
        """Plot the results on the crossvalidation set.

        Parameters
        ----------
        figsize: tuple, default (25,25)
            Figure size, (height, width)

        Returns
        -------
        ax : object
            Figure axis.

        """
        print('[hgboost] >%.0d-fold crossvalidation is performed with [%s]' %(self.cv, self.method))
        disable = (True if (self.verbose==0 or self.verbose>3) else False)
        ax = None
        # Run the cross-validations
        cv_results = {}
        for i in tqdm(np.arange(0, self.cv), disable=disable):
            name = 'cross ' + str(i)
            # Split train-test set
            if ('_clf' in self.method) and not ('_multi' in self.method):
                _, X_test, _, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=None, shuffle=True, stratify=self.y)
                # Evaluate model
                _, cl_results = self._eval(X_test, y_test, self.model, self.space, verbose=0)
                cv_results[name] = cl_results
            elif '_reg' in self.method:
                _, X_test, _, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=None, shuffle=True)
                y_pred = self.predict(X_test, model=self.model)[0]
                cv_results[name] = pd.DataFrame(np.c_[y_test, y_pred], columns=['y', 'y_pred'])

        # Make plots
        if ('_clf' in self.method) and not ('_multi' in self.method):
            ax = cle.plot_cross(cv_results, title=('%.0d-fold crossvalidation results on best-performing %s' %(self.cv, self.method)), figsize=figsize)
        elif '_reg' in self.method:
            fig, ax = plt.subplots(figsize=figsize)
            colors = colourmap.generate(len(cv_results), cmap=cmap)
            for i, key in enumerate(cv_results.keys()):
                sns.regplot('y', 'y_pred', data=cv_results.get(key), ax=ax, color=colors[i, :], label=key)
            ax.legend()
            ax.grid(True)
            ax.set_xlabel('True value')
            ax.set_ylabel('Predicted value')

        return ax

    def plot_validation(self, figsize=(15, 8), cmap='Set2', return_ax=False):
        """Plot the results on the validation set.

        Parameters
        ----------
        figsize: tuple, default (25,25)
            Figure size, (height, width)

        Returns
        -------
        ax : object
            Figure axis.

        """
        if not hasattr(self, 'model'):
            print('[hgboost] >No model found. Hint: fit a model first using xgboost, catboost or lightboost <return>')
            return None
        if self.val_size is None:
            print('[hgboost] >No validation set found. Hint: use the parameter [val_size=0.2] first <return>')
            return None
        ax = None

        title = 'Results on independent validation set'
        if ('_clf' in self.method) and not ('_multi' in self.method):
            if (self.results.get('val_results', None)) is not None:
                ax = cle.plot(self.results['val_results'], title=title)
                if return_ax: return ax
        elif ('_reg' in self.method):
            # fig, ax = plt.subplots(figsize=figsize)
            y_pred = self.predict(self.X_val, model=self.model)[0]
            df = pd.DataFrame(np.c_[self.y_val, y_pred], columns=['y', 'y_pred'])

            fig, ax = plt.subplots(figsize=figsize)
            sns.regplot('y', 'y_pred', data=df, ax=ax, color='k', label='Validation set')
            ax.legend()
            ax.grid(True)
            ax.set_title(title)
            ax.set_xlabel('True value')
            ax.set_ylabel('Predicted value')

        return ax

    def plot_params(self, top_n=10, shade=True, cmap='Set2', figsize=(18, 18), return_ax=False):
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
        # getcolors = colourmap.generate(top_n, cmap='Reds_r')
        ascending = False if self.greater_is_better else True

        # Sort data based on loss
        colname = 'loss'
        colbest = 'best'
        if self.cv is not None:
            colname_cv = 'loss_mean'
            colbest_cv = 'best_cv'
            colnames = [colname_cv, colname]
        else:
            colnames = colname

        # Sort on best loss
        df_summary = self.results['summary'].sort_values(by=colnames, ascending=ascending)
        # Get parameters for best scoring model
        idx_best = np.where(df_summary[colbest])[0]
        if self.cv is not None:
            idx_best_cv = np.where(df_summary[colbest_cv])[0]
        # Collect parameters
        params = np.array([*self.results['params'].keys()])
        color_params = colourmap.generate(len(params), cmap=cmap)
        # Setup figure size
        nrCols = 3
        nrRows = int(np.ceil(len(params) / 3))

        ################### Density plot for each parameter ##################
        fig, ax = plt.subplots(nrRows, nrCols, figsize=figsize)
        i_row = -1
        for i, param in enumerate(params):
            # Get row number
            i_col = np.mod(i, nrCols)
            # Make new column
            if i_col == 0: i_row = i_row + 1
            # Make density
            linefit = sns.distplot(self.results['summary'][param],
                                   hist=False,
                                   kde=True,
                                   rug=True,
                                   color='darkblue',
                                   kde_kws={'shade': shade, 'linewidth': 1, 'color': color_params[i, :]},
                                   rug_kws={'color': 'black'},
                                   ax=ax[i_row][i_col])

            y_data = linefit.get_lines()[0].get_data()[1]
            getvals = df_summary[param].values
            if len(y_data)>0:
                # Plot the top n (not the first because that one is plotted in green)
                ax[i_row][i_col].vlines(getvals[1:top_n], np.min(y_data), np.max(y_data), linewidth=1, colors='k', linestyles='dashed', label='Top ' + str(top_n) + ' models')
                # Plot the best (without cv)
                ax[i_row][i_col].vlines(getvals[idx_best], np.min(y_data), np.max(y_data), linewidth=2, colors='g', linestyles='dashed', label='Best (without cv)')
                # Plot the best (with cv)
                if self.cv is not None:
                    ax[i_row][i_col].vlines(getvals[idx_best_cv], np.min(y_data), np.max(y_data), linewidth=2, colors='r', linestyles='dotted', label='Best ' + str(self.cv) + '-fold cv')

            if self.cv is not None:
                ax[i_row][i_col].set_title(('%s: %.3g (%.0d-fold cv)' %(param, getvals[idx_best_cv], self.cv)))
            else:
                ax[i_row][i_col].set_title(('%s: %.3g' %(param, getvals[idx_best])))
            ax[i_row][i_col].set_ylabel('Density')
            ax[i_row][i_col].grid(True)
            ax[i_row][i_col].legend(loc='upper right')
            # print(param)
            # print(i_col)
            # print(i_row)

        ##################### Scatter plot #####################
        df_sum = self.results['summary'].sort_values(by='tid', ascending=True)
        idx_best = np.where(df_sum[colbest])[0]
        if self.cv is not None:
            idx_best_cv = np.where(df_sum[colbest_cv])[0]

        fig2, ax2 = plt.subplots(nrRows, nrCols, figsize=figsize)
        i_row = -1
        for i, param in enumerate(params):
            # Get row number
            i_col = np.mod(i, nrCols)
            # Make new column
            if i_col == 0: i_row = i_row + 1
            # Make the plot
            sns.regplot('tid', param, data=df_sum, ax=ax2[i_row][i_col], color=color_params[i, :])

            # Scatter top n values
            ax2[i_row][i_col].scatter(df_summary['tid'].values[1:top_n], df_summary[param].values[1:top_n], s=50, color='k', marker='.', label='Top ' + str(top_n) + ' models')

            # Scatter best value
            ax2[i_row][i_col].scatter(df_sum['tid'].values[idx_best], df_sum[param].values[idx_best], s=100, color='g', marker='*', label='Best (without cv)')

            # Scatter best cv
            if self.cv is not None:
                ax2[i_row][i_col].scatter(df_sum['tid'].values[idx_best_cv], df_sum[param].values[idx_best], s=100, color='r', marker='x', label='Best ' + str(self.cv) + '-fold cv')

            # Set labels
            ax2[i_row][i_col].set(xlabel = 'iteration', ylabel = '{}'.format(param), title = '{} over Search'.format(param))
            if self.cv is not None:
                ax2[i_row][i_col].set_title(('%s: %.3g (%.0d-fold cv)' %(param, df_sum[param].values[idx_best_cv], self.cv)))
            else:
                ax2[i_row][i_col].set_title(('%s: %.3g' %(param, df_sum[param].values[idx_best])))

            ax2[i_row][i_col].grid(True)
            ax2[i_row][i_col].legend(loc='upper right')

        if return_ax:
            return ax, ax2

    def plot(self, ylim=None, figsize=(15, 10), return_ax=False):
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
            print('[hgboost] >No model found. Hint: fit a model first using xgboost, catboost or lightboost <return>')
            return None
        ax1, ax2 = None, None

        if hasattr(self.model, 'evals_result'):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)

        tmpdf = self.results['summary'].sort_values(by='tid', ascending=True)

        # Plot results with testsize
        idx = np.where(tmpdf['best'].values)[0]
        ax1.hlines(tmpdf['loss'].iloc[idx], 0, tmpdf['loss'].shape[0], colors='g', linestyles='dashed', label='Best (wihtout cv)')
        ax1.vlines(idx, tmpdf['loss'].min(), tmpdf['loss'].iloc[idx], colors='g', linestyles='dashed')
        best_loss = tmpdf['loss'].iloc[idx]
        title = ('%s (%s: %.3g)' %(self.method, self.eval_metric, best_loss))

        # Plot results with cv
        if self.cv is not None:
            ax1.errorbar(tmpdf['tid'], tmpdf['loss_mean'], tmpdf['loss_std'], marker='s', mfc='red', label=str(self.cv) + '-fold cv for top ' + str(self.top_cv_evals) + ' models')
            idx = np.where(tmpdf['best_cv'].values)[0]
            ax1.hlines(tmpdf['loss_mean'].iloc[idx], 0, tmpdf['loss_mean'].shape[0], colors='r', linestyles='dotted', label='Best (' + str(self.cv) + '-fold cv)')
            ax1.vlines(idx, tmpdf['loss'].min(), tmpdf['loss_mean'].iloc[idx], colors='r', linestyles='dashed')
            best_loss = tmpdf['loss_mean'].iloc[idx]
            title = ('%s (%.0d-fold cv mean %s: %.3g)' %(self.method, self.cv, self.eval_metric, best_loss))
            ax1.set_xlabel('Model number')

        # Plot all other evalution results on the single test-set
        ax1.scatter(tmpdf['tid'].values, tmpdf['loss'].values, s=10, label='All models')

        # Set labels
        ax1.set_title(title)
        ax1.set_ylabel(self.eval_metric)
        ax1.grid(True)
        ax1.legend()
        if ylim is not None: ax1.set_ylim(ylim)

        if hasattr(self.model, 'evals_result'):
            eval_metric = [*self.model.evals_result()['validation_0'].keys()][0]
            ax2.plot([*self.model.evals_result()['validation_0'].values()][0], label='Train error')
            ax2.plot([*self.model.evals_result()['validation_1'].values()][0], label='Test error')
            ax2.set_ylabel(eval_metric)
            ax2.set_title(self.method)
            ax2.grid(True)
            ax2.legend()

        if return_ax:
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
        if verbose>=3: print('[hgboost] >Nothing to download.')
        return None

    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    PATH_TO_DATA = os.path.join(curpath, wget.filename_from_url(url))
    if not os.path.isdir(curpath):
        os.makedirs(curpath, exist_ok=True)

    # Check file exists.
    if not os.path.isfile(PATH_TO_DATA):
        if verbose>=3: print('[hgboost] >Downloading [%s] dataset from github source..' %(data))
        wget.download(url, curpath)

    # Import local dataset
    if verbose>=3: print('[hgboost] >Import dataset [%s]' %(data))
    df = pd.read_csv(PATH_TO_DATA, sep=sep)
    # Return
    return df


# %% Set the search spaces
def _get_params(fn_name, eval_metric=None, verbose=3):
    # choice : categorical variables
    # quniform : discrete uniform (integers spaced evenly)
    # uniform: continuous uniform (floats spaced evenly)
    # loguniform: continuous log uniform (floats spaced evenly on a log scale)
    early_stopping_rounds = 25
    if eval_metric is None: raise ValueError('[hgboost] >eval_metric must be provided.')
    if verbose>=3: print('[hgboost] >Collecting %s parameters.' %(fn_name))

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
        space = {}
        space['model_params'] = xgb_reg_params
        space['fit_params'] = {'early_stopping_rounds': early_stopping_rounds, 'verbose': False}
        # space['fit_params'] = {'eval_metric': eval_metric, 'early_stopping_rounds': 10, 'verbose': False}
        return(space)

    # LightGBM parameters
    if fn_name=='lgb_reg':
        lgb_reg_params = {
            'learning_rate' : hp.quniform('learning_rate', 0.05, 0.31, 0.05),
            'max_depth' : hp.choice('max_depth', np.arange(5, 30, 1, dtype=int)),
            'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
            'subsample': hp.uniform('subsample', 0.8, 1),
            'n_estimators' : hp.choice('n_estimators', range(20, 205, 5)),
        }
        space = {}
        space['model_params'] = lgb_reg_params
        space['fit_params'] = {'eval_metric': 'l2', 'early_stopping_rounds': early_stopping_rounds, 'verbose': False}

        return(space)

    # CatBoost regression parameters
    if fn_name=='ctb_reg':
        ctb_reg_params = {
            'learning_rate' : hp.quniform('learning_rate', 0.05, 0.31, 0.05),
            'max_depth' : hp.choice('max_depth', np.arange(2, 16, 1, dtype=int)),
            'colsample_bylevel' : hp.choice('colsample_bylevel', np.arange(0.3, 0.8, 0.1)),
            'n_estimators' : hp.choice('n_estimators', range(20, 205, 5)),
        }
        space = {}
        space['model_params'] = ctb_reg_params
        space['fit_params'] = {'early_stopping_rounds': early_stopping_rounds, 'verbose': False}
        return(space)

    # CatBoost classification parameters
    if fn_name=='ctb_clf':
        ctb_clf_params = {
            'learning_rate' : hp.choice('learning_rate', np.logspace(np.log10(0.005), np.log10(0.31), base = 10, num = 1000)),
            'depth' : hp.choice('max_depth', np.arange(2, 16, 1, dtype=int)),
            'iterations' : hp.choice('iterations', np.arange(100, 1000, 100)),
            'l2_leaf_reg' : hp.choice('l2_leaf_reg', np.arange(1, 100, 2)),
            'border_count' : hp.choice('border_count', np.arange(5, 200, 1)),
            'thread_count' : 4,
        }
        space = {}
        space['model_params'] = ctb_clf_params
        space['fit_params'] = {'early_stopping_rounds': early_stopping_rounds, 'verbose': False}
        return(space)

    # LightBoost classification parameters
    if fn_name=='lgb_clf':
        lgb_clf_params = {
            'learning_rate' : hp.choice('learning_rate', np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),
            'max_depth' : hp.choice('max_depth', np.arange(5, 75, 1)),
            'boosting_type' : hp.choice('boosting_type', ['gbdt','goss','dart']),
            'num_leaves' : hp.choice('num_leaves', np.arange(100, 1000, 100)),
            'n_estimators' : hp.choice('n_estimators', np.arange(20, 205, 5)),
            'subsample_for_bin' : hp.choice('subsample_for_bin', np.arange(20000, 300000, 20000)),
            'min_child_samples' : hp.choice('min_child_weight', np.arange(20, 500, 5)),
            'reg_alpha' : hp.quniform('reg_alpha', 0, 1, 0.01),
            'reg_lambda' : hp.quniform('reg_lambda', 0, 1, 0.01),
            'colsample_bytree' : hp.quniform('colsample_bytree', 0.6, 1, 0.01),
            'subsample' : hp.quniform('subsample', 0.5, 1, 100),
            'bagging_fraction' : hp.choice('bagging_fraction', np.arange(0.2, 1, 0.2)),
            'is_unbalance' : hp.choice('is_unbalance', [True, False]),
        }
        space = {}
        space['model_params'] = lgb_clf_params
        space['fit_params'] = {'early_stopping_rounds': early_stopping_rounds, 'verbose': False}
        return(space)

    if 'xgb_clf' in fn_name:
        xgb_clf_params = {
            'learning_rate' : hp.choice('learning_rate', np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),
            'max_depth' : hp.choice('max_depth', range(5, 75, 1)),
            'min_child_weight' : hp.quniform('min_child_weight', 1, 10, 1),
            'gamma' : hp.choice('gamma', [0.5, 1, 1.5, 2, 5]),
            'subsample' : hp.quniform('subsample', 0.1, 1, 0.01),
            'n_estimators' : hp.choice('n_estimators', range(20, 205, 5)),
            'booster' : 'gbtree',
            'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1.0, 0.01),
        }

        if fn_name=='xgb_clf':
            # xgb_clf_params['eval_metric'] = hp.choice('eval_metric', ['error', eval_metric])
            xgb_clf_params['objective'] = 'binary:logistic'
            xgb_clf_params['scale_pos_weight'] = hp.choice('scale_pos_weight', [0, 0.5, 1])

        if fn_name=='xgb_clf_multi':
            xgb_clf_params['objective']='multi:softprob'
            # scoring='kappa'

        space = {}
        space['model_params'] = xgb_clf_params
        space['fit_params'] = {'early_stopping_rounds': early_stopping_rounds, 'verbose': False}

        if verbose>=3: print('[hgboost] >Number of variables in search space is [%.0d], loss function: [%s].' %(len([*space['model_params']]), eval_metric))
        return(space)


def _check_input(X, y, pos_label, method, verbose=4):
    # X should be of type dataframe
    if (type(X) is not pd.DataFrame):
        raise ValueError('[hgboost] >Error: dataset X should be of type pd.DataFrame')

    # y should be of type array-like
    if (type(y) is not np.ndarray):
        raise ValueError('[hgboost] >Error: Response variable y should be of type numpy array')

    # Check None and np.nan values in string/numeric type
    if 'str' in str(type(y[0])):
        if any(elem is None for elem in y): raise ValueError('[hgboost] >Error: Response variable y can not have None values.')
    else:
        if np.any(np.isnan(y)): raise ValueError('[hgboost] >Error: Response variable y can not have nan values.')

    if ('_reg' in method):
        pos_label = None

    # Set pos_label and y
    if (pos_label is not None) and ('_clf' in method):
        if verbose>=4: print('[hgboost] >pos_label is used to set [%s].' %(pos_label))
        y = y==pos_label
        pos_label=True

    # Checks pos_label status in case of method is classification
    if ('_clf' in method) and (pos_label is None) and (str(y.dtype)=='bool'):
        pos_label = True
        if verbose>=4: print('[hgboost] >[pos_label] is set to [%s] because [y] is of type [bool].' %(pos_label))

    # Raise ValueError in case of pos_label is not set and not bool.
    if ('_clf' in method) and (pos_label is None) and (len(np.unique(y))==2) and not (str(y.dtype)=='bool'):
        raise ValueError('[hgboost] >Error: In a two-class approach [%s], pos_label needs to be set or of type bool.' %(pos_label))

    # Check label for classificaiton and two-class model
    if ('_clf' in method) and (pos_label is not None) and (len(np.unique(y))==2) and not (np.any(np.isin(y.astype(str), str(pos_label)))):
        raise ValueError('[hgboost] >Error: y contains values %s but none matches pos_label=%s <return>' %(str(np.unique(y)), pos_label))

    # two-class classifier should have 2 classes
    if ('_clf' in method) and not ('_multi' in method) and (len(np.unique(y))>2) and (pos_label is None):
        raise ValueError('[hgboost] >Error: [y] should contain exactly 2 unique classes. Hint: use method="xgb_clf_multi"')

    # multi-class method should have >2 classes
    if ('_clf_multi' in method) and (len(np.unique(y))<=2):
        raise ValueError('[hgboost] >Error: [xgb_clf_multi] requires >2 classes. Hint: use method="xgb_clf"')

    if ('_clf_multi' in method):
        pos_label = None
        if verbose>=4: print('[hgboost] >[pos_label] is set to [None] because [method] is of type [%s].' %(method))

    # Check counts y
    y_counts = np.unique(y, return_counts=True)[1]
    if np.any(y_counts<5) and ('_clf' in method):
        raise ValueError('[hgboost] >Error: [y] contains [%.0d] classes with < 5 samples. Each class should have >=5 samples.' %(sum(y_counts<5)))
    # Check number of classes, should be >1
    if (len(np.unique(y))<=1) and ('_clf' in method):
        raise ValueError('[hgboost] >Error: [y] should have >= 2 classes.')

    # Set X
    X.reset_index(drop=True, inplace=True)
    X.columns = X.columns.values.astype(str)
    if verbose>=4: print('[hgboost] >Reset index for X.')

    # Return
    return X, y, pos_label


def _check_eval_metric(method, eval_metric, greater_is_better, verbose=3):
    # Check the eval_metric
    if (eval_metric is None) and ('_reg' in method):
        eval_metric = 'rmse'
    elif (eval_metric is None) and ('_clf_multi' in method):
        eval_metric = 'kappa'
    elif (eval_metric is None) and ('_clf' in method):
        eval_metric = 'auc'

    # Check the greater_is_better for evaluation metric
    if greater_is_better is None:
        if (eval_metric == 'f1'):
            greater_is_better = True
        elif (eval_metric == 'auc'):
            greater_is_better = True
        elif (eval_metric == 'kappa'):
            greater_is_better = True
        elif (eval_metric == 'rmse'):
            greater_is_better = False
        elif (eval_metric == 'mae'):
            greater_is_better = False
        else:
            if verbose>=2: print('[hgboost] >[%s] is not a implemented option. [greater_is_better] is set to %s' %(eval_metric, str(greater_is_better)))

    # Return
    return eval_metric, greater_is_better
