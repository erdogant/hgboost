# Import the library
from hgboost import hgboost
import unittest
import matplotlib
import clusteval
import numpy as np
matplotlib.use('Agg')  # Use non-interactive backend for tests

class TestHGboost(unittest.TestCase):

    def test_init(self):        
        # Import the library
        from hgboost import hgboost
        
        # Initialize library.
        hgb = hgboost(
            max_eval=25,      # Search space is based  on the number of evaluations.
            threshold=0.5,     # Classification threshold. In case of two-class model this is 0.5.
            cv=5,              # k-folds cross-validation.
            test_size=0.2,     # Percentage split for the testset.
            val_size=0.2,      # Percentage split for the validationset.
            top_cv_evals=10,   # Number of top best performing models that is evaluated.
            is_unbalance=True, # Control the balance of positive and negative weights, useful for unbalanced classes.
            random_state=None, # Fix the random state to create reproducible results.
            n_jobs=-1,         # The number of CPU jobs to run in parallel. -1 means using all processors.
            gpu=False,         # Compute using GPU in case of True.
            verbose='info',         # Print progress to screen.
        )
        
        ###########################################################################
        # Import example and preprocessing
        df = hgb.import_example()
        y = df['Age'].values
        df.drop(['Age', 'PassengerId', 'Name'], axis=1, inplace=True)
        
        # Preprocessing
        X = hgb.preprocessing(df)
        I = ~np.isnan(y)
        X = X.loc[I, :]
        y = y[I]
        
        ###########################################################################
        # Fit model for a regression task.
        hgb.xgboost_reg(X, y)
        hgb.catboost_reg(X, y)
        hgb.lightboost_reg(X, y)
        ###########################################################################
        # Fit ensemble model for classification task.
        hgb.ensemble(X, y, pos_label=1, methods=['xgb_clf', 'ctb_clf', 'lgb_clf'])
        ###########################################################################
        # Fit ensemble model for regression task.
        hgb.ensemble(X, y, methods=['xgb_reg', 'ctb_reg', 'lgb_reg'])
        ###########################################################################