import os, sys
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

# Update path
root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)



def basic_cfe(X_train, y_train, X_test):
    """
    Returns the CounterFactual plan

    Parameters
    --------------------
    X_train : (N_train, d) pd.DataFrame 
        The training inputs used to compute the thresholds
    y_train : (N_train,) pd.DataFrame 
        The training targets used to train a ML model
    X_test : (N_test, d) pd.DataFrame
        The test inputs on which to compute the counterfactuals

    Returns
    --------------------
    modified : (N_test, d) pd.DataFrame
        The action to take on each test input.
        A row could take the form [0.34   (0, 56)   0.68] meaning we recommend to 
        keep x1 and x3 intact and modify x2 so it lies in the range (0, 56).
    """

    # TODO rebalance the training set with SMOTE

    # TODO train a random forest with balanced X_train, y_train

    # TODO Compute counterfactuals on each instance from the test set X_test
    # store them in the pd.DataFrame called modified
    modified = []
    
    return modified
