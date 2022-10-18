"""
Deriving Metric Thresholds from Benchmark Data

Alves, T. L., Ypma, C., & Visser, J. (2010). Deriving metric thresholds from
benchmark data. In ICSM'10 (pp. 1-10). http://doi.org/10.1109/ICSM.2010.5609747
"""

import os
import sys

# Update path
root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif



def compute_weighted_CDF(X, threshold, loc_key="loc"):
    """
    Returns the weights CDFs and threshold for each code metric

    Parameters
    --------------------
        X: (N, d) pd.DataFrame 
            The training data
        threshold : float
            Fraction in (0, 1) of the total LOC we consider risky.
        loc_key : str  
            The column name of the LOC metric

    Returns
    --------------------
        complexity_thresh : dict 
            dict["metric"] returns the metric threshold
        CDFs : dict
            dict["metric"] returns the metric weighted CDF
        sorted_idx : dict
            dict["metric"] returns the indices that would sort the metric
    """
    weights = []
    metrics = X.columns
    
    # Weight entitied by their source lines of code (LOC)
    tot_loc = X.sum()[loc_key]
    weights = X[loc_key] / tot_loc

    # Compute the weighted CDF of the metric
    CDFs = {}
    idx_sorted = {}
    for metric in metrics:
        # Sort the metric value  lowest -> largest
        idx_sorted[metric] = np.argsort(X[metric])
        # The weighted CDF is the cummulative sum of weighs
        CDFs[metric] = np.cumsum(weights[ idx_sorted[metric] ])

    # Find the threshold for complexity
    complexity_thresh = {}
    for metric in metrics:
        # Argmax returns the first index where the value is 1
        idx_cutoff = np.argmax(CDFs[metric] >= threshold / 100)
        # Map back to the unsorted idx
        idx_cutoff = idx_sorted[metric][idx_cutoff]
        complexity_thresh[metric] = X[metric][ idx_cutoff ]

    return complexity_thresh, CDFs, idx_sorted



def apply_threshold(X_row, complexity_thresh):
    new_row = X_row.to_list()
    metrics = complexity_thresh.keys()
    for i, metric in enumerate(metrics):
        threshold = complexity_thresh[metric]
        if threshold is not None:
            try:
                if new_row[i] > threshold:
                    new_row[i] = (0, threshold)
            except:
                pass
    return new_row



def alves(X_train, X_test, threshold):
    """
    Returns the Alves plan

    Parameters
    --------------------
        X_train : (N_train, d) pd.DataFrame 
            The training inputs
        X_test : (N_test, d) pd.DataFrame
            The test inputs
        threshold : float
            Fraction between (0, 1) of total LOC we consider risky.

    Returns
    --------------------
        modified : (N_test, d) pd.DataFrame
            The action to take on each test input.
            A row could take the form [0.34   (0, 56)   0.68] meaning we recomment to 
            keep x1 and x3 intact and modify x2 so it lies in the range (0, 56).
    """
    complexity_thresh, _, _ = compute_weighted_CDF(X_train, threshold)

    # TODO dont modify feature with a p-value > 0.5.
    #pVal = f_classif(train.iloc[:, :-1], train.iloc[:, -1])[1]  # P-Values

    modified = []
    for n in range(X_test.shape[0]):
        new_row = apply_threshold(X_test.iloc[n], complexity_thresh)
        modified.append(new_row)

    return pd.DataFrame(modified, columns=X_test.columns)


if __name__ == "__main__":
    from data.get_data import get_all_projects
    import matplotlib.pyplot as plt

    # Load data
    all_projects = get_all_projects()
    data_files = all_projects["lucene"]
    train_df = pd.read_csv(data_files[0])
    test_df = pd.read_csv(data_files[1])

    # Test the weighted CDF
    complexity_thresh, CDFs, idx_sorted = compute_weighted_CDF(train_df.iloc[:, 1:-1], 90)

    for metric in train_df.columns[1:-1]:
        metric_val = train_df[metric]
        plt.figure()
        plt.step(metric_val[ idx_sorted[metric] ], CDFs[metric], 'b', where='post')
        plt.xticks(ticks=np.linspace(metric_val.min(), metric_val.max(), 10))
        plt.plot([metric_val.min(), complexity_thresh[metric] ], [0.9, 0.9], 'k--')
        plt.plot([complexity_thresh[metric], complexity_thresh[metric] ], [0, 0.9], 'k--')
        plt.xlabel(metric)
        plt.ylabel("Ratio of LOC")
        plt.ylim(0, 1)

    # Test the plan
    modified_test_df = alves(train_df.iloc[:, 1:-1], test_df.iloc[:, 1:-1], 90)
    plt.show()
