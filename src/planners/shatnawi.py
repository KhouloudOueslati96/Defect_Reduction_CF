import os, sys
from plan_utils import apply_threshold
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from imblearn.over_sampling import SMOTE

# Update path
root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from plan_utils import apply_threshold



def VARL(coef, inter, p0=0.05):
    """
    :param coef: Slope of   (Y=aX+b)
    :param inter: Intercept (Y=aX+b)
    :param p0: Confidence Interval. Default p=0.05 (95%)
    :return: VARL threshold

              1   /     /   p0   \             \
    VARL = ------| log | -------  | - intercept |
           slope  \     \ 1 - p0 /             /

    """
    return float( (np.log(p0 / (1 - p0)) - inter) / coef )



def compute_univariate_log_reg(X, y, p):
    """
    Fit a logistic regression on each individual code metric

    Parameters
    --------------------
    X: (N, d) pd.DataFrame
        The training input data
    y: (N,) pd.DataFrame
        The training targets
    loc_key : float
        Value in (0, 1) representing the maximum acceptable 
        conditional probability of defect `P[y=1|x_i=x]`.

    Returns
    --------------------
    complexity_thresh : dict 
        dict["metric"] returns the metric threshold
    coeff : dict
        dict["metric"] returns the slope
    inter : dict
        dict["metric"] returns the intercept

    """
    metrics = X.columns
    inter = {}
    coeff = {}
    pVal = {}
    for metric in metrics:
        smote = SMOTE(sampling_strategy='minority', random_state=1, k_neighbors=2)
        X_aug, y_aug = smote.fit_resample(X[[metric]], y)
        ubr = LogisticRegression(solver='lbfgs')  # Init LogisticRegressor
        ubr.fit(X_aug, y_aug)  # Fit Logit curve
        inter[metric] = ubr.intercept_[0]  # Intercepts
        coeff[metric] = ubr.coef_[0]  # Slopes
        pVal[metric] = f_classif(X_aug, y_aug)[1]  # P-Values
    complexity_thresh = {}
    "Find Thresholds using VARL"
    for metric in metrics:
        if pVal[metric] < 0.1:
            thresh = VARL(coeff[metric], inter[metric], p0=p)
            if thresh > X[metric].min():
                complexity_thresh[metric] = thresh
            else:
                complexity_thresh[metric] = None
        else:
            complexity_thresh[metric] = None

    return complexity_thresh, coeff, inter



def shatnawi(X_train, y_train, X_test, p=0.05):
    """
    Returns the Shatnawi plan

    Parameters
    --------------------
    X_train : (N_train, d) pd.DataFrame 
        The training inputs used to compute the thresholds
    y_train : (N_train,) pd.DataFrame 
        The training targets used to compute the thresholds
    X_test : (N_test, d) pd.DataFrame
        The test inputs on which to apply the plan
    p : float, default=0.05
        Parameter of the approach

    Returns
    --------------------
    modified : (N_test, d) pd.DataFrame
        The action to take on each test input.
        A row could take the form [0.34   (0, 56)   0.68] meaning we recommend to 
        keep x1 and x3 intact and modify x2 so it lies in the range (0, 56).
    """

    complexity_thresh, _, _ = compute_univariate_log_reg(X_train, y_train, p)
    modified = []
    for i in range(X_test.shape[0]):
        new_row = apply_threshold(X_test.iloc[i], complexity_thresh)
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

    # Test the Logistic regression
    X_train = train_df.iloc[:, 1:-1]
    y_train = (train_df.iloc[:, -1]>0).astype(int)
    X_test = test_df.iloc[:, 1:-1]
    X_train_s = X_train.apply(lambda x: (x-x.mean())/ x.std(), axis=0)
    X_test_s = X_test.apply(lambda x: (x-x.mean())/ x.std(), axis=0)
    complexity_thresh, slope, intercept = \
        compute_univariate_log_reg(X_train_s, y_train, p=0.75)

    for metric in train_df.columns[1:-1]:
        metric_val = X_train_s[metric]
        plt.figure()
        # Plot the histograms
        _, _, rects1 = plt.hist(metric_val[train_df.iloc[:, -1]==0], density=True, color="b", alpha=0.25)
        _, _, rects2 = plt.hist(metric_val[train_df.iloc[:, -1]==1], density=True, color="r", alpha=0.25)
        max_height = max([h.get_height() for h in np.concatenate((rects1, rects2))])   
        # Plot the logistic function
        linspace = np.linspace(metric_val.min(), metric_val.max(), 100)
        probability = 1 / (1 + np.exp(-1 * (slope[metric] * linspace + intercept[metric])))
        plt.plot(linspace, probability * max_height, 'k-')

        plt.xticks(ticks=np.linspace(metric_val.min(), metric_val.max(), 10))
        if complexity_thresh[metric] is not None:
            thresh = 0.75 * max_height
            plt.plot([metric_val.min(), complexity_thresh[metric] ], [thresh, thresh], 'k--')
            plt.plot([complexity_thresh[metric], complexity_thresh[metric] ], [0, thresh], 'k--')
        plt.xlabel(metric)
        plt.ylabel("Probability")


    # Test the plan
    modified_test_df = shatnawi(X_train_s, y_train, X_test_s, p=0.75)
    plt.show()
