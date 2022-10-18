import os, sys
import numpy as np
import pandas as pd

# Update path
root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from plan_utils import apply_threshold

def get_percentiles(df):
    percentile_array = []
    q = dict()
    for i in np.arange(0, 100, 1):
        for col in df.columns:
            try:
                q.update({col: np.percentile(df[col].values, q=i)})
            except:
                pass

        elements = dict()
        for col in df.columns:
            try:
                elements.update({col: df.loc[df[col] >= q[col]].median()[col]})
            except:
                pass

        percentile_array.append(elements)

    return percentile_array



def oliveira(X_train, X_test):
    """
    Implements shatnavi's threshold based planner.
    :param train:
    :param test:
    :param rftrain:
    :param tunings:
    :param verbose:
    :return:
    """
    "Helper Functions"

    def compliance_rate(k, train_columns):
        return len([t for t in train_columns if t <= k]) / len(train_columns)

    def penalty_1(Min, compliance):

        comply = Min - compliance
        if comply >= 0:
            return (Min - compliance) / Min
        else:
            return 0

    def penalty_2(k, Med):
        if k > Med:
            return (k - Med) / Med
        else:
            return 0
    
    lo, hi = X_train.min(0), X_train.max(0)
    quantile_array = get_percentiles(X_train)

    pk_best = dict()

    metrics = X_train.columns
    for metric in metrics:
        min_comply = 10e32
        vals = np.empty([10, 100])
        for p_id, p in enumerate(np.arange(0, 100, 10)):
            p = p / 100
            for k_id, k in enumerate(np.linspace(lo[metric], hi[metric], 100)):
                try:
                    med = quantile_array[90][metric]
                    compliance = compliance_rate(k, X_train[metric])
                    penalty1 = penalty_1(compliance=compliance, Min=0.9)
                    penalty2 = penalty_2(k, med)
                    comply_rate_penalty = penalty1 + penalty2
                    vals[p_id, k_id] = comply_rate_penalty

                    if (comply_rate_penalty < min_comply) or (
                            comply_rate_penalty == min_comply and 
                            p >= pk_best[metric][0] and k <= pk_best[metric][1]):
                        min_comply = comply_rate_penalty
                        try:
                            pk_best[metric] = (p, k)
                            #print('p k best', p, k)
                        except KeyError:
                            pk_best.update({metric: (p, k)})
                except:
                    pk_best.update({metric: (p, None)})
        #print('comply', metric, min_comply)

    modified = []
    for i in range(X_test.shape[0]):
        new_row = apply_threshold(X_test.iloc[i], pk_best)
        modified.append(new_row)

    return pd.DataFrame(modified, columns=X_test.columns)