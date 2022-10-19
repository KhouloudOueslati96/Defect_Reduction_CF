import os
import sys
import re
# Update PYTHONPATH
root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

import pandas as pd
from pandas import read_csv, concat


TAXI_CAB = 1729
OVERLAP_RANGE = range(0, 101, 25)

def get_dataframes(filename, binarize=False):
    df = read_csv(filename)
    filenames = df["Name"]
    if binarize:
        df.loc[df['<bug'] > 0, '<bug'] = 1
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    return filenames, X, y



def _effectiveness(dframe, thresh_min, thresh_max):
    overlap = dframe['Overlap']
    heeded = dframe['Heeded']

    a, b, c, d = 0, 0, 0, 0

    for over, heed in zip(overlap, heeded):
        if thresh_min < over <= thresh_max and heed >= 0:
            a += 1
        if thresh_min <= over <= thresh_max and heed < 0:
            b += 1
        if over == 0 and heed < 0:
            c += 1
        if over == 0 and heed >= 0:
            d += 1

    return a, b, c, d



import numpy as np

def overlap(plan,actual): # Jaccard similarity function
    cnt = 20
    right = 0
    # print(plan)
    for i in range(0,len(plan)):
        if isinstance(plan[i], float):
            if np.round(actual[i],4)== np.round(plan[i],4):
                right+=1
        else:
            if actual[i]>=0 and actual[i]<=1:
                if actual[i]>=plan[i][0] and actual[i]<=plan[i][1]:
                    right+=1
            elif actual[i]>1:
                if plan[i][1]>=1:
                    right+=1
            else:
                if plan[i][0]<=0:
                    right+=1
    return right/cnt


def measure_overlap(plan_X, files_test, X_test, y_test, files_valid, X_valid, y_valid):
    """ 
    Measure how much the plan overlaps with actual code changes 
    from test data to valid data
    """
    results = pd.DataFrame()

    # Find modules that appear both in test and validation datasets
    common_modules = list(set(files_test).intersection(set(files_valid)))

    # Intitialize variables to hold information
    improve_heeded = []
    overlap = []

    for module_name in common_modules:
        same = 0  # Keep track of features that follow our recommendations
        # Metric values of classes in the test set
        test_value = X_test.loc[files_test == module_name]
        # Metric values of classes in the planned changes
        plan_value = plan_X.loc[files_test == module_name]
        # Actual metric values the developer's changes yielded
        valid_value = X_valid.loc[files_valid == module_name]

        metrics = X_test.columns
        for metric in metrics:
            try:
                if isinstance(plan_value[metric].values[0], str):
                    # The change recommended lie in a range of values
                    if eval(plan_value[metric].values[0])[0] <= valid_value[metric].values[0] <= \
                            eval(plan_value[metric].values[0])[1]:
                        # If the actual change lies withing the recommended change, then increment the count by 1
                        same += 1
                if isinstance(plan_value[metric].values[0], tuple):
                    # The change recommended lie in a range of values
                    if plan_value[metric].values[0][0] <= valid_value[metric].values[0] <= plan_value[metric].values[0][1]:
                        # If the actual change lies withing the recommended change, then increment the count by 1
                        same += 1
                elif np.round(plan_value[metric].values[0], 4) == np.round(valid_value[metric].values[0], 4):
                    # If we recommend no change and developers didn't change anything, that also counts as an overlap"
                    same += 1

            except IndexError:
                # Catch instances where classes don't match
                pass

        # Find % of overlap for the class.
        overlap.append(int(same / len(metrics) * 100))
        # Variation in the number of bugs between the test and validation versions for that module
        heeded = y_test.loc[files_test == module_name].values[0] - \
                 y_valid.loc[files_valid == module_name].values[0]
        improve_heeded.append(heeded)
    
    # "Save the results ... "
    # validation_common = validation.loc[validation["Name"].isin(common_modules)]

    results['Module'] = common_modules
    results['Overlap'] = overlap
    results['Heeded'] = improve_heeded

    return results#, [_effectiveness(results, thresh_min=lo, thresh_max=hi) for lo, hi in zip(OVERLAP_RANGE[:-1], OVERLAP_RANGE[1:])]
    # return [tuple(map(lambda x: int(100*x/len(validation_common['<bug'].tolist())), _effectiveness(results, thresh_min=lo, thresh_max=hi))) for lo, hi in zip(OVERLAP_RANGE[:-1], OVERLAP_RANGE[1:])]
