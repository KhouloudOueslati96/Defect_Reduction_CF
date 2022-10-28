import os, sys
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from othertools import *


# DiCE imports
import dice_ml
from dice_ml.utils import helpers


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
    sm = SMOTE(random_state=42)
    X_tain_res, y_train_res = sm.fit_resample(X_train, y_train)

    # TODO train a random forest with balanced X_train, y_train
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_tain_res, y_train_res)
    # TODO Compute counterfactuals on each instance from the test set X_test
    train_dataset = X_train.copy()
    train_dataset['bug']=y_train
    
    

    d = dice_ml.Data(dataframe=train_dataset, continuous_features=X_tain_res.columns.to_list(), outcome_name='bug')
    m = dice_ml.Model(model=clf, backend="sklearn")

    # Using method=random for generating CFs
    exp = dice_ml.Dice(d, m, method="genetic")
    #for i in X_test.index:
       
    #    e1 = exp.generate_counterfactuals(X_test[X_test.index==i], total_CFs=1, desired_class="opposite")
    #    print(e1.cf_examples_list[0].final_cfs_df)
        
    e1 = exp.generate_counterfactuals(X_test, total_CFs=1, desired_class="opposite")
    
    
    modified = []
    tmp_ = []

     # store them in the pd.DataFrame called modified

    for i in range(len(e1.cf_examples_list)):
        df_tmp = e1.cf_examples_list[i].final_cfs_df
        tmp_.append(df_tmp)
    
    tmp_ = pd.concat(tmp_).reset_index(drop=True)
    

    for i in range(tmp_.shape[0]):
        tmp_array = []
        for c in train_dataset.columns:
            if c =='bug':
                tmp_array.append(tmp_[c].iloc[i])

            elif tmp_[c].iloc[i] <= X_test[c].iloc[i]:
                tmp_array.append((0, tmp_[c].iloc[i]))

            else:
                tmp_array.append((tmp_[c].iloc[i], X_test[c].max()))

        modified.append(tmp_array)
    modified = pd.DataFrame(modified, columns =train_dataset.columns)
    print(modified)
    return modified
  


 
"""     for i in range(len(e1.cf_examples_list)):
        print("index is {}".format(i))
        df_tmp = e1.cf_examples_list[i].final_cfs_df
        tmp_array = []
        for c in train_dataset.columns:
            print(df_tmp[c][0])

            if c =='bug':
                tmp_array.append(df_tmp[c][0])

            elif df_tmp[c][0] <= X_test[c].iloc[i]:
                tmp_array.append((0, df_tmp[c][0]))

            else:
                tmp_array.append((df_tmp[c][0], X_test[c].max()))

        modified.append(tmp_array)
        
    modified = pd.DataFrame(modified, columns =df_tmp.columns)
    print(modified) """
    
    

def get_actionable_set(train, test, validation):

    start_time = time.time()
    files = [train, test, validation]
    freq = [0] * 20
    deltas = []
    for j in range(0, len(files) - 2):
        df1 = prepareData(files[j])
        df2 = prepareData(files[j + 1])
        for i in range(1, 21):
            col1 = df1.iloc[:, i]
            col2 = df2.iloc[:, i]
            deltas.append(hedge(col1, col2))
    deltas = sorted(range(len(deltas)), key=lambda k: deltas[k], reverse=True)

    actionable = []
    for i in range(0, len(deltas)):
        if i in deltas[0:5]:
            actionable.append(1)
        else:
            actionable.append(0)

    return actionable

def focused_cfe(X_train, y_train, X_test, actionable_set):
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
    sm = SMOTE(random_state=42)
    X_tain_res, y_train_res = sm.fit_resample(X_train, y_train)

    # TODO train a random forest with balanced X_train, y_train
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_tain_res, y_train_res)
    # TODO Compute counterfactuals on each instance from the test set X_test
    train_dataset = X_train.copy()
    train_dataset['bug']=y_train
    
    

    d = dice_ml.Data(dataframe=train_dataset, continuous_features=X_tain_res.columns.to_list(), outcome_name='bug')
    m = dice_ml.Model(model=clf, backend="sklearn")

    # Using method=genetic for generating CFs
    exp = dice_ml.Dice(d, m, method="genetic")
    
    
    e1 = exp.generate_counterfactuals(X_test, total_CFs=1, desired_class="opposite", features_to_vary= actionable_set)
    
    
    # store them in the pd.DataFrame called modified
    modified = []
    for i in range(len(e1.cf_examples_list)):
        df_tmp = e1.cf_examples_list[i].final_cfs_df
        modified.append(df_tmp)

        #tmp_array = []
        #for c in df_tmp.columns:    
        #    if c =='bug':
         #       tmp_array.append(df_tmp[c][0])

          #  elif df_tmp[c][0] <= X_test[c].iloc[i]:
           #     tmp_array.append((0, df_tmp[c][0]))

            #else:
            #    tmp_array.append((df_tmp[c][0], X_test[c].max()))


        #modified.append(tmp_array)
    
    modified = pd.concat(modified).reset_index(drop=True)
    print(modified)
    #modified = pd.DataFrame(modified, columns =df_tmp.columns)
    return modified