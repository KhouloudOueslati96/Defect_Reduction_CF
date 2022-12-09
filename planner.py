from othertools import *
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from XTREE import XTREE
import random
from sklearn.preprocessing import MinMaxScaler


def RandomWalk(data_row, number):
    tem = data_row.copy()
    result = [[0 for m in range(2)] for n in range(20)]
    lis = list(np.arange(20))
    act = random.sample(lis, number)
    rec = [0] * 20
    for j in range(0, len(tem)):
        if j in act:
            rec[j] = 1
            num1 = np.random.rand(1)[0]
            num2 = np.random.rand(1)[0]
            if (num1 <= num2 and tem[j] != 0) or tem[j] == 1:
                result[j][0], result[j][1] = 0, tem[j] - 0.05
            else:
                result[j][0], result[j][1] = tem[j] + 0.05, 1
            tem[j] = (num1 + num2) / 2
        else:
            result[j][0], result[j][1] = tem[j] - 0.05, tem[j] + 0.05
    return tem, result, rec


def RW(name, par, explainer=None, smote=False, small=0.05, act=False, number=5):
    files = [name[0], name[1], name[2]]
    freq = [0] * 20
    deltas = []
    for j in range(0, len(files) - 2):
        df1 = prepareData(files[j])
        df2 = prepareData(files[j + 1])
        for i in range(1, 21):
            col1 = df1.iloc[:, i]
            col2 = df2.iloc[:, i]
            deltas.append(hedge(col1, col2))
    #             if not (hedge(col1,col2,small)):
    #                 freq[i-1]+=1
    deltas = sorted(range(len(deltas)), key=lambda k: deltas[k], reverse=True)
    #     changed = dict()
    #     for i in range(20):
    #         freq[i] = 100*freq[i]/(len(files)-1)
    #         changed.update({df1.columns[i]:freq[i]})
    #     changed = list(changed.values())
    #     actionable = []
    #     for each in changed:
    #         actionable.append(1) if each!=0 else actionable.append(0)
    actionable = []
    for i in range(0, len(deltas)):
        if i in deltas[0:5]:
            actionable.append(1)
        else:
            actionable.append(0)
    print(actionable)
    df1 = prepareData(name[0])
    df2 = prepareData(name[1])
    df3 = prepareData(name[2])
    bug1 = bugs(name[0])
    bug2 = bugs(name[1])
    bug3 = bugs(name[2])
    df11 = df1.iloc[:, 1:]
    df22 = df2.iloc[:, 1:]
    df33 = df3.iloc[:, 1:]

    df1n = norm(df11, df11)
    df2n = norm(df22, df22)
    df3n = norm(df22, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    score = []
    bugchange = []
    size = []
    score2 = []
    matrix = []
    para = 20
    clf1 = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=10, n_jobs=-1)
    #     clf1 =  MLPClassifier(hidden_layer_sizes=[10],max_iter=2000,early_stopping=False,learning_rate='adaptive')
    if smote:
        sm = SMOTE()
        X_train1_s, y_train1_s = sm.fit_resample(X_train1, y_train1)
        clf1.fit(X_train1_s, y_train1_s)
    else:
        clf1.fit(X_train1, y_train1)
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            actual = X_test2.values[j]
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                print('df2', i, 'df3', j)
                temp = X_test1.values[i].copy()
                tem, plan, rec = RandomWalk(temp, number)
                score.append(overlap(plan, actual))
                score2.append(overlap(plan, X_test1.values[i]))
                size.append(size_interval(plan))
                bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added
                tp, tn, fp, fn = abcd(temp, plan, actual, rec)
                print("tp,tn,fp,fn:", tp, tn, fp, fn)
                matrix.append([tp, tn, fp, fn])
                break
    print(name[0])
    print('>>>')
    print('>>>')
    print('>>>')
    return score, bugchange, size, score2, matrix


def planner(name, par, explainer=None, smote=False, small=0.05, act=False):
    # classic LIME
    start_time = time.time()
    files = [name[0], name[1], name[2]]
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
    print(actionable)
    df1 = prepareData(name[0])
    df2 = prepareData(name[1])
    df3 = prepareData(name[2])
    bug1 = bugs(name[0])
    bug2 = bugs(name[1])
    bug3 = bugs(name[2])
    df11 = df1.iloc[:, 1:]
    df22 = df2.iloc[:, 1:]
    df33 = df3.iloc[:, 1:]

    df1n = norm(df11, df11)
    df2n = norm(df22, df22)
    df3n = norm(df22, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    score = []
    bugchange = []
    size = []
    score2 = []
    records = []
    matrix = []
    par = 20
    clf1 = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=10, n_jobs=-1)
    #     clf1 =  MLPClassifier(hidden_layer_sizes=[10],max_iter=2000,early_stopping=False,learning_rate='adaptive')
    if smote:
        sm = SMOTE()
        X_train1_s, y_train1_s = sm.fit_resample(X_train1, y_train1)
        clf1.fit(X_train1_s, y_train1_s)
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train1_s, training_labels=y_train1_s,
                                                           feature_names=df11.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='classification')
    else:
        clf1.fit(X_train1, y_train1)
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train1.values, training_labels=y_train1,
                                                           feature_names=df11.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='classification')
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            actual = X_test2.values[j]
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                #                 print('df2',i,'df3',j)
                #                 if clf1.predict([X_test1.values[i]])==0:
                if True:
                    ins = explainer.explain_instance(data_row=X_test1.values[i], predict_fn=clf1.predict_proba,
                                                     num_features=20,
                                                     num_samples=5000)
                    ind = ins.local_exp[1]
                    temp = X_test1.values[i].copy()
                    if act:
                        tem, plan, rec = flip(temp, ins.as_list(label=1), ind, clf1, df1n.columns, par,
                                              actionable=actionable)
                    else:
                        tem, plan, rec = flip(temp, ins.as_list(label=1), ind, clf1, df1n.columns, par, actionable=None)
                    score.append(overlap(plan, actual))
                    size.append(size_interval(plan))
                    score2.append(overlap(plan, temp))
                    records.append(rec)
                    tp, tn, fp, fn = abcd(temp, plan, actual, rec)
                    print("tp,tn,fp,fn:", tp, tn, fp, fn)
                    matrix.append([tp, tn, fp, fn])
                    bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added
                break
    print("Runtime:", time.time() - start_time)
    print(name[0], par)
    print('>>>')
    print('>>>')
    print('>>>')
    return score, bugchange, size, score2, records, matrix


def TL(name,par,rules,smote=False,act=False):
    start_time = time.time()
    files = [name[0], name[1], name[2]]
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
    print(actionable)
    df1 = prepareData(name[0])
    df2 = prepareData(name[1])
    df3 = prepareData(name[2])
    bug1 = bugs(name[0])
    bug2 = bugs(name[1])
    bug3 = bugs(name[2])
    df11 = df1.iloc[:, 1:]
    df22 = df2.iloc[:, 1:]
    df33 = df3.iloc[:, 1:]

    df1n = norm(df11, df11)
    df2n = norm(df22, df22)
    df3n = norm(df22, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    score = []
    bugchange = []
    size = []
    score2 = []
    records = []
    matrix = []
    seen = []
    seen_id = []
    par = 20
    clf1 = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=10, n_jobs=-1)
    #     clf1 =  MLPClassifier(hidden_layer_sizes=[10],max_iter=2000,early_stopping=False,learning_rate='adaptive')
    #     clf1 = SVC(gamma='auto',probability=True)
    if smote:
        sm = SMOTE()
        X_train1_s, y_train1_s = sm.fit_resample(X_train1, y_train1)
        clf1.fit(X_train1_s, y_train1_s)
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train1_s, training_labels=y_train1_s,
                                                           feature_names=df11.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='classification')
    else:
        clf1.fit(X_train1, y_train1)
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train1.values, training_labels=y_train1,
                                                           feature_names=df11.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='classification')
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            actual = X_test2.values[j]
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                #                 print('df2',i,'df3',j)
                #                 if clf1.predict([X_test1.values[i]])==0:
                if True:
                    ins = explainer.explain_instance(data_row=X_test1.values[i], predict_fn=clf1.predict_proba,
                                                     num_features=20,
                                                     num_samples=5000)
                    ind = ins.local_exp[1]
                    temp = X_test1.values[i].copy()
                    if act:
                        tem, plan, rec = flip(temp, ins.as_list(label=1), ind, clf1, df1n.columns, par,
                                              actionable=actionable)
                    else:
                        tem, plan, rec = flip(temp, ins.as_list(label=1), ind, clf1, df1n.columns, par, actionable=None)
                    if act:
                        if rec in seen_id:
                            supported_plan_id = seen[seen_id.index(rec)]
                            print("Seen.")
                        else:
                            #                             if seen_id:
                            #                                 for i in range(len(seen_id)):
                            #                                     print(rec == seen_id[i])
                            supported_plan_id = find_supported_plan(rec, rules, top=5)
                            seen_id.append(rec.copy())
                            seen.append(supported_plan_id)
                            print("Not seen.", rec)
                            print("seen_id", seen_id)

                        for k in range(len(rec)):
                            if rec[k] != 0:
                                if (k not in supported_plan_id) and ((0 - k) not in supported_plan_id):
                                    plan[k][0], plan[k][1] = tem[k] - 0.05, tem[k] + 0.05
                                    rec[k] = 0

                    score.append(overlap(plan, actual))
                    size.append(size_interval(plan))
                    score2.append(len([n for n in rec if n != 0]))
                    records.append(rec)
                    tp, tn, fp, fn = abcd(temp, plan, actual, rec)
                    # if act:
                    #     print("Supported:", supported_plan_id)
                    # print("tp,tn,fp,fn:", tp, tn, fp, fn)
                    print("")
                    matrix.append([tp, tn, fp, fn])
                    bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added
                break
    print("Runtime:", time.time() - start_time)
    print(name[0], par)
    print('>>>')
    print('>>>')
    print('>>>')
    return score, bugchange, size, score2, records, matrix


def _ent_weight(X, scale):
    try:
        loc = X["loc"].values  # LOC is the 10th index position.
    except KeyError:
        try:
            loc = X["$WCHU_numberOfLinesOfCode"].values
        except KeyError:
            loc = X["$CountLineCode"]

    return X.multiply(loc, axis="index") / scale



def runalves(name, thresh=0.7):
    df1 = prepareData(name[0])
    df2 = prepareData(name[1])
    df3 = prepareData(name[2])
    bug1 = bugs(name[0])
    bug2 = bugs(name[1])
    bug3 = bugs(name[2])
    df11 = df1.iloc[:, 1:]
    df22 = df2.iloc[:, 1:]
    df33 = df3.iloc[:, 1:]

    df1n = norm(df11, df11)
    df2n = norm(df22, df22)
    df3n = norm(df22, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    plans, recs = alves(X_train1, X_test1, y_test1, thresh=thresh)
    score = []
    score2 = []
    bugchange = []
    size = []
    matrix=[]
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            actual = X_test2.values[j]
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                print('df2', i, 'df3', j)
                temp = X_test1.values[i].copy()
                plan = plans.iloc[i, :].values
                rec = recs[i]
                #                 print('actual',actual)
                #                 print('id1',plan[0][0])
                score.append(overlap(plan, actual))
                score2.append(overlap(plan, X_test1.values[i]))
                bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added
                size.append(size_interval(plan))
                tp, tn, fp, fn = abcd(temp, plan, actual, rec)
                print("tp,tn,fp,fn:", tp, tn, fp, fn)
                matrix.append([tp, tn, fp, fn])
                break
    print(name[0])
    print('>>>')
    print('>>>')
    print('>>>')
    return score, bugchange, size, score2, matrix




def runshat(name, p=0.05):
    df1 = prepareData(name[0])
    df2 = prepareData(name[1])
    df3 = prepareData(name[2])
    bug1 = bugs(name[0])
    bug2 = bugs(name[1])
    bug3 = bugs(name[2])
    df11 = df1.iloc[:, 1:]
    df22 = df2.iloc[:, 1:]
    df33 = df3.iloc[:, 1:]

    df1n = norm(df11, df11)
    df2n = norm(df22, df22)
    df3n = norm(df22, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    plans, recs = shatnawi(X_train1, y_train1, X_test1, y_test1, p=p)
    score = []
    score2 = []
    bugchange = []
    size = []
    matrix=[]
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            actual = X_test2.values[j]
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                print('df2', i, 'df3', j)
                temp = X_test1.values[i].copy()
                plan = plans.iloc[i, :].values
                rec = recs[i]
                #                 print("plan",plan)
                #                 print('actual',actual)
                #                 print('id1',plan[0][0])
                score.append(overlap(plan, actual))
                score2.append(overlap(plan, X_test1.values[i]))
                bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added
                size.append(size_interval(plan))
                tp, tn, fp, fn = abcd(temp, plan, actual, rec)
                print("tp,tn,fp,fn:", tp, tn, fp, fn)
                matrix.append([tp, tn, fp, fn])
                break
    print(name[0])
    print('>>>')
    print('>>>')
    print('>>>')
    return score, bugchange, size, score2, matrix


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


def runolive(name):
    df1 = prepareData(name[0])
    df2 = prepareData(name[1])
    df3 = prepareData(name[2])
    bug1 = bugs(name[0])
    bug2 = bugs(name[1])
    bug3 = bugs(name[2])
    df11 = df1.iloc[:, 1:]
    df22 = df2.iloc[:, 1:]
    df33 = df3.iloc[:, 1:]

    df1n = norm(df11, df11)
    df2n = norm(df22, df22)
    df3n = norm(df22, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    plans, recs = oliveira(X_train1, X_test1)
    score = []
    bugchange = []
    size = []
    score2 = []
    matrix=[]
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            actual = X_test2.values[j]
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                print('df2', i, 'df3', j)
                temp = X_test1.values[i].copy()
                plan = plans.iloc[i, :].values
                rec = recs[i]
                #                 print("plan",plan)
                #                 print('actual',actual)
                #                 print('id1',plan[0][0])
                score.append(overlap(plan, actual))
                bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added
                size.append(size_interval(plan))
                score2.append(overlap(plan, X_test1.values[i]))
                tp, tn, fp, fn = abcd(temp, plan, actual, rec)
                print("tp,tn,fp,fn:", tp, tn, fp, fn)
                matrix.append([tp, tn, fp, fn])
                break
    print(name[0])
    print('>>>')
    print('>>>')
    print('>>>')
    return score, bugchange, size, score2, matrix

def xtree(name):
    start = time.time()
    df1 = prepareData(name[0])
    df2 = prepareData(name[1])
    df3 = prepareData(name[2])
    bug1 = bugs(name[0])
    bug2 = bugs(name[1])
    bug3 = bugs(name[2])
    df11 = df1.iloc[:, 1:]
    df22 = df2.iloc[:, 1:]
    df33 = df3.iloc[:, 1:]
    df1n = norm(df11,df11)
    df2n = norm(df11,df22)
    df3n = norm(df11,df33)
    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]
    X_test = pd.concat([X_test1, y_test1], axis=1, ignore_index=True)
    X_test.columns = df1.columns[1:]

    xtree_arplan = XTREE(strategy="closest",alpha=0.95,support_min=int(X_train1.shape[0]/20))
    xtree_arplan = xtree_arplan.fit(X_train1)
    patched_xtree = xtree_arplan.predict(X_test)
    print("Runtime for Xtree:", time.time()-start)
    print(patched_xtree.shape[0], X_test1.shape[0])
    XTREE.pretty_print(xtree_arplan)
    overlap_scores = []
    bcs = []
    size=[]
    score2=[]
    matrix=[]
    for i in range(0, X_test1.shape[0]):
        for j in range(0, X_test2.shape[0]):
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                temp = X_test1.values[i].copy()
                plan = patched_xtree.iloc[i, :-1]
                rec=[0 for n in range(20)]
                for k in range(20):
                    if not isinstance(plan[k], float):
                        if plan[k][0]!=plan[k][1]:
                            rec[k]=1
                # print('**********')
                # print('plan:',plan.values)
                # print('ori:',X_test1.iloc[i,:].values)
                # print('**********')
                actual = X_test2.iloc[j, :]
                overlap_scores.append(overlap1(plan, plan, actual))
                bcs.append(bug3[j]-bug2[i])
                score2.append(overlap(X_test1.values[i], plan, X_test1.values[i]))
                size.append(size_interval(plan))
                tp, tn, fp, fn = abcd(temp, plan, actual, rec)
                print("tp,tn,fp,fn:", tp, tn, fp, fn)
                matrix.append([tp, tn, fp, fn])
                break
    return overlap_scores,bcs,size,score2,matrix


def historical_logs(files, par, explainer=None, smote=False, act=True):
    start_time = time.time()
    deltas = []
    df1 = prepareData(files[0])
    df2 = prepareData(files[1])
    df3 = prepareData(files[2])
    for i in range(1, 21):
        col1 = df1.iloc[:, i]
        col2 = df2.iloc[:, i]
        deltas.append(hedge(col1, col2))
    deltas = np.argsort(-np.array(deltas))

    actionable = []
    for i in range(0, len(deltas)):
        if i in deltas[0:5]:
            actionable.append(1)
        else:
            actionable.append(0)
    print(actionable)
    
    # Too complicated for no reason here ...
    bug1 = bugs(files[0])
    bug2 = bugs(files[1])
    bug3 = bugs(files[2])
    df11 = df1.iloc[:, 1:]
    df22 = df2.iloc[:, 1:]
    df33 = df3.iloc[:, 1:]

    df1n = norm(df11, df11)
    df2n = norm(df11, df22)
    df3n = norm(df11, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    old_change = []
    new_change = []
    par = 0
    clf1 = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=10, n_jobs=-1)
    #     clf1 =  MLPClassifier(hidden_layer_sizes=[10],max_iter=2000,early_stopping=False,learning_rate='adaptive')
    if smote:
        sm = SMOTE()
        X_train1_s, y_train1_s = sm.fit_resample(X_train1, y_train1)
        clf1.fit(X_train1_s, y_train1_s)
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train1_s.values, training_labels=y_train1_s,
                                                           feature_names=df11.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='classification', sample_around_instance=True)
    else:
        clf1.fit(X_train1, y_train1)
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train1.values, training_labels=y_train1,
                                                           feature_names=df11.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='classification', sample_around_instance=True)
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                actual = X_test2.values[j]
                ins = explainer.explain_instance(data_row=X_test1.values[i], predict_fn=clf1.predict_proba,
                                                 num_features=20,
                                                 num_samples=5000)
                ind = ins.local_exp[1]
                temp = X_test1.values[i].copy()
                if act:
                    tem, plan, rec = flip(temp, ins.as_list(label=1), ind, clf1, df1n.columns, 0, actionable=actionable)
                else:
                    tem, plan, rec = flip(temp, ins.as_list(label=1), ind, clf1, df1n.columns, 0, actionable=None)
                o = track1(plan, temp)
                n = track1(plan, actual)
                old_change.append(o)
                new_change.append(n)
                # print(" ", o)
                # print(" ", n)

                break
    print("Runtime:", time.time() - start_time)
    print(files[0], par)
    print('>>>')
    print('>>>')
    print('>>>')
    return old_change, new_change

