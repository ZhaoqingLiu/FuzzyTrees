"""
@author: Zhaoqing Liu
@email : Zhaoqing.Liu-1@student.uts.edu.au
@date  : 21/4/21 11:29 am
@desc  :
"""
import multiprocessing
import os
import time

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from exp_params import ComparisionMode, NUM_CPU_CORES, DS_LOAD_FUNC_CLF, FUZZY_TH, FUZZY_STRIDE
from fuzzy_trees.fuzzy_decision_tree import FuzzyDecisionTreeClassifier
from fuzzy_trees.fuzzy_decision_tree_api import FuzzificationParams, FuzzyDecisionTreeClassifierAPI, CRITERIA_FUNC_CLF, \
    CRITERIA_FUNC_REG
from fuzzy_trees.fuzzy_gbdt import FuzzyGBDTClassifier
from fuzzy_trees.util_data_processing_funcs import extract_fuzzy_features
import fuzzy_trees.util_plotter as plotter


def search_optimum_fuzzy_th(conn_send, comparing_mode, dataset_name, fuzzy_th):
    # Load all data sets.
    ds_df = load_dataset_clf(dataset_name)

    # Execute the experiment according to the parameters.
    X = ds_df.iloc[:, :-1].values
    y = ds_df.iloc[:, -1].values
    accuracy_train_list, accuracy_test_list = get_exp_results_clf(X, y, comparing_mode=comparing_mode, dataset_name=dataset_name, fuzzy_th=fuzzy_th)
    accuracy_train_mean = np.mean(accuracy_train_list)
    accuracy_test_mean = np.mean(accuracy_test_list)

    # Put the result in "queue".
    conn_send.send([[fuzzy_th, accuracy_train_mean], [fuzzy_th, accuracy_test_mean]])


def load_dataset_clf(dataset_name):
    """
    Load data by a dataset name.

    Parameters
    ----------
    dataset_name: a key in exp.params.DS_LOAD_FUNC_CLF

    Returns
    -------
    data: DataFrame
    """
    ds_load_func = None

    if dataset_name in DS_LOAD_FUNC_CLF.keys():
        ds_load_func = DS_LOAD_FUNC_CLF[dataset_name]

    return None if ds_load_func is None else ds_load_func()


def get_exp_results_clf(X, y, comparing_mode, dataset_name, fuzzy_th):
    accuracy_train_list = []
    accuracy_test_list = []

    # Preprocess features for using fuzzy decision tree.
    X_fuzzy_pre = X.copy()
    fuzzification_params = None
    X_dms = None
    if comparing_mode is ComparisionMode.FF3:
        fuzzification_params = FuzzificationParams(conv_k=3)
        X_dms = extract_fuzzy_features(X_fuzzy_pre, conv_k=3, fuzzy_th=fuzzy_th)
    elif comparing_mode is ComparisionMode.FF4:
        fuzzification_params = FuzzificationParams(conv_k=4)
        X_dms = extract_fuzzy_features(X_fuzzy_pre, conv_k=4, fuzzy_th=fuzzy_th)
    elif comparing_mode is ComparisionMode.FF5:
        fuzzification_params = FuzzificationParams(conv_k=5)
        X_dms = extract_fuzzy_features(X_fuzzy_pre, conv_k=5, fuzzy_th=fuzzy_th)
    else:
        fuzzification_params = FuzzificationParams(conv_k=5)
        # - Step 1: Standardise feature scaling.
        # X_fuzzy_pre[:, :] -= X_fuzzy_pre[:, :].min()
        # X_fuzzy_pre[:, :] /= X_fuzzy_pre[:, :].max()
        # - Step 2: Extract fuzzy features.
        X_dms = extract_fuzzy_features(X_fuzzy_pre, conv_k=5, fuzzy_th=fuzzy_th)
    X_plus_dms = np.concatenate((X, X_dms), axis=1)
    print("************* X_plus_dms's shape:", np.shape(X_plus_dms))

    for i in range(10):
        print("%ith comparison on %s" % (i, dataset_name))

        # Split training and test sets by hold-out partition method.
        # X_train, X_test, y_train, y_test = train_test_split(X_fuzzy_pre, y, test_size=0.4)

        kf = KFold(n_splits=2, random_state=i, shuffle=True)
        for train_index, test_index in kf.split(X):
            y_train, y_test = y[train_index], y[test_index]

            # Using a fuzzy decision tree. =============================================================================
            X_train, X_test = X_plus_dms[train_index], X_plus_dms[test_index]
            accuracy_train, accuracy_test = use_fuzzy_trees(comparing_mode=comparing_mode, X_train=X_train, X_test=X_test, y_train=y_train,
                                       y_test=y_test, fuzzification_params=fuzzification_params)
            accuracy_train_list.append(accuracy_train)
            accuracy_test_list.append(accuracy_test)

    print("========================================================================================")
    print(comparing_mode.value, " - ", dataset_name)
    print("Fuzzy:  10-round-mean accuracy:", np.mean(accuracy_train_list), "  std:",
          np.std(accuracy_train_list))
    print("Non-fuzzy:  10-round-mean accuracy:", np.mean(accuracy_test_list), "  std:",
          np.std(accuracy_test_list))
    print("========================================================================================")

    return accuracy_train_list, accuracy_test_list


def use_fuzzy_trees(comparing_mode, X_train, X_test, y_train, y_test, fuzzification_params):
    time_start = time.time()  # Record the start time.

    clf = None
    if comparing_mode is ComparisionMode.NAIVE:
        # My NDT vs. sklearn NDT
        clf = FuzzyDecisionTreeClassifierAPI(fdt_class=FuzzyDecisionTreeClassifier, disable_fuzzy=True,
                                             fuzzification_params=fuzzification_params,
                                             criterion_func=CRITERIA_FUNC_CLF["gini"], max_depth=5)
    elif comparing_mode is ComparisionMode.FF3 or comparing_mode is ComparisionMode.FF4 or comparing_mode is ComparisionMode.FF5:
        # With only Feature Fuzzification vs. NDT
        clf = FuzzyDecisionTreeClassifierAPI(fdt_class=FuzzyDecisionTreeClassifier, disable_fuzzy=True,
                                             fuzzification_params=fuzzification_params,
                                             criterion_func=CRITERIA_FUNC_CLF["gini"], max_depth=5)
    elif comparing_mode is ComparisionMode.FUZZY:
        # FDT vs. NDT
        clf = FuzzyDecisionTreeClassifierAPI(fdt_class=FuzzyDecisionTreeClassifier, disable_fuzzy=False,
                                             fuzzification_params=fuzzification_params,
                                             criterion_func=CRITERIA_FUNC_CLF["gini"], max_depth=5)
    elif comparing_mode is ComparisionMode.BOOSTING:
        # Gradient boosting FDT vs. Gradient boosting NDT
        clf = FuzzyGBDTClassifier(disable_fuzzy=False, fuzzification_params=fuzzification_params,
                                  criterion_func=CRITERIA_FUNC_REG["mse"], learning_rate=0.1, n_estimators=100,
                                  max_depth=5)
    elif comparing_mode is ComparisionMode.MIXED:
        # Gradient boosting Mixture of FDT and NDT vs. Gradient boosting NDT
        clf = FuzzyGBDTClassifier(disable_fuzzy=False, fuzzification_params=fuzzification_params,
                                  is_trees_mixed=True, criterion_func=CRITERIA_FUNC_REG["mse"], learning_rate=0.1,
                                  n_estimators=100,
                                  max_depth=5)  # TODO: Add one more argument "is_trees_mixed", default=False.
    clf.fit(X_train, y_train)
    # clf.print_tree()
    y_pred_train = clf.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    y_pred_test = clf.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print("    Fuzzy accuracy train:", accuracy_train)
    print("    Fuzzy accuracy test:", accuracy_test)
    print('    Elapsed time (FDT-based):', time.time() - time_start, 's')  # Display the time of training a model.

    return accuracy_train, accuracy_test


"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
In master-worker mode, if an execution in the master process depends on the
results of all the child processes, place its code after either Pool().join() 
or Process().join(). 
Never put it in a child process because the child Processes are parallel to 
each other, so you don't know if the other child processes are terminated if
you are in one of the child processes.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
if __name__ == '__main__':
    time_start = time.time()
    print("Main Process (%s) started." % os.getpid())

    # Create a connection between processes.
    # NB: When using Pool create processes, use multiprocessing.Manager().Queue()
    # instead of multiprocessing.Queue() to create connection.
    conn_recv, conn_send = multiprocessing.Pipe()

    # Create a pool containing n (0 - infinity) processes.
    # If the parameter "processes" is None then the number returned by os.cpu_count() is used.
    # Make sure that n is <= the number of CPU cores available.
    # The parameters to the Pool indicate how many parallel processes are called to run the program.
    # The default size of the Pool is the number of compute cores on the CPU, i.e. multiprocessing.cpu_count().
    pool = multiprocessing.Pool(NUM_CPU_CORES)

    # Complete all tasks by the pool.
    # !!! NB: If you want to complete the experiment faster, you can use distributed computing. Or you can divide
    # the task into k groups to execute in k py programs, and then run one on each of k clusters simultaneously.
    for ds_name in DS_LOAD_FUNC_CLF.keys():
        # TODO: 1st task: Searching an optimum fuzzy threshold by a loop according the specified stride.
        while FUZZY_TH <= 0.5:
            # Add a process into the pool. apply_async() is asynchronous equivalent of "apply()" builtin.
            err = pool.apply_async(search_optimum_fuzzy_th, args=(conn_send, ComparisionMode.FUZZY, ds_name, FUZZY_TH,))
            FUZZY_TH += FUZZY_STRIDE

    pool.close()
    pool.join()

    # Read the result and prepare the data for plotting.
    # Read result from the queue "q" and save them.
    x_list_train = []
    y_list_train = []
    x_list_test = []
    y_list_test = []
    while conn_recv.poll():
        res_list = conn_recv.recv()
        x_list_train.append(res_list[0][0])
        y_list_train.append(res_list[0][1])
        x_list_test.append(res_list[1][0])
        y_list_test.append(res_list[1][1])
        print(x_list_train)
        print(y_list_train)
        print(x_list_test)
        print(y_list_test)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # TODO: If needed, sort fuzzy_th_list in ascending order, and then sort accuracy_list in the same order as fuzzy_th_list.
    # Because multiple processes may not return results in an ascending order of fuzzy thresholds.
    print("before sorting:")
    print(x_list_train)
    x_train = sorted(x_list_train)
    y_train = [y for _, y in sorted(zip(x_list_train, y_list_train))]
    print("after sorting:")
    print(y_train)
    x_test = sorted(x_list_test)
    y_test = [y for _, y in sorted(zip(x_list_test, y_list_test))]

    # TODO: Plot the comparison of training error versus test error, and both curves are fuzzy thresholds versus accuracies.
    # Illustrate how the performance on unseen data (test data) is different from the performance on training data.
    assert (len(x_train) > 0 and len(y_train) > 0)
    x_lower_limit, x_upper_limit = np.min(x_train), np.max(x_train)
    y_lower_limit = np.min(y_train) if np.min(y_train) < np.min(y_test) else np.min(y_test)
    y_upper_limit = np.max(y_train) if np.max(y_train) > np.max(y_test) else np.max(y_test)
    print("x_limits and y_limits are:", x_lower_limit, x_upper_limit, y_lower_limit, y_upper_limit)
    plotter.plot_multi_curves(x=[x_train, x_test],
                              y=[y_train, y_test],
                              title="Training Error vs Test Error",
                              x_label="Fuzzy threshold",
                              y_label="Performance",
                              x_limit=(x_lower_limit, x_upper_limit),
                              y_limit=(y_lower_limit - (y_lower_limit / 100), y_upper_limit + (y_upper_limit / 100)),
                              legends=["Train", "Test"])

    print("Total elapsed time: {:.5}s".format(time.time() - time_start))
    print("Main Process (%s) ended." % os.getpid())
