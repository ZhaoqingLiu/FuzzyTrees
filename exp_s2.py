"""
@author: Zhaoqing Liu
@email : Zhaoqing.Liu-1@student.uts.edu.au
@date  : 21/4/21 11:29 am
@desc  :
"""
import multiprocessing
import time

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from exp_params import ComparisionMode, NUM_CPU_CORES, DATASET_NAMES
from fuzzy_trees.fuzzy_decision_tree import FuzzyDecisionTreeClassifier
from fuzzy_trees.fuzzy_decision_tree_api import FuzzificationParams, FuzzyDecisionTreeClassifierAPI, CRITERIA_FUNC_CLF, \
    CRITERIA_FUNC_REG
from fuzzy_trees.fuzzy_gbdt import FuzzyGBDTClassifier
from fuzzy_trees.util_data_processing_funcs import extract_fuzzy_features


def exec_exp_clf(comparing_mode=ComparisionMode.FUZZY, dataset_name="Vehicle"):
    print("******** Task - ", dataset_name, " ********")

    result_df = pd.DataFrame()

    # Load all data sets.
    X_list, y_list, dataset_name_list = load_dataset_classification(dataset_name)
    # Iterate all data sets, and execute the function of the experiment in each iteration.
    for X, y, dataset_name in zip(X_list, y_list, dataset_name_list):
        fuzzy_accuracy_list, naive_accuracy_list = get_exp_results_clf(X, y, comparing_mode=comparing_mode,
                                                                       dataset_name=dataset_name)

        fuzzy_mean = np.mean(fuzzy_accuracy_list)
        fuzzy_std = np.std(fuzzy_accuracy_list)
        naive_mean = np.mean(naive_accuracy_list)
        naive_std = np.std(naive_accuracy_list)
        result_df[dataset_name] = [fuzzy_mean, naive_mean, fuzzy_std, naive_std]

    # Finally, output the results of the experiment.
    result_df.to_csv("exp_results_" + comparing_mode.value + "_" + dataset_name + ".csv")


def get_exp_results_clf(X, y, comparing_mode=ComparisionMode.FUZZY, dataset_name="Vehicle"):
    fuzzy_accuracy_list = []
    naive_accuracy_list = []

    # Preprocess features for using fuzzy decision tree.
    X_fuzzy_pre = X.copy()
    fuzzification_params = None
    X_dms = None
    if comparing_mode is ComparisionMode.FF3:
        fuzzification_params = FuzzificationParams(conv_k=3)
        X_dms = extract_fuzzy_features(X_fuzzy_pre, conv_k=3)
    elif comparing_mode is ComparisionMode.FF4:
        fuzzification_params = FuzzificationParams(conv_k=4)
        X_dms = extract_fuzzy_features(X_fuzzy_pre, conv_k=4)
    elif comparing_mode is ComparisionMode.FF5:
        fuzzification_params = FuzzificationParams(conv_k=5)
        X_dms = extract_fuzzy_features(X_fuzzy_pre, conv_k=5)
    else:
        fuzzification_params = FuzzificationParams(conv_k=5)
        # - Step 1: Standardise feature scaling.
        # X_fuzzy_pre[:, :] -= X_fuzzy_pre[:, :].min()
        # X_fuzzy_pre[:, :] /= X_fuzzy_pre[:, :].max()
        # - Step 2: Extract fuzzy features.
        X_dms = extract_fuzzy_features(X_fuzzy_pre, conv_k=5)
    X_plus_dms = np.concatenate((X, X_dms), axis=1)
    print("************* X_plus_dms's shape:", np.shape(X_plus_dms))

    for i in range(10):
        print("%ith comparison" % i)

        # Split training and test sets by hold-out partition method.
        # X_train, X_test, y_train, y_test = train_test_split(X_fuzzy_pre, y, test_size=0.4)

        kf = KFold(n_splits=10, random_state=i, shuffle=True)
        for train_index, test_index in kf.split(X):
            y_train, y_test = y[train_index], y[test_index]

            # Using a fuzzy decision tree. =============================================================================
            X_train, X_test = X_plus_dms[train_index], X_plus_dms[test_index]
            accuracy = use_fuzzy_trees(comparing_mode=comparing_mode, X_train=X_train, X_test=X_test, y_train=y_train,
                                       y_test=y_test, fuzzification_params=fuzzification_params)
            fuzzy_accuracy_list.append(accuracy)

            # Using a naive decision tree. =============================================================================
            X_train, X_test = X[train_index], X[test_index]
            accuracy = use_naive_trees(comparing_mode=comparing_mode, X_train=X_train, X_test=X_test, y_train=y_train,
                                       y_test=y_test)
            naive_accuracy_list.append(accuracy)

    print("========================================================================================")
    print(comparing_mode.value, " - ", dataset_name)
    print("Fuzzy:  10-round-mean accuracy:", np.mean(fuzzy_accuracy_list), "  std:",
          np.std(fuzzy_accuracy_list))
    print("Non-fuzzy:  10-round-mean accuracy:", np.mean(naive_accuracy_list), "  std:",
          np.std(naive_accuracy_list))
    print("========================================================================================")

    return fuzzy_accuracy_list, naive_accuracy_list


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
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("    Fuzzy accuracy:", accuracy)

    print('    Elapsed time (FDT-based):', time.time() - time_start, 's')  # Display the time of training a model.

    return accuracy


def use_naive_trees(comparing_mode, X_train, X_test, y_train, y_test):
    time_start = time.time()  # Record the start time.

    clf = None
    if comparing_mode is ComparisionMode.NAIVE:
        # My NDT vs. sklearn NDT
        clf = DecisionTreeClassifier(criterion="gini", max_depth=5)
    elif comparing_mode is ComparisionMode.FF3 or comparing_mode is ComparisionMode.FF4 or comparing_mode is ComparisionMode.FF5:
        # With only Feature Fuzzification vs. NDT
        clf = FuzzyDecisionTreeClassifierAPI(fdt_class=FuzzyDecisionTreeClassifier, disable_fuzzy=True,
                                             criterion_func=CRITERIA_FUNC_CLF["gini"], max_depth=5)
    elif comparing_mode is ComparisionMode.FUZZY:
        # FDT vs. NDT
        clf = FuzzyDecisionTreeClassifierAPI(fdt_class=FuzzyDecisionTreeClassifier, disable_fuzzy=True,
                                             criterion_func=CRITERIA_FUNC_CLF["gini"], max_depth=5)
    elif comparing_mode is ComparisionMode.BOOSTING:
        # Gradient boosting FDT vs. Gradient boosting NDT
        clf = FuzzyGBDTClassifier(disable_fuzzy=True, criterion_func=CRITERIA_FUNC_REG["mse"], learning_rate=0.1,
                                  n_estimators=100, max_depth=5)
    elif comparing_mode is ComparisionMode.MIXED:
        # Gradient boosting Mixture of FDT and NDT vs. Gradient boosting NDT
        clf = FuzzyGBDTClassifier(disable_fuzzy=True, is_trees_mixed=True, criterion_func=CRITERIA_FUNC_REG["mse"],
                                  learning_rate=0.1, n_estimators=100,
                                  max_depth=5)
    clf.fit(X_train, y_train)
    # clf.print_tree()
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("    Naive accuracy:", accuracy)

    print('    Elapsed time (NDT-based):', time.time() - time_start, 's')  # Display the time of training a model.

    return accuracy


def load_dataset_classification(dataset_name):
    """
    Load all data sets for classification experiments.

    NB: Before version 1.0 (upgraded by Cython to speed up), use data sets Iris, Wine,
        don't use Breast Cancer (spending nearly 17s/model), Digits (spending nearly 10s/model).

    Returns
    -------
    X_list: list of {array-like, sparse matrix} of shape (n_samples, n_features)
        A list of the features of samples.

    y_list: list of {array-like, sparse matrix} of shape (n_samples, 1)
        A list of the labels of samples.

    dataset_name_list: list of shape (n_data_sets, 1)
        A list of the names of data sets.
    """
    X_list = []
    y_list = []
    dataset_name_list = []

    if dataset_name == "Iris":
        dataset = datasets.load_iris()
        X = dataset.data
        y = dataset.target
        X_list.append(X)
        y_list.append(y)
        dataset_name_list.append("Iris")
    elif dataset_name == "Wine":
        dataset = datasets.load_wine()
        X = dataset.data
        y = dataset.target
        X_list.append(X)
        y_list.append(y)
        dataset_name_list.append("Wine")

    return X_list, y_list, dataset_name_list


if __name__ == '__main__':
    # 1st method: Using multiprocessing.Pool.
    time_start = time.time()

    # Create a pool containing n processes. Make sure that n is <= the number of CPU cores available.
    # The parameters to the Pool indicate how many parallel processes are called to run the program.
    # The default size of the Pool is the number of compute cores on the CPU, i.e. multiprocessing.cpu_count().
    pool = multiprocessing.Pool(NUM_CPU_CORES)

    # Complete all tasks by the pool.
    # !!! NB: If you want to complete the experiment faster, you can use distributed computing. Or you can divide
    # the task into k groups to execute in k py programs, and then run one on each of k clusters simultaneously.
    for ds_name in DATASET_NAMES:
        # Add a process into the pool. apply_async() is asynchronous equivalent of "apply()" builtin.
        pool.apply_async(exec_exp_clf, args=(ComparisionMode.FUZZY, ds_name,))
    pool.close()
    pool.join()

    print("Total elapsed time: {:.5}s".format(time.time() - time_start))

    # =====================================================================================================
    # # 2nd method: Using multiprocessing.Process purely, not multiprocessing.Pool.
    # # Create multiple worker processes (The system marks __main__ as the master process by default),
    # # and each process execute an experiment on one dataset.
    # p1 = multiprocessing.Process(target=exec_exp_clf, args=(ComparisionMode.FUZZY, "Iris",))
    # p2 = multiprocessing.Process(target=exec_exp_clf, args=(ComparisionMode.FUZZY, "Wine",))
    # p3 = multiprocessing.Process(target=exec_exp_clf, args=(ComparisionMode.FUZZY, "Vehicle",))
    # p4 = multiprocessing.Process(target=exec_exp_clf, args=(ComparisionMode.FUZZY, "German_Credit",))
    # p5 = multiprocessing.Process(target=exec_exp_clf, args=(ComparisionMode.FUZZY, "Diabetes",))
    #
    # # Start all processes.
    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()
    # p5.start()
    #
    # # Join all processes to the system resource request queue.
    # p1.join()
    # p2.join()
    # p3.join()
    # p4.join()
    # p5.join()
