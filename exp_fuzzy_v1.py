# _*_coding:utf-8_*_
"""
@author: Zhaoqing Liu
@email: Zhaoqing.Liu-1@student.uts.edu.au
@date: 29/01/2021 9:39 pm
@desc:

TODO:
    1. Done -> Do feature fuzzification in data_preprocessing_funcs.py
    2. Done -> Add function implementing splitting criteria fuzzification in util_criterion_funcs.py
    3. Add data sets Covertype, Pokerhand, Sarcos, Mushroom, Rossmann Store Sales, ......
        What the requirements of citation? e.g. We first consider the simple task of mushroom edibility prediction (Dua and Graff 2017), and Rossmann Store Sales (Kaggle 2019b).
    4. Upgrade program from Python to Cython to speed up the algorithms. (See 005_cython in project Machine-Learning-Algorithms)
"""
import time
from enum import Enum

from sklearn import datasets
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier

from fuzzytrees.fdt_base import FuzzificationParams, FuzzyDecisionTreeWrapper, CRITERIA_FUNC_CLF, CRITERIA_FUNC_REG
from fuzzytrees.fdts import FuzzyCARTClassifier, FuzzyCARTRegressor
from fuzzytrees.fgbdt import FuzzyGBDTClassifier
from fuzzytrees.util_criterion_funcs import calculate_mse, calculate_mae
from fuzzytrees.util_data_processing_funcs import extract_fuzzy_features
import fuzzytrees.util_data_handler as dh


class ComparisionMode(Enum):
    NAIVE = "my_naive_vs_sklearn_naive"
    FF3 = "ff3_vs_naive"  # With only Feature Fuzzification, conv_k=3
    FF4 = "ff4_vs_naive"  # With only Feature Fuzzification, conv_k=4
    FF5 = "ff5_vs_naive"  # With only Feature Fuzzification, conv_k=5
    FUZZY = "fcart_vs_ccart"
    BOOSTING = "fgbdt_vs_nfgbdt"
    MIXED = "mfgbdt_vs_nfgbdt"


def exec_exp_clf(comparing_mode=ComparisionMode.FUZZY):
    result_df = pd.DataFrame()

    # Load data sets.
    dataset_df_list, dataset_name_list = load_dataset_df_classification()
    # Iterate all data sets, and execute the function of the experiment in each iteration.
    for dataset_df, dataset_name in zip(dataset_df_list, dataset_name_list):
        X = dataset_df.iloc[:, :-1].values
        y = dataset_df.iloc[:, -1:].values

        fuzzy_accuracy_list, naive_accuracy_list = get_exp_results_clf(X, y, comparing_mode=comparing_mode)

        fuzzy_mean = np.mean(fuzzy_accuracy_list)
        fuzzy_std = np.std(fuzzy_accuracy_list)
        naive_mean = np.mean(naive_accuracy_list)
        naive_std = np.std(naive_accuracy_list)
        result_df[dataset_name] = [fuzzy_mean, naive_mean, fuzzy_std, naive_std]

    # Load all data sets.
    X_list, y_list, dataset_name_list = load_dataset_classification()
    # Iterate all data sets, and execute the function of the experiment in each iteration.
    for X, y, dataset_name in zip(X_list, y_list, dataset_name_list):
        fuzzy_accuracy_list, naive_accuracy_list = get_exp_results_clf(X, y, comparing_mode=comparing_mode)

        fuzzy_mean = np.mean(fuzzy_accuracy_list)
        fuzzy_std = np.std(fuzzy_accuracy_list)
        naive_mean = np.mean(naive_accuracy_list)
        naive_std = np.std(naive_accuracy_list)
        result_df[dataset_name] = [fuzzy_mean, naive_mean, fuzzy_std, naive_std]

    # Finally, output the results of the experiment.
    result_df.to_csv("exp_results_" + comparing_mode.value + ".csv")


def get_exp_results_clf(X, y, comparing_mode=ComparisionMode.FUZZY):
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
    print(comparing_mode.value, "-FDT's mean accuracy:", np.mean(fuzzy_accuracy_list), "   std:",
          np.std(fuzzy_accuracy_list))
    print(comparing_mode.value, "-NDT's mean accuracy:", np.mean(naive_accuracy_list), "   std:",
          np.std(naive_accuracy_list))
    print("========================================================================================")

    return fuzzy_accuracy_list, naive_accuracy_list


def use_fuzzy_trees(comparing_mode, X_train, X_test, y_train, y_test, fuzzification_params):
    time_start = time.time()  # Record the start time.

    clf = None
    if comparing_mode is ComparisionMode.NAIVE:
        # My NDT vs. sklearn NDT
        clf = FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier, disable_fuzzy=True,
                                       fuzzification_params=fuzzification_params,
                                       criterion_func=CRITERIA_FUNC_CLF["gini"], max_depth=5)
    elif comparing_mode is ComparisionMode.FF3 or comparing_mode is ComparisionMode.FF4 or comparing_mode is ComparisionMode.FF5:
        # With only Feature Fuzzification vs. NDT
        clf = FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier, disable_fuzzy=True,
                                       fuzzification_params=fuzzification_params,
                                       criterion_func=CRITERIA_FUNC_CLF["gini"], max_depth=5)
    elif comparing_mode is ComparisionMode.FUZZY:
        # FDT vs. NDT
        clf = FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier, disable_fuzzy=False,
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
        clf = FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier, disable_fuzzy=True,
                                       criterion_func=CRITERIA_FUNC_CLF["gini"], max_depth=5)
    elif comparing_mode is ComparisionMode.FUZZY:
        # FDT vs. NDT
        clf = FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier, disable_fuzzy=True,
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


def exp_regression():
    data = datasets.load_boston()
    # data = datasets.load_diabetes()
    # data = datasets.load_linnerud()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # fuzzy_model = FuzzyDecisionTreeRegressor()
    fuzzy_model = FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTRegressor)
    fuzzy_model.fit(X_train, y_train)
    # fuzzy_model.print_tree()
    y_pred = fuzzy_model.predict(X_test)

    print("========================================================================================")
    print("r2_score:", r2_score(y_test, y_pred))
    print("MSE:", calculate_mse(y_test, y_pred))
    print("MAE:", calculate_mae(y_test, y_pred))
    print("========================================================================================")


def load_dataset_classification():
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

    dataset = datasets.load_iris()
    X = dataset.data
    y = dataset.target
    X_list.append(X)
    y_list.append(y)
    dataset_name_list.append("Iris")

    # dataset = datasets.load_breast_cancer()
    # X = dataset.data
    # y = dataset.target
    # X_list.append(X)
    # y_list.append(y)
    # dataset_name_list.append("Breast_Cancer")

    # dataset = datasets.load_digits()
    # X = dataset.data
    # y = dataset.target
    # X_list.append(X)
    # y_list.append(y)
    # dataset_name_list.append("Digits")

    dataset = datasets.load_wine()
    X = dataset.data
    y = dataset.target
    X_list.append(X)
    y_list.append(y)
    dataset_name_list.append("Wine")

    return X_list, y_list, dataset_name_list


def load_dataset_df_classification():
    """
    Load all data sets for classification experiments.

    NB: Before version 1.0 (upgraded by Cython to speed up), use data sets Vehicle, German Credit, Diabetes,
        do not use Waveform (spending about 206s/model), Chess (not suitable for fuzzy)

    Returns
    -------
    dataset_df_list: list of DataFrame of shape (n_samples, n_features_and_labels)
        A list of the features and labels of samples.

    dataset_name_list: list of shape (n_data_sets, 1)
        A list of the names of data sets.
    """
    dataset_df_list = []
    dataset_name_list = []

    # Start loading data sets one by one.

    dataset_df = dh.load_vehicle()
    dataset_df_list.append(dataset_df)
    dataset_name_list.append("Vehicle")

    # dataset_df = dh.load_waveform()
    # dataset_df_list.append(dataset_df)
    # dataset_name_list.append("Waveform")

    dataset_df = dh.load_German_credit()
    dataset_df_list.append(dataset_df)
    dataset_name_list.append("German_Credit")

    # dataset_df = dh.load_chess()
    # dataset_df_list.append(dataset_df)
    # dataset_name_list.append("Chess")

    dataset_df = dh.load_diabetes()
    dataset_df_list.append(dataset_df)
    dataset_name_list.append("Diabetes")

    return dataset_df_list, dataset_name_list


def simple_exp_classification_ensemble(X, y):
    # Using a fuzzy decision tree.
    X_fuzzy_pre = X.copy()
    X_fuzzy_pre[:, :] -= X_fuzzy_pre[:, :].min()
    X_fuzzy_pre[:, :] /= X_fuzzy_pre[:, :].max()
    X_fuzzy_dms = extract_fuzzy_features(X)
    X_train, X_test, y_train, y_test = train_test_split(X_fuzzy_dms, y, test_size=0.4)
    clf = FuzzyGBDTClassifier(criterion_func=CRITERIA_FUNC_REG["mse"], learning_rate=0.1, n_estimators=100, max_depth=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    # dataset = datasets.load_iris()
    # X = dataset.data
    # y = dataset.target

    # # Uh-huh...... Try a simple test before experiment.
    # simple_exp_classification_ensemble(X, y)

    # # See what the shape of X after feature fuzzification is and
    # # what the membership degree looks like to determine
    # # how to calculate fuzzy impurity used in splitting.
    # X_fuzzy = extract_fuzzy_features(X)
    # print("Shape(n_samples and n_features):", np.shape(X_fuzzy), "And the data is:")
    # print(X_fuzzy)

    # 1. Multiple experiment, and count the single elapsed time of it. =================================================
    # time_start = time.time()
    # exec_exp_clf(ComparisionMode.NAIVE)
    # print("Elapsed time: {:.5}s".format(time.time() - time_start))

    # time_start = time.time()
    # exec_exp_clf(ComparisionMode.FF3)
    # print("Elapsed time: {:.5}s".format(time.time() - time_start))

    # time_start = time.time()
    # exec_exp_clf(ComparisionMode.FF4)
    # print("Elapsed time: {:.5}s".format(time.time() - time_start))

    # time_start = time.time()
    # exec_exp_clf(ComparisionMode.FF5)
    # print("Elapsed time: {:.5}s".format(time.time() - time_start))

    # time_start = time.time()
    # exec_exp_clf(ComparisionMode.FUZZY)
    # print("Elapsed time: {:.5}s".format(time.time() - time_start))

    time_start = time.time()
    exec_exp_clf(ComparisionMode.BOOSTING)
    print("Elapsed time: {:.5}s".format(time.time() - time_start))

    # time_start = time.time()
    # exec_exp_clf(ComparisionMode.MIXED)
    # print("Elapsed time: {:.5}s".format(time.time() - time_start))

    # 2. Multiple experiment, and count the elapsed time of it repeated N times. =======================================
    # elapsed_time = timeit("exe_exp_clf(ComparisionMode.NAIVE)", globals={"exe_exp_clf": exec_exp_clf, "ComparisionMode": ComparisionMode}, number=2)
    # print("Elapsed time: {:.5}s".format(elapsed_time))
    # elapsed_time = timeit("exe_exp_clf(ComparisionMode.FUZZY)", globals={"exe_exp_clf": exec_exp_clf, "ComparisionMode": ComparisionMode}, number=2)
    # print("Elapsed time: {:.5}s".format(elapsed_time))
    # elapsed_time = timeit("exe_exp_clf(ComparisionMode.BOOSTING)", globals={"exe_exp_clf": exec_exp_clf, "ComparisionMode": ComparisionMode}, number=2)
    # print("Elapsed time: {:.5}s".format(elapsed_time))
    # elapsed_time = timeit("exe_exp_clf(ComparisionMode.MIXED)", globals={"exe_exp_clf": exec_exp_clf, "ComparisionMode": ComparisionMode}, number=2)
    # print("Elapsed time: {:.5}s".format(elapsed_time))

    # 3. Multiple experiment, and count the elapsed time and list its details in order of all functions called. ========
    # profile.run("exec_exp_clf(ComparisionMode.NAIVE)")
    # profile.run("exec_exp_clf(ComparisionMode.FUZZY)")
    # profile.run("exec_exp_clf(ComparisionMode.BOOSTING)")
    # profile.run("exec_exp_clf(ComparisionMode.MIXED)")
    # Single experiment approach =======================================================================================

    # Single experiment approach ===============================================
    # exp_classification()
    # exp_regression()

    # dataset = datasets.load_iris()
    # print(dataset.keys())
    # print("filename:", dataset.filename)
    # X = dataset.data
    # y = dataset.target
    # print("type, type of elements, total number of elements, shape, and number of dimensions:")
    # print(type(X), X.dtype, X.size, X.shape, X.ndim)
    # print(type(y), y.dtype, y.size, y.shape, y.ndim)
    # dataset_np = np.hstack([X, y.reshape(-1, 1)])
    # print("*** dataset_np", type(dataset_np), "***************************************************************")
    # print(dataset_np)
    # dataset_df = pd.DataFrame(dataset_np)
    # dataset_df.columns = [col_name for col_name in dataset.feature_names] + ['label']
    # print("*** dataset_df", type(dataset_df), "***************************************************************")
    # print(dataset_df.head())
    # print("*** dataset's target_names:", dataset.target_names, "; dataset's filename:", dataset.filename)
    #
    # # Test util_criterion_funcs.py
    # y = np.expand_dims(y, axis=1)
    # print(y)
    # print("variance:", calculate_variance(y))
    # print("mean:", calculate_mean(y))

    # # Test dependency injection -----------------------------------------------
    # class AAA:
    #     def __init__(self, friend):
    #         self.friend = friend
    #
    #     def say(self):
    #         print("Hi,", self.friend, ", I'm AAA.")
    #
    # class BBB:
    #     def __init__(self, aaa_class, friend):
    #         self.model = aaa_class(friend)
    #         self.model.say()
    #
    # bbb = BBB(AAA, "Geo")
    #
    #
    # # Test dependency injection -----------------------------------------------
    # dataset_df = pd.DataFrame(X)
    # print(dataset_df)
    # new_cols, _, _ = degree_of_membership_build(r_seed=0, X_df=dataset_df.iloc[:, 0], conv_k=5)
    # print("*********************************************************************")
    # print(new_cols)

    # # Test np.concatenate if two data sets have different number of dimensions.
    # # dataset = np.concatenate((X, y), axis=1)
    # #
    # # print("X:", np.shape(X))
    # # print("y's shape:", np.shape(y))
    # # print(len(np.shape(y)) == 1)
    # y = np.expand_dims(y, axis=1)
    # # print("y's new shape:", np.shape(y))
    # dataset = np.concatenate((X, y), axis=1)
    # # print(dataset)
    #
    # # Test split function
    # subset_true, subset_false = split_dataset(dataset, 0, 5.5)
    # print("*********** type, type of elements, total number of elements, shape, and number of dimensions:")
    # print(type(subset_true), subset_true.dtype, subset_true.size, subset_true.shape, subset_true.ndim)
    # print(type(subset_false), subset_false.dtype, subset_false.size, subset_false.shape, subset_false.ndim)
    # print(subset_true)
    # print(subset_false)

    # # Test criterion functions
    # entropy = calculate_entropy(y)
    # print("entropy:", entropy)
    # gini = calculate_gini(y)
    # print("gini:", gini)
