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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from exp_params import ComparisionMode, NUM_CPU_CORES_REQ, DS_LOAD_FUNC_CLF, FUZZY_STRIDE
from fuzzy_trees.fuzzy_decision_tree import FuzzyDecisionTreeClassifier
from fuzzy_trees.fuzzy_decision_tree_api import FuzzificationParams, FuzzyDecisionTreeClassifierAPI, CRITERIA_FUNC_CLF, \
    CRITERIA_FUNC_REG
from fuzzy_trees.fuzzy_gbdt import FuzzyGBDTClassifier
from fuzzy_trees.util_data_processing_funcs import extract_fuzzy_features
import fuzzy_trees.util_plotter as plotter


# For storing data to plot figures.
DS_PLOT = {}


def search_fuzzy_optimum():
    """
    Task 1: Searching an optimum of fuzzy thresholds by a loop according the specified stride.

    2nd method: Use Pool and Pipe, and get results by Pipe.
    """
    # Create a connection between processes.
    # !!! NB: When using Pipe and the size of the communication message is greater than 65537,
    # it will block the pipe and then the main process will lock up because it will always be
    # waiting for the child process to finish.
    # When using Pool create processes, use multiprocessing.Manager().Queue()
    # instead of multiprocessing.Queue() to create connection.
    conn_recv, conn_send = multiprocessing.Pipe()  # !!!!!!!! Deprecated method.

    # Create a pool containing n (0 - infinity) processes.
    # If the parameter "processes" is None then the number returned by os.cpu_count() is used.
    # Make sure that n is <= the number of CPU cores available.
    # The parameters to the Pool indicate how many parallel processes are called to run the program.
    # The default size of the Pool is the number of compute cores on the CPU, i.e. multiprocessing.cpu_count().
    pool = multiprocessing.Pool(NUM_CPU_CORES_REQ)

    # Complete all tasks by the pool.
    # !!! NB: If you want to complete the experiment faster, you can use distributed computing. Or you can divide
    # the task into k groups to execute in k py programs, and then run one on each of k clusters simultaneously.
    pro_num = 0
    for ds_name in DS_LOAD_FUNC_CLF.keys():
        fuzzy_th = 0
        while fuzzy_th <= 0.5:
            # Add a process into the pool. apply_async() is asynchronous equivalent of "apply()".
            pool.apply_async(search_fuzzy_optimum_on_one_ds, args=(conn_send, ComparisionMode.FUZZY, ds_name, fuzzy_th,))
            fuzzy_th += FUZZY_STRIDE
            pro_num += 1

    pool.close()
    pool.join()

    # Encapsulate each process's result into a data set preparing for plotting.
    ds_plotting = encapsulate_result(conn_recv)
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++", ds_plotting)

    # Plot the comparison of training error versus test error, and both curves are fuzzy thresholds versus accuracies.
    # Illustrate how the performance on unseen data (test data) is different from the performance on training data.
    for (ds_name, coordinates) in ds_plotting.items():
        # x_lower_limit, x_upper_limit = np.min(x_train), np.max(x_train)
        # y_lower_limit = np.min(y_train) if np.min(y_train) < np.min(y_test) else np.min(y_test)
        # y_upper_limit = np.max(y_train) if np.max(y_train) > np.max(y_test) else np.max(y_test)
        # print("x_limits and y_limits are:", x_lower_limit, x_upper_limit, y_lower_limit, y_upper_limit)
        plotter.plot_multi_curves(coordinates=coordinates,
                                  title="Training Error vs Test Error -- {}".format(ds_name),
                                  x_label="Fuzzy threshold",
                                  y_label="Error",
                                  legends=["Train", "Test"])


def search_fuzzy_optimum_on_one_ds(conn_send, comparing_mode, ds_name, fuzzy_th):
    # print("Child process (%s) started.++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" % os.getpid())
    # Load all data sets.
    ds_df = load_dataset_clf(ds_name)

    # Run the experiment according to the parameters.
    X = ds_df.iloc[:, :-1].values
    y = ds_df.iloc[:, -1].values
    accuracy_train_list, accuracy_test_list = get_exp_results_clf(X, y, comparing_mode=comparing_mode, ds_name=ds_name, fuzzy_th=fuzzy_th)

    # Process the result.
    accuracy_train_mean = np.mean(accuracy_train_list)
    accuracy_test_mean = np.mean(accuracy_test_list)

    # Put the result in the connection between the main process and the child processes (in master-worker mode).
    # The 2nd return value in send() should be a 2-dimensional ndarray
    error_train_mean = 1 - accuracy_train_mean
    error_test_mean = 1 - accuracy_test_mean
    # !!! NB: The value in the dictionary to be returned must be a 2-d matrix.
    conn_send.send({ds_name: np.asarray([[fuzzy_th, error_train_mean, fuzzy_th, error_test_mean]])})


def load_dataset_clf(ds_name):
    """
    Load data by a dataset name.

    Parameters
    ----------
    ds_name: a key in exp.params.DS_LOAD_FUNC_CLF

    Returns
    -------
    data: DataFrame
    """
    ds_load_func = None

    if ds_name in DS_LOAD_FUNC_CLF.keys():
        ds_load_func = DS_LOAD_FUNC_CLF[ds_name]

    return None if ds_load_func is None else ds_load_func()


def get_exp_results_clf(X, y, comparing_mode, ds_name, fuzzy_th):
    accuracy_train_list = []
    accuracy_test_list = []

    # Step 1: Preprocess features for using fuzzy decision tree.
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
    print("************* Before, original shape:", np.shape(X))
    print("************* After, fuzzy shape:", np.shape(X_plus_dms))

    # Step 2: Get training and testing result by a model.
    for i in range(10):
        print("%ith comparison on %s" % (i, ds_name))

        # Split training and test sets by hold-out partition method.
        # X_train, X_test, y_train, y_test = train_test_split(X_fuzzy_pre, y, test_size=0.4)

        kf = KFold(n_splits=2, random_state=i, shuffle=True)
        for train_index, test_index in kf.split(X):
            y_train, y_test = y[train_index], y[test_index]

            # Using a fuzzy decision tree. =============================================================================
            X_train, X_test = X_plus_dms[train_index], X_plus_dms[test_index]
            accuracy_train, accuracy_test = exe_by_a_fuzzy_model(comparing_mode=comparing_mode, X_train=X_train, X_test=X_test, y_train=y_train,
                                                                 y_test=y_test, fuzzification_params=fuzzification_params)
            accuracy_train_list.append(accuracy_train)
            accuracy_test_list.append(accuracy_test)

    print("========================================================================================")
    print(comparing_mode.value, " - ", ds_name)
    print("Mean training accuracy:", np.mean(accuracy_train_list), "  std:",
          np.std(accuracy_train_list))
    print("Mean test accuracy:", np.mean(accuracy_test_list), "  std:",
          np.std(accuracy_test_list))
    print("========================================================================================")

    return accuracy_train_list, accuracy_test_list


def exe_by_a_fuzzy_model(comparing_mode, X_train, X_test, y_train, y_test, fuzzification_params):
    time_start = time.time()  # Record the start time.

    # Initialise a fuzzy model.
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
                                  max_depth=5)

    # Fit the initialised model.
    clf.fit(X_train, y_train)
    # clf.print_tree()

    # Get the training accuracy and test accuracy of the fitted (trained) estimator.
    y_pred_train = clf.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    y_pred_test = clf.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    # print("    Fuzzy accuracy train:", accuracy_train)
    # print("    Fuzzy accuracy test:", accuracy_test)
    print('    Elapsed time of a single model (FDT-based):', time.time() - time_start, 's')  # Display the time of training a single model.

    return accuracy_train, accuracy_test


def encapsulate_result(conn_recv):
    """
    Encapsulate each process's result into a data set, preparing for plotting.
    """
    while conn_recv.poll():
        res = conn_recv.recv()

        for (ds_name, coordinates) in res.items():
            if len(np.shape(coordinates)) == 1:
                coordinates = np.expand_dims(coordinates, axis=0)

            if ds_name in DS_PLOT:
                DS_PLOT[ds_name] = np.concatenate((DS_PLOT[ds_name], coordinates), axis=0)
            else:
                DS_PLOT[ds_name] = coordinates

    return DS_PLOT


"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
In master-worker mode, if an execution in the master process depends on the results of all the child processes, place this execution code after either Pool().join() or Process().join(). 
The reason it is not recommended to put execution code in child processes is because child processes run in parallel with each other, so if it is in a child process, it doesn't know if the other child processes have terminated. Although it is possible to obtain the status of all child processes through a two-way Pipe(), that is not a good management mode.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
if __name__ == '__main__':
    print("Main Process (%s) started." % os.getpid())
    time_start = time.time()

    search_fuzzy_optimum()

    print("Total elapsed time: {:.5}s".format(time.time() - time_start))
    print("Main Process (%s) ended." % os.getpid())
