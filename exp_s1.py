"""
@author: Zhaoqing Liu
@email : Zhaoqing.Liu-1@student.uts.edu.au

Notes
-----
There are multiple comparison modes in the experiments and on multiple datasets.

The fuzzification of the fuzzy CART proposed in this paper includes two steps.
One is the fuzzification of the features in the data preprocessing stage
and the other is the fuzzification of the split criteria calculation in the model training stage.
Therefore, in the first two experiments, the first is to verify the impact of
the first step of fuzzification on the performance of a CART classifier,
while the second is to verify the impact of the combination of the two steps of fuzzification.
Also, the third experiment is to see what interesting changes would be made to
the performance of a GBDT classifier with all steps of fuzzification.

1. The first experiment includes:

1.1. The comparison of two non-fuzzy CART classifiers fitted on fuzzy (n_conv=3)
and non-fuzzy training data respectively.

1.2. The comparison of two non-fuzzy CART classifiers fitted on fuzzy (n_conv=4)
and non-fuzzy training data respectively.

1.3. The comparison of two non-fuzzy CART classifiers fitted on fuzzy (n_conv=5)
and non-fuzzy training data respectively.

2. The second experiment is to compare a fuzzy CART classifier with a non-fuzzy one.

3. The third experiment is to compare a fuzzy GBDT classifier with a non-fuzzy one.

Also, see 'util_data_handler.py' for details on the datasets used in the experiments.
"""
import logging
import multiprocessing
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from fuzzytrees.fdt_base import FuzzificationOptions, FuzzyDecisionTreeWrapper, CRITERIA_FUNC_CLF, CRITERIA_FUNC_REG
from fuzzytrees.fdts import FuzzyCARTClassifier
from fuzzytrees.fgbdt import FuzzyGBDTClassifier
from fuzzytrees.settings import DirSave
from fuzzytrees.util_comm import get_now_str, get_timestamp_str
from fuzzytrees.util_data_handler import DS_LOAD_FUNC_CLF, load_data_clf
from fuzzytrees.util_logging import setup_logging
from fuzzytrees.util_preprocessing_funcs import extract_fuzzy_features
import warnings


# =============================================================================
# Environment configuration
# =============================================================================
# Make sure you know what the possible warnings are before you ignore them.
warnings.filterwarnings("ignore")


# =============================================================================
# Global variables
# =============================================================================
# Logger used for logging in production.
# Note: The root logger in `logging` used only for debugging in development.
logger = logging.getLogger("main.core")

n_conv = 3
# Data container used for storing experiments' results.
exp_results = []

# Non-fuzzy classifiers used in comparison modes.
# # k: comparison mode; v: fuzzy classifier instance.
NON_FUZZY_CLFS = {
    # "f3_ds_vs_orig_ds": FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier,
    #                                              disable_fuzzy=True,
    #                                              criterion_func=CRITERIA_FUNC_CLF["gini"],
    #                                              max_depth=5),
    # "f4_ds_vs_orig_ds": FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier,
    #                                              disable_fuzzy=True,
    #                                              criterion_func=CRITERIA_FUNC_CLF["gini"],
    #                                              max_depth=5),
    # "f5_ds_vs_orig_ds": FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier,
    #                                              disable_fuzzy=True,
    #                                              criterion_func=CRITERIA_FUNC_CLF["gini"],
    #                                              max_depth=5),
    # "f6_ds_vs_orig_ds": FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier,
    #                                              disable_fuzzy=True,
    #                                              criterion_func=CRITERIA_FUNC_CLF["gini"],
    #                                              max_depth=5),
    # "fcart_vs_nfcart": FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier,
    #                                             disable_fuzzy=True,
    #                                             criterion_func=CRITERIA_FUNC_CLF["gini"],
    #                                             max_depth=5),
    # "fgbdt_vs_nfgbdt": FuzzyGBDTClassifier(disable_fuzzy=True,
    #                                        criterion_func=CRITERIA_FUNC_REG["mse"],
    #                                        learning_rate=0.1,
    #                                        n_estimators=100,
    #                                        max_depth=5),
}

# Fuzzy classifiers used in comparison modes.
# k: comparison mode; v: fuzzy classifier instance.
FUZZY_CLFS = {
    # "f3_ds_vs_orig_ds": FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier,
    #                                              disable_fuzzy=True,
    #                                              fuzzification_options=FuzzificationOptions(n_conv=3),
    #                                              criterion_func=CRITERIA_FUNC_CLF["gini"],
    #                                              max_depth=5),
    # "f4_ds_vs_orig_ds": FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier,
    #                                              disable_fuzzy=True,
    #                                              fuzzification_options=FuzzificationOptions(n_conv=4),
    #                                              criterion_func=CRITERIA_FUNC_CLF["gini"],
    #                                              max_depth=5),
    # "f5_ds_vs_orig_ds": FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier,
    #                                              disable_fuzzy=True,
    #                                              fuzzification_options=FuzzificationOptions(n_conv=5),
    #                                              criterion_func=CRITERIA_FUNC_CLF["gini"],
    #                                              max_depth=5),
    # "f6_ds_vs_orig_ds": FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier,
    #                                              disable_fuzzy=True,
    #                                              fuzzification_options=FuzzificationOptions(n_conv=6),
    #                                              criterion_func=CRITERIA_FUNC_CLF["gini"],
    #                                              max_depth=5),
    "fcart_vs_nfcart": FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier,
                                                disable_fuzzy=False,
                                                fuzzification_options=FuzzificationOptions(n_conv=n_conv),
                                                criterion_func=CRITERIA_FUNC_CLF["gini"],
                                                max_depth=5),
    # "fgbdt_vs_nfgbdt": FuzzyGBDTClassifier(disable_fuzzy=False,
    #                                        fuzzification_options=FuzzificationOptions(n_conv=3),
    #                                        criterion_func=CRITERIA_FUNC_REG["mse"],
    #                                        learning_rate=0.1,
    #                                        n_estimators=100,
    #                                        max_depth=5),
}


# =============================================================================
# Functions
# =============================================================================
def exp_one_clf(q, ds_name, comparison_mode, clf, with_fuzzy_rules, sn, X, y):
    """Experiment with one classifier."""
    # Record the start time.
    time_start = time.time()

    # 3. Partition the dataset. ====================================================================================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=sn)
    logging.debug("(Dataset: '%s'; Experiment: '%s'; Fuzzy rules: '%r'; SN: '%d-th') training: %s, test: %s",
                  ds_name, comparison_mode, with_fuzzy_rules, sn, y_train.shape, y_test.shape)

    # 4. Train the models. =========================================================================================
    clf.fit(X_train, y_train)

    # 5. Look at the classifier.
    # clf.print_tree()

    # 6. Evaluate the classifier.
    # 6.1. Calculate the test accuracy and train accuracy.
    y_pred_test = clf.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    y_pred_train = clf.predict(X_train)
    acc_train = accuracy_score(y_train, y_pred_train)

    elapsed_time = time.time() - time_start

    # Debug message.
    logging.debug("=" * 100)
    logging.debug("(Dataset: '%s'; Experiment: '%s'; Fuzzy rules: '%r'; SN: '%d-th') %f, %f, %f(s)",
                  ds_name, comparison_mode, with_fuzzy_rules, sn, acc_test, acc_train, elapsed_time)
    logging.debug("=" * 100)

    # Put the results into the queue to send back to main process.
    if not q.full():
        q.put([comparison_mode, ds_name, with_fuzzy_rules, sn, acc_test, acc_train, elapsed_time])


def encapsulate_results(q):
    """Encapsulate the results sent back from all sub-processes."""
    global exp_results

    # 1. Bin each result.
    while not q.empty():
        res = q.get()
        exp_results.append(res)

    # 2. Sort the index by the first column. If the first column has the same items that
    # cannot be sorted, then sort the index by the second column. If the second column
    # has the same items, then sort the index by the third column. If the third column
    # has the same items, then sort the index by the fourth column.
    if exp_results:
        exp_results = np.asarray(exp_results)
        indexes_sorted = np.lexsort((exp_results[:, 3], exp_results[:, 2], exp_results[:, 1], exp_results[:, 0]))
        exp_results = exp_results[indexes_sorted]

    # Debug message.
    logging.debug("Finished encapsulation: %s", np.shape(exp_results))


def output_results():
    """Output the results."""
    global exp_results

    exp_res_df = None
    if len(exp_results) > 0:
        column_names = ["comparison_mode", "ds_name", "with_fuzzy_rules", "SN", "acc_test", "acc_train", "elapsed_time"]
        exp_res_df = pd.DataFrame(data=exp_results, columns=column_names)
        # Specify the dtypes of numeric columns required for mathematical calculation, otherwise an error will occur.
        exp_res_df["acc_test"] = exp_res_df["acc_test"].astype(float)
        exp_res_df["acc_train"] = exp_res_df["acc_train"].astype(float)
        exp_res_df["elapsed_time"] = exp_res_df["elapsed_time"].astype(float)

        # 1. Output all the result records.
        # Mathematical calculation before output: Add a new column to store the mean `acc_test`
        # in the `n_exp` experimental results of one model，and the same calculation for `acc_train`.
        exp_res_df["mean_acc_test"] = exp_res_df.groupby(["comparison_mode",
                                                          "ds_name",
                                                          "with_fuzzy_rules"])["acc_test"].transform("mean")
        exp_res_df["mean_acc_train"] = exp_res_df.groupby(["comparison_mode",
                                                           "ds_name",
                                                           "with_fuzzy_rules"])["acc_train"].transform("mean")
        exp_res_df["mean_acc_test"] = exp_res_df.groupby(["comparison_mode",
                                                          "ds_name",
                                                          "with_fuzzy_rules"])["elapsed_time"].transform("mean")
        filename = DirSave.EVAL_DATA.value + get_now_str(get_timestamp_str()) + "-exp-s1.csv"
        exp_res_df.to_csv(filename)
        # Debug message.
        logging.debug("Finished output: %s", filename)

        # 2. Output the mean-statistics of `acc_train` and `acc_train`.
        exp_res_mean_df = exp_res_df.groupby(["comparison_mode",
                                              "ds_name",
                                              "with_fuzzy_rules"]).agg({"acc_test": "mean",
                                                                        "acc_train": "mean",
                                                                        "elapsed_time": "mean"})
        filename = DirSave.EVAL_DATA.value + get_now_str(get_timestamp_str()) + "-exp-s1-mean-stat.csv"
        exp_res_mean_df.to_csv(filename)
        # Debug message.
        logging.debug(exp_res_mean_df.values)
        logging.debug("Finished output: %s", filename)

        # 3. Output the whole statistics of `acc_train` and `acc_train`.
        exp_res_stat_df = exp_res_df.groupby(["comparison_mode",
                                              "ds_name",
                                              "with_fuzzy_rules"])[["acc_test",
                                                                    "acc_train",
                                                                    "elapsed_time"]].describe()
        filename = DirSave.EVAL_DATA.value + get_now_str(get_timestamp_str()) + "-exp-s1-whole-stat.csv"
        exp_res_stat_df.to_csv(filename)
        # Debug message.
        logging.debug("Finished output: %s", filename)


def exp_clf():
    """
    Main function of the experiment program that compares
    fuzzy and non-fuzzy classifiers through multiple modes.
    """
    # Debug message.
    logging.debug("Start master process '%s'......", os.getpid())

    # In multi-process mode.
    # When using Pool create processes, use multiprocessing.Manager().Queue()
    # instead of multiprocessing.Queue() to create connection.
    with multiprocessing.Manager() as mg:
        # Create a connection used to communicate between main process and its child processes.
        q = mg.Queue()
        # Create a pool for main process to manage its child processes in parallel.
        # If the parameter "processes" is None then the number returned by os.cpu_count() is used.
        pool = multiprocessing.Pool()

        # 1. Load the dataset. =========================================================================================
        for ds_name in DS_LOAD_FUNC_CLF.keys():
            ds_df = load_data_clf(ds_name)
            if ds_df is not None:
                # Resample with stratified sampling method if the sample size is too large to save the experiment time.
                shape = ds_df.shape
                if shape[0] > 1200:
                    ds_df, _ = train_test_split(ds_df, train_size=12000, stratify=ds_df.iloc[:, [-1]])
                    # Debug message.
                    logging.debug("(Dataset: '%s') Shape before resampling: '%s'; after: '%s'",
                                  ds_name, shape, ds_df.shape)

                X = ds_df.iloc[:, :-1].values
                y = ds_df.iloc[:, -1].values

                X_fuzzy_pre = X.copy()
                # Debug message.
                logging.debug(X_fuzzy_pre.dtype)
                # Convert dtype of X_fuzzy_pre to float.
                if not isinstance(X_fuzzy_pre.dtype, float):
                    X_fuzzy_pre = X_fuzzy_pre.astype(float)
                    logging.debug(X_fuzzy_pre.dtype)
                # 2. Preprocess the dataset. ===================================================================
                # 2.1. Do fuzzification preprocessing.
                # Debug message.
                logging.debug("Dataset: '%s'; X before fuzzification: %s", ds_name, np.shape(X_fuzzy_pre))
                # 2.1.1. Standardise feature scaling.
                X_fuzzy_pre[:, :] -= X_fuzzy_pre[:, :].min()
                X_fuzzy_pre[:, :] /= X_fuzzy_pre[:, :].max()
                # 2.1.2. Extract fuzzy features.
                X_dms = extract_fuzzy_features(X_fuzzy_pre, n_conv=n_conv)
                X_plus_dms = np.concatenate((X, X_dms), axis=1)
                # Debug message.
                logging.debug("Dataset: '%s'; X after fuzzification: %s", ds_name, np.shape(X_plus_dms))

                n_exp = 10
                for i in range(n_exp):
                    # Experiment without fuzzy rules.
                    for comparison_mode, clf in NON_FUZZY_CLFS.items():
                        # 3, 4, 5 and 6 steps in sub-processes. ========================================================
                        # pool.apply_async(exp_one_clf,
                        #                  args=(q, ds_name, comparison_mode, clf, False, i, X, y,))
                        if comparison_mode == "fgbdt_vs_nfgbdt":
                            if i < 10:
                                pool.apply_async(exp_one_clf,
                                                 args=(q, ds_name, comparison_mode, clf, False, i, X, y,))
                        else:
                            pool.apply_async(exp_one_clf,
                                             args=(q, ds_name, comparison_mode, clf, False, i, X, y,))

                    # Experiment with fuzzy rules.
                    for comparison_mode, clf in FUZZY_CLFS.items():
                        # 3, 4, 5 and 6 steps in sub-processes. =====================================================
                        # pool.apply_async(exp_one_clf,
                        #                  args=(q, ds_name, comparison_mode, clf, True, i, X_plus_dms, y,))
                        if comparison_mode == "fgbdt_vs_nfgbdt":
                            if i < 10:
                                pool.apply_async(exp_one_clf,
                                                 args=(q, ds_name, comparison_mode, clf, True, i, X_plus_dms, y,))
                        else:
                            pool.apply_async(exp_one_clf,
                                             args=(q, ds_name, comparison_mode, clf, True, i, X_plus_dms, y,))

                        # Following is only used to check for possible exceptions.
                        # res = pool.apply_async(exp_one_clf,
                        #                        args=(q, ds_name, comparison_mode, clf, True, i, X_plus_dms, y,))
                        # res.get()

        pool.close()
        pool.join()

        # Encapsulate the results sent back from all sub-processes.
        encapsulate_results(q)

    # Output the results.
    output_results()

    # Debug message.
    logging.debug("Done master process '%s'.", os.getpid())


if __name__ == '__main__':
    # Step 1: Configure the logging module to prepare for any subsequent procedures.
    filepath = "fuzzytrees/logging_config.yaml"
    setup_logging(filepath)

    # Step 2: Start the main function of the experiment program.
    exp_clf()