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

1.1. The comparison of two non-fuzzy CART classifiers fitted on fuzzy (conv_k=3)
and non-fuzzy training data respectively.

1.2. The comparison of two non-fuzzy CART classifiers fitted on fuzzy (conv_k=4)
and non-fuzzy training data respectively.

1.3. The comparison of two non-fuzzy CART classifiers fitted on fuzzy (conv_k=5)
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

# =============================================================================
# Global variables
# =============================================================================
# Logger used for logging in production.
# Note: The root logger in `logging` used only for debugging in development.
logger = logging.getLogger("main.core")

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
    "fcart_vs_nfcart": FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier,
                                                disable_fuzzy=True,
                                                criterion_func=CRITERIA_FUNC_CLF["gini"],
                                                max_depth=5),
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
    #                                              fuzzification_options=FuzzificationOptions(conv_k=3),
    #                                              criterion_func=CRITERIA_FUNC_CLF["gini"],
    #                                              max_depth=5),
    # "f4_ds_vs_orig_ds": FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier,
    #                                              disable_fuzzy=True,
    #                                              fuzzification_options=FuzzificationOptions(conv_k=4),
    #                                              criterion_func=CRITERIA_FUNC_CLF["gini"],
    #                                              max_depth=5),
    # "f5_ds_vs_orig_ds": FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier,
    #                                              disable_fuzzy=True,
    #                                              fuzzification_options=FuzzificationOptions(conv_k=5),
    #                                              criterion_func=CRITERIA_FUNC_CLF["gini"],
    #                                              max_depth=5),
    # "f6_ds_vs_orig_ds": FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier,
    #                                              disable_fuzzy=True,
    #                                              fuzzification_options=FuzzificationOptions(conv_k=6),
    #                                              criterion_func=CRITERIA_FUNC_CLF["gini"],
    #                                              max_depth=5),
    "fcart_vs_nfcart": FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier,
                                                disable_fuzzy=False,
                                                fuzzification_options=FuzzificationOptions(conv_k=3),
                                                criterion_func=CRITERIA_FUNC_CLF["gini"],
                                                max_depth=5),
    # "fgbdt_vs_nfgbdt": FuzzyGBDTClassifier(disable_fuzzy=False,
    #                                        fuzzification_options=FuzzificationOptions(conv_k=3),
    #                                        criterion_func=CRITERIA_FUNC_REG["mse"],
    #                                        learning_rate=0.1,
    #                                        n_estimators=100,
    #                                        max_depth=5),
}


def exp_one_clf(q, ds_name, comparison_mode, clf, with_fuzzy_rules, X, y, n_exps=10):
    """Experiment with one classifier."""
    # Record the start time.
    time_start = time.time()

    # 2. Preprocess the dataset. =======================================================================================
    if clf is not None and clf.fuzzification_options is not None:
        # 2.1. Do fuzzification preprocessing.
        # Debug message.
        logging.debug("(Dataset: '%s'; Experiment: '%s'; Fuzzy rules: '%r') X before fuzzification: %s",
                      ds_name, comparison_mode, with_fuzzy_rules, np.shape(X))
        X_fuzzy_pre = X.copy()
        # 2.1.1. Standardise feature scaling.
        # X_fuzzy_pre[:, :] -= X_fuzzy_pre[:, :].min()
        # X_fuzzy_pre[:, :] /= X_fuzzy_pre[:, :].max()
        # 2.1.2. Extract fuzzy features.
        conv_k = clf.fuzzification_options.conv_k
        X_dms = extract_fuzzy_features(X_fuzzy_pre, conv_k=conv_k)
        X = np.concatenate((X, X_dms), axis=1)
        # Debug message.
        logging.debug("(Dataset: '%s'; Experiment: '%s'; Fuzzy rules: '%r') X after fuzzification: %s",
                      ds_name, comparison_mode, with_fuzzy_rules, np.shape(X))

    acc_train_list, acc_test_list = [], []
    for i in range(n_exps):
        # 3. Partition the dataset. ====================================================================================
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=i)

        # 4. Train the models. =========================================================================================
        # Debug message.
        logging.debug("(Dataset: '%s'; Experiment: '%s'; Fuzzy rules: '%r') %d-th",
                      ds_name, comparison_mode, with_fuzzy_rules, i)
        clf.fit(X_train, y_train)

        # 5. Look at the classifier.
        # clf.print_tree()

        # 6. Evaluate the classifier.
        # 6.1. Calculate the train accuracy and test accuracy.
        y_pred_train = clf.predict(X_train)
        acc_train = accuracy_score(y_train, y_pred_train)
        y_pred_test = clf.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred_test)

        acc_train_list.append(acc_train)
        acc_test_list.append(acc_test)

    # Prepare the results. =============================================================================================
    mean_acc_train = np.mean(acc_train_list)
    std_acc_train = np.std(acc_train_list)
    mean_acc_test = np.mean(acc_test_list)
    std_acc_test = np.std(acc_test_list)
    elapsed_time = time.time() - time_start

    # Debug message.
    logging.debug("=" * 100)
    logging.debug("(Dataset: '%s'; Experiment: '%s'; Fuzzy rules: '%r') %f, %f, %f, %f, %f(s)",
                  ds_name, comparison_mode, with_fuzzy_rules,
                  mean_acc_test, std_acc_test, mean_acc_train, std_acc_train, elapsed_time)
    logging.debug("=" * 100)

    # Put the results into the queue to send back to main process.
    if not q.full():
        q.put([comparison_mode, ds_name, with_fuzzy_rules,
               mean_acc_test, std_acc_test, mean_acc_train, std_acc_train, str(elapsed_time) + "(s)"])


def encapsulate_results(q):
    """Encapsulate the results sent back from all sub-processes."""
    global exp_results

    # 1. Bin each result.
    while not q.empty():
        res = q.get()
        exp_results.append(res)

    # 2. Sort the index by the first column. If the first column has the same items that
    # cannot be sorted, then sort the index by the second column. If the second column
    # has the same items, then sort the index by the third column.
    if exp_results:
        exp_results = np.asarray(exp_results)
        indexes_sorted = np.lexsort((exp_results[:, 2], exp_results[:, 1], exp_results[:, 0]))
        exp_results = exp_results[indexes_sorted]

    # Debug message.
    logging.debug("Finished encapsulation: %s\n%s", np.shape(exp_results), exp_results)


def output_results():
    """Output the results."""
    global exp_results

    if len(exp_results) > 0:
        column_names = ["comparison_mode", "ds_name", "is_fuzzy_clf", "mean_acc_test",
                        "std_acc_test", "mean_acc_train", "std_acc_train", "elapsed_time"]
        exp_res_df = pd.DataFrame(data=exp_results, columns=column_names)
        filename = DirSave.EVAL_DATA.value + get_now_str(get_timestamp_str()) + "-exp-s1.csv"
        exp_res_df.to_csv(filename)

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

        for ds_name in DS_LOAD_FUNC_CLF.keys():
            # 1. Load the dataset.
            ds_df = load_data_clf(ds_name)
            if ds_df is not None:
                X = ds_df.iloc[:, :-1].values
                y = ds_df.iloc[:, -1].values

                # Experiment without fuzzy rules.
                n_exps = 10
                for comparison_mode, clf in NON_FUZZY_CLFS.items():
                    # Debug message.
                    logging.debug("Dataset: '%s'; Experiment: '%s'; Fuzzy rules: '%r'",
                                  ds_name, comparison_mode, False)

                    n_exps = 1 if comparison_mode == "fgbdt_vs_nfgbdt" else n_exps
                    pool.apply_async(exp_one_clf, args=(q, ds_name, comparison_mode, clf, False, X, y, n_exps,))

                # Experiment with fuzzy rules.
                for comparison_mode, clf in FUZZY_CLFS.items():
                    # Debug message.
                    logging.debug("Dataset: '%s'; Experiment: '%s'; Fuzzy rules: '%r'",
                                  ds_name, comparison_mode, True)

                    n_exps = 1 if comparison_mode == "fgbdt_vs_nfgbdt" else n_exps
                    pool.apply_async(exp_one_clf, args=(q, ds_name, comparison_mode, clf, True, X, y, n_exps,))

                    # Just to check for exception details, if any.
                    # res = pool.apply_async(exp_one_clf, args=(q, ds_name, comparison_mode, clf, True, X, y, n_exps,))
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
