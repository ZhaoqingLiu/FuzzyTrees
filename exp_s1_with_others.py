"""
@author: Zhaoqing Liu
@email : Zhaoqing.Liu-1@student.uts.edu.au

Notes
-----
1. The fourth experiment:
Compare the performance of FDT with published benchmarks, including LightGBM,
XGBoost, CatBoost, MLP, etc. on datasets Covertype, Pokerhand, and Mushroom.
"""
import logging
import multiprocessing
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
from skmultiflow.lazy import SAMKNNClassifier
from skmultiflow.meta import AccuracyWeightedEnsembleClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

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

# Fuzzy classifiers used in comparison.
# k: Model; v: classifier instance.
FUZZY_CLFS = {
    "FDT": None,
}
# LightGBM used in comparison.
# k: Model; v: classifier instance.
NON_FUZZY_CLFS = {
    "LightGBM": None,
    "XGBoost": None,
    "CatBoost": None,
    "HoeffdingTree": None,
    "HoeffdingAdaptiveTree": None,
    "SAMKNN": None,
}


# =============================================================================
# Functions
# =============================================================================
def by_FDT(X_train, X_test, y_train, y_test):
    clf = FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier,
                                   disable_fuzzy=False,
                                   fuzzification_options=FuzzificationOptions(n_conv=n_conv),
                                   criterion_func=CRITERIA_FUNC_CLF["gini"],
                                   max_depth=5)
    clf.fit(X_train, y_train)

    # 5. Look at the classifier.
    # clf.print_tree()

    # 6. Evaluate the classifier.
    # 6.1. Calculate the test accuracy and train accuracy.
    y_pred_test = clf.predict(X_test)
    return accuracy_score(y_test, y_pred_test)


def by_LightGBM(X_train, X_test, y_train, y_test):
    trn_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_test, y_test)
    params_lightgbm = {'num_iterations': 1,
                       # 'num_leaves': 60,
                       # 'min_data_in_leaf': 30,
                       'objective': 'multiclass',
                       'num_class': 7,
                       'max_depth': 5,
                       'learning_rate': 0.01,
                       # "min_sum_hessian_in_leaf": 6,
                       "boosting": "gbdt",
                       "feature_fraction": 0.9,
                       "bagging_freq": 1,
                       "bagging_fraction": 0.8,
                       "bagging_seed": 11,
                       "lambda_l1": 0.1,
                       "verbosity": -1,
                       "nthread": 15,
                       'metric': 'multi_logloss',  # Or: multi_error
                       "random_state": 2021,
                       # 'device': 'gpu'
                       }
    clf = lgb.train(params_lightgbm,
                    trn_data,
                    num_boost_round=1000,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=100,
                    early_stopping_rounds=100)
    y_prob = clf.predict(X_test, num_iteration=clf.best_iteration)
    y_pred = [list(x).index(max(x)) for x in y_prob]
    return accuracy_score(y_pred, y_test)


def by_XGBoost(X_train, X_test, y_train, y_test):
    clf = XGBClassifier(learning_rate=0.01,
                        n_estimators=1,  # 树的个数-10棵树建立xgboost
                        max_depth=5,  # 树的深度
                        min_child_weight=1,  # 叶子节点最小权重
                        gamma=0,  # 惩罚项中叶子结点个数前的参数
                        subsample=0.7,  # 所有样本建立决策树
                        colsample_btree=0.7,  # 所有特征建立决策树
                        scale_pos_weight=1,  # 解决样本个数不平衡的问题
                        random_state=2021,  # 随机数
                        slient=1
                        )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)


def by_CatBoost(X_train, X_test, y_train, y_test):
    clf = CatBoostClassifier(iterations=1, depth=5, learning_rate=0.1,
                             loss_function='MultiClass',
                             logging_level='Verbose')
    clf.fit(X_train, y_train, verbose=False)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_pred, y_test)


def by_HoeffdingTree(X_train, X_test, y_train, y_test):
    clf = HoeffdingTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_pred, y_test)


def by_HoeffdingAdaptiveTree(X_train, X_test, y_train, y_test):
    clf = HoeffdingAdaptiveTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_pred, y_test)


def by_SAMKNN(X_train, X_test, y_train, y_test):
    clf = SAMKNNClassifier(n_neighbors=5, weighting='distance', max_window_size=1000,
                           stm_size_option='maxACCApprox', use_ltm=False)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_pred, y_test)


def by_AccuracyWeightedEnsemble(X_train, X_test, y_train, y_test):
    clf = AccuracyWeightedEnsembleClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_pred, y_test)


def exp_one_clf(q, ds_name, model_name, clf, with_fuzzy_rules, sn, X, y):
    """Experiment with one classifier."""
    # Record the start time.
    time_start = time.time()

    # 3. Partition the dataset. ====================================================================================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=sn)
    logging.debug("(Dataset: '%s'; Experiment: '%s'; Fuzzy rules: '%r'; SN: '%d-th') training: %s, test: %s",
                  ds_name, model_name, with_fuzzy_rules, sn, y_train.shape, y_test.shape)

    # 4. Train the models. =========================================================================================
    acc_test = 0
    if model_name == "FDT":
        acc_test = by_FDT(X_train, X_test, y_train, y_test)
    elif model_name == "LightGBM":
        if ds_name == "Covertype":
            y_train = y_train - 1
            y_test = y_test - 1
        acc_test = by_LightGBM(X_train, X_test, y_train, y_test)
    elif model_name == "XGBoost":
        acc_test = by_XGBoost(X_train, X_test, y_train, y_test)
    elif model_name == "CatBoost":
        acc_test = by_CatBoost(X_train, X_test, y_train, y_test)
    elif model_name == "HoeffdingTree":
        acc_test = by_HoeffdingTree(X_train, X_test, y_train, y_test)
    elif model_name == "HoeffdingAdaptiveTree":
        acc_test = by_HoeffdingAdaptiveTree(X_train, X_test, y_train, y_test)
    elif model_name == "SAMKNN":
        acc_test = by_SAMKNN(X_train, X_test, y_train, y_test)

    elapsed_time = time.time() - time_start

    # Debug message.
    logging.debug("=" * 100)
    logging.debug("(Dataset: '%s'; Experiment: '%s'; Fuzzy rules: '%r'; SN: '%d-th') %f, %f(s)",
                  ds_name, model_name, with_fuzzy_rules, sn, acc_test, elapsed_time)
    logging.debug("=" * 100)

    # Put the results into the queue to send back to main process.
    if not q.full():
        q.put([model_name, ds_name, with_fuzzy_rules, sn, acc_test, elapsed_time])


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
        column_names = ["model_name", "ds_name", "with_fuzzy_rules", "SN", "acc_test", "elapsed_time"]
        exp_res_df = pd.DataFrame(data=exp_results, columns=column_names)
        # Specify the dtypes of numeric columns required for mathematical calculation, otherwise an error will occur.
        exp_res_df["acc_test"] = exp_res_df["acc_test"].astype(float)
        exp_res_df["elapsed_time"] = exp_res_df["elapsed_time"].astype(float)

        # 1. Output all the result records.
        # Mathematical calculation before output: Add a new column to store the mean `acc_test`
        # in the `n_exp` experimental results of one model，and the same calculation for `acc_train`.
        exp_res_df["mean_acc_test"] = exp_res_df.groupby(["model_name",
                                                          "ds_name",
                                                          "with_fuzzy_rules"])["acc_test"].transform("mean")
        exp_res_df["mean_acc_test"] = exp_res_df.groupby(["model_name",
                                                          "ds_name",
                                                          "with_fuzzy_rules"])["elapsed_time"].transform("mean")
        filename = DirSave.EVAL_DATA.value + get_now_str(get_timestamp_str()) + "-exp-s1.csv"
        exp_res_df.to_csv(filename)
        # Debug message.
        logging.debug("Finished output: %s", filename)

        # 2. Output the mean-statistics of `acc_train` and `acc_train`.
        exp_res_mean_df = exp_res_df.groupby(["model_name",
                                              "ds_name",
                                              "with_fuzzy_rules"]).agg({"acc_test": "mean",
                                                                        "elapsed_time": "mean"})
        filename = DirSave.EVAL_DATA.value + get_now_str(get_timestamp_str()) + "-exp-s1-mean-stat.csv"
        exp_res_mean_df.to_csv(filename)
        # Debug message.
        logging.debug(exp_res_mean_df.values)
        logging.debug("Finished output: %s", filename)

        # 3. Output the whole statistics of `acc_train` and `acc_train`.
        exp_res_stat_df = exp_res_df.groupby(["model_name",
                                              "ds_name",
                                              "with_fuzzy_rules"])[["acc_test",
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
                if shape[0] > 1000:
                    ds_df, _ = train_test_split(ds_df, train_size=1000, stratify=ds_df.iloc[:, [-1]])
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

                n_exp = 100
                for i in range(n_exp):
                    # Experiment without fuzzy rules.
                    for comparison_mode, clf in NON_FUZZY_CLFS.items():
                        # 3, 4, 5 and 6 steps in sub-processes. ========================================================
                        # pool.apply_async(exp_one_clf,
                        #                  args=(q, ds_name, comparison_mode, clf, False, i, X, y,))
                        # if comparison_mode == "fgbdt_vs_nfgbdt":
                        #     if i < 10:
                        #         pool.apply_async(exp_one_clf,
                        #                          args=(q, ds_name, comparison_mode, clf, False, i, X, y,))
                        # else:
                        #     pool.apply_async(exp_one_clf,
                        #                      args=(q, ds_name, comparison_mode, clf, False, i, X, y,))

                        # Following is only used to check for possible exceptions.
                        res = pool.apply_async(exp_one_clf,
                                               args=(q, ds_name, comparison_mode, clf, False, i, X, y,))
                        res.get()

                    # Experiment with fuzzy rules.
                    for comparison_mode, clf in FUZZY_CLFS.items():
                        # 3, 4, 5 and 6 steps in sub-processes. =====================================================
                        pool.apply_async(exp_one_clf,
                                         args=(q, ds_name, comparison_mode, clf, True, i, X_plus_dms, y,))
                        # if comparison_mode == "fgbdt_vs_nfgbdt":
                        #     if i < 10:
                        #         pool.apply_async(exp_one_clf,
                        #                          args=(q, ds_name, comparison_mode, clf, True, i, X_plus_dms, y,))
                        # else:
                        #     pool.apply_async(exp_one_clf,
                        #                      args=(q, ds_name, comparison_mode, clf, True, i, X_plus_dms, y,))

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
