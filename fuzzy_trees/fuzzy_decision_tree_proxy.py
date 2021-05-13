# _*_coding:utf-8_*_
"""
@author: Zhaoqing Liu
@email: Zhaoqing.Liu-1@student.uts.edu.au
@date: 03/12/2020 4:42 pm
@desc:
TODO:
    1. Done - Do feature fuzzification in data_preprocessing_funcs.py
    2. Done - Add function implementing splitting criteria fuzzification in util_criterion_funcs.py
    3. Upgrade program from Python to Cython to speed up the algorithms.
"""
import copy
import multiprocessing
import os
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from decimal import Decimal

from sklearn.metrics import accuracy_score, balanced_accuracy_score, average_precision_score, brier_score_loss, f1_score
from sklearn.model_selection import KFold

from fuzzy_trees.settings import NUM_CPU_CORES_REQ, PathSave, EvaluationType
from fuzzy_trees.util_comm import get_timestamp_str, get_today_str
from fuzzy_trees.util_criterion_funcs import calculate_entropy, calculate_gini, calculate_variance, \
    calculate_standard_deviation
from fuzzy_trees.util_data_handler import load_data_clf
from fuzzy_trees.util_data_processing_funcs import extract_fuzzy_features


# =============================================================================
# Types and constants
# =============================================================================
from fuzzy_trees.util_plotter import plot_multi_curves

CRITERIA_FUNC_CLF = {"entropy": calculate_entropy, "gini": calculate_gini}
CRITERIA_FUNC_REG = {"mse": calculate_variance, "mae": calculate_standard_deviation}

# CLF_TYPE = {"ID3": [calculate_entropy, calculate_information_gain],
#              "C45": [calculate_gini, calculate_information_gain_ratio],
#              "CART": [calculate_gini, calculate_impurity_gain,]}


class FuzzificationParams:
    """
    Class that encapsulates all the parameters
    (excluding functions) of the fuzzification
    settings to be used by a fuzzy decision tree.

    Parameters
    ----------

    Attributes
    ----------

    """

    def __init__(self, r_seed=0, conv_size=1, conv_k=3, num_iter=1, feature_filter_func=None,
                 feature_filter_func_param=None, dataset_df=None, dataset_mms_df=None, X_fuzzy_dms=None):
        self.r_seed = r_seed
        self.conv_size = conv_size
        self.conv_k = conv_k
        self.num_iter = num_iter
        self.feature_filter_func = feature_filter_func
        self.feature_filter_func_param = feature_filter_func_param
        self.dataset_df = dataset_df
        self.dataset_mms_df = dataset_mms_df


# =============================================================================
# Decision tree component
# =============================================================================


class Node:
    """
    A Class that encapsulates the data of the node (including root node) and
    leaf node in a decision tree.

    Parameters
    ----------
    split_rule: SplitRule, default=None
        The split rule represented by the feature selected as a node, and
        branching decisions are made based on this rule.

    leaf_value: float, default=None
        The predicted value indicated at a leaf node. In the classification
        tree it is the predicted class, and in the regression tree it is the
        predicted value.
        NB: Only a leaf node has this attribute value.

    leaf_proba: float, default=None
        The predicted probability indicated at a leaf node. Only works in the
        classification tree.
        NB: Only a leaf node has this attribute value.

    branch_true: Node, default=None
        The next node in the decision path when the feature value of a sample
        meets the split rule split_rule.

    branch_false: Node, default=None
        The next node in the decision path when the feature value of a sample
        does not meet the split rule split_rule.
    """

    def __init__(self, split_rule=None, leaf_value=None, leaf_proba=None, branch_true=None, branch_false=None):
        self.split_rule = split_rule
        self.leaf_value = leaf_value
        self.leaf_proba = leaf_proba
        self.branch_true = branch_true
        self.branch_false = branch_false


class SplitRule:
    """
    A Class that encapsulates the data of a split rule, which is one of
    attributes of the node (including root node) in a decision tree.

    Parameters
    ----------
    feature_idx: int, default=None
        The index of the feature selected as the node representing a split rule.

    split_value: float, default=None
        The value from the feature indexed as feature_idx representing a split
        rule, on which branching decisions are made based.
    """

    def __init__(self, feature_idx=None, split_value=None):
        self.feature_idx = feature_idx
        self.split_value = split_value


class BinarySubtrees:
    """
    A class that encapsulates two subtrees under a node, and each subtree has
    two subsets of the samples' features and target values that has been split.

    Parameters
    ----------
    subset_true_X: {array-like, sparse matrix} of shape (n_samples, n_features)
        The subset of feature values of the samples that meet the split_rule
        after splitting.

    subset_true_y: array-like of shape (n_samples,) or (n_samples, n_outputs)
        The subset of target values of the samples that meet the split_rule
        after splitting.

    subset_false_X: {array-like, sparse matrix} of shape (n_samples, n_features)
        The subset of feature values of the samples that do not meet the
        split_rule after splitting.

    subset_false_y: array-like of shape (n_samples,) or (n_samples, n_outputs)
        The subset of target values of the samples that do not meet the
        split_rule after splitting.
    """

    def __init__(self, subset_true_X=None, subset_true_y=None, subset_false_X=None, subset_false_y=None):
        self.subset_true_X = subset_true_X
        self.subset_true_y = subset_true_y
        self.subset_false_X = subset_false_X
        self.subset_false_y = subset_false_y


# =============================================================================
# Interface for decision tree classes
# =============================================================================


class DecisionTreeInterface(metaclass=ABCMeta):
    """
    Interface for decision tree classes based on different algorithms.

    Warning: This interface should not be used directly.
    Use derived algorithm classes instead.

    NB: The purpose of this interface is to establish protocols
    for functions (excluding constructor and attributes) in
    classification decision trees and regression decision trees
    that to be developed.
    """

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def print_tree(self, tree=None, indent="  ", delimiter="=>"):
        pass


# =============================================================================
# Public proxy class for different decision trees
# =============================================================================


class FuzzyDecisionTreeProxy(DecisionTreeInterface):
    """
    Proxy class for different decision trees.

    NB: The role of this class is to unify the external calls of different
    decision tree classes and implement dependency injection for those
    decision tree classes.

    The arguments of the constructors for different decision trees should
    belong to a subset of the following parameters.

    Parameters:
    -----------
    fdt_class: Class, default=None
        The fuzzy decision tree estimator specified.

    disable_fuzzy: bool, default=False
        Set whether the specified fuzzy decision tree uses the fuzzification.
        If disable_fuzzy=True, the specified fuzzy decision tree is equivalent
        to a naive decision tree.

    X_fuzzy_dms: {array-like, sparse matrix} of shape (n_samples, n_features)
        Three-dimensional array, and each element of the first dimension of the
        array is a two-dimensional array of corresponding feature's fuzzy sets.
        Each two-dimensional array is of shape of (n_samples, n_fuzzy_sets), but
        has transformed membership degree of the feature values to corresponding
        fuzzy sets.

    fuzzification_params: FuzzificationParams, default=None
        Class that encapsulates all the parameters of the fuzzification settings
        to be used by the specified fuzzy decision tree.

    criterion_func: {"gini", "entropy"} for a classifier, {"mse", "mae"} for a regressor
        The criterion function used by the function that calculates the impurity
        gain of the target values.

    max_depth: int, default=float("inf")
        The maximum depth of the tree.

    min_samples_split: int, default=2
        The minimum number of samples required to split a node. If a node has a
        sample number above this threshold, it will be split, otherwise it
        becomes a leaf node.

    min_impurity_split: float, default=1e-7
        The minimum impurity required to split a node. If a node's impurity is
        above this threshold, it will be split, otherwise it becomes a leaf node.

    Attributes
    ----------
    root: Node
        The root node of a decision tree.

    _impurity_gain_calculation_func: function
        The function to calculate the impurity gain of the target values.

    _leaf_value_calculation_func: function
        The function to calculate the predicted value if the current node is a
        leaf:
        - In a classification tree, it gives the target value with the highest
         probability.
        - In the regression tree, it gives the average of all the target values.

    _is_one_dim: bool
        The Boolean value that indicates whether the y is a multi-dimensional set,
        which means whether y is one-hot encoded.

    _best_split_rule: SplitRule
        The split rule including the index of the best feature to be used, and
        the best value in the best feature.

    _best_binary_subtrees: BinarySubtrees
        The binary subtrees including two subtrees under a node, and each subtree
        is a subset of the sample that has been split. It is one of attributes of
        the node (including root node) in a decision tree.

    _best_impurity_gain: float
        The best impurity gain calculated based on the current split subtrees
        during a tree building process.

    _fuzzy_sets: {array-like, sparse matrix} of shape (n_features, n_coefficients)
        All the coefficients of the degree of membership sets based on the
        current estimator. They will be used to calculate the degree of membership
        of the features of new samples before predicting those samples. Therefore,
        their life cycle is consistent with that of the current estimator.
        They are generated in the feature fuzzification before training the
        current estimator.
        NB: To be used in version 1.0.

    References
    ----------


    Examples
    --------

    """

    # All parameters in this constructor should have default values.
    def __init__(self, fdt_class=None, disable_fuzzy=False, X_fuzzy_dms=None, fuzzification_params=None,
                 criterion_func=None,
                 max_depth=float("inf"), min_samples_split=2, min_impurity_split=1e-7, **kwargs):
        # Construct a instance of the specified fuzzy decision tree.
        if fdt_class is not None:
            self.estimator = fdt_class(disable_fuzzy=disable_fuzzy, X_fuzzy_dms=X_fuzzy_dms,
                                       fuzzification_params=fuzzification_params, criterion_func=criterion_func,
                                       max_depth=max_depth, min_samples_split=min_samples_split,
                                       min_impurity_split=min_impurity_split, **kwargs)
        self.ds_4_plotting = [[]]
        self.df_4_plotting = None

    def fit(self, X_train, y_train):
        """
        Train a decision tree estimator from the training set (X_train, y_train).

        Parameters:
        -----------
        X_train: {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y_train: array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.
        """
        # Start training to get a fitted estimator.
        try:
            self.estimator.fit(X_train, y_train)
        except Exception as e:
            print(traceback.format_exc())

    def predict(self, X):
        """
        Predict the target values of the input samples X.

        In classification, a predicted target value is the one with the
        largest number of samples of the same class in a leaf.

        In regression, the predicted target value is the mean of the target
        values in a leaf.

        Parameters:
        -----------
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples to be predicted.

        Returns
        -------
        pred_y: list of n_outputs such arrays if n_outputs > 1
            The target values of the input samples.
        """
        try:
            return self.estimator.predict(X)
        except Exception as e:
            print(traceback.format_exc())

    def print_tree(self, tree=None, indent="  ", delimiter="-->"):
        """
        Recursively (in a top-to-bottom approach) print the built decision tree.

        Parameters:
        -----------
        tree: Node
            The root node of a decision tree.

        indent: str
            The indentation symbol used when printing subtrees.

        delimiter: str
            The delimiter between split rules and results.
        """
        try:
            self.estimator.print_tree(tree=tree, indent=indent, delimiter=delimiter)
        except Exception as e:
            print(traceback.format_exc())

    # =============================================================================
    # Pre-Train and Plotting Functions
    # =============================================================================
    def pre_train_clf(self, ds_name_list, conv_k_lim, fuzzy_reg_lim):
        """
        Pretrain a set of FDT classifiers from specified datasets in parallel.

        The fuzzy feature extraction before training is based on specified
        fuzzy regulation coefficients and a number of fuzzy clusters that each
        feature belongs to.

        NB: Use this function to prepare evaluation and plotting data when
        you need to evaluate the effect of different degrees of fuzzification
        on model training.

        Parameters
        ----------
        ds_name_list: array-like
        fuzzy_reg_lim: tuple, (start, stop, step)
        conv_k_lim: tuple, (start, stop, step)

        Returns
        -------

        """
        # Create a connection used to communicate between multi-processes.
        q = multiprocessing.Manager().Queue()

        # Create a pool to manage all child processes in multi-process mode.
        pool = multiprocessing.Pool(processes=NUM_CPU_CORES_REQ)

        # Pretrain classifiers and then get their metrics for evaluation in parallel.
        for ds_name in ds_name_list:
            for conv_k in range(conv_k_lim[0], conv_k_lim[1] + 1, conv_k_lim[2]):
                fuzzy_reg = fuzzy_reg_lim[0]
                while fuzzy_reg <= fuzzy_reg_lim[1]:
                    # Start a child process to fit 10 classifiers and then to get the mean metric of them.
                    pool.apply_async(self._get_one_mean_clf, args=(q, ds_name, conv_k, fuzzy_reg,))
                    fuzzy_reg = float(Decimal(str(fuzzy_reg)) + Decimal(str(fuzzy_reg_lim[2])))

        pool.close()
        pool.join()

        # Encapsulate and save all results from processes.
        self._encapsulate_save_res_clf(q=q)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++", np.shape(self.ds_4_plotting))
        print(self.ds_4_plotting)

    def plot_fuzzy_reg_vs_err_sep(self, filename):
        """
        Separately plot fuzzy regulation coefficient versus training error and
        test error on each number of fuzzy clusters.

        Returns
        -------

        """
        # q.put([[ds_name, conv_k, fuzzy_reg, acc_train_mean, std_train, acc_test_mean, std_test]])
        if np.size(self.ds_4_plotting, axis=0) > 1:
            # Plot based on data in memory.
            # Plot the comparison of training error versus test error, and both curves are fuzzy thresholds versus accuracies.
            # Illustrate how the performance on unseen data (test data) is different from the performance on training data.
            # x_lower_limit, x_upper_limit = np.min(x_train), np.max(x_train)
            # y_lower_limit = np.min(y_train) if np.min(y_train) < np.min(y_test) else np.min(y_test)
            # y_upper_limit = np.max(y_train) if np.max(y_train) > np.max(y_test) else np.max(y_test)
            # print("x_limits and y_limits are:", x_lower_limit, x_upper_limit, y_lower_limit, y_upper_limit)
            self.df_4_plotting = pd.DataFrame()
            assert self.df_4_plotting is not None, "No pre-train data stored in self.df_4_plotting"
            for ds in self.df_4_plotting.groupby(["ds_name"]).groups:
                for (ds_name, idxs) in ds.items():
                    plot_multi_curves(coordinates=data,
                                      title="Fuzzy Threshold vs Error -- {} -- {}".format(comparing_mode.name, ds_name),
                                      x_label="Fuzzy threshold",
                                      y_label="Error Rate",
                                      legends=["Train", "Test"],
                                      f_name=EvaluationType.FUZZY_TH_VS_ACC.value + "_" + comparing_mode.name + "_" + ds_name + ".png")
        else:
            df = pd.read_csv(filename)

    def plot_fuzzy_reg_vs_err_all(self):
        """
        Plot fuzzy regulation coefficient versus training error and test error
        on all numbers of fuzzy clusters together.

        Returns
        -------

        """
        if np.size(self.ds_4_plotting, axis=0) > 1:
            # TODO: Plot based on data in memory.
            pass
        else:
            # TODO: Plot based on data in file.
            pass

    def _get_one_mean_clf(self, q, ds_name, conv_k, fuzzy_reg):
        """
        Get one mean metric of 10 classifiers trained on the specified dataset.

        The fuzzy feature extraction before training is based on specified
        fuzzy regulation coefficients and a number of fuzzy clusters that each
        feature belongs to.

        Parameters
        ----------
        q: multiprocessing.queue.Queue
        ds_name: str
        conv_k: int
        fuzzy_reg: float

        Returns
        -------

        """
        curr_pid = os.getpid()
        print("    |-- ({} Child-process) Train 10 classifiers on: {}.".format(curr_pid, ds_name))
        print("    |-- ({} Child-process) Preprocess fuzzy feature extraction based on parameters: {}, {}.".format(curr_pid, conv_k, fuzzy_reg))

        # Load data.
        df = load_data_clf(ds_name)

        # Preprocess fuzzy feature extraction (only for fuzzy decision tree).
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_fuzzy_pre = X.copy()
        # - Step 1: Standardise feature scaling.
        # X_fuzzy_pre[:, :] -= X_fuzzy_pre[:, :].min()
        # X_fuzzy_pre[:, :] /= X_fuzzy_pre[:, :].max()
        # - Step 2: Extract fuzzy features.
        X_dms = extract_fuzzy_features(X=X_fuzzy_pre, conv_k=conv_k, fuzzy_reg=fuzzy_reg)
        X_plus_dms = np.concatenate((X, X_dms), axis=1)
        # print("************* Shape before fuzzification:", np.shape(X))
        # print("************* Shape after fuzzification:", np.shape(X_plus_dms))

        # Fit a group of models, and then get the mean of their accuracy results.
        acc_train_list = []
        acc_test_list = []
        for i in range(10):
            print("        |-- ({} Child-process) {}-th fitting......".format(curr_pid, i))

            # Split training and test sets by hold-out partition method.
            # X_train, X_test, y_train, y_test = train_test_split(X_fuzzy_pre, y, test_size=0.4)

            kf = KFold(n_splits=2, random_state=i, shuffle=True)
            for train_index, test_index in kf.split(X):
                y_train, y_test = y[train_index], y[test_index]

                # Train and test one model, and then get its accuracy results.
                X_train, X_test = X_plus_dms[train_index], X_plus_dms[test_index]
                accuracy_train, accuracy_test = self._fit_one_clf(X_train=X_train, X_test=X_test,
                                                                  y_train=y_train, y_test=y_test,
                                                                  ds_name=ds_name, conv_k=conv_k, fuzzy_reg=fuzzy_reg)
                acc_train_list.append(accuracy_train)
                acc_test_list.append(accuracy_test)

        # Calculate the mean of the accuracy and errors.
        acc_train_mean = np.mean(acc_train_list)
        std_train = np.std(acc_train_list)
        acc_test_mean = np.mean(acc_test_list)
        std_test = np.std(acc_test_list)
        print("    |-- ========================================================================================")
        print("    |-- ({} Child-process) Train 10 classifiers on: {}.".format(curr_pid, ds_name))
        print("    |-- Mean train acc:", acc_train_mean, "  std:", std_train)
        print("    |-- Mean test acc:", acc_test_mean, "  std:", std_test)
        print("    |-- ========================================================================================")

        # Put the results in the connection between the main process and the child processes.
        # !!! NB: The return value should be a 2-dimensional ndarray. Or, the return value is
        # a dictionary, and its key is the dataset name and its value is a 2-d matrix ndarray.
        if not q.full():
            q.put([[ds_name, conv_k, fuzzy_reg, acc_train_mean, std_train, acc_test_mean, std_test]])

    def _fit_one_clf(self, X_train, X_test, y_train, y_test, ds_name, conv_k, fuzzy_reg):
        """
        Fit one classifier and get its evaluation metrics.

        Evaluation metrics used in classification include accuracy, f1, roc_auc,
        etc.
        See more on https://scikit-learn.org/stable/modules/model_evaluation.html

        Parameters
        ----------
        X_train
        X_test
        y_train
        y_test

        Returns
        -------

        """
        # # Record the start time used to calculate the time spent fitting one model.
        # time_start = time.time()

        # Fit the initialised model.
        self.estimator.fit(X_train, y_train)
        # clf.print_tree()

        # Get the training accuracy and test accuracy of the fitted (trained) estimator.
        y_pred_train = self.predict(X_train)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        # balanced_accuracy_train = balanced_accuracy_score(y_train, y_pred_train)
        # neg_brier_score_train = brier_score_loss(y_train, y_pred_train)
        y_pred_test = self.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        # balanced_accuracy_test = balanced_accuracy_score(y_test, y_pred_test)
        # neg_brier_score_test = brier_score_loss(y_test, y_pred_test)
        # print("    Fuzzy accuracy train:", accuracy_train)
        # print("    Fuzzy accuracy test:", accuracy_test)

        # Pickle the current model.
        joblib.dump(value=self.estimator, filename=PathSave.MODELS + "clf_" + ds_name + "_" + conv_k + "_" + fuzzy_reg + ".mdl")
        # trained_clf = joblib.load(filename=PATH_SAVE_MODELS + "clf-" + ds_name + "-" + conv_k + "-" + fuzzy_reg + ".mdl")

        # # Display the elapsed time.
        # print("            |-- Time elapsed fitting one model:", time.time() - time_start, "s")

        return accuracy_train, accuracy_test

    def _encapsulate_save_res_clf(self, q):
        """
        Encapsulate each process's result into a container for plotting and
        saving into a file.

        Parameters
        ----------
        q: multiprocessing.queue.Queue

        Returns
        -------

        """
        while not q.empty():
            # q.put([[ds_name, conv_k, fuzzy_reg, acc_train_mean, std_train, acc_test_mean, std_test]])
            results = q.get()
            for res in results:
                if len(np.shape(res)) == 1:
                    res = np.expand_dims(res, axis=0)
                if np.size(self.ds_4_plotting) == 0:
                    self.ds_4_plotting = res
                else:
                    self.ds_4_plotting = np.concatenate((self.ds_4_plotting, res), axis=0)

        # Save the experiment's results into a file.
        self.df_4_plotting = pd.DataFrame()
        column_names = ["ds_name", "conv_k", "fuzzy_reg", "acc_train_mean", "std_train", "acc_test_mean", "std_test"]
        self.df_4_plotting = pd.DataFrame(data=self.ds_4_plotting, columns=column_names)
        self.df_4_plotting.to_csv(PathSave.EVAL_DATA + EvaluationType.FUZZY_REG_VS_ACC_ON_CONV_K.value + "_" + get_today_str() + ".csv")

    def _fit_one_regr(self, X_train, X_test, y_train, y_test):
        """
        Fit one regressor and get its evaluation metrics.

        Evaluation metrics used in regression include neg_mean_absolute_error,
        neg_root_mean_squared_error, r2, etc.
        See more on https://scikit-learn.org/stable/modules/model_evaluation.html

        Parameters
        ----------
        X_train
        X_test
        y_train
        y_test

        Returns
        -------

        """
        pass


if __name__ == '__main__':
    pass
