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
import warnings
warnings.filterwarnings("always")
from abc import ABCMeta, abstractmethod
from decimal import Decimal

from deprecated.sphinx import deprecated
from sklearn.metrics import accuracy_score, balanced_accuracy_score, average_precision_score, brier_score_loss, f1_score
from sklearn.model_selection import KFold

from fuzzytrees.settings import NUM_CPU_CORES_REQ, DirSave, EvaluationType, NUM_GRP_MDLS
from fuzzytrees.util_comm import get_timestamp_str, get_today_str
from fuzzytrees.util_criterion_funcs import calculate_entropy, calculate_gini, calculate_variance, \
    calculate_standard_deviation
from fuzzytrees.util_data_handler import load_data_clf
from fuzzytrees.util_data_processing_funcs import extract_fuzzy_features
from fuzzytrees.util_plotter import plot_multi_lines, plot_multi_lines_subplots

# =============================================================================
# Types and constants
# =============================================================================

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
# Public wrapper class for different decision trees
# =============================================================================


class FuzzyDecisionTreeWrapper(DecisionTreeInterface):
    """
    Wrapper class for different decision trees.

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
        self.fdt_class = fdt_class
        self.disable_fuzzy = disable_fuzzy
        self.X_fuzzy_dms = X_fuzzy_dms
        self.fuzzification_params = fuzzification_params
        self.criterion_func = criterion_func
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.kwargs = kwargs

        self.ds_pretrain = None  # A list used to contain data generated by pretraining.
        self.df_pretrain = None  # A dataframe used to contain data generated by pretraining.
        self.filename_ds_pretrain = None  # A name of the file used to save data generated by pretraining.
        self.enable_pkl_mdl = False  # Set whether enable pickling fitted models.

        # Ensure the directories for saving files is existing.
        for item in DirSave:
            if not os.path.exists(item.value):
                os.makedirs(item.value)

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
    # Functions to pretrain FDTs and plot their evaluation
    # =============================================================================
    def pretrain_fuzzy_clf(self, ds_name_list, conv_k_lim, fuzzy_reg_lim):
        """
        Pretrain different groups of FDT classifiers from specified datasets
        in parallel (multi-process/master-worker mode).

        The fuzzy feature extraction before pretraining is based on specified
        fuzzy regulation coefficients and a number of fuzzy clusters that each
        feature belongs to.

        NB: Use this function to prepare evaluation and plotting data when
        you need to evaluate the effect of different degrees of fuzzification
        on model training in advance.

        Parameters
        ----------
        ds_name_list: array-like
        fuzzy_reg_lim: tuple, (start, stop, step)
        conv_k_lim: tuple, (start, stop, step)

        Returns
        -------

        """
        # Create a connection used to communicate between main process and its child processes.
        q = multiprocessing.Manager().Queue()

        # Create a pool for main process to manage its child processes in parallel.
        pool = multiprocessing.Pool(processes=NUM_CPU_CORES_REQ)

        # Pretrain different groups of classifiers and get each group's evaluation scores in parallel.
        for ds_name in ds_name_list:
            for conv_k in range(conv_k_lim[0], conv_k_lim[1] + 1, conv_k_lim[2]):
                fuzzy_reg = fuzzy_reg_lim[0]
                while fuzzy_reg <= fuzzy_reg_lim[1]:
                    # Start a child process to fit a group of classifiers on a specified dataset and
                    # get the mean of their evaluation scores.
                    pool.apply_async(self._get_one_mean_fuzzy_clf, args=(q, ds_name, conv_k, fuzzy_reg,))
                    fuzzy_reg = float(Decimal(str(fuzzy_reg)) + Decimal(str(fuzzy_reg_lim[2])))

        pool.close()
        pool.join()

        # Encapsulate and save all data received from the child processes.
        self._encapsulate_save_data_fuzzy_clf(q=q)

    def _get_one_mean_fuzzy_clf(self, q, ds_name, conv_k, fuzzy_reg):
        """
        Fit a group of fuzzy classifiers on a specified dataset and get the
        mean of their evaluation scores.

        The fuzzy feature extraction before pretraining is based on specified
        fuzzy regulation coefficients and numbers of fuzzy clusters that each
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
        print("    |-- ({} Child-process) Pretrain a group of classifiers on: {}.".format(curr_pid, ds_name))
        print("    |-- ({} Child-process) Preprocess fuzzy feature extraction based on parameters: {}, {}.".format(
            curr_pid, conv_k, fuzzy_reg))

        # Load data.
        df = load_data_clf(ds_name)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Preprocess fuzzy feature extraction (only for fuzzy decision tree).
        X_plus_dms = []
        if fuzzy_reg == 0 or fuzzy_reg == 1:
            self.estimator.disable_fuzzy = True
            X_plus_dms = X
        else:
            self.estimator.disable_fuzzy = self.disable_fuzzy
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
        for i in range(NUM_GRP_MDLS):
            print("        |-- ({} Child-process) {}-th fitting.".format(curr_pid, i))

            # Split training and test sets by hold-out partition method.
            # X_train, X_test, y_train, y_test = train_test_split(X_fuzzy_pre, y, test_size=0.4)
            kf = KFold(n_splits=2, random_state=i, shuffle=True)
            for train_index, test_index in kf.split(X):
                y_train, y_test = y[train_index], y[test_index]

                # Fit a model, and then get its evaluation scores.
                X_train, X_test = X_plus_dms[train_index], X_plus_dms[test_index]
                accuracy_train, accuracy_test = self._fit_one_fuzzy_clf(X_train=X_train, X_test=X_test,
                                                                        y_train=y_train, y_test=y_test,
                                                                        ds_name=ds_name, conv_k=conv_k,
                                                                        fuzzy_reg=fuzzy_reg, sn=i)
                acc_train_list.append(accuracy_train)
                acc_test_list.append(accuracy_test)

        # Calculate the mean of the fitted model's evaluation scores.
        acc_train_mean = np.mean(acc_train_list)
        err_train_mean = 1 - np.abs(np.mean(acc_train_list))
        std_train = np.std(acc_train_list)
        acc_test_mean = np.mean(acc_test_list)
        err_test_mean = 1 - np.abs(np.mean(acc_test_list))
        std_test = np.std(acc_test_list)
        print("    |-- ========================================================================================")
        print("    |-- ({} Child-process) Pretrain a group of classifiers on: {}.".format(curr_pid, ds_name))
        print("    |-- Mean train acc:", acc_train_mean, "  std:", std_train)
        print("    |-- Mean test acc:", acc_test_mean, "  std:", std_test)
        print("    |-- ========================================================================================")

        # Put the data in the connection between the main process and its child processes.
        # !!! NB: The data should be a 2-dimensional ndarray, or a dictionary with key,
        # which is the dataset name, and value, which is a 2-d matrix ndarray.
        if not q.full():
            q.put([[ds_name, conv_k, fuzzy_reg, err_train_mean, std_train, err_test_mean, std_test]])

    def _fit_one_fuzzy_clf(self, X_train, X_test, y_train, y_test, ds_name, conv_k, fuzzy_reg, sn):
        """
        Fit a fuzzy classifier and get its evaluation scores.

        See more about evaluation scores on https://scikit-learn.org/stable/modules/model_evaluation.html

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

        # Fit the initialised model (rebuild a new tree inside).
        self.fit(X_train, y_train)
        # clf.print_tree()

        # Get the evaluation scores of the fitted estimator.
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

        # Pickle the fitted model.
        if self.enable_pkl_mdl:
            filename = DirSave.MODELS.value + get_today_str() + "_" + "clf_" + str(conv_k) + "_" + str(fuzzy_reg) + "_" + ds_name + "_" + str(sn) + ".mdl"
            joblib.dump(value=self.estimator, filename=filename)
            # trained_clf = joblib.load(filename=filename)

        # # Display the elapsed time.
        # print("        |-- ({} Child-process) Time elapsed fitting one model:", time.time() - time_start, "s")

        return accuracy_train, accuracy_test

    def _encapsulate_save_data_fuzzy_clf(self, q):
        """
        Encapsulate and save all data received from the child processes
        when pretraining a group of fuzzy classifiers.

        Save the data in memory for immediate plotting, and a copy of the
        data in a file for future plotting against historical data.

        Parameters
        ----------
        q: multiprocessing.queue.Queue

        Returns
        -------

        """
        # Get data via connection between main process and its child processes.
        while not q.empty():
            # q.put([[ds_name, conv_k, fuzzy_reg, err_train_mean, std_train, err_test_mean, std_test]])
            data = q.get()
            if len(np.shape(data)) == 1:
                data = np.expand_dims(data, axis=0)
            if self.ds_pretrain is None:
                self.ds_pretrain = data
            else:
                self.ds_pretrain = np.concatenate((self.ds_pretrain, data), axis=0)

        # Save the collected data into a file.
        if self.ds_pretrain is not None:
            self.df_pretrain = pd.DataFrame()
            column_names = ["ds_name", "conv_k", "fuzzy_reg", "err_train_mean", "std_train", "err_test_mean",
                            "std_test"]
            self.df_pretrain = pd.DataFrame(data=self.ds_pretrain, columns=column_names)
            filename = DirSave.EVAL_DATA.value + get_today_str() + "_" + EvaluationType.FUZZY_REG_VS_ERR_ON_CONV_K.value + ".csv"
            self.df_pretrain.to_csv(filename)
        print("Main Process (%s) Saved data as the shape:".format(os.getpid()), self.df_pretrain)

    def plot_fuzzy_reg_vs_err(self, filename=None):
        """
        Plot fuzzy regulation coefficient versus training error and
        test error on each numbers of fuzzy clusters respectively.

        Illustrate how the performance on unseen data (test data)
        is different from the performance on training data.

        Parameters
        ----------
        filename: str, default None
            Fetch the data from the specified file if filename is
            not None. Otherwise try from memory and the latest file
            in the default directory in turn.

        Returns
        -------

        """
        # Fetch data for plotting from the specified file if filename is not None.
        if filename is not None:
            self.df_pretrain = pd.read_csv(filename)

        # Otherwise fetch data from memory and the latest file in the default directory in turn.
        if self.df_pretrain is None:
            # NB: The list returned by listdir() is in arbitrary order.
            filename_list = os.listdir(DirSave.EVAL_DATA.value)
            if len(filename_list) > 0:
                filename_list = sorted(filename_list)
                self.df_pretrain = pd.read_csv(DirSave.EVAL_DATA.value + filename_list[-1])

        assert self.df_pretrain is not None, "Not any data for plotting. Please execute the function pretrain() first."
        # q.put([[ds_name, conv_k, fuzzy_reg, err_train_mean, std_train, err_test_mean, std_test]])
        ds_names = self.df_pretrain["ds_name"].unique()
        for ds_name in ds_names:
            df_4_ds_name = self.df_pretrain[self.df_pretrain["ds_name"] == ds_name]
            conv_ks = df_4_ds_name["conv_k"].unique()
            # conv_ks = sorted(conv_ks)  # It doesn't matter if it's drawn in ascending order from conv_k.
            for conv_k in conv_ks:
                df_4_conv_k = df_4_ds_name[df_4_ds_name["conv_k"] == conv_k]
                df_4_conv_k = df_4_conv_k.sort_values(by="fuzzy_reg", ascending=True)  # ascending is True by default.
                coordinates = df_4_conv_k[["fuzzy_reg", "err_train_mean", "err_test_mean"]].astype("float").values
                # print("+++++++++++++++++++++++++++++++++++++++++++++", type(df_4_conv_k["err_train_mean"].values[1]))
                # x_lower_limit, x_upper_limit = np.min(df_4_conv_k[["fuzzy_reg"]].values), np.max(df_4_conv_k[["fuzzy_reg"]].values)
                # y_lower_limit = np.min(df_4_conv_k[["err_train_mean"]].values) if np.min(df_4_conv_k[["err_train_mean"]].values) < np.min(df_4_conv_k[["err_test_mean"]].values) else np.min(df_4_conv_k[["err_test_mean"]].values)
                # y_upper_limit = np.max(df_4_conv_k[["err_train_mean"]].values) if np.max(df_4_conv_k[["err_train_mean"]].values) > np.max(df_4_conv_k[["err_test_mean"]].values) else np.max(df_4_conv_k[["err_test_mean"]].values)
                # print("x_limits and y_limits are:", x_lower_limit, x_upper_limit, y_lower_limit, y_upper_limit)

                plot_multi_lines(coordinates=coordinates,
                                 title="Fuzzy Reg Coeff vs Error - conv_k {} - {}".format(conv_k, ds_name),
                                 x_label="Fuzzy Regulation Coefficient",
                                 y_label="Error Rate",
                                 legends=["Train", "Test"],
                                 fig_name=DirSave.EVAL_FIGURES.value + get_today_str() + "_" + EvaluationType.FUZZY_REG_VS_ERR_ON_CONV_K.value + "_" + str(
                                     conv_k) + "_" + ds_name + ".png")

    def _fit_one_fuzzy_regr(self, X_train, X_test, y_train, y_test):
        """
        Fit a fuzzy regressor and get its evaluation scores.

        See more about evaluation scores on https://scikit-learn.org/stable/modules/model_evaluation.html

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
    # df = pd.DataFrame({"id": [1, 2, 3, 4], "name": ["sam", "sam", "peter", "jack"], "number": [8, 6, 8, 2]})
    # df_after = df[["name", "number"]]
    # print(df_after.values)
    # unique_values = df["name"].unique()
    # for idx, val in enumerate(unique_values):
    #     print(idx, val)
    # print(np.size(unique_values))
    # print(np.shape(unique_values))
    # print(type(unique_values))
    # print(unique_values)
    # for unique_value in unique_values:
    #     print(unique_value)
    #     sub_df = df[df["name"] == unique_value]
    # print(type(sub_df))
    # print(type(sub_df.values))
    # print(type(sub_df["number"]))
    # print(sub_df["number"].values)
    # print(sub_df)
    # sdf = sub_df.sort_values(by="number")
    # print(sdf)

    # ds = df.groupby(["name"]).groups
    # print(ds)
    # for (name, data) in ds.items():
    #     print(name)
    #     print(type(data))
    #     print(type(data.values))
    #     print(type(df.iloc[data, :]))
    #     print(type(df.iloc[data.values, :]))

    # a = np.array([2, 8, 5])
    # print(a)
    # b = a - 1
    # print(b)

    dir = DirSave.EVAL_DATA.value + get_today_str()
    print(os.path.exists(dir))
    if not os.path.exists(dir) or not os.path.isdir(dir):
        os.makedirs(dir)

    pass
