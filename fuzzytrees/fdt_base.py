# _*_coding:utf-8_*_
"""
@author : Zhaoqing Liu
@email  : Zhaoqing.Liu-1@student.uts.edu.au
"""
import logging
import multiprocessing
import os
import traceback
import warnings
from abc import ABCMeta, abstractmethod
from decimal import Decimal
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from fuzzytrees.settings import DirSave, NUM_CPU_CORES_REQ, NUM_GRP_MDLS, EvaluationType
from fuzzytrees.util_comm import get_today_str
from fuzzytrees.util_tree_criterion_funcs import calculate_proba, calculate_entropy, calculate_gini, calculate_variance, \
    calculate_standard_deviation
from fuzzytrees.util_data_handler import load_data_clf
from fuzzytrees.util_preprocessing_funcs import extract_fuzzy_features
from fuzzytrees.util_plotter import plot_multi_lines

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

# =============================================================================
# Types and constants
# =============================================================================

CRITERIA_FUNC_CLF = {"entropy": calculate_entropy, "gini": calculate_gini}
CRITERIA_FUNC_REG = {"mse": calculate_variance, "mae": calculate_standard_deviation}


# CLF_TYPE = {"ID3": [calculate_entropy, calculate_information_gain],
#              "C45": [calculate_gini, calculate_information_gain_ratio],
#              "CART": [calculate_gini, calculate_impurity_gain,]}


class FuzzificationOptions:
    """
    A protocol message class that encapsulates all the options (excluding
    functions) of the fuzzification settings used by a fuzzy model.

    Parameters
    ----------

    Attributes
    ----------

    """

    def __init__(self, r_seed=0, conv_size=1, n_conv=3, num_iter=1, feature_filter_func=None,
                 feature_filter_func_param=None, dataset_df=None, dataset_mms_df=None, X_fuzzy_dms=None):
        self.r_seed = r_seed
        self.conv_size = conv_size
        self.n_conv = n_conv
        self.num_iter = num_iter
        self.feature_filter_func = feature_filter_func
        self.feature_filter_func_param = feature_filter_func_param
        self.dataset_df = dataset_df
        self.dataset_mms_df = dataset_mms_df


class MultiProcessOptions:
    """
    A protocol message class that encapsulates all the options (excluding
    functions) of the multi-process settings.

    Parameters
    ----------
    n_cpu_cores_req : int, default=None
        The number of CPU cores to request. If left to None this is
        automatically set to the number of all CPU cores available.

    allow_growth : bool, default=False
        Whether to dynamically request more CPU resources.
    """

    def __init__(self, n_cpu_cores_req=None, allow_growth=False):
        self.n_cpu_cores_req = n_cpu_cores_req
        self.allow_growth = allow_growth


# =============================================================================
# Decision tree component
# =============================================================================


class Node:
    """
    A Class that encapsulates the data of the node (including root node) and
    leaf node in a decision tree.

    Parameters
    ----------
    split_rule : SplitRule, default=None
        The split rule represented by the feature selected as a node, and
        branching decisions are made based on this rule.

    leaf_value : float, default=None
        The predicted value indicated at a leaf node. In the classification
        tree it is the predicted class, and in the regression tree it is the
        predicted value.
        NB: Only a leaf node has this attribute value.

    leaf_proba : float, default=None
        The predicted probability indicated at a leaf node. Only works in the
        classification tree.
        NB: Only a leaf node has this attribute value.

    branch_true : Node, default=None
        The next node in the decision path when the feature value of a sample
        meets the split rule split_rule.

    branch_false : Node, default=None
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
    feature_idx : int, default=None
        The index of the feature selected as the node representing a split rule.

    split_value : float, default=None
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
    subset_true_X : array-like of shape (n_samples, n_features)
        The subset of feature values of the samples that meet the split_rule
        after splitting.

    subset_true_y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The subset of target values of the samples that meet the split_rule
        after splitting.

    subset_false_X : array-like of shape (n_samples, n_features)
        The subset of feature values of the samples that do not meet the
        split_rule after splitting.

    subset_false_y : array-like of shape (n_samples,) or (n_samples, n_outputs)
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

    Warnings
    --------
    This interface should not be used directly.
    Use derived algorithm classes instead.

    Attention
    ---------
    The purpose of this interface is to establish protocols
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
    def predict_proba(self, X):
        pass

    @abstractmethod
    def print_tree(self, tree=None, indent="  ", delimiter="=>"):
        pass


# =============================================================================
# Base fuzzy decision tree
# =============================================================================

class BaseFuzzyDecisionTree(metaclass=ABCMeta):
    """
    Base fuzzy decision tree class that encapsulates all base functions to be
    inherited by all derived classes (and attributes, if required).

    Warnings
    --------
    This interface should not be used directly.
    Use derived algorithm classes instead.

    Attention
    ---------
    See FuzzyDecisionTreeWrapper for descriptions of all parameters
    and attributes in this class.
    """

    # The parameters in this constructor don't need to have default values.
    def __init__(self, disable_fuzzy, X_fuzzy_dms, fuzzification_options, criterion_func, max_depth, min_samples_split,
                 min_impurity_split,
                 **kwargs):
        self.disable_fuzzy = disable_fuzzy
        self.X_fuzzy_dms = X_fuzzy_dms
        self.fuzzification_options = fuzzification_options
        self.criterion_func = criterion_func
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split

        self.root = None
        self._split_ds_func = None
        self._impurity_gain_calc_func = None
        self._leaf_value_calc_func = None
        self._is_one_dim = None
        self._best_split_rule = None  # To be deprecated in version 1.0.
        self._best_binary_subtrees = None  # To be deprecated in version 1.0.
        self._best_impurity_gain = 0  # To be deprecated in version 1.0.
        self._fuzzy_sets = None
        self.loss_func = None

    def fit(self, X_train, y_train):
        # Store whether y is a multi-dimension set, which means being one-hot encoded.
        self._is_one_dim = len(np.shape(y_train)) == 1

        # # Do feature fuzzification.
        # if not self.disable_fuzzy:

        self.root = self._build_tree(X_train, y_train)

    def predict(self, X):
        # # Do feature fuzzification.
        # if not self.disable_fuzzy:

        y_pred = []
        for x in X:
            y_pred.append(self._predict_one(x))
        return y_pred

    def predict_proba(self, X):
        # # Do feature fuzzification.
        # if not self.disable_fuzzy:

        y_pred_prob = []
        for x in X:
            y_pred_prob.append(self._predict_proba_one(x))
        return y_pred_prob

    def print_tree(self, tree=None, indent="  ", delimiter="=>"):
        if tree is None:
            tree = self.root

        if tree.leaf_value is not None:
            print(tree.leaf_value)
        else:
            # Recursively print sub-nodes.
            # Print the split rule first.
            print("%s:%s? " % (tree.split_rule.feature_idx, tree.split_rule.split_value))

            # Print the sub-node that meets the split rule.
            print("%sTrue%s" % (indent, delimiter), end="")
            self.print_tree(tree.branch_true, indent + indent)

            # Print the other sub-node that do not meet the split rule.
            print("%sFalse%s" % (indent, delimiter), end="")
            self.print_tree(tree.branch_false, indent + indent)

    def _build_tree(self, X, y, current_depth=0):
        """
        Recursively builds a decision tree.

        Attention
        ---------
        Only decision tree components are generated, either
        nodes (including root nodes) or leaf nodes.
        """
        best_split_rule = None
        best_binary_subtrees = None
        best_impurity_gain = 0
        n_samples, _ = np.shape(X)

        # If the current data set meets the split criteria min_samples_split and max_depth,
        # split the data set to prepare all information for a best node.
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # Get the best feature and the best split value based on it
            best_split_rule, best_binary_subtrees, best_impurity_gain = self._get_best_split(X, y)

        # If the best subtrees split above meet the split criterion min_impurity_split,
        # continue growing subtrees and then generate a node.
        if best_impurity_gain > self.min_impurity_split:
            subset_true_X = best_binary_subtrees.subset_true_X
            subset_true_y = best_binary_subtrees.subset_true_y
            branch_true = self._build_tree(subset_true_X, subset_true_y, current_depth + 1)

            subset_false_X = best_binary_subtrees.subset_false_X
            subset_false_y = best_binary_subtrees.subset_false_y
            branch_false = self._build_tree(subset_false_X, subset_false_y, current_depth + 1)

            best_node = Node(split_rule=best_split_rule, branch_true=branch_true, branch_false=branch_false)
            return best_node

        # If none of the above criteria is met, then the current data set can only be a leaf node.
        # Then generate a leaf node.
        leaf_value = self._leaf_value_calc_func(y)
        leaf_proba = calculate_proba(y)
        leaf_node = Node(leaf_value=leaf_value, leaf_proba=leaf_proba)
        return leaf_node

    def _get_best_split(self, X, y):
        """
        Iterate over all feature and calculate the impurity_gain based on its unique
        values. Finally, choose the feature that gives y the maximum gain at
        impurity_gain as the best split.
        """
        best_split_rule = None
        best_binary_subtrees = None
        best_impurity_gain = 0

        # Join the elements in the X and Y by index.
        # Note that both X and y must have same number of dimensions. When after one-hot-encoding, y is already 2-D.
        if len(np.shape(y)) == 1:
            # Do ascending dimension on y, and keep the column arrangement.
            y = np.expand_dims(y, axis=1)
        # Concatenate X and y as last column of X
        ds_train = np.concatenate((X, y), axis=1)

        # Start iterating over all features to get the best split.
        n_samples, n_features = np.shape(X)

        # Calculate the number of iterations over features.
        n_loop = n_features  # For non-fuzzy trees, n_loop is exactly the number of features
        # Fuzzy case: After feature fuzzification, have additional n_conv times of original number of features.
        if not self.disable_fuzzy:
            # NB: If FCM selects n optimal fuzzy sets, the calculation here will be deprecated.
            n_loop = int(n_features / (self.fuzzification_options.n_conv + 1))

        for feature_idx in range(n_loop):
            # Get all unique values of the feature with feature_idx group by value classes.
            feature_values = np.expand_dims(X[:, feature_idx], axis=1)
            unique_values = np.unique(feature_values)

            # Fuzzy case: Calculate p_k based on the current feature's membership degrees instead of probability.
            # Step 1: Calculate the sum of all the membership degrees of the current feature.
            dm_sum = None
            start = None
            # stop = None
            if not self.disable_fuzzy:
                # Columns of the idx-th features's degrees of membership start from
                # "n_loop + feature_idx * self.fuzzification_options.n_conv", and end with
                # "n_loop + (feature_idx + 1) * self.fuzzification_options.n_conv".
                start = n_loop + feature_idx * self.fuzzification_options.n_conv
                # stop = n_loop + (feature_idx + 1) * self.fuzzification_options.n_conv
                dm_sum = np.sum(X[:, start:(start + self.fuzzification_options.n_conv)])

            # Calculate each impurity_gain at each iteration over all unique feature values.
            for unique_value in unique_values:
                subset_true, subset_false = self._split_ds_func(ds_train, feature_idx, unique_value)

                if len(subset_true) > 0 and len(subset_false) > 0:
                    # Select y.
                    # NB: Don't use [:, -1] because y might have been transformed with one-hot-encoding.
                    y_subset_true = subset_true[:, n_features:]
                    y_subset_false = subset_false[:, n_features:]

                    # Fuzzy case: Calculate p_k.
                    p_subset_true_dm = None
                    p_subset_false_dm = None
                    n_conv = None
                    if not self.disable_fuzzy and dm_sum is not None and dm_sum > 0.0:
                        logging.debug("**************** Current feature's membership degrees start from: %d", start)
                        # Step 2: Calculate the membership degrees of each subset of the current feature.
                        subset_true_dm = subset_true[:, start:(start + self.fuzzification_options.n_conv)]
                        subset_true_dm_sum = np.sum(subset_true_dm)
                        subset_false_dm = subset_false[:, start:(start + self.fuzzification_options.n_conv)]
                        subset_false_dm_sum = np.sum(subset_false_dm)
                        # Step 3: Calculate p_k.
                        p_subset_true_dm = subset_true_dm_sum / dm_sum
                        p_subset_false_dm = subset_false_dm_sum / dm_sum

                        if len(np.shape(y_subset_true)) == 1 or len(np.shape(y_subset_false)) == 1:
                            y_subset_true = np.expand_dims(y_subset_true, axis=1)
                            y_subset_false = np.expand_dims(y_subset_false, axis=1)
                        y_subset_true = np.concatenate((subset_true_dm, y_subset_true), axis=1)
                        y_subset_false = np.concatenate((subset_false_dm, y_subset_false), axis=1)
                        logging.debug("**************** Shape of each subset for calculating impurity gain: %s, %s",
                                      np.shape(y_subset_true), np.shape(y_subset_false))

                        n_conv = self.fuzzification_options.n_conv

                    impurity_gain = self._impurity_gain_calc_func(y, y_subset_true, y_subset_false,
                                                                  self.criterion_func,
                                                                  p_subset_true_dm=p_subset_true_dm,
                                                                  p_subset_false_dm=p_subset_false_dm,
                                                                  n_conv=n_conv)
                    if impurity_gain > best_impurity_gain:
                        best_impurity_gain = impurity_gain

                        best_split_rule = SplitRule(feature_idx=feature_idx, split_value=unique_value)

                        subset_true_X = subset_true[:, :n_features]
                        subset_true_y = subset_true[:, n_features:]
                        subset_false_X = subset_false[:, :n_features]
                        subset_false_y = subset_false[:, n_features:]
                        best_binary_subtrees = BinarySubtrees(subset_true_X=subset_true_X,
                                                              subset_true_y=subset_true_y,
                                                              subset_false_X=subset_false_X,
                                                              subset_false_y=subset_false_y)

        return best_split_rule, best_binary_subtrees, best_impurity_gain

    def _predict_one(self, x, tree=None):
        """
        Recursively (in a top-to-bottom approach) search the built
        decision tree and find the leaf that match the sample to be
        predicted, then use the leaf value as the predicted value
        for the sample.
        """
        if tree is None:
            tree = self.root

        if tree.leaf_value is not None:
            return tree.leaf_value

        feature_value = x[tree.split_rule.feature_idx]
        branch = tree.branch_false
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.split_rule.split_value:
                branch = tree.branch_true
        elif feature_value == tree.split_rule.split_value:
            branch = tree.branch_true

        return self._predict_one(x, branch)

    def _predict_proba_one(self, x, tree=None):
        """
        Recursively (in a top-to-bottom approach) search the built
        decision tree and find the leaf that match the sample to be
        predicted, then use the leaf probability as the predicted
        probability for the sample.
        """
        if tree is None:
            tree = self.root

        if tree.leaf_value is not None:
            return tree.leaf_proba

        feature_value = x[tree.split_rule.feature_idx]
        branch = tree.branch_false
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.split_rule.split_value:
                branch = tree.branch_true
        elif feature_value == tree.split_rule.split_value:
            branch = tree.branch_true

        return self._predict_proba_one(x, branch)


# =============================================================================
# Public wrapper class for different decision trees
# =============================================================================
class FuzzyDecisionTreeWrapper(DecisionTreeInterface):
    """
    Wrapper class for different decision trees.

    Attention
    ---------
    The role of this class is to unify the external calls of different
    decision tree classes and implement dependency injection for those
    decision tree classes.

    The arguments of the constructors for different decision trees should
    belong to a subset of the following parameters.

    Parameters
    ----------
    fdt_class : Class, default=None
        The fuzzy decision tree estimator specified.

    disable_fuzzy : bool, default=False
        Set whether the specified fuzzy decision tree uses the fuzzification.
        If disable_fuzzy=True, the specified fuzzy decision tree is equivalent
        to a naive decision tree.

    X_fuzzy_dms : array-like of shape (n_samples, n_features)
        Three-dimensional array, and each element of the first dimension of the
        array is a two-dimensional array of corresponding feature's fuzzy sets.
        Each two-dimensional array is of shape of (n_samples, n_fuzzy_sets), but
        has transformed membership degree of the feature values to corresponding
        fuzzy sets.

    fuzzification_options : FuzzificationOptions, default=None
        Protocol message class that encapsulates all the options of the
        fuzzification settings used by the specified fuzzy decision tree.

    criterion_func : {"gini", "entropy"} for a classifier, {"mse", "mae"} for a regressor
        The criterion function used by the function that calculates the impurity
        gain of the target values.

    max_depth : int, default=float("inf")
        The maximum depth of the tree.

    min_samples_split : int, default=2
        The minimum number of samples required to split a node. If a node has a
        sample number above this threshold, it will be split, otherwise it
        becomes a leaf node.

    min_impurity_split : float, default=1e-7
        The minimum impurity required to split a node. If a node's impurity is
        above this threshold, it will be split, otherwise it becomes a leaf node.

    Attributes
    ----------
    root : Node
        The root node of a decision tree.

    _impurity_gain_calculation_func : function
        The function to calculate the impurity gain of the target values.

    _leaf_value_calculation_func : function
        The function to calculate the predicted value if the current node is a
        leaf:
        In a classification tree, it gives the target value with the highest
        probability.
        In a regression tree, it gives the average of all the target values.

    _is_one_dim : bool
        The Boolean value that indicates whether the y is a multi-dimensional set,
        which means whether y is one-hot encoded.

    _best_split_rule : SplitRule
        The split rule including the index of the best feature to be used, and
        the best value in the best feature.

    _best_binary_subtrees : BinarySubtrees
        The binary subtrees including two subtrees under a node, and each subtree
        is a subset of the sample that has been split. It is one of attributes of
        the node (including root node) in a decision tree.

    _best_impurity_gain : float
        The best impurity gain calculated based on the current split subtrees
        during a tree building process.

    _fuzzy_sets : array-like of shape (n_features, n_coefficients)
        All the coefficients of the degree of membership sets based on the
        current estimator. They will be used to calculate the degree of membership
        of the features of new samples before predicting those samples. Therefore,
        their life cycle is consistent with that of the current estimator.
        They are generated in the feature fuzzification before training the
        current estimator.
        NB: To be used in version 1.0.
    """

    # All parameters in this constructor should have default values.
    def __init__(self, fdt_class=None, disable_fuzzy=False, X_fuzzy_dms=None, fuzzification_options=None,
                 criterion_func=None,
                 max_depth=float("inf"), min_samples_split=2, min_impurity_split=1e-7, **kwargs):
        # Construct a instance of the specified fuzzy decision tree.
        if fdt_class is not None:
            self.estimator = fdt_class(disable_fuzzy=disable_fuzzy, X_fuzzy_dms=X_fuzzy_dms,
                                       fuzzification_options=fuzzification_options, criterion_func=criterion_func,
                                       max_depth=max_depth, min_samples_split=min_samples_split,
                                       min_impurity_split=min_impurity_split, **kwargs)
        self.fdt_class = fdt_class
        self.disable_fuzzy = disable_fuzzy
        self.X_fuzzy_dms = X_fuzzy_dms
        self.fuzzification_options = fuzzification_options
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

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training instances.

        y_train : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values (class labels) as integers or strings.
        """
        # Start training to get a fitted estimator.
        try:
            self.estimator.fit(X_train, y_train)
        except Exception as e:
            logger.exception("Running exception, see details:")
            # logger.error("Running exception, see details:", exc_info=True)

    def predict(self, X):
        """
        Predict the target values of the input samples X.

        In classification, a predicted target value is the one with the
        largest number of samples of the same class in a leaf.

        In regression, the predicted target value is the mean of the target
        values in a leaf.

        Parameters
        -----------
        X : array-like of shape (n_samples, n_features)
            Input instances to be predicted.

        Returns
        -------
        pred_y : list of n_outputs such arrays if n_outputs > 1
            The target values of the input instances.
        """
        try:
            return self.estimator.predict(X)
        except Exception as e:
            logger.exception("Running exception, see details:")
            # logger.error("Running exception, see details:", exc_info=True)

    def predict_proba(self, X):
        """
        Predict the probabilities of the target values of the input samples X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input instances to be predicted.

        Returns
        -------
        pred_y : list of n_outputs such arrays if n_outputs > 1
            The probabilities of the target values of the input instances.
        """
        try:
            return self.estimator.predict_proba(X)
        except Exception as e:
            logger.exception("Running exception, see details:")
            # logger.error("Running exception, see details:", exc_info=True)

    def print_tree(self, tree=None, indent="  ", delimiter="-->"):
        """
        Recursively (in a top-to-bottom approach) print the built decision tree.

        Parameters
        ----------
        tree : Node
            The root node of a decision tree.

        indent : str
            The indentation symbol used when printing subtrees.

        delimiter : str
            The delimiter between split rules and results.
        """
        try:
            self.estimator.print_tree(tree=tree, indent=indent, delimiter=delimiter)
        except Exception as e:
            logger.exception("Running exception, see details:")
            # logger.error("Running exception, see details:", exc_info=True)

    # =============================================================================
    # Functions to search fuzzy parameters for FDTs and plot their evaluation
    # =============================================================================
    def search_fuzzy_params_4_clf(self, ds_name_list, n_conv_lim, fuzzy_reg_lim):
        """
        Search fuzzy parameters for evaluating and choosing through fitting
        a number of groups of FDT classifiers from specified datasets in
        parallel (multi-process/master-worker mode).

        The fuzzy feature extraction before pretraining is based on specified
        fuzzy regulation coefficients and a number of fuzzy clusters that each
        feature belongs to.

        Attention
        ---------
        Use this function to prepare evaluation and plotting data when
        you need to evaluate the effect of different degrees of fuzzification
        on model training in advance.

        Parameters
        ----------
        ds_name_list : array-like

        fuzzy_reg_lim : tuple, (start, stop, step)

        n_conv_lim : tuple, (start, stop, step)
        """
        # Create a connection used to communicate between master process and its sub-processes.
        q = multiprocessing.Manager().Queue()

        # Create a pool for master process to manage its sub-processes in parallel.
        pool = multiprocessing.Pool(processes=NUM_CPU_CORES_REQ)

        # Pretrain different groups of classifiers and get each group's evaluation scores in parallel.
        for ds_name in ds_name_list:
            # Iteratively searching an optimum number of fuzzy clusters and
            # fuzzy regulation coefficient by a specified stride.
            for n_conv in range(n_conv_lim[0], n_conv_lim[1] + 1, n_conv_lim[2]):
                fuzzy_reg = fuzzy_reg_lim[0]
                while fuzzy_reg <= fuzzy_reg_lim[1]:
                    # Start a sub-process to fit a group of classifiers on a specified dataset and
                    # get the mean of their evaluation scores.
                    pool.apply_async(self._get_one_mean_fuzzy_clf, args=(q, ds_name, n_conv, fuzzy_reg,))
                    fuzzy_reg = float(Decimal(str(fuzzy_reg)) + Decimal(str(fuzzy_reg_lim[2])))

        pool.close()
        pool.join()

        # Encapsulate and save all data received from the sub-processes.
        self._encapsulate_save_data_fuzzy_clf(q=q)

    def _get_one_mean_fuzzy_clf(self, q, ds_name, n_conv, fuzzy_reg):
        """
        Fit a group of fuzzy classifiers on a specified dataset and get the
        mean of their evaluation scores.

        The fuzzy feature extraction before pretraining is based on specified
        fuzzy regulation coefficients and numbers of fuzzy clusters that each
        feature belongs to.

        Parameters
        ----------
        q : multiprocessing.queue.Queue

        ds_name : str

        n_conv : int

        fuzzy_reg : float

        Returns
        -------

        """
        curr_pid = os.getpid()
        logging.info("    |-- (%s Child-process) Pretrain a group of classifiers on: %s.", curr_pid, ds_name)
        logging.info("    |-- (%s Child-process) Preprocess fuzzy feature extraction based on parameters: %d, %f.",
                     curr_pid, n_conv, fuzzy_reg)

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
            X_dms = extract_fuzzy_features(X=X_fuzzy_pre, n_conv=n_conv, fuzzy_reg=fuzzy_reg)
            X_plus_dms = np.concatenate((X, X_dms), axis=1)
            # logging.debug("************* Shape before fuzzification: %s", np.shape(X))
            # logging.debug("************* Shape after fuzzification: %s", np.shape(X_plus_dms))

        # Fit a group of models, and then get the mean of their accuracy results.
        acc_train_list = []
        acc_test_list = []
        for i in range(NUM_GRP_MDLS):
            logging.info("        |-- (%s Child-process) %d-th fitting.", curr_pid, i)

            # Split training and test sets by hold-out partition method.
            # X_train, X_test, y_train, y_test = train_test_split(X_fuzzy_pre, y, test_size=0.4)
            kf = KFold(n_splits=2, random_state=i, shuffle=True)
            for train_index, test_index in kf.split(X):
                y_train, y_test = y[train_index], y[test_index]

                # Fit a model, and then get its evaluation scores.
                X_train, X_test = X_plus_dms[train_index], X_plus_dms[test_index]
                accuracy_train, accuracy_test = self._fit_one_fuzzy_clf(X_train=X_train, X_test=X_test,
                                                                        y_train=y_train, y_test=y_test,
                                                                        ds_name=ds_name, n_conv=n_conv,
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
        logging.info("    |-- ========================================================================================")
        logging.info("    |-- (%s Child-process) Pretrain a group of classifiers on: %s.", curr_pid, ds_name)
        logging.info("    |-- Mean train acc: %f; std: %f", acc_train_mean, std_train)
        logging.info("    |-- Mean test acc: %f; std: %f", acc_test_mean, std_test)
        logging.info("    |-- ========================================================================================")

        # Put the data in the connection between the master process and its sub-processes.
        # !!! NB: The data should be a 2-dimensional ndarray, or a dictionary with key,
        # which is the dataset name, and value, which is a 2-d matrix ndarray.
        if not q.full():
            q.put([[ds_name, n_conv, fuzzy_reg, err_train_mean, std_train, err_test_mean, std_test]])

    def _fit_one_fuzzy_clf(self, X_train, X_test, y_train, y_test, ds_name, n_conv, fuzzy_reg, sn):
        """
        Fit a fuzzy classifier and get its evaluation scores.

        See more about evaluation scores on https://scikit-learn.org/stable/modules/model_evaluation.html

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)

        X_test : array-like of shape (n_samples, n_features)

        y_train : array-like of shape (n_samples,)

        y_test : array-like of shape (n_samples,)

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
        # logging.info("    Fuzzy accuracy train: %f", accuracy_train)
        # logging.info("    Fuzzy accuracy test: %f", accuracy_test)

        # Pickle the fitted model.
        if self.enable_pkl_mdl:
            filename = DirSave.MODELS.value + get_today_str() + "_" + "clf_" + str(n_conv) + "_" + str(
                fuzzy_reg) + "_" + ds_name + "_" + str(sn) + ".mdl"
            joblib.dump(value=self.estimator, filename=filename)
            # trained_clf = joblib.load(filename=filename)

        # # Display the elapsed time.
        # logging.info("        |-- (%s Child-process) Time elapsed fitting one model: %f(s)", time.time() - time_start)

        return accuracy_train, accuracy_test

    def _encapsulate_save_data_fuzzy_clf(self, q):
        """
        Encapsulate and save all data received from the sub-processes when
        pretraining a group of fuzzy classifiers.

        Save the data in memory for immediate plotting, and a copy of the
        data in a file for future plotting against historical data.

        Parameters
        ----------
        q : multiprocessing.queue.Queue

        Returns
        -------

        """
        # Get data via connection between master process and its sub-processes.
        while not q.empty():
            # q.put([[ds_name, n_conv, fuzzy_reg, err_train_mean, std_train, err_test_mean, std_test]])
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
            column_names = ["ds_name", "n_conv", "fuzzy_reg", "err_train_mean", "std_train", "err_test_mean",
                            "std_test"]
            self.df_pretrain = pd.DataFrame(data=self.ds_pretrain, columns=column_names)
            filename = DirSave.EVAL_DATA.value + get_today_str() + "_" + EvaluationType.FUZZY_REG_VS_ERR_ON_N_CONV.value + ".csv"
            self.df_pretrain.to_csv(filename)
        logging.info("Main Process %s saved data as the shape: %s", os.getpid(), self.df_pretrain)

    def plot_fuzzy_reg_vs_err(self, filename=None):
        """
        Plot fuzzy regulation coefficient versus training error and
        test error on each numbers of fuzzy clusters respectively.

        Illustrate how the performance on unseen data (test data)
        is different from the performance on training data.

        Parameters
        ----------
        filename : str, default None
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
        # q.put([[ds_name, n_conv, fuzzy_reg, err_train_mean, std_train, err_test_mean, std_test]])
        ds_names = self.df_pretrain["ds_name"].unique()
        for ds_name in ds_names:
            df_4_ds_name = self.df_pretrain[self.df_pretrain["ds_name"] == ds_name]
            n_convs = df_4_ds_name["n_conv"].unique()
            # n_convs = sorted(n_convs)  # It doesn't matter if it's drawn in ascending order from n_conv.
            for n_conv in n_convs:
                df_4_n_conv = df_4_ds_name[df_4_ds_name["n_conv"] == n_conv]
                df_4_n_conv = df_4_n_conv.sort_values(by="fuzzy_reg", ascending=True)  # ascending is True by default.
                coordinates = df_4_n_conv[["fuzzy_reg", "err_train_mean", "err_test_mean"]].astype("float").values
                # logging.debug("+++++++++++++++++++++++++++++++++++++++++++++ %s", type(df_4_n_conv["err_train_mean"].values[1]))
                # x_lower_limit, x_upper_limit = np.min(df_4_n_conv[["fuzzy_reg"]].values), np.max(df_4_n_conv[["fuzzy_reg"]].values)
                # y_lower_limit = np.min(df_4_n_conv[["err_train_mean"]].values) if np.min(df_4_n_conv[["err_train_mean"]].values) < np.min(df_4_n_conv[["err_test_mean"]].values) else np.min(df_4_n_conv[["err_test_mean"]].values)
                # y_upper_limit = np.max(df_4_n_conv[["err_train_mean"]].values) if np.max(df_4_n_conv[["err_train_mean"]].values) > np.max(df_4_n_conv[["err_test_mean"]].values) else np.max(df_4_n_conv[["err_test_mean"]].values)
                # logging.debug("x_limits and y_limits are: %f, %f, %f, %f", x_lower_limit, x_upper_limit, y_lower_limit, y_upper_limit)

                plot_multi_lines(coordinates=coordinates,
                                 title="Fuzzy Reg Coeff vs Error - n_conv {} - {}".format(n_conv, ds_name),
                                 x_label="Fuzzy Regulation Coefficient",
                                 y_label="Error Rate",
                                 legends=["Train", "Test"],
                                 fig_name=DirSave.EVAL_FIGURES.value + get_today_str() + "_" + EvaluationType.FUZZY_REG_VS_ERR_ON_N_CONV.value + "_" + str(
                                     n_conv) + "_" + ds_name + ".png")

    def _fit_one_fuzzy_regr(self, X_train, X_test, y_train, y_test):
        """
        Fit a fuzzy regressor and get its evaluation scores.

        See more about evaluation scores on https://scikit-learn.org/stable/modules/model_evaluation.html

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)

        X_test : array-like of shape (n_samples, n_features)

        y_train : array-like of shape (n_samples,)

        y_test : array-like of shape (n_samples,)

        Returns
        -------

        """
        pass
