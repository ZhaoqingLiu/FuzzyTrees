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
import traceback
import numpy as np
from abc import ABCMeta, abstractmethod
from decimal import Decimal

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from fuzzy_trees.settings import NUM_CPU_CORES_REQ, FUZZY_LIM, FUZZY_STRIDE
from fuzzy_trees.util_criterion_funcs import calculate_entropy, calculate_gini, calculate_variance, \
    calculate_standard_deviation
from fuzzy_trees.util_data_handler import load_dataset_clf
from fuzzy_trees.util_data_processing_funcs import extract_fuzzy_features


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
        self.ds_4_plotting = {}

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
    # Pre-Train and Evaluation Functions
    # =============================================================================
    def pre_train(self, ds_name_list, conv_k_lim, fuzzy_reg_lim):
        """
        Pretrain a set of FDT estimators on specified datasets according to
        different fuzzy regulation coefficients and different number of clusters
        of fuzzification on each feature.

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

        # Create a pool to manage multi-processes.
        pool = multiprocessing.Pool(processes=NUM_CPU_CORES_REQ)

        # Pretrain estimators in parallel.
        for ds_name in ds_name_list:
            for conv_k in range(conv_k_lim[0], conv_k_lim[1] + 1, conv_k_lim[2]):
                fuzzy_reg = fuzzy_reg_lim[0]
                while fuzzy_reg <= fuzzy_reg_lim[1]:
                    # Execute an estimator by a subprocess.
                    pool.apply_async(self._pre_train_one, args=(q, ds_name, conv_k, fuzzy_reg, ))
                    fuzzy_reg = float(Decimal(str(fuzzy_reg)) + Decimal(str(fuzzy_reg_lim[2])))

        pool.close()
        pool.join()

        # Encapsulate and save each process's result.
        # encapsulate_save_result(q)
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++", self.ds_4_plotting)

    def plot_fuzzy_reg_vs_err(self):
        """
        Plot fuzzy regulation coefficient versus training error and test error.
        Returns
        -------

        """
        if not self.ds_4_plotting:
            # TODO: Plot based on data in memory.
            pass
        else:
            # TODO: Plot based on data in file.
            pass

    def _pre_train_one(self, ds_name, conv_k, fuzzy_reg):
        # Load data set.
        ds_df = load_dataset_clf(ds_name)

        # Run the experiment according to the parameters.
        X = ds_df.iloc[:, :-1].values
        y = ds_df.iloc[:, -1].values
        accuracy_train_list, accuracy_test_list = self._exe_pretrain(X, y, ds_name=ds_name, conv_k=conv_k, fuzzy_reg=fuzzy_reg)

        # Process the result.
        accuracy_train_mean = np.mean(accuracy_train_list)
        accuracy_test_mean = np.mean(accuracy_test_list)

        # Put the result in the connection between the main process and the child processes (in master-worker mode).
        # The 2nd return value in send() should be a 2-dimensional ndarray
        error_train_mean = 1 - accuracy_train_mean
        error_test_mean = 1 - accuracy_test_mean
        # !!! NB: The value in the dictionary to be returned must be a 2-d matrix.
        return {ds_name: np.asarray([[fuzzy_reg, error_train_mean, fuzzy_reg, error_test_mean]])}

    def _exe_pretrain(self, X, y, ds_name, conv_k, fuzzy_reg):
        accuracy_train_list = []
        accuracy_test_list = []

        # Step 1: Preprocess features for using fuzzy decision tree.
        X_fuzzy_pre = X.copy()
        # - Step 1: Standardise feature scaling.
        # X_fuzzy_pre[:, :] -= X_fuzzy_pre[:, :].min()
        # X_fuzzy_pre[:, :] /= X_fuzzy_pre[:, :].max()
        # - Step 2: Extract fuzzy features.
        X_dms = extract_fuzzy_features(X_fuzzy_pre, conv_k=conv_k, fuzzy_reg=fuzzy_reg)
        X_plus_dms = np.concatenate((X, X_dms), axis=1)
        # print("************* Before, original shape:", np.shape(X))
        # print("************* After, fuzzy shape:", np.shape(X_plus_dms))

        # Step 2: Get training and testing result by a model.
        pool = multiprocessing.Pool(processes=NUM_CPU_CORES_REQ)
        for i in range(10):
            print("%ith comparison on - %s" % (i, ds_name))

            # Split training and test sets by hold-out partition method.
            # X_train, X_test, y_train, y_test = train_test_split(X_fuzzy_pre, y, test_size=0.4)

            kf = KFold(n_splits=2, random_state=i, shuffle=True)
            for train_index, test_index in kf.split(X):
                y_train, y_test = y[train_index], y[test_index]

                # Using a fuzzy decision tree. =========================================================================
                X_train, X_test = X_plus_dms[train_index], X_plus_dms[test_index]
                # accuracy_train, accuracy_test = self._fit_test(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
                # accuracy_train_list.append(accuracy_train)
                # accuracy_test_list.append(accuracy_test)
                pool.apply_async(self._fit_test, args=(X_train, X_test, y_train, y_test,))

        pool.close()
        pool.join()

        # print("========================================================================================")
        # print("Pretraining on - ", ds_name)
        # print("Mean training accuracy:", np.mean(accuracy_train_list), "  std:", np.std(accuracy_train_list))
        # print("Mean test accuracy:", np.mean(accuracy_test_list), "  std:", np.std(accuracy_test_list))
        # print("========================================================================================")

        return accuracy_train_list, accuracy_test_list

    def _fit_test(self, X_train, X_test, y_train, y_test):
        # time_start = time.time()  # Record the start time.

        # Fit the initialised model.
        self.estimator.fit(X_train, y_train)
        # clf.print_tree()

        # Get the training accuracy and test accuracy of the fitted (trained) estimator.
        y_pred_train = self.predict(X_train)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        y_pred_test = self.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)

        # print("    Fuzzy accuracy train:", accuracy_train)
        # print("    Fuzzy accuracy test:", accuracy_test)

        # # Display the time of training a single model.
        # print('    Elapsed time of a single model (FDT-based):', time.time() - time_start, 's')

        return accuracy_train, accuracy_test


if __name__ == '__main__':
    pass
