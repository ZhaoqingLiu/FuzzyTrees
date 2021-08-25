# _*_coding:utf-8_*_
"""
@author : Zhaoqing Liu
@email  : Zhaoqing.Liu-1@student.uts.edu.au
"""
import logging
import math
from abc import ABCMeta, abstractmethod

import numpy as np


# =============================================================================
# Functions for Classification
# =============================================================================

# # For non-fuzzy decision trees
# def calculate_entropy(y):
#     """
#     Calculate the entropy of y.
#     """
#     entropy = 0
#
#     log2 = lambda x: math.log(x) / math.log(2)
#
#     unique_labels = np.unique(y)
#     for label in unique_labels:
#         count = len(y[y == label])
#         p = count / len(y)
#         entropy += -p * log2(p)
#
#     return entropy
#
#
# # For non-fuzzy decision trees
# def calculate_gini_index(y):
#     """
#     Calculate the Gini index of y.
#     """
#     diffsum = 0
#     for i, yi in enumerate(y[:-1], 1):
#         diffsum += np.sum(np.abs(yi - y[i:]))
#     return diffsum / (len(y) ** 2 * np.mean(y))
#
#
# # For non-fuzzy decision trees
# def calculate_gini(y):
#     """
#     Calculate the Gini impurity of y.
#     """
#     # Implementation based on the 1st Formula:
#     # diff = 0
#     # unique_labels = np.unique(y)
#     # for label in unique_labels:
#     #     count = len(y[y == label])
#     #     p = count / len(y)
#     #     diff += p * p
#     #
#     # return 1 - diff
#
#     # Implementation based on the 2nd Formula:
#     gini = 0
#     unique_labels = np.unique(y)
#     for label in unique_labels:
#         count = len(y[y == label])
#         p = count / len(y)
#         gini += p * (1 - p)
#
#     return gini


# For fuzzy decision trees
def calculate_entropy(y, dm=None):
    """
    Calculate the entropy of y.
    """
    entropy = 0

    log2 = lambda x: math.log(x) / math.log(2)

    unique_labels = np.unique(y)
    for label in unique_labels:
        if dm is not None:
            sum_sub_dm = np.sum(dm[np.where(y == label)[0], :])
            p = sum_sub_dm / np.sum(dm)
            entropy += -p * log2(p)
        else:
            count = len(y[y == label])
            p = count / len(y)
            entropy += -p * log2(p)

    return entropy


# For fuzzy decision trees
def calculate_gini(y, dm=None):
    """
    Calculate the Gini impurity of y.
    """
    # Implementation based on the 1st Formula:
    # diff = 0
    # unique_labels = np.unique(y)
    # for label in unique_labels:
    #     count = len(y[y == label])
    #     p = count / len(y)
    #     diff += p * p
    #
    # return 1 - diff

    # Implementation based on the 2nd Formula:
    gini = 0
    unique_labels = np.unique(y)
    for label in unique_labels:
        if dm is not None:
            sum_sub_dm = np.sum(dm[np.where(y == label)[0], :])
            p = sum_sub_dm / np.sum(dm)
            gini += p * (1 - p)
        else:
            count = len(y[y == label])
            p = count / len(y)
            gini += p * (1 - p)

    return gini


def calculate_impurity_gain(y, sub_y_1, sub_y_2, criterion_func,
                            p_subset_true_dm=None, p_subset_false_dm=None, n_conv=None):
    """
    Calculate the impurity gain, which is equal to the
    impurity of y minus the entropy of sub_y_1 and sub_y_2.
    """
    impurity = criterion_func(y)

    if p_subset_true_dm is not None and p_subset_false_dm is not None and n_conv is not None:
        impurity_sub_1 = p_subset_true_dm * criterion_func(sub_y_1[:, n_conv:], sub_y_1[:, :n_conv])
        impurity_sub_2 = p_subset_false_dm * criterion_func(sub_y_2[:, n_conv:], sub_y_2[:, :n_conv])
        information_gain = impurity - impurity_sub_1 - impurity_sub_2
    else:
        p_1 = len(sub_y_1) / len(y)
        p_2 = len(sub_y_2) / len(y)
        information_gain = impurity - (p_1 * criterion_func(sub_y_1)) - (p_2 * criterion_func(sub_y_2))

    return information_gain


def calculate_impurity_gain_ratio(y, sub_y_1, sub_y_2, X_sub, criterion_func,
                                  p_subset_true_dm=None, p_subset_false_dm=None, n_conv=None):
    """
    Calculate the impurity gain ratio.
    """
    information_gain = calculate_impurity_gain(y=y, sub_y_1=sub_y_1, sub_y_2=sub_y_2, criterion_func=criterion_func,
                                               p_subset_true_dm=p_subset_true_dm,
                                               p_subset_false_dm=p_subset_false_dm,
                                               n_conv=n_conv)
    intrinsic_value = criterion_func(X_sub)
    information_gain_ratio = information_gain / intrinsic_value

    return information_gain_ratio


def calculate_value_by_majority_vote(y):
    """
    Calculate value by majority vote.

    Attention
    ---------
    Used in classification decision tree.
    """
    majority_value = None

    max_count = 0
    unique_labels = np.unique(y)
    for label in unique_labels:
        count = len(y[y == label])
        if count > max_count:
            majority_value = label
            max_count = count

    return majority_value


# =============================================================================
# Functions for Regression
# =============================================================================

def calculate_mse(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred.
    """
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse


def calculate_mae(y_true, y_pred):
    """
    Calculate the Mean Absolute Error between y_true and y_pred.
    """
    mae = np.mean(abs(y_true - y_pred))
    return mae


def calculate_variance(y):
    """
    Calculate the variance of y.
    """
    mean = np.ones(np.shape(y)) * np.mean(y, axis=0)
    n_samples = np.shape(y)[0]
    variance = (1 / n_samples) * np.diag((y - mean).T.dot(y - mean))  # T means transposing a matrix.

    return variance


def calculate_standard_deviation(y):
    """
    Calculate the standard deviation of y.
    """
    std_dev = np.sqrt(calculate_variance(y))

    return std_dev


def calculate_variance_reduction(y, y_sub_1, y_sub_2, criterion_func,
                                 p_subset_true_dm=None, p_subset_false_dm=None, n_conv=None):
    """
    Calculate the variance reduction, which is equal to the
    impurity of y minus the entropy of sub_y_1 and sub_y_2.
    """
    var = criterion_func(y)

    logging.debug("**************** (Shape before) y: %s; y_sub_1: %s; y_sub_2: %s",
                  np.shape(y), np.shape(y_sub_1), np.shape(y_sub_2))
    if p_subset_true_dm is not None and p_subset_false_dm is not None and n_conv is not None:
        # Select y.
        # NB: Don't use [:, -1] because y might have been transformed with one-hot-encoding.
        y_sub_1 = y_sub_1[:, n_conv:]
        y_sub_2 = y_sub_2[:, n_conv:]
    logging.debug("**************** (Shape after) y: %s; y_sub_1: %s; y_sub_2: %s",
                  np.shape(y), np.shape(y_sub_1), np.shape(y_sub_2))

    var_1 = criterion_func(y_sub_1)
    var_2 = criterion_func(y_sub_2)

    if p_subset_true_dm is not None and p_subset_false_dm is not None and n_conv is not None:
        p_1 = p_subset_true_dm
        p_2 = p_subset_false_dm
    else:
        p_1 = len(y_sub_1) / len(y)
        p_2 = len(y_sub_2) / len(y)

    # Calculate the variance reduction
    variance_reduction = var - (p_1 * var_1 + p_2 * var_2)

    return sum(variance_reduction)


def calculate_mean_value(y):
    """
    Calculate the mean of y.

    Parameters
    ----------
    y : array-like of shape (n_samples, n_labels)

    Returns
    -------
    value : array-like of the shape reduced by one dimension,
           at least a 0-d float number
        The mean values.
    """
    value = np.mean(y, axis=0)

    return value if len(value) > 1 else value[0]


# =============================================================================
# Statistical functions
# =============================================================================

def calculate_proba(y):
    """
    Calculate the probabilities of each element in the set.

    Attention
    ---------
    Before counting, the elements will be reordered from smallest to largest.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
    """
    prob_list = []

    label_values = np.unique(y)
    for label in label_values:
        prob_list.append(np.sum(y == label) / np.shape(y)[0])

    # If the number of dimensions of y is greater than 1,
    # the following method may cause a ValueError: object too deep for desired array.
    # if len(np.shape(y)) > 1:
    #     y = np.squeeze(y)
    # dist = np.bincount(y)
    # for count in dist:
    #     logging.debug(count / np.shape(y)[0])

    return prob_list


# =============================================================================
# Loss functions
# =============================================================================

class LossFunction(metaclass=ABCMeta):
    """
    Base loss function class that encapsulates all
    base functions to be inherited by all derived
    function classes.

    Warnings
    --------
    This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def loss(self, y, y_pred):
        pass

    @abstractmethod
    def gradient(self, y, y_pred):
        pass


class LeastSquaresFunction(LossFunction):
    """
    Function class used in a gradient boosting regressor
    (Friedman et al., 1998; Friedman 2001).
    """

    def loss(self, y, y_pred):
        """Lost function is a Least-square equation: L(y, F) = (y - F) ^ 2 / 2"""
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)


class SoftLeastSquaresFunction(LossFunction):
    """
    Function class used in a gradient boosting classifier
    (Friedman et al., 1998; Friedman 2001).
    """

    def loss(self, y, y_pred):
        """
        Lost function (Least-square equation: L(y, F) = (y - F) ^ 2 / 2)
        is not applicable in classification.
        """
        pass

    def gradient(self, y, proba):
        return y - proba


# =============================================================================
# Functions for Boosting Ensembles
# =============================================================================
def sigmoid(y_preds):
    """Sigmoid nonlinear transformation.

    Parameters
    ----------
    y_preds : array-like of one-hot-encoding array
        NB: The input array needs to be of integer dtype, otherwise a
        TypeError is raised.

    Returns
    -------
    array-like of shape (n_samples, )
    """
    return 1 / (1 + np.exp(-y_preds))


def softmax(y_preds):
    """Softmax nonlinear transformation.

    Parameters
    ----------
    y_preds : array-like of one-hot-encoding array
        NB: The input array needs to be of integer dtype, otherwise a
        TypeError is raised.

    Returns
    -------
    array-like of shape (n_samples, )
    """
    dummy = np.exp(y_preds)
    sums = np.sum(dummy, axis=1, keepdims=True)
    return dummy / sums


# =============================================================================
# Functions for Bagging Ensembles
# =============================================================================

def majority_vote(y_preds):
    """
    Get the the final classification result by majority voting method.

    Parameters
    ----------
    y_preds : array-like of shape (n_samples, n_estimators)
        NB: The input array needs to be of integer dtype, otherwise a
        TypeError is raised.

    Returns
    -------
    array-like of shape (n_samples, )
    """
    y_pred = []
    for y_p in y_preds:
        y_pred.append(np.bincount(y_p.astype("int")).argmax())
        # np.bincount(): Count number of occurrences of each value in array of non-negative ints.
        # np.argmax(): Return indices of the maximum values along the given axis.

    return y_pred


def mean_value(y_preds):
    """
    Get the final regression result by averaging method.

    Parameters
    ----------
    y_preds : array-like of shape (n_samples, n_estimators, n_labels)

    Returns
    -------
    y_pred : array-like of the shape (n_samples, n_labels) reduced by one dimension,
           at least array-like of shape (n_samples, )
    """
    y_pred = []
    for y_p in y_preds:
        y_pred.append(calculate_mean_value(y_p))

    return np.array(y_pred)
