# _*_coding:utf-8_*_
"""
@author:
@license: (C) Copyright 2020-,
@file: criterion_functions.py
@date: 10/12/2020 8:27 pm
@IDE:
@desc:
"""
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
            # print("****####  ", dm.shape)
            # print("****####  ", dm[np.where(y == label)[0], :].shape)
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
            # print("****####  ", dm.shape)
            # print("****####  ", dm[np.where(y == label)[0], :].shape)
            sum_sub_dm = np.sum(dm[np.where(y == label)[0], :])
            p = sum_sub_dm / np.sum(dm)
            gini += p * (1 - p)
        else:
            count = len(y[y == label])
            p = count / len(y)
            gini += p * (1 - p)

    return gini


def calculate_impurity_gain(y, sub_y_1, sub_y_2, criterion_func, p_subset_true_dm=None, p_subset_false_dm=None):
    """
    Calculate the impurity gain, which is equal to the
    impurity of y minus the entropy of sub_y_1 and sub_y_2.
    """
    impurity = criterion_func(y)

    if p_subset_true_dm is not None and p_subset_false_dm is not None:
        information_gain = impurity - (p_subset_true_dm * criterion_func(sub_y_1[:, -1], sub_y_1[:, :-1])) - (p_subset_false_dm * criterion_func(sub_y_2[:, -1], sub_y_2[:, :-1]))
    else:
        p_1 = len(sub_y_1) / len(y)
        p_2 = len(sub_y_2) / len(y)
        information_gain = impurity - (p_1 * criterion_func(sub_y_1)) - (p_2 * criterion_func(sub_y_2))

    return information_gain


def calculate_impurity_gain_ratio(y, sub_y_1, sub_y_2, X_sub, criterion_func, p_subset_true_dm=None, p_subset_false_dm=None):
    """
    Calculate the impurity gain ratio.
    """
    information_gain = calculate_impurity_gain(y=y, sub_y_1=sub_y_1, sub_y_2=sub_y_2, criterion_func=criterion_func, p_subset_true_dm=p_subset_true_dm, p_subset_false_dm=p_subset_false_dm)
    intrinsic_value = criterion_func(X_sub)
    information_gain_ratio = information_gain / intrinsic_value

    return information_gain_ratio


def calculate_value_by_majority_vote(y):
    """
    Calculate leaf value by majority vote.

    NB: Used in classification decision tree.
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
    mean = np.ones(np.shape(y)) * y.mean(0)
    n_samples = np.shape(y)[0]
    variance = (1 / n_samples) * np.diag((y - mean).T.dot(y - mean))  # T means transposing a matrix.

    return variance


def calculate_standard_deviation(y):
    """
    Calculate the standard deviation of y.
    """
    std_dev = np.sqrt(calculate_variance(y))

    return std_dev


def calculate_variance_reduction(y, sub_y_1, sub_y_2, criterion_func, p_subset_true_dm=None, p_subset_false_dm=None):
    """
    Calculate the variance reduction, which is equal to the
    impurity of y minus the entropy of sub_y_1 and sub_y_2.
    """
    var = criterion_func(y)
    var_1 = criterion_func(np.expand_dims(sub_y_1[:, -1], axis=1))
    var_2 = criterion_func(np.expand_dims(sub_y_2[:, -1], axis=1))

    if p_subset_true_dm is not None and p_subset_false_dm is not None:
        p_1 = p_subset_true_dm
        p_2 = p_subset_false_dm
    else:
        p_1 = len(sub_y_1) / len(y)
        p_2 = len(sub_y_2) / len(y)

    # Calculate the variance reduction
    variance_reduction = var - (p_1 * var_1 + p_2 * var_2)

    return sum(variance_reduction)


def calculate_mean(y):
    """
    Calculate the mean of y.
    """
    value = np.mean(y, axis=0)

    return value if len(value) > 1 else value[0]


# =============================================================================
# Statistical functions
# =============================================================================

def calculate_proba(y):
    """
    Calculate the probabilities of each element in the set.

    NB: Before counting, the elements will be reordered from smallest to largest.

    Parameters
    ----------
    y: array-like of shape (n_samples,)
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
    #     print(count / np.shape(y)[0])

    return prob_list


# =============================================================================
# Loss functions
# =============================================================================

class LossFunction(metaclass=ABCMeta):
    """
    Base loss function class that encapsulates all
    base functions to be inherited by all derived
    function classes.

    Warning: This class should not be used directly.
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
