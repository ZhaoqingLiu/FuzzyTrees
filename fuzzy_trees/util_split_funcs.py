# _*_coding:utf-8_*_
"""
@author:
@license: (C) Copyright 2020-,
@file: split_functions.py
@date: 11/12/2020 11:12 am
@IDE:
@desc:
"""
import numpy as np


# =============================================================================
# Naive functions
# =============================================================================

def split_dataset(dataset, feature_index, split_value):
    """
    Split a data set into two subsets by a specified value of a specified feature:
    - If the specified feature is numerical data, split the data set into two
      subsets based on whether each value of the specified feature is greater than
      or equal to the split value.
    - If the specified feature is categorical data, split the data set into two
      subsets based on whether each value of the specified feature is the same as
      the split value.

    Parameters
    ----------
    dataset: {array-like, sparse matrix} of shape (n_samples, n_feature)
        The current data set to be split.

    feature_index: int
        The index of the specified feature.

    split_value: int, float, or string
        The specified value of the feature indexed as feature_idx.

    Returns
    -------
    subset_true, subset_false: array-like of shape
        Return a tuple of the two split subsets.
    """
    # Declare a lambda (args: expression), which is an anonymous function,
    # and will define the criteria for slicing the data set to be split.
    split_func = None
    if isinstance(split_value, int) or isinstance(split_value, float):
        split_func = lambda sample: sample[feature_index] >= split_value
    else:
        split_func = lambda sample: sample[feature_index] == split_value

    # Slice out all samples that meet the criteria defined by the lambda.
    subset_true = np.array([sample for sample in dataset if split_func(sample)])
    subset_false = np.array([sample for sample in dataset if not split_func(sample)])

    return subset_true, subset_false




