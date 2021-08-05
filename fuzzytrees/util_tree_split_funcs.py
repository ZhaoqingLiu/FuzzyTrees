# _*_coding:utf-8_*_
"""
@author : Zhaoqing Liu
@email  : Zhaoqing.Liu-1@student.uts.edu.au
"""
import numpy as np


# =============================================================================
# Naive functions
# =============================================================================

def split_ds_2_bin(ds, col_idx, split_val):
    """
    Split a data set into two subsets by a specified value of a specified feature:
    If the specified feature is numerical data, split the data set into two
    subsets based on whether each value of the specified feature is greater than
    or equal to the split value.

    If the specified feature is categorical data, split the data set into two
    subsets based on whether each value of the specified feature is the same as
    the split value.

    Parameters
    ----------
    ds : array-like of shape (n_samples, n_feature)
        The current data set to be split.

    col_idx : int
        The index of the specified column on which the split based.

    split_val : int, float, or string
        The specified value of the column indexed as col_idx.

    Returns
    -------
    subset_true, subset_false : array-like
        Return a tuple of the two split subsets.
    """
    # Declare a lambda (args: expression), which is an anonymous function,
    # and will define the criteria for slicing the data set to be split.
    split_func = None
    if isinstance(split_val, int) or isinstance(split_val, float):
        split_func = lambda sample: sample[col_idx] >= split_val
    else:
        split_func = lambda sample: sample[col_idx] == split_val

    # Slice out all samples that meet the criteria defined by the lambda.
    subset_true = np.array([sample for sample in ds if split_func(sample)])
    subset_false = np.array([sample for sample in ds if not split_func(sample)])

    return subset_true, subset_false


def split_ds_2_multi(ds, col_idx, split_val):
    pass


def split_disc_ds_2_multi(ds, col_idx, split_val):
    pass



