# _*_coding:utf-8_*_
"""
@author: Zhaoqing Liu
@email: Zhaoqing.Liu-1@student.uts.edu.au
@date: 29/01/2021 4:41 am
@desc: 
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


# =============================================================================
# Fuzzy-related functions
# =============================================================================

def degree_of_membership_build(X_df, r_seed, conv_k, fuzzy_th):
    """
    Build the degree of membership set of a feature. That set maps to
    the specified number of fuzzy sets of the feature.
    This is the process of transforming a crisp set into a fuzzy set.

    TODO: To be deprecated in version 1.0.
        This function will be integrated into the FCM module,
        which extracts the fuzzy features of all samples in a
        data set before starting training.

    Parameters:
    -----------
    X_df: DataFrame
        One feature values of the training input samples.
        NB: All features must be normalized by feature scaling.

    r_seed: int
        The random seed.

    conv_k: DataFrame
        The number of convolution over the input sample.

    Returns
    -------
    x_new: {array-like, sparse matrix} of shape (n_samples, n_fuzzy_sets)
        Transformed degree of membership set.
    centriods:

    degree_of_membership_theta:

    """
    # TODO: categorical feature handling
    # TODO: missing value handling
    x_np = X_df.values
    x_np = x_np.reshape(-1, 1)

    # TODO: c-means, self-organize-map
    kmeans = KMeans(n_clusters=conv_k, random_state=r_seed).fit(x_np)
    x_new = kmeans.transform(x_np)
    centriods = kmeans.cluster_centers_

    # get degree of membership distances threshold theta, DOM_theta
    centriods_pair_dist = pairwise_distances(centriods)
    # to fill the diagonal
    centriods_pair_dist[centriods_pair_dist == 0] = 9999
    degree_of_membership_theta = centriods_pair_dist.min(axis=1)

    # convert distance to degree of membership
    for idx, item in enumerate(degree_of_membership_theta):
        x_new[:, idx] = 1 - x_new[:, idx] / item
    # x_new[x_new < 0] = 0

    # TODO: Searching an optimum fuzzy threshold by a loop according the specified stride.
    # np.where(x_new > fuzzy_th, x_new, 0.0)
    # np.where(x_new > fuzzy_th and x_new <= (1 - fuzzy_th), x_new, 1.0)
    x_new[x_new <= fuzzy_th] = 0
    x_new[x_new >= (1 - fuzzy_th)] = 1
    # print("++++++++++++++++++++++++++++++++++++++")
    # print(x_new)
    # print("++++++++++++++++++++++++++++++++++++++")

    return x_new, centriods, degree_of_membership_theta


def extract_fuzzy_features(X, conv_k=5, fuzzy_th=0.0):
    """
    Extract fuzzy features in feature fuzzification to generate degree of
    membership sets of each feature.

    NB: Feature fuzzification must be done in the data preprocessing, that is,
        before training the model and predicting new samples.

    TODO: To be deprecated in version 1.0.
    TODO: To be verified by experiment: When using cross validation, which performance is better doing this before or after the partition of the data sets?
    """
    # print("************* X's shape:", np.shape(X))
    n_samples, n_features = np.shape(X)
    X_fuzzy_dms = np.empty([n_samples, 0])
    for feature_idx in range(n_features):
        X_fuzzy_dm, _, _ = degree_of_membership_build(r_seed=0, X_df=pd.DataFrame(X[:, feature_idx]), conv_k=conv_k,fuzzy_th=fuzzy_th)
        X_fuzzy_dms = np.concatenate((X_fuzzy_dms, X_fuzzy_dm), axis=1)
    # print("************* X_fuzzy_dms's shape:", np.shape(X_fuzzy_dms))
    return X_fuzzy_dms

    # X_df = pd.DataFrame(X)
    # X_fuzzy_dms, _, _ = degree_of_membership_build(r_seed=0, X_df=X_df.iloc[:, 0], conv_k=5)

    # Another try:  ====================================
    # X_fuzzy_dms = []
    # _, n_features = np.shape(X)
    # for feature_idx in range(n_features):
    #     X_fuzzy_dm, _, _ = degree_of_membership_build(r_seed=0, X_df=pd.DataFrame(X[:, feature_idx]), conv_k=5)
    #     X_fuzzy_dms.append(X_fuzzy_dm)
    # print("************* X_fuzzy_dms's elements:", np.asarray(X_fuzzy_dms).shape)
    # return np.asarray(X_fuzzy_dms)


# =============================================================================
# Encoder
# =============================================================================

"""
Encoder class that encodes categorical variables.
Or, use the existing encoder, sklearn.preprocessing.
Feature encoding includes:
    - Ordinal Encoder (OE): Label Encoder, which is used for label coding;
      and Ordinal Encoder, which is used for n-dimensional feature vectors,
      namely n-column feature values.
    - One-Hot Encoder (OHE): encode numerical values.
    - Target Encoder (Mean Encoder): is used when the cardinality of a
      feature (e.g. the number of categories of this feature) is large.
    - CatBoost Encoder: is similar to the Target Encoder.
    - Cyclic Features: is used for uniformly cyclic features.
    - Others, e.g. Feature Hashing, LeaveOut Encoder, etc.
"""


def one_hot_encode(y, n_ohe_col=None):
    """
    One-hot encode nominal values of target values.

    Parameters
    ----------
    y: array-like, 1-dimensional, elements must be numerical value and start from 0
    n_ohe_col: int

    Returns
    -------

    """
    y = y.astype(int)

    if n_ohe_col is None:
        n_ohe_col = np.amax(y) + 1  # np.max is just an alias for np.amax.
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print(y)
        # print(ohe_col_num)
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    one_hot = np.zeros((y.shape[0], n_ohe_col))  # Can also use y.size instead of y.shape[0]
    one_hot[np.arange(y.shape[0]), y] = 1

    return one_hot


def to_nominal(x):
    """
    Transform values from one-hot encoding to nominal.
    """
    return np.argmax(x, axis=1)


def make_diagonal(x):
    """
    Transform a vector into an diagonal matrix.
    """
    m = np.zeros((len(x), len(x)))
    for i in range(len(m[0])):
        m[i, i] = x[i]

    return m
