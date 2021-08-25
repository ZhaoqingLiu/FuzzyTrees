# _*_coding:utf-8_*_
"""
@author : Zhaoqing Liu
@email  : Zhaoqing.Liu-1@student.uts.edu.au
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

"""
Functions to include in this module are for preprocessing data:
    1. Transform numerical variables to fuzzy degree of membership.
    2. Process missing values.
    3. Process outliers.
    4. Transform descriptive variables to numerical variables.
    5. Partitioning training sets and testing sets.
    6. Normalising data.
"""


# =============================================================================
# Fuzzy-related functions
# =============================================================================

def degree_of_membership_build(X_df, r_seed, n_conv, fuzzy_reg):
    """
    Build the degree of membership set of a feature. That set maps to
    the specified number of fuzzy sets of the feature.
    This is the process of transforming a crisp set into a fuzzy set.

    @author : Anjin Liu
    @email : Anjin.Liu@uts.edu.au

    Parameters
    ----------
    X_df : DataFrame
        One feature values of the training input samples.
        NB: All features must be normalized by feature scaling.

    r_seed : int
        The random seed.

    n_conv : int
        The number of convolution over the input sample.

    Returns
    -------
    x_new : array-like of shape (n_samples, n_fuzzy_sets)
        Transformed degree of membership set.

    centriods :

    degree_of_membership_theta :

    """
    x_np = X_df.values
    x_np = x_np.reshape(-1, 1)

    kmeans = KMeans(n_clusters=n_conv, random_state=r_seed).fit(x_np)
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
    x_new[x_new < 0] = 0
    # Followed by an alternative to the above three lines of code, which supports pruning `theta`.
    # theta_f = np.log(fuzzy_reg) - np.log(1 - fuzzy_reg)
    # for idx, item in enumerate(degree_of_membership_theta):
    #     if fuzzy_reg == 0 or fuzzy_reg == 0.0 or fuzzy_reg == 1 or fuzzy_reg == 1.0:
    #         x_new[:, idx] = 1 - x_new[:, idx] / item
    #     else:
    #         x_new[:, idx] = 1 - x_new[:, idx] / item * theta_f
    # x_new[x_new < 0] = 0

    return x_new, centriods, degree_of_membership_theta


def extract_fuzzy_features(X, n_conv=5, fuzzy_reg=0.0):
    """
    Extract fuzzy features in feature fuzzification to generate degree of
    membership sets of each feature.

    Attention
    ---------
    Feature fuzzification must be done in the data preprocessing, that is,
    before training the model and predicting new samples.

    @author: Anjin Liu
    @email: Anjin.Liu@uts.edu.au
    """
    # logging.debug("************* X's shape: %s", np.shape(X))
    n_samples, n_features = np.shape(X)
    X_fuzzy_dms = np.empty([n_samples, 0])
    for feature_idx in range(n_features):
        X_fuzzy_dm, _, _ = degree_of_membership_build(r_seed=0, X_df=pd.DataFrame(X[:, feature_idx]), n_conv=n_conv,
                                                      fuzzy_reg=fuzzy_reg)
        X_fuzzy_dms = np.concatenate((X_fuzzy_dms, X_fuzzy_dm), axis=1)
    # logging.debug("************* X_fuzzy_dms's shape: %s", np.shape(X_fuzzy_dms))
    return X_fuzzy_dms

    # X_df = pd.DataFrame(X)
    # X_fuzzy_dms, _, _ = degree_of_membership_build(r_seed=0, X_df=X_df.iloc[:, 0], n_conv=5)

    # Another try:  ====================================
    # X_fuzzy_dms = []
    # _, n_features = np.shape(X)
    # for feature_idx in range(n_features):
    #     X_fuzzy_dm, _, _ = degree_of_membership_build(r_seed=0, X_df=pd.DataFrame(X[:, feature_idx]), n_conv=5)
    #     X_fuzzy_dms.append(X_fuzzy_dm)
    # logging.debug("************* X_fuzzy_dms's elements: %s", np.asarray(X_fuzzy_dms).shape)
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
    y : array-like, 1-dimensional, elements must be numerical value and start from 0

    n_ohe_col : int

    Returns
    -------

    """
    y = y.astype(int)

    if n_ohe_col is None:
        n_ohe_col = np.amax(y) + 1  # np.max is just an alias for np.amax.

    one_hot = np.zeros((y.shape[0], n_ohe_col))  # Or, use y.size instead of y.shape[0]
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


# =============================================================================
# Functions for Sampling
# =============================================================================
"""
## Definitions
- A population can be defined as including all people or items with the 
  characteristic one wishes to understand.
- Each observation in a population measures one or more properties (such as 
  weight, location, colour) of observable bodies distinguished as independent 
  objects or individuals.
"""


def resample_simple_random(X, y, n_subsets, n_samples_sub=None, replace=False):
    """
    Randomly draw a specified number of collections of independent sample
    subsets from the original sample sets in the simple random sampling method.

    Parameters
    ----------
    X : sequence of array-like of shape (n_samples, n_features)
        The input samples in the format of indexable data-structure, which can
        be arrays, lists, dataframes or scipy sparse matrices with consistent
        first dimension.

    y : array-like of shape (n_samples,)
        The target values (class labels), which are non-negative integers in
        classification and real numbers in regression. Its first dimension has
        to be the same as that of X.

    n_subsets : int
        The number of collections of subsets to generate.

    n_samples_sub : int, default=None
        The sample size in each subset to generate. If left to None this is
        automatically set to the first dimension of X.

    replace : bool, default=False
        Implements resampling with replacement. If False, each sampling for a
        sample subset will implement (sliced) random permutations, which is a
        simple sampling scheme without replacement ('WOR' - no element can be
        selected more than once in the same sample). If True, each sampling for
        a sample subset will implement a bootstrapping sampling scheme with
        replacement ('WR' - an element may appear multiple times in the one
        sample).

    Returns
    -------
    X_subsets, y_subsets : tuple, where each is a sequence of array-like
        The tuple of the generated lists of sample subsets. The first element
        in the tuple is the list of the sample subsets from X and the second
        is the list of sample subsets from y.
    """
    n_samples_super = X.shape[0]
    y = y.reshape(n_samples_super, 1)  # Be equivalent to: np.expand_dims(y, axis=1)
    X_y = np.concatenate((X, y), axis=1)  # Be equivalent to: np.hstack((X, y))
    np.random.shuffle(X_y)  # NB: Bootstrap never uses a random_state, i.e., never np.random.seed(random_state) before.

    if n_samples_sub is None:
        n_samples_sub = n_samples_super
    elif n_samples_sub is not None and n_samples_sub > n_samples_super:
        raise ValueError("The sample is larger than the population.")

    X_subsets = []
    y_subsets = []
    for _ in range(n_subsets):
        # The larger the sample size (starting from > 10,000),
        # the faster numpy.random.choice() is compared to random.sample().
        # When the sample size < 10,000, the reverse applies.
        idxs = np.random.choice(n_samples_super, n_samples_sub, replace=replace)
        X_y_bootstrap = X_y[idxs, :]
        X_bootstrap = X_y_bootstrap[:, :-1]
        y_bootstrap = X_y_bootstrap[:, -1]

        X_subsets.append(X_bootstrap)
        y_subsets.append(y_bootstrap)

    return X_subsets, y_subsets


def resample_bootstrap(X, y, n_subsets, n_samples_sub=None):
    """
    Draw a specified number of collections of independent sample subsets from
    the original sample sets in the bootstrapping sampling method.
    This is a sampling scheme with replacement ('WR' - an element may appear
    multiple times in the one sample).

    Parameters
    ----------
    X : sequence of array-like of shape (n_samples, n_features)
        The input samples in the format of indexable data-structure, which can
        be arrays, lists, dataframes or scipy sparse matrices with consistent
        first dimension.

    y : array-like of shape (n_samples,)
        The target values (class labels), which are non-negative integers in
        classification and real numbers in regression. Its first dimension has
        to be the same as that of X.

    n_subsets : int
        The number of collections of subsets to generate.

    n_samples_sub : int, default=None
        The sample size in each subset to generate. If left to None this is
        automatically set to the first dimension of X.

    Returns
    -------
    X_subsets, y_subsets : tuple, where each is a sequence of array-like
        The tuple of the generated lists of sample subsets. The first element
        in the tuple is the list of the sample subsets from X and the second
        is the list of sample subsets from y.
    """
    return resample_simple_random(X=X, y=y, n_subsets=n_subsets, n_samples_sub=n_samples_sub, replace=True)
