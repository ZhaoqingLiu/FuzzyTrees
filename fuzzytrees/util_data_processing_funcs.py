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

"""
Functions in this module are for preprocessing data:
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

def degree_of_membership_build(X_df, r_seed, conv_k, fuzzy_reg):
    """
    Build the degree of membership set of a feature. That set maps to
    the specified number of fuzzy sets of the feature.
    This is the process of transforming a crisp set into a fuzzy set.

    @author: Anjin Liu
    @email: Anjin.Liu@uts.edu.au

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
    theta_f = np.log(fuzzy_reg) - np.log(1 - fuzzy_reg)
    for idx, item in enumerate(degree_of_membership_theta):
        if fuzzy_reg == 0 or fuzzy_reg == 0.0 or fuzzy_reg == 1 or fuzzy_reg == 1.0:
            x_new[:, idx] = 1 - x_new[:, idx] / item
        else:
            x_new[:, idx] = 1 - x_new[:, idx] / item * theta_f
    x_new[x_new < 0] = 0

    # TODO: Searching an optimum fuzzy threshold by a loop according the specified stride.
    # np.where(x_new > fuzzy_th, x_new, 0.0)
    # np.where(x_new > fuzzy_th and x_new <= (1 - fuzzy_th), x_new, 1.0)
    # x_new[x_new <= fuzzy_th] = 0
    # x_new[x_new > (1 - fuzzy_th)] = 1
    # print("++++++++++++++++++++++++++++++++++++++")
    # print(x_new)
    # print("++++++++++++++++++++++++++++++++++++++")

    return x_new, centriods, degree_of_membership_theta


def extract_fuzzy_features(X, conv_k=5, fuzzy_reg=0.0):
    """
    Extract fuzzy features in feature fuzzification to generate degree of
    membership sets of each feature.

    NB: Feature fuzzification must be done in the data preprocessing, that is,
        before training the model and predicting new samples.

    @author: Anjin Liu
    @email: Anjin.Liu@uts.edu.au

    TODO: To be deprecated in version 1.0.
    TODO: To be verified by experiment: When using cross validation, which performance is better doing this before or after the partition of the data sets?
    """
    # print("************* X's shape:", np.shape(X))
    n_samples, n_features = np.shape(X)
    X_fuzzy_dms = np.empty([n_samples, 0])
    for feature_idx in range(n_features):
        X_fuzzy_dm, _, _ = degree_of_membership_build(r_seed=0, X_df=pd.DataFrame(X[:, feature_idx]), conv_k=conv_k,
                                                      fuzzy_reg=fuzzy_reg)
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


# =============================================================================
# Functions for Sampling
# =============================================================================
"""
## Definitions
- A population can be defined as including all people or items with the 
  characteristic one wishes to understand.
- Each observation measures one or more properties (such as weight, 
  location, colour) of observable bodies distinguished as independent 
  objects or individuals.

## Sampling methods
### 1. Simple random sampling
### 2. Systematic sampling
### 3. Stratified sampling
### 4. Probability-proportional-to-size sampling
### 5. Cluster sampling
### 6. Quota sampling
### 7. Minimax sampling
### 8. Accidental sampling
### 9. Voluntary sampling
### 10. Line-intercept sampling
### 11. Panel sampling
### 12. Snowball sampling
### 13. Theoretical sampling
"""


def resample_bootstrap(X, y, n_subsets, n_samples_sub=None):
    """
    Draw a specified number of collections of independent subsets in
    bootstrapping sampling method.
    This is a sampling scheme with replacement ('WR' - an element may
    appear multiple times in the one sample).

    Parameters
    ----------
    X: sequence of {array-like, sparse matrix} of shape
        (n_samples_super, n_features), at least (n_samples_super,)
        Indexable data-structures can be arrays, lists, dataframes or
        scipy sparse matrices with consistent first dimension.
        The input samples in the format of indexable data-structure.

    y: array-like of shape (n_samples_super,)
        The target values (class labels), which are non-negative integers
        in classification and real numbers in regression.
        Its first dimension is the same as that of X.

    n_subsets: int
        Number of collections of subsets to generate.

    n_samples_sub: int, default=None
        Number of samples in each subset to generate. If left to None
        this is automatically set to the first dimension of X.

    Returns
    -------
    subsets: sequence of array-like
        Bootstrapping subsets generated. The number of samples in each 
        subset is the first dimension of the X.
    """
    n_samples_super = X.shape[0]
    y = y.reshape(n_samples_super, 1)  # Be equivalent to: np.expand_dims(y, axis=1)
    X_y = np.concatenate((X, y), axis=1)  # Be equivalent to: np.hstack((X, y))
    np.random.shuffle(X_y)  # NB: Bootstrap never uses a random_state, i.e., never np.random.seed(random_state) before.

    if n_samples_sub is None:
        n_samples_sub = n_samples_super

    subsets = []
    for _ in range(n_subsets):
        idxs = np.random.choice(n_samples_super, n_samples_sub, replace=True)
        X_y_bootstrap = X_y[idxs, :]
        X_bootstrap = X_y_bootstrap[:, :-1]
        y_bootstrap = X_y_bootstrap[:, -1]

        subsets.append([X_bootstrap, y_bootstrap])

    return subsets


def resample_simple_random(X, y, n_subsets, n_samples_sub=None):
    """
    Randomly draw a specified number of collections of independent
    subsets in simple random sampling method.
    This is a sampling scheme without replacement ('WOR' - no element
    can be selected more than once in the same sample).

    NB: The simple random sampling method first numbers all the
    observations in the population to make them indexable, and then
    randomly draws a number of observations from the indexable
    population to form each sample though a raffle/lots or a list of
    random numbers.

    Parameters
    ----------
    X: sequence of {array-like, sparse matrix} of shape
        (n_samples_super, n_features), at least (n_samples_super,)
        Indexable data-structures can be arrays, lists, dataframes or
        scipy sparse matrices with consistent first dimension.
        The input samples in the format of indexable data-structure.

    y: array-like of shape (n_samples_super,)
        The target values (class labels), which are non-negative integers
        in classification and real numbers in regression.
        Its first dimension is the same as that of X.

    n_subsets: int
        Number of collections of subsets to generate.

    n_samples_sub: int, default=None
        Number of samples in each subset to generate. If left to None
        this is automatically set to the first dimension of X.

    Returns
    -------
    subsets: sequence of array-like
        Bootstrapping subsets generated. The number of samples in each
        subset is the first dimension of the X.
    """
    try:
        n_samples_super = X.shape[0]
        y = y.reshape(n_samples_super, 1)
        X_y = np.concatenate((X, y), axis=1)
        np.random.shuffle(X_y)

        if n_samples_sub is None:
            n_samples_sub = n_samples_super

        subsets = []
        for _ in range(n_subsets):
            idxs = np.random.choice(n_samples_super, n_samples_sub, replace=False)
            X_y_bootstrap = X_y[idxs, :]
            X_bootstrap = X_y_bootstrap[:, :-1]
            y_bootstrap = X_y_bootstrap[:, -1]

            subsets.append([X_bootstrap, y_bootstrap])

        return subsets
    except:
        print('The sample size is greater than the population size.')
