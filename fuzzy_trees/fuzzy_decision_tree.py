# _*_coding:utf-8_*_
"""
@author: Zhaoqing Liu
@email: Zhaoqing.Liu-1@student.uts.edu.au
@date: 11/12/2020 10:00 am
@desc:
"""
from abc import ABCMeta
import numpy as np

from fuzzy_trees.fuzzy_decision_tree_proxy import DecisionTreeInterface, CRITERIA_FUNC_CLF, Node, SplitRule, BinarySubtrees, \
    CRITERIA_FUNC_REG
from fuzzy_trees.util_criterion_funcs import calculate_impurity_gain, calculate_value_by_majority_vote, \
    calculate_variance_reduction, calculate_mean, calculate_proba, calculate_impurity_gain_ratio
from fuzzy_trees.util_split_funcs import split_ds_2_bin, split_ds_2_multi, split_disc_ds_2_multi


# =============================================================================
# Base fuzzy decision tree
# =============================================================================

class FuzzyDecisionTree(metaclass=ABCMeta):
    """
    Base fuzzy decision tree class that encapsulates all base functions to be
    inherited by all derived classes (and attributes, if required).

    Warning: This class should not be used directly.
    Use derived classes instead.

    NB: See FuzzyDecisionTreeClassifierAPI and FuzzyDecisionTreeClassifierAPI
    for descriptions of all parameters and attributes in this class.
    """

    # The parameters in this constructor don't need to have default values.
    def __init__(self, disable_fuzzy, X_fuzzy_dms, fuzzification_params, criterion_func, max_depth, min_samples_split, min_impurity_split,
                 **kwargs):
        self.disable_fuzzy = disable_fuzzy
        self.X_fuzzy_dms = X_fuzzy_dms
        self.fuzzification_params = fuzzification_params
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

        NB: Only decision tree components are generated, either
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
        # Note that both X and y must have same number of dimensions.
        if len(np.shape(y)) == 1:
            # Do ascending dimension on y, and keep the column arrangement.
            y = np.expand_dims(y, axis=1)
        # Concatenate X and y as last column of X
        ds_train = np.concatenate((X, y), axis=1)

        # Start iterating over all features to get the best split.
        n_samples, n_features = np.shape(X)

        # Calculate the number of iterations over features. NB: fuzzy features have more conv_k times of original number of features.
        n_loop = n_features
        if not self.disable_fuzzy:
            n_loop = int(n_features / (self.fuzzification_params.conv_k + 1))  # denominator=conv_k + 1. If the FCM algorithm selects n optimal fuzzy sets, the calculation here will be deprecated.

        for feature_idx in range(n_loop):
            # Calculate the sum of all the membership degrees of the current feature values.
            total_dm = None
            if not self.disable_fuzzy:
                start = (feature_idx + 1) * self.fuzzification_params.conv_k
                stop = (feature_idx + 2) * self.fuzzification_params.conv_k
                total_dm = np.sum(X[:, start:stop])
                # print(feature_idx, "-th feature: total degree of membership:", total_dm)

            # Get all unique values of the feature with feature_idx group by value classes.
            feature_values = np.expand_dims(X[:, feature_idx], axis=1)

            # Calculate impurity_gain in each iteration over all unique feature values.
            unique_values = np.unique(feature_values)
            count = 0
            for unique_value in unique_values:
                count += 1
                subset_true, subset_false = self._split_ds_func(ds_train, feature_idx, unique_value)

                if len(subset_true) > 0 and len(subset_false) > 0:
                    # Calculate the membership probability of each subset according to the fuzzy splitting criterion.
                    p_subset_true_dm = None
                    p_subset_false_dm = None
                    if not self.disable_fuzzy and total_dm is not None and total_dm > 0.0:
                        start = (feature_idx + 1) * self.fuzzification_params.conv_k
                        stop = (feature_idx + 2) * self.fuzzification_params.conv_k
                        subset_true_dm = np.sum(subset_true[:, start:stop])
                        p_subset_true_dm = subset_true_dm / total_dm
                        # print("    ", count, "-th split: subset_true's degree of membership:", subset_true_dm)
                        start = (feature_idx + 1) * self.fuzzification_params.conv_k
                        stop = (feature_idx + 2) * self.fuzzification_params.conv_k
                        subset_false_dm = np.sum(subset_false[:, start:stop])
                        p_subset_false_dm = subset_false_dm / total_dm
                        # print("    ", count, "-th split: subset_false's degree of membership:", subset_false_dm)

                    y_subset_true = subset_true[:, n_loop:]  # For non-fuzzy trees, n_loop is exactly the number of features
                    y_subset_false = subset_false[:, n_loop:]  # For non-fuzzy trees, n_loop is exactly the number of features

                    impurity_gain = self._impurity_gain_calc_func(y, y_subset_true, y_subset_false, self.criterion_func, p_subset_true_dm=p_subset_true_dm, p_subset_false_dm=p_subset_false_dm)
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
# Public estimators
# =============================================================================

class FuzzyDecisionTreeClassifier(FuzzyDecisionTree, DecisionTreeInterface):
    """
    A fuzzy decision tree classifier.

    The CART algorithm can handle both continuous/numerical and discrete/categorical
    variables, and can be used for both classification and regression.

    NB: See FuzzyDecisionTreeProxy for descriptions of all parameters
    and attributes in this class.
    """

    # All parameters in this constructor should have default values.
    def __init__(self, disable_fuzzy=False, X_fuzzy_dms=None, fuzzification_params=None, criterion_func=CRITERIA_FUNC_CLF["gini"], max_depth=float("inf"), min_samples_split=2, min_impurity_split=1e-7, **kwargs):
        super().__init__(disable_fuzzy=disable_fuzzy, X_fuzzy_dms=X_fuzzy_dms, fuzzification_params=fuzzification_params, criterion_func=criterion_func, max_depth=max_depth, min_samples_split=min_samples_split, min_impurity_split=min_impurity_split, **kwargs)

    def fit(self, X_train, y_train):
        self._split_ds_func = split_ds_2_bin
        self._impurity_gain_calc_func = calculate_impurity_gain
        self._leaf_value_calc_func = calculate_value_by_majority_vote
        super().fit(X_train=X_train, y_train=y_train)


class FuzzyDecisionTreeRegressor(FuzzyDecisionTree, DecisionTreeInterface):
    """
    A fuzzy CART decision tree regressor.

    The CART algorithm can handle both continuous/numerical and discrete/categorical
    variables, and can be used for both classification and regression.

    NB: See FuzzyDecisionTreeProxy for descriptions of all parameters
    and attributes in this class.
    """

    # All parameters in this constructor should have default values.
    def __init__(self, disable_fuzzy=False, X_fuzzy_dms=None, fuzzification_params=None, criterion_func=CRITERIA_FUNC_REG["mse"], max_depth=float("inf"), min_samples_split=2, min_impurity_split=1e-7, **kwargs):
        super().__init__(disable_fuzzy=disable_fuzzy, X_fuzzy_dms=X_fuzzy_dms, fuzzification_params=fuzzification_params, criterion_func=criterion_func, max_depth=max_depth, min_samples_split=min_samples_split, min_impurity_split=min_impurity_split, **kwargs)

    def fit(self, X_train, y_train):
        self._split_ds_func = split_ds_2_bin
        self._impurity_gain_calc_func = calculate_variance_reduction
        self._leaf_value_calc_func = calculate_mean
        super().fit(X_train=X_train, y_train=y_train)


class FuzzyID3Classifier(FuzzyDecisionTree, DecisionTreeInterface):
    # Need to modify the code "subset_true, subset_false = split_dataset(ds_train, feature_idx, unique_value)"
    # in baseclass "FuzzyDecisionTree" to using a function object passed externally.
    """
    A fuzzy ID3 decision tree classifier.

    The ID3 algorithm can only handle discrete/categorical variables and can
    only be used for classification.

    NB: See FuzzyDecisionTreeProxy for descriptions of all parameters
    and attributes in this class.
    """

    # All parameters in this constructor should have default values.
    def __init__(self, disable_fuzzy=False, X_fuzzy_dms=None, fuzzification_params=None, criterion_func=CRITERIA_FUNC_CLF["entropy"], max_depth=float("inf"), min_samples_split=2, min_impurity_split=1e-7, **kwargs):
        super().__init__(disable_fuzzy=disable_fuzzy, X_fuzzy_dms=X_fuzzy_dms, fuzzification_params=fuzzification_params, criterion_func=criterion_func, max_depth=max_depth, min_samples_split=min_samples_split, min_impurity_split=min_impurity_split, **kwargs)

    def fit(self, X_train, y_train):
        self._split_ds_func = split_disc_ds_2_multi
        self._impurity_gain_calc_func = calculate_impurity_gain
        self._leaf_value_calc_func = calculate_value_by_majority_vote
        super().fit(X_train=X_train, y_train=y_train)


class FuzzyC45Classifier(FuzzyDecisionTree, DecisionTreeInterface):
    """
    A fuzzy C4.5 decision tree classifier.

    The C4.5 algorithm can handle both continuous/numerical and discrete/categorical
    variables, but can only be used for classification.

    NB: See FuzzyDecisionTreeProxy for descriptions of all parameters
    and attributes in this class.
    """

    # All parameters in this constructor should have default values.
    def __init__(self, disable_fuzzy=False, X_fuzzy_dms=None, fuzzification_params=None, criterion_func=CRITERIA_FUNC_CLF["entropy"], max_depth=float("inf"), min_samples_split=2, min_impurity_split=1e-7, **kwargs):
        super().__init__(disable_fuzzy=disable_fuzzy, X_fuzzy_dms=X_fuzzy_dms, fuzzification_params=fuzzification_params, criterion_func=criterion_func, max_depth=max_depth, min_samples_split=min_samples_split, min_impurity_split=min_impurity_split, **kwargs)

    def fit(self, X_train, y_train):
        self._split_ds_func = split_ds_2_multi
        self._impurity_gain_calc_func = calculate_impurity_gain_ratio
        self._leaf_value_calc_func = calculate_value_by_majority_vote
        super().fit(X_train=X_train, y_train=y_train)


if __name__ == "__main__":
    pass
