"""
@author: Zhaoqing Liu
@email : Zhaoqing.Liu-1@student.uts.edu.au
@date  : 11/12/2020 10:00 am
@desc  :
"""
from fuzzytrees.fdt_base import BaseFuzzyDecisionTree, DecisionTreeInterface, CRITERIA_FUNC_CLF, CRITERIA_FUNC_REG
from fuzzytrees.util_criterion_funcs import calculate_impurity_gain, calculate_value_by_majority_vote, \
    calculate_variance_reduction, calculate_mean, calculate_impurity_gain_ratio
from fuzzytrees.util_split_funcs import split_ds_2_bin, split_disc_ds_2_multi, split_ds_2_multi


# =============================================================================
# Public estimators, including CART, ID3, C4.5
# =============================================================================

class FuzzyCARTClassifier(BaseFuzzyDecisionTree, DecisionTreeInterface):
    """
    A fuzzy decision tree classifier.

    The CART algorithm can handle both continuous/numerical and discrete/categorical
    variables, and can be used for both classification and regression.

    NB: See FuzzyDecisionTreeProxy for descriptions of all parameters
    and attributes in this class.
    """

    # All parameters in this constructor should have default values.
    def __init__(self, disable_fuzzy=False, X_fuzzy_dms=None, fuzzification_params=None,
                 criterion_func=CRITERIA_FUNC_CLF["gini"], max_depth=float("inf"), min_samples_split=2,
                 min_impurity_split=1e-7, **kwargs):
        super().__init__(disable_fuzzy=disable_fuzzy, X_fuzzy_dms=X_fuzzy_dms,
                         fuzzification_params=fuzzification_params, criterion_func=criterion_func, max_depth=max_depth,
                         min_samples_split=min_samples_split, min_impurity_split=min_impurity_split, **kwargs)
        self._split_ds_func = split_ds_2_bin
        self._impurity_gain_calc_func = calculate_impurity_gain
        self._leaf_value_calc_func = calculate_value_by_majority_vote

    # def fit(self, X_train, y_train):
    #     self._split_ds_func = split_ds_2_bin
    #     self._impurity_gain_calc_func = calculate_impurity_gain
    #     self._leaf_value_calc_func = calculate_value_by_majority_vote
    #     super().fit(X_train=X_train, y_train=y_train)


class FuzzyCARTRegressor(BaseFuzzyDecisionTree, DecisionTreeInterface):
    """
    A fuzzy CART decision tree regressor.

    The CART algorithm can handle both continuous/numerical and discrete/categorical
    variables, and can be used for both classification and regression.

    NB: See FuzzyDecisionTreeProxy for descriptions of all parameters
    and attributes in this class.
    """

    # All parameters in this constructor should have default values.
    def __init__(self, disable_fuzzy=False, X_fuzzy_dms=None, fuzzification_params=None,
                 criterion_func=CRITERIA_FUNC_REG["mse"], max_depth=float("inf"), min_samples_split=2,
                 min_impurity_split=1e-7, **kwargs):
        super().__init__(disable_fuzzy=disable_fuzzy, X_fuzzy_dms=X_fuzzy_dms,
                         fuzzification_params=fuzzification_params, criterion_func=criterion_func, max_depth=max_depth,
                         min_samples_split=min_samples_split, min_impurity_split=min_impurity_split, **kwargs)
        self._split_ds_func = split_ds_2_bin
        self._impurity_gain_calc_func = calculate_variance_reduction
        self._leaf_value_calc_func = calculate_mean

    # def fit(self, X_train, y_train):
    #     self._split_ds_func = split_ds_2_bin
    #     self._impurity_gain_calc_func = calculate_variance_reduction
    #     self._leaf_value_calc_func = calculate_mean
    #     super().fit(X_train=X_train, y_train=y_train)


class FuzzyID3Classifier(BaseFuzzyDecisionTree, DecisionTreeInterface):
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
    def __init__(self, disable_fuzzy=False, X_fuzzy_dms=None, fuzzification_params=None,
                 criterion_func=CRITERIA_FUNC_CLF["entropy"], max_depth=float("inf"), min_samples_split=2,
                 min_impurity_split=1e-7, **kwargs):
        super().__init__(disable_fuzzy=disable_fuzzy, X_fuzzy_dms=X_fuzzy_dms,
                         fuzzification_params=fuzzification_params, criterion_func=criterion_func, max_depth=max_depth,
                         min_samples_split=min_samples_split, min_impurity_split=min_impurity_split, **kwargs)

    def fit(self, X_train, y_train):
        self._split_ds_func = split_disc_ds_2_multi
        self._impurity_gain_calc_func = calculate_impurity_gain
        self._leaf_value_calc_func = calculate_value_by_majority_vote
        super().fit(X_train=X_train, y_train=y_train)


class FuzzyC45Classifier(BaseFuzzyDecisionTree, DecisionTreeInterface):
    """
    A fuzzy C4.5 decision tree classifier.

    The C4.5 algorithm can handle both continuous/numerical and discrete/categorical
    variables, but can only be used for classification.

    NB: See FuzzyDecisionTreeProxy for descriptions of all parameters
    and attributes in this class.
    """

    # All parameters in this constructor should have default values.
    def __init__(self, disable_fuzzy=False, X_fuzzy_dms=None, fuzzification_params=None,
                 criterion_func=CRITERIA_FUNC_CLF["entropy"], max_depth=float("inf"), min_samples_split=2,
                 min_impurity_split=1e-7, **kwargs):
        super().__init__(disable_fuzzy=disable_fuzzy, X_fuzzy_dms=X_fuzzy_dms,
                         fuzzification_params=fuzzification_params, criterion_func=criterion_func, max_depth=max_depth,
                         min_samples_split=min_samples_split, min_impurity_split=min_impurity_split, **kwargs)

    def fit(self, X_train, y_train):
        self._split_ds_func = split_ds_2_multi
        self._impurity_gain_calc_func = calculate_impurity_gain_ratio
        self._leaf_value_calc_func = calculate_value_by_majority_vote
        super().fit(X_train=X_train, y_train=y_train)

