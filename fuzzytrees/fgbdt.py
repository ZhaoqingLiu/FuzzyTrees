# _*_coding:utf-8_*_
"""
@author : Zhaoqing Liu
@email  : Zhaoqing.Liu-1@student.uts.edu.au
"""
import logging
from abc import ABCMeta
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.extmath import softmax

from fuzzytrees.fdt_base import FuzzyDecisionTreeWrapper, CRITERIA_FUNC_REG
from fuzzytrees.fdts import FuzzyCARTRegressor
from fuzzytrees.util_tree_criterion_funcs import LeastSquaresFunction, SoftLeastSquaresFunction
from fuzzytrees.util_preprocessing_funcs import one_hot_encode


class FuzzyGBDT(metaclass=ABCMeta):
    """
    Base fuzzy decision tree class that encapsulates all base functions to be
    inherited by all derived classes (and attributes, if required).

    Warnings
    --------
    This class should not be used directly.
    Use derived classes instead.

    Parameters
    ----------
    disable_fuzzy : bool, default=False
        Set whether the specified fuzzy decision tree uses the fuzzification.
        If disable_fuzzy=True, the specified fuzzy decision tree is equivalent
        to a naive decision tree.

    fuzzification_options : FuzzificationOptions, default=None
        Protocol message class that encapsulates all the options of the
        fuzzification settings used by the specified fuzzy decision tree.

    criterion_func : {"mse", "mae"}, default="mse"
        The criterion function used by the function that calculates the impurity
        gain of the target values.
        NB: Only use a criterion function for decision tree regressor.

    learning_rate : float, default=0.1
        The step length taken in the training using the loss of the negative
        gradient descent strategy. It is used to reduce the contribution of
        each tree.
        NB: There is a trade-off between learning_rate and n_estimators.

    n_estimators : int, default=100
        The number of fuzzy decision trees to be used.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if ``n_iter_no_change`` is set to an integer.

    n_iter_no_change : int, default=None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations. The split is stratified.

    max_depth : int, default=3
        The maximum depth of the tree to be trained.

    min_samples_split : int, default=2
        The minimum number of samples required to split a node. If a node has a
        sample number above this threshold, it will be split, otherwise it
        becomes a leaf node.

    min_impurity_split : float, default=1e-7
        The minimum impurity required to split a node. If a node's impurity is
        above this threshold, it will be split, otherwise it becomes a leaf node.

    is_regression : bool, default=True
        True or false depending on if we're doing regression or classification.

    Attributes
    ----------
    _loss_func : LossFunction
        The concrete object of the class LossFunction's derived classes.

    _estimators : ndarray of FuzzyDecisionTreeRegressor
        The collection of sub-estimators as base learners.
    """

    def __init__(self, disable_fuzzy, X_fuzzy_dms, fuzzification_options, criterion_func, learning_rate, n_estimators,
                 validation_fraction, n_iter_no_change, max_depth, min_samples_split, min_impurity_split,
                 is_regression):
        self.disable_fuzzy = disable_fuzzy
        self.X_fuzzy_dms = X_fuzzy_dms
        self.fuzzification_options = fuzzification_options
        self.criterion_func = criterion_func
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.is_regression = is_regression

        self._loss_func = LeastSquaresFunction() if self.is_regression else SoftLeastSquaresFunction()  # (Friedman et al., 1998; Friedman 2001)

        # NB: Use regression trees as base estimators in both regression and classification problems.
        # In classification problems, regression trees can use residuals to learn probabilities of
        # the classifications of samples.
        self._estimators = []
        for i in range(self.n_estimators):
            # self._estimators.append(FuzzyCARTRegressor(disable_fuzzy=self.disable_fuzzy, X_fuzzy_dms=self.X_fuzzy_dms,
            #                                            fuzzification_options=self.fuzzification_options,
            #                                            criterion_func=self.criterion_func, max_depth=self.max_depth,
            #                                            min_samples_split=self.min_samples_split,
            #                                            min_impurity_split=self.min_impurity_split))
            estimator = FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTRegressor,
                                                 disable_fuzzy=disable_fuzzy,
                                                 fuzzification_options=fuzzification_options,
                                                 criterion_func=criterion_func, max_depth=max_depth,
                                                 min_samples_split=min_samples_split,
                                                 min_impurity_split=min_impurity_split)
            self._estimators.append(estimator)

    def fit(self, X_train, y_train):
        """
        Fit the fuzzy gradient boosting model.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Input instances to be predicted.

        y_train : array-like of shape (n_samples,)
            Target values (strings or integers in classification, real numbers
            in regression)
        """
        # Use the first tree to fit the first estimator, and then use it
        # to predict values F_0(x).
        self._estimators[0].fit(X_train, y_train)
        y_pred = self._estimators[0].predict(X_train)
        # logging.debug("0-th estimator produces an initialised constant: %s", y_pred)

        # Then use the other tree iteratively to fit the other estimators by the
        # residuals of the last predictions. The first set of residuals is the
        # true values minus the values F_0(x).
        for i in range(1, self.n_estimators):
            gradient = self._loss_func.gradient(y_train, y_pred)
            self._estimators[i].fit(X_train, gradient)
            y_pred -= np.multiply(self.learning_rate, self._estimators[i].predict(X_train))
            # logging.debug("%d-th estimator produces a residual: %f", i, y_pred)

    def predict(self, X):
        """
        Predict class for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted values.
        """
        # Use the first fitted estimator to predict values F_0(x).
        y_pred = self._estimators[0].predict(X)

        # Then use the other fitting estimators to iteratively predict
        # the residuals and add them up to the values F_0(x).
        for i in range(1, self.n_estimators):
            y_pred -= np.multiply(self.learning_rate, self._estimators[i].predict(X))

        if not self.is_regression:
            # Use softmax function for multiple-class (consider sigmoid function if binary-class).
            y_pred = softmax(y_pred)
            y_pred = np.argmax(y_pred, axis=1)
            # # Use each probability distribution instead.
            # dummy = np.exp(y_pred)
            # sums = np.expand_dims(np.sum(dummy, axis=1), axis=1)
            # if np.all(sums == 0):
            #     y_pred = 0
            # else:
            #     y_pred = np.exp(y_pred) / sums
            # # Select the classification with the highest probability as the prediction.
            # y_pred = np.argmax(y_pred, axis=1)

        return y_pred


class FuzzyGBDTClassifier(FuzzyGBDT):
    """
    Fuzzy gradient boosting decision tree classifier.

    Parameters
    ----------
    disable_fuzzy : bool, default=False
        Set whether the specified fuzzy decision tree uses the fuzzification.
        If disable_fuzzy=True, the specified fuzzy decision tree is equivalent
        to a naive decision tree.

    fuzzification_options : FuzzificationOptions, default=None
        Protocol message class that encapsulates all the options of the
        fuzzification settings used by the specified fuzzy decision tree.

    criterion_func : {"mse", "mae"}, default="mse"
        The criterion function used by the function that calculates the impurity
        gain of the target values.
        NB: Only use a criterion function for decision tree regressor.

    learning_rate : float, default=0.1
        The step length taken in the training using the loss of the negative
        gradient descent strategy. It is used to reduce the contribution of
        each tree.
        NB: There is a trade-off between learning_rate and n_estimators.

    n_estimators : int, default=100
        The number of fuzzy decision trees to be used.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if ``n_iter_no_change`` is set to an integer.

    n_iter_no_change : int, default=None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations. The split is stratified.

    max_depth : int, default=3
        The maximum depth of the tree to be trained.

    min_samples_split : int, default=2
        The minimum number of samples required to split a node. If a node has a
        sample number above this threshold, it will be split, otherwise it
        becomes a leaf node.

    min_impurity_split : float, default=1e-7
        The minimum impurity required to split a node. If a node's impurity is
        above this threshold, it will be split, otherwise it becomes a leaf node.

    Attributes
    ----------
    _loss_func : LossFunction
        The concrete object of the class LossFunction's derived classes.

    _estimators : ndarray of FuzzyDecisionTreeRegressor
        The collection of fitted sub-estimators.
    """

    def __init__(self, disable_fuzzy=False, X_fuzzy_dms=None, fuzzification_options=None,
                 criterion_func=CRITERIA_FUNC_REG["mse"], learning_rate=0.1, n_estimators=100, validation_fraction=0.1,
                 n_iter_no_change=None, max_depth=3, min_samples_split=2, min_impurity_split=1e-7):
        super().__init__(disable_fuzzy=disable_fuzzy, X_fuzzy_dms=X_fuzzy_dms,
                         fuzzification_options=fuzzification_options, criterion_func=criterion_func,
                         learning_rate=learning_rate, n_estimators=n_estimators,
                         validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change,
                         max_depth=max_depth, min_samples_split=min_samples_split,
                         min_impurity_split=min_impurity_split, is_regression=False)

    def fit(self, X_train, y_train):
        logging.debug("**************** Shape before one-hot_encoding: %s", np.shape(y_train))
        if len(np.shape(y_train)) == 1:
            y_train = one_hot_encode(y_train)
        logging.debug("**************** Shape after one-hot_encoding: %s", np.shape(y_train))

        # Here is an alternative encoding method, but requires an additional change of dimension.
        # if len(np.shape(y_train)) == 1:
        #     y_train = np.expand_dims(y_train, axis=1)
        # transformer = OneHotEncoder(handle_unknown='ignore')
        # y_train = transformer.fit_transform(y_train).toarray()

        super().fit(X_train=X_train, y_train=y_train)


class FuzzyGBDTRegressor(FuzzyGBDT):
    """
    Fuzzy gradient boosting decision tree regressor.

    Parameters
    ----------
    disable_fuzzy : bool, default=False
        Set whether the specified fuzzy decision tree uses the fuzzification.
        If disable_fuzzy=True, the specified fuzzy decision tree is equivalent
        to a naive decision tree.

    fuzzification_options : FuzzificationOptions, default=None
        Protocol message class that encapsulates all the options of the
        fuzzification settings used by the specified fuzzy decision tree.

    criterion_func : {"mse", "mae"}, default="mse"
        The criterion function used by the function that calculates the impurity
        gain of the target values.
        NB: Only use a criterion function for decision tree regressor.

    learning_rate : float, default=0.1
        The step length taken in the training using the loss of the negative
        gradient descent strategy. It is used to reduce the contribution of
        each tree.
        NB: There is a trade-off between learning_rate and n_estimators.

    n_estimators : int, default=100
        The number of fuzzy decision trees to be used.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if ``n_iter_no_change`` is set to an integer.

    n_iter_no_change : int, default=None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations. The split is stratified.

    max_depth : int, default=3
        The maximum depth of the tree to be trained.

    min_samples_split : int, default=2
        The minimum number of samples required to split a node. If a node has a
        sample number above this threshold, it will be split, otherwise it
        becomes a leaf node.

    min_impurity_split : float, default=1e-7
        The minimum impurity required to split a node. If a node's impurity is
        above this threshold, it will be split, otherwise it becomes a leaf node.

    Attributes
    ----------
    _loss_func : LossFunction
        The concrete object of the class LossFunction's derived classes.

    _estimators : ndarray of FuzzyDecisionTreeRegressor
        The collection of fitted sub-estimators.
    """

    def __init__(self, disable_fuzzy=False, X_fuzzy_dms=None, fuzzification_options=None,
                 criterion_func=CRITERIA_FUNC_REG["mse"], learning_rate=0.1, n_estimators=100, validation_fraction=0.1,
                 n_iter_no_change=None, max_depth=3, min_samples_split=2, min_impurity_split=1e-7):
        super().__init__(disable_fuzzy=disable_fuzzy, X_fuzzy_dms=X_fuzzy_dms,
                         fuzzification_options=fuzzification_options, criterion_func=criterion_func,
                         learning_rate=learning_rate, n_estimators=n_estimators,
                         validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change,
                         max_depth=max_depth, min_samples_split=min_samples_split,
                         min_impurity_split=min_impurity_split, is_regression=True)
