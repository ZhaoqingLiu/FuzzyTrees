"""
@author: Zhaoqing Liu
@email : Zhaoqing.Liu-1@student.uts.edu.au
@date  : 02/02/2021 3:31 pm
@desc  :
"""
from abc import ABCMeta


class FuzzyRDF(metaclass=ABCMeta):
    """
    Base fuzzy random decision forest class that encapsulates all base functions
    to be inherited by all derived classes (and attributes, if required).

    Warning: This class should not be used directly.
    Use derived classes instead.

    Attributes
    ----------

    """


    def fit(self, X_train, y_train):
        """
        Fit the fuzzy gradient boosting model.

        Parameters
        ----------
        X_train: {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        y_train: array-like of shape (n_samples,)
            Target values (strings or integers in classification, real numbers
            in regression)
        """
        pass

    def predict(self, X):
        """
        Predict class for X.

        Parameters
        ----------
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred: ndarray of shape (n_samples,)
            The predicted values.
        """
        pass


class FuzzyGBDTClassifier(FuzzyRDF):
    """
    A fuzzy random decision forest classifier.

    Parameters:
    -----------

    Attributes
    ----------

    """
    pass


class FuzzyGBDTRegressor(FuzzyRDF):
    """
    A fuzzy random decision forest regressor.

    Parameters:
    -----------

    Attributes
    ----------

    """
    pass
