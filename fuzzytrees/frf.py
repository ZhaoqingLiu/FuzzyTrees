"""
@author: Zhaoqing Liu
@email : Zhaoqing.Liu-1@student.uts.edu.au
@date  : 3/7/21 5:04 pm
@desc  :
@ref   : 
"""
from abc import ABCMeta


class BaseFuzzyRF(metaclass=ABCMeta):
    """
    Base fuzzy random forests (RF) class that encapsulates all base functions
    to be inherited by all derived classes (and attributes, if required).
    This algorithm is a fuzzy extension of the random forests proposed by
    Leo Breiman [1]_ and Adele Cutler [2]_.


    Warning: This class should not be used directly.
    Use derived classes instead.

    References
    ----------
    .. [1] Breiman, L., 2001. Random forests. Machine learning, 45(1),
           pp.5-32.
    .. [2] RColorBrewer, S. and Liaw, M.A., 2018. Package ‘randomForest’.
           University of California, Berkeley: Berkeley, CA, USA.
    """
    pass


class FuzzyRFClassifier(BaseFuzzyRF):
    """
    Fuzzy random forests classifier.

    NB: For classification tasks, the class that is the mode of
    the classes of the individual trees is returned.
    """
    pass


class FuzzyRFRegressor(BaseFuzzyRF):
    """
    Fuzzy random forests regressor.

    NB: For regression tasks, the mean or average prediction of
    the individual trees is returned.
    """
    pass
