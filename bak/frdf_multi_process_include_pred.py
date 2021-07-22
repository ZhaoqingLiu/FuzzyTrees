"""
@author: Zhaoqing Liu
@email : Zhaoqing.Liu-1@student.uts.edu.au
@date  : 02/02/2021 3:31 pm
@desc  :
"""
import ctypes
import multiprocessing
from abc import ABCMeta
import numpy as np

from fuzzytrees.fdt_base import FuzzyDecisionTreeWrapper
from fuzzytrees.fdts import FuzzyCARTClassifier, FuzzyCARTRegressor
from fuzzytrees.util_tree_criterion_funcs import majority_vote, mean_value
from fuzzytrees.util_preprocessing_funcs import resample_bootstrap


class FuzzyRDF(metaclass=ABCMeta):
    """
    Base fuzzy random decision forests (RF) class that encapsulates all
    base functions to be inherited by all derived classes (and attributes,
    if required). This algorithm is a fuzzy extension of the random decision
    forests proposed by Tin Kam Ho [1]_.


    Warning: This class should not be used directly.
    Use derived classes instead.

    Notes
    -----
    About RF
    The first algorithm for random decision forests was created by Tin Kam Ho [1]_
    using the random subspace method [2]_, which, in Ho's formulation, is a way to
    implement the "stochastic discrimination" approach to classification proposed
    by Eugene Kleinberg.
    An extension of the algorithm was developed by Leo Breiman [4]_ and Adele Cutler
    [5]_. The extension combines Breiman's "bagging" idea and random selection of
    features, introduced first by Ho [1]_ and later independently by Amit and
    Geman [3]_ in order to construct a collection of decision trees with controlled
    variance.

    The randomness of RF is reflected in two aspects:
    1. RF uses the bootstrapping sampling method to randomly selects n samples
        from the original dataset to train each tree as the base learner,
        where n is the sample size of the original dataset.
        NB: The sample size of each training dataset is the same as that of
        the original dataset, but the bootstrapping sampling method may make
        the elements in the same training dataset duplicate, or the elements
        in different training datasets duplicate.
    2. During the construction of each tree, RF also randomly selects m
        features of the training dataset, and then searches the optimal
        features from the randomly selected features each time when splitting
        a tree node to find the best splitting point.
        Different RFs have different random feature selection methods. For
        example, Tin Kam Ho's RF adopts tree-level random feature selection,
        i.e. the RF randomly selects m features of the training dataset for
        subsequently splitting all tree nodes. By contrast, Leo Breiman's RF
        adopts node-level random feature selection, i.e. the RF randomly
        selects m features of the training dataset when splitting a node
        every time.
        Let M be the total number of features of data and m be the number
        of selected features. Generally, the value can be tried from the
        following usual practices:
        - For classification problems, :math:`m = 1 / 3 * M`;
        - For regression problems, :math:`m = log_2 (M + 1)`;
        - By defaults, :math:`m = sqrt(M)`.

    References
    ----------
    .. [1] Ho, T.K., 1995, August. Random decision forests. In Proceedings
           of 3rd international conference on document analysis and
           recognition (Vol. 1, pp. 278-282). IEEE.
    .. [2] Ho, T.K., 1998. The random subspace method for constructing
           decision forests. IEEE transactions on pattern analysis and
           machine intelligence, 20(8), pp.832-844.
    .. [3] Amit, Y. and Geman, D., 1997. Shape quantization and recognition
           with randomized trees. Neural computation, 9(7), pp.1545-1588.
    .. [4] Breiman, L., 2001. Random forests. Machine learning, 45(1),
           pp.5-32.
    .. [5] RColorBrewer, S. and Liaw, M.A., 2018. Package ‘randomForest’.
           University of California, Berkeley: Berkeley, CA, USA.
    """

    def __init__(self, disable_fuzzy, fuzzification_options, criterion_func, n_estimators,
                 max_depth, min_samples_split, min_impurity_split, max_features, multi_process_options):
        self.disable_fuzzy = disable_fuzzy
        self.fuzzification_options = fuzzification_options
        self.criterion_func = criterion_func
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.max_features = max_features
        self.multi_process_options = multi_process_options

        self._estimators = []  # Forest initialised in derived classes.
        self._res_func = None

        self._n_processes = None
        if self.multi_process_options is not None:
            self._n_processes = multiprocessing.cpu_count() if self.multi_process_options.n_cpu_cores_req is None else self.multi_process_options.n_cpu_cores_req

    def fit(self, X_train, y_train):
        """
        Fit the fuzzy random decision forest model (in multi-process mode).

        Parameters
        ----------
        X_train: {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        y_train: array-like of shape (n_samples,)
            Target values (non-negative integers in classification,
            real numbers in regression)
            NB: The input array needs to be of integer dtype, otherwise a
            TypeError is raised.
        """
        # Randomly select n_estimators training subsets through bootstrapping sampling.
        X_train_subsets, y_train_subsets = resample_bootstrap(X_train, y_train, n_subsets=self.n_estimators)

        # Get the number of the data features.
        n_features = X_train.shape[1]
        if not self.disable_fuzzy:
            # NB: Except the columns of fuzzy degrees of membership.
            n_features = int(n_features / (self.fuzzification_options.conv_k + 1))

        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        # Train each tree in the forest.
        # NB: Iterate the n_estimators training subsets generated above, training a tree in each iteration.
        if self.multi_process_options:  # When self.multi_process_options is not None
            # In multi-process mode.
            with multiprocessing.Manager() as mg:
                # Create a connection used to communicate between main process and its child processes.
                q = multiprocessing.Manager().Queue()
                # Create a pool for main process to manage its child processes in parallel.
                pool = multiprocessing.Pool(processes=self._n_processes)
                for i in range(self.n_estimators):
                    pool.apply_async(self._fit_one, args=(X_train_subsets[i], y_train_subsets[i], n_features, i, q,))
                pool.close()
                pool.join()

                # Replace all the estimators in the forest with the ones returned by the sub-processes.
                idx = 0
                while not q.empty():
                    estimator = q.get()
                    self._estimators[idx] = estimator
                    print("Replaced the {}-th estimators in the forest.".format(idx))
                    idx += 1
        else:
            # In single-process mode.
            for i in range(self.n_estimators):
                self._fit_one(X_train_subsets[i], y_train_subsets[i], n_features, i)

    def _fit_one(self, X_train_subset, y_train_subset, n_features, i, q=None):
        # Randomly select features.
        idxs = np.random.choice(n_features, self.max_features, replace=True)
        if not self.disable_fuzzy:
            # Select the columns of fuzzy degrees of membership at the same time.
            idxs_cp = np.copy(idxs)
            for idx in idxs_cp:
                # Columns of the idx-th feature's degrees of membership start from
                # "n_original_features + feature_idx * self.fuzzification_options.conv_k + 1", and end with
                # "n_original_features + (feature_idx + 1) * self.fuzzification_options.conv_k + 1".
                start = n_features + idx * self.fuzzification_options.conv_k
                stop = n_features + (idx + 1) * self.fuzzification_options.conv_k
                idxs_dm = np.arange(start=start, stop=stop, step=1, dtype=int)
                idxs = np.concatenate((idxs, idxs_dm), axis=0)
        X_train_subset = X_train_subset[:, idxs]

        # Fit an estimator and record the indexes of fitted features to prepare for predictions.
        self._estimators[i].fit(X_train_subset, y_train_subset)
        self._estimators[i].feature_idxs = idxs

        # In multi-process mode, the trained estimator needs to be passed back to the master process because
        # the sub-process cannot update the global variables in the master process.
        if self.multi_process_options:
            if not q.full():
                q.put(self._estimators[i])

        print("{}-th tree fitting is complete.".format(i))

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
        y_preds = []

        if self.multi_process_options:  # When self.multi_process_options is not None
            # In multi-process mode.
            with multiprocessing.Manager() as mg:
                # Create a connection used to communicate between main process and its child processes.
                q = multiprocessing.Manager().Queue()
                # Create a pool for main process to manage its child processes in parallel.
                pool = multiprocessing.Pool(processes=self._n_processes)
                for i in range(self.n_estimators):
                    pool.apply_async(self._predict_by_one, args=(X, i, q,))

                pool.close()
                pool.join()

                # Add all the predictions returned by the sub-processes into the result set.
                while not q.empty():
                    y_pred = q.get()
                    y_preds.append(y_pred)
        else:
            # In single-process mode.
            for i in range(self.n_estimators):
                y_preds.append(self._predict_by_one(X, i))

        print("Totally {} predictions.".format(len(y_preds)))
        y_preds = np.array(y_preds).T

        return self._res_func(y_preds)

    def _predict_by_one(self, X, i, q=None):
        """
        Predict one result using one estimator in the forest.

        Parameters
        ----------
        X
        i
        q

        Returns
        -------

        """
        idxs = self._estimators[i].feature_idxs
        X_subset = X[:, idxs]
        y_pred = self._estimators[i].predict(X_subset)

        # In multi-process mode, the prediction needs to be passed back to the master process because
        # the sub-process cannot update the global variables in the master process.
        if self.multi_process_options:
            if not q.full():
                q.put(y_pred)
        else:
            return y_pred


class FuzzyRDFClassifier(FuzzyRDF):
    """
    Fuzzy random decision forests classifier.

    NB: For classification tasks, the class that is the mode of
    the classes of the individual trees is returned.

    Parameters:
    -----------
    disable_fuzzy: bool, default=False
        Set whether the specified fuzzy decision tree uses the fuzzification.
        If disable_fuzzy=True, the specified fuzzy decision tree is equivalent
        to a naive decision tree.

    fuzzification_options: FuzzificationOptions, default=None
        Protocol message class that encapsulates all the options of the
        fuzzification settings used by the specified fuzzy decision tree.

    criterion_func: {"gini", "entropy"}, default="gini"
        The criterion function used by the function that calculates the impurity
        gain of the target values.
        NB: Only use a criterion function for decision tree regressor.

    n_estimators: int, default=100
        The number of fuzzy decision trees to be used.

    max_depth: int, default=3
        The maximum depth of the tree to be trained.

    min_samples_split: int, default=2
        The minimum number of samples required to split a node. If a node has a
        sample number above this threshold, it will be split, otherwise it
        becomes a leaf node.

    min_impurity_split: float, default=1e-7
        The minimum impurity required to split a node. If a node's impurity is
        above this threshold, it will be split, otherwise it becomes a leaf node.

    max_features: int, default=None
        The maximum threshold value of the qualified feature number in the
        training dataset when training each fuzzy decision tree.

    multi_process_options: MultiProcessOptions, default=None
        Protocol message class that encapsulates all the options of the
        multi-process settings.

    Attributes
    ----------
    _estimators: ndarray of FuzzyDecisionTreeClassification
        The collection of sub-estimators as base learners.

    _res_func: function, default=None
        In classification, get the final result from the classes given by the
        forest by majority voting method. In regression, calculate the average
        of the predicted values given by the forest as the final result.
    """

    def __init__(self, disable_fuzzy, fuzzification_options, criterion_func, n_estimators=100,
                 max_depth=3, min_samples_split=2, min_impurity_split=1e-7, max_features=None,
                 multi_process_options=None):
        super().__init__(disable_fuzzy=disable_fuzzy,
                         fuzzification_options=fuzzification_options,
                         criterion_func=criterion_func,
                         n_estimators=n_estimators,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_impurity_split=min_impurity_split,
                         max_features=max_features,
                         multi_process_options=multi_process_options)

        # Initialise the forest.
        for _ in range(self.n_estimators):
            estimator = FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier,
                                                 disable_fuzzy=disable_fuzzy,
                                                 fuzzification_options=fuzzification_options,
                                                 criterion_func=criterion_func,
                                                 max_depth=max_depth,
                                                 min_samples_split=min_samples_split,
                                                 min_impurity_split=min_impurity_split)
            self._estimators.append(estimator)

        # Specify to get the final classification result by majority voting method.
        self._res_func = majority_vote

    def fit(self, X_train, y_train):
        # Do some custom things.

        super().fit(X_train=X_train, y_train=y_train)


class FuzzyRDFRegressor(FuzzyRDF):
    """
    Fuzzy random decision forests regressor.

    NB: For regression tasks, the mean or average prediction of
    the individual trees is returned.

    Parameters:
    -----------
    disable_fuzzy: bool, default=False
        Set whether the specified fuzzy decision tree uses the fuzzification.
        If disable_fuzzy=True, the specified fuzzy decision tree is equivalent
        to a naive decision tree.

    fuzzification_options: FuzzificationOptions, default=None
        Protocol message class that encapsulates all the options of the
        fuzzification settings used by the specified fuzzy decision tree.

    criterion_func: {"mse", "mae"}, default="mse"
        The criterion function used by the function that calculates the impurity
        gain of the target values.
        NB: Only use a criterion function for decision tree regressor.

    n_estimators: int, default=100
        The number of fuzzy decision trees to be used.

    max_depth: int, default=3
        The maximum depth of the tree to be trained.

    min_samples_split: int, default=2
        The minimum number of samples required to split a node. If a node has a
        sample number above this threshold, it will be split, otherwise it
        becomes a leaf node.

    min_impurity_split: float, default=1e-7
        The minimum impurity required to split a node. If a node's impurity is
        above this threshold, it will be split, otherwise it becomes a leaf node.

    max_features: int, default=None
        The maximum threshold value of the qualified feature number in the
        training dataset when training each fuzzy decision tree.

    multi_process_options: MultiProcessOptions, default=None
        Protocol message class that encapsulates all the options of the
        multi-process settings.

    Attributes
    ----------
    _estimators: ndarray of FuzzyDecisionTreeRegressor
        The collection of sub-estimators as base learners.

    _res_func: function, default=None
        In classification, get the final result from the classes given by the
        forest by majority voting method. In regression, calculate the average
        of the predicted values given by the forest as the final result.
    """

    def __init__(self, disable_fuzzy, fuzzification_options, criterion_func, n_estimators=100,
                 max_depth=3, min_samples_split=2, min_impurity_split=1e-7, max_features=None,
                 multi_process_options=None):
        super().__init__(disable_fuzzy=disable_fuzzy,
                         fuzzification_options=fuzzification_options,
                         criterion_func=criterion_func,
                         n_estimators=n_estimators,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_impurity_split=min_impurity_split,
                         max_features=max_features,
                         multi_process_options=multi_process_options)

        # Initialise forest.
        for _ in range(self.n_estimators):
            estimator = FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTRegressor,
                                                 disable_fuzzy=disable_fuzzy,
                                                 fuzzification_options=fuzzification_options,
                                                 criterion_func=criterion_func,
                                                 max_depth=max_depth,
                                                 min_samples_split=min_samples_split,
                                                 min_impurity_split=min_impurity_split)
            self._estimators.append(estimator)

        # Specify to get the final regression result by averaging method.
        self._res_func = mean_value

    def fit(self, X_train, y_train):
        # Do some custom things.

        super().fit(X_train=X_train, y_train=y_train)
