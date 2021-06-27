"""
@author: Zhaoqing Liu
@email : Zhaoqing.Liu-1@student.uts.edu.au
@date  : 24/6/21 6:28 pm
@desc  :
"""
import time

import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from fuzzytrees.fdt_base import CRITERIA_FUNC_CLF, FuzzificationOptions, MultiProcessOptions
from fuzzytrees.frdf import FuzzyRDFClassifier
from fuzzytrees.util_data_processing_funcs import extract_fuzzy_features


def main():
    # 1. Load the dataset.
    time_start = time.time()
    ds = datasets.load_digits()
    X = ds.data
    y = ds.target
    print("Time elapsed to load data: {:.5}s".format(time.time() - time_start))

    # 2. Preprocess the dataset.
    # 2.1. Do fuzzification preprocessing.
    time_start = time.time()
    X_fuzzy_pre = X.copy()
    fuzzification_options = FuzzificationOptions(conv_k=5)
    X_dms = extract_fuzzy_features(X_fuzzy_pre, conv_k=fuzzification_options.conv_k)
    X_plus_dms = np.concatenate((X, X_dms), axis=1)
    print("Time elapsed to preprocess fuzzification: {:.5}s".format(time.time() - time_start))

    # 2.2. Do your other data preprocessing,
    # e.g., identifying the feature values and target values, processing the missing values, etc.

    # 3. Partition the dataset.
    time_start = time.time()
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_plus_dms, y, test_size=0.4, random_state=22)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=22)
    print("Time elapsed to partition data: {:.5}s".format(time.time() - time_start))

    # 4. Train the models.
    # 4.1. Using a fuzzy classifier (You can customise the arguments in your constructor and their default values).
    time_start = time.time()
    fclf = FuzzyRDFClassifier(disable_fuzzy=False,
                              fuzzification_options=fuzzification_options,
                              criterion_func=CRITERIA_FUNC_CLF["gini"],
                              n_estimators=1000,
                              max_depth=5,
                              # max_features=int(X_train.shape[1] / 3),
                              multi_process_options=MultiProcessOptions())
    fclf.fit(X_train_f, y_train)
    print("Time elapsed to train a fuzzy classifier: {:.5}s".format(time.time() - time_start))

    # 5. Look at the models.
    # 5.1. Look at the fuzzy model.

    print("========================================================================================")
    # 6. Evaluate the models.
    # 6.1. Evaluate the fuzzy model.
    time_start = time.time()
    y_pred_f = fclf.predict(X_test_f)
    acc_f = accuracy_score(y_test, y_pred_f)
    print("Fuzzy model's accuracy is:", acc_f)
    print("Time elapsed to predict by the fuzzy classifier: {:.5}s".format(time.time() - time_start))

    # 6.3. Do your other evaluations.
    print("========================================================================================")


if __name__ == '__main__':
    main()
