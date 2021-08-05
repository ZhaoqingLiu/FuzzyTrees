"""
@author: Zhaoqing Liu
@email : Zhaoqing.Liu-1@student.uts.edu.au
@desc  : Examples using the algorithms provided by the framework.
"""
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from fuzzytrees.fdt_base import FuzzificationOptions, FuzzyDecisionTreeWrapper, CRITERIA_FUNC_CLF, CRITERIA_FUNC_REG
from fuzzytrees.fdts import FuzzyCARTClassifier, FuzzyCARTRegressor
from fuzzytrees.util_preprocessing_funcs import extract_fuzzy_features
from fuzzytrees.util_tree_criterion_funcs import calculate_mse, calculate_mae


def use_fuzzy_cart_clf():
    """Example using the Fuzzy CART classifier."""
    # 1. Load the dataset.
    X, y = datasets.load_wine(return_X_y=True)

    # 2. Preprocess the dataset.
    # 2.1. Do fuzzification preprocessing.
    X_fuzzy_pre = X.copy()
    fuzzification_options = FuzzificationOptions(conv_k=5)
    X_dms = extract_fuzzy_features(X_fuzzy_pre, conv_k=fuzzification_options.conv_k)
    X_plus_dms = np.concatenate((X, X_dms), axis=1)

    # 2.2. Do your other data preprocessing, e.g. identifying the feature values and target values, processing the missing values, etc.

    # 3. Partition the dataset.
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_plus_dms, y, test_size=0.4, random_state=22)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=22)

    # 4. Train the models.
    # 4.1. Using a fuzzy classifier (You can customise the arguments in your constructor and their default values).
    fclf = FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier, disable_fuzzy=False,
                                    fuzzification_options=fuzzification_options,
                                    criterion_func=CRITERIA_FUNC_CLF["gini"], max_depth=5)
    fclf.fit(X_train_f, y_train_f)

    # 4.2. Using a non-fuzzy classifier (You can customise the arguments in your constructor and their default values).
    clf = FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier, disable_fuzzy=True,
                                   criterion_func=CRITERIA_FUNC_CLF["gini"], max_depth=5)
    clf.fit(X_train, y_train)

    # 5. Look at the models.
    # 5.1. Look at the fuzzy model.
    fclf.print_tree()

    # 5.2. Look at the non-fuzzy model.
    clf.print_tree()

    print("========================================================================================")
    # 6. Evaluate the models.
    # 6.1. Evaluate the fuzzy model.
    y_pred_f = fclf.predict(X_test_f)
    acc_f = accuracy_score(y_test_f, y_pred_f)
    print("Fuzzy model's accuracy is:", acc_f)

    # 6.2. Evaluate the non-fuzzy model.
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Non-fuzzy model's accuracy is:", acc)

    # 6.3. Do your other evaluations.
    print("========================================================================================")


def use_fuzzy_cart_reg():
    """Example using the Fuzzy CART regressor."""
    # 1. Load the dataset.
    X, y = datasets.load_diabetes(return_X_y=True)

    # 2. Preprocess the dataset.
    # 2.1. Do fuzzification preprocessing.
    X_fuzzy_pre = X.copy()
    fuzzification_options = FuzzificationOptions(conv_k=5)
    X_dms = extract_fuzzy_features(X_fuzzy_pre, conv_k=fuzzification_options.conv_k)
    X_plus_dms = np.concatenate((X, X_dms), axis=1)

    # 2.2. Do your other data preprocessing, e.g. identifying the feature values and target values, processing the missing values, etc.

    # 3. Partition the dataset.
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_plus_dms, y, test_size=0.4, random_state=22)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=22)

    # 4. Train the models.
    # 4.1. Using a fuzzy regressor (You can customise the arguments in your constructor and their default values).
    freg = FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTRegressor, disable_fuzzy=False,
                                    fuzzification_options=fuzzification_options,
                                    criterion_func=CRITERIA_FUNC_REG["mse"], max_depth=5)
    freg.fit(X_train_f, y_train_f)

    # 4.2. Using a non-fuzzy regressor (You can customise the arguments in your constructor and their default values).
    reg = FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTRegressor, disable_fuzzy=True,
                                   criterion_func=CRITERIA_FUNC_REG["mse"], max_depth=5)
    reg.fit(X_train, y_train)

    # 5. Look at the models.
    # 5.1. Look at the fuzzy model.
    freg.print_tree()

    # 5.2. Look at the non-fuzzy model.
    reg.print_tree()

    print("========================================================================================")
    # 6. Evaluate the models.
    # 6.1. Evaluate the fuzzy model.
    y_pred_f = freg.predict(X_test_f)
    mse_f = calculate_mse(y_test_f, y_pred_f)
    mae_f = calculate_mae(y_test_f, y_pred_f)
    print("Fuzzy model's average MSE is:", mse_f)
    print("Fuzzy model's average MAE is:", mae_f)

    # 6.2. Evaluate the non-fuzzy model.
    y_pred = reg.predict(X_test)
    mse = calculate_mse(y_test, y_pred)
    mae = calculate_mae(y_test, y_pred)
    print("Non-fuzzy model's average MSE is:", mse)
    print("Non-fuzzy model's average MAE is:", mae)

    # 6.3. Do your other evaluations.
    print("========================================================================================")


if __name__ == '__main__':
    use_fuzzy_cart_clf()
    use_fuzzy_cart_reg()
