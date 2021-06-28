# FuzzyTrees Tutorials

## Installation
###  Getting it
```shell
$ pip install fuzzytrees
```

###  Importing all dependencies
Go to the root directory where FuzzyTrees is installed, and then
```shell
$ pip install -r requirements.txt
```


## Development
If you are developing a new fuzzy decision tree algorithm, 
all you need to do is specify the fuzzy rule-based functions in the constructor \_\_init\_\_() of your fuzzy decision tree class. 
See the utilities [util_data_processing_funcs](./fuzzytrees/util_data_processing_funcs.py), [util_split_funcs](./fuzzytrees/util_split_funcs.py) and [util_criterion_funcs](./fuzzytrees/util_criterion_funcs.py) for details on the fuzzy rules-based functions.

In addition to the utilities provided by FuzzyTrees, if you need to customise your own fuzzy-based functions, 
i.e. fuzzification preprocessing, splitting, splitting criterion calculation, and leaf node value calculation functions, 
you can follow the [API Reference](./docs/index.html) to implement them.

That's all.

### Development example
For example, I'm implementing a fuzzy CART classifier.

```python
from fuzzytrees.fdt_base import BaseFuzzyDecisionTree, DecisionTreeInterface, CRITERIA_FUNC_CLF
from fuzzytrees.util_criterion_funcs import calculate_impurity_gain, calculate_value_by_majority_vote
from fuzzytrees.util_split_funcs import split_ds_2_bin


class FuzzyCARTClassifier(BaseFuzzyDecisionTree, DecisionTreeInterface):

    def __init__(self, disable_fuzzy=False, X_fuzzy_dms=None, fuzzification_options=None,
                 criterion_func=CRITERIA_FUNC_CLF["entropy"], max_depth=float("inf"), min_samples_split=2,
                 min_impurity_split=1e-7, **kwargs):
        super().__init__(disable_fuzzy=disable_fuzzy, X_fuzzy_dms=X_fuzzy_dms,
                         fuzzification_options=fuzzification_options, criterion_func=criterion_func, max_depth=max_depth,
                         min_samples_split=min_samples_split, min_impurity_split=min_impurity_split, **kwargs)
        self._split_ds_func = split_ds_2_bin
        self._impurity_gain_calc_func = calculate_impurity_gain
        self._leaf_value_calc_func = calculate_value_by_majority_vote

    # NB: The functions fit(), predict(), predict_proba() and print_tree() are already defined in the super class BaseFuzzyDecisionTree.
```

Also, I'm developing a fuzzy CART regressor.

```python
from fuzzytrees.fdt_base import BaseFuzzyDecisionTree, DecisionTreeInterface, CRITERIA_FUNC_REG
from fuzzytrees.util_criterion_funcs import calculate_variance_reduction, calculate_mean_value
from fuzzytrees.util_split_funcs import split_ds_2_bin


class FuzzyCARTRegressor(BaseFuzzyDecisionTree, DecisionTreeInterface):

    def __init__(self, disable_fuzzy=False, X_fuzzy_dms=None, fuzzification_options=None,
                 criterion_func=CRITERIA_FUNC_REG["mse"], max_depth=float("inf"), min_samples_split=2,
                 min_impurity_split=1e-7, **kwargs):
        super().__init__(disable_fuzzy=disable_fuzzy, X_fuzzy_dms=X_fuzzy_dms,
                         fuzzification_options=fuzzification_options, criterion_func=criterion_func, max_depth=max_depth,
                         min_samples_split=min_samples_split, min_impurity_split=min_impurity_split, **kwargs)
        self._split_ds_func = split_ds_2_bin
        self._impurity_gain_calc_func = calculate_variance_reduction
        self._leaf_value_calc_func = calculate_mean_value

    # NB: The functions fit(), predict(), predict_proba() and print_tree() are already defined in the super class BaseFuzzyDecisionTree.
```


## Usage
Here are two usage examples following a normal machine learning process, including loading the dataset, preprocessing the dataset, partitioning the dataset, training the model, looking at the model, and evaluating the model.

### Usage example
Let's take machine learning using the fuzzy CART classifier as an example.

```python
from fuzzytrees.fdt_base import FuzzificationOptions, FuzzyDecisionTreeWrapper, CRITERIA_FUNC_CLF
from fuzzytrees.fdts import FuzzyCARTClassifier
from fuzzytrees.util_data_processing_funcs import extract_fuzzy_features
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Load the dataset.
data = datasets.load_wine(as_frame=True).frame
X = data.iloc[:, :-1].values
y = data.iloc[:, -1:].values

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
acc_f = accuracy_score(y_test, y_pred_f)
print("Fuzzy model's accuracy is:", acc_f)

# 6.2. Evaluate the non-fuzzy model.
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Non-fuzzy model's accuracy is:", acc)

# 6.3. Do your other evaluations.
print("========================================================================================")
```

Let's take machine learning using the fuzzy CART regressor as another example.

```python
from fuzzytrees.fdt_base import FuzzificationOptions, FuzzyDecisionTreeWrapper, CRITERIA_FUNC_REG
from fuzzytrees.fdts import FuzzyCARTRegressor
from fuzzytrees.util_data_processing_funcs import extract_fuzzy_features
from fuzzytrees.util_criterion_funcs import calculate_mse, calculate_mae
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

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
freg.fit(X_train_f, y_train)

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
mse_f = calculate_mse(y_test, y_pred_f)
mae_f = calculate_mae(y_test, y_pred_f)
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
```
Done.