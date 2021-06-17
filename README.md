# FuzzyTrees

FuzzyTrees is a framework designed for rapidly developing various fuzzy decision tree algorithms.

First, the framework is a supporting architecture for development. Based on this framework, any developer can extend more components according to a particular fuzzy decision tree to quickly build a complete algorithm scheme.

Second, the framework provides protocols for extending components. You can follow a unified set of APIs to develop algorithms that are easy for other developers to understand.
To easily extend the components, the framework has provided you with a set of supporting and easy-to-use utilities, such as the split metric calculation and split method tools used in ID3, C4.5, and CART algorithms, respectively.

Also, the [fuzzy CART](fuzzytrees/fdt_base.py) and [fuzzy GBDT](fuzzytrees/fgbdt.py) algorithm in the project are implemented based on this framework.


## Installation
###  Getting it
```shell
$ pip install fuzzytrees
```

###  Importing all dependencies
```shell
$ pip install -r requirements.txt
```


## Development example
What you need to do is two steps. First, customise your fuzzy-rule functions (if needed), i.e. the splitting functions, splitting criterion calculation functions, and leaf node value calculation functions. 
Then, customise your fuzzy decision tree algorithm classes. That's it.

### Step 1: Customise your fuzzy-rule functions
Make sure your fuzzy rule functions follow the [API Reference](./docs/index.html).

### Step 2: Custom your fuzzy decision tree algorithm classes
All you need to do is first specify your custom fuzzy-rule functions at the beginning of the function fit().

Taking the classifier class of CART algorithm as an example.
```python
from fuzzytrees.fdt_base import BaseFuzzyDecisionTree, DecisionTreeInterface
from fuzzytrees.util_criterion_funcs import calculate_impurity_gain, calculate_value_by_majority_vote
from fuzzytrees.util_split_funcs import split_ds_2_bin

# Define a classifier.
class FuzzyCARTClassifier(BaseFuzzyDecisionTree, DecisionTreeInterface):
    
    # Add your __init__() here and declare its arguments.

    def fit(self, X_train, y_train):
        self._split_ds_func = split_ds_2_bin
        self._impurity_gain_calc_func = calculate_impurity_gain
        self._leaf_value_calc_func = calculate_value_by_majority_vote
        super().fit(X_train=X_train, y_train=y_train)
        
    # NB: The functions predict(), predict_proba() and print_tree() are already defined in the super class BaseFuzzyDecisionTree.
```

Taking the regressor class of CART algorithm as an example:
```python
from fuzzytrees.fdt_base import BaseFuzzyDecisionTree, DecisionTreeInterface
from fuzzytrees.util_criterion_funcs import calculate_variance_reduction, calculate_mean
from fuzzytrees.util_split_funcs import split_ds_2_bin

# Define a regressor.
class FuzzyCARTRegressor(BaseFuzzyDecisionTree, DecisionTreeInterface):
    
    # Add your __init__() here and declare its arguments.
    
    def fit(self, X_train, y_train):
        self._split_ds_func = split_ds_2_bin
        self._impurity_gain_calc_func = calculate_variance_reduction
        self._leaf_value_calc_func = calculate_mean
        super().fit(X_train=X_train, y_train=y_train)
        
    # NB: The functions predict(), predict_proba() and print_tree() are already defined in the super class BaseFuzzyDecisionTree.
```


## Usage example
Here are two examples of following a normal machine learning process, i.e. getting data, preprocessing the data, partitioning the data set, do the machine learning, and evaluating the trained model.

Taking the classifier class of CART algorithm as an example.
```python
from fuzzytrees.fdt_base import FuzzificationParams, FuzzyDecisionTreeWrapper, CRITERIA_FUNC_CLF
from fuzzytrees.fdts import FuzzyCARTClassifier
from fuzzytrees.util_data_processing_funcs import extract_fuzzy_features
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Getting data.
data = datasets.load_iris(as_frame=True).frame
X = data.iloc[:, :-1].values
y = data.iloc[:, -1:].values

# 2. Preprocessing the data.
# 2.1. Do fuzzification preprocessing.
X_fuzzy_pre = X.copy()
X_dms = extract_fuzzy_features(X_fuzzy_pre, conv_k=5)
X_plus_dms = np.concatenate((X, X_dms), axis=1)
fuzzification_params = FuzzificationParams(conv_k=5)

# 2.2. Do your other data preprocessing, e.g. identifying feature values and target values, processing missing values, etc.

# 3. Partitioning the data.
kf = KFold(n_splits=10, random_state=i, shuffle=True)
for train_index, test_index in kf.split(X):
    y_train, y_test = y[train_index], y[test_index]
    X_train_f, X_test_f = X_plus_dms[train_index], X_plus_dms[test_index]
    X_train, X_test = X[train_index], X[test_index]
    
    # 4. Do the machine learning.
    # 4.1. Using a fuzzy classifier (You can customise the arguments in your constructor and their default values).
    fclf = FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier, disable_fuzzy=False, 
                                    fuzzification_params=fuzzification_params,
                                    criterion_func=CRITERIA_FUNC_CLF["gini"], max_depth=5)
    fclf.fit(X_train_f, y_train)
    fclf.print_tree()
    
    # 4.2. Using a non-fuzzy classifier (You can customise the arguments in your constructor and their default values).
    clf = FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier, disable_fuzzy=True,
                                   criterion_func=CRITERIA_FUNC_CLF["gini"], max_depth=5)
    clf.fit(X_train, y_train)
    clf.print_tree()
    
    # 5. Evaluate the trained model.
    # 5.1. Evaluate the fuzzy model.
    y_pred_f = fclf.predict(X_test_f)
    acc_f = accuracy_score(y_test, y_pred_f)
    
    # 5.2. Evaluate the non-fuzzy model.
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # 5.3. Do your other evaluations.
```

Taking the regressor class of CART algorithm as an example.
```python
from fuzzytrees.fdt_base import FuzzificationParams, FuzzyDecisionTreeWrapper, CRITERIA_FUNC_REG
from fuzzytrees.fdts import FuzzyCARTRegressor
from fuzzytrees.util_data_processing_funcs import extract_fuzzy_features
from fuzzytrees.util_criterion_funcs import calculate_mse, calculate_mae
from sklearn import datasets
from sklearn.model_selection import KFold
import numpy as np

# 1. Getting data.
X, y = datasets.load_boston(return_X_y=True)

# 2. Preprocessing the data.
# 2.1. Do fuzzification preprocessing.
X_fuzzy_pre = X.copy()
X_dms = extract_fuzzy_features(X_fuzzy_pre, conv_k=5)
X_plus_dms = np.concatenate((X, X_dms), axis=1)
fuzzification_params = FuzzificationParams(conv_k=5)

# 2.2. Do your other data preprocessing, e.g. identifying feature values and target values, processing missing values, etc.

# 3. Partitioning the data.
kf = KFold(n_splits=10, random_state=i, shuffle=True)
for train_index, test_index in kf.split(X):
    y_train, y_test = y[train_index], y[test_index]
    X_train_f, X_test_f = X_plus_dms[train_index], X_plus_dms[test_index]
    X_train, X_test = X[train_index], X[test_index]
    
    # 4. Do the machine learning.
    # 4.1. Using a fuzzy regressor (You can customise the arguments in your constructor and their default values).
    freg = FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTRegressor, disable_fuzzy=False,
                                    fuzzification_params=fuzzification_params,
                                    criterion_func=CRITERIA_FUNC_REG["mse"], max_depth=5)
    freg.fit(X_train_f, y_train)
    freg.print_tree()
    
    # 4.2. Using a non-fuzzy regressor (You can customise the arguments in your constructor and their default values).
    reg = FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTRegressor, disable_fuzzy=True,
                                   criterion_func=CRITERIA_FUNC_REG["mse"], max_depth=5)
    reg.fit(X_train, y_train)
    reg.print_tree()
    
    # 5. Evaluate the trained model.
    # 5.1. Evaluate the fuzzy model.
    y_pred_f = freg.predict(X_test_f)
    mse_f = calculate_mse(y_test, y_pred_f)
    mae_f = calculate_mae(y_test, y_pred_f)
    
    # 5.2. Evaluate the non-fuzzy model.
    y_pred = reg.predict(X_test)
    mse = calculate_mse(y_test, y_pred)
    mae = calculate_mae(y_test, y_pred)
    
    # 5.3. Do your other evaluations.
```
Done.

## Documentation & Resources
- [API Reference](./docs/index.html)
- [Tutorials]()


## Credits
Fuzzy Trees was developed by:
- Zhaoqing Liu (FuzzyTrees framework, [fuzzy CART](./fuzzytrees/fdt_base.py), [fuzzy ID3](./fuzzytrees/fdt_base.py), [fuzzy C4.5](./fuzzytrees/fdt_base.py), [fuzzy GBDT](./fuzzytrees/fgbdt.py))
- Anjin Liu ([Fuzzy c-mean algorithm](./fuzzytrees/util_data_processing_funcs.py) for fuzzification in preprocessing)


License
----

MIT License

Copyright (c) 2021 Zhaoqing Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Contact us: Geo.Liu@outlook.com


