# FuzzyTrees

FuzzyTrees is a lightweight framework designed for rapidly developing various fuzzy decision tree algorithms.
It has three features.

- Firstly, FuzzyTrees is a supporting architecture for development. 
Based on the FuzzyTrees, you can quickly extend new components according to a particular fuzzy decision tree requirements and build your own complete algorithm solutions.

- Secondly, FuzzyTrees provides a set of APIs for extending components. 
You can easily understand any algorithm as long as it follows these uniform APIs.
To easily extend new components, FuzzyTrees has provided you with a set of supporting and easy-to-use utilities, e.g. the splitting and splitting criterion calculation functions available in the most popular decision tree algorithms ID3, C4.5, and CART.

- Finally, the [fuzzy CART](fuzzytrees/fdt_base.py) and [fuzzy GBDT](fuzzytrees/fgbdt.py) algorithms in this project are developed based on FuzzyTrees and can be used as examples for developing new algorithms.


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
For example, I'm implementing a fuzzy CART classifier.
```python
from fuzzytrees.fdt_base import BaseFuzzyDecisionTree, DecisionTreeInterface, CRITERIA_FUNC_CLF
from fuzzytrees.util_criterion_funcs import calculate_impurity_gain, calculate_value_by_majority_vote
from fuzzytrees.util_split_funcs import split_ds_2_bin

class FuzzyCARTClassifier(BaseFuzzyDecisionTree, DecisionTreeInterface):
    
    def __init__(self, disable_fuzzy=False, X_fuzzy_dms=None, fuzzification_params=None,
                 criterion_func=CRITERIA_FUNC_CLF["entropy"], max_depth=float("inf"), min_samples_split=2,
                 min_impurity_split=1e-7, **kwargs):
        super().__init__(disable_fuzzy=disable_fuzzy, X_fuzzy_dms=X_fuzzy_dms,
                         fuzzification_params=fuzzification_params, criterion_func=criterion_func, max_depth=max_depth,
                         min_samples_split=min_samples_split, min_impurity_split=min_impurity_split, **kwargs)
        self._split_ds_func = split_ds_2_bin
        self._impurity_gain_calc_func = calculate_impurity_gain
        self._leaf_value_calc_func = calculate_value_by_majority_vote
        
    # NB: The functions fit(), predict(), predict_proba() and print_tree() are already defined in the super class BaseFuzzyDecisionTree.
```

See the [tutorials](./tutorials.md) for more details on developing based on FuzzyTrees.


## Usage example
Let's take machine learning using the fuzzy CART classifier as an example.
```python
from fuzzytrees.fdt_base import FuzzificationParams, FuzzyDecisionTreeWrapper, CRITERIA_FUNC_CLF
from fuzzytrees.fdts import FuzzyCARTClassifier
from fuzzytrees.util_data_processing_funcs import extract_fuzzy_features
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Load the dataset.
data = datasets.load_iris(as_frame=True).frame
X = data.iloc[:, :-1].values
y = data.iloc[:, -1:].values

# 2. Preprocess the dataset.
# 2.1. Do fuzzification preprocessing.
X_fuzzy_pre = X.copy()
X_dms = extract_fuzzy_features(X_fuzzy_pre, conv_k=5)
X_plus_dms = np.concatenate((X, X_dms), axis=1)
fuzzification_params = FuzzificationParams(conv_k=5)

# 2.2. Do your other data preprocessing, e.g. identifying the feature values and target values, processing the missing values, etc.

# 3. Partition the dataset.
kf = KFold(n_splits=10, random_state=i, shuffle=True)
for train_index, test_index in kf.split(X):
    y_train, y_test = y[train_index], y[test_index]
    X_train_f, X_test_f = X_plus_dms[train_index], X_plus_dms[test_index]
    X_train, X_test = X[train_index], X[test_index]
    
    # 4. Train the models.
    # 4.1. Using a fuzzy classifier (You can customise the arguments in your constructor and their default values).
    fclf = FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier, disable_fuzzy=False, 
                                    fuzzification_params=fuzzification_params,
                                    criterion_func=CRITERIA_FUNC_CLF["gini"], max_depth=5)
    fclf.fit(X_train_f, y_train)
    
    # 4.2. Using a non-fuzzy classifier (You can customise the arguments in your constructor and their default values).
    clf = FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier, disable_fuzzy=True,
                                   criterion_func=CRITERIA_FUNC_CLF["gini"], max_depth=5)
    clf.fit(X_train, y_train)
    
    # 5. Look at the models.
    # 5.1. Look at the fuzzy model.
    fclf.print_tree()
    
    # 5.2. Look at the non-fuzzy model.
    clf.print_tree()
    
    # 6. Evaluate the models.
    # 6.1. Evaluate the fuzzy model.
    y_pred_f = fclf.predict(X_test_f)
    acc_f = accuracy_score(y_test, y_pred_f)
    
    # 6.2. Evaluate the non-fuzzy model.
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # 6.3. Do your other evaluations.
```

See the [tutorials](./tutorials.md) for more details on using fuzzy decision trees.


## Documentation & Resources
- [API Reference](./docs/index.html)
- [Tutorials](./tutorials.md)


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


