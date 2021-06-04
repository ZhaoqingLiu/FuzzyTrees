# FuzzyTrees

FuzzyTrees is a framework designed for rapidly developing various fuzzy decision tree algorithms.

First, the framework is a supporting architecture for development. Based on this framework, any developer can extend more components according to a particular fuzzy decision tree to quickly build a complete algorithm scheme.

Second, the framework provides protocols for extending components. You can follow a unified set of APIs to develop algorithms that are easy for other developers to understand.
To easily extend the components, the framework has provided you with a set of supporting and easy-to-use utilities, such as the split metric calculation and split method tools used in ID3, C4.5, and CART algorithms, respectively.

Also, the [fuzzy CART](fuzzytrees/fuzzy_cart.py) and [fuzzy GBDT](fuzzytrees/fuzzy_gbdt.py) algorithm in the project are implemented based on this framework.

Fuzzytrees是一个框架，它为快速开发各种模糊决策树算法而设计。

首先，该框架是一个用于开发的支撑架构。在该框架的基础上，任何开发者都可以以某个特定的模糊决策树为目标去扩展更多的组成部分，从而迅速地构建一个完整的算法方案。

其次，该框架提供扩展组件的协议。你可遵循一组统一的应用程序接口开发出易于其他开发者理解的算法。为了方便地扩展组件，该框架已经给你提供了一组辅助性、支撑性的方便易用的实用工具，例如分别在ID3, C4.5, 和CART算法中使用的分裂指标计算和分裂方法的工具。

此外，项目中的模糊CART和模糊GBDT算法即是在该框架基础上实现的。


## Usage example for development

###  Getting it
```shell
$ pip install fuzzytrees
```

###  Importing its dependencies
```shell
$ pip install -r requirements.txt
```

### Using it
```python
from fuzzytrees.fuzzy_decision_tree_wrapper import DecisionTreeInterface, CRITERIA_FUNC_CLF, CRITERIA_FUNC_REG, Node, SplitRule, BinarySubtrees
from fuzzytrees.util_criterion_funcs import calculate_impurity_gain, calculate_value_by_majority_vote, calculate_variance_reduction, calculate_mean, calculate_proba, calculate_impurity_gain_ratio
from fuzzytrees.util_split_funcs import split_ds_2_bin, split_ds_2_multi, split_disc_ds_2_multi
```

### Custom your algorithm class
Taking the classifier class of CART algorithm as an example:
```python
# Define a classifier.
class FuzzyCARTClassifier(DecisionTreeInterface):
    
    # Add your __init__() here and declare its arguments.

    def fit(self, X_train, y_train):
        self._split_ds_func = split_ds_2_bin
        self._impurity_gain_calc_func = calculate_impurity_gain
        self._leaf_value_calc_func = calculate_value_by_majority_vote
        
        # Add your code for fitting a tree below.

    def predict(self, X):
        # Add your code for predicting a set of samples below.

    def print_tree(self, tree=None, indent="  ", delimiter="=>"):
        # Add your code for printing the fitted tree below.

# Use the classifier (You can customise the arguments in your constructor and their default values).
clf = FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier, disable_fuzzy=False,
                               criterion_func=CRITERIA_FUNC_CLF["gini"], max_depth=5)

# Fit a tree.
clf.fit(X_train, y_train)

# Printing the fitted tree.
clf.print_tree()

# Predict a set of samples.
y_pred = clf.predict(X_test)
```

Taking the regressor class of CART algorithm as an example:
```python
# Define a regressor.
class FuzzyCARTRegressor(DecisionTreeInterface):
    
    # Add your __init__() here and declare its arguments.
    
    def fit(self, X_train, y_train):
        self._split_ds_func = split_ds_2_bin
        self._impurity_gain_calc_func = calculate_variance_reduction
        self._leaf_value_calc_func = calculate_mean
        
        # Add your code for fitting a tree below.

    def predict(self, X):
        # Add your code for predicting a set of samples below.

    def print_tree(self, tree=None, indent="  ", delimiter="=>"):
        # Add your code for printing the fitted tree below.

# Use the regressor (You can customise the arguments in your constructor and their default values).
reg = FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTRegressor, disable_fuzzy=False,
                               criterion_func=CRITERIA_FUNC_REG["mse"], max_depth=5)

# Fit a tree.
reg.fit(X_train, y_train)

# Printing the fitted tree.
reg.print_tree()

# Predict a set of samples.
y_pred = reg.predict(X_test)
```



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


