# FuzzyTrees ![MIT](https://img.shields.io/badge/license-MIT-brightgreen)
FuzzyTrees is a lightweight framework designed for the rapid development of fuzzy decision tree algorithms.
FuzzyTrees offers a variety of benefits including:

- **Support in development solutions**: FuzzyTrees allows the user to extend new components quickly, according to particular fuzzy decision tree requirements, and build complete algorithm solutions.

- **Extending components with a set of APIs**: Any algorithm can be easily understood by following FuzzyTrees’ uniform APIs. To extend new components with ease, FuzzyTrees provides a set of supporting and easy-to-use utilities, e.g. the splitting and splitting criterion calculation functions available in the most widely used decision tree algorithms, CART, ID3, and C4.5.

- **Examples for algorithm development**: The FuzzyTrees algorithms, [fuzzy CART](fuzzytrees/fdt_base.py), [fuzzy GBDT](fuzzytrees/fgbdt.py) and fuzzy RDF[fuzzy RDF](fuzzytrees/frdf.py) can be used as examples for developing new algorithms or for conducting a variety of empirical studies.

## Installation
###  Getting it
```shell
$ pip install fdts
```

###  Importing all dependencies
Go to the root directory where FuzzyTrees is installed, and then
```shell
$ pip install -r requirements.txt
```


## Development example
For example, I'm implementing a fuzzy CART classifier.

```python
from fuzzytrees.fdt_base import BaseFuzzyDecisionTree, DecisionTreeInterface, CRITERIA_FUNC_CLF
from fuzzytrees.util_tree_criterion_funcs import calculate_impurity_gain, calculate_value_by_majority_vote
from fuzzytrees.util_tree_split_funcs import split_ds_2_bin


class FuzzyCARTClassifier(BaseFuzzyDecisionTree, DecisionTreeInterface):

    def __init__(self, disable_fuzzy=False, X_fuzzy_dms=None, fuzzification_options=None,
                 criterion_func=CRITERIA_FUNC_CLF["entropy"], max_depth=float("inf"), min_samples_split=2,
                 min_impurity_split=1e-7, **kwargs):
        super().__init__(disable_fuzzy=disable_fuzzy, X_fuzzy_dms=X_fuzzy_dms,
                         fuzzification_options=fuzzification_options, criterion_func=criterion_func,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split, min_impurity_split=min_impurity_split, **kwargs)
        # Specify the function used to split the dataset at each node.
        self._split_ds_func = split_ds_2_bin
        # Specify the function used to calculate the criteria against which each split point is selected during induction.
        self._impurity_gain_calc_func = calculate_impurity_gain
        # Specify the function used to calculate the value of each leaf node.
        self._leaf_value_calc_func = calculate_value_by_majority_vote

    # NB: The functions fit(), predict(), predict_proba() and print_tree() are already defined in the super class BaseFuzzyDecisionTree.
```

See the [tutorials](./tutorials.md) for more details on developing based on FuzzyTrees.


## Usage example
Let's take machine learning using the fuzzy CART classifier as an example.

```python
from fuzzytrees.fdt_base import FuzzificationOptions, FuzzyDecisionTreeWrapper, CRITERIA_FUNC_CLF
from fuzzytrees.fdts import FuzzyCARTClassifier
from fuzzytrees.util_preprocessing_funcs import extract_fuzzy_features
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

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
```

See the [tutorials](./tutorials.md) for more details on using fuzzy decision trees.


## Documentation & Resources
- [API Reference](./docs/index.html)
- [Tutorials](./tutorials.md)


## Credits
- Zhaoqing Liu (FuzzyTrees framework, [fuzzy CART](fuzzytrees/fdt_base.py), [fuzzy GBDT](fuzzytrees/fgbdt.py), [fuzzy RDF](fuzzytrees/frdf.py))
- Dr. Anjin Liu ([Fuzzy feature extraction](fuzzytrees/util_preprocessing_funcs.py) for fuzzification in preprocessing)
- Dist. Prof. Jie Lu
- A/Prof. Guangquan Zhang


## License
MIT License

Copyright (c) 2021 Australian Artificial Intelligence Institute, UTS

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


Contact us: Zhaoqing.Liu-1@student.uts.edu.au


