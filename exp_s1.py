"""
@author: Zhaoqing Liu
@email : Zhaoqing.Liu-1@student.uts.edu.au

Notes
-----
There are multiple comparison modes in the experiments and on multiple datasets.

The fuzzification of the fuzzy CART proposed in this paper includes two steps.
One is the fuzzification of the features in the data preprocessing stage
and the other is the fuzzification of the split criteria calculation in the model training stage.
Therefore, in the first two experiments, the first is to verify the impact of
the first step of fuzzification on the performance of a CART classifier,
while the second is to verify the impact of the combination of the two steps of fuzzification.
Also, the third experiment is to see what interesting changes would be made to
the performance of a GBDT classifier with all steps of fuzzification.

1. The first experiment includes:

1.1. The comparison of two non-fuzzy CART classifiers fitted on fuzzy (conv_k=3)
and non-fuzzy training data respectively.

1.2. The comparison of two non-fuzzy CART classifiers fitted on fuzzy (conv_k=4)
and non-fuzzy training data respectively.

1.3. The comparison of two non-fuzzy CART classifiers fitted on fuzzy (conv_k=5)
and non-fuzzy training data respectively.

2. The second experiment is to compare a fuzzy CART classifier with a non-fuzzy one.

3. The third experiment is to compare a fuzzy GBDT classifier with a non-fuzzy one.

Also, see 'util_data_handler.py' for details on the datasets used in the experiments.
"""