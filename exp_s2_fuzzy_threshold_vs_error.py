"""
@author: Zhaoqing Liu
@email : Zhaoqing.Liu-1@student.uts.edu.au
@date  : 21/4/21 11:29 am
@desc  :
"""
import multiprocessing
import os
import time
from decimal import Decimal

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from fuzzy_trees.fuzzy_decision_tree import FuzzyDecisionTreeClassifier
from fuzzy_trees.fuzzy_decision_tree_api import FuzzificationParams, FuzzyDecisionTreeAPI, CRITERIA_FUNC_CLF, \
    CRITERIA_FUNC_REG
from fuzzy_trees.fuzzy_gbdt import FuzzyGBDTClassifier
from fuzzy_trees.util_data_processing_funcs import extract_fuzzy_features
import fuzzy_trees.util_plotter as plotter


if __name__ == '__main__':
    print("Main Process (%s) started." % os.getpid())
    time_start = time.time()

    # 1st step: Specify the names of all the datasets on which the model is being trained.
    dataset_name_list = ["Vehicle", "German_Credit", "Diabetes", "Iris", "Wine"]

    # 2nd step: Create a FDT agent, and then use it to pretrain.
    fuzzification_params = FuzzificationParams(conv_k=5)
    clf = FuzzyDecisionTreeAPI(fdt_class=FuzzyDecisionTreeClassifier, disable_fuzzy=False,
                               fuzzification_params=fuzzification_params,
                               criterion_func=CRITERIA_FUNC_CLF["gini"], max_depth=5)
    clf.pre_train(dataset_name_list=dataset_name_list)

    # 3rd step: Show the fuzzy threshold versus training error and test error by the FDT agent.
    clf.show_fuzzy_th_vs_err()

    print("Total elapsed time: {:.5}s".format(time.time() - time_start))
    print("Main Process (%s) ended." % os.getpid())
