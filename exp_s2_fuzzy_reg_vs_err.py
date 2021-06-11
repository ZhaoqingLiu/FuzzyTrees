"""
@author: Zhaoqing Liu
@email : Zhaoqing.Liu-1@student.uts.edu.au
@date  : 21/4/21 11:29 am
@desc  :
"""
import os
import time
from fuzzytrees.fdt_base import FuzzyDecisionTreeWrapper, FuzzificationParams, CRITERIA_FUNC_CLF
from fuzzytrees.fdts import FuzzyCARTClassifier


if __name__ == '__main__':
    print("Main Process (%s) started." % os.getpid())
    # Record the start time used to calculate the time spent running one experiment.
    time_start = time.time()

    # Specify the names of all the datasets on which the model is being trained.
    ds_name_list = ["Vehicle", "German_Credit", "Diabetes", "Iris", "Wine"]
    # ds_name_list = ["Iris"]

    # Create a FDT proxy, and do the pretraining via it.
    clf = FuzzyDecisionTreeWrapper(fdt_class=FuzzyCARTClassifier, disable_fuzzy=False,
                                   fuzzification_params=FuzzificationParams(),
                                   criterion_func=CRITERIA_FUNC_CLF["gini"], max_depth=10)
    clf.search_fuzzy_params_4_clf(ds_name_list=ds_name_list, conv_k_lim=(2, 10, 1), fuzzy_reg_lim=(0.0, 1.0, 0.01))

    # Show the fuzzy regulation coefficient versus training error and test error by the FDT proxy.
    clf.plot_fuzzy_reg_vs_err()

    print("Total elapsed time: {:.5}s".format(time.time() - time_start))
    print("Main Process (%s) ended." % os.getpid())

    """
    1st exp on: 
        1. Configuration:
            ds_name_list = ["Vehicle", "German_Credit", "Diabetes", "Iris", "Wine"]
            conv_k_lim=(2, 10, 1)
            fuzzy_reg_lim=(0.0, 1.0, 0.01)
        2. Hyper-parameters of tree:
            max_depth = 5
        3. exp results:
            [4545 rows x 7 columns] (909 items/ds)
            Total elapsed time: 8046.6s
    
    2nd exp on:
        1. Configuration:
            (Same configuration as that of 1st exp.)
        2.Hyper-parameters of tree:
            max_depth=10
        3. exp results:
            
    """


