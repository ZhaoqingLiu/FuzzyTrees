"""
@author: Zhaoqing Liu
@email : Zhaoqing.Liu-1@student.uts.edu.au
@date  : 21/4/21 11:58 am
@desc  :
"""
import multiprocessing
from enum import Enum

from fuzzy_trees.util_data_handler import load_vehicle, load_German_credit, load_diabetes, load_iris, load_wine


class ComparisionMode(Enum):
    NAIVE = "my_naive_vs_sklearn_naive"
    FF3 = "ff3_vs_naive"  # With only Feature Fuzzification, conv_k=3
    FF4 = "ff4_vs_naive"  # With only Feature Fuzzification, conv_k=4
    FF5 = "ff5_vs_naive"  # With only Feature Fuzzification, conv_k=5
    FUZZY = "fcart_vs_ccart"
    BOOSTING = "fgbdt_vs_nfgbdt"
    MIXED = "mfgbdt_vs_nfgbdt"


# Gets the maximum number of CPU cores available for the current cluster.
# For example, the maximum number of available CPU cores per Mars cluster is 16 for UTS,
# 30 for each Laureate cluster, 26 for each Mercury cluster, and 8 for each Venus cluster.
NUM_CPU_CORES = multiprocessing.cpu_count()


# The data sets on which you want to run experiments.
DS_LOAD_FUNC_CLF = {"Iris": load_iris}
# DS_LOAD_FUNC_CLF = {"Vehicle": load_vehicle, "German_Credit": load_German_credit, "Diabetes": load_diabetes, "Iris": load_iris, "Wine": load_wine}
DS_LOAD_FUNC_REG = {}


# Searching an optimum fuzzy threshold by a loop according the specified stride.
FUZZY_TOL = 0.0
FUZZY_STRIDE = 0.01


print("==================================================================")
print("                    Configuration Information                   ")

print("Comparison experiments include:")
for k, name in ComparisionMode.__members__.items():
    print("                               %s -- %s" % (k, name.value))

print("Number of CPU cores available on the current server:")
print("                               %s" % NUM_CPU_CORES)

print("Data sets to train on:")
for ds_name in DS_LOAD_FUNC_CLF.keys():
    print("                               %s" % (ds_name))
print("==================================================================")
