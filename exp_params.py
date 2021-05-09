"""
@author: Zhaoqing Liu
@email : Zhaoqing.Liu-1@student.uts.edu.au
@date  : 21/4/21 11:58 am
@desc  :
"""
import multiprocessing
from enum import Enum

from fuzzy_trees.util_data_handler import load_vehicle, load_German_credit, load_diabetes, load_iris, load_wine


# Use Enum so that the values are not allowed to be edited and to increase the readability of the code.
class ComparisionMode(Enum):
    NAIVE = "my_naive_vs_sklearn_naive"
    FF3 = "ff3_vs_naive"  # With only Feature Fuzzification, conv_k=3
    FF4 = "ff4_vs_naive"  # With only Feature Fuzzification, conv_k=4
    FF5 = "ff5_vs_naive"  # With only Feature Fuzzification, conv_k=5
    FUZZY = "fcart_vs_ccart"
    BOOSTING = "fgbdt_vs_nfgbdt"
    MIXED = "mfgbdt_vs_nfgbdt"


class EvaluationMode(Enum):
    FUZZY_TH_VS_ACC = "fuzzy_th_vs_accuracy"
    MDL_CXTY_VS_ACC = "mdl_cxty_vs_accuracy"


# Gets the maximum number of CPU cores available for the current cluster.
# For example, the maximum number of available CPU cores per Mars cluster is 16 for UTS,
# 30 for each Laureate cluster, 26 for each Mercury cluster, and 8 for each Venus cluster.
NUM_CPU_CORES_AVAL = multiprocessing.cpu_count()
# NUM_CPU_CORES_REQ = int(NUM_CPU_CORES_AVAL * 1 / 10)
NUM_CPU_CORES_REQ = NUM_CPU_CORES_AVAL

# The data sets on which you want to run experiments.
# DS_LOAD_FUNC_CLF = {"Iris": load_iris}
# DS_LOAD_FUNC_CLF = {"Wine": load_wine}
# DS_LOAD_FUNC_CLF = {"Vehicle": load_vehicle, "German_Credit": load_German_credit, "Diabetes": load_diabetes}
DS_LOAD_FUNC_CLF = {"Vehicle": load_vehicle, "German_Credit": load_German_credit, "Diabetes": load_diabetes, "Iris": load_iris, "Wine": load_wine}
DS_LOAD_FUNC_REG = {}

# Searching an optimum fuzzy threshold by a loop according the specified stride.
FUZZY_STRIDE = 0.01


# Output all of the above preset experiment configuration information before the experiment starts.
# =================================================================================================
for _ in range(80):
    print("=", end="")
print("")

print("{:^80}".format("Experimental Configuration Information"))

print("Comparison experiments include:")
for k, name in ComparisionMode.__members__.items():
    print("{:>80}".format(k + " -- " + name.value))

print("Number of CPU cores available:")
print("{:>80}".format(NUM_CPU_CORES_AVAL))
print("Number of CPU cores currently requested:")
print("{:>80}".format(NUM_CPU_CORES_REQ))

print("Data sets for model training:")
for ds_name in DS_LOAD_FUNC_CLF.keys():
    print("{:>80}".format((ds_name)))

for _ in range(80):
    print("=", end="")
print("")
# =================================================================================================
