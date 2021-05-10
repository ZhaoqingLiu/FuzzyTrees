"""
@author: Zhaoqing Liu
@email : Zhaoqing.Liu-1@student.uts.edu.au
@date  : 10/5/21 11:08 pm
@desc  :
"""
import multiprocessing
from enum import Enum


# Gets the maximum number of CPU cores available for the current cluster.
# For example, the maximum number of available CPU cores per Mars cluster is 16 for UTS,
# 30 for each Laureate cluster, 26 for each Mercury cluster, and 8 for each Venus cluster.
NUM_CPU_CORES_AVAL = multiprocessing.cpu_count()
# NUM_CPU_CORES_REQ = int(NUM_CPU_CORES_AVAL * 1 / 10)
NUM_CPU_CORES_REQ = NUM_CPU_CORES_AVAL


# Model evaluation under different fuzzy thresholds.
FUZZY_LIM = 0.5
FUZZY_STRIDE = 0.01

#
class EvaluationType(Enum):
    FUZZY_TH_VS_ACC = "fuzzy_th_vs_acc"
    MDL_CXTY_VS_ACC = "mdl_cxty_vs_acc"
    FUZZY_TH_VS_ERR = "fuzzy_th_vs_err"
    MDL_CXTY_VS_ERR = "mdl_cxty_vs_err"


# Output all of the above preset experiment configuration information before the experiment starts.
# =================================================================================================
for _ in range(80):
    print("=", end="")
print("")

print("{:^80}".format("Environment Configuration Information"))

print("Number of CPU cores available:")
print("{:>80}".format(NUM_CPU_CORES_AVAL))
print("Number of CPU cores currently requested:")
print("{:>80}".format(NUM_CPU_CORES_REQ))

print("Comparison experiments include:")
for k, name in EvaluationType.__members__.items():
    print("{:>80}".format(k + " -- " + name.value))

for _ in range(80):
    print("=", end="")
print("")
# =================================================================================================