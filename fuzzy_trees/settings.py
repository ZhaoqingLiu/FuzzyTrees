"""
@author: Zhaoqing Liu
@email : Zhaoqing.Liu-1@student.uts.edu.au
@date  : 10/5/21 11:08 pm
@desc  :
"""
import multiprocessing
from enum import Enum
from fuzzy_trees.util_data_handler import *


# =============================================================================
# Stage 1 Experiments
# =============================================================================

# Use Enum so that the values are not allowed to be edited and to increase the readability of the code.
class ComparisionMode(Enum):
    NAIVE = "my_naive_vs_sklearn_naive"
    FF3 = "ff3_vs_naive"  # With only Feature Fuzzification, conv_k=3
    FF4 = "ff4_vs_naive"  # With only Feature Fuzzification, conv_k=4
    FF5 = "ff5_vs_naive"  # With only Feature Fuzzification, conv_k=5
    FUZZY = "fcart_vs_ccart"
    BOOSTING = "fgbdt_vs_nfgbdt"
    MIXED = "mfgbdt_vs_nfgbdt"


# =============================================================================
# Stage 2 Experiments
# =============================================================================

# Gets the maximum number of CPU cores available for the current cluster.
# For example, the maximum number of available CPU cores per Mars cluster is 16 for UTS,
# 30 for each Laureate cluster, 26 for each Mercury cluster, and 8 for each Venus cluster.
NUM_CPU_CORES_AVAL = multiprocessing.cpu_count()
# NUM_CPU_CORES_REQ = int(NUM_CPU_CORES_AVAL * 1 / 10)
NUM_CPU_CORES_REQ = NUM_CPU_CORES_AVAL


# Evaluation types corresponding plotting types.
class EvaluationType(Enum):
    FUZZY_REG_VS_ACC_ON_CONV_K = "fuzzy_reg_vs_acc_on_conv_k"


# File paths to save evaluation data, graphs, serialised models.
class DirSave(Enum):
    EVAL_DATA = "/data/zhaoqliu/my_ml_p/FuzzyDecisionTrees/fuzzy_trees/eval_data/"
    EVAL_FIGURES = "/data/zhaoqliu/my_ml_p/FuzzyDecisionTrees/fuzzy_trees/eval_figures/"
    MODELS = "./data/zhaoqliu/my_ml_p/FuzzyDecisionTrees/fuzzy_trees/pkl_models/"


# Number of a group of models when pretraining.
NUM_GRP_MDLS = 10


# Output all of the above preset experiment configuration information before the experiment starts.
# =================================================================================================
print("=" * 100)

print("{:^100}".format("Environment Configuration Information"))

print("Number of CPU cores available:")
print("{:>100}".format(NUM_CPU_CORES_AVAL))

print("Number of CPU cores currently requested:")
print("{:>100}".format(NUM_CPU_CORES_REQ))

print("(S1 EXP) Comparison experiments include:")
for k, name in ComparisionMode.__members__.items():
    print("{:>100}".format(k + " -- " + name.value))

print("(S2 EXP) experiments include:")
for k, name in EvaluationType.__members__.items():
    print("{:>100}".format(k + " -- " + name.value))

print("(S2 EXP) Path to save files:")
for k, name in DirSave.__members__.items():
    print("{:>100}".format(k + " -- " + name.value))

print("=" * 100)
# =================================================================================================


# =============================================================================
# For temporary use only
# =============================================================================

# Datasets (k: dataset name, v: function for getting data) on which the model is being trained.
# DS_LOAD_FUNC_CLF = {"Iris": load_iris}
# DS_LOAD_FUNC_CLF = {"Wine": load_wine}
# DS_LOAD_FUNC_CLF = {"Vehicle": load_vehicle}
# DS_LOAD_FUNC_CLF = {"German_Credit": load_German_credit}
# DS_LOAD_FUNC_CLF = {"Diabetes": load_diabetes}
DS_LOAD_FUNC_CLF = {"Vehicle": load_vehicle, "German_Credit": load_German_credit, "Diabetes": load_diabetes, "Iris": load_iris, "Wine": load_wine}
DS_LOAD_FUNC_REG = {}

# Model evaluation under different fuzzy regulation coefficients.
FUZZY_LIM = 0.5
FUZZY_STRIDE = 0.01
