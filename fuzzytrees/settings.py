"""
@author : Zhaoqing Liu
@email  : Zhaoqing.Liu-1@student.uts.edu.au
"""
import logging
import multiprocessing
import os
from enum import Enum


# =============================================================================
# Stage 1 Experiments
# =============================================================================

# Use Enum so that the values are not allowed to be edited and to increase the readability of the code.
class ComparisonMode(Enum):
    # NAIVE = "my_naive_vs_sklearn_naive"
    FF3 = "f3_ds_vs_orig_ds"  # With only Feature Fuzzification, n_conv=3
    FF4 = "f4_ds_vs_orig_ds"  # With only Feature Fuzzification, n_conv=4
    FF5 = "f5_ds_vs_orig_ds"  # With only Feature Fuzzification, n_conv=5
    FUZZY = "fdt_vs_nfdt"
    BOOSTING = "fgbdt_vs_nfgbdt"
    # MIXED = "mfgbdt_vs_nfgbdt"


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
    FUZZY_REG_VS_ERR_ON_N_CONV = "fuzzy_reg_vs_err_on_n_conv"


# File paths to save evaluation data, graphs, serialised models.
class DirSave(Enum):
    EVAL_DATA = os.getcwd() + "/fuzzy_trees_v001/data_gen/eval_data/"
    EVAL_FIGURES = os.getcwd() + "/fuzzy_trees_v001/data_gen/eval_figures/"
    MODELS = os.getcwd() + "/fuzzy_trees_v001/data_gen/pkl_models/"


# Number of a group of models when pretraining.
NUM_GRP_MDLS = 10


# Output all of the above preset experiment configuration information before the experiment starts.
# =================================================================================================
print("=" * 100)

print("{:^100}".format("Environment Configuration Information"))

print("Number of CPU cores available:")
print("{:>100}".format(NUM_CPU_CORES_AVAL))

print("Number of CPU cores requested:")
print("{:>100}".format(NUM_CPU_CORES_REQ))

print("(EXP) Comparison experiments:")
for name, item in ComparisonMode.__members__.items():
    print("{:>100}".format(name + " -- " + item.value))

print("(EXP) Search optimal parameters:")
for name, item in EvaluationType.__members__.items():
    print("{:>100}".format(name + " -- " + item.value))

print("Current path:")
print("{:>100}".format(os.getcwd()))

print("Path to save generated files:")
for _, item in DirSave.__members__.items():
    print("{:>100}".format(item.value))

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
# DS_LOAD_FUNC_CLF = {"Vehicle": load_vehicle, "German_Credit": load_German_credit, "Diabetes": load_diabetes, "Iris": load_iris, "Wine": load_wine}
# DS_LOAD_FUNC_REG = {}

# Model evaluation under different fuzzy regulation coefficients.
FUZZY_LIM = 0.5
FUZZY_STRIDE = 0.01
