"""
@author: Zhaoqing Liu
@email : Zhaoqing.Liu-1@student.uts.edu.au
@date  : 27/4/21 2:23 pm
@desc  :
"""
import matplotlib.pyplot as plt


def plot_correlation_curve(alphas_list, errors_list, label_list):
    plt.subplot(2, 1, 1)
    for alphas, errors, label in zip(alphas_list, errors_list, label_list):
        plt.semilogx(alphas, errors, label=label)

    plt.legend(loc="")
    plt.ylim((0, 1.2))
    plt.xlabel("Fuzzy threshold")
    plt.ylabel("Performance")
