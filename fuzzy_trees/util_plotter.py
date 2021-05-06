"""
@author: Zhaoqing Liu
@email : Zhaoqing.Liu-1@student.uts.edu.au
@date  : 27/4/21 2:23 pm
@desc  :
"""
import matplotlib.pyplot as plt
import numpy as np


COLOUR = ["b", "r", "g", "c", "y", "k", "m"]
# The colours above are equivalent to the colours below:
# COLOUR = ["blue", "red", "green", "cyan", "yellow", "black", "magenta"]


def plot_multi_curves(coordinates, title=None, x_label=None, y_label=None, x_limit=None, y_limit=None, legends=None):
    """
    Plot the comparison of multiple curves.
    """
    # 1st step: Create a figure as a canvas.
    plt.figure()

    # 2nd step: Plot on the figure.
    # Plot all curves iteratively.
    # assert len(x) <= len(COLOUR)
    # for i, v in enumerate(x):
    #     plt.plot(v, y[i], color=COLOUR[i], linewidth=1.0, linestyle="-", label=legends[i])
    for i in range(int(coordinates.shape[1] / 2)):
        # Sort fuzzy_th_list in ascending order, and then sort accuracy_list in the same order as fuzzy_th_list.
        # Because multiple processes may not return results in an ascending order of fuzzy thresholds.
        x_list = coordinates[:, i * 2]
        y_list = coordinates[:, i * 2 + 1]
        assert (len(x_list) > 0 and len(y_list) > 0)
        # print("before sorting - x:")
        # print(x_list)
        # print("before sorting - y:")
        # print(y_list)
        x = sorted(x_list)
        y = [y for _, y in sorted(zip(x_list, y_list))]  # Default to the ascending order of 1st list in zip().
        # print("after sorting - x:")
        # print(x)
        # print("after sorting - y:")
        # print(y)

        plt.plot(x, y, color=COLOUR[i], linewidth=1.0, linestyle="-", label=legends[i])

        # Plot the minimum value indicator.
        y_min = np.amin(y_list)
        x_corr_min = [x for _, x in sorted(zip(y_list, x_list))][0]
        plt.scatter(x_corr_min, y_min, s=10, color="black")
        plt.plot([x_corr_min, x_corr_min], [0, y_min], "k--", lw=1.5)
        # Plot the maximum value indicator.
        y_max = np.amax(y_list)
        x_corr_max = [x for _, x in sorted(zip(y_list, x_list))][-1]
        plt.scatter(x_corr_max, y_max, s=10, color="magenta")
        plt.plot([x_corr_max, x_corr_max], [0, y_max], "m--", lw=1.5)

    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(x_limit)
    plt.ylim(y_limit)

    # 3rd step: Show the figure.
    plt.show()


def plot_multi_curves_dyn(q, x_list, y_list, title=None, x_label=None, y_label=None, x_limit=None, y_limit=None):
    pass


if __name__ == '__main__':
    plot_multi_curves([], [], "My Title", "x label", "y label", (10, 20), (-10, 0))
