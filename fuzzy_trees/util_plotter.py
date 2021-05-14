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


def plot_multi_lines(coordinates, title=None, x_label=None, y_label=None, x_limit=None, y_limit=None, legends=None,
                     f_name=None):
    """
    Plot multiple lines in a figure.
    """
    # 1st step: Create a figure as a canvas.
    plt.figure()

    # 2nd step: Plot on the figure.
    # Plot all curves iteratively.
    # assert len(x) <= len(COLOUR)
    # for i, v in enumerate(x):
    #     plt.plot(v, y[i], color=COLOUR[i], linewidth=1.0, linestyle="-", label=legends[i])
    for i in range(0, coordinates.shape[1], 2):
        # Sort fuzzy_th_list in ascending order, and then sort accuracy_list in the same order as fuzzy_th_list.
        # Because multiple processes may not return results in an ascending order of fuzzy thresholds.
        x_list = coordinates[:, i]
        y_list = coordinates[:, i + 1]
        assert (len(x_list) > 0 and len(y_list) > 0), "Each line should have the coordinates of at least two points"
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

        plt.plot(x, y, linewidth=1.0, linestyle="-", label=legends[int(i / 2)])

        # Plot the minimum point and annotate it.
        y_min = np.amin(y)
        x_corr_min = x[int(np.argmin(y))]
        plt.scatter(x_corr_min, y_min, s=10)  # color="black"
        plt.plot([x_corr_min, x_corr_min], [0, y_min], linestyle="--", lw=1.5)  # color="black"
        plt.annotate(r"$Minimum$", xy=(x_corr_min, y_min), xycoords="data", xytext=(+10, -30),
                     textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=0.3"))

        # Plot the maximum point and annotate it.
        y_max = np.amax(y)
        x_corr_max = x[int(np.argmax(y))]
        plt.scatter(x_corr_max, y_max, s=10)  # color="magenta"
        plt.plot([x_corr_max, x_corr_max], [0, y_max], linestyle="--", lw=1.5)  # color="magenta"
        plt.annotate(r"$Maximum$", xy=(x_corr_max, y_max), xycoords="data", xytext=(-70, -30),
                     textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=0.3"))

    plt.grid(True, linestyle="--", linewidth=1, alpha=0.3)
    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(x_limit)
    plt.ylim(y_limit)

    # # Set axes.
    # ax = plt.gca()
    # ax.spines["top"].set_color("none")
    # ax.spines["right"].set_color("none")
    # ax.xaxis.set_ticks_position("bottom")
    # ax.yaxis.set_ticks_position("left")
    # ax.spines["bottom"].set_position(("data", 0))
    # ax.spines["left"].set_position(("data", 0))

    # Save the figure into a file.
    if f_name is not None:
        plt.savefig(fname=f_name)

    # 3rd step: Show the figure.
    plt.show()


def plot_multi_curves_dyn(x_list, y_list, title=None, x_label=None, y_label=None, x_limit=None, y_limit=None):
    pass


def plot_multi_lines_subplots(df, title=None, x_label=None, y_label=None, x_limit=None, y_limit=None,
                              legends=None, f_name=None):
    """
    Plot multiple curves.
    """
    # 1st step: Create a figure as a canvas.
    plt.figure()

    # 2nd step: Plot on the figure.
    # 2.1 Define the figure using subplot mode.
    # q.put([[ds_name, conv_k, fuzzy_reg, acc_train_mean, std_train, acc_test_mean, std_test]])
    conv_ks = df["conv_k"].unique()
    conv_ks = sorted(conv_ks)
    fig, axes = plt.subplots(nrows=np.size(conv_ks), ncols=1)
    # 2.2 Plot all subplots and curves in it iteratively.
    for idx, conv_k in enumerate(conv_ks):
        df_4_conv_k = df[df["conv_k"] == conv_k]
        x_y = df_4_conv_k.sort_values(by="fuzzy_reg")  # ascending is True by default.
        x = x_y["fuzzy_reg"].values
        y_1 = x_y["acc_train_mean"].values
        y_2 = x_y["acc_test_mean"].values
        axes[idx].plot(x, y_1, linewidth=1.0, linestyle="-", label=legends[int(idx % 2)])
        axes[idx].plot(x, y_2, linewidth=1.0, linestyle="-", label=legends[int(idx % 2)])

        axes[idx].grid(True, linestyle="--", linewidth=1, alpha=0.3)
        axes[idx].legend()
        axes[idx].set_title(title + "(conv_k = " + conv_k + ")")
        axes[idx].set_xlabel(x_label)
        axes[idx].set_ylabel(y_label)
        axes[idx].set_xlim(x_limit)
        axes[idx].set_ylim(y_limit)

    # Save the figure into a file.
    if f_name is not None:
        plt.savefig(fname=f_name)

    # 3rd step: Show the figure.
    plt.tight_layout()  # Make the ticks of X-axis and Y-axis have layout space.
    plt.show()


if __name__ == '__main__':
    plot_multi_lines([], [], "My Title", "x label", "y label", (10, 20), (-10, 0))
