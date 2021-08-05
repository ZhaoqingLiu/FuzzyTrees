"""
@author : Zhaoqing Liu
@email  : Zhaoqing.Liu-1@student.uts.edu.au
"""
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("always")


COLOUR = ["b", "r", "g", "c", "y", "k", "m"]


# The colours above are equivalent to the colours below:
# COLOUR = ["blue", "red", "green", "cyan", "yellow", "black", "magenta"]


def plot_multi_lines(coordinates, title=None, x_label=None, y_label=None, x_limit=None, y_limit=None, x_ticks=None, y_ticks=None, legends=None, fig_name=None, enable_max_annot=False, enable_min_annot=False):
    """
    Plot multiple lines in a figure.

    Parameters
    ----------
    coordinates : array-like
        Where, the values (i.e. coordinates[:, 0]) corresponding to the
        X-axis must be numeric and in ascending order.

    title : str, default=None

    x_label : str, default=None

    y_label : str, default=None

    x_limit : tuple, default=None

    y_limit : tuple, default=None

    legends : array-like, default=None

    fig_name : str, default=None
        fig_name is either a text or byte string giving the name (and the
        path if the file isn't in the current working directory) of the
        file to be opened or an integer file descriptor of the file to be
        wrapped.

    Returns
    -------

    """
    # 1st step: Create a figure as a canvas.
    plt.figure()

    # 2nd step: Plot on the figure.
    # Plot all curves iteratively.
    for i in range(1, coordinates.shape[1], 1):
        x = coordinates[:, 0]
        y = coordinates[:, i]
        assert (len(x) > 1 and len(y) > 1), "Each line should have the coordinates of at least two points"
        plt.plot(x, y, linewidth=1.0, linestyle="-", label=legends[int(i - 1)])

        if enable_min_annot:
            # Plot the minimum point and annotate it.
            y_min = np.amin(y)
            x_corr_min = x[int(np.argmin(y))]
            plt.scatter(x_corr_min, y_min, s=10)  # color="black"
            plt.plot([x_corr_min, x_corr_min], [0, y_min], linestyle="--", lw=1.5)  # color="black"
            plt.annotate(r"$Minimum$", xy=(x_corr_min, y_min), xycoords="data", xytext=(+10, -30),
                         textcoords="offset points",
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=0.3"))

        if enable_max_annot:
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
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    # # Set axes.
    # ax = plt.gca()
    # ax.spines["top"].set_color("none")
    # ax.spines["right"].set_color("none")
    # ax.xaxis.set_ticks_position("bottom")
    # ax.yaxis.set_ticks_position("left")
    # ax.spines["bottom"].set_position(("data", 0))
    # ax.spines["left"].set_position(("data", 0))

    # Save the figure into a file.
    if fig_name is not None:
        plt.savefig(fname=fig_name)

    # 3rd step: Show the figure.
    plt.show()

