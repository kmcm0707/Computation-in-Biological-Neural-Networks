import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional


def multi_plot_accuracy(directories, labels, window_size=20, save_dir=None):
    """
        Plot the meta accuracy using a moving average.

    The method first computes a moving average of the meta accuracy values
    stored in a text file located in the results directory. It then plots
    the moving average values against the meta-training episodes. Finally,
    the plot is saved to a file in the results directory.

    :return: None
    """
    # -- plot
    plt.figure()
    average = []
    for directory in directories:
        z = np.loadtxt(directory + "/acc_meta.txt")
        z = z[0:200]
        z = comp_moving_avg(np.nan_to_num(z), window_size)

        average = average + [z]
    average = np.array(average)
    x = np.array(range(average.shape[1])) + int((window_size - 1) / 2)
    print(x.shape, average.shape)
    for i in range(len(directories)):
        plt.plot(x, average[i], label="{} last={:.2f}".format(labels[i], average[i][-1]))
    plt.axhline(y=0.2, color="r", linestyle="--", label="Chance Level")
    plt.xlabel("Meta-Training Episodes")
    plt.ylabel("Meta Accuracy")
    plt.title("Meta Accuracy (Average)")
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig(save_dir + "/meta_accuracy_average_relu", bbox_inches="tight")
    plt.close()


def matrix_plot(matrix, title, save_dir=None):
    """
        Plot a matrix.

    The method plots a matrix and saves the plot to a file in the results
    directory.

    :return: None
    """

    plt.figure()
    plt.matshow(matrix, cmap="seismic")
    plt.title(title)
    plt.colorbar()
    plt.savefig(save_dir + "/matrix_plot", bbox_inches="tight")
    plt.close()


def comp_moving_avg(data, window_size):
    """
        Compute the moving average of a dataset.

    The method computes the moving average of a dataset using a window
    size. The window size determines the number of data points that are
    averaged to produce a single value. The method returns the moving
    average values.

    :return: The moving average values.
    """
    return np.convolve(data, np.ones(window_size), "valid") / window_size


if __name__ == "__main__":
    results_dir = os.getcwd() + "/results"
    baseline = results_dir + "/Baselines/Generalization_rosenbaum_2/0/20241018-013152"
    Khalf = results_dir + "/Mode_1/K_0.5_std"
    K_001_a = results_dir + "/Mode_1/K_0.01"
    add = results_dir + "/Mode_1/add_2"
    add_bias = results_dir + "/Mode_1/mode_1_bias"
    mode_1_1_chemical = results_dir + "/Mode_1/mode_1_1_chemical"
    add_500 = results_dir + "/Mode_1/add_500"
    mode_2 = results_dir + "/Mode_2/mode_2_default"
    mode_2_2_chems = results_dir + "/Mode_2/mode_2_2_chems"
    mode_2_3_chems = results_dir + "/Mode_2/mode_2_3_chems"
    # mode_2_500 = results_dir + "/Mode_2/mode_2_500"
    mode_2_bias = results_dir + "/Mode_2/mode_2_bias"
    sub = results_dir + "/sub/sub_default"
    Backpropagation = results_dir + "/BP"

    P_random = results_dir + "/Mode_1/random_P_0.01"
    P_random_2 = results_dir + "/Mode_1/random_P_0.01_commet"
    P_rosenbaum = results_dir + "/Mode_1/rosenbaum_first"

    baseline_label = "Shervani-Tabar Baseline"
    BP_label = "Backpropagation"
    add_label = "$h(t+1) = yh(t) + zf(Kh(t) + PF)$"
    add_bias_label = "$h(t+1) = yh(t) + zf(Kh(t) + PF + b)$"
    sub_label = "$h(t+1) = yh(s) + sign(h(t))z(f(sign(h(t))(Kh(t) + PF))$\n"
    mode_1_1_chemical_label = "1 chemical"
    mode_2_label = "$h(t+1) = yh(t) + zf(Kzh(t) + PF)$"
    mode_2_bias_label = "$h(t+1) = yh(t) + zf(Kzh(t) + PF + b)$"
    mode_2_2_chems_label = "2 chemicals"
    mode_2_3_chems_label = "3 chemicals"
    P_random_label = "P randomly initialised with 0.01 std"
    P_random_label_2 = "P randomly initialised but first column is positive"
    P_rosenbaum_label = "P initialised with Shervani-Tabar's results in 1st chemical"

    Khalf_label = "K randomly initialized with 0.5 std"  # , add_label, P_rosenbaum, P_random_label_2
    K_001_label = "K randomly initialized with 0.01 std"  # , add, P_rosenbaum, P_random_2

    matrix = [
        [0.0212, 0.0080, -0.0208, -0.0086, 0.0150, 0.0036, 0.0078],
        [0.0057, -0.0024, -0.0199, -0.0161, 0.0087, 0.0035, 0.0010],
        [0.0127, -0.0004, -0.0205, -0.0161, 0.0126, 0.0040, 0.0062],
        [0.0058, -0.0101, 0.0162, -0.0330, -0.0141, 0.0023, 0.0060],
        [0.0226, -0.0037, -0.0208, -0.0162, 0.0092, 0.0009, 0.0084],
    ]
    # matrix_plot(matrix, "P matrix", save_dir=results_dir)

    multi_plot_accuracy(
        [baseline, add, P_rosenbaum, P_random_2],
        [baseline_label, add_label, P_rosenbaum_label, P_random_label_2],
        save_dir=os.getcwd(),
    )
