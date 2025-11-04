import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from options.meta_learner_options import typeOfFeedbackEnum
from torch.nn import functional


class Plot:
    """
        Plot object.

    This class contains functions for reading, processing, and plotting
    results of the meta-training.
    """

    def __init__(self, res_dir, window_size=11):
        """
            Initialize an instance of the Plot class.

        The function initializes the following instance variables:
        - self.res_dir: a string representing the path to the results directory.
        - self.period: an integer representing the size of the moving average
            window for the plots.
        - self.param_len: an integer representing the size of the meta-parameters
            plus two (for loss and accuracy).

        The function takes in three parameters:
        :param res_dir: (str) a string representing the path to the directory where
            results will be saved.
        :param meta_param_size: (int) size of the meta-parameters.
        :param window_size: (int) size of the moving average window.
        """
        self.res_dir = res_dir
        self.period = window_size
        self.param_len = 12

    @staticmethod
    def comp_moving_avg(vector, period):
        """
            Compute moving average.

        The function computes a moving average for the input data vector over
        a given window size. It does so by first calculating the cumulative sum
        of the data vector using numpy's cumsum function. It then subtracts the
        cumulative sum of the data vector up to (period - 1)th index from the
        cumulative sum of the data vector starting from the period-th index.
        Finally, it divides the result by the window size to obtain the moving
        average vector.

        :param vector: (numpy.ndarray) input data,
        :param period: (int) window size,
        :return: numpy.ndarray: a vector of moving average values computed using
            the input data and window size
        """
        ret = np.cumsum(vector, dtype=float)
        ret[period:] = ret[period:] - ret[:-period]

        return ret[period - 1 :] / period

    def meta_accuracy(self):
        """
            Plot the meta accuracy using a moving average.

        The method first computes a moving average of the meta accuracy values
        stored in a text file located in the results directory. It then plots
        the moving average values against the meta-training episodes. Finally,
        the plot is saved to a file in the results directory.

        :return: None
        """
        # -- compute moving average
        z = self.comp_moving_avg(np.nan_to_num(np.loadtxt(self.res_dir + "/acc_meta.txt")), self.period)

        # -- plot
        plt.plot(np.array(range(len(z))) + int((self.period - 1) / 2), z)

        plt.title("Meta Accuracy")
        plt.xlabel("Meta-training episodes")
        plt.ylabel("Meta Accuracy")
        plt.ylim([0, 1])
        plt.savefig(self.res_dir + "/meta_accuracy", bbox_inches="tight")
        plt.close()

    def meta_parameters(self):
        """
            Plot meta-parameters.

        This function reads in the meta-parameters from the `params.txt` file
        in the result directory and plots them. The x-axis of the plot is the
        meta-training episode number, and the y-axis represents the values of
        each plasticity meta-parameter.

        The function performs the following operations:
        1. Read in the meta-parameters from the `params.txt` file.
        2. Convert the read strings into a numpy array, reshaped to be of shape
            `(num_episodes, param_len)`.
        3. Extract the learning rate from the meta-parameters array and plot it.
        4. Extract the other plasticity meta-parameters from the array, plot them
            in separate colors, and add a legend to the plot.

        :return: None
        """
        # -- read meta-params
        with open(self.res_dir + "/params.txt", "r") as f:
            strings = re.findall(r"(-?\d+\.\d+|nan)", f.read())

        y = np.nan_to_num(np.asarray([float(i) for i in strings])).reshape(-1, self.param_len)
        meta_param_lr = y[:, 2]
        meta_param_terms = y[:, 3:]

        # -- plot meta params
        cmap = plt.get_cmap("tab10")

        # -- pseudo-grad term
        plt.plot(range(len(y)), meta_param_lr, color=cmap(0), label=r"$\theta_0$")

        # -- additional terms
        for i in range(meta_param_terms.shape[1]):
            plt.plot(range(len(y)), meta_param_terms[:, i], color=cmap(i + 1), label=r"$\theta_{}$".format(i + 1))

        # -- plot
        plt.legend()
        plt.title("Meta parameters")
        plt.savefig(self.res_dir + "/meta_params", bbox_inches="tight")
        plt.close()

    def meta_angles(self):
        """
            Plot the meta-angles.

        The method loads the meta angles from the text file located in the
        results directory and computes the moving average using the window size
        specified by `self.period`. The method then plots the computed moving
        average values for each angle index against the meta-training episodes
        using the `plt.plot()` function. A horizontal line is also plotted at
        90 degrees as a baseline.

        :return: None
        """
        # -- read angles
        y = np.nan_to_num(np.loadtxt(self.res_dir + "/e_ang_meta.txt"))

        for idx in range(1, y.shape[1] - 1):
            # -- compute moving average
            z = self.comp_moving_avg(y[:, idx], self.period)

            # -- plot
            plt.plot(np.array(range(len(z))) + int((self.period - 1) / 2), z, label=r"$\alpha_{}$".format(idx))

        plt.legend()
        plt.title("Meta Angles")
        plt.ylim([0, 120])
        plt.plot(np.array(range(len(z))) + int((self.period - 1) / 2), 90.0 * np.ones(len(z)), "--", c="gray")
        plt.savefig(self.res_dir + "/meta_angle", bbox_inches="tight")
        plt.close()

    def meta_loss(self):
        """
            Plot meta-loss.

        This function computes a moving average of the meta-loss values saved
        in the `loss_meta.txt` file, and plots the results. It saves the plot
        in the result directory.

        :return: None
        """
        # -- compute moving average
        z = self.comp_moving_avg(np.nan_to_num(np.loadtxt(self.res_dir + "/loss_meta.txt")), self.period)

        # -- plot
        plt.plot(np.array(range(len(z))) + int((self.period - 1) / 2), z)  # , label=label, color=self.color)

        plt.title("Meta Loss")
        plt.xlabel("Meta-training episodes")
        plt.ylabel("Meta Loss")
        plt.ylim([0, 5])
        plt.savefig(self.res_dir + "/meta_loss", bbox_inches="tight")
        plt.close()

    def __call__(self, *args, **kwargs):
        """
            Call function to plot meta-data.

        The function plots meta-data such as accuracy, parameters, angles and
        loss by calling the corresponding functions.

        :param args: any arguments.
        :param kwargs: any keyword arguments.
        :return: None
        """
        self.meta_accuracy()
        """self.meta_parameters()
        self.meta_angles()"""
        self.meta_loss()


def log(data, filename):
    """
        Save data to a file.

    :param data: (list) data to be saved,
    :param filename: (str) path to the file.
    """
    with open(filename, "a") as f:
        np.savetxt(f, np.array(data), newline=" ", fmt="%0.6f")
        f.writelines("\n")


def normalize_vec(vector):
    """
        Normalize input vector.

    :param vector: (torch.Tensor) input vector,
    :return: normalized vector.
    """
    return vector / torch.linalg.norm(vector)


def measure_angle(v1, v2):
    """
        Compute the angle between two vectors.

    :param v1: (torch.Tensor) the first vector,
    :param v2: (torch.Tensor) the second vector,
    :return: the angle in degrees between the two vectors.
    """
    # -- normalize
    n1 = normalize_vec(v1.squeeze())
    n2 = normalize_vec(v2.squeeze())

    return np.nan_to_num((torch.acos(torch.einsum("i, i -> ", n1, n2)) * 180 / torch.pi).cpu().numpy())


def inner_product(v1, v2):
    """
        Compute the inner product between two vectors.

    :param v1: (torch.Tensor) the first vector,
    :param v2: (torch.Tensor) the second vector,
    :return: the inner product between the two vectors.
    """
    # -- normalize
    n1 = normalize_vec(v1.squeeze())
    n2 = normalize_vec(v2.squeeze())

    return torch.einsum("i, i -> ", n1, n2).cpu().numpy()


def accuracy(logits, label):
    """
        Compute accuracy.

    The function computes the accuracy of the predicted logits compared to the
    ground truth labels.

    :param logits: (torch.Tensor) predicted logits,
    :param label: (torch.Tensor) ground truth labels,
    :return: accuracy of the predicted logits.
    """
    # -- obtain predicted class labels
    pred = functional.softmax(logits, dim=1).argmax(dim=1)

    return torch.eq(pred, label).sum().item() / len(label)


def meta_stats(logits, params, label, y, Beta, res_dir, save=True, typeOfFeedback=typeOfFeedbackEnum.FA, dimOut=47):
    """
        Compute meta statistics.

    The function computes various meta statistics, including modulatory signal,
    orthonormality errors, angle between modulator vectors, and accuracy. These
    statistics are then logged to output files.

    :param logits: (torch.Tensor) logits tensor,
    :param params: self.model.named_parameters(),
    :param label: (torch.Tensor) ground truth label tensor,
    :param y: (tuple) tuple of activation tensors,
    :param Beta: (int) smoothness coefficient for the activation function,
    :param res_dir: (str) output directory path for the log files.
    :return: float: computed accuracy value.
    """

    with torch.no_grad():
        # -- modulatory signal
        B = dict({k: v for k, v in params.items() if "feedback" in k})
        output = functional.softmax(logits, dim=1)
        e = [output - functional.one_hot(label, num_classes=dimOut)]
        if typeOfFeedback == typeOfFeedbackEnum.FA:
            for y_, i in zip(reversed(y), reversed(list(B))):
                e.insert(0, torch.matmul(e[0], B[i]) * (1 - torch.exp(-Beta * y_)))
        elif typeOfFeedback == typeOfFeedbackEnum.FA_NO_GRAD:
            for y_, i in zip(reversed(y), reversed(list(B))):
                e.insert(0, torch.matmul(e[0], B[i]))
        elif typeOfFeedback == typeOfFeedbackEnum.DFA:
            for y_, i in zip(reversed(y), reversed(list(B))):
                e.insert(0, torch.matmul(e[-1], B[i]))
        elif typeOfFeedback == typeOfFeedbackEnum.DFA_grad:
            for y_, i in zip(reversed(y), reversed(list(B))):
                e.insert(0, torch.matmul(e[-1], B[i]) * (1 - torch.exp(-Beta * y_)))
        elif typeOfFeedback == typeOfFeedbackEnum.scalar:
            error_scalars = torch.ones((len(label), 1), device=logits.device)
            subset = torch.gather(output, 1, label.unsqueeze(1))
            mask = subset > 0.5
            mask_indices = mask.nonzero(as_tuple=True)[0]
            indices_more_than_half = mask_indices.squeeze()
            error_scalars[indices_more_than_half] = 0.0
            for y_, i in zip(reversed(y), reversed(list(B))):
                e.insert(0, error_scalars * B[i] * (1 - torch.exp(-Beta * y_)))
        elif typeOfFeedback == typeOfFeedbackEnum.scalar_rate:
            error_scalars = torch.ones((len(label), 1), device=logits.device)
            subset = torch.gather(logits, 1, label.unsqueeze(1))
            mask = subset > 0.5
            mask_indices = mask.nonzero(as_tuple=True)[0]
            indices_more_than_half = mask_indices.squeeze()
            error_scalars[indices_more_than_half] = 0.0
            for y_, i in zip(reversed(y), reversed(list(B))):
                e.insert(0, error_scalars * B[i] * (1 - torch.exp(-Beta * y_)))
        elif typeOfFeedback == typeOfFeedbackEnum.scalar_rich:
            error_scalars = torch.ones((len(label), 1), device=logits.device)
            subset = torch.gather(output, 1, label.unsqueeze(1))
            mask = 1 - subset
            error_scalars = mask
            for y_, i in zip(reversed(y), reversed(list(B))):
                e.insert(0, error_scalars * B[i] * (1 - torch.exp(-Beta * y_)))
        elif typeOfFeedback == typeOfFeedbackEnum.zero:
            for y_ in reversed(y):
                e.insert(0, torch.zeros_like(y_, device=logits.device))
        elif typeOfFeedback == typeOfFeedbackEnum.DFA_grad_FA:
            feedback = {name: value for name, value in params.items() if "feedback_FA" in name}
            DFA_feedback = {name: value for name, value in params.items() if "DFA_feedback" in name}
            DFA_error = [functional.softmax(logits, dim=1) - functional.one_hot(label, num_classes=dimOut)]
            for y_, i in zip(reversed(y), reversed(list(DFA_feedback))):
                DFA_error.insert(0, torch.matmul(e[-1], DFA_feedback[i]) * (1 - torch.exp(Beta * y_)))
            for y_, i in zip(reversed(y), reversed(list(feedback))):
                e.insert(0, torch.matmul(e[0], feedback[i]) * (1 - torch.exp(Beta * y_)))
            for i in range(len(DFA_error)):
                e[i] = (DFA_error[i] + e[i]) / 2

        # -- orthonormality errors

        W = [v for k, v in params.items() if "forward" in k]
        E1 = []
        activation = [*y, functional.softmax(logits, dim=0)]
        for i in range(len(activation) - 1):
            E1.append(
                (
                    torch.norm(
                        torch.matmul(activation[i], W[i].T)
                        - torch.matmul(torch.matmul(activation[i + 1], W[i]), W[i].T)
                    )
                    ** 2
                ).item()
            )

        if save:
            log(E1, res_dir + "/E1_meta.txt")

        e_sym = [e[-1]]
        W = dict({k: v for k, v in params.items() if "forward" in k})
        for y_, i in zip(reversed(y), reversed(list(W))):
            e_sym.insert(0, torch.matmul(e_sym[0], W[i]) * (1 - torch.exp(-Beta * y_)))

        # -- angle between modulator vectors e_FA and e_BP
        e_angl = []
        for e_fix_, e_sym_ in zip(e, e_sym):
            e_angl.append(measure_angle(e_fix_.mean(dim=0), e_sym_.mean(dim=0)))

        if save:
            log(e_angl, res_dir + "/e_ang_meta.txt")

        # -- accuracy
        acc = accuracy(logits, label)

        if save:
            log([acc], res_dir + "/acc_meta.txt")

    return acc


def multi_plot_accuracy(directories, window_size=11, save_dir=None):
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
    average = np.array([])
    for directory in directories:
        z = np.loadtxt(directory + "/acc_meta.txt")
        z = Plot.comp_moving_avg(np.nan_to_num(z), window_size)
        average = z if average.shape[0] == 0 else np.average([average, z], axis=0)

    plt.plot(np.array(range(len(average))) + int((window_size - 1) / 2), average, label="Average")
    plt.title("Meta Accuracy (Average)")
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig(save_dir + "/meta_accuracy_average", bbox_inches="tight")
    plt.close()


class ChemicalAnalysis:
    """
    Chemical Analysis object.
    """

    @torch.no_grad()
    def __init__(self, res_dir):
        """
        Initialize an instance of the ChemicalAnalysis class.
        """
        self.past_chemicals = None
        self.past_parameters = None
        self.res_dir = res_dir
        self.time_step = 0
        self.file_name = self.res_dir + "/chemical_autocorrelation"
        self.parameter_file_name = self.res_dir + "/parameter_autocorrelation"

    def reset(self):
        """
        Reset time step counter.
        """
        self.time_step = 0
        self.past_chemicals = {}
        self.past_parameters = {}
        self.past_chemicals_big = []

    @torch.no_grad()
    def chemical_autocorrelation(self, chemicals):
        """
            Compute weight autocorrelation.

        :param chemicals: (dict) dictionary of chemical tensors,

        """
        with torch.no_grad():
            self.time_step += 1
            layer = 0
            for key in chemicals.keys():
                layer += 1
                if key not in self.past_chemicals:
                    self.past_chemicals[key] = chemicals[key].clone()
                    string_of_keys = " ".join(chemicals.keys())
                    with open(self.file_name + "{}.txt".format(str(layer)), "a") as f:
                        np.savetxt(f, [string_of_keys], newline=" ", fmt="%s")
                        f.writelines("\n")
                else:
                    c_current = chemicals[key].squeeze()
                    c_past = self.past_chemicals[key].squeeze()
                    autocorrelations = []
                    for i in range(c_current.shape[0]):
                        c_current_i = normalize_vec(c_current[i].flatten())
                        c_past_i = normalize_vec(c_past[i].flatten())
                        angle = measure_angle(c_current_i, c_past_i)
                        autocorrelations.append(angle)
                    self.past_chemicals[key] = chemicals[key].clone()
                    with open(self.file_name + "{}.txt".format(str(layer)), "a") as f:
                        np.savetxt(f, ["Timestep: {}".format(self.time_step)], newline=" ", fmt="%s")
                        f.writelines("\n")
                        np.savetxt(f, np.array(autocorrelations), newline=" ", fmt="%0.6f")
                        f.writelines("\n")

    @torch.no_grad()
    def parameter_autocorrelation(self, parameters):
        """
            Compute weight autocorrelation.
        :param parameters: (dict) dictionary of parameter tensors,
        """
        with torch.no_grad():
            layer = 0
            for key in parameters.keys():
                layer += 1
                if "forward" in key:
                    if key not in self.past_parameters:
                        with open(self.parameter_file_name + "{}.txt".format(str(layer)), "a") as f:
                            text = "Parameter"
                            np.savetxt(f, [text], newline=" ", fmt="%s")
                            f.writelines("\n")
                        self.past_parameters[key] = parameters[key].clone()
                    else:
                        p_current = parameters[key].squeeze()
                        p_past = self.past_parameters[key].squeeze()
                        autocorrelations = []
                        p_current = normalize_vec(p_current.flatten())
                        p_past = normalize_vec(p_past.flatten())
                        angle = measure_angle(p_current, p_past)
                        autocorrelations.append(angle)
                        self.past_parameters[key] = parameters[key].clone()
                        with open(self.parameter_file_name + "{}.txt".format(str(layer)), "a") as f:
                            np.savetxt(f, ["Timestep: {}".format(self.time_step)], newline=" ", fmt="%s")
                            f.writelines("\n")
                            np.savetxt(f, np.array(autocorrelations), newline=" ", fmt="%0.6f")
                            f.writelines("\n")

    @torch.no_grad()
    def chemical_tracking(self, chemicals, number_of_parameters):
        """
            Track chemicals over time.

        :param chemicals: (dict) dictionary of chemical tensors,

        """
        with torch.no_grad():
            layer = 0
            for key in chemicals.keys():
                layer += 1
                chemical = chemicals[key]
                for i in range(chemical.shape[0]):
                    flattened_chemical = chemical[i].flatten()
                    parameter_numbers = torch.linspace(
                        0, flattened_chemical.shape[0] - 1, number_of_parameters, dtype=torch.int
                    )
                    if self.time_step == 1:
                        with open(
                            self.res_dir + "/chemical_tracking_layer_{}_chemical_{}.txt".format(str(layer), str(i)), "a"
                        ) as f:
                            header = "Parameter_Numbers " + " ".join(
                                [str(int(num.item())) for num in parameter_numbers]
                            )
                            np.savetxt(f, [header], newline=" ", fmt="%s")
                            f.writelines("\n")
                    else:
                        with open(
                            self.res_dir + "/chemical_tracking_layer_{}_chemical_{}.txt".format(str(layer), str(i)), "a"
                        ) as f:
                            np.savetxt(f, flattened_chemical[parameter_numbers].cpu().numpy(), newline=" ", fmt="%0.6f")
                            f.writelines("\n")

    @torch.no_grad()
    def parameter_tracking(self, parameters, number_of_parameters):
        """
            Track parameters over time.

        :param parameters: (dict) dictionary of parameter tensors,

        """
        with torch.no_grad():
            layer = 0
            for key in parameters.keys():
                layer += 1
                if "forward" in key:
                    parameter = parameters[key]
                    flattened_parameter = parameter.flatten()
                    parameter_numbers = torch.linspace(
                        0, flattened_parameter.shape[0] - 1, number_of_parameters, dtype=torch.int
                    )
                    if self.time_step == 1:
                        with open(self.res_dir + "/parameter_tracking_layer_{}.txt".format(str(layer)), "a") as f:
                            header = "Parameter_Numbers " + " ".join(
                                [str(int(num.item())) for num in parameter_numbers]
                            )
                            np.savetxt(f, [header], newline=" ", fmt="%s")
                            f.writelines("\n")
                    else:
                        with open(self.res_dir + "/parameter_tracking_layer_{}.txt".format(str(layer)), "a") as f:
                            np.savetxt(
                                f, flattened_parameter[parameter_numbers].cpu().numpy(), newline=" ", fmt="%0.6f"
                            )
                            f.writelines("\n")

    @torch.no_grad()
    def Kh_Pf_tracking(self, Kh, Pf, number_of_parameters):
        """
            Track Kh and Pf over time.

        :param Kh: (dict) dictionary of Kh tensors,
        :param Pf: (dict) dictionary of Pf tensors,

        """
        with torch.no_grad():
            layer = 0
            for key in Kh.keys():
                layer += 1
                kh = Kh[key]
                pf = Pf[key]
                for i in range(kh.shape[0]):

                    flattened_kh = kh[i].flatten()
                    flattened_pf = pf[i].flatten()
                    parameter_numbers = torch.linspace(
                        0, flattened_kh.shape[0] - 1, number_of_parameters, dtype=torch.int
                    )
                    if self.time_step == 1:
                        with open(
                            self.res_dir + "/Kh_tracking_layer_{}_chemical_{}.txt".format(str(layer), str(i)), "a"
                        ) as f:
                            header = "Parameter_Numbers " + " ".join(
                                [str(int(num.item())) for num in parameter_numbers]
                            )
                            np.savetxt(f, [header], newline=" ", fmt="%s")
                            f.writelines("\n")
                        with open(
                            self.res_dir + "/Pf_tracking_layer_{}_chemical_{}.txt".format(str(layer), str(i)), "a"
                        ) as f:
                            header = "Parameter_Numbers " + " ".join(
                                [str(int(num.item())) for num in parameter_numbers]
                            )
                            np.savetxt(f, [header], newline=" ", fmt="%s")
                            f.writelines("\n")
                    else:
                        with open(
                            self.res_dir + "/Kh_tracking_layer_{}_chemical_{}.txt".format(str(layer), str(i)), "a"
                        ) as f:
                            np.savetxt(f, flattened_kh[parameter_numbers].cpu().numpy(), newline=" ", fmt="%0.6f")
                            f.writelines("\n")
                        with open(
                            self.res_dir + "/Pf_tracking_layer_{}_chemical_{}.txt".format(str(layer), str(i)), "a"
                        ) as f:
                            np.savetxt(f, flattened_pf[parameter_numbers].cpu().numpy(), newline=" ", fmt="%0.6f")
                            f.writelines("\n")

    @torch.no_grad()
    def chemical_parameter_autocorrelation(self, chemicals, parameters):
        """
            Compute weight autocorrelation.
        :param chemicals: (dict) dictionary of chemical tensors,
        :param parameters: (dict) dictionary of parameter tensors,
        """
        with torch.no_grad():
            layer = 0
            for key in chemicals.keys():
                layer += 1
                if "forward" in key:
                    c_current = chemicals[key].squeeze()
                    p_current = parameters[key].squeeze()
                    p_current = normalize_vec(p_current.flatten())
                    autocorrelations = []
                    for i in range(c_current.shape[0]):
                        c_current_i = normalize_vec(c_current[i].flatten())
                        angle = measure_angle(c_current_i, p_current)
                        autocorrelations.append(angle)
                    with open(
                        self.res_dir + "/chemical_parameter_autocorrelation_layer_{}.txt".format(str(layer)), "a"
                    ) as f:
                        np.savetxt(f, ["Timestep: {}".format(self.time_step)], newline=" ", fmt="%s")
                        f.writelines("\n")
                        np.savetxt(f, np.array(autocorrelations), newline=" ", fmt="%0.6f")
                        f.writelines("\n")

    @torch.no_grad()
    def chemical_actual_autocorrelation(self, chemicals, lags=[5, 10, 20, 50, 100], min_time_step=100):
        """
            Compute weight autocorrelation.

        :param chemicals: (dict) dictionary of chemical tensors,

        """
        if self.time_step == 1:
            self.past_chemicals_big = []
        elif self.time_step == min_time_step + 1:
            current_chemicals = {}
            for key in chemicals.keys():
                current_chemicals[key] = chemicals[key].clone()
            self.past_chemicals_big.append(current_chemicals)
            for lag in lags:
                layer = 0
                for key in chemicals.keys():
                    layer += 1
                    with open(
                        self.res_dir
                        + "/chemical_actual_autocorrelation_lag_{}_layer_{}.txt".format(str(lag), str(layer)),
                        "a",
                    ) as f:
                        text_of_keys = " ".join(chemicals.keys())
                        np.savetxt(f, [text_of_keys], newline=" ", fmt="%s")
                        f.writelines("\n")
        elif self.time_step > min_time_step + 1:
            current_chemicals = {}
            for key in chemicals.keys():
                current_chemicals[key] = chemicals[key].clone()
            self.past_chemicals_big.append(current_chemicals)
            if len(self.past_chemicals_big) > (max(lags) + 1):
                self.past_chemicals_big.pop(0)
            for lag in lags:
                if self.time_step - lag - 1 >= min_time_step:
                    layer = 0
                    for key in chemicals.keys():
                        layer += 1
                        c_current = chemicals[key].squeeze()
                        if self.time_step - 1 > max(lags):
                            c_past = self.past_chemicals_big[max(lags) - lag][key].squeeze()
                        else:
                            c_past = self.past_chemicals_big[-lag - 1][key].squeeze()
                        autocorrelations = []
                        for i in range(c_current.shape[0]):
                            c_current_i = normalize_vec(c_current[i].flatten())
                            c_past_i = normalize_vec(c_past[i].flatten())
                            inner_product_value = inner_product(c_current_i, c_past_i)
                            autocorrelations.append(inner_product_value)
                        with open(
                            self.res_dir
                            + "/chemical_actual_autocorrelation_lag_{}_layer_{}.txt".format(str(lag), str(layer)),
                            "a",
                        ) as f:
                            np.savetxt(f, ["Timestep: {}".format(self.time_step)], newline=" ", fmt="%s")
                            f.writelines("\n")
                            np.savetxt(f, np.array(autocorrelations), newline=" ", fmt="%0.6f")
                            f.writelines("\n")


if __name__ == "__main__":
    # -- test code
    """
    directory = os.getcwd() + '/results/rosenbaum_updated'
    plot = Plot(directory, 3)
    plot.meta_accuracy()
    plot.meta_parameters()
    plot.meta_angles()
    plot.meta_loss()
    plot()
    print('Done')
    """

    # -- test multi_plot_accuracy
    directories = [os.curdir + "/results/rosenbaum_updated_5/{}".format(i) for i in range(0, 6)]
    save_dir = os.curdir + "/results/rosenbaum_updated_5"
    multi_plot_accuracy(directories, window_size=3, save_dir=save_dir)
