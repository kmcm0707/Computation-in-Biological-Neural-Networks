import math
from typing import Literal

import numpy as np
import torch
from options.complex_options import (
    complexOptions,
    kMatrixEnum,
    operatorEnum,
    pMatrixEnum,
    vVectorEnum,
    yVectorEnum,
    zVectorEnum,
)
from torch import nn


class ComplexSynapseSlow(nn.Module):
    """
    Complex synapse model.
    The class implements a complex synapse model.
    Which is a biological plausible update rule.
    """

    def __init__(
        self,
        device="cpu",
        numberOfChemicals: int = 1,
        params: dict = {},
        complexOptions: complexOptions = None,
        adaptionPathway: Literal["forward", "feedback"] = "forward",
    ):
        """
        Initialize the complex synapse model.
        :param device: (str) The processing device to use. Default is 'cpu',
        :param numberOfChemicals: (int) The number of chemicals to use. Default is 1,
        :param params: (dict) The model parameters. Default is an empty dictionary,
        :param complexOptions: (complexOptions) The complex options. Default is None,
        :param adaptionPathway: (str) The adaption pathway to use. Default is 'forward'.
        """
        super(ComplexSynapseSlow, self).__init__()

        self.device = device
        self.mode = complexOptions.mode
        self.options = complexOptions
        self.adaptionPathway = adaptionPathway
        self.K_matrix = nn.Parameter()  # K - LxL
        self.v_vector = nn.Parameter()  # v - L
        self.P_matrix = nn.Parameter()  # \theta - Lx10
        self.all_meta_parameters = nn.ParameterList([])  # All updatable meta-parameters except bias

        self.number_chemicals = numberOfChemicals  # L
        self.non_linearity = complexOptions.nonLinear

        self.update_rules = [False] * 10
        for i in self.options.update_rules:
            self.update_rules[i] = True

        self.time_index = 0

        self.init_parameters(params=params)

    @torch.no_grad()
    def init_parameters(self, params: dict = {}):
        """
        Initialize the parameters of the complex synapse model.
        K_matrix: (tensor) The K matrix - dimension (L, L),
        v_vector: (tensor) The v vector - dimension (1, L),
        P_matrix: (tensor) The theta matrix - dimension (L, 10),
        z_vector: (tensor) The z vector - dimension (1, L),
        y_vector: (tensor) The y vector - dimension (1, L),
        """
        ## Initialize the P and K matrices
        if self.options.pMatrix == pMatrixEnum.random:
            self.P_matrix = nn.Parameter(
                torch.nn.init.uniform_(
                    torch.empty(size=(self.number_chemicals, 10), device=self.device),
                    # mean=0,
                    # std=0.001,
                )
            )
            self.P_matrix[:, 0] = torch.abs_(self.P_matrix[:, 0])
            self.P_matrix = torch.nn.Parameter(self.P_matrix)
        elif self.options.pMatrix == pMatrixEnum.rosenbaum_last:
            self.P_matrix = nn.Parameter(
                torch.nn.init.zeros_(torch.empty(size=(self.number_chemicals, 10), device=self.device))
            )
            self.P_matrix[:, 0] = 0.01
            self.P_matrix[-1, 0] = 0.01
            self.P_matrix[-1, 2] = -0.03
            self.P_matrix[-1, 9] = 0.005
        elif self.options.pMatrix == pMatrixEnum.rosenbaum_first:
            self.P_matrix = nn.Parameter(
                torch.nn.init.zeros_(torch.empty(size=(self.number_chemicals, 10), device=self.device))
            )
            self.P_matrix[:, 0] = 0.01
            self.P_matrix[0, 0] = 0.01
            self.P_matrix[0, 2] = -0.03
            self.P_matrix[0, 9] = 0.005
        elif self.options.pMatrix == pMatrixEnum.first_col:
            self.P_matrix = nn.Parameter(
                torch.nn.init.zeros_(torch.empty(size=(self.number_chemicals, 10), device=self.device))
            )
            self.P_matrix[:, 0] = 0.001
        elif self.options.pMatrix == pMatrixEnum.zero:
            self.P_matrix = nn.Parameter(
                torch.nn.init.zeros_(torch.empty(size=(self.number_chemicals, 10), device=self.device))
            )

        if self.options.kMatrix == kMatrixEnum.random:
            self.K_matrix = nn.Parameter(
                torch.nn.init.normal_(
                    torch.empty(
                        size=(self.number_chemicals, self.number_chemicals),
                        device=self.device,
                    ),
                    mean=0,
                    std=1e-3 / self.options.eta,
                )
            )
        elif self.options.kMatrix == kMatrixEnum.xavier:
            self.K_matrix = nn.Parameter(
                torch.nn.init.xavier_normal_(
                    torch.empty(
                        size=(self.number_chemicals, self.number_chemicals),
                        device=self.device,
                    )
                )
            )
        elif self.options.kMatrix == kMatrixEnum.uniform:
            self.K_matrix = nn.Parameter(
                torch.nn.init.uniform_(
                    torch.empty(
                        size=(self.number_chemicals, self.number_chemicals),
                        device=self.device,
                    ),
                    -0.01,
                    0.01,
                )
            )
        elif self.options.kMatrix == kMatrixEnum.zero:
            self.K_matrix = nn.Parameter(
                torch.nn.init.zeros_(
                    torch.empty(
                        size=(self.number_chemicals, self.number_chemicals),
                        device=self.device,
                    )
                )
            )
        elif self.options.kMatrix == kMatrixEnum.identiy:
            identity = torch.eye(self.number_chemicals, device=self.device) * 0.01
            self.K_matrix = nn.Parameter(identity)

        self.all_meta_parameters.append(self.K_matrix)
        self.all_meta_parameters.append(self.P_matrix)

        self.z_vector = torch.tensor([0] * self.number_chemicals, device=self.device)
        self.y_vector = torch.tensor([0] * self.number_chemicals, device=self.device)

        ## Initialize the chemical time constants
        # z = 1 / \tau
        min_tau = self.options.minTau
        max_tau = self.options.maxTau
        base = max_tau / min_tau

        self.tau_vector = min_tau * (base ** torch.linspace(0, 1, self.number_chemicals))
        self.z_vector = 1 / self.tau_vector
        self.y_vector = 1 - self.z_vector

        if self.options.z_vector == zVectorEnum.random:
            self.z_vector = nn.Parameter(
                torch.nn.init.normal_(
                    torch.empty(size=(1, self.number_chemicals), device=self.device),
                    mean=0,
                    std=0.01,
                )
            )
        elif self.options.z_vector == zVectorEnum.all_ones:
            self.z_vector = torch.ones(self.number_chemicals, device=self.device)
        elif self.options.z_vector == zVectorEnum.default:
            pass

        if self.number_chemicals == 1:
            self.y_vector[0] = 1
        elif self.options.y_vector == yVectorEnum.last_one:
            self.y_vector[-1] = 1
        elif self.options.y_vector == yVectorEnum.none:
            pass
        elif self.options.y_vector == yVectorEnum.first_one:
            self.y_vector[0] = 1
        elif self.options.y_vector == yVectorEnum.last_one_and_small_first:
            self.y_vector[-1] = 1
            self.y_vector[0] = self.z_vector[-1]
        elif self.options.y_vector == yVectorEnum.all_ones:
            self.y_vector = torch.ones(self.number_chemicals, device=self.device)
        elif self.options.y_vector == yVectorEnum.half:
            self.y_vector[-1] = 0.5

        self.y_vector = self.y_vector.to(self.device)
        self.y_vector = nn.Parameter(self.y_vector)
        self.z_vector = self.z_vector.to(self.device)
        self.z_vector = nn.Parameter(self.z_vector)

        if self.options.train_z_vector:
            self.all_meta_parameters.append(self.z_vector)

        ## Initialize the v vector
        if self.options.v_vector == vVectorEnum.default:
            self.v_vector = nn.Parameter(
                torch.nn.init.ones_(torch.empty(size=(1, self.number_chemicals), device=self.device))
                / self.number_chemicals
            )
        elif self.options.v_vector == vVectorEnum.random:
            self.v_vector = nn.Parameter(
                torch.nn.init.normal_(
                    torch.empty(size=(1, self.number_chemicals), device=self.device),
                    mean=0,
                    std=1,
                )
            )
            self.v_vector = self.v_vector / torch.norm(self.v_vector, p=2)
        elif self.options.v_vector == vVectorEnum.last_one:
            self.v_vector = nn.Parameter(
                torch.nn.init.zeros_(torch.empty(size=(1, self.number_chemicals), device=self.device))
            )
            self.v_vector[0, -1] = 1
        elif self.options.v_vector == vVectorEnum.random_small:
            self.v_vector = nn.Parameter(
                torch.nn.init.normal_(
                    torch.empty(size=(1, self.number_chemicals), device=self.device),
                    mean=0,
                    std=0.01,
                )
            )
        elif self.options.v_vector == vVectorEnum.random_beta:
            self.v_vector = nn.Parameter(
                self.options.beta
                * torch.nn.init.uniform_(
                    torch.empty(size=(1, self.number_chemicals), device=self.device),
                    # mean=0,
                    # std=1,
                )
                / np.sqrt(self.number_chemicals)
            )
        self.all_meta_parameters.append(self.v_vector)

        ## Initialize the mode
        self.operator = self.options.operator

    @torch.no_grad()
    def reset_time_index(self):
        """
        Reset the time index.
        """
        self.time_index = 0

    @torch.no_grad()
    def update_time_index(self):
        """
        Update the time index.
        """
        self.time_index += 1

    def __call__(
        self,
        params: dict,
        h_parameters: dict,
        activations_and_output: list,
        error: list,
    ):
        """
        :param activations and output: (list) model activations and output - dimension L x (W_1, W_2) (per layer),
        :param error: (list) model error - dimension L x (W_1, W_2) (per layer),
        :param params: (dict) model weights - dimension (W_1, W_2) (per parameter),
        :param h_parameters: (dict) model chemicals - dimension L x (W_1, W_2) (per parameter),
        """

        i = 0
        for name, parameter in params.items():
            if self.adaptionPathway in name:

                h_name = name.replace(self.adaptionPathway, "chemical").split(".")[0]
                if self.adaptionPathway == "feedback":
                    h_name = "feedback_" + h_name
                chemical = h_parameters[h_name]
                if parameter.adapt == self.adaptionPathway and "weight" in name:
                    # Equation 1: h(s+1) = yh(s) + (1/\eta) * zf(Kh(s) + \eta * P * F(Parameter) + b)
                    # Equation 2: w(s) = v * h(s)
                    update_vector = self.calculate_update_vector(error, activations_and_output, parameter, i)
                    # update_vector = update_vector / (torch.amax(update_vector, dim=(1, 2)) + 1e-5)[:, None, None]
                    # update_vector = update_vector / (torch.norm(update_vector, dim=(1, 2), p=2) + 1e-5)[:, None, None]

                    new_chemical = None
                    if (
                        self.operator == operatorEnum.mode_1
                        or self.operator == operatorEnum.mode_3
                        or self.operator == operatorEnum.mode_4
                        or self.operator == operatorEnum.sub_4
                    ):  # mode 1 - was also called add in results
                        new_chemical = torch.einsum("i,ijk->ijk", self.y_vector, chemical) + torch.einsum(
                            "i,ijk->ijk",
                            self.z_vector,
                            self.non_linearity(
                                torch.einsum("ic,ijk->cjk", self.K_matrix, chemical)
                                + torch.einsum("ci,ijk->cjk", self.P_matrix, update_vector)
                            ),
                        )
                    elif self.operator == operatorEnum.sub or self.operator == operatorEnum.sub_4:
                        # Equation 1 - operator = sub: h(s+1) = yh(s) + sign(h(s)) * z( f( sign(h(s)) * (Kh(s) + \theta * F(Parameter) + b) ))
                        new_chemical = torch.einsum("i,ijk->ijk", self.y_vector, chemical) + torch.sign(
                            chemical
                        ) * torch.einsum(
                            "i,ijk->ijk",
                            self.z_vector,
                            self.non_linearity(
                                torch.sign(chemical)
                                * (
                                    torch.einsum("ic,ijk->cjk", self.K_matrix, chemical)
                                    + torch.einsum("ci,ijk->cjk", self.P_matrix, update_vector)
                                )
                            ),
                        )
                    elif self.operator == operatorEnum.mode_2:
                        # Equation 1: h(s+1) = yh(s) + zf(K(zh(s)) + P * F(Parameter) + b)
                        new_chemical = torch.einsum("i,ijk->ijk", self.y_vector, chemical) + torch.einsum(
                            "i,ijk->ijk",
                            self.z_vector,
                            self.non_linearity(
                                self.options.eta
                                * (
                                    torch.einsum(
                                        "ci,ijk->cjk",
                                        self.K_matrix,
                                        torch.einsum("i,ijk->ijk", self.z_vector, chemical),
                                    )
                                    + torch.einsum("ci,ijk->cjk", self.P_matrix, update_vector)
                                )
                            ),
                        )
                    else:
                        raise ValueError("Invalid operator")

                    h_parameters[h_name] = new_chemical
                    if self.operator == operatorEnum.mode_3:
                        # Equation 2: w(s) = w(s) + f(v * h(s))
                        y_schedular = 1
                        z_schedular = 1
                        if self.options.scheduler_t0 != None:
                            z_schedular = self.options.scheduler_t0 / (self.options.scheduler_t0 + self.time_index)
                            y_schedular = 1 - z_schedular

                        new_value = y_schedular * parameter + z_schedular * torch.nn.functional.tanh(
                            torch.einsum("ci,ijk->cjk", self.v_vector, h_parameters[h_name]).squeeze(0)
                        )
                    elif self.operator == operatorEnum.mode_4 or self.operator == operatorEnum.sub_4:
                        v_vector_softmax = torch.nn.functional.softmax(self.v_vector, dim=1)
                        new_value = torch.einsum("ci,ijk->cjk", v_vector_softmax, h_parameters[h_name]).squeeze(0)
                    else:
                        new_value = torch.einsum("ci,ijk->cjk", self.v_vector, h_parameters[h_name]).squeeze(0)

                    params[name] = new_value

                    params[name].adapt = self.adaptionPathway
                i += 1

    @torch.no_grad()
    def initial_update(self, params: dict, h_parameters: dict):
        """
        :param params: (dict) model weights - dimension (W_1, W_2) (per parameter),
        :param h_parameters: (dict) model chemicals - dimension L x (W_1, W_2) (per parameter),

        To connect the forward and chemical parameters.
        """
        for name, parameter in params.items():
            if self.adaptionPathway in name:
                h_name = name.replace(self.adaptionPathway, "chemical").split(".")[0]
                if self.adaptionPathway == "feedback":
                    h_name = "feedback_" + h_name
                if parameter.adapt == self.adaptionPathway and "weight" in name:
                    # Equation 2: w(s) = v * h(s)
                    new_value = torch.einsum("ci,ijk->cjk", self.v_vector, h_parameters[h_name]).squeeze(0)
                    params[name] = new_value

                    params[name].adapt = self.adaptionPathway

    def calculate_update_vector(self, error, activations_and_output, parameter, i) -> torch.Tensor:
        """
        Calculate the update vector for the complex synapse model.
        :param error: (list) model error,
        :param activations_and_output: (list) model activations and output,
        :param parameter: (tensor) model parameter - dimension (W_1, W_2),
        :param i: (int) index of the parameter.
        """
        update_vector = torch.zeros((10, parameter.shape[0], parameter.shape[1]), device=self.device)

        if self.update_rules[0]:
            update_vector[0] = -torch.matmul(error[i + 1].T, activations_and_output[i])  # Pseudo-gradient

        if self.update_rules[1]:
            update_vector[1] = -torch.matmul(activations_and_output[i + 1].T, error[i])

        if self.update_rules[2]:
            update_vector[2] = -torch.matmul(error[i + 1].T, error[i])  # eHebb rule

        if self.update_rules[3]:
            update_vector[3] = -parameter

        if self.update_rules[4]:
            update_vector[4] = -torch.matmul(torch.ones(size=(parameter.shape[0], 1), device=self.device), error[i])

        if self.update_rules[5]:
            """update_vector[5] = -torch.matmul(
                torch.matmul(
                    torch.matmul(
                        error[i + 1].T,
                        torch.ones(size=(1, parameter.shape[0]), device=self.device),
                    ),
                    activations_and_output[i + 1].T,
                ),
                activations_and_output[i],
            )  # = ERROR on high learning rate"""
            """normalised_weight = torch.nn.functional.normalize(parameter.clone(), p=2, dim=1)
            squeeze_activations = activations_and_output[i].clone().squeeze(0)
            normalised_activation = torch.nn.functional.normalize(squeeze_activations, p=2, dim=0)
            output = torch.matmul(normalised_activation, normalised_weight.T)
            max_index_output = torch.argmax(output)  # max index of the output
            update_vector[5][:, max_index_output] = normalised_activation[i] - normalised_weight[:, max_index_output]"""
            softmax_output = torch.nn.functional.softmax(
                torch.matmul(activations_and_output[i + 1].squeeze(0), parameter), dim=0
            )
            diff = parameter - activations_and_output[i].squeeze(0)[None, :]
            update_vector[5] = -torch.matmul(diff, softmax_output[:, None])

        if self.update_rules[6]:
            """update_vector[6] = -torch.matmul(
                torch.matmul(
                    torch.matmul(
                        torch.matmul(
                            activations_and_output[i + 1].T,
                            activations_and_output[i + 1],
                        ),
                        parameter,
                    ),
                    error[i].T,
                ),
                error[i],
            )  # - ERROR"""
            """update_vector[6] = -torch.matmul(
                torch.nn.functional.sigmoid(activations_and_output[i + 1].T),
                torch.nn.functional.sigmoid(activations_and_output[i]),
            )"""

        if self.update_rules[7]:
            update_vector[7] = -torch.matmul(
                torch.matmul(
                    torch.matmul(
                        torch.matmul(error[i + 1].T, activations_and_output[i + 1]),
                        parameter,
                    ),
                    error[i].T,
                ),
                activations_and_output[i],
            )  # - Maybe be bad

        if self.update_rules[8]:
            update_vector[8] = -torch.matmul(
                torch.matmul(
                    torch.matmul(
                        torch.matmul(activations_and_output[i + 1].T, activations_and_output[i]),
                        parameter.T,
                    ),
                    error[i + 1].T,
                ),
                error[i],
            )

        if self.update_rules[9]:
            update_vector[9] = torch.matmul(activations_and_output[i + 1].T, activations_and_output[i]) - torch.matmul(
                torch.matmul(activations_and_output[i + 1].T, activations_and_output[i + 1]),
                parameter,
            )  # Oja's rule

        return update_vector
