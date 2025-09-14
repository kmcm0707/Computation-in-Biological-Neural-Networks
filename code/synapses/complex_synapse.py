from typing import Literal

import numpy as np
import torch
from options.complex_options import (
    complexOptions,
    kMatrixEnum,
    modeEnum,
    operatorEnum,
    pMatrixEnum,
    vVectorEnum,
    yVectorEnum,
    zVectorEnum,
)
from torch import nn


class ComplexSynapse(nn.Module):
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
        :param mode: (str) The update rule to use. Default is 'rosenbaum'.
        :param numberOfChemicals: (int) The number of chemicals to use. Default is 1,
        :param params: (dict) The model parameters. Default is an empty dictionary,
        :param complexOptions: (complexOptions) The complex options. Default is None,
        :param adaptionPathway: (str) The adaption pathway to use. Default is 'forward'.
        """
        super(ComplexSynapse, self).__init__()

        self.device = device
        self.mode = complexOptions.mode
        self.options = complexOptions
        self.adaptionPathway = adaptionPathway

        # h(s+1) = (1-z)h(s) + zf(Kh(s) + \theta * F(Parameter) + b)
        # y = 1-z, y_0 = 1, z_0 = 1
        # w(s) = v * h(s) (if self.number_chemicals = 1)
        self.K_matrix = nn.Parameter()  # K - LxL
        self.v_vector = nn.Parameter()  # v - L
        self.P_matrix = nn.Parameter()  # \theta - Lx10
        self.all_meta_parameters = nn.ParameterList([])  # All updatable meta-parameters except bias
        # self.bias_dictionary = torch.nn.ParameterDict()  # All bias parameters
        # self.all_bias_parameters = nn.ParameterList([])  # All bias parameters if they are used
        self.number_chemicals = numberOfChemicals  # L

        # self.bcm_dict = {}

        self.non_linearity = complexOptions.nonLinear

        self.update_rules = [False] * 10
        if self.mode == modeEnum.rosenbaum:
            self.update_rules[0] = True
            self.update_rules[2] = True
            self.update_rules[9] = True
        elif self.mode == modeEnum.all_rosenbaum:
            self.update_rules = [True] * 10
        else:
            for i in self.options.update_rules:
                self.update_rules[i] = True

        self.time_index = 0

        if self.options.operator == operatorEnum.v_linear:
            self.v_dict = {}

        self.saved_norm = {}

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
        ## Initialize the bias parameters
        """for name, parameter in params:
            if self.adaptionPathway in name and "chemical" not in name:
                h_name = name.replace(self.adaptionPathway, "chemical").split(".")[0]
                if self.adaptionPathway == "feedback":
                    h_name = "feedback_" + h_name
                self.bias_dictionary[h_name] = nn.Parameter(
                    torch.nn.init.zeros_(
                        torch.empty(
                            size=(
                                self.number_chemicals,
                                parameter.shape[0],
                                parameter.shape[1],
                            ),
                            device=self.device,
                            requires_grad=True,
                        )
                    )
                )
                if self.options.operator == operatorEnum.v_linear:
                    self.v_dict[h_name] = (
                        torch.ones(
                            size=(
                                self.number_chemicals,
                                parameter.shape[0],
                                parameter.shape[1],
                            ),
                            device=self.device,
                            requires_grad=False,
                        )
                        / self.number_chemicals
                    )
                self.bias_dictionary[h_name] = nn.Parameter(
                    torch.tensor([0.0] * self.number_chemicals, device=self.device)
                )
                print("BCM")
                self.bcm_dict[h_name] = torch.tensor([0.0] * parameter.shape[0], device=self.device)
                self.saved_norm[h_name] = torch.norm(parameter, p=2)"""

        if self.options.bias:
            raise ("Bias Disabled")
            # self.all_bias_parameters.extend(self.bias_dictionary.values())

        ## Initialize the P and K matrices
        if self.mode == modeEnum.rosenbaum or self.mode == modeEnum.all_rosenbaum:
            self.P_matrix = nn.Parameter(
                torch.nn.init.zeros_(torch.empty(size=(self.number_chemicals, 10), device=self.device))
            )
            self.P_matrix[:, 0] = 1e-3 / self.options.eta

            self.K_matrix = nn.Parameter(
                torch.nn.init.zeros_(
                    torch.empty(
                        size=(self.number_chemicals, self.number_chemicals),
                        device=self.device,
                    )
                )
            )
        else:
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
                self.P_matrix[:, 0] = 0.001 / self.options.eta
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
            elif self.options.kMatrix == kMatrixEnum.identity:
                identity = torch.eye(self.number_chemicals, device=self.device)
                self.K_matrix = nn.Parameter(identity)
            self.all_meta_parameters.append(self.K_matrix)

        self.all_meta_parameters.append(self.P_matrix)

        ## K Mask
        """if self.options.kMasking:
            identity = torch.eye(self.number_chemicals, device=self.device)
            self.K_mask = torch.ones(self.number_chemicals, self.number_chemicals, device=self.device) - identity
        else:
            self.K_mask = torch.ones(self.number_chemicals, self.number_chemicals, device=self.device)"""

        self.z_vector = torch.tensor([0] * self.number_chemicals, device=self.device)
        self.y_vector = torch.tensor([0] * self.number_chemicals, device=self.device)

        ## Initialize the chemical time constants
        # z = 1 / \tau
        self.min_tau = self.options.minTau
        self.max_tau = self.options.maxTau
        if self.options.train_tau:
            self.min_tau = nn.Parameter(torch.tensor(self.min_tau, device=self.device, dtype=torch.float32))
            self.max_tau = nn.Parameter(torch.tensor(self.max_tau, device=self.device, dtype=torch.float32))
            self.all_meta_parameters.append(self.min_tau)
            self.all_meta_parameters.append(self.max_tau)
        base = self.max_tau / self.min_tau
        self.tau_vector = self.min_tau * (base ** torch.linspace(0, 1, self.number_chemicals, device=self.device))
        self.z_vector = 1 / self.tau_vector
        self.y_vector = 1 - self.z_vector

        # self.y_vector = 1 / self.tau_vector
        # self.z_vector = 1 - self.y_vector
        # self.z_vector[0] = 1

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
        elif self.options.y_vector == yVectorEnum.first_one:  # default
            self.y_vector[0] = 1
        elif self.options.y_vector == yVectorEnum.last_one_and_small_first:
            self.y_vector[-1] = 1
            self.y_vector[0] = self.z_vector[-1]
        elif self.options.y_vector == yVectorEnum.all_ones:
            self.y_vector = torch.ones(self.number_chemicals, device=self.device)
        elif self.options.y_vector == yVectorEnum.half:
            self.y_vector[-1] = 0.5

        if not self.options.train_tau:
            self.y_vector = self.y_vector.to(self.device)
            self.y_vector = nn.Parameter(self.y_vector)
            self.z_vector = self.z_vector.to(self.device)
            self.z_vector = nn.Parameter(self.z_vector)
            if self.options.train_z_vector:
                self.all_meta_parameters.append(self.z_vector)

        ## Initialize the v vector
        if self.mode == modeEnum.rosenbaum or self.mode == modeEnum.all_rosenbaum:
            self.v_vector = nn.Parameter(
                torch.nn.init.ones_(torch.empty(size=(1, self.number_chemicals), device=self.device))
                / self.number_chemicals
            )
        else:
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

        ## Attention mechanism
        """if self.operator == operatorEnum.attention:
            self.A = nn.Parameter(
                torch.nn.init.xavier_normal_(
                    torch.empty(
                        size=(self.number_chemicals, self.number_chemicals),
                        device=self.device,
                    ),
                )
            )
            self.all_meta_parameters.append(self.A)

            self.A_bias = nn.Parameter(
                torch.nn.init.xavier_uniform_(
                    torch.empty(
                        size=(1, self.number_chemicals),
                        device=self.device,
                    ),
                )
            ).squeeze_(0)
            self.all_meta_parameters.append(self.A_bias)

            self.B = nn.Parameter(
                torch.nn.init.xavier_normal_(
                    torch.empty(
                        size=(self.number_chemicals, self.number_chemicals),
                        device=self.device,
                    ),
                )
            )
            self.all_meta_parameters.append(self.B)

            self.B_bias = nn.Parameter(
                torch.nn.init.xavier_uniform_(
                    torch.empty(
                        size=(1, self.number_chemicals),
                        device=self.device,
                    ),
                )
            ).squeeze_(0)
            self.all_meta_parameters.append(self.B_bias)
        elif self.operator == operatorEnum.extended_attention:
            self.A = nn.Parameter(
                torch.nn.init.xavier_normal_(
                    torch.empty(
                        size=(self.number_chemicals, self.number_chemicals),
                        device=self.device,
                    ),
                )
            )
            self.all_meta_parameters.append(self.A)

            self.A_bias = nn.Parameter(
                torch.nn.init.xavier_uniform_(
                    torch.empty(
                        size=(1, self.number_chemicals),
                        device=self.device,
                    ),
                )
            ).squeeze_(0)
            self.all_meta_parameters.append(self.A_bias)

            self.B = nn.Parameter(
                torch.nn.init.xavier_normal_(
                    torch.empty(
                        size=(10, self.number_chemicals),
                        device=self.device,
                    ),
                )
            )
            self.all_meta_parameters.append(self.B)

            self.B_bias = nn.Parameter(
                torch.nn.init.xavier_uniform_(
                    torch.empty(
                        size=(1, self.number_chemicals),
                        device=self.device,
                    ),
                )
            ).squeeze_(0)
            self.all_meta_parameters.append(self.B_bias)
        elif self.operator == operatorEnum.attention_2:
            self.A = nn.Parameter(
                torch.nn.init.xavier_normal_(
                    torch.empty(
                        size=(10 + self.number_chemicals, self.number_chemicals),
                        device=self.device,
                    ),
                )
            )
            self.all_meta_parameters.append(self.A)

            self.A_bias = nn.Parameter(
                torch.nn.init.xavier_uniform_(
                    torch.empty(
                        size=(1, self.number_chemicals),
                        device=self.device,
                    ),
                )
            ).squeeze_(0)
            self.all_meta_parameters.append(self.A_bias)

            self.B = nn.Parameter(
                torch.nn.init.xavier_normal_(
                    torch.empty(
                        size=(10 + self.number_chemicals, self.number_chemicals),
                        device=self.device,
                    ),
                )
            )
            self.all_meta_parameters.append(self.B)

            self.B_bias = nn.Parameter(
                torch.nn.init.xavier_uniform_(
                    torch.empty(
                        size=(1, self.number_chemicals),
                        device=self.device,
                    ),
                )
            ).squeeze_(0)
            self.all_meta_parameters.append(self.B_bias)
        elif self.operator == operatorEnum.full_attention:
            self.linear_attention = nn.Linear(
                10 + self.number_chemicals, 3 * (10 + self.number_chemicals), device=self.device
            )
            self.attention = nn.MultiheadAttention(10 + self.number_chemicals, 1, device=self.device)
            self.compress_attention = nn.Linear(10 + self.number_chemicals, self.number_chemicals, device=self.device)

            self.all_meta_parameters.append(self.linear_attention)
            self.all_meta_parameters.append(self.attention)
            self.all_meta_parameters.append(self.compress_attention)
        elif self.operator == operatorEnum.compressed_full_attention:
            self.linear_attention = nn.Linear(11, self.number_chemicals * 3, device=self.device)
            self.attention = nn.MultiheadAttention(self.number_chemicals, 1, device=self.device)
            self.all_meta_parameters.append(self.linear_attention)
            self.all_meta_parameters.append(self.attention)
        elif self.operator == operatorEnum.v_linear:
            # v_+1 = linear(w-h, update, v_-1, time)
            self.linear_v = nn.Linear(10 + 2 * self.number_chemicals + 1, self.number_chemicals, device=self.device)
            self.all_meta_parameters.append(self.linear_v)
        elif self.operator == operatorEnum.compressed_v_linear:
            # v_+1 = linear(update, time)
            self.linear_v = nn.Linear(11, self.number_chemicals, device=self.device)
            self.all_meta_parameters.append(self.linear_v)"""

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
        override_adaption_pathway: Literal["forward", "feedback"] = None,
    ):
        """
        :param activations and output: (list) model activations and output - dimension L x (W_1, W_2) (per layer),
        :param error: (list) model error - dimension L x (W_1, W_2) (per layer),
        :param params: (dict) model weights - dimension (W_1, W_2) (per parameter),
        :param h_parameters: (dict) model chemicals - dimension L x (W_1, W_2) (per parameter),
        """

        """for i in range(len(activations_and_output)):
            # activations_and_output[i] = activations_and_output[i] / torch.norm(activations_and_output[i], p=2)
            activations_and_output[i] = activations_and_output[i] / (
                torch.max(torch.abs(activations_and_output[i])) + 1e-5
            )

        for i in range(len(error)):
            # error[i] = error[i] / torch.norm(error[i], p=2)
            error[i] = error[i] / (torch.max(torch.abs(error[i])) + 1e-5)"""

        if self.options.train_tau and self.time_index == 0:
            base = self.max_tau / self.min_tau
            self.tau_vector = self.min_tau * (base ** torch.linspace(0, 1, self.number_chemicals, device=self.device))
            self.z_vector = 1 / self.tau_vector
            if self.options.y_vector == yVectorEnum.none:
                self.y_vector = 1 - self.z_vector

        i = 0
        currentAdaptionPathway = self.adaptionPathway
        if override_adaption_pathway != None:
            currentAdaptionPathway = override_adaption_pathway
        for name, parameter in params.items():
            if currentAdaptionPathway in name:

                h_name = name.replace(currentAdaptionPathway, "chemical").split(".")[0]
                if currentAdaptionPathway == "feedback":
                    h_name = "feedback_" + h_name

                chemical = h_parameters[h_name]
                if parameter.adapt == currentAdaptionPathway and "weight" in name:
                    # Equation 1: h(s+1) = yh(s) + (1/\eta) * zf(Kh(s) + \eta * P * F(Parameter) + b)
                    # Equation 2: w(s) = v * h(s)
                    update_vector = self.calculate_update_vector(error, activations_and_output, parameter, i, h_name)
                    # update_vector = update_vector / (torch.amax(update_vector, dim=(1, 2)) + 1e-5)[:, None, None]
                    # update_vector = update_vector / (torch.norm(update_vector, dim=(1, 2), p=2) + 1e-5)[:, None, None]

                    new_chemical = None
                    if (
                        self.operator == operatorEnum.mode_1
                        or self.operator == operatorEnum.mode_3
                        or self.operator == operatorEnum.attention
                        or self.operator == operatorEnum.extended_attention
                        or self.operator == operatorEnum.attention_2
                        or self.operator == operatorEnum.full_attention
                        or self.operator == operatorEnum.mode_4
                        or self.operator == operatorEnum.mode_5
                        or self.operator == operatorEnum.mode_6
                        or self.operator == operatorEnum.mode_7
                        or self.operator == operatorEnum.compressed_full_attention
                        or self.operator == operatorEnum.v_linear
                        or self.operator == operatorEnum.compressed_v_linear
                    ):  # mode 1 - was also called add in results
                        new_chemical = torch.einsum("i,ijk->ijk", self.y_vector, chemical) + torch.einsum(
                            "i,ijk->ijk",
                            self.z_vector,
                            self.non_linearity(
                                torch.einsum("ic,ijk->cjk", self.K_matrix, chemical)  # self.K_mask * self.K_matrix
                                + torch.einsum("ci,ijk->cjk", self.P_matrix, update_vector)
                                # + self.bias_dictionary[h_name]  # [:, None, None]
                            ),
                        )
                        if self.operator == operatorEnum.mode_5 or self.operator == operatorEnum.mode_6:
                            parameter_norm = self.saved_norm[h_name]
                            chemical_norms = torch.norm(new_chemical, dim=(1, 2))
                            multiplier = parameter_norm / (chemical_norms)
                            new_chemical = (
                                new_chemical * multiplier[:, None, None]
                            )  # chemical_norms[:, None, None] (mode 5 v2 is commented out)
                        elif self.operator == operatorEnum.mode_7:
                            new_chemical = torch.nn.functional.normalize(new_chemical, p=2, dim=1)
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
                                    + self.bias_dictionary[h_name]
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
                                + self.bias_dictionary[h_name]
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
                    elif self.operator == operatorEnum.attention:
                        # Equation 2: attention mechanism
                        # v(s) = Attention(h(s), w(s-1), v(s-1), Input)
                        # For now v(s) = attention(h(s))
                        # v(s) = (A * h(s) + A_bias) * (B * h(s) + B_bias)
                        # v(s) = softmax(v(s))
                        # w(s) = v(s) * h(s)
                        v_A = torch.einsum("ic,ijk->cjk", self.A, h_parameters[h_name]) + self.A_bias[:, None, None]
                        v_B = torch.einsum("ic,ijk->cjk", self.B, h_parameters[h_name]) + self.B_bias[:, None, None]
                        new_v = torch.nn.functional.softmax(torch.einsum("ijk,ijk->ijk", v_A, v_B), dim=0)
                        new_value = torch.einsum("ijk,ijk->jk", new_v, h_parameters[h_name])
                    elif self.operator == operatorEnum.extended_attention:
                        # Equation 2: attention mechanism
                        # v(s) = (A * h(s) + A_bias) * (B * update + B_bias)
                        # v(s) = softmax(v(s))
                        # w(s) = v(s) * h(s)
                        v_A = torch.einsum("ic,ijk->cjk", self.A, h_parameters[h_name]) + self.A_bias[:, None, None]
                        v_B = torch.einsum("ic,ijk->cjk", self.B, update_vector) + self.B_bias[:, None, None]
                        new_v = torch.nn.functional.softmax(torch.einsum("ijk,ijk->ijk", v_A, v_B), dim=0)
                        new_value = torch.einsum("ijk,ijk->jk", new_v, h_parameters[h_name])
                    elif self.operator == operatorEnum.attention_2:
                        # Equation 2: attention mechanism
                        # v(s) = (A * (h(s), update) + A_bias) * (B * (h(s), update) + B_bias)
                        # v(s) = softmax(v(s))
                        # w(s) = v(s) * h(s)
                        attention_vector = torch.cat((h_parameters[h_name], update_vector), dim=0)
                        attention_vector = (
                            attention_vector / (torch.norm(attention_vector, p=2, dim=(1, 2)) + 1e-5)[:, None, None]
                        )
                        v_A = torch.einsum("ic,ijk->cjk", self.A, attention_vector) + self.A_bias[:, None, None]
                        v_B = torch.einsum("ic,ijk->cjk", self.B, attention_vector) + self.B_bias[:, None, None]
                        new_v = torch.nn.functional.softmax(torch.einsum("ijk,ijk->ijk", v_A, v_B), dim=0)
                        new_value = torch.einsum("ijk,ijk->jk", new_v, h_parameters[h_name])
                    elif self.operator == operatorEnum.full_attention:
                        attention_vector = torch.cat((h_parameters[h_name], update_vector), dim=0)
                        attention_vector = torch.reshape(
                            attention_vector,
                            (1, attention_vector.shape[1] * attention_vector.shape[2], 10 + self.number_chemicals),
                        )
                        linear_attention = self.linear_attention(attention_vector)
                        K = linear_attention[:, :, : 10 + self.number_chemicals]
                        Q = linear_attention[:, :, 10 + self.number_chemicals : 2 * (10 + self.number_chemicals)]
                        V = linear_attention[:, :, 2 * (10 + self.number_chemicals) :]
                        intermeditate_v, _ = self.attention(Q, K, V)
                        new_v = self.compress_attention(intermeditate_v)
                        new_v = new_v.squeeze(0)
                        v_softmax = torch.nn.functional.softmax(new_v, dim=1)
                        v_softmax = torch.reshape(
                            v_softmax,
                            (self.number_chemicals, h_parameters[h_name].shape[1], h_parameters[h_name].shape[2]),
                        )
                        new_value = torch.einsum("ijk,ijk->jk", v_softmax, h_parameters[h_name])
                    elif self.operator == operatorEnum.compressed_full_attention:
                        time_index = (
                            torch.ones(size=(1, update_vector.shape[1], update_vector.shape[2]), device=self.device)
                            * self.time_index
                        )
                        attention_vector = torch.cat((time_index, update_vector), dim=0)
                        attention_vector = torch.reshape(
                            attention_vector,
                            (1, update_vector.shape[1] * update_vector.shape[2], 11),
                        )
                        linear_attention = self.linear_attention(attention_vector)
                        K = linear_attention[:, :, : self.number_chemicals]
                        Q = linear_attention[:, :, self.number_chemicals : 2 * self.number_chemicals]
                        V = linear_attention[:, :, 2 * self.number_chemicals :]
                        v, _ = self.attention(Q, K, V)
                        v = v.squeeze(0)
                        v_softmax = torch.nn.functional.softmax(v, dim=1)
                        v_softmax = torch.reshape(
                            v_softmax,
                            (self.number_chemicals, h_parameters[h_name].shape[1], h_parameters[h_name].shape[2]),
                        )
                        new_value = torch.einsum("ijk,ijk->jk", v_softmax, h_parameters[h_name])
                    elif self.operator == operatorEnum.v_linear:
                        attention_vector = None
                        with torch.no_grad():
                            time_index = (
                                torch.ones(size=(1, update_vector.shape[1], update_vector.shape[2]), device=self.device)
                                * self.time_index
                            )
                            h_diff = h_parameters[h_name] - parameter[None, :, :]
                            attention_vector = torch.cat(
                                (time_index, update_vector, self.v_dict[h_name], h_diff), dim=0
                            )
                            attention_vector = torch.reshape(
                                attention_vector,
                                (update_vector.shape[1] * update_vector.shape[2], 10 + 2 * self.number_chemicals + 1),
                            )
                        new_v = self.linear_v(attention_vector)
                        v_softmax = torch.nn.functional.softmax(new_v, dim=1)
                        v_softmax = torch.reshape(
                            v_softmax,
                            (self.number_chemicals, update_vector.shape[1], update_vector.shape[2]),
                        )
                        new_value = torch.einsum("ijk,ijk->jk", v_softmax, h_parameters[h_name])
                    elif self.operator == operatorEnum.compressed_v_linear:
                        attention_vector = None
                        with torch.no_grad():
                            time_index = (
                                torch.ones(size=(1, update_vector.shape[1], update_vector.shape[2]), device=self.device)
                                * self.time_index
                            )
                            attention_vector = torch.cat((time_index, update_vector), dim=0)
                            attention_vector = torch.reshape(
                                attention_vector,
                                (update_vector.shape[1] * update_vector.shape[2], 11),
                            )
                        new_v = self.linear_v(attention_vector)
                        v_softmax = torch.nn.functional.softmax(new_v, dim=1)
                        v_softmax = torch.reshape(
                            v_softmax,
                            (self.number_chemicals, update_vector.shape[1], update_vector.shape[2]),
                        )
                        new_value = torch.einsum("ijk,ijk->jk", v_softmax, h_parameters[h_name])
                    elif (
                        self.operator == operatorEnum.mode_4
                        or self.operator == operatorEnum.sub_4
                        or self.operator == operatorEnum.mode_5
                        or self.operator == operatorEnum.mode_6
                        or self.operator == operatorEnum.mode_7
                    ):
                        v_vector_softmax = torch.nn.functional.softmax(self.v_vector, dim=1)
                        new_value = torch.einsum("ci,ijk->cjk", v_vector_softmax, h_parameters[h_name]).squeeze(0)
                        if self.operator == operatorEnum.mode_6:
                            parameter_norm = self.saved_norm[h_name]
                            current_norm = torch.norm(new_value, p=2)
                            multiplier = parameter_norm / current_norm
                            new_value = new_value * multiplier
                        elif self.operator == operatorEnum.mode_7:
                            new_value = torch.nn.functional.normalize(new_value, p=2, dim=0)
                    else:
                        new_value = torch.einsum("ci,ijk->cjk", self.v_vector, h_parameters[h_name]).squeeze(0)

                    params[name] = new_value

                    params[name].adapt = currentAdaptionPathway
                i += 1

    @torch.no_grad()
    def initial_update(
        self, params: dict, h_parameters: dict, override_adaption_pathway: Literal["forward", "feedback"] = None
    ):
        """
        :param params: (dict) model weights - dimension (W_1, W_2) (per parameter),
        :param h_parameters: (dict) model chemicals - dimension L x (W_1, W_2) (per parameter),

        To connect the forward and chemical parameters.
        """
        currentAdaptionPathway = self.adaptionPathway
        if override_adaption_pathway != None:
            assert override_adaption_pathway in ["forward", "feedback"]
            currentAdaptionPathway = override_adaption_pathway

        for name, parameter in params.items():
            if currentAdaptionPathway in name:
                h_name = name.replace(currentAdaptionPathway, "chemical").split(".")[0]
                if currentAdaptionPathway == "feedback":
                    h_name = "feedback_" + h_name
                if parameter.adapt == currentAdaptionPathway and "weight" in name:
                    # Equation 2: w(s) = v * h(s)
                    # if self.operator == operatorEnum.mode_7:
                    self.saved_norm[h_name] = torch.norm(parameter, p=2)
                    new_value = torch.einsum("ci,ijk->cjk", self.v_vector, h_parameters[h_name]).squeeze(0)
                    if (
                        self.operator == operatorEnum.mode_5
                        or self.operator == operatorEnum.mode_6
                        or self.operator == operatorEnum.mode_7
                    ):
                        parameter_norm = self.saved_norm[h_name]
                        current_norm = torch.norm(new_value, p=2)
                        multiplier = parameter_norm / current_norm
                        new_value = new_value * multiplier
                    params[name] = new_value

                    params[name].adapt = currentAdaptionPathway

    def calculate_update_vector(self, error, activations_and_output, parameter, i, h_name) -> torch.Tensor:
        """
        Calculate the update vector for the complex synapse model.
        :param error: (list) model error,
        :param activations_and_output: (list) model activations and output,
        :param parameter: (tensor) model parameter - dimension (W_1, W_2),
        :param i: (int) index of the parameter.
        """
        update_vector = torch.zeros((10, parameter.shape[0], parameter.shape[1]), device=self.device)
        # with torch.no_grad():
        if self.update_rules[0]:
            update_vector[0] = -torch.matmul(error[i + 1].T, activations_and_output[i])  # Pseudo-gradient

        if self.update_rules[1]:
            update_vector[1] = -torch.matmul(activations_and_output[i + 1].T, error[i])

        if self.update_rules[2]:
            update_vector[2] = -(
                torch.matmul(error[i + 1].T, error[i])
                - torch.matmul(
                    parameter,
                    torch.matmul(error[i].T, error[i]),
                )
            )  # eHebb rule

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
            update_vector[5][:, max_index_output] = -torch.matmul(diff, softmax_output[:, None])"""
            """softmax_output = torch.nn.functional.softmax(
                torch.matmul(activations_and_output[i + 1].squeeze(0), parameter), dim=0
            )"""
            # softmax_output = torch.nn.functional.softmax(activations_and_output[i + 1].squeeze(0), dim=0)
            # diff = parameter - activations_and_output[i].squeeze(0)[None, :]
            # normalised_weight = torch.nn.functional.normalize(parameter.clone(), p=2, dim=1)
            # squeeze_activations = activations_and_output[i].clone().squeeze(0)
            # normalised_activation = torch.nn.functional.normalize(squeeze_activations, p=2, dim=0)
            # diff = parameter - activations_and_output[i].squeeze(0)
            normalised_weight = torch.nn.functional.normalize(parameter, p=2, dim=0)
            squeeze_activations = activations_and_output[i].squeeze(0)
            normalised_activation = torch.nn.functional.normalize(squeeze_activations, p=2, dim=0)
            output = torch.nn.functional.softplus(
                torch.matmul(normalised_activation, normalised_weight.T),
                beta=10.0,
            )
            softmax_output = torch.nn.functional.softmax(output, dim=0)
            diff = normalised_weight - normalised_activation
            update_vector[5] = -(diff * softmax_output[:, None])
            """norm_uv = torch.norm(update_vector[5], p=2)
            norm_diff = torch.norm(diff, p=2)
            norm_activation = torch.norm(activations_and_output[i].squeeze(0), p=2)
            softmax_output_norm = torch.norm(softmax_output, p=2)
            line = (
                "time: "
                + str(self.time_index)
                + "update vector norm: "
                + str(norm_uv)
                + " diff norm: "
                + str(norm_diff)
                + " activation norm: "
                + str(norm_activation)
                + " softmax output: "
                + str(softmax_output_norm)
            )
            current_cwd = os.getcwd()
            with open(current_cwd + "/check.txt", "a+") as f:
                print(line)
                f.writelines(line + "\n")"""

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
            update_vector[6] = -torch.matmul(
                torch.ones(size=(parameter.shape[0], 1), device=self.device), activations_and_output[i]
            )

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
            """update_vector[9] = torch.matmul(activations_and_output[i + 1].T, activations_and_output[i]) - torch.matmul(
                torch.diag(torch.square(activations_and_output[i + 1].squeeze(0)), diagonal=0),
                parameter,
            )  # Diag Oja's rule"""
            """update_vector[9] = torch.matmul(activations_and_output[i + 1].T, activations_and_output[i]) - torch.matmul(
                torch.triu(torch.matmul(activations_and_output[i + 1].T, activations_and_output[i + 1])),
                parameter,
            )  # Upper Diag Oja's rule"""
            """update_vector[9] = torch.matmul(activations_and_output[i + 1].T, activations_and_output[i]) * (
                (activations_and_output[i + 1].squeeze(0) - self.bcm_dict[h_name])[:, None]
            )
            self.bcm_dict[h_name] = (
                0.001 * (activations_and_output[i + 1].squeeze(0) ** 2 - self.bcm_dict[h_name])
                + self.bcm_dict[h_name]
            )"""

        return update_vector
