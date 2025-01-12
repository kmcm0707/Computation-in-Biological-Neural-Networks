import math

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
from torch.nn import functional


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
    ):
        """
        Initialize the complex synapse model.
        :param device: (str) The processing device to use. Default is 'cpu',
        :param mode: (str) The update rule to use. Default is 'rosenbaum'.
        :param numberOfChemicals: (int) The number of chemicals to use. Default is 1,
        :param options: (dict) The options to use. Default is {}.
        :param params: (dict) The parameters of the normal NN. Default is {}.
        """
        super(ComplexSynapse, self).__init__()

        self.device = device
        self.mode = complexOptions.mode
        self.options = complexOptions

        # h(s+1) = (1-z)h(s) + zf(Kh(s) + \theta * F(Parameter) + b)
        # y = 1-z, y_0 = 1, z_0 = 1
        # w(s) = v * h(s) (if self.number_chemicals = 1)
        self.K_matrix = nn.Parameter()  # K - LxL
        self.v_vector = nn.Parameter()  # v - L
        self.P_matrix = nn.Parameter()  # \theta - Lx10
        self.all_meta_parameters = nn.ParameterList([])  # All updatable meta-parameters except bias
        self.bias_dictionary = torch.nn.ParameterDict()  # All bias parameters
        self.all_bias_parameters = nn.ParameterList([])  # All bias parameters if they are used
        self.number_chemicals = numberOfChemicals  # L

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
        for name, parameter in params:
            if "forward" in name:
                h_name = name.replace("forward", "chemical").split(".")[0]
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

        if self.options.bias:
            self.all_bias_parameters.extend(self.bias_dictionary.values())

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
                self.P_matrix[:, 0] = 0.001

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
            self.all_meta_parameters.append(self.v_vector)

        ## Initialize the mode
        self.operator = self.options.operator

        ## Attention mechanism
        if self.operator == operatorEnum.attention:
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

    def __call__(
        self,
        activations: list,
        output: torch.Tensor,
        label: torch.Tensor,
        params: dict,
        h_parameters: dict,
        beta: int,
    ):
        """
        :param activations: (list) model activations,
        :param output: (tensor) model output - dimension (W_1) (per layer),
        :param label: (tensor) model label - dimension (W_1) (per layer),
        :param params: (dict) model weights - dimension (W_1, W_2) (per parameter),
        :param h_parameters: (dict) model chemicals - dimension L x (W_1, W_2) (per parameter),
        :param beta: (int) smoothness coefficient for non-linearity,
        """

        feedback = {name: value for name, value in params.items() if "feedback" in name}
        error = [functional.softmax(output, dim=1) - functional.one_hot(label, num_classes=47)]
        # add the error for all the layers
        for y, i in zip(reversed(activations), reversed(list(feedback))):
            error.insert(0, torch.matmul(error[0], feedback[i]) * (1 - torch.exp(-beta * y)))
        activations_and_output = [*activations, functional.softmax(output, dim=1)]

        """for i in range(len(activations_and_output)):
            # activations_and_output[i] = activations_and_output[i] / torch.norm(activations_and_output[i], p=2)
            activations_and_output[i] = activations_and_output[i] / (
                torch.max(torch.abs(activations_and_output[i])) + 1e-5
            )

        for i in range(len(error)):
            # error[i] = error[i] / torch.norm(error[i], p=2)
            error[i] = error[i] / (torch.max(torch.abs(error[i])) + 1e-5)"""

        i = 0
        for name, parameter in params.items():
            if "forward" in name:
                h_name = name.replace("forward", "chemical").split(".")[0]
                chemical = h_parameters[h_name]
                if parameter.adapt and "weight" in name:
                    # Equation 1: h(s+1) = yh(s) + zf(Kh(s) + \theta * F(Parameter) + b)
                    # Equation 2: w(s) = v * h(s)
                    update_vector = self.calculate_update_vector(error, activations_and_output, parameter, i)
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
                    ):  # mode 1 - was also called add in results
                        new_chemical = torch.einsum("i,ijk->ijk", self.y_vector, chemical) + torch.einsum(
                            "i,ijk->ijk",
                            self.z_vector,
                            self.non_linearity(
                                torch.einsum("ic,ijk->cjk", self.K_matrix, chemical)
                                + torch.einsum("ci,ijk->cjk", self.P_matrix, update_vector)
                                + self.bias_dictionary[h_name]
                            ),
                        )
                    elif self.operator == operatorEnum.sub:
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
                        new_value = parameter + torch.nn.functional.tanh(
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
                        new_v = torch.reshape(
                            new_v, (self.number_chemicals, h_parameters[h_name].shape[1], h_parameters[h_name].shape[2])
                        )
                        new_value = torch.einsum("ijk,ijk->jk", new_v, h_parameters[h_name])

                    else:
                        new_value = torch.einsum("ci,ijk->cjk", self.v_vector, h_parameters[h_name]).squeeze(0)

                    params[name] = new_value

                    params[name].adapt = True
                i += 1

    @torch.no_grad()
    def initial_update(self, params: dict, h_parameters: dict):
        """
        :param params: (dict) model weights - dimension (W_1, W_2) (per parameter),
        :param h_parameters: (dict) model chemicals - dimension L x (W_1, W_2) (per parameter),

        To connect the forward and chemical parameters.
        """
        for name, parameter in params.items():
            if "forward" in name:
                h_name = name.replace("forward", "chemical").split(".")[0]
                if parameter.adapt and "weight" in name:
                    # Equation 2: w(s) = v * h(s)
                    new_value = torch.einsum("ci,ijk->cjk", self.v_vector, h_parameters[h_name]).squeeze(0)
                    params[name] = new_value

                    params[name].adapt = True

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
            update_vector[5] = -torch.matmul(
                torch.matmul(
                    torch.matmul(
                        error[i + 1].T,
                        torch.ones(size=(1, parameter.shape[0]), device=self.device),
                    ),
                    activations_and_output[i + 1].T,
                ),
                activations_and_output[i],
            )  # = ERROR on high learning rate

        if self.update_rules[6]:
            update_vector[6] = -torch.matmul(
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
            )  # - ERROR

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
