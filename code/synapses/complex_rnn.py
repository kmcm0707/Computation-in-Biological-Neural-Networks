from typing import Literal

import torch
from options.complex_options import operatorEnum, yVectorEnum, zVectorEnum
from options.complex_rnn_options import complexRnnOptions
from torch import nn


class ComplexRnn(nn.Module):
    """
    Complex rnn model.
    The class implements a complex synapse rnn model.
    Which is a biological plausible update rule.
    """

    def __init__(
        self,
        device="cpu",
        numberOfSlowChemicals: int = 1,
        numberOfFastChemicals: int = 1,
        params: dict = {},
        complexRnnOptions: complexRnnOptions = None,
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
        super(ComplexRnn, self).__init__()

        self.device = device
        self.adaptionPathway = adaptionPathway
        self.numberOfSlowChemicals = numberOfSlowChemicals
        self.numberOfFastChemicals = numberOfFastChemicals
        self.params = params

        self.all_meta_parameters = nn.ParameterList([])
        self.non_linearity = complexRnnOptions.nonLinear
        self.options = complexRnnOptions

        self.init_slow_params(params)
        self.init_fast_params(params)

    @torch.no_grad()
    def init_slow_params(self, params: dict):
        """
        Initialize the model parameters.
        :param params: (dict) The model parameters.
        """
        self.params = params
        self.numberUpdateRules = 12
        self.update_rules = [False] * self.numberUpdateRules
        for i in self.options.slow_update_rules:
            self.update_rules[i] = True

        self.K_matrix = nn.Parameter()  # K - LxL
        self.v_vector = nn.Parameter()  # v - L
        self.Q_matrix = nn.Parameter()  # \theta - Lx10

        self.Q_matrix = nn.Parameter(
            torch.nn.init.zeros_(
                torch.empty(size=(self.numberOfFastChemicals, self.numberOfFastChemicals), device=self.device)
            )
        )

        self.K_slow_matrix = nn.Parameter(
            torch.nn.init.zeros_(
                torch.empty(
                    size=(self.numberOfSlowChemicals, self.numberOfSlowChemicals),
                    device=self.device,
                )
            )
        )

        self.v_vector = nn.Parameter(
            torch.nn.init.ones_(torch.empty(size=(1, self.numberOfSlowChemicals), device=self.device))
            / self.numberOfSlowChemicals
        )

        self.all_meta_parameters.append(self.K_slow_matrix)
        self.all_meta_parameters.append(self.v_vector)
        self.all_meta_parameters.append(self.P_matrix)

        ## Initialize the chemical time constants
        # z = 1 / \tau
        min_tau = self.options.minSlowTau
        max_tau = self.options.maxSlowTau
        base = max_tau / min_tau

        self.tau_vector = min_tau * (base ** torch.linspace(0, 1, self.numberOfSlowChemicals))
        self.z_vector = 1 / self.tau_vector
        self.y_vector = 1 - self.z_vector

        # self.y_vector = 1 / self.tau_vector
        # self.z_vector = 1 - self.y_vector
        # self.z_vector[0] = 1

        if self.options.z_vector == zVectorEnum.random:
            self.z_vector = nn.Parameter(
                torch.nn.init.normal_(
                    torch.empty(size=(1, self.numberOfSlowChemicals), device=self.device),
                    mean=0,
                    std=0.01,
                )
            )
        elif self.options.z_vector == zVectorEnum.all_ones:
            self.z_vector = torch.ones(self.numberOfSlowChemicals, device=self.device)
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
            self.y_vector = torch.ones(self.numberOfSlowChemicals, device=self.device)
        elif self.options.y_vector == yVectorEnum.half:
            self.y_vector[-1] = 0.5

        self.z_vector = nn.Parameter(self.z_vector)
        self.z_vector = self.z_vector.to(self.device)
        self.y_vector = nn.Parameter(self.y_vector)
        self.y_vector = self.y_vector.to(self.device)

        self.slow_operator = self.options.slow_operator

    def slow_update(
        self,
        params: dict,
        h_slow_parameters: dict,
        h_fast_parameters: dict,
        conversion_matrix: dict,
    ):
        i = 0
        currentAdaptionPathway = self.adaptionPathway
        for name, parameter in params.items():
            if name in conversion_matrix:
                h_name = self.conversion_matrix[name]
                h_fast_name = "fast_" + h_name
                h_slow_name = "slow_" + h_name

                chemical = h_slow_parameters[h_slow_name]
                chemical_fast = h_fast_parameters[h_fast_name]
                # Equation 1: h(s+1) = yh(s) + (1/\eta) * zf(Kh(s) + Qr )
                # Equation 2: w(s) = v * h(s)

                new_chemical = None
                if (
                    self.slow_operator == operatorEnum.mode_4
                    or self.slow_operator == operatorEnum.mode_5
                    or self.slow_operator == operatorEnum.mode_6
                    or self.slow_operator == operatorEnum.mode_7
                ):
                    new_chemical = torch.einsum("i,ijk->ijk", self.y_vector, chemical) + torch.einsum(
                        "i,ijk->ijk",
                        self.z_vector,
                        self.non_linearity(
                            torch.einsum("ic,ijk->cjk", self.K_slow_matrix, chemical)
                            + torch.einsum(
                                "ic,ijk->cjk",
                                self.Q_matrix,
                                chemical_fast,
                            )
                        ),
                    )
                    if self.slow_operator == operatorEnum.mode_5 or self.slow_operator == operatorEnum.mode_6:
                        parameter_norm = self.saved_norm[h_name]
                        chemical_norms = torch.norm(new_chemical, dim=(1, 2))
                        multiplier = parameter_norm / (chemical_norms)
                        new_chemical = new_chemical * multiplier[:, None, None]
                    elif self.slow_operator == operatorEnum.mode_7:
                        new_chemical = torch.nn.functional.normalize(new_chemical, p=2, dim=1)
                else:
                    raise ValueError("Invalid operator")

                h_slow_parameters[h_name] = new_chemical
                if (
                    self.slow_operator == operatorEnum.mode_4
                    or self.slow_operator == operatorEnum.mode_5
                    or self.slow_operator == operatorEnum.mode_6
                    or self.slow_operator == operatorEnum.mode_7
                ):
                    v_vector_softmax = torch.nn.functional.softmax(self.v_vector, dim=1)
                    new_value = torch.einsum("ci,ijk->cjk", v_vector_softmax, h_slow_parameters[h_name]).squeeze(0)
                    if self.slow_operator == operatorEnum.mode_6:
                        parameter_norm = self.saved_norm[h_name]
                        current_norm = torch.norm(new_value, p=2)
                        multiplier = parameter_norm / current_norm
                        new_value = new_value * multiplier
                    elif self.slow_operator == operatorEnum.mode_7:
                        new_value = torch.nn.functional.normalize(new_value, p=2, dim=0)
                else:
                    new_value = torch.einsum("ci,ijk->cjk", self.v_vector, h_slow_parameters[h_name]).squeeze(0)

                params[name] = new_value

                params[name].adapt = currentAdaptionPathway
                i += 1

    @torch.no_grad()
    def init_fast_params(self, params: dict):
        """
        Initialize the model parameters.
        :param params: (dict) The model parameters.
        """
        self.params = params
        self.numberUpdateRules = 12
        self.update_rules = [False] * self.numberUpdateRules
        for i in self.options.fast_update_rules:
            self.update_rules[i] = True

        self.K_fast_matrix = nn.Parameter()  # K - LxL
        self.P_matrix = nn.Parameter()  # P - Lx10
        self.y_fast_vector = nn.Parameter()  # y - L
        self.z_fast_vector = nn.Parameter()  # z - L

        self.K_fast_matrix = nn.Parameter(
            torch.nn.init.zeros_(
                torch.empty(size=(self.numberOfFastChemicals, self.numberOfFastChemicals), device=self.device)
            )
        )
        self.P_matrix = nn.Parameter(
            torch.nn.init.zeros_(
                torch.empty(size=(self.numberOfFastChemicals, self.numberUpdateRules), device=self.device)
            )
        )
        self.P_matrix[:, 0] = 1e-3

        min_tau = self.options.minFastTau
        max_tau = self.options.maxFastTau
        base = max_tau / min_tau

        self.tau_vector = min_tau * (base ** torch.linspace(0, 1, self.numberOfSlowChemicals))
        self.z_fast_vector = 1 / self.tau_vector
        self.y_fast_vector = 1 - self.z_vector

        if self.options.z_fast_vector == zVectorEnum.all_ones:
            self.z_fast_vector = torch.ones(self.numberOfFastChemicals, device=self.device)

        self.y_fast_vector = nn.Parameter(self.y_fast_vector)
        self.y_fast_vector = self.y_fast_vector.to(self.device)
        self.z_fast_vector = nn.Parameter(self.z_fast_vector)
        self.z_fast_vector = self.z_fast_vector.to(self.device)

        self.fast_operator = self.options.fast_operator

    def fast_update(
        self,
        params: dict,
        h_fast_parameters: dict,
        activations_and_output: dict,
        error: dict,
        conversion_matrix: dict,
    ):
        """
        Fast update.
        The function updates the fast synapse.
        :param params: (dict) The model parameters,
        :param h_parameters: (dict) The hidden parameters,
        :param activations_and_output: (dict) The activations and output,
        :param error: (dict) The error (atm not used).
        """
        for name, parameter in params.items():
            if parameter in conversion_matrix:
                h_name = self.conversion_matrix[name]
                h_name = "fast_" + h_name
                chemical_fast = h_fast_parameters[h_name]
                errors = error[parameter]
                activations_and_outputs = activations_and_output[parameter]
                # Equation 1: h(s+1) = yh(s) +  zf(Kh(s) + PF)

                update_vector = self.calculate_update_vector(errors, activations_and_outputs, parameter)

                new_chemical = None
                if (
                    self.fast_operator == operatorEnum.mode_4
                    or self.fast_operator == operatorEnum.mode_5
                    or self.fast_operator == operatorEnum.mode_6
                    or self.fast_operator == operatorEnum.mode_7
                ):
                    new_chemical = torch.einsum("i,ijk->ijk", self.y_fast_vector, chemical_fast) + torch.einsum(
                        "i,ijk->ijk",
                        self.z_fast_vector,
                        self.non_linearity(
                            torch.einsum("ic,ijk->cjk", self.K_fast_matrix, chemical_fast)
                            + torch.einsum(
                                "ic,ijk->cjk",
                                self.P_matrix,
                                update_vector,
                            )
                        ),
                    )
                    if self.operator == operatorEnum.mode_5 or self.operator == operatorEnum.mode_6:
                        parameter_norm = self.saved_norm[h_name]
                        chemical_norms = torch.norm(new_chemical, dim=(1, 2))
                        multiplier = parameter_norm / (chemical_norms)
                        new_chemical = new_chemical * multiplier[:, None, None]
                    elif self.operator == operatorEnum.mode_7:
                        new_chemical = torch.nn.functional.normalize(
                            new_chemical, p=2, dim=1
                        )  # dim=1 as chemical is 3D
                else:
                    raise ValueError("Invalid operator")
                h_fast_parameters[h_name] = new_chemical

    @torch.no_grad()
    def reset_fast_chemicals(self, h_fast_parameters: dict):
        """
        Reset the fast chemicals.
        :param h_fast_parameters: (dict) The fast chemicals.
        """
        for name, parameter in h_fast_parameters.items():
            h_fast_parameters[name] = torch.zeros_like(parameter)

    def calculate_update_vector(self, error, activations_and_output, parameter) -> torch.Tensor:
        """
        Calculate the update vector for the complex synapse model.
        :param error: (list) model error,
        :param activations_and_output: (list) model activations and output,
        :param parameter: (tensor) model parameter - dimension (W_1, W_2),
        :param i: (int) index of the parameter.
        """
        # error[i] = error_above
        # error[i + 1] = error_below
        # activations_and_output[i] = activation_above
        # activations_and_output[i + 1] = activation_below
        error_below = error[1]
        error_above = error[0]
        activation_above = activations_and_output[0]
        activation_below = activations_and_output[1]
        update_vector = torch.zeros((12, parameter.shape[0], parameter.shape[1]), device=self.device)
        # with torch.no_grad():
        if self.update_rules[0]:
            update_vector[0] = -torch.matmul(error_below.T, activation_above)  # Pseudo-gradient

        if self.update_rules[1]:
            update_vector[1] = -torch.matmul(activation_below.T, error_above)

        if self.update_rules[2]:
            update_vector[2] = -(
                torch.matmul(error_below.T, error_above)
                - torch.matmul(
                    parameter,
                    torch.matmul(error_above.T, error_above),
                )
            )  # eHebb rule

        if self.update_rules[3]:
            update_vector[3] = -parameter

        if self.update_rules[4]:
            update_vector[4] = -torch.matmul(torch.ones(size=(parameter.shape[0], 1), device=self.device), error_above)

        if self.update_rules[5]:
            normalised_weight = torch.nn.functional.normalize(parameter, p=2, dim=0)
            squeeze_activations = activation_above.squeeze(0)
            normalised_activation = torch.nn.functional.normalize(squeeze_activations, p=2, dim=0)
            output = torch.nn.functional.softplus(
                torch.matmul(normalised_activation, normalised_weight.T),
                beta=10.0,
            )
            softmax_output = torch.nn.functional.softmax(output, dim=0)
            diff = normalised_weight - normalised_activation
            update_vector[5] = -(diff * softmax_output[:, None])

        if self.update_rules[6]:
            """update_vector[6] = -torch.matmul(
                torch.matmul(
                    torch.matmul(
                        torch.matmul(
                            activation_below.T,
                            activation_below,
                        ),
                        parameter,
                    ),
                    error_above.T,
                ),
                error_above,
            )  # - ERROR"""
            """update_vector[6] = -torch.matmul(
                torch.nn.functional.sigmoid(activation_below.T),
                torch.nn.functional.sigmoid(activation_above),
            )"""

        if self.update_rules[7]:
            update_vector[7] = -torch.matmul(
                torch.matmul(
                    torch.matmul(
                        torch.matmul(error_below.T, activation_below),
                        parameter,
                    ),
                    error_above.T,
                ),
                activation_above,
            )  # - Maybe be bad

        if self.update_rules[8]:
            update_vector[8] = -torch.matmul(
                torch.matmul(
                    torch.matmul(
                        torch.matmul(activation_below.T, activation_above),
                        parameter.T,
                    ),
                    error_below.T,
                ),
                error_above,
            )

        if self.update_rules[9]:
            update_vector[9] = torch.matmul(activation_below.T, activation_above) - torch.matmul(
                torch.matmul(activation_below.T, activation_below),
                parameter,
            )  # Oja's rule

        if self.update_rules[10]:
            update_vector[10] = -torch.matmul(error_below, torch.ones(size=(1, parameter.shape[0]), device=self.device))

        if self.update_rules[11]:
            update_vector[11] = -torch.matmul(
                activation_below, torch.ones(size=(1, parameter.shape[0]), device=self.device)
            )

        if self.update_rules[12]:
            update_vector[12] = -torch.matmul(
                torch.ones(size=(1, parameter.shape[0]), device=self.device), activation_above
            )

        return update_vector
