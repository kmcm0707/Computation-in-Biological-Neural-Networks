import torch
from options.complex_options import operatorEnum, yVectorEnum, zVectorEnum
from options.kernel_rnn_options import kernelRnnOptions
from torch import nn


class KernelRnn(nn.Module):
    """
    Complex rnn model.
    The class implements a complex synapse rnn model.
    Which is a biological plausible update rule.
    """

    def __init__(
        self,
        device="cpu",
        numberOfSlowChemicals: int = 1,
        params: dict = {},
        kernelRnnOptions: kernelRnnOptions = None,
        conversion_matrix: dict = {},
    ):
        """
        Initialize the complex synapse model.
        :param device: (str) The processing device to use. Default is 'cpu',
        :param mode: (str) The update rule to use. Default is 'rosenbaum'.
        :param numberOfChemicals: (int) The number of chemicals to use. Default is 1,
        :param params: (dict) The model parameters. Default is an empty dictionary,
        :param complexOptions: (complexOptions) The complex options. Default is None,
        """
        super(KernelRnn, self).__init__()

        self.device = device
        self.numberOfSlowChemicals = numberOfSlowChemicals

        self.all_meta_parameters = nn.ParameterList([])
        self.non_linearity = kernelRnnOptions.nonLinear
        self.conversion_matrix = conversion_matrix
        self.options = kernelRnnOptions
        self.time_lag_covariance = kernelRnnOptions.time_lag_covariance
        self.rflo_tau = 30

        self.init_slow_params()
        self.init_fast_params(params)

    @torch.no_grad()
    def init_slow_params(self):
        """
        Initialize the model parameters.
        :param params: (dict) The model parameters.
        """
        self.numberUpdateRules = 14
        self.update_rules = [False] * self.numberUpdateRules
        trueNumberUpdateRules = 0
        for i in self.options.update_rules:
            trueNumberUpdateRules += 1
            self.update_rules[i] = True
        self.numberUpdateRules = trueNumberUpdateRules

        if self.time_lag_covariance is None:
            self.Q_matrix = nn.Parameter(
                torch.nn.init.zeros_(
                    torch.empty(size=(self.numberOfSlowChemicals, self.numberUpdateRules * 2), device=self.device)
                )
            )
            self.Q_matrix[:, 0] = 1e-3
        else:
            self.Q_matrix = nn.Parameter(
                torch.nn.init.zeros_(
                    torch.empty(size=(self.numberOfSlowChemicals, self.numberUpdateRules * 3), device=self.device)
                )
            )
            self.Q_matrix[:, 0] = 1e-3

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
        self.all_meta_parameters.append(self.Q_matrix)

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

        if self.numberOfSlowChemicals == 1:
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

        self.z_vector = self.z_vector.to(self.device)
        self.z_vector = nn.Parameter(self.z_vector)
        self.y_vector = self.y_vector.to(self.device)
        self.y_vector = nn.Parameter(self.y_vector)

        self.slow_operator = self.options.slow_operator

    @torch.no_grad()
    def init_fast_params(self, params: dict):
        """
        Initialize the model parameters.
        :param params: (dict) The model parameters.
        """
        self.mean_update = {}
        self.variance_update = {}
        self.past_updates = {}
        if 13 in self.options.update_rules:
            self.rflo = {}
        for name, parameter in params:
            if name in self.conversion_matrix:
                h_name = self.conversion_matrix[name]
                h_slow_name = "slow_" + h_name
                self.mean_update[h_slow_name] = torch.nn.init.zeros_(
                    torch.empty(
                        size=(self.numberUpdateRules, parameter.shape[0], parameter.shape[1]),
                        device=self.device,
                        requires_grad=True,
                    )
                )
                self.variance_update[h_slow_name] = torch.nn.init.zeros_(
                    torch.empty(
                        size=(self.numberUpdateRules, parameter.shape[0], parameter.shape[1]),
                        device=self.device,
                        requires_grad=True,
                    )
                )
                if self.options.time_lag_covariance is not None:
                    self.past_updates[h_slow_name] = torch.nn.init.zeros_(
                        torch.empty(
                            size=(
                                self.time_lag_covariance,
                                self.numberUpdateRules,
                                parameter.shape[0],
                                parameter.shape[1],
                            ),
                            device=self.device,
                        )
                    )
                    self.past_variance = {}
                    self.past_variance[h_slow_name] = torch.nn.init.zeros_(
                        torch.empty(
                            size=(self.numberUpdateRules, parameter.shape[0], parameter.shape[1]),
                            device=self.device,
                        )
                    )
                    if 13 in self.options.update_rules:
                        self.rflo[h_slow_name] = torch.nn.init.zeros_(
                            torch.empty(
                                size=(1, parameter.shape[1]),
                                device=self.device,
                            )
                        )
        self.time_index = 0

    @torch.no_grad()
    def reset_fast_chemicals(self, params: dict):
        """
        Reset the fast chemicals.
        :param h_fast_parameters: (dict) The fast chemicals.
        """
        for name, parameter in params.items():
            if name in self.conversion_matrix:
                h_name = self.conversion_matrix[name]
                h_slow_name = "slow_" + h_name
                self.mean_update[h_slow_name] = torch.nn.init.zeros_(
                    torch.empty(
                        size=(self.numberUpdateRules, parameter.shape[0], parameter.shape[1]),
                        device=self.device,
                        requires_grad=True,
                    )
                )
                self.variance_update[h_slow_name] = torch.nn.init.zeros_(
                    torch.empty(
                        size=(self.numberUpdateRules, parameter.shape[0], parameter.shape[1]),
                        device=self.device,
                        requires_grad=True,
                    )
                )
                if self.options.time_lag_covariance is not None:
                    self.past_updates[h_slow_name] = torch.nn.init.zeros_(
                        torch.empty(
                            size=(
                                self.time_lag_covariance,
                                self.numberUpdateRules,
                                parameter.shape[0],
                                parameter.shape[1],
                            ),
                            device=self.device,
                        )
                    )
                    self.past_variance[h_slow_name] = torch.nn.init.zeros_(
                        torch.empty(
                            size=(self.numberUpdateRules, parameter.shape[0], parameter.shape[1]),
                            device=self.device,
                        )
                    )
                if 13 in self.options.update_rules:
                    self.rflo[h_slow_name] = torch.nn.init.zeros_(
                        torch.empty(
                            size=(1, parameter.shape[1]),
                            device=self.device,
                        )
                    )
        self.time_index = 0

    def slow_update(
        self,
        params: dict,
        h_slow_parameters: dict,
    ):
        i = 0

        for name, parameter in params.items():
            if name in self.conversion_matrix:
                h_name = self.conversion_matrix[name]
                h_slow_name = "slow_" + h_name
                chemical = h_slow_parameters[h_slow_name]

                """self.mean_update[h_slow_name][3, :, :] = self.mean_update[h_slow_name][3, :, :] / self.time_index
                self.mean_update[h_slow_name][5, :, :] = self.mean_update[h_slow_name][5, :, :] / self.time_index
                self.mean_update[h_slow_name][9, :, :] = self.mean_update[h_slow_name][9, :, :] / self.time_index

                self.variance_update[h_slow_name][3, :, :] = (
                    self.variance_update[h_slow_name][3, :, :] / self.time_index
                )
                self.variance_update[h_slow_name][5, :, :] = (
                    self.variance_update[h_slow_name][5, :, :] / self.time_index
                )
                self.variance_update[h_slow_name][9, :, :] = (
                    self.variance_update[h_slow_name][9, :, :] / self.time_index
                )"""

                """self.mean_update[h_slow_name] = self.mean_update[h_slow_name] / self.time_index
                self.variance_update[h_slow_name] = self.variance_update[h_slow_name] / self.time_index

                self.variance_update[h_slow_name] = (
                    self.variance_update[h_slow_name] - self.mean_update[h_slow_name] ** 2
                )"""
                self.variance_update[h_slow_name] = (self.variance_update[h_slow_name] / self.time_index) - (
                    self.mean_update[h_slow_name] ** 2
                )
                if self.options.time_lag_covariance is None:
                    update = torch.cat((self.mean_update[h_slow_name], self.variance_update[h_slow_name]), dim=0)
                else:
                    self.past_updates[h_slow_name] = (
                        self.past_updates[h_slow_name] / self.time_index - self.mean_update[h_slow_name] ** 2
                    )
                    update = torch.cat(
                        (
                            self.mean_update[h_slow_name],
                            self.variance_update[h_slow_name],
                            self.past_variance[h_slow_name],
                        ),
                        dim=0,
                    )
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
                            + torch.einsum("ci,ijk->cjk", self.Q_matrix, update)
                        ),
                    )
                    if self.slow_operator == operatorEnum.mode_5 or self.slow_operator == operatorEnum.mode_6:
                        parameter_norm = self.saved_norm[h_slow_name]
                        chemical_norms = torch.norm(new_chemical, dim=(1, 2))
                        multiplier = parameter_norm / (chemical_norms)
                        new_chemical = new_chemical * multiplier[:, None, None]
                    elif self.slow_operator == operatorEnum.mode_7:
                        new_chemical = torch.nn.functional.normalize(new_chemical, p=2, dim=1)
                else:
                    raise ValueError("Invalid operator")

                h_slow_parameters[h_slow_name] = new_chemical
                if (
                    self.slow_operator == operatorEnum.mode_4
                    or self.slow_operator == operatorEnum.mode_5
                    or self.slow_operator == operatorEnum.mode_6
                    or self.slow_operator == operatorEnum.mode_7
                ):
                    v_vector_softmax = torch.nn.functional.softmax(self.v_vector, dim=1)
                    new_value = torch.einsum("ci,ijk->cjk", v_vector_softmax, h_slow_parameters[h_slow_name]).squeeze(0)
                    if self.slow_operator == operatorEnum.mode_6:
                        parameter_norm = self.saved_norm[h_slow_name]
                        current_norm = torch.norm(new_value, p=2)
                        multiplier = parameter_norm / current_norm
                        new_value = new_value * multiplier
                    elif self.slow_operator == operatorEnum.mode_7:
                        new_value = torch.nn.functional.normalize(new_value, p=2, dim=0)
                else:
                    new_value = torch.einsum("ci,ijk->cjk", self.v_vector, h_slow_parameters[h_slow_name]).squeeze(0)

                params[name] = new_value
                i += 1

    def fast_update(
        self,
        params: dict,
        activations_and_output: dict,
        error: dict,
    ):
        """
        Fast update.
        The function updates the fast synapse.
        :param params: (dict) The model parameters,
        :param h_parameters: (dict) The hidden parameters,
        :param activations_and_output: (dict) The activations and output,
        :param error: (dict) The error (atm not used).
        """
        self.time_index += 1
        for name, parameter in params.items():
            if name in self.conversion_matrix:
                errors = error[name]
                activations_and_outputs = activations_and_output[name]
                h_name = self.conversion_matrix[name]
                h_slow_name = "slow_" + h_name
                update_vector = self.calculate_update_vector(errors, activations_and_outputs, parameter, h_slow_name)
                mean_removed_update_vector = update_vector - self.mean_update[h_slow_name]
                self.mean_update[h_slow_name] = self.mean_update[h_slow_name] + (
                    mean_removed_update_vector / self.time_index
                )
                self.variance_update[h_slow_name] = self.variance_update[h_slow_name] + update_vector**2
                if self.options.time_lag_covariance is not None and self.time_index > self.time_lag_covariance:
                    self.past_variance[h_slow_name] = self.past_variance[h_slow_name] + (
                        self.past_updates[h_slow_name][0, :, :, :] * update_vector
                    )
                if self.options.time_lag_covariance is not None:
                    self.past_updates[h_slow_name] = torch.roll(self.past_updates[h_slow_name], shifts=-1, dims=0)
                    self.past_updates[h_slow_name][-1, :, :, :] = update_vector

    @torch.no_grad()
    def reset_time_index(self):
        self.time_index = 0

    @torch.no_grad()
    def initial_update(self, params: dict, h_slow_parameters: dict):
        """
        :param params: (dict) model weights - dimension (W_1, W_2) (per parameter),
        :param h_parameters: (dict) model chemicals - dimension L x (W_1, W_2) (per parameter),

        To connect the forward and chemical parameters.
        """
        self.saved_norm = {}
        for name, parameter in params.items():
            if name in self.conversion_matrix:
                h_name = self.conversion_matrix[name]
                h_slow_name = "slow_" + h_name
                # Equation 2: w(s) = v * h(s)
                # if self.operator == operatorEnum.mode_7:
                self.saved_norm[h_slow_name] = torch.norm(parameter, p=2)
                new_value = torch.einsum("ci,ijk->cjk", self.v_vector, h_slow_parameters[h_slow_name]).squeeze(0)
                if (
                    self.slow_operator == operatorEnum.mode_5
                    or self.slow_operator == operatorEnum.mode_6
                    or self.slow_operator == operatorEnum.mode_7
                ):
                    parameter_norm = self.saved_norm[h_slow_name]
                    current_norm = torch.norm(new_value, p=2)
                    multiplier = parameter_norm / current_norm
                    new_value = new_value * multiplier
                params[name] = new_value

    def calculate_update_vector(self, error, activations_and_output, parameter, h_slow_name) -> torch.Tensor:
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
        update_vector = torch.zeros(
            (self.numberUpdateRules, parameter.shape[0], parameter.shape[1]), device=self.device
        )

        i = 0
        # with torch.no_grad():
        if self.update_rules[0]:
            update_vector[i] = -torch.matmul(error_below.T, activation_above)  # Pseudo-gradient
            i += 1

        if self.update_rules[1]:
            update_vector[i] = -torch.matmul(activation_below.T, error_above)
            i += 1

        if self.update_rules[2]:
            update_vector[i] = -(torch.matmul(error_below.T, error_above))  # eHebb rule
            i += 1

        if self.update_rules[3]:
            update_vector[i] = -parameter
            i += 1

        if self.update_rules[4]:
            update_vector[i] = -torch.matmul(torch.ones(size=(parameter.shape[0], 1), device=self.device), error_above)
            i += 1

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
            update_vector[i] = -(diff * softmax_output[:, None])
            i += 1

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
            update_vector[i] = -torch.matmul(
                torch.matmul(
                    torch.matmul(
                        torch.matmul(error_below.T, activation_below),
                        parameter,
                    ),
                    error_above.T,
                ),
                activation_above,
            )  # - Maybe be bad
            i += 1

        if self.update_rules[8]:
            update_vector[i] = -torch.matmul(
                torch.matmul(
                    torch.matmul(
                        torch.matmul(activation_below.T, activation_above),
                        parameter.T,
                    ),
                    error_below.T,
                ),
                error_above,
            )
            i += 1

        if self.update_rules[9]:
            update_vector[i] = torch.matmul(activation_below.T, activation_above) - torch.matmul(
                torch.matmul(activation_below.T, activation_below),
                parameter,
            )  # Oja's rule
            i += 1

        if self.update_rules[10]:
            update_vector[i] = -torch.matmul(
                error_below.T, torch.ones(size=(1, parameter.shape[1]), device=self.device)
            )
            i += 1

        if self.update_rules[11]:
            update_vector[i] = -(
                torch.matmul(activation_below.T, torch.ones(size=(1, parameter.shape[1]), device=self.device))
                - torch.matmul(
                    torch.matmul(activation_below.T, activation_below),
                    parameter,
                )
            )
            i += 1

        if self.update_rules[12]:
            update_vector[i] = -torch.matmul(
                torch.ones(size=(parameter.shape[0], 1), device=self.device), activation_above
            )
            i += 1

        if self.update_rules[13]:
            diff_activation_above = 1 - torch.exp(-10 * activation_above)
            self.rflo[h_slow_name] = (1 - 1 / self.rflo_tau) * self.rflo[h_slow_name] + (
                1 / self.rflo_tau
            ) * diff_activation_above * activation_above
            update_vector[i] = -torch.matmul(error_below.T, self.rflo[h_slow_name])
            i += 1

        return update_vector
