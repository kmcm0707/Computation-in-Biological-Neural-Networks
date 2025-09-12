from typing import Literal

import torch
from options.complex_options import operatorEnum
from options.LSTM_options import lstmOptions
from torch import nn


class LSTMSynapse(nn.Module):
    """
    LSTM synapse model.
    The class implements a LSTM synapse model.
    Which is a biological plausible update rule.
    """

    def __init__(
        self,
        device="cpu",
        numberOfChemicals: int = 1,
        params: dict = {},
        lstmOptions: lstmOptions = None,
        adaptionPathway: Literal["forward", "feedback"] = "forward",
    ):
        """
        Initialize the LSTM synapse model.
        :param device: (str) The processing device to use. Default is 'cpu'
        :param numberOfChemicals: (int) The number of chemicals to use. Default is 1,
        :param params: (dict) The model parameters. Default is an empty dictionary,
        :param lstmOptions: (lstmOptions) The complex options. Default is None,
        :param adaptionPathway: (str) The adaption pathway to use. Default is 'forward'.
        """
        super(LSTMSynapse, self).__init__()

        self.device = device
        self.options = lstmOptions
        self.adaptionPathway = adaptionPathway

        self.all_meta_parameters = nn.ParameterList([])
        self.number_chemicals = numberOfChemicals  # L

        self.update_rules = [False] * 10
        self.total_update_rules = 0

        for i in self.options.update_rules:
            self.update_rules[i] = True
        self.total_update_rules = sum(self.update_rules)

        self.saved_norm = {}
        self.cell_state = {}
        self.time_index = 0

        self.init_parameters(params=params)

    @torch.no_grad()
    def init_parameters(self, params: dict = {}):
        """
        Initialize the parameters of the LSTM synapse model.
        """

        self.lstm_cell = nn.LSTMCell(
            input_size=self.total_update_rules, hidden_size=self.number_chemicals, bias=True, device=self.device
        )

        # Set up gate biases of input b_ii|b_if|b_ig|b_io
        self.lstm_cell.bias_ih[0 * self.number_chemicals : self.number_chemicals] = 0
        self.lstm_cell.bias_ih[2 * self.number_chemicals : 3 * self.number_chemicals] = 0  # Input gate bias set to 0
        self.lstm_cell.bias_ih[self.number_chemicals : 2 * self.number_chemicals] = (
            torch.randn(self.number_chemicals) * 0.1 + 1
        )  # Forget gate bias set to 1

        # Set up gate biases of hidden state b_hi|b_hf|b_hg|b_ho
        self.lstm_cell.bias_hh[0 : self.number_chemicals] = 0  # Input, forget, cell gate, output gate bias set to 0

        # Set up weights of input W_ii|W_if|W_ig|W_io
        self.lstm_cell.weight_ih[2 * self.number_chemicals : 3 * self.number_chemicals, :] = (
            0  # Input gate weights set to 0
        )
        self.lstm_cell.weight_ih[2 * self.number_chemicals : 3 * self.number_chemicals, 0] = (
            1e-3  # Input gate weights set to small value for first input
        )

        # Set up weights of hidden state W_hi|W_hf|W_hg|W_ho
        self.lstm_cell.weight_hh[: self.number_chemicals, :] = 0  # Input gate weights set to 0

        self.all_meta_parameters.append(self.lstm_cell)

        self.v_vector = nn.Parameter(
            torch.nn.init.ones_(torch.empty(size=(1, self.number_chemicals), device=self.device))
            / self.number_chemicals
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
        override_adaption_pathway: Literal["forward", "feedback", None] = None,
    ):

        i = 0
        currentAdaptionPathway = self.adaptionPathway
        if override_adaption_pathway is not None:
            currentAdaptionPathway = override_adaption_pathway
        for name, parameter in params.items():
            if currentAdaptionPathway in name:

                h_name = name.replace(currentAdaptionPathway, "chemical").split(".")[0]
                if currentAdaptionPathway == "feedback":
                    h_name = "feedback_" + h_name

                chemical = h_parameters[h_name]
                parameter_indices = parameter.shape[0] * parameter.shape[1]
                chemical = torch.reshape(chemical, (parameter_indices, self.number_chemicals))
                if parameter.adapt == currentAdaptionPathway and "weight" in name:
                    update_vector = self.calculate_update_vector(error, activations_and_output, parameter, i, h_name)
                    update_vector = torch.reshape(update_vector, (parameter_indices, self.total_update_rules))

                    # print(chemical[0])
                    # print(self.cell_state[h_name][0])

                    new_chemical = None

                    new_chemical, self.cell_state[h_name] = self.lstm_cell(update_vector, (chemical, chemical))
                    # print(new_chemical[0])
                    # print(self.cell_state[h_name][0])
                    new_chemical = torch.reshape(
                        new_chemical, (self.number_chemicals, parameter.shape[0], parameter.shape[1])
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

                    h_parameters[h_name] = new_chemical
                    v_vector_softmax = torch.nn.functional.softmax(self.v_vector, dim=1)
                    new_value = torch.einsum("ci,ijk->cjk", v_vector_softmax, h_parameters[h_name]).squeeze(0)
                    if self.operator == operatorEnum.mode_6:
                        parameter_norm = self.saved_norm[h_name]
                        current_norm = torch.norm(new_value, p=2)
                        multiplier = parameter_norm / current_norm
                        new_value = new_value * multiplier
                    elif self.operator == operatorEnum.mode_7:
                        new_value = torch.nn.functional.normalize(new_value, p=2, dim=0)

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
        if override_adaption_pathway is not None:
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
                    self.cell_state[h_name] = torch.zeros(
                        (parameter.shape[0] * parameter.shape[1], self.number_chemicals), device=self.device
                    )
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
        update_vector = torch.zeros(
            (self.total_update_rules, parameter.shape[0], parameter.shape[1]), device=self.device
        )

        index_update = 0
        if self.update_rules[0]:
            update_vector[index_update] = -torch.matmul(error[i + 1].T, activations_and_output[i])  # Pseudo-gradient
            index_update += 1

        if self.update_rules[1]:
            update_vector[index_update] = -torch.matmul(activations_and_output[i + 1].T, error[i])
            index_update += 1

        if self.update_rules[2]:
            update_vector[index_update] = -(
                torch.matmul(error[i + 1].T, error[i])
                - torch.matmul(
                    parameter,
                    torch.matmul(error[i].T, error[i]),
                )
            )  # eHebb rule
            index_update += 1

        if self.update_rules[3]:
            update_vector[index_update] = -parameter
            index_update += 1

        if self.update_rules[4]:
            update_vector[index_update] = -torch.matmul(
                torch.ones(size=(parameter.shape[0], 1), device=self.device), error[i]
            )
            index_update += 1

        if self.update_rules[5]:
            normalised_weight = torch.nn.functional.normalize(parameter, p=2, dim=0)
            squeeze_activations = activations_and_output[i].squeeze(0)
            normalised_activation = torch.nn.functional.normalize(squeeze_activations, p=2, dim=0)
            output = torch.nn.functional.softplus(
                torch.matmul(normalised_activation, normalised_weight.T),
                beta=10.0,
            )
            softmax_output = torch.nn.functional.softmax(output, dim=0)
            diff = normalised_weight - normalised_activation
            update_vector[index_update] = -(diff * softmax_output[:, None])
            index_update += 1

        if self.update_rules[6]:
            update_vector[index_update] = -torch.matmul(
                torch.ones(size=(parameter.shape[0], 1), device=self.device), activations_and_output[i]
            )
            index_update += 1

        if self.update_rules[7]:
            update_vector[index_update] = -torch.matmul(
                torch.matmul(
                    torch.matmul(
                        torch.matmul(error[i + 1].T, activations_and_output[i + 1]),
                        parameter,
                    ),
                    error[i].T,
                ),
                activations_and_output[i],
            )  # - Maybe be bad
            index_update += 1

        if self.update_rules[8]:
            update_vector[index_update] = -torch.matmul(
                torch.matmul(
                    torch.matmul(
                        torch.matmul(activations_and_output[i + 1].T, activations_and_output[i]),
                        parameter.T,
                    ),
                    error[i + 1].T,
                ),
                error[i],
            )
            index_update += 1

        if self.update_rules[9]:
            update_vector[index_update] = torch.matmul(
                activations_and_output[i + 1].T, activations_and_output[i]
            ) - torch.matmul(
                torch.matmul(activations_and_output[i + 1].T, activations_and_output[i + 1]),
                parameter,
            )  # Oja's rule
            index_update += 1

        return update_vector
