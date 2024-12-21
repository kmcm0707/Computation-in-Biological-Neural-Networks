# Benna and Fusi Synapse
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as functional


class BennaSynapse(nn.Module):
    """
    Benna and Fusi Synapse class.

    """

    def __init__(
        self,
        device: Literal["cpu", "cuda"] = "cpu",
        numberOfChemicals: int = 1,
        non_linearity=torch.nn.functional.tanh,
        options: dict = {},
        params: dict = {},
    ):

        # Initialize the parent class
        super(BennaSynapse, self).__init__()

        # Set the device
        self.device = device

        # Model
        self.numberOfChemicals = numberOfChemicals
        self.non_linearity = non_linearity
        self.all_meta_parameters = nn.ParameterList([])  # All updatable meta-parameters except bias
        self.bias_dictionary = torch.nn.ParameterDict()  # All bias parameters
        self.all_bias_parameters = nn.ParameterList([])  # All bias parameters if they are used
        self.P_matrix = nn.Parameter()

        # Options
        self.options = options

        self.update_rules = [False] * 10
        if self.mode == "rosenbaum":
            self.update_rules[0] = True
            self.update_rules[2] = True
            self.update_rules[9] = True
        elif self.mode == "all_rosenbaum":
            self.update_rules = [True] * 10
        else:
            if "update_rules" in self.options:
                for i in self.options["update_rules"]:
                    self.update_rules[i] = True
            else:
                self.update_rules = [True] * 10

        self.init_parameters(params=params)

    @torch.no_grad()
    def init_parameters(self, params: dict):
        """
        Initialize the model parameters.
        :param params: (dict) model parameters - dimension (W_1, W_2) (per parameter),
        """
        ## Initialize the Bias
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

        if "bias" in self.options:
            if self.options["bias"] == True:
                self.all_bias_parameters.extend(self.bias_dictionary.values())

        min_tau = 1  # Minimum time constant (first beaker)
        if "min_tau" in self.options:
            min_tau = self.options["min_tau"]
        max_tau = 50  # Maximum time constant (last beaker)

        base = max_tau / min_tau
        self.tau = base ** (1 / (2 * self.numberOfChemicals - 2))
        self.C_vector = torch.tensor([self.tau**i for i in range(self.numberOfChemicals)], device=self.device)
        self.G_vector = torch.tensor(
            [1 / min_tau * self.tau ** (i) for i in range(self.numberOfChemicals + 1)],
            device=self.device,
        )

        self.P_matrix = nn.Parameter(
            torch.nn.init.zeros_(torch.empty(size=(self.number_chemicals, 10), device=self.device))
        )
        self.P_matrix[:, 0] = 1e-3
        self.all_meta_parameters.append(self.P_matrix)

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
            activations_and_output[i] = activations_and_output[i] / torch.norm(activations_and_output[i], p=2)"""

        i = 0
        for name, parameter in params.items():
            if "forward" in name:
                h_name = name.replace("forward", "chemical").split(".")[0]
                chemical = h_parameters[h_name]
                if parameter.adapt and "weight" in name:
                    # Equation 1: h(s+1) = yh(s) + zf(Kh(s) + \theta * F(Parameter) + b)
                    # Equation 2: w(s) = v * h(s)
                    update_vector = self.calculate_update_vector(error, activations_and_output, parameter, i)
                    inChange = self.non_linearity(torch.einsum("ci,ijk->cjk", self.P_matrix, update_vector))
                    inFlows = self.G_vector[0 : self.numberOfChemicals - 1] * (
                        chemical[0 : self.numberOfChemicals - 1] - self.C_vector[1 : self.numberOfChemicals]
                    )
                    backFlows = self.G_vector[1 : self.numberOfChemicals] * (
                        chemical[1 : self.numberOfChemicals] - self.C_vector[0 : self.numberOfChemicals - 1]
                    )
                    outChange = self.G_vector[self.numberOfChemicals] * -(chemical[self.numberOfChemicals - 1])

                    newChemical = chemical
                    newChemical[0] = newChemical[0] + (inChange + backFlows[0]) / self.C_vector[0]
                    newChemical[1 : self.numberOfChemicals - 1] = (
                        newChemical[1 : self.numberOfChemicals - 1]
                        + (inFlows + backFlows) / self.C_vector[1 : self.numberOfChemicals - 1]
                    )
                    newChemical[self.numberOfChemicals - 1] = (
                        newChemical[self.numberOfChemicals - 1]
                        + (outChange + inFlows[self.numberOfChemicals - 1]) / self.C_vector[self.numberOfChemicals - 1]
                    )
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
            update_vector[9] = torch.matmul(
                activations_and_output[i + 1].T, activations_and_output[i]
            ) - self.oja_minus_parameter * torch.matmul(
                torch.matmul(activations_and_output[i + 1].T, activations_and_output[i + 1]),
                parameter,
            )  # Oja's rule

        return update_vector
