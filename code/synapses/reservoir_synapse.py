# A reservoir Synapse Model
# https://martinuzzifrancesco.github.io/posts/a-brief-introduction-to-reservoir-computing/
import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn import functional


class ReservoirSynapse(nn.Module):
    """
    Complex synapse model.
    The class implements a complex synapse model.
    Which is a biological plausible update rule.
    """

    def __init__(
        self,
        device="cpu",
        mode="rosenbaum",
        numberOfChemicals: int = 1,
        reservoir_connections=5,
        non_linearity=torch.nn.functional.tanh,
        spectral_radius=0.1,
        params: dict = {},
        **options
    ):
        """
        Initialize the complex synapse model.
        :param device: (str) The processing device to use. Default is 'cpu',
        :param mode: (str) The update rule to use. Default is 'rosenbaum'.
        :param numberOfChemicals: (int) The number of chemicals to use. Default is 1,
        :param non_linearity: (function) The non-linearity to use. Default is torch.nn.functional.tanh,
        :param options: (dict) The options to use. Default is {}.
        :param params: (dict) The parameters of the normal NN. Default is {}.
        """
        super(ReservoirSynapse, self).__init__()

        self.device = device
        self.mode = mode
        self.options = options

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
        self.reservoir_connections = reservoir_connections
        self.reservoir_matrix = None
        self.spec_rad = spectral_radius

        self.non_linearity = non_linearity

        self.update_rules = [False] * 10
        if "update_rules" in self.options:
            for i in self.options["update_rules"]:
                self.update_rules[i] = True
        else:
            self.update_rules = [True] * 10

        self.init_parameters(params=params)

    def reservoir_creation(self):

        seed = self.options["seed"]
        reservoir_matrix = nx.gnm_random_graph(self.number_chemicals, self.reservoir_connections, seed=seed)
        reservoir_matrix = nx.to_numpy_array(reservoir_matrix)
        reservoir_matrix = torch.from_numpy(reservoir_matrix).to_sparse().to(self.device)
        self.reservoir_matrix = reservoir_matrix

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
                            size=(self.number_chemicals, parameter.shape[0], parameter.shape[1]),
                            device=self.device,
                            requires_grad=True,
                        )
                    )
                )

        if "bias" in self.options:
            if self.options["bias"] == True:
                self.all_bias_parameters.extend(self.bias_dictionary.values())

        ## Initialize the P and K matrices
        self.K_matrix = nn.Parameter(
            torch.nn.init.zeros_(torch.empty(size=(self.number_chemicals, self.number_chemicals), device=self.device))
        )
        self.K_matrix = self.K_matrix.to_sparse().to(self.device)
        self.reservoir_creation()
        self.K_matrix += self.reservoir_matrix
        self.K_matrix = self.spec_rad * self.K_matrix / torch.max(torch.abs(torch.eig(self.K_matrix)[0]))

        self.P_matrix = nn.Parameter(
            torch.nn.init.zeros_(torch.empty(size=(self.number_chemicals, 10), device=self.device))
        )
        self.P_matrix[:, 0] = 1e-3

        if "train_K_matrix" in self.options:
            if self.options["train_K_matrix"] == True:
                self.all_meta_parameters.append(self.K_matrix)

        self.all_meta_parameters.append(self.P_matrix)

        self.z_vector = torch.tensor([0] * self.number_chemicals, device=self.device)
        self.y_vector = torch.tensor([0] * self.number_chemicals, device=self.device)

        ## Initialize the chemical time constants
        # z = 1 / \tau
        min_tau = 1
        if "min_tau" in self.options:
            min_tau = self.options["min_tau"]
        max_tau = 50
        base = max_tau / min_tau

        self.tau_vector = min_tau * (base ** torch.linspace(0, 1, self.number_chemicals))
        self.z_vector = 1 / self.tau_vector
        self.y_vector = 1 - self.z_vector

        self.y_vector = self.y_vector.to(self.device)
        self.z_vector = self.z_vector.to(self.device)

        if "train_z_vector" in self.options:
            if self.options["train_z_vector"] == True:
                self.z_vector = nn.Parameter(self.z_vector)
                self.all_meta_parameters.append(self.z_vector)

        ## Initialize the v vector
        self.v_vector = nn.Parameter(
            torch.nn.init.ones_(torch.empty(size=(1, self.number_chemicals), device=self.device))
            / self.number_chemicals
        )
        self.all_meta_parameters.append(self.v_vector)

        ## Initialize the oja minus parameter is trainable
        self.oja_minus_parameter = nn.Parameter(torch.tensor(1, device=self.device).float())
        if "oja_minus_parameter" in self.options:
            if self.options["oja_minus_parameter"] == True:
                self.all_meta_parameters.append(self.oja_minus_parameter)

    def __call__(
        self, activations: list, output: torch.Tensor, label: torch.Tensor, params: dict, h_parameters: dict, beta: int
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
                    new_chemical = None
                    new_chemical = torch.einsum("i,ijk->ijk", self.y_vector, chemical) + torch.einsum(
                        "i,ijk->ijk",
                        self.z_vector,
                        self.non_linearity(
                            torch.einsum("ic,ijk->cjk", self.reservoir_matrix * self.K_matrix, chemical)
                            + torch.einsum("ci,ijk->cjk", self.P_matrix, update_vector)
                            + self.bias_dictionary[h_name]
                        ),
                    )
                    h_parameters[h_name] = new_chemical

                    if self.operator == "mode_3":
                        # Equation 2: w(s) = w(s) + f(v * h(s))
                        new_value = parameter + torch.nn.functional.tanh(
                            torch.einsum("ci,ijk->cjk", self.v_vector, h_parameters[h_name]).squeeze(0)
                        )
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
                    torch.matmul(error[i + 1].T, torch.ones(size=(1, parameter.shape[0]), device=self.device)),
                    activations_and_output[i + 1].T,
                ),
                activations_and_output[i],
            )  # = ERROR on high learning rate

        if self.update_rules[6]:
            update_vector[6] = -torch.matmul(
                torch.matmul(
                    torch.matmul(
                        torch.matmul(activations_and_output[i + 1].T, activations_and_output[i + 1]), parameter
                    ),
                    error[i].T,
                ),
                error[i],
            )  # - ERROR

        if self.update_rules[7]:
            update_vector[7] = -torch.matmul(
                torch.matmul(
                    torch.matmul(torch.matmul(error[i + 1].T, activations_and_output[i + 1]), parameter), error[i].T
                ),
                activations_and_output[i],
            )  # - Maybe be bad

        if self.update_rules[8]:
            update_vector[8] = -torch.matmul(
                torch.matmul(
                    torch.matmul(torch.matmul(activations_and_output[i + 1].T, activations_and_output[i]), parameter.T),
                    error[i + 1].T,
                ),
                error[i],
            )

        if self.update_rules[9]:
            update_vector[9] = torch.matmul(
                activations_and_output[i + 1].T, activations_and_output[i]
            ) - self.oja_minus_parameter * torch.matmul(
                torch.matmul(activations_and_output[i + 1].T, activations_and_output[i + 1]), parameter
            )  # Oja's rule

        return update_vector