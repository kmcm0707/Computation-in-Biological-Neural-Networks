import torch
from torch import nn
from torch.nn import functional

class ComplexSynapse(nn.Module):
    """
        Complex synapse model.
        The class implements a complex synapse model.
        Which is a biological plausible update rule.
    """

    def __init__(self, device='cpu', mode='rosenbaum', numberOfChemicals: int = 1, non_linearity=torch.nn.functional.tanh):
        """
            Initialize the complex synapse model.
            :param device: (str) The processing device to use. Default is 'cpu',
            :param mode: (str) The update rule to use. Default is 'rosenbaum'.
        """
        super(ComplexSynapse, self).__init__()

        self.device = device
        self.mode = mode

        #h(s+1) = (1-z)h(s) + zf(Kh(s) + \theta * F(Parameter) + b)
        # y = 1-z, y_0 = 1, z_0 = 1
        #w(s) = v * h(s) (if self.number_chemicals = 1)
        self.K_matrix = nn.Parameter() # K - LxL
        self.v_vector = nn.Parameter() # v - L
        self.theta_matrix = nn.Parameter() # \theta - Lx10
        self.bias = nn.Parameter() # b #TODO: Add bias
        self.all_meta_parameters = nn.ParameterList([]) # All updatable meta-parameters
        self.number_chemicals = numberOfChemicals # z - L

        self.non_linearity = non_linearity

        self.init_parameters()

    @torch.no_grad()
    def init_parameters(self):
        """
            Initialize the parameters of the complex synapse model.
            K_matrix: (tensor) The K matrix - dimension (L, L),
            v_vector: (tensor) The v vector - dimension (1, L),
            theta_matrix: (tensor) The theta matrix - dimension (L, 10),
            z_vector: (tensor) The z vector - dimension (1, L),
            y_vector: (tensor) The y vector - dimension (1, L),
        """
        
        if self.mode == 'rosenbaum' or self.mode == 'all_rosenbaum':
            self.K_matrix = nn.Parameter(torch.nn.init.zeros_(torch.empty(size=(self.number_chemicals, self.number_chemicals), device=self.device)))
            self.theta_matrix = nn.Parameter(torch.nn.init.zeros_(torch.empty(size=(self.number_chemicals, 10), device=self.device)))
            self.theta_matrix[0,0] = 1e-3
        else:
            self.K_matrix = nn.Parameter(torch.nn.init.zeros_(torch.empty(size=(self.number_chemicals, self.number_chemicals), device=self.device)))
            self.theta_matrix = nn.Parameter(torch.nn.init.zeros_(torch.empty(size=(self.number_chemicals, 10), device=self.device)))
            self.theta_matrix[0,0] = 1e-3
            self.all_meta_parameters.append(self.K_matrix)

        self.all_meta_parameters.append(self.theta_matrix)
       
        self.z_vector = torch.tensor([0] * self.number_chemicals, device=self.device)
        self.y_vector = torch.tensor([0] * self.number_chemicals, device=self.device)
        self.z_vector[0] = 1
        self.y_vector[0] = 1

        if self.number_chemicals > 1:
            ## Initialize the chemical time constants
            # z = 1 / \tau
            min_tau = 1
            max_tau = 30
            exponential = 1.5
            self.tau_vector = torch.tensor([min_tau + (max_tau - min_tau) * (i / (self.number_chemicals - 1)) ** exponential for i in range(self.number_chemicals)], device=self.device).float()
            self.z_vector = 1 / self.tau_vector
            self.y_vector = 1 - self.z_vector
            self.y_vector[0] = 1
        
        self.v_vector = nn.Parameter(torch.nn.init.ones_(torch.empty(size=(1, self.number_chemicals), device=self.device)))
        if self.mode == 'rosenbaum' or self.mode == 'all_rosenbaum':
            pass
        else:
            self.all_meta_parameters.append(self.v_vector)
                
    def __call__(self, activations: list, output: torch.Tensor, label: torch.Tensor, params: dict, h_parameters: dict, beta: int):
        """
        :param activations: (list) model activations,
        :param output: (tensor) model output - dimension (W_1) (per layer),
        :param label: (tensor) model label - dimension (W_1) (per layer),
        :param params: (dict) model weights - dimension (W_1, W_2) (per parameter),
        :param h_parameters: (dict) model chemicals - dimension L x (W_1, W_2) (per parameter),
        :param beta: (int) smoothness coefficient for non-linearity,
        """

        feedback = {name: value for name, value in params.items() if 'feedback' in name}
        error = [functional.softmax(output, dim=1) - functional.one_hot(label, num_classes=47)]
        # add the error for all the layers
        for y, i in zip(reversed(activations), reversed(list(feedback))):
            error.insert(0, torch.matmul(error[0], feedback[i]) * (1 - torch.exp(-beta * y)))
        activations_and_output = [*activations, functional.softmax(output, dim=1)]

        i = 0
        for name, parameter in params.items():
            if 'forward' in name:
                h_name = name.replace('forward', 'chemical').split('.')[0]
                chemical = h_parameters[h_name]
                if parameter.adapt and 'weight' in name:
                    # Equation 1: h(s+1) = (1-z)h(s) + zf(Kh(s) + \theta * F(Parameter))
                    # Equation 2: w(s) = v * h(s)
                    update_vector = self.calculate_update_vector(error, activations_and_output, parameter, i)
                    #unsquezzed_parameter = parameter.unsqueeze(0)
                    new_chemical = torch.einsum('i,ijk->ijk',self.y_vector, chemical) + \
                                    self.non_linearity(torch.einsum('i,ijk->ijk', self.z_vector, torch.einsum('ic,ijk->cjk', self.K_matrix, chemical) + \
                                                    torch.einsum('ci,ijk->cjk', self.theta_matrix, update_vector)))
                    h_parameters[h_name] = new_chemical
                    new_value = torch.einsum('ci,ijk->cjk', self.v_vector, h_parameters[h_name]).squeeze(0)
                    params[name] = new_value

                    params[name].adapt = True
                i += 1

    def calculate_update_vector(self, error, activations_and_output, parameter, i) -> torch.Tensor:
        """
        Calculate the update vector for the complex synapse model.
        :param error: (list) model error,
        :param activations_and_output: (list) model activations and output,
        :param parameter: (tensor) model parameter - dimension (W_1, W_2),
        :param i: (int) index of the parameter.
        """
        update_vector = torch.zeros((10, parameter.shape[0], parameter.shape[1]), device=self.device)
        update_vector[0] = - torch.matmul(error[i + 1].T, activations_and_output[i]) # Pseudo-gradient

        if self.mode != 'rosenbaum':
            update_vector[1] = - torch.matmul(activations_and_output[i+1].T, error[i])

        update_vector[2] = - torch.matmul(error[i + 1].T, error[i]) # eHebb rule

        if self.mode != 'rosenbaum':
            update_vector[3] = - parameter
            update_vector[4] = - torch.matmul(torch.ones(size=(parameter.shape[0], 1), device=self.device), error[i])
            """update_vector[5] = - torch.matmul(torch.matmul(torch.matmul(error[i+1].T, torch.ones(size=(1, parameter.shape[0]), device=self.device)),
                                                  activations_and_output[i+1].T), activations_and_output[i]) # = ERROR on high learning rate
           
            update_vector[6] = - torch.matmul(torch.matmul(torch.matmul(torch.matmul(activations_and_output[i+1].T, activations_and_output[i+1]),
                                                             parameter), error[i].T), error[i]) #- ERROR
            update_vector[7] = - torch.matmul(torch.matmul(torch.matmul(torch.matmul(error[i+1].T, activations_and_output[i+1]),
                                                                parameter), error[i].T), activations_and_output[i]) # - Maybe be bad"""
            update_vector[8] = - torch.matmul(torch.matmul(torch.matmul(torch.matmul(activations_and_output[i+1].T, activations_and_output[i]),
                                                                parameter.T), error[i+1].T), error[i])
            
        update_vector[9] = torch.matmul(activations_and_output[i + 1].T, activations_and_output[i]) - torch.matmul( 
            torch.matmul(activations_and_output[i + 1].T, activations_and_output[i + 1]), parameter) # Oja's rule
        
        return update_vector

            
        