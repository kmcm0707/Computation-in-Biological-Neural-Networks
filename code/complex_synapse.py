import torch
from torch import nn
from torch.nn import functional

class ComplexSynapse(nn.Module):
    """
        Complex synapse model.
        The class implements a complex synapse model.
        Which is a biological plausible update rule.
    """

    def __init__(self, params, device='cpu', mode='rosenbaum'):
        """
            Initialize the complex synapse model.
            :param params: (dict) model weights,
            :param device: (str) The processing device to use. Default is 'cpu',
            :param mode: (str) The update rule to use. Default is 'rosenbaum'.
        """
        super(ComplexSynapse, self).__init__()

        self.device = device
        self.mode = mode

        #h(s+1) = (1-z)h(s) + zf(Kh(s) + PF(\theta) + b)
        # y = 1-z, y_0 = 1, z_0 = 1
        #w(s) = h(s)

        self.K_matrix_dictionary = torch.nn.ParameterDict() # K
        self.P_matrix_dictionary = torch.nn.ParameterDict() # P
        self.bias_dictionary = torch.nn.ParameterDict() # b
        self.number_chemicals = 1 # z

        with torch.no_grad():
            for name, parameter in params.items():
                if 'forward' in name:
                    if 'weight' in name:
                        if self.mode == 'rosenbaum' or self.mode == 'all_rosenbaum':
                            self.K_matrix_dictionary[name] = nn.Parameter(nn.init.zeros_(torch.empty(parameter.shape)))
                            self.bias_dictionary[name] = nn.Parameter(nn.init.zeros_(torch.empty(parameter.shape[0], 10)))
                        else:
                            self.K_matrix_dictionary[name] = nn.Parameter(nn.init.uniform_(torch.empty(10, self.number_chemicals)))
                            self.bias_dictionary[name] = nn.Parameter(nn.init.uniform_(torch.empty(parameter.shape, 10)))

                        self.P_matrix_dictionary[name] = nn.Parameter(nn.init.uniform_(torch.empty(parameter.shape[0], 10)))
                        self.P_matrix_dictionary[name][0, 0] = 1e-3 # Pseudo-inverse
        
        self.z_vector = torch.tensor([1], device=self.device)
        self.y_vector = torch.tensor([1], device=self.device)

        self.thetas = nn.ParameterList([torch.tensor(1e-3).float(), torch.tensor(0.).float() for _ in range(9)])

        self.all_parameters = nn.ParameterList([*self.K_matrix_dictionary.values(), *self.P_matrix_dictionary.values(), *self.bias_dictionary.values(), *self.thetas])
                
    def __call__(self, activations: list, output, label, params, beta: int):
        """
        :param activations: (list) model activations,
        :param output: (tensor) model output,
        :param label: (tensor) model label,
        :param params: (dict) model weights,
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
                if parameter.adapt and 'weight' in name:
                    update_vector = torch.zeros_like((10, parameter.shape))
                    update_vector[0] = - torch.matmul(error[i + 1].T, activations_and_output[i]) # Pseudo-gradient

                    if self.mode != 'rosenbaum':
                        update_vector[1] = - torch.matmul(activations_and_output[i+1].T, error[i])

                    update_vector[2] = - torch.matmul(error[i + 1].T, error[i]) # eHebb rule

                    if self.mode != 'rosenbaum':
                        update_vector[3] = - parameter
                        update_vector[4] = - torch.matmul(torch.ones_like(parameter).T, error[i])
                        update_vector[5] = - torch.matmul(torch.matmul(torch.matmul(error[i+1].T, torch.ones_like(parameter)),
                                                                activations_and_output[i+1].T), activations_and_output[i])
                        update_vector[6] = - torch.matmul(torch.matmul(torch.matmul(torch.matmul(activations_and_output[i+1].T, activations_and_output[i+1]),
                                                                            parameter).T, error[i].T), error[i])
                        update_vector[7] = - torch.matmul(torch.matmul(torch.matmul(torch.matmul(error[i+1].T, activations_and_output[i+1]),
                                                                            parameter).T, error[i].T), activations_and_output[i])
                        update_vector[8] = - torch.matmul(torch.matmul(torch.matmul(torch.matmul(activations_and_output[i+1].T, activations_and_output[i]),
                                                                            parameter), error[i+1].T), error[i])
                        
                    update_vector[9] = torch.matmul(activations_and_output[i + 1].T, activations_and_output[i]) - torch.matmul( 
                        torch.matmul(activations_and_output[i + 1].T, activations_and_output[i + 1]), parameter) # Oja's rule
                    
                    if self.mode == 'rosenbaum' or self.mode == 'all_rosenbaum':
                        params[name] = torch.matmul(self.y_vector, parameter) + \
                                        torch.matmul(self.z_vector, torch.matmul(self.theta_matrix_dictionary, update_vector))

                    params[name].adapt = True
                    i += 1



            
        