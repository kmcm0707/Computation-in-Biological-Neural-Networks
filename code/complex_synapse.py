import torch
from torch import nn
from torch.nn import functional

class ComplexSynapse(nn.Module):
    """
        Complex synapse model.
        The class implements a complex synapse model.
        Which is a biological plausible update rule.
    """

    def __init__(self, device='cpu', mode='rosenbaum'):
        """
            Initialize the complex synapse model.
            :param n_in: (int) number of input units,
            :param n_out: (int) number of output units,
            :param adapt: (bool) whether the synapse is plastic or not,
            :param feedback: (bool) whether the synapse is a feedback synapse or not.
        """
        super(ComplexSynapse, self).__init__()

        self.device = device
        self.mode = mode

        if self.mode == 'rosenbaum':
            # -- meta-learned plasticity coefficients
            self.theta0 = nn.Parameter(torch.tensor(1e-3).float()) # Pseudo-inverse
            self.theta1 = nn.Parameter(torch.tensor(0.).float()) # Hebbian
            self.theta2 = nn.Parameter(torch.tensor(0.).float()) # Oja

            self.theta0.adapt = True
            self.theta1.adapt = True
            self.theta2.adapt = True
            
            self.thetas = nn.ParameterList([self.theta0, self.theta1, self.theta2])
        elif self.mode == 'all_rosenbaum':
            # -- meta-learned plasticity coefficients
            self.theta0 = nn.Parameter(torch.tensor(1e-3).float())
            self.theta1 = nn.Parameter(torch.tensor(0.).float())
            self.theta2 = nn.Parameter(torch.tensor(0.).float())
            self.theta3 = nn.Parameter(torch.tensor(0.).float())
            self.theta4 = nn.Parameter(torch.tensor(0.).float())
            self.theta5 = nn.Parameter(torch.tensor(0.).float())
            self.theta6 = nn.Parameter(torch.tensor(0.).float())
            self.theta7 = nn.Parameter(torch.tensor(0.).float())
            self.theta8 = nn.Parameter(torch.tensor(0.).float())
            self.theta9 = nn.Parameter(torch.tensor(0.).float())

            self.theta0.adapt = True
            self.theta1.adapt = True
            self.theta2.adapt = True
            self.theta3.adapt = True
            self.theta4.adapt = True
            self.theta5.adapt = True
            self.theta6.adapt = True
            self.theta7.adapt = True
            self.theta8.adapt = True
            self.theta9.adapt = True

            self.thetas = nn.ParameterList([self.theta0, self.theta1, self.theta2, self.theta3, self.theta4, self.theta5, self.theta6, self.theta7, self.theta8, self.theta9])

    def __call__(self, activations: list, output, label, params, beta: int):
        """
        :param activations: (list) model activations,
        :param output: (tensor) model output,
        :param label: (tensor) model label,
        :param params: (dict) model weights,
        :param beta: (int) smoothness coefficient for non-linearity,
        """
        if self.mode == 'rosenbaum':
            # -- error
            feedback = {name: value for name, value in params.items() if 'feedback' in name}
            error = [functional.softmax(output, dim=1) - functional.one_hot(label, num_classes=47)]

            # add the error for the first layer
            for y, i in zip(reversed(activations), reversed(list(feedback))):
                error.insert(0, torch.matmul(error[0], feedback[i]) * (1 - torch.exp(-beta * y)))

            # -- weight update
            self.rosenbaum_plastity_rules(activations, output, error, params)
        elif self.mode == 'all_rosenbaum':
            # -- error
            feedback = {name: value for name, value in params.items() if 'feedback' in name}
            error = [functional.softmax(output, dim=1) - functional.one_hot(label, num_classes=47)]

            # add the error for the first layer
            for y, i in zip(reversed(activations), reversed(list(feedback))):
                error.insert(0, torch.matmul(error[0], feedback[i]) * (1 - torch.exp(-beta * y)))

            # -- weight update
            self.all_rosenbaum_plastity_rules(activations, output, error, params)

    def rosenbaum_plastity_rules(self, activations: list, output, error, params):
        """
            Rosenbaum plasticity rules.
            The function implements the Rosenbaum plasticity rules.
            The rules are used to update the model weights.
        :param activations: (list) model activations,
        :param output: (tensor) model output,
        :param params: (dict) model weights,
        :param feedback: (dict) feedback weights,
        :param beta: (int) smoothness coefficient for non-linearity,
        """
        
        activations_and_output = [*activations, functional.softmax(output, dim=1)]

        i = 0
        for name, parameter in params.items():
            if 'forward' in name:
                if parameter.adapt and 'weight' in name:
                    # -- pseudo-gradient
                    update = - self.thetas[0] * torch.matmul(error[i + 1].T, activations_and_output[i])
                    # -- eHebb rule
                    update -= self.thetas[1] * torch.matmul(error[i + 1].T, error[i])
                    # -- Oja's rule
                    update += self.thetas[2] * (torch.matmul(activations_and_output[i + 1].T, activations_and_output[i]) - torch.matmul( 
                        torch.matmul(activations_and_output[i + 1].T, activations_and_output[i + 1]), parameter))

                    # -- weight update
                    params[name] = parameter + update
                    params[name].adapt = True

                i += 1

    def all_rosenbaum_plastity_rules(self, activations: list, output, error, params):

        activations_and_output = [*activations, functional.softmax(output, dim=1)]

        i = 0
        for name, parameter in params.items():
            if 'forward' in name:
                if parameter.adapt and 'weight' in name:
                    update = - self.thetas[0] * torch.matmul(error[i + 1].T, activations_and_output[i])
                    update -= self.thetas[1] * torch.matmul(activations_and_output[i+1].T, error[i])
                    update -= self.thetas[2] * torch.matmul(error[i + 1].T, error[i])
                    update -= self.thetas[3] * parameter
                    update -= self.thetas[4] * torch.matmul(torch.ones_like(parameter).T, error[i])
                    update -= self.thetas[5] * torch.matmul(torch.matmul(torch.matmul(error[i+1].T, torch.ones_like(parameter)),
                                                            activations_and_output[i+1].T), activations_and_output[i])
                    update -= self.thetas[6] * torch.matmul(torch.matmul(torch.matmul(torch.matmul(activations_and_output[i+1].T, activations_and_output[i+1]),
                                                                         parameter).T, error[i].T), error[i])
                    update -= self.thetas[7] * torch.matmul(torch.matmul(torch.matmul(torch.matmul(error[i+1].T, activations_and_output[i+1]),
                                                                         parameter).T, error[i].T), activations_and_output[i])
                    update -= self.thetas[8] * torch.matmul(torch.matmul(torch.matmul(torch.matmul(activations_and_output[i+1].T, activations_and_output[i]),
                                                                         parameter), error[i+1].T), error[i])
                    update += self.thetas[9] * (torch.matmul(activations_and_output[i + 1].T, activations_and_output[i]) - torch.matmul( 
                        torch.matmul(activations_and_output[i + 1].T, activations_and_output[i + 1]), parameter))

                    # -- weight update
                    params[name] = parameter + update
                    params[name].adapt = True

                i += 1



            
        