import torch

from torch.nn import functional


def plasticity_rule(activation, e, params, feedback, Theta, feedbackType):
    """
        Pool of plasticity rules.

    This function receives the network weights as input and updates them based
    on a meta-learned plasticity rule.

    :param activation: (list) model activations,
    :param e: (list) modulatory signals,
    :param params: self.model.named_parameters(),
    :param feedback: (dict) feedback connections,
    :param Theta: (ParameterList) meta-learned plasticity coefficients,
    :param feedbackType: (str) the type of feedback matrix used in the model:
        1) 'sym', which indicates that the feedback matrix is symmetric;
        2) 'fix', which indicates that the feedback matrix is a fixed random matrix.)
    :return: None.
    """
    """ update forward weights """
    i = 0
    with torch.no_grad():
        for name, parameter in params.items():
            if 'linear' in name:
                if parameter.adapt:
                    # -- pseudo-gradient
                    parameter.update = - Theta[0] * torch.matmul(e[i + 1].T, activation[i])
                    # -- eHebb rule
                    parameter.update -= Theta[1] * torch.matmul(e[i + 1].T, e[i])
                    # -- Oja's rule
                    parameter.update += Theta[2] * (torch.matmul(activation[i + 1].T, activation[i]) - torch.matmul( 
                        torch.matmul(activation[i + 1].T, activation[i + 1]), parameter.data))

                    # -- weight update
                    parameter.data += parameter.update

                i += 1

        """ enforce symmetric feedbacks for backprop training """
        if feedbackType == 'sym':
            # -- feedback update (symmetric)
            for i, (name, parameter) in enumerate(feedback.items()):
                parameter.data = params[name.replace('feedback', 'linear')].data # Maybe need .T here to make it symmetric


class RosenbaumOptimizer:
    """
        Adaptation optimizer object.

    The class is responsible for two main operations: computing modulatory
    signals and applying an update rule. The modulatory signals are computed
    based on the current state of the model (activations), and are used to
    adjust the model's parameters. The update rule specifies how these
    adjustments are made.
    """
    def __init__(self, update_rule, feedbackType):
        """
            Initialize the optimizer

        :param update_rule: (function) weight update function,
        :param feedbackType: (str) the type of feedback matrix used in the model:
            1) 'sym', which indicates that the feedback matrix is symmetric;
            2) 'fix', which indicates that the feedback matrix is a fixed random matrix.)
        """
        self.update_rule = update_rule
        self.feedbackType = feedbackType

    def __call__(self, params, logits, label, activation, Beta, Theta):
        """
            Adaptation loop update.

        The following function is an implementation of one step update of the
        model parameters in the adaptation loop. The function performs the
        following operations:
        1) Computes the modulatory signals using the signal from downstream layers,
            feedback connections, and activations. instead of using pre-activations,
            we use g'(z) = 1 - e^(-Beta*y).
        2) Updates the model parameters using the update function.

        :param params: self.model.named_parameters(),
        :param logits: (torch.Tensor) unnormalized prediction values,
        :param label: (torch.Tensor) target class,
        :param activation: (tuple) vector of activations,
        :param Beta: (int) smoothness coefficient for non-linearity,
        :param Theta: (ParameterList) plasticity meta-parameters.
        :return: None.
        """
        # -- error
        feedback = dict({name: value for name, value in params if 'feedback' in name})
        e = [functional.softmax(logits) - functional.one_hot(label, num_classes=47)]
        for y, i in zip(reversed(activation), reversed(list(feedback))):
            e.insert(0, torch.matmul(e[0], feedback[i]) * (1 - torch.exp(-Beta * y)))

        # -- weight update
        self.update_rule([*activation, functional.softmax(logits, dim=1)], e, params, feedback, Theta, self.feedbackType)