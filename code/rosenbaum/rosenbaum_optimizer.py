import torch
from torch.nn import functional


def plasticity_rule(activation, e, params, feedback, Theta, feedbackType):
    """
        Pool of plasticity rules.

    This function receives the network weights as input and updates them based
    on a meta-learned plasticity rule.

    :param activation: (list) model activations,
    :param e: (list) modulatory signals,
    :param params: (dict) model weights,
    :param feedback: (dict) feedback connections,
    :param Theta: (ParameterList) meta-learned plasticity coefficients,
    :param feedbackType: (str) the type of feedback matrix used in the model:
        1) 'sym', which indicates that the feedback matrix is symmetric;
        2) 'fix', which indicates that the feedback matrix is a fixed random matrix.)
    :return: None.
    """
    """ update forward weights """
    i = 0
    for name, parameter in params.items():
        if "forward" in name:
            if parameter.adapt:
                # -- pseudo-gradient
                print(torch.matmul(e[i + 1].T, activation[i]))

                update = -Theta[0] * torch.matmul(e[i + 1].T, activation[i])
                print(update.shape)
                # -- eHebb rule
                update -= Theta[1] * torch.matmul(e[i + 1].T, e[i])
                # -- Oja's rule
                update += Theta[2] * (
                    torch.matmul(activation[i + 1].T, activation[i])
                    - torch.matmul(torch.matmul(activation[i + 1].T, activation[i + 1]), parameter)
                )

                # -- weight update
                params[name] = parameter + update
                params[name].adapt = True

            i += 1

    """ enforce symmetric feedbacks for backprop training """
    if feedbackType == "sym":
        # -- feedback update (symmetric)
        forward = dict({k: v for k, v in params.items() if "forward" in k and "weight" in k})
        for i, ((feedback_name, feedback_value), (forward_name, _)) in enumerate(
            zip(feedback.items(), forward.items())
        ):
            params[feedback_name].data = params[forward_name]
            params[feedback_name].adapt = feedback_value.adapt


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

        :param params: (dict) model weights,
        :param logits: (torch.Tensor) unnormalized prediction values,
        :param label: (torch.Tensor) target class,
        :param activation: (tuple) vector of activations,
        :param Beta: (int) smoothness coefficient for non-linearity,
        :param Theta: (ParameterList) plasticity meta-parameters.
        :return: None.
        """
        # -- error
        feedback = {name: value for name, value in params.items() if "feedback" in name}
        error = [functional.softmax(logits, dim=1) - functional.one_hot(label, num_classes=47)]

        # add the error for the first layer
        for y, i in zip(reversed(activation), reversed(list(feedback))):
            error.insert(0, torch.matmul(error[0], feedback[i]) * (1 - torch.exp(-Beta * y)))

        # -- weight update
        self.update_rule(
            [*activation, functional.softmax(logits, dim=1)], error, params, feedback, Theta, self.feedbackType
        )
