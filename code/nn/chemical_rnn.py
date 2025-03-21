from typing import Literal

import torch
import torch.nn as nn
from options.meta_learner_options import typeOfFeedbackEnum


class ChemicalRnn(nn.Module):
    """
    Chemical Recurrent Neural Network class.
    """

    def __init__(
        self,
        device: Literal["cpu", "cuda"] = "cpu",
        numberOfSlowChemicals: int = 1,
        numberOfFastChemicals: int = 1,
        typOfSlowChemicalFeedback: typeOfFeedbackEnum = typeOfFeedbackEnum.DFA,
        typeOfFastChemicalFeedback: typeOfFeedbackEnum = typeOfFeedbackEnum.DFA,
        dim_in: int = 1,
        dim_out: int = 1,
    ):
        # Initialize the parent class
        super(ChemicalRnn, self).__init__()

        # Set the device
        self.device = device

        # Set the number of slow and fast chemicals
        self.numberOfSlowChemicals = numberOfSlowChemicals
        self.numberOfFastChemicals = numberOfFastChemicals

        # Set the type of feedback for the slow and fast chemicals
        self.typOfSlowChemicalFeedback = typOfSlowChemicalFeedback
        self.typeOfFastChemicalFeedback = typeOfFastChemicalFeedback

        # Set the input and output dimensions
        self.dim_in = dim_in
        self.dim_out = dim_out

        # Model
        self.RNN_forward1 = nn.RNNCell(input_size=self.dim_in, hidden_size=128, bias=False)
        self.RNN_forward2 = nn.RNNCell(input_size=128, hidden_size=128, bias=False)
        self.forward1 = nn.Linear(128, self.dim_out, bias=False)

        # Hidden states
        self.hx1 = torch.zeros(1, 128).to(self.device)
        self.hx2 = torch.zeros(1, 128).to(self.device)

        # Chemicals
        self.slow_RNN1_ih = nn.Parameter(
            torch.zeros((self.numberOfSlowChemicals, 128, self.dim_in), device=self.device)
        )
        self.slow_RNN1_hh = nn.Parameter(torch.zeros((self.numberOfSlowChemicals, 128, 128), device=self.device))
        self.slow_RNN2_ih = nn.Parameter(torch.zeros((self.numberOfSlowChemicals, 128, 128), device=self.device))
        self.slow_RNN2_hh = nn.Parameter(torch.zeros((self.numberOfSlowChemicals, 128, 128), device=self.device))

        self.fast_RNN1_ih = nn.Parameter(
            torch.zeros((self.numberOfFastChemicals, 128, self.dim_in), device=self.device)
        )
        self.fast_RNN1_hh = nn.Parameter(torch.zeros((self.numberOfFastChemicals, 128, 128), device=self.device))
        self.fast_RNN2_ih = nn.Parameter(torch.zeros((self.numberOfFastChemicals, 128, 128), device=self.device))
        self.fast_RNN2_hh = nn.Parameter(torch.zeros((self.numberOfFastChemicals, 128, 128), device=self.device))

        self.chemical1 = nn.Parameter(torch.zeros((self.numberOfSlowChemicals, self.dim_out, 128), device=self.device))

        # DFA feedback
        self.RNN1_ih_feedback = nn.Linear(self.dim_in, dim_out, bias=False)
        self.RNN1_hh_feedback = nn.Linear(128, dim_out, bias=False)
        self.RNN2_ih_feedback = nn.Linear(128, dim_out, bias=False)
        self.RNN2_hh_feedback = nn.Linear(128, dim_out, bias=False)
        self.feedback_chemical1 = nn.Linear(128, dim_out, bias=False)
        self.feedback_order = [
            "feedback_chemical1",
            "RNN2_hh_feedback",
            "RNN2_ih_feedback",
            "RNN1_hh_feedback",
            "RNN1_ih_feedback",
        ]
        self.error_below = [0, 1, 1, 2, 2]
        self.activation_above = [-2, -3, -4, -5, -6]
        self.activation_below = [-1, -2, -2, -4, -4]

        self.feedback_to_parameters = {
            "feedback_chemical1": "forward1",
            "RNN2_hh_feedback": "RNN_forward2.hh",
            "RNN2_ih_feedback": "RNN_forward2.ih",
            "RNN1_hh_feedback": "RNN_forward1.hh",
            "RNN1_ih_feedback": "RNN_forward1.ih",
        }

        # Parameters to chemical
        self.parameters_to_chemical = {
            "RNN_forward1.ih": "RNN1_ih",
            "RNN_forward1.hh": "RNN1_hh",
            "RNN_forward2.ih": "RNN2_ih",
            "RNN_forward2.hh": "RNN2_hh",
            "forward1": "chemical1",
        }

    def forward(self, x):
        assert x.shape[1] == self.dim_in, "Input shape is not correct."
        assert x.shape[0] == self.hx1.shape[0], "Batch size is not correct."

        # Forward pass
        hx1_prev = self.hx1.clone()
        self.hx1 = self.RNN1(x, self.hx1)
        hx2_prev = self.hx2.clone()
        self.hx2 = self.RNN2(self.hx1, self.hx2)
        output = self.forward1(self.hx2)

        return (x, hx1_prev, self.hx1, hx2_prev, self.hx2), output

    def reset_hidden(self, batch_size):
        self.hx1 = torch.zeros(batch_size, 128).to(self.device)
