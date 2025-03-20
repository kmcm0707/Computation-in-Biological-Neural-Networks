from typing import Literal

import torch
import torch.nn as nn
from options.meta_learner_options import typeOfFeedbackEnum


class chemicalRnn(nn.Module):
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
        super(chemicalRnn, self).__init__()

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
        self.RNN1 = nn.RNNCell(input_size=self.dim_in, hidden_size=128, bias=False)
        self.RNN2 = nn.RNNCell(input_size=128, hidden_size=128, bias=False)
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

    def forward(self, x):
        assert x.shape[1] == self.dim_in, "Input shape is not correct."
        assert x.shape[0] == self.hx1.shape[0], "Batch size is not correct."

        self.hx1 = self.RNN1(x, self.hx1)
        self.hx2 = self.RNN2(self.hx1, self.hx2)

        return (x, self.hx1), self.hx2

    def reset_hidden(self, batch_size):
        self.hx1 = torch.zeros(batch_size, 128).to(self.device)
