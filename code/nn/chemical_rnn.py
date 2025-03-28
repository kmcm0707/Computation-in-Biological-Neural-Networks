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
        dim_in: int = 1,
        dim_out: int = 1,
        requireFastChemical: bool = False,
    ):
        # Initialize the parent class
        super(ChemicalRnn, self).__init__()

        # Set the device
        self.device = device

        # Set the number of slow and fast chemicals
        self.numberOfSlowChemicals = numberOfSlowChemicals
        if requireFastChemical:
            self.numberOfFastChemicals = numberOfFastChemicals

        # Set the input and output dimensions
        self.dim_in = dim_in
        self.dim_out = dim_out

        # Model
        self.RNN_forward1_ih = nn.Linear(self.dim_in, 128, bias=False)
        self.RNN_forward1_hh = nn.Linear(128, 128, bias=False)
        # self.RNN_forward2 = nn.RNNCell(input_size=128, hidden_size=128, bias=False)
        self.forward1 = nn.Linear(128, self.dim_out, bias=False)

        # Hidden states
        self.hx1 = torch.zeros(1, 128).to(self.device)
        # self.hx2 = torch.zeros(1, 128).to(self.device)

        # Chemicals
        self.slow_RNN1_ih = nn.Parameter(
            torch.zeros((self.numberOfSlowChemicals, 128, self.dim_in), device=self.device)
        )
        self.slow_RNN1_hh = nn.Parameter(torch.zeros((self.numberOfSlowChemicals, 128, 128), device=self.device))
        # self.slow_RNN2_ih = nn.Parameter(torch.zeros((self.numberOfSlowChemicals, 128, 128), device=self.device))
        # self.slow_RNN2_hh = nn.Parameter(torch.zeros((self.numberOfSlowChemicals, 128, 128), device=self.device))

        if requireFastChemical:
            self.fast_RNN1_ih = nn.Parameter(
                torch.zeros((self.numberOfFastChemicals, 128, self.dim_in), device=self.device)
            )
            self.fast_RNN1_hh = nn.Parameter(torch.zeros((self.numberOfFastChemicals, 128, 128), device=self.device))
            self.fast_RNN2_ih = nn.Parameter(torch.zeros((self.numberOfFastChemicals, 128, 128), device=self.device))
            self.fast_RNN2_hh = nn.Parameter(torch.zeros((self.numberOfFastChemicals, 128, 128), device=self.device))

        self.slow_chemical1 = nn.Parameter(
            torch.zeros((self.numberOfSlowChemicals, self.dim_out, 128), device=self.device)
        )

        # DFA feedback
        self.RNN1_ih_feedback = nn.Linear(self.dim_in, dim_out, bias=False)
        self.RNN1_hh_feedback = nn.Linear(128, dim_out, bias=False)
        # self.RNN2_ih_feedback = nn.Linear(128, dim_out, bias=False)
        # self.RNN2_hh_feedback = nn.Linear(128, dim_out, bias=False)
        self.feedback1 = nn.Linear(128, dim_out, bias=False)

        self.feedback_to_parameters = {
            "feedback1.weight": "forward1.weight",
            "RNN1_hh_feedback.weight": "RNN_forward1_hh.weight",
            "RNN1_ih_feedback.weight": "RNN_forward1_ih.weight",
        }

        # Parameters to chemical
        self.parameters_to_chemical = {
            "RNN_forward1_ih.weight": "RNN1_ih",
            "RNN_forward1_hh.weight": "RNN1_hh",
            "forward1.weight": "chemical1",
        }

        self.error_dict = {
            "RNN_forward1_ih.weight": "feedback1.weight",
            "RNN_forward1_hh.weight": "feedback1.weight",
            "forward1.weight": "last",
        }

    def forward(self, x):
        assert x.shape[1] == self.dim_in, "Input shape is not correct."
        assert x.shape[0] == self.hx1.shape[0], "Batch size is not correct."

        # Forward pass
        hx1_prev = self.hx1
        # hx2_prev = self.hx2
        RNN_forward1_ih_x = self.RNN_forward1_ih(x)
        RNN_forward1_hh_hx1 = self.RNN_forward1_hh(self.hx1)
        self.hx1 = torch.tanh(RNN_forward1_ih_x + RNN_forward1_hh_hx1)
        # self.hx2 = torch.tanh(RNN_forward2_hx1)
        output = self.forward1(self.hx1)

        activations = {
            "RNN_forward1_ih.weight": (x, RNN_forward1_ih_x),
            "RNN_forward1_hh.weight": (hx1_prev, RNN_forward1_hh_hx1),
            "forward1.weight": (self.hx1, output),
        }
        return activations, output

    def reset_hidden(self, batch_size):
        self.hx1 = torch.zeros(batch_size, 128).to(self.device)
        # self.hx2 = torch.zeros(batch_size, 128).to(self.device)
