from typing import Literal

import torch
import torch.nn as nn


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
        biological: bool = False,
        biological_min_tau: int = 1,
        biological_max_tau: int = 56,
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
        self.biological = biological
        self.biological_min_tau = biological_min_tau
        self.biological_max_tau = biological_max_tau

        # Model
        if not biological:
            self.RNN_forward1_ih = nn.Linear(self.dim_in, 128, bias=False)
            self.RNN_forward1_hh = nn.Linear(128, 128, bias=False)
            # self.RNN_forward2 = nn.RNNCell(input_size=128, hidden_size=128, bias=False)
            self.forward1 = nn.Linear(128, self.dim_out, bias=False)
        else:
            base = self.biological_max_tau / self.biological_min_tau
            tau_vector = self.biological_min_tau * (base ** torch.linspace(0, 1, 128, device=self.device))
            self.z_vector = 1 / tau_vector
            self.y_vector = 1 - self.z_vector
            self.y_vector = self.y_vector.to(self.device)
            # self.y_vector = nn.Parameter(self.y_vector)
            self.z_vector = self.z_vector.to(self.device)
            # self.z_vector = nn.Parameter(self.z_vector)

            self.RNN_forward1_ih = nn.Linear(self.dim_in, 128, bias=False)
            self.RNN_forward1_hh = nn.Linear(128, 128, bias=False)
            self.forward1 = nn.Linear(128, self.dim_out, bias=False)
            self.beta = 10
            self.activation = nn.Softplus(beta=self.beta)

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
        hx1_prev = self.hx1.clone()
        # hx2_prev = self.hx2
        RNN_forward1_ih_x = self.RNN_forward1_ih(x)
        if not self.biological:
            RNN_forward1_hh_hx1 = self.RNN_forward1_hh(self.hx1)
            self.hx1 = torch.tanh(RNN_forward1_ih_x + RNN_forward1_hh_hx1)
            # self.hx2 = torch.tanh(RNN_forward2_hx1)
            output = self.forward1(self.hx1)
        else:
            RNN_forward1_hh_hx1 = self.RNN_forward1_hh(self.hx1)
            self.hx1 = self.y_vector * self.hx1 + self.z_vector * self.activation(RNN_forward1_hh_hx1 + self.hx1)
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
