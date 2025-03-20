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
        self.RNN2 = nn.RNNCell(input_size=128, hidden_size=dim_out, bias=False)

        # Hidden states
        self.hx1 = torch.zeros(1, 128).to(self.device)
        self.hx2 = torch.zeros(1, 128).to(self.device)

        # Chemicals
