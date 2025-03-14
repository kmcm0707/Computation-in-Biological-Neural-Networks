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
        numberOfChemicals: int = 1,
        train_feedback: bool = False,
        typeOfFeedback: typeOfFeedbackEnum = typeOfFeedbackEnum.FA,
        dim_in: int = 1,
        dim_out: int = 1,
    ):
        # Initialize the parent class
        super(chemicalRnn, self).__init__()

        # Set the device
        self.device = device
        self.train_feedback = train_feedback
        self.typeOfFeedback = typeOfFeedback
        self.numberOfChemicals = numberOfChemicals
        self.dim_in = dim_in
        self.dim_out = dim_out

        # Model
