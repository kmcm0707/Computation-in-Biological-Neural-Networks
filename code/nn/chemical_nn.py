from typing import Literal

import torch
import torch.nn as nn
from options.meta_learner_options import typeOfFeedbackEnum


class ChemicalNN(nn.Module):
    """

    Chemical Neural Network class.

    """

    def __init__(
        self,
        device: Literal["cpu", "cuda"] = "cpu",
        numberOfChemicals: int = 1,
        small: bool = False,
        train_feedback: bool = False,
        typeOfFeedback: typeOfFeedbackEnum = typeOfFeedbackEnum.FA,
        dim_out: int = 47,
    ):

        # Initialize the parent class
        super(ChemicalNN, self).__init__()

        # Set the device
        self.device = device
        self.small = small  # Small model for testing
        self.train_feedback = train_feedback  # Train feedback for feedback alignment
        self.typeOfFeedback = typeOfFeedback  # Feedback alignment or direct feedback alignment

        if self.typeOfFeedback == "DFA" and self.train_feedback:
            raise ValueError("DFA and train_feedback cannot be used together")

        if self.small and self.train_feedback:
            raise ValueError("Small and train_feedback cannot be used together")

        # Model
        self.dim_out = dim_out

        if self.small:
            self.forward1 = nn.Linear(784, 15, bias=False)
            self.forward2 = nn.Linear(15, 10, bias=False)
            self.forward3 = nn.Linear(10, 5, bias=False)
            self.forward4 = nn.Linear(5, self.dim_out, bias=False)
            """self.forward_layers = nn.ModuleList([self.forward1, self.forward2, self.forward3, self.forward4])"""
        else:
            self.forward1 = nn.Linear(784, 170, bias=False)
            self.forward2 = nn.Linear(170, 130, bias=False)
            self.forward3 = nn.Linear(130, 100, bias=False)
            self.forward4 = nn.Linear(100, 70, bias=False)
            self.forward5 = nn.Linear(70, self.dim_out, bias=False)
            """self.forward_layers = nn.ModuleList([self.forward1, self.forward2, self.forward3, self.forward4, self.forward5])"""

        # Feedback pathway for plasticity
        # Feedback alignment
        if self.typeOfFeedback == typeOfFeedbackEnum.FA or self.typeOfFeedback == typeOfFeedbackEnum.FA_NO_GRAD:
            if self.small:
                self.feedback1 = nn.Linear(784, 15, bias=False)
                self.feedback2 = nn.Linear(15, 10, bias=False)
                self.feedback3 = nn.Linear(10, 5, bias=False)
                self.feedback4 = nn.Linear(5, self.dim_out, bias=False)
            else:
                self.feedback1 = nn.Linear(784, 170, bias=False)
                self.feedback2 = nn.Linear(170, 130, bias=False)
                self.feedback3 = nn.Linear(130, 100, bias=False)
                self.feedback4 = nn.Linear(100, 70, bias=False)
                self.feedback5 = nn.Linear(70, self.dim_out, bias=False)
        # Direct feedback alignment
        elif self.typeOfFeedback == typeOfFeedbackEnum.DFA or self.typeOfFeedback == typeOfFeedbackEnum.DFA_grad:
            if self.small:
                self.feedback1 = nn.Linear(784, self.dim_out, bias=False)
                self.feedback2 = nn.Linear(15, self.dim_out, bias=False)
                self.feedback3 = nn.Linear(10, self.dim_out, bias=False)
                self.feedback4 = nn.Linear(5, self.dim_out, bias=False)
            else:
                self.feedback1 = nn.Linear(784, self.dim_out, bias=False)
                self.feedback2 = nn.Linear(170, self.dim_out, bias=False)
                self.feedback3 = nn.Linear(130, self.dim_out, bias=False)
                self.feedback4 = nn.Linear(100, self.dim_out, bias=False)
                self.feedback5 = nn.Linear(70, self.dim_out, bias=False)
        elif self.typeOfFeedback == typeOfFeedbackEnum.scalar:
            if self.small:
                self.feedback1 = nn.Linear(784, 1, bias=False)
                self.feedback2 = nn.Linear(15, 1, bias=False)
                self.feedback3 = nn.Linear(10, 1, bias=False)
                self.feedback4 = nn.Linear(5, 1, bias=False)
            else:
                self.feedback1 = nn.Linear(784, 1, bias=False)
                self.feedback2 = nn.Linear(170, 1, bias=False)
                self.feedback3 = nn.Linear(130, 1, bias=False)
                self.feedback4 = nn.Linear(100, 1, bias=False)
                self.feedback5 = nn.Linear(70, 1, bias=False)
        elif self.typeOfFeedback == typeOfFeedbackEnum.DFA_grad_FA:
            if self.small:
                self.feedback1 = nn.Linear(784, 15, bias=False)
                self.feedback2 = nn.Linear(15, 10, bias=False)
                self.feedback3 = nn.Linear(10, 5, bias=False)
                self.feedback4 = nn.Linear(5, self.dim_out, bias=False)
                self.DFA_feedback1 = nn.Linear(784, self.dim_out, bias=False)
                self.DFA_feedback2 = nn.Linear(15, self.dim_out, bias=False)
                self.DFA_feedback3 = nn.Linear(10, self.dim_out, bias=False)
                self.DFA_feedback4 = nn.Linear(5, self.dim_out, bias=False)
            else:
                self.feedback_FA1 = nn.Linear(784, 170, bias=False)
                self.feedback_FA2 = nn.Linear(170, 130, bias=False)
                self.feedback_FA3 = nn.Linear(130, 100, bias=False)
                self.feedback_FA4 = nn.Linear(100, 70, bias=False)
                self.feedback_FA5 = nn.Linear(70, self.dim_out, bias=False)
                self.DFA_feedback1 = nn.Linear(784, self.dim_out, bias=False)
                self.DFA_feedback2 = nn.Linear(170, self.dim_out, bias=False)
                self.DFA_feedback3 = nn.Linear(130, self.dim_out, bias=False)
                self.DFA_feedback4 = nn.Linear(100, self.dim_out, bias=False)
                self.DFA_feedback5 = nn.Linear(70, self.dim_out, bias=False)
        else:
            raise ValueError("Invalid type of feedback")

        # Layer normalization
        """self.layer_norm1 = nn.LayerNorm(170)
        self.layer_norm2 = nn.LayerNorm(130)
        self.layer_norm3 = nn.LayerNorm(100)
        self.layer_norm4 = nn.LayerNorm(70)"""

        # h(s) - LxW
        self.numberOfChemicals = numberOfChemicals
        if self.small:
            self.chemical1 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 15, 784), device=self.device))
            self.chemical2 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 10, 15), device=self.device))
            self.chemical3 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 5, 10), device=self.device))
            self.chemical4 = nn.Parameter(torch.zeros(size=(numberOfChemicals, self.dim_out, 5), device=self.device))
            self.chemicals = nn.ParameterList([self.chemical1, self.chemical2, self.chemical3, self.chemical4])
        else:
            self.chemical1 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 170, 784), device=self.device))
            self.chemical2 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 130, 170), device=self.device))
            self.chemical3 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 100, 130), device=self.device))
            self.chemical4 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 70, 100), device=self.device))
            self.chemical5 = nn.Parameter(torch.zeros(size=(numberOfChemicals, self.dim_out, 70), device=self.device))
            self.chemicals = nn.ParameterList(
                [self.chemical1, self.chemical2, self.chemical3, self.chemical4, self.chemical5]
            )

        # h(s) for feedback
        # Only works for non-small models
        if train_feedback:
            self.feedback_chemical1 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 170, 784), device=self.device))
            self.feedback_chemical2 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 130, 170), device=self.device))
            self.feedback_chemical3 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 100, 130), device=self.device))
            self.feedback_chemical4 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 70, 100), device=self.device))
            self.feedback_chemical5 = nn.Parameter(
                torch.zeros(size=(numberOfChemicals, dim_out, 70), device=self.device)
            )
            self.feedback_chemicals = nn.ParameterList(
                [
                    self.feedback_chemical1,
                    self.feedback_chemical2,
                    self.feedback_chemical3,
                    self.feedback_chemical4,
                    self.feedback_chemical5,
                ]
            )

        # Activation function
        self.beta = 10
        self.activation = nn.Softplus(beta=self.beta)

        # Dropout
        self.dropout = nn.Dropout(p=0.2)

    # @torch.compile
    def forward(self, x):
        y0 = x.squeeze(1)

        y1 = self.forward1(y0)
        y1 = self.activation(y1)
        # y1 = self.dropout(y1)

        # y1 = self.layer_norm1(y1)

        y2 = self.forward2(y1)
        y2 = self.activation(y2)
        # y2 = self.dropout(y2)
        # y2 = self.layer_norm2(y2)

        y3 = self.forward3(y2)
        y3 = self.activation(y3)
        # y3 = self.dropout(y3)
        # y3 = self.layer_norm3(y3)

        y4 = self.forward4(y3)

        if not self.small:
            y4 = self.activation(y4)
            # y4 = self.dropout(y4)
            # y4 = self.layer_norm4(y4)
            # y4 = self.layer_norm4(y4)
            y5 = self.forward5(y4)

        if self.small:
            return (y0, y1, y2, y3), y4
        else:
            return (y0, y1, y2, y3, y4), y5
