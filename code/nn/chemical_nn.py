from typing import Literal

import torch
import torch.nn as nn
from options.meta_learner_options import sizeEnum, typeOfFeedbackEnum


class ChemicalNN(nn.Module):
    """

    Chemical Neural Network class.

    """

    @torch.no_grad()
    def __init__(
        self,
        device: Literal["cpu", "cuda"] = "cpu",
        numberOfChemicals: int = 1,
        size: sizeEnum = sizeEnum.normal,
        train_feedback: bool = False,
        typeOfFeedback: typeOfFeedbackEnum = typeOfFeedbackEnum.FA,
        dim_out: int = 47,
        error_control: bool = False,
    ):

        # Initialize the parent class
        super(ChemicalNN, self).__init__()

        # Set the device
        self.device = device
        self.size = size  # Size of the model
        self.train_feedback = train_feedback  # Train feedback for feedback alignment
        self.typeOfFeedback = typeOfFeedback  # Feedback alignment or direct feedback alignment
        self.error_control = error_control  # Error control for feedback alignment

        if self.typeOfFeedback == "DFA" and self.train_feedback:
            raise ValueError("DFA and train_feedback cannot be used together")

        if self.size != sizeEnum.normal and self.train_feedback:
            raise ValueError("Train feedback cannot be used with non-normal sized complex models")

        # Model
        self.dim_out = dim_out

        if self.size == sizeEnum.small:
            self.forward1 = nn.Linear(784, 128, bias=False)
            """self.forward2 = nn.Linear(15, 10, bias=False)
            self.forward3 = nn.Linear(10, 5, bias=False)"""
            self.forward2 = nn.Linear(128, self.dim_out, bias=False)
            """self.forward_layers = nn.ModuleList([self.forward1, self.forward2, self.forward3, self.forward4])"""
        elif self.size == sizeEnum.nine_layer:
            self.forward1 = nn.Linear(784, 650, bias=False)
            self.forward2 = nn.Linear(650, 512, bias=False)
            self.forward3 = nn.Linear(512, 384, bias=False)
            self.forward4 = nn.Linear(384, 256, bias=False)
            self.forward5 = nn.Linear(256, 170, bias=False)
            self.forward6 = nn.Linear(170, 130, bias=False)
            self.forward7 = nn.Linear(130, 100, bias=False)
            self.forward8 = nn.Linear(100, 70, bias=False)
            self.forward9 = nn.Linear(70, dim_out, bias=False)
        elif self.size == sizeEnum.six_layer:
            self.forward1 = nn.Linear(784, 170, bias=False)
            self.forward2 = nn.Linear(170, 150, bias=False)
            self.forward3 = nn.Linear(150, 130, bias=False)
            self.forward4 = nn.Linear(130, 100, bias=False)
            self.forward5 = nn.Linear(100, 70, bias=False)
            self.forward6 = nn.Linear(70, dim_out, bias=False)
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
            if self.size == sizeEnum.small:
                self.feedback1 = nn.Linear(784, 128, bias=False)
                # self.feedback2 = nn.Linear(15, 10, bias=False)
                # self.feedback3 = nn.Linear(10, 5, bias=False)
                self.feedback2 = nn.Linear(128, self.dim_out, bias=False)
            elif self.size == sizeEnum.nine_layer:
                self.feedback1 = nn.Linear(784, 650, bias=False)
                self.feedback2 = nn.Linear(650, 512, bias=False)
                self.feedback3 = nn.Linear(512, 384, bias=False)
                self.feedback4 = nn.Linear(384, 256, bias=False)
                self.feedback5 = nn.Linear(256, 170, bias=False)
                self.feedback6 = nn.Linear(170, 130, bias=False)
                self.feedback7 = nn.Linear(130, 100, bias=False)
                self.feedback8 = nn.Linear(100, 70, bias=False)
                self.feedback9 = nn.Linear(70, self.dim_out, bias=False)
            else:
                self.feedback1 = nn.Linear(784, 170, bias=False)
                self.feedback2 = nn.Linear(170, 130, bias=False)
                self.feedback3 = nn.Linear(130, 100, bias=False)
                self.feedback4 = nn.Linear(100, 70, bias=False)
                self.feedback5 = nn.Linear(70, self.dim_out, bias=False)
        # Direct feedback alignment
        elif self.typeOfFeedback == typeOfFeedbackEnum.DFA or self.typeOfFeedback == typeOfFeedbackEnum.DFA_grad:
            if self.size == sizeEnum.small:
                self.feedback1 = nn.Linear(784, self.dim_out, bias=False)
                self.feedback2 = nn.Linear(15, self.dim_out, bias=False)
                self.feedback3 = nn.Linear(10, self.dim_out, bias=False)
                self.feedback4 = nn.Linear(5, self.dim_out, bias=False)
            elif self.size == sizeEnum.six_layer:
                self.feedback1 = nn.Linear(784, self.dim_out, bias=False)
                self.feedback2 = nn.Linear(170, self.dim_out, bias=False)
                self.feedback3 = nn.Linear(150, self.dim_out, bias=False)
                self.feedback4 = nn.Linear(130, self.dim_out, bias=False)
                self.feedback5 = nn.Linear(100, self.dim_out, bias=False)
                self.feedback6 = nn.Linear(70, self.dim_out, bias=False)
            elif self.size == sizeEnum.nine_layer:
                self.feedback1 = nn.Linear(784, self.dim_out, bias=False)
                self.feedback2 = nn.Linear(650, self.dim_out, bias=False)
                self.feedback3 = nn.Linear(512, self.dim_out, bias=False)
                self.feedback4 = nn.Linear(384, self.dim_out, bias=False)
                self.feedback5 = nn.Linear(256, self.dim_out, bias=False)
                self.feedback6 = nn.Linear(170, self.dim_out, bias=False)
                self.feedback7 = nn.Linear(130, self.dim_out, bias=False)
                self.feedback8 = nn.Linear(100, self.dim_out, bias=False)
                self.feedback9 = nn.Linear(70, self.dim_out, bias=False)
            else:
                self.feedback1 = nn.Linear(784, self.dim_out, bias=False)
                self.feedback2 = nn.Linear(170, self.dim_out, bias=False)
                self.feedback3 = nn.Linear(130, self.dim_out, bias=False)
                self.feedback4 = nn.Linear(100, self.dim_out, bias=False)
                self.feedback5 = nn.Linear(70, self.dim_out, bias=False)
        elif self.typeOfFeedback == typeOfFeedbackEnum.scalar or self.typeOfFeedback == typeOfFeedbackEnum.scalar_rate:
            if self.size == sizeEnum.small:
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
            if self.size == sizeEnum.small:
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
        if self.size == sizeEnum.small:
            self.chemical1 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 128, 784), device=self.device))
            # self.chemical2 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 10, 15), device=self.device))
            # self.chemical3 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 5, 10), device=self.device))
            self.chemical2 = nn.Parameter(torch.zeros(size=(numberOfChemicals, self.dim_out, 128), device=self.device))
            self.chemicals = nn.ParameterList([self.chemical1, self.chemical2])  # , self.chemical3, self.chemical4])
        elif self.size == sizeEnum.nine_layer:
            self.chemical1 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 650, 784), device=self.device))
            self.chemical2 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 512, 650), device=self.device))
            self.chemical3 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 384, 512), device=self.device))
            self.chemical4 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 256, 384), device=self.device))
            self.chemical5 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 170, 256), device=self.device))
            self.chemical6 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 130, 170), device=self.device))
            self.chemical7 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 100, 130), device=self.device))
            self.chemical8 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 70, 100), device=self.device))
            self.chemical9 = nn.Parameter(torch.zeros(size=(numberOfChemicals, self.dim_out, 70), device=self.device))
            self.chemicals = nn.ParameterList(
                [
                    self.chemical1,
                    self.chemical2,
                    self.chemical3,
                    self.chemical4,
                    self.chemical5,
                    self.chemical6,
                    self.chemical7,
                    self.chemical8,
                    self.chemical9,
                ]
            )
        elif self.size == sizeEnum.six_layer:
            self.chemical1 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 170, 784), device=self.device))
            self.chemical2 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 150, 170), device=self.device))
            self.chemical3 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 130, 150), device=self.device))
            self.chemical4 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 100, 130), device=self.device))
            self.chemical5 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 70, 100), device=self.device))
            self.chemical6 = nn.Parameter(torch.zeros(size=(numberOfChemicals, dim_out, 70), device=self.device))
            self.chemicals = nn.ParameterList(
                [
                    self.chemical1,
                    self.chemical2,
                    self.chemical3,
                    self.chemical4,
                    self.chemical5,
                    self.chemical6,
                ]
            )
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
        # self.dropout = nn.Dropout(p=0.2)

    def set_errors(self, errors):
        self.errors = errors

    # @torch.compile
    def forward(self, x):
        if self.size == sizeEnum.small:
            y0 = x.squeeze(1)
            y1 = self.forward1(y0)
            y1 = self.activation(y1)
            y2 = self.forward2(y1)
        elif self.size == sizeEnum.nine_layer:
            y0 = x.squeeze(1)
            y1 = self.forward1(y0)
            y1 = self.activation(y1)
            y2 = self.forward2(y1)
            y2 = self.activation(y2)
            y3 = self.forward3(y2)
            y3 = self.activation(y3)
            y4 = self.forward4(y3)
            y4 = self.activation(y4)
            y5 = self.forward5(y4)
            y5 = self.activation(y5)
            y6 = self.forward6(y5)
            y6 = self.activation(y6)
            y7 = self.forward7(y6)
            y7 = self.activation(y7)
            y8 = self.forward8(y7)
            y8 = self.activation(y8)
            y9 = self.forward9(y8)

            return (y0, y1, y2, y3, y4, y5, y6, y7, y8), y9
        elif self.size == sizeEnum.six_layer:
            y0 = x.squeeze(1)
            y1 = self.forward1(y0)
            y1 = self.activation(y1)
            y2 = self.forward2(y1)
            y2 = self.activation(y2)
            y3 = self.forward3(y2)
            y3 = self.activation(y3)
            y4 = self.forward4(y3)
            y4 = self.activation(y4)
            y5 = self.forward5(y4)
            y5 = self.activation(y5)
            y6 = self.forward6(y5)
            return (y0, y1, y2, y3, y4, y5), y6
        else:

            if self.error_control:
                y0 = x.squeeze(1) + self.errors[0]
                y0 = self.activation(y0)

                y1 = self.forward1(y0) + self.errors[1]
                y1 = self.activation(y1)

                y2 = self.forward2(y1) + self.errors[2]
                y2 = self.activation(y2)

                y3 = self.forward3(y2) + self.errors[3]
                y3 = self.activation(y3)

                y4 = self.forward4(y3) + self.errors[4]
                y4 = self.activation(y4)

                y5 = self.forward5(y4) + self.errors[5]

            else:
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
                y4 = self.activation(y4)
                # y4 = self.dropout(y4)
                # y4 = self.layer_norm4(y4)
                # y4 = self.layer_norm4(y4)
                y5 = self.forward5(y4)

        if self.size == sizeEnum.small:
            return (y0, y1), y2
        else:
            return (y0, y1, y2, y3, y4), y5
