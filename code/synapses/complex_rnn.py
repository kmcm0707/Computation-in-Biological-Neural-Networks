import math
from typing import Literal

import numpy as np
import torch
from options.complex_options import complexOptions
from synapses.complex_synapse import ComplexSynapse
from torch import nn


class ComplexRnn(nn.Module):
    """
    Complex rnn model.
    The class implements a complex synapse rnn model.
    Which is a biological plausible update rule.
    """

    def __init__(
        self,
        device="cpu",
        numberOfSlowChemicals: int = 1,
        numberOfFastChemicals: int = 1,
        params: dict = {},
        complexSlowOptions: complexOptions = None,
        complexFastOptions: complexOptions = None,
        adaptionPathway: Literal["forward", "feedback"] = "forward",
    ):
        """
        Initialize the complex synapse model.
        :param device: (str) The processing device to use. Default is 'cpu',
        :param mode: (str) The update rule to use. Default is 'rosenbaum'.
        :param numberOfChemicals: (int) The number of chemicals to use. Default is 1,
        :param params: (dict) The model parameters. Default is an empty dictionary,
        :param complexOptions: (complexOptions) The complex options. Default is None,
        :param adaptionPathway: (str) The adaption pathway to use. Default is 'forward'.
        """
        super(ComplexRnn, self).__init__()

        self.device = device
        assert complexSlowOptions.mode == "all" and complexFastOptions.mode == "all"
        self.complexSlowOptions = complexSlowOptions
        self.complexFastOptions = complexFastOptions
        self.mode = "all"  # rosenbaum not used
        self.adaptionPathway = adaptionPathway
        self.numberOfSlowChemicals = numberOfSlowChemicals
        self.numberOfFastChemicals = numberOfFastChemicals
        self.params = params

        self.ComplexSlow = ComplexSynapse(
            device=device,
            numberOfChemicals=numberOfSlowChemicals,
            complexOptions=complexSlowOptions,
            params=params,
            adaptionPathway=adaptionPathway,
        )

        self.ComplexFast = ComplexSynapse(
            device=device,
            numberOfChemicals=numberOfFastChemicals,
            complexOptions=complexFastOptions,
            params=params,
            adaptionPathway=adaptionPathway,
        )

    def fast_update(
        self,
        params: dict,
        slow_h_parameters: dict,
        activations_and_output: list,
        error: list,
    ):
        """
        Fast update.
        The function updates the fast synapse.
        :param params: (dict) The model parameters,
        :param h_parameters: (dict) The hidden parameters,
        :param activations_and_output: (list) The activations and output,
        :param error: (list) The error (atm not used).
        """
        self.ComplexFast(
            params=params, h_parameters=slow_h_parameters, activations_and_output=activations_and_output, error=error
        )

    def slow_update(
        self,
        params: dict,
        activations_and_output: list,
        error: list,
    ):
        """
        Slow update.
        The function updates the slow synapse.
        :param params: (dict) The model parameters,
        :param activations_and_output: (list) The activations and output,
        :param error: (list) The error.
        """
        self.ComplexSlow(params=params, activations_and_output=activations_and_output, error=error)
