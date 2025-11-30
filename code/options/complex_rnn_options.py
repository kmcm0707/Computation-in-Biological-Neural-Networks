from enum import Enum

import torch.nn.functional as functional
from options.complex_options import (
    nonLinearEnum,
    operatorEnum,
    yVectorEnum,
    zVectorEnum,
)


def pass_through(input):
    return input


class activationNonLinearEnum(Enum):
    relu = functional.relu
    softplus = "softplus"
    tanh = functional.tanh
    pass_through = pass_through


class complexRnnOptions:
    """
    Options for the complex synapse and individual complex synapse
    """

    def __init__(
        self,
        nonLinear: nonLinearEnum,
        update_rules=None,
        minSlowTau: int = 2,
        maxSlowTau: int = 50,
        minFastTau: int = 2,
        maxFastTau: int = 5,
        y_vector: yVectorEnum = yVectorEnum.none,
        z_vector: zVectorEnum = zVectorEnum.all_ones,
        y_fast_vector: yVectorEnum = yVectorEnum.none,
        z_fast_vector: zVectorEnum = zVectorEnum.all_ones,
        slow_operator: operatorEnum = operatorEnum.mode_6,
        fast_operator: operatorEnum = operatorEnum.mode_4,
    ):
        self.nonLinear = nonLinear
        self.fast_update_rules = update_rules
        self.minSlowTau = minSlowTau
        self.maxSlowTau = maxSlowTau
        self.minFastTau = minFastTau
        self.maxFastTau = maxFastTau
        self.y_vector = y_vector
        self.z_vector = z_vector
        self.y_fast_vector = y_fast_vector
        self.z_fast_vector = z_fast_vector
        self.slow_operator = slow_operator
        self.fast_operator = fast_operator

    def __str__(self):
        string = ""
        for key, value in vars(self).items():
            string += f"{key}: {value}\n"
        return string
