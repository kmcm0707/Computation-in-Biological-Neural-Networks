from typing import Union

from options.complex_options import (
    nonLinearEnum,
    operatorEnum,
    yVectorEnum,
    zVectorEnum,
)


class kernelRnnOptions:
    """
    Options for the complex synapse and individual complex synapse
    """

    def __init__(
        self,
        nonLinear: nonLinearEnum,
        update_rules=None,
        minSlowTau: int = 2,
        maxSlowTau: int = 50,
        y_vector: yVectorEnum = yVectorEnum.none,
        z_vector: zVectorEnum = zVectorEnum.all_ones,
        slow_operator: operatorEnum = operatorEnum.mode_6,
    ):
        self.nonLinear = nonLinear
        self.update_rules = update_rules
        self.minSlowTau = minSlowTau
        self.maxSlowTau = maxSlowTau
        self.y_vector = y_vector
        self.z_vector = z_vector
        self.slow_operator = slow_operator

    def __str__(self):
        string = ""
        for key, value in vars(self).items():
            string += f"{key}: {value}\n"
        return string
