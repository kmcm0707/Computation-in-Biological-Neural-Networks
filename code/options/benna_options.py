from enum import Enum

from options.complex_options import nonLinearEnum


class vVectorReservoirEnum(Enum):
    small_random = "small_random"
    default = "default"


class modeReservoirEnum(Enum):
    mode_1 = "mode_1"
    mode_3 = "mode_3"


class bennaOptions:
    """
    Options for the benna and fusi synapse
    """

    def __init__(
        self,
        non_linearity: nonLinearEnum,
        bias: bool,
        update_rules=None,
        minTau=1,
        maxTau=50,
    ):

        self.non_linearity = non_linearity
        self.bias = bias
        self.update_rules = update_rules
        self.minTau = minTau
        self.maxTau = maxTau

    def __str__(self):
        string = ""
        for key, value in vars(self).items():
            string += f"{key}: {value}\n"
        return string
