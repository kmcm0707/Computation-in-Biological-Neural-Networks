from enum import Enum
from typing import Union

from torch.nn import functional


def pass_through(input):
    return input


class nonLinearEnum(Enum):
    tanh = functional.tanh
    gelu = (
        functional.gelu
    )  # No idea why this is not working but manually setting it to functional.gelu in the code works????
    relu = functional.relu
    sigmoid = functional.sigmoid
    elu = functional.elu
    leaky_relu = functional.leaky_relu
    pass_through = pass_through
    softplus = functional.softplus


class pMatrixEnum(Enum):
    random = "random"
    rosenbaum_last = "rosenbaum_last"
    rosenbaum_first = "rosenbaum_first"
    first_col = "first_col"
    zero = "zero"


class kMatrixEnum(Enum):
    random = "random"
    xavier = "xavier"
    uniform = "uniform"
    zero = "zero"
    identity = "identity"


class zVectorEnum(Enum):
    random = "random"
    all_ones = "all_ones"
    default = "default"


class yVectorEnum(Enum):
    none = "none"
    first_one = "first_one"
    last_one = "last_one"
    last_one_and_small_first = "last_one_and_small_first"
    all_ones = "all_ones"
    half = "half"


class operatorEnum(Enum):
    mode_1 = "mode_1"
    mode_2 = "mode_2"
    mode_3 = "mode_3"
    mode_4 = "mode_4"
    mode_5 = "mode_5"
    mode_6 = "mode_6"
    mode_7 = "mode_7"
    sub = "sub"
    sub_4 = "sub_4"
    attention = "attention"
    extended_attention = "extended_attention"
    attention_2 = "attention_2"
    full_attention = "full_attention"
    compressed_full_attention = "compressed_full_attention"
    v_linear = "v_linear"
    compressed_v_linear = "compressed_v_linear"


class vVectorEnum(Enum):
    default = "default"
    random = "random"
    last_one = "last_one"
    random_small = "random_small"
    random_beta = "random_beta"


class modeEnum(Enum):
    rosenbaum = "rosenbaum"
    all_rosenbaum = "all_rosenbaum"
    all = "all"


class complexOptions:
    """
    Options for the complex synapse and individual complex synapse
    """

    def __init__(
        self,
        nonLinear: nonLinearEnum,
        bias: bool = True,
        update_rules=None,
        pMatrix: pMatrixEnum = pMatrixEnum.first_col,
        kMatrix: kMatrixEnum = kMatrixEnum.random,
        minTau=1,
        maxTau=50,
        y_vector: yVectorEnum = yVectorEnum.first_one,
        z_vector: zVectorEnum = zVectorEnum.default,
        operator: operatorEnum = operatorEnum.mode_1,
        train_z_vector: bool = False,
        mode: modeEnum = modeEnum.all,
        v_vector: vVectorEnum = vVectorEnum.default,
        eta: float = 1e-3,
        beta: float = 0,
        kMasking: bool = False,
        individual_different_v_vector: bool = False,
        scheduler_t0: Union[int, None] = None,
        train_tau: bool = False,
    ):
        self.nonLinear = nonLinear
        self.bias = bias
        self.update_rules = update_rules
        self.pMatrix = pMatrix
        self.kMatrix = kMatrix
        self.minTau = minTau
        self.maxTau = maxTau
        self.y_vector = y_vector
        self.z_vector = z_vector
        self.operator = operator
        self.train_z_vector = train_z_vector
        self.mode = mode
        self.v_vector = v_vector
        self.eta = eta
        self.beta = beta
        self.kMasking = kMasking
        self.individual_different_v_vector = individual_different_v_vector
        self.scheduler_t0 = scheduler_t0
        self.train_tau = train_tau

    def __str__(self):
        string = ""
        for key, value in vars(self).items():
            string += f"{key}: {value}\n"
        return string
