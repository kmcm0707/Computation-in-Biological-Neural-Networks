from enum import Enum

from options.complex_options import nonLinearEnum


class vVectorReservoirEnum(Enum):
    small_random = "small_random"
    default = "default"


class modeReservoirEnum(Enum):
    mode_1 = "mode_1"
    mode_3 = "mode_3"


class reservoirOptions:
    """
    Options for the reservoir synapse
    """

    def __init__(
        self,
        non_linearity: nonLinearEnum,
        reservoir_size: int,
        bias: bool,
        spectral_radius: float,
        unit_connections: int,
        update_rules=None,
        reservoir_seed: int = 0,
        train_K_matrix: bool = False,
        minTau=1,
        maxTau=50,
        v_vector: vVectorReservoirEnum = vVectorReservoirEnum.default,
    ):

        self.non_linearity = non_linearity
        self.reservoir_size = reservoir_size
        self.bias = bias
        self.spectral_radius = spectral_radius
        self.unit_connections = unit_connections
        self.update_rules = update_rules
        self.reservoir_seed = reservoir_seed
        self.train_K_matrix = train_K_matrix
        self.minTau = minTau
        self.maxTau = maxTau
        self.v_vector = v_vector

    def __str__(self):
        string = ""
        for key, value in vars(self).items():
            string += f"{key}: {value}\n"
        return string