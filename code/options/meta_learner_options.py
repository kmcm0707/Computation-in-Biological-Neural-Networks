import datetime
from enum import Enum
from typing import Literal

from synapses.complex_synapse import ComplexSynapse
from synapses.individual_synapse import IndividualSynapse
from synapses.reservoir_synapse import ReservoirSynapse


class schedulerEnum(Enum):
    exponential = "exponential"
    linear = "linear"
    constant = "constant"
    none = "none"


class optimizerEnum(Enum):
    adam = "adam"
    adamW = "adamW"
    sgd = "sgd"


class modelEnum(Enum):
    complex = ComplexSynapse
    reservoir = ReservoirSynapse
    individual = IndividualSynapse


class MetaLearnerOptions:
    """
    Options for the metal learner
    """

    def __init__(
        self,
        scheduler: schedulerEnum,
        metaLossRegularization: int,
        biasLossRegularization: int,
        optimizer: optimizerEnum,
        model: modelEnum,
        results_subdir: str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        seed: int = 0,
        small: bool = False,
        raytune: bool = False,
        save_results: bool = True,
        metatrain_dataset: str = None,
        display: bool = True,
        lr: float = 1e-3,
    ):

        self.model = model
        self.small = small
        self.scheduler = scheduler
        self.metaLossRegularization = metaLossRegularization
        self.biasLossRegularization = biasLossRegularization
        self.optimizer = optimizer
        self.seed = seed
        self.raytune = raytune
        self.save_results = save_results
        self.results_subdir = results_subdir
        self.metatrain_dataset = metatrain_dataset
        self.display = display
        self.lr = lr

    def __str__(self):
        string = ""
        for key, value in vars(self).items():
            string += f"{key}: {value}\n"
        return string
