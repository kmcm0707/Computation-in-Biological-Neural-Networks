import datetime
from enum import Enum
from typing import Literal

from options.meta_learner_options import chemicalEnum, optimizerEnum
from synapses.complex_rnn import ComplexRnn
from synapses.kernel_rnn import KernelRnn


class rnnModelEnum(Enum):
    """
    Enum for the model
    """

    complex = ComplexRnn
    kernel = KernelRnn


class RnnMetaLearnerOptions:
    """
    Options for the metal learner
    """

    def __init__(
        self,
        optimizer: optimizerEnum,
        model: rnnModelEnum,
        results_subdir: str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        seed: int = 0,
        save_results: bool = True,
        metatrain_dataset: str = None,
        display: bool = True,
        lr: float = 1e-3,
        numberOfClasses: int = 5,
        dataset_name: Literal["EMNIST", "FASHION-MNIST"] = "EMNIST",
        chemicalInitialization: chemicalEnum = chemicalEnum.same,
        minTrainingDataPerClass: int = 50,
        maxTrainingDataPerClass: int = 50,
        queryDataPerClass: int = 10,
        rnn_input_size: int = 28,
        datasetDevice: Literal["cpu", "cuda"] = "cuda",
        continueTraining: str = None,
        reset_fast_weights: bool = True,
        requireFastChemical: bool = False,
        dimOut: int = 1,
        biological: bool = False,
        biological_min_tau: int = 1,
        biological_max_tau: int = 56,
    ):

        self.model = model
        self.optimizer = optimizer
        self.seed = seed
        self.save_results = save_results
        self.results_subdir = results_subdir
        self.metatrain_dataset = metatrain_dataset
        self.display = display
        self.lr = lr
        self.numberOfClasses = numberOfClasses
        self.dataset_name = dataset_name
        self.chemicalInitialization = chemicalInitialization
        self.minTrainingDataPerClass = minTrainingDataPerClass
        self.maxTrainingDataPerClass = maxTrainingDataPerClass
        self.rnn_input_size = rnn_input_size
        self.queryDataPerClass = queryDataPerClass
        self.datasetDevice = datasetDevice
        self.continueTraining = continueTraining
        self.reset_fast_weights = reset_fast_weights
        self.requireFastChemical = requireFastChemical
        self.dimOut = dimOut
        self.biological = biological
        self.biological_min_tau = biological_min_tau
        self.biological_max_tau = biological_max_tau

    def __str__(self):
        string = ""
        for key, value in vars(self).items():
            string += f"{key}: {value}\n"
        return string
