from enum import Enum

import jax
from options.meta_learner_options import chemicalEnum


def pass_through(input):
    return input


class JaxActivationNonLinearEnum(Enum):
    softplus = "softplus"
    tanh = jax.nn.tanh
    pass_through = pass_through


class JaxRnnMetaLearnerOptions:
    def __init__(
        self,
        seed: int = 0,
        save_results: bool = True,
        results_subdir: str = "jax_rnn_meta_learner",
        metatrain_dataset: str = "emnist",
        display: bool = True,
        metaLearningRate: float = 0.001,
        numberOfClasses: int = 5,
        dataset_name: str = "emnist",
        chemicalInitialization: chemicalEnum = chemicalEnum.same,
        minTrainingDataPerClass: int = 5,
        maxTrainingDataPerClass: int = 10,
        queryDataPerClass: int = 10,
        input_size: int = 28,
        hidden_size: int = 128,
        output_size: int = 1,
        biological_min_tau: int = 5,
        biological_max_tau: int = 25,
        gradient: bool = False,
        outer_activation=JaxActivationNonLinearEnum.tanh,
        recurrent_activation=JaxActivationNonLinearEnum.softplus,
        number_of_time_steps: int = 10,
        load_model: str = None,
    ):
        self.seed = seed
        self.save_results = save_results
        self.results_subdir = results_subdir
        self.metatrain_dataset = metatrain_dataset
        self.display = display
        self.metaLearningRate = metaLearningRate
        self.numberOfClasses = numberOfClasses
        self.dataset_name = dataset_name
        self.chemicalInitialization = chemicalInitialization
        self.minTrainingDataPerClass = minTrainingDataPerClass
        self.maxTrainingDataPerClass = maxTrainingDataPerClass
        self.queryDataPerClass = queryDataPerClass
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.biological_min_tau = biological_min_tau
        self.biological_max_tau = biological_max_tau
        self.gradient = gradient
        self.outer_activation = outer_activation
        self.recurrent_activation = recurrent_activation
        self.number_of_time_steps = number_of_time_steps
        self.load_model = load_model

    def __str__(self):
        string = ""
        for key, value in vars(self).items():
            string += f"{key}: {value}\n"
        return string
