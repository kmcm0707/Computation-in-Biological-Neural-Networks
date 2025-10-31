import datetime
from typing import Literal

from options.meta_learner_options import (
    chemicalEnum,
    modelEnum,
    sizeEnum,
    typeOfFeedbackEnum,
)


class RunnerOptions:
    """
    Options for the metal learner
    """

    def __init__(
        self,
        model: modelEnum,
        modelPath: str = None,
        results_subdir: str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        seed: int = 0,
        size: sizeEnum = sizeEnum.normal,
        save_results: bool = True,
        metatrain_dataset: str = None,
        display: bool = True,
        numberOfClasses: int = 5,
        dataset_name: Literal["EMNIST", "FASHION-MNIST"] = "FASHION-MNIST",
        chemicalInitialization: chemicalEnum = chemicalEnum.same,
        trainFeedback: bool = False,
        trainSameFeedback: bool = False,
        feedbackModel: modelEnum = modelEnum.complex,
        minTrainingDataPerClass: int = 40,
        maxTrainingDataPerClass: int = 60,
        queryDataPerClass: int = 10,
        typeOfFeedback: typeOfFeedbackEnum = typeOfFeedbackEnum.FA,
        dimOut: int = 47,
        data_repetitions: int = 1,
        wta: bool = False,
        chemical_analysis: bool = False,
    ):

        self.model = model
        self.modelPath = modelPath
        self.size = size
        self.seed = seed
        self.save_results = save_results
        self.results_subdir = results_subdir
        self.metatrain_dataset = metatrain_dataset
        self.display = display
        self.numberOfClasses = numberOfClasses
        self.dataset_name = dataset_name
        self.chemicalInitialization = chemicalInitialization
        self.trainFeedback = trainFeedback
        self.trainSameFeedback = trainSameFeedback
        self.feedbackModel = feedbackModel
        self.minTrainingDataPerClass = minTrainingDataPerClass
        self.maxTrainingDataPerClass = maxTrainingDataPerClass
        self.queryDataPerClass = queryDataPerClass
        self.typeOfFeedback = typeOfFeedback
        self.dimOut = dimOut
        self.data_repetitions = data_repetitions
        self.wta = wta
        self.chemical_analysis = chemical_analysis

    def __str__(self):
        string = ""
        for key, value in vars(self).items():
            string += f"{key}: {value}\n"
        return string
