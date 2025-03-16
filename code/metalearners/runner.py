import argparse
import copy
import datetime
import os
import random
import sys
from multiprocessing import Pool
from typing import Literal, Union

import numpy as np
import torch
from misc.dataset import DataProcess, EmnistDataset, FashionMnistDataset
from misc.utils import Plot, log, meta_stats
from nn.chemical_nn import ChemicalNN
from options.benna_options import bennaOptions
from options.complex_options import (
    complexOptions,
    kMatrixEnum,
    modeEnum,
    nonLinearEnum,
    operatorEnum,
    pMatrixEnum,
    vVectorEnum,
    yVectorEnum,
    zVectorEnum,
)
from options.meta_learner_options import (
    MetaLearnerOptions,
    chemicalEnum,
    modelEnum,
    optimizerEnum,
    schedulerEnum,
    typeOfFeedbackEnum,
)
from options.reservoir_options import (
    modeReservoirEnum,
    reservoirOptions,
    vVectorReservoirEnum,
    yReservoirEnum,
)
from options.runner_options import RunnerOptions
from ray import train
from synapses.benna_synapse import BennaSynapse
from synapses.complex_synapse import ComplexSynapse
from synapses.individual_synapse import IndividualSynapse
from synapses.reservoir_synapse import ReservoirSynapse
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter


class Runner:
    """

    Runner class.

    Runs a pre-trained model on a given dataset.

    """

    def __init__(
        self,
        device: Literal["cpu", "cuda"] = "cpu",
        numberOfChemicals: int = 1,
        runnerOptions: RunnerOptions = None,
        modelOptions: Union[complexOptions, reservoirOptions] = None,
        feedbackModelOptions: Union[complexOptions, reservoirOptions] = None,
    ):

        # -- processor params
        self.device = torch.device(device)
        self.modelOptions = modelOptions
        self.options = runnerOptions
        self.feedbackModelOptions = feedbackModelOptions

        # -- data params
        self.queryDataPerClass = self.options.queryDataPerClass
        self.metatrain_dataset = self.options.metatrain_dataset
        self.data_process = DataProcess(
            minTrainingDataPerClass=self.options.minTrainingDataPerClass,
            maxTrainingDataPerClass=self.options.maxTrainingDataPerClass,
            queryDataPerClass=self.queryDataPerClass,
            dimensionOfImage=28,
            device=self.device,
        )

        # -- model params
        self.numberOfChemicals = numberOfChemicals
        self.model = self.load_model().to(self.device)

        self.loss_func = nn.CrossEntropyLoss()

        # -- set chemical model
        self.UpdateWeights = self.chemical_model_setter(
            options=self.modelOptions, adaptionPathway="forward", typeOfModel=self.options.model
        )

        if self.options.trainFeedback:
            self.UpdateFeedbackWeights = self.chemical_model_setter(
                options=self.feedbackModelOptions, adaptionPathway="feedback", typeOfModel=self.options.feedbackModel
            )

        # -- load chemical model
        state_dict = torch.load(
            self.options.modelPath + "/UpdateWeights.pth", weights_only=True, map_location=self.device
        )
        if self.options.model == modelEnum.individual:
            if "v_vector" in state_dict:
                state_dict["v_dictionary.all"] = state_dict["v_vector"].clone()
                state_dict.pop("v_vector")
        self.UpdateWeights.load_state_dict(state_dict)
        if self.options.trainFeedback:
            feedback_state_dict = torch.load(
                self.options.modelPath + "/UpdateWeights.pth", weights_only=True, map_location=self.device
            )  # UpdateFeedbackWeights
            for key, val in self.UpdateWeights.named_parameters():
                if "bias_dictionary.chemical" in key:
                    name = "bias_dictionary.feedback_chemical" + key[-1]
                    feedback_state_dict[name] = feedback_state_dict[key].clone()
                    feedback_state_dict.pop(key)
            if self.options.feedbackModel == modelEnum.individual:
                if "v_vector" in feedback_state_dict:
                    feedback_state_dict["v_dictionary.all"] = feedback_state_dict["v_vector"].clone()
                    feedback_state_dict.pop("v_vector")
            self.UpdateFeedbackWeights.load_state_dict(feedback_state_dict)

        # -- log params
        self.save_results = self.options.save_results
        self.display = self.options.display
        self.result_directory = os.getcwd() + "/results"
        if self.save_results:
            self.result_directory = os.getcwd() + "/results"
            os.makedirs(self.result_directory, exist_ok=True)
            self.result_directory += (
                "/"
                + self.options.results_subdir
                + "/"
                + str(self.options.seed)
                + "/"
                + str(self.options.minTrainingDataPerClass)
            )
            os.makedirs(self.result_directory, exist_ok=False)
            with open(self.result_directory + "/arguments.txt", "w") as f:
                f.writelines("Number of chemicals: {}\n".format(numberOfChemicals))
                f.writelines("Number of query data per class: {}\n".format(self.queryDataPerClass))
                f.writelines(str(self.options))
                f.writelines(str(modelOptions))
                if self.options.trainFeedback:
                    f.writelines(str(feedbackModelOptions))

            self.average_window = 10
            self.plot = Plot(self.result_directory, self.average_window)
            self.summary_writer = SummaryWriter(log_dir=self.result_directory)

    def chemical_model_setter(
        self, options: Union[complexOptions, reservoirOptions], adaptionPathway="forward", typeOfModel=None
    ):
        model = None
        if typeOfModel == modelEnum.complex:
            model = ComplexSynapse(
                device=self.device,
                numberOfChemicals=self.numberOfChemicals,
                complexOptions=options,
                params=self.model.named_parameters(),
                adaptionPathway=adaptionPathway,
            )
        elif typeOfModel == modelEnum.reservoir:
            model = ReservoirSynapse(
                device=self.device,
                numberOfChemicals=self.numberOfChemicals,
                options=options,
                params=self.model.named_parameters(),
            )
        elif typeOfModel == modelEnum.individual:
            model = IndividualSynapse(
                device=self.device,
                numberOfChemicals=self.numberOfChemicals,
                complexOptions=options,
                params=self.model.named_parameters(),
            )
        elif typeOfModel == modelEnum.benna:
            model = BennaSynapse(
                device=self.device,
                numberOfChemicals=self.numberOfChemicals,
                options=options,
                params=self.model.named_parameters(),
            )
        else:
            raise ValueError("Model not recognized.")
        return model

    def load_model(self):
        """
            Load classifier model

        Loads the classifier network and sets the adaptation, meta-learning,
        and grad computation flags for its variables.

        :param args: (argparse.Namespace) The command-line arguments.
        :return: model with flags , "adapt", set for its parameters
        """

        model = ChemicalNN(
            self.device,
            self.numberOfChemicals,
            small=self.options.small,
            train_feedback=self.options.trainFeedback,
            typeOfFeedback=self.options.typeOfFeedback,
            dim_out=self.options.dimOut,
        )

        # -- learning flags
        for key, val in model.named_parameters():
            if "forward" in key:
                val.adapt = "forward"
            elif "feedback" in key:
                val.adapt, val.requires_grad = "feedback", self.options.trainFeedback

        return model

    @staticmethod
    def weights_init(modules):
        classname = modules.__class__.__name__
        if classname.find("Linear") != -1:

            # -- weights
            nn.init.xavier_uniform_(modules.weight)

            # -- bias
            if modules.bias is not None:
                nn.init.xavier_uniform_(modules.bias)

    @torch.no_grad()
    def chemical_init(self, chemicals):
        if self.options.chemicalInitialization == chemicalEnum.same:
            for chemical in chemicals:
                nn.init.xavier_uniform_(chemical[0])
                for idx in range(chemical.shape[0] - 1):
                    chemical[idx + 1] = chemical[0]
        elif self.options.chemicalInitialization == chemicalEnum.zero:
            for chemical in chemicals:
                nn.init.zeros_(chemical)
        elif self.options.chemicalInitialization == chemicalEnum.different:
            for chemical in chemicals:
                for idx in range(chemical.shape[0]):
                    nn.init.xavier_uniform_(chemical[idx])
            if self.numberOfChemicals > 1:
                assert chemicals[0][0] is not chemicals[0][1]
        else:
            raise ValueError("Invalid Chemical Initialization")

    @torch.no_grad()
    def reinitialize(self):
        """
            Initialize module parameters.

        Initializes and clones the model parameters, creating a separate copy
        of the data in new memory. This duplication enables the modification
        of the parameters using inplace operations, which allows updating the
        parameters with a customized meta-learned optimizer.

        :return: dict: module parameters
        """
        # -- initialize weights
        self.model.apply(self.weights_init)
        self.chemical_init(self.model.chemicals)
        if self.options.trainFeedback:
            self.chemical_init(self.model.feedback_chemicals)

        # -- module parameters
        # -- parameters are not linked to the model even if .clone() is not used
        params = {
            key: val.clone()
            for key, val in dict(self.model.named_parameters()).items()
            if "." in key and "chemical" not in key and "layer_norm" not in key
        }
        h_params = {
            key: val.clone()
            for key, val in dict(self.model.named_parameters()).items()
            if "chemical" in key and "feedback" not in key
        }
        feedback_h_params = None
        if self.options.trainFeedback:
            feedback_h_params = {
                key: val.clone()
                for key, val in dict(self.model.named_parameters()).items()
                if "feedback_chemical" in key
            }

        # -- set adaptation flags for cloned parameters
        for key in params:
            params[key].adapt = dict(self.model.named_parameters())[key].adapt

        return params, h_params, feedback_h_params

    @torch.no_grad()
    def run(self):
        """
            Perform runner

        :return: None
        """
        self.model.train()
        self.UpdateWeights.eval()
        if self.options.trainFeedback:
            self.UpdateFeedbackWeights.eval()

        x_qry = None
        y_qry = None
        current_training_data_per_class = None

        for eps, data in enumerate(self.metatrain_dataset):

            # -- initialize
            # Using a clone of the model parameters to allow for in-place operations
            # Maintains the computational graph for the model as .detach() is not used
            parameters, h_parameters, feedback_params = self.reinitialize()

            if (
                self.options.chemicalInitialization != chemicalEnum.zero
                and self.modelOptions.operator != operatorEnum.mode_3
            ):
                self.UpdateWeights.initial_update(parameters, h_parameters)
                if self.options.trainFeedback and self.feedbackModelOptions.operator != operatorEnum.mode_3:
                    self.UpdateFeedbackWeights.initial_update(parameters, feedback_params)

            # -- reset time index
            self.UpdateWeights.reset_time_index()
            if self.options.trainFeedback:
                self.UpdateFeedbackWeights.reset_time_index()

            # -- training data
            x_trn, y_trn, x_qry, y_qry, current_training_data_per_class = self.data_process(
                data, self.options.numberOfClasses
            )

            """ adaptation """
            for itr_adapt, (x, label) in enumerate(zip(x_trn, y_trn)):

                # -- predict
                y, logits = None, None
                if self.options.trainFeedback:
                    y, logits = torch.func.functional_call(
                        self.model, (parameters, h_parameters, feedback_params), x.unsqueeze(0).unsqueeze(0)
                    )
                else:
                    y, logits = torch.func.functional_call(
                        self.model, (parameters, h_parameters), x.unsqueeze(0).unsqueeze(0)
                    )

                # -- compute error
                activations = y
                output = logits
                params = parameters
                feedback = {name: value for name, value in params.items() if "feedback" in name}
                error = [functional.softmax(output, dim=1) - functional.one_hot(label, num_classes=self.options.dimOut)]
                # add the error for all the layers
                if self.options.typeOfFeedback == typeOfFeedbackEnum.FA:
                    # add the error for all the layers
                    for y, i in zip(reversed(activations), reversed(list(feedback))):
                        error.insert(0, torch.matmul(error[0], feedback[i]) * (1 - torch.exp(-self.model.beta * y)))
                elif self.options.typeOfFeedback == typeOfFeedbackEnum.FA_NO_GRAD:
                    # add the error for all the layers
                    for y, i in zip(reversed(activations), reversed(list(feedback))):
                        error.insert(0, torch.matmul(error[0], feedback[i]))
                elif self.options.typeOfFeedback == typeOfFeedbackEnum.DFA:
                    for y, i in zip(reversed(activations), reversed(list(feedback))):
                        error.insert(0, torch.matmul(error[-1], feedback[i]))
                elif self.options.typeOfFeedback == typeOfFeedbackEnum.DFA_grad:
                    for y, i in zip(reversed(activations), reversed(list(feedback))):
                        error.insert(0, torch.matmul(error[-1], feedback[i]) * (1 - torch.exp(-self.model.beta * y)))
                elif self.options.typeOfFeedback == typeOfFeedbackEnum.scalar:
                    error_scalar = torch.norm(error[0], p=2, dim=1, keepdim=True)
                    for y, i in zip(reversed(activations), reversed(list(feedback))):
                        error.insert(0, torch.matmul(error_scalar, feedback[i]))
                elif self.options.typeOfFeedback == typeOfFeedbackEnum.DFA_grad_FA:
                    DFA_feedback = {name: value for name, value in params.items() if "DFA_feedback" in name}
                    feedback = {name: value for name, value in params.items() if "feedback_FA" in name}
                    DFA_error = [
                        functional.softmax(output, dim=1) - functional.one_hot(label, num_classes=self.options.dimOut)
                    ]
                    for y, i in zip(reversed(activations), reversed(list(DFA_feedback))):
                        DFA_error.insert(
                            0, torch.matmul(error[-1], DFA_feedback[i]) * (1 - torch.exp(-self.model.beta * y))
                        )
                    index_error = len(DFA_error) - 2
                    for y, i in zip(reversed(activations), reversed(list(feedback))):
                        error.insert(
                            0,
                            (
                                torch.matmul(error[0], feedback[i]) * (1 - torch.exp(-self.model.beta * y))
                                + DFA_error[index_error]
                            )
                            / 2,
                        )
                        index_error -= 1
                    """for i in range(len(DFA_error)):
                        # error[i] = (error[i] + DFA_error[i]) / 2
                        if i != 0:
                            error[i] = (error[i] + DFA_error[i]) / np.sqrt(2)"""
                else:
                    raise ValueError("Invalid type of feedback")
                activations_and_output = [*activations, functional.softmax(output, dim=1)]

                # -- update network params
                self.UpdateWeights(
                    params=parameters,
                    h_parameters=h_parameters,
                    error=error,
                    activations_and_output=activations_and_output,
                )

                # -- update time index
                self.UpdateWeights.update_time_index()

                # -- update feedback params
                if self.options.trainFeedback:
                    self.UpdateFeedbackWeights(
                        params=parameters,
                        h_parameters=feedback_params,
                        error=error,
                        activations_and_output=activations_and_output,
                    )

                    # -- update feedback time index
                    self.UpdateFeedbackWeights.update_time_index()

            """ meta test"""
            for key, val in parameters.items():
                print("Before")
                print(torch.norm(val))
            # -- predict
            y, logits = None, None
            if self.options.trainFeedback:
                y, logits = torch.func.functional_call(self.model, (parameters, h_parameters, feedback_params), x_qry)
            else:
                y, logits = torch.func.functional_call(self.model, (parameters, h_parameters), x_qry)

            loss_meta = self.loss_func(logits, y_qry.ravel())

            # -- compute and store meta stats
            acc = meta_stats(
                logits,
                parameters,
                y_qry.ravel(),
                y,
                self.model.beta,
                self.result_directory,
                self.save_results,
                typeOfFeedback=self.options.typeOfFeedback,
            )

            # -- log
            if self.save_results:
                log([loss_meta.item()], self.result_directory + "/loss_meta.txt")

            line = "Runner Episode: {}\tLoss: {:.6f}\tAccuracy: {:.3f} \t Current Training Data Per Class: {}".format(
                eps + 1, loss_meta.item(), acc, current_training_data_per_class
            )
            if self.display:
                print(line)

            if self.save_results:
                self.summary_writer.add_scalar("Loss/meta", loss_meta.item(), eps)
                self.summary_writer.add_scalar("Accuracy/meta", acc, eps)

                with open(self.result_directory + "/params.txt", "a") as f:
                    f.writelines(line + "\n")

        # -- plot
        if self.save_results:
            self.summary_writer.close()
            self.plot()

        print("Runner complete.")


def run(
    seed: int,
    display: bool = True,
    result_subdirectory: str = "testing",
    index: int = 0,
    typeOfFeedback: typeOfFeedbackEnum = typeOfFeedbackEnum.FA,
    modelPath=None,
) -> None:
    """
        Main function for Meta-learning the plasticity rule.

    This function serves as the entry point for meta-learning model training
    and performs the following operations:
    1) Loads and parses command-line arguments,
    2) Loads custom EMNIST dataset using meta-training arguments (K, Q),
    3) Creates tasks for meta-training using `RandomSampler` with specified
        number of classes (M) and episodes,
    4) Initializes and trains a MetaLearner object with the set of tasks.

    :return: None
    """

    # Set the seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # -- load data
    numWorkers = 2
    epochs = 50

    numberOfClasses = None
    # trainingDataPerClass = [90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
    trainingDataPerClass = [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        110,
        120,
        130,
        140,
        150,
        160,
        170,
        180,
        190,
        200,
        225,
        250,
        275,
        300,
        325,
        350,
        375,
    ]
    # trainingDataPerClass = [200, 225, 250, 275, 300, 325, 350, 375]
    # trainingDataPerClass = [200, 250, 300, 350, 375]
    minTrainingDataPerClass = trainingDataPerClass[index]
    maxTrainingDataPerClass = trainingDataPerClass[index]
    queryDataPerClass = 20
    dataset_name = "EMNIST"

    if dataset_name == "EMNIST":
        numberOfClasses = 5
        dataset = EmnistDataset(
            minTrainingDataPerClass=minTrainingDataPerClass,
            maxTrainingDataPerClass=maxTrainingDataPerClass,
            queryDataPerClass=queryDataPerClass,
            dimensionOfImage=28,
        )
        dimOut = 47
    elif dataset_name == "FASHION-MNIST":
        numberOfClasses = 10
        dataset = FashionMnistDataset(
            minTrainingDataPerClass=minTrainingDataPerClass,
            maxTrainingDataPerClass=maxTrainingDataPerClass,
            queryDataPerClass=queryDataPerClass,
            dimensionOfImage=28,
            all_classes=True,
        )
        dimOut = 10

    sampler = RandomSampler(data_source=dataset, replacement=True, num_samples=epochs * numberOfClasses)
    metatrain_dataset = DataLoader(
        dataset=dataset, sampler=sampler, batch_size=numberOfClasses, drop_last=True, num_workers=numWorkers
    )

    # -- options
    model = modelEnum.complex
    modelOptions = None

    if model == modelEnum.complex or model == modelEnum.individual:
        modelOptions = complexOptions(
            nonLinear=nonLinearEnum.tanh,
            update_rules=[0, 1, 2, 3, 4, 5, 8, 9],  # 5
            bias=False,
            pMatrix=pMatrixEnum.first_col,
            kMatrix=kMatrixEnum.zero,
            minTau=2,
            maxTau=1000,
            y_vector=yVectorEnum.none,
            z_vector=zVectorEnum.all_ones,
            operator=operatorEnum.mode_6,
            train_z_vector=False,
            mode=modeEnum.all,
            v_vector=vVectorEnum.default,
            eta=1,
            beta=0,  ## Only for v_vector=random_beta
            kMasking=False,
            individual_different_v_vector=False,  # Individual Model Only
            scheduler_t0=None,  # Only mode_3
        )
    elif model == modelEnum.reservoir:
        modelOptions = reservoirOptions(
            non_linearity=nonLinearEnum.tanh,
            unit_connections=5,
            bias=True,
            spectral_radius=0.9,
            update_rules=[0, 1, 2, 3, 4, 8, 9],
            reservoir_seed=0,
            train_K_matrix=False,
            minTau=1,
            maxTau=50,
            v_vector=vVectorReservoirEnum.default,
            operator=modeReservoirEnum.mode_1,
            y=yReservoirEnum.none,
        )
    elif model == modelEnum.benna:
        modelOptions = bennaOptions(
            non_linearity=nonLinearEnum.tanh,
            bias=False,
            update_rules=[0, 1, 2, 3, 4, 8, 9],
            minTau=1,
            maxTau=50,
        )

    # -- feedback model options
    # feedbackModel = modelEnum.complex
    # feedbackModelOptions = None
    """if feedbackModel == modelEnum.complex or feedbackModel == modelEnum.individual:
        feedbackModelOptions = complexOptions(
            nonLinear=nonLinearEnum.tanh,
            update_rules=[0, 1, 2, 3, 4, 8, 9],
            bias=False,
            pMatrix=pMatrixEnum.zero,
            kMatrix=kMatrixEnum.zero,
            minTau=1 + 1 / 50,  # + 1 / 50,
            maxTau=50,
            y_vector=yVectorEnum.first_one,
            z_vector=zVectorEnum.default,
            operator=operatorEnum.mode_3,
            train_z_vector=False,
            mode=modeEnum.all,
            v_vector=vVectorEnum.default,
            eta=1,
            beta=0,
            kMasking=False,
            individual_different_v_vector=False,  # Individual Model Only
        )
    elif feedbackModel == modelEnum.reservoir:
        feedbackModelOptions = reservoirOptions(
            non_linearity=nonLinearEnum.tanh,
            unit_connections=5,
            bias=True,
            spectral_radius=spectral_radius[index],
            update_rules=[0, 1, 2, 3, 4, 8, 9],
            reservoir_seed=0,
            train_K_matrix=False,
            minTau=1,
            maxTau=50,
            v_vector=vVectorReservoirEnum.default,
            operator=modeReservoirEnum.mode_1,
            y=yReservoirEnum.none,
        )
    elif feedbackModel == modelEnum.benna:
        feedbackModelOptions = bennaOptions(
            non_linearity=nonLinearEnum.tanh,
            bias=False,
            update_rules=[0, 1, 2, 3, 4, 8, 9],
            minTau=1,
            maxTau=50,
        )"""

    feedbackModel = model
    feedbackModelOptions = modelOptions

    # -- path to load model

    # modelPath = os.getcwd() + "/results/DFA_test/1/20250301-165653"
    # modelPath = (
    # r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results\different_y_ind_v_diff_lr\0\0.0009"
    # r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results\Mode_1\baselines\0\3"
    # r"C:\Users\Kyle\Desktop\Computation-in-Biological-Neural-Networks\results\varied_training\1\20250213-134357"
    # r"C:\Users\Kyle\Desktop\Computation-in-Biological-Neural-Networks\results\longer_train_test\1\20250213-154422"
    # r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results\super_varied_longer_train_test\1\20250213-180025"
    # r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results\different_y_0\0\20250203-234503"
    # r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results\individual_no_bias\1\individual_no_bias_recreate\1\20250211-010125"
    # r"C:\Users\Kyle\Desktop\Computation-in-Biological-Neural-Networks\results\rosenbaum_recreate\1\20250215-003840"
    # r"C:\Users\Kyle\Desktop\Computation-in-Biological-Neural-Networks\results\rosenbaum_recreate\1\20250215-010641"
    # r"C:\Users\Kyle\Desktop\Computation-in-Biological-Neural-Networks\results\attention_test\0\20250215-204423"
    # r"C:\Users\Kyle\Desktop\Computation-in-Biological-Neural-Networks\results\y0_extra_long\0\20250216-035231"
    # r"C:\Users\Kyle\Desktop\Computation-in-Biological-Neural-Networks\results\y0_4_extra_long\100_max_tau"
    # r"C:\Users\Kyle\Desktop\Computation-in-Biological-Neural-Networks\results\y0_3_extra_long\1\20250216-185131"
    # r"C:\Users\Kyle\Desktop\Computation-in-Biological-Neural-Networks\results\y0_3_extra_long\1\20250217-005224"
    # )
    # list_of_files = os.listdir(modelPath)
    # modelPath = r"C:\Users\Kyle\Desktop\Results-Computation-In-Biological-NNs\results\different_y_0\100_200_max_tau\y0_5_extra_long_200\500_epochs"

    # -- runner options
    runnerOptions = RunnerOptions(
        model=model,
        modelPath=modelPath,
        results_subdir=result_subdirectory,
        seed=seed,
        small=False,
        save_results=True,
        metatrain_dataset=metatrain_dataset,
        display=display,
        numberOfClasses=numberOfClasses,  # Number of classes in each task (5 for EMNIST, 10 for fashion MNIST)
        dataset_name=dataset_name,
        chemicalInitialization=chemicalEnum.same,
        trainFeedback=False,
        feedbackModel=feedbackModel,
        minTrainingDataPerClass=minTrainingDataPerClass,
        maxTrainingDataPerClass=maxTrainingDataPerClass,
        queryDataPerClass=queryDataPerClass,
        typeOfFeedback=typeOfFeedback,
        dimOut=dimOut,
    )

    #   -- number of chemicals
    numberOfChemicals = 7
    # -- meta-train
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    runner = Runner(
        device=device,
        numberOfChemicals=numberOfChemicals,
        runnerOptions=runnerOptions,
        modelOptions=modelOptions,
        feedbackModelOptions=feedbackModelOptions,
    )

    runner.run()


def runner_main():
    """
        Main function for Meta-learning the plasticity rule.

    This function serves as the entry point for meta-learning model training
    and performs the following operations:
    1) Loads and parses command-line arguments,
    2) Loads custom EMNIST dataset using meta-training arguments (K, Q),
    3) Creates tasks for meta-training using `RandomSampler` with specified
        number of classes (M) and episodes,
    4) Initializes and trains a MetaLearner object with the set of tasks.

    :return: None
    """
    # -- run
    # torch.autograd.set_detect_anomaly(True)
    modelPath_s = [os.getcwd() + "/results/normalise_mode_6_5_chem/0/20250315-233316"]
    for i in range(2):
        for index in range(0, 27):
            run(
                seed=0,
                display=True,
                result_subdirectory=["runner_7_chem_mode_6_800_min_tau_500_2"][i],
                index=index,
                typeOfFeedback=typeOfFeedbackEnum.FA,
                modelPath=modelPath_s[i],
            )
