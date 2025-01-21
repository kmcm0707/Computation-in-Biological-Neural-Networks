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
    modelEnum,
    optimizerEnum,
    schedulerEnum,
)
from options.reservoir_options import (
    modeReservoirEnum,
    reservoirOptions,
    vVectorReservoirEnum,
    yReservoirEnum,
)
from ray import train
from synapses.benna_synapse import BennaSynapse
from synapses.complex_synapse import ComplexSynapse
from synapses.individual_synapse import IndividualSynapse
from synapses.reservoir_synapse import ReservoirSynapse
from torch import nn, optim
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter


class MetaLearner:
    """

    Meta-learner class.

    """

    def __init__(
        self,
        device: Literal["cpu", "cuda"] = "cpu",
        numberOfChemicals: int = 1,
        metaLearnerOptions: MetaLearnerOptions = None,
        modelOptions: Union[complexOptions, reservoirOptions] = None,
    ):

        # -- processor params
        self.device = torch.device(device)
        self.modelOptions = modelOptions
        self.options = metaLearnerOptions

        # -- data params
        self.trainingDataPerClass = 50
        self.queryDataPerClass = 10
        self.metatrain_dataset = metaLearnerOptions.metatrain_dataset
        self.data_process = DataProcess(
            trainingDataPerClass=self.trainingDataPerClass,
            queryDataPerClass=self.queryDataPerClass,
            dimensionOfImage=28,
            device=self.device,
        )

        # -- model params
        self.numberOfChemicals = numberOfChemicals
        if self.device == "cpu":  # Remove if using a newer GPU
            self.model = self.load_model().to(self.device)
        else:
            self.model = self.load_model().to(self.device)

        # -- optimization params
        self.metaLossRegularization = metaLearnerOptions.metaLossRegularization
        self.biasLossRegularization = metaLearnerOptions.biasLossRegularization

        self.loss_func = nn.CrossEntropyLoss()
        if metaLearnerOptions.model == modelEnum.complex:
            self.UpdateWeights = ComplexSynapse(
                device=self.device,
                numberOfChemicals=self.numberOfChemicals,
                complexOptions=self.modelOptions,
                params=self.model.named_parameters(),
            )
        elif metaLearnerOptions.model == modelEnum.reservoir:
            self.UpdateWeights = ReservoirSynapse(
                device=self.device,
                numberOfChemicals=self.numberOfChemicals,
                options=self.modelOptions,
                params=self.model.named_parameters(),
            )
        elif metaLearnerOptions.model == modelEnum.individual:
            self.UpdateWeights = IndividualSynapse(
                device=self.device,
                numberOfChemicals=self.numberOfChemicals,
                complexOptions=self.modelOptions,
                params=self.model.named_parameters(),
            )
        elif metaLearnerOptions.model == modelEnum.benna:
            self.UpdateWeights = BennaSynapse(
                device=self.device,
                numberOfChemicals=self.numberOfChemicals,
                options=self.modelOptions,
                params=self.model.named_parameters(),
            )
        else:
            raise ValueError("Model not recognized.")

        lr = metaLearnerOptions.lr

        if metaLearnerOptions.optimizer == optimizerEnum.sgd:
            self.UpdateMetaParameters = optim.SGD(
                [
                    {
                        "params": self.UpdateWeights.all_bias_parameters.parameters(),
                        "weight_decay": self.biasLossRegularization,
                    },
                    {
                        "params": self.UpdateWeights.all_meta_parameters.parameters(),
                        # "weight_decay": self.metaLossRegularization,
                    },
                ],
                lr=lr,
                momentum=0.8,
                nesterov=True,
            )
        elif metaLearnerOptions.optimizer == optimizerEnum.adam:
            self.UpdateMetaParameters = optim.Adam(
                [
                    {
                        "params": self.UpdateWeights.all_bias_parameters.parameters(),
                        "weight_decay": self.biasLossRegularization,
                    },
                    {
                        "params": self.UpdateWeights.all_meta_parameters.parameters(),
                        # "weight_decay": self.metaLossRegularization,
                    },
                ],
                lr=lr,
            )
        elif metaLearnerOptions.optimizer == optimizerEnum.adamW:
            self.UpdateMetaParameters = optim.AdamW(
                [
                    {
                        "params": self.UpdateWeights.all_bias_parameters.parameters(),
                        "weight_decay": self.biasLossRegularization,
                    },
                    {
                        "params": self.UpdateWeights.all_meta_parameters.parameters(),
                        # "weight_decay": self.metaLossRegularization,
                    },
                ],
                lr=lr,
            )
        else:
            raise ValueError("Optimizer not recognized.")

        # -- scheduler
        if metaLearnerOptions.scheduler == schedulerEnum.exponential:
            self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.UpdateMetaParameters, gamma=0.95)
        elif metaLearnerOptions.scheduler == schedulerEnum.linear:
            self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.UpdateMetaParameters, step_size=30, gamma=0.1)
        elif metaLearnerOptions.scheduler == schedulerEnum.constant:
            self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.UpdateMetaParameters, step_size=100000, gamma=1)
        elif metaLearnerOptions.scheduler == schedulerEnum.none:
            self.scheduler = None

        # -- log params
        self.save_results = metaLearnerOptions.save_results
        self.display = metaLearnerOptions.display
        self.result_directory = os.getcwd() + "/results"
        if self.save_results:
            self.result_directory = os.getcwd() + "/results"
            os.makedirs(self.result_directory, exist_ok=True)
            self.result_directory += (
                "/"
                + metaLearnerOptions.results_subdir
                + "/"
                + str(metaLearnerOptions.seed)
                + "/"
                + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )
            os.makedirs(self.result_directory, exist_ok=False)
            with open(self.result_directory + "/arguments.txt", "w") as f:
                f.writelines("Number of chemicals: {}\n".format(numberOfChemicals))
                f.writelines("Number of training data per class: {}\n".format(self.trainingDataPerClass))
                f.writelines("Number of query data per class: {}\n".format(self.queryDataPerClass))
                f.writelines(str(metaLearnerOptions))
                f.writelines(str(modelOptions))

            self.average_window = 10
            self.plot = Plot(self.result_directory, self.average_window)
            self.summary_writer = SummaryWriter(log_dir=self.result_directory)

    def load_model(self):
        """
            Load classifier model

        Loads the classifier network and sets the adaptation, meta-learning,
        and grad computation flags for its variables.

        :param args: (argparse.Namespace) The command-line arguments.
        :return: model with flags , "adapt", set for its parameters
        """

        model = ChemicalNN(self.device, self.numberOfChemicals, small=self.options.small)

        # -- learning flags
        for key, val in model.named_parameters():
            if "forward" in key:
                val.adapt = True
            elif "feedback" in key:
                val.adapt, val.requires_grad = False, False

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
        """if self.options["chemicals"] == "zeros":
            for chemical in chemicals:
                nn.init.zeros_(chemical)
        elif self.options["chemicals"] == "same":
            for chemical in chemicals:
                nn.init.xavier_uniform_(chemical[0])
                for idx in range(chemical.shape[0] - 1):
                    chemical[idx + 1] = chemical[0]
        else:"""
        for chemical in chemicals:
            nn.init.xavier_uniform_(chemical[0])
            for idx in range(chemical.shape[0] - 1):
                chemical[idx + 1] = chemical[0]

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

        # -- module parameters
        # -- parameters are not linked to the model even if .clone() is not used
        params = {
            key: val.clone()
            for key, val in dict(self.model.named_parameters()).items()
            if "." in key and "chemical" not in key and "layer_norm" not in key
        }
        h_params = {key: val.clone() for key, val in dict(self.model.named_parameters()).items() if "chemical" in key}
        # -- set adaptation flags for cloned parameters
        for key in params:
            params[key].adapt = dict(self.model.named_parameters())[key].adapt

        return params, h_params

    def train(self):
        """
            Perform meta-training.

        This function iterates over episodes to meta-train the model. At each
        episode, it samples a task from the meta-training dataset, initializes
        the model parameters, and clones them. The meta-training data for each
        episode is processed and divided into training and query data. During
        adaptation, the model is updated using `self.UpdateWeights` function, one
        sample at a time, on the training data. In the meta-optimization loop,
        the model is evaluated using the query data, and the plasticity
        meta-parameters are then updated using the `self.UpdateWeights` function.
        Accuracy, loss, and other meta statistics are computed and logged.

        :return: None
        """
        self.model.train()
        self.UpdateWeights.train()

        x_qry = None
        y_qry = None

        for eps, data in enumerate(self.metatrain_dataset):

            # -- initialize
            # Using a clone of the model parameters to allow for in-place operations
            # Maintains the computational graph for the model as .detach() is not used
            parameters, h_parameters = self.reinitialize()
            # if self.options["chemicals"] != "zeros":
            # self.UpdateWeights.initial_update(parameters, h_parameters)

            # -- training data
            x_trn, y_trn, x_qry, y_qry = self.data_process(data, self.options.numberOfClasses)

            """ adaptation """
            for itr_adapt, (x, label) in enumerate(zip(x_trn, y_trn)):

                # -- predict
                y, logits = torch.func.functional_call(
                    self.model, (parameters, h_parameters), x.unsqueeze(0).unsqueeze(0)
                )

                # -- update network params
                self.UpdateWeights(
                    activations=y,
                    output=logits,
                    label=label,
                    params=parameters,
                    h_parameters=h_parameters,
                    beta=self.model.beta,
                )

            """ meta update """
            # -- predict
            y, logits = torch.func.functional_call(self.model, (parameters, h_parameters), x_qry)
            loss_meta = self.loss_func(logits, y_qry.ravel())

            if loss_meta > 1e5 or torch.isnan(loss_meta):
                print(y)
                print(logits)

            # -- compute and store meta stats
            acc = meta_stats(
                logits,
                parameters,
                y_qry.ravel(),
                y,
                self.model.beta,
                self.result_directory,
                self.save_results,
            )

            # -- l1 regularization
            if self.metaLossRegularization > 0:
                P_matrix = self.UpdateWeights.P_matrix
                loss_meta += self.metaLossRegularization * torch.norm(P_matrix, p=1)

            # -- record params
            UpdateWeights_state_dict = copy.deepcopy(self.UpdateWeights.state_dict())

            # -- backprop
            self.UpdateMetaParameters.zero_grad()
            loss_meta.backward()

            # -- gradient clipping
            # torch.nn.utils.clip_grad_norm_(self.UpdateWeights.all_meta_parameters.parameters(), 5000)

            # -- update
            self.UpdateMetaParameters.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # -- log
            if self.save_results:
                log([loss_meta.item()], self.result_directory + "/loss_meta.txt")

            line = "Train Episode: {}\tLoss: {:.6f}\tAccuracy: {:.3f}".format(eps + 1, loss_meta.item(), acc)
            if self.display:
                print(line)

            if self.save_results:
                self.summary_writer.add_scalar("Loss/meta", loss_meta.item(), eps)
                self.summary_writer.add_scalar("Accuracy/meta", acc, eps)
                """for key, val in UpdateWeights_state_dict.items():
                    self.summary_writer.add_tensor(key, val.clone().detach(), eps)"""

                with open(self.result_directory + "/params.txt", "a") as f:
                    f.writelines(line + "\n")

                for key, val in UpdateWeights_state_dict.items():
                    if (
                        "K" in key
                        or "P" in key
                        or "v_vector" in key
                        or "z_vector" in key
                        or "y_vector" in key
                        or "A" in key
                        or "B" in key
                    ):
                        with open(self.result_directory + "/{}.txt".format(key), "a") as f:
                            f.writelines("Episode: {}: {} \n".format(eps + 1, val.clone().detach().cpu().numpy()))

            # -- raytune
            if self.options.raytune:
                if torch.isnan(loss_meta):
                    return train.report({"loss": loss_meta.item(), "accuracy": acc})
                train.report({"loss": loss_meta.item(), "accuracy": acc})

            # -- check for nan
            if torch.isnan(loss_meta):
                print("Meta loss is NaN.")
                break

        # -- plot
        if self.save_results:
            self.summary_writer.close()
            self.plot()

        # -- save
        if self.save_results:
            torch.save(self.UpdateWeights.state_dict(), self.result_directory + "/UpdateWeights.pth")
            torch.save(self.model.state_dict(), self.result_directory + "/model.pth")
            torch.save(self.UpdateMetaParameters.state_dict(), self.result_directory + "/UpdateMetaParameters.pth")
        print("Meta-training complete.")


def run(seed: int, display: bool = True, result_subdirectory: str = "testing", index: int = 0) -> None:
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
    numWorkers = 6
    epochs = 200

    dataset_name = "EMNIST"
    numberOfClasses = None

    if dataset_name == "EMNIST":
        numberOfClasses = 5
        dataset = EmnistDataset(trainingDataPerClass=50, queryDataPerClass=10, dimensionOfImage=28)
    elif dataset_name == "FASHION-MNIST":
        numberOfClasses = 10
        dataset = FashionMnistDataset(trainingDataPerClass=50, queryDataPerClass=10, dimensionOfImage=28)

    sampler = RandomSampler(data_source=dataset, replacement=True, num_samples=epochs * numberOfClasses)
    metatrain_dataset = DataLoader(dataset=dataset, sampler=sampler, batch_size=5, drop_last=True)

    # -- options
    model = modelEnum.individual
    modelOptions = None
    spectral_radius = [0.3, 0.5, 0.7, 0.9, 1.1]

    if model == modelEnum.complex or model == modelEnum.individual:
        modelOptions = complexOptions(
            nonLinear=nonLinearEnum.tanh,
            bias=False,
            update_rules=[0, 1, 2, 3, 4, 5, 6, 8, 9],
            pMatrix=pMatrixEnum.first_col,
            kMatrix=kMatrixEnum.zero,
            minTau=1,
            maxTau=50,
            y_vector=yVectorEnum.first_one,
            z_vector=zVectorEnum.default,
            operator=operatorEnum.mode_1,
            train_z_vector=False,
            mode=modeEnum.all,
            v_vector=vVectorEnum.default,
            eta=1,
        )
    elif model == modelEnum.reservoir:
        modelOptions = reservoirOptions(
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
    elif model == modelEnum.benna:
        modelOptions = bennaOptions(
            non_linearity=nonLinearEnum.tanh,
            bias=False,
            update_rules=[0, 1, 2, 3, 4, 8, 9],
            minTau=1,
            maxTau=50,
        )

    # -- meta-learner options
    metaLearnerOptions = MetaLearnerOptions(
        scheduler=schedulerEnum.none,
        metaLossRegularization=0,  # L1 regularization on P matrix (check 1.5)
        biasLossRegularization=0,
        optimizer=optimizerEnum.adam,
        model=model,
        results_subdir=result_subdirectory,
        seed=seed,
        small=False,
        raytune=False,
        save_results=True,
        metatrain_dataset=metatrain_dataset,
        display=display,
        lr=0.0005,
        numberOfClasses=numberOfClasses,  # Number of classes in each task (5 for EMNIST, 10 for fashion MNIST)
    )

    #   -- number of chemicals
    numberOfChemicals = [1, 2, 4, 5][index]
    # -- meta-train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    metalearning_model = MetaLearner(
        device=device,
        numberOfChemicals=numberOfChemicals,
        metaLearnerOptions=metaLearnerOptions,
        modelOptions=modelOptions,
    )
    metalearning_model.train()


def main():
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
    for i in range(5):
        run(seed=0, display=True, result_subdirectory="mode_1", index=i)


def pass_through(input):
    return input
