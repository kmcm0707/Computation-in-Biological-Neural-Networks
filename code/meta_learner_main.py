import argparse
import copy
import datetime
import os
import random
import sys
from multiprocessing import Pool
from typing import Literal

import numpy as np
import torch
from chemical_nn import ChemicalNN
from complex_synapse import ComplexSynapse
from dataset import DataProcess, EmnistDataset
from individual_synapse import IndividualSynapse
from meta_learner_options import (
    MetaLearnerOptions,
    biasLossRegularizationEnum,
    metaLossRegularizationEnum,
    modelEnum,
    optimizerEnum,
    schedulerEnum,
)
from ray import train
from reservoir_synapse import ReservoirSynapse
from torch import nn, optim
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from utils import Plot, log, meta_stats


class MetaLearner:
    """

    Meta-learner class.

    """

    def __init__(
        self,
        device: Literal["cpu", "cuda"] = "cpu",
        model_type: str = "rosenbaum",
        numberOfChemicals: int = 1,
        non_linearity=torch.nn.functional.tanh,
        options={},
        metaLearnerOptions: MetaLearnerOptions = None,
    ):

        # -- processor params
        self.device = torch.device(device)
        self.model_type = model_type
        self.options = options
        self.small = metaLearnerOptions.small  # -- small model

        # -- raytune
        self.raytune = metaLearnerOptions.raytune

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
            self.UnOptimizedModel = self.load_model().to(self.device)
            self.model = torch.compile(self.UnOptimizedModel, mode="reduce-overhead")
        else:
            self.model = self.load_model().to(self.device)

        # -- optimization params
        self.metaLossRegularization = metaLearnerOptions.metaLossRegularization
        self.biasLossRegularization = metaLearnerOptions.biasLossRegularization

        self.loss_func = nn.CrossEntropyLoss()
        if metaLearnerOptions.model == modelEnum.complex:
            self.UpdateWeights = ComplexSynapse(
                device=self.device,
                mode=self.model_type,
                numberOfChemicals=self.numberOfChemicals,
                non_linearity=non_linearity,
                options=self.options,
                params=self.model.named_parameters(),
            )
        elif metaLearnerOptions.model == modelEnum.reservoir:
            self.UpdateWeights = ReservoirSynapse(
                device=self.device,
                mode=self.model_type,
                numberOfChemicals=self.numberOfChemicals,
                non_linearity=non_linearity,
                options=self.options,
                params=self.model.named_parameters(),
                spectral_radius=options["spectral_radius"],
                reservoir_size=options["reservoir_size"],
            )
        elif metaLearnerOptions.model == modelEnum.individual:
            self.UpdateWeights = IndividualSynapse(
                device=self.device,
                mode=self.model_type,
                numberOfChemicals=self.numberOfChemicals,
                non_linearity=non_linearity,
                options=self.options,
                params=self.model.named_parameters(),
            )
        else:
            raise ValueError("Model not recognized.")

        lr = metaLearnerOptions.lr

        if metaLearnerOptions.optimizer == optimizerEnum.sgd:
            self.UpdateMetaParameters = optim.SGD(
                [
                    {
                        "params": self.UpdateWeights.all_meta_parameters.parameters(),
                        "weight_decay": self.metaLossRegularization,
                    },
                    {
                        "params": self.UpdateWeights.all_bias_parameters.parameters(),
                        "weight_decay": self.biasLossRegularization,
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
                        "params": self.UpdateWeights.all_meta_parameters.parameters(),
                        "weight_decay": self.metaLossRegularization,
                    },
                    {
                        "params": self.UpdateWeights.all_bias_parameters.parameters(),
                        "weight_decay": self.biasLossRegularization,
                    },
                ],
                lr=lr,
            )
        elif metaLearnerOptions.optimizer == optimizerEnum.adamW:
            self.UpdateMetaParameters = optim.AdamW(
                [
                    {
                        "params": self.UpdateWeights.all_meta_parameters.parameters(),
                        "weight_decay": self.metaLossRegularization,
                    },
                    {
                        "params": self.UpdateWeights.all_bias_parameters.parameters(),
                        "weight_decay": self.biasLossRegularization,
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
                f.writelines("Non linearity: {}\n".format(non_linearity))
                f.writelines(str(metaLearnerOptions))

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

        model = ChemicalNN(self.device, self.numberOfChemicals, small=self.small)

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
        if self.options["chemicals"] == "zeros":
            for chemical in chemicals:
                nn.init.zeros_(chemical)
        elif self.options["chemicals"] == "same":
            for chemical in chemicals:
                nn.init.xavier_uniform_(chemical[0])
                for idx in range(chemical.shape[0] - 1):
                    chemical[idx + 1] = chemical[0]
        else:
            for chemical in chemicals:
                nn.init.xavier_uniform_(chemical)

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
            if self.options["chemicals"] != "zeros":
                self.UpdateWeights.initial_update(parameters, h_parameters)

            # -- training data
            x_trn, y_trn, x_qry, y_qry = self.data_process(data, 5)

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

            # -- record params
            UpdateWeights_state_dict = copy.deepcopy(self.UpdateWeights.state_dict())

            # -- backprop
            self.UpdateMetaParameters.zero_grad()
            loss_meta.backward()

            # -- gradient clipping
            # torch.nn.utils.clip_grad_norm_(self.UpdateWeights.all_meta_parameters.parameters(), 5000)

            # -- update
            self.UpdateMetaParameters.step()
            if "scheduler" in self.options:
                if self.options["scheduler"] != "none":
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
                for key, val in UpdateWeights_state_dict.items():
                    self.summary_writer.add_tensor(key, val.clone().detach(), eps)

                with open(self.result_directory + "/params.txt", "a") as f:
                    f.writelines(line + "\n")

            # -- raytune
            if self.raytune:
                if torch.isnan(loss_meta):
                    return train.report({"loss": loss_meta.item(), "accuracy": acc})
                train.report({"loss": loss_meta.item(), "accuracy": acc})

        # -- plot
        if self.save_results:
            self.summary_writer.close()
            self.plot()

        # -- save
        if self.save_results:
            torch.save(
                self.UpdateWeights.state_dict(),
                self.result_directory + "/UpdateWeights.pth",
            )
            torch.save(self.model.state_dict(), self.result_directory + "/model.pth")
        print("Meta-training complete.")


def run(seed: int, display: bool = True, result_subdirectory: str = "testing") -> None:
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
    dataset = EmnistDataset(trainingDataPerClass=50, queryDataPerClass=10, dimensionOfImage=28)
    sampler = RandomSampler(data_source=dataset, replacement=True, num_samples=epochs * 5)
    metatrain_dataset = DataLoader(dataset=dataset, sampler=sampler, batch_size=5, drop_last=True)

    # -- options
    options = {}
    options["lr"] = 4e-4
    options["optimizer"] = "adam"
    options["K_Matrix"] = "n"
    options["P_Matrix"] = "n"
    options["metaLossRegularization"] = 0
    options["update_rules"] = [0, 1, 2, 3, 4, 8, 9]
    options["operator"] = "mode_1"
    options["chemicals"] = "n"
    options["bias"] = False
    options["y_vector"] = "last_one"
    options["min_tau"] = 40
    options["z_vector"] = "all_ones"
    options["train_z_vector"] = True
    options["v_vector"] = "none"
    options["biasLossRegularization"] = 0
    options["small"] = False
    options["train_K_matrix"] = True
    # options['oja_minus_parameter'] = True
    # options['scheduler'] = 'exponential'

    # -- non-linearity
    non_linearity = torch.nn.functional.tanh
    # non_linearity = pass_through

    #   -- number of chemicals
    numberOfChemicals = 3
    # -- meta-train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'

    # -- model
    options["model"] = "reservoir"

    metalearning_model = MetaLearner(
        device=device,
        result_subdirectory=result_subdirectory,
        save_results=True,
        model_type="all",
        metatrain_dataset=metatrain_dataset,
        seed=seed,
        display=display,
        numberOfChemicals=numberOfChemicals,
        non_linearity=non_linearity,
        options=options,
        raytune=False,
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

    """non_linearity = [torch.nn.functional.tanh] * Args.Pool
    results_directory = ['full_attempt/1', 'full_attempt/3', 'full_attempt/5'] * Args.Pool
    chemicals = [1,3,5] """

    # -- run
    run(1, True, "testing")
    """if Args.Pool > 1:
        with Pool(Args.Pool) as P:
            P.starmap(run, zip([0] * Args.Pool, [False]*Args.Pool, results_directory, non_linearity, chemicals))
            P.close()
            P.join()
    else:
        run(0)"""


def pass_through(input):
    return input


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
