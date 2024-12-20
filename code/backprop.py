import argparse
import copy
import datetime
import os
import random
import sys
import warnings
from multiprocessing import Pool
from typing import Literal

import numpy as np
import torch
from dataset import DataProcess, EmnistDataset
from ray import train, tune
from torch import nn, optim
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from utils import Plot, log, meta_stats


class RosenbaumNN(nn.Module):
    """

    Rosenbaum Neural Network class.

    """

    def __init__(self, device: Literal["cpu", "cuda"] = "cpu"):

        # Initialize the parent class
        super(RosenbaumNN, self).__init__()

        # Set the device
        self.device = device

        # Model
        dim_out = 47
        self.forward1 = nn.Linear(784, 170, bias=False)
        self.forward2 = nn.Linear(170, 130, bias=False)
        self.forward3 = nn.Linear(130, 100, bias=False)
        self.forward4 = nn.Linear(100, 70, bias=False)
        self.forward5 = nn.Linear(70, dim_out, bias=False)

        # Activation function
        self.beta = 10
        self.activation = nn.Softplus(beta=self.beta)

    # @torch.compile
    def forward(self, x):
        y0 = x.squeeze(1)

        y1 = self.forward1(y0)
        y1 = self.activation(y1)
        # y1 = self.layer_norm1(y1)

        y2 = self.forward2(y1)
        y2 = self.activation(y2)
        # y2 = self.layer_norm2(y2)

        y3 = self.forward3(y2)
        y3 = self.activation(y3)
        # y3 = self.layer_norm3(y3)

        y4 = self.forward4(y3)
        y4 = self.activation(y4)
        # y4 = self.layer_norm4(y4)

        y5 = self.forward5(y4)

        return (y0, y1, y2, y3, y4), y5


class MetaLearner:
    """

    Meta-learner class.

    """

    def __init__(
        self,
        device: Literal["cpu", "cuda"] = "cpu",
        result_subdirectory: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        save_results: bool = True,
        metatrain_dataset=None,
        seed: int = 0,
    ):

        # -- processor params
        self.device = torch.device(device)

        # -- data params
        self.trainingDataPerClass = 50
        self.queryDataPerClass = 10
        self.metatrain_dataset = metatrain_dataset
        self.data_process = DataProcess(
            trainingDataPerClass=self.trainingDataPerClass,
            queryDataPerClass=self.queryDataPerClass,
            dimensionOfImage=28,
            device=self.device,
        )

        # -- model params
        if self.device == "cpu":  # Remove if using a newer GPU
            self.UnOptimizedmodel = self.load_model().to(self.device)
            self.model = torch.compile(self.UnOptimizedmodel, mode="reduce-overhead")
        else:
            self.model = self.load_model().to(self.device)

        self.loss_func = nn.CrossEntropyLoss()

        lr = 1e-3
        self.UpdateParameters = optim.Adam(self.model.parameters(), lr=lr)

        # -- log params
        self.save_results = save_results
        self.display = True
        self.result_directory = os.getcwd() + "/results"
        if self.save_results:
            self.result_directory = os.getcwd() + "/results"
            os.makedirs(self.result_directory, exist_ok=True)
            self.result_directory += (
                "/"
                + result_subdirectory
                + "/"
                + str(seed)
                + "/"
                + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )
            try:
                os.makedirs(self.result_directory, exist_ok=False)
                with open(self.result_directory + "/arguments.txt", "w") as f:
                    f.writelines("Seed: {}\n".format(seed))
                    f.writelines("Model: {}\n".format("backprop"))
                    f.writelines(
                        "Number of training data per class: {}\n".format(
                            self.trainingDataPerClass
                        )
                    )
                    f.writelines(
                        "Number of query data per class: {}\n".format(
                            self.queryDataPerClass
                        )
                    )
            except FileExistsError:
                warnings.warn(
                    "The directory already exists. The results will be overwritten."
                )

                answer = input("Proceed? (y/n): ")
                while answer.lower() not in ["y", "n"]:
                    answer = input("Please enter 'y' or 'n': ")

                if answer.lower() == "n":
                    exit()
                else:
                    os.rmdir(self.result_directory)
                    os.makedirs(self.result_directory, exist_ok=False)
                    with open(self.result_directory + "/arguments.txt", "w") as f:
                        f.writelines("Seed: {}\n".format(seed))

            self.average_window = 10
            self.summary_writer = SummaryWriter(log_dir=self.result_directory)

    def load_model(self):
        """
            Load classifier model

        Loads the classifier network and sets the adaptation, meta-learning,
        and grad computation flags for its variables.

        :param args: (argparse.Namespace) The command-line arguments.
        :return: model with flags , "adapt", set for its parameters
        """
        model = RosenbaumNN(self.device)
        model.apply(self.weights_init)
        self.UnOptimizedmodelStateDict = copy.deepcopy(model.state_dict())

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

    def train(self):
        """
            Perform meta-training.

        This function iterates over episodes to meta-train the model. At each
        episode, it samples a task from the meta-training dataset, initializes
        the model parameters, and clones them. The meta-training data for each
        episode is processed and divided into training and query data. During
        adaptation, the model is updated using `self.OptimAdpt` function, one
        sample at a time, on the training data. In the meta-optimization loop,
        the model is evaluated using the query data, and the plasticity
        meta-parameters are then updated using the `self.OptimMeta` function.
        Accuracy, loss, and other meta statistics are computed and logged.

        :return: None
        """
        self.model.train()

        x_qry = None
        y_qry = None

        for eps, data in enumerate(self.metatrain_dataset):

            # -- re initialize model
            self.model.train()
            self.model.load_state_dict(
                copy.deepcopy(self.UnOptimizedmodelStateDict), strict=True
            )
            self.UpdateParameters = optim.Adam(self.model.parameters(), lr=1e-3)

            # -- training data
            x_trn, y_trn, x_qry, y_qry = self.data_process(data, 5)

            """ adaptation """
            for itr_adapt, (x, label) in enumerate(zip(x_trn, y_trn)):

                # -- predict
                y, logits = self.model(x.unsqueeze(0).unsqueeze(0))

                # -- update network params
                loss_adapt = self.loss_func(logits, label)

                # -- backprop
                self.UpdateParameters.zero_grad()
                loss_meta = loss_adapt
                loss_meta.backward()
                self.UpdateParameters.step()

            """ meta update """
            # -- predict
            self.model.eval()
            print(x_qry.shape)
            y, logits = self.model(x_qry)

            # -- compute and store meta stats
            pred = torch.argmax(logits, dim=1)
            acc = torch.eq(pred, y_qry.ravel()).sum().item() / len(y_qry.ravel())
            print(torch.eq(pred, y_qry).sum().item())
            print(acc)

            # -- gradient clipping
            # torch.nn.utils.clip_grad_norm_(self.UpdateWeights.parameters(), 0.5)

            # -- log
            if self.save_results:
                log([loss_meta.item()], self.result_directory + "/loss_meta.txt")

            line = "Train Episode: {}\tLoss: {:.6f}\tAccuracy: {:.3f}".format(
                eps + 1, loss_meta.item(), acc
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
    dataset = EmnistDataset(
        trainingDataPerClass=50, queryDataPerClass=10, dimensionOfImage=28
    )
    sampler = RandomSampler(
        data_source=dataset, replacement=True, num_samples=epochs * 5
    )
    metatrain_dataset = DataLoader(
        dataset=dataset, sampler=sampler, batch_size=5, drop_last=True
    )

    metalearning_model = MetaLearner(
        device="cpu",
        result_subdirectory=result_subdirectory,
        save_results=True,
        metatrain_dataset=metatrain_dataset,
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
