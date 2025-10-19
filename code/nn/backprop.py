import datetime
import os
import random
import warnings
from typing import Literal

import numpy as np
import torch
from misc.dataset import DataProcess, EmnistDataset, FashionMnistDataset
from misc.utils import log
from options.meta_learner_options import sizeEnum
from torch import nn, optim
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter


class RosenbaumNN(nn.Module):
    """

    Rosenbaum Neural Network class.

    """

    def __init__(self, device: Literal["cpu", "cuda"] = "cpu", dim_out: int = 47, size: sizeEnum = sizeEnum.normal):

        # Initialize the parent class
        super(RosenbaumNN, self).__init__()

        # Set the device
        self.device = device
        self.size = size
        self.dim_out = dim_out

        # Model
        dim_out = dim_out
        if self.size == sizeEnum.small:
            self.forward1 = nn.Linear(784, 128, bias=False)
            self.forward2 = nn.Linear(128, self.dim_out, bias=False)
        elif self.size == sizeEnum.three_layer_wide:
            self.forward1 = nn.Linear(784, 1500, bias=False)
            self.forward2 = nn.Linear(1500, 1500, bias=False)
            self.forward3 = nn.Linear(1500, self.dim_out, bias=False)
        elif self.size == sizeEnum.three_layer:
            self.forward1 = nn.Linear(784, 512, bias=False)
            self.forward2 = nn.Linear(512, 256, bias=False)
            self.forward3 = nn.Linear(256, self.dim_out, bias=False)
        elif self.size == sizeEnum.nine_layer:
            self.forward1 = nn.Linear(784, 650, bias=False)
            self.forward2 = nn.Linear(650, 512, bias=False)
            self.forward3 = nn.Linear(512, 384, bias=False)
            self.forward4 = nn.Linear(384, 256, bias=False)
            self.forward5 = nn.Linear(256, 170, bias=False)
            self.forward6 = nn.Linear(170, 130, bias=False)
            self.forward7 = nn.Linear(130, 100, bias=False)
            self.forward8 = nn.Linear(100, 70, bias=False)
            self.forward9 = nn.Linear(70, dim_out, bias=False)
        elif self.size == sizeEnum.six_layer:
            self.forward1 = nn.Linear(784, 170, bias=False)
            self.forward2 = nn.Linear(170, 150, bias=False)
            self.forward3 = nn.Linear(150, 130, bias=False)
            self.forward4 = nn.Linear(130, 100, bias=False)
            self.forward5 = nn.Linear(100, 70, bias=False)
            self.forward6 = nn.Linear(70, dim_out, bias=False)
        elif self.size == sizeEnum.ten_layer:
            self.forward1 = nn.Linear(784, 256, bias=False)
            self.forward2 = nn.Linear(256, 200, bias=False)
            self.forward3 = nn.Linear(200, 170, bias=False)
            self.forward4 = nn.Linear(170, 150, bias=False)
            self.forward5 = nn.Linear(150, 130, bias=False)
            self.forward6 = nn.Linear(130, 110, bias=False)
            self.forward7 = nn.Linear(110, 100, bias=False)
            self.forward8 = nn.Linear(100, 90, bias=False)
            self.forward9 = nn.Linear(90, 70, bias=False)
            self.forward10 = nn.Linear(70, dim_out, bias=False)
        else:
            self.forward1 = nn.Linear(784, 170, bias=False)
            self.forward2 = nn.Linear(170, 130, bias=False)
            self.forward3 = nn.Linear(130, 100, bias=False)
            self.forward4 = nn.Linear(100, 70, bias=False)
            self.forward5 = nn.Linear(70, self.dim_out, bias=False)

        # Activation function
        self.beta = 10
        self.activation = nn.Softplus(beta=self.beta)

    # @torch.compile
    def forward(self, x):
        if self.size == sizeEnum.small:
            y0 = x.squeeze(1)
            y1 = self.forward1(y0)
            y1 = self.activation(y1)
            y2 = self.forward2(y1)
            return (y0, y1), y2
        elif self.size == sizeEnum.three_layer_wide or self.size == sizeEnum.three_layer:
            y0 = x.squeeze(1)
            y1 = self.forward1(y0)
            y1 = self.activation(y1)
            y2 = self.forward2(y1)
            y2 = self.activation(y2)
            y3 = self.forward3(y2)
            return (y0, y1, y2), y3
        elif self.size == sizeEnum.nine_layer:
            y0 = x.squeeze(1)
            y1 = self.forward1(y0)
            y1 = self.activation(y1)
            y2 = self.forward2(y1)
            y2 = self.activation(y2)
            y3 = self.forward3(y2)
            y3 = self.activation(y3)
            y4 = self.forward4(y3)
            y4 = self.activation(y4)
            y5 = self.forward5(y4)
            y5 = self.activation(y5)
            y6 = self.forward6(y5)
            y6 = self.activation(y6)
            y7 = self.forward7(y6)
            y7 = self.activation(y7)
            y8 = self.forward8(y7)
            y8 = self.activation(y8)
            y9 = self.forward9(y8)
            return (y0, y1, y2, y3, y4, y5, y6, y7, y8), y9
        elif self.size == sizeEnum.six_layer:
            y0 = x.squeeze(1)
            y1 = self.forward1(y0)
            y1 = self.activation(y1)
            y2 = self.forward2(y1)
            y2 = self.activation(y2)
            y3 = self.forward3(y2)
            y3 = self.activation(y3)
            y4 = self.forward4(y3)
            y4 = self.activation(y4)
            y5 = self.forward5(y4)
            y5 = self.activation(y5)
            y6 = self.forward6(y5)
            return (y0, y1, y2, y3, y4, y5), y6
        elif self.size == sizeEnum.ten_layer:
            y0 = x.squeeze(1)
            y1 = self.forward1(y0)
            y1 = self.activation(y1)
            y2 = self.forward2(y1)
            y2 = self.activation(y2)
            y3 = self.forward3(y2)
            y3 = self.activation(y3)
            y4 = self.forward4(y3)
            y4 = self.activation(y4)
            y5 = self.forward5(y4)
            y5 = self.activation(y5)
            y6 = self.forward6(y5)
            y6 = self.activation(y6)
            y7 = self.forward7(y6)
            y7 = self.activation(y7)
            y8 = self.forward8(y7)
            y8 = self.activation(y8)
            y9 = self.forward9(y8)
            y9 = self.activation(y9)
            y10 = self.forward10(y9)
            return (y0, y1, y2, y3, y4, y5, y6, y7, y8, y9), y10
        else:
            y0 = x.squeeze(1)
            y1 = self.forward1(y0)
            y1 = self.activation(y1)
            y2 = self.forward2(y1)
            y2 = self.activation(y2)
            y3 = self.forward3(y2)
            y3 = self.activation(y3)
            y4 = self.forward4(y3)
            y4 = self.activation(y4)
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
        number_of_classes: int = 5,
        trainingDataPerClass: int = 50,
        dimOut: int = 47,
        numberOfDataRepetitions: int = 1,
        size: sizeEnum = sizeEnum.normal,
    ):

        # -- processor params
        self.device = torch.device(device)
        self.dimOut = dimOut
        self.size = size

        # -- data params
        self.trainingDataPerClass = trainingDataPerClass
        self.queryDataPerClass = 20
        self.metatrain_dataset = metatrain_dataset
        self.data_process = DataProcess(
            minTrainingDataPerClass=self.trainingDataPerClass,
            maxTrainingDataPerClass=self.trainingDataPerClass,
            queryDataPerClass=self.queryDataPerClass,
            dimensionOfImage=28,
            device=self.device,
        )
        self.number_of_classes = number_of_classes
        self.numberOfDataRepetitions = numberOfDataRepetitions

        # -- model params
        if self.device == "cpu":  # Remove if using a newer GPU
            self.UnOptimizedModel = self.load_model().to(self.device)
            self.model = torch.compile(self.UnOptimizedModel, mode="reduce-overhead")
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
            self.result_directory += "/" + result_subdirectory + "/" + str(seed) + "/" + str(self.trainingDataPerClass)
            try:
                os.makedirs(self.result_directory, exist_ok=False)
                with open(self.result_directory + "/arguments.txt", "w") as f:
                    f.writelines("Seed: {}\n".format(seed))
                    f.writelines("Model: {}\n".format("backprop"))
                    f.writelines("Number of training data per class: {}\n".format(self.trainingDataPerClass))
                    f.writelines("Number of query data per class: {}\n".format(self.queryDataPerClass))
            except FileExistsError:
                warnings.warn("The directory already exists. The results will be overwritten.")

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
        model = RosenbaumNN(self.device, self.dimOut, self.size)
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
            Perform Backprop

        :return: None
        """
        self.model.train()

        x_qry = None
        y_qry = None

        for eps, data in enumerate(self.metatrain_dataset):

            # -- re initialize model
            self.model.train()
            self.model.apply(self.weights_init)
            self.UpdateParameters = optim.Adam(self.model.parameters(), lr=1e-3)

            # -- training data
            x_trn, y_trn, x_qry, y_qry, current_training_data = self.data_process(data, self.number_of_classes)
            # x_trn_copy = x_trn.clone()
            # y_trn_copy = y_trn.clone()

            for epoch in range(self.numberOfDataRepetitions):
                """adaptation"""
                for itr_adapt, (x, label) in enumerate(zip(x_trn, y_trn)):

                    # -- predict
                    y, logits = self.model(x.unsqueeze(0).unsqueeze(0))

                    # -- update network params
                    loss_adapt = self.loss_func(logits, label)

                    # -- backprop
                    self.UpdateParameters.zero_grad()
                    loss_adapt.backward()
                    self.UpdateParameters.step()

            # -- predict
            self.model.eval()
            y, logits = self.model(x_qry)

            # -- compute and store stats
            pred = torch.argmax(logits, dim=1)
            acc = torch.eq(pred, y_qry.ravel()).sum().item() / len(y_qry.ravel())
            loss_meta = self.loss_func(logits, y_qry.ravel())

            # -- log
            if self.save_results:
                log([loss_meta.item()], self.result_directory + "/loss_meta.txt")

            line = "Train Episode: {}\tLoss: {:.6f}\tAccuracy: {:.3f}\t Current training data: {}".format(
                eps + 1, loss_meta.item(), acc, current_training_data
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

        # -- save
        """if self.save_results:
            torch.save(
                self.UpdateWeights.state_dict(),
                self.result_directory + "/UpdateWeights.pth",
            )
            torch.save(self.model.state_dict(), self.result_directory + "/model.pth")"""
        print("Meta-training complete.")


def run(
    seed: int, display: bool = True, result_subdirectory: str = "testing", trainingDataPerClass: int = "50"
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
    numWorkers = 3
    epochs = 20
    numberOfClasses = 5
    trainingDataPerClass = trainingDataPerClass
    dimOut = 47
    dataset_name = "FASHION-MNIST"

    if dataset_name == "EMNIST":
        numberOfClasses = 5
        dataset = EmnistDataset(
            minTrainingDataPerClass=trainingDataPerClass,
            maxTrainingDataPerClass=trainingDataPerClass,
            queryDataPerClass=20,
            dimensionOfImage=28,
        )
        dimOut = 47
    elif dataset_name == "FASHION-MNIST":
        numberOfClasses = 10
        dataset = FashionMnistDataset(
            minTrainingDataPerClass=trainingDataPerClass,
            maxTrainingDataPerClass=trainingDataPerClass,
            queryDataPerClass=20,
            dimensionOfImage=28,
            all_classes=True,
        )
        dimOut = 10
    sampler = RandomSampler(data_source=dataset, replacement=True, num_samples=epochs * numberOfClasses)
    metatrain_dataset = DataLoader(
        dataset=dataset, sampler=sampler, batch_size=numberOfClasses, drop_last=True, num_workers=numWorkers
    )

    metalearning_model = MetaLearner(
        device="cuda:0",
        result_subdirectory=result_subdirectory,
        save_results=True,
        metatrain_dataset=metatrain_dataset,
        seed=seed,
        number_of_classes=numberOfClasses,
        trainingDataPerClass=trainingDataPerClass,
        dimOut=dimOut,
        numberOfDataRepetitions=5,
        size=sizeEnum.normal,
    )
    metalearning_model.train()


def backprop_main():
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
    """trainingDataPerClass = [
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
    ]"""
    trainingDataPerClass = [
        10,
        50,
        100,
        150,
        200,
        250,
        300,
        350,
        400,
        450,
        500,
        550,
        600,
        650,
        700,
        750,
        800,
        # 850,
        # 900,
        # 950,
        # 1000,
        # 1050,
        # 1100,
        # 1150,
        # 1200,
        # 1250,
        # 1300,
    ]
    for trainingData in trainingDataPerClass:
        run(
            seed=0,
            display=True,
            result_subdirectory="runner_backprop_5_layer_FASHION-MNIST_5_repetitions",
            trainingDataPerClass=trainingData,
        )
