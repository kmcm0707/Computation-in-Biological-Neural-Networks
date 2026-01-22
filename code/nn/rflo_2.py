import datetime
import os
import random
import warnings
from typing import Literal

import numpy as np
import torch
from misc.dataset import (
    AddBernoulliTaskDataProcess,
    AddBernoulliTaskDataset,
    DataProcess,
    EmnistDataset,
    FashionMnistDataset,
)
from misc.utils import log
from options.complex_options import nonLinearEnum
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter


class RfloRNN(nn.Module):
    """

    Rflo Neural Network class.

    """

    def __init__(
        self,
        device: Literal["cpu", "cuda"] = "cpu",
        dim_out: int = 47,
        dim_in: int = 784,
        biological_min_tau: int = 1,
        biological_max_tau: int = 56,
        biological_nonlinearity: nonLinearEnum = nonLinearEnum.tanh,
        hidden_size: int = 128,
        lr_in: float = 0.001,
        lr_hh: float = 0.001,
        lr_out: float = 0.001,
        reset_modulators: bool = True,
    ):

        # Initialize the parent class
        super(RfloRNN, self).__init__()

        # Set the device
        self.device = device

        # Model
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.biological_min_tau = biological_min_tau
        self.biological_max_tau = biological_max_tau
        self.hidden_size = hidden_size
        self.reset_modulators = reset_modulators
        if biological_nonlinearity == nonLinearEnum.softplus:
            self.beta = 10
            self.biological_nonlinearity = nn.Softplus(beta=self.beta)
        else:
            self.biological_nonlinearity = biological_nonlinearity

        # -- layers
        self.forward1 = nn.Linear(self.dim_in, self.hidden_size, bias=False)
        base = self.biological_max_tau / self.biological_min_tau
        tau_vector = self.biological_min_tau * (base ** torch.linspace(0, 1, self.hidden_size, device=self.device))
        self.z_vector = 1 / tau_vector
        self.y_vector = 1 - self.z_vector
        self.y_vector = self.y_vector.to(self.device)
        # self.y_vector = nn.Parameter(self.y_vector)
        self.z_vector = self.z_vector.to(self.device)
        # self.z_vector = nn.Parameter(self.z_vector)

        self.recurrent1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.forward2 = nn.Linear(self.hidden_size, dim_out, bias=False)

        self.p = torch.zeros(size=(self.hidden_size, self.hidden_size), device=self.device)
        self.q = torch.zeros(size=(self.dim_in, self.hidden_size), device=self.device)

        self.past_x = torch.zeros(size=(1, self.dim_in), device=self.device)

        # self.RNN1_in_feedback = nn.Linear(dim_out, self.dim_in, bias=False)
        self.RNN1_hh_feedback = nn.Linear(dim_out, self.hidden_size, bias=False)

        self.lr_in = lr_in
        self.lr_hh = lr_hh
        self.lr_out = lr_out

    # @torch.compile
    def forward(self, x):
        assert x.shape[1] == self.dim_in, "Input shape is not correct."
        assert x.shape[0] == self.hx1.shape[0], "Batch size is not correct."

        # -- compute hidden states
        self.x = x
        self.out1 = self.forward1(x)

        self.past_hx1 = self.hx1.clone()
        # self.tanh_recurrent = torch.tanh(self.recurrent1(self.hx1))
        self.recurrent1_hx1 = self.recurrent1(self.hx1)
        self.u = self.out1 + self.recurrent1_hx1
        self.hx1 = self.y_vector * self.hx1 + self.z_vector * (
            self.biological_nonlinearity(self.out1 + self.recurrent1_hx1)
        )

        # -- compute output
        output = self.forward2(self.hx1)

        # return (x, self.hx1), output
        return (x, self.hx1), output

    def reset_hidden(self, batch_size):
        # -- hidden states
        self.hx1 = torch.zeros(batch_size, self.hidden_size).to(self.device)
        if self.reset_modulators:
            self.p = torch.zeros(size=(self.hidden_size, self.hidden_size), device=self.device)
            self.q = torch.zeros(size=(self.dim_in, self.hidden_size), device=self.device)
            self.past_x = torch.zeros(size=(batch_size, self.dim_in), device=self.device)
        # self.hx2 = torch.zeros(batch_size, 128).to(self.device)

    def update(self, error):
        # -- update p and q
        self.p = (
            self.y_vector * self.p + self.z_vector * torch.matmul((1 / torch.cosh(self.u) ** 2).T, self.past_hx1).T
        )  # (1 - self.tanh_recurrent ** 2)
        self.q = self.y_vector * self.q + self.z_vector * torch.matmul((1 / torch.cosh(self.u) ** 2).T, self.past_x).T
        self.past_x = self.x.clone()

        # in_error = self.RNN1_in_feedback(error).squeeze(0)
        hh_error = self.RNN1_hh_feedback(error).squeeze(0)

        self.forward1.weight = torch.nn.Parameter(self.forward1.weight - (hh_error * self.q * self.lr_in).T)
        self.recurrent1.weight = torch.nn.Parameter(self.recurrent1.weight - (hh_error * self.p * self.lr_hh).T)
        self.forward2.weight = torch.nn.Parameter(self.forward2.weight - torch.matmul(error.T, self.hx1) * self.lr_out)


class RfloLearner:
    """

    Rflo class.

    """

    def __init__(
        self,
        device: Literal["cpu", "cuda"] = "cpu",
        result_subdirectory: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        save_results: bool = True,
        metatrain_dataset=None,
        dataset_name: str = "EMNIST",
        seed: int = 0,
        number_of_classes: int = 5,
        trainingDataPerClass: int = 50,
        dimOut: int = 47,
        dimIn: int = 28,
        # -- model params
        biological_min_tau: int = 1,
        biological_max_tau: int = 56,
        biological_nonlinearity: nonLinearEnum = nonLinearEnum.tanh,
        hidden_size: int = 128,
        lr_in: float = 0.001,
        lr_hh: float = 0.001,
        lr_out: float = 0.001,
        reset_modulators: bool = True,
    ):

        # -- processor params
        self.device = torch.device(device)

        # -- model params
        self.dimOut = dimOut
        self.dimIn = dimIn
        self.biological_min_tau = biological_min_tau
        self.biological_max_tau = biological_max_tau
        self.biological_nonlinearity = biological_nonlinearity
        self.hidden_size = hidden_size
        self.lr_in = lr_in
        self.lr_hh = lr_hh
        self.lr_out = lr_out
        self.reset_modulators = reset_modulators

        # -- data params
        self.dataset_name = dataset_name
        self.trainingDataPerClass = trainingDataPerClass
        self.queryDataPerClass = 20
        if self.dataset_name == "ADDBERNOULLI":
            self.queryDataPerClass = 100
        self.metatrain_dataset = metatrain_dataset
        if self.dataset_name == "ADDBERNOULLI":
            self.data_process = AddBernoulliTaskDataProcess(
                device=self.device, min_lag_1=5, max_lag_1=5, min_lag_2=8, max_lag_2=8
            )
        else:
            self.data_process = DataProcess(
                minTrainingDataPerClass=self.trainingDataPerClass,
                maxTrainingDataPerClass=self.trainingDataPerClass,
                queryDataPerClass=self.queryDataPerClass,
                dimensionOfImage=28,
                device=self.device,
            )
        self.number_of_classes = number_of_classes

        # -- model params
        if self.device == "cpu":  # Remove if using a newer GPU
            self.UnOptimizedModel = self.load_model().to(self.device)
            self.model = torch.compile(self.UnOptimizedModel, mode="reduce-overhead")
        else:
            self.model = self.load_model().to(self.device)

        self.loss_func = nn.CrossEntropyLoss()

        # lr = 1e-3
        # self.UpdateParameters = optim.Adam(self.model.parameters(), lr=lr)

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
                    for arg in vars(self):
                        if arg not in [
                            "model",
                            "metatrain_dataset",
                            "data_process",
                            "summary_writer",
                            "UpdateParameters",
                            "loss_func",
                        ]:
                            f.writelines("{}: {}\n".format(arg, getattr(self, arg)))
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
        model = RfloRNN(
            self.device,
            dim_out=self.dimOut,
            dim_in=self.dimIn,
            biological_min_tau=self.biological_min_tau,
            biological_max_tau=self.biological_max_tau,
            biological_nonlinearity=self.biological_nonlinearity,
            hidden_size=self.hidden_size,
            lr_in=self.lr_in,
            lr_hh=self.lr_hh,
            lr_out=self.lr_out,
            reset_modulators=self.reset_modulators,
        )
        return model

    @staticmethod
    def weights_init(modules):
        if isinstance(modules, nn.RNNCell):
            # -- weights_ih
            nn.init.xavier_uniform_(modules.weight_ih)
            # -- weights_hh
            nn.init.xavier_uniform_(modules.weight_hh)
            # -- bias
            if modules.bias:
                nn.init.xavier_uniform_(modules.bias)
        if isinstance(modules, nn.Linear):
            nn.init.xavier_uniform_(modules.weight)
            if modules.bias is not None:
                nn.init.xavier_uniform_(modules.bias)

    @torch.no_grad()
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

            # -- reinitialize model
            self.model.train()
            self.model.apply(self.weights_init)

            # -- training data
            if self.dataset_name == "ADDBERNOULLI":
                x_trn, y_trn, x_qry, y_qry, roll_1, roll_2 = self.data_process(
                    data
                )  # current_training_data is current lag
                x_trn = x_trn.unsqueeze(0)
                y_trn = y_trn.unsqueeze(0)
                x_qry = x_qry.unsqueeze(0)
                y_qry = y_qry.unsqueeze(0)

                current_training_data = x_trn.shape[1]
            else:
                x_trn, y_trn, x_qry, y_qry, current_training_data = self.data_process(data, self.number_of_classes)

            """ adaptation """
            for itr_adapt, (x, label) in enumerate(zip(x_trn, y_trn)):

                self.model.reset_hidden(batch_size=1)

                if self.dataset_name != "ADDBERNOULLI":
                    x_reshaped = torch.reshape(x, (784 // self.dimIn, self.dimIn))
                else:
                    x_reshaped = torch.reshape(x, (x.shape[0], self.dimIn))

                for index_inner, input in enumerate(x_reshaped):
                    # -- predict
                    y, logits = self.model(input.unsqueeze(0))

                    # -- update network params
                    if self.dataset_name != "ADDBERNOULLI":
                        error = functional.softmax(logits, dim=1) - functional.one_hot(label, num_classes=self.dimOut)
                    else:
                        error = functional.softmax(logits, dim=1) - label[index_inner, :].unsqueeze(0)

                    # -- rflo update
                    self.model.update(error)

            # -- predict
            self.model.eval()
            if self.dataset_name != "ADDBERNOULLI":
                x_qry = torch.reshape(x_qry, (x_qry.shape[0], 784 // self.dimIn, self.dimIn))
            else:
                x_qry = torch.reshape(x_qry, (x_qry.shape[0], x_qry.shape[1], self.dimIn))

            if self.dataset_name != "ADDBERNOULLI":
                self.model.reset_hidden(batch_size=x_qry.shape[0])

            if self.dataset_name != "ADDBERNOULLI":
                all_logits = torch.zeros(x_qry.shape[0], x_qry.shape[1], self.dimOut).to(self.device)
            else:
                all_logits = torch.zeros(x_qry.shape[0], x_qry.shape[1], self.dimOut).to(self.device)

            for input_index in range(x_qry.shape[1]):
                x = x_qry[:, input_index, :]
                _, logits = self.model(x)
                all_logits[:, input_index, :] = logits

            # -- compute and store stats
            if self.dataset_name == "ADDBERNOULLI":
                all_logits = all_logits.squeeze(-1).squeeze()
                y_qry = y_qry.squeeze()
                print(all_logits.shape, y_qry.shape)
                all_logits_softmax = torch.softmax(all_logits, dim=1)
                print(all_logits_softmax[0:10, :], y_qry[0:10, :])
                loss_meta = self.loss_func(all_logits, y_qry)
                acc = -1  # Accuracy not defined for regression
            else:
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
        print("RFLO complete.")


def run(
    seed: int,
    display: bool = True,
    result_subdirectory: str = "testing",
    trainingDataPerClass: int = "50",
    dimIn: int = 28,
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
    numWorkers = 6
    epochs = 20
    numberOfClasses = 5
    trainingDataPerClass = trainingDataPerClass
    dimOut = 47
    dataset_name = "EMNIST"

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
    elif dataset_name == "ADDBERNOULLI":
        numberOfClasses = 1
        dataset = AddBernoulliTaskDataset(
            minSequenceLength=trainingDataPerClass, maxSequenceLength=trainingDataPerClass, querySequenceLength=100
        )
        dimIn = 2
        dimOut = 2

    sampler = RandomSampler(data_source=dataset, replacement=True, num_samples=epochs * numberOfClasses)
    metatrain_dataset = DataLoader(
        dataset=dataset, sampler=sampler, batch_size=numberOfClasses, drop_last=True, num_workers=numWorkers
    )

    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    metalearning_model = RfloLearner(
        device=device,
        result_subdirectory=result_subdirectory,
        save_results=True,
        metatrain_dataset=metatrain_dataset,
        dataset_name=dataset_name,
        seed=seed,
        number_of_classes=numberOfClasses,
        trainingDataPerClass=trainingDataPerClass,
        dimOut=dimOut,
        dimIn=dimIn,
        # -- model params
        biological_min_tau=1,
        biological_max_tau=28,
        biological_nonlinearity=nonLinearEnum.tanh,
        hidden_size=128,
        lr_in=0.01,
        lr_hh=0.01,
        lr_out=0.01,
        reset_modulators=True,
    )
    metalearning_model.train()


def rflo_main_2():
    """
    Main function for running RFLO experiments.
    """
    # -- run
    dimIn = [28]
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
        #110,
        #120,
        #130,
        #140,
        #150,
        #160,
        #170,
        #180,
        #190,
        #200,
        #225,
        #250,
        #275,
        #300,
        #325,
        #350,
        #375,
    ]
    """trainingDataPerClass = [
        # 10,
        #1,
        #5,
        9,
        10,
        20,
        50,
        75,
        90,
        # 30,
        100,
        150,
        200,
        250,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
        1500,
        2000,
        2500,
        3000,
        4000,
        6000,
        8000,
        # 40,
        # 50,
        # 60,
        # 70,
        # 80,
        # 90,
        10000,
        12000,
        14000,
        15000,
    ]"""
    for dim in dimIn:
        for trainingData in trainingDataPerClass:
            run(
                seed=0,
                display=True,
                result_subdirectory="runner_murray_rflo_28_4/{}".format(dim),
                trainingDataPerClass=trainingData,
                dimIn=dim,
            )
