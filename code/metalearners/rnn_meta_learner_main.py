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
    typeOfFeedbackEnum,
)
from options.reservoir_options import (
    modeReservoirEnum,
    reservoirOptions,
    vVectorReservoirEnum,
    yReservoirEnum,
)
from options.rnn_meta_learner_options import RnnMetaLearnerOptions
from ray import train
from synapses.complex_rnn import ComplexRnn
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter


class RnnMetaLearner:
    """

    Meta-learner class.

    """

    def __init__(
        self,
        device: Literal["cpu", "cuda"] = "cpu",
        numberOfChemicals: int = 1,
        rnnMetaLearnerOptions: RnnMetaLearnerOptions = None,
        modelOptions: Union[complexOptions, reservoirOptions] = None,
    ):

        # -- processor params
        self.str_device = device
        self.device = torch.device(device)
        self.modelOptions = modelOptions
        self.options = rnnMetaLearnerOptions

        # -- data params
        self.queryDataPerClass = self.options.queryDataPerClass
        self.metatrain_dataset = self.options.metatrain_dataset
        self.data_process = DataProcess(
            minTrainingDataPerClass=self.options.minTrainingDataPerClass,
            maxTrainingDataPerClass=self.options.maxTrainingDataPerClass,
            queryDataPerClass=self.queryDataPerClass,
            dimensionOfImage=28,
            device=torch.device(self.options.datasetDevice),
        )

        # -- model params
        self.numberOfChemicals = numberOfChemicals
        if self.device == "cpu":  # Remove if using a newer GPU
            self.model = self.load_model().to(self.device)
        else:
            self.model = self.load_model().to(self.device)

        # -- optimization params
        self.loss_func = nn.CrossEntropyLoss()

        # -- set chemical model
        self.UpdateWeights = self.chemical_model_setter(
            options=self.modelOptions, adaptionPathway="forward", typeOfModel=self.options.model
        )

        lr = self.options.lr

        # -- optimizer
        bias_parameters = list(self.UpdateWeights.all_bias_parameters.parameters())
        meta_parameters = list(self.UpdateWeights.all_meta_parameters.parameters())

        self.UpdateMetaParameters: Union[optim.SGD, optim.Adam, optim.AdamW, optim.NAdam, optim.RAdam, None] = None
        if self.options.optimizer == optimizerEnum.sgd:
            self.UpdateMetaParameters = optim.SGD(
                [
                    {
                        "params": bias_parameters,
                    },
                    {
                        "params": meta_parameters,
                    },
                ],
                lr=lr,
                momentum=0.8,
                nesterov=True,
            )
        elif self.options.optimizer == optimizerEnum.adam:
            self.UpdateMetaParameters = optim.Adam(
                [
                    {
                        "params": bias_parameters,
                    },
                    {
                        "params": meta_parameters,
                    },
                ],
                lr=lr,
            )
        elif self.options.optimizer == optimizerEnum.adamW:
            self.UpdateMetaParameters = optim.AdamW(
                [
                    {
                        "params": bias_parameters,
                    },
                    {
                        "params": meta_parameters,
                    },
                ],
                lr=lr,
            )
        elif self.options.optimizer == optimizerEnum.nadam:
            self.UpdateMetaParameters = optim.NAdam(
                [
                    {
                        "params": bias_parameters,
                    },
                    {
                        "params": meta_parameters,
                    },
                ],
                lr=lr,
            )
        elif self.options.optimizer == optimizerEnum.radam:
            self.UpdateMetaParameters = optim.RAdam(
                [
                    {
                        "params": bias_parameters,
                    },
                    {
                        "params": meta_parameters,
                    },
                ],
                lr=lr,
            )
        else:
            raise ValueError("Optimizer not recognized.")

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
                + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )
            os.makedirs(self.result_directory, exist_ok=False)
            with open(self.result_directory + "/arguments.txt", "w") as f:
                f.writelines("Number of chemicals: {}\n".format(numberOfChemicals))
                f.writelines("Number of query data per class: {}\n".format(self.queryDataPerClass))
                f.writelines(str(self.options))
                f.writelines(str(modelOptions))

            self.average_window = 10
            self.plot = Plot(self.result_directory, self.average_window)
            self.summary_writer = SummaryWriter(log_dir=self.result_directory)

    def chemical_model_setter(
        self, options: Union[complexOptions, reservoirOptions], adaptionPathway="forward", typeOfModel=None
    ):
        model = None
        if typeOfModel == modelEnum.complex:
            model = ComplexRnn(
                device=self.device,
                numberOfChemicals=self.numberOfChemicals,
                complexOptions=options,
                params=self.model.named_parameters(),
                adaptionPathway=adaptionPathway,
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

        # -- continue training
        last_trained_epoch = -1
        if self.options.continueTraining is not None:
            self.UpdateWeights.load_state_dict(
                torch.load(self.options.continueTraining + "/UpdateWeights.pth", weights_only=True)
            )
            self.UpdateMetaParameters.load_state_dict(
                torch.load(self.options.continueTraining + "/UpdateMetaParameters.pth", weights_only=True)
            )
            z = np.loadtxt(self.options.continueTraining + "/acc_meta.txt")
            last_trained_epoch = z.shape[0]

        # -- set model to training mode
        self.model.train()
        self.UpdateWeights.train()
        if self.options.trainFeedback:
            self.UpdateFeedbackWeights.train()

        x_qry = None
        y_qry = None
        current_training_data_per_class = None

        for eps, data in enumerate(self.metatrain_dataset):

            # -- continue training
            if eps < last_trained_epoch:
                print(eps)
                continue

            # -- initialize
            # Using a clone of the model parameters to allow for in-place operations
            # Maintains the computational graph for the model as .detach() is not used
            parameters, h_parameters = self.reinitialize()

            self.UpdateWeights.initial_update(parameters, h_parameters)

            # -- reset time index
            self.UpdateWeights.reset_time_index()

            # -- training data
            x_trn, y_trn, x_qry, y_qry, current_training_data_per_class = self.data_process(
                data, self.options.numberOfClasses
            )

            """ adaptation """
            for itr_adapt, (x, label) in enumerate(zip(x_trn, y_trn)):

                # -- fix device
                if self.str_device != self.options.datasetDevice:
                    x, label = x.to(self.device), label.to(self.device)

                x_reshaped = torch.reshape(x, (784 // self.dimIn, self.dimIn))

                for input in x_reshaped:
                    # -- predict
                    y, logits = torch.func.functional_call(
                        self.model, (parameters, h_parameters), x.unsqueeze(0).unsqueeze(0)
                    )
                    self.model.fast_update()  # TODO

                # -- predict
                y, logits = torch.func.functional_call(
                    self.model, (parameters, h_parameters), x.unsqueeze(0).unsqueeze(0)
                )

                # -- compute slow error
                activations = y
                output = logits
                params = parameters
                feedback = {name: value for name, value in params.items() if "feedback" in name}
                error = [functional.softmax(output, dim=1) - functional.one_hot(label, num_classes=47)]
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
                    DFA_error = [functional.softmax(output, dim=1) - functional.one_hot(label, num_classes=47)]
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

            """ meta update """
            # -- fix device
            if self.device != self.options.datasetDevice:
                x_qry, y_qry = x_qry.to(self.device), y_qry.to(self.device)

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
                self.options.typeOfFeedback,
                self.options.dimOut,
            )

            # -- record params
            UpdateWeights_state_dict = copy.deepcopy(self.UpdateWeights.state_dict())

            # -- backprop
            loss_meta.backward()

            # -- gradient clipping
            # torch.nn.utils.clip_grad_norm_(self.UpdateWeights.all_meta_parameters.parameters(), 5000)

            # -- update
            self.UpdateMetaParameters.step()
            self.UpdateMetaParameters.zero_grad(set_to_none=True)

            # -- log
            if self.save_results:
                log([loss_meta.item()], self.result_directory + "/loss_meta.txt")

            line = "Train Episode: {}\tLoss: {:.6f}\tAccuracy: {:.3f}\tCurrent Training Data Per Class: {}".format(
                eps + 1, loss_meta.item(), acc, current_training_data_per_class
            )
            if self.display:
                print(line)

            if self.save_results:
                self.summary_writer.add_scalar("Loss/meta", loss_meta.item(), eps)
                self.summary_writer.add_scalar("Accuracy/meta", acc, eps)

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
                        or "v_dict" in key
                        or "linear" in key
                    ):
                        with open(self.result_directory + "/{}.txt".format(key), "a") as f:
                            f.writelines("Episode: {}: {} \n".format(eps + 1, val.clone().detach().cpu().numpy()))

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
    numWorkers = 2
    epochs = 500

    dataset_name = "EMNIST"
    minTrainingDataPerClass = 30
    maxTrainingDataPerClass = 110
    queryDataPerClass = 20

    if dataset_name == "EMNIST":
        numberOfClasses = 5
        dataset = EmnistDataset(
            minTrainingDataPerClass=minTrainingDataPerClass,
            maxTrainingDataPerClass=maxTrainingDataPerClass,
            queryDataPerClass=queryDataPerClass,
            dimensionOfImage=28,
        )
    elif dataset_name == "FASHION-MNIST":
        numberOfClasses = 10
        dataset = FashionMnistDataset(
            minTrainingDataPerClass=minTrainingDataPerClass,
            maxTrainingDataPerClass=maxTrainingDataPerClass,
            queryDataPerClass=queryDataPerClass,
            dimensionOfImage=28,
            all_classes=True,
        )

    sampler = RandomSampler(data_source=dataset, replacement=True, num_samples=epochs * numberOfClasses)
    metatrain_dataset = DataLoader(
        dataset=dataset, sampler=sampler, batch_size=numberOfClasses, drop_last=True, num_workers=numWorkers
    )

    # -- options
    model = modelEnum.complex
    modelOptions = None
    spectral_radius = [0.3, 0.5, 0.7, 0.9, 1.1]
    # beta = [1, 0.1, 0.01, 0.001, 0.0001]
    minTau = [10, 20, 30, 40, 50, 60][index]

    if model == modelEnum.complex or model == modelEnum.individual:
        modelOptions = complexOptions(
            nonLinear=nonLinearEnum.tanh,
            update_rules=[0, 1, 2, 3, 4, 5, 8, 9],  # 5,
            bias=False,
            pMatrix=pMatrixEnum.first_col,
            kMatrix=kMatrixEnum.zero,
            minTau=2,  # + 1 / 50,
            maxTau=100,
            y_vector=yVectorEnum.none,
            z_vector=zVectorEnum.all_ones,
            operator=operatorEnum.mode_6,  # mode_5,
            train_z_vector=False,
            mode=modeEnum.all,
            v_vector=vVectorEnum.default,
            eta=1,
            beta=0.01,  ## Only for v_vector=random_beta
            kMasking=False,
            individual_different_v_vector=False,  # Individual Model Only
            scheduler_t0=None,  # Only mode_3
        )

    current_dir = os.getcwd()
    # -- meta-learner options
    metaLearnerOptions = RnnMetaLearnerOptions(
        optimizer=optimizerEnum.adam,
        model=model,
        results_subdir=result_subdirectory,
        seed=seed,
        small=False,
        raytune=False,
        save_results=True,
        metatrain_dataset=metatrain_dataset,
        display=display,
        lr=0.0001,
        numberOfClasses=numberOfClasses,  # Number of classes in each task (5 for EMNIST, 10 for fashion MNIST)
        dataset_name=dataset_name,
        chemicalInitialization=chemicalEnum.same,
        trainFeedback=False,
        feedbackModel=feedbackModel,
        minTrainingDataPerClass=minTrainingDataPerClass,
        maxTrainingDataPerClass=maxTrainingDataPerClass,
        queryDataPerClass=queryDataPerClass,
        datasetDevice="cuda:1",  # if running out of memory, change to "cpu"
        continueTraining=None,
        typeOfFeedback=typeOfFeedbackEnum.FA,
    )

    #   -- number of chemicals
    numberOfChemicals = 5
    # -- meta-train
    # device: Literal["cpu", "cuda"] = "cuda:0" if torch.cuda.is_available() else "cpu"  # cuda:1
    device = "cuda:1"
    metalearning_model = RnnMetaLearner(
        device=device,
        numberOfChemicals=numberOfChemicals,
        metaLearnerOptions=metaLearnerOptions,
        modelOptions=modelOptions,
    )

    metalearning_model.train()
    exit()


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
    for i in range(6):
        run(seed=0, display=True, result_subdirectory="normalise_mode_6_5_chem", index=i)
