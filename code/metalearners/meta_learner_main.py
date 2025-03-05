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
from ray import train
from synapses.benna_synapse import BennaSynapse
from synapses.complex_synapse import ComplexSynapse
from synapses.individual_synapse import IndividualSynapse
from synapses.reservoir_synapse import ReservoirSynapse
from torch import nn, optim
from torch.nn import functional
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
        feedbackModelOptions: Union[complexOptions, reservoirOptions] = None,
    ):

        # -- processor params
        self.str_device = device
        self.device = torch.device(device)
        self.modelOptions = modelOptions
        self.options = metaLearnerOptions
        self.feedbackModelOptions = feedbackModelOptions

        # -- data params
        self.queryDataPerClass = metaLearnerOptions.queryDataPerClass
        self.metatrain_dataset = metaLearnerOptions.metatrain_dataset
        self.data_process = DataProcess(
            minTrainingDataPerClass=metaLearnerOptions.minTrainingDataPerClass,
            maxTrainingDataPerClass=metaLearnerOptions.maxTrainingDataPerClass,
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
        self.metaLossRegularization = metaLearnerOptions.metaLossRegularization
        self.biasLossRegularization = metaLearnerOptions.biasLossRegularization

        self.loss_func = nn.CrossEntropyLoss()

        # -- set chemical model
        self.UpdateWeights = self.chemical_model_setter(
            options=self.modelOptions, adaptionPathway="forward", typeOfModel=self.options.model
        )
        if self.options.trainFeedback:
            self.UpdateFeedbackWeights = self.chemical_model_setter(
                options=self.feedbackModelOptions, adaptionPathway="feedback", typeOfModel=self.options.feedbackModel
            )

        lr = metaLearnerOptions.lr

        # -- optimizer
        bias_parameters = list(self.UpdateWeights.all_bias_parameters.parameters())
        meta_parameters = list(self.UpdateWeights.all_meta_parameters.parameters())
        if self.options.trainFeedback:
            bias_parameters = bias_parameters + list(self.UpdateFeedbackWeights.all_bias_parameters.parameters())
            meta_parameters = meta_parameters + list(self.UpdateFeedbackWeights.all_meta_parameters.parameters())

        self.UpdateMetaParameters: Union[optim.SGD, optim.Adam, optim.AdamW, optim.NAdam, optim.RAdam, None] = None
        if metaLearnerOptions.optimizer == optimizerEnum.sgd:
            self.UpdateMetaParameters = optim.SGD(
                [
                    {
                        "params": bias_parameters,
                        "weight_decay": self.biasLossRegularization,
                    },
                    {
                        "params": meta_parameters,
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
                        "params": bias_parameters,
                        "weight_decay": self.biasLossRegularization,
                    },
                    {
                        "params": meta_parameters,
                        # "weight_decay": self.metaLossRegularization,
                    },
                ],
                lr=lr,
            )
        elif metaLearnerOptions.optimizer == optimizerEnum.adamW:
            self.UpdateMetaParameters = optim.AdamW(
                [
                    {
                        "params": bias_parameters,
                        "weight_decay": self.biasLossRegularization,
                    },
                    {
                        "params": meta_parameters,
                        # "weight_decay": self.metaLossRegularization,
                    },
                ],
                lr=lr,
            )
        elif metaLearnerOptions.optimizer == optimizerEnum.nadam:
            self.UpdateMetaParameters = optim.NAdam(
                [
                    {
                        "params": bias_parameters,
                        "weight_decay": self.biasLossRegularization,
                    },
                    {
                        "params": meta_parameters,
                        # "weight_decay": self.metaLossRegularization,
                    },
                ],
                lr=lr,
            )
        elif metaLearnerOptions.optimizer == optimizerEnum.radam:
            self.UpdateMetaParameters = optim.RAdam(
                [
                    {
                        "params": bias_parameters,
                        "weight_decay": self.biasLossRegularization,
                    },
                    {
                        "params": meta_parameters,
                        # "weight_decay": self.metaLossRegularization,
                    },
                ],
                lr=lr,
            )
        else:
            raise ValueError("Optimizer not recognized.")

        # -- scheduler
        self.scheduler: Union[
            optim.lr_scheduler.ExponentialLR, optim.lr_scheduler.StepLR, optim.lr_scheduler.ConstantLR, None
        ] = None
        if metaLearnerOptions.scheduler == schedulerEnum.exponential:
            self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.UpdateMetaParameters, gamma=0.95)
        elif metaLearnerOptions.scheduler == schedulerEnum.linear:
            self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.UpdateMetaParameters, step_size=30, gamma=0.1)
        elif metaLearnerOptions.scheduler == schedulerEnum.constant:
            self.scheduler = optim.lr_scheduler.ConstantLR(
                optimizer=self.UpdateMetaParameters, total_iters=self.options.epochs
            )
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
                f.writelines("Number of query data per class: {}\n".format(self.queryDataPerClass))
                f.writelines(str(metaLearnerOptions))
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
            if self.options.trainFeedback:
                self.UpdateFeedbackWeights.load_state_dict(
                    torch.load(self.options.continueTraining + "/UpdateFeedbackWeights.pth", weights_only=True)
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

        # scaler = torch.amp.GradScaler("cuda", enabled=True)

        """with torch.autocast(
            device_type=self.str_device,
            dtype=torch.float16,
        ):"""
        for eps, data in enumerate(self.metatrain_dataset):

            # -- continue training
            if eps < last_trained_epoch:
                print(eps)
                continue

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

                # -- fix device
                if self.str_device != self.options.datasetDevice:
                    x, label = x.to(self.device), label.to(self.device)

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
                    for y, i in zip(reversed(activations), reversed(list(feedback))):
                        error.insert(0, torch.matmul(error[0], feedback[i]) * (1 - torch.exp(-self.model.beta * y)))
                    for i in range(len(DFA_error)):
                        # error[i] = (error[i] + DFA_error[i]) / 2
                        if i != 0:
                            error[i] = (error[i] + DFA_error[i]) / np.sqrt(2)
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

            """ meta update """
            # -- fix device
            if self.device != self.options.datasetDevice:
                x_qry, y_qry = x_qry.to(self.device), y_qry.to(self.device)

            # -- predict
            y, logits = None, None
            if self.options.trainFeedback:
                y, logits = torch.func.functional_call(self.model, (parameters, h_parameters, feedback_params), x_qry)
            else:
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
            )

            # -- l1 regularization
            if self.metaLossRegularization > 0:
                P_matrix = self.UpdateWeights.P_matrix
                loss_meta += self.metaLossRegularization * torch.norm(P_matrix, p=1)
                if self.options.trainFeedback:
                    P_matrix = self.UpdateFeedbackWeights.P_matrix
                    loss_meta += self.metaLossRegularization * torch.norm(P_matrix, p=1)

            # -- record params
            UpdateWeights_state_dict = copy.deepcopy(self.UpdateWeights.state_dict())
            UpdateFeedbackWeights_state_dict = None
            if self.options.trainFeedback:
                UpdateFeedbackWeights_state_dict = copy.deepcopy(self.UpdateFeedbackWeights.state_dict())

            # -- backprop
            """scaler.scale(loss_meta).backward()
            scaler.step(self.UpdateMetaParameters)
            scaler.update()"""
            loss_meta.backward()

            # -- gradient clipping
            # torch.nn.utils.clip_grad_norm_(self.UpdateWeights.all_meta_parameters.parameters(), 5000)

            # -- update
            self.UpdateMetaParameters.step()
            if self.scheduler is not None:
                self.scheduler.step()
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

                if self.options.trainFeedback:
                    for key, val in UpdateFeedbackWeights_state_dict.items():
                        if (
                            "K" in key
                            or "P" in key
                            or "v_vector" in key
                            or "z_vector" in key
                            or "y_vector" in key
                            or "A" in key
                            or "B" in key
                            or "linear" in key
                        ):
                            with open(self.result_directory + "/Feedback_{}.txt".format(key), "a") as f:
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
            if self.options.trainFeedback:
                torch.save(
                    self.UpdateFeedbackWeights.state_dict(), self.result_directory + "/UpdateFeedbackWeights.pth"
                )
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
    epochs = 1000

    dataset_name = "EMNIST"
    minTrainingDataPerClass = 30
    maxTrainingDataPerClass = 150
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
    # schedulerT0 = [10, 20, 30, 40][index]
    minTau = [10, 20, 30, 40, 50, 60][index]

    if model == modelEnum.complex or model == modelEnum.individual:
        modelOptions = complexOptions(
            nonLinear=nonLinearEnum.tanh,
            update_rules=[0, 1, 2, 3, 4, 5, 8, 9],
            bias=False,
            pMatrix=pMatrixEnum.first_col,
            kMatrix=kMatrixEnum.zero,
            minTau=2,  # + 1 / 50,
            maxTau=200,
            y_vector=yVectorEnum.none,
            z_vector=zVectorEnum.all_ones,
            operator=operatorEnum.mode_4,
            train_z_vector=False,
            mode=modeEnum.all,
            v_vector=vVectorEnum.default,
            eta=1,
            beta=0.01,  ## Only for v_vector=random_beta
            kMasking=False,
            individual_different_v_vector=False,  # Individual Model Only
            scheduler_t0=None,  # Only mode_3
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
    current_dir = os.getcwd()
    continue_training = current_dir + "/results/5_chem_long/1/20250305-012111"
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
        lr=0.0001,
        numberOfClasses=numberOfClasses,  # Number of classes in each task (5 for EMNIST, 10 for fashion MNIST)
        dataset_name=dataset_name,
        chemicalInitialization=chemicalEnum.same,
        trainFeedback=False,
        feedbackModel=feedbackModel,
        minTrainingDataPerClass=minTrainingDataPerClass,
        maxTrainingDataPerClass=maxTrainingDataPerClass,
        queryDataPerClass=queryDataPerClass,
        datasetDevice="cpu",  # if running out of memory, change to "cpu"
        continueTraining=None,
        typeOfFeedback=typeOfFeedbackEnum.FA,
    )

    #   -- number of chemicals
    numberOfChemicals = 5
    # -- meta-train
    # device: Literal["cpu", "cuda"] = "cuda:1" if torch.cuda.is_available() else "cpu"  # cuda:1
    device = "cpu"
    metalearning_model = MetaLearner(
        device=device,
        numberOfChemicals=numberOfChemicals,
        metaLearnerOptions=metaLearnerOptions,
        modelOptions=modelOptions,
        feedbackModelOptions=feedbackModelOptions,
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
        run(seed=1, display=True, result_subdirectory="5_chem_long", index=i)
