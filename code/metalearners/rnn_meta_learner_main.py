import copy
import datetime
import os
import random
from typing import Literal, Union

import numpy as np
import torch
from misc.dataset import DataProcess, EmnistDataset, FashionMnistDataset
from misc.utils import Plot, accuracy, log
from nn.chemical_rnn import ChemicalRnn
from options.complex_options import (
    nonLinearEnum,
    operatorEnum,
    yVectorEnum,
    zVectorEnum,
)
from options.fast_rnn_options import fastRnnOptions
from options.kernel_rnn_options import kernelRnnOptions
from options.meta_learner_options import chemicalEnum, optimizerEnum
from options.rnn_meta_learner_options import (
    RnnMetaLearnerOptions,
    errorEnum,
    recurrentInitEnum,
    rnnModelEnum,
)
from synapses.fast_rnn import FastRnn
from synapses.kernel_rnn import KernelRnn
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
        numberOfSlowChemicals: int = 1,
        numberOfFastChemicals: int = 1,
        rnnMetaLearnerOptions: RnnMetaLearnerOptions = None,
        modelOptions: Union[kernelRnnOptions] = None,
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
        self.numberOfSlowChemicals = numberOfSlowChemicals
        self.numberOfFastChemicals = numberOfFastChemicals
        if self.device == "cpu":  # Remove if using a newer GPU
            self.model = self.load_model().to(self.device)
        else:
            self.model = self.load_model().to(self.device)

        # -- optimization params
        self.loss_func = nn.CrossEntropyLoss()

        # -- set chemical model
        self.UpdateWeights = self.chemical_model_setter(options=self.modelOptions, typeOfModel=self.options.model)

        lr = self.options.lr

        # -- optimizer
        meta_parameters = list(self.UpdateWeights.all_meta_parameters.parameters())

        self.UpdateMetaParameters: Union[optim.SGD, optim.Adam, optim.AdamW, optim.NAdam, optim.RAdam, None] = None
        if self.options.optimizer == optimizerEnum.adam:
            self.UpdateMetaParameters = optim.Adam(
                [
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
                f.writelines(str(self.options))
                f.writelines(str(modelOptions))

            self.average_window = 10
            self.plot = Plot(self.result_directory, self.average_window)
            self.summary_writer = SummaryWriter(log_dir=self.result_directory)

    def chemical_model_setter(self, options: Union[kernelRnnOptions], typeOfModel=None):
        model = None
        if typeOfModel == rnnModelEnum.kernel:
            model = KernelRnn(
                device=self.device,
                numberOfSlowChemicals=self.numberOfSlowChemicals,
                kernelRnnOptions=options,
                params=self.model.named_parameters(),
                conversion_matrix=self.model.parameters_to_chemical,
            )
        elif typeOfModel == rnnModelEnum.fast:
            model = FastRnn(
                device=self.device,
                numberOfChemicals=self.numberOfSlowChemicals,
                fastRnnOptions=options,
                params=self.model.named_parameters(),
                conversion_matrix=self.model.parameters_to_chemical,
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

        model = ChemicalRnn(
            self.device,
            numberOfSlowChemicals=self.numberOfSlowChemicals,
            numberOfFastChemicals=self.numberOfFastChemicals,
            requireFastChemical=self.options.requireFastChemical,
            dim_in=self.options.rnn_input_size,
            dim_out=self.options.dimOut,
            biological=self.options.biological,
            biological_min_tau=self.options.biological_min_tau,
            biological_max_tau=self.options.biological_max_tau,
            hidden_size=self.options.hidden_size,
        )

        # -- learning flags
        for key, val in model.named_parameters():
            if "feedback" in key:
                val.requires_grad = False

        return model

    @torch.no_grad()
    def weights_init(self, modules):
        # Not Used
        if isinstance(modules, nn.RNNCell):
            # -- weights_ih
            nn.init.xavier_uniform_(modules.weight_ih)
            # -- weights_hh
            nn.init.eye_(modules.weight_hh)
            # -- bias
            if modules.bias:
                nn.init.xavier_uniform_(modules.bias)
        # Used
        if isinstance(modules, nn.Linear):
            if modules.in_features == modules.out_features:
                if self.options.recurrent_init == recurrentInitEnum.identity:
                    nn.init.eye_(modules.weight)
                elif self.options.recurrent_init == recurrentInitEnum.xavierUniform:
                    nn.init.xavier_uniform_(modules.weight)
            else:
                nn.init.xavier_uniform_(modules.weight)
                if modules.bias is not None:
                    nn.init.xavier_uniform_(modules.bias)

    @torch.no_grad()
    def chemical_init(self, chemicals):
        if self.options.chemicalInitialization == chemicalEnum.same:
            for chemical in chemicals:
                if chemical.shape[1] == chemical.shape[2]:
                    if self.options.recurrent_init == recurrentInitEnum.identity:
                        nn.init.eye_(chemical[0])
                    elif self.options.recurrent_init == recurrentInitEnum.xavierUniform:
                        nn.init.xavier_uniform_(chemical[0])
                else:
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

        # -- module parameters
        # -- parameters are not linked to the model even if .clone() is not used
        params = {
            key: val.clone()
            for key, val in dict(self.model.named_parameters()).items()
            if "." in key and "chemical" not in key
        }
        slow_h_params = {
            key: val.clone()
            for key, val in dict(self.model.named_parameters()).items()
            if "slow" in key and "feedback" not in key
        }
        self.chemical_init([val for key, val in slow_h_params.items()])

        fast_h_params = None
        if self.options.requireFastChemical:
            fast_h_params = {
                key: val.clone()
                for key, val in dict(self.model.named_parameters()).items()
                if "fast" in key and "feedback" not in key
            }
            self.chemical_init([val for key, val in fast_h_params.items()])

        return params, slow_h_params, fast_h_params

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
            parameters, slow_h_parameters, fast_h_parameters = self.reinitialize()

            self.UpdateWeights.initial_update(parameters, slow_h_parameters)

            # -- reset time index
            self.UpdateWeights.reset_time_index()

            # -- training data
            x_trn, y_trn, x_qry, y_qry, current_training_data_per_class = self.data_process(
                data, self.options.numberOfClasses
            )

            feedback = {name: value for name, value in parameters.items() if "feedback" in name}

            # -- leaky error
            current_error_dict = {}
            for name, value in feedback.items():
                current_error_dict[name] = torch.matmul(
                    torch.zeros(size=(1, self.options.dimOut), device=self.device), value
                )

            # -- set hidden state
            self.model.reset_hidden(1)

            """ adaptation """
            for itr_adapt, (x, label) in enumerate(zip(x_trn, y_trn)):

                # -- reset time index
                self.UpdateWeights.reset_time_index()

                # -- reset fast weights
                if self.options.reset_fast_weights:
                    if self.options.requireFastChemical:
                        self.UpdateWeights.reset_fast_chemicals(params=parameters, h_fast_parameters=fast_h_parameters)
                    else:
                        self.UpdateWeights.reset_fast_chemicals(params=parameters)

                # -- reset rnn hidden state
                if self.options.hidden_reset:
                    self.model.reset_hidden(1)

                # -- fix device
                if self.str_device != self.options.datasetDevice:
                    x, label = x.to(self.device), label.to(self.device)

                x_reshaped = torch.reshape(x, (784 // self.options.rnn_input_size, self.options.rnn_input_size))

                for index_rnn, input in enumerate(x_reshaped):
                    # -- predict
                    if self.options.requireFastChemical:
                        y_dict, output = torch.func.functional_call(
                            self.model, (parameters, slow_h_parameters, fast_h_parameters), input.unsqueeze(0)
                        )
                    else:
                        y_dict, output = torch.func.functional_call(
                            self.model, (parameters, slow_h_parameters), input.unsqueeze(0)
                        )

                    # -- compute error
                    error_dict = {}
                    if index_rnn == x_reshaped.shape[0] - 1 or self.options.error == errorEnum.all:
                        error = [
                            functional.softmax(output, dim=1)
                            - functional.one_hot(label, num_classes=self.options.dimOut)
                        ]

                        for name, value in feedback.items():
                            current_error_dict[name] = (
                                torch.matmul(error[0], value) + self.options.leaky_error * current_error_dict[name]
                            )

                        for name, value in current_error_dict.items():
                            parameter_name = self.model.feedback_to_parameters[name]
                            error_below = None
                            if self.model.error_dict[parameter_name] != "last":
                                error_below = current_error_dict[self.model.error_dict[parameter_name]]
                            else:
                                error_below = error[0]
                            error_dict[parameter_name] = (value, error_below)
                    else:
                        error = [torch.zeros_like(functional.softmax(output, dim=1))]
                        error_temp_dict = {}

                        for name, value in feedback.items():
                            error_temp_dict[name] = torch.matmul(error[0], value)

                        for name, value in error_temp_dict.items():
                            parameter_name = self.model.feedback_to_parameters[name]
                            error_below = None
                            if self.model.error_dict[parameter_name] != "last":
                                error_below = error_temp_dict[self.model.error_dict[parameter_name]]
                            else:
                                error_below = error[0]
                            error_dict[parameter_name] = (value, error_below)

                    if self.options.requireFastChemical:
                        # -- update network params
                        self.UpdateWeights.fast_update(
                            params=parameters,
                            h_fast_parameters=fast_h_parameters,
                            error=error_dict,
                            activations_and_output=y_dict,
                        )
                    elif self.options.slowIsFast:
                        self.UpdateWeights.fast_update(
                            params=parameters,
                            error=error_dict,
                            h_parameters=slow_h_parameters,
                            activations_and_output=y_dict,
                        )
                    else:
                        self.UpdateWeights.fast_update(
                            params=parameters,
                            error=error_dict,
                            activations_and_output=y_dict,
                        )

                # -- update network params
                if self.options.requireFastChemical:
                    self.UpdateWeights.slow_update(
                        params=parameters,
                        h_slow_parameters=slow_h_parameters,
                        h_fast_parameters=fast_h_parameters,
                    )
                elif self.options.slowIsFast:
                    pass
                else:
                    self.UpdateWeights.slow_update(params=parameters, h_slow_parameters=slow_h_parameters)

            """ meta update """
            # -- fix device
            if self.device != self.options.datasetDevice:
                x_qry, y_qry = x_qry.to(self.device), y_qry.to(self.device)

            x_qry = torch.reshape(
                x_qry, (x_qry.shape[0], 784 // self.options.rnn_input_size, self.options.rnn_input_size)
            )

            # -- reset rnn hidden state
            if self.options.hidden_reset:
                self.model.reset_hidden(x_qry.shape[0])
            else:
                hx1 = self.model.get_hidden()
                self.model.set_hidden(hx1, batch_size=x_qry.shape[0])

            # -- predict
            all_logits = torch.zeros(x_qry.shape[0], x_qry.shape[1], self.options.dimOut).to(self.device)

            if not self.options.test_time_training:
                for input_index in range(x_qry.shape[1]):
                    x_in = x_qry[:, input_index, :]
                    if self.options.requireFastChemical:
                        y, logits = torch.func.functional_call(
                            self.model, (parameters, slow_h_parameters, fast_h_parameters), x_in
                        )
                    else:
                        y, logits = torch.func.functional_call(self.model, (parameters, slow_h_parameters), x_in)
                    all_logits[:, input_index, :] = logits
            else:
                current_parameters = {key: val.clone() for key, val in parameters.items()}
                current_slow_h_parameters = {key: val.clone() for key, val in slow_h_parameters.items()}
                if self.options.requireFastChemical:
                    current_fast_h_parameters = {key: val.clone() for key, val in fast_h_parameters.items()}
                for image_index in range(x_qry.shape[0]):
                    # -- reset time index
                    self.UpdateWeights.reset_time_index()

                    # -- reset fast weights
                    if self.options.reset_fast_weights:
                        if self.options.requireFastChemical:
                            self.UpdateWeights.reset_fast_chemicals(
                                params=current_parameters, h_fast_parameters=current_fast_h_parameters
                            )
                        else:
                            self.UpdateWeights.reset_fast_chemicals(params=current_parameters)

                    # -- reset rnn hidden state
                    if self.options.hidden_reset:
                        self.model.reset_hidden(1)

                    x_reshaped = torch.reshape(
                        x_qry[image_index, :, :], (784 // self.options.rnn_input_size, self.options.rnn_input_size)
                    )

                    for index_rnn, input in enumerate(x_reshaped):
                        # -- predict
                        if self.options.requireFastChemical:
                            y_dict, output = torch.func.functional_call(
                                self.model,
                                (current_parameters, current_slow_h_parameters, current_fast_h_parameters),
                                input.unsqueeze(0),
                            )
                        else:
                            y_dict, output = torch.func.functional_call(
                                self.model, (current_parameters, current_slow_h_parameters), input.unsqueeze(0)
                            )
                        all_logits[image_index, index_rnn, :] = output

                        # -- false error
                        error_dict = {}
                        error = [torch.zeros_like(functional.softmax(output, dim=1))]
                        error_temp_dict = {}

                        for name, value in feedback.items():
                            error_temp_dict[name] = torch.matmul(error[0], value)

                        for name, value in error_temp_dict.items():
                            parameter_name = self.model.feedback_to_parameters[name]
                            error_below = None
                            if self.model.error_dict[parameter_name] != "last":
                                error_below = error_temp_dict[self.model.error_dict[parameter_name]]
                            else:
                                error_below = error[0]
                            error_dict[parameter_name] = (value, error_below)

                        # -- update network params
                        if self.options.requireFastChemical:
                            self.UpdateWeights.fast_update(
                                params=current_parameters,
                                h_fast_parameters=current_fast_h_parameters,
                                error=error_dict,
                                activations_and_output=y_dict,
                            )
                        elif self.options.slowIsFast:
                            self.UpdateWeights.fast_update(
                                params=current_parameters,
                                error=error_dict,
                                h_parameters=current_slow_h_parameters,
                                activations_and_output=y_dict,
                            )
                        else:
                            self.UpdateWeights.fast_update(
                                params=current_parameters,
                                error=error_dict,
                                activations_and_output=y_dict,
                            )

            logits = all_logits[:, -1, :]

            # -- loss
            loss_meta = 0
            if self.options.loss_meta_logits_all:
                for time_index in range(all_logits.shape[1]):
                    logits = all_logits[:, time_index, :]
                    loss_meta += self.loss_func(logits, y_qry.ravel())
                loss_meta = loss_meta / all_logits.shape[1]
            else:
                loss_meta = self.loss_func(logits, y_qry.ravel())

            if loss_meta > 1e5 or torch.isnan(loss_meta):
                print(y)
                print(logits)

            # -- compute and store meta stats
            acc = accuracy(logits, y_qry.ravel())

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
                log([acc], self.result_directory + "/acc_meta.txt")

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
                        or "Q" in key
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
    numWorkers = 0
    epochs = 1000

    dataset_name = "EMNIST"
    minTrainingDataPerClass = 20
    maxTrainingDataPerClass = 70
    queryDataPerClass = 10

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
    model = rnnModelEnum.fast
    modelOptions = None

    if model == rnnModelEnum.kernel:
        modelOptions = kernelRnnOptions(
            nonLinear=nonLinearEnum.tanh,
            update_rules=[0, 1, 2, 4, 5, 9, 12],
            minSlowTau=2,
            maxSlowTau=100,
            y_vector=yVectorEnum.none,
            z_vector=zVectorEnum.all_ones,
            slow_operator=operatorEnum.mode_6,
            time_lag_covariance=None,  ## None to disable
            full_covariance=False,  # True for full covariance, False for diagonal covariance
        )
    elif model == rnnModelEnum.fast:
        modelOptions = fastRnnOptions(
            nonLinear=nonLinearEnum.tanh,
            update_rules=[0, 1, 2, 4, 9, 12],
            minSlowTau=2,
            maxSlowTau=200,
            y_vector=yVectorEnum.none,
            z_vector=zVectorEnum.all_ones,
            operator=operatorEnum.mode_6,
        )

    device: Literal["cpu", "cuda"] = "cuda:1" if torch.cuda.is_available() else "cpu"  # cuda:1
    # device = "cpu"
    # current_dir = os.getcwd()
    # -- meta-learner options
    metaLearnerOptions = RnnMetaLearnerOptions(
        optimizer=optimizerEnum.adam,
        model=model,
        results_subdir=result_subdirectory,
        seed=seed,
        save_results=True,
        metatrain_dataset=metatrain_dataset,
        display=display,
        lr=0.0001,
        numberOfClasses=numberOfClasses,  # Number of classes in each task (5 for EMNIST, 10 for fashion MNIST)
        dataset_name=dataset_name,
        chemicalInitialization=chemicalEnum.same,
        minTrainingDataPerClass=minTrainingDataPerClass,
        maxTrainingDataPerClass=maxTrainingDataPerClass,
        queryDataPerClass=queryDataPerClass,
        rnn_input_size=112,
        datasetDevice=device,  # cuda:1,  # if running out of memory, change to "cpu"
        continueTraining=None,
        reset_fast_weights=False,  # False for fast RNN, True for kernel RNN
        requireFastChemical=False,
        slowIsFast=True,  # True for fast RNN
        dimOut=dimOut,
        biological=True,
        biological_min_tau=1,
        biological_max_tau=7,
        error=errorEnum.all,
        leaky_error=0.0,  # 0.0 for no leaky error
        hidden_reset=True,  # True to reset hidden state between samples
        loss_meta_logits_all=False,  # True to use all logits for meta loss
        hidden_size=128,
        recurrent_init=recurrentInitEnum.xavierUniform,  # identity or xavierUniform
        test_time_training=True,  # True to use test-time training
    )

    #   -- number of chemicals
    numberOfSlowChemicals = 3  # fast uses this
    numberOfFastChemicals = 3
    # -- meta-train

    # device = "cuda:1"
    metalearning_model = RnnMetaLearner(
        device=device,
        numberOfSlowChemicals=numberOfSlowChemicals,
        numberOfFastChemicals=numberOfFastChemicals,
        rnnMetaLearnerOptions=metaLearnerOptions,
        modelOptions=modelOptions,
    )

    metalearning_model.train()
    exit()


def main_rnn():
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
        run(seed=0, display=True, result_subdirectory="mode_4_test_time_compute_fixed", index=i)
