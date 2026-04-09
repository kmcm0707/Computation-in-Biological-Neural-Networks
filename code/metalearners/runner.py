import os
import random
from typing import Literal, Union

import numpy as np
import torch
from misc.dataset import DataProcess, EmnistDataset, FashionMnistDataset
from misc.utils import ChemicalAnalysis, Plot, log, meta_stats
from nn.chemical_nn import ChemicalNN
from options.benna_options import bennaOptions
from options.complex_options import (
    complexOptions,
    gatingEnum,
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
    chemicalEnum,
    modelEnum,
    sizeEnum,
    typeOfFeedbackEnum,
)
from options.reservoir_options import (
    modeReservoirEnum,
    reservoirOptions,
    vVectorReservoirEnum,
    yReservoirEnum,
)
from options.runner_options import RunnerOptions
from synapses.benna_synapse import BennaSynapse
from synapses.complex_synapse import ComplexSynapse
from synapses.individual_synapse import IndividualSynapse
from synapses.reservoir_synapse import ReservoirSynapse
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, RandomSampler


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
        self.metatrain_dataset_1 = self.options.metatrain_dataset_1
        self.metatrain_dataset_2 = self.options.metatrain_dataset_2

        self.data_process_1 = DataProcess(
            minTrainingDataPerClass=self.options.minTrainingDataPerClass_1,
            maxTrainingDataPerClass=self.options.maxTrainingDataPerClass_1,
            queryDataPerClass=self.queryDataPerClass,
            dimensionOfImage=28,
            device=self.device,
            split=self.options.split,
            split_min_number_of_tasks=self.options.split_min_number_of_tasks,
            split_max_number_of_tasks=self.options.split_max_number_of_tasks,
            split_eval=True,
        )
        if self.metatrain_dataset_2 is not None:
            self.data_process_2 = DataProcess(
                minTrainingDataPerClass=self.options.minTrainingDataPerClass_2,
                maxTrainingDataPerClass=self.options.maxTrainingDataPerClass_2,
                queryDataPerClass=self.queryDataPerClass,
                dimensionOfImage=28,
                device=self.device,
                split=self.options.split,
                split_min_number_of_tasks=self.options.split_min_number_of_tasks,
                split_max_number_of_tasks=self.options.split_max_number_of_tasks,
                split_eval=True,
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
        if not self.modelOptions.bias and self.options.model is modelEnum.complex:
            for key in list(state_dict.keys()):
                if "bias" in key:
                    state_dict.pop(key)
        if self.options.model == modelEnum.individual:
            if "v_vector" in state_dict:
                state_dict["v_dictionary.all"] = state_dict["v_vector"].clone()
                state_dict.pop("v_vector")
        self.UpdateWeights.load_state_dict(state_dict)
        if self.options.trainFeedback:
            feedback_state_dict = torch.load(
                self.options.modelPath + "/UpdateFeedbackWeights.pth", weights_only=True, map_location=self.device
            )  # UpdateFeedbackWeights
            for key, val in self.UpdateWeights.named_parameters():
                if self.options.model is modelEnum.complex:
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
        if self.save_results:
            self.result_directory = os.getcwd() + "/results_3"
            os.makedirs(self.result_directory, exist_ok=True)
            self.result_directory += (
                "/"
                + self.options.results_subdir
                + "/"
                + str(self.options.seed)
                + "/"
                + str(
                    self.options.minTrainingDataPerClass_2
                    if self.metatrain_dataset_2 is not None
                    else self.options.minTrainingDataPerClass_1
                )
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
            # self.summary_writer = SummaryWriter(log_dir=self.result_directory)
            self.chemical_analysis = ChemicalAnalysis(self.result_directory)

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
            size=self.options.size,
            train_feedback=self.options.trainFeedback or self.options.trainSameFeedback,
            typeOfFeedback=self.options.typeOfFeedback,
            dim_out=self.options.dimOut,
            wta=self.options.wta,
            low_rank_feedback=self.options.low_Dim_Feedback,
        )

        # -- learning flags
        for key, val in model.named_parameters():
            if "forward" in key:
                val.adapt = "forward"
            elif "feedback" in key:
                val.adapt, val.requires_grad = "feedback", self.options.trainFeedback or self.options.trainSameFeedback

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
        if self.options.trainFeedback or self.options.trainSameFeedback:
            self.chemical_init(self.model.feedback_chemicals)

        # -- module parameters
        # -- parameters are not linked to the model even if .clone() is not used
        params = {
            key: val.clone()
            for key, val in dict(self.model.named_parameters()).items()
            if "." in key and "chemical" not in key and "layer_norm" not in key
        }

        if self.options.typeOfFeedback == typeOfFeedbackEnum.non_linear_DFA:
            feedback_1 = {name: value for name, value in params.items() if "feedback" in name and "_1" in name}
            feedback_2 = {name: value for name, value in params.items() if "feedback" in name and "_2" in name}
            for feedback_1_param, feedback_1_name, feedback_2_param, feedback_2_name in zip(
                feedback_1.values(), feedback_1.keys(), feedback_2.values(), feedback_2.keys()
            ):
                dim_in = feedback_1_param.shape[0]
                dim_out = feedback_2_param.shape[1]
                r = feedback_1_param.shape[1]
                a = (18.0 / (r * (dim_in + dim_out))) ** 0.25
                new_feedback_1_param = torch.empty_like(feedback_1_param).uniform_(-a, a)
                new_feedback_2_param = torch.empty_like(feedback_2_param).uniform_(-a, a)
                params[feedback_1_name] = new_feedback_1_param.clone()
                params[feedback_2_name] = new_feedback_2_param.clone()

        h_params = {
            key: val.clone()
            for key, val in dict(self.model.named_parameters()).items()
            if "chemical" in key and "feedback" not in key
        }
        feedback_h_params = None
        if self.options.trainFeedback or self.options.trainSameFeedback:
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

        """if self.options.symbolic_analysis:
            all_updates = []
            all_activations = []
            all_errors = []"""

        for eps, data in enumerate(
            self.metatrain_dataset_1
            if self.metatrain_dataset_2 is None
            else zip(self.metatrain_dataset_1, self.metatrain_dataset_2)
        ):
            # -- initialize
            # Using a clone of the model parameters to allow for in-place operations
            # Maintains the computational graph for the model as .detach() is not used
            data_1 = data if self.metatrain_dataset_2 is None else data[0]
            data_2 = None if self.metatrain_dataset_2 is None else data[1]

            self.model.train()
            parameters, h_parameters, feedback_params = self.reinitialize()

            if self.options.trajectory_analysis:
                trajectory = []

            """if self.options.symbolic_analysis:
                symbolic_updates = []
                symbolic_activations = []
                symbolic_errors = []
                """

            if (
                self.options.chemicalInitialization != chemicalEnum.zero
                and self.modelOptions.operator != operatorEnum.mode_3
            ):
                self.UpdateWeights.initial_update(parameters, h_parameters)
                if self.options.trainFeedback and self.feedbackModelOptions.operator != operatorEnum.mode_3:
                    self.UpdateFeedbackWeights.initial_update(parameters, feedback_params)
                if self.options.trainSameFeedback and self.modelOptions.operator != operatorEnum.mode_3:
                    self.UpdateWeights.initial_update(parameters, feedback_params, override_adaption_pathway="feedback")

            # -- reset time index
            self.UpdateWeights.reset_time_index()
            if self.options.trainFeedback:
                self.UpdateFeedbackWeights.reset_time_index()

            # -- training data
            (
                x_trn_1,
                y_trn_1,
                x_qry_1,
                y_qry_1,
                current_training_data_per_class_1,
                current_number_of_tasks_1,
                y_qry_task_indices_1,
            ) = self.data_process_1(data_1, self.options.numberOfClasses_1)
            if self.metatrain_dataset_2 is not None:
                (
                    x_trn_2,
                    y_trn_2,
                    x_qry_2,
                    y_qry_2,
                    current_training_data_per_class_2,
                    current_number_of_tasks_2,
                    y_qry_task_indices_2,
                ) = self.data_process_2(data_2, self.options.numberOfClasses_2)
                y_trn_2 = y_trn_2 + self.options.shift_labels_2
                y_qry_2 = y_qry_2 + self.options.shift_labels_2

                x_trn = torch.cat((x_trn_1, x_trn_2), dim=0)
                y_trn = torch.cat((y_trn_1, y_trn_2), dim=0)
                y_trn = y_trn.to(torch.int64)
            else:
                x_trn = x_trn_1
                y_trn = y_trn_1

            # -- reset chemical analysis
            if self.options.chemical_analysis:
                self.chemical_analysis.reset()

            scalar_running_mean = torch.tensor(0.0, device=self.device)
            if self.options.scalar_variance_reduction > 0:
                scalar_running_mean = torch.tensor(0.2, device=self.device)

            """ adaptation """
            for itr_rep in range(self.options.data_repetitions):
                for itr_adapt, (x, label) in enumerate(zip(x_trn, y_trn)):

                    # -- chemical analysis
                    if self.options.chemical_analysis:
                        self.chemical_analysis.chemical_autocorrelation(h_parameters)
                        """self.chemical_analysis.parameter_autocorrelation(parameters)
                        self.chemical_analysis.chemical_parameter_autocorrelation(h_parameters, parameters)
                        self.chemical_analysis.chemical_actual_autocorrelation(
                            h_parameters, lags=[1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50], min_time_step=1
                        )
                        self.chemical_analysis.parameter_actual_autocorrelation(
                            parameters, lags=[1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50], min_time_step=1
                        )
                        self.chemical_analysis.chemical_norms(h_parameters)

                        self.chemical_analysis.chemical_tracking(h_parameters, 40)
                        self.chemical_analysis.parameter_tracking(parameters, 40)"""

                    # -- predict
                    y, logits = None, None
                    if self.options.trainFeedback or self.options.trainSameFeedback:
                        y, logits = torch.func.functional_call(
                            self.model, (parameters, h_parameters, feedback_params), x.unsqueeze(0).unsqueeze(0)
                        )
                    else:
                        y, logits = torch.func.functional_call(
                            self.model, (parameters, h_parameters), x.unsqueeze(0).unsqueeze(0)
                        )

                    # -- compute error
                    activations = y
                    output = functional.softmax(logits, dim=1)
                    params = parameters
                    feedback = {name: value for name, value in params.items() if "feedback" in name}
                    error = [output - functional.one_hot(label, num_classes=self.options.dimOut)]
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
                            error.insert(
                                0, torch.matmul(error[-1], feedback[i]) * (1 - torch.exp(-self.model.beta * y))
                            )
                    elif self.options.typeOfFeedback == typeOfFeedbackEnum.scalar:
                        if output[0][label] > 0.5:
                            error_scalar = torch.tensor(self.options.scalar_min, device=self.device)
                        else:
                            error_scalar = torch.tensor(1.0, device=self.device)
                        if self.options.scalar_variance_reduction > 0:
                            scalar_running_mean = (
                                1 - 1 / self.options.scalar_variance_reduction
                            ) * scalar_running_mean + (1 / self.options.scalar_variance_reduction) * error_scalar
                            error_scalar = error_scalar - scalar_running_mean  # TODO: Check if this works
                        for y, i in zip(reversed(activations), reversed(list(feedback))):
                            error.insert(0, error_scalar * feedback[i] * (1 - torch.exp(-self.model.beta * y)))
                    elif self.options.typeOfFeedback == typeOfFeedbackEnum.scalar_minus_one:
                        if output[0][label] > 0.5:
                            error_scalar = torch.tensor(-1.0, device=self.device)
                        else:
                            error_scalar = torch.tensor(1.0, device=self.device)
                        if self.options.scalar_variance_reduction > 0:
                            scalar_running_mean = (
                                1 - 1 / self.options.scalar_variance_reduction
                            ) * scalar_running_mean + (1 / self.options.scalar_variance_reduction) * error_scalar
                            error_scalar = error_scalar - scalar_running_mean  # TODO: Check if this works
                        for y, i in zip(reversed(activations), reversed(list(feedback))):
                            error.insert(0, error_scalar * feedback[i] * (1 - torch.exp(-self.model.beta * y)))
                    elif self.options.typeOfFeedback == typeOfFeedbackEnum.scalar_rich:
                        error_scalar_val = 1 - output[0][label]
                        error_scalar = torch.tensor(error_scalar_val, device=self.device)
                        for y, i in zip(reversed(activations), reversed(list(feedback))):
                            error.insert(0, error_scalar * feedback[i] * (1 - torch.exp(-self.model.beta * y)))
                    elif self.options.typeOfFeedback == typeOfFeedbackEnum.scalar_rate:
                        if logits[0][label] > 0.5:
                            error_scalar = torch.tensor(0, device=self.device)
                        else:
                            error_scalar = torch.tensor(1.0, device=self.device)
                        for y, i in zip(reversed(activations), reversed(list(feedback))):
                            error.insert(0, error_scalar * feedback[i] * (1 - torch.exp(-self.model.beta * y)))
                    elif self.options.typeOfFeedback == typeOfFeedbackEnum.zero:
                        for y in reversed(activations):
                            error.insert(0, torch.zeros_like(y, device=self.device))
                    elif self.options.typeOfFeedback == typeOfFeedbackEnum.non_linear_DFA:
                        feedback_1 = {
                            name: value for name, value in params.items() if "feedback" in name and "_1" in name
                        }
                        feedback_2 = {
                            name: value for name, value in params.items() if "feedback" in name and "_2" in name
                        }
                        reversed_feedback_2 = list(reversed(list(feedback_2.values())))
                        for index, (y, i) in enumerate(zip(reversed(activations), reversed(list(feedback_1)))):
                            temp_error = torch.matmul(error[-1], feedback_1[i])
                            # temp_error_non_linear = torch.relu(temp_error)  # non-linearity
                            true_error = torch.matmul(temp_error, reversed_feedback_2[index]) * (
                                1 - torch.exp(-self.model.beta * y)
                            )
                            error.insert(0, true_error)
                    elif self.options.typeOfFeedback == typeOfFeedbackEnum.DFA_grad_FA:
                        DFA_feedback = {name: value for name, value in params.items() if "DFA_feedback" in name}
                        feedback = {name: value for name, value in params.items() if "feedback_FA" in name}
                        DFA_error = [
                            functional.softmax(output, dim=1)
                            - functional.one_hot(label, num_classes=self.options.dimOut)
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
                        analysis_mode=self.options.chemical_analysis,
                    )

                    # -- update same model feedback params
                    if self.options.trainSameFeedback:
                        self.UpdateWeights(
                            params=parameters,
                            h_parameters=feedback_params,
                            error=error,
                            activations_and_output=activations_and_output,
                            override_adaption_pathway="feedback",
                            analysis_mode=self.options.chemical_analysis,
                        )

                    if self.options.chemical_analysis:
                        # self.chemical_analysis.Kh_Pf_tracking(self.UpdateWeights.Kh, self.UpdateWeights.Pf, 40)
                        # self.chemical_analysis.Kh_Pf_norm(self.UpdateWeights.Kh, self.UpdateWeights.Pf)
                        self.chemical_analysis.Kh_Pf_second_derivative(self.UpdateWeights.Kh, self.UpdateWeights.Pf, 40)

                    # -- update time index
                    self.UpdateWeights.update_time_index()

                    # -- update feedback params
                    if self.options.trainFeedback:
                        self.UpdateFeedbackWeights(
                            params=parameters,
                            h_parameters=feedback_params,
                            error=error,
                            activations_and_output=activations_and_output,
                            analysis_mode=self.options.chemical_analysis,
                        )

                        # -- update feedback time index
                        self.UpdateFeedbackWeights.update_time_index()

                    if self.options.trajectory_analysis:

                        def flatten_params(params):
                            return torch.cat([p.flatten() for p in params.values()])

                        forward_params = flatten_params({k: v for k, v in parameters.items() if "forward" in k})
                        trajectory.append(forward_params.detach().cpu())

            """ meta test"""
            # -- predict
            self.model.eval()
            y_1, logits_1 = None, None
            if self.options.trainFeedback or self.options.trainSameFeedback:
                y_1, logits_1 = torch.func.functional_call(
                    self.model, (parameters, h_parameters, feedback_params), x_qry_1
                )
            else:
                y_1, logits_1 = torch.func.functional_call(self.model, (parameters, h_parameters), x_qry_1)
            loss_meta_1 = self.loss_func(logits_1, y_qry_1.ravel())

            if self.metatrain_dataset_2 is not None:
                y_2, logits_2 = None, None
                if self.options.trainFeedback or self.options.trainSameFeedback:
                    y_2, logits_2 = torch.func.functional_call(
                        self.model, (parameters, h_parameters, feedback_params), x_qry_2
                    )
                else:
                    y_2, logits_2 = torch.func.functional_call(self.model, (parameters, h_parameters), x_qry_2)
                loss_meta_2 = self.loss_func(logits_2, y_qry_2.ravel())

            # -- compute and store meta stats
            acc_1 = meta_stats(
                logits_1,
                parameters,
                y_qry_1.ravel(),
                y_1,
                self.model.beta,
                self.result_directory,
                self.save_results,
                typeOfFeedback=self.options.typeOfFeedback,
                dimOut=self.options.dimOut,
                calculate_weight_update=True,
                complex_synapse=self.UpdateWeights,
                h_params=h_parameters,
            )
            if self.metatrain_dataset_2 is not None:
                acc_2 = meta_stats(
                    logits_2,
                    parameters,
                    y_qry_2.ravel(),
                    y_2,
                    self.model.beta,
                    self.result_directory,
                    self.save_results,
                    typeOfFeedback=self.options.typeOfFeedback,
                    dimOut=self.options.dimOut,
                    save_index="_2",
                )

            if self.options.split:
                for task in range(current_number_of_tasks_1):
                    task_indices_1 = y_qry_task_indices_1 == task
                    task_indices_1 = task_indices_1[:, 0]
                    meta_stats(
                        logits_1[task_indices_1, :],
                        parameters,
                        y_qry_1.ravel()[task_indices_1],
                        y_1,
                        self.model.beta,
                        self.result_directory,
                        self.save_results,
                        typeOfFeedback=self.options.typeOfFeedback,
                        dimOut=self.options.dimOut,
                        save_index="_task_" + str(task),
                        calculate_only_acc=True,
                    )

            if self.options.chemical_accuracy:
                for chem_index in range(self.numberOfChemicals):
                    forward_params = {k: v.clone() for k, v in parameters.items() if "forward" in k}
                    for k in forward_params:
                        forward_params[k] = h_parameters[k.replace("forward", "chemical").split(".")[0]][
                            chem_index, :, :
                        ].clone()
                        forward_params[k] = torch.nn.functional.normalize(forward_params[k], p=2, dim=0)
                    _, logits = torch.func.functional_call(self.model, (forward_params, h_parameters), x_qry_1)
                    meta_stats(
                        logits,
                        parameters,
                        y_qry_1.ravel(),
                        y_1,
                        self.model.beta,
                        self.result_directory,
                        self.save_results,
                        typeOfFeedback=self.options.typeOfFeedback,
                        dimOut=self.options.dimOut,
                        save_index="_chem_" + str(chem_index),
                        calculate_only_acc=True,
                    )

            # -- log
            if self.save_results:
                log([loss_meta_1.item()], self.result_directory + "/loss_meta.txt")
                if self.metatrain_dataset_2 is not None:
                    log([loss_meta_2.item()], self.result_directory + "/loss_meta_2.txt")

            line = "Runner Episode: {}\tLoss: {:.6f}\tAccuracy: {:.3f} \t Current Training Data Per Class: {}".format(
                eps + 1, loss_meta_1.item(), acc_1, current_training_data_per_class_1
            )
            if self.metatrain_dataset_2 is not None:
                line += "\tLoss_2: {:.6f}\tAccuracy_2: {:.3f}\t Current Training Data Per Class_2: {}".format(
                    loss_meta_2.item(), acc_2, current_training_data_per_class_2
                )
            if self.display:
                print(line)

            if self.save_results:
                # self.summary_writer.add_scalar("Loss/meta", loss_meta.item(), eps)
                # self.summary_writer.add_scalar("Accuracy/meta", acc, eps)

                with open(self.result_directory + "/params.txt", "a") as f:
                    f.writelines(line + "\n")

            if self.options.trajectory_analysis:
                # --- helper: clone params dict ---
                def clone_params(params):
                    return {k: v.clone() for k, v in params.items()}

                # --- filter-wise normalization ---
                def normalize_direction(direction, params, eps=1e-10):
                    new_dir = {}
                    for k in direction:
                        d = direction[k]
                        w = params[k]

                        w_norm = torch.norm(w, dim=1, keepdim=True)
                        d_norm = torch.norm(d, dim=1, keepdim=True)

                        new_dir[k] = d * (w_norm / (d_norm + eps))

                    return new_dir

                def vector_to_params(vec, reference_params):
                    new_params = {}
                    idx = 0
                    for k, v in reference_params.items():
                        numel = v.numel()
                        new_params[k] = vec[idx : idx + numel].view_as(v)
                        idx += numel
                    return new_params

                # --- add directions ---
                def add_direction(params, d1, d2, a, b):
                    return {k: params[k] + a * d1[k] + b * d2[k] for k in params}

                # --- evaluate loss on query set (stable choice) ---
                def eval_loss(params):
                    self.model.eval()

                    with torch.no_grad():
                        if self.options.trainFeedback or self.options.trainSameFeedback:
                            _, logits = torch.func.functional_call(
                                self.model, (params, h_parameters, feedback_params), x_qry_1
                            )
                        else:
                            _, logits = torch.func.functional_call(self.model, (params, h_parameters), x_qry_1)

                        loss = self.loss_func(logits, y_qry_1.ravel())
                        return loss.item()

                # --- base params (FINAL adapted params from last episode) ---
                base_params = clone_params(parameters)

                trajectory = torch.stack(trajectory)  # [T, D]
                # --- center trajectory ---
                final_point = trajectory[-1:].clone()  # shape [1, D]
                trajectory_centered = trajectory - final_point
                # --- PCA via SVD ---
                U, S, Vh = torch.linalg.svd(trajectory_centered, full_matrices=False)
                # top 2 principal directions
                pc1 = Vh[0]
                pc2 = Vh[1]

                base_params_forward = {k: v for k, v in base_params.items() if "forward" in k}
                d1 = vector_to_params(pc1.to(self.device), base_params_forward)
                d2 = vector_to_params(pc2.to(self.device), base_params_forward)

                # --- directions ---
                d1 = normalize_direction(d1, base_params_forward)
                d2 = normalize_direction(d2, base_params_forward)

                # --- grid ---
                def flatten_dict(d):
                    return torch.cat([v.flatten() for v in d.values()])

                d1_vec = flatten_dict(d1).cpu()
                d2_vec = flatten_dict(d2).cpu()

                proj_x = torch.matmul(trajectory_centered.cpu(), d1_vec) / torch.dot(d1_vec, d1_vec)
                proj_y = torch.matmul(trajectory_centered.cpu(), d2_vec) / torch.dot(d2_vec, d2_vec)

                traj_2d = torch.stack([proj_x, proj_y], dim=1).numpy()
                min_traj = traj_2d.min(axis=0)
                max_traj = traj_2d.max(axis=0)
                alphas = np.linspace(-4, 4, 500)
                betas = np.linspace(-4, 4, 500)

                # alpha_0_index = np.argmin(np.abs(alphas.cpu().numpy()))
                # beta_0_index = np.argmin(np.abs(betas.cpu().numpy()))

                loss_grid = np.zeros((len(alphas), len(betas)))

                for i, a in enumerate(alphas):
                    for j, b in enumerate(betas):
                        base_params_forward_temp = {k: v.clone() for k, v in base_params_forward.items()}
                        new_params = add_direction(base_params_forward_temp, d1, d2, a, b)
                        loss_grid[i, j] = eval_loss(new_params)

                with open(self.result_directory + "/loss_landscape.npy", "ab") as f:
                    np.save(f, loss_grid)

                with open(self.result_directory + "/trajectory_pca.npy", "ab") as f:
                    np.save(f, traj_2d)

        # -- plot
        if self.save_results:
            # self.summary_writer.close()
            self.plot()

        print("Runner complete.")


def run(
    seed: int,
    display: bool = True,
    result_subdirectory: str = "testing",
    index: int = 0,
    typeOfFeedback: typeOfFeedbackEnum = typeOfFeedbackEnum.FA,
    modelPath=None,
    numberOfChemicals=1,
    gating=gatingEnum.no_gating,
    operator=operatorEnum.mode_1,
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
    numWorkers = 4
    epochs = 5

    numberOfClasses = None
    # trainingDataPerClass = [90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
    """trainingDataPerClass = [
        0,
        5,
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
        # 275,
        # 300,
        # 325,
        # 350,
        # 375,
    ]"""
    trainingDataPerClass = [30, 50, 80, 100, 250]
    if index >= len(trainingDataPerClass):
        return
    """ trainingDataPerClass = [
        250,
    ]"""
    """trainingDataPerClass = [
        10,       # 30,
        # 40,
        50,
        100,
        150,
        200,
        250,
        300,
        350,
        400,
        500,
        600,
        700,
        800,
        # 900,
        # 950,
        # 1000,
        # 1050,
        # 1100,
        # 1150,
        # 1200,
        # 1250,
        # 1300,
    ]"""
    # trainingDataPerClass = [200, 225, 250, 275, 300, 325, 350, 375]
    # trainingDataPerClass = [200, 250, 300, 350, 375]
    minTrainingDataPerClass = trainingDataPerClass[index]
    maxTrainingDataPerClass = trainingDataPerClass[index]
    queryDataPerClass = 50
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
        epochs = 20
    elif dataset_name == "COMBINED":
        numberOfClasses_1 = 5
        numberOfClasses_2 = 5
        minTrainingDataPerClass_1 = 40
        maxTrainingDataPerClass_1 = 40
        dataset_1 = EmnistDataset(
            minTrainingDataPerClass=minTrainingDataPerClass_1,
            maxTrainingDataPerClass=maxTrainingDataPerClass_1,
            queryDataPerClass=queryDataPerClass,
            dimensionOfImage=28,
        )
        dataset_2 = FashionMnistDataset(
            minTrainingDataPerClass=minTrainingDataPerClass,
            maxTrainingDataPerClass=maxTrainingDataPerClass,
            queryDataPerClass=queryDataPerClass,
            dimensionOfImage=28,
            all_classes=True,
        )
        shift_labels_2 = 47  # EMNIST has 47 classes
        dimOut = 57

    if dataset_name != "COMBINED":
        sampler = RandomSampler(data_source=dataset, replacement=True, num_samples=epochs * numberOfClasses)
        metatrain_dataset = DataLoader(
            dataset=dataset, sampler=sampler, batch_size=numberOfClasses, drop_last=True, num_workers=numWorkers
        )
    else:
        sampler_1 = RandomSampler(data_source=dataset_1, replacement=True, num_samples=epochs * numberOfClasses_1)
        sampler_2 = RandomSampler(data_source=dataset_2, replacement=True, num_samples=epochs * numberOfClasses_2)
        metatrain_dataset_1 = DataLoader(
            dataset=dataset_1,
            sampler=sampler_1,
            batch_size=numberOfClasses_1,
            drop_last=True,
            num_workers=numWorkers,
        )
        metatrain_dataset_2 = DataLoader(
            dataset=dataset_2,
            sampler=sampler_2,
            batch_size=numberOfClasses_2,
            drop_last=True,
            num_workers=numWorkers,
        )

    # -- options
    model = modelEnum.complex
    modelOptions = None

    if model == modelEnum.complex or model == modelEnum.individual:
        modelOptions = complexOptions(
            nonLinear=nonLinearEnum.tanh,
            update_rules=[0, 1, 2, 3, 4, 6, 9],  # 5
            bias=False,
            pMatrix=pMatrixEnum.first_col,
            kMatrix=kMatrixEnum.zero,
            minTau=2,
            maxTau=50,
            y_vector=yVectorEnum.none,
            z_vector=zVectorEnum.default,
            operator=operator,  # _pre_activation,
            train_z_vector=False,
            mode=modeEnum.all,
            v_vector=vVectorEnum.default,
            eta=1,
            beta=0,  ## Only for v_vector=random_beta
            kMasking=False,
            individual_different_v_vector=True,  # Individual Model Only
            scheduler_t0=None,  # Only mode_3
            train_tau=False,
            scale_chemical_weights=False,
            gating=gating,
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

    # -- runner options
    runnerOptions = RunnerOptions(
        model=model,
        modelPath=modelPath,
        results_subdir=result_subdirectory,
        seed=seed,
        size=sizeEnum.normal,
        save_results=True,
        metatrain_dataset_1=metatrain_dataset_1 if dataset_name == "COMBINED" else metatrain_dataset,
        metatrain_dataset_2=metatrain_dataset_2 if dataset_name == "COMBINED" else None,
        shift_labels_2=shift_labels_2 if dataset_name == "COMBINED" else 0,
        display=display,
        numberOfClasses_1=(
            numberOfClasses_1 if dataset_name == "COMBINED" else numberOfClasses
        ),  # Number of classes in each task (5 for EMNIST, 10 for fashion MNIST)
        numberOfClasses_2=numberOfClasses_2 if dataset_name == "COMBINED" else None,
        dataset_name=dataset_name,
        chemicalInitialization=chemicalEnum.same,
        trainFeedback=False,
        trainSameFeedback=False,
        feedbackModel=feedbackModel,
        minTrainingDataPerClass_1=minTrainingDataPerClass_1 if dataset_name == "COMBINED" else minTrainingDataPerClass,
        maxTrainingDataPerClass_1=maxTrainingDataPerClass_1 if dataset_name == "COMBINED" else maxTrainingDataPerClass,
        minTrainingDataPerClass_2=minTrainingDataPerClass if dataset_name == "COMBINED" else None,
        maxTrainingDataPerClass_2=maxTrainingDataPerClass if dataset_name == "COMBINED" else None,
        queryDataPerClass=queryDataPerClass,
        typeOfFeedback=typeOfFeedback,
        dimOut=dimOut,
        data_repetitions=1,
        wta=False,
        chemical_analysis=False,
        scalar_min=0.0,
        scalar_variance_reduction=-1,
        low_Dim_Feedback=-1,
        split=False,
        split_min_number_of_tasks=5,
        split_max_number_of_tasks=5,
        trajectory_analysis=True,
        chemical_accuracy=False,
    )

    #   -- number of chemicals
    numberOfChemicals = numberOfChemicals
    # -- meta-traing
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
    modelPath_s = [
        # os.getcwd() + "/results_3/mode_9_gating_no_W/0/20260324-210116",
        # os.getcwd() + "/results_3/mode_9_gating_lr/1/20260326-004813",
        # os.getcwd() + "/results_3/mode_9_gating_lr_DFA_grad_log/1/20260326-032731",
        # os.getcwd() + "/results_3/mode_9_gating_lr_DFA_grad/1/20260326-032555",
        # os.getcwd() + "/results_3/mode_9_gating_lr_h_DFA_grad/1/20260326-032449",
        # os.getcwd()
        # + "/results_3/mode_9_gating_lr_h_scalar/1/20260326-025622",
        # os.getcwd() + "/results_3/mode_9_rand/0/20251105-152312",
        # os.getcwd() + "/results_3/20251103-214650",
        # os.getcwd()
        # + "/results_3/mode_7_1_chem/1/20260125-202838",
        os.getcwd() + "/results_3/mode_9_scalar_10/1/20251124-005417"
        #os.getcwd() + "/results_3/mode_6_scalar_not_all_ones_same/2/20251123-235027",
        #os.getcwd() + "/results_3/mode_9_scalar_clip/1/20251204-195612",
        #os.getcwd() + "/results_3/error_1_fixed/0/20251009-194350",
    ]
    for i in range(len(modelPath_s)):
        for index_outer in range(0, 25):
            run(
                seed=0,
                display=True,
                result_subdirectory=[
                    "runner_mode_9_trajectory_analysis_true_5_chems_scalar_4",
                    "runner_mode_9_trajectory_analysis_true_3_chems_scalar_4",
                    "runner_mode_9_trajectory_analysis_true_1_chems_scalar_4",
                ][i],
                index=index_outer,
                typeOfFeedback=typeOfFeedbackEnum.scalar,
                modelPath=modelPath_s[i],
                numberOfChemicals=[5, 3, 1][i],
                gating=gatingEnum.no_gating,
                operator=[operatorEnum.mode_9, operatorEnum.mode_9, operatorEnum.mode_9][i],
            )
