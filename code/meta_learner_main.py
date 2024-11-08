import argparse
from multiprocessing import Pool
import os
import random
import sys
from typing import Literal
import numpy as np
import torch
import warnings
import datetime

from torch import nn, optim
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from dataset import EmnistDataset, DataProcess

from complex_synapse import ComplexSynapse

from utils import log, meta_stats, Plot

class RosenbaumChemicalNN(nn.Module):
    """

    Rosenbaum Neural Network class.
    
    """

    def __init__(self, device: Literal['cpu', 'cuda'] = 'cpu', numberOfChemicals: int = 1):

        # Initialize the parent class
        super(RosenbaumChemicalNN, self).__init__()

        # Set the device
        self.device = device

        # Model
        dim_out = 47
        self.forward1 = nn.Linear(784, 170, bias=False)
        self.forward2 = nn.Linear(170, 130, bias=False)
        self.forward3 = nn.Linear(130, 100, bias=False)
        self.forward4 = nn.Linear(100, 70, bias=False)
        self.forward5 = nn.Linear(70, dim_out, bias=False)

        # Feedback pathway (fixed or symmetric) for plasticity
        # Symmetric feedback is used for backpropagation training
        # Fixed feedback is used for feedback alignment meta-learning
        self.feedback1 = nn.Linear(784, 170, bias=False)
        self.feedback2 = nn.Linear(170, 130, bias=False)
        self.feedback3 = nn.Linear(130, 100, bias=False)
        self.feedback4 = nn.Linear(100, 70, bias=False)
        self.feedback5 = nn.Linear(70, dim_out, bias=False)

        # h(s) - LxW
        self.numberOfChemicals = numberOfChemicals
        self.chemical1 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 170, 784), device=self.device))
        self.chemical2 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 130, 170), device=self.device))
        self.chemical3 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 100, 130), device=self.device))
        self.chemical4 = nn.Parameter(torch.zeros(size=(numberOfChemicals, 70, 100), device=self.device))
        self.chemical5 = nn.Parameter(torch.zeros(size=(numberOfChemicals, dim_out, 70), device=self.device))
        self.chemicals = nn.ParameterList([self.chemical1, self.chemical2, self.chemical3, self.chemical4, self.chemical5])

        # Activation function
        self.beta = 10
        self.activation = nn.Softplus(beta=self.beta)

    #@torch.compile
    def forward(self, x):
        y0 = x.squeeze(1)

        y1 = self.activation(self.forward1(y0))
        y2 = self.activation(self.forward2(y1))
        y3 = self.activation(self.forward3(y2))
        y4 = self.activation(self.forward4(y3))
        y5 = self.forward5(y4)

        return (y0, y1, y2, y3, y4), y5


class MetaLearner:
    """

    Meta-learner class.

    """

    def __init__(self, device: Literal['cpu', 'cuda'] = 'cpu', result_subdirectory: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), save_results: bool = True, model_type: str = "rosenbaum", metatrain_dataset = None, seed: int = 0, display: bool = True, numberOfChemicals: int = 1, non_linearity = torch.nn.functional.tanh):

        # -- processor params
        self.device = torch.device(device)
        self.model_type = model_type
        
        # -- data params
        self.trainingDataPerClass = 50
        self.queryDataPerClass = 10
        self.metatrain_dataset = metatrain_dataset
        self.data_process = DataProcess(trainingDataPerClass=self.trainingDataPerClass, queryDataPerClass=self.queryDataPerClass, dimensionOfImage=28, device=self.device)

        # -- model params
        self.numberOfChemicals = numberOfChemicals
        if self.device == 'cpu': # Remove if using a newer GPU
            self.UnOptimizedmodel = self.load_model().to(self.device)
            self.model = torch.compile(self.UnOptimizedmodel, mode='reduce-overhead')
        else:
            self.model = self.load_model().to(self.device)

        # -- optimization params
        self.metaLossRegularization = 0
        self.loss_func = nn.CrossEntropyLoss()
        if False: #Not working atm?
            self.UnoptimizedUpdateWeights = ComplexSynapse(device=self.device, mode=self.model_type, numberOfChemicals=self.numberOfChemicals, non_linearity=non_linearity).to(self.device)
            self.UpdateWeights = torch.compile(self.UnoptimizedUpdateWeights, mode='reduce-overhead')
        else:
            self.UpdateWeights = ComplexSynapse(device=self.device, mode=self.model_type, numberOfChemicals=self.numberOfChemicals, non_linearity=non_linearity).to(self.device)
        self.UpdateMetaParameters = optim.Adam(params=self.UpdateWeights.all_meta_parameters.parameters(), lr=1e-3)
        for param in self.UpdateWeights.all_meta_parameters.parameters():
            print(param)

        # -- log params
        self.save_results = save_results
        self.display = display
        if self.save_results:
            self.result_directory = os.getcwd() + "/results"
            os.makedirs(self.result_directory, exist_ok=True)
            self.result_directory += "/" + result_subdirectory + "/" + str(seed) + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            try:
                os.makedirs(self.result_directory, exist_ok=False)
                with open(self.result_directory + '/arguments.txt', 'w') as f:
                    f.writelines("Seed: {}\n".format(seed))
                    f.writelines("Model: {}\n".format(model_type))
                    f.writelines("Number of chemicals: {}\n".format(numberOfChemicals))
                    f.writelines("Meta loss regularization: {}\n".format(self.metaLossRegularization))
                    f.writelines("Number of training data per class: {}\n".format(self.trainingDataPerClass))
                    f.writelines("Number of query data per class: {}\n".format(self.queryDataPerClass))
                    f.writelines("Non linearity: {}\n".format(non_linearity))
            except FileExistsError:
                warnings.warn("The directory already exists. The results will be overwritten.")

                answer = input("Proceed? (y/n): ")
                while answer.lower() not in ['y', 'n']:
                    answer = input("Please enter 'y' or 'n': ")

                if answer.lower() == 'n':
                    exit()
                else:
                    os.rmdir(self.result_directory)
                    os.makedirs(self.result_directory, exist_ok=False)
                    with open(self.result_directory + '/arguments.txt', 'w') as f:
                        f.writelines("Seed: {}\n".format(seed))
                        f.writelines("Model: {}\n".format(model_type))
                        f.writelines("Number of chemicals: {}\n".format(numberOfChemicals))
                        f.writelines("Meta loss regularization: {}\n".format(self.metaLossRegularization))
                        f.writelines("Number of training data per class: {}\n".format(self.trainingDataPerClass))
                        f.writelines("Number of query data per class: {}\n".format(self.queryDataPerClass))
                        f.writelines("Non linearity: {}\n".format(non_linearity))

            self.average_window = 10
            self.plot = Plot(self.result_directory, self.UpdateWeights.P_matrix.shape[1], self.average_window)
            self.summary_writer = SummaryWriter(log_dir=self.result_directory)

    def load_model(self):
        """
            Load classifier model

        Loads the classifier network and sets the adaptation, meta-learning,
        and grad computation flags for its variables.

        :param args: (argparse.Namespace) The command-line arguments.
        :return: model with flags , "adapt", set for its parameters
        """
        #if self.model_type == "rosenbaum":
        model = RosenbaumChemicalNN(self.device, self.numberOfChemicals)

        # -- learning flags
        for key, val in model.named_parameters():
            if 'forward' in key:
                val.adapt = True
            elif 'feedback' in key:
                val.adapt, val.requires_grad = False, False

        return model

    @staticmethod
    def weights_init(modules):
        classname = modules.__class__.__name__
        if classname.find('Linear') != -1:

            # -- weights
            nn.init.xavier_uniform_(modules.weight.data)

            # -- bias
            if modules.bias is not None:
                nn.init.xavier_uniform_(modules.bias.data)
    
    @staticmethod
    @torch.no_grad()
    def chemical_init(chemicals):
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

        #-- module parameters
        #-- parameters are not linked to the model even if .clone() is not used
        params = {key: val.clone() for key, val in dict(self.model.named_parameters()).items() if '.' in key and 'chemical' not in key}
        h_params = {key: val.clone() for key, val in dict(self.model.named_parameters()).items() if 'chemical' in key}
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
        adaptation, the model is updated using `self.OptimAdpt` function, one
        sample at a time, on the training data. In the meta-optimization loop,
        the model is evaluated using the query data, and the plasticity
        meta-parameters are then updated using the `self.OptimMeta` function.
        Accuracy, loss, and other meta statistics are computed and logged.

        :return: None
        """
        self.model.train()
        self.UpdateWeights.train()
        for eps, data in enumerate(self.metatrain_dataset):

            # -- initialize
            # Using a clone of the model parameters to allow for in-place operations
            # Maintains the computational graph for the model as .detach() is not used
            parameters, h_parameters = self.reinitialize()
            self.UpdateWeights.inital_update(parameters, h_parameters)

            # -- training data
            x_trn, y_trn, x_qry, y_qry = self.data_process(data, 5)

            """ adaptation """
            for itr_adapt, (x, label) in enumerate(zip(x_trn, y_trn)):

                # -- predict
                y, logits = torch.func.functional_call(self.model, (parameters, h_parameters), x.unsqueeze(0).unsqueeze(0))

                # -- update network params
                self.UpdateWeights(activations=y, output=logits, label=label, params=parameters, h_parameters= h_parameters, beta=self.model.beta)

            """ meta update """
            # -- predict
            y, logits = torch.func.functional_call(self.model, (parameters, h_parameters), x_qry)

            # -- L1 regularization
            l1_reg = torch.norm(self.UpdateWeights.P_matrix, 1)

            loss_meta = self.loss_func(logits, y_qry.ravel()) + l1_reg * self.metaLossRegularization

            # -- compute and store meta stats
            acc = meta_stats(logits, parameters, y_qry.ravel(), y, self.model.beta, self.result_directory)

            # -- update params
            theta_temp = [theta.detach().clone() for theta in self.UpdateWeights.P_matrix[0, :]]
            K_norm = torch.norm(self.UpdateWeights.K_matrix.detach().clone(), 1)
            v_values = self.UpdateWeights.v_vector.detach().clone()[0, :]
            self.UpdateMetaParameters.zero_grad()
            loss_meta.backward()

            # -- gradient clipping
            #torch.nn.utils.clip_grad_norm_(self.UpdateWeights.all_meta_parameters.parameters(), 5000)

            # -- update
            self.UpdateMetaParameters.step()

            # -- log
            log([loss_meta.item()], self.result_directory + '/loss_meta.txt')

            line = 'Train Episode: {}\tLoss: {:.6f}\tAccuracy: {:.3f}'.format(eps+1, loss_meta.item(), acc)
            for idx, param in enumerate(theta_temp):
                line += '\tMetaParam_{}: {:.6f}'.format(idx + 1, param.clone().detach().cpu().numpy())
            for idx, v in enumerate(v_values):
                line += '\tV_{}: {:.6f}'.format(idx + 1, v.clone().detach().cpu().numpy())
            line += '\tK_norm: {:.6f}'.format(K_norm.clone().detach().cpu().numpy())
            if self.display:
                print(line)

            if self.save_results:
                self.summary_writer.add_scalar('Loss/meta', loss_meta.item(), eps)
                self.summary_writer.add_scalar('Accuracy/meta', acc, eps)
                for idx, param in enumerate(theta_temp):
                    self.summary_writer.add_scalar('MetaParam_{}'.format(idx + 1), param.clone().detach().cpu().numpy(), eps)
                for idx, v in enumerate(v_values):
                    self.summary_writer.add_scalar('V_{}'.format(idx + 1), v.clone().detach().cpu().numpy(), eps)
                self.summary_writer.add_scalar('K_norm', K_norm.clone().detach().cpu().numpy(), eps)            

                with open(self.result_directory + '/params.txt', 'a') as f:
                    f.writelines(line+'\n')

        # -- plot
        if self.save_results:
            self.summary_writer.close()
            self.plot()

def run(seed: int, display: bool = True, result_subdirectory: str = "testing",  non_linearity =  torch.nn.functional.tanh, numberOfChemicals: int = 1) -> None:
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

    # -- meta-train
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    metalearning_model = MetaLearner(device=device, result_subdirectory=result_subdirectory, save_results=True, model_type="all", metatrain_dataset=metatrain_dataset, seed=seed, display=display, numberOfChemicals=numberOfChemicals, non_linearity=non_linearity)
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

    # -- set up
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Pool', type=int, default=1, help='Number of processes to run in parallel')
    Parser.add_argument('--Num_chem', type=int, default=1, help='Number of chemicals in the complex synapse model')

    Args = Parser.parse_args()
    print(Args)

    """non_linearity = [torch.nn.functional.tanh] * Args.Pool
    results_directory = ['full_attempt/1', 'full_attempt/3', 'full_attempt/5'] * Args.Pool
    chemicals = [1,3,5] """

    # -- run
    run(0, True, "full_attempt_2", torch.nn.functional.tanh, 5)
    """if Args.Pool > 1:
        with Pool(Args.Pool) as P:
            P.starmap(run, zip([0] * Args.Pool, [False]*Args.Pool, results_directory, non_linearity, chemicals))
            P.close()
            P.join()
    else:
        run(0)"""

def pass_through(input):
    return input

if __name__ == '__main__':
    #torch.autograd.set_detect_anomaly(True)
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)