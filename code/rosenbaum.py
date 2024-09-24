import os
import torch
import warnings
import argparse
import datetime

from torch import nn, optim
from torch.utils.data import DataLoader, RandomSampler

from dataset import EmnistDataset, DataProcess

from rosenbaum_optimizer import plasticity_rule, RosenbaumOptimizer

from utils import log, meta_stats, Plot

class RosenbaumNN(nn.Module):
    """

    Rosenbaum Neural Network class.
    
    """

    def __init__(self):

        # Initialize the parent class
        super(RosenbaumNN, self).__init__()

        # Set the seed for reproducibility
        torch.manual_seed(3)

        # Set the device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Model
        dim_out = 47
        self.linear1 = nn.Linear(784, 170, bias=False)
        self.linear2 = nn.Linear(170, 130, bias=False)
        self.linear3 = nn.Linear(130, 100, bias=False)
        self.linear4 = nn.Linear(100, 70, bias=False)
        self.linear5 = nn.Linear(70, dim_out, bias=False)

        # Feedback pathway (fixed or symmetric) for plasticity
        # Symmetric feedback is used for backpropagation training
        # Fixed feedback is used for feedback alignment meta-learning
        self.feedback1 = nn.Linear(784, 170, bias=False)
        self.feedback2 = nn.Linear(170, 130, bias=False)
        self.feedback3 = nn.Linear(130, 100, bias=False)
        self.feedback4 = nn.Linear(100, 70, bias=False)
        self.feedback5 = nn.Linear(70, dim_out, bias=False)

        # Plastisity meta-parameters
        self.theta0 = nn.Parameter(torch.tensor(1e-3).float()) # Pseudo-inverse
        self.theta1 = nn.Parameter(torch.tensor(0.).float()) # Hebbian
        self.theta2 = nn.Parameter(torch.tensor(0.).float()) # Oja
        
        self.theta = nn.ParameterList([self.theta0, self.theta1, self.theta2])

        # Activation function
        self.beta = 10
        self.activation = nn.Softplus(beta=self.beta)

    def forward(self, x):
        y0 = x.squeeze(1)

        y1 = self.activation(self.linear1(y0))
        y2 = self.activation(self.linear2(y1))
        y3 = self.activation(self.linear3(y2))
        y4 = self.activation(self.linear4(y3))
        y5 = self.activation(self.linear5(y4))

        return (y0, y1, y2, y3, y4), y5
    
class RosenbaumMetaLearner:
    """
    Rosenbaum Meta-learner class.

    Class for meta-learning algorithms.

    The MetaLearner class is used to define meta-learning algorithm.
    """
    def __init__(self, metatrain_dataset):
        """
            Initialize the Meta-RosenbaumMetaLearner.

        :param metatrain_dataset: (DataLoader) The meta-training dataset.
        """
        # -- processor params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # -- data params
        self.trainingDataPerClass = 50
        self.queryDataPerClass = 10
        self.database = "emnist"
        self.metatrain_dataset = metatrain_dataset
        self.data_process = DataProcess(K=self.trainingDataPerClass, Q=self.queryDataPerClass, dim=28, device=self.device)

        # -- model params
        self.model = self.load_model().to(self.device)
        self.Theta = nn.ParameterList([*self.model.theta])
        self.feedbackMode = "fixed" # 'sym' or 'fixed'

        # -- optimization params
        self.metaLossRegularization = 0
        self.loss_func = nn.CrossEntropyLoss()
        self.UpdateWeights = RosenbaumOptimizer(plasticity_rule, self.Theta, self.fbk)
        self.UpdateMetaParameters = optim.Adam([{'params': self.model.theta.parameters(), 'lr': 1e-3}])

        # -- log params
        self.result_directory = os.getcwd() + "../results"
        os.makedirs(self.result_directory, exist_ok=True)
        self.average_window = 10
        self.plot = Plot(self.res_dir, len(self.Theta), self.average_window)

    def load_model(self):
        """
            Load classifier model

        Loads the classifier network and sets the adaptation, meta-learning,
        and grad computation flags for its variables.

        :param args: (argparse.Namespace) The command-line arguments.
        :return: model with flags "meta_fwd", "adapt", set for its parameters
        """
        # -- init model
        model = RosenbaumNN()

        # -- learning flags
        for key, val in model.named_parameters():
            if 'linear' in key:
                val.meta_fwd, val.adapt = False, True
            elif 'feedback' in key:
                val.meta_fwd, val.adapt, val.requires_grad = False, False, False
            elif 'theta' in key:
                val.meta_fwd, val.adapt = True, False

        return model
    
    @staticmethod
    def weights_init(m):
        """
            Initialize weight matrices.

        The function initializes weight matrices by filling them with values based on
        the Xavier initialization method proposed by Glorot et al. (2010). The method
        scales the initial values of the weights based on the number of input and output
        units to the layer.
        * Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training
        deep feedforward neural networks." In Proceedings of the thirteenth international
        conference on artificial intelligence and statistics, pp. 249-256. JMLR Workshop
        and Conference Proceedings, 2010.

        :param m: modules in the model.
        """
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:

            # -- weights
            init_range = torch.sqrt(torch.tensor(6.0 / (m.in_features + m.out_features)))
            m.weight.data.uniform_(-init_range, init_range)

            # -- bias
            if m.bias is not None:
                m.bias.data.uniform_(-init_range, init_range)

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

        if self.fbk == 'sym':
            self.model.feedback1.weight.data = self.model.linear1.weight.data
            self.model.feedback2.weight.data = self.model.linear2.weight.data
            self.model.feedback3.weight.data = self.model.linear3.weight.data
            self.model.feedback4.weight.data = self.model.linear4.weight.data
            self.model.feedback5.weight.data = self.model.linear5.weight.data


        # -- clone module parameters
        # params = {key: val.clone() for key, val in dict(self.model.named_parameters()).items() if '.' in key}

        # -- set adaptation flags for cloned parameters
        # for key in params:
        #    params[key].adapt = dict(self.model.named_parameters())[key].adapt

        # return params

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
        for eps, data in enumerate(self.metatrain_dataset):

            # -- initialize
            self.reinitialize()

            # -- training data
            x_trn, y_trn, x_qry, y_qry = self.data_process(data, self.queryDataPerClass)

            """ adaptation """
            for itr_adapt, (x, label) in enumerate(zip(x_trn, y_trn)):

                # -- predict
                y, logits = self.model(x.unsqueeze(0).unsqueeze(0))

                # -- update network params
                self.UpdateWeights(self.model.named_parameters(), logits, label, y, self.model.Beta, self.Theta)

            """ meta update """
            # -- predict
            y, logits = self.model(x_qry.unsqueeze(1))

            # -- L1 regularization
            l1_reg = 0
            for theta in self.model.theta.parameters():
                l1_reg += torch.linalg.norm(theta, 1)

            loss_meta = self.loss_func(logits, y_qry.ravel()) + l1_reg * self.lamb

            # -- compute and store meta stats
            acc = meta_stats(logits, (self.model.named_parameters(), y_qry.ravel(), y, self.model.Beta, self.res_dir))

            # -- update params
            Theta = [p.detach().clone() for p in self.Theta]
            self.UpdateMetaParameters.zero_grad()
            loss_meta.backward()
            self.UpdateMetaParameters.step()

            # -- log
            log([loss_meta.item()], self.res_dir + '/loss_meta.txt')

            line = 'Train Episode: {}\tLoss: {:.6f}\tAccuracy: {:.3f}'.format(eps+1, loss_meta.item(), acc)
            for idx, param in enumerate(Theta):
                line += '\tMetaParam_{}: {:.6f}'.format(idx + 1, param.cpu().numpy())
            print(line)
            with open(self.res_dir + '/params.txt', 'a') as f:
                f.writelines(line+'\n')

        # -- plot
        self.plot()

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

    # -- load data
    queryDataPerClass = 10
    dataset = EmnistDataset(trainingDataPerClass=50, queryDataPerClass=queryDataPerClass, dim=28)
    sampler = RandomSampler(data_source=dataset, replacement=True, num_samples=600 * queryDataPerClass)
    metatrain_dataset = DataLoader(dataset=dataset, sampler=sampler, batch_size=queryDataPerClass, drop_last=True)

    # -- meta-train
    metalearning_model = RosenbaumMetaLearner(metatrain_dataset)
    metalearning_model.train()


if __name__ == '__main__':
    main()
