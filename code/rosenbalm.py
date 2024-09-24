import os
import torch
import warnings
import argparse
import datetime

from torch import nn, optim
from random import randrange
from torch.nn.utils import _stateless
from torch.utils.data import DataLoader, RandomSampler

from dataset import EmnistDataset, DataProcess

class RosenbaumNN(nn.Module):
    """

    Rosenbaum Neural Network class.
    
    """

    def __init__(self, args):

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
    
    """
    







        
    