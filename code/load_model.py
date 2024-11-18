import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

from dataset import DataProcess, EmnistDataset
from complex_synapse import ComplexSynapse
from meta_learner_main import RosenbaumChemicalNN

if __name__ == '__main__':
    device = "cpu"
    options = {}
    options['lr'] = 5e-4
    options['optimizer'] = 'adam'
    options['K_Matrix'] = 'n'
    options['P_Matrix'] = 'n'
    options['metaLossRegularization'] = 0
    options['update_rules'] = [0, 1, 2, 3, 4, 8, 9]
    options['operator'] = 'mode_1'
    options['chemicals'] = 'n'
    options['bias'] = True
    options['y_vector'] = "first_one"
    options['v_vector'] = "none"
    model = RosenbaumChemicalNN(device, 5)
    complex = ComplexSynapse(device=device, mode="all", numberOfChemicals=5, non_linearity=torch.nn.functional.tanh, options=options, 
                                                params=model.named_parameters())
    
    # -- load model
    directory = os.path.join(os.getcwd(), "results")
    directory = os.path.join(directory, "mode_2_bias")
    complex.load_state_dict(torch.load(os.path.join(directory, "UpdateWeights.pth"), map_location=device, weights_only=True))
    model.load_state_dict(torch.load(os.path.join(directory, "model.pth"), map_location=device, weights_only=True))
    print(complex.bias_dictonary["chemical3"])
    print(model.chemical3)

    # -- testing
    chemical2 = nn.Parameter(torch.zeros(size=(5, 130, 170), device="cpu"))
    nn.init.xavier_uniform_(chemical2)
    #print(chemical2)