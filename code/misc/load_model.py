import os

import torch
import torch.nn as nn
from code.synapses.complex_synapse import ComplexSynapse
from code.misc.dataset import DataProcess, EmnistDataset
from code.metalearners.meta_learner_main import RosenbaumChemicalNN
from torch.utils.data import DataLoader, RandomSampler

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
    numberOfChemicals = 1
    model = RosenbaumChemicalNN(device, numberOfChemicals)
    complex = ComplexSynapse(device=device, mode="all", numberOfChemicals=numberOfChemicals, non_linearity=torch.nn.functional.tanh, options=options, 
                                                params=model.named_parameters())
    
    # -- load model
    directory = os.path.join(os.getcwd(), "results/Mode_1")
    directory = os.path.join(directory, "mode_1_1_chemical")
    complex.load_state_dict(torch.load(os.path.join(directory, "UpdateWeights.pth"), map_location=device, weights_only=True))
    model.load_state_dict(torch.load(os.path.join(directory, "model.pth"), map_location=device, weights_only=True))
    
    vector_chemical_1 = complex.bias_dictonary["chemical1"].reshape(-1)
    vector_chemical_1 = torch.abs(vector_chemical_1)
    max_chemical_1 = vector_chemical_1.max()
    index_max_chemical_1 = torch.where(vector_chemical_1 == max_chemical_1)
    print(index_max_chemical_1)
    print(max_chemical_1)

    vector_chemical_2 = complex.bias_dictonary["chemical2"].reshape(-1)
    vector_chemical_2 = torch.abs(vector_chemical_2)
    max_chemical_2 = vector_chemical_2.max()
    index_max_chemical_2 = torch.where(vector_chemical_2 == max_chemical_2)
    print(index_max_chemical_2)
    print(max_chemical_2)

    vector_chemical_3 = complex.bias_dictonary["chemical3"].reshape(-1)
    vector_chemical_3 = torch.abs(vector_chemical_3)
    max_chemical_3 = vector_chemical_3.max()
    index_max_chemical_3 = torch.where(vector_chemical_3 == max_chemical_3)
    print(index_max_chemical_3)
    print(max_chemical_3)

    vector_chemical_4 = complex.bias_dictonary["chemical4"].reshape(-1)
    vector_chemical_4 = torch.abs(vector_chemical_4)
    max_chemical_4 = vector_chemical_4.max()
    index_max_chemical_4 = torch.where(vector_chemical_4 == max_chemical_4)
    print(index_max_chemical_4)
    print(max_chemical_4)

    vector_chemical_5 = complex.bias_dictonary["chemical5"].reshape(-1)
    vector_chemical_5 = torch.abs(vector_chemical_5)
    max_chemical_5 = vector_chemical_5.max()
    index_max_chemical_5 = torch.where(vector_chemical_5 == max_chemical_5)
    print(index_max_chemical_5)
    print(max_chemical_5)

    # -- testing
    chemical2 = nn.Parameter(torch.zeros(size=(5, 130, 170), device="cpu"))
    nn.init.xavier_uniform_(chemical2)
    #print(chemical2)