import os
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler

import torch
from torch.utils.data import DataLoader, RandomSampler

from meta_learner_main import MetaLearner
from dataset import EmnistDataset, DataProcess

def data_loader(epochs):
    dataset = EmnistDataset(trainingDataPerClass=50, queryDataPerClass=10, dimensionOfImage=28)
    sampler = RandomSampler(data_source=dataset, replacement=True, num_samples=epochs * 5)
    metatrain_dataset = DataLoader(dataset=dataset, sampler=sampler, batch_size=5, drop_last=True)
    return metatrain_dataset

def ray_tune_main():
    metatrain_dataset = data_loader(epochs=200)
    config = {
        'non_linearity': tune.grid_search(['relu', 'sigmoid', 'tanh']),
        'optimizer': tune.grid_search(['adam', 'sgd']),
        'lr': tune.grid_search([0.001, 0.01, 0.1]),
        'chemicals': tune.grid_search([1, 3 , 5, 8]),
        'P_Matrix': tune.grid_search(['random', 'non-random']),
        'K_Matrix': tune.grid_search(['random', 'non-random']),
        'Update_rules': tune.grid_search([[0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 9]]),
    }

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=200,
        grace_period=50,
        reduction_factor=2)
    
    currentDir = os.getcwd()
    saveDir = os.path.join(currentDir, '../raytune')
    
    result = tune.run(
        trainable,
        resources_per_trial={"cpu": 2},
        config=config,
        num_samples=10,
        scheduler=scheduler,
        local_dir=saveDir,
        fail_fast=True,
    )
    
    best_trial = result.get_best_trial("accuracy", "max", "last-10-avg")
    print(f"Best trial final loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
    

def trainable(config):
    metatrain_dataset = data_loader(epochs=200)
    args = {}
    args['p_matrix'] = config['P_Matrix']
    args['k_matrix'] = config['K_Matrix']
    args['update_rules'] = config['Update_rules']
    args['lr'] = config['lr']
    args['optimizer'] = config['optimizer']
    args['update_rules'] = config['Update_rules']
    
    num_chemicals = config['chemicals']

    non_linearity = None
    if config['non_linearity'] == 'relu':
        non_linearity = torch.nn.functional.relu
    elif config['non_linearity'] == 'sigmoid':
        non_linearity = torch.nn.functional.sigmoid
    elif config['non_linearity'] == 'tanh':
        non_linearity = torch.nn.functional.tanh
    
    MetaLearner_instance = MetaLearner(model_type='all', device="cpu", numberOfChemicals=num_chemicals, save_results='false', non_linearity=non_linearity, options=args, display=False, save_results=False, metatrain_dataset=metatrain_dataset)
    print(MetaLearner_instance)
