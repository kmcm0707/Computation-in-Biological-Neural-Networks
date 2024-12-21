import os
import random
import numpy as np
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler

import torch
from torch.utils.data import DataLoader, RandomSampler

from code.metalearners.meta_learner_main import MetaLearner
from code.misc.dataset import EmnistDataset, DataProcess

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
        'Update_rules': tune.grid_search([[0, 2, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 9]]),
    }

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=200,
        grace_period=20,
        reduction_factor=3,
    )
    
    currentDir = os.getcwd()
    saveDir = os.path.join(currentDir, 'raytune_results')
    
    result = tune.run(
        trainable,
        resources_per_trial={"cpu": 2},
        config=config,
        num_samples=10,
        scheduler=scheduler,
        storage_path=saveDir,
        fail_fast=True,
        trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
    )
    
    best_trial = result.get_best_trial("accuracy", "max", "last-10-avg")
    print(f"Best trial final loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
    

def trainable(config):
    metatrain_dataset = data_loader(epochs=200)
    args = {}
    args['P_matrix'] = config['P_Matrix']
    args['K_matrix'] = config['K_Matrix']
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
    
    MetaLearner_instance = MetaLearner(model_type='all', device="cpu", numberOfChemicals=num_chemicals, non_linearity=non_linearity, options=args, display=True, save_results=False, metatrain_dataset=metatrain_dataset, raytune=True)
    MetaLearner_instance.train()

if __name__ == "__main__":
    torch.manual_seed(2)
    torch.cuda.manual_seed(2)
    torch.cuda.manual_seed_all(2)
    np.random.seed(2)
    random.seed(2)
    ray_tune_main()