import argparse
import copy
import datetime
import os
from multiprocessing import Pool
from typing import Literal, Union

import numpy as np
import synapses.complex_synapse as complex_synapse
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
from synapses.complex_synapse import ComplexSynapse


def load_model():
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
    numberOfChemicals = 5
    model = ChemicalNN(
        "cuda",
        numberOfChemicals,
        small=False,
        train_feedback=False,
        typeOfFeedback=typeOfFeedbackEnum.FA,
    )
    synapse = ComplexSynapse(
        device="cuda",
        numberOfChemicals=numberOfChemicals,
        complexOptions=modelOptions,
        params=model.named_parameters(),
        adaptionPathway="forward",
    )
    synapse.P_matrix = torch.nn.Parameter(
        torch.tensor(
            [
                [
                    0.00480236,
                    0.01547999,
                    0.01321499,
                    -0.00598501,
                    0.01414017,
                    -0.00078306,
                    0.0,
                    0.0,
                    0.00453079,
                    -0.00031115,
                ],
                [
                    0.00513365,
                    0.0196637,
                    0.01900036,
                    -0.00553879,
                    0.01748486,
                    0.00031542,
                    0.0,
                    0.0,
                    0.0018963,
                    0.00059697,
                ],
                [
                    0.00682191,
                    0.01997717,
                    0.02258199,
                    -0.00578936,
                    0.0173986,
                    0.00139651,
                    0.0,
                    0.0,
                    -0.00058986,
                    0.00058518,
                ],
                [
                    0.01108505,
                    0.01715942,
                    0.02305288,
                    -0.00502636,
                    0.01303791,
                    0.00275446,
                    0.0,
                    0.0,
                    0.00138551,
                    0.0003388,
                ],
                [
                    0.01493413,
                    0.01977227,
                    0.0187822,
                    -0.00181738,
                    0.01061159,
                    -0.0001932,
                    0.0,
                    0.0,
                    0.00215365,
                    0.00184618,
                ],
            ]
        )
    )
    synapse.K_matrix = torch.nn.Parameter(
        torch.tensor(
            [
                [
                    0.00378589,
                    0.00547296,
                    0.00749932,
                    0.00645233,
                    0.00069426,
                ],
                [
                    0.00460122,
                    0.00586227,
                    0.00762682,
                    0.00684959,
                    0.00172009,
                ],
                [
                    0.00587628,
                    0.00630832,
                    0.00732544,
                    0.00677857,
                    0.00141032,
                ],
                [
                    0.00720004,
                    0.00636215,
                    0.00671771,
                    0.0056138,
                    0.00040717,
                ],
                [
                    0.00536641,
                    0.00477057,
                    0.00469338,
                    0.00415535,
                    0.00263022,
                ],
            ]
        )
    )
    synapse.v_vector = torch.nn.Parameter(torch.tensor([[0.18758841, 0.1872707, 0.19293375, 0.21503067, 0.20694348]]))
    torch.save(synapse, "UpdateWeights.pth")
