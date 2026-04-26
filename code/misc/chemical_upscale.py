import os

import numpy as np
import torch
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
from synapses.complex_synapse import ComplexSynapse


def chemical_upscale():
    load_model = (
        os.getcwd()
        + "/results_3/mode_9_3_datasets/2/20260426-171458"  # "/results_3/mode_9_scalar_9_chems_100/0/20260423-234050"
    )  # "/results_3/mode_9_scalar_10/1/20251124-005417"
    oldModelOptions = complexOptions(
        nonLinear=nonLinearEnum.tanh,
        update_rules=[0, 1, 2, 3, 4, 6, 9],  # 5
        bias=False,
        pMatrix=pMatrixEnum.first_col,
        kMatrix=kMatrixEnum.zero,
        minTau=2,
        maxTau=100,
        y_vector=yVectorEnum.none,
        z_vector=zVectorEnum.default,
        operator=operatorEnum.mode_9,  # _pre_activation,
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
        gating=gatingEnum.no_gating,
        disagreement_regularization=False,
    )
    old_chems = 5
    old_state_dict = torch.load(load_model + "/UpdateWeights.pth", weights_only=True, map_location="cpu")
    old_model = ComplexSynapse(
        device="cpu",
        numberOfChemicals=old_chems,
        complexOptions=oldModelOptions,
        params=None,
        adaptionPathway="forward",
    )

    print(old_model.y_vector)

    new_chems = 9
    newModelOptions = complexOptions(
        nonLinear=nonLinearEnum.tanh,
        update_rules=[0, 1, 2, 3, 4, 6, 9],  # 5
        bias=False,
        pMatrix=pMatrixEnum.first_col,
        kMatrix=kMatrixEnum.zero,
        minTau=2,
        maxTau=100,
        y_vector=yVectorEnum.none,
        z_vector=zVectorEnum.default,
        operator=operatorEnum.mode_9,  # _pre_activation,
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
        gating=gatingEnum.no_gating,
        disagreement_regularization=False,
    )
    new_model = ComplexSynapse(
        device="cpu",
        numberOfChemicals=new_chems,
        complexOptions=newModelOptions,
        params=None,
        adaptionPathway="forward",
    )
    print(new_model.y_vector)

    indices_converter = [0, 2, 4, 6, 8]  # [0, 1, 3, 4, 6, 7, 9, 10, 12]
    new_model_state_dict = new_model.state_dict()

    new_model_state_dict["K_matrix"][np.ix_(indices_converter, indices_converter)] = old_state_dict["K_matrix"]
    new_model_state_dict["P_matrix"][indices_converter, :] = old_state_dict["P_matrix"]
    new_model_state_dict["v_vector"][:, indices_converter] = old_state_dict["v_vector"]

    other_indices = [i for i in range(new_chems) if i not in indices_converter]
    new_model_state_dict["v_vector"][:, other_indices] = -1.5

    new_model_state_dict["all_meta_parameters.0"] = new_model_state_dict["K_matrix"]
    new_model_state_dict["all_meta_parameters.1"] = new_model_state_dict["P_matrix"]
    new_model_state_dict["all_meta_parameters.2"] = new_model_state_dict["v_vector"]

    save_model_path = os.getcwd() + "/results_3/mode_9_CB_converted_9_chems"

    os.makedirs(save_model_path, exist_ok=True)
    torch.save(new_model_state_dict, save_model_path + "/UpdateWeights.pth")

    text_file_path = save_model_path + "/model_options.txt"
    with open(text_file_path, "w") as f:
        f.write("Converted from " + load_model + "\n")
