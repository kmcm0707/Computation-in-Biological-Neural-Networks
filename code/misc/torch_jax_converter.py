import os

import equinox as eqx
import jax.numpy as jnp
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
from options.fast_rnn_options import fastRnnOptions
from options.jax_rnn_meat_learner_options import (
    JaxActivationNonLinearEnum,
)
from synapses.jax_fast_rnn import JAXFastRnn


def torch_to_jax_chemical_nn():
    jax_model_options = fastRnnOptions(
            nonLinear=JaxActivationNonLinearEnum.tanh,
            update_rules=[0, 1, 2, 3, 4, 9, 12],  # 4
            minSlowTau=2,
            maxSlowTau=100,
            y_vector=yVectorEnum.none,
            z_vector=zVectorEnum.default,
            operator=operatorEnum.mode_9,
        )
    model_options = complexOptions(
            nonLinear=nonLinearEnum.tanh,
            update_rules=[0, 1, 2, 3, 4, 6, 9],
            bias=False,
            pMatrix=pMatrixEnum.first_col,
            kMatrix=kMatrixEnum.zero,
            minTau=2,  # + 1 / 50,
            maxTau=50,
            y_vector=yVectorEnum.none,
            z_vector=zVectorEnum.default,
            operator=operatorEnum.mode_9,  # _pre_activation,
            train_z_vector=False,
            mode=modeEnum.all,
            v_vector=vVectorEnum.default,
            eta=1,
            beta=0.01,  ## Only for v_vector=random_beta
            kMasking=False,
            individual_different_v_vector=True,  # Individual Model Only
            scheduler_t0=None,  # Only mode_3
            train_tau=False,
            scale_chemical_weights=False,
            gating=gatingEnum.no_gating,
            disagreement_regularization=False,
        )
    
    model_location = os.getcwd() + "/results_3/mode_9_rand/0/20251105-152312/UpdateWeights.pth"
    state_dict = torch.load(model_location, map_location=torch.device("cpu"), weights_only=True)
    P_matrix = state_dict["P_matrix"]
    K_matrix = state_dict["K_matrix"]
    v_vector = state_dict["v_vector"]
    y_vector = state_dict["y_vector"]
    z_vector = state_dict["z_vector"]

    number_of_chemicals = K_matrix.shape[0]
    jax_model = JAXFastRnn(
        numberOfChemicals=number_of_chemicals,
        fastRnnOptionsVal=jax_model_options,
    )


    P_matrix_indicies = [0, 1, 2, 3, 4, 6, 9]
    ## swap columns 6 and 9 to match the order in the JAX model # Note always remove 8 if it was in model
    print("Original P matrix shape:", P_matrix.shape)
    P_matrix[:, [6, 9]] = P_matrix[:, [9, 6]]
    P_matrix = P_matrix[:, P_matrix_indicies]
    
    Q_matrix_jax = jnp.array(P_matrix.detach().numpy())  # Mapping P -> Q
    K_matrix_jax = jnp.array(K_matrix.detach().numpy())  # Mapping K -> K

    v_vector = v_vector.squeeze()  # Remove extra dimensions if present
    y_vector = y_vector.squeeze()
    z_vector = z_vector.squeeze()
    v_vector_jax = jnp.array(v_vector.detach().numpy())
    y_vector_jax = jnp.array(y_vector.detach().numpy())
    z_vector_jax = jnp.array(z_vector.detach().numpy())

    jax_model = eqx.tree_at(
        lambda m: (m.Q_matrix, m.K_matrix, m.v_vector, m.y_vector, m.z_vector),
        jax_model,
        (Q_matrix_jax, K_matrix_jax, v_vector_jax, y_vector_jax, z_vector_jax)
    )

    save_dir = os.getcwd() + "/results_3/mode_9_rand_converted"
    os.makedirs(save_dir, exist_ok=True)

    eqx.tree_serialise_leaves(save_dir + "/meta_learner_model.eqx", jax_model)

    text_file_path = save_dir + "/model_options.txt"
    with open(text_file_path, "w") as f:
        f.write("Converted from " + model_location + "\n")


