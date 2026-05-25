import os

import equinox as eqx
import jax.numpy as jnp
from options.complex_options import (
    operatorEnum,
    yVectorEnum,
    zVectorEnum,
)
from options.fast_rnn_options import fastRnnOptions
from options.jax_rnn_meat_learner_options import (
    JaxActivationNonLinearEnum,
)
from synapses.jax_fast_rnn import JAXFastRnn


def expand_Q_matrix():
    jax_model_options = fastRnnOptions(
            nonLinear=JaxActivationNonLinearEnum.tanh,
            update_rules=[0, 1, 2, 4, 9, 12],  # 4
            minSlowTau=2,
            maxSlowTau=100,
            y_vector=yVectorEnum.none,
            z_vector=zVectorEnum.default,
            operator=operatorEnum.mode_9,
        )
    new_jax_model_options = fastRnnOptions(
            nonLinear=JaxActivationNonLinearEnum.tanh,
            update_rules=[0, 1, 2, 3, 4, 9, 12],  # 4
            minSlowTau=2,
            maxSlowTau=100,
            y_vector=yVectorEnum.none,
            z_vector=zVectorEnum.default,
            operator=operatorEnum.mode_9,
        )

    numberOfChemicals = 3
    jax_model = JAXFastRnn(
        numberOfChemicals=numberOfChemicals,
        fastRnnOptionsVal=jax_model_options,
    )
    model_location = os.getcwd() + "/results_2/jax_rnn_DFA_3_chems/20260221-195452"#/results_3/jax_rnn_1_chem/20260423-005009"
    loaded_metaOptimizer = eqx.tree_deserialise_leaves(
        model_location + "/meta_learner_model.eqx", jax_model
    )
    jax_model = eqx.tree_at(
        lambda m: (m.Q_matrix, m.K_matrix, m.v_vector),
        jax_model,
        (
            loaded_metaOptimizer.Q_matrix,
            loaded_metaOptimizer.K_matrix,
            loaded_metaOptimizer.v_vector,
        ),
    )

    old_Q_matrix = jax_model.Q_matrix
    print("Old Q matrix shape:", old_Q_matrix.shape)
    new_Q_matrix = jnp.zeros((old_Q_matrix.shape[0], len(new_jax_model_options.update_rules)))
    for index, update_rule in enumerate(jax_model_options.update_rules):
        if update_rule in new_jax_model_options.update_rules:
            new_index = new_jax_model_options.update_rules.index(update_rule)
            new_Q_matrix = new_Q_matrix.at[:, new_index].set(old_Q_matrix[:, index])
    new_jax_model = eqx.tree_at(lambda m: (m.Q_matrix, m.K_matrix, m.v_vector), jax_model, (new_Q_matrix, jax_model.K_matrix, jax_model.v_vector))

    

    save_dir = os.getcwd() + "/results_3/jax_rnn_Q_expanded/3"
    os.makedirs(save_dir, exist_ok=True)

    eqx.tree_serialise_leaves(save_dir + "/meta_learner_model.eqx", new_jax_model)
    

    text_file_path = save_dir + "/model_options.txt"
    with open(text_file_path, "w") as f:
        f.write("Converted from " + model_location + "\n")