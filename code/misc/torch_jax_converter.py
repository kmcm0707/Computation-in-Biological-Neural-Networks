import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from nn.chemical_nn import ChemicalNN
from nn.jax_chemical_nn import JAXFeedforwardNN
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
    JaxErrorTypeEnum,
)
from options.meta_learner_options import (
    typeOfFeedbackEnum,
)
from synapses.complex_synapse import ComplexSynapse
from synapses.jax_fast_rnn import JAXFastRnn
from torch import functional, nn
from torch.nn import functional


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
    
    #model_location = os.getcwd() + r"/results_3/mode_9_scalar_9_chems_converted_true\0\20260420-043518/UpdateWeights.pth"#mode_9_scalar_10/1/20251124-005417/UpdateWeights.pth"
    #state_dict = torch.load(model_location, map_location=torch.device("cpu"), weights_only=True)
    outer_location = os.getcwd() + r"\results_3\mode_10_scalar_13_chems_extended_full_sweep"
    tau_mins = os.listdir(outer_location)
    for tau_min in tau_mins:
        tau_min_location = os.path.join(outer_location, tau_min)
        tau_min_location = os.path.join(tau_min_location, "0")
        tau_maxs = os.listdir(tau_min_location)
        for tau_max in tau_maxs:
            tau_max_location = os.path.join(tau_min_location, tau_max)
            state_dict = torch.load(os.path.join(tau_max_location, "UpdateWeights.pth"), map_location=torch.device("cpu"), weights_only=True)
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

            save_dir = os.getcwd() + "/results_4/mode_9_scalar_converted_13_chems_extended_full_sweep/" + tau_min + "/" + tau_max
            os.makedirs(save_dir, exist_ok=True)

            eqx.tree_serialise_leaves(save_dir + "/meta_learner_model.eqx", jax_model)

            model_location = os.path.join(tau_max_location, "UpdateWeights.pth")

            text_file_path = save_dir + "/model_options.txt"
            with open(text_file_path, "w") as f:
                f.write("Converted from " + model_location + "\n")

    tester()

def tester():
    model_location = os.getcwd() + "/results_3/mode_9_scalar_10/1/20251124-005417/UpdateWeights.pth"
    state_dict = torch.load(model_location, map_location=torch.device("cpu"), weights_only=True)
    jax_model_location = os.getcwd() + "/results_3/mode_9_scalar_converted/meta_learner_model.eqx"
    

    jax_model_options = fastRnnOptions(
            nonLinear=JaxActivationNonLinearEnum.tanh,
            update_rules=[0, 1, 2, 3, 4, 9, 12],  # 4
            minSlowTau=2,
            maxSlowTau=50,
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
    number_of_chemicals = state_dict["K_matrix"].shape[0]
    jax_model = JAXFastRnn(
        numberOfChemicals=number_of_chemicals,
        fastRnnOptionsVal=jax_model_options,
    )
    jax_model = eqx.tree_deserialise_leaves(jax_model_location, jax_model)

    typeOfFeedback = typeOfFeedbackEnum.scalar
    torch_NN = ChemicalNN(
        device="cpu",
        numberOfChemicals=number_of_chemicals,
        typeOfFeedback=typeOfFeedback,
    )
    torch_model = ComplexSynapse(
                device="cpu",
                numberOfChemicals=number_of_chemicals,
                complexOptions=model_options,
                params=torch_NN.named_parameters(),
                adaptionPathway="forward",
            )
    torch_model.load_state_dict(state_dict)


    ## Pytorch Setup
    for key, val in torch_NN.named_parameters():
        if "forward" in key or "fc" in key or "conv" in key:
            val.adapt = "forward"
        elif "feedback" in key:
            val.adapt, val.requires_grad = (
                "feedback",
                False,
            )

    def weights_init(modules):
        classname = modules.__class__.__name__
        if classname.find("Linear") != -1:

            # -- weights
            nn.init.xavier_uniform_(modules.weight)

            # -- bias
            if modules.bias is not None:
                nn.init.xavier_uniform_(modules.bias)
        elif classname.find("Conv") != -1:

            # -- weights
            nn.init.xavier_uniform_(modules.weight)

            # -- bias
            if modules.bias is not None:
                nn.init.xavier_uniform_(modules.bias)

    @torch.no_grad()
    def chemical_init(chemicals):
        for chemical in chemicals:
            nn.init.xavier_uniform_(chemical[0])
            for idx in range(chemical.shape[0] - 1):
                chemical[idx + 1] = chemical[0]
    
    torch_NN.apply(weights_init)
    chemical_init(torch_NN.chemicals)
    # -- module parameters
    # -- parameters are not linked to the model even if .clone() is not used
    params = {
        key: val.clone()
        for key, val in dict(torch_NN.named_parameters()).items()
        if "." in key and "chemical" not in key and "layer_norm" not in key
    }
    h_params = {
        key: val.clone()
        for key, val in dict(torch_NN.named_parameters()).items()
        if "chemical" in key and "feedback" not in key
    }

    for key in params:
        params[key].adapt = dict(torch_NN.named_parameters())[key].adapt

    torch_model.initial_update(params, h_params)

    jax_key = jax.random.PRNGKey(0)
    jax_NN = JAXFeedforwardNN(
        dim_out=47,
        key=jax_key,
        gradient=True,
        activation=JaxActivationNonLinearEnum.softplus,
        error_type=JaxErrorTypeEnum.DSEF,
    )
    def sync_model(jax_model, torch_model):
        # Get a flat dictionary of torch parameters for easy lookup
        t_params = {key: val.detach().numpy() for key, val in params.items()}
        
        # 1. Update Forward Weights
        for i in range(1, 6):
            key = f"forward{i}.weight"
            if key in t_params:
                # target the specific weight leaf
                jax_model = eqx.tree_at(
                    lambda m: getattr(getattr(m, f"forward{i}"), "weight"),
                    jax_model,
                    jnp.array(t_params[key])
                )

        new_layers = [
            getattr(jax_model, "forward1"),
            getattr(jax_model, "forward2"),
            getattr(jax_model, "forward3"),
            getattr(jax_model, "forward4"),
            getattr(jax_model, "forward5"),
        ]
        new_layers = tuple(new_layers)
    
        jax_model = eqx.tree_at(lambda m: m.layers, jax_model, new_layers)
                
        # 2. Update Feedback Weights (with Transpose)
        for i in range(1, 6):
            key = f"feedback{i}.weight"
            if key in t_params:
                jax_model = eqx.tree_at(
                    lambda m: getattr(getattr(m, f"feedback{i}"), "weight"),
                    jax_model,
                    jnp.array(t_params[key]).T  # Feedback is usually Transposed
                )
        
        new_feedback_layers = (
            getattr(jax_model, "feedback1"),
            getattr(jax_model, "feedback2"),
            getattr(jax_model, "feedback3"),
            getattr(jax_model, "feedback4"),
            getattr(jax_model, "feedback5"),
        )
        jax_model = eqx.tree_at(lambda m: m.feedback_layers, jax_model, new_feedback_layers)

        return jax_model

    jax_NN = sync_model(jax_NN, torch_NN)

    layers = jax_NN.layers
    synaptic_weights = []
    h_params_list = [h_params[key].detach().numpy() for key in h_params]
    for i, layer in enumerate(layers):
        #temp = getattr(layer, "weight")
        holder = jnp.tile(h_params_list[i][0, :, :].T, (number_of_chemicals, 1, 1))
        synaptic_weights.append(holder)
    synaptic_weights = tuple(synaptic_weights)

    np.testing.assert_allclose(
        synaptic_weights[0][0, :5, :5], synaptic_weights[0][1, :5, :5]
    ), "Chemical weights are not the same across chemicals after tiling"

    ## print out some weights to verify they match
    for i in range(1, 6):
        w_jax = getattr(getattr(jax_NN, f"forward{i}"), "weight")
        w_torch = params[f"forward{i}.weight"].detach().numpy()
        print(f"Forward {i} match: {np.allclose(w_jax, w_torch)}")
    for i in range(1, 6):
        w_jax = getattr(getattr(jax_NN, f"feedback{i}"), "weight")
        w_torch = params[f"feedback{i}.weight"].detach().numpy().T  # Transpose back for comparison
        print(f"Feedback {i} match: {np.allclose(w_jax, w_torch)}")
    
    h_params_list = [h_params[key].detach().numpy() for key in h_params]
    for i in range(0, 5):
        for ii in range(h_params_list[i].shape[0] - 1):
            np.testing.assert_allclose(
                h_params_list[i][ii, :, :],
                synaptic_weights[i][ii, :, :].T,
                atol=1e-5,
                rtol=1e-5,
                err_msg=f"Chemical weights are different at layer {i} chemical {ii}",
            )
    
    torch_NN.train()

    ## Test forward pass to ensure no errors before conversion
    example_input = torch.randn(1, 784)  # Example input tensor
    label = torch.tensor([0])  # Example label

    feedback_params = [param for name, param in torch_NN.named_parameters() if "feedback" in name]
    y, logits = torch.func.functional_call(
        torch_NN, (params, h_params), example_input
    )
    activations = y
    output = functional.softmax(logits, dim=1)
    feedback = {name: value for name, value in params.items() if "feedback" in name}
    error = [output - functional.one_hot(label, num_classes=47)]
    
    if typeOfFeedback == typeOfFeedbackEnum.DFA:
        for y, i in zip(reversed(activations), reversed(list(feedback))):
            error.insert(
                0, torch.matmul(error[-1], feedback[i]) * (1 - torch.exp(-torch_NN.beta * y))
            )
    elif typeOfFeedback == typeOfFeedbackEnum.scalar:
        # error_scalar = torch.norm(error[0], p=2, dim=1, keepdim=True)[0]
        # error_scalar = -error[0][0][label]
        # error_scalar = torch.tanh(error_scalar)  # tanh to avoid exploding gradients
        if output[0][label] > 0.5:
            error_scalar = torch.tensor(0, device="cpu")
        else:
            error_scalar = torch.tensor(1.0, device="cpu")
        for y, i in zip(reversed(activations), reversed(list(feedback))):
            error.insert(0, error_scalar * feedback[i] * (1 - torch.exp(-torch_NN.beta * y)))

    activations_and_output = [*activations, output]
    torch_model(
        params=params,
        h_parameters=h_params,
        error=error,
        activations_and_output=activations_and_output,
    )

    jax_example_input = jnp.array(example_input.squeeze().detach().numpy())
    jax_label = jnp.array(label.detach().numpy())
    jax_label = jax.nn.one_hot(jax_label, num_classes=47)
    jax_label = jax_label.squeeze()  # Remove extra dimensions if present

    jax_output, _, jax_logits, jax_error = jax_NN(jax_example_input, label=jax_label)

    np.testing.assert_allclose(
        jax_output, 
        logits.squeeze().detach().numpy(), 
        atol=1e-5, 
        rtol=1e-5,
        err_msg="Logits are mathematically the same but bit-different"
    )

    assert jax_output.shape == output.squeeze().detach().numpy().shape, f"Shape mismatch for output: {jax_output.shape} vs {output.detach().numpy().shape}"
    np.testing.assert_allclose(
        jax.nn.softmax(jax_output), 
        output.squeeze().detach().numpy(), 
        atol=1e-5, 
        rtol=1e-5,
        err_msg="Logits are mathematically the same but bit-different"
    )

    jax_activation_arr_linear = []
    for j in jax_logits:
        jax_activation_arr_linear.append(j[0])
    
    for j, t in zip(jax_activation_arr_linear, activations):
        assert j.shape == t.squeeze().detach().numpy().shape, f"Shape mismatch for activation: {j.shape} vs {t.detach().numpy().shape}"
        np.testing.assert_allclose(
            j, 
            t.squeeze().detach().numpy(), 
            atol=1e-5, 
            rtol=1e-5,
            err_msg="Activations are mathematically the same but bit-different"
        )

    jax_error_arr_linear = []
    for err in jax_error:
        jax_error_arr_linear.append(err[0])

    for i, (err_jax, err_torch) in enumerate(zip(jax_error_arr_linear, error)):
        assert err_jax.shape == err_torch.squeeze().detach().numpy().shape, f"Shape mismatch for error at layer {i}: {err_jax.shape} vs {err_torch.detach().numpy().shape}"
        np.testing.assert_allclose(
            err_jax, 
            err_torch.squeeze().detach().numpy(), 
            atol=1e-5, 
            rtol=1e-5,
            err_msg=f"Errors are different at layer {i}",
            verbose=True
        )

    def update_layer(w, p, act, err):
        return jax_model(w, p, act, err)

    activations_arr = tuple(jax_logits)
    errors_arr = tuple(jax_error)
    jax_parameters = jax_NN.layers

    results = jax.tree_util.tree_map(update_layer, synaptic_weights, jax_parameters, activations_arr, errors_arr)

    new_parameters = tuple(res[0] for res in results)
    new_synaptic_weights = tuple(res[1] for res in results)

    torch_forward_weights = [param for name, param in params.items() if "forward" in name and "weight" in name]
    for i in range(5):
        w_jax = new_parameters[i].weight
        w_torch = torch_forward_weights[i].detach().numpy()
        np.testing.assert_allclose(w_jax, w_torch, atol=1e-5, rtol=1e-5), f"Updated forward weights are different at layer {i}"

    h_params_list = [h_params[key].detach().numpy() for key in h_params]
    for i in range(5):
        w_jax = np.transpose(new_synaptic_weights[i], (0, 2, 1))  # Transpose back to match torch shape
        w_torch = h_params_list[i]
        np.testing.assert_allclose(w_jax, w_torch, atol=1e-5, rtol=1e-5), f"Updated chemical weights are different at layer {i}"
    


    




    





