import datetime
import os
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from misc.dataset import DataProcess, EmnistDataset
from misc.utils import Plot, accuracy, log
from nn.jax_chemical_rnn import JAXChemicalRNN
from options.complex_options import operatorEnum, yVectorEnum, zVectorEnum
from options.fast_rnn_options import fastRnnOptions
from options.jax_rnn_meat_learner_options import (
    JaxActivationNonLinearEnum,
    JaxRnnMetaLearnerOptions,
)
from options.meta_learner_options import chemicalEnum
from synapses.jax_fast_rnn import JAXFastRnn
from torch.utils.data import DataLoader, RandomSampler


class JaxMetaLearnerRNN:

    def __init__(
        self,
        modelOptions,
        jaxMetaLearnerOptions,
        key: jax.random.key,
        numberOfChemicals: int = 1,
        metaTrainingDataset: Optional[DataLoader] = None,
    ):
        self.jaxMetaLearnerOptions = jaxMetaLearnerOptions
        key1, key2 = jax.random.split(key)
        self.key1 = key1
        self.key2 = key2
        self.rnn = JAXChemicalRNN(
            input_size=self.jaxMetaLearnerOptions.input_size,
            key=key1,
            hidden_size=self.jaxMetaLearnerOptions.hidden_size,
            output_size=self.jaxMetaLearnerOptions.output_size,
            biological_min_tau=self.jaxMetaLearnerOptions.biological_min_tau,
            biological_max_tau=self.jaxMetaLearnerOptions.biological_max_tau,
            gradient=self.jaxMetaLearnerOptions.gradient,
            outer_activation=self.jaxMetaLearnerOptions.outer_activation,
            recurrent_activation=self.jaxMetaLearnerOptions.recurrent_activation,
        )
        self.save_results = self.jaxMetaLearnerOptions.save_results
        self.metaTrainingDataset = metaTrainingDataset
        self.data_process = DataProcess(
            minTrainingDataPerClass=self.jaxMetaLearnerOptions.minTrainingDataPerClass,
            maxTrainingDataPerClass=self.jaxMetaLearnerOptions.maxTrainingDataPerClass,
            queryDataPerClass=self.jaxMetaLearnerOptions.queryDataPerClass,
            dimensionOfImage=28,
            iid=True,
            use_jax=True,
        )

        self.numberOfChemicals = numberOfChemicals
        self.metaOptimizer = JAXFastRnn(numberOfChemicals, modelOptions)

        # -- optimizer --
        trainable_mask = self.get_trainable_mask(self.metaOptimizer)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Max norm of 1.0
            optax.adam(learning_rate=self.jaxMetaLearnerOptions.metaLearningRate),
        )
        dynamic, static = eqx.partition(self.metaOptimizer, trainable_mask)
        self.opt_state = self.optimizer.init(dynamic)
        self.metaOptimizer = eqx.combine(dynamic, static)

        # -- loss function --
        self.loss_function = optax.safe_softmax_cross_entropy

        # -- log params
        self.result_directory = os.getcwd() + "/results_2"
        if self.save_results:
            self.result_directory = os.getcwd() + "/results_2"
            os.makedirs(self.result_directory, exist_ok=True)
            self.result_directory += (
                "/"
                + self.jaxMetaLearnerOptions.results_subdir
                + "/"
                + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )
            os.makedirs(self.result_directory, exist_ok=False)
            with open(self.result_directory + "/arguments.txt", "w") as f:
                f.writelines(str(self.jaxMetaLearnerOptions))
                f.writelines(str(modelOptions))

            self.average_window = 10
            self.plot = Plot(self.result_directory, self.average_window)

    def chemicals_init(self):
        layers = self.rnn.layers
        self.synaptic_weights = []
        for i in layers:
            # self.synaptic_weights.append(jnp.empty((numberOfChemicals, i.in_features, i.out_features)))
            ## Initialize to xavier initialization
            if self.jaxMetaLearnerOptions.chemicalInitialization == chemicalEnum.same:
                temp = jax.nn.initializers.xavier_uniform()(self.key2, (i.in_features, i.out_features))
                holder = jnp.tile(temp, (self.numberOfChemicals, 1, 1))
            elif self.jaxMetaLearnerOptions.chemicalInitialization == chemicalEnum.different:
                holder = jax.nn.initializers.xavier_uniform()(
                    self.key2, (self.numberOfChemicals, i.in_features, i.out_features)
                )
            self.synaptic_weights.append(holder)
        self.synaptic_weights = tuple(self.synaptic_weights)

    def inner_loop_per_image(self, carry, input):
        synaptic_weights, parameters, rnn, hidden_state, metaOptimizer = carry
        x, labels = input
        y, hidden_state, activations_arr, errors_arr = rnn(x, hidden_state, labels)

        new_parameters = list(parameters)
        new_synaptic_weights = list(synaptic_weights)
        for idx, parameter in enumerate(parameters):
            synaptic_weight = synaptic_weights[idx]
            activations_tuple = activations_arr[idx]
            errors_tuple = errors_arr[idx]
            new_parameter, new_synaptic_weight = metaOptimizer(
                synaptic_weight, parameter, activations_tuple, errors_tuple
            )
            new_synaptic_weights[idx] = new_synaptic_weight
            new_parameters[idx] = new_parameter

        new_parameters_tuple = tuple(new_parameters)
        new_synaptic_weights_tuple = tuple(new_synaptic_weights)
        # new_synaptic_weights = tuple(new_synaptic_weights)
        new_rnn = eqx.tree_at(
            lambda r: (r.layers, r.forward1, r.forward2, r.forward3),
            rnn,
            (new_parameters_tuple, new_parameters_tuple[0], new_parameters_tuple[1], new_parameters_tuple[2]),
        )
        return (new_synaptic_weights_tuple, new_parameters_tuple, new_rnn, hidden_state, metaOptimizer), y

    def full_inner_loop(self, carry, input) -> jnp.ndarray:
        synaptic_weights, parameters, rnn, metaOptimizer = carry
        hidden_state = rnn.initialise_hidden_state(batch_size=1)
        x, label = input
        x = jnp.reshape(
            x, (self.jaxMetaLearnerOptions.number_of_time_steps, -1)
        )  # (time_steps, batch_size, input_size)

        label = jnp.repeat(
            label, repeats=self.jaxMetaLearnerOptions.number_of_time_steps, axis=0
        )  # (time_steps, batch_size)

        (new_synaptic_weights, new_parameters, new_rnn, hidden_state, metaOptimizer), y = jax.lax.scan(
            self.inner_loop_per_image,
            (synaptic_weights, parameters, rnn, hidden_state, metaOptimizer),
            (x, label),
        )
        return (new_synaptic_weights, new_parameters, new_rnn, metaOptimizer), y

    def inner_loop_per_image_no_update(self, carry, input):
        hidden_state, rnn = carry
        x = input
        y, hidden_state, _, _ = rnn(x, hidden_state, None)
        return (hidden_state, rnn), y

    def full_inner_loop_no_update(self, input) -> jnp.ndarray:
        x, number_of_timesteps, rnn = input
        hidden_state = rnn.initialise_hidden_state(batch_size=x.shape[0])
        x = jnp.reshape(x, (x.shape[0], number_of_timesteps, -1))  # (batch_size, time_steps, input_size)

        # -- use vmap to process all sequences in the batch --
        def run_sequence(hidden_state, rnn, x):
            return jax.lax.scan(
                self.inner_loop_per_image_no_update,
                (hidden_state, rnn),
                (x),
            )

        (_, _), y = jax.vmap(
            run_sequence,
            in_axes=(0, None, 0),
        )(hidden_state, rnn, x)
        return y

    # @eqx.filter_jit
    def compute_meta_loss(
        self,
        trainable_metaOptimizer,
        fixed_metaOptimizer,
        synaptic_weights,
        rnn_layers,
        rnn,
        x_trn,
        y_trn,
        x_qry,
        y_qry,
    ):

        metaOptimizer = eqx.combine(trainable_metaOptimizer, fixed_metaOptimizer)
        synaptic_weights_tuple = synaptic_weights
        parameters_tuple = rnn_layers
        rnn = rnn

        # -- inner loop --
        checkpointed_inner_loop = jax.checkpoint(self.full_inner_loop)
        carry_init = (
            synaptic_weights_tuple,
            parameters_tuple,
            rnn,
            metaOptimizer,
        )

        (new_synaptic_weights, new_parameters, new_rnn, _), _ = jax.lax.scan(
            checkpointed_inner_loop,
            carry_init,
            (x_trn, y_trn),
        )

        qry_predictions = self.full_inner_loop_no_update(
            (x_qry, self.jaxMetaLearnerOptions.number_of_time_steps, new_rnn)
        )
        qry_predictions = qry_predictions[:, -1, :]  # take last time step
        one_hot_targets = jax.nn.one_hot(y_qry.flatten(), num_classes=self.jaxMetaLearnerOptions.output_size)
        losses = self.loss_function(qry_predictions, one_hot_targets)
        avg_loss = jnp.mean(losses)
        acc = accuracy(qry_predictions, y_qry.ravel(), use_jax=True)
        return avg_loss, acc

    def get_trainable_mask(self, model):

        mask = jax.tree_util.tree_map(lambda _: False, model)

        mask = eqx.tree_at(lambda m: m.Q_matrix, mask, True)
        mask = eqx.tree_at(lambda m: m.K_matrix, mask, True)
        mask = eqx.tree_at(lambda m: m.v_vector, mask, True)

        return mask

    @eqx.filter_jit
    def make_step(
        self,
        dynamic_model,
        static_model,
        synaptic_weights_tuple,
        rnn_layers,
        rnn,
        x_trn,
        y_trn,
        x_qry,
        y_qry,
        opt_state,
    ):

        (loss, acc), grads = jax.value_and_grad(self.compute_meta_loss, has_aux=True)(
            dynamic_model, static_model, synaptic_weights_tuple, rnn_layers, rnn, x_trn, y_trn, x_qry, y_qry
        )

        grads_filtered = eqx.filter(grads, eqx.is_array)
        grad_norm = optax.global_norm(grads_filtered)

        updates, new_opt_state = self.optimizer.update(grads_filtered, opt_state, dynamic_model)
        new_dynamic_model = optax.apply_updates(dynamic_model, updates)

        return new_dynamic_model, new_opt_state, loss, acc, grad_norm

    def train(self):
        current_training_data_per_class = None
        for eps, data in enumerate(self.metaTrainingDataset):
            x_trn, y_trn, x_qry, y_qry, current_training_data_per_class = self.data_process(
                data, self.jaxMetaLearnerOptions.numberOfClasses
            )
            # -- convert to jax arrays --
            x_trn = jnp.array(x_trn)
            y_trn = jnp.array(y_trn)
            x_qry = jnp.array(x_qry)
            y_qry = jnp.array(y_qry)

            # -- weight initialization --
            self.rnn = self.rnn.reset_weights(self.key2)

            # -- initialize chemicals --
            self.chemicals_init()
            new_parameters = list(self.rnn.layers)
            new_synaptic_weights = list(self.synaptic_weights)
            for idx, parameter in enumerate(self.rnn.layers):
                synaptic_weight = self.synaptic_weights[idx]
                new_synaptic_weight, new_parameter = self.metaOptimizer.initialize_parameters(
                    synaptic_weight, parameter
                )
                new_synaptic_weights[idx] = new_synaptic_weight
                new_parameters[idx] = new_parameter
            self.synaptic_weights = tuple(new_synaptic_weights)
            self.new_parameters = tuple(new_parameters)
            self.rnn = eqx.tree_at(
                lambda r: (r.layers, r.forward1, r.forward2, r.forward3),
                self.rnn,
                (self.new_parameters, self.new_parameters[0], self.new_parameters[1], self.new_parameters[2]),
            )

            # -- meta-optimization --
            trainable_mask = self.get_trainable_mask(self.metaOptimizer)
            dynamic_model, static_model = eqx.partition(self.metaOptimizer, trainable_mask)
            new_dynamic_model, self.opt_state, loss, acc, grad_norm = self.make_step(
                dynamic_model,
                static_model,
                self.synaptic_weights,
                self.rnn.layers,
                self.rnn,
                x_trn,
                y_trn,
                x_qry,
                y_qry,
                self.opt_state,
            )

            self.metaOptimizer = eqx.combine(new_dynamic_model, static_model)

            if self.save_results:
                log([loss.item()], self.result_directory + "/loss_meta.txt")
                log([acc], self.result_directory + "/acc_meta.txt")

            line = "Train Episode: {}\tLoss: {:.6f}\tAccuracy: {:.3f}\tGrad Norm: {:.6f}\tTraining/Class: {}".format(
                eps + 1, loss.item(), acc, grad_norm, current_training_data_per_class
            )
            if self.jaxMetaLearnerOptions.display:
                print(line)

            if self.save_results:

                with open(self.result_directory + "/params.txt", "a") as f:
                    f.writelines(line + "\n")

                self.metaOptimizer.Q_matrix.block_until_ready()
                if eps % 10 == 0:
                    with open(self.result_directory + "/{}.txt".format("Q_matrix"), "a") as f:
                        f.writelines("Episode: {}: {} \n".format(eps + 1, np.array(self.metaOptimizer.Q_matrix)))
                    with open(self.result_directory + "/{}.txt".format("K_matrix"), "a") as f:
                        f.writelines("Episode: {}: {} \n".format(eps + 1, np.array(self.metaOptimizer.K_matrix)))
                    with open(self.result_directory + "/{}.txt".format("z_vector"), "a") as f:
                        f.writelines("Episode: {}: {} \n".format(eps + 1, np.array(self.metaOptimizer.z_vector)))
                    with open(self.result_directory + "/{}.txt".format("y_vector"), "a") as f:
                        f.writelines("Episode: {}: {} \n".format(eps + 1, np.array(self.metaOptimizer.y_vector)))
                    with open(self.result_directory + "/{}.txt".format("v_vector"), "a") as f:
                        f.writelines("Episode: {}: {} \n".format(eps + 1, np.array(self.metaOptimizer.v_vector)))
            if jnp.isnan(loss):
                raise ValueError("Loss is NaN")

        if self.save_results:
            self.plot()
            # -- save model --
            eqx.tree_serialise_leaves(self.result_directory + "/meta_learner_model.eqx", self.metaOptimizer)
            # -- save optimizer state --
            eqx.tree_serialise_leaves(self.result_directory + "/meta_learner_optimizer.eqx", self.opt_state)

        print("Training completed.")


def main_jax_rnn_meta_learner():
    key = jax.random.PRNGKey(42)
    # jax.config.update("jax_enable_x64", False)

    # -- load data
    numWorkers = 2
    epochs = 5000

    dataset_name = "EMNIST"
    minTrainingDataPerClass = 5
    maxTrainingDataPerClass = 70
    queryDataPerClass = 10

    if dataset_name == "EMNIST":
        numberOfClasses = 5
        dataset = EmnistDataset(
            minTrainingDataPerClass=minTrainingDataPerClass,
            maxTrainingDataPerClass=maxTrainingDataPerClass,
            queryDataPerClass=queryDataPerClass,
            dimensionOfImage=28,
            use_jax=True,
        )
        dimOut = 47

    sampler = RandomSampler(data_source=dataset, replacement=True, num_samples=epochs * numberOfClasses)
    metatrain_dataset = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=numberOfClasses,
        drop_last=True,
        num_workers=numWorkers,
        persistent_workers=True,
    )

    # -- options
    modelOptions = None
    modelOptions = fastRnnOptions(
        nonLinear=JaxActivationNonLinearEnum.tanh,
        update_rules=[0, 1, 2, 4, 9, 11],  # 4
        minSlowTau=2,
        maxSlowTau=50,
        y_vector=yVectorEnum.none,
        z_vector=zVectorEnum.default,
        operator=operatorEnum.mode_9,
    )
    # cuda:1
    # device = "cpu"
    current_dir = os.getcwd()
    continue_training = current_dir + "/results_2/post_cosyne_rnn_check_mode_9/0/20251206-005104"
    # -- meta-learner options
    metaLearnerOptions = JaxRnnMetaLearnerOptions(
        seed=42,
        save_results=True,
        results_subdir="jax_rnn_6_grad",
        metatrain_dataset="emnist",
        display=True,
        metaLearningRate=0.0001,
        numberOfClasses=numberOfClasses,
        dataset_name=dataset_name,
        chemicalInitialization=chemicalEnum.same,
        minTrainingDataPerClass=minTrainingDataPerClass,
        maxTrainingDataPerClass=maxTrainingDataPerClass,
        queryDataPerClass=queryDataPerClass,
        input_size=int(28 * 28 / 7),
        hidden_size=128,
        output_size=dimOut,
        biological_min_tau=1,
        biological_max_tau=7,
        gradient=True,
        outer_activation=JaxActivationNonLinearEnum.tanh,
        recurrent_activation=JaxActivationNonLinearEnum.softplus,
        number_of_time_steps=7,
    )

    metalearning_model = JaxMetaLearnerRNN(
        modelOptions=modelOptions,
        jaxMetaLearnerOptions=metaLearnerOptions,
        key=key,
        numberOfChemicals=5,
        metaTrainingDataset=metatrain_dataset,
    )

    metalearning_model.train()
