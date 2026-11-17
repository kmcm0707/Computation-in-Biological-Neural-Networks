import os
from pathlib import Path
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from misc.jax_data import DataProcess, EmnistDataset, emnist_batches
from misc.jax_utils import Plot, accuracy, log
from nn.jax_chemical_nn import JAXFeedforwardNN
from options.complex_options import operatorEnum, yVectorEnum, zVectorEnum
from options.fast_rnn_options import fastRnnOptions
from options.jax_rnn_meat_learner_options import (
    JaxActivationNonLinearEnum,
    JaxErrorTypeEnum,
    JaxRnnMetaLearnerOptions,
)
from synapses.jax_fast_rnn import JAXFastRnn

DATASET_NAME = "emnist"
# Save the sampled complex and backpropagation updates during normal runs.
# Set CBNN_SAVE_WEIGHT_DIAGNOSTICS=0 to disable these larger diagnostic files.
SAVE_WEIGHT_DIAGNOSTICS = os.environ.get("CBNN_SAVE_WEIGHT_DIAGNOSTICS", "1") == "1"


def save_npy(path: str, array: np.ndarray) -> None:
    """Write a NumPy array through a binary stream on Windows and POSIX."""
    with open(path, "wb") as stream:
        np.save(stream, array)


def save_weight_diagnostics(result_directory: str, episode: int, complex_history, bp_history) -> None:
    """Save sampled per-layer updates in the padded format used by alignment_angle_bp.py."""
    def as_layer_arrays(history):
        arrays = []
        for layer_history in history:
            array = np.asarray(layer_history)
            # The runner currently evaluates one task per episode. The task
            # axis is retained by the outer scan, so remove it here.
            if array.ndim == 4 and array.shape[0] == 1:
                array = array[0]
            if array.ndim != 3:
                raise ValueError(
                    f"Expected sampled updates shaped (samples, inputs, outputs), got {array.shape}"
                )
            arrays.append(array)
        return arrays

    histories = (as_layer_arrays(complex_history), as_layer_arrays(bp_history))
    if len(histories[0]) != len(histories[1]):
        raise ValueError("Complex and BP histories have different layer counts")
    layer_shapes = np.asarray([array.shape[1:] for array in histories[0]], dtype=np.int32)
    if not np.array_equal(layer_shapes, np.asarray([array.shape[1:] for array in histories[1]])):
        raise ValueError("Complex and BP layer shapes differ")
    if any(array.shape[0] != histories[0][0].shape[0] for arrays in histories for array in arrays):
        raise ValueError("Sample counts differ between saved layers")

    max_rows = int(layer_shapes[:, 0].max())
    max_cols = int(layer_shapes[:, 1].max())
    sample_count = histories[0][0].shape[0]
    for kind, arrays in zip(("weight_changes", "bp_weight_changes"), histories):
        dtype = np.result_type(*(array.dtype for array in arrays))
        padded = np.zeros((len(arrays), sample_count, max_rows, max_cols), dtype=dtype)
        for layer, array in enumerate(arrays):
            rows, cols = array.shape[1:]
            padded[layer, :, :rows, :cols] = array
        save_npy(os.path.join(result_directory, f"{kind}_episode_{episode:04d}.npy"), padded)
        save_npy(
            os.path.join(result_directory, f"{kind}_episode_{episode:04d}_layer_shapes.npy"),
            layer_shapes,
        )


def compute_bp_delta_w(rnn, x_image, y_label, learning_rate: float = 0.01):
    """
    Computes the weight change matrix Delta W that standard Backpropagation
    would prescribe for all layers given the current image and label.
    """
    num_classes = rnn.forward5.out_features
    if y_label.ndim > 0 and y_label.shape[-1] == num_classes:
        y_one_hot = y_label
    else:
        y_one_hot = jax.nn.one_hot(y_label, num_classes=num_classes)

    def loss_fn(model):
        # Use logits directly; this bypasses the network's feedback error path.
        y_pred, _, _, _, _ = model(x_image, None, None)
        return jnp.sum(optax.safe_softmax_cross_entropy(y_pred, y_one_hot))

    grads = eqx.filter_grad(loss_fn)(rnn)

    forward_layers = (
        grads.forward1,
        grads.forward2,
        grads.forward3,
        grads.forward4,
        grads.forward5,
    )
    # Saved matrices use (input, output), matching the chemical update history.
    return tuple(-learning_rate * layer.weight.T for layer in forward_layers)


class JaxMetaLearnerRNN:

    def __init__(
        self,
        modelOptions,
        jaxMetaLearnerOptions,
        key: jax.random.key,
        numberOfChemicals: int = 1,
        metaTrainingDataset: Optional[EmnistDataset] = None,
        tracked_layer: int = 2,
        tracked_neuron: int = 40,
        tracked_input_synapse: int = 8,
        weight_sample_blocksize: int = 3,
        bp_learning_rate: float = 0.01,
        freeze_after_step: int = 50,
    ):
        self.jaxMetaLearnerOptions = jaxMetaLearnerOptions
        self.freeze_after_step = freeze_after_step
        key1, key2 = jax.random.split(key)
        self.key1 = key1
        self.key2 = key2
        self.rnn = JAXFeedforwardNN(
            dim_out=self.jaxMetaLearnerOptions.output_size,
            key=key1,
            input_size=self.jaxMetaLearnerOptions.input_size,
            gradient=self.jaxMetaLearnerOptions.gradient,
            activation=JaxActivationNonLinearEnum.softplus,
            error_type=self.jaxMetaLearnerOptions.error_type,
            low_dim_DFA=self.jaxMetaLearnerOptions.low_dim_DFA,
        )
        self.save_results = self.jaxMetaLearnerOptions.save_results
        self.metaTrainingDataset = metaTrainingDataset
        # Zero-based indexes into the forward layers and their output neurons.
        self.tracked_layer = tracked_layer
        self.tracked_neuron = tracked_neuron
        self.tracked_input_synapse = tracked_input_synapse
        self.weight_sample_blocksize = weight_sample_blocksize
        self.bp_learning_rate = bp_learning_rate
        self.data_process = DataProcess(
            minTrainingDataPerClass=self.jaxMetaLearnerOptions.minTrainingDataPerClass,
            maxTrainingDataPerClass=self.jaxMetaLearnerOptions.maxTrainingDataPerClass,
            queryDataPerClass=self.jaxMetaLearnerOptions.queryDataPerClass,
            dimensionOfImage=28,
            iid=True,
            use_jax=True,
            permutation=self.jaxMetaLearnerOptions.permutation,
        )

        self.numberOfChemicals = numberOfChemicals
        self.metaOptimizer = JAXFastRnn(numberOfChemicals, modelOptions)
        # -- load model --
        self.metaOptimizer = eqx.tree_deserialise_leaves(
            self.jaxMetaLearnerOptions.load_model + "/meta_learner_model.eqx",
            self.metaOptimizer,
        )

        # -- loss function --
        self.loss_function = optax.safe_softmax_cross_entropy

        # -- log params --
        self.result_directory = str(Path(__file__).resolve().parents[2] / "results_4")
        if self.save_results:
            os.makedirs(self.result_directory, exist_ok=True)
            self.result_directory += (
                "/"
                + self.jaxMetaLearnerOptions.results_subdir
                + "/"
                + str(self.jaxMetaLearnerOptions.minTrainingDataPerClass)
            )
            os.makedirs(self.result_directory, exist_ok=True)
            with open(self.result_directory + "/arguments.txt", "w") as f:
                f.writelines(str(self.jaxMetaLearnerOptions))
                f.writelines(str(modelOptions))
            self.accuracy_by_episode_path = os.path.join(self.result_directory, "accuracy_by_episode.txt")
            with open(self.accuracy_by_episode_path, "w", encoding="utf-8") as f:
                f.write("episode\taccuracy\n")
            self.average_window = 10
            self.plot = Plot(self.result_directory, self.average_window)

    @staticmethod
    def make_sample_list(numtrainingimages, blocksize):
        """Return zero-based image indexes at which weight snapshots are taken."""
        if numtrainingimages < 0 or blocksize <= 0:
            raise ValueError("numtrainingimages must be non-negative and blocksize positive")
        return list(range(0, numtrainingimages, blocksize))

    def chemicals_init(self):
        layers = self.rnn.layers
        self.synaptic_weights = []
        for i in layers:
            if self.jaxMetaLearnerOptions.chemicalInitialization == "same":
                temp = jax.nn.initializers.xavier_uniform()(self.key2, (i.in_features, i.out_features))
                holder = jnp.tile(temp, (self.numberOfChemicals, 1, 1))
            elif self.jaxMetaLearnerOptions.chemicalInitialization == "different":
                holder = jax.nn.initializers.xavier_uniform()(
                    self.key2, (self.numberOfChemicals, i.in_features, i.out_features)
                )
            else:
                raise ValueError(
                    f"Unknown chemicalInitialization: {self.jaxMetaLearnerOptions.chemicalInitialization}"
                )
            self.synaptic_weights.append(holder)
        self.synaptic_weights = tuple(self.synaptic_weights)

    def inner_loop_per_image(self, carry, input):
        synaptic_weights, parameters, rnn, hidden_state, metaOptimizer, step_idx, sample_slot = carry
        x, labels = input

        labels = jax.nn.one_hot(labels, num_classes=self.jaxMetaLearnerOptions.output_size)

        y, hidden_state, activations_arr, errors_arr, scaler_error, correct_class_probability = rnn(
            x, hidden_state, labels
        )

        def update_layer(w, p, act, err):
            return metaOptimizer(w, p, act, err)

        activations_arr = tuple(activations_arr)
        errors_arr = tuple(errors_arr)

        results = jax.tree_util.tree_map(update_layer, synaptic_weights, parameters, activations_arr, errors_arr)
        computed_parameters = tuple(res[0] for res in results)
        computed_synaptic_weights = tuple(res[1] for res in results)

        # --- FREEZING LOGIC ---
        # Freeze early layers (all except final readout layer) once `freeze_after_step` images are processed
        should_freeze = step_idx >= self.freeze_after_step

        def apply_freeze(old_tuple, new_tuple):
            frozen_prefix = jax.lax.cond(
                should_freeze,
                lambda: old_tuple[:-1],
                lambda: new_tuple[:-1],
            )
            return frozen_prefix + (new_tuple[-1],)

        new_parameters = apply_freeze(parameters, computed_parameters)
        new_synaptic_weights = apply_freeze(synaptic_weights, computed_synaptic_weights)
        # ----------------------

        num_layers = len(new_parameters)

        new_rnn = eqx.tree_at(
            lambda r: (r.layers, *(getattr(r, f"forward{i+1}") for i in range(num_layers))),
            rnn,
            (new_parameters, *new_parameters),
        )

        tracked_chemicals = new_synaptic_weights[self.tracked_layer][
            :, self.tracked_input_synapse, self.tracked_neuron
        ]

        is_sample_step = jnp.any(step_idx == self.weight_sample_indices)
        sampled_weight_changes = tuple(
            jax.lax.cond(
                is_sample_step,
                lambda new=new_parameter, old=old_parameter: new.weight.T - old.weight.T,
                lambda new=new_parameter: jnp.zeros_like(new.weight.T),
            )
            for new_parameter, old_parameter in zip(new_parameters, parameters)
        )
        sampled_bp_weight_changes = jax.lax.cond(
            is_sample_step,
            lambda: compute_bp_delta_w(rnn, x, labels, learning_rate=self.bp_learning_rate),
            lambda: tuple(jnp.zeros_like(parameter.weight.T) for parameter in parameters),
        )
        sample_slot = sample_slot + is_sample_step.astype(jnp.int32)
        new_carry = (
            new_synaptic_weights,
            new_parameters,
            new_rnn,
            hidden_state,
            metaOptimizer,
            step_idx + 1,
            sample_slot,
        )
        return new_carry, (
            y,
            tracked_chemicals,
            scaler_error,
            correct_class_probability,
            sampled_weight_changes,
            sampled_bp_weight_changes,
        )

    def full_inner_loop(self, carry, input):
        synaptic_weights, parameters, rnn, metaOptimizer = carry
        x, label = input

        number_of_time_steps = self.jaxMetaLearnerOptions.number_of_time_steps
        x = jnp.reshape(x, (-1, number_of_time_steps, self.jaxMetaLearnerOptions.input_size))
        label = jnp.reshape(label, (-1,))
        x = jnp.reshape(x, (-1, self.jaxMetaLearnerOptions.input_size))
        label = jnp.repeat(label, number_of_time_steps)

        sample_indices = jnp.asarray(
            self.make_sample_list(x.shape[0], self.weight_sample_blocksize), dtype=jnp.int32
        )
        self.weight_sample_indices = sample_indices

        step_idx_init = jnp.array(0, dtype=jnp.int32)
        sample_slot_init = jnp.array(0, dtype=jnp.int32)

        (new_synaptic_weights, new_parameters, new_rnn, hidden_state, metaOptimizer, _, _), (
            y,
            tracked_chemicals,
            scaler_error,
            correct_class_probability,
            weight_history,
            bp_weight_history,
        ) = jax.lax.scan(
            self.inner_loop_per_image,
            (synaptic_weights, parameters, rnn, None, metaOptimizer, step_idx_init, sample_slot_init),
            (x, label),
        )

        weight_changes = jax.tree_util.tree_map(
            lambda history: history[sample_indices], weight_history
        )
        bp_weight_changes = jax.tree_util.tree_map(
            lambda history: history[sample_indices], bp_weight_history
        )
        return (
            new_synaptic_weights,
            new_parameters,
            new_rnn,
            metaOptimizer,
        ), (y, tracked_chemicals, scaler_error, correct_class_probability, weight_changes, bp_weight_changes)

    def inner_loop_per_image_no_update(self, carry, input):
        hidden_state, rnn = carry
        x = input
        y, hidden_state, _, _, _, _ = rnn(x, hidden_state, None)
        return (hidden_state, rnn), y

    def full_inner_loop_no_update(self, x, number_of_timesteps, rnn):
        input_size = self.jaxMetaLearnerOptions.input_size
        x = jnp.reshape(x, (x.shape[0], x.shape[1], number_of_timesteps, input_size))
        predict = lambda sequence: jax.vmap(
            lambda steps: jax.vmap(lambda step: rnn(step, None, None)[0])(steps)
        )(sequence)
        return jax.vmap(predict)(x)

    def compute_meta_loss(
        self,
        metaOptimizer,
        synaptic_weights,
        rnn_layers,
        rnn,
        x_trn,
        y_trn,
        x_qry,
        y_qry,
    ):
        carry_init = (
            synaptic_weights,
            rnn_layers,
            rnn,
            metaOptimizer,
        )

        (new_synaptic_weights, new_parameters, new_rnn, _), (
            y,
            tracked_chemicals,
            scaler_error,
            correct_class_probability,
            weight_changes,
            bp_weight_changes,
        ) = jax.lax.scan(
            self.full_inner_loop,
            carry_init,
            (x_trn, y_trn),
        )

        qry_predictions = self.full_inner_loop_no_update(
            x_qry, self.jaxMetaLearnerOptions.number_of_time_steps, new_rnn
        )

        qry_predictions = qry_predictions[:, :, -1, :]
        qry_predictions = qry_predictions.reshape((-1, self.jaxMetaLearnerOptions.output_size))
        query_labels = y_qry.reshape((-1,))
        one_hot_targets = jax.nn.one_hot(query_labels, num_classes=self.jaxMetaLearnerOptions.output_size)
        losses = self.loss_function(qry_predictions, one_hot_targets)
        avg_loss = jnp.mean(losses)
        acc = accuracy(qry_predictions, query_labels, use_jax=True)

        return (
            avg_loss,
            acc,
            tracked_chemicals,
            scaler_error,
            correct_class_probability,
            weight_changes,
            bp_weight_changes,
        )

    @eqx.filter_jit
    def make_step(
        self,
        metaOptimizer,
        synaptic_weights_tuple,
        rnn_layers,
        rnn,
        x_trn,
        y_trn,
        x_qry,
        y_qry,
    ):
        return self.compute_meta_loss(
            metaOptimizer, synaptic_weights_tuple, rnn_layers, rnn, x_trn, y_trn, x_qry, y_qry
        )

    def train(self):
        for eps, data in enumerate(self.metaTrainingDataset):
            x_trn, y_trn, x_qry, y_qry, current_training_data_per_class, _ = self.data_process(
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
            num_layers = len(self.new_parameters)
            self.rnn = eqx.tree_at(
                lambda r: (r.layers, *(getattr(r, f"forward{i+1}") for i in range(num_layers))),
                self.rnn,
                (self.new_parameters, *self.new_parameters),
            )

            # -- meta-optimization / inner loop evaluation --
            (
                loss,
                acc,
                tracked_chemicals,
                scaler_error,
                correct_class_probability,
                weight_changes,
                bp_weight_changes,
            ) = self.make_step(
                self.metaOptimizer,
                self.synaptic_weights,
                self.rnn.layers,
                self.rnn,
                x_trn,
                y_trn,
                x_qry,
                y_qry,
            )

            if self.save_results:
                log([loss.item()], self.result_directory + "/loss_meta.txt")
                log([acc], self.result_directory + "/acc_meta.txt")
                with open(self.accuracy_by_episode_path, "a", encoding="utf-8") as f:
                    f.write(f"{eps + 1}\t{float(acc):.6f}\n")
                self.save_neuron_chemical_plot(
                    tracked_chemicals, scaler_error, correct_class_probability, eps + 1
                )
                if SAVE_WEIGHT_DIAGNOSTICS:
                    save_weight_diagnostics(
                        self.result_directory, eps + 1, weight_changes, bp_weight_changes
                    )
            self.weight_changes = tuple(layer_history[0] for layer_history in weight_changes)

            line = "Runner Episode: {}\tLoss: {:.6f}\tAccuracy: {:.3f}\tTraining/Class: {}".format(
                eps + 1, loss.item(), acc, current_training_data_per_class
            )
            if self.jaxMetaLearnerOptions.display:
                print(line, flush=True)

            if self.save_results:
                with open(self.result_directory + "/params.txt", "a") as f:
                    f.writelines(line + "\n")

        if self.save_results:
            self.plot()
            self.plot_episode_accuracy()
            # -- save model --
            eqx.tree_serialise_leaves(
                self.result_directory + "/meta_learner_model.eqx", self.metaOptimizer
            )

        print("Training completed.", flush=True)

    def plot_episode_accuracy(self):
        """Plot the recorded query accuracy for each training episode."""
        values = np.loadtxt(self.accuracy_by_episode_path, skiprows=1, usecols=1, ndmin=1)
        episodes = np.arange(1, len(values) + 1)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(episodes, values, marker="o", linewidth=1.8)
        ax.set_title(f"Accuracy by Episode ({len(values)} episodes)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Query accuracy")
        ax.set_xticks(episodes)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.result_directory, "accuracy_by_episode.png"), dpi=150)
        plt.close(fig)

    def save_neuron_chemical_plot(
        self, chemical_history, scaler_error_history, correct_class_probability, episode: int
    ):
        """Plot each chemical on the selected input-to-neuron synapse over images."""
        history = np.asarray(chemical_history)
        if history.ndim == 3:
            history = history[0]
        scaler_error = np.asarray(scaler_error_history)
        if scaler_error.ndim == 3:
            scaler_error = scaler_error[0]
        scaler_error = np.squeeze(scaler_error)
        correct_class_probability = np.asarray(correct_class_probability)
        if correct_class_probability.ndim == 3:
            correct_class_probability = correct_class_probability[0]
        correct_class_probability = np.squeeze(correct_class_probability)
        if scaler_error.ndim != 1 or scaler_error.shape[0] != history.shape[0]:
            raise ValueError(
                f"Expected one scaler error per training image ({history.shape[0]}), got {scaler_error.shape}"
            )
        if history.ndim != 2 or history.shape[-1] != self.numberOfChemicals:
            raise ValueError(f"Expected (images, chemicals), got {history.shape}")

        layer = self.rnn.layers[self.tracked_layer]
        if not 0 <= self.tracked_input_synapse < layer.in_features:
            raise ValueError(
                f"Incoming synapse {self.tracked_input_synapse} is out of range "
                f"for layer {self.tracked_layer + 1} with {layer.in_features} inputs"
            )
        if not 0 <= self.tracked_neuron < layer.out_features:
            raise ValueError(
                f"Tracked neuron {self.tracked_neuron} is out of range "
                f"for layer {self.tracked_layer + 1} with {layer.out_features} neurons "
                f"(valid zero-based indexes: 0 to {layer.out_features - 1})"
            )

        csv_path = os.path.join(
            self.result_directory,
            f"neuron_synapse_chemicals_episode_{episode:04d}.csv",
        )
        with open(csv_path, "w", newline="") as csv_file:
            csv_file.write(
                "training_image,input_synapse,output_neuron,chemical,value,scaler_error,correct_class_probability\n"
            )
            for image_index in range(history.shape[0]):
                for chemical_index in range(history.shape[1]):
                    csv_file.write(
                        f"{image_index + 1},{self.tracked_input_synapse},{self.tracked_neuron},"
                        f"{chemical_index},{history[image_index, chemical_index]:.10g},"
                        f"{int(scaler_error[image_index])},{correct_class_probability[image_index]:.10g}\n"
                    )
        images = np.arange(1, history.shape[0] + 1)
        hits = np.flatnonzero(scaler_error == 1)
        fig, ax = plt.subplots(figsize=(11, 6))
        for chemical in range(history.shape[1]):
            ax.plot(images, history[:, chemical], label=f"Chemical {chemical}")
        if hits.size:
            for hit_index, image_index in enumerate(hits):
                ax.axvline(
                    int(image_index) + 1,
                    color="red",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.7,
                    label="Scaler error = 1" if hit_index == 0 else None,
                )
        ax.set_title(
            f"Layer {self.tracked_layer} (zero-based), neuron {self.tracked_neuron}, "
            f"input synapse {self.tracked_input_synapse}: chemical values "
            f"after each training image (episode {episode})"
        )
        ax.set_xlabel("Training image")
        ax.set_ylabel("Synaptic chemical value")
        ax.set_xlim(1, len(images))
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(
            os.path.join(self.result_directory, f"neuron_synapse_chemicals_episode_{episode:04d}.png"),
            dpi=150,
        )
        plt.close(fig)


def jax_runner(index: int, result_subdirectory: str, modelPath: str):
    key = jax.random.PRNGKey(42)

    # -- load data
    training_data = [20, 30, 40, 50, 80, 150, 200, 225, 375]
    if index >= len(training_data):
        return
    epochs = 20

    minTrainingDataPerClass = 70
    maxTrainingDataPerClass = 70
    queryDataPerClass = 20
    numberOfTimeSteps = 1

    numberOfClasses = 5
    dataset = EmnistDataset(
        minTrainingDataPerClass=minTrainingDataPerClass,
        maxTrainingDataPerClass=maxTrainingDataPerClass,
        queryDataPerClass=queryDataPerClass,
        dimensionOfImage=28,
    )
    dimOut = 47
    dimIn = int(28 * 28 / numberOfTimeSteps)

    metatrain_dataset = emnist_batches(dataset, epochs, numberOfClasses, numberOfClasses)

    # -- options
    modelOptions = fastRnnOptions(
        nonLinear=JaxActivationNonLinearEnum.tanh,
        update_rules=[0, 1, 2, 3, 4, 9, 12],
        minSlowTau=2,
        maxSlowTau=50,
        y_vector=yVectorEnum.none,
        z_vector=zVectorEnum.default,
        operator=operatorEnum.mode_9,
    )
    # -- meta-learner options
    metaLearnerOptions = JaxRnnMetaLearnerOptions(
        seed=42,
        save_results=True,
        results_subdir=result_subdirectory,
        metatrain_dataset=DATASET_NAME,
        display=True,
        metaLearningRate=None,
        numberOfClasses=numberOfClasses,
        dataset_name=DATASET_NAME,
        chemicalInitialization="different",
        minTrainingDataPerClass=minTrainingDataPerClass,
        maxTrainingDataPerClass=maxTrainingDataPerClass,
        queryDataPerClass=queryDataPerClass,
        input_size=dimIn,
        hidden_size=128,
        output_size=dimOut,
        biological_min_tau=1,
        biological_max_tau=28,
        gradient=True,
        outer_activation=JaxActivationNonLinearEnum.tanh,
        recurrent_activation=JaxActivationNonLinearEnum.softplus,
        number_of_time_steps=numberOfTimeSteps,
        load_model=modelPath,
        error_type=JaxErrorTypeEnum.DSEF,
        low_dim_DFA=-1,
        permutation=False,
        two_layer_RNN=False,
        feedforward=True,
    )

    metalearning_model = JaxMetaLearnerRNN(
        modelOptions=modelOptions,
        jaxMetaLearnerOptions=metaLearnerOptions,
        key=key,
        numberOfChemicals=13,
        metaTrainingDataset=metatrain_dataset,
        weight_sample_blocksize=3,
        #!!THIS is what you change!
        # step_idx is int32; use its maximum as the "never freeze" threshold.
        freeze_after_step=2**31 - 1,
        #freeze_after_step=250,
    )

    metalearning_model.train()


def main_jax_runner():
    repo_root = Path(__file__).resolve().parents[2]
    model_path = repo_root / "results_4" / "Jax_13_chem_DSEF_full_sweep_200" / "1" / "400"
    checkpoint = model_path / "meta_learner_model.eqx"
    dataset_path = repo_root / "data" / "emnist"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Meta-learner checkpoint not found: {checkpoint}")
    if not dataset_path.is_dir() or not any(dataset_path.iterdir()):
        raise FileNotFoundError(f"EMNIST dataset directory is missing or empty: {dataset_path}")
    jax_runner(
        index=0,
        result_subdirectory="inner_loop",
        modelPath=str(model_path),
    )
