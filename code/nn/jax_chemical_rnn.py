from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from options.jax_rnn_meat_learner_options import JaxActivationNonLinearEnum


def beta_softplus(x: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
    return (1.0 / beta) * jnp.log1p(jnp.exp(beta * x))


class JAXChemicalRNN(eqx.Module):
    forward1: eqx.nn.Linear
    forward2: eqx.nn.Linear
    forward3: eqx.nn.Linear
    pre_feedback1: eqx.nn.Linear
    pre_feedback2: eqx.nn.Linear
    recurrent_feedback: eqx.nn.Linear
    layers: tuple[eqx.nn.Linear]
    feedback_layers: tuple[eqx.nn.Linear]
    gradient: bool = eqx.field(static=True)
    outer_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = eqx.field(static=True)
    recurrent_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = eqx.field(static=True)
    beta: float
    tau: jnp.ndarray
    softplus: Callable[[jnp.ndarray, float], jnp.ndarray] = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        key: jax.random.PRNGKey,
        biological_min_tau: int = 1,
        biological_max_tau: int = 56,
        gradient: bool = False,
        outer_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        recurrent_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ):
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)
        self.forward1 = eqx.nn.Linear(input_size, hidden_size, key=key1, use_bias=False)
        self.forward2 = eqx.nn.Linear(hidden_size, hidden_size, key=key2, use_bias=False)
        self.forward3 = eqx.nn.Linear(hidden_size, output_size, key=key3, use_bias=False)
        self.layers = (self.forward1, self.forward2, self.forward3)

        self.pre_feedback1 = eqx.nn.Linear(output_size, input_size, key=key4, use_bias=False)
        self.pre_feedback2 = eqx.nn.Linear(output_size, hidden_size, key=key5, use_bias=False)
        self.recurrent_feedback = eqx.nn.Linear(output_size, hidden_size, key=key6, use_bias=False)
        self.feedback_layers = (self.pre_feedback1, self.pre_feedback2, self.recurrent_feedback)

        self.gradient = gradient
        self.outer_activation = outer_activation
        if self.outer_activation is JaxActivationNonLinearEnum.softplus:
            self.outer_activation = beta_softplus

        self.recurrent_activation = recurrent_activation
        if self.recurrent_activation is JaxActivationNonLinearEnum.softplus:
            self.recurrent_activation = beta_softplus
        self.beta = 10.0
        self.softplus = beta_softplus

        # -- initialize tau vector -- #TODO: CHECK IF THIS IS CORRECT
        base = biological_max_tau / biological_min_tau
        tau = biological_min_tau * jnp.power(base, jnp.linspace(0, 1, hidden_size))
        # tau = jnp.linspace(biological_min_tau, biological_max_tau, hidden_size)
        self.tau = tau

    def reset_feedback_weights(self, key: jax.random.PRNGKey) -> "JAXChemicalRNN":
        new_pre_feedback1 = eqx.nn.Linear(
            self.pre_feedback1.in_features, self.pre_feedback1.out_features, key=key, use_bias=False
        )
        # new_pre_feedback1 = jax.lax.stop_gradient(new_pre_feedback1.weight)
        new_self = eqx.tree_at(lambda r: r.pre_feedback1, self, new_pre_feedback1)

        new_pre_feedback2 = eqx.nn.Linear(
            self.pre_feedback2.in_features, self.pre_feedback2.out_features, key=key, use_bias=False
        )
        # new_pre_feedback2 = jax.lax.stop_gradient(new_pre_feedback2.weight)
        new_self = eqx.tree_at(lambda r: r.pre_feedback2, self, new_pre_feedback2)

        new_recurrent_feedback = eqx.nn.Linear(
            self.recurrent_feedback.in_features, self.recurrent_feedback.out_features, key=key, use_bias=False
        )
        # new_recurrent_feedback = jax.lax.stop_gradient(new_recurrent_feedback.weight)
        new_self = eqx.tree_at(lambda r: r.recurrent_feedback, self, new_recurrent_feedback)
        return new_self

    def initialise_hidden_state(self, batch_size: int) -> jnp.ndarray:
        if batch_size == 1:
            return jnp.zeros((self.forward2.out_features,))
        return jnp.zeros((batch_size, self.forward2.out_features))

    def __call__(self, x: jnp.ndarray, h: jnp.ndarray, label: jnp.ndarray) -> jnp.ndarray:
        h1 = self.forward1(x)
        h1_activated = self.softplus(h1, beta=self.beta)

        recurrent_input_activated = self.recurrent_activation(h) if self.recurrent_activation else h
        recurrent_input = self.forward2(recurrent_input_activated)

        h_new_pre_tau = h1_activated + recurrent_input
        h_new_pre_tau_activated = self.outer_activation(h_new_pre_tau) if self.outer_activation else h_new_pre_tau
        h_new = h + (1.0 / self.tau) * (-h + h_new_pre_tau_activated)

        y = self.forward3(h_new)

        if label is None:
            return y, h_new, None, None

        softmax_y = jax.nn.softmax(y)
        label_one_hot = jax.nn.one_hot(label, num_classes=y.shape[-1])
        error = softmax_y - label_one_hot

        w_pre1 = jax.lax.stop_gradient(self.pre_feedback1.weight)
        w_pre2 = jax.lax.stop_gradient(self.pre_feedback2.weight)
        w_rec = jax.lax.stop_gradient(self.recurrent_feedback.weight)

        # Use the stopped weights for the feedback calculation
        pre_feedback_input = jnp.dot(w_pre1, error)
        pre_feedback_hidden = jnp.dot(w_pre2, error)
        recurrent_feedback_hidden = jnp.dot(w_rec, error)

        errors = {
            "forward1": (pre_feedback_input, recurrent_feedback_hidden),
            "forward2": (pre_feedback_hidden, recurrent_feedback_hidden),
            "forward3": (recurrent_feedback_hidden, error),
        }

        activations = {
            "forward1": (x, h1_activated),
            "forward2": (recurrent_input_activated, recurrent_input),
            "forward3": (h_new, y),
        }
        if self.gradient:
            gradients = {
                "forward1": (1 - jnp.exp(-self.beta * x), jax.vmap(jax.grad(self.softplus))(h1)),
                "forward2": (
                    jax.vmap(jax.grad(self.recurrent_activation))(h) if self.recurrent_activation else jnp.ones_like(h),                                                                            (
                        jax.vmap(jax.grad(self.outer_activation))(h_new_pre_tau) #* 1.0 / self.tau
                        if self.outer_activation
                        else jnp.ones_like(h_new_pre_tau) #* 1.0 / self.tau
                    ),
                ),
                "forward3": (
                    (
                        jax.vmap(jax.grad(self.outer_activation))(h_new_pre_tau) #* 1.0 / self.tau
                        if self.outer_activation
                        else jnp.ones_like(h_new_pre_tau) #* 1.0 / self.tau
                    ),
                    jnp.ones_like(y),
                ),
            }
            
            errors = {
                "forward1": (
                    errors["forward1"][0] * gradients["forward1"][0],
                    errors["forward1"][1] * gradients["forward1"][1],
                ),
                "forward2": (
                    errors["forward2"][0] * gradients["forward2"][0],
                    errors["forward2"][1] * gradients["forward2"][1],
                ),
                "forward3": (
                    errors["forward3"][0] * gradients["forward3"][0],
                    errors["forward3"][1] * gradients["forward3"][1],
                ),
            }

        activations_arr = [activations["forward1"], activations["forward2"], activations["forward3"]]
        errors_arr = [errors["forward1"], errors["forward2"], errors["forward3"]]

        return y, h_new, activations_arr, errors_arr
