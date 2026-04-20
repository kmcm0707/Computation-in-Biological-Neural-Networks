from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from options.jax_rnn_meat_learner_options import (
    JaxActivationNonLinearEnum,
    JaxErrorTypeEnum,
)


def beta_softplus(x: jnp.ndarray, beta: float = 10) -> jnp.ndarray:
    return (1.0 / beta) * jnp.log1p(jnp.exp(beta * x))


class JAXChemicalRNN(eqx.Module):
    forward1: eqx.nn.Linear
    forward2: eqx.nn.Linear
    forward3: eqx.nn.Linear
    forward4: eqx.nn.Linear
    pre_feedback1: eqx.nn.Linear
    pre_feedback2: eqx.nn.Linear
    pre_feedback3: eqx.nn.Linear
    recurrent_feedback: eqx.nn.Linear
    recurrent_feedback2: eqx.nn.Linear
    layers: tuple[eqx.nn.Linear]
    feedback_layers: tuple[eqx.nn.Linear]
    gradient: bool = eqx.field(static=True)
    outer_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = eqx.field(static=True)
    recurrent_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = eqx.field(static=True)
    beta: float
    tau: jnp.ndarray
    softplus: Callable[[jnp.ndarray, float], jnp.ndarray] = eqx.field(static=True)
    error_type: JaxErrorTypeEnum = eqx.field(static=True)
    low_dim_DFA: int = eqx.field(static=True)
    two_layer_RNN: bool = eqx.field(static=True)

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
        error_type: JaxErrorTypeEnum = JaxErrorTypeEnum.DFA,
        low_dim_DFA: int = -1,
        two_layer_RNN: bool = False,
    ):
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)
        self.two_layer_RNN = two_layer_RNN

        self.forward1 = eqx.nn.Linear(input_size, hidden_size, key=key1, use_bias=False)
        self.forward2 = eqx.nn.Linear(hidden_size, hidden_size, key=key2, use_bias=False)
        self.forward3 = eqx.nn.Linear(hidden_size, output_size, key=key3, use_bias=False)
        if self.two_layer_RNN:
            key3_2, key3_3 = jax.random.split(key3, 2)
            self.forward3 = eqx.nn.Linear(hidden_size, hidden_size, key=key3_2, use_bias=False)
            self.forward4 = eqx.nn.Linear(hidden_size, output_size, key=key3_3, use_bias=False)
        else:
            self.forward4 = None

        self.layers = (self.forward1, self.forward2, self.forward3)
        if self.two_layer_RNN:
            self.layers = (self.forward1, self.forward2, self.forward3, self.forward4)

        if error_type == JaxErrorTypeEnum.DFA:
            error_size = output_size
        elif error_type == JaxErrorTypeEnum.DSEF:
            error_size = 1
        else:
            raise ValueError(f"Unsupported error type: {error_type}")

        self.pre_feedback1 = eqx.nn.Linear(error_size, input_size, key=key4, use_bias=False)
        self.pre_feedback2 = eqx.nn.Linear(error_size, hidden_size, key=key5, use_bias=False)
        self.recurrent_feedback = eqx.nn.Linear(error_size, hidden_size, key=key6, use_bias=False)
        if self.two_layer_RNN:
            key6_2, key6_3 = jax.random.split(key6, 2)
            self.pre_feedback3 = eqx.nn.Linear(error_size, hidden_size, key=key6_2, use_bias=False)
            self.recurrent_feedback2 = eqx.nn.Linear(error_size, hidden_size, key=key6_3, use_bias=False)
        else:
            self.pre_feedback3 = None
            self.recurrent_feedback2 = None

        if not self.two_layer_RNN:
            self.feedback_layers = (self.pre_feedback1, self.pre_feedback2, self.recurrent_feedback)
        else:
            self.feedback_layers = (
                self.pre_feedback1,
                self.pre_feedback2,
                self.recurrent_feedback,
                self.recurrent_feedback2,
            )

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

        self.error_type = error_type
        self.low_dim_DFA = low_dim_DFA

    def low_dim_feedback_initialization(
        self, key: jax.random.PRNGKey, in_features: int, out_features: int
    ) -> jnp.ndarray:
        a = (18.0 / (self.low_dim_DFA * (in_features + out_features))) ** 0.25
        vec_pre1 = jax.random.uniform(key, (self.low_dim_DFA, in_features), minval=-a, maxval=a)
        vec_pre2 = jax.random.uniform(key, (self.low_dim_DFA, out_features), minval=-a, maxval=a)
        full_pre = jnp.dot(vec_pre1.T, vec_pre2)
        return full_pre.T

    def reset_weights(self, key: jax.random.PRNGKey) -> "JAXChemicalRNN":
        new_pre_feedback1 = eqx.nn.Linear(
            self.pre_feedback1.in_features, self.pre_feedback1.out_features, key=key, use_bias=False
        )
        if self.low_dim_DFA > 0:

            new_pre_feedback1 = eqx.tree_at(
                lambda r: r.weight,
                new_pre_feedback1,
                self.low_dim_feedback_initialization(
                    key, self.pre_feedback1.in_features, self.pre_feedback1.out_features
                ),
            )

        # new_pre_feedback1 = jax.lax.stop_gradient(new_pre_feedback1.weight)
        new_self = eqx.tree_at(lambda r: r.pre_feedback1, self, new_pre_feedback1)

        new_pre_feedback2 = eqx.nn.Linear(
            self.pre_feedback2.in_features, self.pre_feedback2.out_features, key=key, use_bias=False
        )
        if self.low_dim_DFA > 0:
            new_pre_feedback2 = eqx.tree_at(
                lambda r: r.weight,
                new_pre_feedback2,
                self.low_dim_feedback_initialization(
                    key, self.pre_feedback2.in_features, self.pre_feedback2.out_features
                ),
            )

        # new_pre_feedback2 = jax.lax.stop_gradient(new_pre_feedback2.weight)
        new_self = eqx.tree_at(lambda r: r.pre_feedback2, new_self, new_pre_feedback2)

        new_recurrent_feedback = eqx.nn.Linear(
            self.recurrent_feedback.in_features, self.recurrent_feedback.out_features, key=key, use_bias=False
        )
        if self.low_dim_DFA > 0:
            new_recurrent_feedback = eqx.tree_at(
                lambda r: r.weight,
                new_recurrent_feedback,
                self.low_dim_feedback_initialization(
                    key, self.recurrent_feedback.in_features, self.recurrent_feedback.out_features
                ),
            )

        new_self = eqx.tree_at(lambda r: r.recurrent_feedback, new_self, new_recurrent_feedback)

        if self.two_layer_RNN:
            new_recurrent_feedback2 = eqx.nn.Linear(
                self.recurrent_feedback2.in_features,
                self.recurrent_feedback2.out_features,
                key=key,
                use_bias=False,
            )
            if self.low_dim_DFA > 0:
                new_recurrent_feedback2 = eqx.tree_at(
                    lambda r: r.weight,
                    new_recurrent_feedback2,
                    self.low_dim_feedback_initialization(
                        key, self.recurrent_feedback2.in_features, self.recurrent_feedback2.out_features
                    ),
                )

            new_self = eqx.tree_at(lambda r: r.recurrent_feedback2, new_self, new_recurrent_feedback2)

            new_pre_feedback3 = eqx.nn.Linear(
                self.pre_feedback3.in_features, self.pre_feedback3.out_features, key=key, use_bias=False
            )
            if self.low_dim_DFA > 0:
                new_pre_feedback3 = eqx.tree_at(
                    lambda r: r.weight,
                    new_pre_feedback3,
                    self.low_dim_feedback_initialization(
                        key, self.pre_feedback3.in_features, self.pre_feedback3.out_features
                    ),
                )
            new_self = eqx.tree_at(lambda r: r.pre_feedback3, new_self, new_pre_feedback3)

        new_linear1 = eqx.nn.Linear(self.forward1.in_features, self.forward1.out_features, key=key, use_bias=False)
        new_self = eqx.tree_at(lambda r: r.forward1, new_self, new_linear1)
        new_linear2 = eqx.nn.Linear(self.forward2.in_features, self.forward2.out_features, key=key, use_bias=False)
        new_self = eqx.tree_at(lambda r: r.forward2, new_self, new_linear2)
        new_linear3 = eqx.nn.Linear(self.forward3.in_features, self.forward3.out_features, key=key, use_bias=False)
        new_self = eqx.tree_at(lambda r: r.forward3, new_self, new_linear3)
        if self.two_layer_RNN:
            new_linear4 = eqx.nn.Linear(self.forward4.in_features, self.forward4.out_features, key=key, use_bias=False)
            new_self = eqx.tree_at(lambda r: r.forward4, new_self, new_linear4)

        return new_self

    def initialise_hidden_state(self, batch_size: int) -> jnp.ndarray:
        if batch_size == 1:
            if self.two_layer_RNN:
                return tuple([jnp.zeros((self.forward2.out_features,)), jnp.zeros((self.forward3.out_features,))])
            return jnp.zeros((self.forward2.out_features,))
        if self.two_layer_RNN:
            return tuple(
                [
                    jnp.zeros((batch_size, self.forward2.out_features)),
                    jnp.zeros((batch_size, self.forward3.out_features)),
                ]
            )
        return jnp.zeros((batch_size, self.forward2.out_features))

    def __call__(self, x: jnp.ndarray, h: jnp.ndarray, label: jnp.ndarray) -> jnp.ndarray:
        h1 = self.forward1(x)
        h1_activated = self.softplus(h1, beta=self.beta)

        if not self.two_layer_RNN:
            recurrent_input = self.forward2(h)
            recurrent_input_activated = (
                self.recurrent_activation(recurrent_input) if self.recurrent_activation else recurrent_input
            )

            h_new_pre_tau = h1_activated + recurrent_input_activated
            h_new_pre_tau_activated = self.outer_activation(h_new_pre_tau) if self.outer_activation else h_new_pre_tau
            h_new = h + (1.0 / self.tau) * (-h + h_new_pre_tau_activated)

            y = self.forward3(h_new)
        else:
            recurrent_input1 = self.forward2(h[0])
            recurrent_input_activated1 = (
                self.recurrent_activation(recurrent_input1) if self.recurrent_activation else recurrent_input1
            )

            h_new_pre_tau1 = h1_activated + recurrent_input_activated1
            h_new_pre_tau_activated1 = (
                self.outer_activation(h_new_pre_tau1) if self.outer_activation else h_new_pre_tau1
            )
            h_new1 = h[0] + (1.0 / self.tau) * (-h[0] + h_new_pre_tau_activated1)

            recurrent_input2 = self.forward3(h[1])
            recurrent_input_activated2 = (
                self.recurrent_activation(recurrent_input2) if self.recurrent_activation else recurrent_input2
            )

            h_new_pre_tau2 = h_new1 + recurrent_input_activated2
            h_new_pre_tau_activated2 = (
                self.outer_activation(h_new_pre_tau2) if self.outer_activation else h_new_pre_tau2
            )
            h_new2 = h[1] + (1.0 / self.tau) * (-h[1] + h_new_pre_tau_activated2)

            y = self.forward4(h_new2)
            h_new = tuple([h_new1, h_new2])

        if label is None:
            return y, h_new, None, None

        softmax_y = jax.nn.softmax(y)
        if self.error_type == JaxErrorTypeEnum.DFA:
            error = softmax_y - label
        elif self.error_type == JaxErrorTypeEnum.DSEF:
            label_index = jnp.argmax(label, axis=-1, keepdims=True)
            y_index = jnp.take_along_axis(softmax_y, label_index, axis=-1)
            error = jax.lax.cond(
                jnp.squeeze(y_index) > 0.5,
                lambda: jnp.zeros_like(label_index),
                lambda: jnp.ones_like(label_index),
            )

        w_pre1 = jax.lax.stop_gradient(self.pre_feedback1.weight)
        w_pre2 = jax.lax.stop_gradient(self.pre_feedback2.weight)
        w_rec = jax.lax.stop_gradient(self.recurrent_feedback.weight)
        if self.two_layer_RNN:
            w_pre3 = jax.lax.stop_gradient(self.pre_feedback3.weight)
            w_rec2 = jax.lax.stop_gradient(self.recurrent_feedback2.weight)

        # Use the stopped weights for the feedback calculation
        pre_feedback_input = jnp.dot(w_pre1, error)
        pre_feedback_hidden = jnp.dot(w_pre2, error)
        recurrent_feedback_hidden = jnp.dot(w_rec, error)
        if self.two_layer_RNN:
            pre_feedback_hidden2 = jnp.dot(w_pre3, error)
            recurrent_feedback_hidden2 = jnp.dot(w_rec2, error)

        if self.error_type == JaxErrorTypeEnum.DSEF:
            error = softmax_y - label

        errors = {
            "forward1": (pre_feedback_input, recurrent_feedback_hidden),
            "forward2": (pre_feedback_hidden, recurrent_feedback_hidden),
            "forward3": (recurrent_feedback_hidden, error),
        }
        if self.two_layer_RNN:
            errors["forward3"] = (pre_feedback_hidden2, recurrent_feedback_hidden2)
            errors["forward4"] = (recurrent_feedback_hidden2, error)

        """activations = {
            "forward1": (x, h1_activated),
            "forward2": (h, recurrent_input_activated),  # (recurrent_input_activated, recurrent_input),
            "forward3": (h_new, y),
        }"""
        if not self.two_layer_RNN:
            activations = {
                "forward1": (x, h1_activated),
                "forward2": (h, recurrent_input_activated),  # (recurrent_input_activated, recurrent_input),
                "forward3": (h_new, y),
            }
        elif self.two_layer_RNN:
            activations = {
                "forward1": (x, h1_activated),
                "forward2": (h[0], recurrent_input_activated1),  # (recurrent_input_activated1, recurrent_input1),
                "forward3": (h[1], recurrent_input_activated2),  # (recurrent_input_activated2, recurrent_input2),
                "forward4": (h_new2, y),
            }

        if self.gradient:
            if not self.two_layer_RNN:
                gradients = {
                    "forward1": (
                        -jnp.expm1(-self.beta * x),
                        jax.vmap(jax.grad(self.softplus))(h1)
                        * jax.vmap(jax.grad(self.outer_activation))(h_new_pre_tau),  # * 1.0 / self.tau,
                    ),
                    "forward2": (
                        (jax.vmap(jax.grad(self.outer_activation))(h_new_pre_tau)),  # * 1.0 / self.tau,
                        (
                            jax.vmap(jax.grad(self.recurrent_activation))(recurrent_input)
                            * jax.vmap(jax.grad(self.outer_activation))(h_new_pre_tau)
                            if self.recurrent_activation
                            else jax.vmap(jax.grad(self.outer_activation))(h_new_pre_tau)  # * 1.0 / self.tau
                        ),
                    ),
                    "forward3": (
                        (jax.vmap(jax.grad(self.outer_activation))(h_new_pre_tau)),  # * 1.0 / self.tau,
                        jnp.ones_like(y),
                    ),
                }
            elif self.two_layer_RNN:
                gradients = {
                    "forward1": (
                        -jnp.expm1(-self.beta * x),
                        jax.vmap(jax.grad(self.softplus))(h1)
                        * jax.vmap(jax.grad(self.outer_activation))(h_new_pre_tau1),  # * 1.0 / self.tau,
                    ),
                    "forward2": (
                        (jax.vmap(jax.grad(self.outer_activation))(h_new_pre_tau1)),  # * 1.0 / self.tau,
                        (
                            jax.vmap(jax.grad(self.recurrent_activation))(recurrent_input1)
                            * jax.vmap(jax.grad(self.outer_activation))(h_new_pre_tau1)
                            if self.recurrent_activation
                            else jax.vmap(jax.grad(self.outer_activation))(h_new_pre_tau1)  # * 1.0 / self.tau
                        ),
                    ),
                    "forward3": (
                        (jax.vmap(jax.grad(self.outer_activation))(h_new_pre_tau2)),  # * 1.0 / self.tau,
                        (
                            jax.vmap(jax.grad(self.recurrent_activation))(recurrent_input2)
                            * jax.vmap(jax.grad(self.outer_activation))(h_new_pre_tau2)
                            if self.recurrent_activation
                            else jax.vmap(jax.grad(self.outer_activation))(h_new_pre_tau2)  # * 1.0 / self.tau
                        ),
                    ),
                    "forward4": (
                        (jax.vmap(jax.grad(self.outer_activation))(h_new_pre_tau2)),  # * 1.0 / self.tau,
                        jnp.ones_like(y),
                    ),
                }

            if not self.two_layer_RNN:
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
            else:
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
                    "forward4": (
                        errors["forward4"][0] * gradients["forward4"][0],
                        errors["forward4"][1] * gradients["forward4"][1],
                    ),
                }

        activations_arr = [activations["forward1"], activations["forward2"], activations["forward3"]]
        if self.two_layer_RNN:
            activations_arr.append(activations["forward4"])
        errors_arr = [errors["forward1"], errors["forward2"], errors["forward3"]]
        if self.two_layer_RNN:
            errors_arr.append(errors["forward4"])

        return y, h_new, activations_arr, errors_arr
