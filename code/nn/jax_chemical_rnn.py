from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp


def beta_softplus(x: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
    return (1.0 / beta) * jnp.log1p(jnp.exp(beta * x))


class JAXChemicalRNN(eqx.Module):
    forward1: eqx.nn.Linear
    forward2: eqx.nn.Linear
    forward3: eqx.nn.Linear

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
        key1, key2, key3 = jax.random.split(key, 3)
        self.forward1 = eqx.nn.Linear(input_size, hidden_size, key=key1)
        self.forward2 = eqx.nn.Linear(hidden_size, hidden_size, key=key2)
        self.forward3 = eqx.nn.Linear(hidden_size, output_size, key=key3)

        self.biological_min_tau = biological_min_tau
        self.biological_max_tau = biological_max_tau
        self.gradient = gradient
        self.outer_activation = outer_activation
        self.recurrent_activation = recurrent_activation
        self.beta = 10.0
        self.softplus = beta_softplus
        tau = jnp.linspace(self.biological_min_tau, self.biological_max_tau, hidden_size)
        self.tau = tau

    def initialise_hidden_state(self, batch_size: int) -> jnp.ndarray:
        return jnp.zeros((batch_size, self.forward2.out_features))

    def __call__(self, x: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        h1 = self.forward1(x)
        h1_activated = self.softplus(h1, beta=self.beta)

        recurrent_input = self.forward2(h)
        recurrent_input_activated = self.softplus(recurrent_input, beta=self.beta)

        h_new_pre_tau = h1_activated + recurrent_input_activated
        h_new_pre_tau_activated = self.outer_activation(h_new_pre_tau) if self.recurrent_activation else h_new_pre_tau
        h_new = h + (1.0 / self.tau) * (-h + h_new_pre_tau_activated)

        y = self.forward3(h_new)

        activations = {
            "forward1": (x, h1_activated),
            "forward2": (h, recurrent_input_activated),
            "forward3": (h_new, y),
        }
        if self.gradient:
            gradients = {
                "forward1": (1 - jnp.exp(-self.beta * x), jax.grad(self.softplus)(h1, h1_activated)),
                "forward2": (),
            }

        return y, h_new


def initialize_rnn(input_size: int, hidden_size: int, output_size: int, seed: int = 0) -> JAXChemicalRNN:
    key = jax.random.PRNGKey(seed)
    rnn = JAXChemicalRNN(input_size, hidden_size, output_size, key)
    return rnn
