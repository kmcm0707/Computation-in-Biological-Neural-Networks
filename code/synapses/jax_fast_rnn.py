import jax.numpy as jnp
from jax import grad, jit, vmap
from flax import nnx

class FastJaxRnn(nnx.Module):
    def __init__(self, numberOfChemicals, fastJaxRnnOptions, conversion_matrix: dict = {}):
        super().__init__()
        self.numberOfChemicals = numberOfChemicals
        self.fastJaxRnnOptions = fastJaxRnnOptions
        self.conversion_matrix = conversion_matrix
        self.non_linearity = fastJaxRnnOptions.non_linearity

        self.numberOfUpdateRules = 14
        self.update_rules = [False] * self.numberOfUpdateRules
        trueNumberOfUpdateRules = 0
        for rule in fastJaxRnnOptions.update_rules:
            trueNumberOfUpdateRules += 1
            self.update_rules[rule] = True
        self.numberOfUpdateRules = trueNumberOfUpdateRules

        self.Q_matrix = self.param('Q_matrix', jnp.zeros, (self.numberOfChemicals, self.numberOfUpdateRules))
        self.Q_matrix = self.Q_matrix.at[:, 0].set(1e-3)  # Baseline

        self.K_matrix = self.param('K_matrix', jnp.zeros, (self.numberOfChemicals, self.numberOfChemicals))

        self.v_vector = self.param('v_vector', jnp.zeros, (self.numberOfChemicals,))

        min_tau = self.options.minSlowTau
        max_tau = self.options.maxSlowTau
        base = max_tau / min_tau

        self.tau_vector = min_tau * jnp.power(base, jnp.linspace(0, 1, self.numberOfChemicals))
        self.z_vector = 1.0 / self.tau_vector
        self.y_vector = 1.0 - self.z_vector

    def __call__(self, activations_and_output, error, states):
        exit()

    def calculate_update_vectors(self, a):
        exit()







