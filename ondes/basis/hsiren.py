"""H-SIREN basis: ``sin(omega * sinh(pre))`` (Cai & Pan 2024).

The ``sinh`` pre-warp expands the dynamic range of the first sinusoidal
argument, which the paper reports improves natural-image fitting by a small
margin over plain SIREN at the same hyperparameters. Init scheme is inherited
unchanged from SIREN (see ``ondes.basis.siren.siren_init``).
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from ondes.basis._base import Basis, Body, _validate_body_args
from ondes.basis.siren import _build_readout, siren_init


class HSIRENLayer(Basis):
    """H-SIREN layer: ``sin(omega * sinh(pre))`` (Cai & Pan 2024)."""

    # Explicit pass-through __init__ so pyright sees concrete signatures on
    # subclasses (eqx.Module + ABC machinery confuses static analysis).
    # DO NOT delete — see DECISIONS.md §"Polymorphism over discriminators".
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        omega_init: float,
        is_first: bool,
        *,
        key: Key[Array, ""],
    ) -> None:
        """Initialise the linear weights and the learnable ``omega``."""
        self.W, self.b = siren_init(in_dim, out_dim, omega_init, is_first, key)
        self.omega = jnp.array(float(omega_init))

    def _activate(self, pre: Float[Array, "out"]) -> Float[Array, "out"]:
        """Apply ``sin(omega * sinh(pre))`` pointwise."""
        return jnp.sin(self.omega * jnp.sinh(pre))


class HSIREN(Body):
    """Stack of ``HSIRENLayer`` s with an internal linear readout (Cai & Pan 2024)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        *,
        key: Key[Array, ""],
        out_features: int | None = None,
        omega_first: float = 6.0,
        omega_hidden: float = 1.0,
    ) -> None:
        """Initialise the H-SIREN body MLP.

        Args:
            in_dim: Coordinate (input) dimension.
            hidden_dim: Width of each hidden layer.
            num_hidden_layers: Number of stacked ``HSIRENLayer`` s.
            key: JAX PRNG key.
            out_features: Readout width; see :class:`ondes.basis.siren.SIREN`.
            omega_first: Initial frequency for the first (input) layer.
            omega_hidden: Initial frequency for subsequent layers.
        """
        out_features = _validate_body_args(num_hidden_layers, out_features)
        keys = jax.random.split(key, num_hidden_layers + 1)
        layers = []
        for i in range(num_hidden_layers):
            in_d = in_dim if i == 0 else hidden_dim
            o = omega_first if i == 0 else omega_hidden
            layers.append(HSIRENLayer(in_d, hidden_dim, o, is_first=(i == 0), key=keys[i]))
        self.layers = tuple(layers)
        self.readout_W, self.readout_b = _build_readout(hidden_dim, omega_hidden, out_features, keys[-1])
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
