"""WIRE basis: ``cos(omega * pre) * exp(-(s * pre) ** 2)`` (Saragadam+ 2023).

WIRE adds a per-layer learnable Gaussian-envelope scalar ``s`` controlling the
window width. ``omega`` (carrier frequency) and ``s`` (envelope width) are
independent; the paper recommends ω=10, s=10 for natural-image fitting. Both
are exposed and learnable.

Init scheme is inherited from SIREN (see ``ondes.basis.siren.siren_init``).
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from ondes.basis._base import Basis, Body, _validate_body_args
from ondes.basis.siren import _build_readout, siren_init


class WIRELayer(Basis):
    """WIRE layer: ``cos(omega * pre) * exp(-(s * pre) ** 2)`` (Saragadam+ 2023).

    Carries an additional learnable scalar ``s`` controlling the Gaussian
    window width. ``s`` lives only on this subclass — SIREN and H-SIREN
    pytrees do not contain an unused ``s`` leaf.
    """

    s: Float[Array, ""]

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        omega_init: float,
        is_first: bool,
        *,
        key: Key[Array, ""],
        s_init: float = 3.0,
    ) -> None:
        """Initialise the linear weights, ``omega``, and the WIRE-specific ``s``."""
        self.W, self.b = siren_init(in_dim, out_dim, omega_init, is_first, key)
        self.omega = jnp.array(float(omega_init))
        self.s = jnp.array(float(s_init))

    def _activate(self, pre: Float[Array, "out"]) -> Float[Array, "out"]:
        """Apply ``cos(omega * pre) * exp(-(s * pre) ** 2)`` pointwise."""
        sz = self.s * pre
        return jnp.cos(self.omega * pre) * jnp.exp(-(sz * sz))


class WIRE(Body):
    """Stack of ``WIRELayer`` s with an internal linear readout (Saragadam+ 2023).

    Accepts ``s_init`` explicitly (forwarded to each ``WIRELayer``) controlling
    the initial Gaussian window width. SIREN and H-SIREN do not.
    """

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
        s_init: float = 3.0,
    ) -> None:
        """Initialise the WIRE body MLP.

        Args:
            in_dim: Coordinate (input) dimension.
            hidden_dim: Width of each hidden layer.
            num_hidden_layers: Number of stacked ``WIRELayer`` s.
            key: JAX PRNG key.
            out_features: Readout width; see :class:`ondes.basis.siren.SIREN`.
            omega_first: Initial frequency for the first (input) layer.
            omega_hidden: Initial frequency for subsequent layers.
            s_init: Initial WIRE Gaussian-window scalar (per layer). Paper
                uses ``s_init=10`` for natural images; default ``3.0`` is a
                gentler envelope suited to smoother synthetic targets.
        """
        out_features = _validate_body_args(num_hidden_layers, out_features)
        keys = jax.random.split(key, num_hidden_layers + 1)
        layers = []
        for i in range(num_hidden_layers):
            in_d = in_dim if i == 0 else hidden_dim
            o = omega_first if i == 0 else omega_hidden
            layers.append(WIRELayer(in_d, hidden_dim, o, is_first=(i == 0), key=keys[i], s_init=s_init))
        self.layers = tuple(layers)
        self.readout_W, self.readout_b = _build_readout(hidden_dim, omega_hidden, out_features, keys[-1])
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
