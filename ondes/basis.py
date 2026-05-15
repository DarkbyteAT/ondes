"""SIREN / H-SIREN / WIRE basis layers and the composed body MLP.

The three supported activations are:

- ``siren``  : ``sin(omega * z)`` (Sitzmann+ 2020)
- ``hsiren`` : ``sin(omega * sinh(z))`` (Cai & Pan 2024)
- ``wire``   : ``cos(omega * z) * exp(-(s * z) ** 2)`` (Saragadam+ 2023)

``omega`` (all bases) and ``s`` (WIRE) are stored as learnable scalars per layer.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


BASIS_KINDS = ("siren", "hsiren", "wire")


def siren_init(in_dim, out_dim, omega, is_first, key):
    """Sample (W, b) under the SIREN initialisation scheme.

    First-layer weights are drawn uniformly from ``[-1/in_dim, 1/in_dim]``;
    subsequent layers from ``[-sqrt(6/in_dim)/omega, +sqrt(6/in_dim)/omega]``,
    which preserves activation variance through the ``sin(omega * .)`` non-linearity.

    Args:
        in_dim: Input dimension of the linear map.
        out_dim: Output dimension of the linear map.
        omega: Frequency scale for the activation that follows this layer.
        is_first: Whether this is the first layer of the network.
        key: JAX PRNG key.

    Returns:
        Tuple ``(W, b)`` with shapes ``(out_dim, in_dim)`` and ``(out_dim,)``.
    """
    bound = 1.0 / in_dim if is_first else jnp.sqrt(6.0 / in_dim) / omega
    k_w, k_b = jax.random.split(key)
    W = jax.random.uniform(k_w, (out_dim, in_dim), minval=-bound, maxval=bound)
    b = jax.random.uniform(k_b, (out_dim,), minval=-bound, maxval=bound)
    return W, b


class BasisLayer(eqx.Module):
    """One basis-MLP layer with kind-dispatched activation.

    Computes ``y = activation(omega * pre [, s * pre])`` where
    ``pre = gamma * (W x + b) + beta``. ``omega`` is learnable for every basis;
    ``s`` is meaningfully used only by WIRE but is stored for every layer so
    the pytree structure is uniform across kinds.

    Note:
        Earlier revisions stored ``omega`` and ``s`` in log-space to enforce
        positivity. That was abandoned because ``dL/d(log omega) = omega * dL/d omega``
        couples the effective step size to the current ``omega`` magnitude.
        The activations are even/symmetric in ``omega`` (sin, sinh-then-sin,
        cos-and-even-``s^2``), so direct parameterisation is mathematically safe.
    """

    W: Float[Array, "out in"]
    b: Float[Array, "out"]
    omega: Float[Array, ""]
    s: Float[Array, ""]
    kind: str = eqx.field(static=True)
    is_first: bool = eqx.field(static=True)

    def __init__(self, in_dim, out_dim, omega_init, kind, is_first, *, key, s_init=3.0):
        """Initialise a single basis layer.

        Args:
            in_dim: Input dimension.
            out_dim: Output dimension.
            omega_init: Initial value for the learnable frequency scalar.
            kind: One of ``BASIS_KINDS``.
            is_first: Whether this layer is the input layer of the body.
            key: JAX PRNG key.
            s_init: Initial value for the learnable WIRE gaussian-width scalar.
        """
        self.W, self.b = siren_init(in_dim, out_dim, omega_init, is_first, key)
        self.omega = jnp.array(float(omega_init))
        self.s = jnp.array(float(s_init))
        self.kind = kind
        self.is_first = is_first

    def __call__(self, x, gamma=None, beta=None):
        """Apply the layer to ``x`` with optional FiLM modulation.

        Args:
            x: Input vector of shape ``(in_dim,)``.
            gamma: Optional multiplicative modulation of shape ``(out_dim,)``.
            beta: Optional additive modulation of shape ``(out_dim,)``.

        Returns:
            Activated output of shape ``(out_dim,)``.
        """
        pre = self.W @ x + self.b
        if gamma is not None:
            pre = gamma * pre
        if beta is not None:
            pre = pre + beta
        if self.kind == "siren":
            return jnp.sin(self.omega * pre)
        if self.kind == "hsiren":
            return jnp.sin(self.omega * jnp.sinh(pre))
        if self.kind == "wire":
            sz = self.s * pre
            return jnp.cos(self.omega * pre) * jnp.exp(-(sz * sz))
        raise ValueError(f"unknown basis kind {self.kind!r}")


class BasisBody(eqx.Module):
    """Stack of ``BasisLayer`` s with a scalar readout head."""

    layers: tuple
    readout_W: Float[Array, "1 hidden"]
    readout_b: Float[Array, ""]
    hidden_dim: int = eqx.field(static=True)
    num_hidden_layers: int = eqx.field(static=True)
    kind: str = eqx.field(static=True)

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_hidden_layers,
        kind="siren",
        *,
        key,
        omega_first=6.0,
        omega_hidden=1.0,
        s_init=3.0,
    ):
        """Initialise the body MLP.

        Args:
            in_dim: Coordinate (input) dimension.
            hidden_dim: Width of each hidden layer.
            num_hidden_layers: Number of stacked ``BasisLayer`` s.
            kind: One of ``BASIS_KINDS``.
            key: JAX PRNG key.
            omega_first: Initial frequency for the first (input) layer.
            omega_hidden: Initial frequency for subsequent layers.
            s_init: Initial WIRE width scalar (used only when ``kind == "wire"``).
        """
        assert kind in BASIS_KINDS, kind
        keys = jax.random.split(key, num_hidden_layers + 1)
        layers = []
        for i in range(num_hidden_layers):
            in_d = in_dim if i == 0 else hidden_dim
            o = omega_first if i == 0 else omega_hidden
            layers.append(BasisLayer(in_d, hidden_dim, o, kind, is_first=(i == 0), key=keys[i], s_init=s_init))
        self.layers = tuple(layers)
        bound = jnp.sqrt(6.0 / hidden_dim) / max(omega_hidden, 1e-3)
        kw, kb = jax.random.split(keys[-1])
        self.readout_W = jax.random.uniform(kw, (1, hidden_dim), minval=-bound, maxval=bound)
        self.readout_b = jax.random.uniform(kb, (), minval=-bound, maxval=bound)
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.kind = kind

    def __call__(self, x, film=None):
        """Forward pass.

        Args:
            x: Coordinate vector of shape ``(in_dim,)``.
            film: Optional FiLM tensor of shape
                ``(num_hidden_layers, 2 * hidden_dim)`` whose halves provide
                ``gamma`` and ``beta`` per layer. When ``None`` no modulation
                is applied.

        Returns:
            Scalar prediction.
        """
        h = x
        for i, layer in enumerate(self.layers):
            if film is not None:
                gamma = film[i, : self.hidden_dim]
                beta = film[i, self.hidden_dim :]
                h = layer(h, gamma=gamma, beta=beta)
            else:
                h = layer(h)
        return (self.readout_W @ h + self.readout_b).squeeze()
