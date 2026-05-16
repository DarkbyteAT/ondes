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
    """Stack of ``BasisLayer`` s with an internal linear readout.

    ``out_features`` controls the readout width and the return shape of
    ``__call__``: ``None`` (default) or ``1`` gives a scalar, integer ``N > 1``
    gives a vector of shape ``(N,)``. The value ``1`` is canonicalised to
    ``None`` at construction so the two scalar-yielding constructions produce
    identical pytrees. The readout is owned by ``ondes`` and is not
    user-extensible — there is no ``head=`` kwarg and no ``Head`` type. To
    attach a distribution head, parameterisation, or other post-trunk
    transform, build a small ``eqx.Module`` wrapper around this body and call
    ``trunk()`` (or ``__call__``) from it.
    """

    layers: tuple
    readout_W: Float[Array, "out hidden"]
    readout_b: Float[Array, "out"]
    hidden_dim: int = eqx.field(static=True)
    num_hidden_layers: int = eqx.field(static=True)
    kind: str = eqx.field(static=True)
    out_features: int | None = eqx.field(static=True)

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
        out_features=None,
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
            out_features: Readout width. ``None`` (default) or ``1`` makes
                ``__call__`` return a scalar; integer ``N > 1`` makes it
                return a vector of shape ``(N,)``. ``1`` is canonicalised to
                ``None`` so the two scalar constructions are indistinguishable.
        """
        assert kind in BASIS_KINDS, kind
        assert num_hidden_layers >= 1, f"num_hidden_layers must be >= 1, got {num_hidden_layers}"
        assert out_features is None or (
            isinstance(out_features, int) and not isinstance(out_features, bool) and out_features >= 1
        ), f"out_features must be None or positive int, got {out_features!r}"
        if out_features == 1:
            out_features = None
        keys = jax.random.split(key, num_hidden_layers + 1)
        layers = []
        for i in range(num_hidden_layers):
            in_d = in_dim if i == 0 else hidden_dim
            o = omega_first if i == 0 else omega_hidden
            layers.append(BasisLayer(in_d, hidden_dim, o, kind, is_first=(i == 0), key=keys[i], s_init=s_init))
        self.layers = tuple(layers)
        # Bound applies per-output-component; independent of out_features.
        bound = jnp.sqrt(6.0 / hidden_dim) / max(omega_hidden, 1e-3)
        kw, kb = jax.random.split(keys[-1])
        out_dim = 1 if out_features is None else out_features
        self.readout_W = jax.random.uniform(kw, (out_dim, hidden_dim), minval=-bound, maxval=bound)
        self.readout_b = jax.random.uniform(kb, (out_dim,), minval=-bound, maxval=bound)
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.kind = kind
        self.out_features = out_features

    def trunk(self, coord, *, film=None):
        """Return pre-readout hidden features.

        Args:
            coord: Coordinate vector of shape ``(in_dim,)``.
            film: Optional FiLM tensor of shape
                ``(num_hidden_layers, 2 * hidden_dim)`` whose halves provide
                ``gamma`` and ``beta`` per layer. ``None`` skips modulation.

        Returns:
            Activations of the final hidden layer, shape ``(hidden_dim,)``.
        """
        h = coord
        for i, layer in enumerate(self.layers):
            if film is not None:
                gamma = film[i, : self.hidden_dim]
                beta = film[i, self.hidden_dim :]
                h = layer(h, gamma=gamma, beta=beta)
            else:
                h = layer(h)
        return h

    def _readout(self, h):
        """Internal linear readout. Not a user extension point."""
        return self.readout_W @ h + self.readout_b

    def __call__(self, coord, *, film=None):
        """Forward pass.

        Args:
            coord: Coordinate vector of shape ``(in_dim,)``.
            film: Optional FiLM tensor of shape
                ``(num_hidden_layers, 2 * hidden_dim)`` whose halves provide
                ``gamma`` and ``beta`` per layer. When ``None`` no modulation
                is applied.

        Returns:
            Scalar when ``out_features`` is ``None`` (or was constructed as
            ``1``); otherwise a vector of shape ``(out_features,)``.
        """
        y = self._readout(self.trunk(coord, film=film))
        if self.out_features is None:
            # squeeze(-1) only collapses the readout's size-1 feature axis;
            # any leading batch dims (e.g. from vmap with batch size 1) survive.
            return y.squeeze(-1)
        return y
