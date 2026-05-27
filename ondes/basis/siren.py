"""SIREN basis: ``sin(omega * pre)`` (Sitzmann+ 2020).

Also owns the SIREN-family init helpers (``siren_init``, ``_build_readout``)
because they were introduced by the SIREN paper and inherited by H-SIREN /
WIRE. New bases that adopt the same init scheme (variance-preserving uniform
on ``[-sqrt(6/in)/omega, +sqrt(6/in)/omega]``) should import from here.
Bases with a different init scheme (e.g. Kaiming for ReLU-like, Gaussian
init for RFF) should own their own.
"""

import jax
import jax.numpy as jnp

from ondes.basis._base import Basis, Body, _validate_body_args


def siren_init(in_dim, out_dim, omega, is_first, key):
    """Sample ``(W, b)`` under the SIREN initialisation scheme.

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


def _build_readout(hidden_dim, omega_hidden, out_features, key):
    """Sample SIREN-style readout weights.

    Shared by SIREN/H-SIREN/WIRE because they all share the SIREN-family
    activation-variance assumption. New basis families (MFN, RFF, …) should
    write their own readout init if they use a different scheme.
    """
    # Bound applies per-output-component; independent of out_features.
    bound = jnp.sqrt(6.0 / hidden_dim) / max(omega_hidden, 1e-3)
    kw, kb = jax.random.split(key)
    out_dim = 1 if out_features is None else out_features
    readout_W = jax.random.uniform(kw, (out_dim, hidden_dim), minval=-bound, maxval=bound)
    readout_b = jax.random.uniform(kb, (out_dim,), minval=-bound, maxval=bound)
    return readout_W, readout_b


class SIRENLayer(Basis):
    """SIREN layer: ``sin(omega * pre)`` (Sitzmann+ 2020)."""

    # Explicit pass-through __init__ so pyright sees concrete signatures on
    # subclasses (eqx.Module + ABC machinery confuses static analysis).
    # DO NOT delete — see DECISIONS.md §"Polymorphism over discriminators".
    def __init__(self, in_dim, out_dim, omega_init, is_first, *, key):
        """Initialise the linear weights and the learnable ``omega``.

        ``is_first`` is a construction-only kwarg consumed by :func:`siren_init`
        to pick the init bound (``1/in_dim`` for the first layer,
        ``sqrt(6/in_dim)/omega`` for the rest). It is *not* stored on the
        layer because the bound has already been baked into ``W``/``b`` —
        the forward pass never needs it. Omitting the field keeps every
        layer in a body pytree-structurally identical, which is the
        precondition for ``jax.lax.scan`` over the full layer stack.
        """
        self.W, self.b = siren_init(in_dim, out_dim, omega_init, is_first, key)
        self.omega = jnp.array(float(omega_init))

    def _activate(self, pre):
        return jnp.sin(self.omega * pre)


class SIREN(Body):
    """Stack of ``SIRENLayer`` s with an internal linear readout (Sitzmann+ 2020)."""

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_hidden_layers,
        *,
        key,
        out_features=None,
        omega_first=6.0,
        omega_hidden=1.0,
    ):
        """Initialise the SIREN body MLP.

        Args:
            in_dim: Coordinate (input) dimension.
            hidden_dim: Width of each hidden layer.
            num_hidden_layers: Number of stacked ``SIRENLayer`` s.
            key: JAX PRNG key.
            out_features: Readout width. ``None`` (default) or ``1`` makes
                ``__call__`` return a scalar; integer ``N > 1`` makes it
                return a vector of shape ``(N,)``. ``1`` is canonicalised to
                ``None`` so the two scalar constructions are indistinguishable.
            omega_first: Initial frequency for the first (input) layer.
            omega_hidden: Initial frequency for subsequent layers.
        """
        out_features = _validate_body_args(num_hidden_layers, out_features)
        keys = jax.random.split(key, num_hidden_layers + 1)
        layers = []
        for i in range(num_hidden_layers):
            in_d = in_dim if i == 0 else hidden_dim
            o = omega_first if i == 0 else omega_hidden
            layers.append(SIRENLayer(in_d, hidden_dim, o, is_first=(i == 0), key=keys[i]))
        self.layers = tuple(layers)
        self.readout_W, self.readout_b = _build_readout(hidden_dim, omega_hidden, out_features, keys[-1])
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
