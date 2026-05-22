"""SIREN / H-SIREN / WIRE basis layers and bodies.

One class per basis family. Each layer subclass inherits the shared linear +
FiLM pre-activation from ``Basis`` and overrides ``_activate`` to produce its
post-activation output. ``WIRELayer`` is the only layer that carries the
WIRE-specific learnable scalar ``s`` — non-WIRE pytrees do not contain an
unused ``s`` leaf.

- ``SIRENLayer``  : ``sin(omega * z)`` (Sitzmann+ 2020)
- ``HSIRENLayer`` : ``sin(omega * sinh(z))`` (Cai & Pan 2024)
- ``WIRELayer``   : ``cos(omega * z) * exp(-(s * z) ** 2)`` (Saragadam+ 2023)

Bodies (``SIREN``, ``HSIREN``, ``WIRE``) each construct the matching layer
class and share trunk/readout machinery via the public ``Body`` base.
"""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


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


class Basis(eqx.Module):
    """ABC for a single basis-MLP layer.

    Holds the linear weights ``(W, b)`` and the learnable frequency scalar
    ``omega`` shared by every basis. Subclasses provide ``_activate(pre)`` to
    produce the post-activation output. Basis-specific fields (e.g. ``s`` for
    WIRE) live on the concrete subclass, not here.

    The ``Basis`` ABC is exported so downstream code (renderers, optimisers,
    test helpers) can express "any basis layer" in a single type. It is not
    instantiable on its own — calling it triggers the ``NotImplementedError``
    in ``_activate``.

    Note:
        Earlier revisions stored ``omega`` (and ``s``) in log-space to enforce
        positivity. That was abandoned because ``dL/d(log omega) = omega * dL/d omega``
        couples the effective step size to the current ``omega`` magnitude.
        The activations are even/symmetric in ``omega`` (sin, sinh-then-sin,
        cos-and-even-``s^2``), so direct parameterisation is mathematically safe.
    """

    W: Float[Array, "out in"]
    b: Float[Array, "out"]
    omega: Float[Array, ""]

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

    def _pre(self, x, gamma=None, beta=None):
        """Apply the linear map and optional FiLM modulation."""
        pre = self.W @ x + self.b
        if gamma is not None:
            pre = gamma * pre
        if beta is not None:
            pre = pre + beta
        return pre

    @abstractmethod
    def _activate(self, pre):
        """Apply the basis-specific activation to ``pre``."""
        raise NotImplementedError

    def __call__(self, x, *, gamma=None, beta=None):
        """Apply the layer to ``x`` with optional FiLM modulation.

        Args:
            x: Input vector of shape ``(in_dim,)``.
            gamma: Optional multiplicative modulation of shape ``(out_dim,)``.
            beta: Optional additive modulation of shape ``(out_dim,)``.

        Returns:
            Activated output of shape ``(out_dim,)``.
        """
        return self._activate(self._pre(x, gamma=gamma, beta=beta))


class SIRENLayer(Basis):
    """SIREN layer: ``sin(omega * pre)`` (Sitzmann+ 2020)."""

    # Explicit pass-through __init__ so pyright sees concrete signatures on
    # subclasses (eqx.Module + ABC machinery confuses static analysis).
    # DO NOT delete — see DECISIONS.md §"Polymorphism over discriminators".
    def __init__(self, in_dim, out_dim, omega_init, is_first, *, key):
        """Initialise the linear weights and the learnable ``omega``."""
        super().__init__(in_dim, out_dim, omega_init, is_first, key=key)

    def _activate(self, pre):
        return jnp.sin(self.omega * pre)


class HSIRENLayer(Basis):
    """H-SIREN layer: ``sin(omega * sinh(pre))`` (Cai & Pan 2024)."""

    # Explicit pass-through __init__ so pyright sees concrete signatures on
    # subclasses (eqx.Module + ABC machinery confuses static analysis).
    # DO NOT delete — see DECISIONS.md §"Polymorphism over discriminators".
    def __init__(self, in_dim, out_dim, omega_init, is_first, *, key):
        """Initialise the linear weights and the learnable ``omega``."""
        super().__init__(in_dim, out_dim, omega_init, is_first, key=key)

    def _activate(self, pre):
        return jnp.sin(self.omega * jnp.sinh(pre))


class WIRELayer(Basis):
    """WIRE layer: ``cos(omega * pre) * exp(-(s * pre) ** 2)`` (Saragadam+ 2023).

    Carries an additional learnable scalar ``s`` controlling the Gaussian
    window width. ``s`` lives only on this subclass — SIREN and H-SIREN
    pytrees do not contain an unused ``s`` leaf.
    """

    s: Float[Array, ""]

    def __init__(self, in_dim, out_dim, omega_init, is_first, *, key, s_init=3.0):
        """Initialise the linear weights, ``omega``, and the WIRE-specific ``s``."""
        super().__init__(in_dim, out_dim, omega_init, is_first, key=key)
        self.s = jnp.array(float(s_init))

    def _activate(self, pre):
        sz = self.s * pre
        return jnp.cos(self.omega * pre) * jnp.exp(-(sz * sz))


@runtime_checkable
class BasisModule(Protocol):
    """Public protocol any basis body conforms to.

    Two equally-valid ways to type against "any basis body" downstream:
    annotate with ``BasisModule`` (this Protocol, structural typing) or
    annotate with ``Body`` (the concrete public base, nominal typing).
    Use ``BasisModule`` for callers that want to accept duck-typed bodies
    (e.g. user-defined wrappers that don't subclass ``Body``); use
    ``Body`` when you specifically want a ``Body`` subclass.
    ``runtime_checkable`` lets callers use ``isinstance(body,
    BasisModule)`` at runtime; prefer static typing where possible.
    """

    out_features: int | None
    hidden_dim: int
    num_hidden_layers: int

    def trunk(self, coord, *, film=None) -> jax.Array:
        """Return pre-readout hidden features (shape ``(hidden_dim,)``)."""
        ...

    def __call__(self, coord, *, film=None) -> jax.Array:
        """Forward pass; scalar when ``out_features is None``, vector otherwise."""
        ...


def _validate_body_args(num_hidden_layers, out_features):
    """Shared constructor preconditions for the body classes.

    Returns ``None`` when ``out_features == 1`` (canonicalisation) so that the
    two scalar-yielding constructions produce identical pytrees.
    """
    assert num_hidden_layers >= 1, f"num_hidden_layers must be >= 1, got {num_hidden_layers}"
    assert out_features is None or (
        isinstance(out_features, int) and not isinstance(out_features, bool) and out_features >= 1
    ), f"out_features must be None or positive int, got {out_features!r}"
    return None if out_features == 1 else out_features


def _build_readout(hidden_dim, omega_hidden, out_features, key):
    """Sample the readout weights with the same SIREN-style bound used elsewhere."""
    # Bound applies per-output-component; independent of out_features.
    bound = jnp.sqrt(6.0 / hidden_dim) / max(omega_hidden, 1e-3)
    kw, kb = jax.random.split(key)
    out_dim = 1 if out_features is None else out_features
    readout_W = jax.random.uniform(kw, (out_dim, hidden_dim), minval=-bound, maxval=bound)
    readout_b = jax.random.uniform(kb, (out_dim,), minval=-bound, maxval=bound)
    return readout_W, readout_b


class Body(eqx.Module):
    """Public base class for any basis body.

    Subclass this directly to implement a new basis family with custom
    activation layers — see ``SIREN`` / ``HSIREN`` / ``WIRE`` for examples
    of the pattern (build a ``layers`` tuple of your own ``Basis`` subclass
    in ``__init__``, call ``_validate_body_args`` and ``_build_readout`` to
    satisfy the ``Body`` invariants, assign the structural fields).

    Symmetric with the ``Basis`` and ``Encoding`` ABCs — downstream consumers
    (e.g. ``loom`` renderers) can type-annotate against ``Body`` to accept
    *any* basis body, including user-defined ones, or against
    ``BasisModule`` for structural-typing flexibility.

    **Don't subclass the concrete bodies externally.** ``SIREN``, ``HSIREN``,
    ``WIRE`` are specific basis instantiations (sin / sinh-sin / Gabor); a
    new basis family (Gabor variants, hash-grid, learned Fourier features as
    activation, etc.) should subclass ``Body`` directly with its own
    ``Basis`` subclass, not subclass an existing concrete body whose
    semantics it doesn't share.

    Each body's ``__init__`` is written out explicitly rather than dispatched
    via a shared helper or class attribute — see the user-memory note
    "Repetition over confusing indirection" (2026-05-17).

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
    out_features: int | None = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    num_hidden_layers: int = eqx.field(static=True)

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


class HSIREN(Body):
    """Stack of ``HSIRENLayer`` s with an internal linear readout (Cai & Pan 2024)."""

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
        """Initialise the H-SIREN body MLP.

        Args:
            in_dim: Coordinate (input) dimension.
            hidden_dim: Width of each hidden layer.
            num_hidden_layers: Number of stacked ``HSIRENLayer`` s.
            key: JAX PRNG key.
            out_features: Readout width; see :class:`SIREN`.
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


class WIRE(Body):
    """Stack of ``WIRELayer`` s with an internal linear readout (Saragadam+ 2023).

    Accepts ``s_init`` explicitly (forwarded to each ``WIRELayer``) controlling
    the initial Gaussian window width. SIREN and H-SIREN do not.
    """

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
        s_init=3.0,
    ):
        """Initialise the WIRE body MLP.

        Args:
            in_dim: Coordinate (input) dimension.
            hidden_dim: Width of each hidden layer.
            num_hidden_layers: Number of stacked ``WIRELayer`` s.
            key: JAX PRNG key.
            out_features: Readout width; see :class:`SIREN`.
            omega_first: Initial frequency for the first (input) layer.
            omega_hidden: Initial frequency for subsequent layers.
            s_init: Initial WIRE Gaussian-window scalar (per layer).
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
