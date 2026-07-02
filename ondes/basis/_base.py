"""Basis-agnostic ABCs, protocol, and shared validation.

Lives at the base of the ``ondes.basis`` package so concrete basis modules
(``siren``, ``hsiren``, ``wire``, and any future additions) can build on the
same ``Body`` / ``Basis`` contracts without circular imports.
"""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
from jaxtyping import Array, Float


class Basis(eqx.Module):
    """ABC for a single basis-MLP layer.

    Holds the linear weights ``(W, b)`` and a frequency scalar ``omega`` carried
    by every basis. Most bases learn ``omega`` and read it in ``_activate``
    (SIREN's ``sin(omega * pre)``); some treat it as an init-only scale the
    forward pass never reads (RFF's placeholder ``1.0``, STAF's frozen
    ``omega_0``). Subclasses provide ``_activate(pre)`` to produce the
    post-activation output. Basis-specific fields (e.g. ``s`` for WIRE) live on
    the concrete subclass, not here.

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

    def _pre(
        self,
        x: Float[Array, "in"],
        gamma: Float[Array, "out"] | None = None,
        beta: Float[Array, "out"] | None = None,
    ) -> Float[Array, "out"]:
        """Apply the linear map and optional FiLM modulation."""
        pre = self.W @ x + self.b
        if gamma is not None:
            pre = gamma * pre
        if beta is not None:
            pre = pre + beta
        return pre

    @abstractmethod
    def _activate(self, pre: Float[Array, "out"]) -> Float[Array, "out"]:
        """Apply the basis-specific activation to ``pre``."""
        raise NotImplementedError

    def __call__(
        self,
        x: Float[Array, "in"],
        *,
        gamma: Float[Array, "out"] | None = None,
        beta: Float[Array, "out"] | None = None,
    ) -> Float[Array, "out"]:
        """Apply the layer to ``x`` with optional FiLM modulation.

        Args:
            x: Input vector of shape ``(in_dim,)``.
            gamma: Optional multiplicative modulation of shape ``(out_dim,)``.
            beta: Optional additive modulation of shape ``(out_dim,)``.

        Returns:
            Activated output of shape ``(out_dim,)``.
        """
        return self._activate(self._pre(x, gamma=gamma, beta=beta))


@runtime_checkable
class BasisModule(Protocol):
    """Public protocol any basis body conforms to.

    Two equally-valid ways to type against "any basis body" downstream:
    annotate with ``BasisModule`` (this Protocol, structural typing) or
    annotate with ``Body`` (the concrete public base, nominal typing).
    Use ``BasisModule`` for callers that want to accept duck-typed bodies
    (e.g. user-defined wrappers that don't subclass ``Body``); use
    ``Body`` when you specifically want a ``Body`` subclass.
    ``runtime_checkable`` lets callers use ``isinstance(body, BasisModule)``
    at runtime; prefer static typing where possible.
    """

    out_features: int | None
    hidden_dim: int
    num_hidden_layers: int

    def trunk(self, coord: Float[Array, "in"], *, film: Float[Array, "n_layers two_hidden"] | None = None) -> jax.Array:
        """Return pre-readout hidden features (shape ``(hidden_dim,)``)."""
        ...

    def __call__(
        self, coord: Float[Array, "in"], *, film: Float[Array, "n_layers two_hidden"] | None = None
    ) -> jax.Array:
        """Forward pass; scalar when ``out_features is None``, vector otherwise."""
        ...


def _validate_body_args(num_hidden_layers: int, out_features: int | None) -> int | None:
    """Shared constructor preconditions for the body classes.

    Returns ``None`` when ``out_features == 1`` (canonicalisation) so that the
    two scalar-yielding constructions produce identical pytrees.

    Args:
        num_hidden_layers: Number of stacked hidden layers; must be ``>= 1``.
        out_features: Readout width or ``None`` for scalar output.

    Returns:
        ``None`` if ``out_features`` is ``None`` or ``1``; otherwise the
        unchanged integer ``out_features``.
    """
    assert num_hidden_layers >= 1, f"num_hidden_layers must be >= 1, got {num_hidden_layers}"
    assert out_features is None or (
        isinstance(out_features, int) and not isinstance(out_features, bool) and out_features >= 1
    ), f"out_features must be None or positive int, got {out_features!r}"
    return None if out_features == 1 else out_features


class Body(eqx.Module):
    """Public base class for any basis body.

    Subclass this directly to implement a new basis family with custom
    activation layers — see ``SIREN`` / ``HSIREN`` / ``WIRE`` for examples
    of the pattern (build a ``layers`` tuple of your own ``Basis`` subclass
    in ``__init__``, call ``_validate_body_args`` and a readout-init helper
    to satisfy the ``Body`` invariants, assign the structural fields).

    Symmetric with the ``Basis`` ABC — downstream consumers (e.g. ``loom``
    renderers) can type-annotate against ``Body`` to accept *any* basis body,
    including user-defined ones, or against ``BasisModule`` for structural-typing
    flexibility.

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

    layers: tuple[Basis, ...]
    readout_W: Float[Array, "out hidden"]
    readout_b: Float[Array, "out"]
    out_features: int | None = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    num_hidden_layers: int = eqx.field(static=True)

    def _check_film_shape(self, film: Float[Array, "n_layers two_hidden"] | None) -> None:
        """Raise ``ValueError`` if ``film`` doesn't match the expected shape.

        Expected shape is ``(num_hidden_layers, 2 * hidden_dim)`` — the first
        half of the trailing axis is ``gamma``, the second half is ``beta``.
        Called at the top of every concrete ``trunk`` implementation so the
        FiLM contract has one source of truth across the basis family
        (``Body.trunk`` for SIREN/HSIREN/WIRE, plus the MFN/PNF/BACON/RFF
        overrides). ``ValueError`` rather than ``assert`` because the shape
        contract is part of the user-facing API and must survive
        ``python -O`` (asserts get stripped under optimisation).
        """
        if film is None:
            return
        expected = (self.num_hidden_layers, 2 * self.hidden_dim)
        if film.shape != expected:
            raise ValueError(f"film must have shape {expected}, got {film.shape}")

    def trunk(
        self,
        coord: Float[Array, "in"],
        *,
        film: Float[Array, "n_layers two_hidden"] | None = None,
    ) -> Float[Array, "hidden"]:
        """Return pre-readout hidden features.

        Args:
            coord: Coordinate vector of shape ``(in_dim,)``.
            film: Optional FiLM tensor of shape
                ``(num_hidden_layers, 2 * hidden_dim)`` whose halves provide
                ``gamma`` and ``beta`` per layer. ``None`` skips modulation.

        Returns:
            Activations of the final hidden layer, shape ``(hidden_dim,)``.
        """
        self._check_film_shape(film)
        h = coord
        for i, layer in enumerate(self.layers):
            if film is not None:
                gamma = film[i, : self.hidden_dim]
                beta = film[i, self.hidden_dim :]
                h = layer(h, gamma=gamma, beta=beta)
            else:
                h = layer(h)
        return h

    def _readout(self, h: Float[Array, "hidden"]) -> Float[Array, "out"]:
        """Internal linear readout. Not a user extension point."""
        return self.readout_W @ h + self.readout_b

    def __call__(
        self,
        coord: Float[Array, "in"],
        *,
        film: Float[Array, "n_layers two_hidden"] | None = None,
    ) -> jax.Array:
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
