"""FINER basis: variable-periodic activation (Liu+ 2024, CVPR).

FINER (`sin(omega * (|pre| + 1) * pre)`) extends SIREN by gating the carrier
frequency with the running pre-activation magnitude. The activation expands
each unit's effective frequency range from a fixed ``[-omega, omega]`` to
``[-omega * (|pre|+1), omega * (|pre|+1)]``, and the per-unit ``|pre|+1``
multiplier is signal-dependent — different signals select different
sub-bands of the variable-periodic function.

Reference implementation: github.com/liuzhen0212/FINER/blob/main/models.py
(``FinerLayer`` class).

Two FINER-specific knobs beyond SIREN's:

- ``first_bias_scale`` — the first-layer bias is initialised uniformly on
  ``[-first_bias_scale, +first_bias_scale]`` instead of SIREN's
  ``[-1/in_dim, +1/in_dim]``. Setting it ``> 1`` ensures the first-layer
  pre-activation magnitudes start large enough to activate the high-frequency
  sub-bands. Paper Section 3 uses values in ``{1, 5, 10, 20}``; default
  ``5`` here matches the paper's natural-image experiments.
- ``scale_req_grad`` — whether the ``|pre| + 1`` factor accumulates gradient.
  Paper default is ``False`` (``jax.lax.stop_gradient`` applied), which
  decouples the gradient signal from the magnitude-gating term.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from ondes.basis._base import Basis, Body, _validate_body_args
from ondes.basis.siren import _build_readout, siren_init


class FINERLayer(Basis):
    """FINER layer: ``sin(omega * (|pre| + 1) * pre)`` (Liu+ 2024).

    Carries the same fields as ``SIRENLayer``. The first-layer bias is the
    only init-scheme difference (constructor argument ``first_bias_scale``).

    ``scale_req_grad`` is a construction-time-only argument that wraps
    ``|pre| + 1`` in ``jax.lax.stop_gradient`` when ``False`` (the paper
    default). It is stored as a static field so the forward pass can branch
    on it without re-tracing; pytree-structural homogeneity across layers
    in one body is preserved because all layers in a body share the same
    ``scale_req_grad`` value (set by ``FINER.__init__``).
    """

    scale_req_grad: bool = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        omega_init: float,
        is_first: bool,
        *,
        key: Key[Array, ""],
        first_bias_scale: float = 5.0,
        scale_req_grad: bool = False,
    ) -> None:
        """Initialise SIREN-style weights, then optionally widen the first-layer bias.

        ``first_bias_scale`` is only consumed when ``is_first`` is True; it has
        no effect on hidden layers (which keep SIREN's ``sqrt(6/in)/omega``
        bias bound). The asymmetry is intentional — the paper places the
        magnitude-boost trick exclusively on the first layer so the
        downstream layers see a wide pre-activation distribution without
        themselves being initialised at high frequencies.

        The wide-bias trick is applied by rescaling the SIREN-drawn ``b_siren``
        rather than re-sampling from the input ``key``. The SIREN first-layer
        bias is uniform on ``[-1/in_dim, +1/in_dim]``, so
        ``b_siren * (first_bias_scale * in_dim)`` is the exact same uniform
        distribution on ``[-first_bias_scale, +first_bias_scale]`` — no extra
        ``jax.random.split`` and no risk of key reuse / cross-leaf correlation
        from re-deriving a second key from one that ``siren_init`` already
        consumed.
        """
        self.W, b_siren = siren_init(in_dim, out_dim, omega_init, is_first, key)
        if is_first:
            self.b = b_siren * (float(first_bias_scale) * in_dim)
        else:
            self.b = b_siren
        self.omega = jnp.array(float(omega_init))
        self.scale_req_grad = bool(scale_req_grad)

    def _activate(self, pre: Float[Array, "out"]) -> Float[Array, "out"]:
        """Apply ``sin(omega * (|pre| + 1) * pre)`` with optional stop-gradient on the scale."""
        scale = jnp.abs(pre) + 1.0
        if not self.scale_req_grad:
            scale = jax.lax.stop_gradient(scale)
        return jnp.sin(self.omega * scale * pre)


class FINER(Body):
    """Stack of ``FINERLayer`` s with an internal linear readout (Liu+ 2024).

    Args:
        in_dim: Coordinate (input) dimension.
        hidden_dim: Width of each hidden layer.
        num_hidden_layers: Number of stacked ``FINERLayer`` s.
        key: JAX PRNG key.
        out_features: Readout width; see :class:`ondes.basis.siren.SIREN`.
        omega_first: Initial frequency for the first (input) layer. Paper
            uses the SIREN-standard ``omega_first = 30``.
        omega_hidden: Initial frequency for subsequent layers. Default
            ``1.0`` matches the SIREN convention; the variable-periodic
            activation does most of the spectral work, so hidden layers can
            run at unit ``omega`` without losing high-frequency coverage.
        first_bias_scale: Width of the first-layer bias uniform-init
            interval. Paper Section 3 tries ``{1, 5, 10, 20}``; default
            ``5.0`` matches the natural-image experiments.
        scale_req_grad: Whether the ``|pre|+1`` factor accumulates gradient.
            Paper default ``False`` (stop_gradient applied).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        *,
        key: Key[Array, ""],
        out_features: int | None = None,
        omega_first: float = 30.0,
        omega_hidden: float = 1.0,
        first_bias_scale: float = 5.0,
        scale_req_grad: bool = False,
    ) -> None:
        """Initialise the FINER body MLP."""
        out_features = _validate_body_args(num_hidden_layers, out_features)
        keys = jax.random.split(key, num_hidden_layers + 1)
        layers = []
        for i in range(num_hidden_layers):
            in_d = in_dim if i == 0 else hidden_dim
            o = omega_first if i == 0 else omega_hidden
            layers.append(
                FINERLayer(
                    in_d,
                    hidden_dim,
                    o,
                    is_first=(i == 0),
                    key=keys[i],
                    first_bias_scale=first_bias_scale,
                    scale_req_grad=scale_req_grad,
                )
            )
        self.layers = tuple(layers)
        self.readout_W, self.readout_b = _build_readout(hidden_dim, omega_hidden, out_features, keys[-1])
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
