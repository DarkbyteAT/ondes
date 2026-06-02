"""BACON basis: Band-limited Coordinate Networks (Lindell+ 2022, CVPR).

BACON is an MFN with two band-limiting modifications:

1. Filter frequencies are sampled from a **discrete** quantised grid (not
   continuous Gaussian/uniform). Each layer's per-input-dim frequency is
   drawn uniformly from ``{-B_i, -B_i + dq, ..., B_i - dq, B_i}`` where
   ``dq`` is the quantisation interval and ``B_i`` is the per-layer
   bandwidth bound.
2. Filter frequencies are **non-trainable** (``eqx.field(static=True)`` not
   used — they're carried as fixed arrays). Only the recurrence linears,
   biases, and readout are learnable. Together with (1) this gives a
   provable analytic per-output bandwidth.

The per-layer bandwidth follows the paper's recommended schedule
``B_i = pi * (max_freq) / (num_hidden_layers + 1)``; the overall output
bandwidth is bounded by the *sum* across layers, which is the
``max_freq`` set at construction (matching BACON's official implementation:
github.com/computational-imaging/bacon, ``modules.py::MultiscaleBACON``).

Multiscale querying — BACON ships intermediate-layer outputs as
band-limited approximations — is not modelled here because the ``Body``
contract has a single output. A multiscale variant would expose ``trunk``
returns per scale; that's downstream-policy territory and can live as an
``examples/`` recipe if/when needed.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ondes.basis._base import Body, _validate_body_args
from ondes.basis.mfn import _mfn_recurrence_init


def _bacon_filter_freqs(in_dim, hidden_dim, bandwidth, quantization_interval, key):
    """Sample integer multiples of ``quantization_interval`` in ``[-bandwidth, bandwidth]``.

    Returns an ``(hidden_dim, in_dim)`` array of filter frequencies for one
    BACON layer. Each entry is uniformly drawn from the discrete set
    ``{-k * dq, ..., -dq, 0, dq, ..., k * dq}`` where ``k = bandwidth / dq``.
    This is the exact procedure in the reference implementation's
    ``FourierLayer.__init__``.
    """
    k = round(bandwidth / quantization_interval)
    if k <= 0:
        # Degenerate per-layer bandwidth (smaller than the quantisation step);
        # use a single integer at the zero frequency rather than crash.
        return jnp.zeros((hidden_dim, in_dim))
    # Draw integers in [0, 2k+1) then shift to [-k, k] inclusive.
    ints = jax.random.randint(key, (hidden_dim, in_dim), 0, 2 * k + 1)
    return (ints.astype(jnp.float32) - k) * quantization_interval


class BACONFilter(eqx.Module):
    """One band-limited Fourier filter for BACON (Lindell+ 2022).

    Carries:

    - ``W``: ``(hidden_dim, in_dim)`` of fixed integer-multiple frequencies.
      Treated as a buffer (no gradient): the analytic bandwidth proof depends
      on filter frequencies being fixed. Using ``eqx.field(static=False)`` and
      a plain ``jnp`` array keeps it inside the pytree (so it serialises and
      jit-traces cleanly) but the reference implementation calls
      ``requires_grad = False`` — in JAX/Equinox the equivalent at training
      time is to mask the W-tree with ``eqx.partition`` before the optimiser
      step. ``BACON.fix_filters_mask`` exposes that mask construction.
    - ``b``: ``(hidden_dim,)`` learnable bias (uniform-phase initialised on
      ``[-pi, pi]`` per the paper).
    """

    W: Float[Array, "hidden in"]
    b: Float[Array, "hidden"]

    def __init__(self, in_dim, hidden_dim, bandwidth, quantization_interval, *, key):
        """Sample fixed quantised frequencies and a uniform-phase learnable bias."""
        k_freq, k_bias = jax.random.split(key)
        self.W = _bacon_filter_freqs(in_dim, hidden_dim, bandwidth, quantization_interval, k_freq)
        self.b = jax.random.uniform(k_bias, (hidden_dim,), minval=-jnp.pi, maxval=jnp.pi)

    def __call__(self, x):
        """Apply the band-limited sinusoidal filter pointwise."""
        return jnp.sin(self.W @ x + self.b)


class BACON(Body):
    """Band-limited MFN body (Lindell+ 2022).

    Per-layer filter bandwidth defaults to ``pi * max_freq / (num_hidden_layers + 1)``,
    matching the reference implementation. With this schedule the overall
    output bandwidth is bounded by ``max_freq`` (the per-layer caps sum
    along the multiplicative recurrence).

    Args:
        in_dim: Coordinate (input) dimension.
        hidden_dim: Width of the recurrence and each filter.
        num_hidden_layers: Number of recurrence steps (the network has
            ``num_hidden_layers + 1`` filters).
        key: JAX PRNG key.
        out_features: Readout width; see :class:`ondes.basis.siren.SIREN`.
        max_freq: Target overall output bandwidth (cycles / coord unit).
            Paper uses ``max_freq = 256`` for natural-image fitting.
        quantization_interval: Spacing of the discrete frequency grid.
            Paper uses ``2 * pi`` (frequencies sampled at integer cycles
            per unit coord). Setting it smaller increases representable
            frequency resolution at the cost of analytic-bandwidth
            tightness.
        weight_scale: Scale on the recurrence-linear weight uniform-init
            bound (same as MFN).

    Note:
        BACON's analytic-bandwidth proof requires the filter frequencies
        (``filters[i].W``) to be *non-trainable*. ``ondes`` does not own
        optimiser plumbing, so this is enforced by the user with
        ``eqx.partition`` (or ``optax.masked``). The
        ``fix_filters_mask()`` classmethod returns the mask pytree
        downstream code can use directly.
    """

    filters: tuple
    recurrence_W: Float[Array, "n_layers hidden hidden"]
    recurrence_b: Float[Array, "n_layers hidden"]
    bandwidths: Float[Array, "n_filters"]

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_hidden_layers,
        *,
        key,
        out_features=None,
        max_freq=256.0,
        quantization_interval=None,
        weight_scale=1.0,
    ):
        """Initialise band-limited filters at the paper-recommended per-layer cap."""
        out_features = _validate_body_args(num_hidden_layers, out_features)
        if quantization_interval is None:
            quantization_interval = 2.0 * float(jnp.pi)
        n_filters = num_hidden_layers + 1
        per_layer_bandwidth = float(jnp.pi) * float(max_freq) / (num_hidden_layers + 1.0)
        # Quantise the per-layer bandwidth to a clean multiple of the interval
        # (matches the reference's `round(... / dq) * dq` line).
        per_layer_bandwidth = round(per_layer_bandwidth / quantization_interval) * quantization_interval

        keys = jax.random.split(key, n_filters + num_hidden_layers + 1)
        filter_keys = keys[:n_filters]
        rec_keys = keys[n_filters : n_filters + num_hidden_layers]
        readout_key = keys[-1]

        self.filters = tuple(
            BACONFilter(in_dim, hidden_dim, per_layer_bandwidth, quantization_interval, key=k) for k in filter_keys
        )
        # Bandwidth per filter is the same in the default schedule; carry it as
        # an array so the analytic cap is recoverable from the body alone.
        self.bandwidths = jnp.full((n_filters,), float(per_layer_bandwidth))

        Ws, bs = [], []
        for i in range(num_hidden_layers):
            W, b = _mfn_recurrence_init(hidden_dim, hidden_dim, weight_scale, rec_keys[i])
            Ws.append(W)
            bs.append(b)
        self.recurrence_W = jnp.stack(Ws)
        self.recurrence_b = jnp.stack(bs)

        out_dim = 1 if out_features is None else out_features
        rw, rb = _mfn_recurrence_init(hidden_dim, out_dim, weight_scale, readout_key)
        self.readout_W = rw
        self.readout_b = rb
        self.layers = ()
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

    @property
    def output_bandwidth(self) -> jax.Array:
        """Analytic upper bound on the network's output frequency content.

        Equal to the sum of per-filter bandwidths along the multiplicative
        recurrence (paper Section 3, "bandwidth analysis"). For the default
        schedule this equals ``max_freq`` (within one quantisation step).

        Returned as a JAX scalar so the property is safe to call inside a
        ``jit``-compiled function (e.g. as part of a bandwidth-regularised
        loss). Cast to Python ``float`` at the call site when an eager
        diagnostic value is wanted.
        """
        return jnp.sum(self.bandwidths)

    def fix_filters_mask(self):
        """Return a pytree mask that selects only the learnable leaves.

        Use with ``eqx.partition(body, body.fix_filters_mask())`` to split the
        body into ``(learnable, fixed)`` halves before applying an optimiser
        update — the fixed half must not receive gradient steps for the
        analytic bandwidth proof to hold.

        Two kinds of leaves end up in the fixed half:

        - Each filter's ``W`` (the discrete-grid frequencies). Required by the
          analytic bandwidth proof.
        - The body-level ``bandwidths`` array (the per-filter cap diagnostic).
          Not a learnable parameter — it's a constant carried alongside the
          filter weights so the analytic cap is recoverable from the body
          alone. Marked ``False`` here so an Adam-style optimiser doesn't
          allocate momentum / second-moment state for it.
        """
        # Start from "everything is learnable", then zero out filter Ws and bandwidths.
        mask = jax.tree_util.tree_map(lambda x: eqx.is_array(x), self)
        # Walk filters: each filter's W is fixed, its b is learnable.
        new_filters = tuple(eqx.tree_at(lambda f: f.W, m, False) for m in mask.filters)
        mask = eqx.tree_at(lambda b: b.filters, mask, new_filters)
        return eqx.tree_at(lambda b: b.bandwidths, mask, False)

    def trunk(self, coord, *, film=None):
        """BACON multiplicative recurrence — identical shape to MFN's.

        Inherits the same FiLM convention: the first filter is unmodulated;
        per-step ``gamma, beta`` rows gate the post-recurrence-linear output.
        """
        self._check_film_shape(film)
        z = self.filters[0](coord)
        for i in range(self.num_hidden_layers):
            pre = self.recurrence_W[i] @ z + self.recurrence_b[i]
            if film is not None:
                gamma = film[i, : self.hidden_dim]
                beta = film[i, self.hidden_dim :]
                pre = gamma * pre + beta
            z = self.filters[i + 1](coord) * pre
        return z
