"""WINNER basis: SIREN with target-aware noise on the first two weights.

Implements *Weight Initialization with Noise for nEural Representations*
(WINNER), arXiv 2509.09719 and reference implementation
``cfdlabtechnion/siren_square`` (`SIREN_square` class in
``modules/networks.py``). WINNER perturbs the **first two** linear-layer
weights of a SIREN body with i.i.d. Gaussian noise scaled by the target
signal's spectral centroid; the rest of the architecture and forward pass
are identical to :class:`~ondes.basis.siren.SIREN`.

The perturbation is target-aware: per-signal scales ``s0, s1`` are derived
from ``spectral_centroid(signal) / n_channels`` through a
:class:`WinnerSchedule` value type. The two reference regimes — audio
(1-D signals) and images (2-D) — are exposed as
:meth:`WinnerSchedule.audio` and :meth:`WinnerSchedule.image` classmethod
factories; downstream callers either pick a factory or build their own
:class:`WinnerSchedule` instance with explicit numbers.

See ``WINNER_DECISIONS.md`` at the repo root for pinned design choices
(omega divisor, centroid double-divide, ``reset_noise`` semantics, bias
init inheritance).
"""

from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from ondes.basis._base import Basis, Body, _validate_body_args
from ondes.basis.siren import SIRENLayer, _build_readout


class WinnerSchedule(eqx.Module):
    """Centroid → ``(s0, s1)`` noise-scale schedule for :class:`WINNER`.

    The reference implementation hardcodes two regime-specific empirical
    formulas; this type lifts them to a frozen pytree value, picked from a
    classmethod factory or constructed by hand. Per
    ``feedback_library_defaults_vs_canonical``, we expose factories rather
    than module-level constants so users have to *name* the regime at the
    call site (and the citation stays adjacent to the numbers).

    Fields:
        s0_max: Saturation amplitude of the ``s0`` schedule.
        a: Exponential-saturation rate of the ``s0`` schedule.
        b: Linear coefficient of the ``s1`` schedule.

    The schedule itself is:
        ``s0 = s0_max · (1 - exp(-a · z))``
        ``s1 = b · z``
    where ``z = centroid / n_channels``. The extra ``/ n_channels`` here is
    intentional: :func:`spectral_centroid` already returns
    ``(Σ f·|X(f)|/Σ|X(f)|) · 2 / n_channels``, so the schedule sees
    ``2 · (channel-averaged-per-slice-mean) / n_channels²``. The reference
    constants are calibrated to that net normalisation — do not collapse
    the double divide. See ``WINNER_DECISIONS.md`` item 2.
    """

    s0_max: float
    a: float
    b: float

    def scales(
        self,
        centroid: Float[Array, ""],
        n_channels: int,
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        """Return the per-signal ``(s0, s1)`` for the given centroid.

        Args:
            centroid: Output of :func:`spectral_centroid` for the target.
            n_channels: Channel count of the target signal (the trailing
                channel axis used in the centroid computation).

        Returns:
            ``(s0, s1)`` as a tuple of 0-d JAX arrays. dtype matches
            ``centroid``.
        """
        z = centroid / n_channels
        s0 = self.s0_max * (1.0 - jnp.exp(-self.a * z))
        s1 = self.b * z
        return s0, s1

    @classmethod
    def audio(cls) -> "WinnerSchedule":
        """Audio (1-D signal) schedule from the reference implementation.

        Source: ``modules/networks.py`` ``set_noise_scales`` branch
        ``if self.in_dim == 1`` — ``a, b = 7, 3``, ``S0 = 3500 · (1 -
        exp(-a · z))``. Note the paper *text* says ``a = 5``; the code
        uses ``a = 7``. We follow the code (see
        ``WINNER_DECISIONS.md`` item 3).
        """
        return cls(s0_max=3500.0, a=7.0, b=3.0)

    @classmethod
    def image(cls) -> "WinnerSchedule":
        """Image (2-D signal) schedule from the reference implementation.

        Source: ``modules/networks.py`` ``set_noise_scales`` branch
        ``if self.in_dim == 2`` — ``a, b = 5, 0.4``, ``S0 = 50 · (1 -
        exp(-a · z))``. Paper text and code agree for the image regime.
        """
        return cls(s0_max=50.0, a=5.0, b=0.4)


def spectral_centroid(
    signal: Float[Array, "..."],
    *,
    freq_axis: int = -2,
    channel_axis: int = -1,
) -> Float[Array, ""]:
    """Channel-averaged spectral centroid of ``signal``, normalised by ``n_ch``.

    Matches the reference (``modules/networks.py`` ``spectral_centroid``)
    per-rank branches by collapsing to a single rank-agnostic operation:

    1. rFFT along ``freq_axis`` and take magnitudes
    2. weight bins by ``rfftfreq(n_freq, d=1)`` and reduce **only** along
       ``freq_axis`` to form a per-slice numerator and denominator
    3. compute the per-slice centroid as ``num / den`` (zero-safe)
    4. mean over all remaining non-channel axes → one centroid per channel
    5. mean across channels
    6. multiply by 2, divide by ``n_channels``

    The per-slice-then-average step (3 → 4) is intentional: the reference's
    2-D image branch (``ndim==2``) sums along the row axis to get a
    per-row centroid, then ``np.mean`` over rows. Summing globally first
    (``Σf·|X| / Σ|X|`` over all axes) gives an energy-weighted average
    that biases toward high-amplitude rows, deviating from the reference
    by a few percent on row-scaled toys.

    The schedule (:meth:`WinnerSchedule.scales`) divides by
    ``n_channels`` *again* — the double-divide is intentional and the
    reference schedule constants are calibrated to it; see
    ``WINNER_DECISIONS.md`` item 2.

    Args:
        signal: Real-valued signal of any rank. The frequency axis and the
            channel axis must both be present.
        freq_axis: Axis along which to take rFFT. Defaults to ``-2``, the
            "image-row" convention from the reference (for a ``(H, W,
            C)`` image: rFFT along ``W``).
        channel_axis: Axis treated as channels for the final average and
            normalisation. Defaults to ``-1``.

    Returns:
        Scalar centroid, dtype matching ``signal``. Returns ``0`` for an
        all-zero spectrum (zero-safe divide).

    Note:
        The reference NumPy code hardcodes per-rank branches (1-D audio,
        2-D image, 3-D volume); this JAX version is rank-agnostic — pass
        whatever ``freq_axis``/``channel_axis`` matches your signal's
        layout. For 1-D mono audio with shape ``(n_samples,)``, reshape
        to ``(n_samples, 1)`` first so the channel axis is explicit, then
        call with ``freq_axis=0`` (not the default, which points at the
        channel axis). Defaults assume the trailing-channel layout
        ``(..., n_freq, n_ch)``.
    """
    if signal.ndim < 2:
        raise ValueError(
            "spectral_centroid expects signal.ndim >= 2 (frequency axis + "
            f"channel axis); got shape {signal.shape}. For 1-D mono audio, "
            "reshape to (n_samples, 1) first."
        )

    n_channels = signal.shape[channel_axis]
    n_freq = signal.shape[freq_axis]

    # rFFT along the frequency axis, magnitude spectrum
    spectrum = jnp.abs(jnp.fft.rfft(signal, axis=freq_axis))
    # rfftfreq with d=1 matches the reference convention; bind dtype to
    # signal.dtype so we don't weak-promote to float64 under jax_enable_x64.
    freq_bins = jnp.fft.rfftfreq(n_freq, d=1.0).astype(signal.dtype)

    # Broadcast freq_bins along the rfft'd axis. After rfft along freq_axis,
    # that axis has length n_freq // 2 + 1; build a shape with 1s everywhere
    # except the freq axis.
    freq_shape = [1] * spectrum.ndim
    freq_shape[freq_axis] = freq_bins.shape[0]
    freq_bins_b = freq_bins.reshape(freq_shape)

    weighted = spectrum * freq_bins_b

    # Reduce along the rfft axis ONLY — gives a per-slice (num, den) pair.
    # Reference's 2-D branch sums along axis=1 to get one centroid per row,
    # then averages rows; the 3-D branch sums along axis=2 to get one
    # centroid per slice, then averages slices. We collapse both into a
    # single per-freq-axis reduction followed by a mean across all other
    # non-channel axes (i.e. rows for 2-D, slices for 3-D).
    num = jnp.sum(weighted, axis=freq_axis)
    den = jnp.sum(spectrum, axis=freq_axis)
    per_slice = jnp.where(den != 0, num / jnp.where(den != 0, den, 1.0), 0.0)

    # After the freq_axis reduction the spectrum has spectrum.ndim - 1 axes.
    # The original channel_axis may now sit at a different position; resolve
    # it into a non-negative index in the original ndim, then shift by 1 if
    # freq_axis was earlier in the ordering.
    norm_channel_axis = channel_axis % spectrum.ndim
    norm_freq_axis = freq_axis % spectrum.ndim
    new_channel_axis = norm_channel_axis if norm_channel_axis < norm_freq_axis else norm_channel_axis - 1

    remaining_axes = tuple(ax for ax in range(per_slice.ndim) if ax != new_channel_axis)
    per_channel = jnp.mean(per_slice, axis=remaining_axes) if remaining_axes else per_slice

    centroid = jnp.mean(per_channel)
    two = jnp.asarray(2.0, signal.dtype)
    return cast(Float[Array, ""], (centroid * two) / n_channels)


def _per_layer_keys(k_layers: Key[Array, ""], num_hidden_layers: int) -> tuple[Key[Array, ""], ...]:
    """Split ``k_layers`` into one subkey per hidden layer.

    Single-sourced so :meth:`WINNER.__init__` and :meth:`WINNER.reset_noise`
    derive the *same* per-layer subkey stream from the same ``k_layers``.
    Without this helper, a future refactor that changes the split count in
    one path but not the other would silently desync the
    "``reset_noise(k)`` reproduces the original ``WINNER(key=k)`` for
    layers 0/1" contract — the regression test for the double-perturbation
    trap would still pass for layers 0/1 only by coincidence, because both
    sides happen to land on the same subkey by index. Pinned helper.
    """
    return tuple(jax.random.split(k_layers, num_hidden_layers))


def _perturb_first_two_layers(
    layers: tuple[Basis, ...],
    s0: Float[Array, ""],
    s1: Float[Array, ""],
    omega_hidden: float,
    key: Key[Array, ""],
) -> tuple[Basis, ...]:
    """Add Gaussian noise to layers 0 and 1's ``W`` only.

    Noise scale is ``s_i / omega_hidden`` for *both* layers (not
    ``omega_first`` for layer 0 — see ``WINNER_DECISIONS.md`` item 1).
    Biases are untouched. Layers ``>= 2`` are returned unchanged.

    Noise dtype is bound to the layer's weight dtype; the caller is
    responsible for passing ``s0``/``s1`` already in that dtype (the
    multiplication is then float-safe under x64).
    """
    k0, k1 = jax.random.split(key)
    scale0 = s0 / omega_hidden
    scale1 = s1 / omega_hidden
    noise0 = jax.random.normal(k0, layers[0].W.shape, dtype=layers[0].W.dtype) * scale0
    noise1 = jax.random.normal(k1, layers[1].W.shape, dtype=layers[1].W.dtype) * scale1
    layer0 = eqx.tree_at(lambda layer: layer.W, layers[0], layers[0].W + noise0)
    layer1 = eqx.tree_at(lambda layer: layer.W, layers[1], layers[1].W + noise1)
    return (layer0, layer1, *layers[2:])


class WINNER(Body):
    """SIREN body with target-aware noise on the first two layer weights.

    Reference: arXiv 2509.09719, ``cfdlabtechnion/siren_square``.

    Construction paths:

    - :meth:`__init__` — explicit ``(s0, s1)``. Target-agnostic. Use when
      you have noise scales from somewhere other than a fresh centroid
      computation (e.g. holding scales fixed across a seed sweep).
    - :meth:`from_signal` classmethod — derives ``(s0, s1)`` from
      ``schedule.scales(spectral_centroid(signal), n_channels)``. Target-
      aware. Use when fitting a single target.

    The forward pass and readout machinery are inherited from
    :class:`~ondes.basis._base.Body` — WINNER's only contribution is the
    init scheme. WINNER is a sibling of :class:`~ondes.basis.siren.SIREN`
    (not a subclass) for the reasons pinned in
    ``WINNER_DECISIONS.md`` item 5.

    Bias init: WINNER inherits ondes' :func:`~ondes.basis.siren.siren_init`
    bias convention — both ``W`` and ``b`` are drawn from the same
    ``U(-bound, +bound)`` per layer (where ``bound = 1/in_dim`` on layer 0
    and ``sqrt(6/in_dim)/omega`` elsewhere). This is a known divergence
    from the PyTorch reference, which uses ``nn.Linear``'s default bias
    init (``U(-1/sqrt(in_dim), +1/sqrt(in_dim))``). The divergence is
    inherited from the existing SIREN, not introduced by WINNER. See
    ``WINNER_DECISIONS.md`` item 8.
    """

    s0: Float[Array, ""]
    s1: Float[Array, ""]
    omega_first: float = eqx.field(static=True)
    omega_hidden: float = eqx.field(static=True)
    in_dim: int = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        *,
        key: Key[Array, ""],
        s0: float | Float[Array, ""],
        s1: float | Float[Array, ""],
        out_features: int | None = None,
        omega_first: float = 30.0,
        omega_hidden: float = 30.0,
    ) -> None:
        """Initialise a WINNER body with explicit noise scales.

        Args:
            in_dim: Coordinate (input) dimension.
            hidden_dim: Width of each hidden layer.
            num_hidden_layers: Number of stacked ``SIRENLayer`` s. Must be
                ``>= 2`` (WINNER perturbs layers 0 and 1).
            key: JAX PRNG key.
            s0: Noise scale for layer 0's weight perturbation. Applied as
                ``s0 / omega_hidden`` Gaussian std.
            s1: Noise scale for layer 1's weight perturbation. Applied as
                ``s1 / omega_hidden`` Gaussian std.
            out_features: Readout width; see :class:`~ondes.basis.siren.SIREN`.
            omega_first: Initial frequency for the first (input) layer.
                Paper-canonical default ``30.0``.
            omega_hidden: Initial frequency for subsequent layers; also
                the divisor in the perturbation noise scale for *both*
                layers (see ``WINNER_DECISIONS.md`` item 1). Paper-
                canonical default ``30.0``.

        Raises:
            ValueError: if ``num_hidden_layers < 2``. WINNER perturbs
                layers 0 and 1, so it needs at least two of them. Raised
                rather than ``assert`` so the precondition survives
                ``python -O`` (which strips assertions); the rest of
                ``ondes.basis._base`` uses the same ``ValueError`` pattern
                (see ``_check_film_shape``).
        """
        if num_hidden_layers < 2:
            raise ValueError(f"WINNER perturbs layers 0 and 1; need num_hidden_layers >= 2, got {num_hidden_layers}")
        out_features = _validate_body_args(num_hidden_layers, out_features)

        # Bind s0/s1 to weight dtype downstream by deferring asarray to
        # whatever dtype the noise weight uses (set in _perturb_first_two_layers).
        # Here we only ensure we have an Array — let JAX pick the dtype so the
        # user can opt into float64 by enabling x64 globally; no silent downcast.
        s0_arr = s0 if isinstance(s0, jax.Array) else jnp.asarray(s0)
        s1_arr = s1 if isinstance(s1, jax.Array) else jnp.asarray(s1)

        # Three-way master split: layers, readout, noise. Keeping the split
        # named-by-purpose (rather than positional in a flat array) is what
        # lets reset_noise reproduce the same layers 0/1 from the same master
        # key — see WINNER_DECISIONS.md item 4 and the reset_noise docstring.
        k_layers, k_readout, k_noise = jax.random.split(key, 3)
        per_layer_keys = _per_layer_keys(k_layers, num_hidden_layers)

        layers_list: list[SIRENLayer] = []
        for i in range(num_hidden_layers):
            in_d = in_dim if i == 0 else hidden_dim
            o = omega_first if i == 0 else omega_hidden
            layers_list.append(SIRENLayer(in_d, hidden_dim, o, is_first=(i == 0), key=per_layer_keys[i]))
        layers = tuple(layers_list)
        # Cast s0/s1 to the weight dtype right at the perturbation site so the
        # mul with the weight-dtype noise is dtype-safe under x64.
        weight_dtype = layers[0].W.dtype
        layers = _perturb_first_two_layers(
            layers,
            s0_arr.astype(weight_dtype),
            s1_arr.astype(weight_dtype),
            omega_hidden,
            k_noise,
        )

        self.layers = layers
        self.readout_W, self.readout_b = _build_readout(hidden_dim, omega_hidden, out_features, k_readout)
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.s0 = s0_arr
        self.s1 = s1_arr
        self.omega_first = omega_first
        self.omega_hidden = omega_hidden
        self.in_dim = in_dim

    @classmethod
    def from_signal(
        cls,
        signal: Float[Array, "..."],
        schedule: WinnerSchedule,
        *,
        in_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        key: Key[Array, ""],
        out_features: int | None = None,
        omega_first: float = 30.0,
        omega_hidden: float = 30.0,
        freq_axis: int = -2,
        channel_axis: int = -1,
    ) -> "WINNER":
        """Build a WINNER from a target signal and a schedule.

        Computes ``centroid = spectral_centroid(signal)``, derives ``(s0,
        s1) = schedule.scales(centroid, n_channels)``, then calls
        :meth:`__init__`.

        Args:
            signal: Target signal whose spectral centroid drives the
                noise scales.
            schedule: Centroid → ``(s0, s1)`` mapping. Usually one of
                :meth:`WinnerSchedule.audio` or
                :meth:`WinnerSchedule.image`.
            in_dim: Coordinate (input) dimension. **Not** read from
                ``signal``; the coordinate space is independent of the
                target's value-space rank.
            hidden_dim: Width of each hidden layer.
            num_hidden_layers: Number of stacked ``SIRENLayer`` s.
            key: JAX PRNG key.
            out_features: Readout width.
            omega_first: First-layer omega.
            omega_hidden: Hidden-layer omega; also the noise-scale
                divisor.
            freq_axis: Frequency axis for the centroid computation.
                Default ``-2`` matches the reference's image convention.
            channel_axis: Channel axis. Default ``-1``.
        """
        n_channels = signal.shape[channel_axis]
        centroid = spectral_centroid(signal, freq_axis=freq_axis, channel_axis=channel_axis)
        s0, s1 = schedule.scales(centroid, n_channels)
        return cls(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            key=key,
            s0=s0,
            s1=s1,
            out_features=out_features,
            omega_first=omega_first,
            omega_hidden=omega_hidden,
        )

    def reset_noise(self, key: Key[Array, ""]) -> "WINNER":
        """Return a new WINNER with layers 0/1 rebuilt under a fresh key.

        Rebuilds layers 0 and 1 from a *fresh* :func:`siren_init` call (so
        the base uniform weights themselves are re-sampled) and then
        applies the perturbation under the new key. Does **not** add noise
        to the already-perturbed weights — that would be the
        double-perturbation trap (see ``WINNER_DECISIONS.md`` item 4).

        Layers ``>= 2`` and the readout are preserved as-is. The stored
        ``(s0, s1, omega_first, omega_hidden, in_dim, ...)`` are unchanged.

        Key threading matches :meth:`__init__` exactly: the master ``key``
        is split into ``(k_layers, k_readout, k_noise)``, and ``k_layers``
        is further split into per-layer subkeys via :func:`_per_layer_keys`
        — the same helper :meth:`__init__` uses. The ``k_readout`` subkey
        is unused here (the readout is preserved as-is) but the split
        topology must match ``__init__`` so the layer-0/1 subkeys line up,
        which is what makes ``WINNER(key=k, ...).reset_noise(k)``
        bit-identical for layers 0 and 1.
        """
        k_layers, _, k_noise = jax.random.split(key, 3)
        per_layer_keys = _per_layer_keys(k_layers, self.num_hidden_layers)
        new_layer0 = SIRENLayer(self.in_dim, self.hidden_dim, self.omega_first, is_first=True, key=per_layer_keys[0])
        new_layer1 = SIRENLayer(
            self.hidden_dim, self.hidden_dim, self.omega_hidden, is_first=False, key=per_layer_keys[1]
        )
        layers = (new_layer0, new_layer1, *self.layers[2:])
        weight_dtype = layers[0].W.dtype
        layers = _perturb_first_two_layers(
            layers,
            self.s0.astype(weight_dtype),
            self.s1.astype(weight_dtype),
            self.omega_hidden,
            k_noise,
        )
        return eqx.tree_at(lambda w: w.layers, self, layers)
