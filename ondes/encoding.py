"""Coordinate pre-encodings for INR bodies.

Four operational encoding classes — one per family. Each is an
``eqx.Module`` that materialises its own parameters at construction and acts
as a callable ``coord → embedded`` map. No kind discriminator, no factory
functions (the class itself is the constructor).

- ``Identity``        : no pre-encoding (the raw coordinate vector).
- ``Gaussian``        : Random Fourier Features (Tancik+ 2020) with a fixed
  ``sigma`` baked into the sampled frequency matrix ``B``.
- ``LearnedGaussian`` : Random Fourier Features with a learnable scalar
  ``sigma`` applied to a unit-scale ``B_raw`` matrix at every call.
- ``Dyadic``          : NeRF-style dyadic positional encoding
  (Mildenhall+ 2020) with ``num_bands`` octaves per coordinate axis.

Each encoding exposes ``out_dim`` so downstream code (renderers, body
constructors) can size the first MLP layer without inspecting internals.
"""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


def nyquist_sigma(shape: tuple) -> float:
    """Per-leaf sigma derived from Nyquist analysis on the longest index axis.

    A weight tensor of size ``N`` along its longest axis carries at most ``~N/2``
    spectral components on a normalised ``[-1, 1]`` grid (Nyquist rate). Setting
    ``sigma ~ (N - 1) / 4`` centres the random Gaussian frequency draw at that
    Nyquist rate, which is the natural lengthscale Tancik (2020) identifies as
    "the integer-grid spacing".

    Args:
        shape: Shape tuple of the weight tensor.

    Returns:
        ``sigma`` value (floored at ``1.0``).
    """
    return float(max(1.0, (max(shape) - 1) / 4))


class Encoding(eqx.Module):
    """ABC for coord pre-encodings.

    Subclasses are operational: each is a callable ``coord → embedded`` map
    and exposes ``out_dim`` so downstream code can size the first body layer
    without inspecting internals.

    Not instantiable on its own — calling it triggers the
    ``NotImplementedError`` in ``__call__``/``out_dim``.
    """

    @property
    @abstractmethod
    def out_dim(self) -> int:
        """Dimension of the encoded coordinate vector."""
        raise NotImplementedError

    @abstractmethod
    def __call__(self, coord):
        """Encode ``coord`` into a ``(out_dim,)`` vector."""
        raise NotImplementedError


class Identity(Encoding):
    """No pre-encoding. ``coord`` flows straight through.

    Identity is the only encoding that takes ``in_dim`` explicitly. Other
    encodings derive ``out_dim`` from their own constructor args
    (``num_freqs`` for Gaussian/LearnedGaussian, ``rank * 2 * num_bands``
    for Dyadic) without needing to know the coord dimension. Identity has
    no such intrinsic; its ``out_dim`` depends entirely on what flows
    through it. Keeping ``in_dim`` on the encoding (rather than inferring
    it at call time) lets downstream consumers size the first MLP layer
    via ``encoding.out_dim`` without already holding a coord.
    """

    in_dim: int = eqx.field(static=True)

    def __init__(self, in_dim: int):
        """Store the coordinate dimension so ``out_dim`` can report it."""
        self.in_dim = in_dim

    @property
    def out_dim(self) -> int:
        """Coordinate dimension passed straight through."""
        return self.in_dim

    def __call__(self, coord):
        """Return ``coord`` unchanged."""
        return coord


class Gaussian(Encoding):
    """Random Fourier Features (Tancik+ 2020) with a fixed sigma.

    The frequency matrix ``B`` is sampled from ``N(0, sigma^2)`` at
    construction. ``sigma`` is not retained on the module after construction;
    it has already been folded into ``B``. To recover the spectral scale, use
    ``jnp.std(self.B)`` (or pre-compute and carry it externally).
    """

    B: Float[Array, "num_freqs rank"]

    def __init__(self, rank: int, num_freqs: int, sigma: float, *, key):
        """Sample ``B`` from ``N(0, sigma^2)`` of shape ``(num_freqs, rank)``."""
        self.B = sigma * jax.random.normal(key, (num_freqs, rank))

    @property
    def out_dim(self) -> int:
        """``2 * num_freqs`` — sin and cos features per sampled frequency."""
        return 2 * self.B.shape[0]

    def __call__(self, coord):
        """Encode ``coord`` as ``[cos(2*pi*B@coord), sin(2*pi*B@coord)]``."""
        angles = 2.0 * jnp.pi * (self.B @ coord)
        return jnp.concatenate([jnp.cos(angles), jnp.sin(angles)])


class LearnedGaussian(Encoding):
    """Random Fourier Features with a learnable scalar sigma.

    ``B_raw`` is sampled once at unit scale; ``sigma`` is a learnable scalar
    applied at every call. ``sigma`` IS a pytree leaf so that
    ``eqx.partition``/``filter_grad`` treat it as trainable.
    """

    B_raw: Float[Array, "num_freqs rank"]
    sigma: Float[Array, ""]

    def __init__(self, rank: int, num_freqs: int, *, key, sigma_init: float = float(np.pi)):
        """Sample unit-scale ``B_raw`` and initialise the learnable ``sigma``."""
        self.B_raw = jax.random.normal(key, (num_freqs, rank))
        self.sigma = jnp.array(float(sigma_init))

    @property
    def out_dim(self) -> int:
        """``2 * num_freqs`` — sin and cos features per sampled frequency."""
        return 2 * self.B_raw.shape[0]

    def __call__(self, coord):
        """Encode ``coord`` with ``B = sigma * B_raw`` applied at call-time."""
        B = self.sigma * self.B_raw
        angles = 2.0 * jnp.pi * (B @ coord)
        return jnp.concatenate([jnp.cos(angles), jnp.sin(angles)])


class Dyadic(Encoding):
    """NeRF positional encoding (Mildenhall+ 2020).

    Per-axis dyadic sinusoidal: ``num_bands`` octaves per coordinate axis,
    output is ``rank * 2 * num_bands`` features (sin and cos per axis per
    band).
    """

    rank: int = eqx.field(static=True)
    num_bands: int = eqx.field(static=True)
    bands: Float[Array, "num_bands"]

    def __init__(self, rank: int, num_bands: int = 4):
        """Store the coordinate rank, octave count, and pre-computed band frequencies."""
        self.rank = rank
        self.num_bands = num_bands
        # Pre-compute once; ``bands`` is a tiny ``(num_bands,)`` pytree leaf
        # so jit/grad see it but the forward pass skips the recomputation.
        self.bands = (2.0 ** jnp.arange(num_bands)) * jnp.pi

    @property
    def out_dim(self) -> int:
        """``rank * 2 * num_bands`` — sin and cos per (axis, band) pair."""
        return self.rank * 2 * self.num_bands

    def __call__(self, coord):
        """Encode ``coord`` with dyadic per-axis sinusoidals at ``2**k * pi`` for k in ``range(num_bands)``."""
        angles = coord[:, None] * self.bands[None, :]
        sins = jnp.sin(angles)
        coss = jnp.cos(angles)
        return jnp.stack([sins, coss], axis=-1).reshape(-1)
