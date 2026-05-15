"""Coordinate pre-encodings for INR bodies.

Three kinds are supported via a string discriminator on ``Encoding.kind``:

- ``"none"``     : identity (no pre-encoding).
- ``"gaussian"`` : Random Fourier Features (Tancik+ 2020) with either a fixed
  ``sigma``, a per-leaf rule (``sigma_from_shape``), or a learnable scalar
  ``sigma`` (``learn_sigma=True``).
- ``"dyadic"``   : NeRF-style positional encoding (Mildenhall+ 2020) with
  ``num_bands`` octaves per coordinate axis.

The factory functions ``gaussian_fixed`` / ``gaussian_from_shape`` /
``gaussian_learn`` / ``dyadic`` are the recommended entry points; ``NO_ENCODING``
is a module-level singleton for the identity case.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


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


@dataclass(frozen=True)
class Encoding:
    """How coordinate inputs are pre-encoded before the body MLP.

    Three kinds, each with its own parameters:

    - ``kind="none"``     : raw coordinate vector (default).
    - ``kind="gaussian"`` : Random Fourier Features (Tancik 2020). Use ``sigma``
      for a fixed scale, ``sigma_from_shape`` for a per-leaf rule, or
      ``learn_sigma=True`` to make ``sigma`` a learnable scalar (initialised at
      ``sigma``, defaults to ``pi``).
    - ``kind="dyadic"``   : NeRF-style dyadic positional encoding with
      ``num_bands`` octaves per coordinate axis.

    Attributes:
        kind: Discriminator string (``"none"`` / ``"gaussian"`` / ``"dyadic"``).
        sigma: Fixed Gaussian scale, or initial value when ``learn_sigma``.
        sigma_from_shape: Callable mapping a weight shape tuple to a ``sigma``.
        learn_sigma: Whether ``sigma`` is a trainable scalar.
        num_bands: Number of octaves for dyadic encoding.
    """

    # TODO(ondes/0.2): consider ABC + Gaussian/Dyadic/Identity subclasses per
    # loom/CRITIQUE.md section 1c. Current string discriminator is a known
    # design smell preserved verbatim during the initial extraction.
    kind: str = "none"
    sigma: float | None = None
    sigma_from_shape: Callable | None = None
    learn_sigma: bool = False
    num_bands: int = 4


NO_ENCODING = Encoding(kind="none")


def gaussian_fixed(sigma: float) -> Encoding:
    """Build a Gaussian RFF encoding with a fixed ``sigma``."""
    return Encoding(kind="gaussian", sigma=float(sigma))


def gaussian_from_shape(rule: Callable) -> Encoding:
    """Build a Gaussian RFF encoding whose ``sigma`` is derived per leaf via ``rule``."""
    return Encoding(kind="gaussian", sigma_from_shape=rule)


def gaussian_learn(init: float = float(np.pi)) -> Encoding:
    """Build a Gaussian RFF encoding with a learnable ``sigma`` initialised at ``init``."""
    return Encoding(kind="gaussian", sigma=init, learn_sigma=True)


def dyadic(L: int = 4) -> Encoding:
    """Build a dyadic positional encoding with ``L`` octaves per axis."""
    return Encoding(kind="dyadic", num_bands=L)
