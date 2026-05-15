"""ondes — Functional INRs in JAX/Equinox.

Implementations of SIREN, H-SIREN, WIRE basis MLPs, Fourier-feature encodings,
and the associated init schemes.
"""

from ondes.basis import BASIS_KINDS, BasisBody, BasisLayer, siren_init
from ondes.encoding import (
    NO_ENCODING,
    Encoding,
    dyadic,
    gaussian_fixed,
    gaussian_from_shape,
    gaussian_learn,
    nyquist_sigma,
)


__all__ = [
    "BASIS_KINDS",
    "BasisBody",
    "BasisLayer",
    "Encoding",
    "NO_ENCODING",
    "dyadic",
    "gaussian_fixed",
    "gaussian_from_shape",
    "gaussian_learn",
    "nyquist_sigma",
    "siren_init",
]
