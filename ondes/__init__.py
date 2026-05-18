"""ondes — Functional INRs in JAX/Equinox.

Polymorphic implementations of SIREN, H-SIREN, WIRE basis MLPs, Fourier-feature
encodings, and the associated init schemes. One class per basis kind, one
class per encoding kind — no string discriminators, no factory functions.
"""

from ondes.basis import (
    HSIREN,
    SIREN,
    WIRE,
    Basis,
    BasisModule,
    HSIRENLayer,
    SIRENLayer,
    WIRELayer,
    siren_init,
)
from ondes.encoding import (
    Dyadic,
    Encoding,
    Gaussian,
    Identity,
    LearnedGaussian,
    nyquist_sigma,
)


__all__ = [
    "Basis",
    "BasisModule",
    "Dyadic",
    "Encoding",
    "Gaussian",
    "HSIREN",
    "HSIRENLayer",
    "Identity",
    "LearnedGaussian",
    "SIREN",
    "SIRENLayer",
    "WIRE",
    "WIRELayer",
    "nyquist_sigma",
    "siren_init",
]
