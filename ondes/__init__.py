"""ondes — Functional INRs in JAX/Equinox.

Polymorphic implementations of SIREN, H-SIREN, WIRE basis MLPs, Fourier-feature
encodings, and the associated init schemes. One class per basis kind, one
class per encoding kind — no string discriminators, no factory functions.
"""

from ondes.basis import (
    BACON,
    FINER,
    HSIREN,
    RFF,
    SIREN,
    WIRE,
    Basis,
    BasisModule,
    Body,
    FINERLayer,
    FourierMFN,
    GaborMFN,
    HSIRENLayer,
    RFFLayer,
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
    "BACON",
    "Basis",
    "BasisModule",
    "Body",
    "Dyadic",
    "Encoding",
    "FINER",
    "FINERLayer",
    "FourierMFN",
    "GaborMFN",
    "Gaussian",
    "HSIREN",
    "HSIRENLayer",
    "Identity",
    "LearnedGaussian",
    "RFF",
    "RFFLayer",
    "SIREN",
    "SIRENLayer",
    "WIRE",
    "WIRELayer",
    "nyquist_sigma",
    "siren_init",
]
