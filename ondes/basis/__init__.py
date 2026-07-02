"""Public ``ondes.basis`` surface.

Re-exports the ABCs (``Basis``, ``Body``, ``BasisModule``), the SIREN-family
init helper (``siren_init``), and the three current concrete basis families
(``SIREN``, ``HSIREN``, ``WIRE``) plus their per-layer classes. Downstream
code should keep importing from ``ondes.basis`` directly — the per-family
submodules are an implementation detail.
"""

from ondes.basis._base import Basis, BasisModule, Body
from ondes.basis.bacon import BACON
from ondes.basis.comb import HarmonicComb, HarmonicCombLayer, JacobiLearnM, JacobiLearnMLayer
from ondes.basis.finer import FINER, FINERLayer
from ondes.basis.hsiren import HSIREN, HSIRENLayer
from ondes.basis.mfn import FourierMFN, GaborMFN
from ondes.basis.pnf import PNF
from ondes.basis.rff import RFF, RFFLayer
from ondes.basis.siren import SIREN, SIRENLayer, siren_init
from ondes.basis.wire import WIRE, WIRELayer


__all__ = [
    "BACON",
    "Basis",
    "BasisModule",
    "Body",
    "FINER",
    "FINERLayer",
    "FourierMFN",
    "GaborMFN",
    "HSIREN",
    "HSIRENLayer",
    "HarmonicComb",
    "HarmonicCombLayer",
    "JacobiLearnM",
    "JacobiLearnMLayer",
    "PNF",
    "RFF",
    "RFFLayer",
    "SIREN",
    "SIRENLayer",
    "WIRE",
    "WIRELayer",
    "siren_init",
]
