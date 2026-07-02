r"""STAF basis: per-layer-shared trainable sinusoidal activation (Morsali+ 2026).

STAF replaces each hidden layer's non-linearity with a learnable finite
sinusoidal mixture

    phi_l(u) = sum_{i=1..tau} C_i * sin(Omega_i * u + Phi_i)

whose amplitude/frequency/phase triple ``(C, Omega, Phi)`` — the fields
``amp``/``freq``/``phase`` here — is **shared across all neurons of layer ``l``**
(the paper's "layer-wise shared activation", used for every experiment; per-neuron
and network-wide are studied only as ablations). Each activation-bearing layer
therefore adds ``3 * tau`` parameters, independent of width — free param matching
against SIREN (their Table 7: STAF tau=5 at 198,975 params vs SIREN 198,915, same
FLOPs; the +60 is ``3 * tau * L`` for their four-activation-layer image config).

Reference: Morsali, Vaez, Soltani, Kazerouni, Taati, Mohammad-Noori,
"A Unified Theory of Sinusoidal Activation Families for Implicit Neural
Representations", TMLR (6/2026), arXiv:2502.00869v3. The method is named STAF;
v1/v2 were titled "STAF: Sinusoidal Trainable Activation Functions". Built on the
SIREN/WIRE/INCODE codebases (project page: alirezamorsali.github.io/staf).

Family nesting (both encoded as tests):

    SIREN         tau=1, phase=0, amp=1, freq=omega           -> sin(omega * u)
    HarmonicComb  freq=(2k+1)*omega, phase=0, amp on the sphere -> odd-harmonic comb

STAF is the *unconstrained free-frequency* member of this family: ``freq`` are
tau independent learnable scalars (no shared fundamental, no harmonic locking),
``phase`` is free, and — the load-bearing difference from
:class:`~ondes.basis.comb.HarmonicComb` — ``amp`` carries **no norm constraint**
during training. STAF's unit variance is an initialisation property only (exact
moment matching, Theorem 3.1), holding in distribution at ``t = 0`` under the
free phases; HarmonicComb instead re-projects ``c`` onto the unit sphere every
forward pass, a for-all-time constraint.

**Canonical-defaults trap.** STAF's canonical image config uses ``omega_0 = 30``
at *every* layer (Appendix C.10 of the v3 TMLR paper: "We set W_0 (or omega_0 in
the code) to 30"), so the defaults here are ``omega_first = omega_hidden = 30.0``
— *not* the ondes SIREN convention of ``6.0`` / ``1.0``. The audio config uses
``omega_0 = 3000`` in the first layer and ``30`` in the hidden layers; denoising
uses ``5``. Set ``omega_first`` / ``omega_hidden`` explicitly when reproducing a
non-image config.

Note on ``omega``: the ``Basis`` ABC's ``omega`` scalar holds STAF's per-layer
``omega_0`` — the init frequency scale, consumed only at construction (the
:func:`~ondes.basis.siren.siren_init` weight bound and the ``freq`` init
``omega_0 * U[0, 1)``). STAF's forward pass reads the learnable ``freq`` vector,
never ``omega``, so ``omega`` receives zero gradient and stays fixed at
``omega_0`` throughout training. The learnable frequencies live in ``freq``.

MLP weights use SIREN uniform init (:func:`~ondes.basis.siren.siren_init`). The
paper specifies no weight-init scheme — Theorem 3.1 is deliberately
weight-agnostic ("does not depend on ... the weight matrices") — so this choice
follows their SIREN-codebase provenance and is not load-bearing for the theory.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from ondes.basis._base import Basis, Body, _validate_body_args
from ondes.basis.siren import _build_readout, siren_init


def _validate_tau(tau: int) -> int:
    """Reject ``tau < 1`` (mirrors ``_validate_body_args``'s layer check).

    A non-positive ``tau`` gives an empty activation sum — the constant ``0``,
    a dead layer with zero gradient. ``assert`` matches the constructor-arg
    precondition convention in ``_base.py`` and ``comb.py``.
    """
    assert tau >= 1, f"tau must be >= 1, got {tau}"
    return int(tau)


def _staf_init(
    tau: int,
    omega_0: float,
    key: Key[Array, ""],
) -> tuple[Float[Array, "tau"], Float[Array, "tau"], Float[Array, "tau"]]:
    r"""Sample STAF's per-layer activation triple ``(amp, freq, phase)`` (Morsali+ 2026).

    For the activation $\phi(u) = \sum_{i=1}^{\tau} C_i \sin(\Omega_i u + \Phi_i)$,
    draws the $\tau$-vectors $C$ (``amp``), $\Omega$ (``freq``), $\Phi$ (``phase``)
    from STAF's initialisation (Listing 1 + Theorem 3.1):

    - $\Omega_i = \omega_0 \cdot U[0, 1)$ — frequencies scaled by ``omega_0``.
    - $\Phi_i \sim U(-\pi, \pi)$ — phases; the variance theorem needs these free
      to centre each term and prevent coherent phase alignment across terms.
    - $C_i = \operatorname{sign}(X_i)\sqrt{|X_i|}$, $X_i \sim \mathrm{Laplace}(0, 2/\tau)$,
      giving $\mathbb{E}[C_i^2] = \mathbb{E}|X_i| = 2/\tau$ — the exact amplitude
      calibration behind the $\mathcal{N}(0, 1)$ post-activations (moment
      matching, not a CLT/large-$\tau$ argument).

    ``jax.random.laplace`` samples the *standard* Laplace (scale $1$); multiplying
    by $2/\tau$ sets the scale, since $\mathbb{E}|bZ| = b$ for
    $Z \sim \mathrm{Laplace}(0, 1)$.

    Args:
        tau: Number of sinusoidal terms in the activation (``>= 1``).
        omega_0: Frequency scale for the ``freq`` init (``30`` canonical for images).
        key: JAX PRNG key.

    Returns:
        Tuple ``(amp, freq, phase)``, each of shape ``(tau,)`` — the paper's
        $(C, \Omega, \Phi)$.
    """
    k_amp, k_freq, k_phase = jax.random.split(key, 3)
    freq = omega_0 * jax.random.uniform(k_freq, (tau,))
    phase = jax.random.uniform(k_phase, (tau,), minval=-jnp.pi, maxval=jnp.pi)
    x = jax.random.laplace(k_amp, (tau,)) * (2.0 / tau)
    amp = jnp.sign(x) * jnp.sqrt(jnp.abs(x))
    return amp, freq, phase


class STAFLayer(Basis):
    r"""STAF layer: per-layer-shared trainable sinusoidal mixture (Morsali+ 2026).

    ``phi(pre) = sum_{i<tau} amp_i * sin(freq_i * pre + phase_i)``, applied
    element-wise. The ``(amp, freq, phase)`` triple (the paper's $(C, \Omega, \Phi)$)
    is a single ``(tau,)`` vector *shared across every neuron* of the layer —
    STAF's "layer-wise shared activation" granularity (Sec 3.4), so the layer adds
    ``3 * tau`` trainable parameters regardless of ``out_dim``.

    All three vectors are unconstrained learnable leaves. In particular ``amp``
    carries **no norm constraint** — STAF's unit-variance property is an
    initialisation statement only (Theorem 3.1), unlike
    :class:`~ondes.basis.comb.HarmonicCombLayer`, which re-normalises its
    coefficients onto the unit sphere every forward pass.

    ``freq`` are ``tau`` *free* frequencies (no shared fundamental, no harmonic
    locking); ``phase`` is free. Family nesting: at ``tau=1`` with ``phase=0``,
    ``amp=1``, ``freq=omega`` the layer reduces exactly to SIREN's
    ``sin(omega * pre)``; clamping ``freq`` to the odd-harmonic lattice
    ``(2k+1)*omega`` with ``phase=0`` and ``amp`` on the unit sphere recovers
    :class:`~ondes.basis.comb.HarmonicCombLayer`.

    The inherited ``omega`` scalar holds ``omega_0``, the init frequency scale.
    It is read only at construction (the :func:`~ondes.basis.siren.siren_init`
    weight bound and the ``freq`` init); the forward pass uses the learnable
    ``freq`` vector, so ``omega`` gets zero gradient and is fixed after init.

    ``amp``/``freq``/``phase`` are ``(tau,)`` and ``tau`` is equal across a body's
    layers, so the per-layer pytree structure stays homogeneous
    (``scan``/``vmap``-compatible); ``tau`` lives in the array shapes, not a
    static field.
    """

    amp: Float[Array, "tau"]
    freq: Float[Array, "tau"]
    phase: Float[Array, "tau"]

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        omega_init: float,
        is_first: bool,
        *,
        key: Key[Array, ""],
        tau: int = 5,
    ) -> None:
        """Initialise the SIREN-style weights and the shared activation triple.

        ``is_first`` is a construction-only kwarg consumed by
        :func:`~ondes.basis.siren.siren_init` to pick the init bound; it is not
        stored (the bound is baked into ``W``/``b``), keeping every layer
        pytree-structurally identical.

        ``omega_init`` (``omega_0``) plays a dual init role: the SIREN weight
        bound and the ``freq`` init scale ``omega_0 * U[0, 1)``.
        """
        init_key, act_key = jax.random.split(key)
        self.W, self.b = siren_init(in_dim, out_dim, omega_init, is_first, init_key)
        self.omega = jnp.array(float(omega_init))
        self.amp, self.freq, self.phase = _staf_init(_validate_tau(tau), omega_init, act_key)

    def _activate(self, pre: Float[Array, "out"]) -> Float[Array, "out"]:
        r"""Sum the shared mixture $\sum_i \text{amp}_i \sin(\text{freq}_i \cdot \text{pre} + \text{phase}_i)$."""
        arg = self.freq * pre[..., None] + self.phase  # (out, tau)
        return jnp.sum(self.amp * jnp.sin(arg), axis=-1)  # (out,)


class STAF(Body):
    r"""Stack of :class:`STAFLayer` s with an internal linear readout (Morsali+ 2026).

    Per-layer-shared trainable sinusoidal activations; each hidden layer owns its
    own ``(amp, freq, phase)`` triple of ``tau`` terms. Shares SIREN's weight init
    and readout because the paper builds on the SIREN codebase and its variance
    theorem is weight-agnostic.

    **Canonical defaults:** ``omega_first = omega_hidden = 30.0`` — STAF's image
    ``omega_0`` from every layer (v3 TMLR paper, App. C.10), *not* ondes' SIREN
    ``6.0`` / ``1.0``. Override for non-image configs (audio ``3000`` first /
    ``30`` hidden; denoising ``5``; the paper uses ``tau=2`` for denoising).
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
        omega_hidden: float = 30.0,
        tau: int = 5,
    ) -> None:
        """Initialise the STAF body MLP.

        Args:
            in_dim: Coordinate (input) dimension.
            hidden_dim: Width of each hidden layer.
            num_hidden_layers: Number of stacked ``STAFLayer`` s.
            key: JAX PRNG key.
            out_features: Readout width; see :class:`ondes.basis.siren.SIREN`.
            omega_first: Init frequency scale for the first layer (canonical ``30``).
            omega_hidden: Init frequency scale for subsequent layers (canonical ``30``).
            tau: Number of sinusoidal terms per activation. Paper default ``5``
                for all tasks except image denoising (``2``); a quality-latency
                knob (their Table 5 sweeps ``2..50``).
        """
        out_features = _validate_body_args(num_hidden_layers, out_features)
        keys = jax.random.split(key, num_hidden_layers + 1)
        layers = []
        for i in range(num_hidden_layers):
            in_d = in_dim if i == 0 else hidden_dim
            o = omega_first if i == 0 else omega_hidden
            layers.append(STAFLayer(in_d, hidden_dim, o, is_first=(i == 0), key=keys[i], tau=tau))
        self.layers = tuple(layers)
        self.readout_W, self.readout_b = _build_readout(hidden_dim, omega_hidden, out_features, keys[-1])
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
