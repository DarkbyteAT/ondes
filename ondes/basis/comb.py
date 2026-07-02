"""Odd-harmonic sine-comb basis family: ``JacobiLearnM`` and ``HarmonicComb``.

Both variants share the same forward shape — a truncated odd-harmonic comb

    phi(pre) = sum_{k<n} c_k * sin((2k+1) * omega * pre)

— and differ only in where the coefficient vector ``c`` lives. The family nests
SIREN and the Jacobi elliptic ``sn`` waveform as special cases::

    SIREN         c = e_0                 (only the fundamental; the m->0 corner)
    JacobiLearnM  c on the sn manifold    (per-neuron modulus m, learnable)
    HarmonicComb  c free on the unit sphere (per-neuron, learnable)

With the fundamental rescaled to ``omega`` for every variant (the ``match_freq``
convention baked into :func:`_odd_freqs`), ``m`` and the comb coefficients become
pure harmonic-content knobs at a fixed base frequency — an apples-to-apples axis
against SIREN. Every variant holds ``||c_j||_2 = 1`` per neuron, which pins the
init variance: under unit-variance pre-activations each ``sin`` term has variance
~1/2 and the odd harmonics are near-orthogonal (their frequencies differ by
``>= 2*omega``, so the cross-terms nearly vanish), giving

    Var[phi_j] ~= (1/2) * ||c_j||^2 = 1/2

for every neuron — approached in the wrapping regime (large ``omega * std(pre)``).
At the default ``omega_hidden=1.0`` the sin-argument std is ~1 and the measured
per-neuron Var sits near 0.44, matching plain SIREN (a lone ``sin`` at ``omega=1``
gives 0.432); it reaches ~0.50 from ``omega`` ~6 up. The comb never *degrades*
SIREN's init variance, so SIREN's depth fixed point and the same SIREN-family init
(:func:`ondes.basis.siren.siren_init`) carry over unchanged.

This family covers periodic *odd* functions only: ``sn``'s ``m -> 1`` limit is
``tanh`` (aperiodic, outside any finite sine comb), so ``m`` is clipped strictly
below 1. Reach for ``tanh`` directly for that regime — it is not in here.

The ``sn`` q-series machinery (:func:`K_agm`, :func:`_q_K`, :func:`_sn_direction`,
:func:`_sn_unit_coeffs`, :func:`_odd_freqs`) lives as module-level free functions,
peers of ``siren_init`` — there is no intermediate comb ABC. Each variant is a
direct :class:`~ondes.basis._base.Basis` / :class:`~ondes.basis._base.Body`
subclass computing the comb inline in ``_activate``.

Choosing between them: ``JacobiLearnM`` is the constrained 1-D ``sn`` dial (each
neuron moves only along the ``sn`` curve — fewer degrees of freedom, stable);
``HarmonicComb`` frees the coefficients onto the full unit sphere (more
expressive, but shows late-training instability under Adam and is not yet
validated as an FWS weight-generator basis). Start with ``JacobiLearnM``.

Defaults follow the ondes ``omega_first=6.0`` / ``omega_hidden=1.0`` convention,
*not* the source experiments' ``omega=20`` (nor canonical SIREN's 30) — set
``omega_first`` / ``omega_hidden`` explicitly when reproducing scratchpad results.

Constraints (both variants):
    Per-neuron learnable activation parameters are known to *hurt* at depth
    (``num_hidden_layers >= 8``): the extra gradient-noise dimension dilutes
    inter-layer coordination. They help at ``L = 2..4``. Prefer SIREN/WIRE for
    deep stacks. See the FWS per-neuron-activation-depth finding (2026-06-20).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from ondes.basis._base import Basis, Body, _validate_body_args
from ondes.basis.siren import _build_readout, siren_init


def _validate_n_terms(n_terms: int) -> int:
    """Reject ``n_terms < 1`` (mirrors ``_validate_body_args``'s layer check).

    A non-positive ``n_terms`` gives an empty harmonic row, whose norm is 0 —
    which would propagate as a silent NaN through :func:`_sn_unit_coeffs` and the
    forward comb. ``assert`` matches the ``_validate_body_args`` convention in
    ``_base.py`` for constructor-arg preconditions.
    """
    assert n_terms >= 1, f"n_terms must be >= 1, got {n_terms}"
    return int(n_terms)


# ---- sn q-series + frequency builders (shared by both comb variants) --------
def K_agm(m: Float[Array, "..."], n_iter: int = 7) -> Float[Array, "..."]:
    """Complete elliptic integral of the first kind ``K(m)`` via the AGM.

    ``K(m) = pi / (2 * AGM(1, sqrt(1 - m)))``. The arithmetic-geometric mean
    converges quadratically; ~7 iterations reach float32 machine epsilon for
    ``m`` in ``[eps, 1 - eps]``. Pure JAX, differentiable, vmap-friendly.

    Both ``K`` and ``dK/dm`` are singular at ``m in {0, 1}``, so callers must
    keep ``m`` off the endpoints — every builder below clips with ``m_eps``
    (see :func:`_q_K`) before calling.

    Args:
        m: Elliptic parameter ``m = k**2``, any shape, strictly in ``(0, 1)``.
        n_iter: AGM iterations. 7 is float32-exact on the clipped domain.

    Returns:
        ``K(m)``, same shape as ``m``.
    """
    a = jnp.ones_like(m)
    b = jnp.sqrt(1.0 - m)
    for _ in range(n_iter):
        a, b = 0.5 * (a + b), jnp.sqrt(a * b)
    return jnp.pi / (2.0 * a)


def _q_K(m: Float[Array, "..."], m_eps: float) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    """Clip ``m`` off the singular endpoints; return ``(m, K(m), q)``.

    ``q = exp(-pi * K(1 - m) / K(m))`` is the elliptic nome.

    Args:
        m: Elliptic parameter, any shape.
        m_eps: Clip margin; ``m`` is confined to ``[m_eps, 1 - m_eps]``.

    Returns:
        The clipped ``m``, ``K(m)``, and the nome ``q`` — all the shape of ``m``.
    """
    m = jnp.clip(m, m_eps, 1.0 - m_eps)
    k = K_agm(m)
    kp = K_agm(1.0 - m)
    q = jnp.exp(-jnp.pi * kp / k)
    return m, k, q


def _sn_direction(m: Float[Array, "..."], n_terms: int, m_eps: float) -> Float[Array, "... n"]:
    """Unnormalised ``sn`` q-series amplitudes — the harmonic *direction*.

        d_k = q^(k + 1/2) / (1 - q^(2k + 1)),   k = 0 .. n_terms - 1.

    The ``k``-independent prefactor ``2*pi / (sqrt(m) * K)`` is deliberately
    dropped: it cancels under L2 normalisation and would otherwise blow up as
    ``m -> 0`` (via the ``1/sqrt(m)`` factor). With this factoring the SIREN
    corner is smooth: as ``m -> 0`` the nome ``q -> 0`` and ``d/||d|| -> e_0``
    cleanly.

    Args:
        m: Elliptic parameter, any shape ``(...)``.
        n_terms: Number of harmonics to retain.
        m_eps: Endpoint clip margin passed to :func:`_q_K`.

    Returns:
        Amplitudes of shape ``(..., n_terms)``, broadcasting against ``m``.
    """
    _, _, q = _q_K(m, m_eps)
    n = jnp.arange(n_terms, dtype=q.dtype)
    qn = q[..., None]
    return qn ** (n + 0.5) / (1.0 - qn ** (2.0 * n + 1.0))


def _unit_normalize(x: Float[Array, "... n"]) -> Float[Array, "... n"]:
    """L2-normalise rows with an eps-floored denominator (NaN-safe at ``x = 0``).

    Plain ``x / ||x||`` has a gauge pathology on the coefficient sphere: the
    gradient is NaN at ``x = 0`` and scales as ``1/||x||``, exploding as a row
    shrinks. Flooring the squared norm at ``finfo(dtype).eps`` — the scale below
    which the row's direction is numerically meaningless anyway — bounds both.
    The distortion on a genuine unit row (``sum(x^2) = 1``) is ``~eps/2``,
    negligible. This may bear on HarmonicComb's documented late-training
    instability (https://trello.com/c/3bNOZ8QQ); empirical confirmation pending.
    """
    eps = jnp.finfo(x.dtype).eps
    return x / jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def _sn_unit_coeffs(m: Float[Array, "..."], n_terms: int, m_eps: float) -> Float[Array, "... n"]:
    """``sn`` harmonic shape renormalised to ``||c||_2 = 1`` per row.

    Same relative harmonic content as ``sn(., m)`` up to a per-neuron scale the
    next linear layer absorbs. Holding ``||c|| = 1`` is the load-bearing
    invariant that pins ``Var[phi] = 1/2`` across variants (see module docstring).

    Uses the eps-floored :func:`_unit_normalize`: the ``sn`` direction ``d`` can
    underflow to zero when the nome ``q`` underflows at extreme ``m_eps``, which
    a bare norm would turn into a divide-by-zero.

    Args:
        m: Elliptic parameter, any shape ``(...)``.
        n_terms: Number of harmonics.
        m_eps: Endpoint clip margin.

    Returns:
        Unit-norm coefficient rows of shape ``(..., n_terms)``.
    """
    return _unit_normalize(_sn_direction(m, n_terms, m_eps))


def _odd_freqs(n_terms: int, omega: Float[Array, ""]) -> Float[Array, "n"]:
    """Per-harmonic frequencies ``(2k + 1) * omega`` at fundamental ``omega``.

    The fundamental is rescaled to ``omega`` for every comb variant (the
    ``match_freq=True`` convention): ``m`` and the comb coefficients then change
    harmonic *content* only, leaving the base frequency fixed — the
    apples-to-apples axis against SIREN. The geometrically-faithful ``sn``
    fundamental ``(pi / 2K(m)) * omega`` is only needed to validate against
    ``scipy.special.ellipj`` and lives with the reference fixture in the tests.

    Args:
        n_terms: Number of odd harmonics.
        omega: Learnable fundamental frequency (0-d array); gradients flow here.

    Returns:
        Frequencies ``[omega, 3*omega, 5*omega, ...]`` of shape ``(n_terms,)``.
    """
    n = jnp.arange(n_terms, dtype=omega.dtype)
    return (2.0 * n + 1.0) * omega


# ---- JacobiLearnM: per-neuron sn modulus, learnable -------------------------
class JacobiLearnMLayer(Basis):
    """Odd-harmonic comb on the ``sn`` manifold with a learnable per-neuron modulus.

    ``phi_j(pre) = sum_{k<n} c_{j,k} sin((2k+1) * omega * pre)`` where the
    coefficient row ``c_j`` is the unit-norm ``sn`` q-series for a per-neuron
    modulus ``m_j = sigmoid(raw_m_j) in (0, 1)``. Training moves each neuron only
    *along* the ``sn`` curve (a 1-D dial on the unit sphere of coefficient space);
    :class:`HarmonicCombLayer` relaxes this to the full sphere.

    ``K(m_j)`` and ``K(1 - m_j)`` are recomputed each forward via the AGM (``m``
    moves with training) and the shape is renormalised to ``||c_j||_2 = 1``, so
    the init variance ``Var[phi] ~= 1/2`` is held throughout training.

    Carries a learnable ``raw_m`` leaf (shape ``(out_dim,)``); SIREN and WIRE
    pytrees do not. ``n_terms`` and ``m_eps`` are static and equal across a
    body's layers, so the per-layer pytree structure stays homogeneous
    (``scan``/``vmap``-compatible).
    """

    raw_m: Float[Array, "out"]
    n_terms: int = eqx.field(static=True)
    m_eps: float = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        omega_init: float,
        is_first: bool,
        *,
        key: Key[Array, ""],
        n_terms: int = 6,
        spread: float = 1.0,
        m_eps: float = 1e-3,
    ) -> None:
        """Initialise the linear weights, learnable ``omega``, and ``raw_m``.

        ``is_first`` is a construction-only kwarg consumed by :func:`siren_init`
        to pick the init bound; it is not stored (the bound is already baked into
        ``W``/``b``), keeping every layer pytree-structurally identical.

        ``raw_m ~ N(0, spread**2)`` puts the initial modulus ``m_j`` roughly
        symmetric about 0.5 — a mix of sine-leaning and harmonic-rich neurons.
        """
        init_key, m_key = jax.random.split(key)
        self.W, self.b = siren_init(in_dim, out_dim, omega_init, is_first, init_key)
        self.omega = jnp.array(float(omega_init))
        self.raw_m = jax.random.normal(m_key, (out_dim,)) * spread
        self.n_terms = _validate_n_terms(n_terms)
        self.m_eps = float(m_eps)

    def _activate(self, pre: Float[Array, "out"]) -> Float[Array, "out"]:
        """Sum the unit-norm ``sn`` comb ``sum_k c_k sin((2k+1) omega * pre)``."""
        m = jax.nn.sigmoid(self.raw_m)
        c = _sn_unit_coeffs(m, self.n_terms, self.m_eps)
        f = _odd_freqs(self.n_terms, self.omega)
        return jnp.sum(c * jnp.sin(f * pre[..., None]), axis=-1)


class JacobiLearnM(Body):
    """Stack of :class:`JacobiLearnMLayer` s with an internal linear readout.

    Per-neuron odd-harmonic comb on the ``sn`` manifold; the modulus ``m_j`` of
    each neuron is learnable. Shares SIREN's init and readout because every
    variant has fundamental ``omega`` and holds ``Var[phi] ~= 1/2`` (see the
    module docstring).

    Constraint:
        Per-neuron learnable activation parameters hurt at
        ``num_hidden_layers >= 8`` (gradient-noise dimension dilutes layer
        coordination). Best at ``L = 2..4``.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        *,
        key: Key[Array, ""],
        out_features: int | None = None,
        omega_first: float = 6.0,
        omega_hidden: float = 1.0,
        n_terms: int = 6,
        spread: float = 1.0,
        m_eps: float = 1e-3,
    ) -> None:
        """Initialise the JacobiLearnM body MLP.

        Args:
            in_dim: Coordinate (input) dimension.
            hidden_dim: Width of each hidden layer.
            num_hidden_layers: Number of stacked ``JacobiLearnMLayer`` s.
            key: JAX PRNG key.
            out_features: Readout width; see :class:`ondes.basis.siren.SIREN`.
            omega_first: Initial fundamental frequency for the first layer.
            omega_hidden: Initial fundamental frequency for subsequent layers.
            n_terms: Number of odd harmonics in each neuron's comb.
            spread: Std of the ``raw_m`` init; wider means more per-neuron
                spread of harmonic content at init.
            m_eps: Clip margin keeping the modulus off the singular endpoints.
        """
        out_features = _validate_body_args(num_hidden_layers, out_features)
        keys = jax.random.split(key, num_hidden_layers + 1)
        layers = []
        for i in range(num_hidden_layers):
            in_d = in_dim if i == 0 else hidden_dim
            o = omega_first if i == 0 else omega_hidden
            layers.append(
                JacobiLearnMLayer(
                    in_d, hidden_dim, o, is_first=(i == 0), key=keys[i], n_terms=n_terms, spread=spread, m_eps=m_eps
                )
            )
        self.layers = tuple(layers)
        self.readout_W, self.readout_b = _build_readout(hidden_dim, omega_hidden, out_features, keys[-1])
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers


# ---- HarmonicComb: per-neuron comb, free on the unit sphere ------------------
class HarmonicCombLayer(Basis):
    """Odd-harmonic sine comb with per-neuron coefficients free on the unit sphere.

        phi_j(pre) = sum_{k<n} c_{j,k} sin((2k+1) * omega * pre),   ||c_j||_2 = 1.

    The full superset that SIREN and ``sn`` live inside:
    ``c_j = e_0`` recovers ``sin(omega * pre)`` (SIREN); ``c_j`` proportional to
    the ``sn`` q-series recovers ``sn(omega * pre, m)``; a free ``c_j`` is an
    arbitrary odd comb.

    Coefficients are trainable but held on the sphere by the reparameterisation
    ``c = raw_c / ||raw_c||`` in the forward pass, so ``Var[phi_j] ~= 1/2`` holds
    throughout training and the SIREN-family init is preserved. The init
    direction is the normalised ``sn(., m0)`` q-series; ``m0`` slides the starting
    comb from pure sine (``m0 -> 0``) to harmonic-rich (``m0 -> 1^-``), and
    ``m0_spread > 0`` gives per-neuron init diversity. ``n_terms`` is the comb's
    *capacity* (sphere dimension), not just a truncation depth.

    Constraints:
        Per-neuron learnable activation parameters hurt at
        ``num_hidden_layers >= 8``. Additionally, HarmonicComb shows
        late-training instability under Adam — it can collapse away from its
        running-best loss; the cause is under investigation (scratchpad
        phase-08 work). Not yet validated as an FWS weight-generator basis.
        Also: Adam allocates moment buffers for the magnitude direction of
        ``raw_c`` that the forward-pass normalisation discards (mild waste; a
        tangent-space update would be cleaner but heavier).
    """

    raw_c: Float[Array, "out n"]
    n_terms: int = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        omega_init: float,
        is_first: bool,
        *,
        key: Key[Array, ""],
        n_terms: int = 6,
        m0: float = 0.5,
        m0_spread: float = 0.0,
        m_eps: float = 1e-3,
    ) -> None:
        """Initialise the linear weights, learnable ``omega``, and the comb ``raw_c``.

        ``is_first`` is a construction-only kwarg (see :class:`JacobiLearnMLayer`).
        The comb is seeded on the ``sn`` submanifold — an arbitrary starting point
        that nests SIREN (``m0 -> 0``) and the ``sn`` waveform (any ``m0``) as
        zero-step cases.
        """
        init_key, c_key = jax.random.split(key)
        self.W, self.b = siren_init(in_dim, out_dim, omega_init, is_first, init_key)
        self.omega = jnp.array(float(omega_init))
        self.n_terms = _validate_n_terms(n_terms)
        if m0_spread > 0.0:
            m0_vec = jnp.clip(m0 + m0_spread * jax.random.normal(c_key, (out_dim,)), m_eps, 1.0 - m_eps)
        else:
            m0_vec = jnp.full((out_dim,), float(m0))
        self.raw_c = _sn_unit_coeffs(m0_vec, n_terms, m_eps)

    def _activate(self, pre: Float[Array, "out"]) -> Float[Array, "out"]:
        """Sum the sphere-normalised comb ``sum_k c_k sin((2k+1) omega * pre)``."""
        c = _unit_normalize(self.raw_c)
        f = _odd_freqs(self.n_terms, self.omega)
        return jnp.sum(c * jnp.sin(f * pre[..., None]), axis=-1)


class HarmonicComb(Body):
    """Stack of :class:`HarmonicCombLayer` s with an internal linear readout.

    Per-neuron odd-harmonic comb with coefficients free on the unit sphere,
    ``sn``-initialised. The widest member of the family — SIREN and the ``sn``
    waveform are special points inside its coefficient sphere.

    Constraints:
        Per-neuron learnable activation parameters hurt at
        ``num_hidden_layers >= 8``. HarmonicComb additionally has late-training
        instability under Adam (collapses from running-best; under
        investigation) and is not yet validated as an FWS weight-generator basis.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        *,
        key: Key[Array, ""],
        out_features: int | None = None,
        omega_first: float = 6.0,
        omega_hidden: float = 1.0,
        n_terms: int = 6,
        m0: float = 0.5,
        m0_spread: float = 0.0,
        m_eps: float = 1e-3,
    ) -> None:
        """Initialise the HarmonicComb body MLP.

        Args:
            in_dim: Coordinate (input) dimension.
            hidden_dim: Width of each hidden layer.
            num_hidden_layers: Number of stacked ``HarmonicCombLayer`` s.
            key: JAX PRNG key.
            out_features: Readout width; see :class:`ondes.basis.siren.SIREN`.
            omega_first: Initial fundamental frequency for the first layer.
            omega_hidden: Initial fundamental frequency for subsequent layers.
            n_terms: Comb capacity (sphere dimension) per neuron.
            m0: Modulus seeding the init comb direction (pure sine at ``m0 -> 0``,
                harmonic-rich near ``m0 -> 1``). Values outside ``(0, 1)``
                saturate silently at the endpoints (clipped to
                ``[m_eps, 1 - m_eps]`` inside the q-series); this is intended —
                ``m0 = 0`` is the documented pure-sine corner — so it is not
                rejected.
            m0_spread: Per-neuron Gaussian spread of ``m0`` at init; ``0`` seeds
                every neuron identically.
            m_eps: Clip margin keeping ``m0`` off the singular endpoints.
        """
        out_features = _validate_body_args(num_hidden_layers, out_features)
        keys = jax.random.split(key, num_hidden_layers + 1)
        layers = []
        for i in range(num_hidden_layers):
            in_d = in_dim if i == 0 else hidden_dim
            o = omega_first if i == 0 else omega_hidden
            layers.append(
                HarmonicCombLayer(
                    in_d,
                    hidden_dim,
                    o,
                    is_first=(i == 0),
                    key=keys[i],
                    n_terms=n_terms,
                    m0=m0,
                    m0_spread=m0_spread,
                    m_eps=m_eps,
                )
            )
        self.layers = tuple(layers)
        self.readout_W, self.readout_b = _build_readout(hidden_dim, omega_hidden, out_features, keys[-1])
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
