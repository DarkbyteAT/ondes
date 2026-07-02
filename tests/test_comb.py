"""Tests for the odd-harmonic sine-comb basis family (ondes.basis.comb).

Four load-bearing properties, each with a test that would catch its failure:

1. **q-series correctness** — the shared ``sn`` machinery reconstructs the true
   Jacobi ``sn`` (validated against ``scipy.special.ellipj``). The reference
   ``JacobiSn`` module lives here, not in the library surface: it is the frozen
   ``ellipj`` fixture (with the geometrically-faithful ``match_freq=False``
   fundamental and the ``use_sin`` short-circuit at the SIREN corner), not a
   basis anyone trains.
2. **variance preservation at init** — unit-norm coefficients pin per-neuron
   ``Var[phi] ~= 1/2`` under unit-variance pre-activations in the wrapping regime.
3. **gradient flow** — the learnable leaves (``omega``, ``raw_m``, ``raw_c``)
   receive finite, non-zero gradients.
4. **pytree homogeneity** — every layer of a body shares one pytree structure,
   the precondition for ``jax.lax.scan`` / ``jax.vmap`` over the layer stack.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ellipj

from ondes.basis import (
    Basis,
    BasisModule,
    Body,
    HarmonicComb,
    HarmonicCombLayer,
    JacobiLearnM,
    JacobiLearnMLayer,
)
from ondes.basis.comb import _q_K, _sn_unit_coeffs


BODY_LAYER_PAIRS = ((JacobiLearnM, JacobiLearnMLayer), (HarmonicComb, HarmonicCombLayer))


# --------------------------------------------------------------------------- #
# Frozen ellipj reference fixture (test-only; not part of the library surface) #
# --------------------------------------------------------------------------- #
def _sn_faithful_coeffs(m, n_terms, m_eps=1e-3):
    """True ``sn`` q-series coefficients, prefactor ``2*pi/(sqrt(m)*K)`` KEPT.

    The library's ``_sn_unit_coeffs`` drops this prefactor (it cancels under L2
    normalisation); keeping it makes the comb evaluate the literal ``sn(., m)``
    bounded in ``[-1, 1]`` — the form that matches ``scipy.special.ellipj``.
    """
    m, k, q = _q_K(jnp.asarray(m, jnp.float32), m_eps)
    n = jnp.arange(n_terms, dtype=q.dtype)
    pref = 2.0 * jnp.pi / (jnp.sqrt(m) * k)
    qn = q[..., None]
    return pref[..., None] * qn ** (n + 0.5) / (1.0 - qn ** (2.0 * n + 1.0))


class JacobiSn(eqx.Module):
    """Frozen reference: literal ``sn(omega * x, m)`` at a fixed scalar ``m``.

    The one non-unit-norm member of the comb family — it evaluates the true
    ``sn`` (range ``[-1, 1]``), so it doubles as the ``scipy.special.ellipj``
    validation point. Coefficients and frequencies are constants computed once
    at ``__init__`` and ``stop_gradient``'d in the forward pass.

    ``match_freq=True`` rescales the fundamental to ``omega`` (harmonic content
    only changes with ``m``); ``match_freq=False`` uses the geometrically
    faithful fundamental ``(pi / 2K) * omega`` needed to reproduce ``ellipj``.
    """

    coeffs: jnp.ndarray
    eff_freqs: jnp.ndarray
    omega: float = eqx.field(static=True)
    m: float = eqx.field(static=True)
    n_terms: int = eqx.field(static=True)
    match_freq: bool = eqx.field(static=True)
    use_sin: bool = eqx.field(static=True)

    def __init__(self, omega=20.0, m=0.5, n_terms=8, match_freq=True, m_eps=1e-3):
        self.omega = float(omega)
        self.m = float(m)
        self.n_terms = int(n_terms)
        self.match_freq = bool(match_freq)
        # sn(., 0) is literally sin — short-circuit avoids the 1/sqrt(m)
        # prefactor at the SIREN corner.
        self.use_sin = float(m) <= m_eps
        n = jnp.arange(n_terms, dtype=jnp.float32)
        odd = 2.0 * n + 1.0
        if self.use_sin:
            self.coeffs = jnp.zeros(n_terms).at[0].set(1.0)
            self.eff_freqs = odd * self.omega
        else:
            mm = jnp.asarray(m, jnp.float32)
            self.coeffs = _sn_faithful_coeffs(mm, n_terms, m_eps)
            _, k, _ = _q_K(mm, m_eps)
            self.eff_freqs = odd * (self.omega if match_freq else (jnp.pi / (2.0 * k)) * self.omega)

    def __call__(self, x):
        coeffs = jax.lax.stop_gradient(self.coeffs)
        eff = jax.lax.stop_gradient(self.eff_freqs)
        return jnp.sum(coeffs * jnp.sin(eff * x[..., None]), axis=-1)


# --------------------------------------------------------------------------- #
# 1. q-series correctness vs scipy.special.ellipj                             #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("m", [0.1, 0.5, 0.9])
def test_jacobi_sn_reference_matches_scipy_ellipj(m: float) -> None:
    # Given: the frozen JacobiSn reference with the geometrically-faithful
    # fundamental (match_freq=False), which reconstructs the literal sn waveform
    # When: evaluating it on a grid and comparing to scipy.special.ellipj
    # Then: the AGM q-series machinery (K_agm, _q_K, _sn_direction) reproduces
    # sn to truncation accuracy. A bug in K(m), the nome q, or the amplitude
    # recurrence would push this far past the tolerance.
    ref = JacobiSn(omega=1.0, m=m, n_terms=16, match_freq=False)
    x = jnp.linspace(-2.0, 2.0, 64)
    comb = ref(x)
    sn_true = ellipj(np.asarray(x), m)[0]
    # atol 1e-4: float32 accumulation through 16 harmonics plus q-series
    # truncation (q < 0.15 for m <= 0.9, so 16 terms is ~1e-6-exact); a broken
    # nome/K would show O(1) error, so this bound is load-bearing.
    assert jnp.allclose(comb, jnp.asarray(sn_true), atol=1e-4)


@pytest.mark.unit
def test_jacobi_sn_use_sin_corner_equals_sine() -> None:
    # Given: JacobiSn at m -> 0 (the SIREN corner), which short-circuits to sin
    # When: evaluating against jnp.sin(omega * x)
    # Then: it is exactly a fundamental sine — confirming the family nests SIREN
    # at m = 0 with no residual harmonic leakage.
    ref = JacobiSn(omega=3.0, m=0.0, n_terms=8)
    assert ref.use_sin
    x = jnp.linspace(-1.0, 1.0, 32)
    assert jnp.allclose(ref(x), jnp.sin(3.0 * x), atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("m", [0.1, 0.5, 0.9])
def test_unit_coeffs_are_normalised_faithful_coeffs(m: float) -> None:
    # Given: the library unit-norm coeffs and the reference faithful coeffs at m
    # When: L2-normalising the faithful coeffs
    # Then: they coincide — the two forms differ only by the dropped
    # 2*pi/(sqrt(m)*K) prefactor, which is exactly what normalisation removes.
    # Ties the library's Var-preserving form to the ellipj-validated one.
    n_terms = 16
    unit = _sn_unit_coeffs(jnp.asarray(m, jnp.float32), n_terms, 1e-3)
    faithful = _sn_faithful_coeffs(jnp.asarray(m, jnp.float32), n_terms)
    normalised = faithful / jnp.linalg.norm(faithful)
    assert jnp.allclose(unit, normalised, atol=1e-3)


# --------------------------------------------------------------------------- #
# 2. variance preservation at init                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("layer_cls", [JacobiLearnMLayer, HarmonicCombLayer])
def test_init_variance_is_one_half_per_neuron(layer_cls: type) -> None:
    # Given: a freshly-initialised comb layer with omega high enough to be in
    # the wrapping regime (omega * std(pre) >> 1), fed unit-variance pre-acts
    # When: computing the per-neuron variance of _activate over many samples
    # Then: Var[phi_j] ~= 1/2 for every neuron. This is the invariant that lets
    # the SIREN-family init carry over unchanged: unit-norm coefficients +
    # wrapping give (1/2)*||c_j||^2 = 1/2. A coefficient-norm regression (rows
    # not on the unit sphere) would move this off 1/2.
    out_dim = 128
    layer = layer_cls(in_dim=3, out_dim=out_dim, omega_init=30.0, is_first=False, key=jax.random.key(0))
    pre = jax.random.normal(jax.random.key(1), (8192, out_dim))
    out = jax.vmap(layer._activate)(pre)
    per_neuron_var = out.var(axis=0)
    # atol 0.05: finite-sample (8192) std of a variance estimate plus the small
    # O(exp(-omega^2)) wrapping-regime correction. A structural break (missing
    # normalisation) would land near ||raw||^2 / 2, well outside this band.
    assert jnp.allclose(per_neuron_var, 0.5, atol=0.05), f"mean Var {float(per_neuron_var.mean())}"


@pytest.mark.unit
@pytest.mark.parametrize("layer_cls", [JacobiLearnMLayer, HarmonicCombLayer])
def test_init_output_is_finite(layer_cls: type) -> None:
    # Given: a comb layer on a deterministic input
    # When: forward-passing the full layer (linear + activation)
    # Then: output is finite and correctly shaped
    layer = layer_cls(in_dim=4, out_dim=16, omega_init=6.0, is_first=True, key=jax.random.key(2))
    y = layer(jnp.linspace(-1.0, 1.0, 4))
    assert y.shape == (16,)
    assert bool(jnp.all(jnp.isfinite(y)))


# --------------------------------------------------------------------------- #
# 3. gradient flow through the learnable leaves                               #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("body_cls,leaf_name", [(JacobiLearnM, "raw_m"), (HarmonicComb, "raw_c")])
def test_gradient_flows_to_learnable_leaves(body_cls: type, leaf_name: str) -> None:
    # Given: a comb body and a trivial scalar loss
    # When: taking filter_grad over the body
    # Then: the activation-specific leaf (raw_m / raw_c) AND omega receive finite,
    # non-zero gradients on every layer. A stop_gradient regression on the comb
    # coefficients (copied from the frozen JacobiSn reference) would zero these.
    body = body_cls(in_dim=2, hidden_dim=32, num_hidden_layers=3, key=jax.random.key(3))
    coord = jnp.array([0.3, -0.4])

    def loss(b, c):
        return b(c).sum()

    grad = eqx.filter_grad(loss)(body, coord)
    for i, layer in enumerate(grad.layers):
        g_leaf = getattr(layer, leaf_name)
        assert bool(jnp.all(jnp.isfinite(g_leaf))), f"layer {i} {leaf_name} grad non-finite"
        assert bool(jnp.any(g_leaf != 0.0)), f"layer {i} {leaf_name} grad all-zero"
        assert bool(jnp.all(jnp.isfinite(layer.omega))), f"layer {i} omega grad non-finite"
        assert bool(jnp.any(layer.omega != 0.0)), f"layer {i} omega grad all-zero"


@pytest.mark.unit
def test_gradient_through_normalisation_is_finite_at_near_zero_row() -> None:
    # Given: a HarmonicComb layer with one raw_c row driven to exactly zero —
    # the gauge-pathology point where plain x/||x|| gives a NaN gradient
    # When: taking the gradient of _activate through the sphere normalisation
    # Then: every gradient stays finite. The eps-floored _unit_normalize is what
    # makes this hold; a bare jnp.linalg.norm denominator would NaN here.
    layer = HarmonicCombLayer(in_dim=1, out_dim=4, omega_init=2.0, is_first=True, key=jax.random.key(30))
    zeroed = layer.raw_c.at[0].set(0.0)
    layer = eqx.tree_at(lambda t: t.raw_c, layer, zeroed)
    pre = jnp.array([0.3, -0.4, 0.5, 0.7])

    grad = eqx.filter_grad(lambda la, p: la._activate(p).sum())(layer, pre)
    assert bool(jnp.all(jnp.isfinite(grad.raw_c))), "normalisation grad non-finite at zero row"


# --------------------------------------------------------------------------- #
# 4. pytree homogeneity across layers (scan/vmap compatibility)               #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("body_cls,layer_cls", BODY_LAYER_PAIRS)
def test_layer_pytree_structure_is_homogeneous_across_body(body_cls: type, layer_cls: type) -> None:
    # Given: a body in the realistic in_dim != hidden_dim shape
    # When: comparing every layer's pytree structure against layer 0
    # Then: all layers share one structure. tree_structure compares static-field
    # values and leaf positions (not array shapes), so this holds despite layer
    # 0's differently-shaped W. It is the precondition for scanning the stack;
    # a per-layer static field (e.g. an is_first flag or a varying n_terms)
    # would break it.
    body = body_cls(in_dim=2, hidden_dim=64, num_hidden_layers=6, key=jax.random.key(4))
    ref = jax.tree_util.tree_structure(body.layers[0])
    for i, layer in enumerate(body.layers):
        assert isinstance(layer, layer_cls)
        assert jax.tree_util.tree_structure(layer) == ref, f"layer {i} structure differs from layer 0"


@pytest.mark.unit
@pytest.mark.parametrize("body_cls,layer_cls", BODY_LAYER_PAIRS)
def test_full_layer_stack_scans_when_in_dim_equals_hidden_dim(body_cls: type, layer_cls: type) -> None:
    # Given: a comb body with in_dim == hidden_dim so all N layers stack
    # (uniform structure AND uniform shapes)
    # When: stacking the array leaves and running jax.lax.scan over the trunk
    # Then: the scanned trunk matches the eager for-loop. Demonstrates the
    # homogeneity invariant is not just structural but scan-usable end to end.
    body = body_cls(in_dim=4, hidden_dim=4, num_hidden_layers=5, key=jax.random.key(5))
    coord = jnp.array([0.1, -0.2, 0.3, -0.4])
    eager = body.trunk(coord)

    _, static = eqx.partition(body.layers[0], eqx.is_array)
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[eqx.filter(la, eqx.is_array) for la in body.layers])

    def step(h, layer_arrays):
        return eqx.combine(layer_arrays, static)(h), None

    scanned, _ = jax.lax.scan(step, coord, stacked)
    assert eager.shape == scanned.shape == (4,)
    assert jnp.allclose(eager, scanned, atol=1e-4)


# --------------------------------------------------------------------------- #
# family nesting: SIREN and sn are special points of the comb                 #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_harmonic_comb_recovers_siren_at_e0() -> None:
    # Given: a HarmonicComb layer whose coefficients are forced to e_0
    # When: comparing _activate to sin(omega * pre)
    # Then: it is exactly a fundamental sine — HarmonicComb genuinely nests
    # SIREN as the c = e_0 corner (tests the "superset" claim, not just runs).
    layer = HarmonicCombLayer(in_dim=1, out_dim=4, omega_init=2.0, is_first=True, key=jax.random.key(6))
    e0 = jnp.zeros_like(layer.raw_c).at[:, 0].set(1.0)
    layer = eqx.tree_at(lambda t: t.raw_c, layer, e0)
    pre = jnp.array([0.3, -0.4, 0.5, 0.7])
    assert jnp.allclose(layer._activate(pre), jnp.sin(2.0 * pre), atol=1e-6)


@pytest.mark.unit
def test_jacobi_learn_m_approaches_sine_as_m_goes_to_zero() -> None:
    # Given: a JacobiLearnM layer with raw_m forced very negative (sigmoid -> 0)
    # When: comparing _activate to sin(omega * pre)
    # Then: it collapses onto the fundamental sine — the sn manifold meets SIREN
    # at m -> 0. Not exact (m is clipped at m_eps, not 0), so a loose tolerance.
    layer = JacobiLearnMLayer(in_dim=1, out_dim=4, omega_init=2.0, is_first=True, key=jax.random.key(7))
    layer = eqx.tree_at(lambda t: t.raw_m, layer, jnp.full_like(layer.raw_m, -20.0))
    pre = jnp.array([0.3, -0.4, 0.5, 0.7])
    assert jnp.allclose(layer._activate(pre), jnp.sin(2.0 * pre), atol=1e-3)


# --------------------------------------------------------------------------- #
# basis-family contracts (mirrors test_basis for the new bodies)              #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("layer_cls", [JacobiLearnMLayer, HarmonicCombLayer])
def test_comb_layer_is_subclass_of_basis_abc(layer_cls: type) -> None:
    # Given: each concrete comb layer
    # When: checking isinstance against the Basis ABC
    # Then: each is a Basis (the polymorphism contract)
    layer = layer_cls(in_dim=2, out_dim=4, omega_init=6.0, is_first=True, key=jax.random.key(0))
    assert isinstance(layer, Basis)


@pytest.mark.unit
@pytest.mark.parametrize("body_cls", [JacobiLearnM, HarmonicComb])
def test_comb_body_conforms_to_body_and_protocol(body_cls: type) -> None:
    # Given: a comb body
    # When: checking against the Body base and the BasisModule protocol
    # Then: it satisfies both — downstream code can type against either.
    body = body_cls(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=jax.random.key(0))
    assert isinstance(body, Body)
    assert isinstance(body, BasisModule)


@pytest.mark.unit
@pytest.mark.parametrize("body_cls", [JacobiLearnM, HarmonicComb])
def test_comb_body_out_features_scalar_and_vector(body_cls: type) -> None:
    # Given: comb bodies with default and integer out_features
    # When: forward-passing
    # Then: default gives a 0-d scalar, out_features=3 gives shape (3,)
    coord = jnp.array([0.1, -0.2])
    scalar = body_cls(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(1))
    vector = body_cls(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(1), out_features=3)
    assert scalar(coord).shape == ()
    assert vector(coord).shape == (3,)


@pytest.mark.unit
@pytest.mark.parametrize("body_cls", [JacobiLearnM, HarmonicComb])
def test_comb_body_is_jit_compilable(body_cls: type) -> None:
    # Given: a comb body and a coordinate
    # When: jitting the call
    # Then: the jit-compiled result matches eager
    body = body_cls(in_dim=2, hidden_dim=32, num_hidden_layers=3, key=jax.random.key(2))
    coord = jnp.array([0.25, -0.5])
    assert jnp.allclose(body(coord), eqx.filter_jit(body)(coord))


@pytest.mark.unit
@pytest.mark.parametrize("body_cls", [JacobiLearnM, HarmonicComb])
def test_comb_body_film_modulation_changes_output(body_cls: type) -> None:
    # Given: a comb body run with and without FiLM
    # When: passing a non-trivial FiLM tensor
    # Then: outputs differ — FiLM is wired through the inherited _pre path
    num_layers, hidden_dim = 2, 8
    body = body_cls(in_dim=2, hidden_dim=hidden_dim, num_hidden_layers=num_layers, key=jax.random.key(3))
    coord = jnp.array([0.3, -0.7])
    film = jnp.ones((num_layers, 2 * hidden_dim)) * 0.5
    assert not jnp.allclose(body(coord), body(coord, film=film))


@pytest.mark.unit
def test_jacobi_learn_m_and_harmonic_comb_are_distinct_types() -> None:
    # Given: the two comb bodies with identical hyperparameters
    # When: checking their types and layer types
    # Then: each is its own class — the kind discriminator is the type, not a
    # string field (no _OddHarmonicComb ABC collapse).
    key = jax.random.key(0)
    jl = JacobiLearnM(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=key)
    hc = HarmonicComb(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=key)
    assert type(jl) is JacobiLearnM
    assert type(hc) is HarmonicComb
    assert type(jl.layers[0]) is JacobiLearnMLayer
    assert type(hc.layers[0]) is HarmonicCombLayer
