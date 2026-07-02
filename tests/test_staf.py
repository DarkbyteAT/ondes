"""Tests for the STAF basis family (ondes.basis.staf).

STAF (Morsali+ 2026, TMLR) is the per-layer-shared trainable sinusoidal mixture
    phi(pre) = sum_{i<tau} amp_i * sin(freq_i * pre + phase_i)
with one (amp, freq, phase) triple shared across every neuron of a layer.

Load-bearing properties, each with a test that would catch its failure:

1. **faithful init** — freq ~ omega_0*U[0,1), phase ~ U(-pi, pi), and
   E[amp^2] = 2/tau (the moment-matching amplitude calibration).
2. **family nesting** — tau=1/phase=0/amp=1/freq=omega reduces exactly to SIREN;
   freq on the odd lattice with phase=0 and unit amp recovers HarmonicComb.
3. **per-layer sharing** — one triple per layer, not per neuron; equal inputs at
   two neurons give equal outputs (a per-neuron parameterisation would not).
4. **param arithmetic** — 3*tau per activation-bearing layer (paper Table 7).
5. **omega is a fixed init scale** — it feeds siren_init and the freq init only,
   so it receives zero gradient; the learnable frequencies live in ``freq``.
6. **pytree homogeneity** — every layer shares one structure (scan/vmap ready).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from ondes.basis import (
    STAF,
    Basis,
    BasisModule,
    Body,
    HarmonicCombLayer,
    STAFLayer,
)
from ondes.basis.comb import _odd_freqs, _unit_normalize
from ondes.basis.staf import _staf_init


# --------------------------------------------------------------------------- #
# 1. faithful init statistics                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("tau", [5, 10])
def test_amp_init_second_moment_matches_two_over_tau(tau: int) -> None:
    # Given: many independent draws of STAF's amplitude vector at fixed tau,
    # where amp_i = sign(X)*sqrt(|X|) with X ~ Laplace(0, 2/tau) so amp^2 = |X|
    # has mean 2/tau and std 2/tau (an Exponential(scale=2/tau)).
    # When: estimating E[amp^2] over the pooled sample.
    # Then: it matches 2/tau within 5 standard errors — the calibration behind
    # STAF's exact N(0,1) post-activations. A missing/​wrong Laplace scale (or a
    # dropped signed-sqrt) shifts this mean far past 5*SE.
    n_keys = 4000
    keys = jax.random.split(jax.random.key(0), n_keys)
    amps, _, _ = jax.vmap(lambda k: _staf_init(tau, 30.0, k))(keys)  # (n_keys, tau)
    mean_sq = float((amps**2).mean())
    n = amps.size
    se = (2.0 / tau) / jnp.sqrt(n)  # standard error of the mean of an Exp(scale=2/tau)
    assert abs(mean_sq - 2.0 / tau) < 5.0 * float(se), f"E[amp^2]={mean_sq}, target={2.0 / tau}"


@pytest.mark.unit
def test_freq_init_lies_in_zero_to_omega0() -> None:
    # Given: a STAF activation triple built at omega_0 = 30 (freq = 30*U[0,1)).
    # When: inspecting the frequency vector.
    # Then: every frequency lies in [0, 30) — the structural range of the scaled
    # uniform. A sign flip or a wrong scale would leave this band.
    omega_0 = 30.0
    _, freq, _ = _staf_init(tau=64, omega_0=omega_0, key=jax.random.key(1))
    assert bool(jnp.all(freq >= 0.0))
    assert bool(jnp.all(freq < omega_0))


@pytest.mark.unit
def test_phase_init_lies_in_minus_pi_to_pi() -> None:
    # Given: a STAF activation triple.
    # When: inspecting the phase vector.
    # Then: every phase lies in [-pi, pi) — required by the variance theorem to
    # centre each sinusoidal term. A mis-scaled uniform would leave this band.
    _, _, phase = _staf_init(tau=64, omega_0=30.0, key=jax.random.key(2))
    assert bool(jnp.all(phase >= -jnp.pi))
    assert bool(jnp.all(phase < jnp.pi))


@pytest.mark.unit
@pytest.mark.parametrize("is_first", [True, False])
def test_weight_init_respects_siren_bound(is_first: bool) -> None:
    # Given: a STAF layer, whose weights use SIREN uniform init with the layer's
    # omega_0 as the frequency scale (first: |W| <= 1/in_dim; hidden:
    # |W| <= sqrt(6/in_dim)/omega_0).
    # When: constructing at omega_0 = 30.
    # Then: |W| and |b| stay within the SIREN bound. Dropping the /omega_0 factor
    # (a 30x larger bound) or using the first-layer bound on a hidden layer both
    # break this — the bound is computed here from the paper formula, not guessed.
    in_dim, out_dim, omega_0 = 3, 64, 30.0
    layer = STAFLayer(in_dim=in_dim, out_dim=out_dim, omega_init=omega_0, is_first=is_first, key=jax.random.key(3))
    bound = 1.0 / in_dim if is_first else float(jnp.sqrt(6.0 / in_dim) / omega_0)
    assert bool(jnp.all(jnp.abs(layer.W) <= bound))
    assert bool(jnp.all(jnp.abs(layer.b) <= bound))


# --------------------------------------------------------------------------- #
# 2. family nesting: SIREN and HarmonicComb are special points of STAF        #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_activation_reduces_to_siren_at_tau_one() -> None:
    # Given: a STAF layer with tau=1 whose triple is forced to (amp=1, phase=0,
    # freq=omega) — the paper's positioning sentence "SIREN is the tau=1 case".
    # When: comparing _activate to sin(omega * pre).
    # Then: it is exactly the fundamental sine. Every op is an identity (x*1, +0,
    # sum of one), so the match is bit-tight — tolerance from dtype eps, not a
    # magic threshold.
    omega = 2.0
    layer = STAFLayer(in_dim=1, out_dim=4, omega_init=omega, is_first=True, key=jax.random.key(4), tau=1)
    layer = eqx.tree_at(
        lambda t: (t.amp, t.phase, t.freq),
        layer,
        (jnp.ones(1), jnp.zeros(1), jnp.array([omega])),
    )
    pre = jnp.array([0.3, -0.4, 0.5, 0.7])
    tol = jnp.finfo(pre.dtype).eps * 8
    assert jnp.allclose(layer._activate(pre), jnp.sin(omega * pre), atol=tol)


@pytest.mark.unit
def test_activation_reduces_to_harmonic_comb_on_odd_lattice() -> None:
    # Given: a STAF layer whose freq is clamped to the odd-harmonic lattice
    # (2k+1)*omega with phase=0 and amp a unit-norm coefficient row, and a
    # HarmonicCombLayer sharing that same row (m0_spread=0 makes every neuron's
    # row identical, i.e. layer-shared like STAF).
    # When: comparing the two _activate outputs.
    # Then: they coincide — STAF genuinely nests HarmonicComb on the locked,
    # fixed-phase, unit-sphere submanifold. Tolerance from dtype eps scaled by
    # the term count (only sum-reassociation noise separates them).
    omega, n = 2.0, 6
    hc = HarmonicCombLayer(in_dim=1, out_dim=5, omega_init=omega, is_first=True, key=jax.random.key(20), n_terms=n)
    shared_c = _unit_normalize(hc.raw_c)[0]  # all rows equal at m0_spread=0
    staf = STAFLayer(in_dim=1, out_dim=5, omega_init=omega, is_first=True, key=jax.random.key(21), tau=n)
    staf = eqx.tree_at(
        lambda t: (t.freq, t.phase, t.amp),
        staf,
        (_odd_freqs(n, staf.omega), jnp.zeros(n), shared_c),
    )
    pre = jnp.array([0.3, -0.4, 0.5, 0.7, -0.2])
    tol = jnp.finfo(pre.dtype).eps * n * 4
    assert jnp.allclose(staf._activate(pre), hc._activate(pre), atol=tol)


@pytest.mark.unit
def test_phase_enters_inside_the_sine_with_correct_sign() -> None:
    # Given: a STAF layer forced to (tau=1, amp=[1], freq=[0], phase=[pi/2]).
    # When: evaluating _activate on any pre.
    # Then: it is exactly 1.0 for every input, since sin(0*pre + pi/2) = 1. Both
    # reduction oracles zero the phase, so a phase bug — added OUTSIDE the sine
    # (=> pi/2), sign-flipped (sin(-pi/2) => -1), or doubled (sin(pi) => 0) —
    # ships green there but fails here. Pins forward-faithfulness of the phase
    # (released repo: bs * sin(ws*u + phis)).
    layer = STAFLayer(in_dim=1, out_dim=4, omega_init=30.0, is_first=True, key=jax.random.key(22), tau=1)
    layer = eqx.tree_at(
        lambda t: (t.amp, t.freq, t.phase),
        layer,
        (jnp.ones(1), jnp.zeros(1), jnp.array([jnp.pi / 2])),
    )
    pre = jnp.array([-0.9, -0.1, 0.3, 0.8])
    tol = jnp.finfo(pre.dtype).eps * 16
    assert jnp.allclose(layer._activate(pre), jnp.ones_like(pre), atol=tol)


@pytest.mark.unit
def test_amp_is_unconstrained_output_scales_linearly() -> None:
    # Given: a STAF layer and a scalar k.
    # When: scaling amp by k (freq/phase/pre fixed).
    # Then: the activation scales by exactly k — _activate is linear in amp, with
    # NO norm re-projection. This pins STAF's deliberate no-sphere-constraint
    # contract (the load-bearing difference from HarmonicComb): a regression to
    # unit-normalised amp would make the output invariant to k, passing every
    # other test but failing here.
    k = 3.7
    layer = STAFLayer(in_dim=1, out_dim=4, omega_init=30.0, is_first=True, key=jax.random.key(23), tau=5)
    pre = jnp.array([0.3, -0.4, 0.5, 0.7])
    base = layer._activate(pre)
    scaled = eqx.tree_at(lambda t: t.amp, layer, layer.amp * k)
    tol = jnp.finfo(pre.dtype).eps * 16 * k
    assert jnp.allclose(scaled._activate(pre), k * base, atol=tol)


# --------------------------------------------------------------------------- #
# 3. per-layer sharing (one triple per layer, not per neuron)                 #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_activation_triple_is_layer_shared_not_per_neuron() -> None:
    # Given: a STAF layer.
    # When: inspecting the triple shapes and feeding pre-activations with two
    # equal entries.
    # Then: (a) amp/freq/phase are (tau,), one triple for the whole layer; and
    # (b) equal inputs at neurons 0 and 3 give bit-identical outputs — the
    # activation is one shared scalar function applied element-wise. A per-neuron
    # parameterisation (shape (out, tau)) would give different outputs even for
    # equal inputs, so this is the behavioural discriminator of the granularity.
    tau = 5
    layer = STAFLayer(in_dim=1, out_dim=8, omega_init=30.0, is_first=True, key=jax.random.key(5), tau=tau)
    assert layer.amp.shape == layer.freq.shape == layer.phase.shape == (tau,)
    pre = jnp.array([0.42, -0.1, 0.9, 0.42, -0.7, 0.3, 0.42, -0.5])  # entries 0,3,6 equal
    out = layer._activate(pre)
    assert float(out[0]) == float(out[3]) == float(out[6])


# --------------------------------------------------------------------------- #
# 4. parameter-count arithmetic                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_param_count_is_three_tau_per_activation_layer() -> None:
    # Given: a STAF body with tau=5 over 4 hidden layers.
    # When: counting the activation leaves (amp + freq + phase) per layer and in
    # total.
    # Then: each layer adds exactly 3*tau, and the body adds 3*tau*L = 60 — the
    # paper's "+60 parameters for tau=5" over SIREN at four activation layers
    # (Table 7). Independent of width, which is the whole point of layer-sharing.
    tau, num_layers = 5, 4
    body = STAF(in_dim=2, hidden_dim=64, num_hidden_layers=num_layers, key=jax.random.key(6), tau=tau)
    per_layer = [lyr.amp.size + lyr.freq.size + lyr.phase.size for lyr in body.layers]
    assert all(p == 3 * tau for p in per_layer)
    assert sum(per_layer) == 3 * tau * num_layers == 60


@pytest.mark.unit
def test_default_tau_is_five() -> None:
    # Given: a STAF layer built with default kwargs.
    # When: inspecting the triple length.
    # Then: tau defaults to 5 — the paper's default for all tasks bar denoising.
    layer = STAFLayer(in_dim=2, out_dim=8, omega_init=30.0, is_first=False, key=jax.random.key(7))
    assert layer.amp.shape == (5,)


# --------------------------------------------------------------------------- #
# 5. gradient flow and the fixed omega leaf                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_gradient_flows_to_activation_leaves() -> None:
    # Given: a STAF body and a trivial scalar loss.
    # When: taking filter_grad over the body.
    # Then: amp, freq, and phase each receive finite, non-zero gradients on every
    # layer. A stop_gradient regression on any of them would zero it out here.
    body = STAF(in_dim=2, hidden_dim=32, num_hidden_layers=3, key=jax.random.key(8))
    coord = jnp.array([0.3, -0.4])
    grad = eqx.filter_grad(lambda b, c: b(c).sum())(body, coord)
    for i, layer in enumerate(grad.layers):
        for name in ("amp", "freq", "phase"):
            g = getattr(layer, name)
            assert bool(jnp.all(jnp.isfinite(g))), f"layer {i} {name} grad non-finite"
            assert bool(jnp.any(g != 0.0)), f"layer {i} {name} grad all-zero"


@pytest.mark.unit
def test_omega_is_a_fixed_dead_leaf() -> None:
    # Given: a STAF body and a scalar loss.
    # When: taking the gradient.
    # Then: omega receives exactly zero gradient on every layer — it is the
    # init-only frequency scale (siren_init bound + freq init), never read at
    # forward time. The learnable frequencies live in ``freq``; a design drift
    # that re-introduced omega into the activation would make this non-zero.
    body = STAF(in_dim=2, hidden_dim=32, num_hidden_layers=3, key=jax.random.key(9))
    coord = jnp.array([0.25, -0.5])
    grad = eqx.filter_grad(lambda b, c: b(c).sum())(body, coord)
    for i, layer in enumerate(grad.layers):
        assert bool(jnp.all(layer.omega == 0.0)), f"layer {i} omega expected fixed (grad 0)"


# --------------------------------------------------------------------------- #
# 6. pytree homogeneity across layers (scan/vmap compatibility)               #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_layer_pytree_structure_is_homogeneous_across_body() -> None:
    # Given: a body in the realistic in_dim != hidden_dim shape.
    # When: comparing every layer's pytree structure against layer 0.
    # Then: all layers share one structure — tau lives in the array shapes, not a
    # static field, so no per-layer discriminator breaks scan over the stack.
    body = STAF(in_dim=2, hidden_dim=64, num_hidden_layers=6, key=jax.random.key(10))
    ref = jax.tree_util.tree_structure(body.layers[0])
    for i, layer in enumerate(body.layers):
        assert isinstance(layer, STAFLayer)
        assert jax.tree_util.tree_structure(layer) == ref, f"layer {i} structure differs from layer 0"


@pytest.mark.unit
def test_full_layer_stack_scans_when_in_dim_equals_hidden_dim() -> None:
    # Given: a STAF body with in_dim == hidden_dim so all N layers stack.
    # When: stacking the array leaves and running jax.lax.scan over the trunk.
    # Then: the scanned trunk matches the eager for-loop — homogeneity is not
    # just structural but scan-usable end to end.
    body = STAF(in_dim=4, hidden_dim=4, num_hidden_layers=5, key=jax.random.key(11))
    coord = jnp.array([0.1, -0.2, 0.3, -0.4])
    eager = body.trunk(coord)

    _, static = eqx.partition(body.layers[0], eqx.is_array)
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[eqx.filter(la, eqx.is_array) for la in body.layers])

    def step(h, layer_arrays):
        return eqx.combine(layer_arrays, static)(h), None

    scanned, _ = jax.lax.scan(step, coord, stacked)
    assert eager.shape == scanned.shape == (4,)
    assert jnp.allclose(eager, scanned, atol=jnp.finfo(coord.dtype).eps * 64)


# --------------------------------------------------------------------------- #
# basis-family contracts (mirrors test_basis / test_comb for the new body)    #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_layer_output_is_finite_and_shaped() -> None:
    # Given: a STAF layer on a deterministic input.
    # When: forward-passing the full layer (linear + activation).
    # Then: output is finite and correctly shaped.
    layer = STAFLayer(in_dim=4, out_dim=16, omega_init=30.0, is_first=True, key=jax.random.key(12))
    y = layer(jnp.linspace(-1.0, 1.0, 4))
    assert y.shape == (16,)
    assert bool(jnp.all(jnp.isfinite(y)))


@pytest.mark.unit
def test_layer_is_subclass_of_basis_abc() -> None:
    # Given: a STAF layer.
    # When: checking isinstance against the Basis ABC.
    # Then: it is a Basis (the polymorphism contract).
    layer = STAFLayer(in_dim=2, out_dim=4, omega_init=30.0, is_first=True, key=jax.random.key(13))
    assert isinstance(layer, Basis)


@pytest.mark.unit
def test_body_conforms_to_body_and_protocol() -> None:
    # Given: a STAF body.
    # When: checking against the Body base and the BasisModule protocol.
    # Then: it satisfies both — downstream code can type against either.
    body = STAF(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=jax.random.key(14))
    assert isinstance(body, Body)
    assert isinstance(body, BasisModule)


@pytest.mark.unit
def test_body_out_features_scalar_and_vector() -> None:
    # Given: STAF bodies with default and integer out_features.
    # When: forward-passing.
    # Then: default gives a 0-d scalar, out_features=3 gives shape (3,).
    coord = jnp.array([0.1, -0.2])
    scalar = STAF(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(15))
    vector = STAF(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(15), out_features=3)
    assert scalar(coord).shape == ()
    assert vector(coord).shape == (3,)


@pytest.mark.unit
def test_out_features_one_canonicalises_to_none() -> None:
    # Given: STAF bodies built with out_features=1 and out_features=None.
    # When: comparing their pytree structures.
    # Then: they are identical — the 1 -> None canonicalisation keeps the two
    # scalar-yielding constructions indistinguishable (serialisation/jit-cache
    # invariant from _validate_body_args).
    one = STAF(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=jax.random.key(16), out_features=1)
    none = STAF(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=jax.random.key(16), out_features=None)
    assert jax.tree_util.tree_structure(one) == jax.tree_util.tree_structure(none)
    assert one.out_features is None


@pytest.mark.unit
def test_body_is_jit_compilable() -> None:
    # Given: a STAF body and a coordinate.
    # When: jitting the call.
    # Then: the jit-compiled result matches eager.
    body = STAF(in_dim=2, hidden_dim=32, num_hidden_layers=3, key=jax.random.key(17))
    coord = jnp.array([0.25, -0.5])
    assert jnp.allclose(body(coord), eqx.filter_jit(body)(coord))


@pytest.mark.unit
def test_body_film_modulation_changes_output() -> None:
    # Given: a STAF body run with and without FiLM.
    # When: passing a non-trivial FiLM tensor.
    # Then: outputs differ — FiLM is wired through the inherited Body.trunk path.
    num_layers, hidden_dim = 2, 8
    body = STAF(in_dim=2, hidden_dim=hidden_dim, num_hidden_layers=num_layers, key=jax.random.key(18))
    coord = jnp.array([0.3, -0.7])
    film = jnp.ones((num_layers, 2 * hidden_dim)) * 0.5
    assert not jnp.allclose(body(coord), body(coord, film=film))


@pytest.mark.unit
def test_body_rejects_non_positive_tau() -> None:
    # Given: a request to build a STAF body with tau=0.
    # When: constructing.
    # Then: it is rejected. tau<1 yields an empty activation sum (the constant 0),
    # a dead layer — caught at construction rather than silently trained.
    with pytest.raises(AssertionError):
        STAF(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=jax.random.key(19), tau=0)
