"""Tests for the BACON basis body (Lindell+ 2022)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from ondes import BACON
from ondes.basis import BasisModule, Body
from ondes.basis.bacon import BACONFilter


def test_bacon_body_conforms_to_basis_module_and_body():
    # Given: a BACON body
    # When: checking the structural and nominal type contracts
    # Then: it satisfies both, so downstream code that types against either
    # picks BACON up without a special case.
    body = BACON(in_dim=2, hidden_dim=16, num_hidden_layers=3, key=jax.random.key(0))
    assert isinstance(body, BasisModule)
    assert isinstance(body, Body)


def test_bacon_body_forward_scalar_shape():
    # Given: a default-out BACON body
    # When: forward-passing
    # Then: output is a 0-d scalar
    body = BACON(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(1))
    y = body(jnp.array([0.1, -0.2]))
    assert y.shape == ()


def test_bacon_body_forward_vector_shape():
    # Given: out_features=3
    # When: forward-passing
    # Then: output shape is (3,)
    body = BACON(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(2), out_features=3)
    y = body(jnp.array([0.1, -0.2]))
    assert y.shape == (3,)


def test_bacon_filter_frequencies_are_integer_multiples_of_quantization():
    # Given: a BACONFilter with known dq
    # When: inspecting filter.W
    # Then: every entry is an integer multiple of dq (the band-limiting proof
    # requires discrete frequencies). A regression that samples from a
    # continuous distribution would silently break BACON's analytic bandwidth.
    dq = 2.0 * float(jnp.pi)
    f = BACONFilter(in_dim=2, hidden_dim=256, bandwidth=10 * dq, quantization_interval=dq, key=jax.random.key(0))
    residuals = f.W / dq - jnp.round(f.W / dq)
    assert float(jnp.max(jnp.abs(residuals))) < 1e-5


def test_bacon_filter_frequencies_within_per_layer_bandwidth():
    # Given: a BACONFilter with bandwidth B
    # When: inspecting filter.W
    # Then: every entry is in [-B, B] (paper-mandated per-layer cap; failure
    # here would silently break the network-level bandwidth bound).
    dq = 2.0 * float(jnp.pi)
    bandwidth = 5 * dq
    f = BACONFilter(in_dim=3, hidden_dim=512, bandwidth=bandwidth, quantization_interval=dq, key=jax.random.key(1))
    assert float(jnp.max(jnp.abs(f.W))) <= bandwidth + 1e-5


def test_bacon_filter_frequencies_include_both_signs_and_zero():
    # Given: a wide filter
    # When: counting unique signs
    # Then: the distribution spans negative, zero, and positive integer
    # multiples — the inclusive {-k,...,k} set the reference samples from.
    dq = 1.0
    f = BACONFilter(in_dim=2, hidden_dim=4096, bandwidth=4 * dq, quantization_interval=dq, key=jax.random.key(2))
    assert float(jnp.min(f.W)) <= -dq
    assert float(jnp.max(f.W)) >= dq
    assert bool(jnp.any(f.W == 0.0))


def test_bacon_per_layer_bandwidth_matches_paper_formula():
    # Given: a BACON body
    # When: inspecting the per-layer bandwidth schedule
    # Then: each bandwidth equals round(pi * max_freq / (N+1) / dq) * dq
    # (the formula the reference uses). Catches a regression that loses the
    # quantisation step in the per-layer schedule.
    max_freq = 64.0
    num_hidden_layers = 3
    dq = 2.0 * float(jnp.pi)
    body = BACON(
        in_dim=2,
        hidden_dim=8,
        num_hidden_layers=num_hidden_layers,
        key=jax.random.key(0),
        max_freq=max_freq,
        quantization_interval=dq,
    )
    expected = round(float(jnp.pi) * max_freq / (num_hidden_layers + 1.0) / dq) * dq
    for bw in body.bandwidths:
        assert float(bw) == pytest.approx(expected)


def test_bacon_output_bandwidth_property_equals_sum_of_per_layer():
    # Given: a BACON body
    # When: reading output_bandwidth
    # Then: equals sum of per-layer bandwidths — this IS the network-level cap
    # the paper proves analytically.
    body = BACON(in_dim=2, hidden_dim=8, num_hidden_layers=4, key=jax.random.key(0), max_freq=128.0)
    assert body.output_bandwidth == pytest.approx(float(jnp.sum(body.bandwidths)))


def test_bacon_has_n_plus_one_filters():
    # Given: a BACON body with num_hidden_layers=4
    # When: counting filters
    # Then: there are 5 (mirrors MFN's invariant — z_0 = g_0(x); z_{i+1} =
    # g_{i+1}(x) * (W_i z_i + b_i)).
    body = BACON(in_dim=2, hidden_dim=16, num_hidden_layers=4, key=jax.random.key(0))
    assert len(body.filters) == 5
    for f in body.filters:
        assert isinstance(f, BACONFilter)


def test_bacon_body_film_modulation_changes_output():
    # Given: a BACON body run with and without FiLM
    # When: passing a non-trivial FiLM tensor
    # Then: outputs differ
    in_dim, hidden_dim, num_layers = 2, 8, 3
    body = BACON(in_dim=in_dim, hidden_dim=hidden_dim, num_hidden_layers=num_layers, key=jax.random.key(4))
    coord = jnp.array([0.3, -0.7])
    film = jnp.ones((num_layers, 2 * hidden_dim)) * 0.5
    plain = body(coord)
    modulated = body(coord, film=film)
    assert not jnp.allclose(plain, modulated)


def test_bacon_body_jit_matches_eager():
    # Given: a BACON body and a coord
    # When: jit-compiling
    # Then: matches eager
    body = BACON(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(5))
    coord = jnp.array([0.1, -0.2])
    eager = body(coord)
    jitted = eqx.filter_jit(body)(coord)
    assert jnp.allclose(eager, jitted)


def test_bacon_body_grad_is_finite_and_nonzero():
    # Given: a BACON body
    # When: taking grad of a sum-loss
    # Then: gradients are finite and at least one carries signal
    body = BACON(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(6))
    coord = jnp.array([0.3, -0.4])

    def loss(b, c):
        return b(c).sum()

    grad = eqx.filter_grad(loss)(body, coord)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(grad) if eqx.is_array(leaf)]
    assert len(leaves) > 0
    for g in leaves:
        assert bool(jnp.all(jnp.isfinite(g)))
    assert any(bool(jnp.any(g != 0)) for g in leaves)


def test_bacon_fix_filters_mask_isolates_filter_weights():
    # Given: a BACON body and its fix-filters mask
    # When: partitioning the body
    # Then: the "fixed" half contains the filter.W arrays *and* the body-level
    # `bandwidths` cap array. The mask is load-bearing: users pass it to
    # optax.masked to keep filter frequencies non-trainable (the band-limit
    # proof depends on it) and to avoid Adam allocating momentum state for the
    # constant `bandwidths` diagnostic.
    body = BACON(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=jax.random.key(7))
    mask = body.fix_filters_mask()
    learnable, fixed = eqx.partition(body, mask)

    # Every filter.W should appear in the fixed half (mask was False there).
    for i, f in enumerate(fixed.filters):
        assert eqx.is_array(f.W)
        # And the learnable half should contain a None / non-array placeholder there.
        assert not eqx.is_array(learnable.filters[i].W)

    # `bandwidths` is a constant diagnostic, not a trainable param.
    assert eqx.is_array(fixed.bandwidths)
    assert not eqx.is_array(learnable.bandwidths)


def test_bacon_body_vmap_over_coords():
    # Given: a batch of coords
    # When: vmapping the body
    # Then: output carries the batch axis
    body = BACON(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(8))
    coords = jax.random.uniform(jax.random.key(80), (4, 2), minval=-1.0, maxval=1.0)
    out = jax.vmap(body)(coords)
    assert out.shape == (4,)


def test_bacon_body_canonicalises_out_features_one():
    # Given: two BACON bodies differing only in out_features=None vs 1
    # When: comparing pytree structures
    # Then: identical
    key = jax.random.key(0)
    a = BACON(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=key, out_features=None)
    b = BACON(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=key, out_features=1)
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)


def test_bacon_body_rejects_zero_hidden_layers():
    # Given: 0 recurrence steps
    # When: constructing
    # Then: assertion fires (shared by all bodies via _validate_body_args)
    with pytest.raises(AssertionError):
        BACON(in_dim=2, hidden_dim=8, num_hidden_layers=0, key=jax.random.key(0))
