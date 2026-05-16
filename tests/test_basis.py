"""Tests for ondes.basis."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from ondes.basis import BASIS_KINDS, BasisBody, BasisLayer, siren_init


def test_siren_init_first_layer_uses_one_over_in_dim_bound():
    # Given: a first-layer init with in_dim=5
    # When: drawing many weight samples
    # Then: |W|, |b| are bounded by 1/in_dim, never by sqrt(6/in_dim)/omega
    in_dim, out_dim, omega = 5, 64, 30.0
    W, b = siren_init(in_dim, out_dim, omega, is_first=True, key=jax.random.PRNGKey(0))
    expected_bound = 1.0 / in_dim
    assert jnp.max(jnp.abs(W)) <= expected_bound + 1e-6
    assert jnp.max(jnp.abs(b)) <= expected_bound + 1e-6
    siren_bound = float(jnp.sqrt(6.0 / in_dim) / omega)
    assert siren_bound < expected_bound  # confirm the two formulas really differ


def test_siren_init_hidden_layer_uses_sqrt6_over_omega_bound():
    # Given: a hidden-layer init with omega large enough to make the SIREN bound tight
    # When: drawing samples
    # Then: |W| is within sqrt(6/in_dim)/omega and strictly exceeds it never
    in_dim, out_dim, omega = 64, 64, 30.0
    W, b = siren_init(in_dim, out_dim, omega, is_first=False, key=jax.random.PRNGKey(1))
    expected_bound = float(jnp.sqrt(6.0 / in_dim) / omega)
    assert jnp.max(jnp.abs(W)) <= expected_bound + 1e-6
    assert jnp.max(jnp.abs(b)) <= expected_bound + 1e-6


def test_basis_layer_each_kind_produces_finite_output():
    # Given: a BasisLayer of each kind on a fixed deterministic input
    # When: forward-passing
    # Then: output is finite and has the right shape
    in_dim, out_dim = 4, 16
    x = jnp.linspace(-1.0, 1.0, in_dim)
    for kind in BASIS_KINDS:
        layer = BasisLayer(in_dim, out_dim, omega_init=6.0, kind=kind, is_first=True, key=jax.random.PRNGKey(7))
        y = layer(x)
        assert y.shape == (out_dim,)
        assert bool(jnp.all(jnp.isfinite(y)))


def test_basis_body_call_is_jit_compilable():
    # Given: a BasisBody and a coordinate input
    # When: jitting the call
    # Then: the jit-compiled function runs and matches the eager call
    body = BasisBody(in_dim=2, hidden_dim=32, num_hidden_layers=3, kind="siren", key=jax.random.PRNGKey(2))
    coord = jnp.array([0.25, -0.5])
    eager = body(coord)
    jitted = eqx.filter_jit(body)(coord)
    assert jnp.allclose(eager, jitted)
    assert eager.shape == ()


@pytest.mark.parametrize("out_features,readout_out_dim", [(None, 1), (5, 5)])
def test_basis_body_parameter_count_matches_analytic_formula(out_features, readout_out_dim):
    # Given: a BasisBody with known dims and a known out_features
    # When: counting learnable float-array leaves
    # Then: total equals sum of layer (W, b, omega, s) + readout (W, b) with the
    # readout sized by readout_out_dim — catches regressions that reshape the
    # readout silently.
    in_dim, hidden_dim, num_layers = 3, 16, 4
    body = BasisBody(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_layers,
        kind="wire",
        key=jax.random.PRNGKey(3),
        out_features=out_features,
    )

    expected = 0
    for i in range(num_layers):
        in_d = in_dim if i == 0 else hidden_dim
        expected += hidden_dim * in_d  # W
        expected += hidden_dim  # b
        expected += 1  # omega
        expected += 1  # s
    expected += readout_out_dim * hidden_dim  # readout_W
    expected += readout_out_dim  # readout_b

    leaves = jax.tree_util.tree_leaves(body)
    total = sum(int(leaf.size) for leaf in leaves if hasattr(leaf, "size"))
    assert total == expected


def test_basis_body_film_modulation_changes_output():
    # Given: a body run with and without FiLM
    # When: passing a non-trivial FiLM tensor
    # Then: outputs differ (modulation is actually wired)
    in_dim, hidden_dim, num_layers = 2, 8, 2
    body = BasisBody(
        in_dim=in_dim, hidden_dim=hidden_dim, num_hidden_layers=num_layers, kind="siren", key=jax.random.PRNGKey(4)
    )
    coord = jnp.array([0.3, -0.7])
    film = jnp.ones((num_layers, 2 * hidden_dim)) * 0.5
    plain = body(coord)
    modulated = body(coord, film=film)
    assert not jnp.allclose(plain, modulated)


def test_basis_body_out_features_none_returns_scalar():
    # Given: a BasisBody constructed with default out_features
    # When: forward-passing
    # Then: output is a 0-d scalar (preserves prior behaviour)
    body = BasisBody(in_dim=2, hidden_dim=16, num_hidden_layers=2, kind="siren", key=jax.random.PRNGKey(5))
    y = body(jnp.array([0.1, -0.2]))
    assert y.shape == ()


def test_basis_body_out_features_int_returns_vector():
    # Given: a BasisBody with out_features=4
    # When: forward-passing
    # Then: output is shape (4,)
    body = BasisBody(
        in_dim=2, hidden_dim=16, num_hidden_layers=2, kind="siren", key=jax.random.PRNGKey(6), out_features=4
    )
    y = body(jnp.array([0.1, -0.2]))
    assert y.shape == (4,)


def test_basis_body_trunk_returns_hidden_features():
    # Given: BasisBody instances with different out_features
    # When: calling trunk()
    # Then: shape is (hidden_dim,) and independent of out_features
    hidden_dim = 32
    coord = jnp.array([0.25, -0.5])
    scalar_body = BasisBody(
        in_dim=2, hidden_dim=hidden_dim, num_hidden_layers=3, kind="siren", key=jax.random.PRNGKey(7)
    )
    vector_body = BasisBody(
        in_dim=2, hidden_dim=hidden_dim, num_hidden_layers=3, kind="siren", key=jax.random.PRNGKey(7), out_features=5
    )
    h_scalar = scalar_body.trunk(coord)
    h_vector = vector_body.trunk(coord)
    assert h_scalar.shape == (hidden_dim,)
    assert h_vector.shape == (hidden_dim,)
    # Trunk activations match: same key, same body layers; out_features only
    # changes the readout shape.
    assert jnp.allclose(h_scalar, h_vector)


@pytest.mark.parametrize("kind", BASIS_KINDS)
def test_basis_body_trunk_is_jit_compilable(kind):
    # Given: a BasisBody of each basis kind
    # When: jitting trunk()
    # Then: jit-compiled result matches eager and has the expected shape
    body = BasisBody(in_dim=2, hidden_dim=16, num_hidden_layers=2, kind=kind, key=jax.random.PRNGKey(8))
    coord = jnp.array([0.3, -0.4])
    eager = body.trunk(coord)
    jitted = eqx.filter_jit(lambda b, c: b.trunk(c))(body, coord)
    assert eager.shape == (16,)
    assert jnp.allclose(eager, jitted)


def test_basis_body_trunk_with_film_modulation_changes_output():
    # Given: trunk called with and without FiLM
    # When: a non-trivial FiLM tensor is supplied
    # Then: trunk outputs differ
    in_dim, hidden_dim, num_layers = 2, 8, 2
    body = BasisBody(
        in_dim=in_dim, hidden_dim=hidden_dim, num_hidden_layers=num_layers, kind="siren", key=jax.random.PRNGKey(9)
    )
    coord = jnp.array([0.3, -0.7])
    film = jnp.ones((num_layers, 2 * hidden_dim)) * 0.5
    plain = body.trunk(coord)
    modulated = body.trunk(coord, film=film)
    assert not jnp.allclose(plain, modulated)


def test_basis_body_out_features_one_returns_scalar():
    # Given: out_features=1 (the canonicalised-to-None boundary)
    # When: forward-passing
    # Then: output is a 0-d scalar — the user-facing contract for 1 is "scalar"
    body = BasisBody(
        in_dim=2, hidden_dim=16, num_hidden_layers=2, kind="siren", key=jax.random.PRNGKey(10), out_features=1
    )
    y = body(jnp.array([0.1, -0.2]))
    assert y.shape == ()


def test_basis_body_rejects_zero_hidden_layers():
    # Given: a request to build a body with no hidden layers
    # When: constructing
    # Then: the constructor rejects it — a 0-layer body would silently break
    # the readout's shape contract (input flows straight to a (out_dim, hidden)
    # matmul that expects post-hidden activations).
    with pytest.raises(AssertionError):
        BasisBody(in_dim=2, hidden_dim=8, num_hidden_layers=0, kind="siren", key=jax.random.PRNGKey(20))


def test_basis_body_out_features_one_canonicalises_to_none():
    # Given: two BasisBody instances differing only in out_features=None vs out_features=1
    # When: comparing their pytree structures
    # Then: structures are identical (canonicalisation is load-bearing for
    # serialisation, jit caching, and tree-equality checks)
    key = jax.random.PRNGKey(0)
    a = BasisBody(in_dim=2, hidden_dim=8, num_hidden_layers=2, out_features=None, key=key)
    b = BasisBody(in_dim=2, hidden_dim=8, num_hidden_layers=2, out_features=1, key=key)
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    assert a.out_features is None
    assert b.out_features is None


def test_basis_body_out_features_two_returns_vector():
    # Given: out_features=2 (the smallest vector-returning width)
    # When: forward-passing
    # Then: output is shape (2,) — catches off-by-one in the squeeze branch
    body = BasisBody(
        in_dim=2, hidden_dim=16, num_hidden_layers=2, kind="siren", key=jax.random.PRNGKey(11), out_features=2
    )
    y = body(jnp.array([0.1, -0.2]))
    assert y.shape == (2,)


def test_basis_body_grad_through_scalar_path():
    # Given: a default (scalar) BasisBody and a trivial loss
    # When: taking jax.grad over the body's parameters
    # Then: gradient is a pytree of finite, non-zero arrays matching the body's leaves
    body = BasisBody(in_dim=2, hidden_dim=16, num_hidden_layers=2, kind="siren", key=jax.random.PRNGKey(12))
    coord = jnp.array([0.3, -0.4])

    def loss(b, c):
        return b(c).sum()

    grad = eqx.filter_grad(loss)(body, coord)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(grad) if eqx.is_array(leaf)]
    assert len(leaves) > 0
    for g in leaves:
        assert bool(jnp.all(jnp.isfinite(g)))
    # At least one leaf must carry signal — full-zero gradient would mean broken plumbing.
    assert any(bool(jnp.any(g != 0)) for g in leaves)


def test_basis_body_grad_through_vector_path():
    # Given: an out_features=4 BasisBody and a trivial loss
    # When: taking jax.grad over the body's parameters
    # Then: gradient is a pytree of finite, non-zero arrays
    body = BasisBody(
        in_dim=2, hidden_dim=16, num_hidden_layers=2, kind="siren", key=jax.random.PRNGKey(13), out_features=4
    )
    coord = jnp.array([0.3, -0.4])

    def loss(b, c):
        return b(c).sum()

    grad = eqx.filter_grad(loss)(body, coord)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(grad) if eqx.is_array(leaf)]
    assert len(leaves) > 0
    for g in leaves:
        assert bool(jnp.all(jnp.isfinite(g)))
    assert any(bool(jnp.any(g != 0)) for g in leaves)


def test_basis_body_vmap_over_coords_scalar():
    # Given: a default (scalar) BasisBody and a batch of coordinates
    # When: vmapping the body over the leading axis
    # Then: output has shape (B,)
    body = BasisBody(in_dim=2, hidden_dim=16, num_hidden_layers=2, kind="siren", key=jax.random.PRNGKey(14))
    coords = jax.random.uniform(jax.random.PRNGKey(140), (7, 2), minval=-1.0, maxval=1.0)
    out = jax.vmap(body)(coords)
    assert out.shape == (7,)


def test_basis_body_vmap_over_coords_vector():
    # Given: an out_features=3 BasisBody and a batch of coordinates
    # When: vmapping the body over the leading axis
    # Then: output has shape (B, 3)
    body = BasisBody(
        in_dim=2, hidden_dim=16, num_hidden_layers=2, kind="siren", key=jax.random.PRNGKey(15), out_features=3
    )
    coords = jax.random.uniform(jax.random.PRNGKey(150), (7, 2), minval=-1.0, maxval=1.0)
    out = jax.vmap(body)(coords)
    assert out.shape == (7, 3)


def test_basis_body_scalar_path_squeeze_only_feature_dim():
    # Given: a default (scalar) BasisBody and a batch of size 1 — the case
    # where unrestricted .squeeze() would silently collapse the batch dim.
    # When: vmapping over a (1, in_dim) coord
    # Then: output has shape (1,), not scalar (). Catches a regression where
    # __call__ uses .squeeze() instead of .squeeze(-1).
    body = BasisBody(in_dim=2, hidden_dim=8, num_hidden_layers=2, kind="siren", key=jax.random.PRNGKey(16))
    coord_batched = jnp.zeros((1, 2))
    out = jax.vmap(body)(coord_batched)
    assert out.shape == (1,), f"expected (1,), got {out.shape}"
