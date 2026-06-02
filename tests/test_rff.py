"""Tests for the RFF basis body (Tancik+ 2020)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from ondes import RFF
from ondes.basis import Basis, BasisModule, Body, RFFLayer


def test_rff_layer_is_basis_subclass():
    # Given: an RFFLayer instance
    # When: checking the inheritance contract
    # Then: it's a Basis (the polymorphism contract — downstream code can
    # express "any basis layer" as Basis in a single type signature).
    layer = RFFLayer(8, 16, key=jax.random.key(0))
    assert isinstance(layer, Basis)


def test_rff_layer_forward_is_relu():
    # Given: an RFFLayer and a pre-activation with both signs
    # When: calling _activate directly
    # Then: it matches plain ReLU exactly (RFF uses no scaling).
    layer = RFFLayer(2, 4, key=jax.random.key(0))
    pre = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert jnp.allclose(layer._activate(pre), jax.nn.relu(pre))


def test_rff_layer_omega_is_unit_placeholder():
    # Given: an RFFLayer (omega is unused by the activation)
    # When: inspecting the omega leaf
    # Then: it's a unit scalar so the leaf still exists for pytree parity with
    # the SIREN family, but its value doesn't influence the forward pass.
    layer = RFFLayer(2, 4, key=jax.random.key(0))
    assert layer.omega.shape == ()
    assert float(layer.omega) == 1.0


def test_rff_body_is_basis_module():
    # Given: an RFF body
    # When: checking the structural and nominal type contracts
    # Then: it conforms to both BasisModule (Protocol) and Body (concrete base).
    body = RFF(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(0))
    assert isinstance(body, BasisModule)
    assert isinstance(body, Body)


def test_rff_body_b_matrix_shape_and_scale():
    # Given: an RFF body with known num_freqs / sigma
    # When: inspecting the encoding's B matrix
    # Then: B has the right shape and a sample standard deviation close to sigma
    # (the bandwidth knob has actually been applied).
    in_dim, num_freqs, sigma = 3, 4096, 7.0
    body = RFF(
        in_dim=in_dim,
        hidden_dim=8,
        num_hidden_layers=1,
        key=jax.random.key(0),
        num_freqs=num_freqs,
        sigma=sigma,
    )
    assert body.B.shape == (num_freqs, in_dim)
    # Sample-std should be close to sigma; tolerance ~5% is comfortably above
    # the asymptotic standard error for N=num_freqs*in_dim Gaussian samples.
    assert jnp.allclose(jnp.std(body.B), sigma, rtol=0.05)


def test_rff_body_forward_scalar_shape():
    # Given: a default-out (scalar) RFF body and a coordinate
    # When: forward-passing
    # Then: output is a 0-d scalar
    body = RFF(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(1))
    y = body(jnp.array([0.1, -0.2]))
    assert y.shape == ()


def test_rff_body_forward_vector_shape():
    # Given: an out_features=4 RFF body
    # When: forward-passing
    # Then: output is shape (4,)
    body = RFF(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(2), out_features=4)
    y = body(jnp.array([0.1, -0.2]))
    assert y.shape == (4,)


def test_rff_body_trunk_returns_hidden_features():
    # Given: an RFF body
    # When: calling trunk()
    # Then: shape is (hidden_dim,) — the encoding adds an internal layer 0
    # but the trunk output is the final hidden activation.
    body = RFF(in_dim=2, hidden_dim=32, num_hidden_layers=3, key=jax.random.key(3))
    h = body.trunk(jnp.array([0.25, -0.5]))
    assert h.shape == (32,)


def test_rff_body_film_modulation_changes_output():
    # Given: an RFF body run with and without FiLM
    # When: passing a non-trivial FiLM tensor
    # Then: outputs differ (modulation is wired through the MLP layers, not the
    # encoding — but at least one MLP layer sees it, so the output must change).
    in_dim, hidden_dim, num_layers = 2, 8, 2
    body = RFF(in_dim=in_dim, hidden_dim=hidden_dim, num_hidden_layers=num_layers, key=jax.random.key(4))
    coord = jnp.array([0.3, -0.7])
    film = jnp.ones((num_layers, 2 * hidden_dim)) * 0.5
    plain = body(coord)
    modulated = body(coord, film=film)
    assert not jnp.allclose(plain, modulated)


def test_rff_body_canonicalises_out_features_one_to_none():
    # Given: two RFF bodies with out_features=None and out_features=1
    # When: comparing their pytree structures
    # Then: identical (the canonicalisation rule applies uniformly across bases)
    key = jax.random.key(0)
    a = RFF(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=key, out_features=None)
    b = RFF(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=key, out_features=1)
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    assert a.out_features is None
    assert b.out_features is None


def test_rff_body_jit_matches_eager():
    # Given: an RFF body and a coordinate
    # When: jit-compiling the call
    # Then: jitted output matches eager
    body = RFF(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(5))
    coord = jnp.array([0.1, -0.2])
    eager = body(coord)
    jitted = eqx.filter_jit(body)(coord)
    assert jnp.allclose(eager, jitted)


def test_rff_body_grad_is_finite_and_nonzero():
    # Given: an RFF body
    # When: taking grad of a sum-loss through the body
    # Then: gradients exist, are finite, and at least one carries signal —
    # full-zero gradient would mean broken plumbing through the encoding or MLP.
    body = RFF(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(6))
    coord = jnp.array([0.3, -0.4])

    def loss(b, c):
        return b(c).sum()

    grad = eqx.filter_grad(loss)(body, coord)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(grad) if eqx.is_array(leaf)]
    assert len(leaves) > 0
    for g in leaves:
        assert bool(jnp.all(jnp.isfinite(g)))
    assert any(bool(jnp.any(g != 0)) for g in leaves)


def test_rff_body_vmap_over_coords():
    # Given: a batch of coordinates and an RFF body
    # When: vmapping the call
    # Then: output has the batch shape
    body = RFF(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(7))
    coords = jax.random.uniform(jax.random.key(70), (7, 2), minval=-1.0, maxval=1.0)
    out = jax.vmap(body)(coords)
    assert out.shape == (7,)


@pytest.mark.parametrize("out_features", [None, 1, 4])
def test_rff_body_rejects_zero_hidden_layers(out_features):
    # Given: a request to build with num_hidden_layers=0
    # When: constructing
    # Then: constructor rejects — the readout shape contract requires at least
    # one hidden layer to feed it.
    with pytest.raises(AssertionError):
        RFF(in_dim=2, hidden_dim=8, num_hidden_layers=0, key=jax.random.key(0), out_features=out_features)


def test_rff_fix_encoding_mask_isolates_B_matrix():
    # Given: an RFF body and its fix-encoding mask
    # When: partitioning the body
    # Then: the "fixed" half contains the encoding `B` and the learnable half
    # has a non-array placeholder there. The mask is load-bearing: users pass
    # it to optax.masked / eqx.partition to keep `B` frozen, matching the
    # Gaussian-RFF paper's "draw once and freeze" convention.
    body = RFF(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=jax.random.key(7), num_freqs=16)
    mask = body.fix_encoding_mask()
    learnable, fixed = eqx.partition(body, mask)

    assert eqx.is_array(fixed.B)
    assert not eqx.is_array(learnable.B)


def test_rff_body_layer_pytree_homogeneous():
    # Given: an RFF body where in_dim == hidden_dim
    # When: comparing the pytree structure of every MLP layer
    # Then: identical (no static-field discriminator on the layer, scan-friendly).
    # Realistic INRs use in_dim != hidden_dim, but the encoding step decouples
    # the coord dim from the MLP input dim, so post-encoding layers can scan
    # whenever encoded_dim == hidden_dim. Pytree homogeneity is independent
    # of array shape — this test asserts the static-fields contract.
    body = RFF(in_dim=2, hidden_dim=64, num_hidden_layers=4, key=jax.random.key(0), num_freqs=32)
    ref = jax.tree_util.tree_structure(body.layers[0])
    for layer in body.layers:
        assert jax.tree_util.tree_structure(layer) == ref
