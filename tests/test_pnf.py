"""Tests for the PNF basis body (Yang+ 2022)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from ondes import PNF
from ondes.basis import BasisModule, Body
from ondes.basis.mfn import FourierFilter


def test_pnf_body_conforms_to_basis_module_and_body():
    # Given: a PNF body
    # When: checking the structural and nominal type contracts
    # Then: it satisfies both, so downstream code that types against either
    # picks PNF up without a special case.
    body = PNF(in_dim=2, hidden_dim=16, num_hidden_layers=3, key=jax.random.key(0))
    assert isinstance(body, BasisModule)
    assert isinstance(body, Body)


def test_pnf_body_forward_scalar_shape():
    # Given: a default-out PNF body
    # When: forward-passing
    # Then: output is a 0-d scalar
    body = PNF(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(1))
    y = body(jnp.array([0.1, -0.2]))
    assert y.shape == ()


def test_pnf_body_forward_vector_shape():
    # Given: out_features=4
    # When: forward-passing
    # Then: output shape (4,)
    body = PNF(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(2), out_features=4)
    y = body(jnp.array([0.1, -0.2]))
    assert y.shape == (4,)


def test_pnf_body_trunk_shape():
    # Given: a PNF body
    # When: calling trunk()
    # Then: shape is (hidden_dim,)
    body = PNF(in_dim=2, hidden_dim=32, num_hidden_layers=3, key=jax.random.key(3))
    h = body.trunk(jnp.array([0.25, -0.5]))
    assert h.shape == (32,)


def test_pnf_has_mix_matrix_per_recurrence_step():
    # Given: a PNF body with num_hidden_layers=4
    # When: inspecting mix_W
    # Then: shape is (4, hidden_dim, hidden_dim) — one mix matrix per step.
    # The mix layer IS PNF's central addition vs MFN; a regression that
    # drops it would silently collapse PNF back to a Fourier MFN.
    body = PNF(in_dim=2, hidden_dim=8, num_hidden_layers=4, key=jax.random.key(0))
    assert body.mix_W.shape == (4, 8, 8)


def test_pnf_has_n_plus_one_fourier_filters():
    # Given: a PNF body with num_hidden_layers=4
    # When: inspecting filters
    # Then: there are 5 FourierFilter instances (same invariant as MFN)
    body = PNF(in_dim=2, hidden_dim=16, num_hidden_layers=4, key=jax.random.key(0))
    assert len(body.filters) == 5
    for f in body.filters:
        assert isinstance(f, FourierFilter)


def test_pnf_mix_matrix_changes_output_vs_zero_mix():
    # Given: a PNF body and the same body with mix_W zeroed
    # When: comparing outputs
    # Then: they differ — the mix layer must actually contribute. With mix_W=0
    # PNF reduces to an MFN (z_{i+1} = g_{i+1}(x) * (W z + b)); the test
    # checks the comparison is non-trivial, so the mix layer is exercised.
    body = PNF(in_dim=2, hidden_dim=16, num_hidden_layers=3, key=jax.random.key(7))
    zeroed_mix = eqx.tree_at(lambda b: b.mix_W, body, jnp.zeros_like(body.mix_W))
    coord = jnp.array([0.3, -0.4])
    assert not jnp.allclose(body(coord), zeroed_mix(coord))


def test_pnf_body_film_modulation_changes_output():
    # Given: a PNF body run with and without FiLM
    # When: a non-trivial FiLM tensor is supplied
    # Then: outputs differ
    in_dim, hidden_dim, num_layers = 2, 8, 3
    body = PNF(in_dim=in_dim, hidden_dim=hidden_dim, num_hidden_layers=num_layers, key=jax.random.key(4))
    coord = jnp.array([0.3, -0.7])
    film = jnp.ones((num_layers, 2 * hidden_dim)) * 0.5
    plain = body(coord)
    modulated = body(coord, film=film)
    assert not jnp.allclose(plain, modulated)


def test_pnf_body_jit_matches_eager():
    # Given: a PNF body and a coord
    # When: jit-compiling
    # Then: matches eager
    body = PNF(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(5))
    coord = jnp.array([0.1, -0.2])
    eager = body(coord)
    jitted = eqx.filter_jit(body)(coord)
    assert jnp.allclose(eager, jitted)


def test_pnf_body_grad_is_finite_and_nonzero():
    # Given: a PNF body
    # When: taking grad of a sum-loss
    # Then: gradients finite and at least one carries signal (including the
    # mix matrix path — a broken mix backward would silently freeze that
    # parameter sub-tree).
    body = PNF(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(6))
    coord = jnp.array([0.3, -0.4])

    def loss(b, c):
        return b(c).sum()

    grad = eqx.filter_grad(loss)(body, coord)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(grad) if eqx.is_array(leaf)]
    assert len(leaves) > 0
    for g in leaves:
        assert bool(jnp.all(jnp.isfinite(g)))
    assert any(bool(jnp.any(g != 0)) for g in leaves)


def test_pnf_mix_matrix_receives_gradient():
    # Given: a PNF body and a sum-loss
    # When: filtering grads to the mix matrix sub-tree
    # Then: at least one entry is non-zero. Specifically tests the mix path —
    # a regression that fails to thread mix_W through the recurrence would
    # leave its gradient at zero even though all other params receive signal.
    body = PNF(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=jax.random.key(11))
    coord = jnp.array([0.3, -0.4])

    def loss(b, c):
        return b(c).sum()

    grad = eqx.filter_grad(loss)(body, coord)
    assert bool(jnp.any(grad.mix_W != 0))


def test_pnf_body_vmap_over_coords():
    # Given: a batch of coords
    # When: vmapping
    # Then: output carries the batch axis
    body = PNF(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(7))
    coords = jax.random.uniform(jax.random.key(70), (5, 2), minval=-1.0, maxval=1.0)
    out = jax.vmap(body)(coords)
    assert out.shape == (5,)


def test_pnf_body_canonicalises_out_features_one():
    # Given: two PNF bodies differing only in out_features=None vs 1
    # When: comparing pytree structures
    # Then: identical
    key = jax.random.key(0)
    a = PNF(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=key, out_features=None)
    b = PNF(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=key, out_features=1)
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)


def test_pnf_body_rejects_zero_hidden_layers():
    # Given: 0 recurrence steps
    # When: constructing
    # Then: assertion fires
    with pytest.raises(AssertionError):
        PNF(in_dim=2, hidden_dim=8, num_hidden_layers=0, key=jax.random.key(0))
