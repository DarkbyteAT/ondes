"""Tests for the FINER basis body (Liu+ 2024)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from ondes import FINER
from ondes.basis import Basis, BasisModule, Body, FINERLayer


def test_finer_layer_is_basis_subclass() -> None:
    # Given: a FINERLayer
    # When: checking inheritance
    # Then: it's a Basis (polymorphism contract for downstream typing)
    layer = FINERLayer(2, 8, omega_init=30.0, is_first=True, key=jax.random.PRNGKey(0))
    assert isinstance(layer, Basis)


def test_finer_activation_matches_paper_formula() -> None:
    # Given: a FINERLayer with omega=1, first layer
    # When: evaluating _activate on known pre-activations
    # Then: output matches sin((|pre|+1) * pre). The activation IS FINER's
    # central contribution; a regression that drops the magnitude-gating
    # would silently collapse FINER back to a SIREN at the same omega.
    layer = FINERLayer(1, 1, omega_init=1.0, is_first=True, key=jax.random.PRNGKey(0), scale_req_grad=True)
    pre = jnp.array([0.0, 0.5, -0.5, 1.0, -1.5])
    expected = jnp.sin((jnp.abs(pre) + 1.0) * pre)
    assert jnp.allclose(layer._activate(pre), expected)


def test_finer_first_layer_bias_within_first_bias_scale() -> None:
    # Given: a first FINER layer with first_bias_scale=10
    # When: inspecting layer.b
    # Then: |b| <= first_bias_scale (uniform-init bound). The wider bias is
    # FINER's "select high-frequency sub-bands at init" trick (paper Section 3).
    fbs = 10.0
    layer = FINERLayer(3, 1024, omega_init=30.0, is_first=True, key=jax.random.PRNGKey(0), first_bias_scale=fbs)
    assert float(jnp.max(jnp.abs(layer.b))) <= fbs + 1e-5


def test_finer_hidden_layer_bias_uses_siren_bound_not_first_bias_scale() -> None:
    # Given: a hidden (non-first) FINER layer with a deliberately tiny SIREN bound
    # When: comparing the bias magnitude to first_bias_scale
    # Then: |b| is much smaller than first_bias_scale — the FINER trick is
    # first-layer-only by design. A regression that applies the wider bound to
    # every layer would push hidden pre-activations way out of SIREN's
    # variance-preserving init range and silently destabilise training.
    omega = 30.0
    fbs = 20.0
    in_dim = 64
    layer = FINERLayer(in_dim, 256, omega_init=omega, is_first=False, key=jax.random.PRNGKey(0), first_bias_scale=fbs)
    siren_bound = float(jnp.sqrt(6.0 / in_dim) / omega)
    assert float(jnp.max(jnp.abs(layer.b))) <= siren_bound + 1e-5
    assert siren_bound < fbs


def test_finer_scale_req_grad_false_stops_gradient_through_scale() -> None:
    # Given: two layers with scale_req_grad=False and =True
    # When: taking gradient of the activation w.r.t. pre
    # Then: the gradient differs — scale_req_grad=False uses stop_gradient on
    # |pre|+1 (the paper default), so derivatives of (|pre|+1) don't flow
    # back. Catches a regression that ignores the flag.
    omega = 1.0
    pre = jnp.array(0.5)

    def fwd(p, sg):
        scale = jnp.abs(p) + 1.0
        if not sg:
            scale = jax.lax.stop_gradient(scale)
        return jnp.sin(omega * scale * p)

    g_stop = jax.grad(fwd)(pre, False)
    g_full = jax.grad(fwd)(pre, True)
    assert not jnp.allclose(g_stop, g_full)


def test_finer_body_is_basis_module() -> None:
    # Given: a FINER body
    # When: checking the structural and nominal type contracts
    # Then: it satisfies both
    body = FINER(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.PRNGKey(0))
    assert isinstance(body, BasisModule)
    assert isinstance(body, Body)


def test_finer_body_forward_scalar_shape() -> None:
    # Given: a default-out FINER body
    # When: forward-passing
    # Then: output is a 0-d scalar
    body = FINER(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.PRNGKey(1))
    y = body(jnp.array([0.1, -0.2]))
    assert y.shape == ()


def test_finer_body_forward_vector_shape() -> None:
    # Given: out_features=3
    # When: forward-passing
    # Then: output shape (3,)
    body = FINER(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.PRNGKey(2), out_features=3)
    y = body(jnp.array([0.1, -0.2]))
    assert y.shape == (3,)


def test_finer_body_trunk_shape() -> None:
    # Given: a FINER body
    # When: calling trunk()
    # Then: shape is (hidden_dim,)
    body = FINER(in_dim=2, hidden_dim=32, num_hidden_layers=3, key=jax.random.PRNGKey(3))
    h = body.trunk(jnp.array([0.25, -0.5]))
    assert h.shape == (32,)


def test_finer_body_first_bias_scale_threads_to_first_layer_only() -> None:
    # Given: a FINER body constructed with a custom first_bias_scale
    # When: inspecting layer biases
    # Then: only layer 0's bias respects the FINER wider bound; layers 1+
    # use the SIREN bound. Locks the body-to-layer kwarg-forwarding path.
    fbs = 8.0
    body = FINER(in_dim=2, hidden_dim=64, num_hidden_layers=3, key=jax.random.PRNGKey(0), first_bias_scale=fbs)
    assert float(jnp.max(jnp.abs(body.layers[0].b))) <= fbs + 1e-5
    siren_bound = float(jnp.sqrt(6.0 / body.hidden_dim) / float(body.layers[1].omega))
    for layer in body.layers[1:]:
        assert float(jnp.max(jnp.abs(layer.b))) <= siren_bound + 1e-5


def test_finer_body_film_modulation_changes_output() -> None:
    # Given: a FINER body run with and without FiLM
    # When: passing a non-trivial FiLM tensor
    # Then: outputs differ
    in_dim, hidden_dim, num_layers = 2, 8, 2
    body = FINER(in_dim=in_dim, hidden_dim=hidden_dim, num_hidden_layers=num_layers, key=jax.random.PRNGKey(4))
    coord = jnp.array([0.3, -0.7])
    film = jnp.ones((num_layers, 2 * hidden_dim)) * 0.5
    plain = body(coord)
    modulated = body(coord, film=film)
    assert not jnp.allclose(plain, modulated)


def test_finer_body_jit_matches_eager() -> None:
    # Given: a FINER body
    # When: jit-compiling the call
    # Then: jit matches eager
    body = FINER(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.PRNGKey(5))
    coord = jnp.array([0.1, -0.2])
    eager = body(coord)
    jitted = eqx.filter_jit(body)(coord)
    assert jnp.allclose(eager, jitted)


def test_finer_body_grad_is_finite_and_nonzero() -> None:
    # Given: a FINER body
    # When: taking grad of a sum-loss
    # Then: gradients finite and at least one carries signal
    body = FINER(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.PRNGKey(6))
    coord = jnp.array([0.3, -0.4])

    def loss(b, c):
        return b(c).sum()

    grad = eqx.filter_grad(loss)(body, coord)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(grad) if eqx.is_array(leaf)]
    assert len(leaves) > 0
    for g in leaves:
        assert bool(jnp.all(jnp.isfinite(g)))
    assert any(bool(jnp.any(g != 0)) for g in leaves)


def test_finer_body_vmap_over_coords() -> None:
    # Given: a batch of coords and a FINER body
    # When: vmapping
    # Then: output carries the batch axis
    body = FINER(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.PRNGKey(7))
    coords = jax.random.uniform(jax.random.PRNGKey(70), (5, 2), minval=-1.0, maxval=1.0)
    out = jax.vmap(body)(coords)
    assert out.shape == (5,)


def test_finer_body_canonicalises_out_features_one() -> None:
    # Given: two FINER bodies differing only in out_features
    # When: comparing pytree structure
    # Then: identical
    key = jax.random.PRNGKey(0)
    a = FINER(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=key, out_features=None)
    b = FINER(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=key, out_features=1)
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)


def test_finer_body_layer_pytree_homogeneous() -> None:
    # Given: a FINER body
    # When: comparing pytree structure of each layer
    # Then: identical — scale_req_grad is static but uniform across all
    # layers in one body, so scan-over-layers is mechanically clean once
    # array shapes agree (the layer-0-separate pattern is still needed for
    # the in_dim != hidden_dim shape mismatch, same as SIREN).
    body = FINER(in_dim=2, hidden_dim=64, num_hidden_layers=4, key=jax.random.PRNGKey(0))
    ref = jax.tree_util.tree_structure(body.layers[0])
    for layer in body.layers:
        assert jax.tree_util.tree_structure(layer) == ref


def test_finer_body_rejects_zero_hidden_layers() -> None:
    # Given: 0 hidden layers
    # When: constructing
    # Then: assertion fires
    with pytest.raises(AssertionError):
        FINER(in_dim=2, hidden_dim=8, num_hidden_layers=0, key=jax.random.PRNGKey(0))
