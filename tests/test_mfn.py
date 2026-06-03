"""Tests for the MFN basis bodies (Fathony+ 2021)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from ondes import FourierMFN, GaborMFN
from ondes.basis import BasisModule, Body
from ondes.basis.mfn import FourierFilter, GaborFilter


MFN_CLASSES = (FourierMFN, GaborMFN)


@pytest.mark.parametrize("body_cls", MFN_CLASSES)
def test_mfn_body_conforms_to_basis_module_and_body(body_cls: type) -> None:
    # Given: an MFN body of each kind
    # When: checking the structural and nominal type contracts
    # Then: each is a BasisModule (Protocol) and a Body (concrete base) — so
    # downstream code that types against either contract picks up MFN bodies
    # without a special case.
    body = body_cls(in_dim=2, hidden_dim=16, num_hidden_layers=3, key=jax.random.key(0))
    assert isinstance(body, BasisModule)
    assert isinstance(body, Body)


@pytest.mark.parametrize("body_cls", MFN_CLASSES)
def test_mfn_body_forward_scalar_shape(body_cls: type) -> None:
    # Given: a default-out MFN body
    # When: forward-passing
    # Then: output is a 0-d scalar
    body = body_cls(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(1))
    y = body(jnp.array([0.1, -0.2]))
    assert y.shape == ()


@pytest.mark.parametrize("body_cls", MFN_CLASSES)
def test_mfn_body_forward_vector_shape(body_cls: type) -> None:
    # Given: an out_features=4 MFN body
    # When: forward-passing
    # Then: output is shape (4,)
    body = body_cls(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(2), out_features=4)
    y = body(jnp.array([0.1, -0.2]))
    assert y.shape == (4,)


@pytest.mark.parametrize("body_cls", MFN_CLASSES)
def test_mfn_body_trunk_shape(body_cls: type) -> None:
    # Given: an MFN body
    # When: calling trunk()
    # Then: shape is (hidden_dim,) (the recurrence carries hidden-dim state)
    body = body_cls(in_dim=2, hidden_dim=32, num_hidden_layers=3, key=jax.random.key(3))
    h = body.trunk(jnp.array([0.25, -0.5]))
    assert h.shape == (32,)


@pytest.mark.parametrize("body_cls", MFN_CLASSES)
def test_mfn_body_film_modulation_changes_output(body_cls: type) -> None:
    # Given: an MFN body run with and without FiLM
    # When: passing a non-trivial FiLM tensor
    # Then: outputs differ — FiLM gates the recurrence-linear output at every step.
    in_dim, hidden_dim, num_layers = 2, 8, 3
    body = body_cls(in_dim=in_dim, hidden_dim=hidden_dim, num_hidden_layers=num_layers, key=jax.random.key(4))
    coord = jnp.array([0.3, -0.7])
    film = jnp.ones((num_layers, 2 * hidden_dim)) * 0.5
    plain = body(coord)
    modulated = body(coord, film=film)
    assert not jnp.allclose(plain, modulated)


@pytest.mark.parametrize("body_cls", MFN_CLASSES)
def test_mfn_body_jit_matches_eager(body_cls: type) -> None:
    # Given: an MFN body and a coordinate
    # When: jit-compiling the call
    # Then: jit-compiled output matches eager
    body = body_cls(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(5))
    coord = jnp.array([0.1, -0.2])
    eager = body(coord)
    jitted = eqx.filter_jit(body)(coord)
    assert jnp.allclose(eager, jitted)


@pytest.mark.parametrize("body_cls", MFN_CLASSES)
def test_mfn_body_grad_is_finite_and_nonzero(body_cls: type) -> None:
    # Given: an MFN body and a sum-loss
    # When: taking grad over the body
    # Then: gradient pytree leaves are finite and at least one carries signal
    body = body_cls(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(6))
    coord = jnp.array([0.3, -0.4])

    def loss(b, c):
        return b(c).sum()

    grad = eqx.filter_grad(loss)(body, coord)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(grad) if eqx.is_array(leaf)]
    assert len(leaves) > 0
    for g in leaves:
        assert bool(jnp.all(jnp.isfinite(g)))
    assert any(bool(jnp.any(g != 0)) for g in leaves)


@pytest.mark.parametrize("body_cls", MFN_CLASSES)
def test_mfn_body_vmap_over_coords(body_cls: type) -> None:
    # Given: a batch of coords
    # When: vmapping the MFN call
    # Then: output shape carries the batch axis
    body = body_cls(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(7))
    coords = jax.random.uniform(jax.random.key(70), (5, 2), minval=-1.0, maxval=1.0)
    out = jax.vmap(body)(coords)
    assert out.shape == (5,)


@pytest.mark.parametrize("body_cls", MFN_CLASSES)
def test_mfn_body_canonicalises_out_features_one(body_cls: type) -> None:
    # Given: two bodies differing only in out_features=None vs 1
    # When: comparing pytree structure
    # Then: identical (canonicalisation rule applies)
    key = jax.random.key(0)
    a = body_cls(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=key, out_features=None)
    b = body_cls(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=key, out_features=1)
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)


@pytest.mark.parametrize("body_cls", MFN_CLASSES)
def test_mfn_body_rejects_zero_hidden_layers(body_cls: type) -> None:
    # Given: a 0-recurrence-step request
    # When: constructing
    # Then: constructor rejects
    with pytest.raises(AssertionError):
        body_cls(in_dim=2, hidden_dim=8, num_hidden_layers=0, key=jax.random.key(0))


def test_fourier_mfn_has_n_plus_one_filters() -> None:
    # Given: a FourierMFN with num_hidden_layers=4
    # When: counting filters
    # Then: there are 5 (= num_hidden_layers + 1) — the recurrence is
    # z_0 = g_0(x); z_{i+1} = g_{i+1}(x) * (W_i z_i + b_i); a regression that
    # mismatches the filter count (e.g. forgets g_0) would silently change
    # the network capacity.
    body = FourierMFN(in_dim=2, hidden_dim=16, num_hidden_layers=4, key=jax.random.key(0))
    assert len(body.filters) == 5
    for f in body.filters:
        assert isinstance(f, FourierFilter)


def test_gabor_mfn_has_n_plus_one_filters_of_gabor_type() -> None:
    # Given: a GaborMFN
    # When: inspecting the filter list
    # Then: every filter is a GaborFilter (catches a type-dispatch regression
    # where _MFNBody's filter construction silently fell back to FourierFilter)
    body = GaborMFN(in_dim=2, hidden_dim=16, num_hidden_layers=3, key=jax.random.key(0))
    assert len(body.filters) == 4
    for f in body.filters:
        assert isinstance(f, GaborFilter)


def test_fourier_filter_bias_uniform_phase() -> None:
    # Given: a FourierFilter with many output units
    # When: checking the bias range
    # Then: every bias is in [-pi, pi] — the uniform-phase init is the paper's
    # randomisation source. A regression that uses N(0, 1) here would cluster
    # phases around 0 and hurt spectral coverage.
    f = FourierFilter(in_dim=2, hidden_dim=2048, input_scale=256.0, n_layers=4, key=jax.random.key(0))
    assert float(jnp.min(f.b)) >= -jnp.pi - 1e-5
    assert float(jnp.max(f.b)) <= jnp.pi + 1e-5
    # And the empirical mean should be close to 0 (uniform).
    assert abs(float(jnp.mean(f.b))) < 0.2


def test_fourier_filter_weight_scale_matches_paper_formula() -> None:
    # Given: a FourierFilter with known input_scale and n_layers
    # When: drawing many filter weights
    # Then: |W| <= input_scale / sqrt(n_layers + 1) exactly (uniform bound)
    input_scale, n_layers = 32.0, 4
    f = FourierFilter(in_dim=4, hidden_dim=256, input_scale=input_scale, n_layers=n_layers, key=jax.random.key(1))
    expected = input_scale / jnp.sqrt(n_layers + 1.0)
    assert float(jnp.max(jnp.abs(f.W))) <= float(expected) + 1e-5


def test_gabor_filter_mu_within_unit_box() -> None:
    # Given: a GaborFilter
    # When: inspecting the centres
    # Then: all mu values are in [-1, 1] (the paper's prior).
    g = GaborFilter(in_dim=3, hidden_dim=512, n_layers=4, key=jax.random.key(0))
    assert float(jnp.min(g.mu)) >= -1.0 - 1e-5
    assert float(jnp.max(g.mu)) <= 1.0 + 1e-5


def test_gabor_filter_gamma_is_strictly_positive() -> None:
    # Given: a GaborFilter (gamma is a Gamma-distributed scale parameter)
    # When: inspecting gamma
    # Then: every entry is strictly positive — negative gamma would invert the
    # Gaussian envelope into a divergent one. A regression that uses N(0, 1)
    # here would silently produce explosions on roughly half the filters.
    g = GaborFilter(in_dim=2, hidden_dim=1024, n_layers=4, key=jax.random.key(1))
    assert bool(jnp.all(g.gamma > 0))


def test_fourier_mfn_no_unused_omega_field_on_filter() -> None:
    # Given: a FourierFilter
    # When: inspecting its fields
    # Then: it carries (W, b) only — no omega leaf, because the filter
    # absorbs the omega scale into W's uniform bound. Catches a regression
    # that adds a redundant omega learnable (would silently break pytree
    # parity with GaborFilter and inflate optimiser state).
    f = FourierFilter(in_dim=2, hidden_dim=4, input_scale=4.0, n_layers=1, key=jax.random.key(0))
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(f) if eqx.is_array(leaf)]
    assert len(leaves) == 2


def test_recurrence_state_stacked_along_axis_0() -> None:
    # Given: a FourierMFN with num_hidden_layers=4
    # When: inspecting the recurrence-linear arrays
    # Then: the W stack has shape (num_hidden_layers, hidden_dim, hidden_dim).
    # The stacked layout is scan-ready (a single jax.lax.scan over axis 0
    # would replay the recurrence without a Python loop), even though the
    # trunk's current implementation uses a Python loop for clarity.
    body = FourierMFN(in_dim=2, hidden_dim=8, num_hidden_layers=4, key=jax.random.key(0))
    assert body.recurrence_W.shape == (4, 8, 8)
    assert body.recurrence_b.shape == (4, 8)
