"""Tests for the FKAN basis family (ondes.basis.fkan, Mehrabian+ 2024).

Load-bearing properties, each with a test that would catch its failure:

1. **First-layer faithfulness** — per-edge coefficient tensor shape and count
   match the paper's Table I arithmetic (2·out·in·K); coefficients are Gaussian
   with the code-derived variance 1/(in·K); the bias is zero-initialised.
2. **Integer-harmonic structure** — the first layer evaluates a Fourier series of
   *every* integer harmonic with the fundamental fixed at 1 (both sine and
   cosine). This is what distinguishes FKAN from the odd-harmonic shared-ω comb.
3. **The two paper-vs-code flags change the forward pass** — ``gated_activation``
   and ``use_layernorm`` each move the output, and ``use_layernorm`` changes the
   pytree (a ``LayerNorm`` submodule appears / disappears).
4. **First-layer-only placement** — the body carries one Fourier feature map and
   ``num_hidden_layers`` plain ``tanh``/gated hidden layers, homogeneous in pytree
   structure.
5. **Body contract** — Body/BasisModule conformance, scalar/vector readout,
   ``out_features`` canonicalisation, FiLM modulation, JIT, gradient flow (incl.
   to the first-layer coefficients), vmap.

FKAN has no full-network SIREN reduction (its hidden activation is ``tanh``-based,
not sinusoidal); the analog of the comb's SIREN corner is the first layer's
reduction to a single integer-harmonic sinusoid, pinned in property 2.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from ondes import FKAN
from ondes.basis import Basis, BasisModule, Body, FKANHiddenLayer
from ondes.basis.fkan import FKANFirstLayer
from ondes.basis.siren import siren_init


EPS = float(jnp.finfo(jnp.float32).eps)


# --------------------------------------------------------------------------- #
# 1. first-layer faithfulness: shape, param count, init distribution          #
# --------------------------------------------------------------------------- #
def test_first_layer_output_shape_and_finite() -> None:
    # Given: a first-layer Fourier feature map in a realistic in != out shape
    # When: forward-passing a coordinate
    # Then: it produces one feature per output neuron, all finite.
    layer = FKANFirstLayer(in_dim=3, out_dim=16, n_freqs=8, key=jax.random.key(0))
    y = layer(jnp.linspace(-1.0, 1.0, 3))
    assert y.shape == (16,)
    assert bool(jnp.all(jnp.isfinite(y)))


def test_first_layer_param_count_matches_paper_table_i() -> None:
    # Given: the paper's canonical image config (in=2, first-layer width 128, K=270)
    # When: counting the coefficient tensors
    # Then: it equals Table I's 2·128·2·270 = 138,240 extra first-layer params.
    # The leading 2 is (cos, sin); the reference tensor is (2, out, in, K). A
    # sin-only or per-neuron (not per-edge) regression would miss this exactly.
    layer = FKANFirstLayer(in_dim=2, out_dim=128, n_freqs=270, key=jax.random.key(0))
    n_coeffs = int(layer.A.size + layer.B.size)
    assert n_coeffs == 2 * 128 * 2 * 270
    assert n_coeffs == 138_240


def test_first_layer_coeff_count_follows_per_edge_formula() -> None:
    # Given: an arbitrary non-canonical config
    # When: counting coefficients
    # Then: it is 2·out·in·K — per-edge (out×in independent series), 2K each.
    # Pins the general shape relationship, not just the canonical point.
    out_dim, in_dim, k = 7, 3, 5
    layer = FKANFirstLayer(in_dim=in_dim, out_dim=out_dim, n_freqs=k, key=jax.random.key(1))
    assert layer.A.shape == (out_dim, in_dim, k)
    assert layer.B.shape == (out_dim, in_dim, k)
    assert int(layer.A.size + layer.B.size) == 2 * out_dim * in_dim * k


def test_first_layer_bias_is_zero_init() -> None:
    # Given: a freshly-initialised first layer
    # When: inspecting the bias
    # Then: it is exactly zero — the code absorbs the k=0 constant term into the
    # bias and seeds it at zero, so the harmonic sum honestly starts at k=1.
    layer = FKANFirstLayer(in_dim=2, out_dim=32, n_freqs=16, key=jax.random.key(2))
    assert layer.bias.shape == (32,)
    assert bool(jnp.all(layer.bias == 0.0))


@pytest.mark.parametrize("coeff_name", ["A", "B"])
def test_first_layer_coeff_init_is_gaussian_with_code_variance(coeff_name: str) -> None:
    # Given: a wide first layer so the coefficient tensor is a large sample
    # When: measuring the sample mean and variance of A (cos) / B (sin)
    # Then: mean ~ 0 and variance ~ 1/(in·K) — the reference code's
    # randn/sqrt(in·K). A wrong normaliser (e.g. 1/in or 1/K alone, or an
    # unscaled randn) shifts the variance well outside the sampling band.
    in_dim, k, out_dim = 2, 270, 256
    layer = FKANFirstLayer(in_dim=in_dim, out_dim=out_dim, n_freqs=k, key=jax.random.key(3))
    coeffs = getattr(layer, coeff_name)
    n = coeffs.size
    expected_var = 1.0 / (in_dim * k)
    # Sampling standard errors for N i.i.d. Gaussian draws: SE(mean)=std/sqrt(N),
    # SE(var)/var = sqrt(2/N). N here is 256·2·270 = 138,240, so SE(mean) ~ 1.2e-4
    # and SE(var)/var ~ 0.0038. Bounds below are many SE wide (a real init bug is
    # an O(1) relative error, not a fractional-SE one).
    se_mean = (expected_var**0.5) / (n**0.5)
    assert abs(float(jnp.mean(coeffs))) < 6.0 * se_mean
    assert bool(jnp.allclose(jnp.var(coeffs), expected_var, rtol=0.05))


# --------------------------------------------------------------------------- #
# 2. integer-harmonic structure (fundamental fixed at 1, sin AND cos)         #
# --------------------------------------------------------------------------- #
def test_first_layer_reduces_to_integer_harmonic_cosine() -> None:
    # Given: a single-edge first layer whose cos coefficients are a one-hot at
    # harmonic index m (k = m+1) with zero sin coefficients and zero bias
    # When: evaluating on a grid
    # Then: it is exactly cos((m+1)·x) — the harmonics are the raw integers k=1..K
    # with the fundamental fixed at 1 (no ω prefactor). A 2k or a shared-ω scaling
    # regression (the comb's structure) would break this.
    m, k = 3, 6  # select the 4th harmonic (k=4)
    layer = FKANFirstLayer(in_dim=1, out_dim=1, n_freqs=k, key=jax.random.key(4))
    a = jnp.zeros_like(layer.A).at[0, 0, m].set(1.0)
    layer = eqx.tree_at(lambda t: (t.A, t.B), layer, (a, jnp.zeros_like(layer.B)))
    x = jnp.linspace(-1.0, 1.0, 32)
    got = jax.vmap(layer)(x[:, None])[:, 0]
    # atol scales with the K-term accumulation in the einsum (eps·K).
    assert bool(jnp.allclose(got, jnp.cos((m + 1) * x), atol=EPS * k))


def test_first_layer_reduces_to_integer_harmonic_sine() -> None:
    # Given: the same one-hot construction but on the sin coefficients B
    # When: evaluating on a grid
    # Then: it is exactly sin((m+1)·x) — pins the sine half of the series (a
    # cos-only regression, or dropping B, would pass the cosine test but fail here).
    m, k = 2, 6  # select the 3rd harmonic (k=3)
    layer = FKANFirstLayer(in_dim=1, out_dim=1, n_freqs=k, key=jax.random.key(5))
    b = jnp.zeros_like(layer.B).at[0, 0, m].set(1.0)
    layer = eqx.tree_at(lambda t: (t.A, t.B), layer, (jnp.zeros_like(layer.A), b))
    x = jnp.linspace(-1.0, 1.0, 32)
    got = jax.vmap(layer)(x[:, None])[:, 0]
    assert bool(jnp.allclose(got, jnp.sin((m + 1) * x), atol=EPS * k))


# --------------------------------------------------------------------------- #
# 3. hidden layer: init bound + the two activation forms                      #
# --------------------------------------------------------------------------- #
def test_hidden_layer_is_basis_subclass() -> None:
    # Given: an FKAN hidden layer
    # When: checking the inheritance contract
    # Then: it's a Basis (the polymorphism contract — the hidden layers are plain
    # linear+pointwise, unlike the first-layer feature map).
    layer = FKANHiddenLayer(8, 16, 30.0, key=jax.random.key(0), gated=False)
    assert isinstance(layer, Basis)


def test_hidden_layer_weight_bound_matches_code_faithful_siren_init() -> None:
    # Given: an FKAN hidden layer at ω₀=30
    # When: inspecting the linear weights
    # Then: they lie within the SIREN hidden bound sqrt(6/in)/ω₀ and reach near it
    # — the code's U(±sqrt(6/in)/ω₀) (the /ω₀ divisor is code-only, paper omits it).
    in_dim, omega = 64, 30.0
    layer = FKANHiddenLayer(in_dim, 64, omega, key=jax.random.key(1), gated=False)
    bound = float(jnp.sqrt(6.0 / in_dim) / omega)
    assert bool(jnp.all(jnp.abs(layer.W) <= bound))
    # Tightness: with 64·64 draws the empirical max should sit close to the bound,
    # confirming it's the actual init bound and not a loose over-estimate.
    assert float(jnp.max(jnp.abs(layer.W))) > 0.9 * bound
    assert float(layer.omega) == omega


def test_hidden_activation_tanh_form_is_exact() -> None:
    # Given: a non-gated hidden layer
    # When: applying _activate to a spread of pre-activations
    # Then: it equals tanh(ω₀·pre) exactly (paper Eq. 4).
    layer = FKANHiddenLayer(4, 4, 30.0, key=jax.random.key(2), gated=False)
    pre = jnp.array([-1.0, -0.3, 0.0, 0.3, 1.0])
    assert bool(jnp.allclose(layer._activate(pre), jnp.tanh(30.0 * pre), atol=8 * EPS))


def test_hidden_activation_gated_form_is_exact() -> None:
    # Given: a gated hidden layer
    # When: applying _activate
    # Then: it equals (pre + tanh(ω₀·pre))·sigmoid(pre) exactly (released code).
    layer = FKANHiddenLayer(4, 4, 30.0, key=jax.random.key(3), gated=True)
    pre = jnp.array([-1.0, -0.3, 0.0, 0.3, 1.0])
    expected = (pre + jnp.tanh(30.0 * pre)) * jax.nn.sigmoid(pre)
    assert bool(jnp.allclose(layer._activate(pre), expected, atol=8 * EPS))


def test_hidden_activation_forms_differ() -> None:
    # Given: two hidden layers identical but for the gated flag
    # When: activating the same pre
    # Then: the outputs differ — the flag is not a no-op (a stale wiring that
    # ignored `gated` would make these coincide).
    pre = jnp.array([-1.0, -0.3, 0.3, 1.0])
    tanh_layer = FKANHiddenLayer(4, 4, 30.0, key=jax.random.key(4), gated=False)
    gated_layer = FKANHiddenLayer(4, 4, 30.0, key=jax.random.key(4), gated=True)
    assert not bool(jnp.allclose(tanh_layer._activate(pre), gated_layer._activate(pre)))


# --------------------------------------------------------------------------- #
# 3b. the two body-level flags change the forward pass                        #
# --------------------------------------------------------------------------- #
def test_gated_activation_flag_changes_body_output() -> None:
    # Given: two FKAN bodies with the SAME key differing only in gated_activation
    # When: forward-passing the same coordinate
    # Then: outputs differ — the hidden-activation choice propagates to the output.
    coord = jnp.array([0.3, -0.4])
    kw = dict(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(6))
    tanh_body = FKAN(**kw, gated_activation=False)
    gated_body = FKAN(**kw, gated_activation=True)
    assert not bool(jnp.allclose(tanh_body(coord), gated_body(coord)))


def test_layernorm_flag_changes_output_and_pytree() -> None:
    # Given: two FKAN bodies with the SAME key differing only in use_layernorm
    # When: forward-passing and inspecting the layernorm field
    # Then: outputs differ; the True body carries an eqx.nn.LayerNorm (weight+bias
    # leaves), the False body carries None. Pins both the forward-pass effect and
    # the structural (pytree) presence of the code-only LayerNorm.
    coord = jnp.array([0.3, -0.4])
    kw = dict(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(7))
    ln_body = FKAN(**kw, use_layernorm=True)
    plain_body = FKAN(**kw, use_layernorm=False)
    assert not bool(jnp.allclose(ln_body(coord), plain_body(coord)))
    assert isinstance(ln_body.layernorm, eqx.nn.LayerNorm)
    assert plain_body.layernorm is None
    ln_leaves = [leaf for leaf in jax.tree_util.tree_leaves(ln_body.layernorm) if eqx.is_array(leaf)]
    assert len(ln_leaves) == 2  # learnable weight + bias


def test_default_config_reproduces_released_code() -> None:
    # Given: an FKAN body built with defaults
    # When: inspecting the activation + norm configuration
    # Then: it is the released-code config (gated hidden activation + LayerNorm)
    # whose numbers are the paper's Table I headline — the documented default.
    body = FKAN(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(8))
    assert isinstance(body.layernorm, eqx.nn.LayerNorm)
    assert all(layer.gated for layer in body.layers)


# --------------------------------------------------------------------------- #
# 4. first-layer-only placement + pytree homogeneity                          #
# --------------------------------------------------------------------------- #
def test_body_has_one_feature_map_and_num_hidden_layers_hidden_layers() -> None:
    # Given: an FKAN body
    # When: inspecting its structure
    # Then: exactly one FKANFirstLayer feature map, and num_hidden_layers plain
    # FKANHiddenLayer hidden layers after it — the "first layer only" placement.
    num_hidden = 3
    body = FKAN(in_dim=2, hidden_dim=16, num_hidden_layers=num_hidden, key=jax.random.key(9))
    assert isinstance(body.first, FKANFirstLayer)
    assert len(body.layers) == num_hidden
    assert all(isinstance(layer, FKANHiddenLayer) for layer in body.layers)


def test_body_hidden_layers_pytree_homogeneous() -> None:
    # Given: an FKAN body where in_dim == hidden_dim
    # When: comparing the pytree structure of every hidden layer
    # Then: identical — the `gated` static field is shared across layers, so a
    # per-layer discriminator can't sneak in. (The trunk is a Python loop like
    # RFF/MFN, so scan-usability is not claimed, only structural homogeneity.)
    body = FKAN(in_dim=4, hidden_dim=4, num_hidden_layers=5, key=jax.random.key(10))
    ref = jax.tree_util.tree_structure(body.layers[0])
    for i, layer in enumerate(body.layers):
        assert jax.tree_util.tree_structure(layer) == ref, f"hidden layer {i} structure differs"


# --------------------------------------------------------------------------- #
# 5. body contract (mirrors test_rff / test_comb)                             #
# --------------------------------------------------------------------------- #
def test_body_conforms_to_body_and_protocol() -> None:
    # Given: an FKAN body
    # When: checking the nominal and structural type contracts
    # Then: it satisfies both Body (base) and BasisModule (Protocol).
    body = FKAN(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=jax.random.key(11))
    assert isinstance(body, Body)
    assert isinstance(body, BasisModule)


def test_body_scalar_and_vector_out() -> None:
    # Given: FKAN bodies with default and integer out_features
    # When: forward-passing
    # Then: default gives a 0-d scalar, out_features=3 gives shape (3,).
    coord = jnp.array([0.1, -0.2])
    scalar = FKAN(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(12))
    vector = FKAN(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(12), out_features=3)
    assert scalar(coord).shape == ()
    assert vector(coord).shape == (3,)


def test_body_canonicalises_out_features_one_to_none() -> None:
    # Given: two FKAN bodies with out_features None and 1 (same layernorm setting)
    # When: comparing pytree structures
    # Then: identical, and both report out_features is None — the canonicalisation
    # rule applies uniformly across bases.
    key = jax.random.key(13)
    a = FKAN(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=key, out_features=None)
    b = FKAN(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=key, out_features=1)
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    assert a.out_features is None
    assert b.out_features is None


def test_body_film_modulation_changes_output() -> None:
    # Given: an FKAN body run with and without FiLM
    # When: passing a non-trivial FiLM tensor of shape (num_hidden_layers, 2·hidden)
    # Then: outputs differ — FiLM is threaded through the hidden layers (the
    # feature map is unmodulated, but the hidden layers see it).
    hidden_dim, num_layers = 8, 2
    body = FKAN(in_dim=2, hidden_dim=hidden_dim, num_hidden_layers=num_layers, key=jax.random.key(14))
    coord = jnp.array([0.3, -0.7])
    film = jnp.ones((num_layers, 2 * hidden_dim)) * 0.5
    assert not bool(jnp.allclose(body(coord), body(coord, film=film)))


def test_body_jit_matches_eager() -> None:
    # Given: an FKAN body and a coordinate
    # When: jit-compiling the call
    # Then: jitted output matches eager.
    body = FKAN(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(15))
    coord = jnp.array([0.25, -0.5])
    assert bool(jnp.allclose(body(coord), eqx.filter_jit(body)(coord)))


def test_body_grad_flows_to_first_layer_coeffs() -> None:
    # Given: an FKAN body and a scalar sum-loss
    # When: taking filter_grad over the body
    # Then: gradients reach the first-layer coefficients A and B (finite and not
    # all-zero) — a stop_gradient regression on the learnable Fourier coefficients
    # would zero these, silently freezing the whole FKAN mechanism.
    body = FKAN(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(16))
    coord = jnp.array([0.3, -0.4])

    def loss(b, c):
        return b(c).sum()

    grad = eqx.filter_grad(loss)(body, coord)
    for name, g in (("A", grad.first.A), ("B", grad.first.B)):
        assert bool(jnp.all(jnp.isfinite(g))), f"first.{name} grad non-finite"
        assert bool(jnp.any(g != 0.0)), f"first.{name} grad all-zero"


def test_body_grad_is_finite_and_nonzero_overall() -> None:
    # Given: an FKAN body
    # When: taking grad of a sum-loss
    # Then: all gradient leaves are finite and at least one carries signal.
    body = FKAN(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(17))
    coord = jnp.array([0.3, -0.4])

    def loss(b, c):
        return b(c).sum()

    grad = eqx.filter_grad(loss)(body, coord)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(grad) if eqx.is_array(leaf)]
    assert len(leaves) > 0
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    assert any(bool(jnp.any(g != 0.0)) for g in leaves)


def test_body_vmap_over_coords() -> None:
    # Given: a batch of coordinates and an FKAN body
    # When: vmapping the call
    # Then: output has the batch shape.
    body = FKAN(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(18))
    coords = jax.random.uniform(jax.random.key(180), (7, 2), minval=-1.0, maxval=1.0)
    assert jax.vmap(body)(coords).shape == (7,)


def test_body_num_hidden_layers_one_is_valid() -> None:
    # Given: an FKAN body with a single hidden layer (feature map + 1 tanh layer)
    # When: forward-passing
    # Then: it produces a scalar — the minimal valid depth (readout needs >=1
    # hidden layer to feed it).
    body = FKAN(in_dim=2, hidden_dim=8, num_hidden_layers=1, key=jax.random.key(19))
    assert body(jnp.array([0.1, -0.2])).shape == ()


@pytest.mark.parametrize("out_features", [None, 1, 4])
def test_body_rejects_zero_hidden_layers(out_features: int | None) -> None:
    # Given: a request to build with num_hidden_layers=0
    # When: constructing
    # Then: rejected — the readout contract needs at least one hidden layer.
    with pytest.raises(AssertionError):
        FKAN(in_dim=2, hidden_dim=8, num_hidden_layers=0, key=jax.random.key(20), out_features=out_features)


def test_body_rejects_non_positive_n_freqs() -> None:
    # Given: a request to build with n_freqs=0
    # When: constructing
    # Then: rejected — an empty harmonic axis would silently zero the feature map.
    with pytest.raises(AssertionError):
        FKAN(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=jax.random.key(21), n_freqs=0)


def test_hidden_init_matches_shared_siren_helper() -> None:
    # Given: the same (in, out, ω, key) fed to FKANHiddenLayer and siren_init
    # When: comparing the sampled weights
    # Then: identical — FKAN reuses the shared SIREN hidden-layer init verbatim
    # (is_first=False), so a divergence here would mean a private re-implementation
    # drifted from the family's one source of truth.
    key = jax.random.key(22)
    layer = FKANHiddenLayer(16, 16, 30.0, key=key, gated=False)
    w_ref, b_ref = siren_init(16, 16, 30.0, is_first=False, key=key)
    assert bool(jnp.allclose(layer.W, w_ref))
    assert bool(jnp.allclose(layer.b, b_ref))
