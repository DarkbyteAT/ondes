"""Tests for ondes.encoding."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from ondes.encoding import (
    Dyadic,
    Encoding,
    Gaussian,
    Identity,
    LearnedGaussian,
    nyquist_sigma,
)


ENCODING_CLASSES = (Identity, Gaussian, LearnedGaussian, Dyadic)


def _build_encoding(
    cls: type[Encoding],
    *,
    rank: int = 3,
    num_freqs: int = 8,
    num_bands: int = 4,
    sigma: float = 2.5,
    key: jax.Array | None = None,
) -> Encoding:
    """Construct an encoding of any kind with sensible defaults for tests."""
    if key is None:
        key = jax.random.key(0)
    if cls is Identity:
        return Identity(in_dim=rank)
    if cls is Gaussian:
        return Gaussian(rank=rank, num_freqs=num_freqs, sigma=sigma, key=key)
    if cls is LearnedGaussian:
        return LearnedGaussian(rank=rank, num_freqs=num_freqs, key=key)
    if cls is Dyadic:
        return Dyadic(rank=rank, num_bands=num_bands)
    raise AssertionError(f"unknown encoding class {cls!r}")


def test_identity_round_trips_coord() -> None:
    # Given: an Identity encoding and a coord
    # When: encoding
    # Then: the coord comes out unchanged and out_dim matches in_dim
    enc = Identity(in_dim=3)
    coord = jnp.array([1.0, 2.0, 3.0])
    out = enc(coord)
    assert enc.out_dim == 3
    assert jnp.array_equal(out, coord)


def test_gaussian_forward_shape_and_finiteness() -> None:
    # Given: a Gaussian encoding with rank=3, num_freqs=8, sigma=2.5
    # When: encoding a coord
    # Then: output shape is (2 * num_freqs,) and values are finite
    enc = Gaussian(rank=3, num_freqs=8, sigma=2.5, key=jax.random.key(0))
    coord = jnp.array([0.1, 0.2, 0.3])
    out = enc(coord)
    assert enc.out_dim == 16
    assert out.shape == (16,)
    assert bool(jnp.all(jnp.isfinite(out)))
    # The first half are cosines (range [-1, 1]), the second half are sines (range [-1, 1])
    assert bool(jnp.all(jnp.abs(out) <= 1.0 + 1e-6))


def test_gaussian_sigma_is_folded_into_B() -> None:
    # Given: a Gaussian with large sigma
    # When: inspecting B's standard deviation
    # Then: it scales ~linearly with sigma. Documents the trade-off of the
    # "fold sigma into B" choice (you can't recover the construction-time
    # sigma scalar afterwards, only the empirical std).
    sigma = 10.0
    enc = Gaussian(rank=4, num_freqs=128, sigma=sigma, key=jax.random.key(0))
    # N(0, sigma^2) ⇒ empirical std ≈ sigma
    assert abs(float(jnp.std(enc.B)) - sigma) < sigma * 0.2


def test_learned_gaussian_forward_shape_and_finiteness() -> None:
    # Given: a LearnedGaussian encoding
    # When: encoding a coord
    # Then: output shape is (2 * num_freqs,), values are finite, sigma is a scalar
    enc = LearnedGaussian(rank=3, num_freqs=8, key=jax.random.key(0))
    coord = jnp.array([0.1, 0.2, 0.3])
    out = enc(coord)
    assert enc.out_dim == 16
    assert out.shape == (16,)
    assert bool(jnp.all(jnp.isfinite(out)))
    assert enc.sigma.shape == ()


def test_learned_gaussian_defaults_sigma_to_pi() -> None:
    # Given: LearnedGaussian with default sigma_init
    # When: inspecting sigma
    # Then: sigma equals pi to float32 precision (matches the prior
    # gaussian_learn factory default; LearnedGaussian materialises sigma as a
    # jnp array, which truncates to float32 by default).
    enc = LearnedGaussian(rank=2, num_freqs=4, key=jax.random.key(0))
    assert math.isclose(float(enc.sigma), math.pi, rel_tol=1e-6)


def test_learned_gaussian_custom_sigma_init() -> None:
    # Given: LearnedGaussian with custom sigma_init
    # When: inspecting sigma
    # Then: sigma is the value passed in
    enc = LearnedGaussian(rank=2, num_freqs=4, key=jax.random.key(0), sigma_init=4.0)
    assert float(enc.sigma) == 4.0


def test_learned_gaussian_sigma_is_in_pytree_gaussian_is_not() -> None:
    # Given: a Gaussian and a LearnedGaussian encoding
    # When: partitioning via eqx.is_array
    # Then: LearnedGaussian's trainable leaves include sigma (a scalar);
    # Gaussian's trainable leaves are just B (no sigma scalar). This is the
    # load-bearing structural invariant: LearnedGaussian's sigma is in the
    # pytree so optimisers can update it; Gaussian's spectral scale is frozen
    # at construction.
    key = jax.random.key(0)
    gauss = Gaussian(rank=3, num_freqs=4, sigma=2.5, key=key)
    learned = LearnedGaussian(rank=3, num_freqs=4, key=key)
    gauss_arrays, _ = eqx.partition(gauss, eqx.is_array)
    learned_arrays, _ = eqx.partition(learned, eqx.is_array)
    gauss_leaves = [leaf for leaf in jax.tree_util.tree_leaves(gauss_arrays) if eqx.is_array(leaf)]
    learned_leaves = [leaf for leaf in jax.tree_util.tree_leaves(learned_arrays) if eqx.is_array(leaf)]
    # Gaussian: just B (one matrix)
    assert len(gauss_leaves) == 1
    assert gauss_leaves[0].shape == (4, 3)
    # LearnedGaussian: B_raw + sigma scalar
    assert len(learned_leaves) == 2
    shapes = sorted([leaf.shape for leaf in learned_leaves], key=lambda s: len(s))
    assert shapes == [(), (4, 3)]


def test_dyadic_forward_shape_and_finiteness() -> None:
    # Given: a Dyadic encoding with rank=2, num_bands=4
    # When: encoding a coord
    # Then: output shape is (rank * 2 * num_bands,) = (16,) and values are finite
    enc = Dyadic(rank=2, num_bands=4)
    coord = jnp.array([0.1, 0.2])
    out = enc(coord)
    assert enc.out_dim == 16
    assert out.shape == (16,)
    assert bool(jnp.all(jnp.isfinite(out)))
    # Sin and cos are bounded by 1
    assert bool(jnp.all(jnp.abs(out) <= 1.0 + 1e-6))


def test_dyadic_default_num_bands_is_four() -> None:
    # Given: Dyadic with no num_bands argument
    # When: inspecting num_bands
    # Then: default is 4 (matches the prior dyadic factory default L=4)
    enc = Dyadic(rank=3)
    assert enc.num_bands == 4
    assert enc.out_dim == 3 * 2 * 4


def test_dyadic_bands_precomputed_as_pytree_leaf() -> None:
    # Given: a Dyadic encoding
    # When: inspecting bands and partitioning via eqx.is_array
    # Then: bands has shape (num_bands,) with values 2**k * pi for k in
    # range(num_bands), and is a real pytree array leaf — proves the
    # precomputation moves work out of __call__ into __init__ without
    # silently dropping the values from the pytree.
    enc = Dyadic(rank=2, num_bands=5)
    expected = (2.0 ** jnp.arange(5)) * jnp.pi
    assert enc.bands.shape == (5,)
    assert jnp.allclose(enc.bands, expected)
    arrays, _ = eqx.partition(enc, eqx.is_array)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(arrays) if eqx.is_array(leaf)]
    assert any(leaf.shape == (5,) for leaf in leaves), "bands not in pytree"


@pytest.mark.parametrize("cls", ENCODING_CLASSES)
def test_encoding_subclass_of_abc(cls: type) -> None:
    # Given: each concrete encoding class
    # When: checking isinstance against the Encoding ABC
    # Then: each is an Encoding — downstream code can express "any encoding"
    # in a single type signature.
    enc = _build_encoding(cls)
    assert isinstance(enc, Encoding)


@pytest.mark.parametrize("cls", ENCODING_CLASSES)
def test_encoding_subclasses_compute_correct_forward(cls: type) -> None:
    # Given: an encoding of each class on a fixed coord
    # When: encoding
    # Then: output shape matches the encoding's reported out_dim and values
    # are finite. Catches subclasses that report an out_dim inconsistent with
    # their forward pass.
    enc = _build_encoding(cls)
    coord = jnp.array([0.1, 0.2, 0.3])
    out = enc(coord)
    assert out.shape == (enc.out_dim,)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_encoding_classes_are_disjoint_types() -> None:
    # Given: one instance of each encoding class
    # When: collecting their types
    # Then: each is its own class — no shared discriminator, no shared
    # concrete base beyond the ABC. Catches a regression where the classes
    # accidentally collapse into one.
    encs = [
        Identity(in_dim=3),
        Gaussian(rank=3, num_freqs=4, sigma=1.0, key=jax.random.key(0)),
        LearnedGaussian(rank=3, num_freqs=4, key=jax.random.key(0)),
        Dyadic(rank=3),
    ]
    classes = {type(e) for e in encs}
    assert classes == {Identity, Gaussian, LearnedGaussian, Dyadic}


def test_non_gaussian_encodings_have_no_sigma_field() -> None:
    # Given: the non-Gaussian encoding classes
    # When: checking attribute presence
    # Then: only Gaussian/LearnedGaussian carry spectral-scale state.
    # Identity and Dyadic pytrees do not contain unused sigma/B leaves.
    assert not hasattr(Identity(in_dim=3), "sigma")
    assert not hasattr(Identity(in_dim=3), "B")
    assert not hasattr(Dyadic(rank=3), "sigma")
    assert not hasattr(Dyadic(rank=3), "B")


def test_nyquist_sigma_uses_longest_axis() -> None:
    # Given: shape (32, 1024) — longest axis is 1024
    # When: computing sigma
    # Then: sigma is (1024 - 1) / 4 = 255.75
    assert nyquist_sigma((32, 1024)) == (1024 - 1) / 4


def test_nyquist_sigma_handles_4d_conv_kernel() -> None:
    # Given: shape (3, 3, 1, 16) — longest axis is 16
    # When: computing sigma
    # Then: sigma is max(1, (16 - 1) / 4) = 3.75
    assert nyquist_sigma((3, 3, 1, 16)) == max(1.0, (16 - 1) / 4)


def test_nyquist_sigma_floor_is_one() -> None:
    # Given: a degenerate shape (1,) where (N-1)/4 = 0
    # When: computing sigma
    # Then: the 1.0 floor kicks in
    assert nyquist_sigma((1,)) == 1.0


def test_nyquist_sigma_composes_with_gaussian() -> None:
    # Given: a Gaussian encoding constructed with sigma=nyquist_sigma(weight_shape)
    # When: inspecting B's std
    # Then: it scales to the Nyquist sigma. Documents the recommended usage
    # pattern (nyquist_sigma is the per-leaf rule a downstream renderer would
    # apply when building one encoding per weight tensor).
    weight_shape = (32, 1024)
    sigma = nyquist_sigma(weight_shape)
    enc = Gaussian(rank=3, num_freqs=256, sigma=sigma, key=jax.random.key(0))
    # Empirical std ≈ sigma to within ~20% on this sample size
    assert abs(float(jnp.std(enc.B)) - sigma) < sigma * 0.2
