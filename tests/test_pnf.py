"""Tests for the PNF basis body (Yang+ 2022)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from ondes import PNF, FourierMFN
from ondes.basis import BasisModule, Body
from ondes.basis.mfn import FourierFilter


def test_pnf_body_conforms_to_basis_module_and_body() -> None:
    # Given: a PNF body
    # When: checking the structural and nominal type contracts
    # Then: it satisfies both, so downstream code that types against either
    # picks PNF up without a special case.
    body = PNF(in_dim=2, hidden_dim=16, num_hidden_layers=3, key=jax.random.key(0))
    assert isinstance(body, BasisModule)
    assert isinstance(body, Body)


def test_pnf_body_forward_scalar_shape() -> None:
    # Given: a default-out PNF body
    # When: forward-passing
    # Then: output is a 0-d scalar
    body = PNF(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(1))
    y = body(jnp.array([0.1, -0.2]))
    assert y.shape == ()


def test_pnf_body_forward_vector_shape() -> None:
    # Given: out_features=4
    # When: forward-passing
    # Then: output shape (4,)
    body = PNF(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(2), out_features=4)
    y = body(jnp.array([0.1, -0.2]))
    assert y.shape == (4,)


def test_pnf_body_trunk_shape() -> None:
    # Given: a PNF body
    # When: calling trunk()
    # Then: shape is (hidden_dim,)
    body = PNF(in_dim=2, hidden_dim=32, num_hidden_layers=3, key=jax.random.key(3))
    h = body.trunk(jnp.array([0.25, -0.5]))
    assert h.shape == (32,)


def test_pnf_has_mix_matrix_per_recurrence_step() -> None:
    # Given: a PNF body with num_hidden_layers=4
    # When: inspecting mix_W
    # Then: shape is (4, hidden_dim, hidden_dim) — one mix matrix per step.
    # The mix layer IS PNF's central addition vs MFN; a regression that
    # drops it would silently collapse PNF back to a Fourier MFN.
    body = PNF(in_dim=2, hidden_dim=8, num_hidden_layers=4, key=jax.random.key(0))
    assert body.mix_W.shape == (4, 8, 8)


def test_pnf_has_n_plus_one_fourier_filters() -> None:
    # Given: a PNF body with num_hidden_layers=4
    # When: inspecting filters
    # Then: there are 5 FourierFilter instances (same invariant as MFN)
    body = PNF(in_dim=2, hidden_dim=16, num_hidden_layers=4, key=jax.random.key(0))
    assert len(body.filters) == 5
    for f in body.filters:
        assert isinstance(f, FourierFilter)


def test_pnf_mix_matrix_changes_output_vs_zero_mix() -> None:
    # Given: a PNF body and the same body with mix_W zeroed
    # When: comparing outputs
    # Then: they differ — the mix layer must actually contribute. The paper-
    # faithful PNF recurrence (Eq. 5, z_{i+1} = g_{i+1}(x) * M_i z_i) is purely
    # multiplicative in z_i with no additive bias, so zeroing mix_W collapses
    # every step after the first to identically zero (the readout then maps
    # 0 to its bias). A non-zero mix_W produces a non-trivially different
    # output; this test pins both that the mix path is wired through the
    # forward pass AND that the recurrence has the bias-free structure (a
    # spurious additive bias inside the multiplication would let zeroed-mix
    # PNF still vary with the coord, breaking the equality with readout_b
    # below).
    body = PNF(in_dim=2, hidden_dim=16, num_hidden_layers=3, key=jax.random.key(7))
    zeroed_mix = eqx.tree_at(lambda b: b.mix_W, body, jnp.zeros_like(body.mix_W))
    coord = jnp.array([0.3, -0.4])
    assert not jnp.allclose(body(coord), zeroed_mix(coord))
    # Zeroed-mix PNF on any coord post-step-0 is readout(0) = readout_b.
    # Check a second distinct coord agrees with the first — bias-free
    # recurrence makes the post-zero-mix output coord-invariant.
    other = jnp.array([-0.1, 0.7])
    assert jnp.allclose(zeroed_mix(coord), zeroed_mix(other)), (
        "zeroed-mix PNF should be coord-invariant under the paper's bias-free "
        "recurrence (every step after the first collapses to 0); a coord-varying "
        "output here means a stray additive bias is hiding inside the multiplication."
    )


def test_pnf_body_film_modulation_changes_output() -> None:
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


def test_pnf_body_jit_matches_eager() -> None:
    # Given: a PNF body and a coord
    # When: jit-compiling
    # Then: matches eager
    body = PNF(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(5))
    coord = jnp.array([0.1, -0.2])
    eager = body(coord)
    jitted = eqx.filter_jit(body)(coord)
    assert jnp.allclose(eager, jitted)


def test_pnf_body_grad_is_finite_and_nonzero() -> None:
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


def test_pnf_mix_matrix_receives_gradient() -> None:
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


def test_pnf_body_vmap_over_coords() -> None:
    # Given: a batch of coords
    # When: vmapping
    # Then: output carries the batch axis
    body = PNF(in_dim=2, hidden_dim=16, num_hidden_layers=2, key=jax.random.key(7))
    coords = jax.random.uniform(jax.random.key(70), (5, 2), minval=-1.0, maxval=1.0)
    out = jax.vmap(body)(coords)
    assert out.shape == (5,)


def test_pnf_body_canonicalises_out_features_one() -> None:
    # Given: two PNF bodies differing only in out_features=None vs 1
    # When: comparing pytree structures
    # Then: identical
    key = jax.random.key(0)
    a = PNF(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=key, out_features=None)
    b = PNF(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=key, out_features=1)
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)


def test_pnf_body_rejects_zero_hidden_layers() -> None:
    # Given: 0 recurrence steps
    # When: constructing
    # Then: assertion fires
    with pytest.raises(AssertionError):
        PNF(in_dim=2, hidden_dim=8, num_hidden_layers=0, key=jax.random.key(0))


def test_pnf_has_no_recurrence_bias_unlike_fourier_mfn() -> None:
    # Given: PNF and FourierMFN at the same arch
    # When: introspecting their pytree leaves
    # Then: FourierMFN carries a `recurrence_b` field (Fathony+ 2021's
    # additive bias inside the multiplication: z_{i+1} = g_{i+1}(x) *
    # (W_i z_i + b_i)), while PNF does NOT — the paper's Eq. 5 has
    # z_{i+1} = g_{i+1}(x) * (M_i z_i), bias-free, and ondes mirrors that.
    # The earlier broken PNF carried both a redundant `recurrence_W` and
    # a stray `recurrence_b`; this test pins their absence so a future
    # refactor can't silently re-introduce them.
    key = jax.random.key(0)
    pnf = PNF(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=key)
    mfn = FourierMFN(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=key)

    assert hasattr(mfn, "recurrence_b"), "FourierMFN should carry a recurrence-bias field"
    assert hasattr(mfn, "recurrence_W"), "FourierMFN should carry a recurrence-linear field"
    assert not hasattr(pnf, "recurrence_b"), (
        "PNF must not carry a `recurrence_b` — the paper's recurrence is bias-free (Eq. 5). "
        "Re-introducing it collapses PNF back to a redundantly-parameterised FourierMFN."
    )
    assert not hasattr(pnf, "recurrence_W"), (
        "PNF must not carry a `recurrence_W` — the single linear is `mix_W` (the paper's W_l). "
        "Carrying both is the redundancy the round-6 review flagged."
    )


def test_pnf_recurrence_differs_from_fourier_mfn_at_matched_filters() -> None:
    # Given: PNF and FourierMFN constructed with identical filter weights
    # (we copy filters from PNF into the FourierMFN body via eqx.tree_at)
    # and identical recurrence linear (mix_W -> recurrence_W). The only
    # structural residual is FourierMFN's `recurrence_b` term, which is
    # randomly initialised and therefore non-zero with probability 1.
    #
    # When: forward-passing both on a common coord
    #
    # Then: outputs differ. The bug-version PNF (`z_{i+1} = g_{i+1} *
    # ((M+W) z + b)`) was algebraically equivalent to a FourierMFN with
    # the same combined linear and identical bias — under matched-filter
    # / matched-W conditions the two would have agreed exactly. The
    # paper-faithful PNF has no bias path, so the outputs must disagree
    # by exactly readout( g_1(x) * b_0 ) at one-layer depth — non-zero
    # by random init.
    in_dim, hidden_dim, num_layers = 2, 8, 1  # 1 layer keeps the math closed-form-checkable.
    key = jax.random.key(42)
    pnf = PNF(in_dim=in_dim, hidden_dim=hidden_dim, num_hidden_layers=num_layers, key=key)
    mfn = FourierMFN(in_dim=in_dim, hidden_dim=hidden_dim, num_hidden_layers=num_layers, key=jax.random.key(43))

    # Splice PNF's filters and mix_W into the MFN body so the only
    # remaining difference is the recurrence-linear `b` (and the
    # respective readouts, which we also align).
    mfn = eqx.tree_at(lambda m: m.filters, mfn, pnf.filters)
    mfn = eqx.tree_at(lambda m: m.recurrence_W, mfn, pnf.mix_W)
    mfn = eqx.tree_at(lambda m: m.readout_W, mfn, pnf.readout_W)
    mfn = eqx.tree_at(lambda m: m.readout_b, mfn, pnf.readout_b)
    # Crucially, mfn.recurrence_b is left at its random init — this is the
    # structural feature that distinguishes PNF (no bias) from FourierMFN
    # (with bias).
    assert bool(jnp.any(mfn.recurrence_b != 0)), "expected the random init to produce a non-zero recurrence_b"

    coord = jnp.array([0.3, -0.4])
    pnf_out = pnf(coord)
    mfn_out = mfn(coord)

    # The two MUST differ. If the bug were back (PNF == FourierMFN up to
    # redundant parameter-relabelling), the bias term would coincide and
    # this would pass falsely; with the paper-faithful PNF, the bias path
    # is a structural absence and the outputs are guaranteed to differ.
    assert not jnp.allclose(pnf_out, mfn_out), (
        f"PNF and FourierMFN at matched filters / matched linear must differ "
        f"(the structural residual is FourierMFN's bias path); got pnf={pnf_out}, mfn={mfn_out}"
    )
    # And the difference is exactly the bias path's contribution at this
    # depth: readout_W @ (g_1(coord) * recurrence_b) (one-layer closed form).
    expected_residual = pnf.readout_W @ (pnf.filters[1](coord) * mfn.recurrence_b[0])
    if pnf.out_features is None:
        expected_residual = expected_residual.squeeze(-1)
    assert jnp.allclose(mfn_out - pnf_out, expected_residual, atol=1e-5), (
        f"the disagreement between PNF and FourierMFN at num_hidden_layers=1 should equal "
        f"readout_W @ (g_1(x) * mfn.recurrence_b[0]); got diff={mfn_out - pnf_out}, "
        f"expected={expected_residual}"
    )
