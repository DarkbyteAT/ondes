"""Tests for the WINNER (siren_square) basis body (arXiv 2509.09719).

Covers the value-type schedule, the spectral-centroid utility, WINNER
construction (perturbation locality, bias invariance, omega-divisor trap
regression, reset_noise rebuild semantics, ``from_signal`` equivalence),
the Theorem 3.1 layer-1 variance smoke test, and one
``@pytest.mark.exploratory`` end-to-end PSNR comparison vs SIREN that's
explicitly gated out of the per-round suite.

All tests use plain ``def test_*`` functions with Given-When-Then
docstrings. Tolerances are dtype- or MC-derived; no magic thresholds.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ondes import SIREN, WINNER, WinnerSchedule, spectral_centroid
from ondes.basis import Basis, BasisModule, Body


# ---------------------------------------------------------------------------
# WinnerSchedule + spectral_centroid (unit)
# ---------------------------------------------------------------------------


def test_winner_schedule_audio_factory_matches_paper_code() -> None:
    """Given: WinnerSchedule.audio.

    When: reading its fields.
    Then: ``(s0_max, a, b) == (3500, 7, 3)`` — matches the reference impl's
    ``set_noise_scales`` branch for ``in_dim == 1``. Note: the paper text
    says ``a = 5`` for audio; the code uses ``a = 7``. We follow the code.
    """
    sched = WinnerSchedule.audio()
    assert sched.s0_max == 3500.0
    assert sched.a == 7.0
    assert sched.b == 3.0


def test_winner_schedule_image_factory_matches_paper_code() -> None:
    """Given: WinnerSchedule.image.

    When: reading its fields.
    Then: ``(s0_max, a, b) == (50, 5, 0.4)`` — paper text and code agree.
    """
    sched = WinnerSchedule.image()
    assert sched.s0_max == 50.0
    assert sched.a == 5.0
    assert sched.b == 0.4


def test_schedule_scales_known_centroid() -> None:
    """Given: a known centroid and channel count.

    When: calling schedule.scales.
    Then: analytic match for both audio and image schedules. ``z =
    centroid / n_channels = 0.1`` here; audio gives ``s0 = 3500 · (1 -
    exp(-0.7))``, image gives ``s0 = 50 · (1 - exp(-0.5))``.
    """
    # Pick centroid and n_ch so z = 0.1
    centroid = jnp.asarray(0.3, dtype=jnp.float32)
    n_ch = 3
    s0_audio, s1_audio = WinnerSchedule.audio().scales(centroid, n_ch)
    s0_image, s1_image = WinnerSchedule.image().scales(centroid, n_ch)

    z = 0.1
    expected_s0_audio = 3500.0 * (1.0 - np.exp(-7.0 * z))
    expected_s1_audio = 3.0 * z
    expected_s0_image = 50.0 * (1.0 - np.exp(-5.0 * z))
    expected_s1_image = 0.4 * z

    tol = jnp.finfo(jnp.float32).eps * 1e4  # mild slack for the exp + multiply chain
    assert jnp.allclose(s0_audio, expected_s0_audio, atol=tol * 3500.0)
    assert jnp.allclose(s1_audio, expected_s1_audio, atol=tol * 3.0)
    assert jnp.allclose(s0_image, expected_s0_image, atol=tol * 50.0)
    assert jnp.allclose(s1_image, expected_s1_image, atol=tol * 0.4)


def test_spectral_centroid_pure_sinusoid() -> None:
    """Given: a single-channel pure sinusoid at integer-cycle frequency ``f``.

    When: calling spectral_centroid.
    Then: the returned value (after the ``*2/n_ch`` normalisation) matches
    ``2 · f / f_s`` within one rFFT bin width. ``f_s = 1`` here so
    ``f_norm = f / N``; one bin is ``1/N``.

    We pick a frequency that lands exactly on a bin (``k = 100`` cycles in
    ``N = 1024`` samples ⇒ ``f_norm = 100/1024``) to avoid spectral
    leakage — a non-integer-cycle sinusoid spreads energy across several
    bins and shifts the centroid by O(1/k) of the fundamental, which
    would mask a real divisor-trap regression with leakage noise.
    """
    N = 1024
    k = 100  # integer cycles → no leakage
    f_norm = k / N
    t = jnp.arange(N, dtype=jnp.float32)
    signal = jnp.sin(2.0 * jnp.pi * f_norm * t)[:, None]  # (N, 1) — one channel

    centroid = spectral_centroid(signal, freq_axis=0, channel_axis=-1)
    # n_ch == 1, so the final returned value is 2 · raw_centroid / 1
    expected = 2.0 * f_norm  # raw centroid is f_norm; n_ch=1 doubles it
    one_bin = 1.0 / N
    assert jnp.allclose(centroid, expected, atol=one_bin * 2.0), (
        f"centroid={float(centroid)} expected≈{expected} within one bin {one_bin}"
    )


def test_spectral_centroid_zero_signal_returns_zero() -> None:
    """Given: an all-zero signal.

    When: calling spectral_centroid.
    Then: returns 0 (zero-safe divide on a zero spectrum). Without the
    safety branch this would NaN.
    """
    signal = jnp.zeros((64, 64, 3), dtype=jnp.float32)
    centroid = spectral_centroid(signal)
    assert jnp.isfinite(centroid)
    assert float(centroid) == 0.0


def test_spectral_centroid_dtype_preserved_float32() -> None:
    """Given: a float32 input.

    When: calling spectral_centroid.
    Then: output dtype is float32 — no silent weak-promotion to float64
    via the ``* 2.0`` normalisation factor (which is the trap pinned by
    reviewer B; ``2.0`` is now built with ``jnp.asarray(2.0, signal.dtype)``
    inside the function).

    The float64 branch is covered by
    ``test_spectral_centroid_dtype_preserved_float64`` under an
    ``enable_x64`` context.
    """
    rng = np.random.default_rng(0)
    arr32 = jnp.asarray(rng.standard_normal((32, 32, 3)).astype(np.float32))
    assert spectral_centroid(arr32).dtype == jnp.float32


def test_spectral_centroid_dtype_preserved_float64() -> None:
    """Given: a float64 input under ``jax_enable_x64``.

    When: calling spectral_centroid.
    Then: output dtype is float64. Catches the dtype-downcast trap pinned
    by reviewer B: an unguarded ``jnp.asarray(s, dtype=jnp.float32)`` or
    ``centroid * 2.0`` would force the result back to float32 under x64
    mode, silently dropping precision.

    Concurrency caveat: ``jax_enable_x64`` is a process-global flag, and
    jax 0.10 does not expose a public context manager for it
    (``jax.experimental.enable_x64`` was removed; only
    ``jax.config.update`` remains). We toggle + restore in a
    ``try / finally`` block, which is safe under serial test execution
    (the default ``uv run pytest`` mode) but races under ``pytest-xdist``
    workers in the same process. If we ever add ``pytest-xdist`` to the
    project, mark this test with ``@pytest.mark.serial`` and configure
    xdist to isolate it. For now: serial-only by convention.
    """
    prev = jax.config.read("jax_enable_x64")
    try:
        jax.config.update("jax_enable_x64", True)
        rng = np.random.default_rng(0)
        arr64 = jnp.asarray(rng.standard_normal((32, 32, 3)))  # numpy default is float64
        assert arr64.dtype == jnp.float64, f"setup error: arr64.dtype={arr64.dtype}"
        out = spectral_centroid(arr64)
        assert out.dtype == jnp.float64, f"got {out.dtype} under jax_enable_x64=True"
    finally:
        jax.config.update("jax_enable_x64", prev)


def _old_global_sum_centroid(signal: jax.Array, *, freq_axis: int = -2, channel_axis: int = -1) -> float:
    """Reference impl of the R1 (pre-fix) ``spectral_centroid``.

    Kept locally as a *contrast oracle* for
    ``test_spectral_centroid_per_channel_then_mean_distinct_from_global_sum``:
    that test must produce a materially different number under this
    function vs the current per-slice-then-mean impl. If the gap
    collapses, the regression test has gone vacuous (as it did in R1 —
    see the round-2 review batch).
    """
    n_channels = signal.shape[channel_axis]
    n_freq = signal.shape[freq_axis]
    spectrum = jnp.abs(jnp.fft.rfft(signal, axis=freq_axis))
    freq_bins = jnp.fft.rfftfreq(n_freq, d=1.0).astype(signal.dtype)
    freq_shape = [1] * spectrum.ndim
    freq_shape[freq_axis] = freq_bins.shape[0]
    freq_bins_b = freq_bins.reshape(freq_shape)
    weighted = spectrum * freq_bins_b
    reduce_axes = tuple(ax for ax in range(spectrum.ndim) if ax != channel_axis % spectrum.ndim)
    num = jnp.sum(weighted, axis=reduce_axes)
    den = jnp.sum(spectrum, axis=reduce_axes)
    per_channel = jnp.where(den != 0, num / jnp.where(den != 0, den, 1.0), 0.0)
    centroid = jnp.mean(per_channel)
    return float((centroid * 2.0) / n_channels)


def test_spectral_centroid_works_under_nondefault_axes() -> None:
    """Given: a ``(C, H, W)`` layout signal with ``channel_axis=0,
    freq_axis=-1`` (the reference's 2-D image branch transposed).

    When: calling spectral_centroid with explicit non-default axes.
    Then: matches the hand-computed expected value. Happy-path coverage
    for the non-default axis kwargs — exercises the rFFT-along-an-
    arbitrary-axis path and the ``n_channels = signal.shape[channel_axis]``
    lookup at a non-trailing position.

    Construction: 3 channels, 4 rows of varying integer-cycle frequencies,
    N=256 samples per row. Single uniform-weight mean over the per-slice
    centroid gives ``mean_all(k_all) / N · 2 / n_ch``. We pick distinct
    per-row, per-channel frequencies so the test isn't accidentally a
    pure single-frequency check.

    Note: this is a happy-path test, not a discriminator for the
    channel-axis-resolution branch. The round-3 review established that
    such a branch is dead code under uniform-weight mean over uniform-
    shape inputs (commutativity collapses any branch flip into the same
    scalar), so ``spectral_centroid`` no longer carries one — the test
    simply confirms the function works under the documented non-default
    axis kwargs.
    """
    N = 256
    # All frequencies must stay strictly below Nyquist (N/2 = 128). A bin
    # above Nyquist aliases under rFFT, which would silently throw the
    # hand-computed expected value off.
    ks_ch0 = [8, 16, 32, 64]
    ks_ch1 = [120, 50, 50, 50]
    ks_ch2 = [40, 40, 40, 100]
    n_ch = 3
    assert max(ks_ch0 + ks_ch1 + ks_ch2) < N // 2, "test frequencies must stay below Nyquist"

    t = jnp.arange(N, dtype=jnp.float32)
    ch0 = jnp.stack([jnp.sin(2.0 * jnp.pi * (k / N) * t) for k in ks_ch0], axis=0)
    ch1 = jnp.stack([jnp.sin(2.0 * jnp.pi * (k / N) * t) for k in ks_ch1], axis=0)
    ch2 = jnp.stack([jnp.sin(2.0 * jnp.pi * (k / N) * t) for k in ks_ch2], axis=0)
    signal = jnp.stack([ch0, ch1, ch2], axis=0)  # (C=3, H=4, W=256)

    centroid = spectral_centroid(signal, freq_axis=-1, channel_axis=0)
    all_ks = ks_ch0 + ks_ch1 + ks_ch2
    expected = (sum(all_ks) / len(all_ks)) / N * 2.0 / n_ch
    one_bin = 1.0 / N
    assert jnp.allclose(centroid, expected, atol=one_bin * 2.0), (
        f"centroid={float(centroid)} expected≈{expected} within one bin {one_bin}"
    )


def test_spectral_centroid_per_channel_then_mean_distinct_from_global_sum() -> None:
    """Given: a single-channel signal where rows have *different
    amplitudes* AND *different frequencies*, so the per-row-then-mean
    and global-sum-then-divide reductions diverge meaningfully.

    When: comparing ondes' ``spectral_centroid`` to
    ``_old_global_sum_centroid`` (the R1 pre-fix impl, kept inline as a
    contrast oracle).
    Then: the two impls produce materially different numbers — by
    construction here, new ≈ 0.322 vs old ≈ 0.763.

    Why this matters: the R1 version of this test broadcast
    *identical-amplitude* rows within each channel
    (``jnp.broadcast_to``), which means amplitude cancelled in the
    per-channel num/den reduction. The 100x cross-channel amplitude
    ratio didn't matter because each channel was reduced
    *independently*. R2 review flagged this as vacuous (A1, C1).

    Correct construction: rows within a single channel have different
    amplitudes correlated with different frequencies, so:
    - Per-row-then-mean (NEW): uniform average ``(1/H) · Σ_r k_r/N``,
      independent of amplitude.
    - Global-sum-then-divide (OLD): amplitude-weighted
      ``Σ_r amp_r·k_r / (N · Σ_r amp_r)`` (since the rFFT magnitude of
      ``amp · sin(2π k t/N)`` is proportional to ``amp``).

    With ``k_r = [5, 10, 50, 100]`` and ``amp_r = [1, 1, 1, 100]``
    (high amp aligned with high freq):
    - NEW = ((5+10+50+100)/4) / N · 2 / 1 = 165/512 ≈ 0.3223
    - OLD = (1·5 + 1·10 + 1·50 + 100·100) / (256 · 103) · 2 ≈ 0.7634
    The 0.44 gap dwarfs the one-bin tolerance (≈ 0.008).
    """
    N = 256
    rows = 4
    ks = [5, 10, 50, 100]
    amps = [1.0, 1.0, 1.0, 100.0]
    t = jnp.arange(N, dtype=jnp.float32)
    rows_stack = jnp.stack(
        [amps[r] * jnp.sin(2.0 * jnp.pi * (ks[r] / N) * t) for r in range(rows)],
        axis=0,
    )
    signal = rows_stack[..., None]  # (rows=4, N=256, channel=1)

    centroid_new = spectral_centroid(signal, freq_axis=-2, channel_axis=-1)
    centroid_old = _old_global_sum_centroid(signal, freq_axis=-2, channel_axis=-1)

    expected_new = sum(ks) / rows / N * 2.0  # n_ch = 1
    expected_old = sum(amps[r] * ks[r] for r in range(rows)) / (N * sum(amps)) * 2.0
    one_bin = 1.0 / N

    # New impl matches per-row-then-mean prediction
    assert jnp.allclose(centroid_new, expected_new, atol=one_bin * 2.0), (
        f"new centroid={float(centroid_new)} expected≈{expected_new}"
    )
    # Old impl matches the global-sum prediction (sanity check on the oracle)
    assert abs(centroid_old - expected_old) < one_bin * 2.0, (
        f"old oracle={centroid_old} expected≈{expected_old}; oracle drift suspected"
    )
    # And they differ — gap must dwarf the one-bin tolerance, otherwise the
    # construction has gone vacuous again.
    gap = abs(float(centroid_new) - centroid_old)
    assert gap > 0.1, (
        f"new={float(centroid_new)} and old={centroid_old} agree to within {gap}; "
        f"test construction is vacuous, no distinguishing power between reductions"
    )


def test_winner_schedule_zero_centroid_gives_zero_scales() -> None:
    """Given: a zero centroid.

    When: calling ``WinnerSchedule.image().scales(0.0, n_channels)``.
    Then: both ``s0`` and ``s1`` are exactly 0 — the WINNER perturbation
    is the identity for an all-zero target. Boundary case from
    ``WINNER.from_signal(jnp.zeros(...), ...)``: ``spectral_centroid``
    returns 0 by the zero-safe divide, schedule must propagate that to
    zero scales so the resulting WINNER is bit-equal to a vanilla SIREN
    init.
    """
    sched = WinnerSchedule.image()
    s0, s1 = sched.scales(jnp.asarray(0.0, dtype=jnp.float32), n_channels=3)
    assert float(s0) == 0.0
    assert float(s1) == 0.0


# ---------------------------------------------------------------------------
# WINNER construction (unit)
# ---------------------------------------------------------------------------


def test_winner_is_body_and_basis_module() -> None:
    """Given: a WINNER.

    When: checking type contracts.
    Then: it satisfies both the structural ``BasisModule`` protocol and
    the nominal ``Body`` base — downstream code that types against either
    accepts a WINNER without changes.
    """
    w = WINNER(in_dim=2, hidden_dim=16, num_hidden_layers=3, key=jax.random.key(0), s0=1.0, s1=0.1)
    assert isinstance(w, BasisModule)
    assert isinstance(w, Body)


def test_winner_layers_are_siren_layers() -> None:
    """Given: a WINNER body.

    When: inspecting layers.
    Then: each layer is a ``Basis`` subclass (specifically ``SIRENLayer``
    by construction).
    """
    w = WINNER(in_dim=2, hidden_dim=16, num_hidden_layers=3, key=jax.random.key(0), s0=1.0, s1=0.1)
    for layer in w.layers:
        assert isinstance(layer, Basis)


def test_winner_init_perturbs_only_first_two_weights() -> None:
    """Given: a WINNER and a same-master-key WINNER with ``s0=s1=0``.

    When: comparing layer weights.
    Then: layers ≥ 2 are bit-equal; layers 0 and 1 differ. The s0=s1=0
    baseline is a vanilla SIREN constructed with WINNER's exact key split,
    so any drift in layers ≥ 2 would indicate the WINNER ``__init__``
    accidentally touches them.
    """
    key = jax.random.key(0)
    in_dim, hidden_dim, n_HL = 2, 32, 4
    w = WINNER(in_dim=in_dim, hidden_dim=hidden_dim, num_hidden_layers=n_HL, key=key, s0=2.0, s1=0.5)
    w_zero = WINNER(in_dim=in_dim, hidden_dim=hidden_dim, num_hidden_layers=n_HL, key=key, s0=0.0, s1=0.0)

    # Layers 0 and 1: weight differs (noise added)
    assert bool(jnp.any(w.layers[0].W != w_zero.layers[0].W))
    assert bool(jnp.any(w.layers[1].W != w_zero.layers[1].W))
    # Layers 2+: bit-equal
    for i in range(2, n_HL):
        assert bool(jnp.all(w.layers[i].W == w_zero.layers[i].W))
        assert bool(jnp.all(w.layers[i].b == w_zero.layers[i].b))


def test_winner_init_leaves_biases_unchanged() -> None:
    """Given: a WINNER and a same-master-key zero-noise WINNER.

    When: comparing biases across every layer.
    Then: every bias is bit-equal. WINNER only perturbs ``W`` on layers
    0/1; biases are sampled by ``siren_init`` and never touched.
    """
    key = jax.random.key(7)
    n_HL = 4
    w = WINNER(in_dim=2, hidden_dim=32, num_hidden_layers=n_HL, key=key, s0=2.0, s1=0.5)
    w_zero = WINNER(in_dim=2, hidden_dim=32, num_hidden_layers=n_HL, key=key, s0=0.0, s1=0.0)
    for i in range(n_HL):
        assert bool(jnp.all(w.layers[i].b == w_zero.layers[i].b))


def test_winner_noise_scales_match_formula() -> None:
    """Given: WINNERs at increasing ``s0``, ``s1``.

    When: comparing the difference layers[0].W - W_zero.W and layers[1].W
        - W_zero.W to the analytic Gaussian std.
    Then: empirical std of the diff ≈ ``s_i / omega_hidden`` within MC
    tolerance.
    """
    key = jax.random.key(11)
    hidden_dim, n_HL = 256, 4
    in_dim = 2
    omega_h = 30.0
    s0, s1 = 10.0, 0.5
    w = WINNER(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=n_HL,
        key=key,
        s0=s0,
        s1=s1,
        omega_hidden=omega_h,
    )
    w_zero = WINNER(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=n_HL,
        key=key,
        s0=0.0,
        s1=0.0,
        omega_hidden=omega_h,
    )

    noise0 = w.layers[0].W - w_zero.layers[0].W
    noise1 = w.layers[1].W - w_zero.layers[1].W

    # MC tolerance: std of sample std on N normal samples ~ sigma / sqrt(2N).
    # noise0 has hidden_dim * in_dim samples; noise1 has hidden_dim * hidden_dim.
    n0_samples = hidden_dim * in_dim
    n1_samples = hidden_dim * hidden_dim
    expected_std0 = s0 / omega_h
    expected_std1 = s1 / omega_h
    tol0 = 4.0 * expected_std0 / np.sqrt(2.0 * n0_samples)  # 4σ MC band
    tol1 = 4.0 * expected_std1 / np.sqrt(2.0 * n1_samples)
    assert jnp.abs(jnp.std(noise0) - expected_std0) < tol0
    assert jnp.abs(jnp.std(noise1) - expected_std1) < tol1


def test_winner_omega_divisor_is_hidden_not_first() -> None:
    """Given: two WINNERs at same s0, s1; one with ``omega_first=30``,
    one with ``omega_first=3000``.

    When: comparing the layer-0 noise std.
    Then: the std is the same (within MC tolerance) — the noise scale uses
    ``omega_hidden`` for BOTH layers, not ``omega_first`` for layer 0.
    Regression test for the trap that the ml-engineer flagged
    (``WINNER_DECISIONS.md`` item 1).

    Without the trap regression: layer 0's noise would scale with
    ``omega_first``, so the ``omega_first=3000`` variant would have noise
    std 100x smaller — easily detected.
    """
    key = jax.random.key(13)
    hidden_dim, n_HL, in_dim = 256, 4, 2
    s0, s1 = 5.0, 0.3
    omega_h = 30.0
    w_a = WINNER(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=n_HL,
        key=key,
        s0=s0,
        s1=s1,
        omega_first=30.0,
        omega_hidden=omega_h,
    )
    w_b = WINNER(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=n_HL,
        key=key,
        s0=s0,
        s1=s1,
        omega_first=3000.0,
        omega_hidden=omega_h,
    )
    w_a_zero = WINNER(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=n_HL,
        key=key,
        s0=0.0,
        s1=0.0,
        omega_first=30.0,
        omega_hidden=omega_h,
    )
    w_b_zero = WINNER(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=n_HL,
        key=key,
        s0=0.0,
        s1=0.0,
        omega_first=3000.0,
        omega_hidden=omega_h,
    )
    noise0_a = w_a.layers[0].W - w_a_zero.layers[0].W
    noise0_b = w_b.layers[0].W - w_b_zero.layers[0].W
    # Both should have std ~ s0/omega_h
    expected = s0 / omega_h
    n_samples = hidden_dim * in_dim
    tol = 4.0 * expected / np.sqrt(2.0 * n_samples)
    assert jnp.abs(jnp.std(noise0_a) - expected) < tol
    assert jnp.abs(jnp.std(noise0_b) - expected) < tol


def test_winner_reproducibility_under_same_key() -> None:
    """Given: two ``__init__`` calls with identical args including key.

    When: comparing every leaf of the resulting pytrees.
    Then: bit-equal. PRNG threading is deterministic.
    """
    key = jax.random.key(99)
    w1 = WINNER(in_dim=2, hidden_dim=32, num_hidden_layers=3, key=key, s0=1.5, s1=0.2)
    w2 = WINNER(in_dim=2, hidden_dim=32, num_hidden_layers=3, key=key, s0=1.5, s1=0.2)
    leaves1 = jax.tree_util.tree_leaves(eqx.filter(w1, eqx.is_array))
    leaves2 = jax.tree_util.tree_leaves(eqx.filter(w2, eqx.is_array))
    assert len(leaves1) == len(leaves2)
    for a, b in zip(leaves1, leaves2, strict=True):
        assert bool(jnp.all(a == b))


def test_winner_from_signal_matches_manual_call() -> None:
    """Given: a target signal + schedule.

    When: comparing ``WINNER.from_signal(signal, sched, key=k, ...)`` to
    a manual ``WINNER(s0=..., s1=..., key=k, ...)`` where the scales come
    from explicitly calling ``schedule.scales(spectral_centroid(...),
    n_ch)``.
    Then: bit-equal pytrees. Catches any drift in how ``from_signal``
    threads the key or computes the centroid.
    """
    key = jax.random.key(5)
    rng = np.random.default_rng(0)
    img = jnp.asarray(rng.standard_normal((32, 32, 3)).astype(np.float32))
    sched = WinnerSchedule.image()
    centroid = spectral_centroid(img)
    s0, s1 = sched.scales(centroid, img.shape[-1])

    w_factory = WINNER.from_signal(
        img,
        sched,
        in_dim=2,
        hidden_dim=16,
        num_hidden_layers=3,
        key=key,
    )
    w_manual = WINNER(
        in_dim=2,
        hidden_dim=16,
        num_hidden_layers=3,
        key=key,
        s0=s0,
        s1=s1,
    )

    leaves_f = jax.tree_util.tree_leaves(eqx.filter(w_factory, eqx.is_array))
    leaves_m = jax.tree_util.tree_leaves(eqx.filter(w_manual, eqx.is_array))
    assert len(leaves_f) == len(leaves_m)
    for a, b in zip(leaves_f, leaves_m, strict=True):
        assert bool(jnp.all(a == b))


def test_winner_reset_noise_rebuilds_not_double_perturbs() -> None:
    """Given: a WINNER ``w`` built with master key ``k0``.

    When: calling ``w.reset_noise(k0)`` and ``w.reset_noise(k1)``.
    Then:
    - ``w.reset_noise(k0)`` is bit-equal to ``w`` for layers 0 and 1 — the
      rebuild from the same key reproduces the original draw plus original
      noise. NOT a double-perturbation.
    - ``w.reset_noise(k1)`` has different layers 0 and 1 than ``w``, but
      layers ≥ 2 and the readout are bit-equal — reset_noise only touches
      the perturbed layers.

    Direct regression test for the double-perturbation trap pinned in
    ``WINNER_DECISIONS.md`` item 4.
    """
    k0 = jax.random.key(101)
    k1 = jax.random.key(202)
    w = WINNER(in_dim=2, hidden_dim=32, num_hidden_layers=4, key=k0, s0=2.0, s1=0.5)
    w_same = w.reset_noise(k0)
    w_new = w.reset_noise(k1)

    # Same-key rebuild matches original on layers 0/1
    assert bool(jnp.all(w.layers[0].W == w_same.layers[0].W))
    assert bool(jnp.all(w.layers[1].W == w_same.layers[1].W))
    assert bool(jnp.all(w.layers[0].b == w_same.layers[0].b))
    assert bool(jnp.all(w.layers[1].b == w_same.layers[1].b))

    # Fresh-key rebuild differs on layers 0/1
    assert bool(jnp.any(w.layers[0].W != w_new.layers[0].W))
    assert bool(jnp.any(w.layers[1].W != w_new.layers[1].W))

    # Layers >= 2 and readout untouched in both cases
    for i in range(2, len(w.layers)):
        assert bool(jnp.all(w.layers[i].W == w_same.layers[i].W))
        assert bool(jnp.all(w.layers[i].W == w_new.layers[i].W))
        assert bool(jnp.all(w.layers[i].b == w_same.layers[i].b))
        assert bool(jnp.all(w.layers[i].b == w_new.layers[i].b))
    assert bool(jnp.all(w.readout_W == w_same.readout_W))
    assert bool(jnp.all(w.readout_W == w_new.readout_W))
    assert bool(jnp.all(w.readout_b == w_same.readout_b))
    assert bool(jnp.all(w.readout_b == w_new.readout_b))


def test_winner_forward_scalar_shape() -> None:
    """Given: a default-out WINNER.

    When: forward-passing on a coord.
    Then: scalar output (0-d).
    """
    w = WINNER(in_dim=2, hidden_dim=16, num_hidden_layers=3, key=jax.random.key(0), s0=1.0, s1=0.1)
    y = w(jnp.array([0.1, -0.2]))
    assert y.shape == ()


def test_winner_vector_readout_shape() -> None:
    """Given: WINNER with ``out_features=4``.

    When: forward-passing on a coord.
    Then: output shape ``(4,)``. Locks the ``out_features`` forwarding
    path through ``__init__`` → ``_validate_body_args`` → ``_build_readout``
    for WINNER (mirrors the corresponding FINER / BACON test).
    """
    w = WINNER(
        in_dim=2,
        hidden_dim=16,
        num_hidden_layers=3,
        key=jax.random.key(0),
        s0=1.0,
        s1=0.1,
        out_features=4,
    )
    y = w(jnp.array([0.1, -0.2]))
    assert y.shape == (4,)


def test_winner_canonicalises_out_features_one() -> None:
    """Given: two WINNERs differing only in ``out_features`` (``None`` vs ``1``).

    When: comparing pytree structure.
    Then: identical. ``_validate_body_args`` canonicalises ``1`` to
    ``None`` so the two scalar-yielding constructions produce
    structurally identical pytrees — load-bearing for vmap / scan over
    the body and for downstream typing.
    """
    key = jax.random.key(0)
    a = WINNER(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=key, s0=1.0, s1=0.1, out_features=None)
    b = WINNER(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=key, s0=1.0, s1=0.1, out_features=1)
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)


def test_winner_rejects_one_hidden_layer() -> None:
    """Given: num_hidden_layers=1.

    When: constructing.
    Then: ``ValueError`` (not ``AssertionError``) fires — WINNER needs at
    least two hidden layers to perturb (layer 0 and layer 1).
    ``ValueError`` survives ``python -O`` (assertions would be stripped),
    matching the ``_check_film_shape`` precedent in
    ``ondes.basis._base``.
    """
    with pytest.raises(ValueError, match="num_hidden_layers >= 2"):
        WINNER(in_dim=2, hidden_dim=16, num_hidden_layers=1, key=jax.random.key(0), s0=1.0, s1=0.1)


# ---------------------------------------------------------------------------
# Paper fidelity (smoke)
# ---------------------------------------------------------------------------


def test_winner_layer1_preactivation_var_scales_with_s1_squared() -> None:
    """Given: WINNERs at varying ``s1`` with all other params fixed.

    When: pushing inputs ``x ~ U(-1, 1)^{d_in}`` through layer-0 sine then
    layer-1 linear, measuring ``Var[omega_hidden · pre1]`` (the actual sine
    argument).
    Then: variance is linear in ``s1²`` with slope ``d_h / 2``. That's the
    WINNER contribution to Theorem 3.1's prediction
    ``Var[ω · pre1] = C + d_h · s1² / 2``. The intercept ``C`` under
    ondes' SIREN init is ``≈ 1`` (not the paper's ``3``) — the discrepancy
    is documented in ``WINNER_DECISIONS.md`` item 7. The test asserts the
    slope only — that's the WINNER-specific load-bearing quantity.

    Clean derivation (matches ``WINNER_DECISIONS.md`` item 7):

    - Layer-0 sine output is uniform-phase so ``Var[h_j] ≈ 1/2``.
    - Layer-1 weight ``W_{ij} ~ U(-c, c) + N(0, (s1/ω)²)`` is i.i.d., so
      ``E[W_{ij}²] = c²/3 + (s1/ω)²``.
    - ``pre1_i = Σ_j W_{ij} · h_j`` sums ``d_h`` independent zero-mean
      terms, so ``Var[ω · pre1_i] = ω² · d_h · E[W²] · Var[h]
      = ω² · d_h · (c²/3 + (s1/ω)²) · (1/2)``.
    - With ``c² = 6 / (d_h · ω²)``, ``ω² · d_h · c² / 3 = 2``, so the
      deterministic part contributes ``1`` (= empirical intercept).
    - The noise part contributes ``d_h · s1² / 2`` (= empirical slope on
      ``s1²``).

    Tolerance: sample variance of a sum of ``d_h = 256`` near-Gaussian
    products over N = 50k samples has standard error ≈ var · sqrt(2/N).
    For ``y_max ≈ 130`` (at ``s1 = 1``) this is ``σ ≈ 0.8``. Propagating
    through ``np.polyfit`` with 6 well-spread design points gives a
    slope standard error of order ``2 σ / span(s1²) ≈ 1.5``, so the
    expected slope `d_h / 2 = 128` lives in a roughly ``±3`` 1σ band.
    We assert ``|slope - d_h/2| < 0.05 · (d_h/2) = 6.4`` (≈ 4σ, comfortable
    structural bound that catches any wrong-axis / wrong-divisor regression
    which would change the slope by tens of percent or flip its sign).
    """
    in_dim, hidden_dim, omega_h = 2, 256, 30.0
    n_HL = 4
    N = 50_000
    x = jax.random.uniform(jax.random.key(31), (N, in_dim), minval=-1.0, maxval=1.0)
    s1_vals = jnp.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    measured_vars = []
    for s1 in s1_vals:
        w = WINNER(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=n_HL,
            key=jax.random.key(42),
            s0=10.0,
            s1=float(s1),
            omega_first=30.0,
            omega_hidden=omega_h,
        )
        h0 = jax.vmap(w.layers[0])(x)
        pre1 = jax.vmap(lambda h, layer=w.layers[1]: layer.W @ h + layer.b)(h0)
        measured_vars.append(float(jnp.var(omega_h * pre1)))
    s1_sq = np.asarray(s1_vals) ** 2
    y = np.asarray(measured_vars)
    slope, intercept = np.polyfit(s1_sq, y, 1)
    expected_slope = hidden_dim / 2.0
    rel_tol = 0.05
    assert abs(slope - expected_slope) < rel_tol * expected_slope, (
        f"slope={slope:.2f}, expected {expected_slope:.2f}; intercept={intercept:.3f}, measured={measured_vars}"
    )
    # Sanity-check the slope's sign — guards against a regression that
    # makes variance shrink with s1 (which a sign-flip in the noise
    # standardisation would produce).
    assert slope > 0.0, f"non-positive slope {slope} suggests a sign-flip regression"


# ---------------------------------------------------------------------------
# Exploratory: end-to-end PSNR comparison vs SIREN
# ---------------------------------------------------------------------------


@pytest.mark.exploratory
def test_winner_vs_siren_high_freq_image() -> None:
    """Given: a synthetic 64×64 ``cos(20πx) · cos(20πy)`` target.

    When: fitting (K=5 seeds) three arms — vanilla SIREN, WINNER (image
    schedule), and a parameter-matched SIREN baseline (same arch as
    WINNER, since WINNER has identical param count).
    Then: log per-seed PSNR; assert structural ``median(WINNER) >
    median(SIREN)``. NO hardcoded dB threshold (per
    ``feedback_no_goodhart_falsifiers``).

    Pre-asserts: ``s0 > 1.0`` and ``s1 > 0.01`` for the chosen target — if
    either fails the target's centroid is too low for WINNER to materially
    differ from SIREN and the test result would be vacuous.

    Marked ``@pytest.mark.exploratory`` — runs post-convergence only, not
    in the per-round gate. Sized for laptop-CPU runtime (~1 min).
    """
    # Build a 64×64 high-frequency cosine target
    side = 64
    grid = jnp.linspace(-1.0, 1.0, side)
    xx, yy = jnp.meshgrid(grid, grid, indexing="xy")
    target = (jnp.cos(20.0 * jnp.pi * xx) * jnp.cos(20.0 * jnp.pi * yy))[:, :, None].astype(jnp.float32)
    coords = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
    values = target.reshape(-1)

    # Pre-assert non-vacuous scales for the image schedule
    sched = WinnerSchedule.image()
    centroid = spectral_centroid(target)
    s0_check, s1_check = sched.scales(centroid, target.shape[-1])
    assert float(s0_check) > 1.0, f"vacuous test: s0={float(s0_check)} too low"
    assert float(s1_check) > 0.01, f"vacuous test: s1={float(s1_check)} too low"

    n_HL, hidden_dim = 4, 128
    n_steps = 600
    lr = 1e-4

    def fit_one(model_factory, seed_key):
        model = model_factory(seed_key)

        @eqx.filter_jit
        def step(m, c, v):
            def loss_fn(mm):
                preds = jax.vmap(mm)(c)
                return jnp.mean((preds - v) ** 2)

            loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
            m_new = jax.tree_util.tree_map(
                lambda p, g: p - lr * g if eqx.is_array(p) and eqx.is_array(g) else p,
                m,
                grads,
            )
            return m_new, loss

        for _ in range(n_steps):
            model, _ = step(model, coords, values)
        preds = jax.vmap(model)(coords)
        mse = float(jnp.mean((preds - values) ** 2))
        psnr = 10.0 * np.log10((2.0**2) / max(mse, 1e-12))  # signal range = 2
        return psnr

    def siren_factory(k):
        return SIREN(
            in_dim=2,
            hidden_dim=hidden_dim,
            num_hidden_layers=n_HL,
            key=k,
            omega_first=30.0,
            omega_hidden=30.0,
        )

    def winner_factory(k):
        return WINNER.from_signal(
            target,
            sched,
            in_dim=2,
            hidden_dim=hidden_dim,
            num_hidden_layers=n_HL,
            key=k,
            omega_first=30.0,
            omega_hidden=30.0,
        )

    siren_psnrs = []
    winner_psnrs = []
    for seed in range(5):
        seed_key = jax.random.key(seed)
        siren_psnrs.append(fit_one(siren_factory, seed_key))
        winner_psnrs.append(fit_one(winner_factory, seed_key))

    print(f"SIREN PSNRs:  {siren_psnrs}")
    print(f"WINNER PSNRs: {winner_psnrs}")
    print(f"medians: SIREN={np.median(siren_psnrs):.2f} dB, WINNER={np.median(winner_psnrs):.2f} dB")

    assert np.median(winner_psnrs) > np.median(siren_psnrs), (
        f"structural invariant violated: median WINNER ({np.median(winner_psnrs):.2f}) "
        f"<= median SIREN ({np.median(siren_psnrs):.2f}) — high-frequency regime"
    )
