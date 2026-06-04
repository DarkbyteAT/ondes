"""Smoke tests: SIREN fits the three synthetic targets used by the example script.

These tests share ``synthetic_target`` and ``train`` with ``examples/fit_image.py``
so a CI regression in either catches the other. They DON'T go through the CLI:
the CLI is per-basis Typer subcommands (no shared ``build_model`` helper —
that pattern would re-introduce the discriminator dispatch DECISIONS.md
forbids). Instead we construct ``ondes.SIREN`` + the example's ``Model`` wrapper
directly, mirroring exactly what the ``siren`` subcommand does internally.

The synthetic targets exercise mixed frequencies, smooth bumps, and sharp
escape boundaries, which is what SIREN-class INRs claim to dominate. We skip
the ``--image`` code path (file I/O).

Optax/typer/pillow are dev-only deps (per DECISIONS.md: ondes library code
has no training-stack or CLI deps; examples + tests may use anything).
"""

import jax
import pytest

import ondes
from examples.fit_image import Model, loss_fn, synthetic_target, train


@pytest.mark.parametrize("target_name", ["sinusoid", "gaussian_bump", "mandelbrot"])
def test_siren_fits_synthetic_target(target_name: str) -> None:
    # Given: one of the three synthetic targets, evaluated on a small 16x16
    # grid; a SIREN with the same defaults as the `siren` subcommand uses
    # (hidden=64, layers=3, ω=30); Adam at lr=1e-3 for 200 steps (the CLI
    # default is 500 — fewer here to keep CI fast).
    coords, target = synthetic_target(target_name, grid_n=16)
    key = jax.random.key(0)
    inr = ondes.SIREN(in_dim=2, hidden_dim=64, num_hidden_layers=3, omega_first=30.0, omega_hidden=30.0, key=key)
    model = Model(inr=inr)

    # When: we train via the same `train(...)` helper the CLI invokes.
    # `train` returns (model, initial_loss, final_loss, chunk_times); we
    # don't assert on per-chunk timing here (system-load-sensitive), so the
    # last tuple element is discarded.
    _, initial_loss, final_loss, _ = train(model, coords, target, steps=200, lr=1e-3)

    # Then: final loss is at most 30% of initial. Loose smoke threshold —
    # catches "training is wired wrong" without claiming convergence.
    assert final_loss < 0.3 * initial_loss, (
        f"{target_name}: expected final_loss < 0.3 * initial_loss, "
        f"got initial={initial_loss:.6f}, final={final_loss:.6f}, "
        f"ratio={final_loss / initial_loss:.3f}"
    )


def test_train_on_step_labels_match_parameter_state() -> None:
    # Given: a tiny SIREN, the sinusoid target, and a 2-step train at chunk_size=2
    # so the entire run is one scan + one final-loss forward pass. The captured
    # on_step calls should be exactly:
    #   step 0  → loss(initial_model)       (seeded before the loop)
    #   step 1  → loss(model after 1 update) (pre-update for j=1 in the scan)
    #   step 2  → loss(trained model)        (post-update final forward pass)
    #
    # The previous label scheme reported losses[j] as step `base + j + 1`,
    # which meant step 1 carried the loss(initial_model) value (a duplicate
    # of step 0) and step `steps` was never emitted. This test pins the
    # corrected alignment so the bug can't silently regress.
    coords, target = synthetic_target("sinusoid", grid_n=8)
    key = jax.random.key(0)
    initial = Model(
        inr=ondes.SIREN(in_dim=2, hidden_dim=8, num_hidden_layers=2, omega_first=30.0, omega_hidden=30.0, key=key)
    )

    captured: list[tuple[int, float]] = []

    def capture(step: int, loss: float) -> None:
        captured.append((step, loss))

    # When: we train for 2 steps with chunk_size=2 (one scan + one final eval).
    # The trailing `_` absorbs the per-chunk-times list train() also returns;
    # this test pins on_step label alignment, not timing.
    trained, _, _, _ = train(initial, coords, target, steps=2, lr=1e-3, chunk_size=2, on_step=capture)

    # Then: exactly three on_step entries, at steps 0, 1, 2.
    assert [s for s, _ in captured] == [0, 1, 2], f"expected steps [0, 1, 2], got {[s for s, _ in captured]}"

    step0_loss = dict(captured)[0]
    step1_loss = dict(captured)[1]
    step2_loss = dict(captured)[2]

    # The label at step 0 must equal the loss on the *initial* model — the seed
    # entry, not a scan-returned value. The label at step `steps` must equal
    # the loss on the *returned* model — the final forward pass after the loop.
    expected_step0 = float(loss_fn(initial, coords, target))
    expected_step2 = float(loss_fn(trained, coords, target))
    # Tight equality (no atol/rtol) — these are the same float computation, not
    # just close enough. A drift here means the labels stopped aligning.
    assert step0_loss == expected_step0, (
        f"step 0 should equal loss(initial_model): got {step0_loss} vs {expected_step0}"
    )
    assert step2_loss == expected_step2, (
        f"step 2 should equal loss(trained_model): got {step2_loss} vs {expected_step2}"
    )

    # The label at step 1 must NOT duplicate step 0 — the old off-by-one bug
    # would emit step1_loss == loss(initial_model) and never emit step2_loss
    # at all. Since training descends on this seed/target, step 1's pre-update
    # loss is for the model after one Adam step, strictly less than the initial
    # loss and strictly greater than the post-two-step loss.
    assert step1_loss != step0_loss, (
        f"step 1 must not duplicate step 0 — that's the regressed off-by-one bug. "
        f"step0={step0_loss}, step1={step1_loss}"
    )
    assert step0_loss > step1_loss > step2_loss, (
        f"expected step0 > step1 > step2 (descending training), "
        f"got step0={step0_loss}, step1={step1_loss}, step2={step2_loss}"
    )


def test_train_rejects_steps_not_divisible_by_chunk_size() -> None:
    # Given: a tiny model and a request for 250 steps in chunks of 100.
    # The scan loop would run (steps // chunk_size) * chunk_size = 200 Adam
    # steps and then emit on_step(steps=250, final_loss), mislabelling the
    # actual post-training state. The precondition check inside train() must
    # fail-fast with a clear error rather than silently under-train.
    coords, target = synthetic_target("sinusoid", grid_n=8)
    key = jax.random.key(0)
    model = Model(
        inr=ondes.SIREN(in_dim=2, hidden_dim=8, num_hidden_layers=2, omega_first=30.0, omega_hidden=30.0, key=key)
    )

    # When/Then: the call raises ValueError with a message naming both numbers.
    with pytest.raises(ValueError, match=r"250.*100") as excinfo:
        train(model, coords, target, steps=250, lr=1e-3, chunk_size=100)
    # Sanity-check the message carries both values explicitly (so the user
    # sees what they passed, not a generic "bad arguments").
    msg = str(excinfo.value)
    assert "250" in msg and "100" in msg, f"expected both 250 and 100 in error message, got: {msg!r}"


def test_train_rejects_non_positive_steps() -> None:
    # Given: a tiny model and a `steps=0` (or negative) request. The naive
    # `n_chunks = steps // chunk_size` would skip the loop, but the post-loop
    # `on_step(steps, final_loss)` would still fire — emitting a duplicate of
    # the step-0 seed and mislabelling the absence of training as "step 0
    # twice". Fail-fast in `train()` rather than silently degenerate.
    coords, target = synthetic_target("sinusoid", grid_n=8)
    key = jax.random.key(0)
    model = Model(
        inr=ondes.SIREN(in_dim=2, hidden_dim=8, num_hidden_layers=2, omega_first=30.0, omega_hidden=30.0, key=key)
    )

    # When/Then: zero steps raises with a clear "must be positive" message.
    with pytest.raises(ValueError, match=r"steps must be positive") as excinfo_zero:
        train(model, coords, target, steps=0, lr=1e-3, chunk_size=10)
    assert "0" in str(excinfo_zero.value)

    # And negative steps fails the same gate (the implementation uses `steps <= 0`
    # so the negative-int case isn't an off-by-one trap).
    with pytest.raises(ValueError, match=r"steps must be positive"):
        train(model, coords, target, steps=-5, lr=1e-3, chunk_size=10)
