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
from examples.fit_image import Model, synthetic_target, train


@pytest.mark.parametrize("target_name", ["sinusoid", "gaussian_bump", "mandelbrot"])
def test_siren_fits_synthetic_target(target_name):
    # Given: one of the three synthetic targets, evaluated on a small 16x16
    # grid; a SIREN with the same defaults as the `siren` subcommand uses
    # (hidden=64, layers=3, ω=30); Adam at lr=1e-3 for 200 steps (the CLI
    # default is 500 — fewer here to keep CI fast).
    coords, target = synthetic_target(target_name, grid_n=16)
    key = jax.random.key(0)
    inr = ondes.SIREN(in_dim=2, hidden_dim=64, num_hidden_layers=3, omega_first=30.0, omega_hidden=30.0, key=key)
    model = Model(inr=inr)

    # When: we train via the same `train(...)` helper the CLI invokes.
    _, initial_loss, final_loss = train(model, coords, target, steps=200, lr=1e-3)

    # Then: final loss is at most 30% of initial. Loose smoke threshold —
    # catches "training is wired wrong" without claiming convergence.
    assert final_loss < 0.3 * initial_loss, (
        f"{target_name}: expected final_loss < 0.3 * initial_loss, "
        f"got initial={initial_loss:.6f}, final={final_loss:.6f}, "
        f"ratio={final_loss / initial_loss:.3f}"
    )
