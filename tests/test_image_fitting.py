"""Smoke tests: the `examples/fit_image.py` CLI fits its synthetic targets.

These tests share `build_model` and `train` with the CLI — the script and the
tests both go through the same construction + training path, so a CI regression
in either catches the other. We skip the `--image` code path (file I/O); the
synthetic targets exercise mixed frequencies, smooth bumps, and sharp escape
boundaries, which is what SIREN-class INRs claim to dominate.

Optax/typer/pillow are dev-only deps (per DECISIONS.md: ondes library code
has no training-stack or CLI deps; examples + tests may use anything).
"""

import jax
import pytest

from examples.fit_image import build_model, synthetic_target, train


@pytest.mark.parametrize("target_name", ["sinusoid", "gaussian_bump", "mandelbrot"])
def test_siren_fits_synthetic_target(target_name):
    # Given: one of the three synthetic targets the CLI supports, evaluated on
    # a small 16x16 grid; a SIREN with the same defaults as the CLI uses
    # (--basis siren --hidden 64 --layers 3 --omega 30); Adam at lr=1e-3 for
    # 200 steps (the CLI default is 500 — fewer here to keep CI fast).
    coords, target = synthetic_target(target_name, grid_n=16)
    key = jax.random.key(0)
    model = build_model(basis="siren", in_dim=2, hidden=64, layers=3, omega=30.0, key=key)

    # When: we train via the same `train(...)` helper the CLI invokes.
    _, initial_loss, final_loss = train(model, coords, target, steps=200, lr=1e-3)

    # Then: final loss is at most 30% of initial. Loose smoke threshold —
    # catches "training is wired wrong" without claiming convergence.
    assert final_loss < 0.3 * initial_loss, (
        f"{target_name}: expected final_loss < 0.3 * initial_loss, "
        f"got initial={initial_loss:.6f}, final={final_loss:.6f}, "
        f"ratio={final_loss / initial_loss:.3f}"
    )
