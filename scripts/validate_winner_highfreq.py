"""Validate WINNER vs SIREN on a synthetic high-frequency target.

Companion to ``scripts/validate_winner_astronaut.py``: same harness
(hidden=128, layers=4, ω=30, lr=5e-4, steps=1000, grid=256, mlx
backend) but the target is the synthetic
``cos(200π·x)·cos(200π·y)`` instead of the astronaut RGB image. The
synthetic target has centroid ≈ 0.229 vs astronaut's 0.043, which
through ``WinnerSchedule.image()`` produces s0 ≈ 15.9 (vs 3.5) and
s1 ≈ 0.031 (vs 0.006) — roughly 5× the perturbation magnitude. The
test is whether the WINNER-vs-SIREN PSNR gap widens materially in the
higher-centroid regime, as the Theorem 3.1 slope-on-s1² prediction
implies.

Run::

    .venv-mlx/bin/python -u scripts/validate_winner_highfreq.py

The ``-u`` flag forces unbuffered stdout so per-line progress prints
land in the redirected log as the run proceeds (the astronaut script
also wires this in-process via ``sys.stdout.reconfigure``).

Outputs land under ``validation_highfreq/`` at the repo root.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image

import ondes


sys.stdout.reconfigure(line_buffering=True)


# ---------------------------------------------------------------------------
# Target generation
# ---------------------------------------------------------------------------


def make_target(grid_n: int, n_pi: float = 200.0) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build ``cos(n_pi·π·x)·cos(n_pi·π·y)`` on a ``grid_n × grid_n`` grid.

    Returns ``(coords, target_flat, image_3d)`` to match the astronaut
    script's layout: ``coords`` is ``(grid_n², in_dim=3)`` with the RGB-
    as-coord convention, ``target_flat`` is the raveled image with the
    same channel structure (broadcast to 3 identical channels so the
    pipeline doesn't change), ``image_3d`` is the ``(H, W, 3)`` form used
    as input to ``ondes.spectral_centroid``.
    """
    xs = jnp.linspace(0.0, 1.0, grid_n)
    xx, yy = jnp.meshgrid(xs, xs, indexing="ij")
    base = jnp.cos(n_pi * jnp.pi * xx) * jnp.cos(n_pi * jnp.pi * yy)
    # broadcast to RGB (3 identical channels) — the harness uses in_dim=3
    image_3d = jnp.stack([base, base, base], axis=-1).astype(jnp.float32)
    # Coords: regular grid in [-1, 1] across (H, W, C); channel axis size is 3.
    h, w, c = image_3d.shape
    cxs = jnp.linspace(-1.0, 1.0, h)
    cys = jnp.linspace(-1.0, 1.0, w)
    ccs = jnp.linspace(-1.0, 1.0, c)
    grid_x, grid_y, grid_c = jnp.meshgrid(cxs, cys, ccs, indexing="ij")
    coords = jnp.stack([grid_x, grid_y, grid_c], axis=-1).reshape(-1, 3)
    target_flat = image_3d.ravel()
    return coords, target_flat, image_3d


# ---------------------------------------------------------------------------
# Training loop (identical to astronaut script)
# ---------------------------------------------------------------------------


def loss_fn(model, coords, target):
    """Mean-squared error between model predictions and the target."""
    pred = jax.vmap(model)(coords)
    return jnp.mean((pred - target) ** 2)


def make_step(optimizer):
    """Build a single jitted Adam step for ``(model, opt_state, coords, target)``."""

    @eqx.filter_jit
    def step(model, opt_state, coords, target):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, coords, target)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    return step


def train_one(model, coords, target, *, steps: int, lr: float, log_every: int = 100):
    """Train ``model`` for ``steps`` Adam steps; return ``(model, psnr_curve)``."""
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    step = make_step(optimizer)

    psnr_curve = []
    for s in range(1, steps + 1):
        model, opt_state, loss = step(model, opt_state, coords, target)
        if s % log_every == 0 or s == 1 or s == steps:
            loss_f = float(loss)
            psnr = -10.0 * np.log10(max(loss_f, 1e-12))
            psnr_curve.append((s, psnr))
    return model, psnr_curve


def final_psnr(model, coords, target) -> float:
    """Compute PSNR (dB) of the model's predictions against the target."""
    pred = jax.vmap(model)(coords)
    mse = float(jnp.mean((pred - target) ** 2))
    return -10.0 * np.log10(max(mse, 1e-12))


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def reconstruct_rgb(model, grid_n: int) -> np.ndarray:
    """Return a ``(grid_n, grid_n, 3)`` float reconstruction."""
    xs = jnp.linspace(-1.0, 1.0, grid_n)
    cs = jnp.linspace(-1.0, 1.0, 3)
    grid_x, grid_y, grid_c = jnp.meshgrid(xs, xs, cs, indexing="ij")
    coords = jnp.stack([grid_x, grid_y, grid_c], axis=-1).reshape(-1, 3)
    pred = np.asarray(jax.vmap(model)(coords)).reshape(grid_n, grid_n, 3)
    return pred


def save_recon_png(model, grid_n: int, target_image_2d: np.ndarray, path: Path) -> None:
    """Reconstruct on a regular grid, rescale to the target's range, write PNG."""
    recon = reconstruct_rgb(model, grid_n)
    t_min, t_max = float(target_image_2d.min()), float(target_image_2d.max())
    scaled = (recon - t_min) / (t_max - t_min) if t_max > t_min else recon
    img = (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


# ---------------------------------------------------------------------------
# Wrapper module (mirror examples/fit_image.py:Model)
# ---------------------------------------------------------------------------


class Model(eqx.Module):
    """Minimal wrapper matching ``examples/fit_image.py``'s composition pattern."""

    inr: ondes.Body

    def __call__(self, coord):
        """Forward pass: coord of shape ``(in_dim,)`` → scalar amplitude."""
        return self.inr(coord)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(out_dir: Path, *, n_seeds: int, steps: int, n_pi: float):
    """Run the SIREN vs WINNER validation on the synthetic high-freq target."""
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_n = 256
    hidden = 128
    n_layers = 4
    omega = 30.0
    lr = 5e-4
    log_every = 100

    print("=" * 70)
    print(f"Validation: WINNER vs SIREN on cos({n_pi}π·x)·cos({n_pi}π·y)")
    print(f"arch: hidden={hidden}, layers={n_layers}, ω={omega}, lr={lr}")
    print(f"budget: steps={steps}, grid={grid_n}, n_seeds={n_seeds}")
    print(f"out_dir: {out_dir}")
    print("=" * 70)

    coords, target_flat, image_3d = make_target(grid_n, n_pi=n_pi)
    print(f"target shape: {tuple(image_3d.shape)}, dtype={image_3d.dtype}")

    sched = ondes.WinnerSchedule.image()
    centroid = ondes.spectral_centroid(image_3d)
    s0, s1 = sched.scales(centroid, image_3d.shape[-1])
    print(f"WinnerSchedule.image: centroid={float(centroid):.4f}, s0={float(s0):.4f}, s1={float(s1):.4f}")
    print()

    siren_psnrs: list[float] = []
    winner_psnrs: list[float] = []
    siren_curve_seed0: list[tuple[int, float]] = []
    winner_curve_seed0: list[tuple[int, float]] = []
    total_start = time.perf_counter()

    for seed in range(n_seeds):
        key = jax.random.key(seed)

        # SIREN arm
        t0 = time.perf_counter()
        siren_inr = ondes.SIREN(
            in_dim=3,
            hidden_dim=hidden,
            num_hidden_layers=n_layers,
            omega_first=omega,
            omega_hidden=omega,
            key=key,
        )
        siren_model, siren_curve = train_one(
            Model(inr=siren_inr),
            coords,
            target_flat,
            steps=steps,
            lr=lr,
            log_every=log_every,
        )
        siren_psnr = final_psnr(siren_model, coords, target_flat)
        t_siren = time.perf_counter() - t0
        siren_psnrs.append(siren_psnr)
        if seed == 0:
            siren_curve_seed0 = siren_curve
            save_recon_png(siren_model, grid_n, np.asarray(image_3d), out_dir / "recon_siren_seed0.png")
        print(f"seed {seed}: SIREN  PSNR={siren_psnr:6.2f} dB  ({t_siren:5.1f}s)")

        # WINNER arm
        t0 = time.perf_counter()
        winner_inr = ondes.WINNER.from_signal(
            image_3d,
            sched,
            in_dim=3,
            hidden_dim=hidden,
            num_hidden_layers=n_layers,
            omega_first=omega,
            omega_hidden=omega,
            key=key,
        )
        winner_model, winner_curve = train_one(
            Model(inr=winner_inr),
            coords,
            target_flat,
            steps=steps,
            lr=lr,
            log_every=log_every,
        )
        winner_psnr = final_psnr(winner_model, coords, target_flat)
        t_winner = time.perf_counter() - t0
        winner_psnrs.append(winner_psnr)
        if seed == 0:
            winner_curve_seed0 = winner_curve
            save_recon_png(winner_model, grid_n, np.asarray(image_3d), out_dir / "recon_winner_seed0.png")
        print(f"seed {seed}: WINNER PSNR={winner_psnr:6.2f} dB  ({t_winner:5.1f}s)")
        print()

    total_wallclock = time.perf_counter() - total_start

    print("=" * 70)
    print(f"{'seed':<6} {'SIREN (dB)':<14} {'WINNER (dB)':<14} {'Δ':<10}")
    print("-" * 70)
    for s, (a, b) in enumerate(zip(siren_psnrs, winner_psnrs, strict=True)):
        print(f"{s:<6} {a:<14.2f} {b:<14.2f} {b - a:<+10.2f}")
    print("-" * 70)
    print(
        f"{'median':<6} {np.median(siren_psnrs):<14.2f} "
        f"{np.median(winner_psnrs):<14.2f} "
        f"{np.median(winner_psnrs) - np.median(siren_psnrs):<+10.2f}"
    )
    print(
        f"{'mean':<6} {np.mean(siren_psnrs):<14.2f} "
        f"{np.mean(winner_psnrs):<14.2f} "
        f"{np.mean(winner_psnrs) - np.mean(siren_psnrs):<+10.2f}"
    )
    print(f"{'std':<6} {np.std(siren_psnrs):<14.4f} {np.std(winner_psnrs):<14.4f}")
    print("=" * 70)
    print(f"total wall clock: {total_wallclock:.1f}s ({total_wallclock / 60:.1f} min)")

    # Write seed-0 PSNR-vs-step curve as CSV
    csv_path = out_dir / "psnr_vs_step_seed0.csv"
    with csv_path.open("w") as f:
        f.write("step,siren_psnr,winner_psnr\n")
        for (s_step, s_psnr), (w_step, w_psnr) in zip(siren_curve_seed0, winner_curve_seed0, strict=True):
            assert s_step == w_step, f"step mismatch: {s_step} vs {w_step}"
            f.write(f"{s_step},{s_psnr:.4f},{w_psnr:.4f}\n")
    print(f"seed-0 PSNR curve: {csv_path}")

    summary_path = out_dir / "summary.txt"
    with summary_path.open("w") as f:
        f.write(f"target: cos({n_pi}π·x)·cos({n_pi}π·y), grid {grid_n}×{grid_n}\n")
        f.write(f"arch: hidden={hidden} layers={n_layers} omega={omega} lr={lr} steps={steps} grid={grid_n}\n")
        f.write(
            f"schedule: WinnerSchedule.image() centroid={float(centroid):.6f} s0={float(s0):.4f} s1={float(s1):.4f}\n"
        )
        f.write(f"n_seeds: {n_seeds}\n\n")
        f.write("per-seed PSNR (dB):\n")
        for s, (a, b) in enumerate(zip(siren_psnrs, winner_psnrs, strict=True)):
            f.write(f"  seed {s}: SIREN={a:.2f}  WINNER={b:.2f}  delta={b - a:+.2f}\n")
        f.write(
            f"\nSIREN  median={np.median(siren_psnrs):.2f} "
            f"mean={np.mean(siren_psnrs):.2f} std={np.std(siren_psnrs):.4f}\n"
        )
        f.write(
            f"WINNER median={np.median(winner_psnrs):.2f} "
            f"mean={np.mean(winner_psnrs):.2f} std={np.std(winner_psnrs):.4f}\n"
        )
        f.write(f"\nmedian delta: {np.median(winner_psnrs) - np.median(siren_psnrs):+.2f} dB\n")
        f.write(f"total wallclock: {total_wallclock:.1f}s\n")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent.parent / "validation_highfreq",
    )
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--n-pi",
        type=float,
        default=200.0,
        help="Frequency multiplier for the cos product target. Default 200.0 → centroid ≈ 0.229.",
    )
    args = parser.parse_args()
    main(args.out_dir, n_seeds=args.n_seeds, steps=args.steps, n_pi=args.n_pi)
