"""Fit a 2D image (real or synthetic) with an ondes INR.

A minimal end-to-end demo of the (trunk, head) composition pattern from the
README: build an ondes basis body, wrap it in a small `eqx.Module`, train with
optax. Run with zero args for a synthetic-sinusoid smoke fit:

    uv run python examples/fit_image.py

Or fit a real image:

    uv run python examples/fit_image.py --image cat.png --steps 2000 --grid 64
"""

from enum import StrEnum
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import typer
from PIL import Image

import ondes


app = typer.Typer(add_completion=False)


# Adam(1e-3) — Sitzmann+ 2020 uses 5e-5 for natural images; 1e-3 works for the
# small synthetic targets in this demo (256-1024 pixels, 200-500 steps).
# Switch to 5e-5 + ~5k steps when fitting megapixel natural images.


class BasisChoice(StrEnum):
    """Typer-friendly basis-kind choice.

    StrEnum so CLI input is validated automatically and `--basis foo` prints
    a clean error instead of raising KeyError. Compares equal to plain strings
    (tests pass `basis="siren"`); enum members are str instances.
    """

    SIREN = "siren"
    HSIREN = "hsiren"
    WIRE = "wire"


class SyntheticChoice(StrEnum):
    """Typer-friendly choice of named synthetic target."""

    SINUSOID = "sinusoid"
    GAUSSIAN_BUMP = "gaussian_bump"
    MANDELBROT = "mandelbrot"


_BASIS_CLASSES = {
    BasisChoice.SIREN: ondes.SIREN,
    BasisChoice.HSIREN: ondes.HSIREN,
    BasisChoice.WIRE: ondes.WIRE,
}


class Model(eqx.Module):
    """Match the README composition pattern: ondes body + user-owned wrapper.

    No head here — the body's scalar readout *is* the value-function output.
    A Gaussian-output variant would wrap `inr.trunk(coord)` with a
    `softplus(σ)` parameterisation (NeRF-W convention) or `exp(0.5·log_var)`
    (classical VAE convention); see DECISIONS.md §"Where head-shaped recipes live".
    """

    inr: ondes.Body

    def __call__(self, coord):
        """Forward pass: coord-of-shape-`(in_dim,)` → scalar amplitude."""
        return self.inr(coord)


def build_model(
    basis: BasisChoice,
    in_dim: int,
    hidden: int,
    layers: int,
    omega: float,
    *,
    key,
    s_init: float | None = None,
):
    """Construct one of {SIREN, HSIREN, WIRE} from the CLI's `--basis` flag.

    The string-to-class lookup is the *user-side* dispatch DECISIONS.md
    explicitly permits: ondes itself has no `kind=` discriminator; the CLI
    parses a string and picks the constructor.

    ``s_init`` is forwarded only to ``WIRE`` — SIREN/HSIREN don't accept it.
    Passing it for non-WIRE bases is silently ignored rather than raising,
    so a single CLI invocation can sweep basis without per-basis arg gymnastics.
    """
    cls = _BASIS_CLASSES[basis]
    # ω=30 — Sitzmann+ 2020 default for natural images. WIRE (Saragadam+ 2023)
    # uses ω=10 with σ=10 for natural images; pass --omega 10 --s-init 10
    # --basis wire to reproduce. H-SIREN (Cai & Pan 2024) uses ω=30 unchanged.
    kwargs = dict(
        in_dim=in_dim,
        hidden_dim=hidden,
        num_hidden_layers=layers,
        omega_first=omega,
        omega_hidden=omega,
        key=key,
    )
    if basis == BasisChoice.WIRE and s_init is not None:
        kwargs["s_init"] = s_init
    inr = cls(**kwargs)
    return Model(inr=inr)


def make_coords(*axes_sizes: int):
    """Build a regular grid in `[-1, 1]^len(axes_sizes)` and return `(prod, n_axes)` coords.

    Per-axis sizes are explicit so spatial axes (grid_n cells) and the channel
    axis (3 cells for RGB) can differ — see the value-function framing in
    DECISIONS.md.

    `[-1, 1]` is the Sitzmann+ 2020 convention; some INR papers use `[0, 1]`
    (matching pixel coordinates). The choice affects ω calibration — keep them
    consistent.
    """
    axes = [jnp.linspace(-1.0, 1.0, n) for n in axes_sizes]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    return jnp.stack([m.ravel() for m in mesh], axis=-1)


def synthetic_target(name: SyntheticChoice, grid_n: int):
    """Return `(coords, values)` for a named synthetic 2D target.

    Accepts the SyntheticChoice enum (via the CLI) or a bare string (via tests).
    Both forms compare equal because SyntheticChoice is `StrEnum`.
    """
    coords = make_coords(grid_n, grid_n)
    x, y = coords[:, 0], coords[:, 1]
    if name == SyntheticChoice.SINUSOID:
        values = jnp.sin(2.0 * jnp.pi * 3.0 * x) * jnp.cos(2.0 * jnp.pi * 3.0 * y)
    elif name == SyntheticChoice.GAUSSIAN_BUMP:
        values = jnp.exp(-5.0 * (x**2 + y**2))
    elif name == SyntheticChoice.MANDELBROT:
        # Escape-time count, normalised; harder than sinusoid (sharp boundary).
        c = x + 1j * y
        z = jnp.zeros_like(c)
        counts = jnp.zeros_like(x)
        for _ in range(32):
            z = z * z + c
            counts = counts + (jnp.abs(z) < 2.0).astype(jnp.float32)
        values = counts / 32.0
    else:
        raise ValueError(f"unknown synthetic target: {name!r}")
    return coords, values


def load_image(path: Path, grid_n: int):
    """Load a PNG/JPG and return `(coords, values)` in the value-function shape.

    RGB is treated as `(x, y, c) → amplitude`: channel is a coord, not an
    output dim (DECISIONS.md §"value-function framing"). Greyscale is
    `(x, y) → amplitude`.

    Palette ('P'), RGBA, and CMYK images are converted to RGB; 1-bit and 'L'
    images stay greyscale. Without explicit mode conversion, PIL returns
    palette indices or 4-channel arrays that don't match the value-function
    contract.
    """
    with Image.open(path) as pil_img:
        if pil_img.mode in ("RGBA", "CMYK", "P", "RGB"):
            pil_img = pil_img.convert("RGB")
        else:
            pil_img = pil_img.convert("L")
        # BILINEAR resampling — the PIL default (NEAREST) introduces aliasing
        # when downsampling natural images to small grids, which makes the
        # coord-to-value mapping unnecessarily noisy and harder for the INR
        # to fit. BILINEAR is the standard low-pass choice.
        img = (
            np.asarray(
                pil_img.resize((grid_n, grid_n), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            / 255.0
        )
    if img.ndim == 2:
        coords = make_coords(grid_n, grid_n)
        return coords, jnp.asarray(img.ravel())
    # RGB: (H, W, 3); coords are (grid_n, grid_n, 3) — channel axis size is 3,
    # not grid_n. The earlier `[linspace(-1,1,grid_n)] * in_dim` form produced
    # a grid_n × grid_n × grid_n cube which is structurally wrong for RGB.
    h, w, c = img.shape
    coords = make_coords(h, w, c)
    return coords, jnp.asarray(img.ravel())


def loss_fn(model, coords, target):
    """Mean-squared error between the INR's predictions and the target."""
    pred = jax.vmap(model)(coords)
    return jnp.mean((pred - target) ** 2)


def train(
    model,
    coords,
    target,
    *,
    steps: int,
    lr: float,
    chunk_size: int = 100,
    on_step=None,
    on_chunk=None,
):
    """Chunked Adam+scan training loop with two callback hooks.

    Returns `(trained_model, initial_loss, final_loss)`.

    - `on_step(step, loss)` — fires once per Adam step from Python, after the
      chunk that contains the step completes. The full chunk's loss history
      comes back from `scan` as an array, so we iterate it in Python and call
      `on_step` for each entry — per-step granularity without ever lowering a
      Python callback into the scan body. (This matters: `io_callback` lowers
      to `EmitPythonCallback`, which JAX only supports on CPU/CUDA/ROCm/TPU
      — not Metal/MPS via jax-mps.)
    - `on_chunk(step, loss, model)` — fires once per chunk from Python with
      the materialised model. Heavy artifacts (matplotlib, recon snapshots)
      belong here.

    The scan runs `chunk_size` steps as one XLA executable; the outer loop
    invokes it `(steps // chunk_size)` times. A remainder is silently dropped
    to keep the JIT cache stable (a final smaller chunk would recompile).
    Default `chunk_size=100` balances on_chunk artifact freshness against
    per-chunk dispatch overhead.
    """
    optimiser = optax.adam(lr)
    opt_state = optimiser.init(eqx.filter(model, eqx.is_inexact_array))
    jitted_loss = eqx.filter_jit(loss_fn)

    # Take model, opt_state, coords, target as explicit args so JAX caches the
    # compiled scan across calls with different array values — closure-capture
    # would mark them as static constants and force recompile.
    @eqx.filter_jit
    def run_chunk(model, opt_state, coords, target):
        def step(carry, _):
            m, o = carry
            loss, grads = eqx.filter_value_and_grad(loss_fn)(m, coords, target)
            updates, o = optimiser.update(grads, o, eqx.filter(m, eqx.is_inexact_array))
            m = eqx.apply_updates(m, updates)
            return (m, o), loss

        return jax.lax.scan(step, (model, opt_state), None, length=chunk_size)

    initial_loss = float(jitted_loss(model, coords, target))
    # Seed step-0 so the loss curve and CSV both have an honest starting point.
    # Skipping this leaves the curve starting at step=1 which hides the
    # "how much did training even help?" baseline.
    if on_step is not None:
        on_step(0, initial_loss)
    n_chunks = max(1, steps // chunk_size)
    for i in range(n_chunks):
        (model, opt_state), losses = run_chunk(model, opt_state, coords, target)
        # Materialise the chunk's per-step losses once and feed them to on_step.
        # Per-step cadence, single host transfer — much cheaper than io_callback.
        if on_step is not None:
            losses_np = np.asarray(losses)
            base = i * chunk_size
            for j, loss_val in enumerate(losses_np):
                on_step(base + j + 1, float(loss_val))
        if on_chunk is not None:
            # Re-evaluate loss outside the scan so on_chunk sees a value
            # consistent with the post-update model state (the in-scan losses
            # are pre-update). Negligible cost — one extra forward pass per chunk.
            chunk_loss = float(jitted_loss(model, coords, target))
            on_chunk(step=(i + 1) * chunk_size, loss=chunk_loss, model=model)
    final_loss = float(jitted_loss(model, coords, target))
    return model, initial_loss, final_loss


@eqx.filter_jit
def _vmap_model(model, coords):
    """Pure JIT'd vmap so cache hits across calls with different model/coords."""
    return jax.vmap(model)(coords)


def reconstruct(model, grid_n: int, in_dim: int):
    """Evaluate the trained model on a regular grid; returns the right-shaped array.

    For `in_dim == 2`: returns `(grid_n, grid_n)`.
    For `in_dim == 3` (RGB-as-coord): returns `(grid_n, grid_n, 3)` — channel
    axis size is 3, matching `load_image`'s coord-shape convention.
    """
    if in_dim == 2:
        coords = make_coords(grid_n, grid_n)
        return np.asarray(_vmap_model(model, coords)).reshape(grid_n, grid_n)
    if in_dim == 3:
        coords = make_coords(grid_n, grid_n, 3)
        return np.asarray(_vmap_model(model, coords)).reshape(grid_n, grid_n, 3)
    raise ValueError(f"unsupported in_dim for reconstruction: {in_dim}")


def _to_uint8_image(arr: np.ndarray, target) -> np.ndarray:
    """Rescale reconstruction by the *target's* range and quantise to uint8.

    Target range (not prediction range) is the ground truth — a non-negative
    target can produce tiny negative overshoots in `arr` from approximation
    noise, and rescaling on `arr.min() < 0` would dim the output spuriously.
    """
    t_min = float(np.asarray(target).min())
    t_max = float(np.asarray(target).max())
    if t_max > t_min:
        rescaled = (arr - t_min) / (t_max - t_min)
    else:
        rescaled = arr
    return (np.clip(rescaled, 0.0, 1.0) * 255).astype(np.uint8)


def _save_recon_png(model, grid_n: int, in_dim: int, target, path: Path) -> None:
    """Reconstruct on a grid and write a PNG. Used for snapshots + final output."""
    recon = reconstruct(model, grid_n, in_dim)
    Image.fromarray(_to_uint8_image(recon, target)).save(path)


def _write_evolution_gif(frame_paths: list[Path], gif_path: Path, *, ms_per_frame: int = 400) -> None:
    """Stitch per-snapshot PNGs into one animated GIF for at-a-glance convergence.

    PNGs remain the source-of-truth artifacts (lossless, scrubbable, side-by-side
    compareable). GIF is the convenience view — quantised palette is lossy but
    fine for a 256x256 thumbnail-grade animation. Rebuilt each snapshot from the
    accumulated frame list; with the default `snapshot_every=5` this is cheap.
    """
    if not frame_paths:
        return
    frames = [Image.open(p).convert("P", palette=Image.Palette.ADAPTIVE) for p in frame_paths]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=ms_per_frame,
        loop=0,
        optimize=True,
    )


def _write_loss_curve(history: list[tuple[int, float]], path: Path, *, title: str) -> None:
    """Render loss + PSNR vs. step as SVG (vector — crisp at any zoom).

    Two stacked subplots sharing the x-axis. PSNR is `-10*log10(MSE)` — strictly
    redundant with the loss, but the linear dB scale makes "how close to
    recognisable?" easier to read than "log MSE is -3.4". Two axes so neither
    metric has to share a misleading scale with the other.

    `title` describes the run (basis · resolution · arch · lr) and is fixed
    across re-renders. Live metrics (current step/loss/PSNR) go to a small
    figure-corner annotation — they're a readout, not a title.

    SVG over PNG: ~25 points of line plot is exactly the regime SVG dominates —
    smaller file, font-correct, infinite zoom. Quick Look refreshes SVGs same
    as PNGs on macOS, so the live-update story is unchanged.

    Lazy-imports matplotlib so test collection doesn't pay its import cost.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless — no Tk/GUI needed
    import matplotlib.pyplot as plt

    steps_, losses = zip(*history, strict=True)
    psnr_vals = [-10.0 * np.log10(max(loss, 1e-12)) for loss in losses]

    fig, (ax_loss, ax_psnr) = plt.subplots(2, 1, figsize=(6, 6), sharex=True, gridspec_kw={"hspace": 0.1})
    ax_loss.set_yscale("log")
    ax_loss.plot(steps_, losses, marker="o", markersize=3, color="tab:blue")
    ax_loss.set_ylabel("MSE (log)")
    ax_loss.grid(True, which="both", alpha=0.3)

    ax_psnr.plot(steps_, psnr_vals, marker="o", markersize=3, color="tab:orange")
    ax_psnr.set_xlabel("step")
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.grid(True, which="both", alpha=0.3)

    fig.suptitle(title)
    last_step, last_loss = history[-1]
    fig.text(
        0.99,
        0.01,
        f"step {last_step}  loss {last_loss:.4g}  PSNR {psnr_vals[-1]:.2f} dB",
        ha="right",
        va="bottom",
        fontsize=8,
        family="monospace",
        color="0.4",
    )
    # matplotlib picks the writer from the suffix; `.svg` → svg backend.
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


@app.command()
def main(
    image: Path | None = typer.Option(
        None,
        help="Path to image; if omitted, use --synthetic.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    synthetic: SyntheticChoice = typer.Option(SyntheticChoice.SINUSOID, help="Named synthetic target."),
    basis: BasisChoice = typer.Option(BasisChoice.SIREN, help="Basis kind."),
    hidden: int = typer.Option(64, help="Hidden dim."),
    layers: int = typer.Option(3, help="Number of hidden layers."),
    omega: float = typer.Option(30.0, help="ω init for both first and hidden layers."),
    s_init: float | None = typer.Option(
        None,
        "--s-init",
        help="WIRE-only: Gaussian envelope width init (paper σ=10 for natural images). Ignored for SIREN/HSIREN.",
    ),
    steps: int = typer.Option(500, help="Training steps."),
    lr: float = typer.Option(
        5e-4,
        help="Adam learning rate. SIREN image-fit regime per Sitzmann+ 2020 is "
        "1e-4 to 5e-4; small synthetic targets tolerate 1e-3.",
    ),
    grid: int = typer.Option(32, help="Grid resolution (synthetic) / resize target (image)."),
    output_dir: Path | None = typer.Option(
        None,
        help="Run directory for artifacts (config, loss curve, recon snapshots, csv). "
        "Defaults to `runs/{ISO-timestamp}/`.",
    ),
    chunk_size: int = typer.Option(
        100,
        help="Adam steps per JIT'd scan chunk. Sets the matplotlib re-render cadence "
        "(loss CSV updates per step via io_callback regardless). "
        "Total steps run = (steps // chunk_size) * chunk_size.",
    ),
    snapshot_every: int = typer.Option(
        1,
        help="Write a recon snapshot every N chunks (default 1 → every chunk; "
        "with chunk_size=100 and steps=2500, ~25 frames in the evolution GIF).",
    ),
    log_every: int = typer.Option(
        10,
        help="Console-print loss every N steps. CSV gets every step regardless.",
    ),
    seed: int = typer.Option(0, help="PRNG seed."),
):
    """Fit an image with an ondes INR and stream loss/recon artifacts to a run dir."""
    import csv
    import json
    from datetime import UTC, datetime

    if output_dir is None:
        # ISO-8601 with `:` replaced — colons break Windows + cause shell-quoting noise.
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
        output_dir = Path("runs") / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.key(seed)
    if image is not None:
        coords, target = load_image(image, grid)
        in_dim = int(coords.shape[-1])
    else:
        coords, target = synthetic_target(synthetic, grid)
        in_dim = 2

    # Save input alongside outputs so each run dir is self-contained — no need
    # to remember which photo produced which reconstruction.
    if in_dim == 2:
        input_arr = np.asarray(target).reshape(grid, grid)
    else:
        input_arr = np.asarray(target).reshape(grid, grid, 3)
    Image.fromarray(_to_uint8_image(input_arr, target)).save(output_dir / "input.png")

    # Config — every CLI knob that affects the run. Reproducibility hook.
    config = {
        "image": str(image) if image is not None else None,
        "synthetic": str(synthetic) if image is None else None,
        "basis": str(basis),
        "hidden": hidden,
        "layers": layers,
        "omega": omega,
        "s_init": s_init,
        "steps": steps,
        "lr": lr,
        "grid": grid,
        "chunk_size": chunk_size,
        "snapshot_every": snapshot_every,
        "log_every": log_every,
        "seed": seed,
        "in_dim": in_dim,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    model = build_model(basis, in_dim, hidden, layers, omega, key=key, s_init=s_init)

    csv_path = output_dir / "loss.csv"
    curve_path = output_dir / "loss_curve.svg"
    gif_path = output_dir / "recon_evolution.gif"
    snapshot_paths: list[Path] = []
    history: list[tuple[int, float]] = []

    # Descriptive run title for the loss-curve SVG. Fixed across re-renders —
    # live metrics go to a corner annotation, not the title.
    target_name = Path(image).name if image is not None else f"synthetic:{synthetic}"
    curve_title = (
        f"{basis} fit · {target_name} · {grid}x{grid} · hidden={hidden} layers={layers} ω={omega:g} · Adam({lr:g})"
    )

    # Open CSV once and write header; per-step appends share the handle. Faster
    # than re-opening per step, and `flush()` per row keeps `tail -f` live. The
    # `with` block guarantees the handle closes even if training raises.
    with csv_path.open("w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["step", "loss", "psnr_db"])

        def on_step(step: int, loss: float) -> None:
            # Per-step, fires from inside scan via io_callback (ordered). Stay cheap:
            # CSV append + history list + occasional console print. Matplotlib lives
            # in on_chunk because rendering blocks the JAX thread.
            psnr = -10.0 * np.log10(max(loss, 1e-12))
            history.append((step, loss))
            csv_writer.writerow([step, f"{loss:.6g}", f"{psnr:.4f}"])
            csv_file.flush()
            # Always log step 0 (the baseline) and step `steps` (the end); otherwise
            # gate to every `log_every` steps to avoid stdout flood at 2500 steps.
            if step == 0 or step == steps or step % log_every == 0:
                typer.echo(f"  step {step:>6d}  loss {loss:.6g}  PSNR {psnr:.2f} dB")

        def on_chunk(*, step: int, loss: float, model) -> None:
            # Per-chunk, fires from Python. Heavy work goes here.
            _write_loss_curve(history, curve_path, title=curve_title)
            chunk_idx = step // chunk_size
            if chunk_idx % snapshot_every == 0:
                snap_path = output_dir / f"recon_step_{step:06d}.png"
                _save_recon_png(model, grid, in_dim, target, snap_path)
                snapshot_paths.append(snap_path)
                # Rebuild GIF cumulatively so the evolution artifact is always
                # current — scrub-via-Finder uses the PNGs, at-a-glance uses the GIF.
                _write_evolution_gif(snapshot_paths, gif_path)

        # Seed the evolution GIF with the random-init reconstruction so frame 0
        # shows the baseline the optimiser starts from. Without this the GIF lies
        # about where training began.
        initial_snap = output_dir / "recon_step_000000.png"
        _save_recon_png(model, grid, in_dim, target, initial_snap)
        snapshot_paths.append(initial_snap)

        typer.echo(f"run dir: {output_dir}")
        typer.echo(f"training {steps} steps in chunks of {chunk_size} (per-step CSV, per-chunk plot)...")
        model, initial_loss, final_loss = train(
            model,
            coords,
            target,
            steps=steps,
            lr=lr,
            chunk_size=chunk_size,
            on_step=on_step,
            on_chunk=on_chunk,
        )
    # PSNR assumes amplitudes are in [0, 1] (images) or roughly so (synthetics in [-1, 1]).
    # For [-1, 1] targets MSE→PSNR uses peak=2; we report peak=1 PSNR consistently and
    # note the convention. Sitzmann+ 2020 reports peak=1 PSNR on normalised images.
    psnr = -10.0 * np.log10(max(final_loss, 1e-12))
    typer.echo(f"initial_loss={initial_loss:.6f}  final_loss={final_loss:.6f}  PSNR={psnr:.2f} dB")

    _save_recon_png(model, grid, in_dim, target, output_dir / "recon_final.png")
    typer.echo(f"artifacts written to {output_dir}")


if __name__ == "__main__":
    app()
