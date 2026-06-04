"""Fit a 2D image (real or synthetic) with an ondes INR.

A minimal end-to-end demo of the (trunk, head) composition pattern from the
README: build an ondes basis body, wrap it in a small `eqx.Module`, train with
optax. One Typer subcommand per basis — each owns the kwargs that *that*
basis takes, no shared discriminator. Pick a subcommand to pick a basis:

    uv run python examples/fit_image.py synthetic --target sinusoid    # SIREN smoke fit
    uv run python examples/fit_image.py siren --image cat.png --steps 2000 --grid 64
    uv run python examples/fit_image.py wire --image cat.png --omega 10 --s-init 10
    uv run python examples/fit_image.py rff  --image cat.png --sigma 10 --num-freqs 256
    uv run python examples/fit_image.py bacon --image cat.png --max-freq 256
    uv run python examples/fit_image.py finer --image cat.png
    uv run python examples/fit_image.py fourier-mfn --image cat.png
    uv run python examples/fit_image.py gabor-mfn   --image cat.png
    uv run python examples/fit_image.py pnf         --image cat.png

The subcommand layout mirrors `ondes`'s public surface: every basis is its
own class with its own constructor kwargs, never a `kind=` string or shared
dict-of-classes (see DECISIONS.md §"Structural design — polymorphism over
discriminators"). The CLI follows the library's discipline so the example
doesn't model a pattern the library refuses to ship.
"""

import time
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import typer
from jaxtyping import Array, Float
from PIL import Image

import ondes


app = typer.Typer(add_completion=False, no_args_is_help=True)


class SyntheticChoice(StrEnum):
    """Typer-friendly choice of named synthetic target."""

    SINUSOID = "sinusoid"
    GAUSSIAN_BUMP = "gaussian_bump"
    MANDELBROT = "mandelbrot"


class Model(eqx.Module):
    """Match the README composition pattern: ondes body + user-owned wrapper.

    No head here — the body's scalar readout *is* the value-function output.
    A Gaussian-output variant would wrap `inr.trunk(coord)` with a
    `softplus(σ)` parameterisation (NeRF-W convention) or `exp(0.5·log_var)`
    (classical VAE convention); see DECISIONS.md §"Where head-shaped recipes live".
    """

    inr: ondes.Body

    def __call__(self, coord: Float[Array, "in"]) -> Float[Array, ""]:
        """Forward pass: coord-of-shape-`(in_dim,)` → scalar amplitude."""
        return self.inr(coord)


def make_coords(*axes_sizes: int) -> Float[Array, "n_points n_axes"]:
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


def synthetic_target(
    name: SyntheticChoice | str, grid_n: int
) -> tuple[Float[Array, "n_points 2"], Float[Array, "n_points"]]:
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


def load_image(path: Path, grid_n: int) -> tuple[Float[Array, "n_points in_dim"], Float[Array, "n_points"]]:
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


def loss_fn(
    model: Model,
    coords: Float[Array, "n_points in"],
    target: Float[Array, "n_points"],
) -> Float[Array, ""]:
    """Mean-squared error between the INR's predictions and the target."""
    pred = jax.vmap(model)(coords)
    return jnp.mean((pred - target) ** 2)


def train(
    model: Model,
    coords: Float[Array, "n_points in"],
    target: Float[Array, "n_points"],
    *,
    steps: int,
    lr: float,
    chunk_size: int = 100,
    on_step: Callable[[int, float], None] | None = None,
    on_chunk: Callable[..., None] | None = None,
) -> tuple[Model, float, float, list[float]]:
    """Chunked Adam+scan training loop with two callback hooks.

    Returns `(trained_model, initial_loss, final_loss, chunk_times)`. The
    `chunk_times` list holds the per-chunk wall-clock in seconds (one entry
    per scan invocation). `chunk_times[0]` includes JIT-compile + first-
    chunk execution; `sum(chunk_times[1:])` is steady-state cost across
    the remaining chunks. Callers split these to report compile-vs-steady
    timing separately. Each entry is measured with `time.perf_counter()`
    around the `run_chunk` call, after `jax.block_until_ready(...)` forces
    XLA's async dispatch to complete — otherwise the timer would only
    capture queue submission, not actual compute.

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
    invokes it `(steps // chunk_size)` times. Default `chunk_size=100`
    balances on_chunk artifact freshness against per-chunk dispatch
    overhead.

    Raises `ValueError` if ``steps <= 0`` (a zero-step request would emit
    the post-loop ``on_step(steps, final_loss)`` as a duplicate of the
    step-0 seed) or if ``steps`` is not divisible by ``chunk_size``
    (silent under-training would mislabel the final post-loop step entry:
    we'd train ``(steps // chunk_size) * chunk_size`` Adam steps but emit
    ``on_step(steps, final_loss)`` claiming the original ``steps`` count).

    Step-label convention: the scan body computes ``loss`` *before*
    ``apply_updates``, so ``losses[j]`` is the loss at the *pre-update*
    parameters for that iteration — i.e. the loss at step ``base + j``,
    not ``base + j + 1``. We therefore relabel scan-returned losses as
    step ``base + j`` and skip ``j == 0`` of chunk 0 because the
    ``initial_loss`` seed at step 0 already covers it. After the loop
    we make one extra forward pass on the trained model and report it
    as step ``steps`` so the post-final-update loss actually lands in
    CSV / curve artifacts. Total entry count is unchanged
    (``steps + 1``) — every label now points at the matching parameter
    state.
    """
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if steps % chunk_size != 0:
        raise ValueError(f"steps ({steps}) must be a multiple of chunk_size ({chunk_size})")
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
    # Both preconditions above guarantee steps >= chunk_size with steps a
    # positive multiple of chunk_size, so the floor-div is correct without
    # a max-clamp — the clamp would have hidden a `steps=0` request behind
    # a silent one-chunk run.
    n_chunks = steps // chunk_size
    chunk_times: list[float] = []
    for i in range(n_chunks):
        t0 = time.perf_counter()
        (model, opt_state), losses = run_chunk(model, opt_state, coords, target)
        # `block_until_ready` forces XLA's async dispatch to complete before
        # we stop the timer. Without it `chunk_times[0]` would only measure
        # queue submission + Python-side overhead, not the actual JIT
        # compile + first-chunk execution we want to attribute to compile
        # cost. Block on both leaves to be safe — `opt_state` is the
        # cheaper of the two (Adam moments) so the wait is dominated by
        # the model leaves either way.
        jax.block_until_ready((model, opt_state))
        chunk_times.append(time.perf_counter() - t0)
        # Materialise the chunk's per-step losses once and feed them to on_step.
        # Per-step cadence, single host transfer — much cheaper than io_callback.
        # `losses[j]` is the pre-update loss at step `base + j` (see docstring).
        # The `base + j > 0` guard skips the very first sample so we don't
        # re-emit step 0 — the `initial_loss` seed above already reported it.
        if on_step is not None:
            losses_np = np.asarray(losses)
            base = i * chunk_size
            for j, loss_val in enumerate(losses_np):
                step = base + j
                if step > 0:
                    on_step(step, float(loss_val))
        if on_chunk is not None:
            # Re-evaluate loss outside the scan so on_chunk sees a value
            # consistent with the post-update model state (the in-scan losses
            # are pre-update). Negligible cost — one extra forward pass per chunk.
            chunk_loss = float(jitted_loss(model, coords, target))
            on_chunk(step=(i + 1) * chunk_size, loss=chunk_loss, model=model)
    final_loss = float(jitted_loss(model, coords, target))
    # Explicit final-step entry: without this the post-update loss at step
    # `steps` is never recorded — the scan only yields pre-update losses, so
    # the model state that ends training would be invisible in CSV / curve.
    if on_step is not None:
        on_step(steps, final_loss)
    return model, initial_loss, final_loss, chunk_times


@eqx.filter_jit
def _vmap_model(model: Model, coords: Float[Array, "n_points in"]) -> Float[Array, "n_points"]:
    """Pure JIT'd vmap so cache hits across calls with different model/coords."""
    return jax.vmap(model)(coords)


def reconstruct(model: Model, grid_n: int, in_dim: int) -> np.ndarray:
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


def _to_uint8_image(arr: np.ndarray, target: Float[Array, "..."]) -> np.ndarray:
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


def _save_recon_png(model: Model, grid_n: int, in_dim: int, target: Float[Array, "..."], path: Path) -> None:
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


def _train_and_save(
    *,
    model: Model,
    basis_label: str,
    image: Path | None,
    synthetic_choice: SyntheticChoice | None,
    in_dim: int,
    coords: Float[Array, "n_points in"],
    target: Float[Array, "n_points"],
    hidden: int,
    layers: int,
    steps: int,
    lr: float,
    grid: int,
    chunk_size: int,
    snapshot_every: int,
    log_every: int,
    output_dir: Path | None,
    seed: int,
    basis_extras: dict[str, Any],
) -> None:
    """Run the training loop and write all per-run artifacts to ``output_dir``.

    Shared body of every per-basis subcommand. Owns:

    - run-directory resolution (timestamped default under ``runs/``)
    - input PNG dump (so each run dir is self-contained)
    - ``config.json`` (every knob that affected the run, including the
      basis-specific ``basis_extras`` so the run is reproducible from the
      file alone)
    - the with-managed CSV handle + per-step / per-chunk callbacks
    - snapshot PNGs + cumulative evolution GIF + final loss SVG
    - the end-of-run reconstruction PNG and console summary

    Each subcommand constructs its `Model` from the basis kwargs that
    *that basis* takes, then passes the coords/target plus all artifact
    knobs here. The helper has no awareness of which basis was picked
    beyond ``basis_label`` (used in the loss-curve title and config.json).
    """
    import csv
    import json
    from datetime import UTC, datetime

    if output_dir is None:
        # ISO-8601 with `:` replaced — colons break Windows + cause shell-quoting noise.
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
        output_dir = Path("runs") / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save input alongside outputs so each run dir is self-contained — no need
    # to remember which photo produced which reconstruction.
    if in_dim == 2:
        input_arr = np.asarray(target).reshape(grid, grid)
    else:
        input_arr = np.asarray(target).reshape(grid, grid, 3)
    Image.fromarray(_to_uint8_image(input_arr, target)).save(output_dir / "input.png")

    # Config — every CLI knob that affects the run. Reproducibility hook.
    config: dict[str, Any] = {
        "basis": basis_label,
        "image": str(image) if image is not None else None,
        "synthetic": str(synthetic_choice) if synthetic_choice is not None else None,
        "hidden": hidden,
        "layers": layers,
        "steps": steps,
        "lr": lr,
        "grid": grid,
        "chunk_size": chunk_size,
        "snapshot_every": snapshot_every,
        "log_every": log_every,
        "seed": seed,
        "in_dim": in_dim,
        # Basis-specific kwargs land in their own sub-dict so config.json
        # readers can pick out "what made this run different" without
        # colliding key names across bases (e.g. siren `omega` vs rff `sigma`).
        "basis_kwargs": basis_extras,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    csv_path = output_dir / "loss.csv"
    curve_path = output_dir / "loss_curve.svg"
    gif_path = output_dir / "recon_evolution.gif"
    snapshot_paths: list[Path] = []
    history: list[tuple[int, float]] = []

    # Descriptive run title for the loss-curve SVG. Fixed across re-renders —
    # live metrics go to a corner annotation, not the title.
    target_name = Path(image).name if image is not None else f"synthetic:{synthetic_choice}"
    curve_title = f"{basis_label} fit · {target_name} · {grid}x{grid} · hidden={hidden} layers={layers} · Adam({lr:g})"

    # Open CSV once and write header; per-step appends share the handle. Faster
    # than re-opening per step, and `flush()` per row keeps `tail -f` live. The
    # `with` block guarantees the handle closes even if training raises.
    with csv_path.open("w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["step", "loss", "psnr_db"])

        def on_step(step: int, loss: float) -> None:
            # Per-step, fires from Python after each chunk's losses come back
            # from scan. Stay cheap: CSV append + history list + occasional
            # console print + log-cadence flush. Matplotlib lives in on_chunk
            # because rendering blocks the JAX thread.
            psnr = -10.0 * np.log10(max(loss, 1e-12))
            history.append((step, loss))
            csv_writer.writerow([step, f"{loss:.6g}", f"{psnr:.4f}"])
            # Flush + console echo share the same `log_every` gate. The
            # per-step CSV row is written every iteration (so the file is
            # complete on a clean exit), but fsync happens on the same
            # cadence the user sees the console heartbeat — `tail -f` and
            # stdout stay in lockstep, and we drop ~99% of the per-step
            # fsyncs that the unconditional flush used to do. The CSV
            # handle's own `with`-managed close still flushes at exit, so
            # nothing is lost on the final partial cadence.
            if step == 0 or step == steps or step % log_every == 0:
                csv_file.flush()
                typer.echo(f"  step {step:>6d}  loss {loss:.6g}  PSNR {psnr:.2f} dB")

        def on_chunk(*, step: int, loss: float, model: Model) -> None:
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
        model, initial_loss, final_loss, chunk_times = train(
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

    # Split compile vs steady-state wall-clock and write `timing.json`. The
    # first chunk pays the JIT-compile cost on top of its `chunk_size` Adam
    # steps; the remaining `n_chunks - 1` chunks are steady-state cost we
    # can extrapolate per-step from. Reporting both separately lets the
    # writeup's per-step inferences cite the steady number while the
    # methodology section names the compile overhead honestly.
    compile_s = chunk_times[0] if chunk_times else 0.0
    steady_s = sum(chunk_times[1:]) if len(chunk_times) > 1 else 0.0
    total_s = sum(chunk_times)
    timing = {
        "compile_s": compile_s,
        "steady_s": steady_s,
        "total_s": total_s,
        "n_chunks": len(chunk_times),
        "chunk_size": chunk_size,
        "steady_steps": max(0, (len(chunk_times) - 1) * chunk_size),
    }
    (output_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    typer.echo(
        f"timing: compile_s={compile_s:.2f}  steady_s={steady_s:.2f}  "
        f"total_s={total_s:.2f}  (n_chunks={len(chunk_times)})"
    )

    _save_recon_png(model, grid, in_dim, target, output_dir / "recon_final.png")
    typer.echo(f"artifacts written to {output_dir}")


def _load_target(
    image: Path | None, synthetic_choice: SyntheticChoice | None, grid: int
) -> tuple[Float[Array, "n_points in"], Float[Array, "n_points"], int]:
    """Return ``(coords, target, in_dim)`` for an image path or synthetic target.

    Exactly one of ``image`` / ``synthetic_choice`` must be set. Per-basis
    subcommands take ``--image`` (only image fits make sense for non-SIREN
    bases in this example); the dedicated ``synthetic`` subcommand passes
    a ``SyntheticChoice``.
    """
    if image is not None:
        coords, target = load_image(image, grid)
        return coords, target, int(coords.shape[-1])
    if synthetic_choice is not None:
        coords, target = synthetic_target(synthetic_choice, grid)
        return coords, target, 2
    raise typer.BadParameter("provide either --image or pick the `synthetic` subcommand.")


# -- Per-basis subcommands ------------------------------------------------- #
#
# One subcommand per basis class. Each owns exactly the kwargs that *that*
# basis takes (no shared ω where there is none, no s_init silently dropped).
# Construction goes straight through ``ondes.{Basis}(...)`` — no enum, no
# discriminator dict. The library exposes the kinds via separate classes;
# the CLI mirrors that.


# Tiny helper for the shared `--image` option — Typer needs an explicit
# `typer.Option(...)` per parameter, so this factory produces a fresh sentinel
# for each subcommand instead of trying to share one across signatures.


def _image_option() -> Any:
    return typer.Option(
        ...,
        help="Path to image to fit.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    )


@app.command()
def siren(
    image: Path = _image_option(),
    hidden: int = typer.Option(64, help="Hidden dim."),
    layers: int = typer.Option(3, help="Number of hidden layers."),
    omega: float = typer.Option(30.0, help="ω init for both first and hidden layers (Sitzmann+ 2020)."),
    steps: int = typer.Option(500, help="Training steps."),
    lr: float = typer.Option(5e-4, help="Adam learning rate (Sitzmann+ 2020 regime: 1e-4 to 5e-4)."),
    grid: int = typer.Option(32, help="Image resize target."),
    output_dir: Path | None = typer.Option(None, help="Run dir for artifacts. Defaults to runs/{ISO-timestamp}/."),
    chunk_size: int = typer.Option(100, help="Adam steps per JIT'd scan chunk."),
    snapshot_every: int = typer.Option(1, help="Write a recon snapshot every N chunks."),
    log_every: int = typer.Option(10, help="Console-print loss every N steps."),
    seed: int = typer.Option(0, help="PRNG seed."),
) -> None:
    """Fit an image with SIREN (Sitzmann+ 2020)."""
    key = jax.random.key(seed)
    coords, target, in_dim = _load_target(image, None, grid)
    inr = ondes.SIREN(
        in_dim=in_dim,
        hidden_dim=hidden,
        num_hidden_layers=layers,
        omega_first=omega,
        omega_hidden=omega,
        key=key,
    )
    _train_and_save(
        model=Model(inr=inr),
        basis_label="siren",
        image=image,
        synthetic_choice=None,
        in_dim=in_dim,
        coords=coords,
        target=target,
        hidden=hidden,
        layers=layers,
        steps=steps,
        lr=lr,
        grid=grid,
        chunk_size=chunk_size,
        snapshot_every=snapshot_every,
        log_every=log_every,
        output_dir=output_dir,
        seed=seed,
        basis_extras={"omega": omega},
    )


@app.command()
def hsiren(
    image: Path = _image_option(),
    hidden: int = typer.Option(64, help="Hidden dim."),
    layers: int = typer.Option(3, help="Number of hidden layers."),
    omega: float = typer.Option(30.0, help="ω init for both first and hidden layers (Cai & Pan 2024 default)."),
    steps: int = typer.Option(500, help="Training steps."),
    lr: float = typer.Option(5e-4, help="Adam learning rate (SIREN regime)."),
    grid: int = typer.Option(32, help="Image resize target."),
    output_dir: Path | None = typer.Option(None, help="Run dir for artifacts. Defaults to runs/{ISO-timestamp}/."),
    chunk_size: int = typer.Option(100, help="Adam steps per JIT'd scan chunk."),
    snapshot_every: int = typer.Option(1, help="Write a recon snapshot every N chunks."),
    log_every: int = typer.Option(10, help="Console-print loss every N steps."),
    seed: int = typer.Option(0, help="PRNG seed."),
) -> None:
    """Fit an image with H-SIREN (Cai & Pan 2024)."""
    key = jax.random.key(seed)
    coords, target, in_dim = _load_target(image, None, grid)
    inr = ondes.HSIREN(
        in_dim=in_dim,
        hidden_dim=hidden,
        num_hidden_layers=layers,
        omega_first=omega,
        omega_hidden=omega,
        key=key,
    )
    _train_and_save(
        model=Model(inr=inr),
        basis_label="hsiren",
        image=image,
        synthetic_choice=None,
        in_dim=in_dim,
        coords=coords,
        target=target,
        hidden=hidden,
        layers=layers,
        steps=steps,
        lr=lr,
        grid=grid,
        chunk_size=chunk_size,
        snapshot_every=snapshot_every,
        log_every=log_every,
        output_dir=output_dir,
        seed=seed,
        basis_extras={"omega": omega},
    )


@app.command()
def wire(
    image: Path = _image_option(),
    hidden: int = typer.Option(64, help="Hidden dim."),
    layers: int = typer.Option(3, help="Number of hidden layers."),
    omega: float = typer.Option(10.0, help="ω init for both first and hidden layers (Saragadam+ 2023 default)."),
    s_init: float = typer.Option(
        10.0, "--s-init", help="Gaussian-envelope width init (paper σ=10 for natural images)."
    ),
    steps: int = typer.Option(500, help="Training steps."),
    lr: float = typer.Option(1e-3, help="Adam learning rate (WIRE codebase default)."),
    grid: int = typer.Option(32, help="Image resize target."),
    output_dir: Path | None = typer.Option(None, help="Run dir for artifacts. Defaults to runs/{ISO-timestamp}/."),
    chunk_size: int = typer.Option(100, help="Adam steps per JIT'd scan chunk."),
    snapshot_every: int = typer.Option(1, help="Write a recon snapshot every N chunks."),
    log_every: int = typer.Option(10, help="Console-print loss every N steps."),
    seed: int = typer.Option(0, help="PRNG seed."),
) -> None:
    """Fit an image with WIRE (Saragadam+ 2023)."""
    key = jax.random.key(seed)
    coords, target, in_dim = _load_target(image, None, grid)
    inr = ondes.WIRE(
        in_dim=in_dim,
        hidden_dim=hidden,
        num_hidden_layers=layers,
        omega_first=omega,
        omega_hidden=omega,
        s_init=s_init,
        key=key,
    )
    _train_and_save(
        model=Model(inr=inr),
        basis_label="wire",
        image=image,
        synthetic_choice=None,
        in_dim=in_dim,
        coords=coords,
        target=target,
        hidden=hidden,
        layers=layers,
        steps=steps,
        lr=lr,
        grid=grid,
        chunk_size=chunk_size,
        snapshot_every=snapshot_every,
        log_every=log_every,
        output_dir=output_dir,
        seed=seed,
        basis_extras={"omega": omega, "s_init": s_init},
    )


@app.command()
def finer(
    image: Path = _image_option(),
    hidden: int = typer.Option(64, help="Hidden dim."),
    layers: int = typer.Option(3, help="Number of hidden layers."),
    omega: float = typer.Option(30.0, help="ω init for the first layer (paper preserves omega_hidden=1.0)."),
    first_bias_scale: float = typer.Option(5.0, help="Uniform bound for the first layer's bias (Liu+ 2024 default)."),
    steps: int = typer.Option(500, help="Training steps."),
    lr: float = typer.Option(5e-4, help="Adam learning rate (FINER image-fit regime)."),
    grid: int = typer.Option(32, help="Image resize target."),
    output_dir: Path | None = typer.Option(None, help="Run dir for artifacts. Defaults to runs/{ISO-timestamp}/."),
    chunk_size: int = typer.Option(100, help="Adam steps per JIT'd scan chunk."),
    snapshot_every: int = typer.Option(1, help="Write a recon snapshot every N chunks."),
    log_every: int = typer.Option(10, help="Console-print loss every N steps."),
    seed: int = typer.Option(0, help="PRNG seed."),
) -> None:
    """Fit an image with FINER (Liu+ 2024)."""
    key = jax.random.key(seed)
    coords, target, in_dim = _load_target(image, None, grid)
    inr = ondes.FINER(
        in_dim=in_dim,
        hidden_dim=hidden,
        num_hidden_layers=layers,
        omega_first=omega,
        first_bias_scale=first_bias_scale,
        key=key,
    )
    _train_and_save(
        model=Model(inr=inr),
        basis_label="finer",
        image=image,
        synthetic_choice=None,
        in_dim=in_dim,
        coords=coords,
        target=target,
        hidden=hidden,
        layers=layers,
        steps=steps,
        lr=lr,
        grid=grid,
        chunk_size=chunk_size,
        snapshot_every=snapshot_every,
        log_every=log_every,
        output_dir=output_dir,
        seed=seed,
        basis_extras={"omega": omega, "first_bias_scale": first_bias_scale},
    )


@app.command()
def rff(
    image: Path = _image_option(),
    hidden: int = typer.Option(64, help="Hidden dim."),
    layers: int = typer.Option(3, help="Number of hidden ReLU layers."),
    sigma: float = typer.Option(
        10.0,
        help="Bandwidth of the Gaussian-RFF projection (Tancik+ 2020 default for natural images).",
    ),
    num_freqs: int = typer.Option(256, help="Number of sampled Fourier frequencies."),
    steps: int = typer.Option(500, help="Training steps."),
    lr: float = typer.Option(1e-4, help="Adam learning rate (Tancik+ 2020 default)."),
    grid: int = typer.Option(32, help="Image resize target."),
    output_dir: Path | None = typer.Option(None, help="Run dir for artifacts. Defaults to runs/{ISO-timestamp}/."),
    chunk_size: int = typer.Option(100, help="Adam steps per JIT'd scan chunk."),
    snapshot_every: int = typer.Option(1, help="Write a recon snapshot every N chunks."),
    log_every: int = typer.Option(10, help="Console-print loss every N steps."),
    seed: int = typer.Option(0, help="PRNG seed."),
) -> None:
    """Fit an image with Random Fourier Features + ReLU MLP (Tancik+ 2020)."""
    key = jax.random.key(seed)
    coords, target, in_dim = _load_target(image, None, grid)
    inr = ondes.RFF(
        in_dim=in_dim,
        hidden_dim=hidden,
        num_hidden_layers=layers,
        sigma=sigma,
        num_freqs=num_freqs,
        key=key,
    )
    _train_and_save(
        model=Model(inr=inr),
        basis_label="rff",
        image=image,
        synthetic_choice=None,
        in_dim=in_dim,
        coords=coords,
        target=target,
        hidden=hidden,
        layers=layers,
        steps=steps,
        lr=lr,
        grid=grid,
        chunk_size=chunk_size,
        snapshot_every=snapshot_every,
        log_every=log_every,
        output_dir=output_dir,
        seed=seed,
        basis_extras={"sigma": sigma, "num_freqs": num_freqs},
    )


@app.command()
def bacon(
    image: Path = _image_option(),
    hidden: int = typer.Option(64, help="Hidden dim."),
    layers: int = typer.Option(3, help="Number of recurrence steps."),
    max_freq: float = typer.Option(
        256.0,
        help="Target overall output bandwidth (cycles/coord unit; Lindell+ 2022 default).",
    ),
    quantization_interval: float = typer.Option(
        2.0 * float(jnp.pi),
        "--quant",
        help="Discrete-frequency grid spacing (paper default 2π).",
    ),
    steps: int = typer.Option(500, help="Training steps."),
    lr: float = typer.Option(1e-3, help="Adam learning rate (BACON image-fit demo default)."),
    grid: int = typer.Option(32, help="Image resize target."),
    output_dir: Path | None = typer.Option(None, help="Run dir for artifacts. Defaults to runs/{ISO-timestamp}/."),
    chunk_size: int = typer.Option(100, help="Adam steps per JIT'd scan chunk."),
    snapshot_every: int = typer.Option(1, help="Write a recon snapshot every N chunks."),
    log_every: int = typer.Option(10, help="Console-print loss every N steps."),
    seed: int = typer.Option(0, help="PRNG seed."),
) -> None:
    """Fit an image with BACON (Lindell+ 2022)."""
    key = jax.random.key(seed)
    coords, target, in_dim = _load_target(image, None, grid)
    inr = ondes.BACON(
        in_dim=in_dim,
        hidden_dim=hidden,
        num_hidden_layers=layers,
        max_freq=max_freq,
        quantization_interval=quantization_interval,
        key=key,
    )
    _train_and_save(
        model=Model(inr=inr),
        basis_label="bacon",
        image=image,
        synthetic_choice=None,
        in_dim=in_dim,
        coords=coords,
        target=target,
        hidden=hidden,
        layers=layers,
        steps=steps,
        lr=lr,
        grid=grid,
        chunk_size=chunk_size,
        snapshot_every=snapshot_every,
        log_every=log_every,
        output_dir=output_dir,
        seed=seed,
        basis_extras={"max_freq": max_freq, "quantization_interval": quantization_interval},
    )


@app.command(name="fourier-mfn")
def fourier_mfn(
    image: Path = _image_option(),
    hidden: int = typer.Option(64, help="Hidden dim."),
    layers: int = typer.Option(3, help="Number of recurrence steps."),
    input_scale: float = typer.Option(256.0, help="Filter-frequency uniform-init scale (Fathony+ 2021 default)."),
    weight_scale: float = typer.Option(1.0, help="Recurrence-linear uniform-init scale (paper default)."),
    steps: int = typer.Option(500, help="Training steps."),
    lr: float = typer.Option(1e-3, help="Adam learning rate (MFN image-fit default)."),
    grid: int = typer.Option(32, help="Image resize target."),
    output_dir: Path | None = typer.Option(None, help="Run dir for artifacts. Defaults to runs/{ISO-timestamp}/."),
    chunk_size: int = typer.Option(100, help="Adam steps per JIT'd scan chunk."),
    snapshot_every: int = typer.Option(1, help="Write a recon snapshot every N chunks."),
    log_every: int = typer.Option(10, help="Console-print loss every N steps."),
    seed: int = typer.Option(0, help="PRNG seed."),
) -> None:
    """Fit an image with Fourier MFN (Fathony+ 2021)."""
    key = jax.random.key(seed)
    coords, target, in_dim = _load_target(image, None, grid)
    inr = ondes.FourierMFN(
        in_dim=in_dim,
        hidden_dim=hidden,
        num_hidden_layers=layers,
        input_scale=input_scale,
        weight_scale=weight_scale,
        key=key,
    )
    _train_and_save(
        model=Model(inr=inr),
        basis_label="fourier-mfn",
        image=image,
        synthetic_choice=None,
        in_dim=in_dim,
        coords=coords,
        target=target,
        hidden=hidden,
        layers=layers,
        steps=steps,
        lr=lr,
        grid=grid,
        chunk_size=chunk_size,
        snapshot_every=snapshot_every,
        log_every=log_every,
        output_dir=output_dir,
        seed=seed,
        basis_extras={"input_scale": input_scale, "weight_scale": weight_scale},
    )


@app.command(name="gabor-mfn")
def gabor_mfn(
    image: Path = _image_option(),
    hidden: int = typer.Option(64, help="Hidden dim."),
    layers: int = typer.Option(3, help="Number of recurrence steps."),
    alpha: float = typer.Option(6.0, help="Gamma-prior shape on filter scales (Fathony+ 2021 default)."),
    beta: float = typer.Option(1.0, help="Gamma-prior rate on filter scales (paper default)."),
    weight_scale: float = typer.Option(1.0, help="Recurrence-linear uniform-init scale (paper default)."),
    steps: int = typer.Option(500, help="Training steps."),
    lr: float = typer.Option(1e-3, help="Adam learning rate (MFN image-fit default)."),
    grid: int = typer.Option(32, help="Image resize target."),
    output_dir: Path | None = typer.Option(None, help="Run dir for artifacts. Defaults to runs/{ISO-timestamp}/."),
    chunk_size: int = typer.Option(100, help="Adam steps per JIT'd scan chunk."),
    snapshot_every: int = typer.Option(1, help="Write a recon snapshot every N chunks."),
    log_every: int = typer.Option(10, help="Console-print loss every N steps."),
    seed: int = typer.Option(0, help="PRNG seed."),
) -> None:
    """Fit an image with Gabor MFN (Fathony+ 2021)."""
    key = jax.random.key(seed)
    coords, target, in_dim = _load_target(image, None, grid)
    inr = ondes.GaborMFN(
        in_dim=in_dim,
        hidden_dim=hidden,
        num_hidden_layers=layers,
        alpha=alpha,
        beta=beta,
        weight_scale=weight_scale,
        key=key,
    )
    _train_and_save(
        model=Model(inr=inr),
        basis_label="gabor-mfn",
        image=image,
        synthetic_choice=None,
        in_dim=in_dim,
        coords=coords,
        target=target,
        hidden=hidden,
        layers=layers,
        steps=steps,
        lr=lr,
        grid=grid,
        chunk_size=chunk_size,
        snapshot_every=snapshot_every,
        log_every=log_every,
        output_dir=output_dir,
        seed=seed,
        basis_extras={"alpha": alpha, "beta": beta, "weight_scale": weight_scale},
    )


@app.command()
def pnf(
    image: Path = _image_option(),
    hidden: int = typer.Option(64, help="Hidden dim."),
    layers: int = typer.Option(3, help="Number of recurrence steps."),
    input_scale: float = typer.Option(256.0, help="Filter-frequency uniform-init scale (matches MFN)."),
    weight_scale: float = typer.Option(1.0, help="Recurrence-linear + mix-layer uniform-init scale (paper default)."),
    steps: int = typer.Option(500, help="Training steps."),
    lr: float = typer.Option(1e-3, help="Adam learning rate (Yang+ 2022 default)."),
    grid: int = typer.Option(32, help="Image resize target."),
    output_dir: Path | None = typer.Option(None, help="Run dir for artifacts. Defaults to runs/{ISO-timestamp}/."),
    chunk_size: int = typer.Option(100, help="Adam steps per JIT'd scan chunk."),
    snapshot_every: int = typer.Option(1, help="Write a recon snapshot every N chunks."),
    log_every: int = typer.Option(10, help="Console-print loss every N steps."),
    seed: int = typer.Option(0, help="PRNG seed."),
) -> None:
    """Fit an image with PNF (Yang+ 2022)."""
    key = jax.random.key(seed)
    coords, target, in_dim = _load_target(image, None, grid)
    inr = ondes.PNF(
        in_dim=in_dim,
        hidden_dim=hidden,
        num_hidden_layers=layers,
        input_scale=input_scale,
        weight_scale=weight_scale,
        key=key,
    )
    _train_and_save(
        model=Model(inr=inr),
        basis_label="pnf",
        image=image,
        synthetic_choice=None,
        in_dim=in_dim,
        coords=coords,
        target=target,
        hidden=hidden,
        layers=layers,
        steps=steps,
        lr=lr,
        grid=grid,
        chunk_size=chunk_size,
        snapshot_every=snapshot_every,
        log_every=log_every,
        output_dir=output_dir,
        seed=seed,
        basis_extras={"input_scale": input_scale, "weight_scale": weight_scale},
    )


@app.command()
def synthetic(
    target: SyntheticChoice = typer.Option(SyntheticChoice.SINUSOID, help="Named synthetic target."),
    hidden: int = typer.Option(64, help="Hidden dim."),
    layers: int = typer.Option(3, help="Number of hidden layers."),
    omega: float = typer.Option(30.0, help="SIREN ω init."),
    steps: int = typer.Option(200, help="Training steps (low default — synthetics fit fast)."),
    lr: float = typer.Option(1e-3, help="Adam learning rate (synthetic targets tolerate 1e-3)."),
    grid: int = typer.Option(32, help="Grid resolution."),
    output_dir: Path | None = typer.Option(None, help="Run dir for artifacts. Defaults to runs/{ISO-timestamp}/."),
    chunk_size: int = typer.Option(100, help="Adam steps per JIT'd scan chunk."),
    snapshot_every: int = typer.Option(1, help="Write a recon snapshot every N chunks."),
    log_every: int = typer.Option(10, help="Console-print loss every N steps."),
    seed: int = typer.Option(0, help="PRNG seed."),
) -> None:
    """Fit a named synthetic target (sinusoid / gaussian_bump / mandelbrot) with SIREN.

    A self-contained smoke fit — no external image needed. Uses SIREN because
    it's the basis the synthetic-target tests pin to; pick a different basis
    by running the corresponding subcommand against an image instead.
    """
    key = jax.random.key(seed)
    coords, target_arr, in_dim = _load_target(None, target, grid)
    inr = ondes.SIREN(
        in_dim=in_dim,
        hidden_dim=hidden,
        num_hidden_layers=layers,
        omega_first=omega,
        omega_hidden=omega,
        key=key,
    )
    _train_and_save(
        model=Model(inr=inr),
        basis_label="siren",
        image=None,
        synthetic_choice=target,
        in_dim=in_dim,
        coords=coords,
        target=target_arr,
        hidden=hidden,
        layers=layers,
        steps=steps,
        lr=lr,
        grid=grid,
        chunk_size=chunk_size,
        snapshot_every=snapshot_every,
        log_every=log_every,
        output_dir=output_dir,
        seed=seed,
        basis_extras={"omega": omega, "synthetic_target": str(target)},
    )


if __name__ == "__main__":
    app()
