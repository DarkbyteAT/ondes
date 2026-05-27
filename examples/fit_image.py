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
from jaxtyping import Array, Float, Key
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

    def __call__(self, coord: Float[Array, "in"]) -> jax.Array:
        """Forward pass: coord-of-shape-`(in_dim,)` → scalar amplitude."""
        return self.inr(coord)


def build_model(
    basis: BasisChoice,
    in_dim: int,
    hidden: int,
    layers: int,
    omega: float,
    *,
    key: Key[Array, ""],
) -> "Model":
    """Construct one of {SIREN, HSIREN, WIRE} from the CLI's `--basis` flag.

    The string-to-class lookup is the *user-side* dispatch DECISIONS.md
    explicitly permits: ondes itself has no `kind=` discriminator; the CLI
    parses a string and picks the constructor.
    """
    cls = _BASIS_CLASSES[basis]
    inr = cls(
        in_dim=in_dim,
        hidden_dim=hidden,
        num_hidden_layers=layers,
        # ω=30 — Sitzmann+ 2020 default for natural images. WIRE (Saragadam+ 2023)
        # uses ω=10 with σ=10 for natural images; pass --omega 10 + --basis wire
        # to reproduce. H-SIREN (Cai & Pan 2024) uses ω=30 unchanged.
        omega_first=omega,
        omega_hidden=omega,
        key=key,
    )
    return Model(inr=inr)


def make_coords(*axes_sizes: int) -> Float[Array, "prod n_axes"]:
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
    name: SyntheticChoice | str,
    grid_n: int,
) -> tuple[Float[Array, "n in_dim"], Float[Array, "n"]]:
    """Return `(coords, values)` for a named synthetic 2D target.

    Accepts the SyntheticChoice enum (via the CLI) or a bare string (via tests).
    Both forms compare equal because SyntheticChoice is `(str, Enum)`.
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


def load_image(
    path: Path,
    grid_n: int,
) -> tuple[Float[Array, "n in_dim"], Float[Array, "n"]]:
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
    model: "Model",
    coords: Float[Array, "n in_dim"],
    target: Float[Array, "n"],
) -> Float[Array, ""]:
    """Mean-squared error between the INR's predictions and the target."""
    pred = jax.vmap(model)(coords)
    return jnp.mean((pred - target) ** 2)


def train(
    model: "Model",
    coords: Float[Array, "n in_dim"],
    target: Float[Array, "n"],
    *,
    steps: int,
    lr: float,
) -> tuple["Model", float, float]:
    """Adam + scan training loop. Returns (trained_model, initial_loss, final_loss).

    The whole loop runs as a single `jax.lax.scan` so XLA compiles all `steps`
    iterations into one executable — no Python loop overhead, no per-step
    dispatch latency. The trade-off is that `steps` becomes a JIT-time constant;
    re-running with a different `steps` triggers a recompile.
    """
    optimiser = optax.adam(lr)
    opt_state = optimiser.init(eqx.filter(model, eqx.is_inexact_array))
    jitted_loss = eqx.filter_jit(loss_fn)

    # Take model, opt_state, coords, target as explicit args so JAX caches the
    # compiled scan across calls with different array values — closure-capture
    # of these arrays would mark them as static constants and force recompile
    # per call, which is fine for one-off scripts but a footgun for users
    # copy-pasting this loop into their own training code.
    @eqx.filter_jit
    def run_scan(model, opt_state, coords, target):
        def step(carry, _):
            m, o = carry
            loss, grads = eqx.filter_value_and_grad(loss_fn)(m, coords, target)
            updates, o = optimiser.update(grads, o, eqx.filter(m, eqx.is_inexact_array))
            m = eqx.apply_updates(m, updates)
            return (m, o), loss

        return jax.lax.scan(step, (model, opt_state), None, length=steps)

    initial_loss = float(jitted_loss(model, coords, target))
    (model, _), _ = run_scan(model, opt_state, coords, target)
    final_loss = float(jitted_loss(model, coords, target))
    return model, initial_loss, final_loss


@eqx.filter_jit
def _vmap_model(model: "Model", coords: Float[Array, "n in_dim"]) -> jax.Array:
    """Pure JIT'd vmap so cache hits across calls with different model/coords."""
    return jax.vmap(model)(coords)


def reconstruct(model: "Model", grid_n: int, in_dim: int) -> np.ndarray:
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
    steps: int = typer.Option(500, help="Training steps."),
    lr: float = typer.Option(1e-3, help="Adam learning rate."),
    grid: int = typer.Option(32, help="Grid resolution (synthetic) / resize target (image)."),
    output: Path | None = typer.Option(None, help="Save reconstructed image PNG here."),
    seed: int = typer.Option(0, help="PRNG seed."),
) -> None:
    """Fit an image with an ondes INR and report initial/final loss + PSNR."""
    key = jax.random.key(seed)
    if image is not None:
        coords, target = load_image(image, grid)
        in_dim = int(coords.shape[-1])
    else:
        coords, target = synthetic_target(synthetic, grid)
        in_dim = 2

    model = build_model(basis, in_dim, hidden, layers, omega, key=key)
    model, initial_loss, final_loss = train(model, coords, target, steps=steps, lr=lr)
    # PSNR assumes amplitudes are in [0, 1] (images) or roughly so (synthetics in [-1, 1]).
    # For [-1, 1] targets MSE→PSNR uses peak=2; we report peak=1 PSNR consistently and
    # note the convention. Sitzmann+ 2020 reports peak=1 PSNR on normalised images.
    psnr = -10.0 * np.log10(max(final_loss, 1e-12))
    typer.echo(f"initial_loss={initial_loss:.6f}  final_loss={final_loss:.6f}  PSNR={psnr:.2f} dB")

    if output is not None:
        # reconstruct() returns the correctly-shaped array per in_dim — no extra
        # reshape needed. The earlier `recon.reshape(grid, grid, grid)` was an
        # artefact of make_coords producing a grid_n^3 cube for RGB.
        recon = reconstruct(model, grid, in_dim)
        # Rescale by the *target's* known range, not the prediction's. A
        # non-negative target (image, mandelbrot, gaussian_bump) can produce
        # tiny negative overshoots in `recon` from approximation noise; rescaling
        # on `recon.min() < 0` would dim those outputs spuriously. The target
        # range is the ground truth.
        t_min = float(np.asarray(target).min())
        t_max = float(np.asarray(target).max())
        if t_max > t_min:
            arr = (recon - t_min) / (t_max - t_min)
        else:
            arr = recon
        arr = np.clip(arr, 0.0, 1.0)
        # Ensure parent dir exists — typer doesn't validate output paths the
        # same way --image does, so a user running with --output recons/foo.png
        # in a fresh repo would otherwise hit FileNotFoundError from PIL.
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((arr * 255).astype(np.uint8)).save(output)
        typer.echo(f"saved reconstruction to {output}")


if __name__ == "__main__":
    app()
