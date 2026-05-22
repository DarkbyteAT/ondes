"""Smoke test: a SIREN fits a small synthetic image.

This is the canonical INR use case ondes was designed for. The test constructs
a small SIREN body, fits a 16x16 sinusoidal target via Adam on MSE, and
asserts the final loss is substantially lower than the initial loss. The
threshold is loose (3x reduction) — a smoke check that training is wired up
end-to-end, not a convergence claim.

Optax is a *dev* dependency only — ondes itself has no training-stack deps
(per DECISIONS.md; the library owns coord-to-value primitives, downstream
owns optimisers). Test code can use anything; library code cannot.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

import ondes


def test_siren_fits_small_sinusoidal_image():
    # Given: a 16x16 grid sampled from sin(2*pi*3*x) * cos(2*pi*3*y) on [-1, 1]^2,
    # a small SIREN with image-fitting defaults (omega=30 per Sitzmann 2020),
    # and Adam at lr=1e-3.
    grid_n = 16
    coords_1d = jnp.linspace(-1.0, 1.0, grid_n)
    xs, ys = jnp.meshgrid(coords_1d, coords_1d, indexing="ij")
    coords = jnp.stack([xs.ravel(), ys.ravel()], axis=-1)  # (256, 2)
    target = (jnp.sin(2.0 * jnp.pi * 3.0 * xs) * jnp.cos(2.0 * jnp.pi * 3.0 * ys)).ravel()

    key = jax.random.PRNGKey(0)
    siren = ondes.SIREN(
        in_dim=2,
        hidden_dim=32,
        num_hidden_layers=2,
        key=key,
        omega_first=30.0,
        omega_hidden=30.0,
    )

    def loss_fn(model, coords, target):
        pred = jax.vmap(model)(coords)
        return jnp.mean((pred - target) ** 2)

    optimiser = optax.adam(1e-3)
    opt_state = optimiser.init(eqx.filter(siren, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, coords, target):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, coords, target)
        updates, opt_state = optimiser.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # When: we measure initial loss and train for 200 Adam steps.
    initial_loss = float(loss_fn(siren, coords, target))
    for _ in range(200):
        siren, opt_state, _ = step(siren, opt_state, coords, target)
    final_loss = float(loss_fn(siren, coords, target))

    # Then: final loss is at most 30% of initial — substantial learning,
    # not a convergence claim. A genuine wiring break would leave the ratio at ~1.
    assert final_loss < 0.3 * initial_loss, (
        f"image fit smoke test: expected final_loss < 0.3 * initial_loss, "
        f"got initial={initial_loss:.4f}, final={final_loss:.4f}, "
        f"ratio={final_loss / initial_loss:.3f}"
    )
