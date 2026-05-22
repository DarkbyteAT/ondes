"""Scannability tests for SIREN/HSIREN/WIRE bodies.

Within one body, all layers are the same concrete class and (since #8
dropped the ``is_first`` field) share an identical pytree structure. The
remaining blocker for "stack all N layers" is purely a *shape* mismatch:
layer 0's ``W`` has shape ``(hidden_dim, in_dim)`` while layers 1..N-1
have ``(hidden_dim, hidden_dim)``. When ``in_dim == hidden_dim`` the
stack succeeds and scan-over-all-N-layers runs uniformly; in the
realistic case (``in_dim`` is 2 or 3 for coord inputs, ``hidden_dim`` is
64+), the layer-0-separate pattern is still required.

These tests demonstrate the scan pattern and assert numerical match with
the eager (for-loop) forward pass.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from ondes import HSIREN, SIREN, WIRE
from ondes.basis import HSIRENLayer, SIRENLayer, WIRELayer


def _stack_arrays(layers):
    """Stack a list of homogeneous layers along axis 0 of each array leaf.

    Only stacks the array-typed leaves; static fields (e.g. ``is_first``) must
    be uniform across all layers in the input list or this raises.
    """
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *layers)


def _scan_trunk(body, coord, *, film=None):
    """Reimplement Body.trunk via jax.lax.scan over layers 1..N-1.

    Layer 0 is applied eagerly because its ``W`` has shape
    ``(hidden_dim, in_dim)`` while layers 1..N-1 have
    ``(hidden_dim, hidden_dim)``; that shape mismatch (not any pytree
    structural difference — see #8) is the remaining blocker for stacking
    the whole layer stack uniformly.
    """
    # Layer 0: handled separately (different static field, different in_dim).
    if film is not None:
        gamma_0 = film[0, : body.hidden_dim]
        beta_0 = film[0, body.hidden_dim :]
        h = body.layers[0](coord, gamma=gamma_0, beta=beta_0)
    else:
        h = body.layers[0](coord)

    if len(body.layers) == 1:
        return h

    # Layers 1..N-1: stack and scan.
    hidden = list(body.layers[1:])
    stacked_params, static = eqx.partition(hidden[0], eqx.is_array)
    # Replace stacked_params' leaves with stacked-along-axis-0 versions.
    stacked_params = _stack_arrays([eqx.filter(layer, eqx.is_array) for layer in hidden])

    if film is None:

        def step(h, layer_arrays):
            layer = eqx.combine(layer_arrays, static)
            return layer(h), None

        h, _ = jax.lax.scan(step, h, stacked_params)
    else:
        # FiLM rows 1..N-1 of the film tensor zip with the scan iters.
        film_rest = film[1:]

        def step(h, layer_and_film):
            layer_arrays, film_row = layer_and_film
            layer = eqx.combine(layer_arrays, static)
            gamma = film_row[: body.hidden_dim]
            beta = film_row[body.hidden_dim :]
            return layer(h, gamma=gamma, beta=beta), None

        h, _ = jax.lax.scan(step, h, (stacked_params, film_rest))

    return h


@pytest.mark.parametrize("body_cls,layer_cls", [(SIREN, SIRENLayer), (HSIREN, HSIRENLayer), (WIRE, WIRELayer)])
def test_scan_matches_eager_for_loop(body_cls, layer_cls):
    # Given: a body of each basis class with multiple hidden layers
    # When: comparing the eager trunk() against the scan-rebuilt trunk
    # Then: outputs match to float32 precision. Demonstrates that scan is
    # mechanically viable over layers 1..N-1 once layer 0 is applied
    # separately to absorb its (hidden, in_dim) → (hidden, hidden) shape
    # transition.
    body = body_cls(in_dim=2, hidden_dim=64, num_hidden_layers=6, key=jax.random.key(0))
    coord = jnp.array([0.1, -0.2])

    eager = body.trunk(coord)
    scanned = _scan_trunk(body, coord)

    assert eager.shape == scanned.shape == (64,)
    # atol=1e-4 (not 1e-6) because scan and the eager for-loop compile to
    # slightly different XLA fusion orders; the resulting float32
    # re-association noise accumulates through 6 nested non-linearities,
    # especially for H-SIREN (sin∘sinh has steeper derivatives than sin).
    # A bug in scan plumbing would show ≫ 1e-4 deltas; this tolerance is
    # still strict enough to be load-bearing.
    assert jnp.allclose(eager, scanned, atol=1e-4)


def test_scan_matches_eager_with_film_modulation():
    # Given: a SIREN body + a FiLM tensor covering all layers
    # When: comparing eager trunk(film=...) against the scan-rebuilt version
    # Then: outputs match. Demonstrates the FiLM-conditioned scan also works
    # — the gamma/beta rows zip with the scan iters as a second carry.
    in_dim, hidden_dim, num_layers = 2, 32, 5
    body = SIREN(in_dim=in_dim, hidden_dim=hidden_dim, num_hidden_layers=num_layers, key=jax.random.key(1))
    coord = jnp.array([0.3, -0.4])
    film = 0.5 * jnp.ones((num_layers, 2 * hidden_dim))

    eager = body.trunk(coord, film=film)
    scanned = _scan_trunk(body, coord, film=film)

    assert jnp.allclose(eager, scanned, atol=1e-4)


def test_scan_all_n_layers_matches_eager_when_in_dim_equals_hidden_dim():
    # Given: a SIREN body with in_dim == hidden_dim so all N layers stack
    # When: scanning the full layer stack in one go (no layer-0 special case)
    # Then: output matches the eager trunk. Post-#8 the pytree structure is
    # uniform across all layers, so this is the genuinely-clean case the
    # is_first removal enables: layer 0 and layers 1..N-1 are mechanically
    # identical, the only constraint was shape, and equal in_dim removes
    # even that.
    body = SIREN(in_dim=4, hidden_dim=4, num_hidden_layers=5, key=jax.random.key(4))
    coord = jnp.array([0.1, -0.2, 0.3, -0.4])

    eager = body.trunk(coord)

    # Stack ALL N layers, scan from raw coord.
    params, static = eqx.partition(body.layers[0], eqx.is_array)
    stacked = _stack_arrays([eqx.filter(layer, eqx.is_array) for layer in body.layers])

    def step(h, layer_arrays):
        layer = eqx.combine(layer_arrays, static)
        return layer(h), None

    scanned, _ = jax.lax.scan(step, coord, stacked)

    assert eager.shape == scanned.shape == (4,)
    assert jnp.allclose(eager, scanned, atol=1e-4)


def test_naive_stack_all_layers_fails_on_shape_mismatch_when_in_dim_differs():
    # Given: a realistic SIREN body where in_dim != hidden_dim (coord inputs)
    # When: attempting to stack ALL N layers via jax.tree.map
    # Then: it raises a *shape* error — layer 0's W is (hidden, in_dim) while
    # layers 1..N-1 have (hidden, hidden). Post-#8 the pytree structures are
    # uniform (is_first is no longer a field), so the static-discriminator
    # issue is gone; what remains is a genuine shape-level constraint.
    # Documents why the layer-0-separate scan pattern persists for the
    # common in_dim != hidden_dim case.
    body = SIREN(in_dim=2, hidden_dim=64, num_hidden_layers=4, key=jax.random.key(2))
    with pytest.raises(ValueError, match="same shape"):
        _stack_arrays(list(body.layers))


def test_all_layers_stack_cleanly_when_in_dim_equals_hidden_dim():
    # Given: a SIREN body where in_dim == hidden_dim (artificial; not the typical
    # INR shape, but exercises the "all layers genuinely uniform" code path)
    # When: stacking ALL N layers via jax.tree.map
    # Then: the stack succeeds. Post-#8 the pytree structure is uniform across
    # layers; the only residual constraint is shape, and equal in_dim removes
    # that too. Documents that scan-over-the-full-stack is mechanically clean
    # whenever shapes align — no static-field discriminator left to block it.
    body = SIREN(in_dim=4, hidden_dim=4, num_hidden_layers=4, key=jax.random.key(3))
    stacked = _stack_arrays(list(body.layers))
    # Every array leaf gains a leading axis of size num_hidden_layers.
    assert stacked.W.shape == (4, 4, 4)
    assert stacked.b.shape == (4, 4)
    assert stacked.omega.shape == (4,)
