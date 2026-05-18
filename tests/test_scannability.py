"""Scannability tests for SIREN/HSIREN/WIRE bodies.

The polymorphism refactor produces uniform per-class layer pytree structure:
within one body, all layers are the same concrete class and have the same
array-leaf shapes. This *enables* ``jax.lax.scan`` over the layer stack — but
with one caveat: the first layer has ``is_first=True`` (a static bool field)
while subsequent layers have ``is_first=False``, so a naive
``jax.tree.map(stack, *body.layers)`` fails on the static-field mismatch.

The realistic scan pattern: apply layer 0 separately, then scan layers
1..N-1. These tests demonstrate that pattern and assert numerical match
with the eager (for-loop) forward pass.
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

    Layer 0 is applied eagerly because its ``is_first=True`` static field
    differs from the rest. The remaining N-1 layers all have ``is_first=False``
    and stack cleanly.
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
    # mechanically viable over layers 1..N-1 once you partition by is_first.
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


def test_naive_stack_all_layers_fails_on_is_first_mismatch():
    # Given: a SIREN body where layer 0 has is_first=True and the rest have is_first=False
    # When: attempting to stack ALL layers via jax.tree.map
    # Then: it raises — the static `is_first` field discriminates layer 0 from
    # the rest at the pytree level. Documents WHY the realistic scan pattern
    # has to apply layer 0 separately rather than scanning over all N layers
    # uniformly. (If a future refactor unified the init bound, this test would
    # need updating.)
    body = SIREN(in_dim=4, hidden_dim=4, num_hidden_layers=4, key=jax.random.key(2))
    with pytest.raises(ValueError, match="Mismatch"):
        _stack_arrays(list(body.layers))


def test_homogeneous_layers_stack_cleanly():
    # Given: a list of layers all constructed with is_first=False
    # When: stacking via jax.tree.map
    # Then: stacking succeeds — confirms the ONLY blocker for naive
    # stack-everything is is_first, not anything intrinsic to the polymorphism
    # design. Within "all hidden layers" the design is genuinely scan-friendly.
    keys = jax.random.split(jax.random.key(3), 4)
    layers = [SIRENLayer(4, 4, omega_init=1.0, is_first=False, key=keys[i]) for i in range(4)]
    stacked = _stack_arrays(layers)
    # Each array leaf gains a leading axis of size 4.
    assert stacked.W.shape == (4, 4, 4)
    assert stacked.b.shape == (4, 4)
    assert stacked.omega.shape == (4,)
