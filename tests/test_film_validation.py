"""Cross-basis FiLM-shape contract.

Every concrete ondes basis body accepts ``film`` of shape
``(num_hidden_layers, 2 * hidden_dim)``. The check lives in one place
(``Body._check_film_shape``) and every concrete ``trunk`` calls it as
its first line. This test parametrises across the full basis family and
asserts the contract holds — re-introducing per-body shape duplication
(or skipping the check in a new body subclass) shows up here as a
``DID NOT RAISE`` failure.
"""

import jax
import jax.numpy as jnp
import pytest

import ondes


# Every shipped body. New bases (subclasses of ``ondes.Body``) should be
# added here so the FiLM contract is automatically exercised against them.
_BODY_CLASSES = (
    ondes.SIREN,
    ondes.HSIREN,
    ondes.WIRE,
    ondes.FINER,
    ondes.RFF,
    ondes.BACON,
    ondes.PNF,
    ondes.FourierMFN,
    ondes.GaborMFN,
)


def _build(body_cls: type[ondes.Body]) -> ondes.Body:
    """Construct a tiny instance with the kwargs each body's __init__ accepts.

    Each class has its own constructor surface, so we instantiate per-cls
    here rather than chasing a unified factory — which would re-introduce
    the discriminator dispatch DECISIONS.md explicitly forbids.
    """
    key = jax.random.key(0)
    common = dict(in_dim=2, hidden_dim=8, num_hidden_layers=2, key=key)
    if body_cls is ondes.WIRE:
        return body_cls(**common, s_init=1.0)
    return body_cls(**common)


@pytest.mark.parametrize("body_cls", _BODY_CLASSES)
def test_trunk_rejects_film_of_wrong_shape(body_cls: type[ondes.Body]) -> None:
    # Given: a tiny body of one of the nine shipped basis kinds; a coord;
    # and a FiLM tensor whose shape disagrees with the expected
    # (num_hidden_layers, 2 * hidden_dim) = (2, 16). Use (3, 16) as the
    # bad shape — wrong on the leading axis, same dtype/strides so the
    # failure is structural not numerical.
    body = _build(body_cls)
    coord = jnp.array([0.1, -0.2])
    wrong_film = jnp.zeros((3, 2 * 8))

    # When/Then: calling trunk with the wrong-shape film raises ValueError
    # with a message naming the expected and actual shapes. Same error
    # message across all bodies, since the check lives on Body once.
    with pytest.raises(ValueError, match=r"film must have shape"):
        body.trunk(coord, film=wrong_film)


@pytest.mark.parametrize("body_cls", _BODY_CLASSES)
def test_trunk_rejects_film_of_wrong_trailing_dim(body_cls: type[ondes.Body]) -> None:
    # Given: a body and a FiLM whose trailing axis is wrong (not 2*hidden_dim).
    # When/Then: still raises — covers the "split gamma | beta" assumption,
    # not just the leading-axis count.
    body = _build(body_cls)
    coord = jnp.array([0.1, -0.2])
    wrong_film = jnp.zeros((2, 2 * 8 + 1))  # off-by-one on the trailing dim

    with pytest.raises(ValueError, match=r"film must have shape"):
        body.trunk(coord, film=wrong_film)


@pytest.mark.parametrize("body_cls", _BODY_CLASSES)
def test_trunk_accepts_well_shaped_film(body_cls: type[ondes.Body]) -> None:
    # Given: a body and a FiLM with the contract-conformant shape.
    # When: calling trunk
    # Then: returns hidden features without raising — the check is
    # narrow enough not to reject the valid case.
    body = _build(body_cls)
    coord = jnp.array([0.1, -0.2])
    good_film = jnp.zeros((body.num_hidden_layers, 2 * body.hidden_dim))
    h = body.trunk(coord, film=good_film)
    assert h.shape == (body.hidden_dim,)
