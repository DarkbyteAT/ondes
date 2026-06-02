"""RFF basis: Gaussian Fourier-feature encoding + plain ReLU MLP (Tancik+ 2020).

Reference: "Fourier Features Let Networks Learn High Frequency Functions in
Low Dimensional Domains" (NeurIPS 2020, arXiv:2006.10739). The encoding step
is ``[cos(2*pi*B*x), sin(2*pi*B*x)]`` with ``B`` sampled from
``N(0, sigma^2 * I)``; ``sigma`` is the bandwidth knob.

Unlike SIREN/H-SIREN/WIRE the activation is a plain pointwise ReLU and the
"basis" lives in the input encoding rather than in the activation. The body
still conforms to the same ``Body`` contract (trunk → readout, optional FiLM,
scalar/vector ``out_features``) so downstream renderers can treat all bases
interchangeably.

Init follows the paper's reference TF/PyTorch implementations: encoding
``B`` is Gaussian with std ``sigma``; MLP weights use Kaiming-uniform suited
to the ReLU non-linearity; biases are zeroed.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ondes.basis._base import Basis, Body, _validate_body_args


def _kaiming_uniform_relu(in_dim, out_dim, key):
    """Kaiming-uniform init for a ReLU-followed linear layer.

    Bound is ``sqrt(6 / in_dim)``, the He (Kaiming) bound that keeps activation
    variance stable through a ReLU. Used by RFFLayer and by the RFF body's
    readout because both feed into ReLU (or, for the readout, a linear sink
    that benefits from the same variance assumption).
    """
    bound = jnp.sqrt(6.0 / in_dim)
    W = jax.random.uniform(key, (out_dim, in_dim), minval=-bound, maxval=bound)
    b = jnp.zeros((out_dim,))
    return W, b


class RFFLayer(Basis):
    """RFF hidden layer: ``relu(W @ pre + b)``.

    The ``omega`` field is unused (RFF has no per-layer learnable frequency;
    the spectral scale lives in the encoding's ``B`` matrix). It is retained
    at value ``1.0`` for pytree-structure parity with the rest of the
    ``Basis`` family — downstream code that gathers omegas across a mixed-basis
    pytree doesn't have to special-case RFF.
    """

    def __init__(self, in_dim, out_dim, *, key):
        """Initialise the linear weights (Kaiming-uniform) and a unit ``omega`` placeholder."""
        self.W, self.b = _kaiming_uniform_relu(in_dim, out_dim, key)
        self.omega = jnp.array(1.0)

    def _activate(self, pre):
        return jax.nn.relu(pre)


class RFF(Body):
    """RFF body: Gaussian-RFF encoding then a plain ReLU MLP (Tancik+ 2020).

    The encoding's ``B`` matrix is constructed internally to keep the body
    self-contained — RFF is a packaged "Gaussian features → ReLU MLP"
    architecture in the paper, not a generic encoding + MLP composition. (For
    the latter, compose ``ondes.Gaussian`` with a user-built MLP directly.)

    Args:
        in_dim: Coordinate (input) dimension.
        hidden_dim: Width of each hidden layer.
        num_hidden_layers: Number of hidden ReLU layers (not counting the
            encoding, which is layer 0, or the readout).
        num_freqs: Number of sampled Fourier frequencies. The encoded
            coordinate has ``2 * num_freqs`` dimensions (sin + cos).
        sigma: Bandwidth knob for the Gaussian frequency draw. Tancik+ 2020
            recommend ``sigma=10`` for natural-image fitting at the default
            ``num_freqs=256``; reduce for smoother targets, increase for
            sharper.

    Note:
        The encoding matrix ``B`` is sampled at construction and is meant to
        be **non-trainable** (the Gaussian-RFF paper draws it once and freezes
        it; the bandwidth knob is ``sigma``, baked into ``B`` at construction,
        not a separate gradient). ``ondes`` does not own optimiser plumbing,
        so this is enforced by the user with ``eqx.partition`` (or
        ``optax.masked``). The ``fix_encoding_mask()`` method returns a mask
        pytree downstream code can use directly — symmetric with
        :meth:`ondes.basis.bacon.BACON.fix_filters_mask`.
    """

    B: Float[Array, "num_freqs rank"]

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_hidden_layers,
        *,
        key,
        out_features=None,
        num_freqs=256,
        sigma=10.0,
    ):
        """Initialise the encoding's ``B`` matrix and the ReLU-MLP layers + readout."""
        out_features = _validate_body_args(num_hidden_layers, out_features)
        k_enc, *rest = jax.random.split(key, num_hidden_layers + 2)
        self.B = float(sigma) * jax.random.normal(k_enc, (num_freqs, in_dim))
        encoded_dim = 2 * num_freqs

        layers = []
        for i in range(num_hidden_layers):
            in_d = encoded_dim if i == 0 else hidden_dim
            layers.append(RFFLayer(in_d, hidden_dim, key=rest[i]))
        self.layers = tuple(layers)

        rw, rb = _kaiming_uniform_relu(hidden_dim, 1 if out_features is None else out_features, rest[-1])
        self.readout_W = rw
        self.readout_b = rb
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

    def fix_encoding_mask(self):
        """Return a pytree mask that selects only the learnable (non-encoding-B) leaves.

        Use with ``eqx.partition(body, body.fix_encoding_mask())`` to split the
        body into ``(learnable, fixed)`` halves before applying an optimiser
        update — the fixed half (encoding matrix ``B``) must not receive
        gradient steps to match the Gaussian-RFF paper's frozen-features
        convention.
        """
        # Start from "everything is learnable", then zero out the encoding B.
        mask = jax.tree_util.tree_map(lambda x: eqx.is_array(x), self)
        return eqx.tree_at(lambda b: b.B, mask, False)

    def _encode(self, coord):
        angles = 2.0 * jnp.pi * (self.B @ coord)
        return jnp.concatenate([jnp.cos(angles), jnp.sin(angles)])

    def trunk(self, coord, *, film=None):
        """Apply Fourier-feature encoding then the ReLU MLP.

        ``film`` is threaded layer-by-layer through the MLP exactly as in the
        SIREN-family bodies — the encoding step is not modulated (it's a
        deterministic feature map, not a layer with linear weights and bias
        that FiLM would gate).
        """
        h = self._encode(coord)
        for i, layer in enumerate(self.layers):
            if film is not None:
                gamma = film[i, : self.hidden_dim]
                beta = film[i, self.hidden_dim :]
                h = layer(h, gamma=gamma, beta=beta)
            else:
                h = layer(h)
        return h
