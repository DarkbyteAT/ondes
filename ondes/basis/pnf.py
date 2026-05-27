"""PNF basis: Polynomial Neural Fields (Yang+ 2022, NeurIPS).

"Polynomial Neural Fields for Subband Decomposition and Manipulation"
(arXiv:2302.04862). PNF generalises MFN with an explicit *mix* matrix
between recurrence steps:

```
    z_0 = g_0(x)
    z_{i+1} = g_{i+1}(x) * (M_i z_i + W_i z_i + b_i)
```

where ``g_i`` are Fourier filters (sinusoidal, same as ``FourierMFN``) and
``M_i`` is a per-step linear mixing matrix the paper calls the *mix layer*.
The mix-then-multiply structure exposes the polynomial composition: at
each step the running state is a polynomial of the filter outputs, and
the mix layer controls which monomials get amplified — this is the
mechanism behind the paper's "interpretable composition" claim.

PAPER AMBIGUITY: the full paper architecture (``SubbandNet`` in the
reference: github.com/stevenygd/PNF/blob/main/models/sbn_ndims.py) is a
**multi-output, multi-subband** network — Algorithm 1 of the paper
decomposes the signal into overlapping frequency bands and produces one
output per band. ``ondes.basis.Body`` has a single output by contract,
so this implementation models the per-band trunk (a single Fourier-PNF
with the mix layer) and leaves the subband-decomposition policy to
downstream consumers. A multi-output subband variant would expose
``trunk`` returns per band; that's outside the Body contract and
belongs as an ``examples/`` recipe if needed.

The single-band Fourier-PNF here matches the reference's per-band trunk
(``MultLayer`` + ``MixLayer`` alternating between ``FourierFilter`` calls)
in structure and initialisation.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ondes.basis._base import Body, _validate_body_args
from ondes.basis.mfn import FourierFilter, _mfn_recurrence_init


class PNF(Body):
    """Single-band Fourier-PNF body (Yang+ 2022).

    The recurrence is ``z_{i+1} = g_{i+1}(x) * (M_i z_i + W_i z_i + b_i)``
    with one Fourier filter per step (``num_hidden_layers + 1`` total) and
    one *mix matrix* ``M_i`` per recurrence step (``num_hidden_layers``
    total). The mix matrix is the only structural difference from
    ``FourierMFN`` — semantically, PNF augments MFN's recurrence-linear with
    a second linear path whose role is to expose polynomial-composition
    interpretability (paper Section 3.1).

    Args:
        in_dim: Coordinate (input) dimension.
        hidden_dim: Width of the recurrence and each filter.
        num_hidden_layers: Number of recurrence steps.
        key: JAX PRNG key.
        out_features: Readout width; see :class:`ondes.basis.siren.SIREN`.
        input_scale: Filter-frequency uniform-init scale (paper default
            uses values comparable to MFN's ``input_scale = 256`` for
            natural images).
        weight_scale: Scale on recurrence-linear and mix-layer uniform-init
            bounds (same convention as MFN).
    """

    filters: tuple
    recurrence_W: Float[Array, "n_layers hidden hidden"]
    recurrence_b: Float[Array, "n_layers hidden"]
    mix_W: Float[Array, "n_layers hidden hidden"]

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_hidden_layers,
        *,
        key,
        out_features=None,
        input_scale=256.0,
        weight_scale=1.0,
    ):
        """Initialise N+1 Fourier filters and the (mix, recurrence-linear, readout) stack."""
        out_features = _validate_body_args(num_hidden_layers, out_features)
        n_filters = num_hidden_layers + 1
        keys = jax.random.split(key, n_filters + 2 * num_hidden_layers + 1)
        filter_keys = keys[:n_filters]
        rec_keys = keys[n_filters : n_filters + num_hidden_layers]
        mix_keys = keys[n_filters + num_hidden_layers : n_filters + 2 * num_hidden_layers]
        readout_key = keys[-1]

        self.filters = tuple(
            FourierFilter(in_dim, hidden_dim, input_scale, num_hidden_layers, key=k) for k in filter_keys
        )

        rec_Ws, rec_bs = [], []
        for i in range(num_hidden_layers):
            W, b = _mfn_recurrence_init(hidden_dim, hidden_dim, weight_scale, rec_keys[i])
            rec_Ws.append(W)
            rec_bs.append(b)
        self.recurrence_W = jnp.stack(rec_Ws)
        self.recurrence_b = jnp.stack(rec_bs)

        mix_Ws = []
        for i in range(num_hidden_layers):
            mW, _ = _mfn_recurrence_init(hidden_dim, hidden_dim, weight_scale, mix_keys[i])
            mix_Ws.append(mW)
        self.mix_W = jnp.stack(mix_Ws)

        out_dim = 1 if out_features is None else out_features
        rw, rb = _mfn_recurrence_init(hidden_dim, out_dim, weight_scale, readout_key)
        self.readout_W = rw
        self.readout_b = rb
        self.layers = ()
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

    def trunk(self, coord, *, film=None):
        """PNF mix-then-multiply recurrence.

        ``film`` is applied to the post-(mix + recurrence-linear) sum, gating
        the polynomial-composition state before the next filter multiplies in.
        The first filter is unmodulated — same convention as MFN/BACON.
        """
        z = self.filters[0](coord)
        for i in range(self.num_hidden_layers):
            pre = self.mix_W[i] @ z + self.recurrence_W[i] @ z + self.recurrence_b[i]
            if film is not None:
                gamma = film[i, : self.hidden_dim]
                beta = film[i, self.hidden_dim :]
                pre = gamma * pre + beta
            z = self.filters[i + 1](coord) * pre
        return z
