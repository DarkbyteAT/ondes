"""PNF basis: Polynomial Neural Fields (Yang+ 2022, NeurIPS).

"Polynomial Neural Fields for Subband Decomposition and Manipulation"
(arXiv:2302.04862). The paper defines the per-band trunk (Section 3.3.3,
Eq. 5) as a Fourier-filter recurrence with one linear (the paper calls
it ``W_l``, the "mix layer") between filter multiplications:

```
    z_0 = g_0(x)                       # Eq. 5 with k=1: Z_{j,1} = G_j(x, 0, Δ_1)
    z_{i+1} = g_{i+1}(x) * (M_i z_i)   # Eq. 5 with k>=2: Z_{j,k} = G_j(x, 0, Δ_k) · W_l · Z_{j,k-1}
```

where ``g_i`` are Fourier filters (``G_j`` in the paper — same sinusoidal
shape as ``FourierMFN``'s) and ``M_i`` is the per-step linear ``W_l``.
Theorem 2 ties the structure to polynomial composition: an elementwise
product of two basis-limited PNFs is a PNF limited to the product of
their subbands. Stacking these factors produces a finite linear sum of
Fourier basis functions over a controlled subband.

**Structural difference from FourierMFN.** The paper's recurrence has
no additive bias inside the multiplication and no separate recurrence-
linear path: the single linear is *the* mix layer. ``FourierMFN`` uses
``z_{i+1} = g_{i+1}(x) * (W_i z_i + b_i)`` — same multiplicative-filter
shape, but with an extra learnable bias inside the multiplication.
The bias is exactly the mechanism that makes ``FourierMFN`` produce a
constant-shifted polynomial at every step; PNF deliberately drops it
to keep the per-step output a pure polynomial in the filter outputs,
which is what makes Theorem 2's subband-product claim hold. Earlier
revisions of this module carried a second linear plus a bias inside
the multiplication, which (a) was algebraically redundant with the
mix linear and (b) broke the paper-faithful "pure linear, no bias"
structure — the recurrence now matches Eq. 5 directly.

The paper's full architecture (``SubbandNet`` in the reference:
github.com/stevenygd/PNF/blob/main/models/sbn_ndims.py) is a
**multi-output, multi-subband** network — Algorithm 1 / Eq. 4
decomposes the signal into overlapping frequency bands and produces one
output per band. ``ondes.basis.Body`` has a single output by contract,
so this implementation models the per-band trunk (a single Fourier-PNF
with the mix layer) and leaves the subband-decomposition policy to
downstream consumers. A multi-output subband variant would expose
``trunk`` returns per band; that's outside the Body contract and
belongs as an ``examples/`` recipe if needed.

The single-band Fourier-PNF here matches the reference's per-band trunk
(``MultLayer(MixLayer(z), g(x))`` alternating between ``FourierFilter``
calls — ``MixLayer`` is a bias-free linear in the reference) in
structure and initialisation.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ondes.basis._base import Body, _validate_body_args
from ondes.basis.mfn import FourierFilter, _mfn_recurrence_init


class PNF(Body):
    """Single-band Fourier-PNF body (Yang+ 2022).

    The recurrence is ``z_{i+1} = g_{i+1}(x) * (M_i z_i)`` (paper Eq. 5
    with the single-band substitution ``W_l = M_i``), with one Fourier
    filter per step (``num_hidden_layers + 1`` total) and one mix
    matrix ``M_i`` per recurrence step (``num_hidden_layers`` total).

    The mix matrix is the *only* learnable linear between filters; there
    is deliberately no additive bias inside the multiplication (see the
    module docstring for the paper-faithfulness argument).

    Args:
        in_dim: Coordinate (input) dimension.
        hidden_dim: Width of the recurrence and each filter.
        num_hidden_layers: Number of recurrence steps.
        key: JAX PRNG key.
        out_features: Readout width; see :class:`ondes.basis.siren.SIREN`.
        input_scale: Filter-frequency uniform-init scale (paper default
            uses values comparable to MFN's ``input_scale = 256`` for
            natural images).
        weight_scale: Scale on mix-layer uniform-init bound (same
            convention as MFN's recurrence-linear init; we reuse
            ``_mfn_recurrence_init`` for the bound formula and discard
            its bias output — the paper's mix layer is bias-free).
    """

    filters: tuple
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
        """Initialise N+1 Fourier filters and the (mix, readout) stack."""
        out_features = _validate_body_args(num_hidden_layers, out_features)
        n_filters = num_hidden_layers + 1
        keys = jax.random.split(key, n_filters + num_hidden_layers + 1)
        filter_keys = keys[:n_filters]
        mix_keys = keys[n_filters : n_filters + num_hidden_layers]
        readout_key = keys[-1]

        self.filters = tuple(
            FourierFilter(in_dim, hidden_dim, input_scale, num_hidden_layers, key=k) for k in filter_keys
        )

        mix_Ws = []
        for i in range(num_hidden_layers):
            # _mfn_recurrence_init returns (W, b); discard b — the paper's
            # mix layer is bias-free (Eq. 5), and the reference's MixLayer
            # is a bias=None Conv1d.
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
        """PNF mix-then-multiply recurrence (paper Eq. 5).

        ``film`` gates the post-mix state ``M_i z_i`` before the next
        filter multiplies in. The first filter is unmodulated — same
        convention as MFN/BACON. Note that FiLM is an ondes-side
        downstream-extension hook, not in the paper; bias-free mix is
        preserved by leaving ``beta=0`` in the FiLM tensor.
        """
        self._check_film_shape(film)
        z = self.filters[0](coord)
        for i in range(self.num_hidden_layers):
            pre = self.mix_W[i] @ z
            if film is not None:
                gamma = film[i, : self.hidden_dim]
                beta = film[i, self.hidden_dim :]
                pre = gamma * pre + beta
            z = self.filters[i + 1](coord) * pre
        return z
